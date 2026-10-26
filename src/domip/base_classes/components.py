import inspect
import time
import traceback
from abc import abstractmethod
from itertools import product
from threading import Lock
from typing import Dict, Type, List, Callable, Tuple, Union

import torch
from tensordict import TensorDict

from domip.base_classes.goals import visualize_goal_derivatives

from domip.base_classes.connections import Connection
from domip.base_classes.derivatives import DerivativeDict, compute_derivatives, \
    get_steepest_gradient_from_derivative_block
from domip.base_classes.goals import evaluate_goals, Goal
from domip.base_classes.util import collect_args, collect_inputs, collect_derivatives, gather_partial_derivatives, \
    generate_outputdict, collect_shapes, collect_required_signs , collect_flipped_signs
from domip.tower_of_london_direct.global_params import NUM_BLOCKS
from domip.base_classes.visualizations import (
    visualize_partial_derivatives_between_external,
    visualize_component_derivatives_external,
    collect_partial_derivative_chain_by_pairs,
    visualize_chain_derivative_product_external,
)

import matplotlib.pyplot as plt
import datetime

class Component:

    name: str
    lock: Lock
    dtype: torch.dtype
    device: torch.device
    connections: Dict[str, Type[Connection]]
    quantities: TensorDict
    timestamp: torch.Tensor
    goals: Dict[str, Type[Goal]]
    partial_derivatives: TensorDict
    derivatives_forward_mode: DerivativeDict
    derivatives_backward_mode: DerivativeDict
    finished_forward_derivatives: DerivativeDict
    other_synchronization_functions: List[Callable]

    def __init__(self, name:str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        self.name = name
        self.lock = Lock()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.connections = {k: connections[k](device, dtype, mockbuild) for k in connections.keys()}
        if goals is None:
            self.goals = dict()
        else:
            self.goals = {k: goals[k](device, dtype, mockbuild) for k in goals.keys()}
        self.quantities = TensorDict(dict(), device=device, batch_size=())
        self.timestamp = torch.zeros(1, device=device, dtype=dtype)
        self.partial_derivatives = TensorDict(dict(), device=device, batch_size=())
        self.derivatives_forward_mode = DerivativeDict()
        self.derivatives_backward_mode = DerivativeDict()
        self.finished_forward_derivatives = DerivativeDict()
        self.other_synchronization_functions = list()
        self.initial_definitions()
        self.initialized_all_quantities = False

    def get_dt(self, new_time: torch.tensor) -> torch.tensor:
        return new_time - self.timestamp

    @abstractmethod
    def initialize_quantities(self) -> bool:
        return False

    @abstractmethod
    def initial_definitions(self) -> None:
        pass

    def update(self, new_time: torch.tensor) -> None:
        if not self.initialized_all_quantities:
            self._attempt_init(new_time)
        else:
            self._attempt_update(new_time)
        self._attempt_synch(new_time)

    def _attempt_synch(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.timestamp = new_time
                self.synchronize()
        except Exception:
            print("Could not synchronize " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _attempt_update(self, new_time: torch.tensor) -> None:
        raise NotImplementedError()

    def synchronize(self) -> None:
        for c in self.connections.values():
            try:
                with c.lock:
                    c.derivatives_forward_mode.update_(self.derivatives_forward_mode)
                    c.derivatives_backward_mode.update_(self.derivatives_backward_mode)
                    # FIXME this is a hotfix for the weird dimension issues"
                    if self.name != "Planner":
                        c.connected_quantities.update_(self.quantities)
                    else:
                        c.connected_quantities.update(self.quantities)
                    for k in self.quantities.keys():
                        c.connected_quantities_timestamps[k] = self.timestamp
                        c.connected_quantities_initialized[k] = self.initialized_all_quantities
            except (ValueError, KeyError):
                print("During Synchronization with Connection "+str(c.name)+" encountered following Issue:")
                traceback.print_exc()
                print("New Updated Values from Compoenent:")
                print(self.quantities)
                print("Connection Previous Value:")
                with c.lock:
                    print(c.connected_quantities)
        for sync_func in self.other_synchronization_functions:
            sync_func(self)

    def _attempt_init(self, new_time: torch.tensor) -> None:
        print(self.name + " not initialized yet")
        try:
            with self.lock:
                self.initialized_all_quantities = self.initialize_quantities()
        except Exception:
            print("Could not initialize " + str(self.name) + ":")
            traceback.print_exc()


class SensorComponent(Component):

    def __init__(self, name: str, connections: Dict[str, Callable],
                 dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

        super().__init__(name, connections, None, dtype, device, mockbuild=mockbuild)

    @abstractmethod
    def obtain_measurements(self) -> bool:
        return False

    def initialize_quantities(self) -> bool:
        return self.obtain_measurements()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.obtain_measurements()
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()


class ActionComponent(Component):

    def __init__(self, name: str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, forward_mode:bool=False, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):
        super().__init__(name, connections, goals, dtype, device, mockbuild=mockbuild)
        self.active = False
        self.forward_mode = forward_mode

    def start(self) -> None:
        try:
            self._start()
            self.active = True
        except Exception:
            print("Could not start " + str(self.name) + ":")
            traceback.print_exc()

    def stop(self) -> None:
        try:
            self._stop()
            self.active = False
        except Exception:
            print("Could not stop " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def _start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _stop(self) -> None:
        raise NotImplementedError

    def get_steepest_gradient(self, quantity : str, time_threshold: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
        try:
            if self.forward_mode:
                my_derivatives = self.finished_forward_derivatives[quantity]
            else:
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                my_derivatives = relevant_backward_derivatives[quantity]
        except KeyError:
            print("No Gradients for quantity "+quantity+ " available yet")
            return torch.zeros_like(self.quantities[quantity]), [], []

        if time_threshold is None:
            time_threshold = torch.zeros(1, dtype=self.dtype, device=self.device)
        steepest_grad, timestamps, trace = get_steepest_gradient_from_derivative_block(my_derivatives, time_threshold)
        print("Steepest Gradient Selection")
        print("Options:")
        print(my_derivatives)
        print("Selected:")
        print(str(trace))
        return steepest_grad, timestamps, trace

    @abstractmethod
    def determine_new_actions(self) -> None:
        raise NotImplementedError()

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            with self.lock:
                self.determine_new_actions()
                if self.active:
                    self.send_action_values()
                else:
                    # print(self.name + " not sending action values, because it is deactivated")
                    pass
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    @abstractmethod
    def send_action_values(self) -> None:
        raise NotImplementedError()


def gradient_descent(gain, last_action, steepest_grad):
    if torch.norm(steepest_grad) > 0 and torch.norm(last_action) > 0:
        # gradient descent
        new_action = last_action / torch.norm(last_action) - gain * steepest_grad / torch.norm(
            steepest_grad)
    elif torch.norm(steepest_grad) > 0:
        new_action = - steepest_grad / torch.norm(
            steepest_grad)
    else:
        # no action
        new_action = torch.zeros_like(last_action[:3])
    return new_action


class EstimationComponent(Component):

    def __init__(self, name: str, connections: Dict[str, Callable], goals : Union[None,Dict[str, Callable]] = None, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, no_differentiation : bool = False, prevent_loops_in_differentiation : bool = True, max_length_differentiation_trace : Union[int, None] = None, num_blocks = NUM_BLOCKS):
        self.num_blocks = num_blocks
        self.no_differentiation = no_differentiation
        self.prevent_loops_in_differentiation = prevent_loops_in_differentiation
        self.max_length_differentiation_trace = max_length_differentiation_trace
        self.wait_viz_counter = 0
        super().__init__(name, connections, goals, dtype, device, mockbuild=mockbuild)
        if not mockbuild:
            f_func, output_signature_diff, output_signature_non_diff = self.define_estimation_function_f()
            self.f_input_signature = inspect.getfullargspec(f_func).args
            self.f_output_signature_diff = output_signature_diff
            self.f_output_signature_non_diff = output_signature_non_diff
            # self.f_func = torch.func.functionalize(f_func)
            self.f_func = f_func
            self.idxs_diff_against_args = None
            self.goal_values = None
            self.latest_goal_derivatives = None

    @abstractmethod
    def define_estimation_function_f(self) -> Tuple[Callable, List[str], List[str]]:
        pass

    def _attempt_update(self, new_time: torch.tensor) -> None:
        try:
            self.update_state_and_derivatives(new_time)
        except Exception:
            print("Could not update " + str(self.name) + ":")
            traceback.print_exc()

    def update_state_and_derivatives(self, new_time: torch.tensor) -> None:
        start = time.time()
        with self.lock:
            all_inputs = collect_inputs(self.quantities, self.connections, self.get_dt(new_time))
            relevant_forward_derivatives, relevant_backward_derivatives = collect_derivatives(self.connections)
            required_signs_dict = collect_required_signs(self.connections)
            flipped_signs_dict = collect_flipped_signs(self.connections)
        args, diff_argnums, input_names_diff = collect_args(all_inputs, self.f_input_signature,
                                                            self.idxs_diff_against_args, self.f_output_signature_diff)
        up = time.time()
        outputs, partial_jacs = self.inference_and_differentiation(args, diff_argnums)
        inf = time.time()
        output_dict = generate_outputdict(outputs, self.f_output_signature_non_diff)
        backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives = self.derivative_processing(
            all_inputs, input_names_diff, new_time, output_dict, partial_jacs, relevant_backward_derivatives,
            relevant_forward_derivatives, required_signs_dict, flipped_signs_dict)
        now = time.time()
        if self.wait_viz_counter > 5:
            #print("With:")
            """ for k, d in partial_derivatives.items(include_nested=True, leaves_only=True):
                values = torch.nonzero(d)
                print(values.shape)
                #print(k.shape)
                print(k)
                for v in values:
                    idx = tuple(v.tolist())
                    print(v, d[idx])
                print("Zero entries:")
                if d.ndim == 0:
                    if d.item() == 0:
                        print(torch.empty(0, dtype=torch.long, device=d.device), d)
                else:
                    for idx in product(*(range(size) for size in d.shape)):
                        if d[idx].item() == 0:
                            print(torch.tensor(idx, dtype=torch.long, device=d.device), d[idx])
            print("Print Components")
            print(self.name, ": ", up - start, inf - up, now - inf)
            print("Component " + str(self.name))
            print("From:")
            print(relevant_backward_derivatives)
            visualize_component_derivatives_external(relevant_backward_derivatives, self.name)
            visualize_partial_derivatives_between_external(partial_derivatives, self.name, 'likelihood_slots', 'likelihood_free', first_input_shape=(6,3), second_input_shape=(6,))
            visualize_partial_derivatives_between_external(partial_derivatives, self.name, 'likelihood_free', 'likelihood_slots', first_input_shape=(6,), second_input_shape=(6,3))
            visualize_partial_derivatives_between_external(partial_derivatives, self.name, 'likelihood_slots', 'likelihood_movable', first_input_shape=(6,3), second_input_shape=(6,))
            visualize_partial_derivatives_between_external(partial_derivatives, self.name, 'likelihood_movable', 'likelihood_slots', first_input_shape=(6,), second_input_shape=(6,3))
            try:
                if self.latest_goal_derivatives is not None:
                    goal_leaves = [
                        (k, t)
                        for k, t in self.latest_goal_derivatives.items(include_nested=True, leaves_only=True)
                        if isinstance(t, torch.Tensor) and t.numel() > 0
                    ]
                    if len(goal_leaves) > 0:
                        for goal_key, goal_tensor in goal_leaves:
                            if isinstance(goal_key, tuple) and len(goal_key) >= 2 and str(goal_key[1]) == 'likelihood_slots':
                                chain_pairs = [('likelihood_slots', 'likelihood_free')]
                                try:
                                    chain_tensors, _ = collect_partial_derivative_chain_by_pairs(
                                        partial_derivatives,
                                        chain_pairs,
                                    )
                                    visualize_chain_derivative_product_external(
                                        goal_derivative=goal_tensor,
                                        partial_derivative_chain=chain_tensors,
                                        component_name=f"{self.name} - Chain Product",
                                        keep_batch_dims=1,
                                        compact=True,
                                        annotate=True,
                                        show_intermediate=True,
                                    )
                                except KeyError:
                                    pass
            except Exception:
                print("Could not visualize chain-rule derivative product:")
                traceback.print_exc() """
            self.wait_viz_counter = 0
        else:
            self.wait_viz_counter += 1
        
            #print({k: v for k, v in partial_derivatives.items(include_nested=True, leaves_only=True)})
        """ print("To Derivatives:")
        print(backward_derivatives) """
        with self.lock:
            self.quantities = self.quantities.update(output_dict)
            self.partial_derivatives = self.partial_derivatives.update(partial_derivatives)
            self.derivatives_forward_mode = self.derivatives_forward_mode.update(forward_derivatives)
            self.derivatives_backward_mode = self.derivatives_backward_mode.update(backward_derivatives)
            self.finished_forward_derivatives = self.finished_forward_derivatives.update(
                finished_forward_derivatives)

    def derivative_processing(self, all_inputs, input_names_diff, new_time, output_dict, partial_jacs,
                              relevant_backward_derivatives, relevant_forward_derivatives, required_signs_dict , flipped_signs_dict):
        if self.no_differentiation:
            return DerivativeDict(), DerivativeDict(), DerivativeDict(), DerivativeDict()
        in_out_shapes = collect_shapes(all_inputs, output_dict)
        partial_derivatives = gather_partial_derivatives(partial_jacs, input_names_diff, self.f_output_signature_diff,
                                                         in_out_shapes)
        goal_derivatives, self.goal_values = evaluate_goals(self.goals.values(), output_dict)
        self.latest_goal_derivatives = goal_derivatives
        #if self.wait_viz_counter > 5:
        #    visualize_goal_derivatives(goal_derivatives, self.name + " - Goal Derivatives")
        for k, v in self.goal_values.items():
            if self.goals[k].fulfilled_value is not None:
                if v <= self.goals[k].fulfilled_value:
                    print("Finished Goal "+str(k))
                    self.goals[k].is_active = False
        forward_derivatives, backward_derivatives, finished_forward_derivatives = compute_derivatives(
            partial_derivatives, relevant_forward_derivatives, relevant_backward_derivatives, goal_derivatives,
            input_names_diff, self.f_output_signature_diff, new_time, required_signs_dict, flipped_signs_dict,
            prevent_repeats=self.prevent_loops_in_differentiation, max_length_trace=self.max_length_differentiation_trace)
        return backward_derivatives, finished_forward_derivatives, forward_derivatives, partial_derivatives

    def inference_and_differentiation(self, args, diff_argnums):
        if self.no_differentiation:
            _, outputs = self.f_func(*args)
            partial_jacs = tuple()
        else:
            partial_jacs, outputs = torch.func.jacfwd(self.f_func, argnums=diff_argnums, has_aux=True, randomness="same")(
                *args)
        return outputs, partial_jacs