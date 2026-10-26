import torch

from domip.base_classes.components import EstimationComponent
from domip.inference.util import gradient_preserving_clipping
from domip.tower_of_london_move_gridsearch.global_params import NUM_PEGS, NUM_BLOCKS
from domip.tower_of_london_move_gridsearch.sensors import SLOT_TO_PEG_HEIGHT, peg_height_to_slot
from domip.base_classes.util import collect_derivatives

# Constants
NUM_SLOTS = 6
NUM_COLORS = 3  # 0=Blue, 1=Yellow, 2=Red


class GlobalStateEstimator(EstimationComponent):
    def initialize_quantities(self):
        c = self.connections["SlotObservation"]
        with c.lock:
            if c.connected_quantities_initialized["below_state_sensed"]:
                below_state_sensed = c.connected_quantities["below_state_sensed"]
            else:
                return False
        self.quantities["likelihood_slots"] = below_state_sensed[:, 1:]
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["GlobalStateConnection"].c_func
        c_func_observation = self.connections["SlotObservation"].c_func

        def f_func(likelihood_slots, action_move, likelihood_movable, below_state_sensed, likelihood_free):
            # Update state from constraints
            innovation = c_func(likelihood_slots, action_move, likelihood_movable, likelihood_free)
            new_likelihood = gradient_preserving_clipping(likelihood_slots + innovation, 0.01, 0.99)
            #print(f"After applying constraint innovation:\n{new_likelihood}")
            #print(f"below_state_sensed (float one-hot) in f_func:\n{below_state_sensed.shape}, values:\n{below_state_sensed}")
            # Measurement update from sensor (already in float form)
            new_likelihood = new_likelihood + c_func_observation(new_likelihood, below_state_sensed).detach() # no grad from sensing
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.001, 0.999)

            """ with torch.no_grad():
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                self.visualize_component_derivatives(relevant_backward_derivatives, self.name) """

            return (new_likelihood,), (new_likelihood,)
        
        return f_func, ["likelihood_slots"], ["likelihood_slots"]

    def initial_definitions(self):
        self.quantities["likelihood_slots"] = torch.zeros((NUM_SLOTS, NUM_COLORS), dtype=self.dtype, device=self.device)


class MovableLikelihoodEstimator(EstimationComponent):

    def initialize_quantities(self):
        c = self.connections["MovableConstraint"]
        with c.lock:
            if c.connected_quantities_initialized["likelihood_slots"]:
                likelihood_slots = c.connected_quantities["likelihood_slots"]
                c_func = c.c_func
            else:
                return False
        
        self.quantities["likelihood_movable"] = torch.clip(
            c_func(torch.zeros(NUM_SLOTS, dtype=self.dtype, device=self.device), likelihood_slots), 0.01, 0.99)

        #print(f"Initialized likelihood_movable from constraint:\n{self.quantities['likelihood_movable']}")
        return True

    def define_estimation_function_f(self):
        c_func = self.connections["MovableConstraint"].c_func

        def f_func(likelihood_movable, likelihood_slots):
            # Get innovation from constraint connection
            innovation = c_func(likelihood_movable, likelihood_slots)
            
            # Update likelihood_movable by applying innovation
            new_likelihood = likelihood_movable + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.01, 0.99)

            """ with torch.no_grad():
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                self.visualize_component_derivatives(relevant_backward_derivatives, self.name) """

            
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_movable"], ["likelihood_movable"]

    def initial_definitions(self):
        self.quantities["likelihood_movable"] = torch.zeros(NUM_SLOTS, dtype=self.dtype, device=self.device)


class FreeLikelihoodEstimator(EstimationComponent):

    def initialize_quantities(self):
        """Initialize from FreeConstraint connection which reads global state."""
        c = self.connections["FreeConstraint"]
        with c.lock:
            if c.connected_quantities_initialized["likelihood_slots"]:
                likelihood_slots = c.connected_quantities["likelihood_slots"]
                c_func = c.c_func
            else:
                return False

        self.quantities["likelihood_free"] = torch.clip(
            c_func(torch.zeros(NUM_SLOTS, dtype=self.dtype, device=self.device), likelihood_slots), 0.01, 0.99)
        return True

    def define_estimation_function_f(self):

        c_func = self.connections["FreeConstraint"].c_func

        def f_func(likelihood_free, likelihood_slots):
            # Get innovation from constraint connection
            innovation = c_func(likelihood_free, likelihood_slots)
            
            # Update likelihood_free by applying innovation
            new_likelihood = likelihood_free + innovation
            new_likelihood = gradient_preserving_clipping(new_likelihood, 0.01, 0.99)

            """ with torch.no_grad():
                _, relevant_backward_derivatives = collect_derivatives(self.connections)
                self.visualize_component_derivatives(relevant_backward_derivatives, self.name) """
            
            return (new_likelihood,), (new_likelihood,)

        return f_func, ["likelihood_free"], ["likelihood_free"]

    def initial_definitions(self):
        self.quantities["likelihood_free"] = torch.zeros(NUM_SLOTS, dtype=self.dtype, device=self.device)

