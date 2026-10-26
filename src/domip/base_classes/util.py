from __future__ import annotations

from typing import Tuple, List, Dict, Iterable

import torch
from tensordict import TensorDict

from domip.base_classes.derivatives import DerivativeDict


def collect_args(all_inputs, f_signature, diff_argnums, f_output_signature_diff) -> Tuple[List[torch.Tensor], Tuple[int], List[str]]:
    inputs = all_inputs.select(*f_signature).values()
    if diff_argnums is None:
        diff_argnums = tuple([i for i, arg in enumerate(f_signature) if arg not in [*f_output_signature_diff, "dt"]])
    input_names_diff = [f_signature[idx] for idx in diff_argnums]
    return inputs, diff_argnums, input_names_diff


def collect_inputs(quantities, connections, dt) -> TensorDict:
    collected_inputs = quantities.clone()
    for c in connections.values():
        with c.lock:
            collected_inputs = collected_inputs.update(c.connected_quantities)
    collected_inputs["dt"] = dt
    return collected_inputs


def collect_derivatives(connections) -> Tuple[DerivativeDict, DerivativeDict]:
    collected_forward = DerivativeDict()
    collected_backward = DerivativeDict()
    for c in connections.values():
        with c.lock:
            collected_forward = collected_forward.update(c.derivatives_forward_mode)
            collected_backward = collected_backward.update(c.derivatives_backward_mode)
    return collected_forward, collected_backward


def gather_partial_derivatives(partial_jacs:Dict[int, Dict[int, torch.Tensor]], input_names_diff:List[str], output_names_diff:List[str], in_and_out_shapes: Dict[str, torch.Size]) -> TensorDict:
    partial_derivatives = TensorDict({}, batch_size=())
    for o_idx, o in enumerate(output_names_diff):
        for i_idx, i in enumerate(input_names_diff):
            if len(output_names_diff) > 1 or sum(in_and_out_shapes[output_names_diff[0]]) > 1:
                # gets packed up as dict output -> dict input from torch.jacfwd/torch.jacrev
                partial_jac = partial_jacs[o_idx][i_idx]
            else:
                # gets packed up as single dict from torch.jacfwd/torch.jacrev
                partial_jac = partial_jacs[i_idx]
            # depending on exact specification, torch can have issues for scalar values
            # this is error handling for these
            if in_and_out_shapes[i] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(0)
            if in_and_out_shapes[o] == partial_jac.shape:
                partial_jac = partial_jac.unsqueeze(-1)
            if sum(partial_jac.shape) == 0:
                partial_jac = partial_jac.view(1, 1)
            partial_derivatives[o, i] = partial_jac
    return partial_derivatives


def generate_outputdict(outputs: Iterable[torch.Tensor], output_names: Iterable[str]) -> TensorDict:
    return TensorDict({n: t for n, t in zip(output_names, outputs)}, batch_size=())


def collect_shapes(all_inputs, output_dict):
    in_out_shapes = dict()
    for k, v in output_dict.items():
        in_out_shapes[k] = v.shape
    for k, v in all_inputs.items():
        in_out_shapes[k] = v.shape
    return in_out_shapes


def collect_required_signs(connections : Dict[str, Connection]) -> Dict[str, Dict[str, int]]:
    required_signs_dict : Dict[str, Dict[str, int]] = dict()
    for c in connections.values():
        for k, v in c.required_signs_dict.items():
            if k in required_signs_dict:
                required_signs_dict[k].update(v)
            else:
                required_signs_dict[k] = v
    return required_signs_dict

def collect_flipped_signs(connections : Dict[str, Connection]) -> Dict[str, Dict[str, int]]:
    flipped_signs_dict : Dict[str, Dict[str, int]] = dict()
    for c in connections.values():
        for k, v in c.flipped_signs_dict.items():
            if k in flipped_signs_dict:
                flipped_signs_dict[k].update(v)
            else:
                flipped_signs_dict[k] = v
    return flipped_signs_dict
