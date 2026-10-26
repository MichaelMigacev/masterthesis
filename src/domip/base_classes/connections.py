from abc import abstractmethod
from threading import Lock
from typing import Callable, Tuple, Dict, Union

import torch
from tensordict import TensorDict

from domip.base_classes.derivatives import DerivativeDict
from domip.blocksworld_experiment.global_params import NUM_BLOCKS


class Connection:

    device: torch.device
    dtype: torch.dtype
    connected_quantities_initialized: dict[str, bool]
    connected_quantities_timestamps: TensorDict
    name: str
    lock: Lock
    connected_quantities: TensorDict
    derivatives_backward_mode: DerivativeDict
    derivatives_forward_mode: DerivativeDict
    c_func: Callable

    def __init__(self, name:str, quantity_names_and_shapes: Dict[str, Tuple[int]], dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False, required_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None, flipped_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None, num_blocks=NUM_BLOCKS):
        self.num_blocks = num_blocks
        self.name = name
        self.lock = Lock()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.connected_quantities = TensorDict(dict(), device=device, batch_size=())
        self.connected_quantities_timestamps = TensorDict(dict(), device=device, batch_size=())
        self.connected_quantities_initialized = dict()
        self.required_signs_dict = dict()
        self.flipped_signs_dict = dict()
        self.derivatives_forward_mode = DerivativeDict()
        self.derivatives_backward_mode = DerivativeDict()
        self.initial_definitions(quantity_names_and_shapes, required_signs_dict, flipped_signs_dict)
        if not mockbuild:
            self.c_func = torch.func.functionalize(self.define_implicit_connection_function())

    def initial_definitions(self, quantity_names_and_shapes: Dict[str, Tuple[int]], required_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None, flipped_signs_dict : Union[None, Dict[str, Dict[str, int]]] = None):
        if required_signs_dict is not None:
            self.required_signs_dict = required_signs_dict
        if flipped_signs_dict is not None:
            self.flipped_signs_dict = flipped_signs_dict
        for n, s in quantity_names_and_shapes.items():
            self.connected_quantities[n] = torch.zeros(s, dtype=self.dtype, device=self.device)
            self.connected_quantities_timestamps[n] = torch.zeros(1, dtype=self.dtype, device=self.device)
            self.connected_quantities_initialized[n] = False

    @abstractmethod
    def define_implicit_connection_function(self):
        pass