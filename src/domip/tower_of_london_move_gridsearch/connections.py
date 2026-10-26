from typing import Union
from pathlib import Path

import numpy
import torch
import matplotlib.pyplot as plt
import datetime

from domip.base_classes.connections import Connection
from domip.tower_of_london_move_gridsearch.global_params import NUM_BLOCKS, NUM_PEGS
from domip.inference.util import gradient_preserving_clipping
from domip.tower_of_london_move_gridsearch.sensors import SLOT_TO_PEG_HEIGHT, peg_height_to_slot

# Constants matching estimators
NUM_SLOTS = 6
NUM_COLORS = 3  # 0=empty, 1=BlUE, 2=YELLOW, 3=RED


def build_connections(alpha: float = 1.0, beta: float = 1.0):
    connections = {
        "GlobalStateConnection": lambda device, dtype, mockbuild: GlobalStateConnection("GlobalStateConnection", device=device, dtype=dtype, mockbuild=mockbuild),
        "SlotObservation": lambda device, dtype, mockbuild: SlotObservation("SlotObservation", device=device, dtype=dtype, mockbuild=mockbuild),
        "FreeConstraint": lambda device, dtype, mockbuild: FreeConstraint("FreeConstraint", device=device, dtype=dtype, mockbuild=mockbuild, alpha=alpha),
        "MovableConstraint": lambda device, dtype, mockbuild: MovableConstraint("MovableConstraint", device=device, dtype=dtype, mockbuild=mockbuild, beta=beta),
    }
    return connections

class GlobalStateConnection(Connection):
    def __init__(self, name: str, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None,
                 mockbuild : bool = False):
        super().__init__(name, {
            "likelihood_slots": (NUM_SLOTS, NUM_COLORS),
            "action_move": (NUM_SLOTS, NUM_SLOTS),
            "likelihood_movable": (NUM_SLOTS,),
            "likelihood_free": (NUM_SLOTS,)
        }, dtype=dtype, device=device, mockbuild=mockbuild)
        #required_signs_dict={"likelihood_slots": {"action_move": -1}}
    
    def define_implicit_connection_function(self):
        def connection_func(likelihood_slots, action_move, likelihood_movable, likelihood_free):
            #likelihood_slots_hard = (likelihood_slots > 0.5).detach() * likelihood_slots

            self_mask = [[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                             [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
                
            self_mask = torch.tensor(self_mask, dtype=self.dtype, device=self.device)

            valid_move = torch.einsum("i, j -> ij", likelihood_movable, likelihood_free) * 0.95
            valid_move_masked = torch.einsum("ij, ij -> ij", valid_move, self_mask)
            valid_actions = torch.einsum("ij, ij -> ij", valid_move_masked, action_move)
            contributions = torch.einsum("ji, jk -> ijk", valid_actions, likelihood_slots)
            #valid_actions = torch.einsum("ij, jk -> ijk", valid_move, likelihood_slots_hard)
            #contributions = torch.einsum("ijk, ij -> ijk", valid_actions, action_move)
            #contributions = torch.einsum("ijk, ij -> ijk", contributions, self_mask)
            pos_contributions = torch.einsum("ijk -> ik", contributions)
            neg_contributions = torch.einsum("ijk -> jk", contributions)
            #pos_contributions_scaled = torch.where(pos_contributions > 0.5, pos_contributions / pos_contributions, pos_contributions)
            #neg_contributions_scaled = torch.where(neg_contributions > 0.5, neg_contributions / neg_contributions, neg_contributions)
            innovation = likelihood_slots - pos_contributions + neg_contributions
            return innovation
        
        return connection_func


class SlotObservation(Connection):
    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False):
        super().__init__(name, {
            "likelihood_slots": (NUM_SLOTS, NUM_COLORS),
            "below_state_sensed": (NUM_SLOTS, NUM_COLORS + 1), 
            "action_move": (NUM_SLOTS, NUM_SLOTS)
        }, dtype=dtype, device=device, mockbuild=mockbuild)

    def define_implicit_connection_function(self):

        def connection_func(likelihood_slots, below_state_sensed):
            innovation = below_state_sensed[:, 1:] - likelihood_slots
            return innovation

        return connection_func



class FreeConstraint(Connection):

    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, alpha: float = 1.0):
        super().__init__(name, {
            "likelihood_free": (NUM_SLOTS,),
            "likelihood_slots": (NUM_SLOTS, NUM_COLORS)
        }, dtype=dtype, device=device, mockbuild=mockbuild, flipped_signs_dict={"likelihood_free": {"likelihood_slots": 1}})
        self.alpha = alpha

    def define_implicit_connection_function(self):
        def connection_func(likelihood_free, likelihood_slots):
            alpha = self.alpha
            # SELECTORS
            # selector below
            selector_below = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]

            selector_below = torch.tensor(selector_below, dtype=self.dtype, device=self.device)
            
            has_slot_below = torch.einsum("jk -> j", selector_below)
            has_floor_below = 1.0 - has_slot_below

            # selector clear
            selector_clear = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

            selector_clear = torch.tensor(selector_clear, dtype=self.dtype, device=self.device)
            

            has_slot_above = torch.einsum("jk -> j", selector_clear)
            has_sky_above = 1.0 - has_slot_above

            # selector 2 clear
            selector_clear_2 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

            selector_clear_2 = torch.tensor(selector_clear_2, dtype=self.dtype, device=self.device)
            

            has_slot_2_above = torch.einsum("jk -> j", selector_clear_2)
            has_sky_2_above = 1.0 - has_slot_2_above




            likelihood_slots_hard = likelihood_slots * (likelihood_slots > 0.5).detach()



            occupancy = torch.einsum("jk -> j", likelihood_slots_hard) 
            clearance = 1.0 - occupancy
            #occupancy_hard = torch.einsum("jk -> j", likelihood_slots_hard)
            #occupancy_scaled = torch.where(occupancy > 0.5, occupancy / occupancy, occupancy)

            has_occupied_below = torch.einsum("jk, k -> j", selector_below, occupancy)
            has_stuff_below = has_floor_below + has_occupied_below

            has_clear_slot_above = torch.einsum("jk, k -> j", selector_clear, clearance)
            has_occupied_slot_above = torch.einsum("jk, k -> j", selector_clear, occupancy)

            has_clear_slot_2_above = torch.einsum("jk, k -> j", selector_clear_2, clearance)

            #has_clear_above = has_sky_above + has_clear_slot_above

            new_likelihood_free = clearance * has_stuff_below * has_clear_slot_above.detach() + clearance * has_stuff_below * has_sky_above.detach()
            new_likelihood_free += occupancy.detach() * has_occupied_slot_above.detach() * has_clear_slot_2_above
            new_likelihood_free -= occupancy.detach() * has_occupied_slot_above.detach()
            new_likelihood_free += occupancy.detach() * has_clear_slot_above * has_clear_slot_2_above.detach()
            new_likelihood_free -= occupancy.detach() * has_clear_slot_above.detach()
            new_likelihood_free += occupancy.detach() * has_clear_slot_above * has_sky_2_above.detach()
            new_likelihood_free -= occupancy.detach() * has_clear_slot_above.detach()
            #new_likelihood_free += has_stuff_below.detach() * clearance * has_clear_above
            #new_likelihood_free += has_stuff_below * clearance 
            innovation = new_likelihood_free - likelihood_free
            innovation = innovation * alpha
            return innovation

        return connection_func





class MovableConstraint(Connection):
    def __init__(self, name: str, dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, beta: float = 1.0):
        super().__init__(name, {
            "likelihood_movable": (NUM_SLOTS,),
            "likelihood_slots": (NUM_SLOTS, NUM_COLORS)
        }, dtype=dtype, device=device, mockbuild=mockbuild, flipped_signs_dict={"likelihood_movable": {"likelihood_slots": 1}})
        self.beta = beta
        #, required_signs_dict={"likelihood_movable": {"likelihood_slots": 1}}

    def define_implicit_connection_function(self):
        def connection_func(likelihood_movable, likelihood_slots):
            beta = self.beta
            # SELECTORS
            selector_clear = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

            selector_clear = torch.tensor(selector_clear, dtype=self.dtype, device=self.device)
            

            has_slot_above = torch.einsum("jk -> j", selector_clear)
            has_sky_above = 1.0 - has_slot_above

            # selector 2 clear
            selector_clear_2 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

            selector_clear_2 = torch.tensor(selector_clear_2, dtype=self.dtype, device=self.device)
            

            has_slot_2_above = torch.einsum("jk -> j", selector_clear_2)
            has_sky_2_above = 1.0 - has_slot_2_above

            likelihood_slots_hard = likelihood_slots * (likelihood_slots > 0.5).detach()

            occupancy = torch.einsum("jk -> j", likelihood_slots_hard)
            #occupancy_hard = torch.einsum("jk -> j", likelihood_slots_hard)
            #occupancy_scaled = torch.where(occupancy > 0.5, occupancy / occupancy, occupancy)
            clearance = 1.0 - occupancy
            has_clear_slot_above = torch.einsum("jk, k -> j", selector_clear, clearance)
            has_occupied_slot_above = torch.einsum("jk, k -> j", selector_clear, occupancy)

            has_clear_slot_2_above = torch.einsum("jk, k -> j", selector_clear_2, clearance)
            #print("Has clear slot above:")
            #print(has_clear_slot_above)

            #has_clear_above = has_sky_above + has_clear_slot_above

            new_likelihood_movable = has_sky_above * occupancy.detach() + has_clear_slot_above * occupancy.detach()
            new_likelihood_movable += has_occupied_slot_above.detach() * occupancy.detach() * has_clear_slot_2_above
            new_likelihood_movable -= has_occupied_slot_above.detach() * occupancy.detach() * has_clear_slot_2_above.detach()
            innovation = new_likelihood_movable - likelihood_movable
            innovation = innovation * beta
            return innovation

        return connection_func


def compute_slot_clearance(likelihood_slots, device, dtype):
    """
    Compute likelihood that each slot is clear (can be picked from).
    A slot is clear if there is nothing on top of it.
    Fully differentiable using vectorized matrix operations.
    
    Args:
        likelihood_slots: (NUM_SLOTS, NUM_COLORS) tensor with color probabilities
        device: torch device
        dtype: torch dtype
    
    Returns:
        clear: (NUM_SLOTS,) tensor with P(slot is clear/pickable)
    """
    # Create selector matrix: selector[i, j] = 1 if slot j is directly above slot i
    selector = torch.zeros(NUM_SLOTS, NUM_SLOTS, dtype=dtype, device=device)
    
    for slot_idx in range(NUM_SLOTS):
        peg, height = SLOT_TO_PEG_HEIGHT[slot_idx]
        try:
            slot_above = peg_height_to_slot(peg, height + 1)
            selector[slot_idx, slot_above] = 1.0
        except KeyError:
            # No slot above this one (slot is at max height for its peg)
            pass
    
    # Extract emptiness: P(slot is empty) = likelihood_slots[:, 0]
    empty_prob = likelihood_slots[:, 0]  # (NUM_SLOTS,)
    
    # Matrix multiplication: for each slot, get emptiness of slot above
    above_empty = torch.matmul(selector, empty_prob)  # (NUM_SLOTS,)
    
    # For slots with nothing above (no slot above), always clear (1.0)
    # For slots with another slot above, use emptiness of that slot
    has_slot_above = torch.sum(selector, dim=1)  # (NUM_SLOTS,)
    clear = above_empty * has_slot_above + (1.0 - has_slot_above)
    
    return clear


def compute_slot_below_occupancy(likelihood_slots, device, dtype):
    """
    Compute likelihood that each slot has something valid below it.
    Fully differentiable using vectorized matrix operations.
    
    Args:
        likelihood_slots: (NUM_SLOTS, NUM_COLORS) tensor with color probabilities
        device: torch device
        dtype: torch dtype
    
    Returns:
        below_valid: (NUM_SLOTS,) tensor with P(slot below is occupied)
    """
    # Create selector matrix: selector[i, j] = 1 if slot j is directly below slot i
    selector = torch.zeros(NUM_SLOTS, NUM_SLOTS, dtype=dtype, device=device)
    
    for slot_idx in range(NUM_SLOTS):
        peg, height = SLOT_TO_PEG_HEIGHT[slot_idx]
        if height > 0:  # Has something below
            slot_below = peg_height_to_slot(peg, height - 1)
            selector[slot_idx, slot_below] = 1.0
    
    # Extract occupancy: P(slot is occupied) = 1 - P(slot is empty)
    occupancy = 1.0 - likelihood_slots[:, 0]  # (NUM_SLOTS,)
    
    # Matrix multiplication: for each slot, get occupancy of slot below
    below_occupied = torch.matmul(selector, occupancy)  # (NUM_SLOTS,)
    
    # For slots with floor beneath (no slot below), always valid (1.0)
    # For slots with another slot below, use occupancy of that slot
    has_slot_below = torch.sum(selector, dim=1)  # (NUM_SLOTS,)
    below_valid = below_occupied * has_slot_below + (1.0 - has_slot_below)
    
    return below_valid

