import random
from typing import Dict, Callable, Union

import torch

from domip.base_classes.connections import Connection
from domip.base_classes.util import collect_derivatives
from domip.base_classes.components import ActionComponent, visualize_component_derivatives_external
from domip.tower_of_london_move_gridsearch.sensors import SLOT_TO_PEG_HEIGHT, peg_height_to_slot
from domip.tower_of_london_move_gridsearch.connections import compute_slot_below_occupancy, compute_slot_clearance
from domip.tower_of_london_move_gridsearch.global_params import (
    REQUIRE_POSITIVE_GRADIENT_THRESHOLD,
    POSITIVE_GRADIENT_THRESHOLD,
    REQUIRE_ANY_POSITIVE_NONZERO_GRADIENT,
    ALREADY_TRIED_THRESHOLD,
)

# Constants
NUM_SLOTS = 6


class MoveAction(ActionComponent):
    """Unified move action for Tower of London with 6-slot representation.
    
    Action format: (6, 6) matrix where action[i, j] = 1 means move disk from slot i to slot j.
    
    Uses gradient-based action selection: selects the move with the steepest gradient
    """

    def __init__(self, name: str, connections: Dict[str, Connection], goals: Union[None, Dict[str, Callable]] = None,
                 dtype: torch.dtype = torch.get_default_dtype(), device: torch.device = torch.get_default_device(),
                 mockbuild: bool = False, send_action_func: Union[None, Dict[str, Callable]] = None):
        self.num_slots = NUM_SLOTS
        super().__init__(name, connections, goals=goals, dtype=dtype, device=device, mockbuild=mockbuild)
        
        if mockbuild:
            return

        self.internal_action = None
        self.send_actions = False
        self.wait_for_action_counter = 0
        self.tried_actions = dict()  # Maps state hash to tried moves
        self._send_action_func = send_action_func

    def _start(self):
        self.send_actions = (self._send_action_func is not None)

    def _stop(self):
        self.send_actions = False

    def send_action_values(self) -> None:
        if self.internal_action is not None and self.send_actions:
            self._send_action_func(self.internal_action)

    def determine_new_actions(self):
        try:
            _, relevant_backward_derivatives = collect_derivatives(self.connections)
            my_derivatives = relevant_backward_derivatives["action_move"]
            #visualize_backward_derivatives(relevant_backward_derivatives)
            
        except KeyError:
            print("No Derivatives for Actions yet!")
            return
        
        # Wait a bit before selecting new action (allow gradient accumulation)
        if self.wait_for_action_counter > 5:
            # Check if gradients are fresh enough
            #visualize_component_derivatives_external(relevant_backward_derivatives, self.name)
            oldest_stamp_for_each_grad = torch.cat(
                [torch.min(torch.cat(stamps)).unsqueeze(0) for stamps in my_derivatives.timestamps])
            age_of_newest_grad = torch.tensor(self.timestamp, dtype=self.dtype, device=self.device) - torch.max(
                oldest_stamp_for_each_grad)
            if age_of_newest_grad > 0.2:
                return
            
            # Get gradient steepness (absolute value for finding max gradient)
            gradient_steepness = my_derivatives.derivatives_tensor  # (NUM_SLOTS, NUM_SLOTS)

            # Read current sensed state and build a stable key for tried-actions
            with self.connections["SlotObservation"].lock:
                slots = self.connections["SlotObservation"].connected_quantities["below_state_sensed"]
                clear = compute_slot_clearance(slots, self.device, self.dtype)
                below_valid = compute_slot_below_occupancy(slots, self.device, self.dtype)
                already_tried_counts = torch.zeros((NUM_SLOTS, NUM_SLOTS), dtype=self.dtype, device=self.device)
                matched_state_key = None
                for k, v in self.tried_actions.items():
                    if torch.equal(k, slots):
                        already_tried_counts = v
                        matched_state_key = k
                        break
            # Penalize proportionally: move tried n times gets multiplied by threshold**n
            gradient_steepness *= torch.pow(
                torch.tensor(ALREADY_TRIED_THRESHOLD, dtype=self.dtype, device=self.device),
                already_tried_counts,
            )
            chosen_gradient_idx = None
            chosen_move = None
            while (
                torch.sum(torch.abs(gradient_steepness)) > 0
                and (
                    not REQUIRE_ANY_POSITIVE_NONZERO_GRADIENT
                    or torch.any(gradient_steepness > 0)
                )
                and (
                    not REQUIRE_POSITIVE_GRADIENT_THRESHOLD
                    or torch.any(gradient_steepness > POSITIVE_GRADIENT_THRESHOLD)
                )
            ):
                #visualize_backward_derivatives(relevant_backward_derivatives)
                possible_steepest_gradient_idxs = (gradient_steepness == torch.max(gradient_steepness)).nonzero()
                n_idxs = possible_steepest_gradient_idxs.shape[0]
                chosen_gradient_idx = possible_steepest_gradient_idxs[random.randint(0, n_idxs - 1)]
                action_idx = chosen_gradient_idx[2:4]
                #print(f"Considered Action {action_idx}, \n clear_start={clear[action_idx[0]]}, \n clear_goal={clear[action_idx[1]]}, \n below_valid_goal={below_valid[action_idx[1]]}, \n gradient={gradient_steepness[*chosen_gradient_idx]}")
                
                # Find position with highest gradient
                from_slot = action_idx[0].item()
                to_slot = action_idx[1].item()
                # Inline validity checks (no external helper)
                # 1) source occupied
                if slots is not None:
                    source_occupied = (slots[from_slot, 0] < 0.5)

                # 2) source clear
                source_clear = (clear[from_slot] > 0.5)

                # 3) target empty
                if slots is not None:
                    target_empty = (slots[to_slot, 0] > 0.5)

                # 4) target below valid
                target_below_valid = (below_valid[to_slot] > 0.5)

                same_pole_move_mask = [[0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
                
                torch_same_pole_move_mask = torch.tensor(same_pole_move_mask, dtype=self.dtype, device=self.device)

                same_pole_move = torch_same_pole_move_mask[from_slot, to_slot] == 0.0

                if source_occupied and source_clear and target_empty and target_below_valid and not same_pole_move:
                    #print(f"Chosen move: {from_slot} → {to_slot}")
                    #print(f"Parameters: occupied={source_occupied}, clear={source_clear}, empty={target_empty}, below_valid={target_below_valid}")
                    chosen_move = (from_slot, to_slot)
                    #visualize_component_derivatives_external(relevant_backward_derivatives, self.name)
                    break
                #else:
                    #print(f"Invalid move: {from_slot} → {to_slot} (occupied={source_occupied}, clear={source_clear}, empty={target_empty}, below_valid={target_below_valid})")
                    #visualize_component_derivatives_external(relevant_backward_derivatives, self.name)

                # Zero out this move and try the next highest gradient
                gradient_steepness[*chosen_gradient_idx] = 0.0
            
            if chosen_gradient_idx is None or chosen_move is None or torch.sum(gradient_steepness) == 0:
                #print("No valid moves available!")
                self.internal_action = None
                return
            
            # Create action tensor
            from_slot, to_slot = chosen_move
            self.internal_action = torch.zeros((NUM_SLOTS, NUM_SLOTS), dtype=self.dtype, device=self.device)
            self.internal_action[from_slot, to_slot] = 1.0
            updated_counts = already_tried_counts.clone()
            updated_counts[from_slot, to_slot] += 1.0
            state_key = matched_state_key if matched_state_key is not None else slots.clone()
            self.tried_actions[state_key] = updated_counts
            
            #print(f"Action: {self.internal_action}")
            
            self.wait_for_action_counter = 0
        else:
            self.wait_for_action_counter += 1
            self.internal_action = None

    def initial_definitions(self):
        """Initialize action quantity."""
        self.quantities["action_move"] = torch.zeros((NUM_SLOTS, NUM_SLOTS), dtype=self.dtype, device=self.device)

    def initialize_quantities(self) -> bool:
        """Initialize action to ones."""
        self.quantities["action_move"] = torch.ones((NUM_SLOTS, NUM_SLOTS), dtype=self.dtype, device=self.device)*1.0
        return True


def visualize_backward_derivatives(relevant_backward_derivatives):
    """
    Visualizes the backward derivatives for the 'action_move' quantity as a heatmap with annotated values.
    
    Args:
        relevant_backward_derivatives: DerivativeDict containing the backward derivatives.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if 'action_move' in relevant_backward_derivatives:
        block = relevant_backward_derivatives['action_move']
        tensor = block.derivatives_tensor
        
        if tensor is not None and tensor.numel() > 0:
            # Assuming the tensor is 2D (NUM_SLOTS x NUM_SLOTS)
            data = tensor.detach().cpu().numpy()
            real_data = data[0, 0, :, :]
            
            plt.figure(figsize=(10, 8))
            im = plt.imshow(real_data, cmap='coolwarm', aspect='equal', vmin=-np.max(np.abs(real_data)), vmax=np.max(np.abs(real_data)))
            plt.colorbar(im, label='Derivative Value')
            plt.title('Backward Derivatives for action_move')
            plt.xlabel('To Slot')
            plt.ylabel('From Slot')
            plt.xticks(range(real_data.shape[1]), range(real_data.shape[1]))
            plt.yticks(range(real_data.shape[0]), range(real_data.shape[0]))
            plt.grid(False)
            
            # Add text annotations
            for i in range(real_data.shape[0]):
                for j in range(real_data.shape[1]):
                    val = real_data[i, j]
                    # Choose text color for contrast: white for dark backgrounds, black for light
                    text_color = 'white' if abs(val) > np.max(np.abs(real_data)) / 2 else 'black'
                    plt.text(j, i, f'{val:.3f}', ha='center', va='center', color=text_color, fontsize=10, weight='bold')
            
            plt.tight_layout()
            plt.show()
        else:
            print("Derivatives tensor is None or empty")
    else:
        print("No 'action_move' found in relevant_backward_derivatives")

