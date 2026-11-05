# This file contains code written by AI.

from typing import Dict, Union, Tuple

import torch

#from domip.blocksworld_experiment.sensors import BlocksBelowSensor
from domip.base_classes.components import SensorComponent
from domip.base_classes.connections import Connection
from domip.tower_of_london_move_gridsearch.global_params import NUM_BLOCKS

# Slot mapping for Tower of London with 6 discrete positions:
# Peg 1 (height 1): Slot 0
# Peg 2 (height 2): Slots 1-2 (heights 0-1)
# Peg 3 (height 3): Slots 3-5 (heights 0-2)
# Total: 6 slots

SLOT_TO_PEG_HEIGHT = {
    0: (0, 0),
    1: (1, 0),
    2: (1, 1),
    3: (2, 0),
    4: (2, 1),
    5: (2, 2),
}

PEG_HEIGHT_TO_SLOT = {v: k for k, v in SLOT_TO_PEG_HEIGHT.items()}

# Color encoding: 0=empty, 1=RED, 2=BLUE, 3=YELLOW
COLOR_TO_NUMBER = {
    "B": 1,
    "Y": 2,
    "R": 3,
    0: 0,
}

NUMBER_TO_COLOR = {v: k for k, v in COLOR_TO_NUMBER.items()}


# representation where a block is only ON the block directly below it, not on any under that
def block_to_direct_representation(block):
    initial_state = torch.zeros((NUM_BLOCKS, NUM_BLOCKS))
    color_to_number = {
        "B": 3,
        "Y": 4,
        "R": 5
    }
    for i in range(0, 3):
        for j in range(0, 3):
            disk = block[i][j]
            if disk == 0: continue
            if i+1 <= 2:
                under_disk = color_to_number[block[i+1][j]]
            else:
                under_disk = j
            initial_state[color_to_number[disk], under_disk] = 1.0
    return initial_state

def slot_to_peg_height(slot_idx: int) -> Tuple[int, int]:
    """Convert slot index (0-5) to (peg, height)."""
    return SLOT_TO_PEG_HEIGHT[slot_idx]


def peg_height_to_slot(peg: int, height: int) -> int:
    return PEG_HEIGHT_TO_SLOT[(peg, height)]


def state_to_one_hot(state_int: torch.Tensor, num_colors: int = 4, dtype=None, device=None) -> torch.Tensor:
    """Convert integer slot state to float one-hot encoding.
    
    Args:
        state_int: (6,) integer tensor with color indices (0=empty, 1=RED, 2=BLUE, 3=YELLOW)
        num_colors: Number of color classes (default 4)
        dtype: Desired output dtype (default torch.float32)
        device: Desired output device (default same as input)
    
    Returns:
        state_float: (6, 4) float tensor with one-hot encoding, shape (NUM_SLOTS, NUM_COLORS)
    """
    if device is None:
        device = state_int.device
    if dtype is None:
        dtype = torch.float32
    
    # Convert to one-hot using torch.nn.functional.one_hot
    state_float = torch.nn.functional.one_hot(
        state_int.long(), num_classes=num_colors
    ).float().to(dtype=dtype, device=device)
    
    return state_float


def block_config_to_slots(block):
    """Convert Tower of London block configuration (3x3 grid) to 6-slot representation.
    
    Args:
        block: 3x3 grid where block[i][j] is a color string ("R", "B", "Y") or 0
               Indexing: i=row (0=top, 2=bottom), j=column (peg index)
    
    Returns:
        slots: torch.Tensor of shape (6,) where each element is the color occupying that slot,
               or 0 if empty. Color encoding: 1=RED, 2=BLUE, 3=YELLOW, 0=empty
    """
    slots = torch.zeros(6, dtype=torch.long)
    
    # Iterate through pegs (columns)
    for peg_idx in range(3):
        # Iterate through heights (rows, from bottom to top since block[0] is top)
        for height in range(3):
            row_idx = 2 - height  # Convert height to row index (bottom-up to top-down)
            disk_color = block[row_idx][peg_idx]
            
            if disk_color != 0 and disk_color in COLOR_TO_NUMBER:
                slot_idx = peg_height_to_slot(peg_idx, height)
                slots[slot_idx] = COLOR_TO_NUMBER[disk_color]
    
    return slots


def generate_all_blocks():
    """Generate all 36 initial block configurations for Tower of London."""
    blocks = {}
    positioning = [[[0, 0, 2], [0, 0, 3], [0, 0, 1]],
                   [[0, 0, 0], [0, 0, 3], [2, 0, 1]],
                   [[0, 0, 0], [0, 0, 3], [0, 2, 1]],
                   [[0, 0, 0], [0, 3, 0], [0, 2, 1]],
                   [[0, 0, 0], [0, 0, 0], [3, 2, 1]],
                   [[0, 0, 0], [0, 1, 0], [3, 2, 0]]]
    colors = [["R", "B", "Y"],
              ["R", "Y", "B"],
              ["B", "Y", "R"],
              ["B", "R", "Y"],
              ["Y", "R", "B"],
              ["Y", "B", "R"]]
    for i in range(0, 6):
        for j in range(0, 6):
            name = "Pos" + str(i + 1) + str(j + 1) + "Init"
            color = colors[i]
            position = positioning[j]
            block = [[color[0] if item == 1 else color[1] if item == 2 else color[2] if item == 3 else item for item
                    in sublist] for sublist in position]
            blocks[name] = block
    return blocks


def generate_all_initial_states_slots():
    """Generate all 36 initial states in the 6-slot tensor format."""
    blocks = generate_all_blocks()
    slots_dict = {k: block_config_to_slots(v) for k, v in blocks.items()}
    return slots_dict


class DisksBelowSensor(SensorComponent):
    """Sensor for Tower of London game using 6-slot positional representation.
    
    Outputs state as a (6,) integer tensor where each element is the color occupying that slot:
    - 0: empty
    - 1: RED
    - 2: BLUE
    - 3: YELLOW
    
    Slots are indexed as: [Peg1-H0, Peg2-H0, Peg2-H1, Peg3-H0, Peg3-H1, Peg3-H2]
    """
    
    def __init__(self, name: str, connections: Dict[str, Connection],
                 dtype: Union[torch.dtype, None] = None, device: Union[torch.device, None] = None,
                 mockbuild: bool = False, setup: str = "Pos23Init", ROS_MODE: bool = False):
        self.quantity_name = "below_state_sensed"
        self.num_blocks = 6  # 6 discrete positions
        self.num_pegs = 3
        self.max_height = 3
        self.num_colors = 4  # 0=empty, 1=RED, 2=BLUE, 3=YELLOW
        
        # Call parent init (will set self.dtype, self.device, etc.)
        super().__init__(name, connections, dtype, device, mockbuild)

        # Generate all initial states in the new slot format
        self.all_initial_states_slots = generate_all_initial_states_slots()
        
        # Initialize from the setup configuration
        if setup in self.all_initial_states_slots:
            initial_slots = self.all_initial_states_slots[setup].clone().detach()
        else:
            # Fallback: empty state
            initial_slots = torch.zeros(6, dtype=torch.long)
        
        # Store integer state for internal logic
        self.initial_state_int = torch.tensor(initial_slots, dtype=torch.long, device=device)
        self.current_sim_state_int = self.initial_state_int.clone()
        
        # Convert to float one-hot for sensor output (what connections expect)
        self.initial_state = state_to_one_hot(self.initial_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)
        
        # Store float one-hot as the sensor quantity
        self.quantities[self.quantity_name] = self.initial_state
        #print(f"Initialized sensor with state (float one-hot):\n{self.quantities[self.quantity_name]}")
        
        self.current_sim_state = self.initial_state
        self.sim_timestamp = self.timestamp
        self.taken_valid_actions = 0
        
        if mockbuild:
            return

    def is_slot_occupied(self, slot_idx: int) -> bool:
        """Check if a slot is occupied (non-zero)."""
        return self.current_sim_state_int[slot_idx].item() != 0

    def is_slot_empty(self, slot_idx: int) -> bool:
        """Check if a slot is empty (zero)."""
        return self.current_sim_state_int[slot_idx].item() == 0

    def get_disk_above(self, slot_idx: int) -> int:
        """Get the slot index of the disk directly above (if any), or -1."""
        peg, height = slot_to_peg_height(slot_idx)
        try:  # Can potentially have something above
            slot_above = peg_height_to_slot(peg, height + 1)
            if not self.is_slot_empty(slot_above):
                return slot_above
        except KeyError:
            return -1
        return -1

    def get_disk_below(self, slot_idx: int) -> int:
        """Get the slot index of the disk directly below (if any), or -1."""
        peg, height = slot_to_peg_height(slot_idx)
        if height > 0:  # Can have something below
            slot_below = peg_height_to_slot(peg, height - 1)
            if not self.is_slot_empty(slot_below):
                return slot_below
        return -1

    def can_move_from_slot(self, slot_idx: int) -> bool:
        """Check if a disk can be moved from this slot (must be occupied and nothing on top)."""
        if not self.is_slot_occupied(slot_idx):
            return False
        if self.get_disk_above(slot_idx) != -1:
            return False
        return True

    def can_move_to_slot(self, slot_idx: int) -> bool:
        """Check if a disk can be moved to this slot (must be empty and respect height rules)."""
        if not self.is_slot_empty(slot_idx):
            return False
        peg, height = slot_to_peg_height(slot_idx)
        # Can only place on empty slot, or slot must be the lowest available on its peg
        if height > 0:
            slot_below = peg_height_to_slot(peg, height - 1)
            if self.is_slot_empty(slot_below):
                return False
        return True
    
    def obtain_measurements(self) -> bool:
        self.timestamp = torch.tensor(self.timestamp, dtype=self.dtype, device=self.device)
        self.quantities[self.quantity_name] = self.current_sim_state
        return True

    def take_action(self, action_value):
        """Apply a move action to the current state.
        
        Args:
            action_value: Tensor of shape (6, 6) where action_value[i, j] = 1 means move disk from slot i to slot j.
                         Only one action should be active (sum should be <= 1).
        
        Returns:
            Updated state tensor as float one-hot encoding (6, 4).
        """
        if torch.max(action_value) != 1:
            # No valid action
            #print("No valid action taken.")
            return state_to_one_hot(self.current_sim_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)

        # Extract the single move (from_slot, to_slot)
        indices = torch.where(action_value == 1)
        if len(indices[0]) == 0:
            return state_to_one_hot(self.current_sim_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)

        from_slot = indices[0][0].item()
        to_slot = indices[1][0].item()

        # Validate move
        if not self.can_move_from_slot(from_slot):
            #print(f"Invalid move: cannot move from slot {from_slot} (not occupied or blocked)")
            return state_to_one_hot(self.current_sim_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)

        if not self.can_move_to_slot(to_slot):
            #
            #print(f"Invalid move: cannot move to slot {to_slot} (occupied or violates stacking rules)")
            return state_to_one_hot(self.current_sim_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)

        # Execute move (work with integer state)
        disk_color = self.current_sim_state_int[from_slot].item()
        self.current_sim_state_int[from_slot] = 0
        self.current_sim_state_int[to_slot] = disk_color
        self.taken_valid_actions += 1
        
        #print(f"Move {from_slot} → {to_slot} (disk color {disk_color}). Total valid actions: {self.taken_valid_actions}")
        #print(f"Current state (int): {self.current_sim_state_int}")
        
        # Return float one-hot representation
        state_float = state_to_one_hot(self.current_sim_state_int, num_colors=self.num_colors, dtype=self.dtype, device=self.device)
        self.quantities[self.quantity_name] = state_float.clone()  # Update sensor output
        #print(f"Updated sensor state (float one-hot):\n{self.quantities[self.quantity_name]}")
        self.current_sim_state = state_float
        return self.current_sim_state
    
    def initial_definitions(self) -> None:
        self.quantities[self.quantity_name] = torch.empty(
            (self.num_blocks, self.num_blocks), dtype=self.dtype, device=self.device)
