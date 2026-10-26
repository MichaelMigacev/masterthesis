from dataclasses import dataclass
from typing import List, Tuple
from copy import deepcopy


@dataclass
class State:
    """Represents a Tower of London state.
    
    Standard Tower of London puzzle:
    - Peg 0 (left): height 1
    - Peg 1 (middle): height 2
    - Peg 2 (right): height 3
    
    Beads are numbered 0 (smallest) to 2 (largest).
    Beads are stored as lists from bottom to top.
    """
    pegs: List[List[int]]  # List of pegs, each containing list of bead IDs
    peg_heights: List[int] = None  # Max height for each peg
    state_id: str = None  # Official problem ID as 
    
    def __post_init__(self):
        """Initialize peg heights if not provided."""
        if self.peg_heights is None:
            self.peg_heights = [1, 2, 3]  # Standard Tower of London
        if self.state_id is None:
            # Try to assign official ID if matches
            key = tuple(tuple(peg) for peg in self.pegs)
            self.state_id = OFFICIAL_STATES.get(key, "UNOFFICIAL")
            #print(f"Assigned State ID: {self.state_id} for key {key}")
    
    def __hash__(self):
        """Make state hashable for use in sets/dicts."""
        return hash(tuple(tuple(peg) for peg in self.pegs))
    
    def __eq__(self, other):
        """Compare states."""
        return isinstance(other, State) and self.pegs == other.pegs
    
    def is_valid_move(self, from_peg: int, to_peg: int) -> bool:
        """Check if move is legal.
        
        Tower of London rules:
        - All beads are the same size (IDs represent color only)
        - Can move any top bead to any non-full peg
        - Cannot move from empty peg
        - Cannot move to same peg
        """
        if from_peg == to_peg or from_peg < 0 or to_peg < 0:
            return False
        if from_peg >= len(self.pegs) or to_peg >= len(self.pegs):
            return False
        if not self.pegs[from_peg]:  # Source peg empty
            return False
        if len(self.pegs[to_peg]) >= self.peg_heights[to_peg]:  # Dest peg full
            return False
        
        # All beads are same size, so we can always move top bead to non-full peg
        return True
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal (from_peg, to_peg) moves."""
        moves = []
        for from_peg in range(len(self.pegs)):
            for to_peg in range(len(self.pegs)):
                if self.is_valid_move(from_peg, to_peg):
                    moves.append((from_peg, to_peg))
        return moves
    
    def apply_move(self, from_peg: int, to_peg: int) -> 'State':
        """Apply a move and return new state."""
        if not self.is_valid_move(from_peg, to_peg):
            raise ValueError(f"Invalid move: {from_peg} -> {to_peg}")

        new_state = State(deepcopy(self.pegs), self.peg_heights)
        
        bead = new_state.pegs[from_peg].pop()
        new_state.pegs[to_peg].append(bead)

        new_state.state_id = OFFICIAL_STATES.get(
        tuple(tuple(peg) for peg in new_state.pegs), 
        "UNOFFICIAL"
        )
        
        return new_state
    

OFFICIAL_STATES = {
    ((), (), (2, 1, 0)): "TOL-011",
    ((0, ), (), (2, 1)): "TOL-012",
    ((), (0, ), (2, 1)): "TOL-013",
    ((), (0, 1), (2, )): "TOL-014",
    ((1, ), (0, ), (2, )): "TOL-015",
    ((1, ), (0, 2, ), ()): "TOL-016",
    ((), (), (2, 0, 1)): "TOL-021",
    ((1, ), (), (2, 0)): "TOL-022",
    ((), (1, ), (2, 0)): "TOL-023",
    ((), (1, 0), (2, )): "TOL-024",
    ((0, ), (1, ), (2, )): "TOL-025",
    ((0, ), (1, 2, ), ()): "TOL-026",
    ((), (), (0, 2, 1)): "TOL-031",
    ((1, ), (), (0, 2)): "TOL-032",
    ((), (1, ), (0, 2)): "TOL-033",
    ((), (1, 2), (0, )): "TOL-034",
    ((2, ), (1, ), (0, )): "TOL-035",
    ((2, ), (1, 0, ), ()): "TOL-036",
    ((), (), (0, 1, 2)): "TOL-041",
    ((2, ), (), (0, 1)): "TOL-042",
    ((), (2, ), (0, 1)): "TOL-043",
    ((), (2, 1), (0, )): "TOL-044",
    ((1, ), (2, ), (0, )): "TOL-045",
    ((1, ), (2, 0, ), ()): "TOL-046",
    ((), (), (1, 0, 2)): "TOL-051",
    ((2, ), (), (1, 0)): "TOL-052",
    ((), (2, ), (1, 0)): "TOL-053",
    ((), (2, 0), (1, )): "TOL-054",
    ((0, ), (2, ), (1, )): "TOL-055",
    ((0, ), (2, 1, ), ()): "TOL-056",
    ((), (), (1, 2, 0)): "TOL-061",
    ((0, ), (), (1, 2)): "TOL-062",
    ((), (0, ), (1, 2)): "TOL-063",
    ((), (0, 2), (1, )): "TOL-064",
    ((2, ), (0, ), (1, )): "TOL-065",
    ((2, ), (0, 1, ), ()): "TOL-066",
}