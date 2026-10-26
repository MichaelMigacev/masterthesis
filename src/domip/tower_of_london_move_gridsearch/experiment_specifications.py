from typing import Callable, Type

import torch
import sys

from domip.base_classes.goals import Goal
from domip.tower_of_london_move_gridsearch.actions import MoveAction
from domip.tower_of_london_move_gridsearch.connections import build_connections
from domip.tower_of_london_move_gridsearch.estimators import GlobalStateEstimator, MovableLikelihoodEstimator, FreeLikelihoodEstimator
from domip.tower_of_london_move_gridsearch.global_params import NUM_BLOCKS, NUM_PEGS
from domip.tower_of_london_move_gridsearch.automatic_goal import *
from domip.tower_of_london_move_gridsearch.sensors import DisksBelowSensor


def get_building_functions_basic_blocks_world(
    init_setup: str = "Pos23Init",
    goal: str = "Pos24Goal",
    alpha: float = 1.0,
    beta: float = 1.0,
):
    connection_builders = build_connections(alpha=alpha, beta=beta)
    goal_class = getattr(sys.modules[__name__], goal)

    diff_depth = 6
    component_builders = {
        "DisksBelowSensor": lambda mockbuild: DisksBelowSensor(
            "DisksBelowSensor",
            connections={k: connection_builders[k] for k in ("SlotObservation",)},
            device=torch.device("cpu"),
            dtype=torch.double,
            mockbuild=mockbuild,
            setup=init_setup
        ),
        "GlobalStateEstimator": lambda mockbuild: GlobalStateEstimator(
            "GlobalStateEstimator",
            connections={k: connection_builders[k] for k in ("SlotObservation", "FreeConstraint", "MovableConstraint", "GlobalStateConnection")},
            goals={goal: lambda d, t, mockbuild=False: goal_class(is_active=True, dtype=t, device=d,mockbuild=mockbuild)}, 
            device=torch.device("cpu"), 
            dtype=torch.double, 
            mockbuild=mockbuild,
            prevent_loops_in_differentiation=True,
            max_length_differentiation_trace=diff_depth, 
            num_blocks=NUM_BLOCKS),
        "MovableLikelihoodEstimator": lambda mockbuild: MovableLikelihoodEstimator(
            "MovableLikelihoodEstimator",
            connections={k: connection_builders[k] for k in("MovableConstraint", "GlobalStateConnection")},
            device=torch.device("cpu"), dtype=torch.double,
            mockbuild=mockbuild, 
            prevent_loops_in_differentiation=True,
            max_length_differentiation_trace=diff_depth, 
            num_blocks=NUM_BLOCKS),
        "MoveAction": lambda mockbuild: MoveAction(
            "MoveAction",
            connections={k: connection_builders[k] for k in("GlobalStateConnection", "SlotObservation")},
            device=torch.device("cpu"), 
            dtype=torch.double,
            mockbuild=mockbuild),
        "FreeLikelihoodEstimator": lambda mockbuild: FreeLikelihoodEstimator(
            "FreeLikelihoodEstimator",
            connections={k: connection_builders[k] for k in("FreeConstraint", "GlobalStateConnection")},
            device=torch.device("cpu"), dtype=torch.double,
            mockbuild=mockbuild, 
            prevent_loops_in_differentiation=True,
            max_length_differentiation_trace=diff_depth,
            num_blocks=NUM_BLOCKS),
    }

    frame_rates = {"DisksBelowSensor": 10, "GlobalStateEstimator": 10, "MovableLikelihoodEstimator": 10, "MoveAction": 10, "FreeLikelihoodEstimator": 10,}
    return component_builders, connection_builders, frame_rates
