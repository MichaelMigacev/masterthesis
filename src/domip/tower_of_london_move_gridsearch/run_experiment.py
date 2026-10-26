import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import os
import datetime

from domip.tower_of_london_move_gridsearch.experiment_specifications import get_building_functions_basic_blocks_world
from domip.middleware.python_sequential import build_components, run_component_sequence
from domip.tower_of_london_move_gridsearch.sensors import state_to_one_hot
from domip.tower_of_london_move_gridsearch.setup_lists import *

from domip.tower_of_london_move_gridsearch.global_params import MAKE_CORRELATION_ANALYSIS, MAKE_GRAPHS, MAKE_CSV

GRIDSEARCH_SEED = 111

""" def map_to_string(x):
    mapping = {
        0: "0",
        1: "1",
        2: "2",
        3: "R",
        4: "B",
        5: "Y",
    }
    return mapping[x] """

def convert_matrix_to_string_indexes(matrix):
    index = torch.nonzero(matrix)[0]
    from_slot = index[0].item()
    to_slot = index[1].item()
    return [str(from_slot), str(to_slot)]

def build_sensor_state_lookup(sensor):
    lookup = {}
    for key, slots_state in sensor.all_initial_states_slots.items():
        state = state_to_one_hot(
            slots_state.clone().detach(),
            num_colors=sensor.num_colors,
            dtype=sensor.dtype,
            device=sensor.device,
        )
        lookup[int(key[3:5])] = state.clone().detach()
    return lookup

def get_number_from_sensor_state(state, sensor_state_lookup):
    for state_number, known_state in sensor_state_lookup.items():
        if torch.equal(known_state, state):
            return state_number
    return -1

def get_sequence_from_sensor_states(states, sensor_state_lookup):
    sequence = []
    for _, state in states:
        sequence.append(get_number_from_sensor_state(state, sensor_state_lookup))
    return sequence

def start_all_components(functions_constructor : Callable):
    component_builders, connection_builders, frame_rates = functions_constructor()
    return build_components(component_builders)


def run_experiment(functions_constructor : Callable):
    components = start_all_components(functions_constructor=functions_constructor)
    sensor_state_lookup = build_sensor_state_lookup(components["DisksBelowSensor"])

    def determine_goals_fulfilled(cs):
        for _, comp in cs.items():
            for _, goal in comp.goals.items():
                if goal.is_active:
                    return False
        return True
    sim_states = []
    sim_states_list = []
    num_sim_states = 1
    sim_states.append(("0", torch.tensor(components["DisksBelowSensor"].initial_state.clone().detach())))
    for i in range(1000):
        if determine_goals_fulfilled(components):
            return sim_states, num_sim_states, sim_states_list, sensor_state_lookup, True
        components = run_component_sequence(components, torch.tensor(i * 0.01))

        action = components["MoveAction"].internal_action
        if action is not None:
            sim_state = components["DisksBelowSensor"].take_action(action)
            if sim_state is not None:
                sim_states.append((str(num_sim_states), torch.tensor(sim_state.clone().detach())))
                sim_states_list.append(convert_matrix_to_string_indexes(action))
                num_sim_states += 1

    return sim_states, num_sim_states, sim_states_list, sensor_state_lookup, False

def _format_float_for_filename(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _reset_gridsearch_seed(seed: int = GRIDSEARCH_SEED):
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_and_collect_results(
    init_setup,
    goal,
    experiment_results,
    experiment_name,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    _reset_gridsearch_seed()
    setup_func = lambda: get_building_functions_basic_blocks_world(
        init_setup=init_setup,
        goal=goal,
        alpha=alpha,
        beta=beta,
    )
    print("Start exp " + init_setup + "_" + goal + f" (alpha={alpha}, beta={beta})")
    sim_states, num_sim_states, sim_states_list, sensor_state_lookup, success = run_experiment(functions_constructor=setup_func)
    experiment_results["init"].append(init_setup[3:5])
    experiment_results["goal"].append(goal[3:5])
    experiment_results["num_moves"].append(num_sim_states - 1)
    experiment_results["finished"].append(success)
    experiment_results["moves taken"].append(sim_states_list)
    experiment_results["sequence"].append(get_sequence_from_sensor_states(sim_states, sensor_state_lookup))
    experiment_results["alpha"].append(alpha)
    experiment_results["beta"].append(beta)
    print(f"  Result: {success}, Moves: {num_sim_states - 1}")


def run_i_i(
    init_setup_list,
    goal_list,
    experiment_name=None,
    alpha: float = 1.0,
    beta: float = 1.0,
    csv_name: str = None,
):
    if experiment_name is None:
        ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
        experiment_name = ct
    experiment_results = {
        "init": [],
        "goal": [],
        "num_moves": [],
        "finished": [],
        "moves taken": [],
        "sequence": [],
        "alpha": [],
        "beta": [],
    }

    for init_setup, goal in zip(init_setup_list, goal_list):
        run_and_collect_results(
            init_setup,
            goal,
            experiment_results,
            experiment_name,
            alpha=alpha,
            beta=beta,
        )
    if MAKE_CSV:
        csv_path = export_csv(data=experiment_results, experiment_name=experiment_name, csv_name=csv_name)
    else:
        csv_path = "", ""
    return csv_path

def run_i_j(init_setup_list, goal_list, experiment_name=None, alpha: float = 1.0, beta: float = 1.0):
    if experiment_name is None:
        ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
        experiment_name = ct
    experiment_results = {
        "init": [],
        "goal": [],
        "num_moves": [],
        "finished": [],
        "moves taken": [],
        "sequence": [],
        "alpha": [],
        "beta": [],
    }

    for init_setup in init_setup_list:
        for goal in goal_list:
            run_and_collect_results(
                init_setup,
                goal,
                experiment_results,
                experiment_name,
                alpha=alpha,
                beta=beta,
            )
    return experiment_results

def run_all():
    init_setup_list, goal_list = setup_all()

    run_i_j(init_setup_list, goal_list)


def run_standard():
    init_setup_list, goal_list = setup_standard()
    output_dir, filename = run_i_i(init_setup_list, goal_list)
    print(f"Completed standard run. CSV: {output_dir}/{filename}")


def run_multiple(init=None, goal=None, alpha: float = 1.0, beta: float = 1.0, experiment_name=None, csv_name: str = None):
    if init is None:
        init = [54, 42]
    if goal is None:
        goal = [31, 23]
    init_setup_list, goal_list = setup_multiple(init, goal)

    return run_i_i(
        init_setup_list,
        goal_list,
        experiment_name=experiment_name,
        alpha=alpha,
        beta=beta,
        csv_name=csv_name,
    )


def run_multiple_gridsearch(
    init,
    goal,
    alpha_values,
    beta_values,
    csv_name_prefix: str,
    experiment_name: str = "gridsearch",
):
    if csv_name_prefix is None or csv_name_prefix.strip() == "":
        raise ValueError("csv_name_prefix must be a non-empty string")

    output_paths = []
    for alpha in alpha_values:
        for beta in beta_values:
            alpha_tag = _format_float_for_filename(float(alpha))
            beta_tag = _format_float_for_filename(float(beta))
            csv_name = f"{csv_name_prefix}_alpha-{alpha_tag}_beta-{beta_tag}"
            output_path = run_multiple(
                init=init,
                goal=goal,
                alpha=float(alpha),
                beta=float(beta),
                experiment_name=experiment_name,
                csv_name=csv_name,
            )
            print("Alpha:")
            print(alpha)
            print("Beta:")
            print(beta)
            output_paths.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "output_dir": output_path[0],
                "filename": output_path[1],
            })

    return output_paths


def run_gridsearch_example_12x12():
    init = [54, 42, 34, 12, 55, 16, 25, 36, 33, 53, 23, 55, 42, 54, 52, 55, 22, 34, 42, 46, 23, 43, 13, 22]
    goal = [31, 23, 13, 65, 41, 24, 32, 15, 11, 13, 43, 15, 21, 34, 12, 35, 41, 53, 63, 25, 61, 64, 32, 55]

    #alpha_values = [1.0]
    #beta_values = [1.0]
    #alpha_values = [0.1, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0]
    alpha_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0]
    beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0]
    #beta_values = [0.1]

    print("Starting 12x12 gridsearch example...")
    outputs = run_multiple_gridsearch(
        init=init,
        goal=goal,
        alpha_values=alpha_values,
        beta_values=beta_values,
        csv_name_prefix="gridsearch_bmsa",
        experiment_name="gridsearch_0-10000",
    )
    print(f"Finished 12x12 gridsearch example with {len(outputs)} runs.")
    return outputs

def run_single(init=11, goal=12):
    init_setup_list, goal_list = setup_single(init=init, goal=goal)
    output_dir, filename = run_i_i(init_setup_list, goal_list)
    print(f"Single run completed. CSV: {output_dir}/{filename}")

def export_csv(data, experiment_name, csv_name: str = None):
    ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha
    sha = "default"
    if csv_name is None:
        filename = "TOL-experiments-" + ct + "-" + str(sha) + ".csv"
    else:
        filename = csv_name + ".csv"

    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "tower_of_london_plots" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / filename, index=False)
    return str(output_dir), filename

if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    torch._dynamo.config.capture_func_transforms = True

    torch.use_deterministic_algorithms(True)
    torch.set_printoptions(profile="full", precision=20)

    seed = 111
    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #save_path = str(Path.cwd() / 'domip' / 'tower_of_london_specifications' / '24_problem_states.csv')
    #df = pd.read_csv(save_path)
    #print(df)
    #init = df["init"].tolist()
    #print(init)
    #goal = df["goal"].tolist()
    #print(goal)
    init = [54, 42, 34, 12, 55, 16, 25, 36, 33, 53, 23, 55, 42, 54, 52, 55, 22, 34, 42, 46, 23, 43, 13, 22]
    goal = [31, 23, 13, 65, 41, 24, 32, 15, 11, 13, 43, 15, 21, 34, 12, 35, 41, 53, 63, 25, 61, 64, 32, 55]

    #run_multiple(init, goal)
    # run_multiple()
    #print("Starting single run...")
    #run_single(34, 13)
    run_gridsearch_example_12x12()
    #print("Starting multiple run...")
    #run_multiple(init, goal)
    print("Finito")
