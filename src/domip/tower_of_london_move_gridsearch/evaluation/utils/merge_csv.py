import datetime
import pandas as pd
from pathlib import Path

from domip.tower_of_london_direct.evaluation.create_csv_from_data import *


def export_csv(data, filename, save_path):
    ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    filename = filename+ "-" + ct + ".csv"
    df = pd.DataFrame(data)
    df.to_csv(save_path + "//" + filename , index=False)
    return save_path + "//" + filename

def import_csv(filename):
    df = pd.read_csv(filename)
    return df

def get_standard_tests_from_aicon(filename, save_path):
    standard_tests_dict = get_dict_from_config_strings()
    readings = import_csv(filename)
    frames = []
    for init, goal in zip(standard_tests_dict["init"], standard_tests_dict["goal"]):
        frame = readings[(readings['init'] == init) & (readings['goal'] == goal)]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = pd.concat([result, percentage_solved(Path.cwd().parent / 'tower_of_london_plots' / 'human_data' / 'data4berlin.csv')], axis=1)
    return export_csv(result, "standard_tests_aicon", save_path)

def get_standard_tests(save_path):
    standard_tests_dict = get_dict_from_config_strings()
    export_csv(standard_tests_dict, "standard_tests", save_path)

def percentage_solved(filename):
    readings = import_csv(filename)
    percentage = {
        "mean": [],
        "mean_moves" : [],
        "mean_moves_on_success": [],
    }
    length = readings.shape[0]
    for x in ["05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28"]:
        percentage["mean"].append(readings[readings["L" + x] < 3.0].shape[0] / length)
        if readings[readings["L" + x] < 3.0].shape[0]:
            filtered_df = readings[readings["L" + x] < 3.0]
            length_filtered = filtered_df.shape[0]
            percentage["mean_moves_on_success"].append(filtered_df["Z" + x].sum() / length_filtered)
        percentage["mean_moves"].append(readings["Z" + x].sum() / length)

    return pd.DataFrame(percentage)


def merge_csv(filename1, filename2, save_path):
    file1 = pd.read_csv(filename1)
    file2 = pd.read_csv(filename2)
    result = pd.merge(file1, file2, how="outer", on=["init", "goal"])
    result = result.sort_values(by=["number"])
    return export_csv(result, "merge_csv", save_path)