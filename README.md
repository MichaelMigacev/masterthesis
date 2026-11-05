# Masterthesis: Investigating the Relationship Between a Reactive Robotics Model and Human Clinical Populations via Failure Modes on a Planning Task

**Authors:** Michael Migacev

## Description

This Repository contains code used for the writing of the master thesis.

## Overview

#### Tower of London Plots

Contains the results of an AICON model applied to the Tower of London problems. Each alpha-beta combination produces one .csv-file summarizing AICON's performance with that parameter combination.
The original code to produce the results is in the `tower_of_london_move_gridsearch` folder.

#### Evaluation against Baselines

The code to evaluate against other methods is in the `evaluation` folder in `tower_of_london_move_gridsearch`. It leans on human data, an example of which can be found in the `tower_of_london_human_data` folder.

#### Evaluation for Classification

All methods for the evaluation of AICON as an assisting tool in classification is contained in the `evaluation` folder under `indiv_data`.

#### Availability of Human Data

As I work with clinical data for which I have no permission to distribute, no human data is provided with the thesis or the thesis code. But to increase the legibility of the code an example of human data is provided in the `tower_of_london_human_data` folder.

#### Prior Work and AI Use

This code builds on Vito Mengers' implementation of the AICON framework. Thus the files outside `tower_of_london_move_gridsearch` remained largely unchanged.

Some AI has been used, primarily for autocompleting and repetitive code sequences, such as imports exports and print commands. Files were I have used the assistance of AI tools of any sort contain a mention of that in the description at the top of the file.
Files without such a mention can be regarded as created without my use of any AI tools or assistance. The thouroughness of the documentation has been done to the best of my abilities.

## Running the Code

### AICON

To prepare running the AICON code I recommend to setup a conda environment using [environment_tol.yml](./environment_tol.yml)<br>
The environment file should be OS independent.

To perform the full AICON run of all alpha and beta combinations you can run [run_experiment.py](./src/domip/tower_of_london_move_gridsearch/run_experiment.py):

```
python src\domip\tower_of_london_move_gridsearch\evaluation\run_experiment.py
```


### Analysis

#### Please keep in mind that no human data is made available with this repository

To run the analysis you can prepare the human data with the [problem_ordering_preparation.ipynb](./src/domip/tower_of_london_move_gridsearch/evaluation/utils/problem_ordering_preparation.ipynb) file. An example for the human data is provided [here](./src/tower_of_london_human_data/EXAMPLE_HUMAN_DATA.csv)

After preparing the human data you can either compare the human ordering to a fixed baseline such as [bfs_bidirectional](./src/tower_of_london_human_data/bfs_bidirectional_avg_nodes_for_po.csv) by running the [po_pipeline_pp_mult_val.py](./src/domip/tower_of_london_move_gridsearch/evaluation/po_pipeline_pp_mult_val.py):

```
python src\domip\tower_of_london_move_gridsearch\evaluation\po_pipeline_pp_mult_val.py "folder_with_human_orderings" --metric-path "file_for_baseline_orderings" --output-folder "output_folder_name" --permutation-runs 200 --seed 42 --splits-csv-path "csv-file_of_your_splits"
```

or compare it to the model data by running [top_analysis_average.py](./src/domip/tower_of_london_move_gridsearch/evaluation/top_analysis_average.py):

```
python "path_to/top_analysis_average.py" --reference-orderings-path "folder_with_human_orderings" --model-orderings-path "file_of_gridsearch_output" --cross-validation-path "csv-file_of_your_splits" --output-root "path_to_output" --folder-name "output_name" --seed 42 --permutation-runs 200 --metric-mode extra_number_of_moves --moves-till-failure none
```

### Provided Results

This repository includes the resulting data from running the AICON model to enable reproducing the results without setting up AICON.
The data is provided in the [gridsearchl](./tower_of_london_plots/gridsearch_0-10000) folder
