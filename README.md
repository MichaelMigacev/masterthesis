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

Some AI has been used for autocompleting and repetitive code sequences, such as imports exports and print commands. Files were I have used the assistance of AI tools of any sort contain a mention of that in the description at the top of the file.
Files without such a mention can be regarded as created without my use of any AI tools or assistance. The thouroughness of the documentation has been done to the best of my abilities.
