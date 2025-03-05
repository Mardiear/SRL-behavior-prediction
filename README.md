# SRL-behavior-prediction
This repository contains the code to process the StudentLife Dataset, run sedentary behaviour prediction experiments and process the results.

## Process the dataset

- Download the dataset
- Use the functionality located in `/src/preprocessing` to generate the processed datasets.
  - `get_studentlife_dataset()` generates the dataset based on a granularity (`nb_min`).
  - `generate_clean_dataset()` cleans the dataset by removing inconsistencies. 
  - `generate_lagged_dataset()` adds lags to a cleaned dataset.
  - `get_user_data()` and `get_not_user_data()` are used to retrieve the data that will be used for training and testing, depending on the nature of the experiment (personal or impersonal).
  - `time_series_split()` splits the training and testing datasets into `k` validation splits.

## Tune the neural networks

Use the functionality located in `/src/tunning` to tune the hyperparameters of each type of neural network. The tuning is done by using Bayes normalization. As hyperparameter search is expensive, checkpoints of this process are stored in `/pkl/tunning/`.


## Run the experiments

Use the functionality located in `/src/experiments` to run the experiments.

- `run_all_experiments()` runs all the experiments using the values returned by `get_experiment_combinations()`.
- `run_experiment()` runs a single experiment by creating `PersonalExperiment` and `ImpersonalExperiment` objects.

## Process the experiment results
Use the functionality located in `/src/experiments/experiments_results.py` to process the experiment results. The experiment results for each of the experiments are stored as 
pickle files in `/pkl/experiments`.

- `generate_df_from_experiments()` gathers all the experiments and puts them in a single Dataframe.
- Get insight of the results by calling `rank_results_agg_func()` providing a comparison column, a rank by column, a based on column and an aggregation function.
- Use a wide variety of plotting functions (the ones that begin `plot_`).
- Use `get_p_value()` to run statistic tests.
