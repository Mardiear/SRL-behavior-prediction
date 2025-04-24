# SRL-Behavior-Prediction

This repository contains the code to process the StudentLife Dataset, run sedentary behaviour prediction experiments and process the results.

The dataset is put on the “master“ Branche.

## Repository Structure

```
SRL-Behavior-Prediction/
├── results/                  # Experiment results and visualizations
│   ├── figures/              # Plots and graphs
│   └── metrics/              # Performance metrics (e.g., MSE, MAE)
│
├── src/                      # Source code
│   ├── preprocessing/        # Data preprocessing scripts
│   ├── tuning/               # Hyperparameter tuning scripts
│   ├── experiments/          # Experiment execution scripts
│   └── utils/                # Utility functions and helpers
│
├── pkl/                      # Pickle files for checkpoints and intermediate results
│   ├── tuning/               # Hyperparameter tuning checkpoints
│   └── experiments/          # Experiment results
│
├── README.md                 # Project overview and instructions
└── requirements.txt          # Python dependencies
```

## Workflow

### 1. Data Preprocessing
The raw sensor data is preprocessed to extract meaningful features for predicting SRL behaviors. The preprocessing steps include:

- Extracting physical activity, social interaction, sleep, location, and app usage features.
- Cleaning and normalizing the data.
- Generating time windows for training and testing deep learning models.

#### Key Scripts:
- `data_preprocessing.py`: Extracts and processes features from raw sensor data.
- `time_windows.py`: Generates time windows for model training and testing.

#### Example Usage:

```
python src/preprocessing/data_preprocessing.py
python src/preprocessing/time_windows.py
```
### 2. Hyperparameter Tuning

Hyperparameter tuning is performed for each deep learning architecture (MLP, LSTM, CNN, TCN) using Bayesian optimization. The tuning process evaluates different configurations to identify the best-performing hyperparameters.

#### Key Scripts:
- `tuning/tune_mlp.py`: Tunes hyperparameters for the MLP model.
- `tuning/tune_lstm.py`: Tunes hyperparameters for the LSTM model.
- `tuning/tune_cnn.py`: Tunes hyperparameters for the CNN model.
- `tuning/tune_tcn.py`: Tunes hyperparameters for the TCN model.

#### Example Usage:
```
python src/tuning/tune_mlp.py
python src/tuning/tune_lstm.py
python src/tuning/tune_cnn.py
python src/tuning/tune_tcn.py
```
### 3. Model Training and Evaluation

The four deep learning models are trained and evaluated using the preprocessed dataset. The models are compared based on their performance in predicting future SRL scores.

#### Key Scripts:
- `train_models.py`: Trains MLP, LSTM, CNN, and TCN models.
- `evaluate_models.py`: Evaluates the trained models and computes performance metrics (e.g., MSE, MAE).

#### Example Usage:
```
python src/experiments/train_models.py
python src/experiments/evaluate_models.py
```

### 4. Experiment Results

The results of the experiments are processed and analyzed to gain insights into the performance of the models. Statistical tests and visualizations are used to compare the models and evaluate the impact of individual differences.

#### Key Scripts:
- `experiments_results.py`: Processes experiment results and generates visualizations.
- `plot_results.py`: Creates plots and graphs for the results.

#### Example Usage:
```
python src/experiments/experiments_results.py
python src/experiments/plot_results.py
```

## Key Features

### Time Windows Definition

The dataset is divided into time windows to predict future SRL scores based on historical data. The time window size and prediction period are key parameters:

- **Time Window Size**: Defines the length of historical data used for predictions (e.g., 7, 14, 21, or 30 days).
- **Prediction Period**: Defines the number of days ahead to predict (e.g., 1 day).

For each time bucket, historical features are combined with the current bucket's features to construct a feature matrix. The target variable is the OSLQ score at the current time step.

## Methodology

To reduce computational costs, surrogate experiments are conducted:

- **Leave-One-Subject-Out (LOSO)**: Evaluates the impact of individual differences by leaving one subject out for testing.
- **Specific Leave-One-Subject-Out (SLOSO)**: Focuses on specific subjects to analyze individual characteristics.
- **Hyperparameter Optimization**: Performed for MLP, LSTM, CNN, and TCN models to identify the best configurations.

A total of 96 architectures are generated during hyperparameter tuning, with 100 iterations for each tuning process.

## Usage

### Install Dependencies:
```
pip install -r requirements.txt
```
### Download the Dataset:

Place the raw sensor data in the `data/raw/` directory.

### Run the Pipeline:

#### Preprocess the data:

```
python src/preprocessing/data_preprocessing.py
python src/preprocessing/time_windows.py
```

### Tune hyperparameters:

```
python src/tuning/tune_mlp.py
python src/tuning/tune_lstm.py
python src/tuning/tune_cnn.py
python src/tuning/tune_tcn.py
```

### Train and evaluate models:

```
python src/experiments/train_models.py
python src/experiments/evaluate_models.py
```

### Analyze results:

```
python src/experiments/experiments_results.py
python src/experiments/plot_results.py
```

## Results

The results of the experiments are stored in the `results/` directory, including performance metrics, visualizations, and statistical tests. Key insights include:

- Comparison of model performance (MLP, LSTM, CNN, TCN).
- Impact of time window size and prediction period on model accuracy.
- Analysis of individual differences in SRL behavior prediction.
