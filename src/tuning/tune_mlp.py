# src/tuning/tune_mlp.py
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('../data/processed/time_windows_dataset.csv')
X = data.drop(columns=['OSLQ_score']).values
y = data['OSLQ_score'].values

# Define hyperparameter space
space = [
    Integer(1, 5, name='num_layers'),
    Integer(32, 256, name='units_per_layer'),
    Real(0.0, 0.5, name='dropout_rate'),
    Categorical([True, False], name='batch_normalization')
]


# Define the model-building function
def build_mlp(num_layers, units_per_layer, dropout_rate, batch_normalization):
    model = Sequential()
    for i in range(num_layers):
        model.add(Dense(units_per_layer, activation='relu'))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    return model


# Define the objective function for Bayesian optimization
@use_named_args(space)
def objective(**params):
    model = build_mlp(**params)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    return loss


# Perform Bayesian optimization
res = gp_minimize(objective, space, n_calls=100, random_state=42)

# Print the best hyperparameters
print("Best hyperparameters:")
print(f"Number of layers: {res.x[0]}")
print(f"Units per layer: {res.x[1]}")
print(f"Dropout rate: {res.x[2]}")
print(f"Batch normalization: {res.x[3]}")
print(f"Best MSE: {res.fun}")