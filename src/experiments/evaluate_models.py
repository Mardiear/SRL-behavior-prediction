# src/experiments/evaluate_models.py
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model and prints performance metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MSE: {mse}, MAE: {mae}')

if __name__ == "__main__":
    # Load test data
    X_test = np.load('../data/processed/X_test.npy')
    y_test = np.load('../data/processed/y_test.npy')

    # Load trained models
    mlp_model = load_model('../models/mlp_model.h5')
    lstm_model = load_model('../models/lstm_model.h5')
    cnn_model = load_model('../models/cnn_model.h5')
    tcn_model = load_model('../models/tcn_model.h5')

    # Evaluate models
    print("Evaluating MLP...")
    evaluate_model(mlp_model, X_test, y_test)

    print("Evaluating LSTM...")
    evaluate_model(lstm_model, X_test, y_test)

    print("Evaluating CNN...")
    evaluate_model(cnn_model, X_test, y_test)

    print("Evaluating TCN...")
    evaluate_model(tcn_model, X_test, y_test)