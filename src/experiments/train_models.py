# src/experiments/train_models.py
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd

def build_mlp(input_shape):
    """
    Builds a Multilayer Perceptron (MLP) model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    return model

def build_lstm(input_shape):
    """
    Builds a Long Short-Term Memory (LSTM) model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    return model

def build_cnn(input_shape):
    """
    Builds a Convolutional Neural Network (CNN) model.
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(1)
    ])
    return model

def build_tcn(input_shape):
    """
    Builds a Temporal Convolutional Network (TCN) model.
    """
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal'),
        Flatten(),
        Dense(1)
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains a model and evaluates its performance.
    """
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    return model

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('../data/processed/time_windows_dataset.csv')
    X = data.drop(columns=['OSLQ_score']).values
    y = data['OSLQ_score'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for LSTM, CNN, and TCN
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train MLP
    print("Training MLP...")
    mlp_model = build_mlp(X_train.shape[1])
    train_model(mlp_model, X_train, y_train, X_test, y_test)
    mlp_model.save('../models/mlp_model.h5')

    # Train LSTM
    print("Training LSTM...")
    lstm_model = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    train_model(lstm_model, X_train_lstm, y_train, X_test_lstm, y_test)
    lstm_model.save('../models/lstm_model.h5')

    # Train CNN
    print("Training CNN...")
    cnn_model = build_cnn((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    train_model(cnn_model, X_train_lstm, y_train, X_test_lstm, y_test)
    cnn_model.save('../models/cnn_model.h5')

    # Train TCN
    print("Training TCN...")
    tcn_model = build_tcn((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    train_model(tcn_model, X_train_lstm, y_train, X_test_lstm, y_test)
    tcn_model.save('../models/tcn_model.h5')