# src/utils/utils.py
import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads a dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """
    Saves a dataset to a CSV file.
    """
    data.to_csv(file_path, index=False)

def save_numpy_array(array, file_path):
    """
    Saves a NumPy array to a file.
    """
    np.save(file_path, array)

def load_numpy_array(file_path):
    """
    Loads a NumPy array from a file.
    """
    return np.load(file_path)