# src/preprocessing/time_windows.py
import pandas as pd

def generate_time_windows(data, time_window_size=7, prediction_period=1):
    """
    Generates a time windows dataset for predicting future SRL (OSLQ scores) based on historical data.
    """
    data = data.sort_values(by=['userId', 'time'])
    features, targets = [], []

    for user in data['userId'].unique():
        user_data = data[data['userId'] == user].copy()
        user_data = user_data.set_index('time')

        for i in range(time_window_size, len(user_data) - prediction_period + 1):
            historical_features = user_data.iloc[i - time_window_size:i].drop(columns=['userId', 'OSLQ_score']).values.flatten()
            target = user_data.iloc[i + prediction_period - 1]['OSLQ_score']
            features.append(historical_features)
            targets.append(target)

    feature_columns = [f'feature_{i}' for i in range(len(features[0]))]
    time_windows = pd.DataFrame(features, columns=feature_columns)
    time_windows['OSLQ_score'] = targets
    return time_windows

if __name__ == "__main__":
    data = pd.read_csv('../data/processed/preprocessed_data.csv')
    data['time'] = pd.to_datetime(data['time'])
    time_windows = generate_time_windows(data, time_window_size=7, prediction_period=1)
    time_windows.to_csv('../data/processed/time_windows_dataset.csv', index=False)