# src/preprocessing/data_preprocessing.py
import pandas as pd
from datetime import datetime


def extract_physical_activity_features():
    """
    Extracts physical activity features from raw sensor data.
    """
    df = pd.read_csv('../data/raw/raw_activity_data.csv')
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1d')
    activity_features = df.groupby(['userId', 'time', 'inference']).size().unstack(fill_value=0).reset_index()
    activity_features.rename(columns={0: 'stationaryCount', 1: 'walkingCount', 2: 'runningCount', 3: 'unknownCount'},
                             inplace=True)
    activity_features['totalActivityCount'] = activity_features[
        ['stationaryCount', 'walkingCount', 'runningCount']].sum(axis=1)
    activity_features['activityMajor'] = activity_features[['stationaryCount', 'walkingCount', 'runningCount']].idxmax(
        axis=1)
    return activity_features


def extract_social_interaction_features():
    """
    Extracts social interaction features from raw sensor data.
    """
    call_df = pd.read_csv('../data/raw/raw_call_data.csv')
    sms_df = pd.read_csv('../data/raw/raw_sms_data.csv')
    screen_df = pd.read_csv('../data/raw/raw_screen_data.csv')

    call_df['time'] = pd.to_datetime(call_df['time'], unit='s').dt.floor('1d')
    call_features = call_df.groupby(['userId', 'time', 'direction']).size().unstack(fill_value=0).reset_index()
    call_features.rename(columns={'incoming': 'incomingCallCount', 'outgoing': 'outgoingCallCount'}, inplace=True)

    sms_df['time'] = pd.to_datetime(sms_df['time'], unit='s').dt.floor('1d')
    sms_features = sms_df.groupby(['userId', 'time', 'direction']).size().unstack(fill_value=0).reset_index()
    sms_features.rename(columns={'incoming': 'incomingSMSCount', 'outgoing': 'outgoingSMSCount'}, inplace=True)

    screen_df['time'] = pd.to_datetime(screen_df['time'], unit='s').dt.floor('1d')
    screen_features = screen_df.groupby(['userId', 'time']).size().reset_index(name='screenOnCount')
    screen_features['screenOnFrequency'] = screen_features['screenOnCount'] / 24

    social_features = pd.merge(call_features, sms_features, on=['userId', 'time'], how='outer').fillna(0)
    social_features = pd.merge(social_features, screen_features, on=['userId', 'time'], how='outer').fillna(0)
    social_features['totalCommunicationCount'] = social_features[
        ['incomingCallCount', 'outgoingCallCount', 'incomingSMSCount', 'outgoingSMSCount']].sum(axis=1)
    social_features['communicationMajor'] = social_features[
        ['incomingCallCount', 'outgoingCallCount', 'incomingSMSCount', 'outgoingSMSCount']].idxmax(axis=1)
    return social_features


def extract_sleep_features():
    """
    Extracts sleep features from raw sensor data.
    """
    sleep_df = pd.read_csv('../data/raw/raw_sleep_data.csv')
    sleep_df['time'] = pd.to_datetime(sleep_df['time'], unit='s').dt.floor('1d')
    sleep_features = sleep_df.groupby(['userId', 'time']).agg({
        'duration': 'sum',  # Total sleep duration
        'start_time': 'min',  # Sleep start time
        'end_time': 'max',  # Sleep end time
        'quality': 'mean',  # Average sleep quality
        'episodes': 'count',  # Total sleep episodes
        'state': lambda x: x.mode()[0]  # Dominant sleep state
    }).reset_index()
    sleep_features.rename(columns={
        'duration': 'sleepDuration',
        'start_time': 'sleepStartTime',
        'end_time': 'sleepEndTime',
        'quality': 'sleepQuality',
        'episodes': 'totalSleepEpisodes',
        'state': 'dominantSleepState'
    }, inplace=True)
    sleep_features['Sleepduration_sum'] = sleep_features['sleepDuration']
    sleep_features['Sleepquality_avg'] = sleep_features['sleepQuality']
    return sleep_features


def extract_location_features():
    """
    Extracts location features from raw sensor data.
    """
    location_df = pd.read_csv('../data/raw/raw_location_data.csv')
    location_df['time'] = pd.to_datetime(location_df['time'], unit='s').dt.floor('1d')
    location_df['location_type'] = location_df['location'].apply(lambda x: categorize_location(x))
    location_features = location_df.groupby(['userId', 'time', 'location_type']).size().unstack(
        fill_value=0).reset_index()
    location_features.rename(columns={
        'dormitory': 'dormitoryTime',
        'classroom': 'classroomTime',
        'library': 'libraryTime',
        'study_room': 'studyRoomTime',
        'other': 'otherLocationTime'
    }, inplace=True)
    location_features['totalLocationTime'] = location_features[
        ['dormitoryTime', 'classroomTime', 'libraryTime', 'studyRoomTime', 'otherLocationTime']].sum(axis=1)
    location_features['dominantLocation'] = location_features[
        ['dormitoryTime', 'classroomTime', 'libraryTime', 'studyRoomTime', 'otherLocationTime']].idxmax(axis=1)
    location_features['LocationTime_sum'] = location_features['totalLocationTime']
    return location_features


def extract_app_usage_features():
    """
    Extracts app usage features from raw sensor data.
    """
    app_df = pd.read_csv('../data/raw/raw_app_usage_data.csv')
    app_df['time'] = pd.to_datetime(app_df['time'], unit='s').dt.floor('1d')
    app_df['app_category'] = app_df['app_name'].apply(lambda x: categorize_app(x))
    app_features = app_df.groupby(['userId', 'time', 'app_category']).size().unstack(fill_value=0).reset_index()
    app_features.rename(columns={
        'education': 'educationAppTime',
        'social': 'socialAppTime',
        'entertainment': 'entertainmentAppTime',
        'game': 'gameAppTime',
        'other': 'otherAppTime'
    }, inplace=True)
    app_features['totalAppTime'] = app_features[
        ['educationAppTime', 'socialAppTime', 'entertainmentAppTime', 'gameAppTime', 'otherAppTime']].sum(axis=1)
    app_features['dominantAppCategory'] = app_features[
        ['educationAppTime', 'socialAppTime', 'entertainmentAppTime', 'gameAppTime', 'otherAppTime']].idxmax(axis=1)
    app_features['Appusage_time_sum'] = app_features['totalAppTime']
    return app_features


def categorize_location(location):
    """
    Categorizes a location into one of the predefined types.
    """
    if 'dormitory' in location.lower():
        return 'dormitory'
    elif 'classroom' in location.lower():
        return 'classroom'
    elif 'library' in location.lower():
        return 'library'
    elif 'study_room' in location.lower():
        return 'study_room'
    else:
        return 'other'


def categorize_app(app_name):
    """
    Categorizes an app into one of the predefined categories.
    """
    if 'education' in app_name.lower():
        return 'education'
    elif 'social' in app_name.lower():
        return 'social'
    elif 'entertainment' in app_name.lower():
        return 'entertainment'
    elif 'game' in app_name.lower():
        return 'game'
    else:
        return 'other'


def preprocess_data():
    """
    Preprocesses all raw sensor data and saves the preprocessed data to a CSV file.
    """
    physical_activity_features = extract_physical_activity_features()
    social_interaction_features = extract_social_interaction_features()
    sleep_features = extract_sleep_features()
    location_features = extract_location_features()
    app_usage_features = extract_app_usage_features()
    all_features = pd.merge(physical_activity_features, social_interaction_features, on=['userId', 'time'], how='outer')
    all_features = pd.merge(all_features, sleep_features, on=['userId', 'time'], how='outer')
    all_features = pd.merge(all_features, location_features, on=['userId', 'time'], how='outer')
    all_features = pd.merge(all_features, app_usage_features, on=['userId', 'time'], how='outer')
    all_features.to_csv('../data/processed/preprocessed_data.csv', index=False)


if __name__ == "__main__":
    preprocess_data()