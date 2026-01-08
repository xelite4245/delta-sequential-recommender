import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np



# Mutual Information Feature Selection

def get_mi_scores(X, y):
    """
    Calculate Mutual Information scores between features and the target variable.

    Parameters:
    X (DataFrame): Feature set.
    y (Series): Target variable.

    Returns:
    Series: Mutual Information scores for each feature.
    """

    mi_scores = mutual_info_regression(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_series

def plot_mi_scores(mi_scores):
    """
    Plot Mutual Information scores.

    Parameters:
    mi_scores (Series): Mutual Information scores for each feature.
    """

    plt.figure(figsize=(10, 6))
    sns.barplot(x=mi_scores.values, y=mi_scores.index)
    plt.title("Mutual Information Scores")
    plt.xlabel("MI Score")
    plt.ylabel("Features")
    plt.show()


#Time-Based Feature Engineering

def add_time_features(df):
    """
    Add time-based features to the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    Returns:
    DataFrame: Modified DataFrame with new time features.
    """

    df['date'] = pd.to_datetime(df['date'])
    first_workout_date = df['date'].min()
    df['days_since_first_workout'] = (df['date'] - first_workout_date).dt.days

    df = df.sort_values(by='date')
    df['days_since_last_workout'] = df['date'].diff().dt.days.fillna(0).astype(int)

    return df

def add_days_since_last_workout(df):
    """
    Add a feature indicating days since the last workout session.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    Returns:
    DataFrame: Modified DataFrame with new feature.
    """

    df = df.sort_values(by='date')
    df['days_since_last_workout'] = df['date'].diff().dt.days.fillna(0).astype(int)

    return df

def add_session_number_per_exercise(df):
    """
    Add a feature indicating the session number per exercise.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    Returns:
    DataFrame: Modified DataFrame with new feature.
    """

    df = df.sort_values(by=['exercise_normalized', 'date'])
    df['session_number'] = (
        df.groupby('exercise_normalized')['date']
        .rank(method='dense')
        .astype(int)
    )

    return df

def add_rolling_avg_load_last_n_sessions(df, n=3):
    """
    Add a feature indicating the rolling average load over the last n sessions.

    Parameters:
    df (DataFrame): The DataFrame to modify.
    n (int): Number of sessions to consider for rolling average.

    Returns:
    DataFrame: Modified DataFrame with new feature.
    """

    df = df.sort_values(by=['exercise_normalized', 'date'])
    df['rolling_avg_load_last_{}_sessions'.format(n)] = (
        df.groupby('exercise_normalized')['effective_load']
        .rolling(window=n, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df

def add_rolling_trend_load(df, n=5):
    """
    Add a feature indicating the rolling trend (slope) of load over the last n sessions.

    Parameters:
    df (DataFrame): The DataFrame to modify.
    n (int): Number of sessions to consider for rolling trend.

    Returns:
    DataFrame: Modified DataFrame with new feature.
    """

    def compute_slope(x):
        if len(x) < 2:
            return 0
        y = x.values
        x_vals = np.arange(len(y))
        A = np.vstack([x_vals, np.ones(len(x_vals))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    df = df.sort_values(by=['exercise_normalized', 'session_number'])
    df['rolling_trend_load'] = (
        df.groupby('exercise_normalized')['effective_load']
        .rolling(window=n, min_periods=2)
        .apply(compute_slope, raw=False)
        .reset_index(level=0, drop=True)
    )

    return df




# RPE Feature Engineering

def handle_missing_rpe(df):
    """
    Handle missing RPE values by creating an indicator column and imputing with median RPE.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    Returns:
    DataFrame: Modified DataFrame with missing RPE handled.
    """

    df['rpe_missing'] = df['rpe'].isna().astype(int)
    median_rpe = df['rpe'].median()
    df['rpe'] = df['rpe'].fillna(median_rpe)

    return df

def bin_rpe(df):
    """
    Bin RPE values into categorical bins: Low, Medium, High.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    Returns:
    DataFrame: Modified DataFrame with binned RPE feature.
    """

    bins = [0, 5, 7, 10]
    labels = ['Low', 'Medium', 'High']
    df['rpe_binned'] = pd.cut(df['rpe'], bins=bins, labels=labels, include_lowest=True)
    return df

def encode_rpe_ordinal(df):
    """
    Encode binned RPE as an ordinal numerical variable.

    Parameters:
    df (DataFrame): The DataFrame to modify RPE feature.
    0 - 5 : Low
    6 - 7 : Medium
    8 - 10: High

    Returns:
    DataFrame: Modified DataFrame with ordinal encoded RPE feature.
    """

    rpe_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['rpe_ordinal'] = df['rpe_binned'].map(rpe_mapping).astype(int)

    return df



# Top set intensity prediction feature engineering

def filter_top_set_sessions(df):
    """
    Filter the DataFrame to only include top set sessions.
    This will be done by grouping by days_since_first_workout, workout_name, and exercise_normalized
    and selecting the row with the maximum set_volume within each group.

    Then create a bool column 'is_top_set' to indicate whether a row is a top set or not.

    Parameters:
    df (DataFrame): The DataFrame to filter.

    Returns:
    DataFrame: Dataframe with a new column 'is_top_set' indicating top set sessions.
    """
    df = df.copy()

    df['is_top_set'] = False

    idx = (
        df
        .groupby(['days_since_first_workout', 'workout_name', 'exercise_normalized'])['set_volume']
        .idxmax()
    )

    df.loc[idx, 'is_top_set'] = True

    return df

def add_reps_binned(df):
    """
    Bin reps into categorical bins: Low, Medium, High.

    Parameters:
    df (DataFrame): The DataFrame to modify.

    rep_range_buckets:
    1-5: Strength
    6-15: Hypertrophy
    15+: Endurance

    Returns:
    DataFrame: Modified DataFrame with binned reps feature.
    """

    bins = [0, 5, 15, np.inf]
    labels = ['Strength', 'Hypertrophy', 'Endurance']
    df['reps_binned'] = pd.cut(df['reps'], bins=bins, labels=labels, include_lowest=True)
    return df
