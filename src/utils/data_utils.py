"""
Utility functions for data preprocessing and manipulation
for the NPRI Time Series Analysis project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os


def load_data(filepath):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath)


def create_lag_features(df, target_column, lag_periods=7, group_cols=None):
    """
    Create lag features for time series analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_column : str
        Column to create lag features for
    lag_periods : int, default=7
        Number of lag periods to create
    group_cols : list, default=None
        Columns to group by before creating lags
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added lag features
    """
    df_copy = df.copy()
    
    if group_cols is not None:
        for lag in range(1, lag_periods + 1):
            lag_col_name = f'{target_column}_lag_{lag}'
            df_copy[lag_col_name] = df_copy.groupby(group_cols)[target_column].shift(lag)
    else:
        for lag in range(1, lag_periods + 1):
            lag_col_name = f'{target_column}_lag_{lag}'
            df_copy[lag_col_name] = df_copy[target_column].shift(lag)
    
    return df_copy


def one_hot_encode_categories(df, categorical_cols, drop_first=False):
    """
    One-hot encode categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns to encode
    drop_first : bool, default=False
        Whether to drop the first category in each feature
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one-hot encoded categories
    """
    df_copy = df.copy()
    
    for col in categorical_cols:
        if col in df_copy.columns:
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy.drop(col, axis=1, inplace=True)
    
    return df_copy


def prepare_time_series_split(df, test_size=0.2, time_col='Reporting_Year'):
    """
    Split data for time series validation, ensuring chronological order is maintained.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    test_size : float, default=0.2
        Proportion of data to use for testing
    time_col : str, default='Reporting_Year'
        Column containing time information
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Train and test dataframes
    """
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    return train_df, test_df


def save_processed_data(df, output_path, filename):
    """
    Save processed data to CSV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe to save
    output_path : str
        Directory to save the file
    filename : str
        Name of the output file
        
    Returns:
    --------
    str
        Path to the saved file
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    output_file = os.path.join(output_path, filename)
    df.to_csv(output_file, index=False)
    
    return output_file
