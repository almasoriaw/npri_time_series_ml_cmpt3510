"""
Utility functions for model training, evaluation, and forecasting
for the NPRI Time Series Analysis project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os


def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, substance_name, target_type):
    """
    Train and evaluate a regression model for time series forecasting.
    
    Parameters:
    -----------
    model : sklearn estimator
        The regression model to train
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target values
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target values
    model_name : str
        Name of the model for reporting
    substance_name : str
        Name of the substance being modeled
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
        
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and evaluation metrics
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Create results dictionary
    results = {
        'model': model,
        'model_name': model_name,
        'substance_name': substance_name,
        'target_type': target_type,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    return results


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Determine model performance category
    if r2 >= 0.5:
        category = "Well performed Model"
    else:
        category = "Ill performed Model"
    
    # Create metrics dictionary
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'category': category
    }
    
    return metrics


def timeseries_cv(model, X, y, n_splits=5):
    """
    Perform time series cross-validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        The regression model to evaluate
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target values
    n_splits : int, default=5
        Number of splits for time series cross-validation
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation metrics
    """
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize lists to store metrics
    cv_rmse = []
    cv_mae = []
    cv_r2 = []
    
    # Perform cross-validation
    for train_idx, test_idx in tscv.split(X):
        # Split data
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train_cv, y_train_cv)
        
        # Make predictions
        y_pred_cv = model.predict(X_test_cv)
        
        # Calculate metrics
        cv_rmse.append(np.sqrt(mean_squared_error(y_test_cv, y_pred_cv)))
        cv_mae.append(mean_absolute_error(y_test_cv, y_pred_cv))
        cv_r2.append(r2_score(y_test_cv, y_pred_cv))
    
    # Calculate average metrics
    avg_metrics = {
        'mean_rmse': np.mean(cv_rmse),
        'std_rmse': np.std(cv_rmse),
        'mean_mae': np.mean(cv_mae),
        'std_mae': np.std(cv_mae),
        'mean_r2': np.mean(cv_r2),
        'std_r2': np.std(cv_r2)
    }
    
    return avg_metrics


def save_model(model, model_name, substance_name, target_type, models_dir='../../models'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    model_name : str
        Name of the model
    substance_name : str
        Name of the substance
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    models_dir : str, default='../../models'
        Directory to save models
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Create filename
    filename = f"{model_name}_{substance_name}_{target_type}.joblib"
    filepath = os.path.join(models_dir, filename)
    
    # Save model
    joblib.dump(model, filepath)
    
    return filepath


def load_model(model_name, substance_name, target_type, models_dir='../../models'):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    substance_name : str
        Name of the substance
    target_type : str
        Type of target (Air_Releases, Land_Releases, or Water_Releases)
    models_dir : str, default='../../models'
        Directory where models are saved
        
    Returns:
    --------
    object
        Loaded model
    """
    # Create filename
    filename = f"{model_name}_{substance_name}_{target_type}.joblib"
    filepath = os.path.join(models_dir, filename)
    
    # Load model
    model = joblib.load(filepath)
    
    return model


def forecast_future(model, last_data, periods=5, year_start=2024):
    """
    Generate forecasts for future periods.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained forecasting model
    last_data : pandas.DataFrame
        Last available data point(s) used for forecasting
    periods : int, default=5
        Number of periods to forecast (years)
    year_start : int, default=2024
        Starting year for forecasting
        
    Returns:
    --------
    pandas.DataFrame
        Forecasted values for future periods
    """
    # Initialize results dataframe
    forecast_df = pd.DataFrame()
    
    # Copy the last data point as a starting point
    forecast_data = last_data.copy()
    
    # Get feature columns (excluding target)
    feature_cols = forecast_data.columns
    
    # Generate forecasts for each period
    forecasts = []
    
    for i in range(periods):
        current_year = year_start + i
        
        # Update year-related features
        if 'Reporting_Year' in forecast_data.columns:
            forecast_data['Reporting_Year'] = current_year
        
        if 'Year_Diff' in forecast_data.columns:
            min_year = year_start - forecast_data['Year_Diff'].iloc[0]
            forecast_data['Year_Diff'] = current_year - min_year
        
        # Make prediction for current period
        prediction = model.predict(forecast_data)
        
        # Create result row
        result = {
            'Forecast_Year': current_year,
            'Predicted_Value': prediction[0]
        }
        
        # Update lag features for next period if they exist
        lag_cols = [col for col in forecast_data.columns if '_lag_' in col]
        if lag_cols:
            # Sort lag columns by lag number (descending)
            lag_cols = sorted(lag_cols, key=lambda x: int(x.split('_')[-1]), reverse=True)
            
            # Shift lag values (lag_1 <- prediction, lag_2 <- lag_1, etc.)
            for i in range(len(lag_cols) - 1):
                forecast_data[lag_cols[i+1]] = forecast_data[lag_cols[i]]
            
            # Set lag_1 to current prediction
            forecast_data[lag_cols[-1]] = prediction[0]
        
        # Add to results
        forecasts.append(result)
    
    # Convert to dataframe
    forecast_df = pd.DataFrame(forecasts)
    
    return forecast_df
