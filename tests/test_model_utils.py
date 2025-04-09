"""
Tests for model utility functions in the NPRI Time Series Analysis project.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestRegressor

# Add the source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model_utils import calculate_metrics, forecast_future


class TestModelUtils(unittest.TestCase):
    """Test cases for model utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample actual and predicted values
        self.y_true = np.array([10, 15, 20, 25, 30])
        self.y_pred = np.array([12, 14, 21, 24, 31])
        
        # Create sample data for forecasting
        self.forecast_data = pd.DataFrame({
            'Reporting_Year': [2023],
            'Value_lag_1': [30],
            'Value_lag_2': [25],
            'Industry_Sector_Mining': [1],
            'Industry_Sector_Oil': [0]
        })
        
        # Create sample model
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Fit with dummy data
        X_dummy = np.array([[2023, 30, 25, 1, 0],
                           [2022, 25, 20, 1, 0],
                           [2021, 20, 15, 1, 0]])
        y_dummy = np.array([35, 30, 25])
        self.model.fit(X_dummy, y_dummy)
    
    def test_calculate_metrics(self):
        """Test the calculate_metrics function."""
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        # Check if metrics were calculated
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Check specific values (with tolerance)
        self.assertAlmostEqual(metrics['rmse'], 1.414, places=2)
        self.assertAlmostEqual(metrics['mae'], 1.2, places=2)
        # R² should be very high for this test data
        self.assertGreater(metrics['r2'], 0.95)
    
    def test_forecast_future(self):
        """Test the forecast_future function."""
        forecast_df = forecast_future(
            self.model, 
            self.forecast_data,
            periods=3,  # shorter forecast for testing
            year_start=2024
        )
        
        # Check if forecast dataframe has correct structure
        self.assertEqual(len(forecast_df), 3)
        self.assertIn('Forecast_Year', forecast_df.columns)
        self.assertIn('Predicted_Value', forecast_df.columns)
        
        # Check years
        self.assertEqual(forecast_df['Forecast_Year'].tolist(), [2024, 2025, 2026])
        
        # All predictions should be numeric
        self.assertTrue(all(isinstance(x, (int, float)) for x in forecast_df['Predicted_Value']))


if __name__ == '__main__':
    unittest.main()
