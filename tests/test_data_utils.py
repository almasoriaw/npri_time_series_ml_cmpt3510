"""
Tests for data utility functions in the NPRI Time Series Analysis project.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the source directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_utils import create_lag_features, one_hot_encode_categories


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample dataframe
        self.df = pd.DataFrame({
            'Year': [2018, 2019, 2020, 2021, 2022, 2023,
                     2018, 2019, 2020, 2021, 2022, 2023],
            'Industry': ['Mining', 'Mining', 'Mining', 'Mining', 'Mining', 'Mining',
                        'Oil', 'Oil', 'Oil', 'Oil', 'Oil', 'Oil'],
            'Facility': ['A', 'A', 'A', 'A', 'A', 'A',
                         'B', 'B', 'B', 'B', 'B', 'B'],
            'Value': [10, 15, 20, 25, 30, 35,
                      5, 10, 15, 20, 25, 30]
        })
    
    def test_create_lag_features(self):
        """Test the create_lag_features function."""
        # Create lag features
        result_df = create_lag_features(
            self.df, 
            'Value', 
            lag_periods=2, 
            group_cols=['Industry', 'Facility']
        )
        
        # Check if lag columns were created
        self.assertIn('Value_lag_1', result_df.columns)
        self.assertIn('Value_lag_2', result_df.columns)
        
        # Check specific lag values for facility A
        facility_a = result_df[result_df['Facility'] == 'A']
        # 2020 should have 2019 as lag 1 (15)
        self.assertEqual(
            facility_a.loc[facility_a['Year'] == 2020, 'Value_lag_1'].iloc[0], 
            15
        )
        # 2021 should have 2020 as lag 1 (20) and 2019 as lag 2 (15)
        self.assertEqual(
            facility_a.loc[facility_a['Year'] == 2021, 'Value_lag_1'].iloc[0], 
            20
        )
        self.assertEqual(
            facility_a.loc[facility_a['Year'] == 2021, 'Value_lag_2'].iloc[0], 
            15
        )
    
    def test_one_hot_encode_categories(self):
        """Test the one_hot_encode_categories function."""
        # Encode 'Industry' column
        result_df = one_hot_encode_categories(self.df, ['Industry'])
        
        # Check if one-hot columns were created
        self.assertIn('Industry_Mining', result_df.columns)
        self.assertIn('Industry_Oil', result_df.columns)
        
        # Check values
        self.assertEqual(result_df.loc[0, 'Industry_Mining'], 1)
        self.assertEqual(result_df.loc[0, 'Industry_Oil'], 0)
        self.assertEqual(result_df.loc[6, 'Industry_Mining'], 0)
        self.assertEqual(result_df.loc[6, 'Industry_Oil'], 1)


if __name__ == '__main__':
    unittest.main()
