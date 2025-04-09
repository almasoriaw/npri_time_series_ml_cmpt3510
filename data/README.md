# NPRI Dataset Information

## Overview

This directory contains the National Pollutant Release Inventory (NPRI) dataset used for time series analysis and forecasting. The main dataset (`df_merged_releases.csv`) is a comprehensive collection of pollutant release data that includes:

1. Historical NPRI data (pre-2023)
2. Additional 2023 data that was merged as part of the CMPT2400 project

## Dataset Structure

### Raw Data

The `raw` directory contains the original, unmodified datasets:

- `df_merged_releases.csv` (~190MB): The complete dataset with historical and 2023 data merged during the CMPT2400 project

### Processed Data

The `processed` directory contains cleaned and transformed versions of the data that are ready for modeling:

- Preprocessed datasets with feature engineering
- Train/test splits for model evaluation
- Industry-specific subsets for targeted analysis

## Data Dictionary

The dataset includes the following key columns:

| Column Name | Description | Data Type |
|-------------|-------------|-----------|
| Reporting_Year | Year when the data was reported | Integer |
| Facility_Name | Name of the reporting facility | String |
| Substance_Name | Name of the pollutant substance | String |
| Industry_Sector | Industry sector classification | String |
| Total_Air_Releases | Amount released to air (tonnes) | Float |
| Total_Land_Releases | Amount released to land (tonnes) | Float |
| Total_Water_Releases | Amount released to water (tonnes) | Float |
| [Additional lag features] | Time-lagged variables created for forecasting | Float |

## Data Preparation Pipeline

1. **Data Loading**: Import the merged dataset from CMPT2400
2. **Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create lag variables and time-based features
4. **Encoding**: One-hot encode categorical variables
5. **Splitting**: Create chronological train/test splits for time series validation

## Usage Notes

When using this data, be aware that:

1. The dataset includes both historical NPRI data and 2023 data merged from the CMPT2400 project
2. Some pollutant measurements may have different reporting thresholds over time
3. Industry classifications have been standardized to ensure consistency

## Data Sources

- National Pollutant Release Inventory (NPRI), Environment and Climate Change Canada
- Additional 2023 data incorporated from the CMPT2400 project
