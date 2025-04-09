# NPRI Time Series Analysis and Forecasting

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Last Updated](https://img.shields.io/badge/last%20updated-April%202025-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

## 📋 Project Overview

This project is a direct continuation of the CMPT2400 NPRI Data Analysis project, where we initially processed and analyzed the National Pollutant Release Inventory (NPRI) dataset and merged it with newly available data for 2023. The NPRI is Canada's comprehensive public inventory tracking pollutant releases, disposals, and transfers across industries nationwide.

Building on that foundation, this advanced project applies sophisticated time series analysis and machine learning techniques to:

1. **Analyze Historical Trends**: Identify patterns in pollutant release data (2013-2023)
2. **Develop Predictive Models**: Create industry-specific forecasting models
3. **Generate Future Projections**: Forecast pollutant releases for 2024-2028
4. **Identify Critical Industries**: Highlight sectors with the highest projected increases or decreases in emissions

The analysis focuses on three key environmental metrics:

- **Air Releases**: Pollutants released directly into the atmosphere
- **Land Releases**: Pollutants discharged onto land surfaces
- **Water Releases**: Pollutants released into water bodies

## 🔑 Key Findings

- **Forecast Period**: 2024-2028 (5-year forecast horizon)
- **Industry with Highest Projected Growth in Emissions**: "Cement, Lime, and Other Non-Metallic Minerals"
- **Industry with Largest Projected Decline**: "Metals (Except Aluminum, and Iron and Steel)"
- **Top Polluting Industries**:
  1. Oil and Gas Extraction (excluding oil sands)
  2. Wood Products
  3. Mining and Quarrying
  4. Electricity
  5. Pulp and Paper

## 🧠 Methodology

### Data Preprocessing
- **Continuation from CMPT2400**: Leveraged the enriched dataset from the previous project where we merged historical NPRI data with 2023 records
- Advanced data cleaning tailored for time series forecasting versus the initial exploratory analysis done in CMPT2400
- Feature engineering specifically for forecasting:
  - Created lag features (1-7 years) to capture temporal dependencies
  - Developed time-based features to identify long-term trends
  - Constructed industry-specific variables to model sector-based patterns
- One-hot encoding of industry sectors (21 distinct categories) to enable sector-specific modeling

### Modeling Approach
- **Models Evaluated**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² Score

### Validation Strategy
- Time series cross-validation using TimeSeriesSplit
- Historical data segmentation to maintain temporal integrity

## 📊 Repository Structure

```
npri_time_series/
│
├── data/                       # Data directory
│   ├── raw/                    # Raw, immutable data dump
│   └── processed/              # Cleaned, transformed data ready for modeling
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Initial data exploration
│   ├── 02_data_preprocessing.ipynb  # Data cleaning and feature engineering
│   └── 03_modeling_evaluation.ipynb # Model building and evaluation
│
├── src/                        # Source code
│   ├── __init__.py             # Make src a Python package
│   ├── data/                   # Scripts for data preparation
│   ├── models/                 # Model training and prediction scripts
│   ├── features/               # Feature engineering scripts
│   └── utils/                  # Utility functions
│
├── models/                     # Saved model objects
│
├── reports/                    # Reports and results
│   └── figures/                # Generated visualizations
│
├── requirements.txt            # Project dependencies
│
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Dependencies listed in requirements.txt

### 📂 Dataset

The full `df_merged_releases.csv` (181MB) is not stored in the repository due to GitHub file size limits.  
You can download it [here](https://drive.google.com/file/d/10ufcTVPHdto-9PrQfZ-ZdNqNhDm5bf58/view?usp=drive_link).

Place it in the `/data/` folder before running the notebook or scripts.

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/almasoriaw/npri_time_series_ml_cmpt3510.git
   cd npri_time_series_ml_cmpt3510
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the notebooks or scripts:
   ```
   jupyter notebook notebooks/
   # or
   python src/main.py
   ```

## 📈 Results

Building on the exploratory analysis from CMPT2400, this project successfully delivers industry-specific forecasts for pollution release trends across Canada for 2024-2028, providing actionable insights for environmental policy, industrial regulation, and sustainability planning.

### Key Forecast Achievements
- **Industry Growth Projections**: Predicted a 32.8% increase in emissions from the Cement & Non-Metallic Minerals sector
- **Industry Decline Projections**: Forecasted a 26.4% decrease in the Metals sector (excluding Aluminum, Iron and Steel)
- **Regional Impact Assessment**: Identified provinces most affected by changing pollution patterns
- **Substance-Specific Forecasts**: Created targeted models for high-priority pollutants

### Model Performance Analysis
- **Well-performing Models**: 
  - Air release models show strong predictive power (e.g., propylene: R² = 0.955, calcium fluoride: R² = 0.966)
  - Time series patterns in the Oil & Gas sector were captured with high accuracy (RMSE < 2.3%)
- **Challenging Models**: 
  - Some substances with irregular release patterns proved difficult to model (hexachlorobenzene: R² = 0.048)
  - Industries with frequent facility openings/closures showed less predictable patterns

## 🔄 Future Work

### Advanced Modeling Extensions
- Implement deep learning approaches (LSTM, Transformer models) for improved time series forecasting accuracy
- Develop specialized models for the top 5 industries with highest projected changes in emissions
- Explore multi-target models that can simultaneously forecast air, land, and water releases

### Data Enrichment Opportunities
- Integrate economic indicators (GDP, employment rates) to analyze correlations with pollution trends
- Incorporate climate data to explore relationships between weather patterns and emission levels
- Add regulatory timeline data to evaluate policy impacts on industrial emissions

### Visualization and Interpretation
- Create an interactive dashboard for dynamic exploration of forecast results
- Develop province-level visualizations to identify regional emission hotspots
- Build explanatory tools to interpret model predictions for non-technical stakeholders

### CMPT2400 Connection
- Extend the data pipeline created in CMPT2400 to automatically integrate future NPRI data releases
- Refine the merging methodology developed in the previous project for improved data quality

## 📚 Data Source

### NPRI Dataset (2013-2023)
This project uses the National Pollutant Release Inventory (NPRI), Environment and Climate Change Canada's comprehensive public inventory tracking industrial pollutant releases, disposals, and transfers for recycling.

### Data Continuation from CMPT2400
The CMPT2400 project established the data pipeline for NPRI data processing and merged historical data (2013-2022) with the newly available 2023 data. This merged dataset (`df_merged_releases.csv`, ~190MB) serves as the foundation for this advanced time series analysis project.

### Key Dataset Characteristics
- **Time Range**: Annual data from 2013-2023
- **Facilities Covered**: Over 7,000 industrial facilities across Canada
- **Pollutants Tracked**: More than 320 substances of environmental concern
- **Industries**: 21 distinct industrial sectors from manufacturing to mining

## 📃 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For questions or feedback about this project:
- Open an issue on this GitHub repository, contact me --> [LinkedIn](https://linkedin.com/in/almasoria)
