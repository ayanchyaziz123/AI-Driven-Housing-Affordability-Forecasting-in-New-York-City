# AI-Driven Housing Affordability Forecasting in New York City

An end-to-end machine learning framework for forecasting neighborhood-level housing affordability across NYC boroughs (2005–2027), with explainable AI insights and gentrification risk analysis.

---

## Overview

This project integrates demographic, economic, transit, and crime data from four NYC boroughs — **Manhattan, Bronx, Brooklyn, and Queens** — to:

- Forecast housing affordability stress using XGBoost, Random Forest, and LSTM models
- Analyze the income gap between all households and renter households
- Predict affordability trends through 2027
- Identify gentrification risk patterns by borough
- Provide explainable feature attributions via SHAP (XAI)

---

## Project Structure

```
.
├── NYC_Housing_Affordability_Forecasting.ipynb   # Main analysis notebook
├── borough-medianhouseholdincome2024.csv
├── borough-medianhouseholdincomerenters2024.csv
├── borough-population.csv
├── borough-populationdensity1000personspersquaremile.csv
├── borough-homeownershiprate.csv
├── borough-seriouscrimerateper1000residents.csv
├── borough-car-freecommuteofcommuters.csv
├── borough-meantraveltimetoworkminutes.csv
├── borough-higher-costhomepurchaseloansofhomepurchaseloans.csv
├── borough-higher-costrefinanceloansofrefinanceloans.csv
├── borough-residentialunitswithin12mileofasubwaystation.csv
├── borough-residentialunitswithin14mileofapark.csv
└── README.md
```

**Generated outputs** (after running the notebook):

| File | Description |
|------|-------------|
| `nyc_affordability_full_dataset.csv` | Merged and engineered feature dataset |
| `nyc_affordability_forecast_2024_2027.csv` | Borough-level affordability forecasts |
| `model_performance_results.csv` | Train/test metrics for all models |
| `shap_feature_importance.csv` | Mean absolute SHAP values per feature |
| `fig1_income_trends.png` – `fig18_summary_dashboard.png` | All visualizations |

---

## Data Sources

All data is sourced at the NYC borough level and covers 2005–2023:

| Dataset | Variable |
|---------|----------|
| Median Household Income (2024$) | `median_household_income` |
| Median Renter Household Income (2024$) | `renter_median_income` |
| Population | `population` |
| Population Density (per 1,000 sq mi) | `pop_density_1000sqmi` |
| Homeownership Rate | `homeownership_rate` |
| Serious Crime Rate (per 1,000 residents) | `crime_rate_per1000` |
| Car-Free Commute Share | `car_free_commute_pct` |
| Mean Travel Time to Work (minutes) | `mean_travel_time_min` |
| Higher-Cost Home Purchase Loans (%) | `high_cost_purchase_loan_pct` |
| Higher-Cost Refinance Loans (%) | `high_cost_refi_loan_pct` |
| Residential Units within ½ mile of subway | `subway_proximity_pct` |
| Residential Units within ¼ mile of park | `park_proximity_pct` |

---

## Feature Engineering

The following composite indicators are derived from the raw data:

| Feature | Description |
|---------|-------------|
| `renter_income_ratio` | Renter median income / Overall median income |
| `cost_pressure_index` | Average of high-cost purchase and refinance loan rates |
| `affordability_stress` | Cost pressure index / Renter income ratio — the **primary target variable** |
| `income_gap` | Overall median income − Renter median income |
| `renter_share` | 1 − Homeownership rate |
| `income_growth_yoy` | Year-over-year % change in median household income |
| `*_lag1` | One-year lags for key features |

---

## Models

| Model | Type | Purpose |
|-------|------|---------|
| **Random Forest** | Ensemble regression | Baseline with strong interpretability |
| **XGBoost** | Gradient-boosted trees | Primary forecasting model |
| **LSTM** | Recurrent neural network | Sequential time-series patterns (requires TensorFlow) |

Models are evaluated using RMSE, MAE, and R² on a chronological train/test split (last 20% of years held out as test).

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is applied to the XGBoost model to:

- Rank features by their mean absolute contribution to predictions
- Visualize directional effects via beeswarm and dependence plots
- Explain individual predictions with waterfall charts

Top affordability drivers identified include renter income ratio, cost pressure index, income gap, and lagged affordability stress.

---

## Forecasting

An iterative forecasting approach uses XGBoost to project **affordability stress through 2027** for each borough:

1. Extrapolate feature trends using 3-year linear regression per borough
2. Feed projected features into the trained XGBoost model
3. Use the previous year's predicted stress as a lag feature for each next step

---

## Gentrification Risk

A composite **Gentrification Risk Score (0–1)** is computed from five normalized indicators:

- Rising household income growth
- Declining crime rate (area becoming more desirable)
- High renter share (displacement risk)
- Declining homeownership rate
- High car-free commute share (transit desirability)

---

## Key Findings

- **Bronx and Brooklyn** show the highest affordability stress for renters
- Renter income has grown slower than overall income, widening the inequality gap
- High-cost loan concentration is a strong predictor of affordability stress spikes
- Transit access (car-free commute) correlates with gentrification pressure
- XGBoost achieves the best predictive performance on the test set

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
tensorflow   # optional — required for LSTM model
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap tensorflow
```

---

## Usage

1. Place all CSV data files in the same directory as the notebook
2. Open `NYC_Housing_Affordability_Forecasting.ipynb` in Jupyter or VS Code
3. Run all cells sequentially
4. Output figures and CSV exports will be saved to the working directory

---

## Policy Implications

- Targeted rent stabilization is needed in high-stress boroughs (Bronx, Brooklyn)
- Income support programs for renters are critical where renter income lags overall growth
- Affordable housing incentives should focus on boroughs with rising gentrification risk scores
- Transit investment should be coupled with displacement-prevention policies in well-connected neighborhoods
