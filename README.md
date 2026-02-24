# AI-Driven Housing Affordability Forecasting in New York City
### An NTA-Level Panel Analysis Using Ensemble Machine Learning

> **Published as:** IEEE-format research paper — *AI-Driven Housing Affordability Forecasting in New York City: An NTA-Level Panel Analysis Using Ensemble Machine Learning*
> **Author:** Rahman Azizur · Department of Urban Informatics, New York University

---

## Overview

This project applies ensemble machine learning to forecast severe housing cost burden at the **Neighborhood Tabulation Area (NTA)** level across New York City from 2012 to 2022, with projections through 2025. It is the first study to apply gradient-boosted ensemble models — Random Forest, XGBoost, and LightGBM — to an NTA-level panel dataset integrating Census Bureau ACS data, NYC eviction records, and the Zillow Observed Rent Index (ZORI).

The primary target variable is `rent_burden_50plus_pct`: the share of renter-occupied households spending **50% or more of gross income on housing** — the HUD threshold for "severely cost-burdened."

---

## Key Results

| Model | Test R² | RMSE | MAE | Rank |
|---|---|---|---|---|
| **LightGBM** | **0.9214** | **0.036** | **0.019** | **1st** |
| XGBoost | 0.8981 | 0.041 | 0.019 | 2nd |
| Random Forest | 0.8895 | 0.042 | 0.019 | 3rd |

- **Best model:** LightGBM (Test R²=0.9214, 5-fold CV Mean R²=0.865 ± 0.045)
- **Top predictor (SHAP):** `rent_burden_30plus_pct` — confirming that moderate burden strongly predicts severe burden
- **Ablation:** Rental market features drive the largest R² drop (ΔR²=0.038)
- **Spatial:** Moran's I = −0.080 — no significant spatial autocorrelation in residuals
- **Forecast:** All four boroughs deteriorate through 2025; Queens shows highest relative increase (+6.7%)

---

## Dataset

### Panel Structure

| Attribute | Value |
|---|---|
| Total observations | 2,512 NTA-year rows |
| Neighborhoods (NTAs) | 239 (Bronx: 59 · Brooklyn: 76 · Manhattan: 55 · Queens: 49) |
| Time period | 2012–2022 (11 annual waves) |
| Total features | 49 (raw + engineered) |
| Target variable | `rent_burden_50plus_pct` (ACS Table B25070) |
| Target mean ± SD | 0.225 ± 0.128 (range: 0.000–1.000) |
| Train / Val / Test | 1,912 / 239 / 478 observations |

### Data Sources

| Source | Records | Variables |
|---|---|---|
| U.S. Census Bureau ACS 5-Year Estimates (2012–2022) | 23,059 tract-year records | Income, rent, vacancy, crowding, Gini, commuting |
| NYC Eviction Records — NYC Open Data (`6z8x-wfsh`) | 91,198 residential filings | NTA-year eviction filing rates |
| Zillow Observed Rent Index — ZORI (2014–2022) | 1,840 ZIP-year records | Median asking rent, smoothed & seasonally adjusted |
| NYC DCP NTA Crosswalk — NYC Open Data (`hm78-6dwm`) | 2,327 census tracts → 263 NTAs | Geographic aggregation weights |

### Descriptive Statistics (Key Variables)

| Variable | Mean | Median | Std Dev | Min | Max | Skewness |
|---|---|---|---|---|---|---|
| rent_burden_50plus_pct | 0.225 | 0.260 | 0.128 | 0.000 | 1.000 | −0.46 |
| rent_burden_30plus_pct | 0.419 | 0.492 | 0.216 | 0.000 | 1.000 | −1.05 |
| median_hh_income ($000s) | 66.3 | 60.2 | 32.6 | 2.5 | 225.3 | +1.34 |
| median_gross_rent ($) | 1,491 | 1,402 | 456 | 616 | 3,501 | +1.25 |
| renter_income_ratio | 0.851 | 0.865 | 0.098 | 0.393 | 1.102 | −0.57 |
| vacancy_rate | 0.075 | 0.067 | 0.086 | 0.000 | 1.000 | +6.45 |
| unemployment_rate | 0.077 | 0.071 | 0.075 | 0.000 | 1.000 | +5.61 |
| gini_coefficient | 0.455 | 0.456 | 0.063 | 0.015 | 0.692 | −1.66 |
| severe_crowding_rate | 0.001 | 0.000 | 0.002 | 0.000 | 0.023 | +4.22 |

---

## Repository Structure

```
.
├── NYC_Housing_IEEE_Final.ipynb          # Main notebook — run top-to-bottom
├── build_ieee_paper_v2.py               # Generates IEEE Word paper (Final)
├── build_ieee_paper.py                  # Earlier version of paper builder
├── generate_figures.py                  # Generates all 10 publication figures
├── rebuild_notebook_v2.py               # Regenerates the .ipynb from scratch
├── rebuild_notebook.py                  # Earlier notebook builder
│
├── IEEE_Housing_Affordability_NYC_Final.docx   # Full IEEE paper (output)
│
├── expanded_data/
│   ├── nta_panel_final.csv             # Main panel dataset (2,512 × 49)
│   ├── nta_acs_panel.csv               # ACS variables aggregated to NTA-year
│   ├── acs_tract_raw.csv               # Raw ACS census tract data
│   ├── evictions_by_nta_year.csv       # Eviction filings per NTA-year
│   ├── tract_nta_crosswalk.csv         # Census tract → NTA mapping
│   ├── zillow_zori_zip_annual.csv      # ZORI annual averages by ZIP code
│   └── zillow_zori_citywide.csv        # ZORI citywide index
│
└── figures/                            # Publication-quality PNGs (generated)
    ├── fig1_distribution.png           # Target distribution + borough violins
    ├── fig2_trends.png                 # Rent burden trends 2012–2022
    ├── fig3_scatter.png                # Income/rent vs burden scatter
    ├── fig4_correlation.png            # Pearson correlation heatmap
    ├── fig5_model_comparison.png       # Model performance comparison
    ├── fig6_cv_performance.png         # 5-fold CV results
    ├── fig7_shap.png                   # SHAP feature importance
    ├── fig8_ablation.png               # Ablation study
    ├── fig9_spatial.png                # Moran's I spatial residuals
    └── fig10_forecast.png              # Borough-level 2023–2025 forecast
```

---

## Methodology

### Pipeline Overview

```
Raw Data (ACS + Evictions + ZORI)
        ↓
  NTA Aggregation (population-weighted)
        ↓
  Feature Engineering (30 modeling features)
        ↓
  Temporal Split: Train (≤2019) | Val (2020) | Test (2021–2022)
        ↓
  Preprocessing: inf → NaN → Median Imputation (train-fit only)
        ↓
  Model Training: Random Forest · XGBoost · LightGBM (n_estimators=50)
        ↓
  Evaluation: R² · RMSE · MAE + 5-fold TimeSeriesSplit CV
        ↓
  SHAP TreeExplainer + Ablation Study + Moran's I
        ↓
  Borough-Level Forecast 2023–2025
```

### Feature Groups (30 modeling features)

| Group | Count | Examples |
|---|---|---|
| Income | 4 | median_hh_income, renter_income_ratio, income_gap |
| Rental Market | 4 | median_gross_rent, rent_burden_30plus_pct |
| Housing Stock | 5 | vacancy_rate, severe_crowding_rate, renter_share |
| Labor / Inequality | 3 | unemployment_rate, gini_coefficient |
| Engineered Composites | 4 | market_tightness, rent_to_income_ratio, renter_vulnerability |
| Temporal Lags (t−1) | 6 | income_lag1, rent_lag1, burden_30pct_lag1 |
| Growth Rates (YoY Δ) | 2 | income_growth_yoy, rent_growth_yoy |
| Identifiers | 2 | borough_code, covid_year |

### Null Value Handling

| Null Type | Treatment |
|---|---|
| `inf` / `-inf` from ratio features | Replaced with `NaN` |
| Missing ACS features | Median imputation — fitted on training set only |
| Missing ZORI (2012–2013) | Median imputation |
| Missing lag features (year 2012) | Median imputation |
| Missing target variable | Row dropped before splitting |

> **No leakage:** The `SimpleImputer` is fitted exclusively on 2012–2019 training data and applied without refitting to validation and test sets.

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn scipy python-docx
```

Or with Anaconda:
```bash
conda install pandas numpy scikit-learn matplotlib seaborn scipy
pip install xgboost lightgbm shap python-docx
```

### 2. Run the Notebook

Open `NYC_Housing_IEEE_Final.ipynb` in Jupyter and run all cells top-to-bottom.

```bash
jupyter notebook NYC_Housing_IEEE_Final.ipynb
```

> Uses `n_estimators=50` throughout — designed to complete without freezing.

### 3. Generate Figures Only

```bash
python generate_figures.py
# Saves 10 PNGs to ./figures/
```

### 4. Rebuild the IEEE Paper (Word document)

```bash
python build_ieee_paper_v2.py
# Outputs: IEEE_Housing_Affordability_NYC_Final.docx
```

### 5. Regenerate the Notebook from Scratch

```bash
python rebuild_notebook_v2.py
# Outputs: NYC_Housing_IEEE_Final.ipynb (40 cells)
```

---

## Notebook Sections

| Section | Content |
|---|---|
| 1 | Setup & library imports |
| 2 | Data loading & overview |
| 3 | Descriptive statistics (live-computed, matches paper Table III) |
| 4 | Exploratory Data Analysis — 6 inline figures |
| 5 | Feature engineering — 4 composite features |
| 6 | Temporal train / validation / test split |
| 7 | Model training + comparison chart + actual vs. predicted |
| 8 | 5-fold TimeSeriesSplit cross-validation |
| 9 | SHAP / built-in feature importance |
| 10 | Leave-one-group-out ablation study |
| 11 | Spatial autocorrelation — Moran's I |
| 12 | Borough deep dive (2022) |
| 13 | Borough-level forecast 2023–2025 |
| Final | Results summary printed to output |

---

## SHAP Feature Importance (Top 10)

| Rank | Feature | Mean \|SHAP\| | Direction |
|---|---|---|---|
| 1 | rent_burden_30plus_pct | 0.0412 | + |
| 2 | renter_income_ratio | 0.0318 | − |
| 3 | median_gross_rent | 0.0287 | + |
| 4 | median_hh_income | 0.0241 | − |
| 5 | rent_burden_30plus_pct_lag1 | 0.0198 | + |
| 6 | unemployment_rate | 0.0176 | + |
| 7 | rent_to_income_ratio | 0.0154 | + |
| 8 | renter_median_income | 0.0142 | − |
| 9 | eviction_rate | 0.0121 | + |
| 10 | vacancy_rate | 0.0098 | − |

---

## Borough-Level Forecast (2023–2025)

> Baseline assumptions: rent +3%/yr · HH income +2.5%/yr · renter income +2%/yr · unemployment constant

| Borough | 2022 Actual | 2025 Forecast | Δ Change | Risk Level |
|---|---|---|---|---|
| **Bronx** | 0.2998 | 0.3107 | +3.6% | Critical |
| Queens | 0.2256 | 0.2406 | +6.7% | High |
| Brooklyn | 0.2541 | 0.2701 | +6.3% | High |
| Manhattan | 0.2090 | 0.2162 | +3.4% | Moderate |

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| xgboost | ≥ 1.6 |
| lightgbm | ≥ 3.3 |
| shap | ≥ 0.41 |
| matplotlib | ≥ 3.4 |
| seaborn | ≥ 0.11 |
| scipy | ≥ 1.7 |
| python-docx | ≥ 0.8.11 |

---

## Limitations

- **Causal inference not supported** — SHAP importance does not imply that intervening on a feature produces a proportional change in burden
- **ACS margins of error** are not propagated into prediction uncertainty intervals
- **Spatial analysis** operates at borough level (n=4 units); NTA-level Moran's I with k-nearest-neighbor weights is future work
- **Forecast** assumes a single baseline economic scenario — no confidence intervals or scenario analysis
- **Missing features:** rent stabilization status, NYCHA inventory, HPD complaint rates, and building-level characteristics are not included

---

## Citation

If you use this dataset, code, or paper in your research, please cite:

```bibtex
@article{azizur2024nyc_housing,
  title     = {AI-Driven Housing Affordability Forecasting in New York City:
               An NTA-Level Panel Analysis Using Ensemble Machine Learning},
  author    = {Azizur, Rahman},
  journal   = {IEEE-Format Research Paper},
  year      = {2024},
  note      = {Department of Urban Informatics, New York University}
}
```

---

## Data Sources & Licenses

| Source | License |
|---|---|
| U.S. Census Bureau ACS | Public domain (U.S. Government) |
| NYC Eviction Records | NYC Open Data — Open Data Commons Public Domain |
| Zillow ZORI | Zillow Research — free for non-commercial use with attribution |
| NYC DCP NTA Crosswalk | NYC Open Data — Open Data Commons Public Domain |

---

## Contact

**Rahman Azizur**
Department of Urban Informatics, New York University
New York, NY 10012
r.azizur@nyu.edu
