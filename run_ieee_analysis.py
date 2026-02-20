"""
AI-Driven Housing Affordability Forecasting - NYC
IEEE-Standard Analysis (Standalone Script)
Run from Terminal: /opt/anaconda3/bin/python run_ieee_analysis.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display needed — saves files directly
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print(f'LightGBM available: {lgb.__version__}')
except ImportError:
    LGB_AVAILABLE = False
    print('LightGBM not available — skipping.')

try:
    import shap
    SHAP_AVAILABLE = True
    print(f'SHAP available: {shap.__version__}')
except ImportError:
    SHAP_AVAILABLE = False
    print('SHAP not available — skipping.')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')

OUTPUT_DIR = '.'
DATA_PATH = 'expanded_data/nta_panel_final.csv'

# ─────────────────────────────────────────────
# SECTION 1: Load Data
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 1: Loading Data')
print('='*60)

df = pd.read_csv(DATA_PATH)
print(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
print(f'Boroughs: {sorted(df["borough_name"].unique().tolist())}')
print(f'Years: {sorted(df["year"].unique().tolist())}')
print(f'NTAs: {df["nta_code"].nunique()} unique NTA codes')

# ─────────────────────────────────────────────
# SECTION 2: EDA Figures
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 2: EDA Figures')
print('='*60)

from scipy.stats import gaussian_kde

# Fig 1: Distribution
col = 'rent_burden_30plus_pct'
data_clean = df[col].dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(data_clean, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(data_clean.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Mean: {data_clean.mean():.3f}')
axes[0].axvline(data_clean.median(), color='orange', linestyle='--', linewidth=1.8, label=f'Median: {data_clean.median():.3f}')
axes[0].set_xlabel('Rent Burden 30+ Pct'); axes[0].set_ylabel('Frequency')
axes[0].set_title('Fig 1a: Histogram of Rent Burden (30%+)'); axes[0].legend()
kde = gaussian_kde(data_clean)
x_vals = np.linspace(data_clean.min(), data_clean.max(), 300)
axes[1].plot(x_vals, kde(x_vals), color='steelblue', linewidth=2)
axes[1].fill_between(x_vals, kde(x_vals), alpha=0.3, color='steelblue')
axes[1].axvline(data_clean.mean(), color='red', linestyle='--', linewidth=1.8, label=f'Mean: {data_clean.mean():.3f}')
axes[1].axvline(data_clean.median(), color='orange', linestyle='--', linewidth=1.8, label=f'Median: {data_clean.median():.3f}')
axes[1].set_xlabel('Rent Burden 30+ Pct'); axes[1].set_ylabel('Density')
axes[1].set_title('Fig 1b: KDE of Rent Burden (30%+)'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig1_burden_distribution.png'), bbox_inches='tight')
plt.close(); print('Fig 1 saved.')

# Fig 2: Boxplot by borough
borough_order = sorted(df['borough_name'].dropna().unique().tolist())
plot_data = [df.loc[df['borough_name'] == b, 'rent_burden_50plus_pct'].dropna().values for b in borough_order]
fig, ax = plt.subplots(figsize=(11, 6))
bp = ax.boxplot(plot_data, labels=borough_order, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
colors = ['#4472C4', '#ED7D31', '#A9D18E', '#FF0000']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xlabel('Borough'); ax.set_ylabel('Rent Burden 50%+')
ax.set_title('Fig 2: Severe Rent Burden by Borough (2012-2022)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig2_burden_by_borough.png'), bbox_inches='tight')
plt.close(); print('Fig 2 saved.')

# Fig 3: NTA heatmap
nta_avg = df.groupby('nta_name')['rent_burden_50plus_pct'].mean().dropna()
top20_ntas = nta_avg.nlargest(20).index.tolist()
pivot_df = df[df['nta_name'].isin(top20_ntas)].pivot_table(
    index='nta_name', columns='year', values='rent_burden_50plus_pct', aggfunc='mean')
pivot_df = pivot_df.loc[nta_avg[top20_ntas].sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.4, linecolor='white', cbar_kws={'label': 'Rent Burden 50%+'}, ax=ax)
ax.set_title('Fig 3: NTA Severe Rent Burden Heatmap (Top 20 NTAs)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig3_nta_heatmap.png'), bbox_inches='tight')
plt.close(); print('Fig 3 saved.')

# Fig 4: Scatter matrix
scatter_cols = ['median_hh_income', 'median_gross_rent', 'rent_burden_50plus_pct',
                'unemployment_rate', 'severe_crowding_rate']
available_scatter = [c for c in scatter_cols if c in df.columns]
scatter_data = df[available_scatter].dropna()
rename_map = {'median_hh_income': 'HH Income', 'median_gross_rent': 'Gross Rent',
              'rent_burden_50plus_pct': 'Burden 50%+', 'unemployment_rate': 'Unemp Rate',
              'severe_crowding_rate': 'Crowding'}
scatter_data_renamed = scatter_data.rename(columns=rename_map)
fig = plt.figure(figsize=(13, 11))
pd.plotting.scatter_matrix(scatter_data_renamed, alpha=0.25, figsize=(13, 11),
                            diagonal='kde', color='steelblue')
plt.suptitle('Fig 4: Scatter Matrix - Key Housing Variables', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig4_scatter_matrix.png'), bbox_inches='tight')
plt.close(); print('Fig 4 saved.')

# Fig 5: Time trends
trend = df.groupby(['borough_name', 'year'])['rent_burden_30plus_pct'].mean().reset_index()
colors_list = ['#4472C4', '#ED7D31', '#A9D18E', '#FF4444']
fig, ax = plt.subplots(figsize=(13, 6))
for i, boro in enumerate(sorted(trend['borough_name'].unique())):
    boro_data = trend[trend['borough_name'] == boro].sort_values('year')
    ax.plot(boro_data['year'], boro_data['rent_burden_30plus_pct'],
            marker='o', linewidth=2.2, markersize=5, color=colors_list[i], label=boro)
ax.axvspan(2019.5, 2021.5, alpha=0.15, color='red', label='COVID-19 Period')
ax.set_xlabel('Year'); ax.set_ylabel('Mean Rent Burden (30%+)')
ax.set_title('Fig 5: Rent Burden Trend by Borough (2012-2022)')
ax.legend(); ax.set_xticks(sorted(df['year'].unique()))
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig5_time_trends.png'), bbox_inches='tight')
plt.close(); print('Fig 5 saved.')

# ─────────────────────────────────────────────
# SECTION 3: Target Variable
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 3: Target Variable')
print('='*60)

df['affordability_stress'] = (df['median_gross_rent'] / df['renter_median_income'].clip(lower=1) * 12)
q99 = df['affordability_stress'].quantile(0.99)
df['affordability_stress'] = df['affordability_stress'].clip(upper=q99)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
t1 = df['rent_burden_50plus_pct'].dropna()
axes[0].hist(t1, bins=40, color='#C0392B', edgecolor='white', alpha=0.8)
axes[0].axvline(t1.mean(), color='navy', linestyle='--', linewidth=2, label=f'Mean: {t1.mean():.3f}')
axes[0].set_xlabel('Rent Burden 50%+'); axes[0].set_title('Primary Target'); axes[0].legend()
t2 = df['affordability_stress'].dropna()
axes[1].hist(t2, bins=40, color='#2980B9', edgecolor='white', alpha=0.8)
axes[1].axvline(t2.mean(), color='navy', linestyle='--', linewidth=2, label=f'Mean: {t2.mean():.3f}')
axes[1].set_xlabel('Affordability Stress'); axes[1].set_title('Secondary Target'); axes[1].legend()
plt.suptitle('Fig 6: Target Variable Distributions', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig6_target_distributions.png'), bbox_inches='tight')
plt.close(); print('Fig 6 saved.')

# ─────────────────────────────────────────────
# SECTION 4: Feature Engineering
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 4: Feature Engineering')
print('='*60)

df['market_tightness'] = 1.0 / (df['vacancy_rate'].clip(lower=0.001) + 0.005)
df['rent_to_income_ratio'] = df['median_gross_rent'] * 12.0 / df['renter_median_income'].clip(lower=1)
df['housing_burden_composite'] = df['unemployment_rate'] + df['severe_crowding_rate']
df['renter_vulnerability'] = (
    (1.0 - df['renter_income_ratio'].clip(lower=0, upper=1)) * 0.5
    + df['unemployment_rate'] * 0.3
    + df['severe_crowding_rate'] * 0.2
)
print('Engineered features created.')

# Fig 7: Correlation heatmap
corr_cols = ['median_hh_income', 'renter_median_income', 'median_gross_rent',
             'rent_burden_30plus_pct', 'rent_burden_50plus_pct', 'vacancy_rate',
             'unemployment_rate', 'severe_crowding_rate', 'gini_coefficient',
             'market_tightness', 'rent_to_income_ratio', 'renter_vulnerability']
available_corr = [c for c in corr_cols if c in df.columns]
corr_matrix = df[available_corr].corr().round(2)
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax)
ax.set_title('Fig 7: Correlation Matrix of Key Features')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig7_correlation_heatmap.png'), bbox_inches='tight')
plt.close(); print('Fig 7 saved.')

# ─────────────────────────────────────────────
# SECTION 5: Train / Val / Test Split
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 5: Train/Val/Test Split')
print('='*60)

TARGET = 'rent_burden_50plus_pct'

FEATURES = [
    'median_hh_income', 'renter_median_income', 'renter_income_ratio', 'income_gap',
    'median_gross_rent', 'median_contract_rent', 'rent_burden_30plus_pct',
    'renter_share', 'homeownership_rate', 'vacancy_rate', 'unemployment_rate',
    'severe_crowding_rate', 'transit_commute_rate', 'gini_coefficient',
    'eviction_rate', 'market_tightness', 'rent_to_income_ratio',
    'housing_burden_composite', 'renter_vulnerability', 'income_growth_yoy',
    'rent_growth_yoy', 'median_hh_income_lag1', 'renter_median_income_lag1',
    'unemployment_rate_lag1', 'rent_burden_30plus_pct_lag1', 'vacancy_rate_lag1',
    'median_gross_rent_lag1', 'borough_code', 'year', 'covid_year'
]
FEATURES = [f for f in FEATURES if f in df.columns]
print(f'Features used: {len(FEATURES)}')

# Deduplicate column selection
_all_cols = list(dict.fromkeys(FEATURES + [TARGET, 'year', 'nta_name', 'borough_name']))
df_model = df[_all_cols].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f'Model rows: {len(df_model)}')

train_mask = df_model['year'] <= 2019
val_mask   = df_model['year'] == 2020
test_mask  = df_model['year'] >= 2021

train_df = df_model[train_mask].copy()
val_df   = df_model[val_mask].copy()
test_df  = df_model[test_mask].copy()

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].values
X_val   = val_df[FEATURES].copy()
y_val   = val_df[TARGET].values
X_test  = test_df[FEATURES].copy()
y_test  = test_df[TARGET].values

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)
X_trainval  = np.vstack([X_train_imp, X_val_imp])
y_trainval  = np.concatenate([y_train, y_val])

print(f'Train: {X_train_imp.shape}, Val: {X_val_imp.shape}, Test: {X_test_imp.shape}')

# ─────────────────────────────────────────────
# SECTION 6: Model Training
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 6: Model Training')
print('='*60)

def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name='Model'):
    pred_train = model.predict(X_tr)
    pred_test  = model.predict(X_te)
    train_r2   = r2_score(y_tr, pred_train)
    test_r2    = r2_score(y_te, pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_tr, pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_te, pred_test))
    train_mae  = mean_absolute_error(y_tr, pred_train)
    test_mae   = mean_absolute_error(y_te, pred_test)
    print(f'  {model_name}: Train R2={train_r2:.4f} | Test R2={test_r2:.4f} | Test RMSE={test_rmse:.4f}')
    return {'model': model_name, 'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae, 'pred_test': pred_test}

model_results = []

# Random Forest
print('Training Random Forest (n=50)...')
rf_model = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
rf_model.fit(X_trainval, y_trainval)
rf_results = evaluate_model(rf_model, X_trainval, y_trainval, X_test_imp, y_test, 'Random Forest')
model_results.append(rf_results)

# XGBoost
print('Training XGBoost (n=50)...')
xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, random_state=42, verbosity=0)
xgb_model.fit(X_trainval, y_trainval)
xgb_results = evaluate_model(xgb_model, X_trainval, y_trainval, X_test_imp, y_test, 'XGBoost')
model_results.append(xgb_results)

# LightGBM
lgb_results = None
lgb_model = None
if LGB_AVAILABLE:
    print('Training LightGBM (n=50)...')
    lgb_model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                    num_leaves=15, min_child_samples=20, subsample=0.8,
                                    n_jobs=-1, random_state=42, verbose=-1)
    lgb_model.fit(X_trainval, y_trainval)
    lgb_results = evaluate_model(lgb_model, X_trainval, y_trainval, X_test_imp, y_test, 'LightGBM')
    model_results.append(lgb_results)

# Stacking Ensemble
print('Training Stacking Ensemble...')
base_estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, random_state=42, verbosity=0)),
    ('rf',  RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42))
]
if LGB_AVAILABLE and lgb_model is not None:
    base_estimators.append(('lgb', lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                                        num_leaves=15, min_child_samples=20,
                                                        subsample=0.8, n_jobs=-1, random_state=42, verbose=-1)))
stack_model = StackingRegressor(estimators=base_estimators, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
stack_model.fit(X_trainval, y_trainval)
stack_results = evaluate_model(stack_model, X_trainval, y_trainval, X_test_imp, y_test, 'Stacking Ensemble')
model_results.append(stack_results)

# Fig 8: Model Comparison
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'pred_test'} for r in model_results])
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ['test_r2', 'test_rmse', 'test_mae']
titles  = ['Test R2 (higher=better)', 'Test RMSE (lower=better)', 'Test MAE (lower=better)']
colors_bar = ['#2ECC71', '#E74C3C', '#3498DB', '#9B59B6']
for ax, metric, title in zip(axes, metrics, titles):
    bars = ax.bar(results_df['model'], results_df[metric],
                  color=colors_bar[:len(results_df)], edgecolor='white', alpha=0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(results_df['model'], rotation=15, ha='right', fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f'{h:.4f}',
                ha='center', va='bottom', fontsize=8)
plt.suptitle('Fig 8: Model Comparison (Test Set 2021-2022)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig8_model_comparison.png'), bbox_inches='tight')
plt.close(); print('Fig 8 saved.')

# Fig 9: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test, xgb_results['pred_test'], alpha=0.4, color='steelblue', s=18)
mn = min(y_test.min(), xgb_results['pred_test'].min())
mx = max(y_test.max(), xgb_results['pred_test'].max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Rent Burden 50%+'); ax.set_ylabel('Predicted')
ax.set_title(f'Fig 9: XGBoost Actual vs Predicted  R2={xgb_results["test_r2"]:.4f}')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig9_actual_vs_predicted.png'), bbox_inches='tight')
plt.close(); print('Fig 9 saved.')

# ─────────────────────────────────────────────
# SECTION 7: Time-Series Cross-Validation
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 7: 5-Fold TimeSeriesSplit CV')
print('='*60)

df_full_cv = df_model[FEATURES + [TARGET]].copy()
df_full_cv.replace([np.inf, -np.inf], np.nan, inplace=True)
X_full = df_full_cv[FEATURES].values
y_full = df_full_cv[TARGET].values
imp_full = SimpleImputer(strategy='median')
X_full_imp = imp_full.fit_transform(X_full)

tscv5 = TimeSeriesSplit(n_splits=5)
cv_results = {'XGBoost': {'r2': [], 'rmse': []}, 'RandomForest': {'r2': [], 'rmse': []}}

for fold_i, (tr_idx, te_idx) in enumerate(tscv5.split(X_full_imp)):
    X_tr_cv, X_te_cv = X_full_imp[tr_idx], X_full_imp[te_idx]
    y_tr_cv, y_te_cv = y_full[tr_idx], y_full[te_idx]
    xgb_cv = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=0)
    xgb_cv.fit(X_tr_cv, y_tr_cv)
    pred_cv = xgb_cv.predict(X_te_cv)
    cv_results['XGBoost']['r2'].append(r2_score(y_te_cv, pred_cv))
    cv_results['XGBoost']['rmse'].append(np.sqrt(mean_squared_error(y_te_cv, pred_cv)))
    rf_cv = RandomForestRegressor(n_estimators=50, max_depth=6, n_jobs=-1, random_state=42)
    rf_cv.fit(X_tr_cv, y_tr_cv)
    pred_rf_cv = rf_cv.predict(X_te_cv)
    cv_results['RandomForest']['r2'].append(r2_score(y_te_cv, pred_rf_cv))
    cv_results['RandomForest']['rmse'].append(np.sqrt(mean_squared_error(y_te_cv, pred_rf_cv)))
    print(f'  Fold {fold_i+1}: XGB R2={cv_results["XGBoost"]["r2"][-1]:.4f}  RF R2={cv_results["RandomForest"]["r2"][-1]:.4f}')

for model_nm in ['XGBoost', 'RandomForest']:
    r2_arr = cv_results[model_nm]['r2']
    print(f'  {model_nm}: Mean R2={np.mean(r2_arr):.4f} +/- {np.std(r2_arr):.4f}')

# Fig 10
folds = list(range(1, 6))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(folds, cv_results['XGBoost']['r2'],    'o-', color='#E74C3C', linewidth=2, label='XGBoost')
axes[0].plot(folds, cv_results['RandomForest']['r2'],'s-', color='#3498DB', linewidth=2, label='Random Forest')
axes[0].set_xlabel('CV Fold'); axes[0].set_ylabel('R2'); axes[0].set_title('CV R2 per Fold'); axes[0].legend()
axes[1].plot(folds, cv_results['XGBoost']['rmse'],    'o-', color='#E74C3C', linewidth=2, label='XGBoost')
axes[1].plot(folds, cv_results['RandomForest']['rmse'],'s-', color='#3498DB', linewidth=2, label='Random Forest')
axes[1].set_xlabel('CV Fold'); axes[1].set_ylabel('RMSE'); axes[1].set_title('CV RMSE per Fold'); axes[1].legend()
plt.suptitle('Fig 10: TimeSeriesSplit CV Performance (5 Folds)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig10_cv_performance.png'), bbox_inches='tight')
plt.close(); print('Fig 10 saved.')

# ─────────────────────────────────────────────
# SECTION 8: SHAP
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 8: SHAP Explainability')
print('='*60)

shap_importance_df = None
if SHAP_AVAILABLE:
    shap_n = min(300, X_test_imp.shape[0])
    X_shap = X_test_imp[:shap_n]
    explainer  = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=FEATURES, show=False, plot_size=None)
    plt.title('Fig 11: SHAP Beeswarm (XGBoost, Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig11_shap_beeswarm.png'), bbox_inches='tight')
    plt.close(); print('Fig 11 saved.')

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs_shap}
                                       ).sort_values('mean_abs_shap', ascending=False)
    top_shap = shap_importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_shap['feature'][::-1], top_shap['mean_abs_shap'][::-1], color='steelblue', alpha=0.85)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Fig 12: SHAP Feature Importance - Top 20')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig12_shap_bar.png'), bbox_inches='tight')
    plt.close(); print('Fig 12 saved.')

    top3 = shap_importance_df['feature'].head(3).tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, feat in enumerate(top3):
        fi = FEATURES.index(feat)
        axes[i].scatter(X_shap[:, fi], shap_values[:, fi], alpha=0.3, s=12, color='steelblue')
        axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[i].set_xlabel(feat); axes[i].set_ylabel('SHAP Value')
        axes[i].set_title(f'Dependence: {feat}')
    plt.suptitle('Fig 13: SHAP Dependence Plots - Top 3 Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig13_shap_dependence.png'), bbox_inches='tight')
    plt.close(); print('Fig 13 saved.')
    print(f'  Top features: {shap_importance_df["feature"].head(5).tolist()}')
else:
    fi_df = pd.DataFrame({'feature': FEATURES, 'importance': xgb_model.feature_importances_}
                          ).sort_values('importance', ascending=False)
    shap_importance_df = fi_df
    print('SHAP skipped — using XGBoost built-in feature importances.')

# ──��──────────────────────────────────────────
# SECTION 9: Ablation Study
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 9: Ablation Study')
print('='*60)

feature_groups = {
    'income_features':    ['median_hh_income', 'renter_median_income', 'renter_income_ratio', 'income_gap',
                            'income_growth_yoy', 'median_hh_income_lag1', 'renter_median_income_lag1'],
    'rental_market':      ['median_gross_rent', 'median_contract_rent', 'rent_burden_30plus_pct',
                            'rent_to_income_ratio', 'rent_growth_yoy', 'median_gross_rent_lag1',
                            'rent_burden_30plus_pct_lag1'],
    'labor_market':       ['unemployment_rate', 'unemployment_rate_lag1', 'housing_burden_composite'],
    'housing_stock':      ['renter_share', 'homeownership_rate', 'vacancy_rate', 'severe_crowding_rate',
                            'market_tightness', 'vacancy_rate_lag1', 'transit_commute_rate'],
    'temporal_lags':      ['median_hh_income_lag1', 'renter_median_income_lag1', 'unemployment_rate_lag1',
                            'rent_burden_30plus_pct_lag1', 'vacancy_rate_lag1', 'median_gross_rent_lag1'],
    'spatial_identifiers':['borough_code', 'year', 'covid_year']
}
for grp in feature_groups:
    feature_groups[grp] = [f for f in feature_groups[grp] if f in FEATURES]

def train_eval_xgb_abl(feats_subset):
    feat_idx = [FEATURES.index(f) for f in feats_subset if f in FEATURES]
    if not feat_idx:
        return 0.0
    m = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(X_trainval[:, feat_idx], y_trainval)
    return r2_score(y_test, m.predict(X_test_imp[:, feat_idx]))

baseline_r2 = xgb_results['test_r2']
ablation_results = []
for grp_name, grp_feats in feature_groups.items():
    reduced = [f for f in FEATURES if f not in grp_feats]
    abl_r2  = train_eval_xgb_abl(reduced) if reduced else 0.0
    drop    = baseline_r2 - abl_r2
    ablation_results.append({'feature_group': grp_name, 'num_features_removed': len(grp_feats),
                               'test_r2_all': round(baseline_r2, 5), 'test_r2_without': round(abl_r2, 5),
                               'r2_drop': round(drop, 5)})
    print(f'  Without {grp_name}: R2={abl_r2:.4f}  drop={drop:.4f}')

ablation_df = pd.DataFrame(ablation_results).sort_values('r2_drop', ascending=False)

# Fig 14
colors_abl = ['#C0392B' if d > 0 else '#2ECC71' for d in ablation_df['r2_drop']]
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(ablation_df['feature_group'], ablation_df['r2_drop'], color=colors_abl, edgecolor='white', alpha=0.85)
ax.axhline(0, color='black', linewidth=1.2)
ax.set_xlabel('Feature Group Removed'); ax.set_ylabel('R2 Drop')
ax.set_title('Fig 14: Ablation Study — R2 Drop When Removing Feature Groups')
plt.xticks(rotation=20, ha='right')
for bar, val in zip(bars, ablation_df['r2_drop']):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig14_ablation_study.png'), bbox_inches='tight')
plt.close(); print('Fig 14 saved.')

# ─────────────────────────────────────────────
# SECTION 10: Spatial Analysis — Moran's I
# ─────────────────────────────────────────────
print('\n' + '='*60)
print("SECTION 10: Moran's I Spatial Analysis")
print('='*60)

residuals_test = y_test - xgb_results['pred_test']
test_df_res = test_df.copy()
test_df_res['residual'] = residuals_test
boro_residuals = test_df_res.groupby('borough_name')['residual'].mean()

borough_list_sorted = sorted(boro_residuals.index.tolist())
n_boro = len(borough_list_sorted)
adjacency = {'Bronx': ['Manhattan', 'Queens'], 'Brooklyn': ['Manhattan', 'Queens'],
             'Manhattan': ['Bronx', 'Brooklyn', 'Queens'], 'Queens': ['Bronx', 'Brooklyn', 'Manhattan']}
W = np.zeros((n_boro, n_boro))
for i, b_i in enumerate(borough_list_sorted):
    for j, b_j in enumerate(borough_list_sorted):
        if b_j in adjacency.get(b_i, []):
            W[i, j] = 1.0
row_sums = W.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
W_norm = W / row_sums
res_vals = np.array([boro_residuals.get(b, 0.0) for b in borough_list_sorted])
z = res_vals - res_vals.mean()
n_f = float(len(z))
morans_I = (n_f * np.sum(W_norm * np.outer(z, z))) / (W_norm.sum() * np.sum(z**2) + 1e-12)
E_I = -1.0 / (n_f - 1)

from scipy import stats
W_sym = W_norm + W_norm.T
S1 = 0.5 * np.sum(W_sym ** 2)
S2 = np.sum((W_norm.sum(axis=1) + W_norm.sum(axis=0)) ** 2)
S0 = W_norm.sum()
b2 = (np.sum(z**4) / n_f) / ((np.sum(z**2) / n_f)**2 + 1e-12)
Var_I = ((n_f * ((n_f**2 - 3*n_f + 3)*S1 - n_f*S2 + 3*S0**2)
          - b2 * ((n_f**2 - n_f)*S1 - 2*n_f*S2 + 6*S0**2))
         / ((n_f-1)*(n_f-2)*(n_f-3)*S0**2 + 1e-12)) - E_I**2
z_score = (morans_I - E_I) / np.sqrt(max(Var_I, 1e-12))
p_value  = 2 * (1 - stats.norm.cdf(abs(z_score)))
print(f"  Moran's I={morans_I:.4f}  Z={z_score:.4f}  p={p_value:.4f}")

# Fig 15
colors_res = ['#C0392B' if r > 0 else '#2ECC71' for r in boro_residuals.values]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(boro_residuals.index, boro_residuals.values, color=colors_res, edgecolor='white', alpha=0.8)
axes[0].axhline(0, color='black', linewidth=1.2)
axes[0].set_title('Borough-Level Mean Residuals'); axes[0].tick_params(axis='x', rotation=15)
axes[1].hist(residuals_test, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[1].axvline(residuals_test.mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {residuals_test.mean():.4f}')
axes[1].set_title('Residual Distribution'); axes[1].legend()
plt.suptitle(f"Fig 15: Spatial Residuals — Moran's I={morans_I:.4f}, p={p_value:.4f}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig15_spatial_residuals.png'), bbox_inches='tight')
plt.close(); print('Fig 15 saved.')

# ─────────────────────────────────────────────
# SECTION 11: Borough Deep Dive
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 11: Borough Deep Dive')
print('='*60)

year_2022 = df[df['year'] == 2022][['nta_code', 'nta_name', 'borough_name', 'rent_burden_50plus_pct']].dropna()
nta_ranking_2022 = year_2022.sort_values('rent_burden_50plus_pct', ascending=False).reset_index(drop=True)
nta_ranking_2022['rank'] = nta_ranking_2022.index + 1
print('Top 10 worst NTAs in 2022:')
print(nta_ranking_2022[['rank', 'nta_name', 'borough_name', 'rent_burden_50plus_pct']].head(10).to_string(index=False))

nta_2012 = df[df['year'] == 2012][['nta_name', 'borough_name', 'rent_burden_50plus_pct']].set_index('nta_name')
nta_2022i = df[df['year'] == 2022][['nta_name', 'borough_name', 'rent_burden_50plus_pct']].set_index('nta_name')
common_ntas = nta_2012.index.intersection(nta_2022i.index)
change_df = pd.DataFrame({'burden_2012': nta_2012.loc[common_ntas, 'rent_burden_50plus_pct'],
                           'burden_2022': nta_2022i.loc[common_ntas, 'rent_burden_50plus_pct'],
                           'borough':     nta_2012.loc[common_ntas, 'borough_name']}).dropna()
change_df['change'] = change_df['burden_2022'] - change_df['burden_2012']
improved = change_df.sort_values('change').head(5)
worsened = change_df.sort_values('change', ascending=False).head(5)

top10_worst = nta_ranking_2022.head(10)
colors_boro = {'Bronx': '#4472C4', 'Brooklyn': '#ED7D31', 'Manhattan': '#A9D18E', 'Queens': '#FF4444'}
bar_colors  = [colors_boro.get(b, 'gray') for b in top10_worst['borough_name']]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(top10_worst['nta_name'][::-1], top10_worst['rent_burden_50plus_pct'][::-1],
             color=bar_colors[::-1], edgecolor='white', alpha=0.85)
axes[0].set_xlabel('Rent Burden 50%+'); axes[0].set_title('Top 10 Worst NTAs (2022)')
patches = [mpatches.Patch(color=v, label=k) for k, v in colors_boro.items()]
axes[0].legend(handles=patches, fontsize=8, loc='lower right')

combined_change = pd.concat([worsened, improved]).sort_values('change')
bar_cols2 = ['#2ECC71' if c < 0 else '#C0392B' for c in combined_change['change']]
axes[1].barh(combined_change.index, combined_change['change'], color=bar_cols2, edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='black', linewidth=1.2)
axes[1].set_xlabel('Change in Rent Burden (2012–2022)'); axes[1].set_title('Improved vs Worsened NTAs')

plt.suptitle('Fig 16: Borough Deep Dive — NTA Rent Burden Rankings', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig16_borough_deepdive.png'), bbox_inches='tight')
plt.close(); print('Fig 16 saved.')

# ─────────────────────────────────────────────
# SECTION 12: Forecasting 2023–2025
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 12: Forecasting 2023-2025')
print('='*60)

latest_year_data = df[df['year'] == 2022].copy()
_fc_cols = list(dict.fromkeys(FEATURES + [TARGET]))
borough_medians = latest_year_data.groupby('borough_name')[_fc_cols].median().reset_index()
hist_trend = df.groupby(['borough_name', 'year'])[[TARGET]].median().reset_index()

forecast_rows = []
current_state = borough_medians.copy()

for forecast_year in [2023, 2024, 2025]:
    next_state = current_state.copy()
    next_state['year'] = forecast_year
    if 'covid_year' in next_state.columns:
        next_state['covid_year'] = 0
    for lag_col, src_col in [('median_hh_income_lag1', 'median_hh_income'),
                               ('renter_median_income_lag1', 'renter_median_income'),
                               ('unemployment_rate_lag1', 'unemployment_rate'),
                               ('rent_burden_30plus_pct_lag1', 'rent_burden_30plus_pct'),
                               ('vacancy_rate_lag1', 'vacancy_rate'),
                               ('median_gross_rent_lag1', 'median_gross_rent')]:
        if lag_col in FEATURES and src_col in next_state.columns:
            next_state[lag_col] = current_state[src_col]
    if 'median_hh_income' in next_state.columns:
        next_state['median_hh_income'] *= 1.025
    if 'renter_median_income' in next_state.columns:
        next_state['renter_median_income'] *= 1.02
    if 'median_gross_rent' in next_state.columns:
        next_state['median_gross_rent'] *= 1.03
    if 'vacancy_rate' in next_state.columns:
        next_state['market_tightness'] = 1.0 / (next_state['vacancy_rate'].clip(lower=0.001) + 0.005)
    if 'renter_median_income' in next_state.columns and 'median_gross_rent' in next_state.columns:
        next_state['rent_to_income_ratio'] = next_state['median_gross_rent'] * 12 / next_state['renter_median_income'].clip(lower=1)

    X_forecast = next_state[FEATURES].copy()
    X_forecast.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_fc_imp = imputer.transform(X_forecast)
    preds = xgb_model.predict(X_fc_imp)
    for idx_r, (_, row) in enumerate(next_state.iterrows()):
        forecast_rows.append({'borough_name': row['borough_name'], 'year': forecast_year,
                               'predicted_rent_burden_50plus': round(float(preds[idx_r]), 5)})
    current_state = next_state.copy()
    print(f'  {forecast_year}: done.')

forecast_df = pd.DataFrame(forecast_rows)
print(forecast_df.pivot(index='borough_name', columns='year', values='predicted_rent_burden_50plus').round(4))

# Fig 17
fig, ax = plt.subplots(figsize=(13, 7))
for i, boro in enumerate(sorted(forecast_df['borough_name'].unique())):
    hist_boro = hist_trend[hist_trend['borough_name'] == boro].sort_values('year')
    ax.plot(hist_boro['year'], hist_boro[TARGET], marker='o', linewidth=2, markersize=5,
            color=colors_list[i % len(colors_list)], label=boro)
    fc_boro = forecast_df[forecast_df['borough_name'] == boro].sort_values('year')
    last_yr  = hist_boro['year'].max()
    last_val = hist_boro[hist_boro['year'] == last_yr][TARGET].values[0]
    fc_yrs  = [last_yr] + fc_boro['year'].tolist()
    fc_vals = [last_val] + fc_boro['predicted_rent_burden_50plus'].tolist()
    ax.plot(fc_yrs, fc_vals, marker='D', linewidth=2, markersize=6, linestyle='--',
            color=colors_list[i % len(colors_list)], alpha=0.8)
ax.axvspan(2019.5, 2021.5, alpha=0.12, color='red', label='COVID-19')
ax.axvspan(2022.5, 2025.5, alpha=0.08, color='blue', label='Forecast Zone')
ax.axvline(x=2022.5, color='blue', linestyle=':', linewidth=1.8)
ax.set_xlabel('Year'); ax.set_ylabel('Rent Burden 50%+')
ax.set_title('Fig 17: Borough-Level Forecast (2012-2022 Historical + 2023-2025 Forecast)')
ax.legend(loc='upper right', fontsize=9)
ax.set_xticks(list(range(2012, 2026))); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ieee_fig17_forecast.png'), bbox_inches='tight')
plt.close(); print('Fig 17 saved.')

# ─────────────────────────────────────────────
# SECTION 13: Export CSV Results
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('SECTION 13: Exporting Results')
print('='*60)

perf_export = pd.DataFrame([{k: v for k, v in r.items() if k != 'pred_test'} for r in model_results])
perf_export.to_csv(os.path.join(OUTPUT_DIR, 'ieee_model_performance.csv'), index=False)
print('Saved: ieee_model_performance.csv')

shap_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'ieee_shap_importance.csv'), index=False)
print('Saved: ieee_shap_importance.csv')

ablation_df.to_csv(os.path.join(OUTPUT_DIR, 'ieee_ablation_results.csv'), index=False)
print('Saved: ieee_ablation_results.csv')

nta_ranking_2022.to_csv(os.path.join(OUTPUT_DIR, 'ieee_nta_rankings_2022.csv'), index=False)
print('Saved: ieee_nta_rankings_2022.csv')

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
best_row = perf_export.loc[perf_export['test_r2'].idxmax()]
print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
print(f'Dataset: {df.shape[0]} rows x {df.shape[1]} columns | {df["nta_code"].nunique()} NTAs | {df["year"].min()}-{df["year"].max()}')
print(f'\nModel Results (Test Set 2021-2022):')
for _, row in perf_export.iterrows():
    marker = '  <-- BEST' if row['model'] == best_row['model'] else ''
    print(f'  {row["model"]:<22}: R2={row["test_r2"]:.4f}  RMSE={row["test_rmse"]:.4f}{marker}')
print(f'\nCV (XGBoost 5-fold): Mean R2={np.mean(cv_results["XGBoost"]["r2"]):.4f} +/- {np.std(cv_results["XGBoost"]["r2"]):.4f}')
print(f"Moran's I={morans_I:.4f}  p={p_value:.4f}")
print(f'\nFigures saved: ieee_fig1 through ieee_fig17')
print(f'CSVs saved:    ieee_model_performance, ieee_shap_importance, ieee_ablation_results, ieee_nta_rankings_2022')
print('='*60)
print('DONE!')
