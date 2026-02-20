"""
Rebuilds NYC_Housing_IEEE_Expanded.ipynb — clean, complete, no stacking, n_estimators=50
User will run the notebook themselves.
"""
import json, textwrap

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(src).strip()}

cells = []

# ─── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md("""
# AI-Driven Housing Affordability Forecasting in New York City
## IEEE-Standard Analysis | NTA-Level Panel Dataset (2012–2022)

**Dataset:** 2,512 observations · 239 Neighborhood Tabulation Areas · 4 Boroughs
**Target:** Severe Rent Burden (50%+ of income on rent)
**Models:** Random Forest · XGBoost · LightGBM
**Authors:** NYC Housing Research Lab
"""))

# ─── SECTION 1: SETUP ─────────────────────────────────────────────────────────
cells.append(md("## Section 1 — Setup & Data Loading"))

cells.append(code("""
%matplotlib inline
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

try:
    import lightgbm as lgb
    LGB = True
    print(f"LightGBM {lgb.__version__} available")
except ImportError:
    LGB = False
    print("LightGBM not available — will use RF + XGBoost only")

try:
    import shap
    SHAP = True
    print(f"SHAP {shap.__version__} available")
except ImportError:
    SHAP = False
    print("SHAP not available — using built-in feature importance")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 12})
sns.set_style("whitegrid")
sns.set_palette("tab10")

COLORS      = ["#4472C4", "#ED7D31", "#A9D18E", "#FF4444"]
BORO_COLOR  = {"Bronx": "#4472C4", "Brooklyn": "#ED7D31",
               "Manhattan": "#A9D18E", "Queens": "#FF4444"}

print("All imports OK")
"""))

cells.append(code("""
df = pd.read_csv("expanded_data/nta_panel_final.csv")

print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Boroughs  : {sorted(df.borough_name.dropna().unique().tolist())}")
print(f"Years     : {sorted(df.year.dropna().unique().tolist())}")
print(f"NTAs      : {df.nta_code.nunique()}")
print()
print("Missing values (top 10):")
print(df.isnull().sum().sort_values(ascending=False).head(10))
df.head(3)
"""))

# ─── SECTION 2: EDA ───────────────────────────────────────────────────────────
cells.append(md("## Section 2 — Exploratory Data Analysis"))

# Fig 1 — Burden Distribution
cells.append(code("""
# ── Fig 1: Rent Burden Distribution ──────────────────────────────────────────
col  = "rent_burden_30plus_pct"
data = df[col].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(data, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(data.mean(),   color="red",    ls="--", lw=2, label=f"Mean:   {data.mean():.3f}")
axes[0].axvline(data.median(), color="orange", ls="--", lw=2, label=f"Median: {data.median():.3f}")
axes[0].set(xlabel="Rent Burden 30%+ Share", ylabel="Frequency",
            title="(a) Histogram")
axes[0].legend()

# KDE
kde = gaussian_kde(data)
x   = np.linspace(data.min(), data.max(), 300)
axes[1].plot(x, kde(x), color="steelblue", lw=2.5)
axes[1].fill_between(x, kde(x), alpha=0.25, color="steelblue")
axes[1].axvline(data.mean(),   color="red",    ls="--", lw=2, label=f"Mean:   {data.mean():.3f}")
axes[1].axvline(data.median(), color="orange", ls="--", lw=2, label=f"Median: {data.median():.3f}")
axes[1].set(xlabel="Rent Burden 30%+ Share", ylabel="Density", title="(b) KDE")
axes[1].legend()

plt.suptitle("Fig 1: Rent Burden (30%+) Distribution Across All NTAs (2012–2022)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# Fig 2 — Boxplot by Borough
cells.append(code("""
# ── Fig 2: Rent Burden by Borough ────────────────────────────────────────────
boros = sorted(df.borough_name.dropna().unique())

fig, ax = plt.subplots(figsize=(11, 6))
bp = ax.boxplot(
    [df.loc[df.borough_name == b, "rent_burden_50plus_pct"].dropna().values for b in boros],
    labels=boros, patch_artist=True,
    medianprops=dict(color="black", lw=2.5),
    flierprops=dict(marker="o", markersize=2, alpha=0.4)
)
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)

ax.set(xlabel="Borough", ylabel="Severely Cost-Burdened Share (50%+)",
       title="Fig 2: Severe Rent Burden by Borough (2012–2022)")
ax.yaxis.grid(True, ls="--", alpha=0.6)

# Annotate medians
for i, boro in enumerate(boros):
    med = df.loc[df.borough_name == boro, "rent_burden_50plus_pct"].dropna().median()
    ax.text(i + 1, med + 0.005, f"{med:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()
"""))

# Fig 3 — Time Trends
cells.append(code("""
# ── Fig 3: Time Trends by Borough ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i, boro in enumerate(sorted(df.borough_name.dropna().unique())):
    bd = df[df.borough_name == boro]
    for ax, col in zip(axes, ["rent_burden_30plus_pct", "rent_burden_50plus_pct"]):
        tr = bd.groupby("year")[col].mean().reset_index().sort_values("year")
        ax.plot(tr.year, tr[col], marker="o", lw=2, markersize=5,
                color=COLORS[i], label=boro)

for ax, title in zip(axes, ["Cost-Burdened (30%+) Share", "Severely Cost-Burdened (50%+) Share"]):
    ax.axvspan(2019.5, 2021.5, alpha=0.13, color="red", label="COVID-19 Period")
    ax.set(xlabel="Year", ylabel="Share", title=title)
    ax.legend(fontsize=9)
    ax.set_xticks(sorted(df.year.dropna().unique()))
    plt.setp(ax.get_xticklabels(), rotation=45)

plt.suptitle("Fig 3: Rent Burden Trends by Borough (2012–2022)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# Fig 4 — NTA Heatmap
cells.append(code("""
# ── Fig 4: NTA Severe Burden Heatmap (Top 20) ────────────────────────────────
nta_avg = df.groupby("nta_name")["rent_burden_50plus_pct"].mean().dropna()
top20   = nta_avg.nlargest(20).index.tolist()

pivot = (df[df.nta_name.isin(top20)]
         .pivot_table(index="nta_name", columns="year",
                      values="rent_burden_50plus_pct", aggfunc="mean"))
pivot = pivot.loc[nta_avg[top20].sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Severely Cost-Burdened Share (50%+)"}, ax=ax)
ax.set(title="Fig 4: Severe Rent Burden Heatmap — Top 20 Neighborhoods",
       xlabel="Year", ylabel="NTA Name")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""))

# Fig 5 — Worst NTAs per Borough
cells.append(code("""
# ── Fig 5: Worst 10 NTAs per Borough (2022) ──────────────────────────────────
yr22 = df[df.year == 2022].dropna(subset=["rent_burden_50plus_pct"])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, boro in zip(axes.flatten(), sorted(yr22.borough_name.unique())):
    top = (yr22[yr22.borough_name == boro]
           .sort_values("rent_burden_50plus_pct", ascending=False)
           .head(10))
    c = BORO_COLOR.get(boro, "gray")
    ax.barh(top["nta_name"][::-1].values,
            top["rent_burden_50plus_pct"][::-1].values,
            color=c, edgecolor="white", alpha=0.85)
    ax.set(xlabel="Severely Cost-Burdened Share (50%+)",
           title=f"{boro} — Top 10 Worst NTAs (2022)")
    for j, (_, row) in enumerate(top[::-1].iterrows()):
        ax.text(row.rent_burden_50plus_pct + 0.003, j,
                f"{row.rent_burden_50plus_pct:.3f}", va="center", fontsize=8)

plt.suptitle("Fig 5: Top 10 Most Burdened NTAs per Borough (2022)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
"""))

# Fig 6 — Scatter plots
cells.append(code("""
# ── Fig 6: Key Drivers — Scatter Plots ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, boro in enumerate(sorted(df.borough_name.dropna().unique())):
    s = df[df.borough_name == boro].dropna(subset=["median_hh_income",
                                                    "median_gross_rent",
                                                    "rent_burden_50plus_pct"])
    axes[0].scatter(s.median_hh_income / 1000, s.rent_burden_50plus_pct,
                    alpha=0.2, s=10, color=COLORS[i], label=boro)
    axes[1].scatter(s.median_gross_rent, s.rent_burden_50plus_pct,
                    alpha=0.2, s=10, color=COLORS[i], label=boro)

axes[0].set(xlabel="Median HH Income ($000s)", ylabel="Severely Cost-Burdened Share (50%+)",
            title="(a) Income vs. Severe Burden")
axes[1].set(xlabel="Median Gross Rent ($)", ylabel="Severely Cost-Burdened Share (50%+)",
            title="(b) Gross Rent vs. Severe Burden")
axes[0].legend(fontsize=9)
axes[1].legend(fontsize=9)

plt.suptitle("Fig 6: Key Drivers of Severe Rent Burden",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# Fig 7 — Correlation Heatmap
cells.append(code("""
# ── Fig 7: Correlation Heatmap ────────────────────────────────────────────────
corr_cols = [
    "median_hh_income", "median_gross_rent", "rent_burden_30plus_pct",
    "rent_burden_50plus_pct", "vacancy_rate", "unemployment_rate",
    "severe_crowding_rate", "gini_coefficient", "eviction_rate",
    "renter_share", "homeownership_rate"
]
avail = [c for c in corr_cols if c in df.columns]

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones(len(avail), dtype=bool))
sns.heatmap(df[avail].corr().round(2), mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor="gray", ax=ax)
ax.set_title("Fig 7: Correlation Matrix of Key Housing Variables", fontsize=13)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""))

# Fig 8 — Improved vs Worsened NTAs
cells.append(code("""
# ── Fig 8: Most Improved vs. Most Worsened NTAs (2012 → 2022) ───────────────
b12 = (df[df.year == 2012]
       .drop_duplicates("nta_name")
       .set_index("nta_name")["rent_burden_50plus_pct"])
b22 = (df[df.year == 2022]
       .drop_duplicates("nta_name")
       .set_index("nta_name")["rent_burden_50plus_pct"])
bmap = (df[df.year == 2022]
        .drop_duplicates("nta_name")
        .set_index("nta_name")["borough_name"]
        .to_dict())

common = b12.index.intersection(b22.index)
change = (b22[common] - b12[common]).dropna().sort_values()
imp = change.head(10)
wor = change.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, data, title, tcolor in zip(
        axes, [imp, wor],
        ["Top 10 Most Improved (2012→2022)", "Top 10 Most Worsened (2012→2022)"],
        ["#27AE60", "#C0392B"]):
    cb = [BORO_COLOR.get(bmap.get(n, ""), "steelblue") for n in data.index]
    ax.barh(range(len(data)), data.values, color=cb, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.axvline(0, color="black", lw=1.2)
    ax.set(xlabel="Change in Severe Rent Burden Share (2012 → 2022)", title=title)
    ax.title.set_color(tcolor)

patches = [mpatches.Patch(color=v, label=k) for k, v in BORO_COLOR.items()]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))
plt.suptitle("Fig 8: NTA Change in Severe Rent Burden (2012 → 2022)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# Fig 9 — Additional EDA: Rent vs Income divergence
cells.append(code("""
# ── Fig 9: Rent vs Income Divergence (Indexed to 2012=100) ──────────────────
fig, ax = plt.subplots(figsize=(12, 6))
years = sorted(df.year.dropna().unique())

for i, boro in enumerate(sorted(df.borough_name.dropna().unique())):
    bd = df[df.borough_name == boro].groupby("year")[
        ["median_gross_rent", "median_hh_income"]].mean()
    if 2012 not in bd.index:
        continue
    rent_idx   = bd.median_gross_rent / bd.median_gross_rent.loc[2012] * 100
    income_idx = bd.median_hh_income  / bd.median_hh_income.loc[2012]  * 100
    ax.plot(bd.index, rent_idx,   ls="-",  color=COLORS[i], lw=2,
            label=f"{boro} Rent")
    ax.plot(bd.index, income_idx, ls="--", color=COLORS[i], lw=1.5,
            alpha=0.7, label=f"{boro} Income")

ax.axhline(100, color="black", lw=1, ls=":")
ax.axvspan(2019.5, 2021.5, alpha=0.12, color="red")
ax.set(xlabel="Year", ylabel="Index (2012 = 100)",
       title="Fig 9: Rent vs. Income Growth Divergence by Borough (2012–2022)")
ax.legend(fontsize=7.5, ncol=2, loc="upper left")
ax.set_xticks(years)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""))

# ─── SECTION 3: FEATURE ENGINEERING ─────────────────────────────────────────
cells.append(md("## Section 3 — Feature Engineering"))

cells.append(code("""
# ── Engineered Features ──────────────────────────────────────────────────────
df["market_tightness"]         = 1.0 / (df.vacancy_rate.clip(lower=0.001) + 0.005)
df["rent_to_income_ratio"]     = (df.median_gross_rent * 12 /
                                   df.renter_median_income.clip(lower=1))
df["housing_burden_composite"] = df.unemployment_rate + df.severe_crowding_rate
df["renter_vulnerability"]     = (
    (1 - df.renter_income_ratio.clip(0, 1)) * 0.5
    + df.unemployment_rate * 0.3
    + df.severe_crowding_rate * 0.2
)

TARGET = "rent_burden_50plus_pct"

candidate_features = [
    # Income
    "median_hh_income", "renter_median_income", "renter_income_ratio",
    "income_gap", "income_growth_yoy", "median_hh_income_lag1",
    "renter_median_income_lag1",
    # Rental market
    "median_gross_rent", "median_contract_rent", "rent_burden_30plus_pct",
    "rent_to_income_ratio", "rent_growth_yoy", "median_gross_rent_lag1",
    "rent_burden_30plus_pct_lag1",
    # Labor
    "unemployment_rate", "unemployment_rate_lag1", "housing_burden_composite",
    # Housing stock
    "renter_share", "homeownership_rate", "vacancy_rate", "severe_crowding_rate",
    "market_tightness", "vacancy_rate_lag1", "transit_commute_rate",
    # Inequality / evictions
    "gini_coefficient", "eviction_rate",
    # Engineered
    "renter_vulnerability",
    # Temporal / spatial IDs
    "borough_code", "year", "covid_year",
]
FEATURES = [f for f in candidate_features if f in df.columns]
print(f"Features selected : {len(FEATURES)}")
print(FEATURES)
"""))

# ─── SECTION 4: SPLIT ─────────────────────────────────────────────────────────
cells.append(md("## Section 4 — Train / Validation / Test Split"))

cells.append(code("""
# Deduplicate columns (year appears in both FEATURES and the extra cols list)
_cols    = list(dict.fromkeys(FEATURES + [TARGET, "year", "nta_name", "borough_name"]))
df_model = df[_cols].copy().dropna(subset=[TARGET])
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

train_df = df_model[df_model.year <= 2019]
val_df   = df_model[df_model.year == 2020]
test_df  = df_model[df_model.year >= 2021]

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(train_df[FEATURES])
X_val   = imputer.transform(val_df[FEATURES])
X_test  = imputer.transform(test_df[FEATURES])

y_train = train_df[TARGET].values
y_val   = val_df[TARGET].values
y_test  = test_df[TARGET].values

# Combined train+val for final model fit
X_tv = np.vstack([X_train, X_val])
y_tv = np.concatenate([y_train, y_val])

print(f"Train : {X_train.shape}  (years ≤ 2019)")
print(f"Val   : {X_val.shape}   (year  = 2020)")
print(f"Test  : {X_test.shape}  (years ≥ 2021)")
print(f"TV    : {X_tv.shape}    (train+val combined)")
"""))

# ─── SECTION 5: MODEL TRAINING ───────────────────────────────────────────────
cells.append(md("## Section 5 — Model Training"))

cells.append(code("""
# ── Helper: fit and evaluate a model ────────────────────────────────────────
def eval_model(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)
    result = {
        "model"     : name,
        "train_r2"  : round(r2_score(y_tr, p_tr),    4),
        "test_r2"   : round(r2_score(y_te, p_te),    4),
        "test_rmse" : round(float(np.sqrt(mean_squared_error(y_te, p_te))), 5),
        "test_mae"  : round(float(mean_absolute_error(y_te, p_te)),         5),
        "pred"      : p_te,
        "fitted"    : model,
    }
    print(f"  {name:<22}: Train R2={result['train_r2']:.4f}  "
          f"Test R2={result['test_r2']:.4f}  RMSE={result['test_rmse']:.5f}")
    return result

results = []
print("Training models (n_estimators=50, should complete in a few seconds)...")
print()

# Random Forest
rf = RandomForestRegressor(
    n_estimators=50, max_depth=8, min_samples_leaf=3,
    n_jobs=-1, random_state=42
)
results.append(eval_model(rf, X_tv, y_tv, X_test, y_test, "Random Forest"))

# XGBoost
xgb_m = xgb.XGBRegressor(
    n_estimators=50, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbosity=0
)
results.append(eval_model(xgb_m, X_tv, y_tv, X_test, y_test, "XGBoost"))

# LightGBM
if LGB:
    lgb_m = lgb.LGBMRegressor(
        n_estimators=50, max_depth=5, learning_rate=0.1,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    )
    results.append(eval_model(lgb_m, X_tv, y_tv, X_test, y_test, "LightGBM"))

res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred", "fitted")}
                        for r in results])
print()
print("=" * 60)
print(res_df[["model", "train_r2", "test_r2", "test_rmse", "test_mae"]].to_string(index=False))
print("=" * 60)

# Keep best model for downstream analysis
best_res = max(results, key=lambda r: r["test_r2"])
best_model = best_res["fitted"]
best_pred  = best_res["pred"]
print(f"\\nBest model: {best_res['model']}  (Test R2={best_res['test_r2']:.4f})")
"""))

# Fig 10 — Model Comparison
cells.append(code("""
# ── Fig 10: Model Comparison Bar Charts ─────────────────────────────────────
bar_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#9B59B6"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, title in zip(
        axes,
        ["test_r2",  "test_rmse",  "test_mae"],
        ["Test R² (↑ better)", "Test RMSE (↓ better)", "Test MAE (↓ better)"]):
    bars = ax.bar(res_df["model"], res_df[metric],
                  color=bar_colors[:len(res_df)], edgecolor="white", alpha=0.9, width=0.55)
    ax.set_title(title, fontsize=11)
    ax.set_xticklabels(res_df["model"], rotation=20, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(h * 0.01, 0.001),
                f"{h:.4f}", ha="center", va="bottom", fontsize=8.5)

plt.suptitle("Fig 10: Model Comparison on Test Set (2021–2022)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# Fig 11 — Actual vs Predicted
cells.append(code("""
# ── Fig 11: Actual vs. Predicted (Best Model) ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter
mn = min(y_test.min(), best_pred.min())
mx = max(y_test.max(), best_pred.max())
axes[0].scatter(y_test, best_pred, alpha=0.4, color="steelblue", s=18)
axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Prediction")
axes[0].set(xlabel="Actual Rent Burden 50%+", ylabel="Predicted",
            title=f"(a) Actual vs. Predicted — {best_res['model']}")
axes[0].legend()
r2 = r2_score(y_test, best_pred)
axes[0].text(0.05, 0.90, f"R² = {r2:.4f}", transform=axes[0].transAxes,
             fontsize=11, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# Residuals
resid = y_test - best_pred
axes[1].scatter(best_pred, resid, alpha=0.4, color="darkorange", s=18)
axes[1].axhline(0, color="red", ls="--", lw=2)
axes[1].set(xlabel="Predicted Value", ylabel="Residual (Actual − Predicted)",
            title="(b) Residuals vs. Fitted")

plt.suptitle(f"Fig 11: Actual vs. Predicted & Residuals — {best_res['model']}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# ─── SECTION 6: CROSS-VALIDATION ─────────────────────────────────────────────
cells.append(md("## Section 6 — 5-Fold TimeSeriesSplit Cross-Validation"))

cells.append(code("""
# ── TimeSeriesSplit CV ─────────────────────────────────��──────────────────────
df_cv = df_model[FEATURES + [TARGET]].copy().replace([np.inf, -np.inf], np.nan)
Xf    = SimpleImputer(strategy="median").fit_transform(df_cv[FEATURES])
yf    = df_cv[TARGET].values

tscv = TimeSeriesSplit(n_splits=5)

cv_models = {
    "XGBoost"     : xgb.XGBRegressor(n_estimators=50, max_depth=5, verbosity=0, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
}
if LGB:
    cv_models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=50, max_depth=5, num_leaves=31, verbose=-1, random_state=42)

cv_results = {nm: {"r2": [], "rmse": []} for nm in cv_models}

for fold, (tr_idx, te_idx) in enumerate(tscv.split(Xf)):
    line = f"Fold {fold + 1}: "
    for nm, m in cv_models.items():
        import copy
        m_fold = copy.deepcopy(m)
        m_fold.fit(Xf[tr_idx], yf[tr_idx])
        p = m_fold.predict(Xf[te_idx])
        r2   = r2_score(yf[te_idx], p)
        rmse = float(np.sqrt(mean_squared_error(yf[te_idx], p)))
        cv_results[nm]["r2"].append(r2)
        cv_results[nm]["rmse"].append(rmse)
        line += f"  {nm} R2={r2:.4f}"
    print(line)

print()
for nm in cv_models:
    r2s = cv_results[nm]["r2"]
    print(f"  {nm:<22}: Mean R2={np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
"""))

# Fig 12 — CV Results
cells.append(code("""
# ── Fig 12: CV Performance ────────────────────────────────────────────────────
folds   = list(range(1, 6))
cv_clrs = ["#E74C3C", "#3498DB", "#2ECC71"]
cv_mks  = ["o", "s", "^"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, nm in enumerate(cv_models):
    c, mk = cv_clrs[i], cv_mks[i]
    axes[0].plot(folds, cv_results[nm]["r2"],   mk + "-", color=c, lw=2, markersize=7, label=nm)
    axes[1].plot(folds, cv_results[nm]["rmse"],  mk + "-", color=c, lw=2, markersize=7, label=nm)
    axes[0].axhline(np.mean(cv_results[nm]["r2"]),  color=c, ls="--", alpha=0.45)
    axes[1].axhline(np.mean(cv_results[nm]["rmse"]), color=c, ls="--", alpha=0.45)

axes[0].set(xlabel="CV Fold", ylabel="R²",   title="R² per Fold"); axes[0].legend()
axes[1].set(xlabel="CV Fold", ylabel="RMSE", title="RMSE per Fold"); axes[1].legend()
for ax in axes: ax.set_xticks(folds)

plt.suptitle("Fig 12: 5-Fold TimeSeriesSplit Cross-Validation Performance",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# ─── SECTION 7: FEATURE IMPORTANCE ──────────────────────────────────────────
cells.append(md("## Section 7 — Feature Importance (SHAP / Built-in)"))

cells.append(code("""
# ── Fig 13: Feature Importance ────────────────────────────────────────────────
if SHAP:
    explainer  = shap.TreeExplainer(xgb_m)
    shap_vals  = explainer.shap_values(X_test[:300])
    mean_shap  = np.abs(shap_vals).mean(0)
    fi_df = pd.DataFrame({"feature": FEATURES, "importance": mean_shap})
    fi_df = fi_df.sort_values("importance", ascending=False)
    xlabel_str = "Mean |SHAP Value|"
    title_str  = "Fig 13: SHAP Feature Importance — Top 15 (XGBoost)"
else:
    fi_df = pd.DataFrame({"feature": FEATURES,
                           "importance": xgb_m.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=False)
    xlabel_str = "Feature Importance (Gain)"
    title_str  = "Fig 13: XGBoost Feature Importance — Top 15"

top15 = fi_df.head(15)
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top15["feature"][::-1].values, top15["importance"][::-1].values,
               color="steelblue", alpha=0.85, edgecolor="white")
ax.set(xlabel=xlabel_str, title=title_str)
for bar in bars:
    w = bar.get_width()
    ax.text(w + max(w * 0.01, 1e-4), bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.show()

print("Top 5 features:")
for _, row in fi_df.head(5).iterrows():
    print(f"  {row.feature:<35}: {row.importance:.4f}")
"""))

# ─── SECTION 8: ABLATION ─────────────────────────────────────────────────────
cells.append(md("## Section 8 — Ablation Study"))

cells.append(code("""
# ── Ablation: remove one feature group at a time ─────────────────────────────
groups = {
    "income_features": ["median_hh_income", "renter_median_income", "renter_income_ratio",
                         "income_gap", "income_growth_yoy",
                         "median_hh_income_lag1", "renter_median_income_lag1"],
    "rental_market"  : ["median_gross_rent", "median_contract_rent", "rent_burden_30plus_pct",
                         "rent_to_income_ratio", "rent_growth_yoy",
                         "median_gross_rent_lag1", "rent_burden_30plus_pct_lag1"],
    "labor_market"   : ["unemployment_rate", "unemployment_rate_lag1",
                         "housing_burden_composite"],
    "housing_stock"  : ["renter_share", "homeownership_rate", "vacancy_rate",
                         "severe_crowding_rate", "market_tightness",
                         "vacancy_rate_lag1", "transit_commute_rate"],
    "temporal_lags"  : ["median_hh_income_lag1", "renter_median_income_lag1",
                         "unemployment_rate_lag1", "rent_burden_30plus_pct_lag1",
                         "vacancy_rate_lag1", "median_gross_rent_lag1"],
    "spatial_ids"    : ["borough_code", "year", "covid_year"],
}
# Keep only features actually in FEATURES
groups = {k: [f for f in v if f in FEATURES] for k, v in groups.items()}

baseline_r2 = best_res["test_r2"]
abl_rows    = []

for grp, feats in groups.items():
    reduced = [f for f in FEATURES if f not in feats]
    if not reduced:
        abl_r2 = 0.0
    else:
        idx   = [FEATURES.index(f) for f in reduced]
        m_abl = xgb.XGBRegressor(n_estimators=50, max_depth=5, verbosity=0, random_state=42)
        m_abl.fit(X_tv[:, idx], y_tv)
        abl_r2 = r2_score(y_test, m_abl.predict(X_test[:, idx]))
    drop = round(baseline_r2 - abl_r2, 4)
    abl_rows.append({"group": grp, "features_removed": len(feats),
                     "r2_without": round(abl_r2, 4), "r2_drop": drop})
    print(f"  Without {grp:<18}: R2={abl_r2:.4f}  drop={drop:+.4f}")

abl_df = pd.DataFrame(abl_rows).sort_values("r2_drop", ascending=False)
print()
print(abl_df.to_string(index=False))
"""))

# Fig 14 — Ablation
cells.append(code("""
# ── Fig 14: Ablation Study Chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bar_c = ["#C0392B" if d > 0 else "#2ECC71" for d in abl_df["r2_drop"]]
bars  = ax.bar(abl_df["group"], abl_df["r2_drop"],
               color=bar_c, edgecolor="white", alpha=0.85, width=0.5)
ax.axhline(0, color="black", lw=1.2)
ax.set(xlabel="Feature Group Removed",
       ylabel="R² Drop (Baseline − Without Group)",
       title="Fig 14: Ablation Study — Feature Group Importance")
plt.xticks(rotation=20, ha="right")
for bar, val in zip(bars, abl_df["r2_drop"]):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            h + (0.001 if h >= 0 else -0.004),
            f"{val:+.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()
"""))

# ─── SECTION 9: SPATIAL ANALYSIS ─────────────────────────────────────────────
cells.append(md("## Section 9 — Spatial Analysis (Moran's I)"))

cells.append(code("""
# ── Residuals & Moran's I ─────────────────────────────────────────────────────
resid    = y_test - best_pred
tcopy    = test_df.copy()
tcopy["resid"] = resid

# Borough-level mean residuals
bres = tcopy.groupby("borough_name")["resid"].mean()
print("Borough mean residuals:")
print(bres.round(5))

# Moran's I (manual — borough contiguity)
bl  = sorted(bres.index)
adj = {
    "Bronx"    : ["Manhattan", "Queens"],
    "Brooklyn" : ["Manhattan", "Queens"],
    "Manhattan": ["Bronx", "Brooklyn", "Queens"],
    "Queens"   : ["Bronx", "Brooklyn", "Manhattan"],
}
n = len(bl)
W = np.zeros((n, n))
for i, a in enumerate(bl):
    for j, b in enumerate(bl):
        if b in adj.get(a, []):
            W[i, j] = 1.0
rs = W.sum(1, keepdims=True); rs[rs == 0] = 1
W = W / rs

z_arr    = np.array([bres.get(b, 0.0) for b in bl])
z_arr    = z_arr - z_arr.mean()
morans_I = (len(z_arr) * np.sum(W * np.outer(z_arr, z_arr))
            / (W.sum() * np.sum(z_arr ** 2) + 1e-12))
E_I      = -1 / (len(z_arr) - 1)
print(f"\\nMoran's I = {morans_I:.4f}  (Expected under no autocorrelation = {E_I:.4f})")
if morans_I > E_I:
    print("→ Positive spatial autocorrelation detected (similar values cluster)")
else:
    print("→ Negative/no spatial autocorrelation")
"""))

# Fig 15 — Spatial
cells.append(code("""
# ── Fig 15: Spatial Residual Analysis ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Borough mean residuals bar
colors_b = ["#C0392B" if r > 0 else "#2ECC71" for r in bres.values]
axes[0].bar(bres.index, bres.values, color=colors_b, edgecolor="white", alpha=0.85)
axes[0].axhline(0, color="black", lw=1.2)
for i, (boro, v) in enumerate(bres.items()):
    axes[0].text(i, v + (0.001 if v >= 0 else -0.003),
                 f"{v:.4f}", ha="center", fontsize=9)
axes[0].set(xlabel="Borough", ylabel="Mean Residual",
            title=f"(a) Borough Mean Residuals  |  Moran's I = {morans_I:.4f}")

# Residual histogram
axes[1].hist(resid, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0,           color="red",    ls="--", lw=2, label="Zero")
axes[1].axvline(resid.mean(), color="orange", ls="--", lw=2,
                label=f"Mean: {resid.mean():.4f}")
axes[1].set(xlabel="Residual (Actual − Predicted)", ylabel="Count",
            title="(b) Residual Distribution")
axes[1].legend()

plt.suptitle("Fig 15: Spatial Residual Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

# ─── SECTION 10: BOROUGH DEEP DIVE ───────────────────────────────────────────
cells.append(md("## Section 10 — Borough Deep Dive (2022)"))

cells.append(code("""
# ── Fig 16: Borough Deep Dive ─────────────────────────────────────────────────
yr22   = df[df.year == 2022][["nta_name", "borough_name",
                               "rent_burden_50plus_pct",
                               "median_hh_income",
                               "median_gross_rent"]].dropna()
ranking = yr22.sort_values("rent_burden_50plus_pct", ascending=False).reset_index(drop=True)
ranking.index += 1
top10   = ranking.head(10)

fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Top 10 NTAs
bar_c = [BORO_COLOR.get(b, "gray") for b in top10["borough_name"][::-1].values]
axes[0].barh(top10["nta_name"][::-1].values,
             top10["rent_burden_50plus_pct"][::-1].values,
             color=bar_c, edgecolor="white", alpha=0.85)
for j, (_, row) in enumerate(top10[::-1].iterrows()):
    axes[0].text(row.rent_burden_50plus_pct + 0.002, j,
                 f"{row.rent_burden_50plus_pct:.3f}", va="center", fontsize=9)
patches = [mpatches.Patch(color=v, label=k) for k, v in BORO_COLOR.items()]
axes[0].legend(handles=patches, fontsize=9)
axes[0].set(xlabel="Severely Cost-Burdened Share (50%+)",
            title="(a) Top 10 Most Burdened NTAs (2022)")

# Boxplot by borough
boros = sorted(yr22.borough_name.unique())
bp = axes[1].boxplot(
    [yr22.loc[yr22.borough_name == b, "rent_burden_50plus_pct"].values for b in boros],
    labels=boros, patch_artist=True,
    medianprops=dict(color="black", lw=2)
)
for patch, c in zip(bp["boxes"], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.75)
axes[1].set(xlabel="Borough", ylabel="Severely Cost-Burdened Share",
            title="(b) Distribution by Borough (2022)")

plt.suptitle("Fig 16: Borough Deep Dive — Rent Burden 2022",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# Summary table
print("\\nBorough Summary (2022):")
print(yr22.groupby("borough_name")[["rent_burden_50plus_pct",
                                    "median_hh_income",
                                    "median_gross_rent"]].median().round(3))
"""))

# ─── SECTION 11: FORECASTING ─────────────────────────────────────────────────
cells.append(md("## Section 11 — Forecasting 2023–2025"))

cells.append(code("""
# ── Build 2023-2025 Forecast ─────────────────────────────────────────────────
_fc_cols = list(dict.fromkeys(FEATURES + [TARGET]))
bmed     = df[df.year == 2022].groupby("borough_name")[_fc_cols].median().reset_index()
hist     = (df.groupby(["borough_name", "year"])[[TARGET]]
            .median().reset_index())

fc_rows = []
cur     = bmed.copy()

for yr in [2023, 2024, 2025]:
    nxt = cur.copy()
    nxt["year"] = yr
    if "covid_year" in nxt.columns:
        nxt["covid_year"] = 0

    # Advance lag features
    lag_pairs = [
        ("median_hh_income_lag1",       "median_hh_income"),
        ("renter_median_income_lag1",    "renter_median_income"),
        ("unemployment_rate_lag1",       "unemployment_rate"),
        ("rent_burden_30plus_pct_lag1",  "rent_burden_30plus_pct"),
        ("vacancy_rate_lag1",            "vacancy_rate"),
        ("median_gross_rent_lag1",       "median_gross_rent"),
    ]
    for lag, src in lag_pairs:
        if lag in FEATURES and src in nxt.columns:
            nxt[lag] = cur[src]

    # Apply simple growth assumptions
    growth = {"median_hh_income": 1.025, "renter_median_income": 1.020,
              "median_gross_rent": 1.030, "median_contract_rent": 1.025}
    for col, g in growth.items():
        if col in nxt.columns:
            nxt[col] = nxt[col] * g

    Xfc  = nxt[FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    preds = best_model.predict(imputer.transform(Xfc))

    for idx_r, (_, row) in enumerate(nxt.iterrows()):
        fc_rows.append({
            "borough_name": row.borough_name,
            "year"        : yr,
            "predicted"   : round(float(preds[idx_r]), 4),
        })
    cur = nxt.copy()

fc_df = pd.DataFrame(fc_rows)
print("Forecast (median per borough):")
print(fc_df.pivot(index="borough_name", columns="year",
                  values="predicted").round(4))
"""))

# Fig 17 — Forecast
cells.append(code("""
# ── Fig 17: Forecast Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

for i, boro in enumerate(sorted(fc_df.borough_name.unique())):
    h  = hist[hist.borough_name == boro].sort_values("year")
    fc = fc_df[fc_df.borough_name == boro].sort_values("year")

    # Historical line
    ax.plot(h.year, h[TARGET], marker="o", lw=2, markersize=5,
            color=COLORS[i], label=boro)

    # Forecast continuation
    last_year  = h.year.max()
    last_val   = h.loc[h.year == last_year, TARGET].values[0]
    fc_years   = [last_year] + fc.year.tolist()
    fc_vals    = [last_val]  + fc.predicted.tolist()
    ax.plot(fc_years, fc_vals, "D--", lw=2, markersize=6,
            color=COLORS[i], alpha=0.80)

ax.axvspan(2019.5, 2021.5, alpha=0.12, color="red",  label="COVID-19")
ax.axvspan(2022.5, 2025.5, alpha=0.07, color="blue", label="Forecast Zone")
ax.axvline(2022.5, color="blue", ls=":", lw=1.8)
ax.set(xlabel="Year", ylabel="Severely Cost-Burdened Share (50%+)",
       title="Fig 17: Borough-Level Forecast 2023–2025")
ax.legend(loc="upper right", fontsize=9)
ax.set_xticks(range(2012, 2026))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""))

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
cells.append(md("## Final Summary"))

cells.append(code("""
best_row = res_df.loc[res_df.test_r2.idxmax()]

print("=" * 65)
print("  FINAL SUMMARY — NYC Housing Affordability Forecasting (IEEE)")
print("=" * 65)
print(f"  Dataset : {df.shape[0]:,} rows | {df.nta_code.nunique()} NTAs | "
      f"{int(df.year.min())}–{int(df.year.max())}")

print()
print("  Model Performance (Test Set 2021–2022):")
for _, r in res_df.iterrows():
    mark = "  ← BEST" if r.model == best_row.model else ""
    print(f"    {r.model:<22}: R2={r.test_r2:.4f}  "
          f"RMSE={r.test_rmse:.5f}  MAE={r.test_mae:.5f}{mark}")

print()
print("  Cross-Validation (5-Fold TimeSeriesSplit):")
for nm in cv_models:
    r2s = cv_results[nm]["r2"]
    print(f"    {nm:<22}: R2={np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

print()
print("  Top Feature Groups (Ablation):")
for _, r in abl_df.head(3).iterrows():
    print(f"    Remove {r.group:<18}: R2 drop = {r.r2_drop:+.4f}")

print()
print(f"  Moran's I (spatial autocorrelation) = {morans_I:.4f}")

print()
print("  2025 Forecast (median severe burden per borough):")
fc25 = fc_df[fc_df.year == 2025].set_index("borough_name")["predicted"]
for b, v in fc25.items():
    print(f"    {b:<15}: {v:.4f}")

print("=" * 65)
"""))

# ─── BUILD NOTEBOOK ───────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "cells": cells
}

out = "NYC_Housing_IEEE_Expanded.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {out} with {len(cells)} cells")
print("Sections:")
for c in cells:
    if c["cell_type"] == "markdown":
        src = c["source"]
        if src.startswith("## "):
            print(f"  {src.splitlines()[0]}")
