"""
Generates all 10 publication-quality figures into ./figures/
Run: /opt/anaconda3/bin/python generate_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import copy, os, warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb; LGB = True
except ImportError:
    LGB = False

os.makedirs("figures", exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "serif",
    "font.size"       : 10,
    "axes.titlesize"  : 12,
    "axes.titleweight": "bold",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 150,
    "savefig.dpi"     : 150,
    "savefig.bbox"    : "tight",
})

BORO_COLOR = {
    "Bronx":     "#E63946",
    "Brooklyn":  "#457B9D",
    "Manhattan": "#2A9D8F",
    "Queens":    "#E9C46A",
}
COLORS = list(BORO_COLOR.values())
BOROS  = ["Bronx", "Brooklyn", "Manhattan", "Queens"]
TARGET = "rent_burden_50plus_pct"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("expanded_data/nta_panel_final.csv")
print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Target Distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig1_distribution.png ...")
data = df[TARGET].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Severe Rent Burden (≥50%) Distribution — 2,512 NTA-Year Observations",
             fontsize=12, fontweight="bold", y=1.02)

axes[0].hist(data, bins=50, color="#457B9D", edgecolor="white", alpha=0.8, density=True)
xk = np.linspace(data.min(), data.max(), 300)
axes[0].plot(xk, gaussian_kde(data)(xk), color="#E63946", lw=2.5, label="KDE")
axes[0].axvline(data.mean(),   color="#E63946", ls="--", lw=2, label=f"Mean {data.mean():.3f}")
axes[0].axvline(data.median(), color="#E9C46A", ls="--", lw=2, label=f"Median {data.median():.3f}")
axes[0].set(xlabel="Severely Cost-Burdened Share (≥50%)", ylabel="Density",
            title=f"(a) Histogram + KDE  |  Skewness = {stats.skew(data):+.3f}")
axes[0].legend(fontsize=9)

parts = axes[1].violinplot(
    [df[df.borough_name == b][TARGET].dropna().values for b in BOROS],
    positions=range(4), showmedians=True, showextrema=True)
for pc, c in zip(parts["bodies"], COLORS):
    pc.set_facecolor(c); pc.set_alpha(0.7)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(BOROS)
axes[1].set(xlabel="Borough", ylabel="Severely Cost-Burdened Share (≥50%)",
            title="(b) Distribution by Borough")

plt.tight_layout()
plt.savefig("figures/fig1_distribution.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Time Trends
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig2_trends.png ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Rent Burden Trends by Borough (2012–2022) — Shaded Bands = ±1 SD",
             fontsize=12, fontweight="bold", y=1.02)

for i, boro in enumerate(BOROS):
    bd = df[df.borough_name == boro]
    for ax, col in zip(axes, ["rent_burden_30plus_pct", TARGET]):
        grp = bd.groupby("year")[col]
        mn, sd = grp.mean(), grp.std()
        ax.plot(mn.index, mn.values, "o-", lw=2.2, markersize=5, color=COLORS[i], label=boro)
        ax.fill_between(mn.index, mn - sd, mn + sd, alpha=0.12, color=COLORS[i])

for ax, title in zip(axes, ["Cost-Burdened (≥30%) Share", "Severely Cost-Burdened (≥50%) Share"]):
    ax.axvspan(2019.5, 2021.5, alpha=0.15, color="#E63946", label="COVID-19")
    ax.set(xlabel="Year", ylabel="Share of Renter Households", title=title)
    ax.set_xticks(sorted(df.year.dropna().unique()))
    plt.setp(ax.get_xticklabels(), rotation=40)
    ax.legend(fontsize=8.5)

plt.tight_layout()
plt.savefig("figures/fig2_trends.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Scatter Plots
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig3_scatter.png ...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Primary Demand & Supply Drivers of Severe Rent Burden",
             fontsize=12, fontweight="bold", y=1.02)

for i, boro in enumerate(BOROS):
    s = df[df.borough_name == boro].dropna(subset=["median_hh_income","median_gross_rent",TARGET])
    axes[0].scatter(s.median_hh_income / 1000, s[TARGET], alpha=0.25, s=12, color=COLORS[i], label=boro)
    axes[1].scatter(s.median_gross_rent,        s[TARGET], alpha=0.25, s=12, color=COLORS[i], label=boro)

for ax, xcol, div, xlabel in zip(
        axes, ["median_hh_income","median_gross_rent"], [1000,1],
        ["Median HH Income ($000s)","Median Gross Rent ($)"]):
    c = df.dropna(subset=[xcol, TARGET])
    x = c[xcol].values / div; y = c[TARGET].values
    m, b, r, *_ = stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 200)
    ax.plot(xl, m*xl+b, "k--", lw=2, label=f"OLS  r={r:.2f}")
    ax.set(xlabel=xlabel, ylabel="Severely Cost-Burdened Share (≥50%)")
    ax.legend(fontsize=8.5)

axes[0].set_title("(a) Income vs. Severe Burden")
axes[1].set_title("(b) Gross Rent vs. Severe Burden")
plt.tight_layout()
plt.savefig("figures/fig3_scatter.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig4_correlation.png ...")
corr_vars = [TARGET,"rent_burden_30plus_pct","median_hh_income","median_gross_rent",
             "renter_income_ratio","vacancy_rate","unemployment_rate",
             "gini_coefficient","severe_crowding_rate","renter_share"]
avail = [c for c in corr_vars if c in df.columns]
corr  = df[avail].corr().round(2)
mask  = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
            cbar_kws={"label":"Pearson r","shrink":0.8}, ax=ax)
ax.set_title("Pearson Correlation Matrix — 10 Key Housing Variables (n=2,512)", fontsize=12, fontweight="bold")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("figures/fig4_correlation.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS (needed for figs 5–10)
# ══════════════════════════════════════════════════════════════════════════════
print("Engineering features and training models ...")

df["market_tightness"]         = 1.0 / (df["vacancy_rate"].clip(lower=0.001) + 0.005)
df["rent_to_income_ratio"]     = df["median_gross_rent"] * 12 / df["renter_median_income"].clip(lower=1)
df["housing_burden_composite"] = df["unemployment_rate"] + df["severe_crowding_rate"]
df["renter_vulnerability"]     = ((1 - df["renter_income_ratio"].clip(0,1))*0.5
                                   + df["unemployment_rate"]*0.3
                                   + df["severe_crowding_rate"]*0.2)

CANDIDATE = [
    "median_hh_income","renter_median_income","renter_income_ratio","income_gap","income_growth_yoy",
    "median_hh_income_lag1","renter_median_income_lag1",
    "median_gross_rent","median_contract_rent","rent_burden_30plus_pct","rent_to_income_ratio",
    "rent_growth_yoy","median_gross_rent_lag1","rent_burden_30plus_pct_lag1",
    "unemployment_rate","unemployment_rate_lag1","housing_burden_composite",
    "renter_share","homeownership_rate","vacancy_rate","severe_crowding_rate",
    "market_tightness","vacancy_rate_lag1","transit_commute_rate",
    "gini_coefficient","eviction_rate","renter_vulnerability",
    "borough_code","year","covid_year",
]
FEATURES = [f for f in CANDIDATE if f in df.columns]

_cols    = list(dict.fromkeys(FEATURES + [TARGET,"year","nta_name","borough_name"]))
df_model = df[_cols].copy().dropna(subset=[TARGET])
df_model.replace([np.inf,-np.inf], np.nan, inplace=True)

train_df = df_model[df_model.year <= 2019]
val_df   = df_model[df_model.year == 2020]
test_df  = df_model[df_model.year >= 2021]

imputer  = SimpleImputer(strategy="median")
X_train  = imputer.fit_transform(train_df[FEATURES])
X_val    = imputer.transform(val_df[FEATURES])
X_test   = imputer.transform(test_df[FEATURES])
y_train  = train_df[TARGET].values; y_val = val_df[TARGET].values; y_test = test_df[TARGET].values
X_tv = np.vstack([X_train, X_val]); y_tv = np.concatenate([y_train, y_val])

results = []
for name, model in [
    ("Random Forest", RandomForestRegressor(n_estimators=50,max_depth=8,min_samples_leaf=3,n_jobs=-1,random_state=42)),
    ("XGBoost",       xgb.XGBRegressor(n_estimators=50,max_depth=5,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,random_state=42,verbosity=0)),
]:
    model.fit(X_tv, y_tv)
    p_tr = model.predict(X_tv); p_te = model.predict(X_test)
    results.append({"model":name,"train_r2":round(r2_score(y_tv,p_tr),4),
                    "test_r2":round(r2_score(y_test,p_te),4),
                    "test_rmse":round(float(np.sqrt(mean_squared_error(y_test,p_te))),5),
                    "test_mae":round(float(np.mean(np.abs(y_test-p_te))),5),
                    "pred":p_te,"fitted":model})
    print(f"  {name}: Test R²={results[-1]['test_r2']:.4f}")

if LGB:
    lgb_m = lgb.LGBMRegressor(n_estimators=50,max_depth=5,num_leaves=31,learning_rate=0.1,
                                subsample=0.8,colsample_bytree=0.8,n_jobs=-1,random_state=42,verbose=-1)
    lgb_m.fit(X_tv, y_tv)
    p_tr = lgb_m.predict(X_tv); p_te = lgb_m.predict(X_test)
    results.append({"model":"LightGBM","train_r2":round(r2_score(y_tv,p_tr),4),
                    "test_r2":round(r2_score(y_test,p_te),4),
                    "test_rmse":round(float(np.sqrt(mean_squared_error(y_test,p_te))),5),
                    "test_mae":round(float(np.mean(np.abs(y_test-p_te))),5),
                    "pred":p_te,"fitted":lgb_m})
    print(f"  LightGBM: Test R²={results[-1]['test_r2']:.4f}")

res_df     = pd.DataFrame([{k:v for k,v in r.items() if k not in ("pred","fitted")} for r in results])
best_res   = max(results, key=lambda r: r["test_r2"])
best_pred  = best_res["pred"]
xgb_res    = next(r for r in results if r["model"]=="XGBoost")
xgb_m      = xgb_res["fitted"]

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig5_model_comparison.png ...")
pal = ["#2A9D8F","#E63946","#457B9D"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Performance on Held-Out Test Set (2021–2022, n=478)",
             fontsize=12, fontweight="bold")
for ax, metric, title in zip(axes,
        ["test_r2","test_rmse","test_mae"],
        ["Test R² (↑ better)","Test RMSE (↓ better)","Test MAE (↓ better)"]):
    bars = ax.bar(res_df["model"], res_df[metric], color=pal[:len(res_df)],
                  edgecolor="white", alpha=0.9, width=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xticklabels(res_df["model"], rotation=20, ha="right", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+max(h*0.01,0.001),
                f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig5_model_comparison.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — CV Performance
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig6_cv_performance.png ...")
from sklearn.model_selection import TimeSeriesSplit
df_cv = df_model[FEATURES+[TARGET]].copy().replace([np.inf,-np.inf],np.nan)
Xf    = SimpleImputer(strategy="median").fit_transform(df_cv[FEATURES])
yf    = df_cv[TARGET].values
tscv  = TimeSeriesSplit(n_splits=5)

cv_models_dict = {
    "XGBoost":       xgb.XGBRegressor(n_estimators=50,max_depth=5,verbosity=0,random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50,max_depth=8,n_jobs=-1,random_state=42),
}
if LGB:
    cv_models_dict["LightGBM"] = lgb.LGBMRegressor(n_estimators=50,max_depth=5,num_leaves=31,verbose=-1,random_state=42)

cv_res = {nm:{"r2":[],"rmse":[]} for nm in cv_models_dict}
for fold, (tr,te) in enumerate(tscv.split(Xf)):
    for nm, base in cv_models_dict.items():
        m = copy.deepcopy(base); m.fit(Xf[tr], yf[tr]); p = m.predict(Xf[te])
        cv_res[nm]["r2"].append(r2_score(yf[te],p))
        cv_res[nm]["rmse"].append(float(np.sqrt(mean_squared_error(yf[te],p))))

cv_pal = ["#E63946","#457B9D","#2A9D8F"]; marks = ["o","s","^"]
folds  = list(range(1,6))
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("5-Fold TimeSeriesSplit Cross-Validation Performance", fontsize=12, fontweight="bold")
for i,nm in enumerate(cv_models_dict):
    axes[0].plot(folds, cv_res[nm]["r2"],  marks[i]+"-", color=cv_pal[i], lw=2, markersize=7, label=nm)
    axes[1].plot(folds, cv_res[nm]["rmse"], marks[i]+"-", color=cv_pal[i], lw=2, markersize=7, label=nm)
    axes[0].axhline(np.mean(cv_res[nm]["r2"]),  color=cv_pal[i], ls="--", alpha=0.4)
    axes[1].axhline(np.mean(cv_res[nm]["rmse"]), color=cv_pal[i], ls="--", alpha=0.4)
for ax, lbl in zip(axes,["R²","RMSE"]):
    ax.set(xlabel="CV Fold", ylabel=lbl); ax.set_xticks(folds); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig6_cv_performance.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — SHAP / Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig7_shap.png ...")
try:
    import shap
    explainer = shap.TreeExplainer(xgb_m)
    shap_vals = explainer.shap_values(X_test[:300])
    mean_shap = np.abs(shap_vals).mean(0)
    fi_df     = pd.DataFrame({"feature":FEATURES,"importance":mean_shap})
    x_label   = "Mean |SHAP Value|"
    title_str = "SHAP Feature Importance — Top 15 (XGBoost)"
except Exception:
    fi_df     = pd.DataFrame({"feature":FEATURES,"importance":xgb_m.feature_importances_})
    x_label   = "Feature Importance (Gain)"
    title_str = "XGBoost Feature Importance (Gain) — Top 15"

fi_df = fi_df.sort_values("importance", ascending=False)
top15 = fi_df.head(15)
colors_fi = ["#E63946" if i<3 else "#457B9D" if i<8 else "#2A9D8F" for i in range(len(top15))]
fig, ax = plt.subplots(figsize=(11,7))
ax.barh(top15["feature"][::-1].values, top15["importance"][::-1].values,
        color=colors_fi[::-1], alpha=0.88, edgecolor="white")
for bar in ax.patches:
    w = bar.get_width()
    ax.text(w+max(w*0.01,1e-4), bar.get_y()+bar.get_height()/2,
            f"{w:.4f}", va="center", fontsize=8.5)
ax.set(xlabel=x_label, title=title_str)
plt.tight_layout()
plt.savefig("figures/fig7_shap.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig8_ablation.png ...")
groups = {
    "Rental Market"  :["median_gross_rent","median_contract_rent","rent_burden_30plus_pct",
                        "rent_to_income_ratio","rent_growth_yoy","median_gross_rent_lag1","rent_burden_30plus_pct_lag1"],
    "Income Features":["median_hh_income","renter_median_income","renter_income_ratio",
                        "income_gap","income_growth_yoy","median_hh_income_lag1","renter_median_income_lag1"],
    "Labor Market"   :["unemployment_rate","unemployment_rate_lag1","housing_burden_composite"],
    "Housing Stock"  :["renter_share","homeownership_rate","vacancy_rate","severe_crowding_rate",
                        "market_tightness","vacancy_rate_lag1","transit_commute_rate"],
    "Temporal Lags"  :["median_hh_income_lag1","renter_median_income_lag1","unemployment_rate_lag1",
                        "rent_burden_30plus_pct_lag1","vacancy_rate_lag1","median_gross_rent_lag1"],
    "Spatial IDs"    :["borough_code","year","covid_year"],
}
groups   = {k:[f for f in v if f in FEATURES] for k,v in groups.items()}
baseline = best_res["test_r2"]
abl_rows = []
for grp, feats in groups.items():
    reduced = [f for f in FEATURES if f not in feats]
    if not reduced: abl_r2 = 0.0
    else:
        idx = [FEATURES.index(f) for f in reduced]
        m   = xgb.XGBRegressor(n_estimators=50,max_depth=5,verbosity=0,random_state=42)
        m.fit(X_tv[:,idx], y_tv)
        abl_r2 = r2_score(y_test, m.predict(X_test[:,idx]))
    abl_rows.append({"Group":grp,"R² Without":round(abl_r2,4),"ΔR² Drop":round(baseline-abl_r2,4)})

abl_df = pd.DataFrame(abl_rows).sort_values("ΔR² Drop", ascending=False)
bar_c  = ["#E63946" if d>0.005 else "#E9C46A" if d>0 else "#2A9D8F" for d in abl_df["ΔR² Drop"]]

fig, axes = plt.subplots(1, 2, figsize=(15,5))
fig.suptitle("Ablation Study — Feature Group Contribution (XGBoost)", fontsize=12, fontweight="bold")
axes[0].bar(abl_df["Group"], abl_df["ΔR² Drop"], color=bar_c, edgecolor="white", alpha=0.9, width=0.5)
axes[0].axhline(0, color="black", lw=1.2)
axes[0].set(xlabel="Feature Group Removed", ylabel=f"R² Drop (Baseline={baseline:.4f})", title="(a) R² Drop")
plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")
for p, val in zip(axes[0].patches, abl_df["ΔR² Drop"]):
    h = p.get_height()
    axes[0].text(p.get_x()+p.get_width()/2, h+(0.001 if h>=0 else -0.003),
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].bar(abl_df["Group"], abl_df["R² Without"], color="#457B9D", edgecolor="white", alpha=0.9, width=0.5)
axes[1].axhline(baseline, color="#E63946", ls="--", lw=2, label=f"Baseline R²={baseline:.4f}")
axes[1].set(xlabel="Feature Group Removed", ylabel="Remaining R²", title="(b) Remaining R²")
axes[1].legend(fontsize=9)
plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig("figures/fig8_ablation.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Spatial / Moran's I
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig9_spatial.png ...")
resid_arr = y_test - best_pred
tcopy = test_df.copy(); tcopy["resid"] = resid_arr
bres  = tcopy.groupby("borough_name")["resid"].mean()
bl    = sorted(bres.index)
adj   = {"Bronx":["Manhattan","Queens"],"Brooklyn":["Manhattan","Queens"],
         "Manhattan":["Bronx","Brooklyn","Queens"],"Queens":["Bronx","Brooklyn","Manhattan"]}
W = np.array([[1.0 if bl[j] in adj.get(bl[i],[]) else 0.0
               for j in range(4)] for i in range(4)], dtype=float)
rs = W.sum(1, keepdims=True); rs[rs==0]=1; W=W/rs
z = np.array([bres.get(b,0.0) for b in bl]); z = z - z.mean()
I = (4 * np.sum(W * np.outer(z,z)) / (W.sum()*np.sum(z**2)+1e-12))

fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle(f"Spatial Autocorrelation Analysis — Moran's I = {I:.4f}", fontsize=12, fontweight="bold")
bar_c2 = ["#E63946" if v>0 else "#2A9D8F" for v in bres.values]
axes[0].bar(bres.index, bres.values, color=bar_c2, edgecolor="white", alpha=0.88)
axes[0].axhline(0, color="black", lw=1.5)
for i,(boro,v) in enumerate(bres.items()):
    axes[0].text(i, v+(0.0008 if v>=0 else -0.0025), f"{v:.4f}", ha="center", fontsize=9.5, fontweight="bold")
axes[0].set(xlabel="Borough", ylabel="Mean Residual", title="(a) Borough Mean Residuals")
axes[1].hist(resid_arr, bins=50, color="#457B9D", edgecolor="white", alpha=0.82)
axes[1].axvline(0, color="#E63946", ls="--", lw=2.2, label="Zero")
axes[1].axvline(resid_arr.mean(), color="#E9C46A", ls="--", lw=2,
                label=f"Mean: {resid_arr.mean():.5f}")
axes[1].set(xlabel="Residual", ylabel="Count", title="(b) Residual Distribution")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig9_spatial.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Forecast
# ══════════════════════════════════════════════════════════════════════════════
print("Generating fig10_forecast.png ...")
_fc_cols = list(dict.fromkeys(FEATURES+[TARGET]))
bmed     = df[df.year==2022].groupby("borough_name")[_fc_cols].median().reset_index()
hist     = df.groupby(["borough_name","year"])[[TARGET]].median().reset_index()
growth   = {"median_hh_income":1.025,"renter_median_income":1.020,
            "median_gross_rent":1.030,"median_contract_rent":1.025}
lag_pairs = [("median_hh_income_lag1","median_hh_income"),
             ("renter_median_income_lag1","renter_median_income"),
             ("unemployment_rate_lag1","unemployment_rate"),
             ("rent_burden_30plus_pct_lag1","rent_burden_30plus_pct"),
             ("vacancy_rate_lag1","vacancy_rate"),
             ("median_gross_rent_lag1","median_gross_rent")]
fc_rows = []; cur = bmed.copy()
for yr in [2023,2024,2025]:
    nxt = cur.copy(); nxt["year"]=yr
    if "covid_year" in nxt.columns: nxt["covid_year"]=0
    for lag,src in lag_pairs:
        if lag in FEATURES and src in nxt.columns: nxt[lag]=cur[src].values
    for col,g in growth.items():
        if col in nxt.columns: nxt[col]=nxt[col]*g
    Xfc   = nxt[FEATURES].copy().replace([np.inf,-np.inf],np.nan)
    preds = best_res["fitted"].predict(imputer.transform(Xfc))
    for idx_r,(_,row) in enumerate(nxt.iterrows()):
        fc_rows.append({"borough_name":row.borough_name,"year":yr,"predicted":round(float(preds[idx_r]),4)})
    cur = nxt.copy()
fc_df = pd.DataFrame(fc_rows)

fig, ax = plt.subplots(figsize=(13,7))
for i,boro in enumerate(BOROS):
    h  = hist[hist.borough_name==boro].sort_values("year")
    fc = fc_df[fc_df.borough_name==boro].sort_values("year")
    ax.plot(h.year, h[TARGET], "o-", lw=2.2, markersize=5, color=COLORS[i], label=boro)
    last_y = h.year.max(); last_v = h.loc[h.year==last_y,TARGET].values[0]
    ax.plot([last_y]+fc.year.tolist(), [last_v]+fc.predicted.tolist(),
            "D--", lw=2, markersize=6, color=COLORS[i], alpha=0.85)
ax.axvspan(2019.5,2021.5,alpha=0.13,color="#E63946",label="COVID-19")
ax.axvspan(2022.5,2025.5,alpha=0.07,color="#457B9D",label="Forecast Zone")
ax.axvline(2022.5,color="#457B9D",ls=":",lw=2)
ax.set(xlabel="Year", ylabel="Severely Cost-Burdened Share (≥50%)",
       title="Borough-Level Severe Rent Burden Forecast (2023–2025)")
ax.legend(loc="upper right",fontsize=9)
ax.set_xticks(range(2012,2026))
plt.xticks(rotation=40)
plt.tight_layout()
plt.savefig("figures/fig10_forecast.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
print("\nAll figures generated successfully!")
figs = sorted(os.listdir("figures"))
for f in figs:
    size = os.path.getsize(f"figures/{f}") // 1024
    print(f"  figures/{f}  ({size} KB)")
