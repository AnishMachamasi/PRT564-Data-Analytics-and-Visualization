"""
03_regression.py
================
Regression model training for the NHS Age-based Obesity dataset.

Inputs  : regression_dataset.csv
Outputs : outputs/regression/
            - model_metrics.csv
            - oof_predictions.csv   (out-of-fold predictions for all models)
            - model_coefficients.csv (linear models)
            - rf_feature_importance.csv

Models
------
  1. Linear Regression  – baseline, interpretable
  2. Ridge (α=1.0)      – L2 regularisation, handles multicollinearity
  3. Lasso (α=0.001)    – L1 regularisation, built-in feature selection
  4. Random Forest      – non-linear ensemble, captures age interactions

All models evaluated with 10-fold cross-validation.
"""

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
TARGET  = "nhs_bmi__obese_class_1_rate"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
DATA   = ROOT /"Assignment"/"outputs_regression_dataset"/ "regression_dataset.csv"
OUTDIR = ROOT / "outputs" / "regression"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & FEATURE PREPARATION  (mirrors preprocessing step)
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA)

rate_cols = [c for c in df_raw.columns if c.endswith("_rate")]
df = df_raw[["age"] + rate_cols].copy()

leak_patterns = ["nhs_bmi__"]
feature_cols = [
    c for c in rate_cols
    if c != TARGET and not any(p in c for p in leak_patterns)
]
X_raw = df[feature_cols].copy()
y     = df[TARGET].copy()

# Near-zero variance filter
vt = VarianceThreshold(threshold=1e-4)
vt.fit(X_raw)
feature_cols = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]
X_raw = X_raw[feature_cols]

# Multicollinearity filter (|r| > 0.97)
corr_abs = X_raw.corr().abs()
upper    = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
drop_set = {col for col in upper.columns if any(upper[col] > 0.97)}
feature_cols = [c for c in feature_cols if c not in drop_set]
X_raw = X_raw[feature_cols]

# Scale
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols, index=X_raw.index)

n, p = X.shape
print(f"Samples   : {n}")
print(f"Features  : {p}")
print(f"Target    : {TARGET}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
CV = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge (α=1.0)":     Ridge(alpha=1.0),
    "Lasso (α=0.001)":   Lasso(alpha=0.001, max_iter=10_000),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=300,
                             max_depth=6,
                             min_samples_leaf=3,
                             max_features="sqrt",
                             random_state=42,
                             n_jobs=-1,
                         ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATED TRAINING & METRIC COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
def adjusted_r2(r2, n, p):
    """Adjusted R² penalises for number of predictors."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


results  = {}   # model → metric dict
preds    = {}   # model → OOF predictions array
cv_scores = {}  # model → per-fold metric dict

for name, model in models.items():
    # OOF predictions
    y_pred = cross_val_predict(model, X, y, cv=CV)
    preds[name] = y_pred

    # Per-fold scores via cross_validate for variance reporting
    fold_scores = cross_validate(
        model, X, y, cv=CV,
        scoring=["neg_mean_absolute_error",
                 "neg_root_mean_squared_error",
                 "r2"],
        return_train_score=False,
    )
    cv_scores[name] = fold_scores

    mae   = mean_absolute_error(y, y_pred)
    rmse  = np.sqrt(mean_squared_error(y, y_pred))
    r2    = r2_score(y, y_pred)
    adj   = adjusted_r2(r2, n, p)

    # Fold-level std for reporting
    mae_std  = fold_scores["test_neg_mean_absolute_error"].std()
    rmse_std = fold_scores["test_neg_root_mean_squared_error"].std()
    r2_std   = fold_scores["test_r2"].std()

    results[name] = {
        "MAE":         round(mae,  6),
        "MAE_std":     round(abs(mae_std),  6),
        "RMSE":        round(rmse, 6),
        "RMSE_std":    round(abs(rmse_std), 6),
        "R²":          round(r2,   4),
        "R²_std":      round(r2_std, 4),
        "Adj. R²":     round(adj,  4),
    }
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"  MAE     = {mae:.6f}  (±{abs(mae_std):.6f})")
    print(f"  RMSE    = {rmse:.6f}  (±{abs(rmse_std):.6f})")
    print(f"  R²      = {r2:.4f}    (±{r2_std:.4f})")
    print(f"  Adj. R² = {adj:.4f}")

metrics_df = pd.DataFrame(results).T
metrics_df.to_csv(OUTDIR / "model_metrics.csv")
print(f"\nSaved: model_metrics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE OOF PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
oof_df = pd.DataFrame({"age": df["age"].values, "actual": y.values})
for name, pred in preds.items():
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "").replace("α", "a")
    oof_df[f"pred_{safe}"] = pred
oof_df.to_csv(OUTDIR / "oof_predictions.csv", index=False)
print("Saved: oof_predictions.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LINEAR MODEL COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────
coef_records = {}
for name in ["Linear Regression", "Ridge (α=1.0)", "Lasso (α=0.001)"]:
    m = models[name]
    m.fit(X, y)
    coef_records[name] = pd.Series(m.coef_, index=feature_cols)

coef_df = pd.DataFrame(coef_records)
coef_df["abs_linear"] = coef_df["Linear Regression"].abs()
coef_df = coef_df.sort_values("abs_linear", ascending=False).drop(columns="abs_linear")
coef_df.to_csv(OUTDIR / "model_coefficients.csv")
print("Saved: model_coefficients.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 6. RANDOM FOREST FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
rf_fitted = RandomForestRegressor(
    n_estimators=300, max_depth=6, min_samples_leaf=3,
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_fitted.fit(X, y)

importances = pd.Series(rf_fitted.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
importances.to_csv(OUTDIR / "rf_feature_importance.csv", header=["importance"])
print("Saved: rf_feature_importance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def short(col):
    return col.replace("nhs_", "").replace("_rate", "").replace("__", " | ")

# ── 7a. Metrics bar chart (with std error bars) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_names = list(models.keys())

for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
    vals = metrics_df[metric].values.astype(float)
    errs = metrics_df[f"{metric}_std"].values.astype(float)
    bars = ax.bar(model_names, vals, color=PALETTE, edgecolor="white",
                  width=0.55, yerr=errs, capsize=5,
                  error_kw=dict(elinewidth=1.2, ecolor="grey"))
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    for bar, v in zip(bars, vals):
        offset = 0.002 if metric != "R²" else 0.01
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Model Comparison – 10-fold CV (error bars = ±1 std)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "01_model_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_model_metrics.png")


# ── 7b. Residual plots (2×2) ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for ax, (name, colour) in zip(axes, zip(model_names, PALETTE)):
    residuals = y.values - preds[name]
    ax.scatter(preds[name], residuals, alpha=0.65, s=28,
               color=colour, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.3)
    # Lowess-style smoothed trend line using rolling mean
    order = np.argsort(preds[name])
    smooth_x = preds[name][order]
    smooth_y = pd.Series(residuals[order]).rolling(10, center=True).mean().values
    ax.plot(smooth_x, smooth_y, color="darkred", linewidth=1.5, alpha=0.6)
    ax.set_title(f"Residuals – {name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Residual")
    rmse_val = np.sqrt(np.mean(residuals**2))
    ax.text(0.97, 0.05, f"RMSE={rmse_val:.4f}",
            transform=ax.transAxes, ha="right", fontsize=9, color="grey")

plt.suptitle("Residual Plots – 10-fold CV Predictions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "02_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_residuals.png")


# ── 7c. Actual vs Predicted for best model ────────────────────────────────────
best_name = metrics_df["R²"].idxmax()
best_pred = preds[best_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter
axes[0].scatter(y, best_pred, alpha=0.75, s=35, color=PALETTE[2],
                edgecolor="white", linewidth=0.4)
lims = [min(y.min(), best_pred.min()) * 0.95,
        max(y.max(), best_pred.max()) * 1.05]
axes[0].plot(lims, lims, "r--", linewidth=1.3, label="Perfect prediction")
axes[0].set_xlabel("Actual Obesity Class 1 Rate")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Actual vs Predicted\n{best_name}  |  R²={metrics_df.loc[best_name,'R²']:.4f}")
axes[0].legend()

# Age-indexed prediction overlay
axes[1].plot(df["age"], y, "o-", label="Actual", color=PALETTE[0],
             markersize=3, linewidth=1.3, alpha=0.85)
axes[1].plot(df["age"], best_pred, "s--", label=f"Predicted ({best_name})",
             color=PALETTE[2], markersize=3, linewidth=1.3, alpha=0.85)
axes[1].fill_between(df["age"], y, best_pred, alpha=0.12, color="grey")
axes[1].set_xlabel("Age (years)")
axes[1].set_ylabel("Rate")
axes[1].set_title("Actual vs Predicted by Age")
axes[1].legend(fontsize=8)

plt.suptitle(f"Best Model: {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "03_best_model_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: 03_best_model_predictions.png  (best model: {best_name})")


# ── 7d. Random Forest feature importance ──────────────────────────────────────
top_imp = importances.head(20).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    [short(c) for c in top_imp.index],
    top_imp.values,
    color=PALETTE[0], edgecolor="white"
)
ax.set_title("Random Forest – Top-20 Feature Importances", fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
for bar, v in zip(bars, top_imp.values):
    ax.text(v + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTDIR / "04_rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_rf_feature_importance.png")


# ── 7e. Lasso coefficient path (feature selection insight) ───────────────────
lasso_coef = coef_df["Lasso (α=0.001)"].sort_values()
non_zero   = lasso_coef[lasso_coef != 0].sort_values()

if not non_zero.empty:
    fig, ax = plt.subplots(figsize=(10, max(4, len(non_zero) * 0.35)))
    colors_bar = [PALETTE[3] if v < 0 else PALETTE[2] for v in non_zero.values]
    ax.barh([short(c) for c in non_zero.index], non_zero.values,
            color=colors_bar, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Lasso (α=0.001) – Non-zero Coefficients", fontsize=12, fontweight="bold")
    ax.set_xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(OUTDIR / "05_lasso_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_lasso_coefficients.png")
    print(f"  Lasso selected {len(non_zero)} / {p} features")
else:
    print("  All Lasso coefficients are zero – try a smaller alpha")


print(f"\nAll regression outputs saved to: {OUTDIR}")
