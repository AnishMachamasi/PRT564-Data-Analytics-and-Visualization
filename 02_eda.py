"""
02_eda.py
=========
Exploratory Data Analysis for the NHS Age-based Regression Dataset.

Inputs  : regression_dataset.csv   (output of data_processing step)
Outputs : figures saved to outputs/eda/
          eda_summary.csv

Sections
--------
1. Load preprocessed feature matrix
2. Target distribution & age trend
3. Full correlation heatmap
4. Top-5 features vs target scatter plots
5. Feature distributions (boxplots)
6. Age-stratified heatmap of key rates
"""

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
TARGET   = "nhs_bmi__obese_class_1_rate"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent        # project root
DATA   = ROOT /"Assignment"/"outputs_regression_dataset"/ "regression_dataset.csv"
OUTDIR = ROOT / "outputs" / "eda"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA)
print(f"Dataset shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
print(f"Age range     : {df_raw['age'].min()} – {df_raw['age'].max()}")
print(f"Missing values: {df_raw.isnull().sum().sum()}")

# Keep only rate columns (population-normalised, age-comparable)
rate_cols = [c for c in df_raw.columns if c.endswith("_rate")]
df = df_raw[["age"] + rate_cols].copy()

# Remove BMI siblings → data leakage for our target
leak_patterns = ["nhs_bmi__"]
feature_cols = [
    c for c in rate_cols
    if c != TARGET and not any(p in c for p in leak_patterns)
]
X_raw = df[feature_cols].copy()
y     = df[TARGET].copy()

# Remove near-zero-variance features
vt = VarianceThreshold(threshold=1e-4)
vt.fit(X_raw)
feature_cols = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]
X_raw = X_raw[feature_cols]

# Remove highly correlated pairs (|r| > 0.97)
corr_abs = X_raw.corr().abs()
upper    = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
drop_set = {col for col in upper.columns if any(upper[col] > 0.97)}
feature_cols = [c for c in feature_cols if c not in drop_set]
X_raw = X_raw[feature_cols]

print(f"\nFeatures used for EDA : {len(feature_cols)}")

def short(col):
    """Human-readable column label."""
    return col.replace("nhs_", "").replace("_rate", "").replace("__", " | ")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGET DISTRIBUTION & AGE TREND
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histogram
axes[0].hist(y, bins=20, color=PALETTE[0], edgecolor="white", linewidth=0.6)
axes[0].axvline(y.mean(), color="red", linestyle="--", linewidth=1.3, label=f"Mean={y.mean():.3f}")
axes[0].axvline(y.median(), color="orange", linestyle=":", linewidth=1.3, label=f"Median={y.median():.3f}")
axes[0].set_title("Distribution of Obesity Class 1 Rate")
axes[0].set_xlabel("Rate (proportion of age-group population)")
axes[0].set_ylabel("Count")
axes[0].legend(fontsize=8)

# Line plot vs age
axes[1].plot(df["age"], y, "o-", color=PALETTE[0], markersize=3, linewidth=1.4, alpha=0.85)
axes[1].fill_between(df["age"], y, alpha=0.12, color=PALETTE[0])
axes[1].set_title("Obesity Class 1 Rate by Age")
axes[1].set_xlabel("Age (years)")
axes[1].set_ylabel("Rate")

# QQ plot to check normality
from scipy import stats as scipy_stats

(osm, osr), (slope, intercept, r) = scipy_stats.probplot(y, dist="norm")
axes[2].plot(osm, osr, "o", markersize=4, color=PALETTE[1], alpha=0.8)
axes[2].plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.2)
axes[2].set_title(f"Q-Q Plot (target)\nShapiro-Wilk p={scipy_stats.shapiro(y).pvalue:.4f}")
axes[2].set_xlabel("Theoretical Quantiles")
axes[2].set_ylabel("Sample Quantiles")

plt.suptitle("Target Variable: Obesity Class 1 Rate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "01_target_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_target_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
heatmap_df = X_raw.copy()
heatmap_df["[TARGET]"] = y.values
corr_heat   = heatmap_df.corr()
short_lbls  = [short(c) for c in corr_heat.columns]

fig, ax = plt.subplots(figsize=(15, 12))
mask = np.triu(np.ones_like(corr_heat, dtype=bool))
sns.heatmap(
    corr_heat, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.25, linecolor="white",
    ax=ax, xticklabels=short_lbls, yticklabels=short_lbls,
    annot_kws={"size": 6.5}, vmin=-1, vmax=1,
)
ax.set_title("Pearson Correlation Heatmap – Retained Features + Target",
             fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=45, ha="right", fontsize=7.5)
plt.yticks(fontsize=7.5)
plt.tight_layout()
plt.savefig(OUTDIR / "02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOP-5 FEATURES vs TARGET (scatter + regression line + confidence band)
# ─────────────────────────────────────────────────────────────────────────────
corr_with_target = X_raw.corrwith(y).abs().sort_values(ascending=False)
top5 = corr_with_target.head(5).index.tolist()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, top5):
    x_vals = X_raw[col].values
    # scatter
    ax.scatter(x_vals, y.values, alpha=0.65, s=22, color=PALETTE[1], edgecolor="white", linewidth=0.3)
    # OLS line + 95 % CI band
    slope, intercept, r_val, p_val, stderr = scipy_stats.linregress(x_vals, y.values)
    xl  = np.linspace(x_vals.min(), x_vals.max(), 200)
    yl  = slope * xl + intercept
    n   = len(x_vals)
    se  = stderr * np.sqrt(1/n + (xl - x_vals.mean())**2 / ((n-1)*x_vals.var()))
    t95 = scipy_stats.t.ppf(0.975, df=n-2)
    ax.plot(xl, yl, "r-", linewidth=1.3)
    ax.fill_between(xl, yl - t95*se, yl + t95*se, alpha=0.15, color="red")
    ax.set_xlabel(short(col), fontsize=7.5)
    ax.set_ylabel("Obesity Rate" if ax is axes[0] else "")
    ax.set_title(f"|r| = {corr_with_target[col]:.3f}\np = {p_val:.3f}", fontsize=8.5)

plt.suptitle("Top-5 Correlated Features vs Obesity Class 1 Rate",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTDIR / "03_top5_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_top5_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE DISTRIBUTION BOXPLOTS  (grouped by domain)
# ─────────────────────────────────────────────────────────────────────────────
def domain(col):
    parts = col.replace("nhs_", "").split("__")
    return parts[0] if len(parts) > 1 else "other"

domain_map = {c: domain(c) for c in feature_cols}
domains    = sorted(set(domain_map.values()))

fig, axes = plt.subplots(len(domains), 1,
                          figsize=(14, 3.5 * len(domains)),
                          constrained_layout=True)
if len(domains) == 1:
    axes = [axes]

for ax, dom in zip(axes, domains):
    cols = [c for c in feature_cols if domain_map[c] == dom]
    data = [X_raw[c].dropna().values for c in cols]
    bp   = ax.boxplot(data, patch_artist=True, vert=True,
                      medianprops=dict(color="red", linewidth=1.5))
    for patch, colour in zip(bp["boxes"], plt.cm.tab20.colors):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([short(c) for c in cols], rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Domain: {dom}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Rate")

plt.suptitle("Feature Distributions by Domain (Boxplots)", fontsize=13,
             fontweight="bold")
plt.savefig(OUTDIR / "04_feature_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_feature_boxplots.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGE-STRATIFIED HEATMAP  (key rates across age bands)
# ─────────────────────────────────────────────────────────────────────────────
# Bin ages into 10-year bands
age_bins   = list(range(16, 100, 10)) + [100]
age_labels = [f"{a}–{a+9}" for a in age_bins[:-1]]
df2 = df.copy()
df2["age_band"] = pd.cut(df2["age"], bins=age_bins, labels=age_labels, right=False)

# Pick a representative subset of features
highlight_cols = top5 + [TARGET]
agg = df2.groupby("age_band", observed=True)[highlight_cols].mean()

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    agg.T, annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.4, linecolor="white", ax=ax,
    xticklabels=age_labels,
    yticklabels=[short(c) for c in highlight_cols],
    annot_kws={"size": 8},
)
ax.set_title("Mean Rate by Age Band – Top Features + Target",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Age Band")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(OUTDIR / "05_age_band_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_age_band_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EDA SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
desc = X_raw[feature_cols].describe().T
desc["corr_with_target"] = X_raw.corrwith(y)
desc["abs_corr"]         = desc["corr_with_target"].abs()
desc = desc.sort_values("abs_corr", ascending=False)
desc.to_csv(OUTDIR / "eda_summary.csv")
print("Saved: eda_summary.csv")

print("\n── Top-10 features by |correlation with target| ──")
print(desc[["mean", "std", "corr_with_target"]].head(10).to_string())
print(f"\nAll EDA outputs saved to: {OUTDIR}")
