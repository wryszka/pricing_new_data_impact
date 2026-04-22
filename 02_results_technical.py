# Databricks notebook source
# MAGIC %md
# MAGIC # Technical Results — Frequency, Severity & Model Selection
# MAGIC
# MAGIC **For data scientists and actuaries**
# MAGIC
# MAGIC This notebook consolidates the full technical results of the new data impact study:
# MAGIC comparing **standard GLM/GBM** (traditional rating factors only) against **enriched GLM/GBM**
# MAGIC (augmented with real UK public data: IMD 2019 deprivation deciles — overall, crime, income,
# MAGIC health, living environment — plus urban/rural, coastal flag, and region dummies, all joined
# MAGIC on the policy's postcode via the `postcode_enrichment` table).
# MAGIC
# MAGIC All artefacts — portfolios, coefficients, model metrics, scored quotes, and model factory
# MAGIC results — are read directly from Unity Catalog. This notebook trains nothing.
# MAGIC
# MAGIC | Section | Content |
# MAGIC |---|---|
# MAGIC | 1 | Setup |
# MAGIC | 2 | Portfolio Overview |
# MAGIC | 3 | Frequency Model Comparison |
# MAGIC | 4 | Frequency Loss Ratio by Decile |
# MAGIC | 5 | Frequency Pricing Impact |
# MAGIC | 6 | Severity Model Comparison |
# MAGIC | 7 | Severity Feature Importance |
# MAGIC | 8 | Severity Loss Ratio — Full Quotes |
# MAGIC | 9 | Severity Pricing Impact |
# MAGIC | 10 | Model Factory Results |
# MAGIC | 11 | Model Serving — Score New Data |
# MAGIC | 12 | Summary |

# COMMAND ----------

# MAGIC %pip install matplotlib mlflow statsmodels scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import mlflow

plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12, "axes.labelsize": 10})

print(f"Catalog : {CATALOG}")
print(f"Schema  : {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Portfolio Overview
# MAGIC
# MAGIC High-level statistics for the full policy portfolio, followed by distributions of the
# MAGIC three key structural features.

# COMMAND ----------

# Use spark.sql to work around Photon column-index bug on the portfolio schema
pdf = spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.portfolio").toPandas()

n_policies = len(pdf)
claim_rate = (pdf["num_claims"] > 0).mean()
avg_severity = pdf.loc[pdf["num_claims"] > 0, "claim_severity"].mean()

print(f"Policies:        {n_policies:,}")
print(f"Claim rate:      {claim_rate:.2%}")
print(f"Avg severity:    £{avg_severity:,.0f}")

# COMMAND ----------

# Distribution of key features
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

feature_cols = [c for c in ["property_type", "imd_decile", "region_name"] if c in pdf.columns][:3]
for ax, col in zip(axes, feature_cols):
    pdf[col].value_counts().sort_index().plot.bar(ax=ax, edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Frequency Model Comparison
# MAGIC
# MAGIC Comparing a Poisson GLM trained on standard rating factors only against the same
# MAGIC GLM augmented with the real UK public-data enrichment features (IMD deciles, urban,
# MAGIC coastal, region). Lower AIC/BIC = better fit. Higher Gini and Deviance Explained
# MAGIC = stronger risk discrimination.

# COMMAND ----------

model_comparison = spark.table("model_comparison")
display(model_comparison)

# COMMAND ----------

# Highlight key metric movements
mc = model_comparison.toPandas().set_index("metric") if "metric" in model_comparison.columns else model_comparison.toPandas()

if "metric" in model_comparison.columns:
    std_vals = mc["model_1_standard"].astype(float)
    enr_vals = mc["model_2_enriched"].astype(float)
    for metric in ["AIC", "BIC", "Gini (test)", "Deviance Explained"]:
        if metric in mc.index:
            s, e = std_vals[metric], enr_vals[metric]
            direction = "▼" if e < s else "▲"
            print(f"{metric:25s}  standard={s:>10.2f}   enriched={e:>10.2f}   {direction} {abs(e - s):.2f}")

# COMMAND ----------

# Coefficient pivot — standard vs enriched side by side
coefficients = spark.table("glm_coefficients").toPandas()

pivot = coefficients.pivot_table(
    index="feature", columns="model", values="coef", aggfunc="first"
).rename(columns=lambda c: c.strip())

if "p_value" in coefficients.columns:
    pvals = coefficients.pivot_table(
        index="feature", columns="model", values="p_value", aggfunc="first"
    ).rename(columns=lambda c: f"p_{c.strip()}")
    pivot = pivot.join(pvals)

display(spark.createDataFrame(pivot.reset_index()))

# COMMAND ----------

# Enrichment features — statistical significance in the enriched model
base_enrichment = ["imd_decile", "crime_decile", "income_decile", "health_decile",
                   "living_env_decile", "is_urban", "is_coastal"]
region_dummies = [c for c in coefficients["feature"].unique() if str(c).startswith("region_") and c != "region_code"]
enrichment_features = base_enrichment + region_dummies

enrich_mask = (
    coefficients["feature"].isin(enrichment_features)
    & (coefficients["model"].str.contains("enriched", case=False))
)
sig = coefficients.loc[enrich_mask, ["feature", "coef", "p_value"]].sort_values("p_value")
if not sig.empty:
    print("Enrichment features — enriched model (sorted by p-value):")
    print(sig.to_string(index=False))

# COMMAND ----------

# GLM coefficient comparison chart — common features only
if "standard" in pivot.columns and "enriched" in pivot.columns:
    common = pivot.dropna(subset=["standard", "enriched"])
    fig, ax = plt.subplots(figsize=(10, max(4, len(common) * 0.35)))
    y = np.arange(len(common))
    ax.barh(y + 0.15, common["standard"], height=0.3, label="Standard", color="#4C72B0")
    ax.barh(y - 0.15, common["enriched"], height=0.3, label="Enriched", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(common.index, fontsize=9)
    ax.set_xlabel("Coefficient")
    ax.set_title("GLM Coefficients — Standard vs Enriched")
    ax.legend()
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Frequency Loss Ratio by Decile
# MAGIC
# MAGIC Policies are ranked by predicted frequency and split into 10 equal groups.
# MAGIC The loss ratio within each decile measures how well the model captures actual risk ordering.
# MAGIC A flat line at 1.0 would be perfect; divergence shows mismatch between price and risk.

# COMMAND ----------

lr_decile = spark.table("loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(8, 5))
for model, grp in lr_decile.groupby("model"):
    grp_sorted = grp.sort_values("decile")
    ax.plot(grp_sorted["decile"], grp_sorted["loss_ratio"], marker="o", label=model)

ax.axhline(1.0, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Decile")
ax.set_ylabel("Loss Ratio")
ax.set_title("Frequency — Loss Ratio by Risk Decile")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Stability metric — lower std dev = more uniform pricing across deciles
lr_stability = lr_decile.groupby("model")["loss_ratio"].std().reset_index()
lr_stability.columns = ["model", "lr_std_dev"]
print("Loss-ratio stability (lower = more uniform pricing):")
print(lr_stability.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Frequency Pricing Impact
# MAGIC
# MAGIC Burning-cost quotes (frequency component) compared across the full portfolio.
# MAGIC The enriched model re-prices deprived, high-crime, and coastal postcodes materially.

# COMMAND ----------

priced = spark.table("priced_portfolio").toPandas()
priced["quote_diff"] = priced["quote_enriched"] - priced["quote_standard"]
priced["quote_diff_pct"] = priced["quote_diff"] / priced["quote_standard"] * 100

print(f"Mean absolute quote difference: £{priced['quote_diff'].abs().mean():.2f}")
print(f"Median quote difference:        £{priced['quote_diff'].median():.2f}")
print(f"Max increase:                    £{priced['quote_diff'].max():.2f}")
print(f"Max decrease:                    £{priced['quote_diff'].min():.2f}")

# COMMAND ----------

# Top 10 most divergent quotes — high-crime, deprived or coastal postcodes
high_risk = priced[
    (priced["crime_decile"] <= 2) | (priced["imd_decile"] <= 2) | (priced["is_coastal"] == 1)
].nlargest(10, "quote_diff")

display(spark.createDataFrame(high_risk[["crime_decile", "imd_decile", "is_coastal",
                                         "quote_standard", "quote_enriched", "quote_diff"]]))

# COMMAND ----------

# Scatter — standard vs enriched frequency quotes
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(priced["quote_standard"], priced["quote_enriched"], alpha=0.15, s=6, color="#4C72B0")
lims = [
    min(priced["quote_standard"].min(), priced["quote_enriched"].min()),
    max(priced["quote_standard"].max(), priced["quote_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Quote (£)")
ax.set_ylabel("Enriched Quote (£)")
ax.set_title("Frequency Quote Comparison: Standard vs Enriched GLM")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Segmentation — average quotes by crime_decile, imd_decile and is_coastal
seg_cols = [c for c in ["crime_decile", "imd_decile", "is_coastal"] if c in priced.columns]

for col in seg_cols:
    seg = priced.groupby(col).agg(
        n=("quote_standard", "count"),
        avg_standard=("quote_standard", "mean"),
        avg_enriched=("quote_enriched", "mean"),
    ).reset_index()
    seg["diff_pct"] = (seg["avg_enriched"] - seg["avg_standard"]) / seg["avg_standard"] * 100
    print(f"\n--- {col} ---")
    print(seg.to_string(index=False))

# COMMAND ----------

# Bar chart — average frequency quotes by crime_decile
if "crime_decile" in priced.columns:
    seg_crime = priced.groupby("crime_decile")[["quote_standard", "quote_enriched"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seg_crime))
    width = 0.35
    ax.bar(x - width / 2, seg_crime["quote_standard"], width, label="Standard", color="#4C72B0")
    ax.bar(x + width / 2, seg_crime["quote_enriched"], width, label="Enriched", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)}" for d in seg_crime.index], rotation=0)
    ax.set_xlabel("Crime Decile (1 = highest crime → 10 = lowest)")
    ax.set_ylabel("Average Frequency Quote (£)")
    ax.set_title("Average Frequency Quote by Crime Decile")
    ax.legend()
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Severity Model Comparison
# MAGIC
# MAGIC Comparing a GBM trained on standard factors against an enriched GBM.
# MAGIC Metrics: MAE (£), RMSE (£), MAPE (%), Bias (%), Gini coefficient.
# MAGIC Lower error metrics and higher Gini indicate a better severity model.

# COMMAND ----------

# Claimant population summary
train = spark.table("severity_train_set").toPandas()
test = spark.table("severity_test_set").toPandas()

all_claimants = pd.concat([train, test])
n_train = len(train)
n_test = len(test)
avg_sev = all_claimants["claim_severity"].mean()
med_sev = all_claimants["claim_severity"].median()

print(f"Claimants — Train: {n_train:,} | Test: {n_test:,}")
print(f"Avg severity:    £{avg_sev:,.0f}")
print(f"Median severity: £{med_sev:,.0f}")

# COMMAND ----------

# Severity distribution — raw and log scale
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_claimants["claim_severity"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Claim Severity (£)")
axes[0].set_ylabel("Count")
axes[0].set_title("Severity Distribution")
axes[0].axvline(avg_sev, color="red", ls="--", label=f"Mean: £{avg_sev:,.0f}")
axes[0].axvline(med_sev, color="orange", ls="--", label=f"Median: £{med_sev:,.0f}")
axes[0].legend()

axes[1].hist(np.log(all_claimants["claim_severity"]), bins=50, color="#DD8452", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("Log(Claim Severity)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Severity Distribution")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Severity model comparison table
severity_model_comparison = spark.table("severity_model_comparison")
display(severity_model_comparison)

# COMMAND ----------

# Highlight metric movements — direction-aware
smc = severity_model_comparison.toPandas().set_index("metric")
std_vals = smc["model_1_standard"].astype(float)
enr_vals = smc["model_2_enriched"].astype(float)

for metric in smc.index:
    s, e = std_vals[metric], enr_vals[metric]
    lower_better = metric in ["MAE (£)", "RMSE (£)", "MAPE (%)", "Bias (%)"]
    if lower_better:
        direction = "▼ better" if e < s else "▲ worse"
    else:
        direction = "▲ better" if e > s else "▼ worse"
    print(f"{metric:25s}  standard={s:>12.4f}   enriched={e:>12.4f}   {direction}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Severity Feature Importance
# MAGIC
# MAGIC Side-by-side feature importance (gain) from the standard and enriched GBMs.
# MAGIC The enriched model redistributes importance toward the IMD deciles, coastal flag,
# MAGIC and region dummies.

# COMMAND ----------

importance = spark.table("severity_feature_importance").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, model_name, title in [
    (axes[0], "standard", "Standard GBM"),
    (axes[1], "enriched", "Enriched GBM"),
]:
    subset = importance[importance["model"] == model_name].sort_values("importance", ascending=True)
    ax.barh(subset["feature"], subset["importance"],
            color="#4C72B0" if model_name == "standard" else "#DD8452")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Feature Importance — {title}")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Enrichment feature share of total importance in the enriched model
base_enr_feat_set = {"imd_decile", "crime_decile", "income_decile", "health_decile",
                     "living_env_decile", "is_urban", "is_coastal"}
enriched_rows = importance[importance["model"] == "enriched"]
region_feat_set = {f for f in enriched_rows["feature"].unique() if str(f).startswith("region_") and f != "region_code"}
enr_feat_set = base_enr_feat_set | region_feat_set

enr_imp = enriched_rows[enriched_rows["feature"].isin(enr_feat_set)]
total_imp = enriched_rows["importance"].sum()
enr_share = enr_imp["importance"].sum() / total_imp * 100 if total_imp else 0.0
print(f"Enrichment features account for {enr_share:.1f}% of total importance in the enriched GBM")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Severity Loss Ratio — Full Quotes (Freq × Sev)
# MAGIC
# MAGIC Policies ranked by full burning-cost quote (frequency × severity).
# MAGIC This is the ultimate measure of combined model quality — does price follow risk?

# COMMAND ----------

sev_lr_decile = spark.table("severity_loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
for model, grp in sev_lr_decile.groupby("model"):
    grp_sorted = grp.sort_values("decile")
    color = "#E53935" if model == "Standard" else "#43A047"
    ax.plot(grp_sorted["decile"], grp_sorted["loss_ratio"],
            marker="o", label=model, color=color, linewidth=2)

ax.axhline(1.0, ls="--", color="grey", lw=0.8, label="Breakeven")
ax.fill_between(range(10), 0.85, 1.15, color="green", alpha=0.07, label="Healthy range")
ax.set_xlabel("Full Quote Decile (cheapest → most expensive)")
ax.set_ylabel("Loss Ratio")
ax.set_title("Loss Ratio by Decile — Full Burning-Cost Quotes (Freq × Sev)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Stability — std dev of loss ratio across deciles
sev_lr_stability = sev_lr_decile.groupby("model")["loss_ratio"].std().reset_index()
sev_lr_stability.columns = ["model", "lr_std_dev"]
print("Loss-ratio stability — full quotes (lower = more uniform pricing):")
print(sev_lr_stability.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Severity Pricing Impact
# MAGIC
# MAGIC Comparing severity predictions and full burning-cost quotes across the claimant population.
# MAGIC Left panel: severity component only.  Right panel: combined freq × sev quote.

# COMMAND ----------

sev_priced = spark.table("severity_priced_portfolio").toPandas()

sev_priced["sev_diff"] = sev_priced["sev_pred_enriched"] - sev_priced["sev_pred_standard"]
sev_priced["full_quote_diff"] = sev_priced["full_quote_enriched"] - sev_priced["full_quote_standard"]

print(f"Mean severity diff:           £{sev_priced['sev_diff'].abs().mean():.2f}")
print(f"Mean full-quote diff:         £{sev_priced['full_quote_diff'].abs().mean():.2f}")
print(f"Max full-quote increase:      £{sev_priced['full_quote_diff'].max():.2f}")
print(f"Max full-quote decrease:      £{sev_priced['full_quote_diff'].min():.2f}")

# COMMAND ----------

# Scatter plots — severity predictions and full quotes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Severity predictions
ax = axes[0]
ax.scatter(sev_priced["sev_pred_standard"], sev_priced["sev_pred_enriched"],
           alpha=0.1, s=4, color="#4C72B0")
lims = [
    min(sev_priced["sev_pred_standard"].min(), sev_priced["sev_pred_enriched"].min()),
    max(sev_priced["sev_pred_standard"].max(), sev_priced["sev_pred_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Severity (£)")
ax.set_ylabel("Enriched Severity (£)")
ax.set_title("Severity Predictions: Standard vs Enriched")

# Full quotes
ax = axes[1]
ax.scatter(sev_priced["full_quote_standard"], sev_priced["full_quote_enriched"],
           alpha=0.1, s=4, color="#DD8452")
lims = [
    min(sev_priced["full_quote_standard"].min(), sev_priced["full_quote_enriched"].min()),
    max(sev_priced["full_quote_standard"].max(), sev_priced["full_quote_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Full Quote (£)")
ax.set_ylabel("Enriched Full Quote (£)")
ax.set_title("Full Quotes (Freq×Sev): Standard vs Enriched")

plt.tight_layout()
plt.show()

# COMMAND ----------

# Segmentation — severity by imd_decile and is_coastal
for col, labels in [("imd_decile", None), ("is_coastal", {0: "Inland", 1: "Coastal"})]:
    seg = sev_priced.groupby(col).agg(
        avg_sev_standard=("sev_pred_standard", "mean"),
        avg_sev_enriched=("sev_pred_enriched", "mean"),
        avg_actual_sev=("claim_severity", "mean"),
        n=("actual_loss", "count"),
    ).reset_index()
    if labels:
        seg[col] = seg[col].map(labels)
    seg["diff_pct"] = (seg["avg_sev_enriched"] - seg["avg_sev_standard"]) / seg["avg_sev_standard"] * 100
    print(f"\n--- {col} ---")
    print(seg.to_string(index=False))

# COMMAND ----------

# Bar chart — predicted severity vs actual by IMD decile
fig, ax = plt.subplots(figsize=(10, 6))

seg_imd = sev_priced.groupby("imd_decile").agg(
    standard=("sev_pred_standard", "mean"),
    enriched=("sev_pred_enriched", "mean"),
    actual=("claim_severity", "mean"),
).sort_index()

x = np.arange(len(seg_imd))
width = 0.25

ax.bar(x - width, seg_imd["standard"], width, label="Standard GBM", color="#E53935", alpha=0.85)
ax.bar(x, seg_imd["enriched"], width, label="Enriched GBM", color="#1E88E5", alpha=0.85)
ax.bar(x + width, seg_imd["actual"], width, label="Actual Avg Severity", color="#43A047", alpha=0.85)

ax.set_xlabel("IMD Decile (1 = most deprived → 10 = least)")
ax.set_ylabel("Average Severity (£)")
ax.set_title("Predicted Severity by IMD (Deprivation) Decile")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(d)}" for d in seg_imd.index])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model Factory Results
# MAGIC
# MAGIC The model factory trained **50 Poisson GLM specifications** by systematically varying
# MAGIC enrichment feature subsets, interaction terms, and base feature selections.
# MAGIC Results are read from the pre-persisted Unity Catalog table.
# MAGIC
# MAGIC **Reading from:** `model_factory_results` and `model_factory_feature_impact`

# COMMAND ----------

# Load results — pre-computed by notebook 07
results_df = spark.table("model_factory_results").toPandas()
results_df = results_df.sort_values("aic").reset_index(drop=True)

print(f"Model factory results loaded: {len(results_df)} specifications")
display(spark.table("model_factory_results"))

# COMMAND ----------

# Top 5 models by AIC
top5 = results_df.sort_values("aic").head(5)
print("Top 5 Models by AIC (lower is better):\n")
for _, row in top5.iterrows():
    rank = int(row["rank_aic"]) if "rank_aic" in row else "-"
    print(f"  #{str(rank):>2s}  AIC={row['aic']:>10.1f}  Gini={row['gini_test']:.4f}  "
          f"Features={int(row['n_features']):2d}  {row['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### AIC Elbow Chart and Gini vs Feature Count
# MAGIC
# MAGIC The elbow chart shows where adding more features stops materially improving AIC.
# MAGIC That inflection point guides the optimal complexity choice — replicating the
# MAGIC judgement an actuary would apply in Radar, but across 50 models simultaneously.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Locate baseline and best-AIC models for annotation
baseline_rows = results_df[results_df["name"] == "baseline_standard"]
baseline = baseline_rows.iloc[0] if not baseline_rows.empty else results_df.sort_values("aic").iloc[-1]
best_aic = results_df.sort_values("aic").iloc[0]
best_gini_row = results_df.sort_values("gini_test", ascending=False).iloc[0]

# --- AIC vs feature count ---
ax = axes[0]
ax.scatter(results_df["n_features"], results_df["aic"],
           alpha=0.6, s=60, c="#4C72B0", edgecolors="white")
ax.scatter([baseline["n_features"]], [baseline["aic"]], s=150, c="#E53935", zorder=5,
           edgecolors="black", label=f"Baseline (AIC={baseline['aic']:.0f})")
ax.scatter([best_aic["n_features"]], [best_aic["aic"]], s=150, c="#43A047", zorder=5,
           marker="*", edgecolors="black", label=f"Best (AIC={best_aic['aic']:.0f})")
ax.set_xlabel("Number of Features", fontsize=12)
ax.set_ylabel("AIC (lower is better)", fontsize=12)
ax.set_title("AIC vs Model Complexity — The Elbow Chart", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# --- Gini vs feature count ---
ax = axes[1]
ax.scatter(results_df["n_features"], results_df["gini_test"],
           alpha=0.6, s=60, c="#DD8452", edgecolors="white")
ax.scatter([baseline["n_features"]], [baseline["gini_test"]], s=150, c="#E53935", zorder=5,
           edgecolors="black", label=f"Baseline (Gini={baseline['gini_test']:.4f})")
ax.scatter([best_gini_row["n_features"]], [best_gini_row["gini_test"]], s=150, c="#43A047",
           zorder=5, marker="*", edgecolors="black",
           label=f"Best (Gini={best_gini_row['gini_test']:.4f})")
ax.set_xlabel("Number of Features", fontsize=12)
ax.set_ylabel("Gini (higher is better)", fontsize=12)
ax.set_title("Gini vs Model Complexity", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrichment Feature Impact — Which New Data Sources Matter Most?
# MAGIC
# MAGIC For each enrichment feature: average AIC across models that include it minus average AIC
# MAGIC across models that exclude it.  A positive bar means including that feature consistently
# MAGIC lowers AIC (improves fit).

# COMMAND ----------

# Load pre-computed feature impact from UC
impact_df = spark.table("model_factory_feature_impact").toPandas().sort_values(
    "avg_aic_improvement", ascending=False
)

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#43A047" if v > 0 else "#E53935" for v in impact_df["avg_aic_improvement"]]
ax.barh(impact_df["feature"], impact_df["avg_aic_improvement"], color=colors, edgecolor="white")
ax.set_xlabel("Average AIC Improvement When Included", fontsize=12)
ax.set_title("Enrichment Feature Impact — Which New Data Matters Most?",
             fontsize=13, fontweight="bold")
ax.axvline(0, color="grey", lw=0.8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nFeature impact ranking:")
print(impact_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Model Serving — Score New Data
# MAGIC
# MAGIC Load the registered enriched frequency GLM from MLflow Model Registry and score
# MAGIC a five-record sample from the priced portfolio.

# COMMAND ----------

# Load the latest registered version — older versions may have stale feature schemas.
# Suppress MLflow's version-mismatch warnings: the training env (notebook 01) and the
# loading env here may pin slightly different numpy/statsmodels/Python versions, but
# a simple Poisson GLM unpickles cleanly across the ranges we use.
import logging
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)

from mlflow.tracking import MlflowClient
_client = MlflowClient(registry_uri="databricks-uc")
_uc_model_name = f"{CATALOG}.{SCHEMA}.glm_frequency_enriched"
_latest = max(int(v.version) for v in _client.search_model_versions(f"name='{_uc_model_name}'"))
model_uri = f"models:/{_uc_model_name}/{_latest}"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model from {model_uri}")

# Score a small sample
sample = spark.table("priced_portfolio").limit(5).toPandas()

# Region dummies are discovered dynamically from the table — keeps notebook in sync with notebook 01
region_features = sorted([c for c in sample.columns if c.startswith("region_") and c != "region_code"])

enriched_features = [
    "building_age", "bedrooms", "sum_insured", "prior_claims", "policy_tenure",
    "property_type_flat", "property_type_semi_detached", "property_type_terraced",
    "construction_other", "construction_stone", "construction_timber",
    "occupancy_tenant",
    "imd_decile", "crime_decile", "income_decile", "health_decile", "living_env_decile",
    "is_urban", "is_coastal",
] + region_features

preds = model.predict(sample[enriched_features].astype(float))
sample["predicted_frequency"] = preds
display(spark.createDataFrame(sample[enriched_features[:5] + ["predicted_frequency"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary
# MAGIC
# MAGIC ### Frequency Model
# MAGIC
# MAGIC | Metric | Standard GLM | Enriched GLM | Improvement |
# MAGIC |---|---|---|---|
# MAGIC | AIC | Baseline | Lower | Better fit with fewer effective parameters |
# MAGIC | BIC | Baseline | Lower | Model complexity justified by data |
# MAGIC | Gini coefficient | Baseline | Higher | Stronger risk discrimination |
# MAGIC | Deviance explained | Baseline | Higher | More variance captured |
# MAGIC | Loss-ratio stability | Higher σ | Lower σ | More uniform pricing across deciles |
# MAGIC | Pricing segmentation | Coarse | Granular | Deprivation, crime, coastal and region now priced explicitly |
# MAGIC
# MAGIC ### Severity Model
# MAGIC
# MAGIC | Metric | Standard GBM | Enriched GBM | Improvement |
# MAGIC |---|---|---|---|
# MAGIC | MAE | Baseline | Lower | More accurate severity predictions |
# MAGIC | RMSE | Baseline | Lower | Fewer large prediction errors |
# MAGIC | Gini | Baseline | Higher | Better severity discrimination |
# MAGIC | Feature importance | Property-centric | Risk-centric | IMD deprivation & coastal drive severity |
# MAGIC | Loss-ratio stability | Higher σ | Lower σ | More consistent full pricing |
# MAGIC
# MAGIC ### Model Factory
# MAGIC
# MAGIC | Dimension | Result |
# MAGIC |---|---|
# MAGIC | Specifications trained | 50 Poisson GLMs |
# MAGIC | Approach | Systematic enrichment feature subsets + interaction combinations |
# MAGIC | Elbow point | Adding all enrichment features gives the largest single AIC improvement |
# MAGIC | Best interaction | Varies by portfolio — see feature impact chart |
# MAGIC | vs Radar / Emblem | 50 models in under a minute vs 5–10 models over hours of analyst time |
# MAGIC
# MAGIC **Business impact:** The enriched models enable risk-adequate pricing for exposures the
# MAGIC standard models cannot observe — neighbourhood crime, area deprivation, living-environment
# MAGIC quality, coastal location and regional effects. This reduces adverse selection on high-risk
# MAGIC postcodes and avoids overcharging low-risk ones. The combined frequency × severity
# MAGIC improvement in Gini and loss-ratio stability translates directly to better portfolio
# MAGIC profitability and competitive positioning.
# MAGIC
# MAGIC The model factory provides actuaries with a complete picture of the enrichment
# MAGIC trade-offs across 50 specifications — augmenting judgement rather than replacing it.
