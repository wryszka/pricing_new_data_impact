# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: Impact of New Data on Severity GBM — Home Insurance
# MAGIC
# MAGIC This notebook walks through the results of a severity modelling exercise comparing a
# MAGIC **standard GBM** (traditional rating factors only) against an **enriched GBM** (augmented
# MAGIC with external data: flood risk, crime index, subsidence, etc.).
# MAGIC
# MAGIC All artefacts are materialised in Unity Catalog by notebook 04.

# COMMAND ----------

# MAGIC %pip install matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Setup
CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12, "axes.labelsize": 10})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Claimant Population Overview

# COMMAND ----------

train = spark.table("severity_train_set").toPandas()
test = spark.table("severity_test_set").toPandas()

n_train = len(train)
n_test = len(test)
avg_sev = pd.concat([train, test])["claim_severity"].mean()
med_sev = pd.concat([train, test])["claim_severity"].median()

print(f"Claimants — Train: {n_train:,} | Test: {n_test:,}")
print(f"Avg severity:    £{avg_sev:,.0f}")
print(f"Median severity: £{med_sev:,.0f}")

# COMMAND ----------

# Severity distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

all_claimants = pd.concat([train, test])

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

# MAGIC %md
# MAGIC ## 2. Model Comparison

# COMMAND ----------

model_comparison = spark.table("severity_model_comparison")
display(model_comparison)

# COMMAND ----------

mc = model_comparison.toPandas().set_index("metric")

std_vals = mc["model_1_standard"].astype(float)
enr_vals = mc["model_2_enriched"].astype(float)

for metric in mc.index:
    s, e = std_vals[metric], enr_vals[metric]
    lower_better = metric in ["MAE (£)", "RMSE (£)", "MAPE (%)", "Bias (%)"]
    if lower_better:
        direction = "▼ better" if e < s else "▲ worse"
    else:
        direction = "▲ better" if e > s else "▼ worse"
    print(f"{metric:25s}  standard={s:>12.4f}   enriched={e:>12.4f}   {direction}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Importance

# COMMAND ----------

importance = spark.table("severity_feature_importance").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, model_name, title in [
    (axes[0], "standard", "Standard GBM"),
    (axes[1], "enriched", "Enriched GBM"),
]:
    subset = importance[importance["model"] == model_name].sort_values("importance", ascending=True)
    ax.barh(subset["feature"], subset["importance"], color="#4C72B0" if model_name == "standard" else "#DD8452")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Feature Importance — {title}")

plt.tight_layout()
plt.show()

# Enrichment feature share of total importance
enr_features = {"flood_risk_zone", "crime_index", "distance_fire_station_km", "annual_rainfall_mm", "subsidence_risk"}
enr_imp = importance[(importance["model"] == "enriched") & (importance["feature"].isin(enr_features))]
total_imp = importance[importance["model"] == "enriched"]["importance"].sum()
enr_share = enr_imp["importance"].sum() / total_imp * 100
print(f"Enrichment features account for {enr_share:.1f}% of total importance in the enriched model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Loss Ratio by Decile — Full Quotes (Freq × Sev)

# COMMAND ----------

lr_decile = spark.table("severity_loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(10, 5))
for model, grp in lr_decile.groupby("model"):
    grp_sorted = grp.sort_values("decile")
    color = "#E53935" if model == "Standard" else "#43A047"
    ax.plot(grp_sorted["decile"], grp_sorted["loss_ratio"], marker="o", label=model, color=color, linewidth=2)

ax.axhline(1.0, ls="--", color="grey", lw=0.8, label="Breakeven")
ax.fill_between(range(10), 0.85, 1.15, color="green", alpha=0.07, label="Healthy range")
ax.set_xlabel("Full Quote Decile (cheapest → most expensive)")
ax.set_ylabel("Loss Ratio")
ax.set_title("Loss Ratio by Decile — Full Burning-Cost Quotes (Freq × Sev)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.show()

# Stability
lr_stability = lr_decile.groupby("model")["loss_ratio"].std().reset_index()
lr_stability.columns = ["model", "lr_std_dev"]
print("Loss-ratio stability (lower = more uniform pricing):")
print(lr_stability.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pricing Impact — Severity Component

# COMMAND ----------

priced = spark.table("severity_priced_portfolio").toPandas()

priced["sev_diff"] = priced["sev_pred_enriched"] - priced["sev_pred_standard"]
priced["full_quote_diff"] = priced["full_quote_enriched"] - priced["full_quote_standard"]

print(f"Mean severity diff:           £{priced['sev_diff'].abs().mean():.2f}")
print(f"Mean full-quote diff:         £{priced['full_quote_diff'].abs().mean():.2f}")
print(f"Max full-quote increase:      £{priced['full_quote_diff'].max():.2f}")
print(f"Max full-quote decrease:      £{priced['full_quote_diff'].min():.2f}")

# COMMAND ----------

# Scatter — standard vs enriched severity predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Severity predictions
ax = axes[0]
ax.scatter(priced["sev_pred_standard"], priced["sev_pred_enriched"], alpha=0.1, s=4, color="#4C72B0")
lims = [
    min(priced["sev_pred_standard"].min(), priced["sev_pred_enriched"].min()),
    max(priced["sev_pred_standard"].max(), priced["sev_pred_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Severity (£)")
ax.set_ylabel("Enriched Severity (£)")
ax.set_title("Severity Predictions: Standard vs Enriched")

# Full quotes
ax = axes[1]
ax.scatter(priced["full_quote_standard"], priced["full_quote_enriched"], alpha=0.1, s=4, color="#DD8452")
lims = [
    min(priced["full_quote_standard"].min(), priced["full_quote_enriched"].min()),
    max(priced["full_quote_standard"].max(), priced["full_quote_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Full Quote (£)")
ax.set_ylabel("Enriched Full Quote (£)")
ax.set_title("Full Quotes (Freq×Sev): Standard vs Enriched")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Segmentation — Severity by Risk Factor

# COMMAND ----------

for col, labels in [("flood_risk_zone", None), ("subsidence_risk", {0: "No Risk", 1: "At Risk"})]:
    seg = priced.groupby(col).agg(
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

# Bar chart — severity by flood risk zone
fig, ax = plt.subplots(figsize=(10, 6))

seg_flood = priced.groupby("flood_risk_zone").agg(
    standard=("sev_pred_standard", "mean"),
    enriched=("sev_pred_enriched", "mean"),
    actual=("claim_severity", "mean"),
).sort_index()

x = np.arange(len(seg_flood))
width = 0.25

ax.bar(x - width, seg_flood["standard"], width, label="Standard GBM", color="#E53935", alpha=0.85)
ax.bar(x, seg_flood["enriched"], width, label="Enriched GBM", color="#1E88E5", alpha=0.85)
ax.bar(x + width, seg_flood["actual"], width, label="Actual Avg Severity", color="#43A047", alpha=0.85)

ax.set_xlabel("Flood Risk Zone (1 = Low → 4 = High)")
ax.set_ylabel("Average Severity (£)")
ax.set_title("Predicted Severity by Flood Risk Zone")
ax.set_xticks(x)
ax.set_xticklabels(seg_flood.index)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Standard GBM | Enriched GBM | Improvement |
# MAGIC |---|---|---|---|
# MAGIC | MAE | Baseline | Lower | More accurate severity predictions |
# MAGIC | RMSE | Baseline | Lower | Fewer large prediction errors |
# MAGIC | Gini | Baseline | Higher | Better severity discrimination |
# MAGIC | Feature importance | Property-centric | Risk-centric | Flood & subsidence drive severity |
# MAGIC | Loss-ratio stability | Higher σ | Lower σ | More consistent full pricing |
# MAGIC
# MAGIC **Key insight:** Adding the enrichment data improves both frequency AND severity models.
# MAGIC The combined effect (freq × sev) produces materially better full burning-cost quotes,
# MAGIC with tighter loss ratios and more accurate risk segmentation.
