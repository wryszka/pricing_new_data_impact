# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: Impact of New Data on GLM Pricing — Home Insurance
# MAGIC
# MAGIC This notebook walks through the results of a modelling exercise comparing a **standard GLM** (traditional rating factors only) against an **enriched GLM** (augmented with external data: flood risk, crime index, subsidence, etc.).
# MAGIC All artefacts — portfolios, coefficients, model metrics, and scored quotes — are already materialised in Unity Catalog.

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
# MAGIC ## 1. Portfolio Overview

# COMMAND ----------

portfolio = spark.table("portfolio")
pdf = portfolio.toPandas()

n_policies = len(pdf)
claim_rate = (pdf["num_claims"] > 0).mean()
avg_severity = pdf.loc[pdf["num_claims"] > 0, "claim_severity"].mean()

print(f"Policies:        {n_policies:,}")
print(f"Claim rate:      {claim_rate:.2%}")
print(f"Avg severity:    £{avg_severity:,.0f}")

# COMMAND ----------

# Distribution of key features
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

feature_cols = [c for c in ["property_type", "flood_risk_zone", "construction"] if c in pdf.columns][:3]
for ax, col in zip(axes, feature_cols):
    pdf[col].value_counts().sort_index().plot.bar(ax=ax, edgecolor="white")
    ax.set_title(col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Comparison

# COMMAND ----------

model_comparison = spark.table("model_comparison")
display(model_comparison)

# COMMAND ----------

mc = model_comparison.toPandas().set_index("metric") if "metric" in model_comparison.columns else model_comparison.toPandas()

# Quick highlights
if "metric" in model_comparison.columns:
    std_vals = mc["model_1_standard"].astype(float)
    enr_vals = mc["model_2_enriched"].astype(float)
    for metric in ["AIC", "BIC", "Gini (test)", "Deviance Explained"]:
        if metric in mc.index:
            s, e = std_vals[metric], enr_vals[metric]
            direction = "▼" if e < s else "▲"
            print(f"{metric:25s}  standard={s:>10.2f}   enriched={e:>10.2f}   {direction} {abs(e - s):.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Coefficient Analysis

# COMMAND ----------

coefficients = spark.table("glm_coefficients").toPandas()

# Pivot to show standard vs enriched side by side
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

# Enrichment features — statistical significance
enrichment_features = ["flood_risk_zone", "crime_index", "subsidence_risk",
                       "distance_fire_station_km", "annual_rainfall_mm"]
enrich_mask = coefficients["feature"].isin(enrichment_features) & (coefficients["model"].str.contains("enriched", case=False))
sig = coefficients.loc[enrich_mask, ["feature", "coef", "p_value"]].sort_values("p_value")
if not sig.empty:
    print("Enrichment features — enriched model")
    print(sig.to_string(index=False))

# COMMAND ----------

# Coefficient comparison chart
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
# MAGIC ## 4. Loss Ratio by Decile

# COMMAND ----------

lr_decile = spark.table("loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(8, 5))
for model, grp in lr_decile.groupby("model"):
    grp_sorted = grp.sort_values("decile")
    ax.plot(grp_sorted["decile"], grp_sorted["loss_ratio"], marker="o", label=model)

ax.axhline(1.0, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Decile")
ax.set_ylabel("Loss Ratio")
ax.set_title("Loss Ratio by Risk Decile")
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# Stability metric — LR standard deviation across deciles
lr_stability = lr_decile.groupby("model")["loss_ratio"].std().reset_index()
lr_stability.columns = ["model", "lr_std_dev"]
print("Loss-ratio stability (lower = more uniform pricing):")
print(lr_stability.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Pricing Impact

# COMMAND ----------

priced = spark.table("priced_portfolio").toPandas()
priced["quote_diff"] = priced["quote_enriched"] - priced["quote_standard"]
priced["quote_diff_pct"] = priced["quote_diff"] / priced["quote_standard"] * 100

print(f"Mean absolute quote difference: £{priced['quote_diff'].abs().mean():.2f}")
print(f"Median quote difference:        £{priced['quote_diff'].median():.2f}")
print(f"Max increase:                    £{priced['quote_diff'].max():.2f}")
print(f"Max decrease:                    £{priced['quote_diff'].min():.2f}")

# COMMAND ----------

# Where pricing diverges most — high flood risk and subsidence
high_risk = priced[
    (priced["flood_risk_zone"] >= 3) | (priced["subsidence_risk"] == 1)
].nlargest(10, "quote_diff")

display(spark.createDataFrame(high_risk[["flood_risk_zone", "subsidence_risk",
                                          "quote_standard", "quote_enriched", "quote_diff"]]))

# COMMAND ----------

# Scatter — standard vs enriched quotes
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(priced["quote_standard"], priced["quote_enriched"], alpha=0.15, s=6, color="#4C72B0")
lims = [
    min(priced["quote_standard"].min(), priced["quote_enriched"].min()),
    max(priced["quote_standard"].max(), priced["quote_enriched"].max()),
]
ax.plot(lims, lims, ls="--", color="grey", lw=0.8)
ax.set_xlabel("Standard Quote (£)")
ax.set_ylabel("Enriched Quote (£)")
ax.set_title("Quote Comparison: Standard vs Enriched GLM")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Segmentation Deep Dive

# COMMAND ----------

seg_cols = [c for c in ["flood_risk_zone", "subsidence_risk"] if c in priced.columns]

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

# Bar chart — average quotes by flood_risk_zone
if "flood_risk_zone" in priced.columns:
    seg_flood = priced.groupby("flood_risk_zone")[["quote_standard", "quote_enriched"]].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(seg_flood))
    width = 0.35
    ax.bar(x - width / 2, seg_flood["quote_standard"], width, label="Standard", color="#4C72B0")
    ax.bar(x + width / 2, seg_flood["quote_enriched"], width, label="Enriched", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_flood.index, rotation=45, ha="right")
    ax.set_ylabel("Average Quote (£)")
    ax.set_title("Average Quote by Flood Risk Zone")
    ax.legend()
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Serving — Score New Data

# COMMAND ----------

import mlflow

model_uri = f"models:/{CATALOG}.{SCHEMA}.glm_frequency_enriched/1"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model from {model_uri}")

# Score a small sample
sample = spark.table("priced_portfolio").limit(5).toPandas()
enriched_features = [
    "building_age", "bedrooms", "sum_insured", "prior_claims", "policy_tenure",
    "property_type_flat", "property_type_semi_detached", "property_type_terraced",
    "construction_other", "construction_stone", "construction_timber",
    "occupancy_tenant",
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk",
]
preds = model.predict(sample[enriched_features].astype(float))
sample["predicted_frequency"] = preds
display(spark.createDataFrame(sample[enriched_features[:5] + ["predicted_frequency"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Standard GLM | Enriched GLM | Improvement |
# MAGIC |---|---|---|---|
# MAGIC | AIC | Baseline | Lower | Better fit with fewer effective parameters |
# MAGIC | BIC | Baseline | Lower | Model complexity justified by data |
# MAGIC | Gini coefficient | Baseline | Higher | Stronger risk discrimination |
# MAGIC | Deviance explained | Baseline | Higher | More variance captured |
# MAGIC | Loss-ratio stability | Higher σ | Lower σ | More uniform pricing across deciles |
# MAGIC | Pricing segmentation | Coarse | Granular | Flood, subsidence, crime now priced explicitly |
# MAGIC
# MAGIC **Business impact:** The enriched model enables risk-adequate pricing for perils the standard model cannot observe — flood, subsidence, crime exposure. This reduces adverse selection on high-risk properties and avoids overcharging low-risk ones. The improvement in Gini and loss-ratio stability translates directly to better portfolio profitability and competitive positioning.
