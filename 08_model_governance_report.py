# Databricks notebook source
# MAGIC %md
# MAGIC # Model Governance Report
# MAGIC ## Home Insurance Pricing — Enriched GLM & Severity Model Review
# MAGIC
# MAGIC **Document Purpose:** This notebook generates a structured governance report for internal
# MAGIC actuarial review and regulatory submission. It consolidates evidence from the full modelling
# MAGIC pipeline — GLM frequency models, LightGBM severity models, and the 50-specification model
# MAGIC factory — into a single auditable artefact.
# MAGIC
# MAGIC | Field | Value |
# MAGIC |---|---|
# MAGIC | **Report Date** | Auto-generated at runtime |
# MAGIC | **Catalog** | `lr_serverless_aws_us_catalog.pricing_new_data_impact` |
# MAGIC | **Models in Scope** | GLM Frequency (Standard & Enriched), LightGBM Severity (Standard & Enriched) |
# MAGIC | **Model Factory** | 50 Poisson GLM specifications, systematic search over enrichment feature subsets |
# MAGIC | **Intended Audience** | Chief Actuary, Pricing Committee, Internal Audit, Regulatory Submissions |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **How to use this notebook:**
# MAGIC Run all cells in order. The final cells produce a governance summary table written back to
# MAGIC Unity Catalog (`model_governance_summary`) and a copy-pasteable text report.

# COMMAND ----------

# %pip install matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"USE {CATALOG}.{SCHEMA}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
print(f"Governance report generated: {REPORT_DATE}")
print(f"Catalog: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 1 — Model Inventory
# MAGIC
# MAGIC This section documents the full search space explored during model development.
# MAGIC Every specification trained is recorded in Unity Catalog for auditability.

# COMMAND ----------

# Load the model factory results — all 50 GLM specifications
factory_df = spark.table("model_factory_results").toPandas()

# Summary statistics of the search space
n_models = len(factory_df)
n_features_min = int(factory_df["n_features"].min())
n_features_max = int(factory_df["n_features"].max())
aic_range_low = factory_df["aic"].min()
aic_range_high = factory_df["aic"].max()
gini_range_low = factory_df["gini_test"].min()
gini_range_high = factory_df["gini_test"].max()

print("=" * 60)
print("MODEL INVENTORY — Search Space Summary")
print("=" * 60)
print(f"  Total model specifications trained : {n_models}")
print(f"  Feature count range                : {n_features_min} – {n_features_max}")
print(f"  AIC range                          : {aic_range_low:,.1f} – {aic_range_high:,.1f}")
print(f"  Gini range (test set)              : {gini_range_low:.4f} – {gini_range_high:.4f}")
print()

# Summarise by spec category (inferred from name prefix)
def categorise_spec(name):
    if name == "baseline_standard":
        return "Baseline (standard only)"
    elif name.startswith("standard_plus_") and name.count("_") <= 4:
        return "Standard + 1 enrichment feature"
    elif name.startswith("standard_plus_"):
        return "Standard + 2 enrichment features"
    elif name.startswith("enrich_3"):
        return "Standard + 3 enrichment features"
    elif name.startswith("enrich_4"):
        return "Standard + 4 enrichment features"
    elif name == "full_enrichment":
        return "Standard + all 5 enrichment features"
    elif name.startswith("full_plus_"):
        return "Full enrichment + 1 interaction"
    elif name.startswith("full_ix_"):
        return "Full enrichment + 2 interactions"
    elif name.startswith("full_no_"):
        return "Full enrichment minus one base feature"
    elif name == "kitchen_sink":
        return "All features + all interactions"
    else:
        return "Other"

factory_df["category"] = factory_df["name"].apply(categorise_spec)

category_summary = (
    factory_df.groupby("category")
    .agg(
        count=("name", "count"),
        best_aic=("aic", "min"),
        avg_gini=("gini_test", "mean"),
    )
    .reset_index()
    .sort_values("best_aic")
    .rename(columns={
        "category": "Specification Category",
        "count": "# Specs",
        "best_aic": "Best AIC in Category",
        "avg_gini": "Avg Gini",
    })
)

print("Specification Categories:")
display(spark.createDataFrame(category_summary))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full Model Inventory Table
# MAGIC
# MAGIC All 50 specifications, ranked by AIC (lower is better).

# COMMAND ----------

inventory_display = factory_df[[
    "rank_aic", "spec_id", "name", "description", "n_features",
    "aic", "bic", "gini_test", "deviance_explained"
]].sort_values("rank_aic").rename(columns={
    "rank_aic": "AIC Rank",
    "spec_id": "Spec ID",
    "name": "Model Name",
    "description": "Description",
    "n_features": "# Features",
    "aic": "AIC",
    "bic": "BIC",
    "gini_test": "Gini (test)",
    "deviance_explained": "Deviance Explained",
})

display(spark.createDataFrame(inventory_display))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 2 — Recommended Model & Selection Rationale
# MAGIC
# MAGIC The recommended model is identified as the specification with the lowest AIC across the
# MAGIC full factory search. AIC balances goodness-of-fit against model complexity, making it
# MAGIC appropriate for comparing nested and non-nested GLM specifications.

# COMMAND ----------

# Identify best model by AIC
best_model = factory_df.sort_values("aic").iloc[0]
baseline_model = factory_df[factory_df["name"] == "baseline_standard"].iloc[0]

# Rank the best model by other criteria
factory_sorted_bic = factory_df.sort_values("bic").reset_index(drop=True)
factory_sorted_gini = factory_df.sort_values("gini_test", ascending=False).reset_index(drop=True)

best_bic_rank = int(factory_sorted_bic[factory_sorted_bic["name"] == best_model["name"]].index[0]) + 1
best_gini_rank = int(factory_sorted_gini[factory_sorted_gini["name"] == best_model["name"]].index[0]) + 1

aic_improvement = baseline_model["aic"] - best_model["aic"]
gini_improvement = best_model["gini_test"] - baseline_model["gini_test"]
deviance_improvement = best_model["deviance_explained"] - baseline_model["deviance_explained"]

print("=" * 60)
print("RECOMMENDED MODEL — Selection Summary")
print("=" * 60)
print(f"  Recommended model   : {best_model['name']}")
print(f"  Description         : {best_model['description']}")
print(f"  Number of features  : {int(best_model['n_features'])}")
print()
print("  Performance vs Baseline:")
print(f"    AIC  : {best_model['aic']:>10,.1f}  (baseline: {baseline_model['aic']:,.1f}, improvement: {aic_improvement:,.1f})")
print(f"    BIC  : {best_model['bic']:>10,.1f}  (baseline: {baseline_model['bic']:,.1f})")
print(f"    Gini : {best_model['gini_test']:>10.4f}  (baseline: {baseline_model['gini_test']:.4f}, improvement: +{gini_improvement:.4f})")
print(f"    Dev. : {best_model['deviance_explained']:>10.4f}  (baseline: {baseline_model['deviance_explained']:.4f}, improvement: +{deviance_improvement:.4f})")
print()
print("  Rankings across 50 specifications:")
print(f"    AIC rank  : #{int(best_model['rank_aic'])} of {n_models}")
print(f"    BIC rank  : #{best_bic_rank} of {n_models}")
print(f"    Gini rank : #{best_gini_rank} of {n_models}")

# COMMAND ----------

# Selection rationale chart — AIC ranking with baseline and best highlighted
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: AIC distribution
ax = axes[0]
colors_bar = ["#43A047" if row["name"] == best_model["name"]
              else "#E53935" if row["name"] == "baseline_standard"
              else "#90CAF9"
              for _, row in factory_df.sort_values("rank_aic").iterrows()]

ax.barh(
    range(len(factory_df)),
    factory_df.sort_values("rank_aic")["aic"],
    color=colors_bar,
    edgecolor="white",
    height=0.7,
)
ax.set_yticks([0, len(factory_df) // 2, len(factory_df) - 1])
ax.set_yticklabels(["Best (Rank 1)", f"Median (Rank {n_models//2})", f"Worst (Rank {n_models})"])
ax.set_xlabel("AIC (lower is better)", fontsize=11)
ax.set_title("AIC Distribution — All 50 Specifications", fontsize=12, fontweight="bold")
ax.axvline(best_model["aic"], color="#43A047", linestyle="--", alpha=0.7, label=f"Best: {best_model['aic']:,.0f}")
ax.axvline(baseline_model["aic"], color="#E53935", linestyle="--", alpha=0.7, label=f"Baseline: {baseline_model['aic']:,.0f}")
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)

# Right: Multi-criteria ranking comparison (AIC vs Gini rank)
ax = axes[1]
ax.scatter(factory_df["rank_aic"], factory_df["rank_gini"], alpha=0.5, s=50, c="#4C72B0", edgecolors="white")

# Highlight best and baseline
ax.scatter(
    [best_model["rank_aic"]], [best_gini_rank],
    s=200, c="#43A047", zorder=5, marker="*", edgecolors="black",
    label=f"Recommended: AIC #{int(best_model['rank_aic'])}, Gini #{best_gini_rank}",
)
baseline_gini_rank = int(factory_sorted_gini[factory_sorted_gini["name"] == "baseline_standard"].index[0]) + 1
ax.scatter(
    [int(baseline_model["rank_aic"])], [baseline_gini_rank],
    s=150, c="#E53935", zorder=5, edgecolors="black",
    label=f"Baseline: AIC #{int(baseline_model['rank_aic'])}, Gini #{baseline_gini_rank}",
)

ax.set_xlabel("AIC Rank (1 = best)", fontsize=11)
ax.set_ylabel("Gini Rank (1 = best)", fontsize=11)
ax.set_title("AIC vs Gini Ranking — Model Consistency", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# Diagonal — if AIC and Gini perfectly agree, all points lie on this line
ax.plot([1, n_models], [1, n_models], "k--", alpha=0.2, linewidth=1)
ax.set_xlim(0, n_models + 1)
ax.set_ylim(0, n_models + 1)

plt.suptitle("Section 2 — Model Selection Evidence", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selection Rationale Summary
# MAGIC
# MAGIC The recommended model was selected based on the following criteria:
# MAGIC
# MAGIC 1. **AIC (primary criterion):** Penalises complexity; the recommended model achieves the lowest
# MAGIC    AIC across the full 50-specification search space.
# MAGIC
# MAGIC 2. **BIC consistency:** BIC applies a heavier complexity penalty than AIC. A high BIC rank
# MAGIC    confirms the model is not over-fitted — it genuinely improves fit relative to its size.
# MAGIC
# MAGIC 3. **Gini coefficient (discrimination):** Measures the model's ability to rank risks from
# MAGIC    low to high claim probability on the held-out test set. Higher Gini indicates better
# MAGIC    risk segmentation.
# MAGIC
# MAGIC 4. **Parsimony:** Where two models achieve similar AIC, the simpler model is preferred
# MAGIC    for interpretability and regulatory robustness.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 3 — Feature Justification
# MAGIC
# MAGIC For each feature in the recommended model, we document the statistical evidence (coefficient,
# MAGIC p-value) alongside a plain-English business rationale. This section is intended to support
# MAGIC regulatory review and satisfy requirements around model explainability.

# COMMAND ----------

# Business rationale for each feature — plain English for governance documentation
FEATURE_RATIONALE = {
    "const": "Intercept — captures the baseline log-frequency for the reference risk profile.",
    "building_age": (
        "Older buildings have higher claim frequency due to deteriorating structures, "
        "outdated wiring, and aging plumbing systems. A positive coefficient is expected "
        "and consistent with industry experience."
    ),
    "bedrooms": (
        "Proxy for property size and value at risk. Larger properties have more rooms and "
        "contents exposed to loss events; also correlates with occupant density."
    ),
    "sum_insured": (
        "Declared rebuild value. Higher sums insured are associated with larger, more complex "
        "properties that may carry greater physical risk."
    ),
    "prior_claims": (
        "Number of claims in prior policy years. Strong predictor of future claim behaviour — "
        "reflects both inherent risk characteristics of the property and policyholder behaviour."
    ),
    "policy_tenure": (
        "Years the policyholder has been with the insurer. Longer-tenured customers typically "
        "exhibit lower claim frequency, consistent with adverse selection dynamics at inception."
    ),
    "property_type_flat": (
        "Flats (apartments) have shared structural elements and communal areas that alter "
        "the nature and frequency of claims relative to detached houses (reference category)."
    ),
    "property_type_semi_detached": (
        "Semi-detached properties share a party wall, which can affect subsidence, dampness, "
        "and escape of water claims compared to the detached reference category."
    ),
    "property_type_terraced": (
        "Terraced properties have shared walls on both sides; claim patterns differ from "
        "detached properties particularly for escape of water and structural claims."
    ),
    "construction_other": (
        "Non-standard construction materials (not brick, stone, or timber) may have unusual "
        "risk characteristics that are harder to underwrite; a positive loading is expected."
    ),
    "construction_stone": (
        "Stone construction is common in older properties; while durable, such properties "
        "often have lower rebuild costs and different claim profiles to brick (reference)."
    ),
    "construction_timber": (
        "Timber-frame construction carries elevated fire risk and is more susceptible to "
        "moisture ingress. A positive coefficient versus the brick reference is expected."
    ),
    "occupancy_tenant": (
        "Tenanted properties exhibit higher claim frequency in most home insurance portfolios, "
        "attributed to differences in care of the property and incentive alignment with the insurer."
    ),
    "flood_risk_zone": (
        "Flood risk zone (1–4, low to high) derived from Environmental Agency data. "
        "Properties in higher flood zones have materially elevated claim frequency for "
        "escape of water and structural damage events."
    ),
    "crime_index": (
        "Area-level crime index (0–100). Higher crime areas are associated with greater "
        "frequency of theft, accidental damage, and malicious damage claims."
    ),
    "distance_fire_station_km": (
        "Distance to the nearest fire station in kilometres. Greater distances increase "
        "fire claim severity and may increase frequency for properties where delayed response "
        "leads to greater secondary damage."
    ),
    "annual_rainfall_mm": (
        "Average annual rainfall at the property location (mm). Higher rainfall increases "
        "subsidence risk from ground movement, escape of water frequency, and storm damage."
    ),
    "subsidence_risk": (
        "Binary indicator of properties located on subsidence-prone ground (clay, chalk). "
        "Strong predictor of structural claims; materially significant in London and the South East."
    ),
}

# Load GLM coefficients from the enriched model
coef_df = spark.table("glm_coefficients").toPandas()
coef_enriched = coef_df[coef_df["model"] == "enriched"].copy()

# Flag statistical significance
coef_enriched["significance"] = coef_enriched["p_value"].apply(
    lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
)

# Add business rationale
coef_enriched["business_rationale"] = coef_enriched["feature"].map(
    lambda f: FEATURE_RATIONALE.get(f, "No rationale documented.")
)

# Build the feature justification table
feature_table = coef_enriched[[
    "feature", "coef", "std_err", "p_value", "significance", "business_rationale"
]].rename(columns={
    "feature": "Feature",
    "coef": "Coefficient",
    "std_err": "Std Error",
    "p_value": "P-Value",
    "significance": "Sig.",
    "business_rationale": "Business Rationale",
})

print("Feature Justification Table — Enriched GLM (Frequency Model)")
print(f"Significance codes: *** p<0.001 | ** p<0.01 | * p<0.05 | ns not significant")
print()
display(spark.createDataFrame(feature_table))

# COMMAND ----------

# Coefficient forest plot — shows direction and uncertainty for all features
fig, ax = plt.subplots(figsize=(12, 9))

plot_df = coef_enriched[coef_enriched["feature"] != "const"].sort_values("coef")
y_pos = range(len(plot_df))

# Colour: positive = risk-increasing (red), negative = risk-reducing (blue)
colors_coef = ["#E53935" if c > 0 else "#1E88E5" for c in plot_df["coef"]]

ax.barh(list(y_pos), plot_df["coef"], xerr=1.96 * plot_df["std_err"],
        color=colors_coef, edgecolor="white", height=0.6, capsize=3, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_df["feature"], fontsize=10)
ax.set_xlabel("Coefficient (log scale — Poisson GLM)", fontsize=11)
ax.set_title(
    "Section 3 — Feature Coefficients with 95% Confidence Intervals\n"
    "Enriched GLM Frequency Model",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

# Mark enrichment features
enrichment_features = [
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk"
]
for i, feat in enumerate(plot_df["feature"]):
    if feat in enrichment_features:
        ax.get_yticklabels()[i].set_color("#6A1B9A")
        ax.get_yticklabels()[i].set_fontweight("bold")

# Legend for colour coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#E53935", label="Risk-increasing (positive coefficient)"),
    Patch(facecolor="#1E88E5", label="Risk-reducing (negative coefficient)"),
    Patch(facecolor="#6A1B9A", label="Enrichment feature (purple label)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 4 — Performance Evidence
# MAGIC
# MAGIC This section presents the full performance evidence for the frequency and severity models.
# MAGIC We compare standard vs enriched specifications on held-out test data across multiple metrics.

# COMMAND ----------

# Load frequency model comparison
freq_comp_df = spark.table("model_comparison").toPandas()

# Load severity model comparison
sev_comp_df = spark.table("severity_model_comparison").toPandas()

print("Frequency Model Comparison (Poisson GLM):")
display(spark.createDataFrame(freq_comp_df))

# COMMAND ----------

print("Severity Model Comparison (LightGBM):")
display(spark.createDataFrame(sev_comp_df))

# COMMAND ----------

# Loss ratio stability analysis — std dev across deciles is the key metric
lr_freq_df = spark.table("loss_ratio_by_decile").toPandas()
lr_sev_df = spark.table("severity_loss_ratio_by_decile").toPandas()

print("Loss Ratio Stability (Frequency Models):")
print("-" * 50)
for model_name in lr_freq_df["model"].unique():
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    lr_std = subset["loss_ratio"].std()
    lr_mean = subset["loss_ratio"].mean()
    lr_min = subset["loss_ratio"].min()
    lr_max = subset["loss_ratio"].max()
    print(f"  {model_name:12s} | Mean LR: {lr_mean:.3f} | Std Dev: {lr_std:.3f} | Range: {lr_min:.3f}–{lr_max:.3f}")

print()
print("Loss Ratio Stability (Severity Models):")
print("-" * 50)
for model_name in lr_sev_df["model"].unique():
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    lr_std = subset["loss_ratio"].std()
    lr_mean = subset["loss_ratio"].mean()
    lr_min = subset["loss_ratio"].min()
    lr_max = subset["loss_ratio"].max()
    print(f"  {model_name:12s} | Mean LR: {lr_mean:.3f} | Std Dev: {lr_std:.3f} | Range: {lr_min:.3f}–{lr_max:.3f}")

# COMMAND ----------

# Combined performance visualisation
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- Top left: Frequency LR by decile ---
ax1 = fig.add_subplot(gs[0, 0])
for model_name, color, ls in [("Standard", "#E53935", "-"), ("Enriched", "#43A047", "-")]:
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    if not subset.empty:
        ax1.plot(subset["decile"], subset["loss_ratio"], marker="o", label=model_name,
                 color=color, linewidth=2, linestyle=ls)
ax1.axhline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_xlabel("Premium Decile")
ax1.set_ylabel("Loss Ratio")
ax1.set_title("Frequency Model — Loss Ratio by Decile", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# --- Top right: Severity LR by decile ---
ax2 = fig.add_subplot(gs[0, 1])
for model_name, color, ls in [("Standard", "#E53935", "-"), ("Enriched", "#43A047", "-")]:
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    if not subset.empty:
        ax2.plot(subset["decile"], subset["loss_ratio"], marker="o", label=model_name,
                 color=color, linewidth=2, linestyle=ls)
ax2.axhline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=1)
ax2.set_xlabel("Premium Decile")
ax2.set_ylabel("Loss Ratio")
ax2.set_title("Severity Model — Loss Ratio by Decile", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# --- Bottom left: LR std dev comparison (stability bar chart) ---
ax3 = fig.add_subplot(gs[1, 0])
stability_data = []
for model_name in lr_freq_df["model"].unique():
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    stability_data.append({"model": f"Freq\n{model_name}", "lr_std": subset["loss_ratio"].std()})
for model_name in lr_sev_df["model"].unique():
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    stability_data.append({"model": f"Sev\n{model_name}", "lr_std": subset["loss_ratio"].std()})

stab_df = pd.DataFrame(stability_data)
bar_colors = ["#E53935" if "Standard" in m else "#43A047" for m in stab_df["model"]]
ax3.bar(stab_df["model"], stab_df["lr_std"], color=bar_colors, edgecolor="white", width=0.5)
ax3.set_ylabel("Loss Ratio Std Dev (lower = more stable)")
ax3.set_title("Pricing Stability — LR Volatility Across Deciles", fontweight="bold")
ax3.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
legend_els = [Patch(facecolor="#E53935", label="Standard"), Patch(facecolor="#43A047", label="Enriched")]
ax3.legend(handles=legend_els, fontsize=9)

# --- Bottom right: AIC/Gini improvement summary ---
ax4 = fig.add_subplot(gs[1, 1])
metrics_labels = ["AIC\nimprovement", "Gini\nimprovement", "Deviance\nimprovement"]
# Parse metrics from the comparison table
try:
    aic_std = float(freq_comp_df[freq_comp_df["metric"] == "AIC"]["model_1_standard"].values[0])
    aic_enr = float(freq_comp_df[freq_comp_df["metric"] == "AIC"]["model_2_enriched"].values[0])
    gini_std = float(freq_comp_df[freq_comp_df["metric"] == "Gini (test)"]["model_1_standard"].values[0])
    gini_enr = float(freq_comp_df[freq_comp_df["metric"] == "Gini (test)"]["model_2_enriched"].values[0])
    dev_std = float(freq_comp_df[freq_comp_df["metric"] == "Deviance Explained"]["model_1_standard"].values[0])
    dev_enr = float(freq_comp_df[freq_comp_df["metric"] == "Deviance Explained"]["model_2_enriched"].values[0])
    improvement_vals = [
        (aic_std - aic_enr) / abs(aic_std) * 100,  # AIC reduction % (positive = improvement)
        (gini_enr - gini_std) / gini_std * 100,     # Gini improvement %
        (dev_enr - dev_std) / dev_std * 100,         # Deviance improvement %
    ]
    bar_colors_imp = ["#43A047" if v > 0 else "#E53935" for v in improvement_vals]
    ax4.bar(metrics_labels, improvement_vals, color=bar_colors_imp, edgecolor="white", width=0.5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_ylabel("% Improvement (Enriched vs Standard)")
    ax4.set_title("Frequency Model — Enrichment Uplift", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)
except Exception as e:
    ax4.text(0.5, 0.5, f"Metric parsing error:\n{e}", ha="center", va="center", transform=ax4.transAxes)

plt.suptitle("Section 4 — Performance Evidence Summary", fontsize=14, fontweight="bold")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Summary Interpretation
# MAGIC
# MAGIC | Criterion | Standard Model | Enriched Model | Verdict |
# MAGIC |---|---|---|---|
# MAGIC | **AIC** | Higher | Lower | Enriched model fits better, penalised for complexity |
# MAGIC | **Gini coefficient** | Lower | Higher | Enriched model discriminates risk more effectively |
# MAGIC | **Deviance explained** | Lower | Higher | More variance in claims is explained |
# MAGIC | **LR stability (std dev)** | Higher | Lower | Enriched model produces more stable pricing |
# MAGIC | **LR monotonicity** | Partially monotone | More monotone | Enriched model ranks risk more consistently |
# MAGIC
# MAGIC A lower loss-ratio standard deviation across deciles indicates that the enriched model is
# MAGIC pricing risk more fairly — high-risk deciles are not systematically under-priced and
# MAGIC low-risk deciles are not systematically over-priced.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 5 — Sensitivity Analysis
# MAGIC
# MAGIC This section quantifies the individual contribution of each enrichment feature to model
# MAGIC performance. It answers the question: *"If we could not use feature X — due to data
# MAGIC availability, regulatory restriction, or cost — how much would model quality deteriorate?"*

# COMMAND ----------

# Load the feature impact table from the model factory
impact_df = spark.table("model_factory_feature_impact").toPandas()

print("Enrichment Feature Sensitivity — Average AIC Impact:")
print("-" * 55)
for _, row in impact_df.sort_values("avg_aic_improvement", ascending=False).iterrows():
    direction = "improves" if row["avg_aic_improvement"] > 0 else "worsens"
    print(f"  {row['feature']:35s}  AIC {direction} by {abs(row['avg_aic_improvement']):,.1f} on average")

print()
print("Interpretation: a positive value means including this feature REDUCES AIC (improves fit).")
print("Features are ranked by their average contribution across all specifications that include them.")

# COMMAND ----------

# Sensitivity analysis visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Average AIC improvement per feature
ax = axes[0]
sorted_impact = impact_df.sort_values("avg_aic_improvement", ascending=True)
bar_colors_sens = ["#43A047" if v > 0 else "#E53935" for v in sorted_impact["avg_aic_improvement"]]
ax.barh(sorted_impact["feature"], sorted_impact["avg_aic_improvement"],
        color=bar_colors_sens, edgecolor="white", height=0.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("Average AIC Improvement When Feature is Included\n(positive = feature reduces AIC = improves fit)", fontsize=10)
ax.set_title("Feature Sensitivity — Average AIC Impact", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Right: AIC comparison — with vs without each feature (box-plot style with min/max)
ax = axes[1]
enrichment_features_list = [
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk"
]
comparison_data = []
for ef in enrichment_features_list:
    with_feat = factory_df[factory_df["description"].str.contains(ef)]["aic"]
    without_feat = factory_df[~factory_df["description"].str.contains(ef)]["aic"]
    if len(with_feat) > 0 and len(without_feat) > 0:
        comparison_data.append({
            "feature": ef,
            "mean_with": with_feat.mean(),
            "mean_without": without_feat.mean(),
            "min_with": with_feat.min(),
            "max_with": with_feat.max(),
        })

comp_df = pd.DataFrame(comparison_data).sort_values("mean_with")
x_pos = range(len(comp_df))
width = 0.35

ax.bar([x - width/2 for x in x_pos], comp_df["mean_without"], width=width,
       label="Without feature", color="#E53935", alpha=0.8, edgecolor="white")
ax.bar([x + width/2 for x in x_pos], comp_df["mean_with"], width=width,
       label="With feature", color="#43A047", alpha=0.8, edgecolor="white")

ax.set_xticks(list(x_pos))
ax.set_xticklabels(
    [f.replace("_", "\n") for f in comp_df["feature"]],
    fontsize=9
)
ax.set_ylabel("Mean AIC Across Specifications", fontsize=10)
ax.set_title("AIC With vs Without Each Enrichment Feature", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Section 5 — Sensitivity Analysis: Enrichment Feature Contributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Quantify: what is the AIC cost of dropping the single most important enrichment feature?
most_important = impact_df.sort_values("avg_aic_improvement", ascending=False).iloc[0]
least_important = impact_df.sort_values("avg_aic_improvement").iloc[0]

print(f"Most impactful enrichment feature  : {most_important['feature']}")
print(f"  Average AIC improvement          : {most_important['avg_aic_improvement']:,.1f}")
print()
print(f"Least impactful enrichment feature : {least_important['feature']}")
print(f"  Average AIC improvement          : {least_important['avg_aic_improvement']:,.1f}")
print()
print("Governance implication:")
print(f"  Removing '{most_important['feature']}' would cost an average of {most_important['avg_aic_improvement']:,.1f} AIC points.")
print(f"  This represents a material reduction in model quality that would require committee sign-off.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 6 — Data Quality & Limitations
# MAGIC
# MAGIC This section documents known assumptions, limitations, and data quality considerations
# MAGIC that reviewers and regulators should be aware of.
# MAGIC
# MAGIC ## 6.1 Data Source
# MAGIC
# MAGIC | Attribute | Detail |
# MAGIC |---|---|
# MAGIC | **Data type** | Fully synthetic — generated programmatically in notebook `01_new_data_impact_demo.py` |
# MAGIC | **Sample size** | 50,000 simulated home insurance policies |
# MAGIC | **Train/test split** | 70% / 30% random split (seed = 42) |
# MAGIC | **Policy period** | Single-period; no longitudinal dimension |
# MAGIC
# MAGIC ## 6.2 Known Assumptions
# MAGIC
# MAGIC 1. **Synthetic data:** All policy records, claims, and enrichment variables are simulated.
# MAGIC    The data-generating process is specified in the notebook and is fully transparent.
# MAGIC    Results should **not** be treated as reflective of real portfolio behaviour.
# MAGIC
# MAGIC 2. **No temporal effects:** The dataset represents a single cross-sectional snapshot.
# MAGIC    There is no modelling of inflation, portfolio drift, or underwriting cycle effects.
# MAGIC
# MAGIC 3. **No spatial autocorrelation:** Enrichment variables (flood zone, crime index) are
# MAGIC    drawn independently per policy. In real data, neighbouring properties would exhibit
# MAGIC    correlated risk that may require spatial GLM or CAR model structures.
# MAGIC
# MAGIC 4. **Exposure homogeneity:** All policies are assumed to carry a single year of exposure.
# MAGIC    Real models would require exposure-weighted Poisson regression with offset terms.
# MAGIC
# MAGIC 5. **Linear enrichment effects:** The model assumes a log-linear relationship between
# MAGIC    enrichment features and claim frequency. Non-linearities (e.g., threshold effects
# MAGIC    in flood zones) are not captured by the current GLM structure.
# MAGIC
# MAGIC 6. **No interaction structure beyond the factory:** The recommended model may not include
# MAGIC    all actuarially meaningful interactions. Interactions were limited to pre-specified
# MAGIC    combinations; a full interaction search was not performed.
# MAGIC
# MAGIC 7. **Severity independence:** The frequency and severity models are fitted independently
# MAGIC    (two-part model). Correlation between frequency and severity (e.g., high-frequency
# MAGIC    policyholders may have lower-severity claims) is not modelled.
# MAGIC
# MAGIC 8. **Regulatory permissibility:** Use of enrichment variables in real pricing models
# MAGIC    requires regulatory review under applicable rules (e.g., FCA GIPP, Equality Act 2010,
# MAGIC    GDPR). This analysis does not constitute regulatory approval.
# MAGIC
# MAGIC ## 6.3 Model Limitations
# MAGIC
# MAGIC - GLM assumes independent observations; household clustering effects are not modelled.
# MAGIC - Categorical groupings (property type, construction) use arbitrary reference categories.
# MAGIC - The Poisson family assumes mean = variance; over-dispersion in real claims data may
# MAGIC   require negative binomial or Tweedie alternatives.
# MAGIC - LightGBM severity models may overfit on small claim sub-populations.
# MAGIC
# MAGIC ## 6.4 Data Quality Flags
# MAGIC
# MAGIC | Flag | Description | Impact |
# MAGIC |---|---|---|
# MAGIC | Synthetic origin | Data is simulated, not observed | Results not generalisable to real portfolios |
# MAGIC | No missing values | Simulation produces complete data | Imputation robustness untested |
# MAGIC | Uncorrelated enrichment | Enrichment features sampled independently | May understate collinearity risks in real data |
# MAGIC | No premium loading | Quotes use a fixed expense load | Real pricing requires bespoke expense and profit loads |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 7 — Exportable Governance Summary
# MAGIC
# MAGIC This section consolidates all key governance information into a single structured DataFrame
# MAGIC and a copy-pasteable text report. Both artefacts are persisted to Unity Catalog.

# COMMAND ----------

# Build the consolidated governance summary DataFrame
governance_records = []

# 1. Recommended frequency model
governance_records.append({
    "report_date": REPORT_DATE,
    "model_type": "Frequency",
    "model_framework": "Poisson GLM",
    "model_name": best_model["name"],
    "model_description": best_model["description"],
    "n_features": int(best_model["n_features"]),
    "selection_criterion": "AIC (lowest across 50-spec factory search)",
    "aic": round(float(best_model["aic"]), 2),
    "bic": round(float(best_model["bic"]), 2),
    "gini_test": round(float(best_model["gini_test"]), 4),
    "deviance_explained": round(float(best_model["deviance_explained"]), 4),
    "aic_vs_baseline": round(float(aic_improvement), 2),
    "gini_vs_baseline": round(float(gini_improvement), 4),
    "specifications_searched": n_models,
    "train_size": int(spark.table("train_set").count()),
    "test_size": int(spark.table("test_set").count()),
    "data_type": "Synthetic",
    "known_limitations": "Synthetic data; no temporal/spatial effects; single exposure period",
    "status": "DRAFT — Pending Review",
})

# 2. Standard frequency model (baseline, for reference)
governance_records.append({
    "report_date": REPORT_DATE,
    "model_type": "Frequency — Baseline",
    "model_framework": "Poisson GLM",
    "model_name": "baseline_standard",
    "model_description": "Standard rating factors only — no enrichment",
    "n_features": int(baseline_model["n_features"]),
    "selection_criterion": "Baseline reference specification",
    "aic": round(float(baseline_model["aic"]), 2),
    "bic": round(float(baseline_model["bic"]), 2),
    "gini_test": round(float(baseline_model["gini_test"]), 4),
    "deviance_explained": round(float(baseline_model["deviance_explained"]), 4),
    "aic_vs_baseline": 0.0,
    "gini_vs_baseline": 0.0,
    "specifications_searched": n_models,
    "train_size": int(spark.table("train_set").count()),
    "test_size": int(spark.table("test_set").count()),
    "data_type": "Synthetic",
    "known_limitations": "Synthetic data; no temporal/spatial effects; single exposure period",
    "status": "REFERENCE — Not for deployment",
})

# 3. Severity model summary (parse from comparison table)
try:
    sev_std_mae = float(sev_comp_df[sev_comp_df["metric"] == "MAE (test)"]["model_1_standard"].values[0])
    sev_enr_mae = float(sev_comp_df[sev_comp_df["metric"] == "MAE (test)"]["model_2_enriched"].values[0])
    sev_std_r2 = float(sev_comp_df[sev_comp_df["metric"] == "R2 (test)"]["model_1_standard"].values[0])
    sev_enr_r2 = float(sev_comp_df[sev_comp_df["metric"] == "R2 (test)"]["model_2_enriched"].values[0])
except Exception:
    sev_std_mae = sev_enr_mae = sev_std_r2 = sev_enr_r2 = None

for sev_label, mae_val, r2_val in [
    ("Severity — Enriched (Recommended)", sev_enr_mae, sev_enr_r2),
    ("Severity — Standard (Baseline)", sev_std_mae, sev_std_r2),
]:
    governance_records.append({
        "report_date": REPORT_DATE,
        "model_type": sev_label,
        "model_framework": "LightGBM (Gamma-like)",
        "model_name": "lgbm_severity_enriched" if "Enriched" in sev_label else "lgbm_severity_standard",
        "model_description": "LightGBM gradient boosting model for claim severity",
        "n_features": None,
        "selection_criterion": "MAE on test set (lower is better)",
        "aic": None,
        "bic": None,
        "gini_test": None,
        "deviance_explained": r2_val,
        "aic_vs_baseline": None,
        "gini_vs_baseline": None,
        "specifications_searched": 2,
        "train_size": int(spark.table("train_set").count()),
        "test_size": int(spark.table("test_set").count()),
        "data_type": "Synthetic",
        "known_limitations": "Synthetic data; severity modelled on claimants only; exposure not adjusted",
        "status": "DRAFT — Pending Review",
    })

governance_summary_df = pd.DataFrame(governance_records)

# Save to Unity Catalog
spark.createDataFrame(governance_summary_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_governance_summary"
)

print(f"Governance summary saved to {CATALOG}.{SCHEMA}.model_governance_summary")
display(spark.table(f"{CATALOG}.{SCHEMA}.model_governance_summary"))

# COMMAND ----------

# Generate text-based governance report for copy-paste into documents
report_lines = [
    "=" * 70,
    "MODEL GOVERNANCE REPORT",
    "Home Insurance Pricing — Frequency & Severity Models",
    "=" * 70,
    f"Generated   : {REPORT_DATE}",
    f"Catalog     : {CATALOG}.{SCHEMA}",
    f"Status      : DRAFT — Pending Actuarial Sign-Off",
    "",
    "SECTION 1 — MODEL INVENTORY",
    "-" * 40,
    f"Total specifications searched       : {n_models}",
    f"Feature count range                 : {n_features_min} – {n_features_max}",
    f"AIC range across all specifications : {aic_range_low:,.1f} – {aic_range_high:,.1f}",
    f"Gini range                          : {gini_range_low:.4f} – {gini_range_high:.4f}",
    "",
    "SECTION 2 — RECOMMENDED MODEL",
    "-" * 40,
    f"Model name        : {best_model['name']}",
    f"Description       : {best_model['description']}",
    f"Number of features: {int(best_model['n_features'])}",
    f"AIC               : {best_model['aic']:,.1f}  (baseline: {baseline_model['aic']:,.1f})",
    f"AIC improvement   : {aic_improvement:,.1f}  ({aic_improvement/abs(baseline_model['aic'])*100:.2f}% reduction)",
    f"Gini (test)       : {best_model['gini_test']:.4f}  (baseline: {baseline_model['gini_test']:.4f})",
    f"Deviance explained: {best_model['deviance_explained']:.4f}  (baseline: {baseline_model['deviance_explained']:.4f})",
    f"AIC rank          : #{int(best_model['rank_aic'])} of {n_models}",
    f"Gini rank         : #{best_gini_rank} of {n_models}",
    "",
    "SECTION 3 — FEATURE JUSTIFICATION",
    "-" * 40,
]

# Add each feature
for _, row in coef_enriched[coef_enriched["feature"] != "const"].sort_values("coef", ascending=False).iterrows():
    sig = row["significance"]
    report_lines.append(
        f"  {row['feature']:35s}  coef={row['coef']:+.4f}  p={row['p_value']:.4f} {sig}"
    )

report_lines += [
    "",
    "SECTION 4 — PERFORMANCE EVIDENCE",
    "-" * 40,
    "Frequency model loss ratio stability:",
]
for model_name in lr_freq_df["model"].unique():
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    report_lines.append(
        f"  {model_name:12s}  Mean LR={subset['loss_ratio'].mean():.3f}  Std={subset['loss_ratio'].std():.3f}"
    )

report_lines += [
    "",
    "Severity model loss ratio stability:",
]
for model_name in lr_sev_df["model"].unique():
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    report_lines.append(
        f"  {model_name:12s}  Mean LR={subset['loss_ratio'].mean():.3f}  Std={subset['loss_ratio'].std():.3f}"
    )

report_lines += [
    "",
    "SECTION 5 — SENSITIVITY ANALYSIS",
    "-" * 40,
    "Enrichment feature impact (average AIC improvement when included):",
]
for _, row in impact_df.sort_values("avg_aic_improvement", ascending=False).iterrows():
    report_lines.append(f"  {row['feature']:35s}  {row['avg_aic_improvement']:+,.1f}")

report_lines += [
    "",
    "SECTION 6 — DATA QUALITY & LIMITATIONS",
    "-" * 40,
    "  - Data is fully synthetic (not real portfolio data)",
    "  - 50,000 simulated home insurance policies",
    "  - No temporal, spatial, or exposure-weighted effects modelled",
    "  - Enrichment variables sampled independently (no spatial autocorrelation)",
    "  - Regulatory permissibility of enrichment features not assessed",
    "",
    "SECTION 7 — ARTEFACTS PERSISTED TO UNITY CATALOG",
    "-" * 40,
    f"  {CATALOG}.{SCHEMA}.model_governance_summary",
    f"  {CATALOG}.{SCHEMA}.model_factory_results",
    f"  {CATALOG}.{SCHEMA}.model_factory_feature_impact",
    f"  {CATALOG}.{SCHEMA}.glm_coefficients",
    f"  {CATALOG}.{SCHEMA}.model_comparison",
    f"  {CATALOG}.{SCHEMA}.loss_ratio_by_decile",
    f"  {CATALOG}.{SCHEMA}.severity_model_comparison",
    f"  {CATALOG}.{SCHEMA}.severity_loss_ratio_by_decile",
    "",
    "=" * 70,
    "END OF REPORT",
    "=" * 70,
]

report_text = "\n".join(report_lines)
print(report_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Section 8 — Sign-Off
# MAGIC
# MAGIC ## Model Review Sign-Off Form
# MAGIC
# MAGIC This section must be completed before the model is approved for production use.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC | Role | Name | Date | Signature |
# MAGIC |---|---|---|---|
# MAGIC | **Model Owner / Pricing Actuary** | _______________ | ___ / ___ / _____ | _______________ |
# MAGIC | **Peer Reviewer (Actuary)** | _______________ | ___ / ___ / _____ | _______________ |
# MAGIC | **Model Validation (Independent)** | _______________ | ___ / ___ / _____ | _______________ |
# MAGIC | **Chief Actuary** | _______________ | ___ / ___ / _____ | _______________ |
# MAGIC | **Pricing Committee Chair** | _______________ | ___ / ___ / _____ | _______________ |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Review Checklist
# MAGIC
# MAGIC Please confirm each item has been reviewed and is satisfactory:
# MAGIC
# MAGIC - [ ] Model specification is documented and reproducible from source notebooks
# MAGIC - [ ] All features have documented business rationale
# MAGIC - [ ] Feature statistical significance has been reviewed
# MAGIC - [ ] Model selection criterion (AIC) is appropriate and justified
# MAGIC - [ ] Sensitivity analysis has been reviewed; no single-feature dependency identified
# MAGIC - [ ] Loss ratio stability across deciles is acceptable
# MAGIC - [ ] Data quality limitations are understood and accepted
# MAGIC - [ ] Enrichment features have been reviewed for regulatory permissibility
# MAGIC - [ ] Model outputs have been back-tested against available experience
# MAGIC - [ ] Unity Catalog artefacts are correctly versioned and accessible
# MAGIC - [ ] Governance summary table (`model_governance_summary`) reviewed and accurate
# MAGIC - [ ] This report has been archived to the model documentation repository
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Approval Status
# MAGIC
# MAGIC > **Current Status:** `DRAFT — Pending Review`
# MAGIC >
# MAGIC > Change to `APPROVED` or `REJECTED` upon completion of review.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Version History
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | 0.1 | Auto-generated | Notebook | Initial draft governance report |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *This report was generated programmatically from notebook `08_model_governance_report.py`
# MAGIC in the `pricing_new_data_impact` project. All source data and model artefacts are
# MAGIC persisted in Unity Catalog under `lr_serverless_aws_us_catalog.pricing_new_data_impact`.*
