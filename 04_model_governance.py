# Databricks notebook source
# MAGIC %md
# MAGIC # Model Governance Report & PDF Export
# MAGIC ## Home Insurance Pricing - Audit Trail, Evidence & Sign-Off
# MAGIC
# MAGIC **Document Purpose:** This notebook consolidates model governance documentation
# MAGIC and PDF report generation into a single auditable artefact.
# MAGIC
# MAGIC | Field | Value |
# MAGIC |---|---|
# MAGIC | **Report Date** | Auto-generated at runtime |
# MAGIC | **Catalog** | `lr_serverless_aws_us_catalog.pricing_new_data_impact` |
# MAGIC | **Models in Scope** | GLM Frequency (Standard & Enriched), LightGBM Severity (Standard & Enriched) |
# MAGIC | **Model Factory** | 50 Poisson GLM specifications |
# MAGIC | **Intended Audience** | Chief Actuary, Pricing Committee, Internal Audit, Regulatory Submissions |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Structure:**
# MAGIC - **Part 1:** Model inventory, selection rationale, feature justification, performance evidence, sensitivity analysis
# MAGIC - **Part 2:** PDF report export (saved to Unity Catalog volume)
# MAGIC - **Part 3:** Sign-off template

# COMMAND ----------

# MAGIC %pip install matplotlib mlflow fpdf2

# COMMAND ----------


dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1 — Governance Report

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"USE {CATALOG}.{SCHEMA}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow.deployments
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
print(f"Governance report generated: {REPORT_DATE}")
print(f"Catalog: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Model Inventory
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
    elif name == "standard_plus_region":
        return "Standard + region dummies only"
    elif name.startswith("standard_plus_") and name.count("_") <= 3:
        return "Standard + 1 enrichment feature"
    elif name.startswith("standard_plus_"):
        return "Standard + 2 enrichment features"
    elif name.startswith("enrich_3"):
        return "Standard + 3 enrichment features"
    elif name.startswith("enrich_4"):
        return "Standard + 4 enrichment features"
    elif name == "full_enrichment_no_region":
        return "Standard + all enrichment features (no region)"
    elif name == "full_enrichment_with_region":
        return "Standard + all enrichment features (with region)"
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
# MAGIC ## 3. Recommended Model & Selection Rationale
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
colors_bar = [
    "#43A047" if row["name"] == best_model["name"]
    else "#E53935" if row["name"] == "baseline_standard"
    else "#90CAF9"
    for _, row in factory_df.sort_values("rank_aic").iterrows()
]

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
ax.plot([1, n_models], [1, n_models], "k--", alpha=0.2, linewidth=1)
ax.set_xlim(0, n_models + 1)
ax.set_ylim(0, n_models + 1)

plt.suptitle("Section 3 — Model Selection Evidence", fontsize=13, fontweight="bold")
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
# MAGIC ## 4. Feature Justification
# MAGIC
# MAGIC For each feature in the recommended model, we document the statistical evidence (coefficient,
# MAGIC p-value) alongside a plain-English business rationale. This section is intended to support
# MAGIC regulatory review and satisfy requirements around model explainability.

# COMMAND ----------

# Business rationale for each feature
FEATURE_RATIONALE = {
    "const": "Intercept - captures the baseline log-frequency for the reference risk profile.",
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
        "Number of claims in prior policy years. Strong predictor of future claim behaviour - "
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
    "imd_decile": (
        "Index of Multiple Deprivation (England, 2019), LSOA-level overall decile. "
        "Decile 1 = most deprived, 10 = least deprived. Lower deciles are associated with "
        "higher claim frequency (theft, escape of water, accidental damage) via correlates of "
        "housing quality and security. A negative coefficient (lower decile -> higher log-frequency) is expected."
    ),
    "imd_score": (
        "Underlying IMD 2019 composite score (continuous). Captures the same deprivation signal "
        "as imd_decile at finer granularity. Typically strongly correlated with imd_decile; "
        "expected direction: higher score (more deprived) -> higher claim frequency."
    ),
    "crime_decile": (
        "Crime domain of the IMD 2019 at LSOA level. Decile 1 = most crime-deprived neighbourhoods. "
        "Strong predictor of theft and malicious damage claim frequency and, to a lesser extent, "
        "severity. Negative coefficient expected."
    ),
    "income_decile": (
        "Income deprivation decile (IMD 2019). Correlates with many downstream risk factors "
        "including housing quality, occupancy patterns, and prior-claim behaviour. "
        "Negative coefficient expected."
    ),
    "health_decile": (
        "Health deprivation and disability decile (IMD 2019). Acts as a weak positive correlate of "
        "escape-of-water and accidental-damage claims through occupancy and care patterns. "
        "Coefficient is typically modest and may be partially absorbed by imd_decile."
    ),
    "living_env_decile": (
        "Living environment deprivation decile (IMD 2019) — captures housing condition, "
        "air quality, and road safety. Strong predictor of maintenance-related claims "
        "(escape of water, subsidence precursors). Negative coefficient expected."
    ),
    "is_urban": (
        "Urban/rural indicator derived from the ONS 2011 Rural-Urban Classification "
        "(urban = A1/B1/C1/C2 bands at LSOA level). Urban properties exhibit higher claim frequency "
        "for theft and escape of water. Positive coefficient expected."
    ),
    "is_coastal": (
        "Coastal flag derived from coastal English local authorities. Coastal properties experience "
        "elevated weather-related and water-ingress claims (storm, salt corrosion). "
        "Positive coefficient expected for both frequency and severity."
    ),
    "region_east_of_england": (
        "ONS Region dummy. Regional geographic effect relative to the baseline region "
        "(typically East Midlands). Captures residual geographic heterogeneity after IMD "
        "and urban/coastal controls."
    ),
    "region_london": (
        "ONS Region dummy (London). London typically shows elevated claim frequency "
        "(theft, escape of water) relative to the baseline region."
    ),
    "region_north_east": (
        "ONS Region dummy (North East). Captures residual regional effect after deprivation "
        "and urban/coastal controls."
    ),
    "region_north_west": (
        "ONS Region dummy (North West). Typically shows elevated frequency relative to the baseline, "
        "driven by weather exposure and housing stock composition."
    ),
    "region_south_east": (
        "ONS Region dummy (South East). Regional effect beyond deprivation and coastal controls."
    ),
    "region_south_west": (
        "ONS Region dummy (South West). Weather-exposed coastal geography tends to lift frequency "
        "relative to the baseline region."
    ),
    "": (
        "ONS Region dummy (Wales). Captures residual regional effect; note Welsh IMD is not "
        "directly comparable to English IMD 2019, so the dummy carries additional signal."
    ),
    "region_west_midlands": (
        "ONS Region dummy (West Midlands). Regional geographic effect relative to the baseline region."
    ),
    "region_yorkshire": (
        "ONS Region dummy (Yorkshire and The Humber). Regional geographic effect relative to the baseline region."
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
print("Significance codes: *** p<0.001 | ** p<0.01 | * p<0.05 | ns not significant")
print()
display(spark.createDataFrame(feature_table))

# COMMAND ----------

# Coefficient forest plot
fig, ax = plt.subplots(figsize=(12, 9))

plot_df = coef_enriched[coef_enriched["feature"] != "const"].sort_values("coef")
y_pos = range(len(plot_df))

colors_coef = ["#E53935" if c > 0 else "#1E88E5" for c in plot_df["coef"]]

ax.barh(list(y_pos), plot_df["coef"], xerr=1.96 * plot_df["std_err"],
        color=colors_coef, edgecolor="white", height=0.6, capsize=3, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(plot_df["feature"], fontsize=10)
ax.set_xlabel("Coefficient (log scale — Poisson GLM)", fontsize=11)
ax.set_title(
    "Section 4 — Feature Coefficients with 95% Confidence Intervals\n"
    "Enriched GLM Frequency Model",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="x", alpha=0.3)

enrichment_features = [
    "imd_decile", "imd_score", "crime_decile", "income_decile",
    "health_decile", "living_env_decile", "is_urban", "is_coastal",
]
for i, feat in enumerate(plot_df["feature"]):
    if feat in enrichment_features or (feat.startswith("region_") and feat != "region_code"):
        ax.get_yticklabels()[i].set_color("#6A1B9A")
        ax.get_yticklabels()[i].set_fontweight("bold")

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
# MAGIC ## 5. Performance Evidence
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

# Loss ratio stability analysis
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

# Top left: Frequency LR by decile
ax1 = fig.add_subplot(gs[0, 0])
for model_name, color in [("Standard", "#E53935"), ("Enriched", "#43A047")]:
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    if not subset.empty:
        ax1.plot(subset["decile"], subset["loss_ratio"], marker="o", label=model_name,
                 color=color, linewidth=2)
ax1.axhline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=1)
ax1.set_xlabel("Premium Decile")
ax1.set_ylabel("Loss Ratio")
ax1.set_title("Frequency Model — Loss Ratio by Decile", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Top right: Severity LR by decile
ax2 = fig.add_subplot(gs[0, 1])
for model_name, color in [("Standard", "#E53935"), ("Enriched", "#43A047")]:
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    if not subset.empty:
        ax2.plot(subset["decile"], subset["loss_ratio"], marker="o", label=model_name,
                 color=color, linewidth=2)
ax2.axhline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=1)
ax2.set_xlabel("Premium Decile")
ax2.set_ylabel("Loss Ratio")
ax2.set_title("Severity Model — Loss Ratio by Decile", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Bottom left: LR std dev comparison
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
from matplotlib.patches import Patch as _Patch
legend_els = [_Patch(facecolor="#E53935", label="Standard"), _Patch(facecolor="#43A047", label="Enriched")]
ax3.legend(handles=legend_els, fontsize=9)

# Bottom right: AIC/Gini improvement summary
ax4 = fig.add_subplot(gs[1, 1])
metrics_labels = ["AIC\nimprovement", "Gini\nimprovement", "Deviance\nimprovement"]
try:
    aic_std = float(freq_comp_df[freq_comp_df["metric"] == "AIC"]["model_1_standard"].values[0])
    aic_enr = float(freq_comp_df[freq_comp_df["metric"] == "AIC"]["model_2_enriched"].values[0])
    gini_std = float(freq_comp_df[freq_comp_df["metric"] == "Gini (test)"]["model_1_standard"].values[0])
    gini_enr = float(freq_comp_df[freq_comp_df["metric"] == "Gini (test)"]["model_2_enriched"].values[0])
    dev_std = float(freq_comp_df[freq_comp_df["metric"] == "Deviance Explained"]["model_1_standard"].values[0])
    dev_enr = float(freq_comp_df[freq_comp_df["metric"] == "Deviance Explained"]["model_2_enriched"].values[0])
    improvement_vals = [
        (aic_std - aic_enr) / abs(aic_std) * 100,
        (gini_enr - gini_std) / gini_std * 100,
        (dev_enr - dev_std) / dev_std * 100,
    ]
    bar_colors_imp = ["#43A047" if v > 0 else "#E53935" for v in improvement_vals]
    ax4.bar(metrics_labels, improvement_vals, color=bar_colors_imp, edgecolor="white", width=0.5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.set_ylabel("% Improvement (Enriched vs Standard)")
    ax4.set_title("Frequency Model — Enrichment Uplift", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)
except Exception as e:
    ax4.text(0.5, 0.5, f"Metric parsing error:\n{e}", ha="center", va="center", transform=ax4.transAxes)

plt.suptitle("Section 5 — Performance Evidence Summary", fontsize=14, fontweight="bold")
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
# MAGIC ## 6. Sensitivity Analysis
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

# COMMAND ----------

# Sensitivity analysis visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
sorted_impact = impact_df.sort_values("avg_aic_improvement", ascending=True)
bar_colors_sens = ["#43A047" if v > 0 else "#E53935" for v in sorted_impact["avg_aic_improvement"]]
ax.barh(sorted_impact["feature"], sorted_impact["avg_aic_improvement"],
        color=bar_colors_sens, edgecolor="white", height=0.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("Average AIC Improvement When Feature is Included\n(positive = feature reduces AIC = improves fit)", fontsize=10)
ax.set_title("Feature Sensitivity — Average AIC Impact", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
enrichment_features_list = [
    "imd_decile", "crime_decile", "income_decile", "health_decile",
    "living_env_decile", "is_urban", "is_coastal",
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

plt.suptitle("Section 6 — Sensitivity Analysis: Enrichment Feature Contributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Quantify: what is the AIC cost of dropping the most important enrichment feature?
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
# MAGIC ## 7. Data Quality & Limitations
# MAGIC
# MAGIC ## 7.1 Data Source
# MAGIC
# MAGIC | Attribute | Detail |
# MAGIC |---|---|
# MAGIC | **Policy & claim records** | Synthetic — generated programmatically in notebook `01_build_all_models.py` |
# MAGIC | **Enrichment features** | **Real UK public data** — ONSPD postcode directory, MHCLG IMD 2019 (LSOA deciles), ONS 2011 Rural-Urban Classification, coastal local-authority derivation |
# MAGIC | **Sample size** | 50,000 simulated home insurance policies, each keyed to a real UK postcode |
# MAGIC | **Train/test split** | 70% / 30% random split (seed = 42) |
# MAGIC | **Policy period** | Single-period; no longitudinal dimension |
# MAGIC
# MAGIC ## 7.2 Known Assumptions
# MAGIC
# MAGIC 1. **Synthetic policies and claims:** Policyholders, properties, and claim outcomes are simulated.
# MAGIC    The enrichment features attached to each policy are real (IMD 2019, ONSPD, ONS RUC), but the
# MAGIC    claim-generating process is modelled and not reflective of a real portfolio.
# MAGIC
# MAGIC 2. **No temporal effects:** The dataset represents a single cross-sectional snapshot. IMD 2019
# MAGIC    deprivation inputs are also a single-vintage snapshot; they do not track deprivation change over time.
# MAGIC
# MAGIC 3. **Spatial smoothing at LSOA level:** IMD deciles are constant within an LSOA, so within-LSOA
# MAGIC    heterogeneity in risk is not captured by these features.
# MAGIC
# MAGIC 4. **Exposure homogeneity:** All policies are assumed to carry a single year of exposure.
# MAGIC
# MAGIC 5. **Linear enrichment effects:** Non-linearities (e.g., threshold effects in deprivation deciles)
# MAGIC    are not captured by the current GLM structure; deciles are treated as ordinal-continuous.
# MAGIC
# MAGIC 6. **No interaction structure beyond the factory:** Interactions were limited to pre-specified
# MAGIC    combinations; a full interaction search was not performed.
# MAGIC
# MAGIC 7. **Severity independence:** The frequency and severity models are fitted independently.
# MAGIC
# MAGIC 8. **Regulatory permissibility:** Use of area-level deprivation and postcode-derived features in
# MAGIC    real pricing models requires regulatory review under applicable rules (e.g., FCA GIPP,
# MAGIC    Equality Act 2010, GDPR). Deprivation indices can act as proxies for protected characteristics
# MAGIC    and must be documented and justified accordingly.
# MAGIC
# MAGIC ## 7.3 Data Quality Flags
# MAGIC
# MAGIC | Flag | Description | Impact |
# MAGIC |---|---|---|
# MAGIC | Synthetic claims | Claim frequency and severity are simulated | Results not generalisable to real portfolio behaviour |
# MAGIC | Real public enrichment | IMD 2019, ONSPD, and RUC are real UK public datasets | Collinearity between deprivation deciles present and must be monitored |
# MAGIC | LSOA granularity | Enrichment deciles are constant within each LSOA | Within-LSOA risk variation not captured |
# MAGIC | England-only IMD | IMD 2019 coverage is English LSOAs | Welsh / Scottish / NI postcodes rely on region effects; not directly comparable |
# MAGIC | No premium loading | Quotes use a fixed expense load | Real pricing requires bespoke expense and profit loads |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 8. Exportable Summary — Save to Unity Catalog

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
    "data_type": "Synthetic policies & claims; real UK public enrichment (IMD 2019, ONSPD, ONS RUC)",
    "known_limitations": "Synthetic claim process; LSOA-level enrichment granularity; single exposure period; no temporal dynamics",
    "status": "DRAFT - Pending Review",
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
    "data_type": "Synthetic policies & claims; real UK public enrichment (IMD 2019, ONSPD, ONS RUC)",
    "known_limitations": "Synthetic claim process; LSOA-level enrichment granularity; single exposure period; no temporal dynamics",
    "status": "REFERENCE — Not for deployment",
})

# 3. Severity model summaries
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
        "data_type": "Synthetic policies & claims; real UK public enrichment (IMD 2019, ONSPD, ONS RUC)",
        "known_limitations": "Synthetic claim process; severity modelled on claimants only; exposure not adjusted",
        "status": "DRAFT - Pending Review",
    })

governance_summary_df = pd.DataFrame(governance_records)

# Save to Unity Catalog
spark.createDataFrame(governance_summary_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_governance_summary"
)

print(f"Governance summary saved to {CATALOG}.{SCHEMA}.model_governance_summary")
display(spark.table(f"{CATALOG}.{SCHEMA}.model_governance_summary"))

# COMMAND ----------

# Generate text-based governance report for copy-paste
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

report_lines += ["", "Severity model loss ratio stability:"]
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
    "  - Policies and claims are synthetic (not real portfolio experience)",
    "  - Enrichment features are REAL UK public data: IMD 2019 (MHCLG), ONSPD (ONS), ONS RUC 2011",
    "  - 50,000 simulated home insurance policies keyed to real UK postcodes",
    "  - No temporal or exposure-weighted effects modelled",
    "  - Enrichment deciles constant within LSOA (spatial granularity limit)",
    "  - Regulatory permissibility of deprivation-based features requires review (FCA GIPP, Equality Act, GDPR)",
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
# MAGIC # Part 2 — PDF Export

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Generate PDF Report
# MAGIC
# MAGIC This section uses **fpdf2** to produce a professional, self-contained PDF document suitable
# MAGIC for archiving, email distribution, or regulatory submission. The PDF mirrors the structure
# MAGIC of Part 1 and includes:
# MAGIC
# MAGIC - Title page with report metadata
# MAGIC - Model inventory summary
# MAGIC - Recommended model details and improvement over baseline
# MAGIC - Feature justification table (coefficient, p-value, rationale)
# MAGIC - Performance metrics comparison table (frequency + severity)
# MAGIC - Sensitivity analysis table
# MAGIC - Limitations section
# MAGIC - Sign-off page with blank fields

# COMMAND ----------

from fpdf import FPDF

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/reports"
PDF_PATH = f"{VOLUME_PATH}/model_governance_report_{REPORT_DATE}.pdf"

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
BLUE   = (13, 71, 161)   # dark blue — headings
GREY   = (97, 97, 97)    # mid grey — secondary text
LGREEN = (232, 245, 233) # light green — highlight rows
WHITE  = (255, 255, 255)
BLACK  = (33, 33, 33)
HEADER_BLUE = (21, 101, 192)

def _safe(val, fmt=None):
    """Convert a value to a safely printable string, handling None/NaN."""
    if val is None:
        return "N/A"
    try:
        import math
        if math.isnan(float(val)):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if fmt:
        try:
            return fmt.format(val)
        except Exception:
            pass
    return _ascii(str(val))


def _ascii(text):
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    return (str(text)
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2192", "->")  # right arrow
            .replace("\u2190", "<-")  # left arrow
            .replace("\u00d7", "x")   # multiplication sign
            .replace("\u2265", ">=")  # ≥
            .replace("\u2264", "<=")  # ≤
            .replace("\u2248", "~")   # ≈
            )


class GovernancePDF(FPDF):
    """Custom FPDF subclass with consistent header/footer and helper methods."""

    def header(self):
        if self.page_no() == 1:
            return  # No running header on title page
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Model Governance Report - Home Insurance Pricing  |  {REPORT_DATE}  |  DRAFT",
                  new_x="LMARGIN", new_y="NEXT", align="R")
        self.set_draw_color(*GREY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)
        self.set_text_color(*BLACK)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}  |  {CATALOG}.{SCHEMA}  |  CONFIDENTIAL - DRAFT", align="C")
        self.set_text_color(*BLACK)

    # --- Section heading ---
    def section_heading(self, text, level=1):
        self.ln(4)
        if level == 1:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*BLUE)
            self.set_fill_color(227, 242, 253)
            self.cell(0, 10, _ascii(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*HEADER_BLUE)
            self.cell(0, 8, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*BLACK)
        self.ln(1)

    # --- Key-value row ---
    def kv_row(self, key, value, indent=6):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GREY)
        self.set_x(self.l_margin + indent)
        self.cell(55, 6, _ascii(key) + ":")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*BLACK)
        self.cell(0, 6, _ascii(str(value)), new_x="LMARGIN", new_y="NEXT")

    # --- Table with headers ---
    def table(self, headers, rows, col_widths, header_fill=HEADER_BLUE, stripe=True):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        self.set_fill_color(*header_fill)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, _ascii(h), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*BLACK)
        for i, row in enumerate(rows):
            if stripe and i % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(*WHITE)
            for cell_val, w in zip(row, col_widths):
                self.cell(w, 6, _safe(cell_val), border=1, fill=True)
            self.ln()
        self.set_fill_color(*WHITE)

    # --- Bullet point ---
    def bullet(self, text, indent=8):
        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin + indent)
        remaining = self.w - self.l_margin - indent - 5 - self.r_margin
        self.cell(5, 6, "-")
        self.multi_cell(remaining, 6, _ascii(text))


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------
pdf = GovernancePDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(12, 12, 12)

# ===========================================================================
# PAGE 1 — Title page
# ===========================================================================
pdf.add_page()

pdf.ln(20)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(*BLUE)
pdf.cell(0, 14, "Model Governance Report", new_x="LMARGIN", new_y="NEXT", align="C")

pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(*HEADER_BLUE)
pdf.cell(0, 10, "Home Insurance Pricing", new_x="LMARGIN", new_y="NEXT", align="C")

pdf.ln(3)
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(*GREY)
pdf.cell(0, 8, "Enriched GLM Frequency & Severity Model Review", new_x="LMARGIN", new_y="NEXT", align="C")

pdf.ln(8)
pdf.set_draw_color(*BLUE)
pdf.set_line_width(0.8)
pdf.line(30, pdf.get_y(), 180, pdf.get_y())
pdf.ln(10)

pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(*BLACK)

meta = [
    ("Report Date",   REPORT_DATE),
    ("Status",        "DRAFT - Pending Actuarial Sign-Off"),
    ("Catalog",       CATALOG),
    ("Schema",        SCHEMA),
    ("Models in Scope", "GLM Frequency (Standard & Enriched), LightGBM Severity (Standard & Enriched)"),
    ("Model Factory", f"{n_models} Poisson GLM specifications"),
    ("Intended Audience", "Chief Actuary, Pricing Committee, Internal Audit, Regulatory Submissions"),
]
for key, val in meta:
    pdf.kv_row(key, val, indent=30)

pdf.ln(12)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*GREY)
pdf.multi_cell(0, 6,
    "This document was automatically generated from the model factory pipeline. "
    "All metrics are derived from Unity Catalog tables at runtime. "
    "This report must be reviewed and signed off by the relevant parties before use in production or regulatory submission.",
    align="C"
)

# ===========================================================================
# PAGE 2 — Model Inventory
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 1 — Model Inventory")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "The model factory conducted a systematic search over 50 Poisson GLM specifications, "
    "varying the set of enrichment features included alongside standard rating factors. "
    "All specifications are recorded in Unity Catalog for full auditability."
)
pdf.ln(3)

pdf.section_heading("1.1  Search Space Summary", level=2)
pdf.kv_row("Total specifications trained", str(n_models))
pdf.kv_row("Feature count range", f"{n_features_min} – {n_features_max}")
pdf.kv_row("AIC range", f"{aic_range_low:,.1f} – {aic_range_high:,.1f}")
pdf.kv_row("Gini range (test set)", f"{gini_range_low:.4f} – {gini_range_high:.4f}")
pdf.ln(4)

pdf.section_heading("1.2  Top 10 Specifications by AIC", level=2)
top10_rows = []
for _, r in factory_df.sort_values("rank_aic").head(10).iterrows():
    top10_rows.append([
        int(r["rank_aic"]),
        r["name"][:28],
        f"{r['aic']:,.1f}",
        f"{r['bic']:,.1f}",
        f"{r['gini_test']:.4f}",
        int(r["n_features"]),
    ])

pdf.table(
    headers=["Rank", "Model Name", "AIC", "BIC", "Gini (test)", "# Feat"],
    rows=top10_rows,
    col_widths=[13, 68, 22, 22, 22, 16],
)

# ===========================================================================
# PAGE 3 — Recommended Model
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 2 — Recommended Model & Selection Rationale")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "The recommended model is the specification with the lowest AIC across the full factory search. "
    "AIC (Akaike Information Criterion) balances goodness-of-fit against model complexity, penalising "
    "additional parameters. This makes it appropriate for comparing both nested and non-nested GLM specifications."
)
pdf.ln(3)

pdf.section_heading("2.1  Recommended Model Details", level=2)
pdf.kv_row("Model Name",       best_model["name"])
pdf.kv_row("Description",      best_model["description"])
pdf.kv_row("Number of Features", str(int(best_model["n_features"])))
pdf.kv_row("AIC Rank",         f"#{int(best_model['rank_aic'])} of {n_models}")
pdf.kv_row("BIC Rank",         f"#{best_bic_rank} of {n_models}")
pdf.kv_row("Gini Rank",        f"#{best_gini_rank} of {n_models}")
pdf.ln(4)

pdf.section_heading("2.2  Performance vs Baseline", level=2)
perf_rows = [
    ["AIC",               f"{best_model['aic']:,.1f}",              f"{baseline_model['aic']:,.1f}",              f"{aic_improvement:+,.1f}"],
    ["BIC",               f"{best_model['bic']:,.1f}",              f"{baseline_model['bic']:,.1f}",              "—"],
    ["Gini (test)",       f"{best_model['gini_test']:.4f}",         f"{baseline_model['gini_test']:.4f}",         f"{gini_improvement:+.4f}"],
    ["Deviance Explained",f"{best_model['deviance_explained']:.4f}",f"{baseline_model['deviance_explained']:.4f}",f"{deviance_improvement:+.4f}"],
    ["# Features",        str(int(best_model['n_features'])),       str(int(baseline_model['n_features'])),       f"+{int(best_model['n_features']) - int(baseline_model['n_features'])}"],
]
pdf.table(
    headers=["Metric", "Recommended", "Baseline", "Improvement"],
    rows=perf_rows,
    col_widths=[48, 42, 42, 36],
)
pdf.ln(4)

pdf.section_heading("2.3  Selection Criteria", level=2)
criteria = [
    "AIC (primary criterion): Penalises complexity; lowest AIC across the full 50-spec search.",
    "BIC consistency: Heavier complexity penalty confirms the model is not over-fitted.",
    "Gini coefficient: Measures risk ranking discrimination on held-out test data.",
    "Parsimony: Where models achieve similar AIC, the simpler model is preferred for regulatory robustness.",
]
for c in criteria:
    pdf.bullet(c)

# ===========================================================================
# PAGE 4 — Feature Justification
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 3 — Feature Justification")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "For each feature in the recommended enriched model, statistical evidence (coefficient, p-value) "
    "is presented alongside plain-English business rationale to support regulatory review and model explainability requirements. "
    "Enrichment features are marked with (*). Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant."
)
pdf.ln(3)

feat_rows = []
for _, row in coef_enriched.sort_values("coef", ascending=False).iterrows():
    feat_name = row["feature"]
    is_enrich = (
        feat_name in [
            "imd_decile", "imd_score", "crime_decile", "income_decile",
            "health_decile", "living_env_decile", "is_urban", "is_coastal",
        ]
        or (feat_name.startswith("region_") and feat_name != "region_code")
    )
    label = f"(*) {feat_name}" if is_enrich else feat_name
    # Truncate rationale for table
    rationale = FEATURE_RATIONALE.get(feat_name, "")
    rationale_short = rationale[:70] + "..." if len(rationale) > 70 else rationale
    feat_rows.append([
        label[:32],
        f"{row['coef']:+.4f}",
        f"{row['p_value']:.4f}",
        row["significance"],
        rationale_short,
    ])

pdf.table(
    headers=["Feature", "Coefficient", "P-Value", "Sig.", "Business Rationale (summary)"],
    rows=feat_rows,
    col_widths=[40, 22, 20, 16, 88],
)

pdf.ln(3)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(*GREY)
pdf.cell(0, 5, "(*) denotes enrichment feature.  Full rationale text is available in the interactive notebook.")
pdf.set_text_color(*BLACK)

# ===========================================================================
# PAGE 5 — Performance Evidence
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 4 — Performance Evidence")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "Performance is assessed on held-out test data (30% of 50,000 policies). "
    "The enriched model is compared against the standard baseline across AIC, Gini, deviance explained, "
    "and loss ratio stability by premium decile."
)
pdf.ln(3)

pdf.section_heading("4.1  Frequency Model Comparison (Poisson GLM)", level=2)
try:
    freq_table_rows = []
    for _, r in freq_comp_df.iterrows():
        freq_table_rows.append([
            str(r.get("metric", "")),
            _safe(r.get("model_1_standard")),
            _safe(r.get("model_2_enriched")),
        ])
    pdf.table(
        headers=["Metric", "Standard", "Enriched"],
        rows=freq_table_rows,
        col_widths=[80, 48, 40],
    )
except Exception as e:
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"[Frequency comparison table unavailable: {e}]", new_x="LMARGIN", new_y="NEXT")

pdf.ln(4)

pdf.section_heading("4.2  Severity Model Comparison (LightGBM)", level=2)
try:
    sev_table_rows = []
    for _, r in sev_comp_df.iterrows():
        sev_table_rows.append([
            str(r.get("metric", "")),
            _safe(r.get("model_1_standard")),
            _safe(r.get("model_2_enriched")),
        ])
    pdf.table(
        headers=["Metric", "Standard", "Enriched"],
        rows=sev_table_rows,
        col_widths=[80, 48, 40],
    )
except Exception as e:
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"[Severity comparison table unavailable: {e}]", new_x="LMARGIN", new_y="NEXT")

pdf.ln(4)

pdf.section_heading("4.3  Loss Ratio Stability", level=2)
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "A lower standard deviation of loss ratio across premium deciles indicates more stable pricing. "
    "The enriched model should exhibit lower LR volatility, confirming improved risk segmentation."
)
pdf.ln(2)

lr_rows = []
for model_name in lr_freq_df["model"].unique():
    subset = lr_freq_df[lr_freq_df["model"] == model_name]
    lr_rows.append([
        f"Frequency — {model_name}",
        f"{subset['loss_ratio'].mean():.3f}",
        f"{subset['loss_ratio'].std():.3f}",
        f"{subset['loss_ratio'].min():.3f}",
        f"{subset['loss_ratio'].max():.3f}",
    ])
for model_name in lr_sev_df["model"].unique():
    subset = lr_sev_df[lr_sev_df["model"] == model_name]
    lr_rows.append([
        f"Severity — {model_name}",
        f"{subset['loss_ratio'].mean():.3f}",
        f"{subset['loss_ratio'].std():.3f}",
        f"{subset['loss_ratio'].min():.3f}",
        f"{subset['loss_ratio'].max():.3f}",
    ])

pdf.table(
    headers=["Model", "Mean LR", "Std Dev", "Min LR", "Max LR"],
    rows=lr_rows,
    col_widths=[72, 24, 24, 24, 24],
)

# ===========================================================================
# PAGE 6 — Sensitivity Analysis
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 5 — Sensitivity Analysis")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "This section quantifies the contribution of each enrichment feature to model fit. "
    "The 'average AIC improvement' measures the reduction in AIC when a feature is included, "
    "averaged across all specifications in the factory that contain it. "
    "A larger positive value indicates a more important feature."
)
pdf.ln(3)

sens_rows = []
for _, row in impact_df.sort_values("avg_aic_improvement", ascending=False).iterrows():
    direction = "Improves fit" if row["avg_aic_improvement"] > 0 else "Worsens fit"
    sens_rows.append([
        row["feature"],
        f"{row['avg_aic_improvement']:+,.1f}",
        direction,
    ])

pdf.table(
    headers=["Enrichment Feature", "Avg AIC Improvement", "Direction"],
    rows=sens_rows,
    col_widths=[88, 52, 46],
)

pdf.ln(4)
pdf.section_heading("5.1  Governance Implication", level=2)
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    f"The most impactful enrichment feature is '{most_important['feature']}' with an average AIC improvement "
    f"of {most_important['avg_aic_improvement']:,.1f} points. Removal of this feature would represent a material "
    f"reduction in model quality and would require Pricing Committee sign-off. "
    f"The least impactful feature is '{least_important['feature']}' ({least_important['avg_aic_improvement']:,.1f} points) "
    f"and could be removed with lower governance concern if data availability constraints arise."
)

# ===========================================================================
# PAGE 7 — Data Quality & Limitations
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 6 — Data Quality & Limitations")

pdf.section_heading("6.1  Data Source", level=2)
datasource_rows = [
    ["Policies & claims",  "Synthetic — generated programmatically in 01_build_all_models.py"],
    ["Enrichment features","Real UK public data: IMD 2019 (MHCLG), ONSPD (ONS), ONS RUC 2011, coastal LAs"],
    ["Sample size",        "50,000 simulated home insurance policies keyed to real UK postcodes"],
    ["Train/test split",   "70% / 30% random split (seed = 42)"],
    ["Policy period",      "Single-period; no longitudinal dimension"],
]
pdf.table(
    headers=["Attribute", "Detail"],
    rows=datasource_rows,
    col_widths=[60, 126],
)
pdf.ln(4)

pdf.section_heading("6.2  Known Assumptions & Limitations", level=2)
limitations = [
    "Synthetic policies & claims: Claim process is simulated. Results are illustrative, not indicative of real portfolio behaviour.",
    "Real enrichment features: IMD 2019, ONSPD, and ONS RUC 2011 are real UK public datasets; collinearity between deprivation deciles is material and must be monitored.",
    "No temporal effects: Single cross-sectional snapshot; no inflation, portfolio drift, or underwriting cycle.",
    "LSOA granularity: Enrichment deciles are constant within each LSOA; within-LSOA risk variation is not captured.",
    "England-only IMD: IMD 2019 covers English LSOAs; non-English postcodes rely on region dummies only.",
    "Exposure homogeneity: All policies carry a single year of exposure (no offset term required).",
    "Linear enrichment effects: Non-linearities (e.g., threshold effects in deprivation deciles) are not captured.",
    "No interaction search: Interactions were limited to pre-specified combinations only.",
    "Severity independence: Frequency and severity models are fitted independently (two-part model).",
    "Regulatory permissibility: Deprivation-based and postcode-derived features require regulatory review (FCA GIPP, Equality Act, GDPR) as they may proxy for protected characteristics.",
    "GLM over-dispersion: The Poisson family assumes mean = variance; over-dispersion untested.",
    "LightGBM severity: May overfit on small claim sub-populations.",
]
for lim in limitations:
    pdf.bullet(lim)

pdf.ln(4)
pdf.section_heading("6.3  Data Quality Flags", level=2)
dq_rows = [
    ["Synthetic claims",       "Claim frequency/severity are simulated",                "Results not generalisable to real portfolios"],
    ["Real public enrichment", "IMD 2019, ONSPD, ONS RUC 2011 are real UK public data", "Collinearity between deprivation deciles must be monitored"],
    ["LSOA granularity",       "Enrichment deciles constant within each LSOA",          "Within-LSOA risk variation not captured"],
    ["England-only IMD",       "IMD 2019 covers English LSOAs",                         "Welsh/Scottish/NI postcodes rely on region dummies only"],
    ["No premium loading",     "Quotes use a fixed expense load",                       "Real pricing requires bespoke expense/profit loads"],
]
pdf.table(
    headers=["Flag", "Description", "Impact"],
    rows=dq_rows,
    col_widths=[42, 72, 72],
)

# ===========================================================================
# PAGE 8 — Sign-Off
# ===========================================================================
pdf.add_page()
pdf.section_heading("Section 7 — Sign-Off & Approval")

pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "This section must be completed before the model is approved for production use. "
    "All signatories confirm they have reviewed the full report and are satisfied with the "
    "modelling approach, evidence presented, and governance documentation."
)
pdf.ln(4)

pdf.section_heading("7.1  Sign-Off Form", level=2)
signoff_rows = [
    ["Model Owner / Pricing Actuary",  "___________________________", "___ / ___ / _____", "___________________"],
    ["Peer Reviewer (Actuary)",         "___________________________", "___ / ___ / _____", "___________________"],
    ["Model Validation (Independent)",  "___________________________", "___ / ___ / _____", "___________________"],
    ["Chief Actuary",                   "___________________________", "___ / ___ / _____", "___________________"],
    ["Pricing Committee Chair",         "___________________________", "___ / ___ / _____", "___________________"],
]
pdf.table(
    headers=["Role", "Name", "Date", "Signature"],
    rows=signoff_rows,
    col_widths=[60, 60, 34, 32],
    stripe=False,
)
pdf.ln(6)

pdf.section_heading("7.2  Review Checklist", level=2)
checklist = [
    "Model specification is documented and reproducible from source notebooks",
    "All features have documented business rationale",
    "Feature statistical significance has been reviewed",
    "Model selection criterion (AIC) is appropriate and justified",
    "Sensitivity analysis reviewed; no single-feature dependency identified",
    "Loss ratio stability across deciles is acceptable",
    "Data quality limitations are understood and accepted",
    "Enrichment features reviewed for regulatory permissibility",
    "Model outputs have been back-tested against available experience",
    "Unity Catalog artefacts are correctly versioned and accessible",
    "Governance summary table (model_governance_summary) reviewed and accurate",
    "This report has been archived to the model documentation repository",
]
for item in checklist:
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(pdf.l_margin + 8)
    pdf.cell(8, 6, "[ ]")
    remaining_w = pdf.w - pdf.get_x() - pdf.r_margin
    pdf.multi_cell(remaining_w, 6, _ascii(item))

pdf.ln(5)
pdf.section_heading("7.3  Approval Status", level=2)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(183, 28, 28)
pdf.cell(0, 8, "Current Status:  DRAFT - Pending Review", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*GREY)
pdf.cell(0, 6, "Change to APPROVED or REJECTED upon completion of review.", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(*BLACK)

# ---------------------------------------------------------------------------
# Save the PDF
# ---------------------------------------------------------------------------
pdf.output(PDF_PATH)

print(f"PDF report saved to Unity Catalog volume:")
print(f"  {PDF_PATH}")
print()
print("To access:")
print(f"  1. Catalog Explorer > {CATALOG} > {SCHEMA} > Volumes > reports")
print(f"  2. Or: SELECT * FROM list('{VOLUME_PATH}')")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3 — Sign-Off Template

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Model Review Sign-Off
# MAGIC
# MAGIC This section must be completed before the model is approved for production use.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Sign-Off Form
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
# MAGIC ### Review Checklist
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
# MAGIC - [ ] PDF report has been generated and distributed to all signatories
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Approval Status
# MAGIC
# MAGIC > **Current Status:** `DRAFT - Pending Review`
# MAGIC >
# MAGIC > Change to `APPROVED` or `REJECTED` upon completion of review.
# MAGIC >
# MAGIC > **PDF Report:** Saved to UC volume at `lr_serverless_aws_us_catalog.pricing_new_data_impact.reports`
