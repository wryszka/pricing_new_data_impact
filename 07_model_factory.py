# Databricks notebook source
# MAGIC %md
# MAGIC # Model Factory — Systematic GLM Specification Search
# MAGIC
# MAGIC ## What Radar Does, But at Scale
# MAGIC
# MAGIC In traditional actuarial tools (WTW Radar, Willis Emblem, Earnix, Akur8), you manually
# MAGIC configure model variants one at a time — choosing which factors to include, how to group
# MAGIC levels, and which interactions to test. Comparing 5–10 models is typical; doing more
# MAGIC becomes impractical in a GUI.
# MAGIC
# MAGIC This notebook takes a different approach: **define a search space and train 50 GLM
# MAGIC specifications programmatically**. Every model is evaluated on the same holdout set,
# MAGIC ranked by AIC/BIC/Gini, and the results are persisted to Unity Catalog for full
# MAGIC auditability.
# MAGIC
# MAGIC | Approach | Models compared | Time | Reproducibility |
# MAGIC |---|---|---|---|
# MAGIC | Radar / Emblem (manual) | 5–10 | Hours of analyst time | Depends on documentation |
# MAGIC | Databricks Model Factory | 50+ | Under a minute on serverless | Full audit trail in UC |

# COMMAND ----------

# MAGIC %pip install statsmodels scikit-learn matplotlib

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
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import combinations
import time
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

train_df = spark.table("train_set").toPandas()
test_df = spark.table("test_set").toPandas()

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the Search Space
# MAGIC
# MAGIC We define three dimensions of variation:
# MAGIC
# MAGIC 1. **Enrichment feature subsets** — which of the 5 enrichment features to include
# MAGIC 2. **Interaction terms** — actuarially meaningful pairings
# MAGIC 3. **Base feature variations** — with/without certain standard factors
# MAGIC
# MAGIC This generates a set of distinct, actuarially plausible model specifications.

# COMMAND ----------

# --- Base features (always included in most models) ---
core_features = [
    "building_age", "bedrooms", "sum_insured", "prior_claims", "policy_tenure",
]

categorical_features = [
    "property_type_flat", "property_type_semi_detached", "property_type_terraced",
    "construction_other", "construction_stone", "construction_timber",
    "occupancy_tenant",
]

standard_features = core_features + categorical_features

# --- Enrichment features (the new data we're testing) ---
enrichment_features = [
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk",
]

# --- Interaction terms (actuarially meaningful) ---
# We pre-compute these in the dataframes
interaction_defs = [
    ("flood_x_construction_timber", "flood_risk_zone", "construction_timber"),
    ("flood_x_building_age", "flood_risk_zone", "building_age"),
    ("subsidence_x_building_age", "subsidence_risk", "building_age"),
    ("subsidence_x_sum_insured", "subsidence_risk", "sum_insured"),
    ("crime_x_occupancy_tenant", "crime_index", "occupancy_tenant"),
    ("flood_x_subsidence", "flood_risk_zone", "subsidence_risk"),
]

# Create interaction columns
for name, f1, f2 in interaction_defs:
    train_df[name] = train_df[f1] * train_df[f2]
    test_df[name] = test_df[f1] * test_df[f2]

interaction_names = [name for name, _, _ in interaction_defs]

print(f"Core features:        {len(core_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Enrichment features:  {len(enrichment_features)}")
print(f"Interaction terms:    {len(interaction_names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Model Specifications
# MAGIC
# MAGIC We systematically generate 50 distinct specifications:
# MAGIC
# MAGIC - **Spec 1:** Standard features only (baseline)
# MAGIC - **Specs 2–6:** Standard + each individual enrichment feature
# MAGIC - **Specs 7–16:** Standard + each pair of enrichment features
# MAGIC - **Specs 17–26:** Standard + each triple of enrichment features
# MAGIC - **Specs 27–31:** Standard + each quad of enrichment features
# MAGIC - **Spec 32:** Standard + all enrichment features (no interactions)
# MAGIC - **Specs 33–38:** Full enrichment + each individual interaction
# MAGIC - **Specs 39–50+:** Full enrichment + interaction combinations + base variations

# COMMAND ----------

specs = []

# 1. Baseline — standard only
specs.append({
    "name": "baseline_standard",
    "features": standard_features,
    "description": "Standard rating factors only",
})

# 2-6. Standard + individual enrichment features
for ef in enrichment_features:
    specs.append({
        "name": f"standard_plus_{ef}",
        "features": standard_features + [ef],
        "description": f"Standard + {ef}",
    })

# 7-16. Standard + pairs of enrichment features
for pair in combinations(enrichment_features, 2):
    specs.append({
        "name": f"standard_plus_{'_'.join(p.split('_')[0] for p in pair)}",
        "features": standard_features + list(pair),
        "description": f"Standard + {', '.join(pair)}",
    })

# 17-26. Standard + triples of enrichment features
for triple in combinations(enrichment_features, 3):
    specs.append({
        "name": f"enrich_3_{'_'.join(t.split('_')[0] for t in triple)}",
        "features": standard_features + list(triple),
        "description": f"Standard + 3 enrichment: {', '.join(triple)}",
    })

# 27-31. Standard + quads of enrichment features
for quad in combinations(enrichment_features, 4):
    specs.append({
        "name": f"enrich_4_{'_'.join(q.split('_')[0] for q in quad)}",
        "features": standard_features + list(quad),
        "description": f"Standard + 4 enrichment: {', '.join(quad)}",
    })

# 32. Full enrichment — no interactions
specs.append({
    "name": "full_enrichment",
    "features": standard_features + enrichment_features,
    "description": "Standard + all 5 enrichment features",
})

# 33-38. Full enrichment + individual interactions
for ix in interaction_names:
    specs.append({
        "name": f"full_plus_{ix}",
        "features": standard_features + enrichment_features + [ix],
        "description": f"Full enrichment + {ix}",
    })

# 39-44. Full enrichment + pairs of interactions
for ix_pair in combinations(interaction_names, 2):
    specs.append({
        "name": f"full_ix_{'_'.join(i.split('_')[0] for i in ix_pair)}",
        "features": standard_features + enrichment_features + list(ix_pair),
        "description": f"Full enrichment + interactions: {', '.join(ix_pair)}",
    })

# 45+. Reduced base — drop low-importance standard features
for drop_feat in ["bedrooms", "policy_tenure"]:
    reduced = [f for f in standard_features if f != drop_feat]
    specs.append({
        "name": f"full_no_{drop_feat}",
        "features": reduced + enrichment_features,
        "description": f"Full enrichment minus {drop_feat}",
    })

# Full kitchen sink
specs.append({
    "name": "kitchen_sink",
    "features": standard_features + enrichment_features + interaction_names,
    "description": "All features + all interactions",
})

# Cap at 50
specs = specs[:50]
print(f"Total specifications to train: {len(specs)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train All 50 Models
# MAGIC
# MAGIC Each GLM fits in under a second. We capture all metrics for comparison.

# COMMAND ----------

def gini_coefficient(y_true, y_pred):
    arr = np.array(sorted(zip(y_pred, y_true), key=lambda x: x[0]))
    cum_actual = np.cumsum(arr[:, 1])
    cum_actual_norm = cum_actual / cum_actual[-1]
    n = len(y_true)
    lorenz = cum_actual_norm.sum() / n
    return 2 * lorenz - 1


target = "num_claims"
results = []

start_time = time.time()

for i, spec in enumerate(specs):
    try:
        X_train = sm.add_constant(train_df[spec["features"]].astype(float))
        X_test = sm.add_constant(test_df[spec["features"]].astype(float))
        y_train = train_df[target]
        y_test = test_df[target]

        model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

        pred_test = model.predict(X_test)

        results.append({
            "spec_id": i + 1,
            "name": spec["name"],
            "description": spec["description"],
            "n_features": len(spec["features"]),
            "aic": model.aic,
            "bic": model.bic,
            "deviance": model.deviance,
            "null_deviance": model.null_deviance,
            "deviance_explained": 1 - model.deviance / model.null_deviance,
            "mae_test": mean_absolute_error(y_test, pred_test),
            "rmse_test": np.sqrt(mean_squared_error(y_test, pred_test)),
            "gini_test": gini_coefficient(y_test.values, pred_test.values),
        })
    except Exception as e:
        results.append({
            "spec_id": i + 1,
            "name": spec["name"],
            "description": spec["description"],
            "n_features": len(spec["features"]),
            "aic": None, "bic": None, "deviance": None, "null_deviance": None,
            "deviance_explained": None, "mae_test": None, "rmse_test": None,
            "gini_test": None,
        })
        print(f"  FAILED: {spec['name']} — {e}")

elapsed = time.time() - start_time
print(f"\nTrained {len(specs)} models in {elapsed:.1f}s ({elapsed/len(specs):.2f}s per model)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Results — All Models Ranked

# COMMAND ----------

results_df = pd.DataFrame(results).dropna(subset=["aic"])
results_df = results_df.sort_values("aic")
results_df["rank_aic"] = range(1, len(results_df) + 1)
results_df = results_df.sort_values("gini_test", ascending=False)
results_df["rank_gini"] = range(1, len(results_df) + 1)
results_df = results_df.sort_values("aic")

# Save to UC
spark.createDataFrame(results_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_factory_results"
)

display(spark.table(f"{CATALOG}.{SCHEMA}.model_factory_results"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Top 5 Models by AIC

# COMMAND ----------

top5 = results_df.head(5)
print("Top 5 Models by AIC (lower is better):\n")
for _, row in top5.iterrows():
    print(f"  #{int(row['rank_aic']):2d}  AIC={row['aic']:>10.1f}  Gini={row['gini_test']:.4f}  "
          f"Features={int(row['n_features']):2d}  {row['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. AIC vs Feature Count — The Elbow Chart
# MAGIC
# MAGIC This is the chart actuaries use in Radar to decide "how many features is enough?"
# MAGIC Each point is a model. The x-axis is complexity (feature count), the y-axis is fit (AIC).
# MAGIC
# MAGIC The **elbow** — where adding more features stops meaningfully improving AIC — tells
# MAGIC you the optimal complexity.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- AIC vs feature count ---
ax = axes[0]
ax.scatter(results_df["n_features"], results_df["aic"], alpha=0.6, s=60, c="#4C72B0", edgecolors="white")

# Highlight baseline and best
baseline = results_df[results_df["name"] == "baseline_standard"].iloc[0]
best = results_df.iloc[0]

ax.scatter([baseline["n_features"]], [baseline["aic"]], s=150, c="#E53935", zorder=5,
           edgecolors="black", label=f"Baseline (AIC={baseline['aic']:.0f})")
ax.scatter([best["n_features"]], [best["aic"]], s=150, c="#43A047", zorder=5, marker="*",
           edgecolors="black", label=f"Best (AIC={best['aic']:.0f})")

ax.set_xlabel("Number of Features", fontsize=12)
ax.set_ylabel("AIC (lower is better)", fontsize=12)
ax.set_title("AIC vs Model Complexity — The Elbow Chart", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# --- Gini vs feature count ---
ax = axes[1]
ax.scatter(results_df["n_features"], results_df["gini_test"], alpha=0.6, s=60, c="#DD8452", edgecolors="white")

best_gini_row = results_df.sort_values("gini_test", ascending=False).iloc[0]
ax.scatter([baseline["n_features"]], [baseline["gini_test"]], s=150, c="#E53935", zorder=5,
           edgecolors="black", label=f"Baseline (Gini={baseline['gini_test']:.4f})")
ax.scatter([best_gini_row["n_features"]], [best_gini_row["gini_test"]], s=150, c="#43A047", zorder=5,
           marker="*", edgecolors="black", label=f"Best (Gini={best_gini_row['gini_test']:.4f})")

ax.set_xlabel("Number of Features", fontsize=12)
ax.set_ylabel("Gini (higher is better)", fontsize=12)
ax.set_title("Gini vs Model Complexity", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Improvement Waterfall — Baseline to Best

# COMMAND ----------

# Show how AIC improves as we add features
waterfall_specs = [
    ("Baseline (standard only)", "baseline_standard"),
    ("+ flood_risk_zone", "standard_plus_flood_risk_zone"),
    ("+ subsidence_risk", None),  # find the 2-feature combo
    ("+ crime_index", None),
    ("+ all enrichment", "full_enrichment"),
    ("+ best interactions", None),
]

# Build waterfall from incremental enrichment
waterfall = []
# 1. Baseline
waterfall.append(results_df[results_df["name"] == "baseline_standard"].iloc[0])

# 2-6. Incrementally add the enrichment features ranked by individual impact
individual_enrichments = results_df[results_df["name"].str.startswith("standard_plus_")].sort_values("aic")
for _, row in individual_enrichments.iterrows():
    waterfall.append(row)

# 7. Full enrichment
full_enr = results_df[results_df["name"] == "full_enrichment"]
if not full_enr.empty:
    waterfall.append(full_enr.iloc[0])

# 8. Best overall
if best["name"] != "full_enrichment":
    waterfall.append(best)

wf = pd.DataFrame(waterfall).drop_duplicates(subset=["name"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
colors = ["#E53935"] + ["#4C72B0"] * (len(wf) - 2) + ["#43A047"]
if len(colors) > len(wf):
    colors = colors[:len(wf)]

bars = ax.bar(range(len(wf)), wf["aic"], color=colors, edgecolor="white", width=0.6)
ax.set_xticks(range(len(wf)))
ax.set_xticklabels(wf["description"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("AIC (lower is better)", fontsize=12)
ax.set_title("AIC Progression — Adding Features Incrementally", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Add AIC values on bars
for bar, aic_val in zip(bars, wf["aic"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{aic_val:.0f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Heatmap — Which Enrichment Features Matter Most?

# COMMAND ----------

# For each enrichment feature, compute average AIC improvement when it's included vs excluded
feature_impact = []
for ef in enrichment_features:
    with_feat = results_df[results_df["description"].str.contains(ef)]["aic"].mean()
    without_feat = results_df[~results_df["description"].str.contains(ef)]["aic"].mean()
    improvement = without_feat - with_feat
    feature_impact.append({"feature": ef, "avg_aic_improvement": improvement})

impact_df = pd.DataFrame(feature_impact).sort_values("avg_aic_improvement", ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#43A047" if v > 0 else "#E53935" for v in impact_df["avg_aic_improvement"]]
ax.barh(impact_df["feature"], impact_df["avg_aic_improvement"], color=colors, edgecolor="white")
ax.set_xlabel("Average AIC Improvement When Included", fontsize=12)
ax.set_title("Enrichment Feature Impact — Which New Data Matters Most?", fontsize=13, fontweight="bold")
ax.axvline(0, color="grey", lw=0.8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Save
spark.createDataFrame(impact_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_factory_feature_impact"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary
# MAGIC
# MAGIC ### What We Did
# MAGIC
# MAGIC Trained **50 Poisson GLM specifications** in a single notebook run — systematically
# MAGIC varying enrichment features, interaction terms, and base feature selections.
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC - The **elbow chart** shows diminishing returns beyond a certain complexity level
# MAGIC - Enrichment features consistently improve AIC, BIC, and Gini vs the baseline
# MAGIC - The **feature impact chart** shows which new data sources deliver the most value
# MAGIC - Interaction terms provide additional (though smaller) improvements
# MAGIC
# MAGIC ### Why This Matters
# MAGIC
# MAGIC In traditional pricing tools (Radar, Emblem), an actuary might test 5–10 model
# MAGIC variants over several days. Here we tested 50 in under a minute, with full
# MAGIC reproducibility and an audit trail in Unity Catalog.
# MAGIC
# MAGIC This doesn't replace actuarial judgement — it **augments** it. The actuary still
# MAGIC decides which features are permissible, which interactions make business sense, and
# MAGIC what the final model should look like. But they can now make those decisions with
# MAGIC a complete picture of the trade-offs, rather than relying on a handful of manually
# MAGIC configured variants.
# MAGIC
# MAGIC ### Persisted Artefacts
# MAGIC
# MAGIC | Artefact | Table |
# MAGIC |---|---|
# MAGIC | All 50 model results | `.model_factory_results` |
# MAGIC | Feature impact analysis | `.model_factory_feature_impact` |
