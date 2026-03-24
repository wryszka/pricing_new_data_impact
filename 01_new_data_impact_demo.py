# Databricks notebook source
# MAGIC %md
# MAGIC # Does More Data Build Better Models?
# MAGIC ## Home Insurance GLM Comparison
# MAGIC
# MAGIC This notebook demonstrates that enriching a GLM with additional data sources
# MAGIC leads to measurably better risk segmentation and pricing accuracy.
# MAGIC
# MAGIC | | Model 1 — Standard | Model 2 — Enriched |
# MAGIC |---|---|---|
# MAGIC | **Features** | Classic rating factors | Standard + geo/risk enrichment |
# MAGIC | **Goal** | Baseline performance | Show uplift from new data |
# MAGIC
# MAGIC **All artefacts are persisted to Unity Catalog:**
# MAGIC - Tables → `lr_serverless_aws_us_catalog.pricing_new_data_impact`
# MAGIC - Models → same catalog, registered via MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup — Create Schema & Configure MLflow

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Home Insurance Portfolio

# COMMAND ----------

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(42)
N = 50_000

# --- Standard rating factors ---
property_type = np.random.choice(
    ["detached", "semi_detached", "terraced", "flat"], N, p=[0.2, 0.3, 0.3, 0.2]
)
construction = np.random.choice(
    ["brick", "timber", "stone", "other"], N, p=[0.5, 0.2, 0.2, 0.1]
)
year_built = np.random.randint(1900, 2024, N)
building_age = 2025 - year_built
bedrooms = np.random.choice([1, 2, 3, 4, 5], N, p=[0.1, 0.25, 0.35, 0.2, 0.1])
sum_insured = np.round(
    np.random.lognormal(mean=12.2, sigma=0.4, size=N), -3
)
occupancy = np.random.choice(["owner", "tenant"], N, p=[0.65, 0.35])
prior_claims = np.random.poisson(0.15, N)
policy_tenure = np.random.randint(0, 15, N)

# --- Enrichment factors (Model 2 will use these) ---
flood_risk_zone = np.random.choice([1, 2, 3, 4], N, p=[0.50, 0.25, 0.15, 0.10])
crime_index = np.round(np.random.beta(2, 5, N) * 100, 1)
distance_fire_station_km = np.round(np.random.exponential(3, N), 1)
annual_rainfall_mm = np.round(np.random.normal(800, 200, N).clip(300, 1600), 0)
subsidence_risk = np.random.choice([0, 1], N, p=[0.85, 0.15])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simulate Claims (Ground Truth Depends on ALL Factors)
# MAGIC
# MAGIC The key insight: the **true** data-generating process uses enrichment variables.
# MAGIC Model 1 cannot see them, so it will underperform.

# COMMAND ----------

# --- Encode categoricals for the true DGP ---
prop_effect = {"detached": 0.1, "semi_detached": 0.0, "terraced": -0.05, "flat": -0.1}
cons_effect = {"brick": -0.1, "timber": 0.2, "stone": 0.0, "other": 0.15}
occ_effect = {"owner": -0.05, "tenant": 0.1}

prop_vec = np.array([prop_effect[p] for p in property_type])
cons_vec = np.array([cons_effect[c] for c in construction])
occ_vec = np.array([occ_effect[o] for o in occupancy])

# True log-frequency depends on ALL factors
log_freq = (
    -2.5
    + prop_vec
    + cons_vec
    + occ_vec
    + 0.003 * building_age
    + 0.05 * prior_claims
    - 0.01 * policy_tenure
    # --- enrichment effects (hidden from Model 1) ---
    + 0.25 * (flood_risk_zone - 1) / 3
    + 0.005 * crime_index
    + 0.02 * (distance_fire_station_km > 5).astype(float)
    + 0.0003 * (annual_rainfall_mm - 800)
    + 0.3 * subsidence_risk
)

claim_freq = np.exp(log_freq)
num_claims = np.random.poisson(claim_freq)

# Severity for those with claims
log_sev = (
    7.5
    + 0.15 * (flood_risk_zone - 1) / 3
    + 0.1 * subsidence_risk
    + 0.00001 * sum_insured / 1000
    + np.random.normal(0, 0.3, N)
)
claim_severity = np.where(num_claims > 0, np.exp(log_sev), 0)
total_loss = num_claims * claim_severity

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Assemble Dataset & Save to Unity Catalog

# COMMAND ----------

df = pd.DataFrame({
    "property_type": property_type,
    "construction": construction,
    "building_age": building_age,
    "bedrooms": bedrooms,
    "sum_insured": sum_insured,
    "occupancy": occupancy,
    "prior_claims": prior_claims,
    "policy_tenure": policy_tenure,
    "flood_risk_zone": flood_risk_zone,
    "crime_index": crime_index,
    "distance_fire_station_km": distance_fire_station_km,
    "annual_rainfall_mm": annual_rainfall_mm,
    "subsidence_risk": subsidence_risk,
    "num_claims": num_claims,
    "claim_severity": claim_severity,
    "total_loss": total_loss,
})

# One-hot encode for modelling (cast to int to avoid uint8 arrow issues)
df_encoded = pd.get_dummies(df, columns=["property_type", "construction", "occupancy"], drop_first=True)
bool_cols = df_encoded.select_dtypes(include=["bool", "uint8"]).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Save raw portfolio to UC
spark.createDataFrame(df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.portfolio")

print(f"Portfolio: {N:,} policies | Claim rate: {(num_claims > 0).mean():.1%}")
print(f"Avg frequency: {num_claims.mean():.3f} | Avg severity (claimants): £{claim_severity[num_claims > 0].mean():,.0f}")
display(spark.table(f"{CATALOG}.{SCHEMA}.portfolio").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train / Test Split & Feature Sets

# COMMAND ----------

standard_features = [
    "building_age", "bedrooms", "sum_insured", "prior_claims", "policy_tenure",
    "property_type_flat", "property_type_semi_detached", "property_type_terraced",
    "construction_other", "construction_stone", "construction_timber",
    "occupancy_tenant",
]

enriched_features = standard_features + [
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk",
]

train_df, test_df = train_test_split(df_encoded, test_size=0.3, random_state=42)
print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

# Save train/test splits to UC
spark.createDataFrame(train_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.train_set")
spark.createDataFrame(test_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.test_set")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Helper Functions

# COMMAND ----------

def gini_coefficient(y_true, y_pred):
    """Ordered Lorenz / Gini for model discrimination."""
    arr = np.array(sorted(zip(y_pred, y_true), key=lambda x: x[0]))
    cum_actual = np.cumsum(arr[:, 1])
    cum_actual_norm = cum_actual / cum_actual[-1]
    n = len(y_true)
    lorenz = cum_actual_norm.sum() / n
    return 2 * lorenz - 1


def fit_and_log_glm(train, test, features, model_name, label="num_claims"):
    """Fit a Poisson GLM, log to MLflow, register in UC."""
    X_train = sm.add_constant(train[features].astype(float))
    X_test = sm.add_constant(test[features].astype(float))
    y_train = train[label]
    y_test = test[label]

    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    deviance_explained = 1 - model.deviance / model.null_deviance
    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    gini = gini_coefficient(y_test.values, pred_test.values)

    # Log to MLflow
    uc_model_name = f"{CATALOG}.{SCHEMA}.{model_name}"
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "family": "Poisson",
            "link": "log",
            "n_features": len(features),
            "features": ", ".join(features),
            "n_train": len(train),
            "n_test": len(test),
        })
        mlflow.log_metrics({
            "aic": model.aic,
            "bic": model.bic,
            "deviance": model.deviance,
            "null_deviance": model.null_deviance,
            "deviance_explained": deviance_explained,
            "mae_test": mae,
            "rmse_test": rmse,
            "gini_test": gini,
        })

        # Log model as pyfunc for UC registration — signature required
        from mlflow.models.signature import infer_signature
        sample_input = test[features].head(5).astype(float)
        sample_output = pd.Series(model.predict(sm.add_constant(sample_input)), name="predicted_frequency")
        signature = infer_signature(sample_input, sample_output)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=GLMWrapper(model, features),
            registered_model_name=uc_model_name,
            signature=signature,
        )

        run_id = run.info.run_id

    return {
        "model": model,
        "run_id": run_id,
        "uc_model_name": uc_model_name,
        "aic": model.aic,
        "bic": model.bic,
        "deviance": model.deviance,
        "null_deviance": model.null_deviance,
        "deviance_explained": deviance_explained,
        "mae_test": mae,
        "rmse_test": rmse,
        "gini_test": gini,
        "pred_test": pred_test,
        "y_test": y_test,
    }


class GLMWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a statsmodels GLM so it can be logged & served via MLflow."""

    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, context, model_input, params=None):
        X = sm.add_constant(model_input[self.features].astype(float))
        return self.model.predict(X).values

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Train & Register — Model 1 (Standard)

# COMMAND ----------

experiment_path = f"/Users/laurence.ryszka@databricks.com/pricing_new_data_impact/experiments"
mlflow.set_experiment(experiment_path)

print("Training Model 1 (Standard features)...")
m1 = fit_and_log_glm(train_df, test_df, standard_features, "glm_frequency_standard")
print(f"  Registered → {m1['uc_model_name']}")
print(f"  Gini: {m1['gini_test']:.4f} | Deviance explained: {m1['deviance_explained']:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Train & Register — Model 2 (Enriched)

# COMMAND ----------

print("Training Model 2 (Enriched features)...")
m2 = fit_and_log_glm(train_df, test_df, enriched_features, "glm_frequency_enriched")
print(f"  Registered → {m2['uc_model_name']}")
print(f"  Gini: {m2['gini_test']:.4f} | Deviance explained: {m2['deviance_explained']:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Side-by-Side Metric Comparison

# COMMAND ----------

metrics = ["aic", "bic", "deviance", "null_deviance", "deviance_explained", "mae_test", "rmse_test", "gini_test"]
labels = ["AIC", "BIC", "Deviance", "Null Deviance", "Deviance Explained", "MAE (test)", "RMSE (test)", "Gini (test)"]

comparison = pd.DataFrame({
    "metric": labels,
    "model_1_standard": [f"{m1[m]:.4f}" for m in metrics],
    "model_2_enriched": [f"{m2[m]:.4f}" for m in metrics],
})

spark_comp = spark.createDataFrame(comparison)
display(spark_comp)

# Save comparison table
spark_comp.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.model_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Lift Charts

# COMMAND ----------

def plot_lift_chart(y_true, y_pred, n_bins=10, label="Model"):
    df_lift = pd.DataFrame({"actual": y_true.values, "predicted": y_pred.values})
    df_lift["decile"] = pd.qcut(df_lift["predicted"], n_bins, labels=False, duplicates="drop")
    grouped = df_lift.groupby("decile").agg(
        avg_predicted=("predicted", "mean"),
        avg_actual=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()
    return grouped

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, res, title in [
    (axes[0], m1, "Model 1 — Standard"),
    (axes[1], m2, "Model 2 — Enriched"),
]:
    lift = plot_lift_chart(res["y_test"], res["pred_test"], label=title)
    ax.bar(lift["decile"] - 0.15, lift["avg_actual"], width=0.3, label="Actual", alpha=0.7, color="#2196F3")
    ax.bar(lift["decile"] + 0.15, lift["avg_predicted"], width=0.3, label="Predicted", alpha=0.7, color="#FF9800")
    ax.set_title(title)
    ax.set_xlabel("Risk Decile (low → high)")
    ax.set_ylabel("Average Claim Frequency")
    ax.legend()

plt.suptitle("Lift Chart — Predicted vs Actual Frequency by Decile", fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Generate Quotes & Save Priced Portfolio

# COMMAND ----------

avg_severity = df.loc[df["num_claims"] > 0, "claim_severity"].mean()
expense_load = 1.35

test_out = test_df.copy()
test_out["pred_freq_standard"] = m1["pred_test"].values
test_out["pred_freq_enriched"] = m2["pred_test"].values
test_out["quote_standard"] = np.round(m1["pred_test"].values * avg_severity * expense_load, 2)
test_out["quote_enriched"] = np.round(m2["pred_test"].values * avg_severity * expense_load, 2)
test_out["actual_loss"] = test_out["total_loss"]

# Save the full priced test set
spark.createDataFrame(test_out).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.priced_portfolio")

display(spark.table(f"{CATALOG}.{SCHEMA}.priced_portfolio").select(
    "building_age", "bedrooms", "sum_insured", "flood_risk_zone", "subsidence_risk",
    "quote_standard", "quote_enriched", "actual_loss"
).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Loss Ratio Analysis by Decile

# COMMAND ----------

lr_rows = []
for model_name, quote_col in [("Standard", "quote_standard"), ("Enriched", "quote_enriched")]:
    temp = test_out.copy()
    temp["decile"] = pd.qcut(temp[quote_col], 10, labels=False, duplicates="drop")
    grouped = temp.groupby("decile").agg(
        total_premium=(quote_col, "sum"),
        total_loss=("actual_loss", "sum"),
        policy_count=("actual_loss", "count"),
    ).reset_index()
    grouped["loss_ratio"] = grouped["total_loss"] / grouped["total_premium"]
    grouped["model"] = model_name
    lr_rows.append(grouped)

lr_all = pd.concat(lr_rows)
spark.createDataFrame(lr_all).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.loss_ratio_by_decile")

# Visualise
fig, ax = plt.subplots(figsize=(10, 5))
for model, color in [("Standard", "#E53935"), ("Enriched", "#43A047")]:
    subset = lr_all[lr_all["model"] == model]
    ax.plot(subset["decile"], subset["loss_ratio"], marker="o", label=model, color=color, linewidth=2)
ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5, label="Breakeven (LR=1)")
ax.set_xlabel("Premium Decile (cheapest → most expensive)")
ax.set_ylabel("Loss Ratio")
ax.set_title("Loss Ratio by Quote Decile — Standard vs Enriched")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Coefficient Comparison

# COMMAND ----------

coef_m1 = m1["model"].summary2().tables[1].reset_index()
coef_m1.columns = ["feature", "coef", "std_err", "z", "p_value", "ci_low", "ci_high"]
coef_m1["model"] = "standard"

coef_m2 = m2["model"].summary2().tables[1].reset_index()
coef_m2.columns = ["feature", "coef", "std_err", "z", "p_value", "ci_low", "ci_high"]
coef_m2["model"] = "enriched"

coef_all = pd.concat([coef_m1, coef_m2])
for c in ["coef", "std_err", "z", "p_value", "ci_low", "ci_high"]:
    coef_all[c] = coef_all[c].astype(float)
spark.createDataFrame(coef_all).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.glm_coefficients")
display(spark.table(f"{CATALOG}.{SCHEMA}.glm_coefficients"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Persisted Artefacts
# MAGIC
# MAGIC | Artefact | Location |
# MAGIC |---|---|
# MAGIC | Raw portfolio | `lr_serverless_aws_us_catalog.pricing_new_data_impact.portfolio` |
# MAGIC | Train set | `lr_serverless_aws_us_catalog.pricing_new_data_impact.train_set` |
# MAGIC | Test set | `lr_serverless_aws_us_catalog.pricing_new_data_impact.test_set` |
# MAGIC | Priced portfolio | `lr_serverless_aws_us_catalog.pricing_new_data_impact.priced_portfolio` |
# MAGIC | Model comparison | `lr_serverless_aws_us_catalog.pricing_new_data_impact.model_comparison` |
# MAGIC | Loss ratios | `lr_serverless_aws_us_catalog.pricing_new_data_impact.loss_ratio_by_decile` |
# MAGIC | GLM coefficients | `lr_serverless_aws_us_catalog.pricing_new_data_impact.glm_coefficients` |
# MAGIC | Model 1 (Standard) | `lr_serverless_aws_us_catalog.pricing_new_data_impact.glm_frequency_standard` |
# MAGIC | Model 2 (Enriched) | `lr_serverless_aws_us_catalog.pricing_new_data_impact.glm_frequency_enriched` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC | Dimension | Model 1 (Standard) | Model 2 (Enriched) | Verdict |
# MAGIC |---|---|---|---|
# MAGIC | **Deviance explained** | Lower | Higher | More data captures more variance |
# MAGIC | **AIC / BIC** | Higher | Lower | Better fit, even penalising complexity |
# MAGIC | **Gini** | Lower | Higher | Better risk discrimination |
# MAGIC | **Lift chart** | Flatter | Steeper | Enriched model separates good/bad risks |
# MAGIC | **Loss ratios** | More volatile | More stable | Fairer pricing across deciles |
# MAGIC
# MAGIC **Conclusion:** Adding flood, crime, subsidence, weather, and proximity data
# MAGIC materially improves the GLM's ability to price home insurance risk —
# MAGIC leading to better segmentation, fairer premiums, and lower adverse selection.
