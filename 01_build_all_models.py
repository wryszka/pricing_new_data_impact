# Databricks notebook source
# MAGIC %md
# MAGIC # Build All Models — Frequency, Severity & Model Factory
# MAGIC
# MAGIC This is the **one-time pipeline** that generates all data and models used in the
# MAGIC pricing new-data-impact demonstration. Run this notebook once to populate Unity Catalog
# MAGIC with every table, registered model, and evaluation artefact. The walkthrough notebooks
# MAGIC consume these outputs for presentation.
# MAGIC
# MAGIC | Section | What it does |
# MAGIC |---|---|
# MAGIC | 1. Setup | pip install, schema creation, MLflow config |
# MAGIC | 2. Synthetic Portfolio | Generate 50 k home insurance policies |
# MAGIC | 3. Train/Test Split | 70/30 split, save to UC |
# MAGIC | 4. Helper Functions | Gini, GLMWrapper, fit_and_log_glm |
# MAGIC | 5. Frequency GLMs | Train & register standard and enriched Poisson GLMs |
# MAGIC | 6. Frequency Metrics | Comparison table, priced portfolio, loss ratios, coefficients |
# MAGIC | 7. Severity GBMs | Filter claimants, train LightGBM (Gamma) standard & enriched |
# MAGIC | 8. Full Burning-Cost Quotes | freq × sev quotes and loss ratios |
# MAGIC | 9. Model Factory | 50 GLM specifications — search, train, rank, feature impact |
# MAGIC | 10. Artefact Summary | Print every table and model persisted to UC |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & pip install

# COMMAND ----------

# MAGIC %pip install statsmodels scikit-learn mlflow matplotlib lightgbm

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
import mlflow
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

mlflow.set_registry_uri("databricks-uc")

experiment_path = f"/Users/laurence.ryszka@databricks.com/pricing_new_data_impact/experiments"
mlflow.set_experiment(experiment_path)

print(f"Catalog : {CATALOG}")
print(f"Schema  : {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Synthetic Portfolio

# COMMAND ----------

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

# --- Encode categoricals for the true DGP ---
prop_effect = {"detached": 0.1, "semi_detached": 0.0, "terraced": -0.05, "flat": -0.1}
cons_effect = {"brick": -0.1, "timber": 0.2, "stone": 0.0, "other": 0.15}
occ_effect  = {"owner": -0.05, "tenant": 0.1}

prop_vec = np.array([prop_effect[p] for p in property_type])
cons_vec = np.array([cons_effect[c] for c in construction])
occ_vec  = np.array([occ_effect[o]  for o in occupancy])

# True log-frequency depends on ALL factors (enrichment effects hidden from Model 1)
log_freq = (
    -2.5
    + prop_vec
    + cons_vec
    + occ_vec
    + 0.003 * building_age
    + 0.05  * prior_claims
    - 0.01  * policy_tenure
    + 0.25  * (flood_risk_zone - 1) / 3
    + 0.005 * crime_index
    + 0.02  * (distance_fire_station_km > 5).astype(float)
    + 0.0003 * (annual_rainfall_mm - 800)
    + 0.3   * subsidence_risk
)

claim_freq = np.exp(log_freq)
num_claims = np.random.poisson(claim_freq)

log_sev = (
    7.5
    + 0.15 * (flood_risk_zone - 1) / 3
    + 0.1  * subsidence_risk
    + 0.00001 * sum_insured / 1000
    + np.random.normal(0, 0.3, N)
)
claim_severity = np.where(num_claims > 0, np.exp(log_sev), 0)
total_loss = num_claims * claim_severity

# COMMAND ----------

df = pd.DataFrame({
    "property_type":           property_type,
    "construction":            construction,
    "building_age":            building_age,
    "bedrooms":                bedrooms,
    "sum_insured":             sum_insured,
    "occupancy":               occupancy,
    "prior_claims":            prior_claims,
    "policy_tenure":           policy_tenure,
    "flood_risk_zone":         flood_risk_zone,
    "crime_index":             crime_index,
    "distance_fire_station_km": distance_fire_station_km,
    "annual_rainfall_mm":      annual_rainfall_mm,
    "subsidence_risk":         subsidence_risk,
    "num_claims":              num_claims,
    "claim_severity":          claim_severity,
    "total_loss":              total_loss,
})

# One-hot encode for modelling (cast to int to avoid uint8 arrow issues)
df_encoded = pd.get_dummies(df, columns=["property_type", "construction", "occupancy"], drop_first=True)
bool_cols = df_encoded.select_dtypes(include=["bool", "uint8"]).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Save raw portfolio to UC
spark.createDataFrame(df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.portfolio")

print(f"Portfolio: {N:,} policies | Claim rate: {(num_claims > 0).mean():.1%}")
print(f"Avg frequency: {num_claims.mean():.3f} | Avg severity (claimants): £{claim_severity[num_claims > 0].mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train/Test Split & Feature Sets

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
print("Saved: train_set, test_set")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Helper Functions

# COMMAND ----------

def gini_coefficient(y_true, y_pred):
    """Ordered Lorenz / Gini for model discrimination."""
    arr = np.array(sorted(zip(y_pred, y_true), key=lambda x: x[0]))
    cum_actual = np.cumsum(arr[:, 1])
    cum_actual_norm = cum_actual / cum_actual[-1]
    n = len(y_true)
    lorenz = cum_actual_norm.sum() / n
    return 2 * lorenz - 1


class GLMWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a statsmodels GLM so it can be logged and served via MLflow."""

    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, context, model_input, params=None):
        X = sm.add_constant(model_input[self.features].astype(float))
        return self.model.predict(X).values


def fit_and_log_glm(train, test, features, model_name, label="num_claims"):
    """Fit a Poisson GLM, log to MLflow, register in UC."""
    from mlflow.models.signature import infer_signature

    X_train = sm.add_constant(train[features].astype(float))
    X_test  = sm.add_constant(test[features].astype(float))
    y_train = train[label]
    y_test  = test[label]

    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    deviance_explained = 1 - model.deviance / model.null_deviance
    mae  = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    gini = gini_coefficient(y_test.values, pred_test.values)

    uc_model_name = f"{CATALOG}.{SCHEMA}.{model_name}"

    sample_input  = test[features].head(5).astype(float)
    sample_output = pd.Series(model.predict(sm.add_constant(sample_input)), name="predicted_frequency")
    signature = infer_signature(sample_input, sample_output)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "family":     "Poisson",
            "link":       "log",
            "n_features": len(features),
            "features":   ", ".join(features),
            "n_train":    len(train),
            "n_test":     len(test),
        })
        mlflow.log_metrics({
            "aic":                model.aic,
            "bic":                model.bic,
            "deviance":           model.deviance,
            "null_deviance":      model.null_deviance,
            "deviance_explained": deviance_explained,
            "mae_test":           mae,
            "rmse_test":          rmse,
            "gini_test":          gini,
        })
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=GLMWrapper(model, features),
            registered_model_name=uc_model_name,
            signature=signature,
        )
        run_id = run.info.run_id

    return {
        "model":              model,
        "run_id":             run_id,
        "uc_model_name":      uc_model_name,
        "aic":                model.aic,
        "bic":                model.bic,
        "deviance":           model.deviance,
        "null_deviance":      model.null_deviance,
        "deviance_explained": deviance_explained,
        "mae_test":           mae,
        "rmse_test":          rmse,
        "gini_test":          gini,
        "pred_test":          pred_test,
        "y_test":             y_test,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Frequency GLMs — Standard & Enriched

# COMMAND ----------

print("Training Frequency GLM — Model 1 (Standard features)...")
m1 = fit_and_log_glm(train_df, test_df, standard_features, "glm_frequency_standard")
print(f"  Registered → {m1['uc_model_name']}")
print(f"  Gini: {m1['gini_test']:.4f} | Deviance explained: {m1['deviance_explained']:.2%}")

# COMMAND ----------

print("Training Frequency GLM — Model 2 (Enriched features)...")
m2 = fit_and_log_glm(train_df, test_df, enriched_features, "glm_frequency_enriched")
print(f"  Registered → {m2['uc_model_name']}")
print(f"  Gini: {m2['gini_test']:.4f} | Deviance explained: {m2['deviance_explained']:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Frequency Metrics & Quotes

# COMMAND ----------

# --- Side-by-side metric comparison ---
metrics = ["aic", "bic", "deviance", "null_deviance", "deviance_explained", "mae_test", "rmse_test", "gini_test"]
labels  = ["AIC", "BIC", "Deviance", "Null Deviance", "Deviance Explained", "MAE (test)", "RMSE (test)", "Gini (test)"]

comparison = pd.DataFrame({
    "metric":           labels,
    "model_1_standard": [f"{m1[m]:.4f}" for m in metrics],
    "model_2_enriched": [f"{m2[m]:.4f}" for m in metrics],
})

spark_comp = spark.createDataFrame(comparison)
spark_comp.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.model_comparison")
print("Saved: model_comparison")

# COMMAND ----------

# --- Priced portfolio ---
avg_severity  = df.loc[df["num_claims"] > 0, "claim_severity"].mean()
expense_load  = 1.35

test_out = test_df.copy()
test_out["pred_freq_standard"] = m1["pred_test"].values
test_out["pred_freq_enriched"] = m2["pred_test"].values
test_out["quote_standard"]     = np.round(m1["pred_test"].values * avg_severity * expense_load, 2)
test_out["quote_enriched"]     = np.round(m2["pred_test"].values * avg_severity * expense_load, 2)
test_out["actual_loss"]        = test_out["total_loss"]

spark.createDataFrame(test_out).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.priced_portfolio")
print("Saved: priced_portfolio")

# COMMAND ----------

# --- Loss ratio by decile (frequency quotes) ---
lr_rows = []
for model_name, quote_col in [("Standard", "quote_standard"), ("Enriched", "quote_enriched")]:
    temp = test_out.copy()
    temp["decile"] = pd.qcut(temp[quote_col], 10, labels=False, duplicates="drop")
    grouped = temp.groupby("decile").agg(
        total_premium=(quote_col,     "sum"),
        total_loss=   ("actual_loss", "sum"),
        policy_count= ("actual_loss", "count"),
    ).reset_index()
    grouped["loss_ratio"] = grouped["total_loss"] / grouped["total_premium"]
    grouped["model"]      = model_name
    lr_rows.append(grouped)

lr_all = pd.concat(lr_rows)
spark.createDataFrame(lr_all).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.loss_ratio_by_decile")
print("Saved: loss_ratio_by_decile")

# COMMAND ----------

# --- GLM coefficients ---
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
print("Saved: glm_coefficients")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Train Severity GBMs — Standard & Enriched

# COMMAND ----------

# Load portfolio and filter to claimants only
portfolio  = spark.table(f"{CATALOG}.{SCHEMA}.portfolio").toPandas()

df_enc_sev = pd.get_dummies(portfolio, columns=["property_type", "construction", "occupancy"], drop_first=True)
bool_cols_sev = df_enc_sev.select_dtypes(include=["bool", "uint8"]).columns
df_enc_sev[bool_cols_sev] = df_enc_sev[bool_cols_sev].astype(int)

claimants = df_enc_sev[df_enc_sev["num_claims"] > 0].copy()
print(f"Total policies: {len(df_enc_sev):,}")
print(f"Claimants:      {len(claimants):,} ({len(claimants)/len(df_enc_sev):.1%})")
print(f"Avg severity:   £{claimants['claim_severity'].mean():,.0f}")
print(f"Median severity: £{claimants['claim_severity'].median():,.0f}")

# COMMAND ----------

sev_target = "claim_severity"

sev_train_df, sev_test_df = train_test_split(claimants, test_size=0.3, random_state=42)
print(f"Severity train: {len(sev_train_df):,} | Test: {len(sev_test_df):,}")

spark.createDataFrame(sev_train_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_train_set")
spark.createDataFrame(sev_test_df).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_test_set")
print("Saved: severity_train_set, severity_test_set")

# COMMAND ----------

def evaluate_severity_model(y_true, y_pred, model_name):
    """Compute severity-specific evaluation metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    gini = gini_coefficient(y_true, y_pred)
    bias = (y_pred.mean() - y_true.mean()) / y_true.mean()
    return {
        "model":          model_name,
        "mae":            mae,
        "rmse":           rmse,
        "mape":           mape,
        "gini":           gini,
        "mean_predicted": y_pred.mean(),
        "mean_actual":    y_true.mean(),
        "bias_pct":       bias * 100,
    }

# COMMAND ----------

lgb_params = {
    "objective":        "gamma",
    "metric":           "gamma_deviance",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "verbose":          -1,
    "seed":             42,
}

# --- Model 1: Standard ---
ds_train_std = lgb.Dataset(sev_train_df[standard_features], label=sev_train_df[sev_target])
ds_val_std   = lgb.Dataset(sev_test_df[standard_features],  label=sev_test_df[sev_target], reference=ds_train_std)

print("Training Severity GBM — Model 1 (Standard features)...")
model_sev_std = lgb.train(
    lgb_params,
    ds_train_std,
    num_boost_round=500,
    valid_sets=[ds_val_std],
    callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
)

pred_sev_std    = model_sev_std.predict(sev_test_df[standard_features])
metrics_sev_std = evaluate_severity_model(sev_test_df[sev_target].values, pred_sev_std, "Standard")
print(f"  MAE: £{metrics_sev_std['mae']:,.0f} | RMSE: £{metrics_sev_std['rmse']:,.0f} | Gini: {metrics_sev_std['gini']:.4f}")

# COMMAND ----------

# --- Model 2: Enriched ---
ds_train_enr = lgb.Dataset(sev_train_df[enriched_features], label=sev_train_df[sev_target])
ds_val_enr   = lgb.Dataset(sev_test_df[enriched_features],  label=sev_test_df[sev_target], reference=ds_train_enr)

print("Training Severity GBM — Model 2 (Enriched features)...")
model_sev_enr = lgb.train(
    lgb_params,
    ds_train_enr,
    num_boost_round=500,
    valid_sets=[ds_val_enr],
    callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
)

pred_sev_enr    = model_sev_enr.predict(sev_test_df[enriched_features])
metrics_sev_enr = evaluate_severity_model(sev_test_df[sev_target].values, pred_sev_enr, "Enriched")
print(f"  MAE: £{metrics_sev_enr['mae']:,.0f} | RMSE: £{metrics_sev_enr['rmse']:,.0f} | Gini: {metrics_sev_enr['gini']:.4f}")

# COMMAND ----------

# --- Severity metric comparison ---
sev_metric_names = ["MAE (£)", "RMSE (£)", "MAPE (%)", "Gini", "Mean Predicted (£)", "Mean Actual (£)", "Bias (%)"]
sev_metric_keys  = ["mae", "rmse", "mape", "gini", "mean_predicted", "mean_actual", "bias_pct"]

sev_comparison = pd.DataFrame({
    "metric":           sev_metric_names,
    "model_1_standard": [f"{metrics_sev_std[k]:.4f}" for k in sev_metric_keys],
    "model_2_enriched": [f"{metrics_sev_enr[k]:.4f}" for k in sev_metric_keys],
})

spark.createDataFrame(sev_comparison).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_model_comparison")
print("Saved: severity_model_comparison")

# COMMAND ----------

# --- Feature importance ---
imp_std = pd.DataFrame({
    "feature":    standard_features,
    "importance": model_sev_std.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=True)

imp_enr = pd.DataFrame({
    "feature":    enriched_features,
    "importance": model_sev_enr.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=True)

imp_std["model"] = "standard"
imp_enr["model"] = "enriched"
imp_all = pd.concat([imp_std, imp_enr])
spark.createDataFrame(imp_all).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_feature_importance")
print("Saved: severity_feature_importance")

# Top enrichment features by gain
print("\nTop enrichment features by gain (enriched model):")
print(imp_enr[imp_enr["feature"].isin(enriched_features)].sort_values("importance", ascending=False).head(5).to_string(index=False))

# COMMAND ----------

# --- Severity by risk segment ---
priced_freq = spark.table(f"{CATALOG}.{SCHEMA}.priced_portfolio").toPandas()

for col in ["flood_risk_zone", "subsidence_risk"]:
    seg = priced_freq.groupby(col).agg(
        avg_actual_sev=("claim_severity", "mean"),
        n=("actual_loss", "count"),
    ).reset_index()
    print(f"\n--- {col} ---")
    print(seg.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Full Burning-Cost Quotes

# COMMAND ----------

# Predict severity for ALL test policies (not just claimants) using the freq test set
sev_pred_std_full = model_sev_std.predict(priced_freq[standard_features])
sev_pred_enr_full = model_sev_enr.predict(priced_freq[enriched_features])

priced_full = priced_freq.copy()
priced_full["sev_pred_standard"]  = sev_pred_std_full
priced_full["sev_pred_enriched"]  = sev_pred_enr_full
priced_full["full_quote_standard"] = np.round(
    priced_full["pred_freq_standard"] * sev_pred_std_full * expense_load, 2
)
priced_full["full_quote_enriched"] = np.round(
    priced_full["pred_freq_enriched"] * sev_pred_enr_full * expense_load, 2
)

spark.createDataFrame(priced_full).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_priced_portfolio")
print("Saved: severity_priced_portfolio")
print(f"Avg full quote (standard): £{priced_full['full_quote_standard'].mean():.2f}")
print(f"Avg full quote (enriched): £{priced_full['full_quote_enriched'].mean():.2f}")
print(f"Avg actual loss:           £{priced_full['actual_loss'].mean():.2f}")

# COMMAND ----------

# --- Loss ratio by decile — full quotes ---
sev_lr_rows = []
for model_name, quote_col in [("Standard", "full_quote_standard"), ("Enriched", "full_quote_enriched")]:
    temp = priced_full.copy()
    temp["decile"] = pd.qcut(temp[quote_col], 10, labels=False, duplicates="drop")
    grouped = temp.groupby("decile").agg(
        total_premium=(quote_col,     "sum"),
        total_loss=   ("actual_loss", "sum"),
        policy_count= ("actual_loss", "count"),
    ).reset_index()
    grouped["loss_ratio"] = grouped["total_loss"] / grouped["total_premium"]
    grouped["model"]      = model_name
    sev_lr_rows.append(grouped)

sev_lr_all = pd.concat(sev_lr_rows)
spark.createDataFrame(sev_lr_all).write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.severity_loss_ratio_by_decile")
print("Saved: severity_loss_ratio_by_decile")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Factory — 50 GLM Specifications

# COMMAND ----------

# Load train/test sets (already in memory, but reload from UC to keep this section self-contained)
mf_train_df = spark.table(f"{CATALOG}.{SCHEMA}.train_set").toPandas()
mf_test_df  = spark.table(f"{CATALOG}.{SCHEMA}.test_set").toPandas()

print(f"Model Factory — Train: {len(mf_train_df):,} | Test: {len(mf_test_df):,}")

# COMMAND ----------

# --- Search space definition ---
core_features = [
    "building_age", "bedrooms", "sum_insured", "prior_claims", "policy_tenure",
]

categorical_features = [
    "property_type_flat", "property_type_semi_detached", "property_type_terraced",
    "construction_other", "construction_stone", "construction_timber",
    "occupancy_tenant",
]

mf_standard_features = core_features + categorical_features

enrichment_features = [
    "flood_risk_zone", "crime_index", "distance_fire_station_km",
    "annual_rainfall_mm", "subsidence_risk",
]

interaction_defs = [
    ("flood_x_construction_timber", "flood_risk_zone",  "construction_timber"),
    ("flood_x_building_age",        "flood_risk_zone",  "building_age"),
    ("subsidence_x_building_age",   "subsidence_risk",  "building_age"),
    ("subsidence_x_sum_insured",    "subsidence_risk",  "sum_insured"),
    ("crime_x_occupancy_tenant",    "crime_index",      "occupancy_tenant"),
    ("flood_x_subsidence",          "flood_risk_zone",  "subsidence_risk"),
]

for name, f1, f2 in interaction_defs:
    mf_train_df[name] = mf_train_df[f1] * mf_train_df[f2]
    mf_test_df[name]  = mf_test_df[f1]  * mf_test_df[f2]

interaction_names = [name for name, _, _ in interaction_defs]

print(f"Core features:        {len(core_features)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Enrichment features:  {len(enrichment_features)}")
print(f"Interaction terms:    {len(interaction_names)}")

# COMMAND ----------

# --- Generate specifications ---
specs = []

# 1. Baseline
specs.append({
    "name":        "baseline_standard",
    "features":    mf_standard_features,
    "description": "Standard rating factors only",
})

# 2-6. Standard + individual enrichment features
for ef in enrichment_features:
    specs.append({
        "name":        f"standard_plus_{ef}",
        "features":    mf_standard_features + [ef],
        "description": f"Standard + {ef}",
    })

# 7-16. Standard + pairs of enrichment features
for pair in combinations(enrichment_features, 2):
    specs.append({
        "name":        f"standard_plus_{'_'.join(p.split('_')[0] for p in pair)}",
        "features":    mf_standard_features + list(pair),
        "description": f"Standard + {', '.join(pair)}",
    })

# 17-26. Standard + triples of enrichment features
for triple in combinations(enrichment_features, 3):
    specs.append({
        "name":        f"enrich_3_{'_'.join(t.split('_')[0] for t in triple)}",
        "features":    mf_standard_features + list(triple),
        "description": f"Standard + 3 enrichment: {', '.join(triple)}",
    })

# 27-31. Standard + quads of enrichment features
for quad in combinations(enrichment_features, 4):
    specs.append({
        "name":        f"enrich_4_{'_'.join(q.split('_')[0] for q in quad)}",
        "features":    mf_standard_features + list(quad),
        "description": f"Standard + 4 enrichment: {', '.join(quad)}",
    })

# 32. Full enrichment — no interactions
specs.append({
    "name":        "full_enrichment",
    "features":    mf_standard_features + enrichment_features,
    "description": "Standard + all 5 enrichment features",
})

# 33-38. Full enrichment + individual interactions
for ix in interaction_names:
    specs.append({
        "name":        f"full_plus_{ix}",
        "features":    mf_standard_features + enrichment_features + [ix],
        "description": f"Full enrichment + {ix}",
    })

# 39-44. Full enrichment + pairs of interactions
for ix_pair in combinations(interaction_names, 2):
    specs.append({
        "name":        f"full_ix_{'_'.join(i.split('_')[0] for i in ix_pair)}",
        "features":    mf_standard_features + enrichment_features + list(ix_pair),
        "description": f"Full enrichment + interactions: {', '.join(ix_pair)}",
    })

# 45+. Reduced base — drop low-importance standard features
for drop_feat in ["bedrooms", "policy_tenure"]:
    reduced = [f for f in mf_standard_features if f != drop_feat]
    specs.append({
        "name":        f"full_no_{drop_feat}",
        "features":    reduced + enrichment_features,
        "description": f"Full enrichment minus {drop_feat}",
    })

# Kitchen sink
specs.append({
    "name":        "kitchen_sink",
    "features":    mf_standard_features + enrichment_features + interaction_names,
    "description": "All features + all interactions",
})

# Cap at 50
specs = specs[:50]
print(f"Total specifications to train: {len(specs)}")

# COMMAND ----------

# --- Train all 50 models ---
mf_target = "num_claims"
mf_results = []
start_time = time.time()

for i, spec in enumerate(specs):
    try:
        X_train = sm.add_constant(mf_train_df[spec["features"]].astype(float))
        X_test  = sm.add_constant(mf_test_df[spec["features"]].astype(float))
        y_train = mf_train_df[mf_target]
        y_test  = mf_test_df[mf_target]

        model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
        pred_test = model.predict(X_test)

        mf_results.append({
            "spec_id":           i + 1,
            "name":              spec["name"],
            "description":       spec["description"],
            "n_features":        len(spec["features"]),
            "aic":               model.aic,
            "bic":               model.bic,
            "deviance":          model.deviance,
            "null_deviance":     model.null_deviance,
            "deviance_explained": 1 - model.deviance / model.null_deviance,
            "mae_test":          mean_absolute_error(y_test, pred_test),
            "rmse_test":         np.sqrt(mean_squared_error(y_test, pred_test)),
            "gini_test":         gini_coefficient(y_test.values, pred_test.values),
        })
    except Exception as e:
        mf_results.append({
            "spec_id": i + 1, "name": spec["name"], "description": spec["description"],
            "n_features": len(spec["features"]),
            "aic": None, "bic": None, "deviance": None, "null_deviance": None,
            "deviance_explained": None, "mae_test": None, "rmse_test": None, "gini_test": None,
        })
        print(f"  FAILED: {spec['name']} — {e}")

elapsed = time.time() - start_time
print(f"\nTrained {len(specs)} models in {elapsed:.1f}s ({elapsed/len(specs):.2f}s per model)")

# COMMAND ----------

# --- Rank and save results ---
results_df = pd.DataFrame(mf_results).dropna(subset=["aic"])
results_df = results_df.sort_values("aic")
results_df["rank_aic"]  = range(1, len(results_df) + 1)
results_df = results_df.sort_values("gini_test", ascending=False)
results_df["rank_gini"] = range(1, len(results_df) + 1)
results_df = results_df.sort_values("aic")

spark.createDataFrame(results_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_factory_results"
)
print("Saved: model_factory_results")

# Top 5 by AIC
top5 = results_df.head(5)
print("\nTop 5 Models by AIC (lower is better):\n")
for _, row in top5.iterrows():
    print(f"  #{int(row['rank_aic']):2d}  AIC={row['aic']:>10.1f}  Gini={row['gini_test']:.4f}  "
          f"Features={int(row['n_features']):2d}  {row['description']}")

# COMMAND ----------

# --- Feature impact analysis ---
feature_impact = []
for ef in enrichment_features:
    with_feat    = results_df[results_df["description"].str.contains(ef)]["aic"].mean()
    without_feat = results_df[~results_df["description"].str.contains(ef)]["aic"].mean()
    improvement  = without_feat - with_feat
    feature_impact.append({"feature": ef, "avg_aic_improvement": improvement})

impact_df = pd.DataFrame(feature_impact).sort_values("avg_aic_improvement", ascending=False)
spark.createDataFrame(impact_df).write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.model_factory_feature_impact"
)
print("Saved: model_factory_feature_impact")
print("\nEnrichment feature impact (avg AIC improvement when included):")
print(impact_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary of All Persisted Artefacts

# COMMAND ----------

summary = [
    # Tables
    ("TABLE", "portfolio",                      "Raw 50k synthetic home insurance portfolio"),
    ("TABLE", "train_set",                      "Frequency model train split (70%)"),
    ("TABLE", "test_set",                       "Frequency model test split (30%)"),
    ("TABLE", "model_comparison",               "Frequency GLM metric comparison (standard vs enriched)"),
    ("TABLE", "priced_portfolio",               "Test set with frequency-only quotes"),
    ("TABLE", "loss_ratio_by_decile",           "Frequency quote loss ratios by decile"),
    ("TABLE", "glm_coefficients",               "GLM coefficients for both frequency models"),
    ("TABLE", "severity_train_set",             "Severity GBM train split (claimants only, 70%)"),
    ("TABLE", "severity_test_set",              "Severity GBM test split (claimants only, 30%)"),
    ("TABLE", "severity_model_comparison",      "Severity GBM metric comparison (standard vs enriched)"),
    ("TABLE", "severity_feature_importance",    "LightGBM feature importance (gain) for both severity models"),
    ("TABLE", "severity_priced_portfolio",      "Full freq × sev burning-cost quotes"),
    ("TABLE", "severity_loss_ratio_by_decile",  "Full quote loss ratios by decile"),
    ("TABLE", "model_factory_results",          "All 50 GLM specification results ranked by AIC"),
    ("TABLE", "model_factory_feature_impact",   "Average AIC improvement per enrichment feature"),
    # UC-registered models
    ("MODEL", "glm_frequency_standard",         "Poisson GLM — standard rating factors"),
    ("MODEL", "glm_frequency_enriched",         "Poisson GLM — standard + geo/risk enrichment"),
]

print(f"\n{'='*80}")
print(f"  BUILD COMPLETE — All artefacts persisted to {CATALOG}.{SCHEMA}")
print(f"{'='*80}\n")
print(f"  {'TYPE':<8}  {'ARTEFACT':<40}  DESCRIPTION")
print(f"  {'-'*8}  {'-'*40}  {'-'*35}")
for kind, name, desc in summary:
    print(f"  {kind:<8}  {name:<40}  {desc}")

print(f"\n  Total tables : {sum(1 for k, _, _ in summary if k == 'TABLE')}")
print(f"  Total models : {sum(1 for k, _, _ in summary if k == 'MODEL')}")
print(f"{'='*80}\n")
