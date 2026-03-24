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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Home Insurance Portfolio

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
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
)  # ~£200k median
occupancy = np.random.choice(["owner", "tenant"], N, p=[0.65, 0.35])
prior_claims = np.random.poisson(0.15, N)
policy_tenure = np.random.randint(0, 15, N)

# --- Enrichment factors (Model 2 will use these) ---
flood_risk_zone = np.random.choice([1, 2, 3, 4], N, p=[0.50, 0.25, 0.15, 0.10])
crime_index = np.round(np.random.beta(2, 5, N) * 100, 1)  # 0-100, skewed low
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
    + 0.25 * (flood_risk_zone - 1) / 3        # flood zone is a strong driver
    + 0.005 * crime_index                       # moderate effect
    + 0.02 * (distance_fire_station_km > 5)     # step effect beyond 5km
    + 0.0003 * (annual_rainfall_mm - 800)       # mild weather effect
    + 0.3 * subsidence_risk                     # significant binary risk
)

claim_freq = np.exp(log_freq)
num_claims = np.random.poisson(claim_freq)

# Severity for those with claims — also influenced by enrichment
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
# MAGIC ## 3. Assemble Dataset

# COMMAND ----------

df = pd.DataFrame({
    # Standard
    "property_type": property_type,
    "construction": construction,
    "building_age": building_age,
    "bedrooms": bedrooms,
    "sum_insured": sum_insured,
    "occupancy": occupancy,
    "prior_claims": prior_claims,
    "policy_tenure": policy_tenure,
    # Enrichment
    "flood_risk_zone": flood_risk_zone,
    "crime_index": crime_index,
    "distance_fire_station_km": distance_fire_station_km,
    "annual_rainfall_mm": annual_rainfall_mm,
    "subsidence_risk": subsidence_risk,
    # Target
    "num_claims": num_claims,
    "claim_severity": claim_severity,
    "total_loss": total_loss,
})

# One-hot encode categoricals
df_encoded = pd.get_dummies(df, columns=["property_type", "construction", "occupancy"], drop_first=True)

print(f"Portfolio: {N:,} policies")
print(f"Claim rate: {(num_claims > 0).mean():.1%}")
print(f"Average frequency: {num_claims.mean():.3f}")
print(f"Average severity (claimants): £{claim_severity[num_claims > 0].mean():,.0f}")
display(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define Feature Sets

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Frequency GLMs (Poisson)

# COMMAND ----------

def fit_poisson_glm(train, test, features, label="num_claims"):
    X_train = sm.add_constant(train[features].astype(float))
    X_test = sm.add_constant(test[features].astype(float))
    y_train = train[label]
    y_test = test[label]

    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    results = {
        "model": model,
        "aic": model.aic,
        "bic": model.bic,
        "deviance": model.deviance,
        "null_deviance": model.null_deviance,
        "deviance_explained": 1 - model.deviance / model.null_deviance,
        "mae_test": mean_absolute_error(y_test, pred_test),
        "rmse_test": np.sqrt(mean_squared_error(y_test, pred_test)),
        "pred_test": pred_test,
        "y_test": y_test,
    }
    return results

print("Training Model 1 (Standard features)...")
m1_freq = fit_poisson_glm(train_df, test_df, standard_features)

print("Training Model 2 (Enriched features)...")
m2_freq = fit_poisson_glm(train_df, test_df, enriched_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Frequency Model Comparison

# COMMAND ----------

comparison = pd.DataFrame({
    "Metric": ["AIC", "BIC", "Deviance", "Null Deviance", "Deviance Explained (%)", "MAE (test)", "RMSE (test)"],
    "Model 1 — Standard": [
        f"{m1_freq['aic']:,.1f}", f"{m1_freq['bic']:,.1f}",
        f"{m1_freq['deviance']:,.1f}", f"{m1_freq['null_deviance']:,.1f}",
        f"{m1_freq['deviance_explained']:.2%}",
        f"{m1_freq['mae_test']:.4f}", f"{m1_freq['rmse_test']:.4f}",
    ],
    "Model 2 — Enriched": [
        f"{m2_freq['aic']:,.1f}", f"{m2_freq['bic']:,.1f}",
        f"{m2_freq['deviance']:,.1f}", f"{m2_freq['null_deviance']:,.1f}",
        f"{m2_freq['deviance_explained']:.2%}",
        f"{m2_freq['mae_test']:.4f}", f"{m2_freq['rmse_test']:.4f}",
    ],
})

spark_comp = spark.createDataFrame(comparison)
display(spark_comp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Gini Coefficient (Frequency)

# COMMAND ----------

def gini_coefficient(y_true, y_pred):
    """Ordered Lorenz / Gini for model discrimination."""
    arr = np.array(sorted(zip(y_pred, y_true), key=lambda x: x[0]))
    cum_actual = np.cumsum(arr[:, 1])
    cum_actual_norm = cum_actual / cum_actual[-1]
    n = len(y_true)
    lorenz = cum_actual_norm.sum() / n
    return 2 * lorenz - 1

gini_m1 = gini_coefficient(m1_freq["y_test"].values, m1_freq["pred_test"].values)
gini_m2 = gini_coefficient(m2_freq["y_test"].values, m2_freq["pred_test"].values)

print(f"Gini — Model 1 (Standard):  {gini_m1:.4f}")
print(f"Gini — Model 2 (Enriched):  {gini_m2:.4f}")
print(f"Gini uplift:                 {(gini_m2 - gini_m1) / abs(gini_m1):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Lift Charts

# COMMAND ----------

def plot_lift_chart(y_true, y_pred, n_bins=10, label="Model"):
    """Double lift chart — predicted vs actual by decile."""
    df_lift = pd.DataFrame({"actual": y_true.values, "predicted": y_pred.values})
    df_lift["decile"] = pd.qcut(df_lift["predicted"], n_bins, labels=False, duplicates="drop")
    grouped = df_lift.groupby("decile").agg(
        avg_predicted=("predicted", "mean"),
        avg_actual=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()
    return grouped

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, freq, title in [
    (axes[0], m1_freq, "Model 1 — Standard"),
    (axes[1], m2_freq, "Model 2 — Enriched"),
]:
    lift = plot_lift_chart(freq["y_test"], freq["pred_test"], label=title)
    ax.bar(lift["decile"] - 0.15, lift["avg_actual"], width=0.3, label="Actual", alpha=0.7)
    ax.bar(lift["decile"] + 0.15, lift["avg_predicted"], width=0.3, label="Predicted", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Risk Decile (low → high)")
    ax.set_ylabel("Average Claim Frequency")
    ax.legend()

plt.suptitle("Lift Chart — Predicted vs Actual Frequency by Decile", fontsize=13)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Generate Quotes & Compare Pricing

# COMMAND ----------

# Pure premium = predicted frequency × predicted severity
# For simplicity, use a flat average severity and vary only by frequency model

avg_severity = df.loc[df["num_claims"] > 0, "claim_severity"].mean()
expense_load = 1.35  # 35% loading for expenses + profit

test_df = test_df.copy()
test_df["quote_m1"] = np.round(m1_freq["pred_test"] * avg_severity * expense_load, 2)
test_df["quote_m2"] = np.round(m2_freq["pred_test"] * avg_severity * expense_load, 2)
test_df["actual_loss"] = test_df["total_loss"]

# Show sample quotes
sample = test_df[["building_age", "bedrooms", "sum_insured", "flood_risk_zone",
                   "subsidence_risk", "quote_m1", "quote_m2", "actual_loss"]].head(20)
display(spark.createDataFrame(sample))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Pricing Accuracy: Actual vs Quoted Loss Ratios by Decile

# COMMAND ----------

for model_name, quote_col in [("Model 1 — Standard", "quote_m1"), ("Model 2 — Enriched", "quote_m2")]:
    test_df["decile"] = pd.qcut(test_df[quote_col], 10, labels=False, duplicates="drop")
    lr_by_decile = test_df.groupby("decile").agg(
        total_premium=(quote_col, "sum"),
        total_loss=("actual_loss", "sum"),
    )
    lr_by_decile["loss_ratio"] = lr_by_decile["total_loss"] / lr_by_decile["total_premium"]
    print(f"\n{model_name} — Loss Ratio by Quote Decile:")
    print(lr_by_decile[["loss_ratio"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Summary Coefficient Table

# COMMAND ----------

coef_m1 = m1_freq["model"].summary2().tables[1].reset_index()
coef_m1.columns = ["Feature", "Coef", "StdErr", "z", "P>|z|", "CI_low", "CI_high"]
coef_m1["Model"] = "Standard"

coef_m2 = m2_freq["model"].summary2().tables[1].reset_index()
coef_m2.columns = ["Feature", "Coef", "StdErr", "z", "P>|z|", "CI_low", "CI_high"]
coef_m2["Model"] = "Enriched"

coef_all = pd.concat([coef_m1, coef_m2])
display(spark.createDataFrame(coef_all))

# COMMAND ----------

# MAGIC %md
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
