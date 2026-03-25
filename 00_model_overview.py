# Databricks notebook source
# MAGIC %md
# MAGIC # Model Overview — New Data Impact on Insurance Pricing
# MAGIC
# MAGIC This notebook provides a reference guide to the modelling approach, data, and artefacts
# MAGIC used in this project. No code is executed — this is documentation only.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Project Goal
# MAGIC
# MAGIC Demonstrate that enriching a home insurance pricing model with additional external data
# MAGIC sources leads to measurably better risk segmentation, pricing accuracy, and loss-ratio
# MAGIC stability.
# MAGIC
# MAGIC We compare two models trained on the **same portfolio** but with **different feature sets**
# MAGIC to isolate the impact of the new data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Type: Poisson GLM — Claims Frequency
# MAGIC
# MAGIC Both models are **Poisson Generalised Linear Models (GLMs)** with a **log link function**.
# MAGIC
# MAGIC This is the standard actuarial approach for modelling **claims frequency** — predicting
# MAGIC how many claims a policy will generate in a given period.
# MAGIC
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | **Distribution** | Poisson |
# MAGIC | **Link function** | Log (ensures predictions are always ≥ 0) |
# MAGIC | **Target variable** | `num_claims` — integer count of claims per policy |
# MAGIC | **Implementation** | `statsmodels.GLM` (not sklearn) — gives actuarial-grade outputs |
# MAGIC | **Training data** | 35,000 policies (70% of 50,000 synthetic portfolio) |
# MAGIC | **Test data** | 15,000 policies (30%) |
# MAGIC
# MAGIC ### Why Poisson GLM?
# MAGIC
# MAGIC - **Industry standard** — regulators and actuaries expect GLMs for pricing transparency
# MAGIC - **Interpretable coefficients** — each feature has a multiplicative effect on frequency
# MAGIC   (e.g., flood zone 4 multiplies expected claims by e^coefficient)
# MAGIC - **Statistical rigour** — p-values, confidence intervals, AIC/BIC for model selection
# MAGIC - **Log link** — naturally handles the non-negative, right-skewed nature of claim counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Sets
# MAGIC
# MAGIC ### Model 1 — Standard (12 features)
# MAGIC
# MAGIC Traditional rating factors that any insurer would have:
# MAGIC
# MAGIC | Feature | Type | Description |
# MAGIC |---|---|---|
# MAGIC | `building_age` | Continuous | Years since construction (2025 − year_built) |
# MAGIC | `bedrooms` | Discrete | Number of bedrooms (1–5) |
# MAGIC | `sum_insured` | Continuous | Rebuild cost (£), log-normal distributed |
# MAGIC | `prior_claims` | Discrete | Historical claim count (Poisson λ=0.15) |
# MAGIC | `policy_tenure` | Discrete | Years as customer (0–14) |
# MAGIC | `property_type_*` | Binary | One-hot: flat, semi-detached, terraced (detached = baseline) |
# MAGIC | `construction_*` | Binary | One-hot: other, stone, timber (brick = baseline) |
# MAGIC | `occupancy_tenant` | Binary | 1 = tenant, 0 = owner-occupied |
# MAGIC
# MAGIC ### Model 2 — Enriched (17 features)
# MAGIC
# MAGIC Everything in Model 1, **plus** external/geospatial enrichment data:
# MAGIC
# MAGIC | Feature | Type | Description |
# MAGIC |---|---|---|
# MAGIC | `flood_risk_zone` | Ordinal (1–4) | EA-style flood zone: 1 = low, 4 = high |
# MAGIC | `crime_index` | Continuous (0–100) | Neighbourhood crime score (Beta distribution) |
# MAGIC | `distance_fire_station_km` | Continuous | Distance to nearest fire station (Exponential) |
# MAGIC | `annual_rainfall_mm` | Continuous | Local annual rainfall (Normal, clipped 300–1600) |
# MAGIC | `subsidence_risk` | Binary (0/1) | Ground subsidence indicator (15% prevalence) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data-Generating Process
# MAGIC
# MAGIC The synthetic data is constructed so that the **true risk depends on ALL features**,
# MAGIC including the enrichment variables. This is the key design choice:
# MAGIC
# MAGIC ```
# MAGIC log(expected_claims) =
# MAGIC     −2.5                                          (baseline)
# MAGIC   + property_type effect                          (−0.1 to +0.1)
# MAGIC   + construction effect                           (−0.1 to +0.2)
# MAGIC   + occupancy effect                              (−0.05 to +0.1)
# MAGIC   + 0.003 × building_age
# MAGIC   + 0.05  × prior_claims
# MAGIC   − 0.01  × policy_tenure
# MAGIC   + 0.25  × (flood_risk_zone − 1) / 3            ← hidden from Model 1
# MAGIC   + 0.005 × crime_index                           ← hidden from Model 1
# MAGIC   + 0.02  × I(distance_fire_station > 5km)        ← hidden from Model 1
# MAGIC   + 0.0003 × (annual_rainfall − 800)              ← hidden from Model 1
# MAGIC   + 0.3   × subsidence_risk                       ← hidden from Model 1
# MAGIC ```
# MAGIC
# MAGIC Because Model 1 cannot see the enrichment features, it **systematically misprices**
# MAGIC policies where those features matter most — high flood zones, subsidence areas, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pricing Pipeline
# MAGIC
# MAGIC The notebooks implement a standard actuarial pricing pipeline:
# MAGIC
# MAGIC ```
# MAGIC 1. Data Generation     → 50,000 synthetic home policies
# MAGIC 2. Train/Test Split    → 70/30 stratified split
# MAGIC 3. Frequency GLM       → Poisson GLM for claim counts (this project)
# MAGIC 4. Quote Generation    → predicted_frequency × avg_severity × expense_load (1.35)
# MAGIC 5. Evaluation          → AIC, BIC, Gini, MAE, RMSE, lift charts, loss ratios
# MAGIC 6. Persistence         → All artefacts saved to Unity Catalog + MLflow
# MAGIC ```
# MAGIC
# MAGIC ### Note on Severity
# MAGIC
# MAGIC This project currently models **frequency only**. Severity (claim cost given a claim
# MAGIC has occurred) is approximated using the portfolio average. A dedicated severity model
# MAGIC would be the natural next step — see the discussion below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics
# MAGIC
# MAGIC | Metric | What it measures | Good direction |
# MAGIC |---|---|---|
# MAGIC | **AIC** | Model fit penalised for complexity | Lower is better |
# MAGIC | **BIC** | Like AIC with stricter complexity penalty | Lower is better |
# MAGIC | **Deviance Explained** | % of variance in claims captured by the model | Higher is better |
# MAGIC | **Gini Coefficient** | Risk discrimination — ability to rank policies by risk | Higher is better |
# MAGIC | **MAE** | Average absolute prediction error | Lower is better |
# MAGIC | **RMSE** | Like MAE but penalises large errors more | Lower is better |
# MAGIC | **Loss Ratio by Decile** | Premium adequacy across risk segments | Stable around 1.0 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Artefacts
# MAGIC
# MAGIC All outputs are persisted to `lr_serverless_aws_us_catalog.pricing_new_data_impact`:
# MAGIC
# MAGIC | Artefact | Table / Model |
# MAGIC |---|---|
# MAGIC | Raw portfolio | `.portfolio` |
# MAGIC | Training set | `.train_set` |
# MAGIC | Test set | `.test_set` |
# MAGIC | Priced portfolio (with quotes) | `.priced_portfolio` |
# MAGIC | Metric comparison | `.model_comparison` |
# MAGIC | Loss ratios by decile | `.loss_ratio_by_decile` |
# MAGIC | GLM coefficients | `.glm_coefficients` |
# MAGIC | Frequency model — Standard | MLflow: `.glm_frequency_standard` |
# MAGIC | Frequency model — Enriched | MLflow: `.glm_frequency_enriched` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Guide
# MAGIC
# MAGIC | Notebook | Audience | Description |
# MAGIC |---|---|---|
# MAGIC | **00_model_overview** | Everyone | This notebook — project and model documentation |
# MAGIC | **01_new_data_impact_demo** | Data scientists | Generates data, trains both GLMs, persists all artefacts |
# MAGIC | **02_demo_run_standard** | Data scientists | Walkthrough of results with charts and coefficient analysis |
# MAGIC | **03_demo_run_eli5** | Non-technical stakeholders | Same results, explained in plain English with glossary |
