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
# MAGIC 3. Frequency GLM       → Poisson GLM for claim counts
# MAGIC 4. Severity GBM        → LightGBM with Gamma objective for claim costs (claimants only)
# MAGIC 5. Quote Generation    → predicted_frequency × predicted_severity × expense_load (1.35)
# MAGIC 6. Evaluation          → AIC, BIC, Gini, MAE, RMSE, MAPE, lift charts, loss ratios
# MAGIC 7. Persistence         → All artefacts saved to Unity Catalog + MLflow (frequency models)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Severity Model: LightGBM with Gamma Objective
# MAGIC
# MAGIC The severity component predicts **how much a claim will cost**, conditional on a claim
# MAGIC having occurred. It uses the same standard-vs-enriched feature comparison as frequency.
# MAGIC
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | **Algorithm** | LightGBM (Gradient Boosted Machine) |
# MAGIC | **Objective** | Gamma — for strictly positive, right-skewed data |
# MAGIC | **Metric** | Gamma deviance |
# MAGIC | **Target variable** | `claim_severity` — cost per claim (£) |
# MAGIC | **Training population** | Claimants only (policies with `num_claims > 0`) |
# MAGIC | **Early stopping** | 50 rounds patience on validation gamma deviance |
# MAGIC
# MAGIC ### Why GBM for Severity?
# MAGIC
# MAGIC - **Non-linear interactions** — flood zone 4 + subsidence together is disproportionately
# MAGIC   expensive; a GBM captures these interactions naturally without manual feature engineering
# MAGIC - **Handles heterogeneity** — claim costs vary enormously (a burst pipe vs. full subsidence
# MAGIC   repair); GBMs handle this heavy-tailed distribution better than linear models
# MAGIC - **Complementary to GLM** — using GLM for frequency (transparent, regulatory-friendly) and
# MAGIC   GBM for severity (captures complex cost patterns) is a common actuarial approach
# MAGIC
# MAGIC ### GBM Hyperparameters
# MAGIC
# MAGIC | Parameter | Value | Rationale |
# MAGIC |---|---|---|
# MAGIC | `learning_rate` | 0.05 | Conservative; avoids overfitting on relatively small claimant population |
# MAGIC | `num_leaves` | 31 | Default; balanced complexity |
# MAGIC | `min_child_samples` | 50 | Prevents overfitting on rare segments |
# MAGIC | `subsample` | 0.8 | Row sampling for regularisation |
# MAGIC | `colsample_bytree` | 0.8 | Feature sampling for regularisation |
# MAGIC | `num_boost_round` | Up to 500 | Capped; early stopping triggers before this |
# MAGIC
# MAGIC ### Combined Pricing
# MAGIC
# MAGIC The full burning-cost quote combines both models:
# MAGIC
# MAGIC > **Quote = Predicted Frequency × Predicted Severity × Expense Load (1.35)**
# MAGIC
# MAGIC This replaces the flat portfolio-average severity used in notebooks 01–03, giving
# MAGIC risk-differentiated severity that properly prices expensive perils like flood and subsidence.

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
# MAGIC | **MAPE** | Prediction error as a percentage (severity) | Lower is better |
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
# MAGIC | Severity train set | `.severity_train_set` |
# MAGIC | Severity test set | `.severity_test_set` |
# MAGIC | Severity model comparison | `.severity_model_comparison` |
# MAGIC | Severity feature importance | `.severity_feature_importance` |
# MAGIC | Severity priced portfolio (freq×sev) | `.severity_priced_portfolio` |
# MAGIC | Severity loss ratios | `.severity_loss_ratio_by_decile` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Guide
# MAGIC
# MAGIC | Notebook | Audience | Description |
# MAGIC |---|---|---|
# MAGIC | **00_model_overview** | Everyone | This notebook — project and model documentation |
# MAGIC | **01_new_data_impact_demo** | Data scientists | Generates data, trains both frequency GLMs, persists all artefacts |
# MAGIC | **02_demo_run_standard** | Data scientists | Frequency walkthrough with charts and coefficient analysis |
# MAGIC | **03_demo_run_eli5** | Non-technical stakeholders | Frequency results explained in plain English with glossary |
# MAGIC | **04_severity_gbm_demo** | Data scientists | Trains both severity GBMs, generates full freq×sev quotes |
# MAGIC | **05_severity_demo_standard** | Data scientists | Severity walkthrough with feature importance and loss ratios |
# MAGIC | **06_severity_demo_eli5** | Non-technical stakeholders | Severity results explained in plain English with glossary |
