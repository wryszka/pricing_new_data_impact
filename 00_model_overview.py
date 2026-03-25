# Databricks notebook source
# MAGIC %md
# MAGIC # Model Overview — New Data Impact on Insurance Pricing
# MAGIC
# MAGIC This notebook is the **starting point** for the project. It documents the modelling
# MAGIC approach, data, artefacts, and how to navigate the demo. No code is executed.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Quick Start
# MAGIC
# MAGIC | Step | What to do |
# MAGIC |---|---|
# MAGIC | **1. Build everything** | Run **01_build_all_models** once. This generates the data, trains all models, and persists everything to Unity Catalog. Takes ~3 minutes. |
# MAGIC | **2. Pick your audience** | Open **02** (technical) or **03** (executive). Both read from the same UC tables — no retraining needed. |
# MAGIC | **3. Governance & review** | Open **04** for the audit report, PDF export, and interactive AI review agent. |
# MAGIC | **4. Self-service** | Use the **Lakeview dashboard** or **Genie room** for ad-hoc exploration. |

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
# MAGIC ## What Gets Built
# MAGIC
# MAGIC Notebook 01 builds **everything** in a single run:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  1. Data Generation          50,000 synthetic policies      │
# MAGIC │  2. Train/Test Split         70/30                          │
# MAGIC │  3. Frequency GLMs           Standard vs Enriched (Poisson) │
# MAGIC │  4. Severity GBMs            Standard vs Enriched (Gamma)   │
# MAGIC │  5. Model Factory            50 GLM specifications ranked   │
# MAGIC │  6. Full Quotes              Freq × Sev × Expense Load      │
# MAGIC │  7. Persist to UC            15 tables + 2 MLflow models    │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Type 1: Poisson GLM — Claims Frequency
# MAGIC
# MAGIC Both frequency models are **Poisson Generalised Linear Models (GLMs)** with a **log link**.
# MAGIC
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | **Distribution** | Poisson |
# MAGIC | **Link function** | Log (ensures predictions are always ≥ 0) |
# MAGIC | **Target variable** | `num_claims` — integer count of claims per policy |
# MAGIC | **Implementation** | `statsmodels.GLM` — actuarial-grade outputs with p-values, CIs |
# MAGIC | **Training data** | 35,000 policies (70% of 50,000 synthetic portfolio) |
# MAGIC | **Test data** | 15,000 policies (30%) |
# MAGIC
# MAGIC ### Why Poisson GLM?
# MAGIC
# MAGIC - **Industry standard** — regulators and actuaries expect GLMs for pricing transparency
# MAGIC - **Interpretable coefficients** — each feature has a multiplicative effect on frequency
# MAGIC - **Statistical rigour** — p-values, confidence intervals, AIC/BIC for model selection
# MAGIC - **Log link** — naturally handles non-negative, right-skewed claim counts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Type 2: LightGBM — Claims Severity
# MAGIC
# MAGIC The severity component predicts **how much a claim will cost**, conditional on a claim
# MAGIC having occurred.
# MAGIC
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | **Algorithm** | LightGBM (Gradient Boosted Machine) |
# MAGIC | **Objective** | Gamma — for strictly positive, right-skewed claim costs |
# MAGIC | **Metric** | Gamma deviance |
# MAGIC | **Target variable** | `claim_severity` — cost per claim (£) |
# MAGIC | **Training population** | Claimants only (policies with `num_claims > 0`) |
# MAGIC | **Early stopping** | 50 rounds patience on validation set |
# MAGIC
# MAGIC ### Why GBM for Severity?
# MAGIC
# MAGIC - **Non-linear interactions** — flood + subsidence together is disproportionately expensive
# MAGIC - **Handles heterogeneity** — burst pipe (£2k) vs subsidence (£50k) in the same model
# MAGIC - **Complementary to GLM** — GLM for frequency (transparent) + GBM for severity (flexible)
# MAGIC   is a common actuarial pattern
# MAGIC
# MAGIC ### GBM Hyperparameters
# MAGIC
# MAGIC | Parameter | Value | Rationale |
# MAGIC |---|---|---|
# MAGIC | `learning_rate` | 0.05 | Conservative; avoids overfitting on small claimant population |
# MAGIC | `num_leaves` | 31 | Default; balanced complexity |
# MAGIC | `min_child_samples` | 50 | Prevents overfitting on rare segments |
# MAGIC | `subsample` | 0.8 | Row sampling for regularisation |
# MAGIC | `colsample_bytree` | 0.8 | Feature sampling for regularisation |
# MAGIC | `num_boost_round` | Up to 500 | Capped; early stopping triggers before this |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Factory — 50 GLM Specifications
# MAGIC
# MAGIC Instead of manually configuring model variants one at a time (as in WTW Radar, Emblem,
# MAGIC or Earnix), notebook 01 **programmatically generates and trains 50 GLM specifications**:
# MAGIC
# MAGIC | Search dimension | Variants |
# MAGIC |---|---|
# MAGIC | Enrichment feature subsets | All 31 combinations of 5 features |
# MAGIC | Interaction terms | 6 actuarially meaningful pairings (flood×construction, subsidence×age, etc.) |
# MAGIC | Interaction combinations | Pairs of interaction terms |
# MAGIC | Base feature variations | Dropping low-importance standard features |
# MAGIC | Kitchen sink | All features + all interactions |
# MAGIC
# MAGIC All 50 models are ranked by AIC, BIC, and Gini. The results power the elbow chart,
# MAGIC feature impact analysis, and governance report.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined Pricing Formula
# MAGIC
# MAGIC > **Quote = Predicted Frequency × Predicted Severity × Expense Load (1.35)**
# MAGIC
# MAGIC This produces a full **burning-cost premium** with risk-differentiated severity,
# MAGIC rather than the flat portfolio-average severity used in simpler approaches.

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
# MAGIC Severity also depends on enrichment features:
# MAGIC
# MAGIC ```
# MAGIC log(claim_severity) =
# MAGIC     7.5
# MAGIC   + 0.15 × (flood_risk_zone − 1) / 3
# MAGIC   + 0.1  × subsidence_risk
# MAGIC   + 0.00001 × sum_insured / 1000
# MAGIC   + noise ~ N(0, 0.3)
# MAGIC ```
# MAGIC
# MAGIC Because Model 1 cannot see the enrichment features, it **systematically misprices**
# MAGIC policies where those features matter most.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Metrics
# MAGIC
# MAGIC | Metric | What it measures | Good direction | Used for |
# MAGIC |---|---|---|---|
# MAGIC | **AIC** | Model fit penalised for complexity | Lower | Frequency (model factory ranking) |
# MAGIC | **BIC** | Like AIC with stricter complexity penalty | Lower | Frequency |
# MAGIC | **Deviance Explained** | % of variance captured by the model | Higher | Frequency |
# MAGIC | **Gini Coefficient** | Risk discrimination — rank ordering ability | Higher | Both |
# MAGIC | **MAE** | Average absolute prediction error | Lower | Both |
# MAGIC | **RMSE** | Like MAE but penalises large errors more | Lower | Both |
# MAGIC | **MAPE** | Prediction error as a percentage | Lower | Severity |
# MAGIC | **Bias** | Systematic over/under-prediction | Closer to 0 | Severity |
# MAGIC | **Loss Ratio by Decile** | Premium adequacy across risk segments | Stable ~1.0 | Both |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Artefacts
# MAGIC
# MAGIC All outputs are persisted to `lr_serverless_aws_us_catalog.pricing_new_data_impact`:
# MAGIC
# MAGIC ### Data Tables
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Raw portfolio (50k policies) | `.portfolio` | 01 |
# MAGIC | Training set (35k, one-hot encoded) | `.train_set` | 01 |
# MAGIC | Test set (15k, one-hot encoded) | `.test_set` | 01 |
# MAGIC | Severity training set (claimants only) | `.severity_train_set` | 01 |
# MAGIC | Severity test set (claimants only) | `.severity_test_set` | 01 |
# MAGIC
# MAGIC ### Frequency Model Outputs
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Side-by-side metric comparison | `.model_comparison` | 01 |
# MAGIC | GLM coefficients (both models) | `.glm_coefficients` | 01 |
# MAGIC | Loss ratios by decile | `.loss_ratio_by_decile` | 01 |
# MAGIC | Priced portfolio (freq-only quotes) | `.priced_portfolio` | 01 |
# MAGIC
# MAGIC ### Severity Model Outputs
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Side-by-side metric comparison | `.severity_model_comparison` | 01 |
# MAGIC | Feature importance (both models) | `.severity_feature_importance` | 01 |
# MAGIC | Full priced portfolio (freq × sev quotes) | `.severity_priced_portfolio` | 01 |
# MAGIC | Loss ratios by decile (full quotes) | `.severity_loss_ratio_by_decile` | 01 |
# MAGIC
# MAGIC ### Model Factory Outputs
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | All 50 GLM results ranked | `.model_factory_results` | 01 |
# MAGIC | Feature impact analysis | `.model_factory_feature_impact` | 01 |
# MAGIC
# MAGIC ### Governance Outputs
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Governance summary | `.model_governance_summary` | 04 |
# MAGIC | PDF report | Volume: `.reports/model_governance_report_<date>.pdf` | 04 |
# MAGIC
# MAGIC ### Registered MLflow Models
# MAGIC
# MAGIC | Model | UC Path | Type |
# MAGIC |---|---|---|
# MAGIC | Frequency GLM — Standard | `lr_serverless_aws_us_catalog.pricing_new_data_impact.glm_frequency_standard` | Poisson GLM (pyfunc) |
# MAGIC | Frequency GLM — Enriched | `lr_serverless_aws_us_catalog.pricing_new_data_impact.glm_frequency_enriched` | Poisson GLM (pyfunc) |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Guide
# MAGIC
# MAGIC | # | Notebook | Audience | What it does | Run order |
# MAGIC |---|---|---|---|---|
# MAGIC | **00** | `model_overview` | Everyone | This notebook — documentation and run guide | Read anytime |
# MAGIC | **01** | `build_all_models` | Run once | Generates data, trains freq GLMs + sev GBMs + 50-model factory, persists all to UC | **Run first** |
# MAGIC | **02** | `results_technical` | Data scientists, actuaries | Full technical walkthrough — metrics, coefficients, feature importance, model factory charts, model serving | After 01 |
# MAGIC | **03** | `results_executive` | Business stakeholders | Plain-English walkthrough — same data, no jargon, with glossary | After 01 |
# MAGIC | **04** | `model_governance` | Governance / regulatory | Model governance report with PDF export to UC volume | After 01 |
# MAGIC | **05** | `model_review_agent` | Actuaries | Interactive AI agent for model Q&A — powered by Foundation Model API | After 01 |
# MAGIC
# MAGIC ### Additional Assets
# MAGIC
# MAGIC | Asset | Type | Description |
# MAGIC |---|---|---|
# MAGIC | **Lakeview Dashboard** | AI/BI Dashboard | Interactive exploration of model comparison, loss ratios, and pricing impact |
# MAGIC | **Genie Room** | AI/BI Genie | Natural language Q&A over the model factory and pricing tables |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Flow Suggestions
# MAGIC
# MAGIC ### For a technical audience (actuaries, data scientists)
# MAGIC 1. Start with **00** (this notebook) for context
# MAGIC 2. Walk through **01** to show the pipeline
# MAGIC 3. Open **02** for the deep dive — coefficients, feature importance, model factory elbow chart
# MAGIC 4. Show **04** for governance — the PDF report
# MAGIC 5. Demo **05** — the interactive review agent
# MAGIC 6. End with the **Genie room** — "ask it anything about the models"
# MAGIC
# MAGIC ### For a business audience (underwriters, executives)
# MAGIC 1. Start with **03** — the full story in plain English
# MAGIC 2. Show the **Lakeview dashboard** for interactive exploration
# MAGIC 3. Optionally show the **Genie room** for self-service
# MAGIC
# MAGIC ### For a governance/regulatory audience
# MAGIC 1. Start with **00** for the technical specification
# MAGIC 2. Open **04** — walk through the governance report and generate the PDF
# MAGIC 3. Download the PDF from Catalog Explorer > Volumes > reports
# MAGIC 4. Open **05** — demo the review agent answering "what if" questions
