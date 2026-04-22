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
# MAGIC | **1. Build the enrichment table** | Run **00a_build_postcode_enrichment** once. Builds the `postcode_enrichment` table from ONSPD + IMD 2019. ~7 min first run (downloads 2.3 GB ONSPD), ~3 min subsequent runs. Only needs to be rerun if the source files change. |
# MAGIC | **2. Build everything** | Run **01_build_all_models** once. Samples real postcodes, trains all models, and persists everything to Unity Catalog. Takes ~10 minutes. |
# MAGIC | **3. Pick your audience** | Open **02** (technical) or **03** (executive). Both read from the same UC tables — no retraining needed. |
# MAGIC | **4. Governance & review** | Open **04** for the audit report, PDF export, and interactive AI review agent. |
# MAGIC | **5. Self-service** | Use the **Lakeview dashboard** or **Genie room** for ad-hoc exploration. |
# MAGIC
# MAGIC **Geographic scope:** The enrichment table covers **England only**. Scotland uses SIMD,
# MAGIC Wales uses WIMD, and Northern Ireland uses NIMDM — these deprivation indices are not
# MAGIC directly comparable to the English IMD and are not included in this demo.

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
# MAGIC The pipeline runs in two stages:
# MAGIC
# MAGIC ### Stage 1 — Notebook 00a (one-off, reusable)
# MAGIC
# MAGIC Builds `postcode_enrichment` — a single wide table keyed on UK postcode, containing real
# MAGIC public data for every English postcode: IMD 2019 deciles (overall, crime, income, health,
# MAGIC living environment), ONS Rural-Urban Classification, coastal flag derived from coastal
# MAGIC local authorities, region name, LSOA code, and local authority code. Sourced from:
# MAGIC
# MAGIC - **ONSPD** — Office for National Statistics Postcode Directory (lat/long, LSOA, LA, region)
# MAGIC - **IMD 2019** — Ministry of Housing, Communities & Local Government English Indices of Deprivation
# MAGIC - **ONS RUC 2011** — Rural-Urban Classification of Output Areas
# MAGIC
# MAGIC ### Stage 2 — Notebook 01 (pricing pipeline)
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │  1. Sample Postcodes         200k real English postcodes    │
# MAGIC │  2. Synthetic Rating Factors Property type, age, sum ins.   │
# MAGIC │  3. Simulate Claims          Poisson freq + Gamma sev       │
# MAGIC │  4. Train/Test Split         70/30                          │
# MAGIC │  5. Frequency GLMs           Standard vs Enriched (Poisson) │
# MAGIC │  6. Severity GBMs            Standard vs Enriched (Gamma)   │
# MAGIC │  7. Model Factory            50 GLM specifications ranked   │
# MAGIC │  8. Full Quotes              Freq × Sev × Expense Load      │
# MAGIC │  9. Persist to UC            15 tables + 2 MLflow models    │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC Each demo policy is assigned a real English postcode and inherits its real
# MAGIC enrichment features. Only the property-level rating factors (construction, bedrooms,
# MAGIC sum insured) are synthetic — in a real deployment they come from the insurer's book.

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
# MAGIC | **Training data** | 140,000 policies (70% of the 200,000 sampled portfolio) |
# MAGIC | **Test data** | 60,000 policies (30%) |
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
# MAGIC - **Non-linear interactions** — high deprivation + coastal location is disproportionately expensive
# MAGIC - **Handles heterogeneity** — small theft (£500) vs major escape-of-water (£20k+) in the same model
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
# MAGIC | Enrichment feature subsets | Singles, pairs and triples of 6 enrichment features (IMD, crime, income, health, urban, coastal) |
# MAGIC | Region dummies | With / without the 8-region-dummy group |
# MAGIC | Interaction terms | 6 actuarially meaningful pairings (crime×urban, imd×coastal, coastal×sum_insured, etc.) |
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
# MAGIC ### Model 2 — Enriched (standard + real UK public data)
# MAGIC
# MAGIC Everything in Model 1, **plus** real geospatial enrichment joined from the
# MAGIC `postcode_enrichment` table on each policy's postcode:
# MAGIC
# MAGIC | Feature | Type | Source | Description |
# MAGIC |---|---|---|---|
# MAGIC | `imd_decile` | Ordinal (1–10) | IMD 2019 | Overall deprivation decile (1 = most deprived, 10 = least) |
# MAGIC | `imd_score` | Continuous | IMD 2019 | Raw IMD score (higher = more deprived) |
# MAGIC | `crime_decile` | Ordinal (1–10) | IMD 2019 | Crime deprivation decile (1 = highest crime) |
# MAGIC | `income_decile` | Ordinal (1–10) | IMD 2019 | Income deprivation decile (1 = most income-deprived) |
# MAGIC | `health_decile` | Ordinal (1–10) | IMD 2019 | Health deprivation decile (1 = most health-deprived) |
# MAGIC | `living_env_decile` | Ordinal (1–10) | IMD 2019 | Living environment decile (1 = worst) |
# MAGIC | `is_urban` | Binary (0/1) | ONS RUC 2011 | Urban/rural classification |
# MAGIC | `is_coastal` | Binary (0/1) | Derived | Derived from coastal local authorities |
# MAGIC | `region_*` | Binary | ONSPD | One-hot encoded region dummies (8 dummies; East Midlands = baseline) |
# MAGIC
# MAGIC The portfolio table also carries identifier columns — `postcode`, `lat`, `long`,
# MAGIC `lsoa_code`, `local_authority_code`, `urban_rural_band` — for downstream geospatial
# MAGIC analysis and joining to other datasets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data-Generating Process
# MAGIC
# MAGIC Claims are simulated so the **true risk depends on ALL features**, including the real
# MAGIC enrichment variables attached to each postcode. The DGP is calibrated to reproduce
# MAGIC published UK home-insurance market statistics (~15% claim rate, ~£2,600 average
# MAGIC severity, ABI market data).
# MAGIC
# MAGIC ```
# MAGIC log(expected_claims) =
# MAGIC     −2.95                                         (baseline, tunes overall freq to ~15%)
# MAGIC   + property_type effect                          (−0.10 to +0.10)
# MAGIC   + construction effect                           (−0.10 to +0.20)
# MAGIC   + occupancy effect                              (−0.05 to +0.10)
# MAGIC   + 0.003 × building_age
# MAGIC   + 0.05  × prior_claims
# MAGIC   − 0.01  × policy_tenure
# MAGIC   + region_effect                                 (−0.02 to +0.10; London highest)
# MAGIC   + 0.80  × crime_inv                             ← hidden from Model 1
# MAGIC   + 0.45  × imd_inv                               ← hidden from Model 1
# MAGIC   + 0.25  × living_inv                            ← hidden from Model 1
# MAGIC   + 0.30  × is_coastal                            ← hidden from Model 1
# MAGIC   + 0.12  × is_urban                              ← hidden from Model 1
# MAGIC
# MAGIC   where  crime_inv   = (11 − crime_decile) / 10     (0..1, 1 = highest crime)
# MAGIC          imd_inv     = (11 − imd_decile) / 10       (0..1, 1 = most deprived)
# MAGIC          living_inv  = (11 − living_env_decile) / 10
# MAGIC ```
# MAGIC
# MAGIC Severity also depends on the real enrichment features:
# MAGIC
# MAGIC ```
# MAGIC log(claim_severity) =
# MAGIC     7.5
# MAGIC   + 0.35 × is_coastal                             (water ingress, salt damage)
# MAGIC   + 0.30 × imd_inv                                (poor maintenance → costlier repairs)
# MAGIC   + 0.15 × crime_inv                              (higher theft claim sizes)
# MAGIC   + 0.00001 × sum_insured / 1000
# MAGIC   + noise ~ N(0, 0.35)
# MAGIC ```
# MAGIC
# MAGIC Because Model 1 cannot see the enrichment features, it **systematically misprices**
# MAGIC policies in high-crime, deprived, or coastal postcodes.

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
# MAGIC ### Enrichment Reference Table
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Real UK postcode enrichment (~1.5M English postcodes) | `.postcode_enrichment` | 00a |
# MAGIC
# MAGIC ### Data Tables
# MAGIC
# MAGIC | Artefact | Table | Created by |
# MAGIC |---|---|---|
# MAGIC | Raw portfolio (200k policies sampled from real postcodes) | `.portfolio` | 01 |
# MAGIC | Training set (140k, one-hot encoded) | `.train_set` | 01 |
# MAGIC | Test set (60k, one-hot encoded) | `.test_set` | 01 |
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
# MAGIC | **00a** | `build_postcode_enrichment` | Run once | Builds `postcode_enrichment` (1.5M English postcodes) from ONSPD + IMD 2019 + RUC 2011. Only rerun when source files change. | **Run first** |
# MAGIC | **01** | `build_all_models` | Run once | Samples real postcodes, trains freq GLMs + sev GBMs + 50-model factory, persists all to UC | **Run second** |
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
