# Databricks notebook source
# MAGIC %md
# MAGIC # Model Review Agent - Interactive Governance Assistant
# MAGIC
# MAGIC This notebook provides an **interactive AI agent** powered by the Databricks Foundation
# MAGIC Model API (Claude Sonnet 4) that actuaries can use to query and review pricing model
# MAGIC results using natural language.
# MAGIC
# MAGIC ## What You Can Ask
# MAGIC
# MAGIC The underlying models were trained on a synthetic portfolio enriched with **real UK public data**
# MAGIC (MHCLG IMD 2019 deprivation deciles, ONSPD postcode directory, ONS 2011 Rural-Urban Classification,
# MAGIC and derived coastal flags). The agent reasons over all 50 GLM specifications and can help you:
# MAGIC
# MAGIC - *"What are the top 5 models by Gini coefficient?"*
# MAGIC - *"What happens if I can't use crime_decile data?"*
# MAGIC - *"Compare the baseline model against the best enriched model"*
# MAGIC - *"Which real-UK enrichment features give the biggest improvement?"*
# MAGIC - *"How does adding is_coastal change model fit?"*
# MAGIC - *"Generate a summary for model governance review"*

# COMMAND ----------

# MAGIC %pip install matplotlib mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data & Build Helper Functions

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow.deployments

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Load all tables the agent will reason over
model_results_df = spark.table("model_factory_results").toPandas()
feature_impact_df = spark.table("model_factory_feature_impact").toPandas()
glm_coefficients_df = spark.table("glm_coefficients").toPandas()

# Load comparison tables if they exist
try:
    model_comparison_df = spark.table("model_comparison").toPandas()
    has_model_comparison = True
except Exception:
    model_comparison_df = pd.DataFrame()
    has_model_comparison = False

try:
    severity_comparison_df = spark.table("severity_model_comparison").toPandas()
    has_severity_comparison = True
except Exception:
    severity_comparison_df = pd.DataFrame()
    has_severity_comparison = False

print(f"model_factory_results        : {len(model_results_df):>4} rows  |  columns: {list(model_results_df.columns)}")
print(f"model_factory_feature_impact : {len(feature_impact_df):>4} rows  |  columns: {list(feature_impact_df.columns)}")
print(f"glm_coefficients             : {len(glm_coefficients_df):>4} rows  |  columns: {list(glm_coefficients_df.columns)}")
print(f"model_comparison             : {len(model_comparison_df):>4} rows  (available={has_model_comparison})")
print(f"severity_comparison          : {len(severity_comparison_df):>4} rows  (available={has_severity_comparison})")

# COMMAND ----------

display(model_results_df.sort_values("rank_aic").head(10))

# COMMAND ----------

display(feature_impact_df.sort_values("avg_aic_improvement", ascending=False))

# COMMAND ----------

def top_models(n: int = 5, metric: str = "aic") -> pd.DataFrame:
    """Return the top N models ranked by the given metric.

    For AIC/BIC/deviance/mae/rmse — lower is better (ascending sort).
    For gini_test — higher is better (descending sort).
    """
    lower_is_better = {"aic", "bic", "deviance_explained", "mae_test", "rmse_test"}
    higher_is_better = {"gini_test"}

    metric_col = metric.lower()
    if metric_col not in model_results_df.columns:
        available = [c for c in model_results_df.columns if c not in ("name", "description")]
        return pd.DataFrame({"error": [f"Unknown metric '{metric}'. Available: {available}"]})

    ascending = metric_col not in higher_is_better
    ranked = model_results_df.sort_values(metric_col, ascending=ascending).head(n)
    cols = ["name", "n_features", "aic", "bic", "gini_test", "deviance_explained", "mae_test", "rmse_test", "description"]
    cols = [c for c in cols if c in ranked.columns]
    return ranked[cols].reset_index(drop=True)


def compare_models(model_a_name: str, model_b_name: str) -> pd.DataFrame:
    """Return a side-by-side comparison of two named models."""
    row_a = model_results_df[model_results_df["name"] == model_a_name]
    row_b = model_results_df[model_results_df["name"] == model_b_name]

    missing = []
    if row_a.empty:
        missing.append(model_a_name)
    if row_b.empty:
        missing.append(model_b_name)
    if missing:
        all_names = model_results_df["name"].tolist()
        return pd.DataFrame({"error": [f"Model(s) not found: {missing}. Available names: {all_names[:10]}..."]})

    metrics = ["n_features", "aic", "bic", "gini_test", "deviance_explained", "mae_test", "rmse_test", "rank_aic", "rank_gini"]
    metrics = [m for m in metrics if m in model_results_df.columns]

    comparison = pd.DataFrame({
        "metric": metrics,
        model_a_name: [row_a.iloc[0][m] for m in metrics],
        model_b_name: [row_b.iloc[0][m] for m in metrics],
    })

    numeric_a = comparison[model_a_name]
    numeric_b = comparison[model_b_name]
    try:
        comparison["difference (B - A)"] = numeric_b.values - numeric_a.values
    except Exception:
        pass

    return comparison


def drop_feature_impact(feature_name: str) -> pd.DataFrame:
    """Show impact summary for models that include vs exclude a given feature."""
    feature_lower = feature_name.lower()

    fi_row = feature_impact_df[feature_impact_df["feature"].str.lower() == feature_lower]

    mask_with = (
        model_results_df["description"].str.lower().str.contains(feature_lower, na=False) |
        model_results_df["name"].str.lower().str.contains(feature_lower, na=False)
    )
    models_with = model_results_df[mask_with]
    models_without = model_results_df[~mask_with]

    summary_rows = []

    if not fi_row.empty:
        summary_rows.append({
            "stat": "avg_aic_improvement_from_feature",
            "value": fi_row.iloc[0]["avg_aic_improvement"],
        })

    summary_rows.append({"stat": "models_including_feature", "value": len(models_with)})
    summary_rows.append({"stat": "models_excluding_feature", "value": len(models_without)})

    if not models_with.empty:
        summary_rows.append({"stat": "mean_aic_WITH_feature",  "value": round(models_with["aic"].mean(), 2)})
        summary_rows.append({"stat": "mean_gini_WITH_feature", "value": round(models_with["gini_test"].mean(), 4)})
    if not models_without.empty:
        summary_rows.append({"stat": "mean_aic_WITHOUT_feature",  "value": round(models_without["aic"].mean(), 2)})
        summary_rows.append({"stat": "mean_gini_WITHOUT_feature", "value": round(models_without["gini_test"].mean(), 4)})

    return pd.DataFrame(summary_rows)


def model_detail(model_name: str) -> dict:
    """Return full detail for a single named model including its GLM coefficients."""
    row = model_results_df[model_results_df["name"] == model_name]
    if row.empty:
        all_names = model_results_df["name"].tolist()
        return {"error": f"Model '{model_name}' not found. Available: {all_names[:10]}..."}

    detail = row.iloc[0].to_dict()

    if "model" in glm_coefficients_df.columns:
        coefs = glm_coefficients_df[glm_coefficients_df["model"] == model_name]
    else:
        coefs = pd.DataFrame()

    detail["coefficients"] = coefs.to_dict(orient="records") if not coefs.empty else []
    return detail

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. System Prompt & Query Agent Function

# COMMAND ----------

def _build_data_summary() -> str:
    """Produce a compact text summary of all loaded data for injection into the prompt."""

    lines = []

    lines.append("=== TABLE: model_factory_results ===")
    lines.append(f"Rows: {len(model_results_df)}  |  Each row is one GLM specification.")
    lines.append("Columns: name, description, n_features, aic, bic, gini_test, deviance_explained, mae_test, rmse_test, rank_aic, rank_gini")
    lines.append("")
    lines.append("Top 10 models by AIC:")
    top10 = model_results_df.sort_values("rank_aic").head(10)
    for _, r in top10.iterrows():
        lines.append(
            f"  [{int(r['rank_aic'])}] {r['name']}  aic={r['aic']:.1f}  bic={r.get('bic', float('nan')):.1f}"
            f"  gini={r['gini_test']:.4f}  n_features={int(r['n_features'])}  — {r['description']}"
        )
    lines.append("")

    lines.append("=== TABLE: model_factory_feature_impact ===")
    lines.append("Columns: feature, avg_aic_improvement  (improvement = AIC drop when feature is added; more negative = better)")
    lines.append("Top features by avg_aic_improvement (most valuable first):")
    top_fi = feature_impact_df.sort_values("avg_aic_improvement").head(15)
    for _, r in top_fi.iterrows():
        lines.append(f"  {r['feature']:<40} avg_aic_improvement={r['avg_aic_improvement']:.2f}")
    lines.append("")

    lines.append("=== TABLE: glm_coefficients ===")
    lines.append("Columns: feature, coef, std_err, z, p_value, ci_low, ci_high, model")
    lines.append(f"Total coefficient rows: {len(glm_coefficients_df)}")
    if not glm_coefficients_df.empty and "model" in glm_coefficients_df.columns:
        models_with_coefs = glm_coefficients_df["model"].unique().tolist()
        lines.append(f"Models with stored coefficients: {models_with_coefs[:10]}")
    lines.append("")

    if has_model_comparison and not model_comparison_df.empty:
        lines.append("=== TABLE: model_comparison ===")
        lines.append(f"Rows: {len(model_comparison_df)}  |  Columns: {list(model_comparison_df.columns)}")
        lines.append(model_comparison_df.to_string(index=False, max_rows=10))
        lines.append("")

    if has_severity_comparison and not severity_comparison_df.empty:
        lines.append("=== TABLE: severity_model_comparison ===")
        lines.append(f"Rows: {len(severity_comparison_df)}  |  Columns: {list(severity_comparison_df.columns)}")
        lines.append(severity_comparison_df.to_string(index=False, max_rows=10))
        lines.append("")

    aic_min = model_results_df["aic"].min()
    aic_max = model_results_df["aic"].max()
    gini_min = model_results_df["gini_test"].min()
    gini_max = model_results_df["gini_test"].max()
    lines.append("=== OVERALL STATISTICS ===")
    lines.append(f"AIC range across all specs: {aic_min:.1f} — {aic_max:.1f}  (improvement = {aic_max - aic_min:.1f})")
    lines.append(f"Gini range: {gini_min:.4f} — {gini_max:.4f}  (improvement = {gini_max - gini_min:.4f})")
    lines.append(f"Number of enrichment features tested: {len(feature_impact_df)}")

    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are an expert actuarial assistant helping insurance pricing actuaries review
and govern Generalised Linear Model (GLM) specifications.

You have access to the results of a model factory that trained and evaluated 50 Poisson GLM
specifications on a home insurance portfolio. The models are frequency models (claim count)
using a log link function — the standard actuarial approach.

## Portfolio & Feature Context

- Policies and claims are **synthetic** (simulated), but each policy is keyed to a real UK
  postcode.
- **Enrichment features are real UK public data**:
  * `imd_decile`, `imd_score` — Index of Multiple Deprivation (MHCLG, 2019) at LSOA level.
    Decile 1 = most deprived, 10 = least. Lower decile typically = higher claim frequency.
  * `crime_decile` — Crime domain of IMD 2019. Strong predictor of theft / malicious damage.
  * `income_decile` — Income deprivation decile (IMD 2019). Correlates with housing quality,
    occupancy patterns, and prior-claim behaviour.
  * `health_decile` — Health & disability deprivation decile (IMD 2019). Weak positive correlate
    of escape-of-water and accidental-damage claims.
  * `living_env_decile` — Living environment deprivation (housing condition, air quality, road
    safety). Strong predictor of maintenance-related claims.
  * `is_urban` — Urban flag from ONS 2011 Rural-Urban Classification (urban = A1/B1/C1/C2 bands).
    Urban properties exhibit higher theft and escape-of-water frequency.
  * `is_coastal` — Derived from coastal English local authorities. Coastal properties show
    elevated weather / water-ingress claims.
  * `region_*` — One-hot ONS 9-region dummies. Captures residual geographic heterogeneity
    after deprivation and urban/coastal controls.
- Standard rating factors also present: `building_age`, `bedrooms`, `sum_insured`, `prior_claims`,
  `policy_tenure`, property_type / construction / occupancy dummies.

## Your Role

- Answer questions about model performance, feature importance, and specification comparisons
- Explain actuarial concepts clearly when asked
- Help actuaries make governance decisions about which model to deploy
- Highlight trade-offs between interpretability, predictive power, and data availability
- Flag regulatory considerations (e.g., deprivation-based features potentially proxying protected
  characteristics under FCA GIPP / Equality Act 2010 / GDPR)

## Metric Guidance

| Metric | Direction | Meaning |
|---|---|---|
| AIC / BIC | Lower is better | Information-theoretic fit; penalises complexity |
| Gini (test set) | Higher is better | Ranking discrimination on holdout data |
| Deviance explained | Higher is better | % of null deviance explained (like R2) |
| MAE / RMSE (test) | Lower is better | Absolute / squared prediction error |

## Data Available

The following tables are loaded and summarised below. Use this data to answer questions precisely.

{data_summary}

## Behavioural Guidelines

- Be concise but complete. Actuaries are time-pressed professionals.
- When recommending a model, justify your recommendation with specific metric values.
- If a question references a feature or model name, match it against the data provided.
- If asked for a governance narrative, write a professional paragraph suitable for a model
  committee or sign-off document.
- Do not fabricate metric values. Only quote numbers present in the data above.
- If the data is insufficient to answer, say so clearly and suggest what additional analysis
  would be needed.
"""


def build_full_system_prompt() -> str:
    data_summary = _build_data_summary()
    return SYSTEM_PROMPT_TEMPLATE.format(data_summary=data_summary)


print("System prompt built successfully.")
print(f"Approximate token count (chars/4): ~{len(build_full_system_prompt()) // 4:,}")

# COMMAND ----------

def query_agent(question: str, max_tokens: int = 1024, verbose: bool = True) -> str:
    """Send a natural-language question to the model review agent.

    Uses the Databricks Foundation Model API (Claude Sonnet 4) with the full
    model factory data injected as context.

    Parameters
    ----------
    question : str
        The actuarial question to ask.
    max_tokens : int
        Maximum tokens in the response (default 1024).
    verbose : bool
        If True, prints the question and response to stdout.

    Returns
    -------
    str
        The agent's response.
    """
    system_prompt = build_full_system_prompt()
    client = mlflow.deployments.get_deploy_client("databricks")

    primary_endpoint = "databricks-claude-sonnet-4"
    fallback_endpoint = "databricks-meta-llama-3-1-70b-instruct"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    answer = None
    endpoint_used = None

    for endpoint in [primary_endpoint, fallback_endpoint]:
        try:
            response = client.predict(
                endpoint=endpoint,
                inputs={"messages": messages, "max_tokens": max_tokens},
            )
            answer = response["choices"][0]["message"]["content"]
            endpoint_used = endpoint
            break
        except Exception as e:
            if endpoint == primary_endpoint:
                print(f"  [info] Primary endpoint '{primary_endpoint}' unavailable ({type(e).__name__}), trying fallback...")
            else:
                raise RuntimeError(
                    f"Both endpoints failed. Last error on '{endpoint}': {e}"
                ) from e

    if verbose:
        print("=" * 72)
        print(f"QUESTION  [{endpoint_used}]")
        print("-" * 72)
        print(question)
        print()
        print("ANSWER")
        print("-" * 72)
        print(answer)
        print("=" * 72)

    return answer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Example Queries
# MAGIC
# MAGIC The cells below demonstrate the range of questions actuaries can ask.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.1 — Top Models by AIC

# COMMAND ----------

print("Helper output — top_models(5, 'aic'):")
display(top_models(5, "aic"))

# COMMAND ----------

_ = query_agent(
    "What are the top 5 models by AIC? Summarise what features they use and whether "
    "the improvement over the baseline is meaningful."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.2 — Feature Unavailability Scenario

# COMMAND ----------

print("Helper output — drop_feature_impact('crime_decile'):")
display(drop_feature_impact("crime_decile"))

# COMMAND ----------

_ = query_agent(
    "What happens if I can't use crime_decile (IMD 2019 crime domain) data? "
    "Which models should I consider instead, and how much predictive power do we lose? "
    "Please give specific AIC and Gini numbers."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.3 — Baseline vs Best Model Comparison

# COMMAND ----------

baseline_name = model_results_df.sort_values("n_features").iloc[0]["name"]
best_aic_name = model_results_df.sort_values("rank_aic").iloc[0]["name"]

print(f"Baseline model (fewest features): {baseline_name}")
print(f"Best model by AIC rank:           {best_aic_name}")

print("\nHelper output — compare_models():")
display(compare_models(baseline_name, best_aic_name))

# COMMAND ----------

_ = query_agent(
    f"Compare the baseline model '{baseline_name}' against the best-performing model '{best_aic_name}'. "
    "What is the improvement in AIC, BIC, and Gini? Is the additional complexity justified for "
    "a pricing model that will be reviewed by a regulator?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.4 — Most Valuable Enrichment Features

# COMMAND ----------

print("Helper output — feature_impact_df (top 10):")
display(feature_impact_df.sort_values("avg_aic_improvement").head(10))

# COMMAND ----------

_ = query_agent(
    "Which real-UK enrichment features (IMD 2019 deciles, ONSPD-derived urban/coastal flags, "
    "ONS region dummies) give the biggest improvement to model fit? "
    "Rank the top 5 and explain what each feature likely represents in the context of home insurance. "
    "Which features would you prioritise for a first-phase data enrichment project, and what "
    "governance considerations (e.g., deprivation as potential proxy for protected characteristics) apply?"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 13.5 — Governance Narrative for Committee

# COMMAND ----------

_ = query_agent(
    "Generate a model governance summary suitable for a pricing committee sign-off document. "
    "It should cover: (1) the modelling approach and data used, (2) the range of specifications tested, "
    "(3) the recommended model with justification, (4) key risks and limitations, and "
    "(5) suggested monitoring metrics post-deployment. "
    "Write in formal, professional language appropriate for a regulatory audience.",
    max_tokens=2048,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interactive Use Guide
# MAGIC
# MAGIC To use the agent interactively, call `query_agent()` with your own question:
# MAGIC
# MAGIC ```python
# MAGIC query_agent("Your question here")
# MAGIC ```
# MAGIC
# MAGIC **Example questions to try:**
# MAGIC
# MAGIC ```python
# MAGIC # Compare two specific models
# MAGIC query_agent("Compare model_spec_12 against model_spec_25 — which has better holdout Gini?")
# MAGIC
# MAGIC # Explore a specific feature
# MAGIC query_agent("How important is imd_decile across the model factory?")
# MAGIC query_agent("Does is_coastal add meaningful lift on top of the region dummies?")
# MAGIC
# MAGIC # Get coefficient details for the best model
# MAGIC best = model_results_df.sort_values("rank_aic").iloc[0]["name"]
# MAGIC detail = model_detail(best)
# MAGIC query_agent(f"Interpret the GLM coefficients for model '{best}': {detail['coefficients'][:5]}")
# MAGIC
# MAGIC # Ask about regulatory risk
# MAGIC query_agent("Are there any features in the top models that might raise regulatory concerns?")
# MAGIC
# MAGIC # Request a data quality narrative
# MAGIC query_agent("Which models are most robust to potential data quality issues in the enrichment features?")
# MAGIC ```
# MAGIC
# MAGIC ## Helper Functions Reference
# MAGIC
# MAGIC | Function | Purpose |
# MAGIC |---|---|
# MAGIC | `top_models(n, metric)` | Top N models by any metric (aic, bic, gini_test, etc.) |
# MAGIC | `compare_models(name_a, name_b)` | Side-by-side metric comparison |
# MAGIC | `drop_feature_impact(feature)` | AIC/Gini impact of excluding a feature |
# MAGIC | `model_detail(name)` | Full spec + coefficients for one model |
# MAGIC | `query_agent(question)` | Ask the AI agent anything |
# MAGIC
# MAGIC The agent has access to all model factory results in its context window, so questions
# MAGIC about specific model names, features, or metrics will be answered using the actual data
# MAGIC rather than general knowledge.

# COMMAND ----------
