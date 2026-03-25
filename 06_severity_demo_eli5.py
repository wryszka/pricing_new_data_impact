# Databricks notebook source
# MAGIC %md
# MAGIC # How Much Will a Claim Cost? Standard vs Enriched Severity Models
# MAGIC
# MAGIC In the previous notebooks, we looked at **how often** claims happen (frequency).
# MAGIC Now we ask the second big question: **when a claim does happen, how much will it cost?**
# MAGIC
# MAGIC This is called **severity modelling**, and it's the other half of insurance pricing.
# MAGIC
# MAGIC We've trained two models:
# MAGIC - **Model 1** uses only the data we've always had (property type, age, size, etc.)
# MAGIC - **Model 2** adds new external data (flood risk, crime, subsidence, etc.)
# MAGIC
# MAGIC **No data-science background required.** Everything is explained in plain English.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is a Severity Model?
# MAGIC
# MAGIC A **severity model** predicts **how expensive a claim will be**, given that one has
# MAGIC already happened. It answers: "If this property makes a claim, how much will we pay out?"
# MAGIC
# MAGIC Combined with the frequency model (which predicts *how often* claims happen), we get
# MAGIC the full picture:
# MAGIC
# MAGIC > **Expected Cost = How Often × How Much**
# MAGIC >
# MAGIC > Or in insurance terms: **Pure Premium = Frequency × Severity**
# MAGIC
# MAGIC This is how insurers calculate the base price before adding expenses and profit margin.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why a GBM Instead of a GLM?
# MAGIC
# MAGIC For frequency, we used a **GLM** (Generalised Linear Model) — the industry standard.
# MAGIC For severity, we're using a **GBM** (Gradient Boosted Machine). Here's why:
# MAGIC
# MAGIC | | GLM | GBM |
# MAGIC |---|---|---|
# MAGIC | **How it works** | One simple formula | Hundreds of small decision trees working together |
# MAGIC | **Strengths** | Transparent, easy to explain to regulators | Captures complex patterns the formula would miss |
# MAGIC | **Best for** | Frequency (simple, well-behaved counts) | Severity (messy, varied claim costs) |
# MAGIC
# MAGIC Claim costs are messy — a burst pipe costs £2,000 but a subsidence claim costs £50,000.
# MAGIC A GBM is better at spotting these complex patterns, especially when they depend on
# MAGIC combinations of factors (e.g., old timber house + flood zone = very expensive).
# MAGIC
# MAGIC We use **LightGBM** with a **Gamma distribution**, which is designed for strictly
# MAGIC positive, right-skewed data — exactly what claim costs look like.

# COMMAND ----------

# MAGIC %pip install matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Claimants

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pyspark.sql.functions as F

train = spark.table(f"{CATALOG}.{SCHEMA}.severity_train_set").toPandas()
test = spark.table(f"{CATALOG}.{SCHEMA}.severity_test_set").toPandas()
all_claimants = pd.concat([train, test])

n = len(all_claimants)
avg_sev = all_claimants["claim_severity"].mean()
med_sev = all_claimants["claim_severity"].median()

print(f"Number of claimants: {n:,}")
print(f"Average claim cost:  £{avg_sev:,.0f}")
print(f"Median claim cost:   £{med_sev:,.0f}")
print(f"\n  → The average is higher than the median because a few very expensive claims")
print(f"    (like subsidence) pull the average up. This is typical in insurance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Scoreboard
# MAGIC
# MAGIC Just like before, we compare the two models. Here's what each metric means:
# MAGIC
# MAGIC | Metric | What it tells you | Good direction |
# MAGIC |---|---|---|
# MAGIC | **MAE** | On average, how many pounds off is the prediction? | **Lower is better** |
# MAGIC | **RMSE** | Like MAE, but punishes big mistakes more | **Lower is better** |
# MAGIC | **MAPE** | How far off are predictions as a percentage? | **Lower is better** |
# MAGIC | **Gini** | How well can the model tell cheap claims from expensive ones? | **Higher is better** |
# MAGIC | **Bias** | Does the model systematically over- or under-predict? 0% = perfectly balanced | **Closer to 0 is better** |

# COMMAND ----------

model_comparison = spark.table(f"{CATALOG}.{SCHEMA}.severity_model_comparison")
display(model_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the Scoreboard
# MAGIC
# MAGIC **Model 2 wins again.** The enriched severity model is more accurate because it can
# MAGIC see the factors that make claims expensive:
# MAGIC
# MAGIC - A flood claim is far more costly than a burst pipe
# MAGIC - Subsidence repair can run to tens of thousands of pounds
# MAGIC - Properties far from fire stations suffer more damage before help arrives
# MAGIC
# MAGIC Model 1 has to guess at these costs; Model 2 can price them directly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Drives Claim Costs?
# MAGIC
# MAGIC One advantage of the GBM is that it tells us **which features matter most** for
# MAGIC predicting how expensive a claim will be. This is called **feature importance**.

# COMMAND ----------

importance = spark.table(f"{CATALOG}.{SCHEMA}.severity_feature_importance").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, model_name, title, color in [
    (axes[0], "standard", "Model 1 — Standard", "#E53935"),
    (axes[1], "enriched", "Model 2 — Enriched", "#1E88E5"),
]:
    subset = importance[importance["model"] == model_name].sort_values("importance", ascending=True)
    ax.barh(subset["feature"], subset["importance"], color=color, alpha=0.85)
    ax.set_xlabel("Importance")
    ax.set_title(title)

plt.suptitle("What Drives Claim Costs? Feature Importance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What This Tells Us
# MAGIC
# MAGIC In **Model 1**, the most important features are things like `sum_insured` and
# MAGIC `building_age` — the model is doing its best with what it has, but it's missing
# MAGIC the real drivers of expensive claims.
# MAGIC
# MAGIC In **Model 2**, `flood_risk_zone` and `subsidence_risk` shoot up the rankings.
# MAGIC These are the features that really explain *why* some claims cost so much more
# MAGIC than others. The model can now see what was previously invisible.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Flood Risk Story — Severity Edition
# MAGIC
# MAGIC We saw earlier that Model 1 can't price flood risk for frequency. The same is true
# MAGIC for severity — and arguably the impact is even bigger, because flood claims are
# MAGIC **much more expensive** than average.

# COMMAND ----------

priced = spark.table(f"{CATALOG}.{SCHEMA}.severity_priced_portfolio").toPandas()

flood_seg = priced.groupby("flood_risk_zone").agg(
    avg_sev_model1=("sev_pred_standard", "mean"),
    avg_sev_model2=("sev_pred_enriched", "mean"),
    avg_actual_sev=("claim_severity", "mean"),
    n=("actual_loss", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(flood_seg))
width = 0.25

ax.bar(x - width, flood_seg["avg_sev_model1"], width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(x, flood_seg["avg_sev_model2"], width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar(x + width, flood_seg["avg_actual_sev"], width, label="Actual Average Cost", color="#43A047", alpha=0.85)

ax.set_xlabel("Flood Risk Zone (1 = Low Risk → 4 = High Risk)", fontsize=12)
ax.set_ylabel("Average Claim Cost (£)", fontsize=12)
ax.set_title("How Do the Models Predict Claim Costs by Flood Zone?", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(flood_seg["flood_risk_zone"])
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What This Shows
# MAGIC
# MAGIC **Model 1 predicts roughly the same claim cost regardless of flood zone** — it can't
# MAGIC see the data. But in reality, flood zone 4 claims cost significantly more than zone 1.
# MAGIC
# MAGIC **Model 2 correctly predicts higher costs for higher flood zones.** The blue bars
# MAGIC track much more closely to the green bars (actual costs).

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Subsidence Story — Severity Edition
# MAGIC
# MAGIC Subsidence claims are some of the most expensive in home insurance — often £30,000+
# MAGIC for underpinning work. Let's see how the models handle it.

# COMMAND ----------

sub_seg = priced.groupby("subsidence_risk").agg(
    avg_sev_model1=("sev_pred_standard", "mean"),
    avg_sev_model2=("sev_pred_enriched", "mean"),
    avg_actual_sev=("claim_severity", "mean"),
    n=("actual_loss", "count"),
).reset_index()
sub_seg["subsidence_risk"] = sub_seg["subsidence_risk"].map({0: "No Subsidence Risk", 1: "Subsidence Risk"})

fig, ax = plt.subplots(figsize=(9, 6))
x_pos = range(len(sub_seg))
width = 0.25

ax.bar([p - width for p in x_pos], sub_seg["avg_sev_model1"], width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(list(x_pos), sub_seg["avg_sev_model2"], width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar([p + width for p in x_pos], sub_seg["avg_actual_sev"], width, label="Actual Average Cost", color="#43A047", alpha=0.85)

ax.set_xlabel("Subsidence Risk", fontsize=12)
ax.set_ylabel("Average Claim Cost (£)", fontsize=12)
ax.set_title("How Do the Models Predict Subsidence Claim Costs?", fontsize=14, fontweight="bold")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(sub_seg["subsidence_risk"], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

m1_diff = sub_seg.iloc[1]["avg_sev_model1"] - sub_seg.iloc[0]["avg_sev_model1"]
m2_diff = sub_seg.iloc[1]["avg_sev_model2"] - sub_seg.iloc[0]["avg_sev_model2"]
actual_diff = sub_seg.iloc[1]["avg_actual_sev"] - sub_seg.iloc[0]["avg_actual_sev"]

print(f"Predicted cost difference for subsidence-risk properties:")
print(f"  Model 1: £{m1_diff:+,.0f} extra  (barely notices)")
print(f"  Model 2: £{m2_diff:+,.0f} extra  (sees the risk)")
print(f"  Reality: £{actual_diff:+,.0f} extra  (this is what we need to price for)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Full Picture — Frequency × Severity
# MAGIC
# MAGIC Now we can combine both models to see the **total pricing impact**:
# MAGIC
# MAGIC > **Full Quote = Predicted Frequency × Predicted Severity × Expense Loading**
# MAGIC
# MAGIC This gives us a complete burning-cost premium that accounts for both *how often*
# MAGIC and *how much*.

# COMMAND ----------

# Loss ratio comparison — full quotes
lr_data = spark.table(f"{CATALOG}.{SCHEMA}.severity_loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(12, 6))

for model_name in ["Standard", "Enriched"]:
    subset = lr_data[lr_data["model"] == model_name].sort_values("decile")
    marker = "s" if model_name == "Standard" else "o"
    color = "#E53935" if model_name == "Standard" else "#1E88E5"
    ax.plot(subset["decile"], subset["loss_ratio"], marker=marker, color=color,
            linewidth=2.5, markersize=8, label=model_name)

ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.6, linewidth=1, label="Breakeven (LR = 1.0)")
ax.fill_between(range(10), 0.85, 1.15, color="green", alpha=0.07, label="Healthy range")

ax.set_xlabel("Premium Decile (1 = cheapest → 10 = most expensive)", fontsize=12)
ax.set_ylabel("Loss Ratio", fontsize=12)
ax.set_title("Loss Ratio — Full Burning-Cost Quotes (Frequency × Severity)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.set_xticks(range(int(lr_data["decile"].min()), int(lr_data["decile"].max()) + 1))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What This Shows
# MAGIC
# MAGIC When we combine **better frequency predictions** with **better severity predictions**,
# MAGIC the improvement in pricing is even more dramatic:
# MAGIC
# MAGIC - **Standard model** (red) — loss ratios swing wildly. Some segments hugely profitable,
# MAGIC   others deeply loss-making. This means we're overcharging some customers and undercharging others.
# MAGIC - **Enriched model** (blue) — loss ratios stay much closer to breakeven across all
# MAGIC   deciles. Every customer segment is priced more fairly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What This Means for the Business
# MAGIC
# MAGIC Adding new data improves **both halves** of the pricing equation:
# MAGIC
# MAGIC | Component | Without new data | With new data |
# MAGIC |---|---|---|
# MAGIC | **Frequency** (how often) | Misses flood/subsidence/crime risk | Prices frequency accurately |
# MAGIC | **Severity** (how much) | Flat average — all claims treated equally | Knows flood claims cost more than burst pipes |
# MAGIC | **Combined quote** | Over/underprices by large margins | Tight, risk-adequate pricing |
# MAGIC
# MAGIC The combined effect is **multiplicative** — getting both frequency AND severity right
# MAGIC is much more powerful than improving either one alone. A property in flood zone 4 with
# MAGIC subsidence risk claims **more often** AND each claim **costs more**. Only the enriched
# MAGIC model captures both effects.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Glossary — Severity-Specific Terms
# MAGIC
# MAGIC | Term | Definition |
# MAGIC |---|---|
# MAGIC | **Severity** | The cost of a claim, given that one has occurred. Measured in pounds. |
# MAGIC | **GBM** | Gradient Boosted Machine — a type of machine learning model that combines many small decision trees. Good at capturing complex, non-linear patterns. |
# MAGIC | **LightGBM** | A fast, efficient implementation of GBM made by Microsoft. Widely used in insurance pricing. |
# MAGIC | **Gamma distribution** | A statistical distribution for positive, right-skewed data. Perfect for claim costs (always positive, often with a long tail of expensive claims). |
# MAGIC | **Feature importance** | A score showing how much each input contributes to the model's predictions. Higher = more influential. |
# MAGIC | **Pure premium** | Frequency × Severity — the expected claims cost per policy before loading. |
# MAGIC | **Burning cost** | Another term for pure premium — what the policy "burns" in claims on average. |
# MAGIC | **MAPE** | Mean Absolute Percentage Error — prediction error as a percentage. 10% means the model is typically off by 10%. |
