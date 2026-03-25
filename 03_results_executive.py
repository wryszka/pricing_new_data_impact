# Databricks notebook source

# MAGIC %pip install matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Setup — catalog and schema config (hidden in presentation)
CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

# COMMAND ----------

# MAGIC %md
# MAGIC # Can Better Data Make Better Insurance Prices?
# MAGIC
# MAGIC **No data-science background required. Everything is explained in plain English.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC We built two pricing models for home insurance:
# MAGIC
# MAGIC - **Model 1 — Standard** uses only the data we've always had: property type, age, size, occupancy, and claims history.
# MAGIC - **Model 2 — Enriched** adds new external data sources: flood risk, crime rates, distance to fire stations, annual rainfall, and whether the ground is at risk of sinking.
# MAGIC
# MAGIC This notebook tells the full story — from raw data to final pricing — and shows exactly what the new data buys us.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is a GLM? What is a GBM?
# MAGIC
# MAGIC We use **two different types of model** together. Each is the best tool for its job.
# MAGIC
# MAGIC ### GLM — Generalised Linear Model (for frequency)
# MAGIC
# MAGIC A **GLM** is the standard mathematical tool insurers have used for decades to calculate premiums.
# MAGIC Think of it as a recipe: you put in information about a property (size, age, location) and it
# MAGIC tells you the expected cost. The "ingredients" are called **rating factors** or **features**.
# MAGIC
# MAGIC We use a specific type called a **Poisson GLM** to predict how *often* claims happen, because
# MAGIC we're counting events — and counts are always whole numbers (0, 1, 2, ...). It uses a **log link**,
# MAGIC a mathematical trick that guarantees predictions are always positive. (You can't have a negative
# MAGIC number of claims!)
# MAGIC
# MAGIC - **Poisson** = good at predicting counts (how many claims will this policy have?)
# MAGIC - **Log link** = guarantees the answer is never negative
# MAGIC
# MAGIC ### GBM — Gradient Boosted Machine (for severity)
# MAGIC
# MAGIC A **GBM** is a machine learning model that combines hundreds of small decision trees.
# MAGIC We use it to predict *how much* a claim will cost — a job where GLMs struggle, because
# MAGIC claim costs are messy and vary enormously (a burst pipe might cost £2,000; subsidence
# MAGIC can cost £50,000).
# MAGIC
# MAGIC | | GLM | GBM |
# MAGIC |---|---|---|
# MAGIC | **How it works** | One formula | Hundreds of small decision trees working together |
# MAGIC | **Strengths** | Transparent, easy to explain to regulators | Captures complex patterns a formula would miss |
# MAGIC | **Best for** | Frequency (well-behaved counts) | Severity (messy, varied claim costs) |
# MAGIC
# MAGIC ### Together they give us the full price
# MAGIC
# MAGIC > **Expected Cost = How Often × How Much**
# MAGIC >
# MAGIC > Or in insurance terms: **Pure Premium = Frequency × Severity**
# MAGIC
# MAGIC This is how insurers calculate the base price before adding expenses and profit margin.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our Test Portfolio
# MAGIC
# MAGIC We created a portfolio of **50,000 home insurance policies** with realistic characteristics.
# MAGIC Let's load it and take a look.
# MAGIC
# MAGIC Here's what each column means:
# MAGIC
# MAGIC | Column | What it means |
# MAGIC |---|---|
# MAGIC | `property_type` | Is it a detached house, semi-detached, terraced, or a flat? |
# MAGIC | `construction` | What is the house built from? Brick, timber, stone, or other? |
# MAGIC | `building_age` | How old is the building (in years)? |
# MAGIC | `bedrooms` | How many bedrooms does it have? |
# MAGIC | `sum_insured` | How much would it cost to rebuild the property? |
# MAGIC | `occupancy` | Does the owner live there, or is it rented out? |
# MAGIC | `prior_claims` | How many claims has this policyholder made before? |
# MAGIC | `policy_tenure` | How many years has the customer been with us? |
# MAGIC | `flood_risk_zone` | How likely is the area to flood? 1 = low, 4 = high |
# MAGIC | `crime_index` | How much crime happens in the area? 0–100 scale |
# MAGIC | `distance_fire_station_km` | How far is the nearest fire station? |
# MAGIC | `annual_rainfall_mm` | How much rain does the area get per year? |
# MAGIC | `subsidence_risk` | Is the ground at risk of sinking? 0 = no, 1 = yes |
# MAGIC | `num_claims` | How many claims were made on this policy (what the frequency model predicts) |
# MAGIC | `total_loss` | Total cost of those claims in pounds |

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

portfolio = spark.table(f"{CATALOG}.{SCHEMA}.portfolio")

print(f"Total policies: {portfolio.count():,}")

stats = portfolio.agg(
    F.count("*").alias("total_policies"),
    F.sum(F.when(F.col("num_claims") > 0, 1).otherwise(0)).alias("policies_with_claims"),
    F.mean("num_claims").alias("avg_claims_per_policy"),
    F.mean("total_loss").alias("avg_loss_per_policy"),
).collect()[0]

claim_rate = stats["policies_with_claims"] / stats["total_policies"]
print(f"\nClaim rate: {claim_rate:.1%}")
print(f"  → Roughly {claim_rate:.0%} of policyholders made at least one claim.")
print(f"Average claims per policy: {stats['avg_claims_per_policy']:.3f}")
print(f"Average loss per policy:   £{stats['avg_loss_per_policy']:,.0f}")

display(portfolio.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Data Does Each Model Use?
# MAGIC
# MAGIC Both frequency and severity models come in a standard and enriched version.
# MAGIC The only difference between them is **what information they are allowed to see**.
# MAGIC
# MAGIC | Model 1 — What we've always had | Model 2 — With new data added |
# MAGIC |---|---|
# MAGIC | Property type — Is it a house, flat, or terraced? | Everything in Model 1, **PLUS:** |
# MAGIC | Construction — Brick, timber, stone? | Flood risk zone — How likely is the area to flood? Scale 1–4 |
# MAGIC | Building age — How old is the property? | Crime index — How much crime in the neighbourhood? 0–100 |
# MAGIC | Bedrooms — How many bedrooms? | Distance to fire station — How far away is help? |
# MAGIC | Sum insured — Rebuild cost of the property | Annual rainfall — How wet is the area? |
# MAGIC | Occupancy — Owner-occupied or rented? | Subsidence risk — Is the ground sinking? Yes/No |
# MAGIC | Prior claims — How claims-happy is the customer? | |
# MAGIC | Policy tenure — How long have they been with us? | |
# MAGIC
# MAGIC ### Why do these extra features matter?
# MAGIC
# MAGIC Imagine two identical houses — same size, same age, same construction. But one sits in
# MAGIC a flood plain and the other is on a hilltop. **They have very different risk profiles**,
# MAGIC but Model 1 has no way of knowing that. It would charge them the same price.
# MAGIC
# MAGIC Model 2 can see the flood risk, the crime rate, and whether the ground is sinking —
# MAGIC so it charges a price that actually reflects reality. That's fairer for the customer
# MAGIC and better for the business.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1 — How Often Do Claims Happen?
# MAGIC ## The Frequency Story
# MAGIC
# MAGIC Our **Poisson GLM** predicts the *number* of claims a policy will generate.
# MAGIC Getting this right is the first building block of accurate pricing.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How Do We Know Which Model is Better?
# MAGIC
# MAGIC We use several "scores" to compare the models. Here's what each one means:
# MAGIC
# MAGIC | Metric | What it tells you | Good direction |
# MAGIC |---|---|---|
# MAGIC | **AIC / BIC** | Like a score for how well the model fits the data, with a penalty for being too complicated. Think of it as "accuracy minus a complexity tax." | **Lower is better** |
# MAGIC | **Deviance Explained** | What percentage of the variation in claims can the model explain? Like an exam score — 50% means you got half the answers right. | **Higher is better** |
# MAGIC | **Gini Coefficient** | How well can the model tell apart high-risk and low-risk policies? 0 = can't tell the difference; 1 = sorts perfectly. | **Higher is better** |
# MAGIC | **MAE** | On average, how far off are the predictions (in number of claims)? | **Lower is better** |
# MAGIC | **RMSE** | Like MAE, but punishes big mistakes more heavily. | **Lower is better** |

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Frequency Scoreboard
# MAGIC
# MAGIC Let's load the side-by-side comparison and see which model comes out on top.

# COMMAND ----------

model_comparison = spark.table(f"{CATALOG}.{SCHEMA}.model_comparison")
display(model_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 2 wins on every metric.** Here's what that means in plain English:
# MAGIC
# MAGIC - **AIC and BIC are lower** — the enriched model fits the data better, even after being
# MAGIC   penalised for using more ingredients. The extra data is genuinely useful, not just noise.
# MAGIC - **Deviance Explained is higher** — Model 2 understands more of *why* some properties
# MAGIC   have more claims than others.
# MAGIC - **Gini is higher** — Model 2 is better at sorting policies from low-risk to high-risk.
# MAGIC   This is critical: if you can't tell who's risky, you can't charge them appropriately.
# MAGIC - **MAE and RMSE are lower** — Model 2's predictions are closer to reality.
# MAGIC
# MAGIC In short: **more data = better predictions = better pricing.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is a Loss Ratio?
# MAGIC
# MAGIC The **loss ratio** tells you whether your pricing is working. It's a simple formula:
# MAGIC
# MAGIC > **Loss Ratio = Claims Paid ÷ Premiums Collected**
# MAGIC
# MAGIC - **1.0** = breaking even — every pound collected went back out in claims.
# MAGIC - **Above 1.0** = losing money (paying out more than you're taking in).
# MAGIC - **Below 1.0** = profitable on that segment.
# MAGIC
# MAGIC Ideally you want the loss ratio to be **stable and predictable** across all customer
# MAGIC segments. Wild swings mean some customers are being overcharged (they'll leave) and
# MAGIC others are being undercharged (they'll cost you money).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loss Ratio by Premium Decile
# MAGIC
# MAGIC We split all policies into 10 groups (**deciles**) based on the premium the model would
# MAGIC charge — from the cheapest 10% to the most expensive 10%. Then we check: for each group,
# MAGIC how do the claims compare to the premiums?

# COMMAND ----------

lr_data = spark.table(f"{CATALOG}.{SCHEMA}.loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(12, 6))

colours = {"Standard": "#E53935", "Enriched": "#1E88E5"}
markers = {"Standard": "s", "Enriched": "o"}

for model_name in ["Standard", "Enriched"]:
    subset = lr_data[lr_data["model"] == model_name].sort_values("decile")
    ax.plot(
        subset["decile"], subset["loss_ratio"],
        marker=markers[model_name], color=colours[model_name],
        linewidth=2.5, markersize=8, label=model_name,
    )

ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.6, linewidth=1, label="Breakeven (LR = 1.0)")
ax.fill_between(range(10), 0.85, 1.15, color="green", alpha=0.07, label="Healthy range")

ax.set_xlabel("Premium Decile (1 = cheapest policies → 10 = most expensive)", fontsize=12)
ax.set_ylabel("Loss Ratio", fontsize=12)
ax.set_title("Loss Ratio by Premium Decile — Which Model Prices More Consistently?", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i+1}" for i in range(10)])
ax.set_ylim(0, max(lr_data["loss_ratio"].max() * 1.15, 2.0))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Notice how **Model 1's loss ratio swings wildly** — some customer segments are hugely
# MAGIC overcharged (loss ratio well below 1) and others are dramatically undercharged
# MAGIC (loss ratio well above 1). This means:
# MAGIC - Overcharged customers will **shop around and leave** for a competitor.
# MAGIC - Undercharged customers will **stay** — but they cost us money.
# MAGIC - This is the classic **adverse selection death spiral**.
# MAGIC
# MAGIC **Model 2 is much more stable.** The loss ratio stays closer to breakeven across all
# MAGIC deciles, meaning every customer segment is priced more fairly. That's better for
# MAGIC customers AND better for the business.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Price Flood Risk
# MAGIC
# MAGIC Model 1 cannot see flood risk data. Model 2 can. Let's see what difference that makes.

# COMMAND ----------

priced = spark.table(f"{CATALOG}.{SCHEMA}.priced_portfolio").toPandas()

flood_comparison = priced.groupby("flood_risk_zone").agg(
    avg_quote_model1=("quote_standard", "mean"),
    avg_quote_model2=("quote_enriched", "mean"),
    avg_actual_loss=("actual_loss", "mean"),
    policy_count=("actual_loss", "count"),
).reset_index()

flood_comparison = flood_comparison.rename(columns={
    "avg_quote_model1": "Avg Quote — Model 1 (£)",
    "avg_quote_model2": "Avg Quote — Model 2 (£)",
    "avg_actual_loss": "Avg Actual Loss (£)",
    "policy_count": "Number of Policies",
    "flood_risk_zone": "Flood Risk Zone",
})

print("Average premium quote by flood risk zone:")
display(spark.createDataFrame(flood_comparison))

fig, ax = plt.subplots(figsize=(10, 6))

x = flood_comparison["Flood Risk Zone"]
width = 0.25

ax.bar(x - width, flood_comparison["Avg Quote — Model 1 (£)"], width,
       label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(x, flood_comparison["Avg Quote — Model 2 (£)"], width,
       label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar(x + width, flood_comparison["Avg Actual Loss (£)"], width,
       label="Actual Average Loss", color="#43A047", alpha=0.85)

ax.set_xlabel("Flood Risk Zone (1 = Low Risk → 4 = High Risk)", fontsize=12)
ax.set_ylabel("Amount (£)", fontsize=12)
ax.set_title("How Do the Models Price Flood Risk? (Frequency)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xticks([1, 2, 3, 4])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 charges roughly the same price regardless of flood risk** — it simply cannot
# MAGIC see that data. Whether your house is on a hilltop or in a flood plain, you get a similar quote.
# MAGIC
# MAGIC **Model 2 correctly charges more for high-flood-risk properties.** The blue bars track
# MAGIC much more closely to the green bars (actual losses). That's what good pricing looks like.
# MAGIC
# MAGIC For the business, this means:
# MAGIC - We stop **underpricing** flood-prone properties (saving money on claims).
# MAGIC - We stop **overpricing** safe properties (keeping good customers).

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Price Subsidence Risk
# MAGIC
# MAGIC **Subsidence** means the ground under the house is sinking — often due to clay soil
# MAGIC drying out, tree roots, or old mining activity. It's one of the most expensive types
# MAGIC of claim, often costing tens of thousands of pounds to fix.

# COMMAND ----------

subsidence_comparison = priced.groupby("subsidence_risk").agg(
    avg_quote_model1=("quote_standard", "mean"),
    avg_quote_model2=("quote_enriched", "mean"),
    avg_actual_loss=("actual_loss", "mean"),
    policy_count=("actual_loss", "count"),
).reset_index()

subsidence_comparison["subsidence_risk"] = subsidence_comparison["subsidence_risk"].map(
    {0: "No Subsidence Risk", 1: "Subsidence Risk"}
)

subsidence_comparison = subsidence_comparison.rename(columns={
    "avg_quote_model1": "Avg Quote — Model 1 (£)",
    "avg_quote_model2": "Avg Quote — Model 2 (£)",
    "avg_actual_loss": "Avg Actual Loss (£)",
    "policy_count": "Number of Policies",
    "subsidence_risk": "Subsidence Risk",
})

print("Average premium quote by subsidence risk:")
display(spark.createDataFrame(subsidence_comparison))

no_risk = subsidence_comparison[subsidence_comparison["Subsidence Risk"] == "No Subsidence Risk"]
yes_risk = subsidence_comparison[subsidence_comparison["Subsidence Risk"] == "Subsidence Risk"]

m1_diff = yes_risk["Avg Quote — Model 1 (£)"].values[0] - no_risk["Avg Quote — Model 1 (£)"].values[0]
m2_diff = yes_risk["Avg Quote — Model 2 (£)"].values[0] - no_risk["Avg Quote — Model 2 (£)"].values[0]
actual_diff = yes_risk["Avg Actual Loss (£)"].values[0] - no_risk["Avg Actual Loss (£)"].values[0]

print(f"\nPrice difference for subsidence-risk properties:")
print(f"  Model 1 charges £{m1_diff:+,.2f} extra  (it can barely tell the difference)")
print(f"  Model 2 charges £{m2_diff:+,.2f} extra  (it sees the risk)")
print(f"  Actual losses are £{actual_diff:+,.2f} higher  (this is the reality we need to price for)")

fig, ax = plt.subplots(figsize=(9, 6))

categories = subsidence_comparison["Subsidence Risk"]
x_pos = range(len(categories))
width = 0.25

ax.bar([p - width for p in x_pos], subsidence_comparison["Avg Quote — Model 1 (£)"],
       width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(list(x_pos), subsidence_comparison["Avg Quote — Model 2 (£)"],
       width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar([p + width for p in x_pos], subsidence_comparison["Avg Actual Loss (£)"],
       width, label="Actual Average Loss", color="#43A047", alpha=0.85)

ax.set_xlabel("Subsidence Risk", fontsize=12)
ax.set_ylabel("Amount (£)", fontsize=12)
ax.set_title("How Do the Models Price Subsidence Risk? (Frequency)", fontsize=14, fontweight="bold")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 has no idea subsidence is happening.** It charges nearly the same price whether
# MAGIC or not the ground is at risk of sinking. But the actual claims are significantly higher
# MAGIC for subsidence-risk properties.
# MAGIC
# MAGIC **Model 2 increases the price for at-risk properties**, bringing the quote much closer
# MAGIC to the real cost. Without this adjustment, we'd be systematically losing money on every
# MAGIC subsidence-risk policy we insure.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2 — How Much Do Claims Cost?
# MAGIC ## The Severity Story
# MAGIC
# MAGIC Knowing *how often* claims happen is only half the picture. We also need to know
# MAGIC *how expensive* each claim will be. That's what the severity model does.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why a GBM Instead of a GLM for Severity?
# MAGIC
# MAGIC Claim costs are messy — a burst pipe costs £2,000 but a subsidence claim costs £50,000.
# MAGIC A GBM is better at spotting these complex patterns, especially when they depend on
# MAGIC combinations of factors (e.g., old timber house + flood zone = very expensive).
# MAGIC
# MAGIC We use **LightGBM** with a **Gamma distribution**, which is designed for strictly
# MAGIC positive, right-skewed data — exactly what claim costs look like.

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Claimants

# COMMAND ----------

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
print(f"    (like subsidence) pull the average up. This is completely normal in insurance.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Severity Scoreboard
# MAGIC
# MAGIC Here's what the metrics mean for severity:
# MAGIC
# MAGIC | Metric | What it tells you | Good direction |
# MAGIC |---|---|---|
# MAGIC | **MAE** | On average, how many pounds off is the prediction? | **Lower is better** |
# MAGIC | **RMSE** | Like MAE, but punishes big mistakes more heavily | **Lower is better** |
# MAGIC | **MAPE** | How far off are predictions as a percentage? 10% = typically off by 10% | **Lower is better** |
# MAGIC | **Gini** | How well can the model tell cheap claims from expensive ones? | **Higher is better** |
# MAGIC | **Bias** | Does the model systematically over- or under-predict? 0% = perfectly balanced | **Closer to 0 is better** |

# COMMAND ----------

severity_comparison = spark.table(f"{CATALOG}.{SCHEMA}.severity_model_comparison")
display(severity_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 2 wins again.** The enriched severity model is more accurate because it can
# MAGIC see the factors that make claims expensive:
# MAGIC
# MAGIC - A flood claim is far more costly than a burst pipe
# MAGIC - Subsidence repair can run to tens of thousands of pounds
# MAGIC - Properties far from fire stations suffer more damage before help arrives
# MAGIC
# MAGIC Model 1 has to guess at these costs. Model 2 can price them directly.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What Drives Claim Costs? — Feature Importance
# MAGIC
# MAGIC One advantage of GBMs is that they tell us **which features matter most** for predicting
# MAGIC how expensive a claim will be. This is called **feature importance**.

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
# MAGIC In **Model 1**, the most important features are things like `sum_insured` and
# MAGIC `building_age` — the model is doing its best with what it has, but it's missing
# MAGIC the real drivers of expensive claims.
# MAGIC
# MAGIC In **Model 2**, `flood_risk_zone` and `subsidence_risk` shoot up the rankings.
# MAGIC These are the features that really explain *why* some claims cost so much more
# MAGIC than others. The model can now see what was previously invisible.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Predict Claim Costs by Flood Zone
# MAGIC
# MAGIC We saw that Model 1 can't price flood risk for frequency. The same is true
# MAGIC for severity — and arguably the impact is even bigger, because flood claims are
# MAGIC **much more expensive** than average.

# COMMAND ----------

priced_sev = spark.table(f"{CATALOG}.{SCHEMA}.severity_priced_portfolio").toPandas()

flood_seg = priced_sev.groupby("flood_risk_zone").agg(
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
ax.set_title("How Do the Models Predict Claim Costs by Flood Zone? (Severity)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(flood_seg["flood_risk_zone"])
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 predicts roughly the same claim cost regardless of flood zone** — it can't see
# MAGIC the data. But in reality, flood zone 4 claims cost significantly more than zone 1.
# MAGIC
# MAGIC **Model 2 correctly predicts higher costs for higher flood zones.** The blue bars track
# MAGIC much more closely to the green bars (actual costs).

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Predict Subsidence Claim Costs
# MAGIC
# MAGIC Subsidence claims are some of the most expensive in home insurance — often £30,000+
# MAGIC for underpinning work. Let's see how the severity models handle it.

# COMMAND ----------

sub_seg = priced_sev.groupby("subsidence_risk").agg(
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
ax.set_title("How Do the Models Predict Subsidence Claim Costs? (Severity)", fontsize=14, fontweight="bold")
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
# MAGIC ---
# MAGIC # Part 3 — The Full Picture
# MAGIC ## Frequency × Severity Combined
# MAGIC
# MAGIC Now we combine both models to see the **total pricing impact**:
# MAGIC
# MAGIC > **Full Quote = Predicted Frequency × Predicted Severity × Expense Loading**
# MAGIC
# MAGIC This gives us a complete burning-cost premium that accounts for both *how often*
# MAGIC claims happen and *how much* they cost when they do.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Full Burning-Cost Loss Ratio
# MAGIC
# MAGIC The chart below shows what happens to the loss ratio when we use the combined
# MAGIC frequency × severity quote — the final number a customer would actually be charged.

# COMMAND ----------

lr_data_sev = spark.table(f"{CATALOG}.{SCHEMA}.severity_loss_ratio_by_decile").toPandas()

fig, ax = plt.subplots(figsize=(12, 6))

for model_name in ["Standard", "Enriched"]:
    subset = lr_data_sev[lr_data_sev["model"] == model_name].sort_values("decile")
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
ax.set_xticks(range(int(lr_data_sev["decile"].min()), int(lr_data_sev["decile"].max()) + 1))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC When we combine better frequency predictions with better severity predictions,
# MAGIC the improvement is even more dramatic than either model alone:
# MAGIC
# MAGIC - **Standard model** (red) — loss ratios swing wildly. Some segments hugely profitable,
# MAGIC   others deeply loss-making.
# MAGIC - **Enriched model** (blue) — loss ratios stay much closer to breakeven across all
# MAGIC   deciles. Every customer segment is priced more fairly.
# MAGIC
# MAGIC ### Why is the combined improvement so powerful?
# MAGIC
# MAGIC The effect is **multiplicative**. A property in flood zone 4 with subsidence risk
# MAGIC claims **more often** AND each claim **costs more**. Under the standard model, we miss
# MAGIC both effects. Under the enriched model, we capture both — and the two corrections
# MAGIC multiply together into a significantly more accurate final price.
# MAGIC
# MAGIC | Component | Without new data | With new data |
# MAGIC |---|---|---|
# MAGIC | **Frequency** (how often) | Misses flood/subsidence/crime risk | Prices frequency accurately |
# MAGIC | **Severity** (how much) | Flat average — all claims treated equally | Knows flood claims cost far more than burst pipes |
# MAGIC | **Combined quote** | Over/underprices by large margins | Tight, risk-adequate pricing |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # What This Means for the Business
# MAGIC
# MAGIC Adding new data sources improves **both halves** of the pricing equation and delivers
# MAGIC four concrete benefits:
# MAGIC
# MAGIC ### 1. Better Risk Selection
# MAGIC We can avoid underpricing the riskiest properties. Right now, a house in a flood zone
# MAGIC or on sinking ground gets the same price as a safe one. With the enriched model, we
# MAGIC charge appropriately — or decline if the risk is too high.
# MAGIC
# MAGIC ### 2. Fairer Pricing
# MAGIC Low-risk customers pay less, high-risk customers pay more. This is fairer for everyone
# MAGIC and makes our product more competitive for the customers we actually want.
# MAGIC
# MAGIC ### 3. Less Adverse Selection
# MAGIC If our competitors are already using flood, subsidence, and crime data (and many are),
# MAGIC they'll cherry-pick our underpriced low-risk customers and leave us with the expensive ones.
# MAGIC Matching their data means we stop being the "insurer of last resort."
# MAGIC
# MAGIC ### 4. More Stable Profitability
# MAGIC Loss ratios are more predictable across all customer segments — for both frequency and
# MAGIC severity. No more nasty surprises when a segment turns out to be dramatically underpriced
# MAGIC on one or both dimensions.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Glossary
# MAGIC
# MAGIC | Term | Definition |
# MAGIC |---|---|
# MAGIC | **GLM** | Generalised Linear Model — the standard mathematical tool for calculating insurance premiums. Uses a formula to relate rating factors to expected claims. |
# MAGIC | **GBM** | Gradient Boosted Machine — a machine learning model that combines hundreds of small decision trees. Better than a GLM at capturing complex, non-linear patterns in claim costs. |
# MAGIC | **LightGBM** | A fast, efficient implementation of GBM made by Microsoft. Widely used in insurance pricing. |
# MAGIC | **Poisson** | A type of GLM suited to counting events (like the number of claims on a policy). |
# MAGIC | **Gamma distribution** | A statistical distribution for positive, right-skewed data. Perfect for claim costs (always positive, often with a long tail of expensive claims). |
# MAGIC | **Log link** | A mathematical function ensuring model predictions are always positive (you can't have negative claims). |
# MAGIC | **Frequency** | How often claims happen — predicted using the Poisson GLM. |
# MAGIC | **Severity** | The cost of a claim, given that one has occurred. Measured in pounds — predicted using the GBM. |
# MAGIC | **Pure premium** | Frequency × Severity — the expected claims cost per policy before any loading for expenses or profit. |
# MAGIC | **Burning cost** | Another term for pure premium — what the policy "burns" in claims on average. |
# MAGIC | **Expense loading** | A multiplier applied to the pure premium to cover business costs (admin, commissions, profit margin). |
# MAGIC | **Rating factor / Feature** | A piece of information used by the model to calculate a price (e.g., building age, flood risk). |
# MAGIC | **Feature importance** | A score showing how much each input feature contributes to a model's predictions. Higher = more influential. |
# MAGIC | **Loss ratio** | Claims paid divided by premiums collected. Below 1.0 = profitable; above 1.0 = losing money. |
# MAGIC | **Decile** | One of 10 equal-sized groups, used to compare performance across the range of predictions. |
# MAGIC | **Gini coefficient** | Measures how well a model can sort policies from low-risk to high-risk (or cheap claims from expensive ones). 0 = no sorting ability; 1 = perfect. |
# MAGIC | **AIC** | Akaike Information Criterion — a score for model quality that penalises unnecessary complexity. Lower is better. |
# MAGIC | **BIC** | Bayesian Information Criterion — similar to AIC but with a stricter penalty for complexity. Lower is better. |
# MAGIC | **Deviance** | A technical measure of how far the model's predictions are from reality. Lower deviance = better fit. |
# MAGIC | **Deviance explained** | The percentage of variation in claims that the model can account for. Higher is better. |
# MAGIC | **MAE** | Mean Absolute Error — the average size of prediction errors. Lower is better. |
# MAGIC | **RMSE** | Root Mean Squared Error — like MAE but penalises large errors more heavily. Lower is better. |
# MAGIC | **MAPE** | Mean Absolute Percentage Error — prediction error expressed as a percentage. 10% means the model is typically off by 10%. Lower is better. |
# MAGIC | **Adverse selection** | When competitors use better data, they attract your best (cheapest) customers and leave you with the worst (most expensive). |
# MAGIC | **Enrichment data** | Additional data sourced from third parties (e.g., flood maps, crime databases) that improves pricing accuracy. |
# MAGIC | **Subsidence** | When the ground under a building sinks, often causing structural damage. Very expensive to repair — typically £30,000+. |
