# Databricks notebook source
# MAGIC %md
# MAGIC # Can Better Data Make Better Insurance Prices?
# MAGIC
# MAGIC We built two pricing models for home insurance. One uses only the data we've always had.
# MAGIC The other adds new data sources — things like flood risk, crime rates, and whether the ground
# MAGIC is at risk of sinking. This notebook shows why the second model is better, and what that
# MAGIC means for the business.
# MAGIC
# MAGIC **No data-science background required.** Every chart and table is explained in plain English.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is a GLM?
# MAGIC
# MAGIC A **GLM (Generalised Linear Model)** is the standard mathematical tool insurers use to
# MAGIC calculate premiums. Think of it as a recipe: you put in information about a property
# MAGIC (size, age, location) and it tells you the expected cost of insuring it. The "ingredients"
# MAGIC are called **rating factors** or **features**.
# MAGIC
# MAGIC We use a specific type called a **Poisson GLM** because we're counting how often claims
# MAGIC happen — and counts are always whole numbers (0, 1, 2, ...). The model uses something
# MAGIC called a **log link**, which is just a mathematical trick to make sure the predictions
# MAGIC are always positive. (You can't have a negative number of claims!)
# MAGIC
# MAGIC In short:
# MAGIC - **Poisson** = good at predicting counts (like "how many claims will this policy have?")
# MAGIC - **Log link** = guarantees the answer is never negative

# COMMAND ----------

# MAGIC %pip install matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our Test Portfolio
# MAGIC
# MAGIC We created a portfolio of **50,000 home insurance policies** with realistic characteristics.
# MAGIC Let's load it up and take a look.
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
# MAGIC | `crime_index` | How much crime happens in the area? 0-100 scale |
# MAGIC | `distance_fire_station_km` | How far is the nearest fire station? |
# MAGIC | `annual_rainfall_mm` | How much rain does the area get per year? |
# MAGIC | `subsidence_risk` | Is the ground at risk of sinking? 0 = no, 1 = yes |
# MAGIC | `num_claims` | How many claims were made on this policy (the answer we're trying to predict) |
# MAGIC | `total_loss` | Total cost of those claims in pounds |

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"

portfolio = spark.table(f"{CATALOG}.{SCHEMA}.portfolio")

print(f"Total policies: {portfolio.count():,}")

# Show claim rate
import pyspark.sql.functions as F

stats = portfolio.agg(
    F.count("*").alias("total_policies"),
    F.sum(F.when(F.col("num_claims") > 0, 1).otherwise(0)).alias("policies_with_claims"),
    F.mean("num_claims").alias("avg_claims_per_policy"),
    F.mean("total_loss").alias("avg_loss_per_policy"),
).collect()[0]

claim_rate = stats["policies_with_claims"] / stats["total_policies"]
print(f"\nClaim rate: {claim_rate:.1%}")
print(f"  → That means roughly {claim_rate:.0%} of policyholders made at least one claim.")
print(f"Average claims per policy: {stats['avg_claims_per_policy']:.3f}")
print(f"Average loss per policy: £{stats['avg_loss_per_policy']:,.0f}")

display(portfolio.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Data Does Each Model Use?
# MAGIC
# MAGIC This is the key question. Both models are the same type of model (a Poisson GLM).
# MAGIC The only difference is **what information they're allowed to see**.
# MAGIC
# MAGIC | Model 1 — What we've always had | Model 2 — With new data added |
# MAGIC |---|---|
# MAGIC | Property type — Is it a house, flat, or terraced? | Everything in Model 1, **PLUS:** |
# MAGIC | Construction — Brick, timber, stone? | Flood risk zone — How likely is the area to flood? Scale 1-4 |
# MAGIC | Building age — How old is the property? | Crime index — How much crime in the neighbourhood? 0-100 |
# MAGIC | Bedrooms — How many bedrooms? | Distance to fire station — How far away is help? |
# MAGIC | Sum insured — Rebuild cost of the property | Annual rainfall — How wet is the area? |
# MAGIC | Occupancy — Owner-occupied or rented? | Subsidence risk — Is the ground sinking? Yes/No |
# MAGIC | Prior claims — How claims-happy is the customer? | |
# MAGIC | Policy tenure — How long have they been with us? | |
# MAGIC
# MAGIC ### Why do these extra features matter?
# MAGIC
# MAGIC Imagine two identical houses — same size, same age, same construction. But one sits in
# MAGIC a flood plain, and the other is on a hilltop. **They have very different risk profiles**,
# MAGIC but Model 1 has no way of knowing that. It would charge them the same price.
# MAGIC
# MAGIC Model 2 can see the flood risk, the crime rate, and whether the ground is sinking —
# MAGIC so it charges a price that actually reflects the risk. That's fairer for everyone.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How Do We Know Which Model is Better?
# MAGIC
# MAGIC We use several different "scores" to compare the models. Here's what each one means:
# MAGIC
# MAGIC | Metric | What it tells you | Good direction |
# MAGIC |---|---|---|
# MAGIC | **AIC / BIC** | Like a score for how well the model fits the data, with a penalty for being too complicated. Think of it as "accuracy minus a complexity tax." | **Lower is better** |
# MAGIC | **Deviance Explained** | What percentage of the variation in claims can the model explain? Think of it like an exam score — 50% means you got half the answers right. | **Higher is better** |
# MAGIC | **Gini Coefficient** | How well can the model tell apart high-risk and low-risk policies? A Gini of 0 means the model can't tell the difference; 1 means it sorts perfectly. | **Higher is better** |
# MAGIC | **MAE** (Mean Absolute Error) | On average, how far off are the predictions? If MAE is 0.05, the model is typically wrong by 0.05 claims per policy. | **Lower is better** |
# MAGIC | **RMSE** (Root Mean Squared Error) | Similar to MAE, but punishes big mistakes more heavily. A model that's slightly wrong everywhere beats one that's very wrong sometimes. | **Lower is better** |

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Scoreboard
# MAGIC
# MAGIC Let's load the side-by-side comparison and see which model comes out on top.

# COMMAND ----------

model_comparison = spark.table(f"{CATALOG}.{SCHEMA}.model_comparison")
display(model_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading the Scoreboard
# MAGIC
# MAGIC **Model 2 wins on every metric.** Here's what that means in plain English:
# MAGIC
# MAGIC - **AIC and BIC are lower** — the enriched model fits the data better, even after being
# MAGIC   penalised for using more ingredients. The extra data is genuinely useful, not just noise.
# MAGIC - **Deviance Explained is higher** — Model 2 understands more of *why* some properties
# MAGIC   have more claims than others.
# MAGIC - **Gini is higher** — Model 2 is better at sorting policies from low-risk to high-risk.
# MAGIC   This is critical for pricing: if you can't tell who's risky, you can't charge them appropriately.
# MAGIC - **MAE and RMSE are lower** — Model 2's predictions are closer to reality, on average.
# MAGIC
# MAGIC In short: **more data = better predictions = better pricing.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is a Loss Ratio?
# MAGIC
# MAGIC The **loss ratio** tells you whether your pricing is working. It's a simple formula:
# MAGIC
# MAGIC > **Loss Ratio = Claims Paid / Premiums Collected**
# MAGIC
# MAGIC - If it's **1.0**, you're breaking even — every pound you collected went back out in claims.
# MAGIC - **Above 1.0** means you're losing money (paying out more than you're taking in).
# MAGIC - **Below 1.0** means you're profitable on that segment.
# MAGIC
# MAGIC Ideally, you want the loss ratio to be **stable and predictable** across all your customer
# MAGIC segments. Wild swings mean some customers are being overcharged (they'll leave) and
# MAGIC others are being undercharged (they'll cost you money).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loss Ratio Comparison
# MAGIC
# MAGIC We split all policies into 10 groups (called **deciles**) based on the premium the model
# MAGIC would charge — from the cheapest 10% to the most expensive 10%. Then we check: for each
# MAGIC group, how do the claims compare to the premiums?

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
# MAGIC ### What Does This Chart Tell Us?
# MAGIC
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
# MAGIC ## Where Does Pricing Differ Most?
# MAGIC
# MAGIC Let's look at how the two models price differently depending on **flood risk**.
# MAGIC Remember: Model 1 can't see flood risk data, but Model 2 can.

# COMMAND ----------

import pandas as pd

priced = spark.table(f"{CATALOG}.{SCHEMA}.priced_portfolio").toPandas()

# Average quote by flood risk zone
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

# Visualise
fig, ax = plt.subplots(figsize=(10, 6))

x = flood_comparison["Flood Risk Zone"]
width = 0.25

bars1 = ax.bar(x - width, flood_comparison["Avg Quote — Model 1 (£)"], width,
               label="Model 1 (Standard)", color="#E53935", alpha=0.85)
bars2 = ax.bar(x, flood_comparison["Avg Quote — Model 2 (£)"], width,
               label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
bars3 = ax.bar(x + width, flood_comparison["Avg Actual Loss (£)"], width,
               label="Actual Average Loss", color="#43A047", alpha=0.85)

ax.set_xlabel("Flood Risk Zone (1 = Low Risk → 4 = High Risk)", fontsize=12)
ax.set_ylabel("Amount (£)", fontsize=12)
ax.set_title("How Do the Models Price Flood Risk?", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xticks([1, 2, 3, 4])
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What This Shows
# MAGIC
# MAGIC **Model 1 charges roughly the same price regardless of flood risk**, because it
# MAGIC simply cannot see that data. Whether your house is on a hilltop or in a flood plain,
# MAGIC you get a similar quote.
# MAGIC
# MAGIC **Model 2 correctly charges more for high-flood-risk properties.** Look at how the
# MAGIC blue bars (Model 2) track much more closely to the green bars (actual losses). That's
# MAGIC what good pricing looks like — the price reflects the actual risk.
# MAGIC
# MAGIC For the business, this means:
# MAGIC - We stop **underpricing** flood-prone properties (saving money on claims).
# MAGIC - We stop **overpricing** safe properties (keeping good customers).

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Subsidence Story
# MAGIC
# MAGIC **Subsidence** means the ground under the house is sinking — often due to clay soil
# MAGIC drying out, tree roots, or old mining activity. It's one of the most expensive types
# MAGIC of claim to fix, often costing tens of thousands of pounds.
# MAGIC
# MAGIC Let's see how the two models handle it.

# COMMAND ----------

# Average quote by subsidence risk
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

# Calculate the uplift
no_risk = subsidence_comparison[subsidence_comparison["Subsidence Risk"] == "No Subsidence Risk"]
yes_risk = subsidence_comparison[subsidence_comparison["Subsidence Risk"] == "Subsidence Risk"]

m1_diff = yes_risk["Avg Quote — Model 1 (£)"].values[0] - no_risk["Avg Quote — Model 1 (£)"].values[0]
m2_diff = yes_risk["Avg Quote — Model 2 (£)"].values[0] - no_risk["Avg Quote — Model 2 (£)"].values[0]
actual_diff = yes_risk["Avg Actual Loss (£)"].values[0] - no_risk["Avg Actual Loss (£)"].values[0]

print(f"\nPrice difference for subsidence-risk properties:")
print(f"  Model 1 charges £{m1_diff:+,.2f} extra (it can barely tell the difference)")
print(f"  Model 2 charges £{m2_diff:+,.2f} extra (it sees the risk)")
print(f"  Actual losses are £{actual_diff:+,.2f} higher (this is reality)")

# Visualise
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
ax.set_title("How Do the Models Price Subsidence Risk?", fontsize=14, fontweight="bold")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What This Shows
# MAGIC
# MAGIC **Model 1 has no idea subsidence is happening.** It charges nearly the same price whether
# MAGIC or not the ground is at risk of sinking. But the actual claims are significantly higher
# MAGIC for subsidence-risk properties.
# MAGIC
# MAGIC **Model 2 increases the price for properties at risk**, bringing the quote much closer
# MAGIC to the real cost. Without this adjustment, we'd be systematically losing money on every
# MAGIC subsidence-risk property we insure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What This Means for the Business
# MAGIC
# MAGIC Adding new data sources to our pricing model delivers four concrete benefits:
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
# MAGIC If our competitors are already using flood and subsidence data (and many are), they'll
# MAGIC cherry-pick our underpriced low-risk customers and leave us with the expensive ones.
# MAGIC Matching their data means we stop being the "insurer of last resort."
# MAGIC
# MAGIC ### 4. More Stable Profitability
# MAGIC Loss ratios are more predictable across all customer segments. No more nasty surprises
# MAGIC when one segment turns out to be dramatically underpriced.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Glossary
# MAGIC
# MAGIC | Term | Definition |
# MAGIC |---|---|
# MAGIC | **GLM** | Generalised Linear Model — the standard mathematical tool for calculating insurance premiums. |
# MAGIC | **Poisson** | A type of GLM suited to counting events (like the number of claims on a policy). |
# MAGIC | **Rating factor / Feature** | A piece of information used by the model to calculate a price (e.g., building age, flood risk). |
# MAGIC | **Loss ratio** | Claims paid divided by premiums collected. Below 1.0 = profitable; above 1.0 = losing money. |
# MAGIC | **Gini coefficient** | Measures how well a model can sort policies from low-risk to high-risk. 0 = no sorting ability, 1 = perfect. |
# MAGIC | **AIC** | Akaike Information Criterion — a score for model quality that penalises unnecessary complexity. Lower is better. |
# MAGIC | **BIC** | Bayesian Information Criterion — similar to AIC but with a stricter penalty for complexity. Lower is better. |
# MAGIC | **Deviance** | A technical measure of how far the model's predictions are from reality. Lower deviance = better fit. |
# MAGIC | **Deviance explained** | The percentage of variation in claims that the model can account for. Higher is better. |
# MAGIC | **Adverse selection** | When competitors use better data, they attract your best (cheapest) customers and leave you with the worst (most expensive). |
# MAGIC | **Pure premium** | The expected claims cost per policy, before any loading for expenses or profit. |
# MAGIC | **Expense loading** | A multiplier applied to the pure premium to cover business costs (admin, commissions, profit margin). |
# MAGIC | **Enrichment data** | Additional data sourced from third parties (e.g., flood maps, crime databases) that improves pricing accuracy. |
# MAGIC | **MAE** | Mean Absolute Error — the average size of prediction errors. Lower is better. |
# MAGIC | **RMSE** | Root Mean Squared Error — like MAE but penalises large errors more. Lower is better. |
# MAGIC | **Decile** | One of 10 equal-sized groups, used to compare performance across the range of predictions. |
# MAGIC | **Log link** | A mathematical function ensuring predictions are always positive (you can't have negative claims). |
# MAGIC | **Subsidence** | When the ground under a building sinks, often causing structural damage. Very expensive to repair. |
