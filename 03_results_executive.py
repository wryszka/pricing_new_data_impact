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
# MAGIC - **Model 2 — Enriched** adds real public UK data about **where each property is located**:
# MAGIC   area deprivation, neighbourhood crime levels, living-environment quality, urban vs rural,
# MAGIC   and whether the postcode sits on the coast. All of it sourced from official government
# MAGIC   datasets and joined on each policy's postcode.
# MAGIC
# MAGIC This notebook tells the full story — from raw data to final pricing — and shows exactly what the new data buys us.
# MAGIC
# MAGIC **Geographic note:** This analysis covers **England only**. Scotland has its own
# MAGIC deprivation index (SIMD), Wales uses WIMD, and Northern Ireland uses NIMDM — these are
# MAGIC separate datasets that would be added in a full production rollout.

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
# MAGIC claim costs are messy and vary enormously (a small theft claim might cost £500;
# MAGIC a major escape-of-water or storm-damage claim can run to £20,000+).
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
# MAGIC We built a portfolio of **200,000 home insurance policies**. Each one is assigned a real
# MAGIC English postcode, and inherits its real public-data location profile from that postcode.
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
# MAGIC | `postcode` | Real UK postcode — used to look up the location features below |
# MAGIC | `imd_decile` | How deprived is the area? 1 = most deprived 10% in England, 10 = least |
# MAGIC | `crime_decile` | How much crime does the area have? 1 = worst 10%, 10 = safest |
# MAGIC | `income_decile` | How income-deprived is the area? 1 = most deprived |
# MAGIC | `health_decile` | How health-deprived is the area? 1 = worst health outcomes |
# MAGIC | `living_env_decile` | Quality of the living environment — housing condition, air quality etc. 1 = worst |
# MAGIC | `is_urban` | Is the postcode in an urban area? 1 = urban, 0 = rural |
# MAGIC | `is_coastal` | Does the postcode sit in a coastal local authority? 1 = coastal, 0 = inland |
# MAGIC | `region_name` | Which of England's 9 government regions (London, South West, etc.) |
# MAGIC | `num_claims` | How many claims were made on this policy (what the frequency model predicts) |
# MAGIC | `total_loss` | Total cost of those claims in pounds |

# COMMAND ----------

import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Use spark.sql to work around Photon column-index bug on the portfolio schema
portfolio = spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.portfolio")

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
# MAGIC | Construction — Brick, timber, stone? | Area deprivation — how deprived is the postcode overall (IMD decile 1–10) |
# MAGIC | Building age — How old is the property? | Crime level — is the neighbourhood high-crime or low-crime? (decile 1–10) |
# MAGIC | Bedrooms — How many bedrooms? | Income deprivation — how income-deprived is the area? |
# MAGIC | Sum insured — Rebuild cost of the property | Health deprivation — how is the area's general health? |
# MAGIC | Occupancy — Owner-occupied or rented? | Living-environment quality — housing stock, air quality |
# MAGIC | Prior claims — How claims-happy is the customer? | Urban vs rural — is the postcode in a city/town or the countryside? |
# MAGIC | Policy tenure — How long have they been with us? | Coastal — does the postcode sit on the English coast? |
# MAGIC | | Region — which of England's 9 regions (London, South West, etc.) |
# MAGIC
# MAGIC ### Why do these extra features matter?
# MAGIC
# MAGIC Imagine two identical houses — same size, same age, same construction. But one sits in
# MAGIC a deprived inner-city postcode with high crime, and the other sits in a leafy village.
# MAGIC **They have very different risk profiles**, but Model 1 has no way of knowing that.
# MAGIC It would charge them the same price.
# MAGIC
# MAGIC Model 2 can see the crime deprivation, overall deprivation, and whether the postcode
# MAGIC is on the coast (where weather damage is more likely) — so it charges a price that
# MAGIC actually reflects reality. That's fairer for the customer and better for the business.
# MAGIC
# MAGIC ### Where does this "new data" come from?
# MAGIC
# MAGIC It's all public, free UK government data:
# MAGIC
# MAGIC - **IMD 2019** — the Ministry of Housing, Communities & Local Government's English
# MAGIC   Indices of Deprivation. Produces seven deprivation deciles for every neighbourhood
# MAGIC   in England. We use five of them: overall, crime, income, health, and living environment.
# MAGIC - **ONSPD** — the Office for National Statistics Postcode Directory. Links every UK
# MAGIC   postcode to its neighbourhood, local authority, and region.
# MAGIC - **ONS Rural-Urban Classification** — official urban/rural flag for every postcode.
# MAGIC - **Coastal local authorities** — list of English local authorities with coastline.

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
# MAGIC ### How the Models Price Crime-Deprived Postcodes
# MAGIC
# MAGIC Model 1 cannot see the area's crime level. Model 2 can. Let's see what difference that makes.
# MAGIC
# MAGIC The **crime decile** puts every English neighbourhood into one of ten bands. **Decile 1 =
# MAGIC the worst 10% for crime**, decile 10 = the safest 10%. High-crime postcodes drive more
# MAGIC theft and malicious-damage claims.

# COMMAND ----------

priced = spark.table(f"{CATALOG}.{SCHEMA}.priced_portfolio").toPandas()

crime_comparison = priced.groupby("crime_decile").agg(
    avg_quote_model1=("quote_standard", "mean"),
    avg_quote_model2=("quote_enriched", "mean"),
    avg_actual_loss=("actual_loss", "mean"),
    policy_count=("actual_loss", "count"),
).reset_index()

crime_comparison = crime_comparison.rename(columns={
    "avg_quote_model1": "Avg Quote — Model 1 (£)",
    "avg_quote_model2": "Avg Quote — Model 2 (£)",
    "avg_actual_loss": "Avg Actual Loss (£)",
    "policy_count": "Number of Policies",
    "crime_decile": "Crime Decile",
})

print("Average premium quote by crime decile:")
display(spark.createDataFrame(crime_comparison))

fig, ax = plt.subplots(figsize=(10, 6))

x = crime_comparison["Crime Decile"].astype(int)
width = 0.25

ax.bar(x - width, crime_comparison["Avg Quote — Model 1 (£)"], width,
       label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(x, crime_comparison["Avg Quote — Model 2 (£)"], width,
       label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar(x + width, crime_comparison["Avg Actual Loss (£)"], width,
       label="Actual Average Loss", color="#43A047", alpha=0.85)

ax.set_xlabel("Crime Decile (1 = highest crime → 10 = lowest crime)", fontsize=12)
ax.set_ylabel("Amount (£)", fontsize=12)
ax.set_title("How Do the Models Price Crime Deprivation? (Frequency)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xticks(list(range(1, 11)))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 charges roughly the same price regardless of crime level** — it simply cannot
# MAGIC see that data. Whether your house is on a quiet cul-de-sac or a street with heavy theft
# MAGIC activity, you get a similar quote.
# MAGIC
# MAGIC **Model 2 correctly charges more for high-crime postcodes.** The blue bars track
# MAGIC much more closely to the green bars (actual losses). That's what good pricing looks like.
# MAGIC
# MAGIC For the business, this means:
# MAGIC - We stop **underpricing** high-crime postcodes (saving money on theft and malicious-damage claims).
# MAGIC - We stop **overpricing** low-crime postcodes (keeping good customers from walking).

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Price Coastal Properties
# MAGIC
# MAGIC **Coastal postcodes** are more exposed to weather-related damage — wind, driving rain,
# MAGIC flooding, and salt-related corrosion. Many also sit in older housing stock where water
# MAGIC ingress is a bigger problem. This is a real and well-documented risk factor in home
# MAGIC insurance pricing.

# COMMAND ----------

coastal_comparison = priced.groupby("is_coastal").agg(
    avg_quote_model1=("quote_standard", "mean"),
    avg_quote_model2=("quote_enriched", "mean"),
    avg_actual_loss=("actual_loss", "mean"),
    policy_count=("actual_loss", "count"),
).reset_index()

coastal_comparison["is_coastal"] = coastal_comparison["is_coastal"].map(
    {0: "Inland", 1: "Coastal"}
)

coastal_comparison = coastal_comparison.rename(columns={
    "avg_quote_model1": "Avg Quote — Model 1 (£)",
    "avg_quote_model2": "Avg Quote — Model 2 (£)",
    "avg_actual_loss": "Avg Actual Loss (£)",
    "policy_count": "Number of Policies",
    "is_coastal": "Location",
})

print("Average premium quote — coastal vs inland:")
display(spark.createDataFrame(coastal_comparison))

inland = coastal_comparison[coastal_comparison["Location"] == "Inland"]
coastal = coastal_comparison[coastal_comparison["Location"] == "Coastal"]

m1_diff = coastal["Avg Quote — Model 1 (£)"].values[0] - inland["Avg Quote — Model 1 (£)"].values[0]
m2_diff = coastal["Avg Quote — Model 2 (£)"].values[0] - inland["Avg Quote — Model 2 (£)"].values[0]
actual_diff = coastal["Avg Actual Loss (£)"].values[0] - inland["Avg Actual Loss (£)"].values[0]

print(f"\nPrice difference for coastal properties:")
print(f"  Model 1 charges £{m1_diff:+,.2f} extra  (it can barely tell the difference)")
print(f"  Model 2 charges £{m2_diff:+,.2f} extra  (it sees the risk)")
print(f"  Actual losses are £{actual_diff:+,.2f} higher  (this is the reality we need to price for)")

fig, ax = plt.subplots(figsize=(9, 6))

categories = coastal_comparison["Location"]
x_pos = range(len(categories))
width = 0.25

ax.bar([p - width for p in x_pos], coastal_comparison["Avg Quote — Model 1 (£)"],
       width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(list(x_pos), coastal_comparison["Avg Quote — Model 2 (£)"],
       width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar([p + width for p in x_pos], coastal_comparison["Avg Actual Loss (£)"],
       width, label="Actual Average Loss", color="#43A047", alpha=0.85)

ax.set_xlabel("Location", fontsize=12)
ax.set_ylabel("Amount (£)", fontsize=12)
ax.set_title("How Do the Models Price Coastal vs Inland? (Frequency)", fontsize=14, fontweight="bold")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 has no idea whether the property is on the coast.** It charges nearly the
# MAGIC same price inland or on the seafront. But the actual claims are significantly higher
# MAGIC for coastal properties.
# MAGIC
# MAGIC **Model 2 increases the price for coastal properties**, bringing the quote much closer
# MAGIC to the real cost. Without this adjustment, we'd be systematically losing money on every
# MAGIC coastal policy we insure.

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
# MAGIC Claim costs are messy — a small theft claim costs £500 but a major escape-of-water or
# MAGIC storm-damage claim can run to £20,000+. A GBM is better at spotting these complex
# MAGIC patterns, especially when they depend on combinations of factors (e.g., old timber
# MAGIC coastal property in a deprived postcode = much more expensive than average).
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
print(f"    (major escape-of-water, storm damage, large theft) pull the average up.")
print(f"    This is completely normal in insurance.")

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
# MAGIC - Claims in the most deprived postcodes cost more — poorer maintenance, older housing,
# MAGIC   higher repair costs relative to property value
# MAGIC - Coastal claims tend to be larger — water ingress, storm damage, salt-related corrosion
# MAGIC - Theft claims in high-crime areas are larger on average — organised thefts, more valuable items taken
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
# MAGIC In **Model 2**, `imd_decile` (area deprivation), `is_coastal`, and `crime_decile`
# MAGIC shoot up the rankings. These are the features that really explain *why* some claims
# MAGIC cost so much more than others. The model can now see what was previously invisible.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Predict Claim Costs by Deprivation Decile
# MAGIC
# MAGIC We saw that Model 1 can't price neighbourhood effects for frequency. The same is true
# MAGIC for severity — and arguably the impact is even bigger, because claims in deprived areas
# MAGIC are **meaningfully more expensive** than in affluent ones.

# COMMAND ----------

priced_sev = spark.table(f"{CATALOG}.{SCHEMA}.severity_priced_portfolio").toPandas()

imd_seg = priced_sev.groupby("imd_decile").agg(
    avg_sev_model1=("sev_pred_standard", "mean"),
    avg_sev_model2=("sev_pred_enriched", "mean"),
    avg_actual_sev=("claim_severity", "mean"),
    n=("actual_loss", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(imd_seg))
width = 0.25

ax.bar(x - width, imd_seg["avg_sev_model1"], width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(x, imd_seg["avg_sev_model2"], width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar(x + width, imd_seg["avg_actual_sev"], width, label="Actual Average Cost", color="#43A047", alpha=0.85)

ax.set_xlabel("IMD Decile (1 = most deprived → 10 = least deprived)", fontsize=12)
ax.set_ylabel("Average Claim Cost (£)", fontsize=12)
ax.set_title("How Do the Models Predict Claim Costs by Deprivation? (Severity)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(d)}" for d in imd_seg["imd_decile"]])
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Model 1 predicts roughly the same claim cost regardless of deprivation** — it can't
# MAGIC see the data. But in reality, claims in the most deprived decile cost significantly
# MAGIC more than claims in the least deprived decile.
# MAGIC
# MAGIC **Model 2 correctly predicts higher costs in deprived postcodes.** The blue bars track
# MAGIC much more closely to the green bars (actual costs).

# COMMAND ----------

# MAGIC %md
# MAGIC ### How the Models Predict Coastal Claim Costs
# MAGIC
# MAGIC Coastal claims are typically larger than inland claims — wind-driven rain, storm damage,
# MAGIC and salt corrosion all push repair bills up. Let's see how the severity models handle it.

# COMMAND ----------

coast_seg = priced_sev.groupby("is_coastal").agg(
    avg_sev_model1=("sev_pred_standard", "mean"),
    avg_sev_model2=("sev_pred_enriched", "mean"),
    avg_actual_sev=("claim_severity", "mean"),
    n=("actual_loss", "count"),
).reset_index()
coast_seg["is_coastal"] = coast_seg["is_coastal"].map({0: "Inland", 1: "Coastal"})

fig, ax = plt.subplots(figsize=(9, 6))
x_pos = range(len(coast_seg))
width = 0.25

ax.bar([p - width for p in x_pos], coast_seg["avg_sev_model1"], width, label="Model 1 (Standard)", color="#E53935", alpha=0.85)
ax.bar(list(x_pos), coast_seg["avg_sev_model2"], width, label="Model 2 (Enriched)", color="#1E88E5", alpha=0.85)
ax.bar([p + width for p in x_pos], coast_seg["avg_actual_sev"], width, label="Actual Average Cost", color="#43A047", alpha=0.85)

ax.set_xlabel("Location", fontsize=12)
ax.set_ylabel("Average Claim Cost (£)", fontsize=12)
ax.set_title("How Do the Models Predict Coastal Claim Costs? (Severity)", fontsize=14, fontweight="bold")
ax.set_xticks(list(x_pos))
ax.set_xticklabels(coast_seg["is_coastal"], fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

m1_diff = coast_seg.iloc[1]["avg_sev_model1"] - coast_seg.iloc[0]["avg_sev_model1"]
m2_diff = coast_seg.iloc[1]["avg_sev_model2"] - coast_seg.iloc[0]["avg_sev_model2"]
actual_diff = coast_seg.iloc[1]["avg_actual_sev"] - coast_seg.iloc[0]["avg_actual_sev"]

print(f"Predicted cost difference for coastal properties:")
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
# MAGIC The effect is **multiplicative**. A property in a deprived, high-crime, coastal postcode
# MAGIC claims **more often** AND each claim **costs more**. Under the standard model, we miss
# MAGIC both effects. Under the enriched model, we capture both — and the two corrections
# MAGIC multiply together into a significantly more accurate final price.
# MAGIC
# MAGIC | Component | Without new data | With new data |
# MAGIC |---|---|---|
# MAGIC | **Frequency** (how often) | Misses deprivation, crime and coastal exposure | Prices frequency accurately |
# MAGIC | **Severity** (how much) | Flat average — all claims treated equally | Knows coastal and deprived-area claims cost far more |
# MAGIC | **Combined quote** | Over/underprices by large margins | Tight, risk-adequate pricing |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # What This Means for the Business
# MAGIC
# MAGIC ### The big story
# MAGIC
# MAGIC > **Adding real, public data about where each property is located — deprivation, crime,
# MAGIC > urban/rural, coastal, region — makes our pricing significantly more accurate. And
# MAGIC > because it's all free government data, the only cost is the pipeline that joins it.**
# MAGIC
# MAGIC Adding new data sources improves **both halves** of the pricing equation and delivers
# MAGIC four concrete benefits:
# MAGIC
# MAGIC ### 1. Better Risk Selection
# MAGIC We can avoid underpricing the riskiest postcodes. Right now, a house on a high-crime,
# MAGIC deprived street gets the same price as one in a leafy village. With the enriched model,
# MAGIC we charge appropriately — or decline if the risk is too high.
# MAGIC
# MAGIC ### 2. Fairer Pricing
# MAGIC Low-risk customers pay less, high-risk customers pay more. This is fairer for everyone
# MAGIC and makes our product more competitive for the customers we actually want.
# MAGIC
# MAGIC ### 3. Less Adverse Selection
# MAGIC If our competitors are already using IMD, crime, and coastal data (and most are),
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
# MAGIC | **Rating factor / Feature** | A piece of information used by the model to calculate a price (e.g., building age, IMD decile, coastal flag). |
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
# MAGIC | **Enrichment data** | Additional data joined onto each policy (e.g., IMD deciles, urban/rural flag) that improves pricing accuracy. In this demo, sourced entirely from free UK government open data. |
# MAGIC | **IMD** | English **Indices of Multiple Deprivation 2019** — the government's official measure of neighbourhood deprivation in England. Publishes separate deciles for overall deprivation, crime, income, health, living environment, and more. |
# MAGIC | **Decile** (IMD context) | One of 10 bands that every English neighbourhood is sorted into. Decile 1 = the most deprived 10%; decile 10 = the least deprived 10%. We follow the IMD convention exactly. |
# MAGIC | **LSOA** | **Lower-layer Super Output Area** — a small statistical neighbourhood (avg. ~1,500 residents) used to publish IMD and other ONS statistics. Every postcode maps to exactly one LSOA. |
# MAGIC | **ONSPD** | **Office for National Statistics Postcode Directory** — free UK government file that links every postcode to its LSOA, local authority, region, and grid reference. |
# MAGIC | **ONS RUC** | **Office for National Statistics Rural-Urban Classification** — official flag classifying every area as urban or rural. |
# MAGIC | **Coastal** | A postcode whose local authority has coastline. Coastal properties face more weather-related exposure (wind, driving rain, storm surge). |
# MAGIC | **Region** | One of England's 9 government regions: London, South East, South West, East of England, East Midlands, West Midlands, Yorkshire and The Humber, North West, North East. |
