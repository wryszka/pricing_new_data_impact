# Demo Track — New Data Impact on Insurance Pricing

A working guide for delivering this demo live. Covers prep, flow, talking points,
and common Q&A. Tune the length by picking which notebooks to show.

---

## Audience variants at a glance

| Audience | Length | Notebooks | Dashboard | Focus |
|---|---|---|---|---|
| **Executive / business** | 15 min | 03 only | Yes | Pricing story in plain English |
| **Actuaries / pricing team** | 30 min | 01 (skim) → 02 → (05 briefly) | Optional | Methodology + interactive Q&A |
| **Governance / risk / regulatory** | 20 min | 00 → 04 → PDF export | No | Audit trail + sign-off flow |
| **Full technical deep-dive** | 45 min | 00 → 00a → 01 → 02 → 05 | Optional | Whole pipeline end-to-end |

---

## Before the demo

**Do these once before the session:**

1. Confirm the workspace is alive: `https://fevm-lr-serverless-aws-us.cloud.databricks.com`
2. Verify the `postcode_enrichment` table exists (built by **00a**). Takes ~2 minutes if you need to rebuild.
3. Run **01_build_all_models** to refresh models and metrics. Takes ~10 minutes. All downstream notebooks read from its UC tables, so you don't need to rerun them.
4. Open the Lakeview dashboard once to let it warm up (removes the first-load lag).
5. Have the GitHub repo open in another tab: `https://github.com/wryszka/pricing_new_data_impact`

**Hero numbers to memorise** (from the latest run):

- Claim rate: **~16%** (observed; ABI-realistic for UK home)
- Avg severity: **~£2,900** (observed; ABI-realistic)
- Frequency: Deviance Explained **1.0% → 5.3%** (5× improvement)
- Frequency: Gini **0.11 → 0.25** (2.3× improvement)
- Severity: MAE **£951 → £829** (-13%)

---

## The narrative arc

The whole demo tells one story in three acts:

1. **Setup the question.** You already have years of internal data — policies, claims, customer records. Does adding *external* data (postcode-level public data) make your pricing meaningfully better?
2. **Show the answer.** We took real UK public data, sampled real English postcodes into a 200k-policy portfolio, trained standard vs enriched frequency + severity models, and measured lift on the same holdout.
3. **Show the proof.** Both models pass the usual actuarial checks. The enriched model has materially better Gini, better loss-ratio stability, and sensible coefficients. We can also generate a full governance-grade PDF report in one click.

The question behind the question an actuary will ask: **"Is this reproducible, auditable, and safe to put in front of a regulator?"** Every notebook is designed to answer *yes*.

---

## Opening (90 sec — use every session)

> "Before we show anything — the question we're answering isn't *'does Databricks work'*. You've seen that. The question is: **can external data meaningfully improve your pricing models, and can we do it in a way your model governance team will actually sign off on?**
>
> We've taken real UK public data — the ONS postcode directory, the English Indices of Deprivation, the ONS rural-urban classification — and built a reusable enrichment layer covering 1.5 million English postcodes. Every policy in our demo portfolio inherits real features from its actual location. We then train two frequency and two severity models — standard rating factors vs standard + enrichment — and measure the lift.
>
> Claims are simulated, because real claims data isn't public. But everything upstream is real, and the methodology drops directly onto a real book of business."

---

## Demo track — the five-notebook flow

### Notebook 00 — Model Overview (optional, 2 min)

**Skip for executives.** Useful for technical audiences as orientation.

- Open it, scroll to the "Notebook Guide" table.
- Point out: *"We've got five notebooks. One-time build, two walkthroughs, one governance doc, one AI agent."*
- Don't run any cells. Move on.

### Notebook 00a — Build Postcode Enrichment (optional, 3 min)

**Show to technical audiences who care about data engineering.**

- Scroll through the markdown — highlight the data sources (ONSPD, IMD 2019, ONS RUC).
- Point at the `postcode_enrichment` table in Catalog Explorer: ~1.5M rows, 16 columns.
- Talking point: *"This is the reusable piece. One insurer, one industry, one table. Any UK home/motor/health insurer can plug their policy data on top of this."*
- Don't rerun it live — the download alone takes ~2 min.

### Notebook 01 — Build All Models (run once before, skim in demo, ~5 min)

- Jump to **Section 2c — Simulate claims**. Show the DGP formula. Talking point: *"We calibrate to published UK market statistics — ~15% claim rate, ~£2,600 average severity, both in line with ABI figures. The enrichment features drive both frequency and severity, and the baseline model can't see them."*
- Jump to **Section 5 — Train Frequency GLMs**. Show the output: Gini 0.11 (standard) → 0.25 (enriched). Pause here — this is the money shot.
- Jump to **Section 9 — Model Factory**. Talking point: *"Instead of manually trying 5 model variants in Radar, we programmatically train 50 Poisson GLM specifications in under a minute. Every spec is ranked by AIC, BIC, and Gini. Full audit trail in Unity Catalog."*
- Don't linger. The point is that the build is fast, reproducible, and persisted.

### Notebook 02 — Results Technical (15 min — the core for technical audiences)

**This is the main event for actuaries.**

1. **Portfolio overview** — distribution of key features, claim rate, average severity.
2. **Model Comparison table** (Section 2). Read the AIC/BIC/Gini/Deviance Explained numbers out loud.
3. **Coefficient Analysis** (Section 3). Show the forest plot. Point at:
   - `crime_decile` — negative coefficient (low decile = high crime = more claims). Significant at p<0.001.
   - `is_coastal` — positive. Coastal → more claims.
   - `imd_decile` — negative. More deprived → more claims.
   Talking point: *"These are all directionally correct by any UK pricing actuary's intuition. And they're all statistically significant."*
4. **Loss Ratio by Decile** (Section 4). The enriched model's LR stays much closer to 1.0 across deciles. This is the point where business listeners "get it" — tighter LR = fairer pricing, less adverse selection.
5. **Feature Importance** (Section 7, severity). The GBM tells us which real features drive severity. Usually `imd_decile` and `is_coastal` lead.
6. **Model Factory elbow chart** (Section 8). AIC vs feature count. Point at the bend: *"This is how you answer 'how many features is enough?' in one chart. Radar makes you guess manually."*
7. **Model serving** (Section 10). Load the registered enriched model from UC, score five new policies. Talking point: *"That model is versioned, lineaged, and servable as a REST endpoint the minute governance approves it."*

### Notebook 03 — Results Executive (15 min — the whole demo for business audiences)

**Use this as the ENTIRE demo for non-technical audiences.**

Read through it like a story — the notebook is written as a continuous ELI5 narrative with a glossary. Key beats:

1. *"What data does each model use?"* — make the asymmetry concrete.
2. *"The scoreboard"* — show the model comparison table, then stop. The numbers speak.
3. *"How do the models price crime-deprived postcodes?"* — the bar chart where the standard model is flat and the enriched one tracks actual losses. This is where non-technical people have the "oh" moment.
4. *"Loss ratio by decile"* — stability story. Standard = volatile, Enriched = steady.
5. *"Coastal properties"* — same story, different feature.
6. *"What this means for the business"* — four concrete benefits. Don't read them verbatim, summarise.

### Notebook 04 — Model Governance (10 min — for risk/compliance audiences)

1. Run Sections 1-6 — inventory, rationale, feature justification, evidence, sensitivity. Let the charts do the talking.
2. Jump to **Section 9 — Generate PDF Report**. Run the cell. Show the output path in the UC volume. *Open the PDF live* — this is the "click moment" governance audiences love.
3. Point at the **sign-off section** — roles, checklist, version history. Talking point: *"Every row in here is populated from Unity Catalog tables. The report regenerates on demand. You never have to ask the team to rewrite the governance document — the document is the pipeline."*

### Notebook 05 — Model Review Agent (5 min — optional closer for actuaries)

1. Run the example queries (they're already in the notebook). Read one response out loud.
2. Then type a new query live — e.g. `query_agent("What happens if the regulator bans us from using crime_decile? Show the three best alternative models.")`
3. Talking point: *"This is your junior actuary on demand. It reasons over all 50 models and gives you a defensible answer in the time it takes to type the question. Every call is audit-logged."*

### Lakeview dashboard (5 min — any audience)

- Executive Summary page → scoreboard.
- Pricing Distribution → scatter plot of standard vs enriched quotes.
- Risk Segmentation → shows the pricing difference by crime decile / coastal / urban.

Talking point: *"For self-service, you don't have to open a notebook at all. Same underlying tables, different audience."*

---

## Closing (60 sec)

> "The pattern we just walked through — **real external data → automated enrichment → side-by-side model comparison → governance sign-off** — drops onto any real book of business. The code is in GitHub. The catalog is in your workspace. The models are MLflow-registered. The audit trail is Unity Catalog. Nothing I showed you is a slide.
>
> Next step: swap the synthetic policies for a real portfolio slice, rerun the pipeline, and see the same story on your data."

---

## Q&A prep

**Q: Why not use real claims data?**
A: UK insurers don't publish it. Our framing: real features, simulated claims calibrated to ABI market statistics. The methodology is what's portable — the specific lift will depend on the insurer's actual experience.

**Q: How do we extend this to Scotland/Wales/NI?**
A: Different deprivation indices (SIMD, WIMD, NIMDM). Structurally the same pipeline, just need to add the regional enrichment sources to 00a. Maybe half a day of work.

**Q: Can this replace Radar/Emblem?**
A: Not trying to. Position as complementary — Databricks does the data engineering and enrichment layer, feeds curated features into your existing Radar/Emblem workflow if you want.

**Q: What about regulatory scrutiny on new variables (e.g. crime, IMD)?**
A: Good question — that's exactly why the governance notebook exists. Every feature has a documented business rationale, p-value, and confidence interval. Model factory shows you alternative specs if one feature becomes impermissible.

**Q: What's the scale limit?**
A: Current demo: 200k policies, 50 model variants, under 10 min end-to-end on serverless. Scaling to 2M+ policies works but requires Spark ML's GLM instead of statsmodels for the frequency model.

**Q: Can the agent actually train models, or just describe them?**
A: Describe only. It's an advisor, not an actor. Humans still approve and deploy. Every interaction is audit-logged.

**Q: How fresh is the data?**
A: ONSPD refreshes quarterly; IMD 2019 is current (next release 2025). Notebook 00a is idempotent — rerun when source data updates.

---

## Troubleshooting mid-demo

- **Notebook 01 is slow** — you probably didn't prewarm. Reference the UC tables that already exist from your last successful run.
- **Dashboard shows "no data"** — the underlying tables haven't been refreshed since the last 01 run. Rerun 01.
- **PDF cell fails** — typically a Unicode character slipped into a new feature rationale. `_ascii()` in notebook 04 handles most cases.
- **Agent returns an error** — the Foundation Model API endpoint may be unavailable; skip to the next scenario and come back.
- **Workspace token expired** — say "this is my demo environment, not prod" and reauth in another tab.

---

## What NOT to do

- Don't promise retraining happens in seconds on real data. Be explicit: real retraining is batched, overnight.
- Don't compare lift numbers to the audience's own book without heavy caveats — our DGP is calibrated to market averages, not their specific experience.
- Don't open every notebook. Pick the three that fit the audience and own them.
- Don't read the notebooks out loud — paraphrase. Reading word-for-word feels mechanical.
