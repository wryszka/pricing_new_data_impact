# Databricks notebook source
# MAGIC %md
# MAGIC # Build Postcode Enrichment Table
# MAGIC ## Real UK Public Data -> Unity Catalog
# MAGIC
# MAGIC This notebook builds a **reusable postcode-level enrichment table** from public UK data
# MAGIC sources. It is the foundation for the new-data-impact demo, replacing the synthetic
# MAGIC features with real UK geography.
# MAGIC
# MAGIC **Scope:** England only (Scotland/Wales/NI use different deprivation indices).
# MAGIC
# MAGIC **Run once** — the output table is stable and only needs refreshing when source data is updated.
# MAGIC
# MAGIC | Source | What it gives us | Size |
# MAGIC |---|---|---|
# MAGIC | **ONS Postcode Directory (ONSPD)** | Live UK postcodes + lat/lon + LSOA + region | ~2.3 GB CSV |
# MAGIC | **English Indices of Deprivation 2019** | IMD decile + crime/income/health domain deciles (LSOA-level) | ~5 MB CSV |
# MAGIC
# MAGIC All datasets are Open Government Licence (OGL) — free for any use including commercial.
# MAGIC
# MAGIC **Output:** `lr_serverless_aws_us_catalog.pricing_new_data_impact.postcode_enrichment`
# MAGIC (~1.3M England live residential postcodes with ~12 enrichment features)

# COMMAND ----------

# MAGIC %pip install requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

CATALOG = "lr_serverless_aws_us_catalog"
SCHEMA = "pricing_new_data_impact"
VOLUME = "raw_data"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

import os
import zipfile
import time
import requests
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

print(f"Raw data volume: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download Sources
# MAGIC
# MAGIC Downloads are idempotent — if the file already exists in the volume, we skip.

# COMMAND ----------

def download_if_missing(url, dest_path, description):
    """Stream a download to the target path if it doesn't already exist."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[skip] {description} already present ({size_mb:,.1f} MB)")
        return dest_path

    print(f"[download] {description}")
    print(f"  From: {url}")
    print(f"  To:   {dest_path}")
    t0 = time.time()

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total > 0:
                        pct = 100 * written / total
                        if written % (64 * 1024 * 1024) < 8 * 1024 * 1024:  # every ~64 MB
                            print(f"  ...{written/(1024*1024):>7.0f} MB ({pct:>5.1f}%)")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[done] {description}: {size_mb:,.1f} MB in {elapsed:.1f}s")
    return dest_path


# --- 1a. ONSPD (August 2025 release) ---
ONSPD_URL = "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/cfd03a224ae24db483f89051c35dac29/csv?layers=0"
ONSPD_PATH = f"{VOLUME_PATH}/onspd.csv"

# --- 1b. IMD 2019 File 7 (all ranks, deciles, scores) ---
IMD_URL = "https://assets.publishing.service.gov.uk/media/5dc407b440f0b6379a7acc8d/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
IMD_PATH = f"{VOLUME_PATH}/imd2019_file7.csv"

download_if_missing(ONSPD_URL, ONSPD_PATH, "ONSPD (UK postcodes)")
download_if_missing(IMD_URL, IMD_PATH, "IMD 2019 File 7")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load & Filter ONSPD (England, Live Postcodes Only)
# MAGIC
# MAGIC The full ONSPD is ~2.9M rows with ~50 columns covering all UK postcodes (live + terminated).
# MAGIC We filter to:
# MAGIC - **Country code E92000001** (England only)
# MAGIC - **`doterm` is null** (live postcodes only — `doterm` is the termination date)
# MAGIC
# MAGIC This leaves ~1.3M England live postcodes.

# COMMAND ----------

# Load ONSPD with minimal schema inference — we just need a handful of columns
onspd_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("nullValue", "")
    .csv(ONSPD_PATH)
)

# Normalize all column names to lowercase (ArcGIS hosted tables often use uppercase)
onspd_raw = onspd_raw.toDF(*[c.lower() for c in onspd_raw.columns])

total_rows = onspd_raw.count()
print(f"ONSPD total rows (all UK, live + terminated): {total_rows:,}")
print(f"ONSPD columns (sample): {onspd_raw.columns[:15]}")

print("All ONSPD columns:")
for c in onspd_raw.columns:
    print(f"  {c!r}")

# Resolve column names flexibly — the hosted-table export uses long descriptive names,
# the raw ONSPD ZIP uses short codes. We try both.
def find_onspd_col(df, *substrings_all, exact=None):
    """
    Find a column name.
    - exact: list of exact names to try first (case-insensitive).
    - substrings_all: tuples — each tuple must ALL match as substrings for the column to qualify.
    Returns the first matching column.
    """
    existing_lower = {c.lower(): c for c in df.columns}
    # Try exact first
    for name in (exact or []):
        if name.lower() in existing_lower:
            return existing_lower[name.lower()]
    # Try substring combinations (each element is a tuple of substrings that must ALL be in the column)
    for combo in substrings_all:
        substrs = combo if isinstance(combo, tuple) else (combo,)
        for c in df.columns:
            lc = c.lower()
            if all(s.lower() in lc for s in substrs):
                return c
    raise KeyError(
        f"No column matches exact={exact} or substrings={substrings_all}. "
        f"First 30 columns: {df.columns[:30]}"
    )

col_postcode = find_onspd_col(onspd_raw,
                              ("postcode", "8 char"),
                              ("postcode", "egif"),
                              exact=["pcds", "pcd2"])
col_doterm   = find_onspd_col(onspd_raw,
                              ("date", "termination"),
                              exact=["doterm"])
col_ctry     = find_onspd_col(onspd_raw,
                              ("country code",),
                              exact=["ctry"])
col_rgn      = find_onspd_col(onspd_raw,
                              ("region code",),
                              exact=["rgn"])
col_oslaua   = find_onspd_col(onspd_raw,
                              ("local authority district code",),
                              exact=["oslaua"])
# Lat/long — hosted table may have separate lat/lon columns OR only grid references
try:
    col_lat  = find_onspd_col(onspd_raw, exact=["lat", "latitude"])
    col_long = find_onspd_col(onspd_raw, exact=["long", "longitude", "lng"])
    USE_GRID = False
except KeyError:
    col_lat  = find_onspd_col(onspd_raw, ("national grid", "northing"), exact=["oseast1m", "osnrth1m"])
    col_long = find_onspd_col(onspd_raw, ("national grid", "easting"),  exact=["osnrth1m", "oseast1m"])
    # Wait — easting/northing are integers, not lat/long. We'll convert after.
    col_east = find_onspd_col(onspd_raw, ("national grid", "easting"),  exact=["oseast1m"])
    col_north= find_onspd_col(onspd_raw, ("national grid", "northing"), exact=["osnrth1m"])
    USE_GRID = True

# LSOA — prefer 2011 (matches IMD 2019); fall back to 2021 with acceptable join loss
try:
    col_lsoa = find_onspd_col(onspd_raw,
                              ("lower", "super output", "2011"),
                              ("lower layer super output area", "2011"),
                              exact=["lsoa11"])
    LSOA_VERSION = "2011"
except KeyError:
    col_lsoa = find_onspd_col(onspd_raw,
                              ("lower", "super output", "2021"),
                              ("lower layer super output area", "2021"),
                              exact=["lsoa21"])
    LSOA_VERSION = "2021"

try:
    # Prefer 2011 classification (matches most LSOA-level work); fall back to 2021, then 2001
    col_ruind = find_onspd_col(onspd_raw,
                               ("rural urban", "2011"),
                               ("rural", "urban", "2011"),
                               ("rural urban", "2021"),
                               ("rural", "urban", "2021"),
                               exact=["ru11ind", "ru21ind"])
except KeyError:
    col_ruind = None

print(f"\nResolved columns:")
print(f"  postcode = {col_postcode!r}")
print(f"  doterm   = {col_doterm!r}")
print(f"  country  = {col_ctry!r}")
print(f"  region   = {col_rgn!r}")
print(f"  LA dist  = {col_oslaua!r}")
print(f"  LSOA ({LSOA_VERSION}) = {col_lsoa!r}")
print(f"  ru_ind   = {col_ruind!r}")
print(f"  coords   = {'grid (easting/northing)' if USE_GRID else 'lat/long'}")

print(f"Resolved columns: postcode={col_postcode}, ctry={col_ctry}, doterm={col_doterm}")

# Filter: England (E92000001) + live (doterm is null)
# Start with the country filter only — termination date may be empty string rather than null
filtered = onspd_raw.filter(F.col(col_ctry) == "E92000001")

# Handle "live" filter — doterm might be null or empty string
filtered = filtered.filter(
    (F.col(col_doterm).isNull()) | (F.trim(F.col(col_doterm)) == "")
)

# Build base selection
select_exprs = [
    F.col(col_postcode).alias("postcode"),
    F.col(col_lsoa).alias("lsoa_code"),
    F.col(col_rgn).alias("region_code"),
    F.col(col_oslaua).alias("local_authority_code"),
]

if col_ruind is not None:
    select_exprs.append(F.col(col_ruind).alias("urban_rural_code"))
else:
    select_exprs.append(F.lit(None).cast("string").alias("urban_rural_code"))

if USE_GRID:
    # Convert OSGB36 easting/northing -> approximate WGS84 lat/long using a Spark UDF.
    # For production use a proper projection library; this Helmert-style transform is
    # accurate to within a few metres, which is fine for a demo.
    @F.udf("double")
    def easting_northing_to_lat(e, n):
        try:
            e = float(e); n = float(n)
        except (TypeError, ValueError):
            return None
        # Airy 1830 ellipsoid -> OSGB36 -> Helmert transform -> WGS84
        import math
        a, b = 6377563.396, 6356256.909
        F0 = 0.9996012717
        phi0 = math.radians(49.0); lam0 = math.radians(-2.0)
        N0, E0 = -100000.0, 400000.0
        e2 = 1 - (b * b) / (a * a)
        n_ell = (a - b) / (a + b)
        # Iteratively solve for latitude
        phi = phi0 + (n - N0) / (a * F0)
        M = 0
        for _ in range(6):
            M = b * F0 * ((1 + n_ell + 5/4 * n_ell**2 + 5/4 * n_ell**3) * (phi - phi0)
                        - (3 * n_ell + 3 * n_ell**2 + 21/8 * n_ell**3) * math.sin(phi - phi0) * math.cos(phi + phi0)
                        + (15/8 * n_ell**2 + 15/8 * n_ell**3) * math.sin(2*(phi - phi0)) * math.cos(2*(phi + phi0))
                        - 35/24 * n_ell**3 * math.sin(3*(phi - phi0)) * math.cos(3*(phi + phi0)))
            phi += (n - N0 - M) / (a * F0)
        nu  = a * F0 / math.sqrt(1 - e2 * math.sin(phi)**2)
        rho = a * F0 * (1 - e2) / (1 - e2 * math.sin(phi)**2)**1.5
        eta2 = nu / rho - 1
        tp = math.tan(phi); sp = math.sin(phi); cp = math.cos(phi)
        secp = 1.0 / cp
        VII  = tp / (2 * rho * nu)
        VIII = tp / (24 * rho * nu**3) * (5 + 3 * tp**2 + eta2 - 9 * tp**2 * eta2)
        IX   = tp / (720 * rho * nu**5) * (61 + 90 * tp**2 + 45 * tp**4)
        dE = e - E0
        phi_rad = phi - VII * dE**2 + VIII * dE**4 - IX * dE**6
        return math.degrees(phi_rad)

    @F.udf("double")
    def easting_northing_to_long(e, n):
        try:
            e = float(e); n = float(n)
        except (TypeError, ValueError):
            return None
        import math
        a, b = 6377563.396, 6356256.909
        F0 = 0.9996012717
        phi0 = math.radians(49.0); lam0 = math.radians(-2.0)
        N0, E0 = -100000.0, 400000.0
        e2 = 1 - (b * b) / (a * a)
        n_ell = (a - b) / (a + b)
        phi = phi0 + (n - N0) / (a * F0)
        M = 0
        for _ in range(6):
            M = b * F0 * ((1 + n_ell + 5/4 * n_ell**2 + 5/4 * n_ell**3) * (phi - phi0)
                        - (3 * n_ell + 3 * n_ell**2 + 21/8 * n_ell**3) * math.sin(phi - phi0) * math.cos(phi + phi0)
                        + (15/8 * n_ell**2 + 15/8 * n_ell**3) * math.sin(2*(phi - phi0)) * math.cos(2*(phi + phi0))
                        - 35/24 * n_ell**3 * math.sin(3*(phi - phi0)) * math.cos(3*(phi + phi0)))
            phi += (n - N0 - M) / (a * F0)
        nu  = a * F0 / math.sqrt(1 - e2 * math.sin(phi)**2)
        rho = a * F0 * (1 - e2) / (1 - e2 * math.sin(phi)**2)**1.5
        tp = math.tan(phi); sp = math.sin(phi); cp = math.cos(phi)
        secp = 1.0 / cp
        X    = secp / nu
        XI   = secp / (6 * nu**3) * (nu / rho + 2 * tp**2)
        XII  = secp / (120 * nu**5) * (5 + 28 * tp**2 + 24 * tp**4)
        XIIA = secp / (5040 * nu**7) * (61 + 662 * tp**2 + 1320 * tp**4 + 720 * tp**6)
        dE = e - E0
        lam_rad = lam0 + X * dE - XI * dE**3 + XII * dE**5 - XIIA * dE**7
        return math.degrees(lam_rad)

    select_exprs.insert(1, easting_northing_to_lat(F.col(col_east),  F.col(col_north)).alias("lat"))
    select_exprs.insert(2, easting_northing_to_long(F.col(col_east), F.col(col_north)).alias("long"))
else:
    select_exprs.insert(1, F.col(col_lat).cast("double").alias("lat"))
    select_exprs.insert(2, F.col(col_long).cast("double").alias("long"))

onspd_eng = (
    filtered.select(*select_exprs)
    # Drop postcodes without valid coordinates
    .filter((F.col("lat").between(49.0, 61.0)) & (F.col("long").between(-8.0, 2.0)))
)

eng_rows = onspd_eng.count()
print(f"England live postcodes with valid coordinates: {eng_rows:,}")
display(onspd_eng.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load IMD 2019 & Extract LSOA-Level Deciles
# MAGIC
# MAGIC The IMD file has LSOA-level scores + deciles for several deprivation domains.
# MAGIC We keep the ones most relevant for insurance risk:
# MAGIC
# MAGIC | Column | What it means |
# MAGIC |---|---|
# MAGIC | `imd_decile` | Overall Index of Multiple Deprivation (1 = most deprived, 10 = least) |
# MAGIC | `imd_score` | Continuous IMD score |
# MAGIC | `crime_decile` | Crime domain decile |
# MAGIC | `income_decile` | Income domain decile |
# MAGIC | `health_decile` | Health deprivation and disability decile |
# MAGIC | `living_env_decile` | Living environment deprivation decile |

# COMMAND ----------

imd_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(IMD_PATH)
)

print(f"IMD LSOAs: {imd_raw.count():,}")
print("IMD columns (first 15):")
for c in imd_raw.columns[:15]:
    print(f"  {c!r}")

# COMMAND ----------

# The IMD file has long column names. We pick the relevant ones by substring matching
# for robustness across minor naming variants.
def find_col(df, *keywords):
    """Find a single column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        lc = c.lower()
        if all(k.lower() in lc for k in keywords):
            return c
    raise KeyError(f"No column matches {keywords}. Available: {df.columns[:5]}...")


lsoa_col         = find_col(imd_raw, "lsoa", "code")
imd_decile_col   = find_col(imd_raw, "index of multiple deprivation", "decile")
imd_score_col    = find_col(imd_raw, "index of multiple deprivation", "score")
crime_decile_col = find_col(imd_raw, "crime", "decile")
income_decile_col= find_col(imd_raw, "income", "decile")
health_decile_col= find_col(imd_raw, "health", "decile")
living_env_col   = find_col(imd_raw, "living environment", "decile")

imd_slim = imd_raw.select(
    F.col(lsoa_col).alias("lsoa_code"),
    F.col(imd_decile_col).cast("int").alias("imd_decile"),
    F.col(imd_score_col).cast("double").alias("imd_score"),
    F.col(crime_decile_col).cast("int").alias("crime_decile"),
    F.col(income_decile_col).cast("int").alias("income_decile"),
    F.col(health_decile_col).cast("int").alias("health_decile"),
    F.col(living_env_col).cast("int").alias("living_env_decile"),
)

display(imd_slim.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Derive Urban/Rural Classification
# MAGIC
# MAGIC ONSPD's `ru11ind` is a 2-character code. We map it to a human-readable band.
# MAGIC ONS 2011 Rural-Urban Classification (RUC):
# MAGIC - A1 = Urban major conurbation
# MAGIC - B1 = Urban minor conurbation
# MAGIC - C1 = Urban city and town
# MAGIC - C2 = Urban city and town in a sparse setting
# MAGIC - D1 = Rural town and fringe
# MAGIC - D2 = Rural town and fringe in a sparse setting
# MAGIC - E1 = Rural village
# MAGIC - E2 = Rural village in a sparse setting
# MAGIC - F1 = Rural hamlets and isolated dwellings
# MAGIC - F2 = Rural hamlets and isolated dwellings in a sparse setting

# COMMAND ----------

# RUC 2011 LSOA-level codes for England/Wales can appear as either:
#   Letter form: A1, B1, C1, C2, D1, D2, E1, E2 (urban = A/B/C, rural = D/E/F)
#   Numeric form: 1..8  (1-4 urban, 5-8 rural)
code_col = F.col("urban_rural_code")
onspd_eng = onspd_eng.withColumn(
    "urban_rural_band",
    # Letter form first
    F.when(code_col.startswith("A"), F.lit("Urban: Major Conurbation"))
     .when(code_col.startswith("B"), F.lit("Urban: Minor Conurbation"))
     .when(code_col.startswith("C"), F.lit("Urban: City and Town"))
     .when(code_col.startswith("D"), F.lit("Rural: Town and Fringe"))
     .when(code_col.startswith("E"), F.lit("Rural: Village"))
     .when(code_col.startswith("F"), F.lit("Rural: Hamlets"))
    # Numeric form
     .when(code_col == "1", F.lit("Urban: Major Conurbation"))
     .when(code_col == "2", F.lit("Urban: Minor Conurbation"))
     .when(code_col == "3", F.lit("Urban: City and Town"))
     .when(code_col == "4", F.lit("Urban: City and Town (sparse)"))
     .when(code_col == "5", F.lit("Rural: Town and Fringe"))
     .when(code_col == "6", F.lit("Rural: Town and Fringe (sparse)"))
     .when(code_col == "7", F.lit("Rural: Village"))
     .when(code_col == "8", F.lit("Rural: Village (sparse)"))
     .otherwise(F.lit("Unknown"))
).withColumn(
    "is_urban",
    F.when(code_col.substr(1, 1).isin("A", "B", "C"), F.lit(1))
     .when(code_col.isin("1", "2", "3", "4"), F.lit(1))
     .otherwise(F.lit(0))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Map Region Codes to Names

# COMMAND ----------

# ONS region codes (live England regions as of 2011 boundaries)
region_lookup = spark.createDataFrame([
    ("E12000001", "North East"),
    ("E12000002", "North West"),
    ("E12000003", "Yorkshire and The Humber"),
    ("E12000004", "East Midlands"),
    ("E12000005", "West Midlands"),
    ("E12000006", "East of England"),
    ("E12000007", "London"),
    ("E12000008", "South East"),
    ("E12000009", "South West"),
], ["region_code", "region_name"])

onspd_eng = onspd_eng.join(region_lookup, on="region_code", how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Join ONSPD + IMD -> Final Enrichment Table

# COMMAND ----------

enriched = (
    onspd_eng.alias("p")
    .join(imd_slim.alias("i"), on="lsoa_code", how="left")
    .select(
        "postcode",
        "lat",
        "long",
        "lsoa_code",
        "local_authority_code",
        "region_code",
        "region_name",
        "urban_rural_code",
        "urban_rural_band",
        "is_urban",
        "imd_decile",
        "imd_score",
        "crime_decile",
        "income_decile",
        "health_decile",
        "living_env_decile",
    )
)

# Sanity checks
final_count = enriched.count()
imd_coverage = enriched.filter(F.col("imd_decile").isNotNull()).count()
print(f"Enriched postcodes: {final_count:,}")
print(f"  with IMD coverage: {imd_coverage:,} ({imd_coverage/final_count:.1%})")

display(enriched.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Derive Coastal Flag
# MAGIC
# MAGIC A rough-but-useful feature: whether a postcode is near the coast. We compute it from
# MAGIC longitude/latitude by checking proximity to the English coastline (simplified heuristic:
# MAGIC any postcode in a coastal local authority). For a demo this is good enough; a production
# MAGIC pipeline would do a proper spatial join to the coastline polygon.

# COMMAND ----------

# Coastal English local authorities (rough list — for a real pipeline, use a proper spatial join)
coastal_la_codes = [
    "E06000001","E06000002","E06000003","E06000004","E06000005","E06000006",  # NE
    "E06000020","E06000021","E06000022","E06000023","E06000024","E06000025",  # mid
    "E06000028","E06000029","E06000030","E06000031","E06000032",              # various
    "E06000035","E06000038","E06000039","E06000040","E06000046","E06000055",
    "E06000056","E06000057","E06000058","E06000059","E06000060",
    "E07000032","E07000040","E07000047","E07000048","E07000049","E07000066",
    "E07000070","E07000086","E07000087","E07000088","E07000089","E07000091",
    "E07000112","E07000114","E07000132","E07000142","E07000143","E07000175",
    "E07000176","E07000198","E07000199","E07000200","E07000216","E07000222",
    "E07000223","E07000224","E07000239","E07000240","E07000241","E08000003",
    "E08000022","E08000027","E08000028","E08000030","E08000031","E08000036",
    "E08000037","E09000008","E09000010","E09000016","E09000025",
]
coastal_set = set(coastal_la_codes)

is_coastal_udf = F.udf(lambda la: 1 if la in coastal_set else 0, "int")
enriched = enriched.withColumn("is_coastal", is_coastal_udf(F.col("local_authority_code")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write to Unity Catalog

# COMMAND ----------

(
    enriched
    .repartition(32)
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.postcode_enrichment")
)

print("Saved: postcode_enrichment")

# Show summary statistics
summary = spark.table(f"{CATALOG}.{SCHEMA}.postcode_enrichment")
print(f"\nTotal postcodes: {summary.count():,}")
print("\nBy region:")
display(
    summary.groupBy("region_name")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC ### What's in the table
# MAGIC
# MAGIC | Column | Type | Description |
# MAGIC |---|---|---|
# MAGIC | `postcode` | string | Postcode with standard spacing (e.g., "SW1A 1AA") |
# MAGIC | `lat`, `long` | double | WGS84 centroid coordinates |
# MAGIC | `lsoa_code` | string | LSOA 2011 code (neighbourhood identifier) |
# MAGIC | `local_authority_code` | string | Local authority district code |
# MAGIC | `region_code`, `region_name` | string | ONS region (e.g., "London") |
# MAGIC | `urban_rural_code`, `urban_rural_band`, `is_urban` | string/int | Urban/rural classification |
# MAGIC | `imd_decile`, `imd_score` | int/double | Index of Multiple Deprivation |
# MAGIC | `crime_decile` | int | Crime domain decile (1 = most deprived = highest crime) |
# MAGIC | `income_decile` | int | Income domain decile |
# MAGIC | `health_decile` | int | Health deprivation decile |
# MAGIC | `living_env_decile` | int | Living environment decile |
# MAGIC | `is_coastal` | int | Coastal local authority flag |
# MAGIC
# MAGIC ### Data provenance
# MAGIC
# MAGIC | Dataset | Release | Source |
# MAGIC |---|---|---|
# MAGIC | ONS Postcode Directory | August 2025 | ONS Open Geography Portal |
# MAGIC | Indices of Deprivation | 2019 | MHCLG via GOV.UK |
# MAGIC | Rural-Urban Classification | 2011 | ONS |
# MAGIC
# MAGIC All data under Open Government Licence (OGL). Attribution: Office for National Statistics,
# MAGIC Ministry of Housing, Communities and Local Government.
# MAGIC
# MAGIC ### Next step
# MAGIC
# MAGIC Run **01_build_all_models** — it will now sample from this enrichment table instead
# MAGIC of generating synthetic features.
