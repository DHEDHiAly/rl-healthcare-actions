"""Per-patient data integration: all MIMIC tables → unified patient profiles.
Single-pass scanning where possible to avoid repeated GB-level CSV reads."""
import polars as pl
from pathlib import Path
from src.config import MIMIC_DATA_DIR

HOSP = Path(MIMIC_DATA_DIR) / "hosp"
ICU = Path(MIMIC_DATA_DIR) / "icu"
PROFILE_PATH = Path("data/patient_profiles.parquet")

# ── Helpers ──────────────────────────────────────────────────────
_CHART_SOFA_KEPT = {220052, 223901, 198, 184, 454, 3420, 190, 223835,
                    220339, 3348, 437, 224685, 224684, 224688, 224689, 188}
_LAB_SOFA_KEPT = {50885, 51265, 50912, 50817}


def _keyed_cache(path, key_cols, **scan_kw):
    """Memoized loader: reads a CSV lazily, caches result per call."""
    cache = getattr(_keyed_cache, "_cache", {})
    key = str(path)
    if key not in cache:
        cache[key] = pl.scan_csv(path, infer_schema_length=5000, **scan_kw)
        _keyed_cache._cache = cache
    return cache[key]


def _extract_base():
    return pl.scan_csv(HOSP / "admissions.csv.gz", infer_schema_length=5000).join(
        pl.scan_csv(HOSP / "patients.csv.gz", infer_schema_length=5000), on="subject_id", how="left"
    ).select("subject_id", "hadm_id", "anchor_age", "gender", "race",
             "admission_type", "hospital_expire_flag", "admittime", "dischtime").collect()


# ── 1. Microbiology ──────────────────────────────────────────────
def extract_microbio():
    return pl.scan_csv(HOSP / "microbiologyevents.csv.gz", infer_schema_length=2000
                       ).drop_nulls("hadm_id").filter(pl.col("org_name").is_not_null()
                       ).group_by("hadm_id").agg([
        pl.count().alias("micro_cultures_total"),
        pl.col("org_name").n_unique().alias("micro_unique_orgs"),
        pl.col("spec_type_desc").first().alias("micro_primary_specimen"),
        pl.col("org_name").mode().first().alias("micro_most_common_org"),
        pl.col("interpretation").filter(pl.col("interpretation") == "R").count().alias("micro_resistant_count"),
    ]).collect()


# ── 2. Input events ──────────────────────────────────────────────
def extract_io():
    io_in = pl.scan_csv(ICU / "inputevents.csv.gz", infer_schema_length=2000,
                        schema_overrides={"totalamount": pl.Float32}
                        ).group_by("hadm_id").agg([
        pl.col("totalamount").sum().alias("io_total_input"),
        pl.count().alias("io_input_events"),
    ]).collect()
    io_out = pl.scan_csv(ICU / "outputevents.csv.gz", infer_schema_length=2000,
                         schema_overrides={"value": pl.Float32}
                         ).group_by("hadm_id").agg([
        pl.col("value").sum().alias("io_total_output"),
        pl.count().alias("io_output_events"),
    ]).collect()
    merged = io_in.join(io_out, on="hadm_id", how="outer").with_columns([
        pl.col("io_total_input").fill_null(0.0),
        pl.col("io_total_output").fill_null(0.0),
        pl.col("io_input_events").fill_null(0),
        pl.col("io_output_events").fill_null(0),
    ]).with_columns(
        (pl.col("io_total_input") - pl.col("io_total_output")).alias("io_net_balance")
    )
    return merged


# ── 3. ICU stays + Transfers ─────────────────────────────────────
def extract_icustays():
    return pl.scan_csv(ICU / "icustays.csv.gz", infer_schema_length=2000
                       ).group_by("hadm_id").agg([
        pl.count().alias("icu_stay_count"),
        pl.col("los").sum().alias("icu_total_los_days"),
        pl.col("first_careunit").first().alias("icu_first_unit"),
    ]).collect()

def extract_transfers():
    return pl.scan_csv(HOSP / "transfers.csv.gz", infer_schema_length=2000
                       ).drop_nulls("hadm_id").group_by("hadm_id").agg([
        pl.count().alias("transfers_total"),
        pl.col("careunit").n_unique().alias("transfers_unique_units"),
    ]).collect()


# ── 4. Services + DRG ────────────────────────────────────────────
def extract_services():
    return pl.scan_csv(HOSP / "services.csv.gz", infer_schema_length=2000
                       ).drop_nulls("hadm_id").group_by("hadm_id").agg([
        pl.col("curr_service").first().alias("service_primary"),
    ]).collect()

def extract_drg():
    return pl.scan_csv(HOSP / "drgcodes.csv.gz", infer_schema_length=2000
                       ).drop_nulls("hadm_id").group_by("hadm_id").agg([
        pl.col("drg_code").first().alias("drg_code"),
        pl.col("description").first().alias("drg_description"),
        pl.col("drg_severity").max().alias("drg_severity_max"),
        pl.col("drg_mortality").max().alias("drg_mortality_max"),
    ]).collect()


# ── 5. Charlson (single scan of diagnoses) ───────────────────────
CHARLSON_RULES = [
    ("charlson_mi", ["410", "412", "I21", "I22", "I252"]),
    ("charlson_chf", ["428", "I50", "I110", "I130", "I132"]),
    ("charlson_pvd", ["440", "441", "442", "443", "444", "V434", "I70", "I71", "I72", "I73", "I74", "I77", "K551", "Z958", "Z959"]),
    ("charlson_cva", ["430", "431", "432", "433", "434", "435", "436", "437", "438", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "G45", "G46"]),
    ("charlson_dementia", ["290", "F00", "F01", "F02", "F03", "G30", "G311"]),
    ("charlson_copd", ["490", "491", "492", "493", "494", "495", "496", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"]),
    ("charlson_rheumatic", ["710", "714", "725", "M05", "M06", "M32", "M33", "M34", "M35"]),
    ("charlson_pud", ["531", "532", "533", "534", "K25", "K26", "K27", "K28"]),
    ("charlson_mild_liver", ["570", "571", "573", "B18", "K70", "K71", "K73", "K74", "K76", "K77"]),
    ("charlson_diabetes", ["250", "E10", "E11", "E12", "E13", "E14"]),
    ("charlson_diabetes_comp", ["2504", "2505", "2506", "2507", "E102", "E103", "E104", "E105", "E106", "E107", "E108", "E112", "E113", "E114", "E115", "E116", "E117", "E118"]),
    ("charlson_paralysis", ["342", "343", "344", "G04", "G11", "G80", "G81", "G82", "G83"]),
    ("charlson_renal", ["582", "583", "585", "586", "588", "I12", "I13", "N00", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N11", "N12", "N14", "N17", "N18", "N19", "Z49"]),
    ("charlson_cancer", ["140", "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "198", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208", "2386", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]),
    ("charlson_metastatic", ["196", "197", "198", "199", "C77", "C78", "C79", "C80"]),
    ("charlson_severe_liver", ["4560", "4561", "4562", "5722", "5723", "5724", "5728", "I850", "I859", "I864", "K729", "K766", "K767"]),
    ("charlson_hiv", ["042", "043", "044", "B20", "B21", "B22", "B23", "B24", "Z21"]),
]

def extract_charlson():
    codes = pl.scan_csv(HOSP / "diagnoses_icd.csv.gz", infer_schema_length=2000
                        ).drop_nulls("hadm_id").select("hadm_id", "icd_code").collect()
    result = codes.select("hadm_id").unique()
    for name, prefixes in CHARLSON_RULES:
        pat = "^(" + "|".join(prefixes) + ")"
        flagged = codes.filter(pl.col("icd_code").cast(pl.Utf8).str.to_uppercase().str.replace(".", "").str.contains(pat)
                               ).select("hadm_id").unique().with_columns(pl.lit(1).alias(name))
        result = result.join(flagged, on="hadm_id", how="left").with_columns(pl.col(name).fill_null(0))
    score_cols = [c for c in result.columns if c.startswith("charlson_")]
    return result.with_columns(pl.sum_horizontal(score_cols).alias("charlson_score"))


# ── 6. SOFA + GCS + Vent (single scans) ──────────────────────────
def extract_clinical():
    """SOFA, GCS, vent settings from single lab + chart scans."""
    lab = pl.scan_csv(HOSP / "labevents.csv.gz", infer_schema_length=2000)
    chart = pl.scan_csv(ICU / "chartevents.csv.gz", infer_schema_length=2000)
    lab_f = lab.filter(pl.col("itemid").is_in(_LAB_SOFA_KEPT)).select("hadm_id", "itemid", "valuenum").drop_nulls().collect()
    chart_f = chart.filter(pl.col("itemid").is_in(_CHART_SOFA_KEPT)).select("hadm_id", "itemid", "valuenum").drop_nulls().collect()

    # SOFA components
    bili = lab_f.filter(pl.col("itemid") == 50885).group_by("hadm_id").agg(pl.max("valuenum").alias("bili_max"))
    plt = lab_f.filter(pl.col("itemid") == 51265).group_by("hadm_id").agg(pl.min("valuenum").alias("plt_min"))
    cr = lab_f.filter(pl.col("itemid") == 50912).group_by("hadm_id").agg(pl.max("valuenum").alias("cr_max"))
    map_bp = chart_f.filter(pl.col("itemid") == 220052).group_by("hadm_id").agg(pl.min("valuenum").alias("map_min"))
    sofa = bili.join(plt, on="hadm_id", how="outer").join(cr, on="hadm_id", how="outer").join(map_bp, on="hadm_id", how="outer")
    sofa = sofa.with_columns([
        pl.when(pl.col("bili_max") < 1.2).then(0).when(pl.col("bili_max") < 2.0).then(1).when(pl.col("bili_max") < 6.0).then(2).when(pl.col("bili_max") < 12.0).then(3).otherwise(4).alias("sofa_liver"),
        pl.when(pl.col("plt_min") > 150).then(0).when(pl.col("plt_min") > 100).then(1).when(pl.col("plt_min") > 50).then(2).when(pl.col("plt_min") > 20).then(3).otherwise(4).alias("sofa_coag"),
        pl.when(pl.col("cr_max") < 1.2).then(0).when(pl.col("cr_max") < 2.0).then(1).when(pl.col("cr_max") < 3.5).then(2).when(pl.col("cr_max") < 5.0).then(3).otherwise(4).alias("sofa_renal"),
        pl.when(pl.col("map_min") >= 70).then(0).otherwise(1).alias("sofa_cardio"),
    ]).with_columns(
        (pl.col("sofa_liver").fill_null(0) + pl.col("sofa_coag").fill_null(0) +
         pl.col("sofa_renal").fill_null(0) + pl.col("sofa_cardio").fill_null(0)).alias("sofa_score")
    )

    # GCS
    gcs = chart_f.filter(pl.col("itemid").is_in([198, 184, 454, 223901])).group_by("hadm_id").agg([
        pl.min("valuenum").alias("gcs_min"), pl.mean("valuenum").alias("gcs_mean"),
    ])

    # Ventilator (pivot via group + join)
    vent_map = {3420: "fio2", 190: "fio2", 223835: "fio2", 220339: "peep", 3348: "peep",
                437: "peep", 224685: "tv", 224684: "tv", 224688: "rate", 224689: "rate", 188: "rate"}
    vent_raw = chart_f.filter(pl.col("itemid").is_in(list(vent_map.keys())))
    vent_raw = vent_raw.with_columns(pl.col("itemid").replace_strict(vent_map).alias("vparam"))
    vent = vent_raw.group_by(["hadm_id", "vparam"]).agg(pl.mean("valuenum").alias("val"))
    vent_pivoted = vent.collect().pivot(values="val", index="hadm_id", columns="vparam", aggregate_function="first")
    vent_pivoted = vent_pivoted.rename({c: f"vent_{c}" for c in vent_pivoted.columns if c != "hadm_id"})

    merged = sofa.join(gcs, on="hadm_id", how="outer").join(vent_pivoted, on="hadm_id", how="outer")
    return merged


# ── 7. Master build ──────────────────────────────────────────────
def build_profile(force=False):
    if PROFILE_PATH.exists() and not force:
        return pl.read_parquet(PROFILE_PATH)
    print("Building patient profiles (this takes ~5 min on first run)...")
    base = _extract_base()
    print(f"  Base: {base.height:,} admissions")
    extractors = [
        ("Microbio", extract_microbio), ("IO Balance", extract_io),
        ("ICU", extract_icustays), ("Transfers", extract_transfers),
        ("Services", extract_services), ("DRG", extract_drg),
        ("Charlson", extract_charlson), ("Clinical", extract_clinical),
    ]
    for name, fn in extractors:
        try:
            df = fn()
            if df is not None and df.height > 0:
                dup = [c for c in df.columns if c != "hadm_id" and c in base.columns]
                if dup:
                    df = df.drop([c for c in dup if c in df.columns])
                base = base.join(df, on="hadm_id", how="left")
            print(f"  {name}: {df.height if df is not None else 0:,}")
        except Exception as e:
            print(f"  {name}: SKIPPED — {e}")
            import traceback; traceback.print_exc()
    base.write_parquet(PROFILE_PATH)
    print(f"  Saved ({base.height:,} rows, {len(base.columns)} cols)")
    return base


def query_patient(hadm_id: int):
    row = build_profile().filter(pl.col("hadm_id") == hadm_id)
    return row.to_dicts()[0] if row.height > 0 else None


def query_group(field: str, value):
    return build_profile().filter(pl.col(field) == value)


def profile_summary(profile=None):
    if profile is None:
        profile = build_profile()
    print(f"Patient profiles: {profile.height:,}")
    print(f"  Age: {profile['anchor_age'].mean():.0f} ± {profile['anchor_age'].std():.0f}")
    print(f"  Mortality: {profile['hospital_expire_flag'].sum():,} ({100*profile['hospital_expire_flag'].mean():.1f}%)")
    for col in ["charlson_score", "sofa_score"]:
        if col in profile.columns:
            vals = profile[col].drop_nulls()
            print(f"  {col}: {vals.mean():.1f} ± {vals.std():.1f} (n={len(vals):,})")
    if "icu_stay_count" in profile.columns:
        icu_pct = 100 * profile["icu_stay_count"].not_null().sum() / profile.height
        print(f"  ICU: {profile['icu_stay_count'].sum():,} stays ({icu_pct:.1f}%)")
    if "drg_severity_max" in profile.columns:
        sv = profile["drg_severity_max"].drop_nulls()
        print(f"  DRG severity: {sv.mean():.1f} (n={len(sv):,})")
    if "io_net_balance" in profile.columns:
        net = profile["io_net_balance"].drop_nulls()
        print(f"  Net fluid: {net.mean():.0f} ± {net.std():.0f} mL" if len(net) > 0 else "  Net fluid: N/A")
