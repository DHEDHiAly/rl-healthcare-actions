"""Extract all hospital admissions from MIMIC-IV (ICU + floor)."""

import polars as pl
from src.config import MIMIC_DATA_DIR, COHORT_MODE


def extract_cohort() -> pl.DataFrame:
    adm = pl.read_csv(f"{MIMIC_DATA_DIR}/hosp/admissions.csv.gz", try_parse_dates=True)
    pat = pl.read_csv(f"{MIMIC_DATA_DIR}/hosp/patients.csv.gz")

    if COHORT_MODE == "all_hosp":
        # All admissions with any data — ICU + hospital floor
        labev = pl.scan_csv(f"{MIMIC_DATA_DIR}/hosp/labevents.csv.gz", try_parse_dates=True)
        lab_with_hadm = (
            labev.filter(pl.col("valuenum").is_not_null() & pl.col("hadm_id").is_not_null())
            .select(pl.col("hadm_id").cast(pl.Int64))
            .unique()
            .collect()
        )
        lab_no_hadm = (
            labev.filter(
                pl.col("valuenum").is_not_null()
                & pl.col("hadm_id").is_null()
                & pl.col("subject_id").is_not_null()
            )
            .select(pl.col("subject_id").cast(pl.Int64))
            .unique()
            .collect()
        )
        lab_hadm_set = set(lab_with_hadm["hadm_id"].to_list())
        lab_subj_set = set(lab_no_hadm["subject_id"].to_list())
        # Admissions that either have labs directly or their subject has labs with null hadm_id
        adm_with_labs = adm.filter(
            pl.col("hadm_id").is_in(list(lab_hadm_set))
            | pl.col("subject_id").cast(pl.Int64).is_in(list(lab_subj_set))
        )
        # Also include admissions with prescriptions (floor meds)
        from pathlib import Path

        rx_path = f"{MIMIC_DATA_DIR}/hosp/prescriptions.csv.gz"
        if Path(rx_path).exists():
            rx = (
                pl.scan_csv(rx_path)
                .filter(pl.col("hadm_id").is_not_null())
                .select(pl.col("hadm_id").cast(pl.Int64))
                .unique()
                .collect()
            )
            rx_hadm_set = set(rx["hadm_id"].to_list())
            adm_with_rx = adm.filter(pl.col("hadm_id").is_in(list(rx_hadm_set)))
            # Union: admissions with labs OR prescriptions
            all_hadm = set(adm_with_labs["hadm_id"].to_list()) | rx_hadm_set
            cohort_ids = (
                adm.filter(pl.col("hadm_id").is_in(list(all_hadm))).select("hadm_id").unique()
            )
        else:
            cohort_ids = adm_with_labs.select("hadm_id").unique()
    elif COHORT_MODE == "all_icu":
        labev = pl.scan_csv(f"{MIMIC_DATA_DIR}/hosp/labevents.csv.gz", try_parse_dates=True)
        cohort_ids = (
            labev.filter(pl.col("valuenum").is_not_null() & pl.col("hadm_id").is_not_null())
            .select(pl.col("hadm_id").cast(pl.Int64))
            .unique()
            .collect()
        )
    else:
        cohort_ids = adm.select("hadm_id").unique()

    cohort = (
        cohort_ids.join(
            adm.select(
                "hadm_id",
                "subject_id",
                "admittime",
                "dischtime",
                "deathtime",
                "hospital_expire_flag",
            ),
            on="hadm_id",
            how="inner",
        )
        .join(pat.select("subject_id", "anchor_age", "gender"), on="subject_id", how="left")
        .with_columns(
            ((pl.col("dischtime") - pl.col("admittime")).dt.total_seconds() / 86400.0).alias(
                "los_days"
            )
        )
        .filter(pl.col("dischtime") > pl.col("admittime"))
    )

    return cohort.sort("hadm_id")


if __name__ == "__main__":
    from pathlib import Path

    cohort = extract_cohort()
    out = Path("data")
    out.mkdir(exist_ok=True)
    cohort.write_parquet(out / "cohort.parquet")
    cohort.write_csv(out / "cohort.csv")
    print(f"Cohort: {cohort.height} admissions")
    print(f"Mortality: {cohort['hospital_expire_flag'].sum()}")
    print(f"Mean LOS: {cohort['los_days'].mean():.1f} days")
