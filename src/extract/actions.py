"""Extract action bundles from prescriptions + chartevents (no inputevents table)."""

from typing import Optional
import polars as pl
from src.config import MIMIC_DATA_DIR, BIN_HOURS, ACTION_BUNDLES, PRECEDENCE


def extract_actions_from_prescriptions(hadm_ids: Optional[set] = None) -> pl.DataFrame:
    scan = pl.scan_csv(f"{MIMIC_DATA_DIR}/hosp/prescriptions.csv.gz", try_parse_dates=True)
    if hadm_ids is not None:
        hadm_str = [str(h) for h in hadm_ids]
        scan = scan.filter(pl.col("hadm_id").cast(pl.Utf8).is_in(hadm_str))

    rx = scan.select(
        pl.col("hadm_id").cast(pl.Int64),
        pl.col("starttime"),
        pl.col("drug"),
    ).collect()
    rx = rx.filter(pl.col("starttime").is_not_null())
    rx = rx.with_columns(pl.col("drug").str.to_lowercase())

    # Multi-match: a drug can match multiple bundles; assign all matching action_ids
    # then de-duplicate by (hadm_id, starttime, action_id)
    action_id = pl.lit(None).cast(pl.Int8)
    for aid in sorted(ACTION_BUNDLES.keys()):
        if aid == 0:
            continue
        cfg = ACTION_BUNDLES[aid]
        keywords = cfg.get("drugs", [])
        if not keywords:
            continue
        # Build a combined OR expression for all keywords in this bundle
        kw_mask = pl.lit(False)
        for kw in keywords:
            kw_mask = kw_mask | pl.col("drug").str.contains(kw.lower())
        action_id = pl.when(kw_mask).then(pl.lit(aid, dtype=pl.Int8)).otherwise(action_id)

    rx = rx.with_columns(action_id.alias("action_id")).filter(pl.col("action_id").is_not_null())

    return rx.select("hadm_id", "starttime", "action_id").unique()


def extract_actions_from_chartevents(hadm_ids: Optional[set] = None) -> pl.DataFrame:
    """Load pre-extracted blood product events from chartevents.

    chartevents.csv.gz is truncated — blood products were extracted via manual CSV scan
    and saved to data/blood_products_chartevents.parquet.
    """
    from pathlib import Path

    p = Path("data/blood_products_chartevents.parquet")
    if not p.exists():
        return pl.DataFrame(schema={"hadm_id": pl.Int64, "starttime": pl.Datetime, "action_id": pl.Int8})

    bp = pl.read_parquet(p)
    bp = bp.with_columns(pl.col("charttime").str.to_datetime(), pl.col("action_id").cast(pl.Int8))
    if hadm_ids is not None:
        bp = bp.filter(pl.col("hadm_id").is_in(list(hadm_ids)))

    bp = bp.rename({"charttime": "starttime"})
    return bp.select("hadm_id", "starttime", "action_id").unique()


def bin_actions(actions: pl.DataFrame, admittimes: pl.DataFrame) -> pl.DataFrame:
    """Assign bin_idx and apply precedence rule."""
    joined = actions.join(admittimes, on="hadm_id", how="inner")
    joined = joined.with_columns(
        ((pl.col("starttime") - pl.col("admittime")).dt.total_seconds() / 3600.0 / BIN_HOURS)
        .floor()
        .cast(pl.Int32)
        .alias("bin_idx")
    ).filter(pl.col("bin_idx") >= 0)

    # Precedence: lowest precedence rank wins per (hadm_id, bin_idx)
    # PRECEDENCE is ordered by priority — map action_id to its rank
    prec_rank = {a: i for i, a in enumerate(PRECEDENCE)}
    with_rank = joined.with_columns(
        pl.col("action_id").replace(prec_rank, default=len(PRECEDENCE)).alias("prec_rank")
    )
    precedenced = with_rank.sort("hadm_id", "bin_idx", "prec_rank").group_by("hadm_id", "bin_idx").agg(
        pl.col("action_id").first().alias("action_id")
    )

    return precedenced.sort("hadm_id", "bin_idx")


def extract_all_actions(hadm_ids: Optional[set] = None, admittimes: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    rx = extract_actions_from_prescriptions(hadm_ids)
    ce = extract_actions_from_chartevents(hadm_ids)

    all_actions = pl.concat([rx, ce], how="diagonal").unique()

    if admittimes is not None and all_actions.height > 0:
        binned = bin_actions(all_actions, admittimes)
    else:
        binned = all_actions

    # Add Hgb-jump inferred RBC transfusions (already binned)
    hgb = extract_rbc_from_hgb_jump(hadm_ids)
    if hgb.height > 0:
        binned = pl.concat([binned, hgb.select("hadm_id", "bin_idx", "action_id")], how="diagonal").unique()
        # Re-apply precedence
        prec_rank = {a: i for i, a in enumerate(PRECEDENCE)}
        with_rank = binned.with_columns(
            pl.col("action_id").replace(prec_rank, default=len(PRECEDENCE)).alias("prec_rank")
        )
        binned = with_rank.sort("hadm_id", "bin_idx", "prec_rank").group_by("hadm_id", "bin_idx").agg(
            pl.col("action_id").first().alias("action_id")
        )

    return binned.sort("hadm_id", "bin_idx")


def extract_rbc_from_hgb_jump(hadm_ids: Optional[set] = None) -> pl.DataFrame:
    """Infer RBC transfusion (action 1) from Hgb jump >= 1.5 g/dL between bins.

    inputevents table is unavailable, so RBC transfusion is detected via
    unexpected Hgb increase rather than direct order record.
    ponytail: hgb_jump proxy for RBC transfusion, verify if inputevents obtained
    """
    from pathlib import Path
    from src.config import LAB_FEATURES

    labs = Path("data/labs_binned.parquet")
    if not labs.exists():
        return pl.DataFrame(schema={"hadm_id": pl.Int64, "starttime": pl.Utf8, "action_id": pl.Int8})

    df = pl.read_parquet(labs)
    if hadm_ids is not None:
        df = df.filter(pl.col("hadm_id").is_in(list(hadm_ids)))

    if "hemoglobin" not in df.columns:
        return pl.DataFrame(schema={"hadm_id": pl.Int64, "starttime": pl.Utf8, "action_id": pl.Int8})

    df = df.sort("hadm_id", "bin_idx").with_columns(
        pl.col("hemoglobin").diff().over("hadm_id").alias("hgb_delta")
    )
    jumps = df.filter(pl.col("hgb_delta") >= 1.5)
    if jumps.height == 0:
        return pl.DataFrame(schema={"hadm_id": pl.Int64, "starttime": pl.Utf8, "action_id": pl.Int8})

    jumps = jumps.with_columns(pl.lit(1).cast(pl.Int8).alias("action_id"))
    return pl.DataFrame({"hadm_id": jumps["hadm_id"], "action_id": jumps["action_id"], "bin_idx": jumps["bin_idx"]})


if __name__ == "__main__":
    from pathlib import Path

    out = Path("data")
    out.mkdir(exist_ok=True)

    cohort = pl.read_parquet(out / "cohort.parquet")
    hadm_ids = set(cohort["hadm_id"].to_list())
    admittimes = cohort.select("hadm_id", "admittime")

    actions = extract_all_actions(hadm_ids, admittimes)
    print(f"Action rows (binned): {actions.height:,}")
    if "bin_idx" in actions.columns:
        counts = actions.group_by("action_id").len()
        print(f"Action distribution:\n{counts}")

    actions.write_parquet(out / "actions_binned.parquet")
