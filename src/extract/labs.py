"""Extract 38 lab biomarkers from labevents + 7 vitals from chartevents, binned into 4-hour intervals.

Handles hospital floor patients where labevents.hadm_id is null by matching
subject_id + charttime to admission time windows.
"""

import csv
from datetime import datetime
from typing import Optional
import polars as pl
from src.config import MIMIC_DATA_DIR, BIN_HOURS, LAB_FEATURES, VITAL_FEATURES


def _build_subject_windows(admittimes: pl.DataFrame) -> dict:
    """Build {subject_id: (starts, ends, hadm_ids)} with unix timestamps for bisect lookup.

    ponytail: sorted arrays + bisect over per-row datetime parse — 100x faster for 25M rows
    """
    import bisect

    windows = {}
    for row in admittimes.iter_rows(named=True):
        sid = int(row["subject_id"])
        hadm = int(row["hadm_id"])
        adm_ts = row["admittime"].timestamp() - 3600  # 1h before admission
        disch_ts = row["dischtime"].timestamp() + 3600  # 1h after discharge
        if sid not in windows:
            windows[sid] = {"starts": [], "ends": [], "hadms": [], "los": []}
        w = windows[sid]
        w["starts"].append(adm_ts)
        w["ends"].append(disch_ts)
        w["hadms"].append(hadm)
        w["los"].append(disch_ts - adm_ts)
    # Sort each subject's admissions by start time
    for sid, w in windows.items():
        order = sorted(range(len(w["starts"])), key=lambda i: w["starts"][i])
        w["starts"] = [w["starts"][i] for i in order]
        w["ends"] = [w["ends"][i] for i in order]
        w["hadms"] = [w["hadms"][i] for i in order]
        w["los"] = [w["los"][i] for i in order]
    return windows


def _assign_hadm_charttime(ct: datetime, subj_windows: dict) -> int:
    """Find hadm_id whose admission window contains this charttime. Returns 0 if not found."""
    import bisect
    ct_ts = ct.timestamp()
    starts = subj_windows["starts"]
    idx = bisect.bisect_right(starts, ct_ts) - 1
    if idx < 0:
        return 0
    best_hadm = 0
    best_los = float("inf")
    for i in range(max(0, idx - 1), min(len(starts), idx + 3)):
        if starts[i] <= ct_ts <= subj_windows["ends"][i]:
            if subj_windows["los"][i] < best_los:
                best_los = subj_windows["los"][i]
                best_hadm = subj_windows["hadms"][i]
    return best_hadm


def extract_labs(
    hadm_ids: Optional[set] = None, admittimes: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    itemids = set(v["itemid"] for v in LAB_FEATURES.values())
    id_to_label = {v["itemid"]: k for k, v in LAB_FEATURES.items()}

    # Build subject windows for null-hadm assignment
    subj_windows = None
    subject_filter = None
    if admittimes is not None and hadm_ids is not None:
        adm_subset = admittimes.filter(pl.col("hadm_id").is_in(list(hadm_ids)))
        subj_windows = _build_subject_windows(adm_subset)
        subject_filter = set(subj_windows.keys())

    # Phase 1: extract labs WITH hadm_id using lazy scan (fast, column-pruned)
    scan = pl.scan_csv(f"{MIMIC_DATA_DIR}/hosp/labevents.csv.gz", try_parse_dates=True)
    scan_with = scan.filter(
        pl.col("itemid").is_in(list(itemids))
        & pl.col("valuenum").is_not_null()
        & (pl.col("valuenum") > 0)
        & pl.col("hadm_id").is_not_null()
    )
    if hadm_ids is not None:
        hadm_str = [str(h) for h in hadm_ids]
        scan_with = scan_with.filter(pl.col("hadm_id").is_in(hadm_str))

    labs_known = scan_with.select(
        pl.col("hadm_id").cast(pl.Int64),
        pl.col("itemid"),
        pl.col("charttime"),
        pl.col("valuenum"),
    ).collect()

    print(
        f"  Labs with hadm_id: {labs_known.height:,} rows, {labs_known['hadm_id'].n_unique():,} admissions"
    )

    # Phase 2: scan for labs WITHOUT hadm_id — stream CSV to avoid OOM on 73M null rows
    assigned_count = 0
    if subj_windows and subject_filter:
        import gzip

        print("  Scanning labevents for null-hadm rows (hospital floor patients)...")
        rows_out = []
        lab_path = f"{MIMIC_DATA_DIR}/hosp/labevents.csv.gz"
        with gzip.open(lab_path, "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_idx = {h.lower(): i for i, h in enumerate(header)}
            subj_i = col_idx.get("subject_id", 1)
            hadm_i = col_idx.get("hadm_id", 2)
            itemid_i = col_idx.get("itemid", 4)
            charttime_i = col_idx.get("charttime", 3)
            valuenum_i = col_idx.get("valuenum", 7)

            count = 0
            try:
                for row in reader:
                    count += 1
                    if count % 10_000_000 == 0:
                        print(f"    {count:,} rows scanned, {len(rows_out):,} assigned")
                    try:
                        hadm_raw = row[hadm_i]
                    except IndexError:
                        continue
                    if hadm_raw.strip() != "":
                        continue
                    try:
                        itemid = int(row[itemid_i])
                    except (ValueError, IndexError):
                        continue
                    if itemid not in itemids:
                        continue
                    try:
                        subj = int(row[subj_i])
                    except (ValueError, IndexError):
                        continue
                    if subj not in subject_filter:
                        continue
                    try:
                        vn = float(row[valuenum_i])
                    except (ValueError, IndexError):
                        continue
                    if vn <= 0:
                        continue
                    try:
                        ct_str = row[charttime_i]
                    except IndexError:
                        continue
                    # Parse charttime and find matching admission
                    from datetime import datetime

                    try:
                        ct = datetime.strptime(ct_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    hadm_assigned = _assign_hadm_charttime(ct, subj_windows[subj])
                    if hadm_assigned is not None:
                        rows_out.append((hadm_assigned, itemid, ct_str, vn))
                        assigned_count += 1
            except (EOFError, csv.Error):
                print(f"    EOF at {count:,} rows")

        if rows_out:
            labs_assigned = pl.DataFrame(
                {
                    "hadm_id": [r[0] for r in rows_out],
                    "itemid": [r[1] for r in rows_out],
                    "charttime": [r[2] for r in rows_out],
                    "valuenum": [r[3] for r in rows_out],
                },
                schema={
                    "hadm_id": pl.Int64,
                    "itemid": pl.Int64,
                    "charttime": pl.Utf8,
                    "valuenum": pl.Float64,
                },
            )
            labs_assigned = labs_assigned.with_columns(pl.col("charttime").str.to_datetime())
            labs_known = pl.concat([labs_known, labs_assigned], how="diagonal")

    print(f"  Null-hadm labs assigned: {assigned_count:,}")

    label_map = pl.DataFrame(
        {
            "itemid": list(id_to_label.keys()),
            "label": list(id_to_label.values()),
        }
    )
    labs_known = labs_known.join(label_map, on="itemid", how="inner")

    return labs_known


def extract_vitals_from_chartevents(hadm_ids: Optional[set] = None) -> pl.DataFrame:
    from pathlib import Path

    vital_itemids = {v["itemid"]: k for k, v in VITAL_FEATURES.items()}
    id_set = set(vital_itemids.keys())
    hadm_filter = hadm_ids if hadm_ids is not None else None

    cache = Path("data/vitals_chartevents.parquet")
    if cache.exists():
        v = pl.read_parquet(cache)
        if hadm_filter is not None:
            v = v.filter(pl.col("hadm_id").is_in(list(hadm_filter)))
        return v

    chartevents_path = f"{MIMIC_DATA_DIR}/icu/chartevents.csv.gz"
    if not Path(chartevents_path).exists():
        return pl.DataFrame(
            schema={
                "hadm_id": pl.Int64,
                "itemid": pl.Int64,
                "charttime": pl.Datetime,
                "valuenum": pl.Float64,
            }
        )

    import gzip

    rows = []
    with gzip.open(chartevents_path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_idx = {h.lower(): i for i, h in enumerate(header)}

        hadm_i = col_idx.get("hadm_id", 1)
        itemid_i = col_idx.get("itemid", 5)
        charttime_i = col_idx.get("charttime", 4)
        valuenum_i = col_idx.get("valuenum", 8)

        count = 0
        try:
            for row in reader:
                count += 1
                if count % 5_000_000 == 0:
                    print(f"  chartevents: {count:,} rows scanned, {len(rows):,} vitals matched")
                try:
                    itemid = int(row[itemid_i])
                except (ValueError, IndexError):
                    continue
                if itemid not in id_set:
                    continue
                try:
                    hadm = int(row[hadm_i])
                except (ValueError, IndexError):
                    continue
                if hadm_filter is not None and hadm not in hadm_filter:
                    continue
                try:
                    vn = float(row[valuenum_i])
                except (ValueError, IndexError):
                    continue
                try:
                    ct = row[charttime_i]
                except IndexError:
                    continue
                if vn <= 0:
                    continue
                rows.append((hadm, itemid, ct, vn))
        except (EOFError, csv.Error):
            print(f"  chartevents: EOF at {count:,} rows (truncated file)")

    if not rows:
        return pl.DataFrame(
            schema={
                "hadm_id": pl.Int64,
                "itemid": pl.Int64,
                "charttime": pl.Datetime,
                "valuenum": pl.Float64,
            }
        )

    vitals = pl.DataFrame(
        {
            "hadm_id": [r[0] for r in rows],
            "itemid": [r[1] for r in rows],
            "charttime": [r[2] for r in rows],
            "valuenum": [r[3] for r in rows],
        },
        schema={
            "hadm_id": pl.Int64,
            "itemid": pl.Int64,
            "charttime": pl.Utf8,
            "valuenum": pl.Float64,
        },
    )
    vitals = vitals.with_columns(pl.col("charttime").str.to_datetime())
    vitals = vitals.with_columns(pl.col("itemid").cast(pl.Int64))

    cache.parent.mkdir(parents=True, exist_ok=True)
    vitals.write_parquet(str(cache))
    print(f"  Cached {vitals.height:,} vital rows to {cache}")

    label_map = pl.DataFrame(
        {
            "itemid": list(vital_itemids.keys()),
            "label": list(vital_itemids.values()),
        }
    )
    vitals = vitals.join(label_map, on="itemid", how="inner")
    return vitals


def pivot_and_bin(
    labs: pl.DataFrame, admittimes: pl.DataFrame, vitals: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    joined = labs.join(admittimes, on="hadm_id", how="inner")
    joined = joined.with_columns(
        ((pl.col("charttime") - pl.col("admittime")).dt.total_seconds() / 3600.0 / BIN_HOURS)
        .floor()
        .cast(pl.Int32)
        .alias("bin_idx")
    ).filter(pl.col("bin_idx") >= 0)

    wide = joined.pivot(
        on="label", index=["hadm_id", "bin_idx"], values="valuenum", aggregate_function="last"
    )

    if vitals is not None and vitals.height > 0:
        v_joined = vitals.join(admittimes, on="hadm_id", how="inner")
        v_joined = v_joined.with_columns(
            ((pl.col("charttime") - pl.col("admittime")).dt.total_seconds() / 3600.0 / BIN_HOURS)
            .floor()
            .cast(pl.Int32)
            .alias("bin_idx")
        ).filter(pl.col("bin_idx") >= 0)

        if "label" in v_joined.columns:
            vital_wide = v_joined.pivot(
                on="label",
                index=["hadm_id", "bin_idx"],
                values="valuenum",
                aggregate_function="last",
            )
            all_bins = pl.concat(
                [
                    wide.select("hadm_id", "bin_idx"),
                    vital_wide.select("hadm_id", "bin_idx"),
                ]
            ).unique()
            wide = all_bins.join(wide, on=["hadm_id", "bin_idx"], how="left").join(
                vital_wide, on=["hadm_id", "bin_idx"], how="left"
            )
        else:
            vital_agg = v_joined.group_by(["hadm_id", "bin_idx"]).agg(
                pl.col("valuenum").mean().alias("vital_value")
            )
            wide = (
                pl.concat(
                    [
                        wide.select("hadm_id", "bin_idx"),
                        vital_agg.select("hadm_id", "bin_idx"),
                    ]
                )
                .unique()
                .join(wide, on=["hadm_id", "bin_idx"], how="left")
                .join(vital_agg, on=["hadm_id", "bin_idx"], how="left")
            )

    return wide.sort("hadm_id", "bin_idx")


if __name__ == "__main__":
    from pathlib import Path

    out = Path("data")
    out.mkdir(exist_ok=True)

    cohort = pl.read_parquet(out / "cohort.parquet")
    hadm_ids = set(cohort["hadm_id"].to_list())
    admittimes = cohort.select("hadm_id", "subject_id", "admittime")

    labs = extract_labs(hadm_ids, admittimes)
    print(f"Raw lab rows for cohort: {labs.height:,}")

    print("Extracting vitals from chartevents (may take a while)...")
    vitals = extract_vitals_from_chartevents(hadm_ids)
    print(f"Raw vital rows for cohort: {vitals.height:,}")

    binned = pivot_and_bin(labs, admittimes, vitals)
    print(f"Binned rows: {binned.height:,}")
    print(f"Admissions with data: {binned['hadm_id'].n_unique()}")

    binned.write_parquet(out / "labs_binned.parquet")
