"""Run full pipeline: MIMIC extraction or bring your own CSV/parquet."""

import sys
import time
import os
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _read_data(path: str) -> pl.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pl.read_csv(str(p), try_parse_dates=True)
    return pl.read_parquet(str(p))


def main():
    out = Path(os.environ.get("RL_DATA_DIR", "data"))
    out.mkdir(exist_ok=True)
    t0 = time.time()

    data_path = os.environ.get("RL_DATA_PATH")

    if data_path:
        # ── Bring-your-own-data mode ──────────────────────────────
        print(f"=== Reading data from {data_path} ===")
        df = _read_data(data_path)
        print(f"  {df.height:,} rows, {df.width} columns")

        if "action_id" in df.columns and "hadm_id" in df.columns:
            df.write_parquet(out / "trajectories_v1.parquet")
            print(f"  Saved as trajectory ({df['hadm_id'].n_unique()} admissions)")
        else:
            print("  No action_id column found; saving as raw data. Run features to build trajectories.")
            df.write_parquet(out / "raw_data.parquet")
    else:
        # ── MIMIC extraction mode ─────────────────────────────────
        from src.cohort.extract import extract_cohort
        from src.extract.labs import extract_labs, extract_vitals_from_chartevents, pivot_and_bin
        from src.extract.actions import extract_all_actions
        from src.pipeline.trajectory import build_trajectories

        print("=== Step 1: Cohort extraction ===")
        cohort = extract_cohort()
        cohort.write_parquet(out / "cohort.parquet")
        cohort.write_csv(out / "cohort.csv")
        print(f"  {cohort.height} admissions, mortality={cohort['hospital_expire_flag'].sum()}, mean LOS={cohort['los_days'].mean():.1f}d")

        hadm_ids = set(cohort["hadm_id"].to_list())
        admittimes = cohort.select("hadm_id", "subject_id", "admittime", "dischtime")

        print(f"\n=== Step 2: Lab extraction ===")
        labs = extract_labs(hadm_ids, admittimes)
        print(f"  {labs.height:,} raw lab rows")

        print(f"\n=== Step 3: Vitals extraction ===")
        vitals = extract_vitals_from_chartevents(hadm_ids)
        print(f"  {vitals.height:,} raw vital rows")

        print(f"\n=== Step 4: Pivot and bin ===")
        binned = pivot_and_bin(labs, admittimes, vitals)
        binned.write_parquet(out / "labs_binned.parquet")
        print(f"  {binned.height:,} binned rows, {binned['hadm_id'].n_unique()} admissions")

        print(f"\n=== Step 5: Action extraction ===")
        actions = extract_all_actions(hadm_ids, admittimes)
        actions.write_parquet(out / "actions_binned.parquet")
        print(f"  {actions.height:,} action rows")
        if "action_id" in actions.columns:
            print(f"  Distribution: {actions.group_by('action_id').len().sort('action_id').to_dict(as_series=False)}")

        print(f"\n=== Step 6: Trajectory assembly ===")
        traj = build_trajectories(
            str(out / "cohort.csv"),
            str(out / "labs_binned.parquet"),
            str(out / "actions_binned.parquet"),
            str(out / "trajectories_v1.parquet"),
        )
        print(f"  {traj.height:,} transitions, {traj['hadm_id'].n_unique()} admissions")
        if "action_id" in traj.columns:
            print(f"  Actions: {traj.group_by('action_id').len().sort('action_id').to_dict(as_series=False)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Healthcare Actions data pipeline")
    parser.add_argument("--out", default=None, help="Output directory (defaults to $RL_DATA_DIR or data/)")
    parser.add_argument("--mimic-dir", default=None, help="MIMIC data directory")
    parser.add_argument("--data", default=None, help="Input CSV/parquet (skips MIMIC extraction)")
    parser.add_argument("--skip-cohort", action="store_true", help="Skip cohort extraction")
    parser.add_argument("--skip-actions", action="store_true", help="Skip action extraction")
    args = parser.parse_args()

    if args.out:
        os.environ["RL_DATA_DIR"] = args.out
    if args.mimic_dir:
        os.environ["MIMIC_DATA_DIR"] = args.mimic_dir
    if args.data:
        os.environ["RL_DATA_PATH"] = args.data
    main()
