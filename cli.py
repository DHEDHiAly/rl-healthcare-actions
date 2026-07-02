#!/usr/bin/env python3
"""Unified CLI entry point for RL Healthcare Actions."""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.rl.train import ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(description="RL Healthcare Actions — training and evaluation")
    parser.add_argument("--data-dir", default=None, help="Data directory (overrides $RL_DATA_DIR)")
    parser.add_argument("--mimic-dir", default=None, help="MIMIC data directory (overrides $MIMIC_DATA_DIR)")
    parser.add_argument("--config", default=None, help="Config JSON path (overrides $RL_CONFIG)")
    parser.add_argument("--device", default=None, help="Device override")

    sub = parser.add_subparsers(dest="command", required=True)

    p_cohort = sub.add_parser("cohort", help="Extract cohort from MIMIC")
    p_cohort.add_argument("--min-bins", type=int, default=6)

    p_labs = sub.add_parser("labs", help="Extract labs and vitals from MIMIC")

    p_actions = sub.add_parser("actions", help="Extract actions from MIMIC")

    p_pipeline = sub.add_parser("pipeline", help="Run full pipeline: cohort → labs → actions → trajectory")
    p_pipeline.add_argument("--skip-cohort", action="store_true")
    p_pipeline.add_argument("--skip-actions", action="store_true")
    p_pipeline.add_argument("--data", default=None, help="Input CSV/parquet (skips MIMIC extraction)")

    p_features = sub.add_parser("features", help="Feature engineering and splits")
    p_features.add_argument("--seed", type=int, default=42)
    p_features.add_argument("--data", default=None, help="Input CSV/parquet (default: data/trajectories_v1.parquet)")

    p_train = sub.add_parser("train", help="Train IQL + BC models")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    p_train.add_argument("--batch-size", type=int, default=2048)
    p_train.add_argument("--iql-only", action="store_true")
    p_train.add_argument("--data", default=None, help="Data CSV/parquet (runs feature pipeline first if needed)")
    p_train.add_argument("--arch", choices=list(ARCHITECTURES.keys()), default="mlp",
                        help="Network architecture: mlp (256x2), deep (512x3), wide (512x2), lstm")
    p_train.add_argument("--hidden-sizes", type=int, nargs="+", default=None,
                        help="Custom hidden layer sizes (overrides --arch)")

    p_eval = sub.add_parser("eval", help="Run evaluation suite")
    p_eval.add_argument("--subtask", choices=["all", "ope", "safety", "phenotype", "plausibility", "bootstrap"],
                        default="all")

    p_infer = sub.add_parser("infer", help="Batch inference: score millions of patient states")
    p_infer.add_argument("--input", required=True, help="Input parquet/csv with state vectors")
    p_infer.add_argument("--output", default="data/inference_results.parquet", help="Output path")
    p_infer.add_argument("--model-dir", default=None, help="Model directory")
    p_infer.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    p_infer.add_argument("--batch-size", type=int, default=4096, help="Inference batch size")
    p_infer.add_argument("--state-cols", nargs="*", help="State column names (infer from dataset if omitted)")
    p_infer.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    p_infer.add_argument("--lab-cols", nargs="*", help="Lab column names for safety masking")
    p_infer.add_argument("--patient-id-col", default=None, help="Patient ID column for output")
    p_infer.add_argument("--safety", action="store_true", help="Apply safety constraints")

    p_serve = sub.add_parser("serve", help="Start REST API server for inference")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--model-dir", default=None)
    p_serve.add_argument("--state-dim", type=int, default=64)
    p_serve.add_argument("--preload", action="store_true", help="Load model on startup")

    p_test = sub.add_parser("test", help="Run test suite")
    p_test.add_argument("--filter", default=None, help="Test filter expression (e.g. 'phase4')")
    p_test.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.data_dir:
        os.environ["RL_DATA_DIR"] = args.data_dir
    if args.mimic_dir:
        os.environ["MIMIC_DATA_DIR"] = args.mimic_dir
    if args.config:
        os.environ["RL_CONFIG"] = args.config

    if args.command == "cohort":
        from src.cohort.extract import extract_cohort
        from pathlib import Path
        out = Path(os.environ.get("RL_DATA_DIR", "data"))
        out.mkdir(exist_ok=True)
        cohort = extract_cohort()
        cohort.write_parquet(out / "cohort.parquet")
        cohort.write_csv(out / "cohort.csv")
        print(f"Cohort: {cohort.height} admissions, mortality={cohort['hospital_expire_flag'].sum()}")

    if hasattr(args, 'data') and args.data:
        os.environ["RL_DATA_PATH"] = args.data
    elif args.command == "labs":
        from src.extract.labs import extract_labs, extract_vitals_from_chartevents, pivot_and_bin
        from pathlib import Path
        out = Path(os.environ.get("RL_DATA_DIR", "data"))
        out.mkdir(exist_ok=True)
        cohort = pl.read_parquet(out / "cohort.parquet")
        hadm_ids = set(cohort["hadm_id"].to_list())
        admittimes = cohort.select("hadm_id", "subject_id", "admittime", "dischtime")
        import polars as pl
        labs = extract_labs(hadm_ids, admittimes)
        vitals = extract_vitals_from_chartevents(hadm_ids)
        binned = pivot_and_bin(labs, admittimes, vitals)
        binned.write_parquet(out / "labs_binned.parquet")
        print(f"Labs binned: {binned.height:,} rows, {binned['hadm_id'].n_unique()} admissions")
    elif args.command == "actions":
        from src.extract.actions import extract_all_actions
        from pathlib import Path
        out = Path(os.environ.get("RL_DATA_DIR", "data"))
        out.mkdir(exist_ok=True)
        import polars as pl
        cohort = pl.read_parquet(out / "cohort.parquet")
        hadm_ids = set(cohort["hadm_id"].to_list())
        admittimes = cohort.select("hadm_id", "admittime")
        actions = extract_all_actions(hadm_ids, admittimes)
        actions.write_parquet(out / "actions_binned.parquet")
        print(f"Actions: {actions.height:,} rows")
    elif args.command == "pipeline":
        from run_pipeline import main as pipeline_main
        pipeline_main()
    elif args.command == "features":
        from src.pipeline.features import build_dataset
        traj_path = getattr(args, 'data', None) or os.environ.get("RL_DATA_PATH", "data/trajectories_v1.parquet")
        result = build_dataset(traj_path=traj_path, seed=args.seed)
        for k, v in result.items():
            if k != "zstats":
                print(f"  {k}: {v}")
    elif args.command == "train":
        from src.rl.train import train_iql, train_bc, auto_device, save_history
        import torch
        from src.config import N_ACTIONS
        data_dir = os.environ.get("RL_DATA_DIR", "data")
        if hasattr(args, 'data') and args.data:
            os.environ["RL_DATA_PATH"] = args.data
            from src.pipeline.features import build_dataset
            build_dataset(traj_path=args.data, seed=42)
        out = Path(f"{data_dir}/models")
        out.mkdir(parents=True, exist_ok=True)
        device = args.device or auto_device()
        print(f"Device: {device} | Seeds: {args.seeds}")

        arch_kwargs = {"arch": args.arch}
        if args.hidden_sizes:
            arch_kwargs["hidden_sizes"] = args.hidden_sizes

        for seed in args.seeds:
            print(f"\n=== Training IQL (seed={seed}, arch={args.arch}, n_actions={N_ACTIONS}) ===")
            result = train_iql(f"{data_dir}/dataset_v1/train.parquet", f"{data_dir}/dataset_v1/val.parquet",
                               epochs=args.epochs, seed=seed, device=device, batch_size=args.batch_size,
                               **arch_kwargs)
            torch.save(result["model"].q_net.state_dict(), out / f"iql_q_seed{seed}.pt")
            torch.save(result["model"].v_net.state_dict(), out / f"iql_v_seed{seed}.pt")
            save_history(result["history"], str(out / f"iql_history_seed{seed}.json"))
            print(f"  Final val loss: {result['history']['val'][-1]}")

            if not args.iql_only:
                print(f"=== Training BC (seed={seed}, arch={args.arch}) ===")
                bc = train_bc(f"{data_dir}/dataset_v1/train.parquet", f"{data_dir}/dataset_v1/val.parquet",
                              epochs=args.epochs, seed=seed, device=device, batch_size=args.batch_size,
                              **arch_kwargs)
                torch.save(bc["model"].net.state_dict(), out / f"bc_seed{seed}.pt")
                save_history(bc["history"], str(out / f"bc_history_seed{seed}.json"))
                print(f"  Final val loss: {bc['history']['val'][-1]}")
    elif args.command == "eval":
        from src.rl.evaluate import evaluate_all
        evaluate_all(device=args.device)
    elif args.command == "infer":
        import polars as pl
        import numpy as np
        from src.rl.inference import InferenceEnsemble, batch_inference, export_results, build_safety_mask
        from src.rl.dataset import _state_columns
        from src.config import N_ACTIONS, ACTION_BUNDLES

        model_dir = args.model_dir or os.environ.get("RL_DATA_DIR", "data") + "/models"
        output = args.output

        inp = Path(args.input)
        if inp.suffix == ".csv":
            df = pl.read_csv(str(inp))
        else:
            df = pl.read_parquet(str(inp))

        print(f"Loaded {df.height:,} rows from {args.input}")
        n_actions_infer = N_ACTIONS

        if args.state_cols:
            state_cols = args.state_cols
        else:
            state_cols = _state_columns(df)
            if not state_cols or state_cols[0] not in df.columns:
                state_cols = [c for c in df.columns if c not in (
                    "hadm_id", "bin_idx", "action_id", "reward", "subject_id",
                    "admittime", "dischtime", "charttime", "gender",
                )]
        print(f"State dim: {len(state_cols)}")

        states_np = df.select(state_cols).fill_null(0.0).to_numpy().astype(np.float32)

        state_dim = states_np.shape[1]
        ensemble = InferenceEnsemble(state_dim=state_dim, n_actions=n_actions_infer, model_dir=model_dir, seeds=args.seeds)

        safety_mask = None
        if args.safety:
            lab_cols = args.lab_cols or [c for c in df.columns if c != "hadm_id"]
            labs_np = df.select(lab_cols).fill_null(0.0).to_numpy().astype(np.float32) if lab_cols else None
            hadm_ids = df["hadm_id"].to_numpy() if "hadm_id" in df.columns else None
            diagnoses_hadm = set()
            safety_mask = build_safety_mask(
                n_rows=df.height, n_actions=n_actions_infer,
                labs=labs_np, lab_names=lab_cols if lab_cols else None,
                diagnoses_hadm=diagnoses_hadm, hadm_ids=hadm_ids,
            )
            print("Safety constraints applied")

        results = batch_inference(
            states_np, ensemble,
            batch_size=args.batch_size,
            safety_mask=safety_mask,
        )

        patient_ids = df[args.patient_id_col].to_numpy() if args.patient_id_col and args.patient_id_col in df.columns else None
        export_results(results, output, patient_ids=patient_ids, format=args.format)

        print(f"  Recommended action distribution:")
        action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()}
        unique, counts = np.unique(results["recommended_action"], return_counts=True)
        for a, c in sorted(zip(unique, counts)):
            print(f"    {action_names.get(int(a), f'action_{int(a)}')}: {c:,} ({100*c/len(results['recommended_action']):.1f}%)")
    elif args.command == "serve":
        model_dir = args.model_dir or os.environ.get("RL_DATA_DIR", "data") + "/models"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.rl.server import app, _ensemble, InferenceEnsemble
        import uvicorn
        if args.preload:
            _ensemble = InferenceEnsemble(state_dim=args.state_dim, model_dir=model_dir,
                                          seeds=[0, 1, 2, 3, 42])
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "test":
        import subprocess
        cmd = ["python3", "-m", "pytest", "tests/"]
        if args.filter:
            cmd.extend(["-k", args.filter])
        if args.verbose:
            cmd.append("-v")
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
