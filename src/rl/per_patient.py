"""Per-patient evaluation: score each patient, generate portfolios."""
import torch
import numpy as np
import polars as pl
from pathlib import Path
from src.config import N_ACTIONS, ACTION_BUNDLES
from src.rl.dataset import _state_columns
from src.rl.inference import InferenceEnsemble
from src.rl.interpret import interpret_patient, format_report


def score_patient(hadm_id, ensemble, df, state_cols, device="mps"):
    adm = df.filter(pl.col("hadm_id") == hadm_id).sort("bin_idx")
    if adm.height == 0:
        return None
    states_np = adm.select(state_cols).fill_null(0.0).to_numpy().astype(np.float32)
    states_t = torch.FloatTensor(states_np).to(device)
    pred = ensemble.predict(states_t)
    q_vals = pred["q_values"]
    actions = adm["action_id"].to_numpy().astype(int)
    n_bins = len(actions)
    q_clinician = q_vals[np.arange(n_bins), actions]
    q_best = q_vals.max(axis=1)
    best_actions = q_vals.argmax(axis=1)
    advantage = q_best - q_clinician
    agreement = (best_actions == actions).mean()
    return {
        "hadm_id": hadm_id, "n_bins": n_bins,
        "q_clinician_mean": float(np.mean(q_clinician)),
        "q_best_mean": float(np.mean(q_best)),
        "advantage_mean": float(np.mean(advantage)),
        "advantage_std": float(np.std(advantage)),
        "agreement_rate": float(agreement),
        "best_action_first": int(best_actions[0]),
        "best_action_first_name": ACTION_BUNDLES.get(int(best_actions[0]), {}).get("name", f"action_{int(best_actions[0])}"),
        "best_action_last": int(best_actions[-1]),
        "best_action_last_name": ACTION_BUNDLES.get(int(best_actions[-1]), {}).get("name", f"action_{int(best_actions[-1])}"),
        "first_q_range": [float(q_vals[0].min()), float(q_vals[0].max())],
    }


def score_all_patients(split="test", device="mps", max_patients=None):
    ds_dir = Path("data/dataset_v1")
    df = pl.read_parquet(str(ds_dir / f"{split}.parquet"))
    state_cols = _state_columns(df)
    states_np = df.select(state_cols).fill_null(0.0).to_numpy().astype(np.float32)
    ensemble = InferenceEnsemble(state_dim=len(state_cols), device=device)
    n_patients = df["hadm_id"].n_unique()
    if max_patients:
        keep = df.select("hadm_id").unique().sort("hadm_id").head(max_patients)
        df = df.filter(pl.col("hadm_id").is_in(keep["hadm_id"]))
    hadm_ids = df["hadm_id"].to_numpy()
    n = len(df)
    batch_size = 8192
    all_q = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.FloatTensor(states_np[start:end]).to(device)
        pred = ensemble.predict(batch)
        all_q.append(pred["q_values"])
    q_vals = np.concatenate(all_q, axis=0)
    actions = df["action_id"].to_numpy().astype(int)
    rewards = df["reward"].to_numpy()
    q_clinician = q_vals[np.arange(n), actions]
    q_best = q_vals.max(axis=1)
    best_actions = q_vals.argmax(axis=1)
    advantage = q_best - q_clinician
    agreement = (best_actions == actions).astype(float)
    survival = (rewards > 5).astype(float)
    result_df = pl.DataFrame({
        "hadm_id": hadm_ids,
        "q_clinician": q_clinician,
        "q_best": q_best,
        "advantage": advantage,
        "agreement": agreement,
        "survival_bin": survival,
        "reward": rewards,
        "action": actions,
        "best_action": best_actions,
    })
    per_patient = result_df.group_by("hadm_id").agg([
        pl.len().alias("n_bins"),
        pl.col("q_clinician").mean().alias("q_clinician_mean"),
        pl.col("q_best").mean().alias("q_best_mean"),
        pl.col("advantage").mean().alias("advantage_mean"),
        pl.col("advantage").std().alias("advantage_std"),
        pl.col("agreement").mean().alias("agreement_rate"),
        pl.col("survival_bin").max().alias("survived"),
        pl.col("reward").sum().alias("reward_total"),
        pl.col("reward").mean().alias("reward_mean"),
        pl.col("action").n_unique().alias("n_unique_actions"),
        pl.col("action").mode().first().alias("most_common_action"),
        pl.col("best_action").first().alias("best_action_first"),
        pl.col("best_action").last().alias("best_action_last"),
    ])
    action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()}
    per_patient = per_patient.with_columns([
        pl.col("most_common_action").replace_strict(action_names, default="unknown").alias("most_common_action_name"),
        pl.col("best_action_first").replace_strict(action_names, default="unknown").alias("best_action_first_name"),
        pl.col("best_action_last").replace_strict(action_names, default="unknown").alias("best_action_last_name"),
    ])
    return per_patient


def patient_summary(per_patient, top_k=10):
    print(f"Per-patient scores ({len(per_patient)} patients):")
    print(f"  Q clinician mean:  {per_patient['q_clinician_mean'].mean():.3f} ± {per_patient['q_clinician_mean'].std():.3f}")
    print(f"  Q best mean:       {per_patient['q_best_mean'].mean():.3f} ± {per_patient['q_best_mean'].std():.3f}")
    print(f"  Advantage mean:    {per_patient['advantage_mean'].mean():.3f} ± {per_patient['advantage_mean'].std():.3f}")
    print(f"  Agreement rate:    {per_patient['agreement_rate'].mean():.3f} ± {per_patient['agreement_rate'].std():.3f}")
    print(f"  Survival rate:     {per_patient['survived'].mean():.3f}")
    print(f"  Avg bins/patient:  {per_patient['n_bins'].mean():.1f}")
    print()
    print(f"Agreement distribution:")
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
    labels = ["0-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
    for i in range(len(labels)):
        lo, hi = bins[i], bins[i+1]
        cnt = per_patient.filter((pl.col("agreement_rate") >= lo) & (pl.col("agreement_rate") < hi)).height
        if lo == 0.9:
            cnt = per_patient.filter(pl.col("agreement_rate") >= 0.9).height
        print(f"  Agreement {labels[i]}: {cnt}")
    print()
    print(f"Top {top_k} patients requiring different action (highest advantage):")
    top = per_patient.sort("advantage_mean", descending=True).head(top_k)
    for r in top.iter_rows(named=True):
        print(f"  hadm={r['hadm_id']}: advantage={r['advantage_mean']:.3f} agreement={r['agreement_rate']:.1%} "
              f"recommends={r['best_action_first_name']}")


def compare_scenarios(device="mps", max_pairs=5):
    df = pl.read_parquet("data/dataset_v1/test.parquet")
    state_cols = _state_columns(df)
    ensemble = InferenceEnsemble(state_dim=len(state_cols), device=device)
    young = df.filter((pl.col("anchor_age") < 35) & (pl.col("hemoglobin_z").is_not_null())).select("hadm_id").unique().to_series().to_list()
    old = df.filter((pl.col("anchor_age") > 65) & (pl.col("hemoglobin_z").is_not_null())).select("hadm_id").unique().to_series().to_list()
    pairs = []
    for yh in young[:200]:
        yadm = df.filter(pl.col("hadm_id") == yh).sort("bin_idx")
        yage = yadm[0, "anchor_age"]
        yhb = yadm.select(pl.col("hemoglobin_z").drop_nulls()).to_series()
        if len(yhb) == 0: continue
        yhb_val = yhb[0]
        if abs(yhb_val) > 2: continue
        for oh in old:
            oadm = df.filter(pl.col("hadm_id") == oh).sort("bin_idx")
            oage = oadm[0, "anchor_age"]
            ohb = oadm.select(pl.col("hemoglobin_z").drop_nulls()).to_series()
            if len(ohb) == 0: continue
            if abs(yhb_val - ohb[0]) < 0.3:
                ys = score_patient(yh, ensemble, df, state_cols, device)
                os = score_patient(oh, ensemble, df, state_cols, device)
                pairs.append({
                    "young": yh, "old": oh,
                    "young_age": yage, "old_age": oage,
                    "hb_z": round(yhb_val, 2),
                    "y_best": ys["best_action_first_name"] if ys else "?",
                    "o_best": os["best_action_first_name"] if os else "?",
                    "y_q": round(ys["first_q_range"][1], 1) if ys else 0,
                    "o_q": round(os["first_q_range"][1], 1) if os else 0,
                    "y_n_bins": ys["n_bins"] if ys else 0,
                    "o_n_bins": os["n_bins"] if os else 0,
                })
                if len(pairs) >= max_pairs: break
        if len(pairs) >= max_pairs: break
    for p in pairs:
        same = "SAME" if p["y_best"] == p["o_best"] else "DIFF"
        print(f"[{same}] Hb z={p['hb_z']} | Young(age {p['young_age']}): {p['y_best']} (Q={p['y_q']}, {p['y_n_bins']} bins) vs "
              f"Old(age {p['old_age']}): {p['o_best']} (Q={p['o_q']}, {p['o_n_bins']} bins)")
