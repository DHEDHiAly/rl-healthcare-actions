"""Phase 4: Off-Policy Evaluation, safety audit, and statistical rigor."""

import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
import csv
import json
from typing import Dict, List, Optional
from pathlib import Path
from src.rl.train import IQL, BehaviorCloning, wis_ope, fqe_ope, policy_value, auto_device, MLP
from src.rl.dataset import FlatDataset
from src.config import SAFETY_CONSTRAINTS, ACTION_BUNDLES, N_ACTIONS

import os

DATA_DIR = Path(os.environ.get("RL_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data"))
DS_DIR = DATA_DIR / "dataset_v1"
MODEL_DIR = DATA_DIR / "models"
MIMIC_DIR = Path(os.environ.get("MIMIC_DATA_DIR", "/Users/farasatdhedhi/mimic_pipeline/mimiciv_3.1"))


def _get_state_dim() -> int:
    p = DS_DIR / "train.parquet"
    if not p.exists():
        return 100
    ds = FlatDataset(str(p))
    return ds.state_dim


def _infer_arch_from_checkpoint(path, state_dim, n_actions, device):
    chk = torch.load(str(path), map_location=device, weights_only=True)
    keys = list(chk.keys())
    has_lstm = any("lstm" in k for k in keys)
    if has_lstm:
        return IQL(state_dim=state_dim, n_actions=n_actions, device=device, arch="lstm")

    # Collect hidden sizes from all Linear weight keys except the last (output) layer
    weight_keys = [k for k in keys if k.endswith(".weight")]
    last_weight = weight_keys[-1] if weight_keys else None
    hidden_sizes = []
    for k in weight_keys:
        if k == last_weight:
            continue
        hidden_sizes.append(chk[k].shape[0])

    hs = hidden_sizes or [256, 256]
    return IQL(state_dim=state_dim, n_actions=n_actions, device=device, arch="mlp", hidden_sizes=hs)


def _load_best_iql(device: Optional[str] = None, prefix: str = "iql", seed: int = 42):
    device = device or auto_device()
    state_dim = _get_state_dim()
    q_path = MODEL_DIR / f"{prefix}_q_seed{seed}.pt"
    m = _infer_arch_from_checkpoint(q_path, state_dim, N_ACTIONS, device)
    m.q_net.load_state_dict(torch.load(str(q_path), map_location=device, weights_only=True))
    m.v_net.load_state_dict(torch.load(str(MODEL_DIR / f"{prefix}_v_seed{seed}.pt"), map_location=device, weights_only=True))
    return m


def _load_bc(device: Optional[str] = None, prefix: str = "bc", seed: int = 42):
    device = device or auto_device()
    state_dim = _get_state_dim()
    m = BehaviorCloning(state_dim=state_dim, n_actions=N_ACTIONS, device=device)
    m.net.load_state_dict(torch.load(str(MODEL_DIR / f"{prefix}_seed{seed}.pt"), map_location=device, weights_only=True))
    return m


def _load_pi_beta():
    bp_path = DS_DIR / "behavior_policy.csv"
    if not bp_path.exists():
        return np.ones(N_ACTIONS) / N_ACTIONS
    with open(str(bp_path)) as f:
        reader = csv.DictReader(f)
        bp = {int(r["action_id"]): float(r["pi_beta"]) for r in reader}
    return np.array([bp.get(a, 1e-8) for a in range(N_ACTIONS)])


class DirectMethodOPE:
    def __init__(self, state_dim: int, n_actions: int = N_ACTIONS, hidden: int = 256, lr: float = 3e-4, dropout: float = 0.2, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net = MLP(state_dim, hidden, n_actions, dropout).to(self.device)
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def fit(self, ds: FlatDataset, epochs: int = 20, batch_size: int = 2048):
        self.q_net.train()
        n = len(ds)
        for epoch in range(epochs):
            perm = torch.randperm(n, device=ds.states.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                q = self.q_net(ds.states[idx]).gather(1, ds.actions[idx].unsqueeze(1))
                loss = F.mse_loss(q, ds.rewards[idx].unsqueeze(1))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                epoch_loss += loss.detach().item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                print(f"  DM epoch {epoch+1}/{epochs} loss={epoch_loss/n_batches:.4f}")

    def evaluate(self, policy, ds: FlatDataset, n_episodes: int = 2000) -> float:
        self.q_net.eval()
        dev = self.device
        dones_np = ds.dones.cpu().numpy()
        ep_ends = np.where(dones_np == 1.0)[0] + 1
        ep_starts = np.concatenate([[0], ep_ends[:-1]])
        n_ep = min(len(ep_starts), n_episodes)
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
        chosen.sort()

        values = []
        for i in chosen:
            s = ep_starts[i]
            e = ep_ends[i] if i < len(ep_ends) else len(dones_np)
            states = ds.states[s:e].to(dev)
            with torch.no_grad():
                pi_e = policy.policy(states)
                q = self.q_net(states)
                v_pi_e = (pi_e * q).sum(dim=1).mean().item()
            values.append(v_pi_e)
        return float(np.mean(values))


def _episode_bounds(ds: FlatDataset):
    dones_np = ds.dones.cpu().numpy()
    ep_ends = np.where(dones_np == 1.0)[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    return ep_starts, ep_ends


def apply_safety_mask(pi: np.ndarray, test: pl.DataFrame, malignancy_hadm: set) -> np.ndarray:
    """Zero out policy probability on unsafe actions given state context, then renormalize."""
    if not SAFETY_CONSTRAINTS:
        return pi
    test_hadm = test["hadm_id"].to_numpy()

    if any(c["id"] == "S1" for c in SAFETY_CONSTRAINTS):
        malign_mask = np.isin(test_hadm, list(malignancy_hadm))
        pi[malign_mask, 5] = 0.0

    if any(c["id"] == "S3" for c in SAFETY_CONSTRAINTS):
        if "platelets" in test.columns:
            plt_vals = test["platelets"].fill_null(999.0).to_numpy()
            pi[plt_vals >= 50.0, 2] = 0.0

    if any(c["id"] == "S4" for c in SAFETY_CONSTRAINTS):
        if "inr" in test.columns:
            inr_vals = test["inr"].fill_null(0.0).to_numpy()
            pi[inr_vals <= 2.0, 3] = 0.0

    if any(c["id"] == "S5" for c in SAFETY_CONSTRAINTS):
        if "mean_bp" in test.columns:
            map_vals = test["mean_bp"].fill_null(100.0).to_numpy()
            pi[map_vals >= 65.0, 9] = 0.0

    if any(c["id"] == "S6" for c in SAFETY_CONSTRAINTS):
        if "glucose" in test.columns:
            glu_vals = test["glucose"].fill_null(100.0).to_numpy()
            pi[glu_vals < 70.0, 11] = 0.0

    if any(c["id"] == "S7" for c in SAFETY_CONSTRAINTS):
        if "creatinine" in test.columns and "mean_bp" in test.columns:
            cr_vals = test["creatinine"].fill_null(0.0).to_numpy()
            map_vals2 = test["mean_bp"].fill_null(100.0).to_numpy()
            pi[(cr_vals > 4.0) & (map_vals2 < 65.0), 12] = 0.0

    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    pi = pi / row_sums
    return pi


def run_ope_suite(device: Optional[str] = None) -> Dict:
    device = device or auto_device()
    dev = torch.device(device)
    iql = _load_best_iql(device)
    bc = _load_bc(device) if (MODEL_DIR / "bc_seed42.pt").exists() else None
    pi_beta = _load_pi_beta()

    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test_ds.to_device(dev)

    results = {}
    for name, policy in [("iql", iql)]:
        wis = wis_ope(policy, test_ds, pi_beta, n_episodes=2000)
        fqe = fqe_ope(policy, test_ds, n_episodes=2000)
        pv = policy_value(policy, test_ds, n_episodes=2000)
        results[name] = {"wis": wis, "fqe": fqe, "pv": pv}

    if bc is not None:
        results["bc"] = {"wis": wis_ope(bc, test_ds, pi_beta, n_episodes=2000),
                          "fqe": fqe_ope(bc, test_ds, n_episodes=2000),
                          "pv": policy_value(bc, test_ds, n_episodes=2000)}

    state_dim = _get_state_dim()
    dm = DirectMethodOPE(state_dim=state_dim, device=device)
    dm.fit(test_ds, epochs=20, batch_size=2048)
    results["iql"]["dm"] = dm.evaluate(iql, test_ds, n_episodes=2000)
    if bc is not None:
        results["bc"]["dm"] = dm.evaluate(bc, test_ds, n_episodes=2000)

    ep_starts, ep_ends = _episode_bounds(test_ds)
    rng = np.random.default_rng(42)
    n_ep = min(len(ep_starts), 2000)
    chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
    chosen.sort()
    beh_returns = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(test_ds)
        g = float(np.sum(test_ds.discounts[s:e].cpu().numpy() * test_ds.rewards[s:e].cpu().numpy()))
        beh_returns.append(g)
    results["behavior"] = {"fqe": float(np.mean(beh_returns))}

    return results


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95) -> Dict:
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(np.mean(sample))
    lower = np.percentile(boot, (1 - ci) / 2 * 100)
    upper = np.percentile(boot, (1 + ci) / 2 * 100)
    return {"mean": float(np.mean(values)), "ci_lower": float(lower), "ci_upper": float(upper), "n_resamples": n_resamples}


def policy_efficacy_bootstrap(device: Optional[str] = None) -> Dict:
    device = device or auto_device()
    dev = torch.device(device)
    iql = _load_best_iql(device)
    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test_ds.to_device(dev)

    ep_starts, ep_ends = _episode_bounds(test_ds)
    rng = np.random.default_rng(42)
    n_ep = min(len(ep_starts), 5000)
    chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
    chosen.sort()

    iql_returns = []
    behavior_returns = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(test_ds)
        rewards = test_ds.rewards[s:e].cpu().numpy()
        discounts = test_ds.discounts[s:e].cpu().numpy()
        g = float(np.sum(discounts * rewards))
        behavior_returns.append(g)
        states = test_ds.states[s:e]
        pi = iql.policy(states).cpu().numpy()
        actions = test_ds.actions[s:e].cpu().numpy()
        pi_a = pi[np.arange(len(actions)), actions]
        iql_returns.append(g * pi_a.mean())

    iql_ci = bootstrap_ci(np.array(iql_returns))
    beh_ci = bootstrap_ci(np.array(behavior_returns))
    overlaps = iql_ci["ci_lower"] < beh_ci["ci_upper"] and beh_ci["ci_lower"] < iql_ci["ci_upper"]
    return {"iql": iql_ci, "behavior": beh_ci, "overlaps": overlaps, "iql_mean_higher": iql_ci["mean"] > beh_ci["mean"]}


def _load_malignancy_hadm() -> set:
    s1 = [c for c in SAFETY_CONSTRAINTS if c.get("id") == "S1"]
    if not s1:
        return set()
    malignancy_icd10_prefixes = s1[0].get("icd_prefix", [])
    diag_path = MIMIC_DIR / "hosp" / "diagnoses_icd.csv.gz"
    if not diag_path.exists():
        return set()
    diag = pl.scan_csv(str(diag_path), infer_schema_length=10000).collect()
    malignancy_hadm = set()
    for row in diag.iter_rows(named=True):
        code = str(row["icd_code"]).strip()
        version = row["icd_version"]
        if version == 10:
            for prefix in malignancy_icd10_prefixes:
                if code.startswith(prefix):
                    malignancy_hadm.add(row["hadm_id"])
                    break
        elif version == 9:
            if code.startswith("1") and len(code) >= 3 and 140 <= int(code[:3]) <= 209:
                malignancy_hadm.add(row["hadm_id"])
    return malignancy_hadm


def safety_audit(device: Optional[str] = None) -> Dict:
    """Check safety constraints after applying post-hoc action masking."""
    violations = {"n_violations": 0, "violations": [], "constraints_checked": [], "pre_mask_violations": {}, "post_mask_violations": 0}
    if not SAFETY_CONSTRAINTS:
        return violations

    device = device or auto_device()
    dev = torch.device(device)
    iql = _load_best_iql(device)
    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test = pl.read_parquet(str(DS_DIR / "test.parquet"))
    malignancy_hadm = _load_malignancy_hadm()

    with torch.no_grad():
        pi_raw = iql.policy(test_ds.states.to(dev)).cpu().numpy()
    pi_safe = apply_safety_mask(pi_raw.copy(), test, malignancy_hadm)
    policy_actions = pi_safe.argmax(axis=1)

    violations = []
    test_hadm = test["hadm_id"].to_numpy()

    for c in SAFETY_CONSTRAINTS:
        cid = c["id"]
        if cid == "S1":
            for idx in range(len(test)):
                if policy_actions[idx] == 5 and test_hadm[idx] in malignancy_hadm:
                    violations.append({"constraint": "S1", "hadm_id": int(test_hadm[idx]), "action": 5, "bin_idx": int(test["bin_idx"][idx])})

        elif cid == "S3":
            if "platelets" in test.columns:
                plt_vals = test["platelets"].fill_null(999.0).to_numpy()
                for idx in range(len(test)):
                    if policy_actions[idx] == 2 and plt_vals[idx] >= 50.0:
                        violations.append({"constraint": "S3", "hadm_id": int(test_hadm[idx]), "action": 2, "platelets": float(plt_vals[idx]), "bin_idx": int(test["bin_idx"][idx])})

        elif cid == "S4":
            if "inr" in test.columns:
                inr_vals = test["inr"].fill_null(0.0).to_numpy()
                for idx in range(len(test)):
                    if policy_actions[idx] == 3 and inr_vals[idx] <= 2.0:
                        violations.append({"constraint": "S4", "hadm_id": int(test_hadm[idx]), "action": 3, "inr": float(inr_vals[idx]), "bin_idx": int(test["bin_idx"][idx])})

        elif cid == "S5":
            if "mean_bp" in test.columns:
                map_vals = test["mean_bp"].fill_null(100.0).to_numpy()
                for idx in range(len(test)):
                    if policy_actions[idx] == 9 and map_vals[idx] >= 65.0:
                        violations.append({"constraint": "S5", "hadm_id": int(test_hadm[idx]), "action": 9, "mean_bp": float(map_vals[idx]), "bin_idx": int(test["bin_idx"][idx])})

        elif cid == "S6":
            if "glucose" in test.columns:
                glu_vals = test["glucose"].fill_null(100.0).to_numpy()
                for idx in range(len(test)):
                    if policy_actions[idx] == 11 and glu_vals[idx] < 70.0:
                        violations.append({"constraint": "S6", "hadm_id": int(test_hadm[idx]), "action": 11, "glucose": float(glu_vals[idx]), "bin_idx": int(test["bin_idx"][idx])})

        elif cid == "S7":
            if "creatinine" in test.columns and "mean_bp" in test.columns:
                cr_vals = test["creatinine"].fill_null(0.0).to_numpy()
                map_vals2 = test["mean_bp"].fill_null(100.0).to_numpy()
                for idx in range(len(test)):
                    if policy_actions[idx] == 12 and cr_vals[idx] > 4.0 and map_vals2[idx] < 65.0:
                        violations.append({"constraint": "S7", "hadm_id": int(test_hadm[idx]), "action": 12, "creatinine": float(cr_vals[idx]), "bin_idx": int(test["bin_idx"][idx])})

    return {
        "n_violations": len(violations),
        "violations": violations[:20],
        "constraints_checked": [c["id"] for c in SAFETY_CONSTRAINTS],
        "pre_mask_violations": {},
        "post_mask_violations": len(violations),
    }


def phenotype_stratification(device: Optional[str] = None) -> Dict:
    device = device or auto_device()
    dev = torch.device(device)
    iql = _load_best_iql(device)
    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test_ds.to_device(dev)
    test = pl.read_parquet(str(DS_DIR / "test.parquet"))
    diag_path = MIMIC_DIR / "hosp" / "diagnoses_icd.csv.gz"

    if not diag_path.exists():
        return {"error": "diagnoses_icd.csv.gz not found"}

    diag = pl.scan_csv(str(diag_path), infer_schema_length=10000).filter(pl.col("seq_num") == 1).collect()
    primary = diag.select(["hadm_id", "icd_code", "icd_version"]).unique(subset=["hadm_id"])
    primary = primary.with_columns(pl.col("hadm_id").cast(pl.Int64))
    test_with_diag = test.join(primary, on="hadm_id", how="left")
    test_with_diag = test_with_diag.with_columns(
        pl.col("icd_code").cast(pl.Utf8).str.slice(0, 3).alias("icd_group")
    )

    groups = test_with_diag.group_by("icd_group").agg(pl.len().alias("n")).sort("n", descending=True).head(50)
    test_hadm_np = test["hadm_id"].to_numpy()
    ep_starts, ep_ends = _episode_bounds(test_ds)
    ep_hadm = test_hadm_np[ep_starts]

    ep_beh_returns = np.zeros(len(ep_starts))
    ep_iql_returns = np.zeros(len(ep_starts))
    for i in range(len(ep_starts)):
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(test_ds)
        rewards = test_ds.rewards[s:e].cpu().numpy()
        discounts = test_ds.discounts[s:e].cpu().numpy()
        g = float(np.sum(discounts * rewards))
        ep_beh_returns[i] = g
        states = test_ds.states[s:e]
        pi = iql.policy(states).cpu().numpy()
        actions = test_ds.actions[s:e].cpu().numpy()
        pi_a = pi[np.arange(len(actions)), actions]
        ep_iql_returns[i] = g * pi_a.mean()

    ep_df = pl.DataFrame({
        "hadm_id": ep_hadm,
        "beh_return": ep_beh_returns,
        "iql_return": ep_iql_returns,
    })
    icd_group_map = test_with_diag.select(["hadm_id", "icd_group"]).unique(subset=["hadm_id"])
    ep_df = ep_df.join(icd_group_map, on="hadm_id", how="left")

    results = []
    for group in groups["icd_group"].to_list():
        group_eps = ep_df.filter(pl.col("icd_group") == group)
        if group_eps.height < 10:
            continue
        beh_mean = float(group_eps["beh_return"].mean())
        iql_mean = float(group_eps["iql_return"].mean())
        results.append({
            "icd_group": group,
            "n_episodes": group_eps.height,
            "behavior_mean": round(beh_mean, 4),
            "iql_mean": round(iql_mean, 4),
            "iql_better": iql_mean > beh_mean,
        })
    failing = [r for r in results if not r["iql_better"]]
    return {"groups_checked": len(results), "top_groups": results[:10], "failing_groups": failing}


def plausibility_check(device: Optional[str] = None, n_trajectories: int = 5) -> Dict:
    device = device or auto_device()
    dev = torch.device(device)
    iql = _load_best_iql(device)
    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test_ds.to_device(dev)
    test = pl.read_parquet(str(DS_DIR / "test.parquet"))

    ep_starts, ep_ends = _episode_bounds(test_ds)
    ep_returns = []
    for i in range(len(ep_starts)):
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(test_ds)
        rewards = test_ds.rewards[s:e].cpu().numpy()
        discounts = test_ds.discounts[s:e].cpu().numpy()
        ep_returns.append(float(np.sum(discounts * rewards)))

    top_idx = np.argsort(ep_returns)[::-1][:n_trajectories]
    action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()}
    test_hadm_np = test["hadm_id"].to_numpy()
    trajectories = []
    for rank, idx in enumerate(top_idx):
        s = ep_starts[idx]
        e = ep_ends[idx] if idx < len(ep_ends) else len(test_ds)
        states = test_ds.states[s:e]
        with torch.no_grad():
            pi = iql.policy(states).cpu().numpy()
        policy_acts = pi.argmax(axis=1).tolist()
        hadm = int(test_hadm_np[s])
        ret = ep_returns[idx]
        trajectories.append({
            "rank": rank + 1,
            "hadm_id": hadm,
            "return": round(ret, 2),
            "length": e - s,
            "top_policy_actions": [action_names.get(a, str(a)) for a in set(policy_acts)],
        })
    return {"trajectories": trajectories}


def evaluate_all(device: Optional[str] = None) -> Dict:
    device = device or auto_device()
    print(f"Phase 4 evaluation — device: {device}")

    print("Running OPE suite...", flush=True)
    ope = run_ope_suite(device)

    print("Running policy efficacy bootstrap...", flush=True)
    efficacy = policy_efficacy_bootstrap(device)

    print("Running phenotype stratification...", flush=True)
    phenotype = phenotype_stratification(device)

    print("Running safety audit...", flush=True)
    safety = safety_audit(device)

    print("Running plausibility check...", flush=True)
    plausibility = plausibility_check(device)

    print("Computing bootstrap CIs...", flush=True)
    dev = torch.device(device)
    test_ds = FlatDataset(str(DS_DIR / "test.parquet"))
    test_ds.to_device(dev)
    iql = _load_best_iql(device)
    ep_starts, ep_ends = _episode_bounds(test_ds)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(ep_starts), size=min(len(ep_starts), 3000), replace=False)
    chosen.sort()
    pv_values = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(test_ds)
        g = float(np.sum(test_ds.discounts[s:e].cpu().numpy() * test_ds.rewards[s:e].cpu().numpy()))
        pi = iql.policy(test_ds.states[s:e]).cpu().numpy()
        a = test_ds.actions[s:e].cpu().numpy()
        pv_values.append(g * pi[np.arange(len(a)), a].mean())
    boot = bootstrap_ci(np.array(pv_values), n_resamples=1000)

    report = {
        "ope": ope,
        "efficacy": efficacy,
        "phenotype": phenotype,
        "safety": safety,
        "plausibility": plausibility,
        "bootstrap_ci": boot,
        "n_actions": N_ACTIONS,
    }
    out = MODEL_DIR / "phase4_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report saved to {out}")

    print("\n=== Phase 4 Summary ===")
    for policy_name in ["iql", "bc"]:
        if policy_name in ope:
            print(f"{policy_name}: WIS={ope[policy_name]['wis']:.2f} FQE={ope[policy_name]['fqe']:.2f} PV={ope[policy_name]['pv']:.4f}")
    print(f"Policy efficacy: IQL mean={efficacy['iql']['mean']:.4f}, Behavior mean={efficacy['behavior']['mean']:.4f}, CIs overlap={efficacy['overlaps']}")
    print(f"Phenotype groups: {phenotype.get('groups_checked', 0)} checked, {len(phenotype.get('failing_groups', []))} failing")
    print(f"Safety violations: {safety['n_violations']}")
    print(f"Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL models for healthcare actions")
    parser.add_argument("--data-dir", default="data/dataset_v1", help="Dataset directory")
    parser.add_argument("--model-dir", default="data/models", help="Model directory")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda/mps)")
    parser.add_argument("--subtask", choices=["all", "ope", "safety", "phenotype", "plausibility", "bootstrap"],
                        default="all", help="Which evaluation to run")
    args = parser.parse_args()
    DS_DIR = DATA_DIR / args.data_dir if args.data_dir != "data/dataset_v1" else Path(args.data_dir)
    MODEL_DIR = Path(args.model_dir)
    evaluate_all(device=args.device)
