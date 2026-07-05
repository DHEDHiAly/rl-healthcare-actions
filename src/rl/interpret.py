import torch
import numpy as np
import polars as pl
from pathlib import Path
from src.config import MIMIC_DATA_DIR, ACTION_BUNDLES, LAB_FEATURES, VITAL_FEATURES
from src.rl.dataset import FlatDataset, _state_columns
from src.rl.train import IQL
from src.rl.inference import InferenceEnsemble
from src.rl.world_model import load_world_model, counterfactual as cf_simulate

DS_DIR = Path("data/dataset_v1")
MODELS_DIR = Path("data/models")
MIMIC_HOSP = Path(MIMIC_DATA_DIR) / "hosp"
MIMIC_ICU = Path(MIMIC_DATA_DIR) / "icu"


def _load_patient_data(hadm_id: int):
    diag = pl.scan_csv(str(MIMIC_HOSP / "diagnoses_icd.csv.gz"), infer_schema_length=10000)
    proc = pl.scan_csv(str(MIMIC_HOSP / "procedures_icd.csv.gz"), infer_schema_length=10000)
    adm = pl.scan_csv(str(MIMIC_HOSP / "admissions.csv.gz"), infer_schema_length=10000)
    pat = pl.scan_csv(str(MIMIC_HOSP / "patients.csv.gz"), infer_schema_length=10000)
    diag_d = pl.scan_csv(str(MIMIC_HOSP / "d_icd_diagnoses.csv.gz"), infer_schema_length=10000)

    diag_p = diag.filter(pl.col("hadm_id") == hadm_id).collect()
    proc_p = proc.filter(pl.col("hadm_id") == hadm_id).collect()
    adm_p = adm.filter(pl.col("hadm_id") == hadm_id).collect()
    sid = adm_p.select("subject_id").unique()
    pat_p = pat.filter(pl.col("subject_id") == sid.item()).collect()

    diag_codes = [str(r.get("icd_code", "?")) for r in diag_p.iter_rows(named=True)]
    proc_codes = [str(r.get("icd_code", "?")) for r in proc_p.iter_rows(named=True)]

    patient = {}
    if pat_p.height > 0:
        p = pat_p.row(0, named=True)
        patient = {"age": p.get("anchor_age"), "gender": p.get("gender")}
    if adm_p.height > 0:
        a = adm_p.row(0, named=True)
        patient["race"] = a.get("race")
        patient["admission_type"] = a.get("admission_type")
        if a.get("dischtime") and a.get("admittime"):
            from datetime import datetime
            dt = datetime.strptime(str(a["dischtime"]), "%Y-%m-%d %H:%M:%S")
            at = datetime.strptime(str(a["admittime"]), "%Y-%m-%d %H:%M:%S")
            los = (dt - at).total_seconds() / 86400.0
            patient["los_days"] = round(los, 1)

    return {"diagnoses": diag_codes, "procedures": proc_codes, "patient": patient}


def _action_name(aid):
    return ACTION_BUNDLES.get(aid, {}).get("name", f"action_{aid}")


def compute_action_attribution(state, action, q_net, device="mps"):
    s = torch.FloatTensor(state).unsqueeze(0).to(device)
    s.requires_grad_(True)
    q = q_net(s)
    qa = q[0, action]
    qa.backward()
    grad = s.grad.squeeze().detach()
    x = s.squeeze().detach()
    attr = (grad * x).cpu().numpy()
    return attr


def interpret_patient(hadm_id, device="mps"):
    # Load patient data
    info = _load_patient_data(hadm_id)
    test = pl.read_parquet(str(DS_DIR / "test.parquet"))
    adm = test.filter(pl.col("hadm_id") == hadm_id).sort("bin_idx")
    if adm.height == 0:
        adm = pl.read_parquet(str(DS_DIR / "train.parquet")).filter(pl.col("hadm_id") == hadm_id).sort("bin_idx")
    if adm.height == 0:
        return {"error": f"hadm_id {hadm_id} not found in dataset"}

    state_cols = _state_columns(adm)
    feat_names = state_cols
    z_cols = [c for c in feat_names if c.endswith("_z")]
    missing_cols = [c for c in feat_names if c.endswith("_missing")]

    # Load ensemble
    ensemble = InferenceEnsemble(state_dim=len(state_cols), device=device)

    # Load world model
    wm = load_world_model(device)

    # For each bin, generate recommendation
    bins = []
    for bin_idx in range(adm.height):
        row = adm.row(bin_idx, named=True)
        state = np.array([row[c] if row[c] is not None else 0.0 for c in state_cols], dtype=np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Ensemble prediction
        pred = ensemble.predict(state_t)
        q_values = pred["q_values"][0]
        ci_lower = pred["q_ci_lower"][0]
        ci_upper = pred["q_ci_upper"][0]
        best_action = int(np.argmax(q_values))
        best_q = float(q_values[best_action])
        ci_low = float(ci_lower[best_action])
        ci_high = float(ci_upper[best_action])

        # Current action and reward
        current_action = int(row.get("action_id", -1))
        current_reward = float(row.get("reward", 0.0))

        # Feature attribution (why this action?)
        attr = compute_action_attribution(state, best_action, ensemble.q_nets[0], device)

        # Top contributing features (Gradient x Input = contribution in Q-value units)
        feat_contrib = [(feat_names[i], float(attr[i])) for i in range(len(attr))]
        feat_contrib = [(n, v) for n, v in feat_contrib if abs(v) > 1e-4]
        feat_contrib.sort(key=lambda x: abs(x[1]), reverse=True)

        # Abnormal labs
        abnormal_labs = []
        for zc in z_cols:
            zv = row.get(zc, 0.0)
            if zv is not None and abs(zv) > 2.0:
                lab_name = zc.replace("_z", "")
                abnormal_labs.append({"lab": lab_name, "z_score": round(zv, 2)})

        # Missing labs
        missing_labs = []
        for mc in missing_cols:
            if row.get(mc, 0) == 1:
                lab_name = mc.replace("_missing", "")
                missing_labs.append(lab_name)

        # Counterfactual: best action vs current action
        cf = None
        if wm is not None and current_action >= 0 and best_action != current_action:
            cf = cf_simulate(wm, state, best_action, current_action, device)

        # Action ranking
        action_ranking = []
        ranked = sorted(enumerate(q_values), key=lambda x: x[1], reverse=True)
        for aid, qv in ranked[:5]:
            action_ranking.append({
                "action_id": aid,
                "action_name": _action_name(aid),
                "q_value": round(float(qv), 2),
                "ci_95": [round(float(ci_lower[aid]), 2), round(float(ci_upper[aid]), 2)],
            })

        bins.append({
            "bin_idx": bin_idx,
            "time_from_admission_hrs": bin_idx * 4,
            "current_action": {"id": current_action, "name": _action_name(current_action)},
            "current_reward": current_reward,
            "recommendation": {
                "action_id": best_action,
                "action_name": _action_name(best_action),
                "q_value": best_q,
                "ci_95": [round(ci_low, 2), round(ci_high, 2)],
            },
            "action_ranking": action_ranking,
            "abnormal_labs": abnormal_labs[:10],
            "missing_labs": missing_labs[:5],
            "top_features": [{"feature": fn, "contribution": round(fc, 4)} for fn, fc in feat_contrib[:5]],
            "counterfactual": cf,
        })

    return {
        "patient": info["patient"],
        "diagnoses": info["diagnoses"][:20],
        "procedures": info["procedures"][:10],
        "total_bins": adm.height,
        "state_dim": len(state_cols),
        "bin_recommendations": bins,
    }


def _fmt_abnormal(labs):
    return ", ".join(f'{l["lab"]}(z={l["z_score"]:+.1f})' for l in labs)

def _fmt_features(feats):
    return ", ".join(f'{f["feature"]}({f["contribution"]:+.3f})' for f in feats)

def _fmt_actions(actions):
    return " | ".join(f'{a["action_name"]}({a["q_value"]})' for a in actions)

def format_report(result, max_bins=5):
    p = result["patient"]
    lines = []
    lines.append(f"Patient: age={p.get('age', '?')}, gender={p.get('gender', '?')}, race={p.get('race', '?')}")
    if p.get("admission_type"):
        lines.append(f"Admission: {p['admission_type']}, LOS={p.get('los_days', '?')} days")
    if result["diagnoses"]:
        diag_str = ", ".join(result["diagnoses"][:10])
        lines.append(f"Diagnoses ({len(result['diagnoses'])}): {diag_str}")
    if result["procedures"]:
        proc_str = ", ".join(result["procedures"][:5])
        lines.append(f"Procedures ({len(result['procedures'])}): {proc_str}")
    lines.append(f"Total timesteps: {result['total_bins']} (4-hour bins, {result['total_bins'] * 4}h window)")
    lines.append("")

    for b in result["bin_recommendations"][:max_bins]:
        lines.append(f"--- T+{b['time_from_admission_hrs']}h (bin {b['bin_idx']}) ---")
        rec = b["recommendation"]
        lines.append(f"  Clinician: {b['current_action']['name']} (reward={b['current_reward']:.2f})")
        ci = f"[{rec['ci_95'][0]:.2f}, {rec['ci_95'][1]:.2f}]"
        lines.append(f"  Recommend: {rec['action_name']} (Q={rec['q_value']:.2f}, 95% CI {ci})")
        if b["abnormal_labs"]:
            lines.append(f"  Abnormal labs: {_fmt_abnormal(b['abnormal_labs'])}")
        if b["missing_labs"]:
            ml_str = ", ".join(b["missing_labs"])
            lines.append(f"  Missing labs: {ml_str}")
        lines.append(f"  Key drivers: {_fmt_features(b['top_features'])}")
        if b["action_ranking"]:
            top3 = b["action_ranking"][:3]
            lines.append(f"  Top 3: {_fmt_actions(top3)}")
        if b["counterfactual"]:
            cf = b["counterfactual"]
            a_name = _action_name(cf["action_a"])
            b_name = _action_name(cf["action_b"])
            lines.append(f"  Counterfactual: {a_name} predicted reward={cf['reward_a']:.2f} vs {b_name}={cf['reward_b']:.2f} (delta={cf['delta']:+.2f})")
        lines.append("")

    return "\n".join(lines)
