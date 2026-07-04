import os
import json
import polars as pl
from typing import Dict, List, Optional, Any

BIN_HOURS = 4
MIN_BINS = 6
LOCF_MAX_GAP_HOURS = 24

MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR", "/Users/farasatdhedhi/mimic_pipeline/mimiciv_3.1")

# Columns to exclude when auto-detecting features from data
AUTO_META_COLS = {
    "hadm_id", "subject_id", "bin_idx", "action_id", "reward",
    "admittime", "dischtime", "charttime", "deathtime",
    "gender", "anchor_age", "hospital_expire_flag", "los_days",
    "itemid", "valuenum", "drug", "starttime",
    "icd_code", "icd_version", "seq_num", "label",
    "patient_id", "admission_id", "admit_time", "discharge_time",
    "death_time", "expire_flag", "chart_time", "item_id",
    "value_num", "drug_name", "start_time",
    "los_days",
}

# ==========================================================
#  Config loading
# ==========================================================
# 1) $RL_CONFIG = path to JSON file with your config
# 2) Individual env vars overrides per key (RL_CFG_<KEY>)
# 3) Default = MIMIC-hematology config (backward compat)
# ==========================================================

_MIMIC_LAB_FEATURES = {
    "wbc": {"itemid": 51301, "unit": "K/uL", "lo": 4.5, "hi": 11.0},
    "rbc": {"itemid": 51279, "unit": "M/uL", "lo": 4.0, "hi": 5.5},
    "hemoglobin": {"itemid": 51222, "unit": "g/dL", "lo": 12.0, "hi": 16.0},
    "hematocrit": {"itemid": 51221, "unit": "%", "lo": 36.0, "hi": 46.0},
    "mcv": {"itemid": 51250, "unit": "fL", "lo": 80.0, "hi": 100.0},
    "mch": {"itemid": 51248, "unit": "pg", "lo": 27.0, "hi": 33.0},
    "mchc": {"itemid": 51249, "unit": "g/dL", "lo": 32.0, "hi": 36.0},
    "platelets": {"itemid": 51265, "unit": "K/uL", "lo": 150.0, "hi": 400.0},
    "serum_iron": {"itemid": 50952, "unit": "ug/dL", "lo": 50.0, "hi": 170.0},
    "ferritin": {"itemid": 50924, "unit": "ng/mL", "lo": 10.0, "hi": 200.0},
    "tibc": {"itemid": 50953, "unit": "ug/dL", "lo": 240.0, "hi": 450.0},
    "pt": {"itemid": 51274, "unit": "sec", "lo": 11.0, "hi": 13.5},
    "ptt": {"itemid": 51275, "unit": "sec", "lo": 25.0, "hi": 35.0},
    "inr": {"itemid": 51237, "unit": "ratio", "lo": 0.8, "hi": 1.2},
    "fibrinogen": {"itemid": 51214, "unit": "mg/dL", "lo": 200.0, "hi": 400.0},
    "d_dimer": {"itemid": 51196, "unit": "ng/mL FEU", "lo": 0.0, "hi": 500.0},
    "reticulocyte": {"itemid": 51283, "unit": "%", "lo": 0.5, "hi": 2.0},
    "haptoglobin": {"itemid": 50935, "unit": "mg/dL", "lo": 30.0, "hi": 200.0},
    "ldh": {"itemid": 50954, "unit": "U/L", "lo": 100.0, "hi": 250.0},
    "bilirubin_total": {"itemid": 50885, "unit": "mg/dL", "lo": 0.1, "hi": 1.2},
    "bilirubin_direct": {"itemid": 50883, "unit": "mg/dL", "lo": 0.0, "hi": 0.3},
    "b12": {"itemid": 51010, "unit": "pg/mL", "lo": 200.0, "hi": 900.0},
    "folate": {"itemid": 50925, "unit": "ng/mL", "lo": 3.0, "hi": 20.0},
    "transferrin_sat": {"itemid": 51746, "unit": "%", "lo": 20.0, "hi": 50.0},
    "creatinine": {"itemid": 50912, "unit": "mg/dL", "lo": 0.5, "hi": 1.3},
    "bun": {"itemid": 51006, "unit": "mg/dL", "lo": 7.0, "hi": 20.0},
    "potassium": {"itemid": 50971, "unit": "mEq/L", "lo": 3.5, "hi": 5.0},
    "sodium": {"itemid": 50983, "unit": "mEq/L", "lo": 136.0, "hi": 145.0},
    "chloride": {"itemid": 50902, "unit": "mEq/L", "lo": 98.0, "hi": 106.0},
    "bicarbonate": {"itemid": 50882, "unit": "mEq/L", "lo": 22.0, "hi": 26.0},
    "alt": {"itemid": 50861, "unit": "IU/L", "lo": 7.0, "hi": 56.0},
    "ast": {"itemid": 50878, "unit": "IU/L", "lo": 10.0, "hi": 40.0},
    "albumin": {"itemid": 50862, "unit": "g/dL", "lo": 3.5, "hi": 5.5},
    "troponin_t": {"itemid": 50931, "unit": "ng/mL", "lo": 0.0, "hi": 0.01},
    "lactate": {"itemid": 50813, "unit": "mmol/L", "lo": 0.5, "hi": 2.0},
    "ph": {"itemid": 50820, "unit": "", "lo": 7.35, "hi": 7.45},
    "pco2": {"itemid": 50818, "unit": "mmHg", "lo": 35.0, "hi": 45.0},
    "po2": {"itemid": 50817, "unit": "mmHg", "lo": 80.0, "hi": 100.0},
    "spo2_lab": {"itemid": 50821, "unit": "%", "lo": 95.0, "hi": 100.0},
    "crp": {"itemid": 50889, "unit": "mg/L", "lo": 0.0, "hi": 10.0},
    "glucose": {"itemid": 50893, "unit": "mg/dL", "lo": 70.0, "hi": 100.0},
}

_MIMIC_VITAL_FEATURES = {
    "heart_rate": {"itemid": 220045, "unit": "bpm", "lo": 60.0, "hi": 100.0},
    "systolic_bp": {"itemid": 220050, "unit": "mmHg", "lo": 90.0, "hi": 140.0},
    "diastolic_bp": {"itemid": 220051, "unit": "mmHg", "lo": 60.0, "hi": 90.0},
    "mean_bp": {"itemid": 220052, "unit": "mmHg", "lo": 70.0, "hi": 105.0},
    "resp_rate": {"itemid": 220210, "unit": "/min", "lo": 12.0, "hi": 20.0},
    "spo2": {"itemid": 220277, "unit": "%", "lo": 95.0, "hi": 100.0},
    "temperature": {"itemid": 223761, "unit": "F", "lo": 97.0, "hi": 99.5},
}

_MIMIC_ACTION_BUNDLES = {
    0: {"name": "no_intervention", "source": "derived", "itemids": [], "drugs": []},
    1: {"name": "rbc_transfusion", "source": "prescriptions+chartevents", "itemids": [220997, 226267], "drugs": ["packed red blood", "prbc", "red blood cell"]},
    2: {"name": "platelet_transfusion", "source": "prescriptions+chartevents", "itemids": [225075, 225076], "drugs": ["platelet pheresis", "platelet"]},
    3: {"name": "ffp_cryo", "source": "prescriptions+chartevents", "itemids": [220989, 225771, 224929], "drugs": ["fresh frozen plasma", "ffp", "cryoprecipitate"]},
    4: {"name": "iv_iron", "source": "prescriptions", "drugs": ["iron sucrose", "ferric carboxymaltose", "iron dextran", "ferumoxytol", "iron (iv)"]},
    5: {"name": "esa", "source": "prescriptions", "drugs": ["epoetin", "epogen", "procrit", "aranesp", "darbepoetin", "erythropoietin"]},
    6: {"name": "fluid_resuscitation", "source": "prescriptions+chartevents", "itemids": [225158, 225159, 226391, 226392, 220862, 220986, 223258], "drugs": ["sodium chloride 0.9%", "lactated ringer", "normal saline", "d5w", "dextrose 5%"]},
    7: {"name": "electrolyte_correction", "source": "prescriptions", "drugs": ["potassium chloride", "magnesium sulfate", "calcium gluconate", "sodium phosphate", "potassium phosphate", "calcium chloride", "sodium bicarbonate"]},
    8: {"name": "anticoag_hold", "source": "prescriptions+chartevents", "itemids": [225152], "drugs": ["warfarin", "coumadin", "rivaroxaban", "apixaban", "dabigatran", "enoxaparin", "lovenox", "heparin"]},
    9: {"name": "vasopressor", "source": "prescriptions", "drugs": ["norepinephrine", "levophed", "vasopressin", "phenylephrine", "neosynephrine", "dobutamine", "dopamine", "epinephrine"]},
    10: {"name": "antibiotic", "source": "prescriptions", "drugs": ["vancomycin", "meropenem", "piperacillin", "cefepime", "ceftriaxone", "cefazolin", "ampicillin", "metronidazole", "azithromycin", "levofloxacin", "ciprofloxacin", "zosyn"]},
    11: {"name": "insulin", "source": "prescriptions", "drugs": ["insulin regular", "insulin glargine", "insulin lispro", "insulin aspart", "insulin humalog", "insulin novolog", "humulin", "novolin"]},
    12: {"name": "diuretic", "source": "prescriptions", "drugs": ["furosemide", "lasix", "bumetanide", "mannitol", "spironolactone", "acetazolamide"]},
    13: {"name": "steroid", "source": "prescriptions", "drugs": ["prednisone", "methylprednisolone", "hydrocortisone", "dexamethasone", "solumedrol", "prednisolone"]},
    14: {"name": "sedation_analgesia", "source": "prescriptions", "drugs": ["propofol", "midazolam", "versed", "fentanyl", "dexmedetomidine", "precedex", "hydromorphone", "dilaudid", "morphine", "ketamine"]},
    15: {"name": "cardiac_rxn", "source": "prescriptions", "drugs": ["amiodarone", "metoprolol", "lopressor", "diltiazem", "cardizem", "nitroglycerin", "nitroprusside", "dobutamine"]},
}

_MIMIC_SAFETY_CONSTRAINTS = [
    {"id": "S1", "rule": "No ESA with active malignancy", "icd_prefix": ["C00", "C97"]},
    {"id": "S2", "rule": "No anticoag hold + surgery within 48h"},
    {"id": "S3", "rule": "Platelet transfusion only if Plt < 50K", "action": 2, "lab": "platelets", "threshold": 50.0, "direction": "below"},
    {"id": "S4", "rule": "FFP only if INR > 2.0", "action": 3, "lab": "inr", "threshold": 2.0, "direction": "above"},
    {"id": "S5", "rule": "No vasopressor if MAP >= 65 mmHg", "action": 9, "lab": "mean_bp", "threshold": 65.0, "direction": "above"},
    {"id": "S6", "rule": "No insulin if glucose < 70 mg/dL", "action": 11, "lab": "glucose", "threshold": 70.0, "direction": "below"},
    {"id": "S7", "rule": "No diuretic if creatinine > 4.0 + hypotension", "action": 12},
]

_MIMIC_REWARD_WEIGHTS = {
    "balanced": {"w1": 10.0, "w2": 1.0, "w3": 0.5, "w4": 5.0},
    "conservative": {"w1": 15.0, "w2": 0.5, "w3": 0.3, "w4": 8.0},
    "lab_focused": {"w1": 8.0, "w2": 0.5, "w3": 1.0, "w4": 3.0},
}

PRECEDENCE = [1, 2, 3, 9, 5, 4, 10, 11, 15, 8, 13, 14, 12, 7, 6, 0]

# ==========================================================
#  Config loading + public names
# ==========================================================

def _load_raw_config() -> Dict[str, Any]:
    path = os.environ.get("RL_CONFIG")
    if path:
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = {}
    overrides = {}
    for key in ("lab_features", "vital_features", "action_bundles", "safety_constraints",
                "reward_weights", "precedence", "n_actions", "cohort_mode"):
        env_val = os.environ.get(f"RL_CFG_{key.upper()}")
        if env_val is not None:
            try:
                overrides[key] = json.loads(env_val)
            except json.JSONDecodeError:
                overrides[key] = env_val
    if overrides:
        raw = {**raw, **overrides}
    return raw

_RAW_CONFIG: Dict[str, Any] = _load_raw_config()

def get(key: str, default=None) -> Any:
    return _RAW_CONFIG.get(key, default)

# Public names — consumers import these directly.
# When $RL_CONFIG is set, values come from there; otherwise MIMIC defaults.

def lab_features() -> Dict[str, dict]:
    cfg = get("lab_features")
    return cfg if cfg else dict(_MIMIC_LAB_FEATURES)

def vital_features() -> Dict[str, dict]:
    cfg = get("vital_features")
    return cfg if cfg else dict(_MIMIC_VITAL_FEATURES)

def action_bundles() -> Dict[int, dict]:
    cfg = get("action_bundles")
    if cfg:
        return {int(k): v for k, v in cfg.items()}
    return dict(_MIMIC_ACTION_BUNDLES)

def safety_constraints() -> List[dict]:
    cfg = get("safety_constraints")
    return cfg if cfg else list(_MIMIC_SAFETY_CONSTRAINTS)

def reward_weights() -> Dict[str, dict]:
    cfg = get("reward_weights")
    return cfg if cfg else dict(_MIMIC_REWARD_WEIGHTS)

def n_actions() -> int:
    cfg_n = get("n_actions")
    if cfg_n is not None:
        return int(cfg_n)
    return len(action_bundles())

# Backward-compat module-level names (evaluated at import time)
# These stay as-is when no RL_CONFIG is set, so old consumer code works.
# New code should use the () functions for dynamic reloading.

# ponytail: module-level aliases for backward compat
LAB_FEATURES = lab_features()
VITAL_FEATURES = vital_features()
ACTION_BUNDLES = action_bundles()
SAFETY_CONSTRAINTS = safety_constraints()
REWARD_WEIGHTS = reward_weights()
N_ACTIONS = n_actions()

COHORT_MODE = get("cohort_mode", "all_hosp")

TRANSFERRIN_SAT = {"numerator": "serum_iron", "denominator": "tibc", "lo": 20.0, "hi": 50.0}

# ==========================================================
#  Auto-detection helpers — data-driven
# ==========================================================

def auto_feature_cols(df: pl.DataFrame, exclude: Optional[set] = None) -> List[str]:
    """Discover feature columns from a DataFrame: all numeric cols not in meta set."""
    exc = set(AUTO_META_COLS)
    if exclude:
        exc |= exclude
    # z-scored cols already identified
    z_cols = [c for c in df.columns if c.endswith("_z")]
    if z_cols:
        return z_cols
    # missing mask cols
    missing_cols = [c for c in df.columns if c.endswith("_missing")]
    if missing_cols:
        base = set(c.replace("_missing", "") for c in missing_cols)
        return sorted(base)
    # numeric columns that aren't meta
    numeric = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64) and c not in exc]
    return numeric

def auto_action_ids(df: pl.DataFrame) -> List[int]:
    """Discover action IDs from a DataFrame's action_id column."""
    if "action_id" in df.columns:
        vals = df["action_id"].unique().to_list()
        return sorted(vals)
    return [0]

def auto_feature_names_from_dict(features: dict) -> List[str]:
    return list(features.keys())
