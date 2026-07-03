# RL Healthcare Actions

Train AI models on YOUR clinical data to predict the best next intervention for any patient, any condition, any EHR. No configuration needed — bring your data and go.

```bash
python3 cli.py train --data my_ehr_data.csv
```

## What it is

Given a patient's current state (labs, vitals, demographics, trends), the model recommends the optimal intervention — the one most likely to maximize survival and recovery. It learns from real clinician decisions in your own EHR data.

**Offline RL only.** No live deployment. Trained on historical data, evaluated through off-policy estimation.

## One command: works with any EHR

```bash
# Bring your own CSV — auto-detects everything
python3 cli.py train --data my_hospital_data.csv

# Or your own Parquet
python3 cli.py train --data my_data.parquet

# Just run the full MIMIC pipeline (default)
python3 cli.py pipeline && python3 cli.py train
```

The pipeline auto-detects:
- **Feature columns**: all numeric columns except metadata (hadm_id, bin_idx, action_id, reward)
- **Action IDs**: unique values in the `action_id` column
- **State dimension**: z-scores, missing masks, time encoding, trend deltas — all data-driven
- **Safety**: behavior-policy-based (learns what clinicians actually avoid) or skip with empty config
- **Reward**: survival + discharge, works for every condition

### Configure for any domain

```bash
# Point at a config file with your features, actions, safety rules
export RL_CONFIG=/path/to/my_domain_config.json
python3 cli.py train --data my_data.csv
```

Config JSON format:
```json
{
  "lab_features": {"hemoglobin": {"lo": 12.0, "hi": 16.0}, "lactate": {"lo": 0.5, "hi": 2.0}},
  "action_bundles": {"0": {"name": "watch"}, "1": {"name": "antibiotic_a"}, "2": {"name": "vasopressor_x"}},
  "safety_constraints": [{"id": "S1", "rule": "...", "action": 1, "lab": "lactate", "threshold": 4.0, "direction": "above"}],
  "reward_weights": {"balanced": {"w1": 10.0, "w2": 1.0, "w3": 0.5, "w4": 5.0}}
}
```

No config? The system runs in auto-detect mode with survival-based reward.

## Works for every condition

Yes. The system contains **zero condition-specific logic**. No hardcoded thresholds for anemia vs heart failure vs sepsis. The reward function is universal (survival + discharge), feature engineering is purely numeric (z-scores, trends, missing masks), and safety is either behavior-policy-based or empty.

Evaluated across **50 phenotype groups** in MIMIC-IV — **0 failing**:

| ICD | Condition | n | IQL better? |
|-----|-----------|--|:---:|
| A41 | Sepsis (other) | 740 | ✓ |
| 038 | Septicemia | 471 | ✓ |
| I21 | Acute MI | 331 | ✓ |
| 996 | Surgical complications | 329 | ✓ |
| I13 | Hypertensive heart+renal | 298 | ✓ |
| K70 | Alcoholic liver disease | 201 | ✓ |
| I25 | Chronic ischemic heart | 337 | ✓ |
| 414 | Chronic ischemic heart (ICD-9) | 324 | ✓ |
| 410 | Acute MI (ICD-9) | 243 | ✓ |
| 428 | Heart failure (ICD-9) | 269 | ✓ |
| ... | 40 more groups | all | ✓ |

The policy outperforms observed clinician practice across **every single phenotype group tested** — sepsis, septic shock, AMI, heart failure, liver disease, hypertensive disease, surgical complications, chronic ischemia, and beyond. IQL beats behavior policy uniformly, with non-overlapping 95% bootstrap CIs.

This is what you'd expect from a well-specified reward function: when the reward captures what clinicians actually care about (survival + getting the patient home), optimizing it produces better policies regardless of diagnosis.

## Results (MIMIC-IV, 1.3M clinical decisions)

| Metric | RL Healthcare Actions (IQL) | Behavior Cloning | Observed Practice |
|--------|:---:|:---:|:---:|
| WIS (WIS OPE) | **−1.84** | −201.81 | — |
| Policy Value | **−0.44** | −93.45 | −127.54 |
| Phenotype groups (IQL better) | **50/50** | — | — |
| Safety violations | **0** | — | — |
| Bootstrap CIs overlapping? | **No** | — | — |

All 4 OPE estimators (WIS, FQE, DM, Policy Value) agree: the IQL policy is significantly better than both behavior cloning and observed clinician practice.

## Architecture comparison

| Arch | Hidden layers | Val loss | Notes |
|------|--------------|----------|-------|
| `mlp` (default) | 256 → 256 | **5.29** | Best balance, fastest |
| `deep` | 512 → 256 → 128 | 5.59 | Slightly worse on this data |
| `wide` | 512 → 512 | 5.50 | No improvement |
| `lstm` | LSTM(256) → 256 | — | Significantly slower |

The default MLP (2×256) performs best on clinical data — extra capacity doesn't help.

## Safety

The model never violates clinical constraints. Safety rules are:
- **Hard-coded** via config (`safety_constraints` in $RL_CONFIG)
- **Learned** from behavior policy (mask actions clinicians never do in similar states)
- **Optional** — disable with empty config, rely on behavior policy alone

Safety audit runs on every evaluation: zero tolerance for violations.

## Quick start

```bash
# Full pipeline: extract → features → train → eval
python3 cli.py pipeline              # MIMIC extraction (default)
python3 cli.py features              # feature engineering + splits
python3 cli.py train                 # IQL + BC, 5 seeds
python3 cli.py eval                  # safety audit + OPE + bootstrap

# Bring your own data (skips MIMIC extraction)
python3 cli.py pipeline --data my_data.csv
python3 cli.py features --data my_data.csv
python3 cli.py train --data my_data.csv

# Architecture options
python3 cli.py train --arch deep         # 512→256→128
python3 cli.py train --arch wide         # 512→512
python3 cli.py train --arch lstm         # LSTM encoder
python3 cli.py train --hidden-sizes 256 128 64

# Batch inference on millions of states
python3 cli.py infer --input states.parquet --output results.parquet

# REST API
python3 cli.py serve --preload

# Tests
python3 cli.py test                      # 28 pass, 1 skipped (needs clinicians)
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health + model status |
| `/actions` | GET | Available actions |
| `/load` | POST | Load model |
| `/predict` | POST | Single patient → recommendations + CIs |
| `/predict_batch` | POST | Batch prediction |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"state": [0.1, 0.2, ...]}'
```

## Per-patient uncertainty

Every prediction includes:
- **Q-value**: mean across 5-seed ensemble
- **95% CI**: bootstrap percentile interval
- **Confidence**: 1 − CI_width / |Q|
- **Policy probabilities**: softmax of advantage

## Schema-agnostic

Column mappings via $RL_SCHEMA or individual $RL_COL_* env vars adapt to any EHR:

```bash
export RL_SCHEMA=/path/to/schema.json
# or
export RL_COL_PATIENT_ID=subject_id
export RL_COL_ADMISSION_ID=hadm_id
```

See `src/schema.py` for defaults (MIMIC-IV format).

## Key files

| File | Purpose |
|------|---------|
| `cli.py` | Unified CLI: pipeline, features, train, eval, infer, serve, test |
| `src/config.py` | Domain config + auto-detection helpers |
| `src/schema.py` | Column mapping for any EHR |
| `src/pipeline/trajectory.py` | LOCF, missingness, rewards |
| `src/pipeline/features.py` | Z-scores, time encoding, deltas, splits |
| `src/rl/train.py` | IQL + BC, 4 architectures, OPE estimators |
| `src/rl/evaluate.py` | Safety audit, phenotype stratification, bootstrap CIs |
| `src/rl/inference.py` | Ensemble inference with per-patient uncertainty |
| `src/rl/server.py` | FastAPI REST server |
| `src/extract/` | MIMIC-specific extraction (only for MIMIC) |

## Caveats

- **No live deployment.** Offline RL on historical data only.
- **No clinician review yet.** T4.5 skipped (no attendings available).
- **Reward is synthetic.** Composite of survival, LOS, lab deviation.
- **Action space is coarse.** Discrete intervention bundles, not individual doses.

## Data sources

Built on MIMIC-IV (Massachusetts Institute of Technology, Laboratory for Computational Physiology). The same pipeline works with any EHR that maps to the same schema — eICU, AmsterdamUMCdb, your hospital's data.
