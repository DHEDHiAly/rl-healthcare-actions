# Data Pipeline

## Flow
1. Load MIMIC parquet files
2. Extract features (labs, vitals, demographics, trends)
3. Build trajectories (per-admission sequences)
4. Save to `data/dataset_v1/`

See `src/pipeline/` for implementation.
