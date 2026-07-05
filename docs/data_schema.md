# Data Schema

## Input Format
CSV or Parquet with columns:
- `subject_id`: patient identifier
- `hadm_id`: admission identifier
- `charttime`: timestamp
- Lab values: one column per lab
- Vital signs: one column per vital

See `src/config.py` for lab mappings.
