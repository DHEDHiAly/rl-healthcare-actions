# Hyperparameter Tuning

## Selected Values
| Param | Value | Rationale |
|-------|-------|-----------|
| gamma | 0.95 | Healthcare actions matter for 4-12h |
| tau | 0.7 | IQL expectile |
| CQL alpha | 0.5 | Action differentiation |
| lr | 3e-4 | Standard for Adam |
| batch | 2048 | GPU memory limit |
