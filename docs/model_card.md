# Model Card: rl-healthcare-actions

## Model Details
- Algorithm: IQL + CQL
- State: 113-dim (labs, vitals, trends, demographics)
- Actions: 16 discrete intervention bundles
- Training: Offline RL on MIMIC-IV

## Intended Use
Clinical decision support for ICU intervention recommendations.

## Limitations
- Retrospective data only
- Requires clinician review before deployment
