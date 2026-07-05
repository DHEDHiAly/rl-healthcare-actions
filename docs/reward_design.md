# Reward Function Design

## Formula
- Per-bin penalty: `-0.5 * (avg_lab_deviation / 45)`
- Terminal survive: `+10 - 1*LOS_norm`
- Terminal die: `-5`

## Rationale
Laboratory values normalized and penalized per timestep.
Survival bonus incentivizes keeping patients alive.
