import polars as pl
from pathlib import Path
from src.config import BIN_HOURS, MIN_BINS, LOCF_MAX_GAP_HOURS, LAB_FEATURES, VITAL_FEATURES, REWARD_WEIGHTS


def load_cohort(path: str) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def load_labs(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_actions(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def _all_feature_cols() -> list:
    return list(LAB_FEATURES.keys()) + list(VITAL_FEATURES.keys())


def _data_feature_cols(df: pl.DataFrame) -> list:
    """Auto-detect feature columns from data when config has none."""
    from src.config import auto_feature_cols
    cfg = list(LAB_FEATURES.keys()) + list(VITAL_FEATURES.keys())
    existing = [c for c in cfg if c in df.columns]
    if existing:
        return existing
    return auto_feature_cols(df)


def compute_transferrin_sat(df: pl.DataFrame) -> pl.DataFrame:
    if "serum_iron" in df.columns and "tibc" in df.columns:
        df = df.with_columns(
            ((pl.col("serum_iron") / pl.col("tibc")) * 100).alias("transferrin_sat")
        )
    return df


def locf(df: pl.DataFrame, max_gap_bins: int = LOCF_MAX_GAP_HOURS // BIN_HOURS) -> pl.DataFrame:
    feat_cols = _data_feature_cols(df)
    sorted_df = df.sort(["hadm_id", "bin_idx"])
    for c in feat_cols:
        measured_bin = pl.when(pl.col(c).is_not_null()).then(pl.col("bin_idx")).otherwise(None)
        sorted_df = sorted_df.with_columns(
            measured_bin.forward_fill().over("hadm_id").alias(f"_src_bin_{c}")
        )
    sorted_df = sorted_df.with_columns(
        [pl.col(c).forward_fill().over("hadm_id") for c in feat_cols]
    )
    for c in feat_cols:
        dist = pl.col("bin_idx") - pl.col(f"_src_bin_{c}")
        orig_null = sorted_df[f"_src_bin_{c}"] != sorted_df["bin_idx"]
        sorted_df = sorted_df.with_columns(
            pl.when(
                orig_null & (dist > max_gap_bins)
            )
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
        )
    drop_cols = [f"_src_bin_{c}" for c in feat_cols]
    return sorted_df.drop(drop_cols)


def compute_missingness_mask(df: pl.DataFrame) -> pl.DataFrame:
    feat_cols = _data_feature_cols(df)
    return df.with_columns(
        [pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_missing") for c in feat_cols]
    )


def compute_lab_deviation(df: pl.DataFrame) -> pl.DataFrame:
    all_features = {**LAB_FEATURES, **VITAL_FEATURES}
    if not all_features:
        return df.with_columns(pl.lit(0.0).alias("lab_deviation"))
    dev_cols = []
    for name, cfg in all_features.items():
        if name not in df.columns:
            continue
        mid = (cfg["lo"] + cfg["hi"]) / 2.0
        half = (cfg["hi"] - cfg["lo"]) / 2.0
        if half == 0:
            continue
        dev_cols.append(((pl.col(name) - mid).abs() / half).fill_null(0.0).alias(f"_dev_{name}"))

    if not dev_cols:
        return df.with_columns(pl.lit(0.0).alias("lab_deviation"))

    df = df.with_columns(dev_cols)
    dev_names = [c for c in df.columns if c.startswith("_dev_")]
    n_feats = max(len(dev_names), 1)
    mean_dev = sum(pl.col(c).clip(0, 10) for c in dev_names) / n_feats
    df = df.with_columns(mean_dev.alias("lab_deviation"))
    return df.drop(dev_names)


def assign_default_action(actions: pl.DataFrame, bins: pl.DataFrame) -> pl.DataFrame:
    all_bins = bins.select("hadm_id", "bin_idx").unique()
    filled = all_bins.join(actions, on=["hadm_id", "bin_idx"], how="left")
    return filled.with_columns(pl.col("action_id").fill_null(0).cast(pl.Int8))


def compute_rewards(
    trajectory: pl.DataFrame,
    cohort: pl.DataFrame,
    profile: str = "balanced",
) -> pl.DataFrame:
    w = REWARD_WEIGHTS[profile]
    cohort_meta = cohort.select("hadm_id", "hospital_expire_flag", "los_days")

    trajectory = trajectory.join(cohort_meta, on="hadm_id", how="left")
    mean_los = pl.col("los_days").mean()
    std_los = pl.col("los_days").std()
    los_norm = (pl.col("los_days") - mean_los) / (std_los + 1e-8)

    has_lab_dev = "lab_deviation" in trajectory.columns
    if has_lab_dev:
        trajectory = trajectory.with_columns(
            (-w["w3"] * pl.col("lab_deviation")).alias("per_bin_reward")
        )
    else:
        trajectory = trajectory.with_columns(pl.lit(0.0).alias("per_bin_reward"))

    max_bin = trajectory.group_by("hadm_id").agg(pl.col("bin_idx").max().alias("max_bin"))
    trajectory = trajectory.join(max_bin, on="hadm_id", how="left")

    is_terminal = pl.col("bin_idx") == pl.col("max_bin")
    survived = pl.col("hospital_expire_flag") == 0

    terminal_survive = w["w1"] - w["w2"] * los_norm
    terminal_die = pl.lit(-w["w4"])

    trajectory = trajectory.with_columns(
        pl.when(is_terminal & survived)
        .then(terminal_survive)
        .when(is_terminal & ~survived)
        .then(terminal_die)
        .otherwise(0.0)
        .alias("terminal_reward")
    )

    trajectory = trajectory.with_columns(
        (pl.col("per_bin_reward") + pl.col("terminal_reward")).alias("reward")
    )

    return trajectory.drop(["per_bin_reward", "terminal_reward", "max_bin", "los_days", "hospital_expire_flag"])


def build_trajectories(
    cohort_path: str,
    labs_path: str,
    actions_path: str,
    output_path: str,
    profile: str = "balanced",
) -> pl.DataFrame:
    cohort = load_cohort(cohort_path)
    labs = load_labs(labs_path)
    actions = load_actions(actions_path)

    feat_cols = list(LAB_FEATURES.keys()) + list(VITAL_FEATURES.keys())
    has_config_feats = any(c in labs.columns for c in feat_cols)

    wide = compute_transferrin_sat(labs)
    wide = locf(wide)
    wide = compute_missingness_mask(wide)
    wide = compute_lab_deviation(wide)

    full_actions = assign_default_action(actions, wide)
    trajectory = wide.join(full_actions, on=["hadm_id", "bin_idx"], how="left")

    trajectory = compute_rewards(trajectory, cohort, profile)

    hadm_counts = trajectory.group_by("hadm_id").agg(pl.len().alias("n_bins"))
    valid = hadm_counts.filter(pl.col("n_bins") >= MIN_BINS)
    trajectory = trajectory.join(valid.select("hadm_id"), on="hadm_id", how="inner")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_parquet(output_path)
    return trajectory
