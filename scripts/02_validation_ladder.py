"""
02_validation_ladder.py — the central experiment.

Holds data, features, encoding and hyperparameters fixed and varies only how the
test set is drawn. Produces the validation ladder (Table 1, Tables S12 and S13)
and the complete 2 x 2 factorial of augmentation and partitioning (Table S27).

The 40 resamples form a resampling distribution over partitions of one fixed
dataset. They are not 40 independent validations, and the reported 2.5-97.5
intervals are empirical percentiles of that distribution, not confidence
intervals for a population parameter.

Outputs
    results/02_ladder_scores.csv    per-resample scores, long format
    results/02_ladder_summary.csv   Table 1 / Table S12
    results/02_error_metrics.csv    Table S13
    results/02_validation.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (AUG_NOISE_SD, CATEGORICAL_FEATURES, FEATURES, GROUP_COL,
                    NUMERIC_FEATURES, N_RESAMPLES, SEED, TARGET, TEST_SIZE,
                    get_logger, load_modelling_set, summarise, write_table)

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger("02_validation")


def make_model(kind: str) -> Pipeline:
    """
    One-hot encoding and median imputation are steps inside the Pipeline and are
    therefore fitted on the training fold of each split and applied unchanged to
    the test fold. No scaling or feature selection is used.
    """
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ])
    if kind == "RandomForest":
        est = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    elif kind == "CatBoost":
        try:
            from catboost import CatBoostRegressor
            est = CatBoostRegressor(verbose=0, random_seed=SEED, allow_writing_files=False)
        except ImportError:
            log.warning("catboost not installed — substituting GradientBoostingRegressor")
            from sklearn.ensemble import GradientBoostingRegressor
            est = GradientBoostingRegressor(random_state=SEED)
    elif kind == "Ridge":
        est = Ridge(alpha=1.0, random_state=SEED)
    elif kind == "MeanPredictor":
        est = DummyRegressor(strategy="mean")
    else:
        raise ValueError(f"unknown model: {kind!r}")
    return Pipeline([("pre", pre), ("est", est)])


def augment(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Reproduce the noise-augmentation practice audited in the paper: duplicate the
    dataset and add Gaussian noise (sd 0.05, relative) to the numeric predictors
    and the target. Categorical predictors are copied exactly, so each synthetic
    record is near-identical to its parent in the predictor space.
    """
    synth = df.copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        sd = AUG_NOISE_SD * np.nanstd(df[col].to_numpy(dtype=float))
        synth[col] = synth[col] + rng.normal(0.0, sd, size=len(synth))
    synth[TARGET] = synth[TARGET].clip(lower=0.0)
    synth["_synthetic"] = 1
    original = df.copy()
    original["_synthetic"] = 0
    return pd.concat([original, synth], ignore_index=True)


def splitter(protocol: str, seed: int):
    if protocol == "study_grouped":
        return GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)


def run_protocol(df: pd.DataFrame, model_kind: str, protocol: str,
                 augmented: bool) -> list[dict]:
    """Evaluate one cell of the design over the full resampling distribution."""
    rows = []
    for seed in range(N_RESAMPLES):
        rng = np.random.default_rng(seed)
        work = augment(df, rng) if augmented else df.copy()
        groups = work[GROUP_COL].to_numpy()
        X, y = work[FEATURES], work[TARGET].to_numpy(dtype=float)

        sp = splitter(protocol, seed)
        train_idx, test_idx = next(sp.split(X, y, groups=groups if protocol == "study_grouped" else None))

        model = make_model(model_kind)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        obs = y[test_idx]

        rows.append({
            "model": model_kind, "protocol": protocol,
            "augmented": int(augmented), "seed": seed,
            "r2": r2_score(obs, pred),
            "rmse": float(np.sqrt(mean_squared_error(obs, pred))),
            "mae": mean_absolute_error(obs, pred),
            "n_test": int(len(test_idx)),
            "n_test_studies": int(pd.unique(groups[test_idx]).size),
        })
    return rows


def main() -> None:
    df = load_modelling_set()
    log.info("modelling set: %d observations, %d studies", len(df), df[GROUP_COL].nunique())
    log.info("target SD = %.4f dm3 H2 g-1 substrate", df[TARGET].std())

    records: list[dict] = []

    # ── the three protocols reported as the validation ladder ───────────────
    ladder = [
        ("noise_augmented_random", True, "row_wise"),
        ("row_wise", False, "row_wise"),
        ("study_grouped", False, "study_grouped"),
    ]
    for label, aug, protocol in ladder:
        for kind in ("RandomForest", "CatBoost", "Ridge", "MeanPredictor"):
            if label != "study_grouped" and kind in ("Ridge", "MeanPredictor"):
                continue  # baselines are reported under grouping, where they matter
            out = run_protocol(df, kind, protocol, aug)
            for r in out:
                r["ladder"] = label
            records += out
            s = summarise([r["r2"] for r in out])
            log.info("%-24s %-14s median R2 = %+.3f  [%+.3f, %+.3f]",
                     label, kind, s["median"], s["pct_lo"], s["pct_hi"])

    # ── full 2 x 2 factorial: augmentation x partitioning (Table S27) ───────
    log.info("--- full factorial ---")
    factorial: list[dict] = []
    for aug in (False, True):
        for protocol in ("row_wise", "study_grouped"):
            out = run_protocol(df, "RandomForest", protocol, aug)
            for r in out:
                r["ladder"] = f"factorial_aug{int(aug)}_{protocol}"
            factorial += out
            s = summarise([r["r2"] for r in out])
            log.info("augmentation=%-5s  %-14s median R2 = %+.3f  [%+.3f, %+.3f]",
                     aug, protocol, s["median"], s["pct_lo"], s["pct_hi"])
    records += factorial

    scores = pd.DataFrame(records)
    write_table(scores, "02_ladder_scores.csv")

    summary_rows = []
    for (ladder_name, model_kind), block in scores.groupby(["ladder", "model"], sort=False):
        s = summarise(block["r2"])
        summary_rows.append({"ladder": ladder_name, "model": model_kind, **s})
    write_table(pd.DataFrame(summary_rows), "02_ladder_summary.csv")

    # ── Table S13: grouped-protocol error metrics ───────────────────────────
    grouped = scores[(scores.ladder == "study_grouped")]
    err_rows = []
    target_sd = float(df[TARGET].std())
    for kind, block in grouped.groupby("model", sort=False):
        rmse = summarise(block["rmse"])
        mae = summarise(block["mae"])
        err_rows.append({
            "model": kind,
            "rmse_median": rmse["median"], "rmse_lo": rmse["pct_lo"], "rmse_hi": rmse["pct_hi"],
            "mae_median": mae["median"],
            "target_sd": target_sd,
            "rmse_over_sd": rmse["median"] / target_sd,
            "pct_reduction_vs_sd": 100.0 * (1.0 - rmse["median"] / target_sd),
        })
    write_table(pd.DataFrame(err_rows), "02_error_metrics.csv")

    # DummyRegressor leakage check (Supplementary Note S22): the fitted constant
    # must equal the training-fold mean, not the full-dataset mean.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=0)
    tr, _ = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
    dummy = make_model("MeanPredictor").fit(df[FEATURES].iloc[tr], df[TARGET].iloc[tr])
    log.info("leakage check: dummy constant = %.5f, training-fold mean = %.5f, "
             "full-dataset mean = %.5f",
             float(dummy.named_steps["est"].constant_.ravel()[0]),
             float(df[TARGET].iloc[tr].mean()), float(df[TARGET].mean()))


if __name__ == "__main__":
    main()
