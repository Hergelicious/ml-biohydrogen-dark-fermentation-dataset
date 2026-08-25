"""
05_robustness.py — does the grouped result depend on the choices we made?

Four objections are tested: that the collapse reflects the two algorithms
chosen, the default hyperparameters, the category mapping, or one influential
study. None of them survives.

Outputs
    results/05_robustness_algorithms.csv   Table S18
    results/05_robustness_tuning.csv       Table S19
    results/05_robustness_encoding.csv     Table S20
    results/05_robustness_influence.csv    Table S21
    results/05_robustness.log
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
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (CATEGORICAL_FEATURES, FEATURES, GROUP_COL, NUMERIC_FEATURES,
                    N_RESAMPLES, SEED, TARGET, TEST_SIZE, get_logger,
                    load_modelling_set, summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("05_robustness")

TUNING_SPLITS = 12


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ])


def estimator(kind: str):
    if kind == "RandomForest":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1)
    if kind == "Ridge":
        return Ridge(alpha=1.0, random_state=SEED)
    if kind == "MeanPredictor":
        return DummyRegressor(strategy="mean")
    if kind == "CatBoost":
        try:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(verbose=0, random_seed=SEED, allow_writing_files=False)
        except ImportError:
            log.warning("catboost unavailable — skipping")
            return None
    if kind == "XGBoost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0)
        except ImportError:
            log.warning("xgboost unavailable — skipping")
            return None
    if kind == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)
        except ImportError:
            log.warning("lightgbm unavailable — skipping")
            return None
    raise ValueError(kind)


def grouped_scores(df: pd.DataFrame, kind: str, n: int = N_RESAMPLES) -> list[float]:
    est = estimator(kind)
    if est is None:
        return []
    scores = []
    for seed in range(n):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
        model = Pipeline([("pre", preprocessor()), ("est", estimator(kind))])
        model.fit(df[FEATURES].iloc[tr], df[TARGET].iloc[tr])
        scores.append(r2_score(df[TARGET].iloc[te], model.predict(df[FEATURES].iloc[te])))
    return scores


def main() -> None:
    df = load_modelling_set()
    log.info("modelling set: %d observations nested within %d studies",
             len(df), df[GROUP_COL].nunique())

    # ── 1. algorithms (Table S18) ───────────────────────────────────────────
    rows = []
    for kind in ["RandomForest", "MeanPredictor", "LightGBM", "Ridge", "CatBoost", "XGBoost"]:
        s = grouped_scores(df, kind)
        if not s:
            continue
        rows.append({"model": kind, **summarise(s)})
        log.info("  %-14s median grouped R2 = %+.3f  [%+.3f, %+.3f]",
                 kind, rows[-1]["median"], rows[-1]["pct_lo"], rows[-1]["pct_hi"])
    write_table(pd.DataFrame(rows), "05_robustness_algorithms.csv")

    # ── 2. hyperparameter tuning (Table S19) ────────────────────────────────
    # The inner search runs on the training partition only, with an inner grouped
    # split, so no test study can influence tuning.
    grid = {"est__n_estimators": [200, 500, 1000],
            "est__max_depth": [None, 5, 10, 20],
            "est__max_features": ["sqrt", "log2", 1.0],
            "est__min_samples_leaf": [1, 2, 5]}
    default_scores, tuned_scores = [], []
    for seed in range(TUNING_SPLITS):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
        Xtr, ytr = df[FEATURES].iloc[tr], df[TARGET].iloc[tr]
        Xte, yte = df[FEATURES].iloc[te], df[TARGET].iloc[te]
        gtr = df[GROUP_COL].iloc[tr]

        base = Pipeline([("pre", preprocessor()), ("est", estimator("RandomForest"))])
        base.fit(Xtr, ytr)
        default_scores.append(r2_score(yte, base.predict(Xte)))

        search = RandomizedSearchCV(
            Pipeline([("pre", preprocessor()), ("est", estimator("RandomForest"))]),
            grid, n_iter=10,
            cv=GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=seed),
            random_state=seed, n_jobs=-1, scoring="r2")
        search.fit(Xtr, ytr, groups=gtr)
        tuned_scores.append(r2_score(yte, search.best_estimator_.predict(Xte)))

    write_table(pd.DataFrame([
        {"arm": "Random Forest, default", **summarise(default_scores)},
        {"arm": "Random Forest, tuned", **summarise(tuned_scores)},
    ]), "05_robustness_tuning.csv")
    log.info("  tuning: default %+.3f vs tuned %+.3f over %d grouped splits",
             np.median(default_scores), np.median(tuned_scores), TUNING_SPLITS)

    # ── 3. category encoding (Table S20) ────────────────────────────────────
    rows = []
    for encoding, label in [("raw", "raw literature strings"),
                            ("coarse", "coarse three-class substrate + mode"),
                            ("mapped", "mapping used in this work")]:
        alt = load_modelling_set(encoding=encoding)
        s = grouped_scores(alt, "RandomForest")
        rows.append({"encoding": label,
                     "substrate_levels": int(alt["substrate_class"].nunique()),
                     **summarise(s)})
        log.info("  %-36s levels=%3d  median R2 = %+.3f",
                 label, rows[-1]["substrate_levels"], rows[-1]["median"])
    write_table(pd.DataFrame(rows), "05_robustness_encoding.csv")

    # ── 4. leave-one-study-out influence (Table S21) ────────────────────────
    counts = df[GROUP_COL].value_counts()
    largest = counts.head(6).index.tolist()
    rows = [{"study_removed": "none (full set)", "rows_removed": 0, "n": len(df),
             **summarise(grouped_scores(df, "RandomForest"))}]
    for study in largest:
        sub = df[df[GROUP_COL] != study]
        rows.append({"study_removed": study, "rows_removed": int(counts[study]),
                     "n": len(sub), **summarise(grouped_scores(sub, "RandomForest"))})
        log.info("  removing %-30s (%2d rows) -> median R2 = %+.3f",
                 str(study)[:30], counts[study], rows[-1]["median"])
    write_table(pd.DataFrame(rows), "05_robustness_influence.csv")
    log.info("done")


if __name__ == "__main__":
    main()
