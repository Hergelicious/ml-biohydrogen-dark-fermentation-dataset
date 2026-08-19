#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_validation_ladder.py

Central validation experiment for the dark-fermentation hydrogen-yield
modelling framework.

The data, feature construction, encoding and model hyperparameters are held
fixed while only the test-set construction protocol is varied.

Validation protocols
---------------------

1. augmented
   The modelling dataset is duplicated. Gaussian noise (sigma = 0.05) is
   added to numerical predictors and to the target in the duplicated copy.
   The resulting dataset is randomly split 80/20.

2. rowwise
   The original observations are randomly split 80/20.

3. grouped
   The original observations are split by source study. All observations
   belonging to a given study are assigned entirely to either the training
   or test set.

Each protocol is repeated N_RESAMPLES times.

For the grouped protocol, the following models are evaluated:

    - Random Forest
    - CatBoost
    - Ridge (one-hot)
    - Mean predictor

The validation ladder therefore separates apparent predictive performance
under random/augmented splitting from performance under study-grouped
validation.

Outputs
-------

results/02_ladder_scores.csv
    One row per protocol, model and resampling iteration.

results/02_ladder_summary.csv
    Median R2 and 2.5th-97.5th percentile intervals for each protocol/model.

results/02_error_metrics.csv
    Median grouped RMSE and MAE for each baseline model.

This script is intended to be run from the repository structure:

    repo/
    ├── data/
    ├── src/
    │   ├── common.py
    │   └── 02_validation_ladder.py
    └── results/

Run with:

    python src/02_validation_ladder.py

The script does not contain hard-coded user-specific paths.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    ShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from catboost import CatBoostRegressor


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# This file is located in:
#
#     repo/src/02_validation_ladder.py
#
# Therefore:
#
#     SRC_DIR  = repo/src
#     ROOT     = repo
#
# This makes the script portable and avoids hard-coded
# computer-specific paths.

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ============================================================
# 3. VERIFY PROJECT STRUCTURE
# ============================================================

print("=" * 70)
print("VALIDATION LADDER")
print("=" * 70)

print("\nRepository:")
print(ROOT)

if not ROOT.exists():
    raise FileNotFoundError(
        f"\nRepository directory was not found:\n{ROOT}"
    )

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"\ndata/ directory was not found:\n{DATA_DIR}"
    )

if not (SRC_DIR / "common.py").exists():
    raise FileNotFoundError(
        f"\ncommon.py was not found at:\n{SRC_DIR / 'common.py'}"
    )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nProject structure verified.")


# ============================================================
# 4. IMPORT SHARED DEFINITIONS
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from common import (
    CAT_FEATURES,
    NUM_FEATURES,
    N_RESAMPLES,
    NOISE_SD,
    SEED,
    TEST_FRACTION,
    design_matrix,
    fmt,
    load_modelling_data,
)


# ============================================================
# 5. PREPROCESSOR
# ============================================================

def preprocessor():
    """
    Construct the preprocessing pipeline.

    Categorical variables are one-hot encoded.
    Numerical variables are median-imputed.

    The transformer is fitted separately within each model fit,
    preventing information from the test set from entering the
    preprocessing stage.
    """

    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CAT_FEATURES,
            ),
            (
                "num",
                SimpleImputer(
                    strategy="median"
                ),
                NUM_FEATURES,
            ),
        ]
    )


# ============================================================
# 6. MODELS
# ============================================================

MODELS = {

    "Random Forest": lambda: make_pipeline(
        preprocessor(),
        RandomForestRegressor(
            n_estimators=400,
            random_state=SEED,
            n_jobs=-1,
        ),
    ),

    "CatBoost": lambda: make_pipeline(
        preprocessor(),
        CatBoostRegressor(
            iterations=400,
            verbose=0,
            random_seed=SEED,
            thread_count=4,
            allow_writing_files=False,
        ),
    ),

    "Ridge (one-hot)": lambda: make_pipeline(
        preprocessor(),
        RidgeCV(
            alphas=np.logspace(
                -3,
                3,
                13,
            )
        ),
    ),

    "Mean predictor": lambda: DummyRegressor(
        strategy="mean"
    ),
}


# Models used in the validation ladder itself.
LADDER_MODELS = [
    "Random Forest",
    "CatBoost",
]


# Models used for the grouped validation baseline comparison.
BASELINE_MODELS = [
    "Random Forest",
    "CatBoost",
    "Ridge (one-hot)",
    "Mean predictor",
]


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

def augment(X, y, seed):
    """
    Create the noise-augmented dataset.

    The original observations are retained and duplicated.

    Gaussian noise with standard deviation NOISE_SD is added to:

        - numerical predictors
        - target values

    Categorical predictors are duplicated unchanged because Gaussian
    perturbation of categorical labels is not meaningful.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : numpy.ndarray
        Target vector.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_augmented : pandas.DataFrame
        Original plus noise-perturbed observations.

    y_augmented : numpy.ndarray
        Original plus noise-perturbed targets.
    """

    rng = np.random.default_rng(seed)

    Xs = X.copy()

    for column in NUM_FEATURES:
        Xs[column] = (
            Xs[column]
            + NOISE_SD
            * rng.normal(
                size=len(Xs)
            )
        )

    ys = (
        y
        + NOISE_SD
        * rng.normal(
            size=len(y)
        )
    )

    X_augmented = pd.concat(
        [
            X,
            Xs,
        ],
        ignore_index=True,
    )

    y_augmented = np.concatenate(
        [
            y,
            ys,
        ]
    )

    return (
        X_augmented,
        y_augmented,
    )


# ============================================================
# 8. TEST-SET INDICES
# ============================================================

def get_split_indices(
    protocol,
    X,
    y,
    groups,
    seed,
):
    """
    Generate train/test indices for a validation protocol.

    Parameters
    ----------
    protocol : str
        One of:
            "rowwise"
            "grouped"

    X : pandas.DataFrame
        Feature matrix.

    y : numpy.ndarray
        Target vector.

    groups : numpy.ndarray
        Source-study identifiers.

    seed : int
        Random seed.

    Returns
    -------
    train_idx : numpy.ndarray
    test_idx : numpy.ndarray
    """

    if protocol == "grouped":

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_FRACTION,
            random_state=seed,
        )

        return next(
            splitter.split(
                X,
                y,
                groups,
            )
        )

    splitter = ShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=seed,
    )

    return next(
        splitter.split(X)
    )


# ============================================================
# 9. MODEL SCORING
# ============================================================

def score(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Fit a model and calculate R2, RMSE and MAE.

    Returns
    -------
    tuple
        (R2, RMSE, MAE)
    """

    model = MODELS[
        model_name
    ]()

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    rmse = float(
        mean_squared_error(
            y_test,
            prediction,
        ) ** 0.5
    )

    mae = float(
        mean_absolute_error(
            y_test,
            prediction,
        )
    )

    return (
        r2,
        rmse,
        mae,
    )


# ============================================================
# 10. LOAD MODELLING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODELLING DATA")
print("=" * 70)

d = load_modelling_data()

X, y, groups = design_matrix(d)

print(
    "\nObservations:",
    len(d),
)

print(
    "Source studies:",
    d.ref.nunique(),
)

print(
    "Target mean:",
    f"{y.mean():.4f}",
)

print(
    "Target SD:",
    f"{y.std(ddof=1):.4f}",
    "dm3 H2 g-1",
)


# ============================================================
# 11. RUN VALIDATION LADDER
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION LADDER")
print("=" * 70)

rows = []


for protocol in [
    "augmented",
    "rowwise",
    "grouped",
]:

    if protocol == "grouped":
        models = BASELINE_MODELS
    else:
        models = LADDER_MODELS

    print(
        f"\nProtocol: {protocol}"
    )

    for model_name in models:

        for seed in range(
            N_RESAMPLES
        ):

            # ------------------------------------------------
            # Augmented protocol
            # ------------------------------------------------

            if protocol == "augmented":

                X_augmented, y_augmented = augment(
                    X,
                    y,
                    seed,
                )

                train_idx, test_idx = train_test_split(
                    np.arange(
                        len(X_augmented)
                    ),
                    test_size=TEST_FRACTION,
                    random_state=seed,
                )

                r2, rmse, mae = score(
                    model_name,
                    X_augmented.iloc[train_idx],
                    y_augmented[train_idx],
                    X_augmented.iloc[test_idx],
                    y_augmented[test_idx],
                )

                n_test = len(
                    test_idx
                )

                n_test_studies = np.nan

            # ------------------------------------------------
            # Original rowwise / grouped protocols
            # ------------------------------------------------

            else:

                train_idx, test_idx = get_split_indices(
                    protocol,
                    X,
                    y,
                    groups,
                    seed,
                )

                r2, rmse, mae = score(
                    model_name,
                    X.iloc[train_idx],
                    y[train_idx],
                    X.iloc[test_idx],
                    y[test_idx],
                )

                n_test = len(
                    test_idx
                )

                if protocol == "grouped":

                    n_test_studies = len(
                        set(
                            groups[
                                test_idx
                            ]
                        )
                    )

                else:

                    n_test_studies = np.nan

            rows.append(
                {
                    "protocol": protocol,
                    "model": model_name,
                    "resample": seed,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "n_test": n_test,
                    "n_test_studies": n_test_studies,
                }
            )

        # ----------------------------------------------------
        # Print running model summary
        # ----------------------------------------------------

        values = [
            row["r2"]
            for row in rows
            if (
                row["protocol"] == protocol
                and row["model"] == model_name
            )
        ]

        print(
            "  %-11s %-18s R2 %s"
            % (
                protocol,
                model_name,
                fmt(values),
            )
        )


# ============================================================
# 12. SAVE VALIDATION SCORES
# ============================================================

scores = pd.DataFrame(
    rows
)

scores_path = (
    RESULTS_DIR
    / "02_ladder_scores.csv"
)

scores.to_csv(
    scores_path,
    index=False,
)


# ============================================================
# 13. CREATE VALIDATION SUMMARY
# ============================================================

summary = (
    scores
    .groupby(
        [
            "protocol",
            "model",
        ]
    )
    .r2
    .agg(
        median="median",
        lo=lambda s: np.percentile(
            s,
            2.5,
        ),
        hi=lambda s: np.percentile(
            s,
            97.5,
        ),
        n="size",
    )
    .reset_index()
)

summary_path = (
    RESULTS_DIR
    / "02_ladder_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# 14. CREATE GROUPED ERROR METRICS
# ============================================================

err = (
    scores[
        scores.protocol == "grouped"
    ]
    .groupby("model")[
        [
            "rmse",
            "mae",
        ]
    ]
    .median()
    .reset_index()
)

err["target_sd"] = (
    y.std(
        ddof=1
    )
)

error_path = (
    RESULTS_DIR
    / "02_error_metrics.csv"
)

err.to_csv(
    error_path,
    index=False,
)


# ============================================================
# 15. PRINT GROUPED ERROR METRICS
# ============================================================

print(
    "\nGrouped-protocol error metrics "
    "(median over %d resamples):"
    % N_RESAMPLES
)

print(
    err.round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 16. GROUPED TEST PARTITIONS
# ============================================================

grouped = scores[
    scores.protocol == "grouped"
]

print(
    "\nGrouped test partitions: "
    "%.0f rows from %.0f studies on average"
    % (
        grouped.n_test.mean(),
        grouped.n_test_studies.mean(),
    )
)


# ============================================================
# 17. ATTRIBUTION OF R2 INFLATION
# ============================================================

augmented_median = (
    scores[
        scores.protocol == "augmented"
    ]
    .groupby("model")
    .r2
    .median()
)

rowwise_median = (
    scores[
        scores.protocol == "rowwise"
    ]
    .groupby("model")
    .r2
    .median()
)

grouped_median = (
    scores[
        scores.protocol == "grouped"
    ]
    .groupby("model")
    .r2
    .median()
)


print(
    "\nAttribution of the inflation "
    "(median R2):"
)

for model_name in LADDER_MODELS:

    augmentation_effect = (
        augmented_median[model_name]
        - rowwise_median[model_name]
    )

    rowwise_effect = (
        rowwise_median[model_name]
        - grouped_median[model_name]
    )

    grouped_baseline = (
        grouped_median[model_name]
    )

    print(
        "  %-14s augmentation %+0.2f | "
        "row-wise splitting %+0.2f | "
        "grouped baseline %+0.2f"
        % (
            model_name,
            augmentation_effect,
            rowwise_effect,
            grouped_baseline,
        )
    )


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FILES WRITTEN"
)

print(
    "=" * 70
)

print(
    "\n1.",
    scores_path,
)

print(
    "2.",
    summary_path,
)

print(
    "3.",
    error_path,
)

print(
    "\nValidation ladder complete."
)

print(
    "=" * 70
)
