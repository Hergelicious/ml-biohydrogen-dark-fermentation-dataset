#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_robustness.py

STEP 05 — ROBUSTNESS ANALYSIS

Purpose
-------
Assess the robustness of the study-grouped hydrogen-yield prediction
framework using the harmonised modelling dataset.

Analyses
--------
A. Algorithm robustness under repeated study-grouped holdout:
   - Random Forest
   - CatBoost
   - XGBoost
   - Ridge
   - Mean predictor

B. Random Forest hyperparameter robustness:
   - Default Random Forest
   - Tuned Random Forest
   - Hyperparameter tuning performed only within the training partition

C. Alternative substrate/category encodings:
   - Paper mapping
   - Raw literature strings
   - Coarse substrate grouping

D. Leave-one-study-out influence:
   - Full dataset
   - Removal of each of the six largest studies

Important
---------
The modelling target is explicitly set to `y`.

The response variable used for ML is:

    ly = log1p(y)

Study-level grouping is defined by the `ref` column.

Outputs
-------
results/05_robustness_algorithms.csv
results/05_robustness_tuning.csv
results/05_robustness_encoding.csv
results/05_robustness_influence.csv
"""


# ============================================================
# 1. IMPORTS
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore")


# ============================================================
# 2. OPTIONAL ML PACKAGES
# ============================================================

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True

except ImportError:

    HAS_CATBOOST = False

    print(
        "WARNING: CatBoost is not installed. "
        "CatBoost will be skipped."
    )


try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True

except ImportError:

    HAS_XGBOOST = False

    print(
        "WARNING: XGBoost is not installed. "
        "XGBoost will be skipped."
    )


# ============================================================
# 3. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS = PROJECT_DIR / "results"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. SETTINGS
# ============================================================

SEED = 42

TEST_FRACTION = 0.20

N_SPLITS = 40

TARGET_COLUMN = "y"

GROUP_COLUMN = "ref"


# ============================================================
# 5. DATA FILE
# ============================================================

DATA_FILE = (
    DATA_DIR /
    "dataset_modelling_224.csv"
)


# ============================================================
# 6. COLUMN DEFINITIONS
# ============================================================

CAT_FEATURES = [
    "sub_cat",
    "inoc_cat",
    "react_cat",
]

NUM_FEATURES = [
    "pHn",
    "Tn",
]


# ============================================================
# 7. DATA LOADING
# ============================================================

def load_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "\nDataset not found:\n"
            f"{DATA_FILE}\n\n"
            "Please check that "
            "'dataset_modelling_224.csv' is located in "
            "the project's data/ directory."
        )

    print("\nUsing dataset:")
    print(DATA_FILE)

    d = pd.read_csv(
        DATA_FILE
    )

    print(
        "\nDataset shape:",
        d.shape
    )

    return d


# ============================================================
# 8. TARGET VALIDATION
# ============================================================

def validate_target(d):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "TARGET VALIDATION"
    )

    print(
        "=" * 70
    )

    if TARGET_COLUMN not in d.columns:

        raise ValueError(
            "\nTarget column "
            f"'{TARGET_COLUMN}' "
            "was not found.\n\n"
            "Available columns:\n"
            +
            "\n".join(
                map(str, d.columns)
            )
        )

    target = pd.to_numeric(
        d[TARGET_COLUMN],
        errors="coerce"
    )

    print(
        "\nTarget column:",
        TARGET_COLUMN
    )

    print(
        "Non-missing values:",
        target.notna().sum()
    )

    print(
        "Missing values:",
        target.isna().sum()
    )

    print(
        "Minimum:",
        target.min()
    )

    print(
        "Maximum:",
        target.max()
    )

    print(
        "Median:",
        target.median()
    )

    if (target.dropna() < 0).any():

        raise ValueError(
            "\nTarget contains negative values.\n"
            "log1p transformation cannot be applied."
        )

    if target.notna().sum() == 0:

        raise ValueError(
            "\nTarget column contains no usable numeric values."
        )

    return target


# ============================================================
# 9. PREPARE DATA
# ============================================================

def prepare_data(d):

    d = d.copy()

    print(
        "\n" +
        "=" * 70
    )

    print(
        "PREPARING DATA"
    )

    print(
        "=" * 70
    )

    if GROUP_COLUMN not in d.columns:

        raise ValueError(
            "The dataset must contain a "
            f"'{GROUP_COLUMN}' column identifying the study."
        )

    y_raw = validate_target(
        d
    )

    d["y_raw"] = y_raw

    d["ly"] = np.log1p(
        d["y_raw"]
    )

    print(
        "\nTarget transformation:"
    )

    print(
        "Raw target = y"
    )

    print(
        "ML target  = log1p(y)"
    )

    if "sub_cat" not in d.columns:

        if "Substrate" in d.columns:

            d["sub_cat"] = (
                d["Substrate"]
                .astype(str)
                .str.strip()
            )

        else:

            raise ValueError(
                "Neither 'sub_cat' nor "
                "'Substrate' exists."
            )

    if "inoc_cat" not in d.columns:

        if "Microbial Inoculum" in d.columns:

            d["inoc_cat"] = (
                d["Microbial Inoculum"]
                .astype(str)
                .str.strip()
            )

        else:

            d["inoc_cat"] = (
                "Unspecified"
            )

    if "react_cat" not in d.columns:

        if "Reactor / Mode" in d.columns:

            d["react_cat"] = (
                d["Reactor / Mode"]
                .astype(str)
                .str.strip()
            )

        else:

            d["react_cat"] = (
                "Unspecified"
            )

    if "mode" not in d.columns:

        if "Reactor / Mode" in d.columns:

            d["mode"] = (
                d["Reactor / Mode"]
                .astype(str)
                .str.strip()
            )

        else:

            d["mode"] = (
                "Unspecified"
            )

    if "Tn" not in d.columns:

        possible_T = [
            "Temp (°C)",
            "Temperature",
            "temperature",
            "Temperature_C",
            "T",
            "Temp",
        ]

        found = None

        for c in possible_T:

            if c in d.columns:

                found = c
                break

        if found is None:

            raise ValueError(
                "Temperature column not found."
            )

        d["Tn"] = pd.to_numeric(
            d[found],
            errors="coerce"
        )

    if "pHn" not in d.columns:

        possible_pH = [
            "pH",
            "PH",
            "pH_value",
        ]

        found = None

        for c in possible_pH:

            if c in d.columns:

                found = c
                break

        if found is None:

            raise ValueError(
                "pH column not found."
            )

        d["pHn"] = pd.to_numeric(
            d[found],
            errors="coerce"
        )

    needed = [
        GROUP_COLUMN,
        "sub_cat",
        "inoc_cat",
        "react_cat",
        "mode",
        "Tn",
        "pHn",
        "y_raw",
        "ly",
    ]

    temp = d[
        needed
    ].copy()

    before = len(temp)

    temp = temp.dropna(
        subset=[
            "ly",
            GROUP_COLUMN,
        ]
    ).reset_index(
        drop=True
    )

    removed = before - len(temp)

    print(
        "\nRows removed because of "
        "missing target/study:",
        removed
    )

    for c in [
        "sub_cat",
        "inoc_cat",
        "react_cat",
        "mode",
    ]:

        temp[c] = (
            temp[c]
            .fillna("Unspecified")
            .astype(str)
            .str.strip()
        )

    for c in [
        "Tn",
        "pHn",
    ]:

        temp[c] = pd.to_numeric(
            temp[c],
            errors="coerce"
        )

    print(
        "\nPrepared dataset:",
        len(temp),
        "observations"
    )

    print(
        "Number of studies:",
        temp[GROUP_COLUMN].nunique()
    )

    print(
        "\nTarget statistics on raw scale:"
    )

    print(
        temp["y_raw"].describe()
    )

    print(
        "\nTarget statistics on log1p scale:"
    )

    print(
        temp["ly"].describe()
    )

    return temp


# ============================================================
# 10. PREPROCESSOR
# ============================================================

def make_preprocessor(
    cats=CAT_FEATURES,
    nums=NUM_FEATURES,
):

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                list(cats),
            ),

            (
                "num",
                SimpleImputer(
                    strategy="median"
                ),
                list(nums),
            ),
        ]
    )


# ============================================================
# 11. DESIGN MATRIX
# ============================================================

def design_matrix(d):

    X = d[
        CAT_FEATURES +
        NUM_FEATURES
    ].copy()

    y = d[
        "ly"
    ].to_numpy()

    groups = d[
        GROUP_COLUMN
    ].to_numpy()

    for c in CAT_FEATURES:

        X[c] = (
            X[c]
            .fillna("Unspecified")
            .astype(str)
        )

    for c in NUM_FEATURES:

        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

        X[c] = X[c].fillna(
            X[c].median()
        )

    return X, y, groups


# ============================================================
# 12. SUMMARY FUNCTIONS
# ============================================================

def summarise(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    median = np.median(
        values
    )

    lo = np.percentile(
        values,
        2.5
    )

    hi = np.percentile(
        values,
        97.5
    )

    return (
        median,
        lo,
        hi
    )


def fmt(values):

    m, lo, hi = summarise(
        values
    )

    return (
        f"median={m:.3f}, "
        f"95% interval="
        f"[{lo:.3f}, {hi:.3f}]"
    )


# ============================================================
# 13. GROUPED SPLITS
# ============================================================

def grouped_splits(
    X,
    y,
    groups,
    n=N_SPLITS,
):

    splitter = GroupShuffleSplit(
        n_splits=n,
        test_size=TEST_FRACTION,
        random_state=SEED,
    )

    return splitter.split(
        X,
        y,
        groups
    )


# ============================================================
# 14. MODEL DEFINITIONS
# ============================================================

def random_forest():

    return make_pipeline(

        make_preprocessor(),

        RandomForestRegressor(
            n_estimators=400,
            random_state=SEED,
            n_jobs=-1,
        )
    )


def ridge_model():

    return make_pipeline(

        make_preprocessor(),

        RidgeCV(
            alphas=np.logspace(
                -3,
                3,
                13
            )
        )
    )


def catboost_model():

    if not HAS_CATBOOST:

        return None

    return make_pipeline(

        make_preprocessor(),

        CatBoostRegressor(
            iterations=400,
            verbose=0,
            random_seed=SEED,
            thread_count=4,
            allow_writing_files=False,
        )
    )


def xgboost_model():

    if not HAS_XGBOOST:

        return None

    return make_pipeline(

        make_preprocessor(),

        XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            random_state=SEED,
            n_jobs=4,
            verbosity=0,
        )
    )


def mean_model():

    return DummyRegressor(
        strategy="mean"
    )


# ============================================================
# 15. ALGORITHM ROBUSTNESS
# ============================================================

def run_algorithms(
    X,
    y,
    groups,
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "A. ALGORITHM ROBUSTNESS"
    )

    print(
        "=" * 70
    )

    models = {
        "Random Forest":
            random_forest,

        "Ridge":
            ridge_model,

        "Mean predictor":
            mean_model,
    }

    if HAS_CATBOOST:

        models[
            "CatBoost"
        ] = catboost_model

    if HAS_XGBOOST:

        models[
            "XGBoost"
        ] = xgboost_model

    rows = []

    for name, factory in models.items():

        print(
            f"\nRunning {name}..."
        )

        scores = []

        for tr, te in grouped_splits(
            X,
            y,
            groups
        ):

            model = factory()

            model.fit(
                X.iloc[tr],
                y[tr]
            )

            pred = model.predict(
                X.iloc[te]
            )

            score = r2_score(
                y[te],
                pred
            )

            scores.append(
                score
            )

        m, lo, hi = summarise(
            scores
        )

        rows.append(
            {
                "model":
                    name,

                "median_r2":
                    m,

                "ci_lo":
                    lo,

                "ci_hi":
                    hi,

                "n_splits":
                    len(scores),
            }
        )

        print(
            f"{name:20s} "
            f"{fmt(scores)}"
        )

    out = pd.DataFrame(
        rows
    )

    output = (
        RESULTS /
        "05_robustness_algorithms.csv"
    )

    out.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    return out


# ============================================================
# 16. TUNED RANDOM FOREST
# ============================================================

def run_tuning(
    X,
    y,
    groups,
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "B. TUNED RANDOM FOREST"
    )

    print(
        "=" * 70
    )

    grid = {

        "randomforestregressor__n_estimators":
            [200, 400],

        "randomforestregressor__max_depth":
            [None, 6, 12, 20],

        "randomforestregressor__max_features":
            [0.4, 0.6, 0.8, 1.0],

        "randomforestregressor__min_samples_leaf":
            [1, 2, 4, 8],
    }

    tuned_scores = []

    default_scores = []

    outer_splitter = GroupShuffleSplit(
        n_splits=12,
        test_size=TEST_FRACTION,
        random_state=SEED,
    )

    for i, (tr, te) in enumerate(
        outer_splitter.split(
            X,
            y,
            groups
        ),
        1
    ):

        print(
            f"Outer split {i}/12",
            end="\r"
        )

        inner_cv = GroupShuffleSplit(
            n_splits=3,
            test_size=0.25,
            random_state=SEED,
        )

        search = RandomizedSearchCV(
            estimator=random_forest(),

            param_distributions=grid,

            n_iter=10,

            scoring="r2",

            cv=inner_cv,

            random_state=SEED,

            n_jobs=-1,

            refit=True,
        )

        search.fit(
            X.iloc[tr],
            y[tr],
            groups=groups[tr],
        )

        tuned_pred = search.predict(
            X.iloc[te]
        )

        tuned_scores.append(
            r2_score(
                y[te],
                tuned_pred
            )
        )

        default = random_forest()

        default.fit(
            X.iloc[tr],
            y[tr]
        )

        default_pred = default.predict(
            X.iloc[te]
        )

        default_scores.append(
            r2_score(
                y[te],
                default_pred
            )
        )

    print("\n")

    rows = []

    for label, scores in [
        (
            "default",
            default_scores
        ),
        (
            "tuned",
            tuned_scores
        ),
    ]:

        m, lo, hi = summarise(
            scores
        )

        rows.append(
            {
                "arm":
                    label,

                "median_r2":
                    m,

                "ci_lo":
                    lo,

                "ci_hi":
                    hi,

                "n_splits":
                    len(scores),
            }
        )

        print(
            f"{label:10s} "
            f"{fmt(scores)}"
        )

    out = pd.DataFrame(
        rows
    )

    output = (
        RESULTS /
        "05_robustness_tuning.csv"
    )

    out.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    return out


# ============================================================
# 17. ALTERNATIVE ENCODINGS
# ============================================================

def run_encoding(
    d,
    y,
    groups,
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "C. CATEGORY ENCODING ROBUSTNESS"
    )

    print(
        "=" * 70
    )

    COARSE = {

        "Simple sugars/starch":
            "Soluble carbohydrate",

        "Food/kitchen waste":
            "Complex waste",

        "Municipal solid waste":
            "Complex waste",

        "Mixed organic waste":
            "Complex waste",

        "Coffee mucilage":
            "Complex waste",

        "Manure/slurry":
            "Complex waste",

        "Sewage sludge":
            "Complex waste",

        "Industrial wastewater":
            "Complex waste",

        "Other":
            "Complex waste",

        "Lignocellulosic":
            "Recalcitrant",

        "Algal biomass":
            "Recalcitrant",

        "Glycerol/biodiesel":
            "Soluble carbohydrate",
    }

    paper = d[
        CAT_FEATURES +
        NUM_FEATURES
    ].copy()

    raw = pd.DataFrame({

        "sub_cat":
            d["sub_cat"].astype(str),

        "inoc_cat":
            d["inoc_cat"].astype(str),

        "react_cat":
            d["mode"].astype(str),

        "pHn":
            d["pHn"],

        "Tn":
            d["Tn"],
    })

    coarse = pd.DataFrame({

        "sub_cat":
            d["sub_cat"]
            .map(COARSE)
            .fillna("Complex waste"),

        "inoc_cat":
            d["inoc_cat"].astype(str),

        "react_cat":
            d["mode"].astype(str),

        "pHn":
            d["pHn"],

        "Tn":
            d["Tn"],
    })

    variants = {

        "paper mapping":
            paper,

        "raw literature strings":
            raw,

        "coarse 3-class substrate + mode":
            coarse,
    }

    rows = []

    for label, frame in variants.items():

        print(
            f"\nRunning encoding: {label}"
        )

        frame = frame.copy()

        for c in NUM_FEATURES:

            frame[c] = pd.to_numeric(
                frame[c],
                errors="coerce"
            )

            frame[c] = frame[c].fillna(
                frame[c].median()
            )

        for c in CAT_FEATURES:

            frame[c] = (
                frame[c]
                .fillna("Unspecified")
                .astype(str)
            )

        scores = []

        for tr, te in grouped_splits(
            frame,
            y,
            groups
        ):

            model = make_pipeline(

                make_preprocessor(),

                RandomForestRegressor(
                    n_estimators=400,
                    random_state=SEED,
                    n_jobs=-1,
                )
            )

            model.fit(
                frame.iloc[tr],
                y[tr]
            )

            pred = model.predict(
                frame.iloc[te]
            )

            scores.append(
                r2_score(
                    y[te],
                    pred
                )
            )

        m, lo, hi = summarise(
            scores
        )

        rows.append(
            {
                "encoding":
                    label,

                "n_substrate_levels":
                    frame["sub_cat"].nunique(),

                "median_r2":
                    m,

                "ci_lo":
                    lo,

                "ci_hi":
                    hi,

                "n_splits":
                    len(scores),
            }
        )

        print(
            f"{label:35s} "
            f"{fmt(scores)}"
        )

    out = pd.DataFrame(
        rows
    )

    output = (
        RESULTS /
        "05_robustness_encoding.csv"
    )

    out.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    return out


# ============================================================
# 18. LEAVE-ONE-STUDY-OUT INFLUENCE
# ============================================================

def run_influence(
    d
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "D. LEAVE-ONE-STUDY-OUT INFLUENCE"
    )

    print(
        "=" * 70
    )

    rows = []

    X, y, groups = design_matrix(
        d
    )

    base_scores = []

    for tr, te in grouped_splits(
        X,
        y,
        groups
    ):

        model = random_forest()

        model.fit(
            X.iloc[tr],
            y[tr]
        )

        pred = model.predict(
            X.iloc[te]
        )

        base_scores.append(
            r2_score(
                y[te],
                pred
            )
        )

    m, lo, hi = summarise(
        base_scores
    )

    rows.append(
        {
            "study_removed":
                "none (full set)",

            "rows_removed":
                0,

            "n":
                len(d),

            "n_studies":
                d[GROUP_COLUMN].nunique(),

            "median_r2":
                m,

            "ci_lo":
                lo,

            "ci_hi":
                hi,
        }
    )

    print(
        f"{'none (full set)':35s} "
        f"n={len(d):3d} "
        f"{fmt(base_scores)}"
    )

    largest = (
        d[GROUP_COLUMN]
        .value_counts()
        .head(6)
        .index
    )

    print(
        "\nSix largest studies:"
    )

    for study in largest:

        count = (
            d[GROUP_COLUMN]
            == study
        ).sum()

        print(
            f"  {str(study):40s} "
            f"{count} observations"
        )

    for study in largest:

        print(
            f"\nRemoving study: {study}"
        )

        keep = (
            d[GROUP_COLUMN]
            != study
        )

        dk = (
            d.loc[keep]
            .reset_index(drop=True)
        )

        Xk, yk, gk = design_matrix(
            dk
        )

        scores = []

        for tr, te in grouped_splits(
            Xk,
            yk,
            gk
        ):

            model = random_forest()

            model.fit(
                Xk.iloc[tr],
                yk[tr]
            )

            pred = model.predict(
                Xk.iloc[te]
            )

            scores.append(
                r2_score(
                    yk[te],
                    pred
                )
            )

        m, lo, hi = summarise(
            scores
        )

        rows.append(
            {
                "study_removed":
                    study,

                "rows_removed":
                    int(
                        (~keep).sum()
                    ),

                "n":
                    len(dk),

                "n_studies":
                    dk[
                        GROUP_COLUMN
                    ].nunique(),

                "median_r2":
                    m,

                "ci_lo":
                    lo,

                "ci_hi":
                    hi,
            }
        )

        print(
            f"{str(study)[:35]:35s} "
            f"n={len(dk):3d} "
            f"{fmt(scores)}"
        )

    out = pd.DataFrame(
        rows
    )

    output = (
        RESULTS /
        "05_robustness_influence.csv"
    )

    out.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    return out


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

def print_final_summary(
    algorithm_results,
    tuning_results,
    encoding_results,
    influence_results,
):

    print(
        "\n" +
        "=" * 70
    )

    print(
        "FINAL ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nA. Algorithm robustness"
    )

    print(
        algorithm_results.to_string(
            index=False
        )
    )

    print(
        "\nB. Random Forest tuning"
    )

    print(
        tuning_results.to_string(
            index=False
        )
    )

    print(
        "\nC. Encoding robustness"
    )

    print(
        encoding_results.to_string(
            index=False
        )
    )

    print(
        "\nD. Leave-one-study-out influence"
    )

    print(
        influence_results.to_string(
            index=False
        )
    )


# ============================================================
# 20. MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 05 — ROBUSTNESS ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nProject directory:"
    )

    print(
        PROJECT_DIR
    )

    print(
        "\nResults directory:"
    )

    print(
        RESULTS
    )

    print(
        "\nDataset:"
    )

    print(
        DATA_FILE
    )

    print(
        "\nTarget:"
    )

    print(
        TARGET_COLUMN
    )

    print(
        "\nResponse transformation:"
    )

    print(
        "log1p(y)"
    )

    raw = load_data()

    d = prepare_data(
        raw
    )

    X, y, groups = design_matrix(
        d
    )

    algorithm_results = run_algorithms(
        X,
        y,
        groups
    )

    tuning_results = run_tuning(
        X,
        y,
        groups
    )

    encoding_results = run_encoding(
        d,
        y,
        groups
    )

    influence_results = run_influence(
        d
    )

    print_final_summary(
        algorithm_results,
        tuning_results,
        encoding_results,
        influence_results,
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        "STEP 05 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFiles written:"
    )

    print(
        RESULTS /
        "05_robustness_algorithms.csv"
    )

    print(
        RESULTS /
        "05_robustness_tuning.csv"
    )

    print(
        RESULTS /
        "05_robustness_encoding.csv"
    )

    print(
        RESULTS /
        "05_robustness_influence.csv"
    )


# ============================================================
# 21. RUN
# ============================================================

if __name__ == "__main__":

    main()
