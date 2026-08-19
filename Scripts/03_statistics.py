#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_statistics.py

Statistical analyses that account for the non-independence of observations
within source studies.

Analyses
--------

1. Variance decomposition
   Decomposes harmonised hydrogen yield into within-study and between-study
   components using a direct decomposition, and estimates a random-intercept
   mixed-effects model on log(1 + yield).

2. Learning curve
   Evaluates grouped test R2 as the number of training studies increases.

3. Mixed-effects meta-regression
   Estimates the effects of substrate class, operating mode, temperature and
   pH while accounting for study-level clustering through a random intercept.

4. Random-effects pooled yields
   Estimates pooled hydrogen yield for each substrate class using a
   study-level random intercept.

Outputs
-------

results/03_variance_components.csv
    Within-study variance, between-study variance and ICC.

results/03_learning_curve.csv
    Grouped test R2 across different numbers of training studies.

results/03_metaregression.csv
    Mixed-effects meta-regression coefficients, standard errors, p-values
    and 95% confidence intervals.

results/03_temperature_effect.csv
    Model-implied yield difference between mesophilic and thermophilic
    reference temperatures.

results/03_pooled_yields.csv
    Random-effects pooled yield estimates by substrate class.

Repository structure
--------------------

    repo/
    ├── data/
    ├── src/
    │   ├── common.py
    │   ├── 01_audit_dataset.py
    │   ├── 02_validation_ladder.py
    │   ├── 03_statistics.py
    │   └── 04_energy_balance.py
    └── results/

Run from the repository with:

    python src/03_statistics.py

The script contains no user-specific absolute paths and is therefore
portable across systems.
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# This script is stored in:
#
#     repo/src/03_statistics.py
#
# Therefore:
#
#     SRC_DIR = repo/src
#     ROOT    = repo

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ============================================================
# 3. VERIFY PROJECT STRUCTURE
# ============================================================

print("=" * 70)
print("STATISTICAL ANALYSES")
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
    N_LC_REPEATS,
    RESULTS,
    SEED,
    TARGET,
    design_matrix,
    load_modelling_data,
    variance_components,
)


# ============================================================
# 5. ANALYSIS SETTINGS
# ============================================================

warnings.filterwarnings(
    "ignore"
)

REFERENCE_SUBSTRATE = (
    "Food/kitchen waste"
)

MIN_STUDIES_FOR_POOLING = 3

LC_HOLDOUT_STUDIES = 16

LC_GRID = [
    10,
    20,
    30,
    40,
    50,
    60,
    65,
]

T_MESO = 37.0
T_THERMO = 55.0


# ============================================================
# 6. BASELINE YIELD
# ============================================================

def baseline_yield(d):
    """
    Return the median harmonised yield of the modelling dataset.

    The modelling-set median is used as the reference yield when translating
    the temperature coefficient from the log1p scale back to the original
    hydrogen-yield scale.

    This avoids making the temperature-effect estimate dependent on the
    arbitrary choice of regression reference category.
    """

    return float(
        d[TARGET].median()
    )


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

def rf():
    """
    Construct the Random Forest pipeline.

    Categorical variables are one-hot encoded and numerical variables are
    median-imputed. The preprocessing is fitted within each model fit.
    """

    return make_pipeline(
        ColumnTransformer(
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
        ),
        RandomForestRegressor(
            n_estimators=400,
            random_state=SEED,
            n_jobs=-1,
        ),
    )


# ============================================================
# 8. RANDOM-INTERCEPT MIXED MODEL
# ============================================================

def mixed_intercept(
    frame,
    group_col="ref",
    response="ly",
):
    """
    Fit a random-intercept mixed-effects model.

    The grouping variable is passed as an array after resetting the dataframe
    index to avoid index-alignment problems in statsmodels.
    """

    frame = (
        frame
        .reset_index(drop=True)
    )

    return smf.mixedlm(
        f"{response} ~ 1",
        frame,
        groups=frame[group_col].values,
    ).fit(
        reml=True
    )


# ============================================================
# 9. LOAD MODELLING DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODELLING DATA")
print("=" * 70)

d = load_modelling_data()

d = d.assign(
    ly=np.log1p(
        d[TARGET]
    )
)

X, y, groups = design_matrix(
    d
)

print(
    "\nObservations:",
    len(d),
)

print(
    "Source studies:",
    d.ref.nunique(),
)

print(
    "Target median:",
    f"{d[TARGET].median():.4f}",
)

print(
    "Target mean:",
    f"{d[TARGET].mean():.4f}",
)


# ============================================================
# 10. VARIANCE COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("1. VARIANCE COMPONENTS")
print("=" * 70)

within, between, icc = variance_components(
    d
)

m0 = mixed_intercept(
    d
)

tau2 = float(
    m0.cov_re.iloc[0, 0]
)

sigma2 = float(
    m0.scale
)

icc_mixed = (
    tau2
    /
    (tau2 + sigma2)
)

vc = pd.DataFrame(
    [
        {
            "method":
                "direct decomposition (raw yield)",
            "within":
                within,
            "between":
                between,
            "icc":
                icc,
        },
        {
            "method":
                "random-intercept mixed model (log1p yield)",
            "within":
                sigma2,
            "between":
                tau2,
            "icc":
                icc_mixed,
        },
    ]
)

variance_path = (
    RESULTS_DIR
    /
    "03_variance_components.csv"
)

vc.to_csv(
    variance_path,
    index=False,
)

print(
    vc.round(5)
    .to_string(
        index=False
    )
)

print(
    "\nBetween-study variance accounts for "
    "%.1f%% of the raw-yield variance."
    % (
        100 * icc
    )
)

print(
    "This is the component that study-level holdout "
    "validation is designed to remove."
)


# ============================================================
# 11. LEARNING CURVE
# ============================================================

print("\n" + "=" * 70)
print("2. LEARNING CURVE")
print("=" * 70)

studies = np.array(
    sorted(
        d.ref.unique()
    )
)

rng = np.random.default_rng(
    0
)

lc_rows = []

for k in LC_GRID:

    for rep in range(
        N_LC_REPEATS
    ):

        perm = rng.permutation(
            studies
        )

        held = set(
            perm[
                :LC_HOLDOUT_STUDIES
            ]
        )

        pool = list(
            perm[
                LC_HOLDOUT_STUDIES:
                LC_HOLDOUT_STUDIES + k
            ]
        )

        train_mask = np.isin(
            groups,
            pool,
        )

        test_mask = np.isin(
            groups,
            list(held),
        )

        if (
            train_mask.sum() < 20
            or test_mask.sum() < 8
        ):
            continue

        model = rf().fit(
            X[train_mask],
            y[train_mask],
        )

        prediction = model.predict(
            X[test_mask]
        )

        lc_rows.append(
            {
                "train_studies":
                    k,
                "repeat":
                    rep,
                "n_train":
                    int(
                        train_mask.sum()
                    ),
                "n_test":
                    int(
                        test_mask.sum()
                    ),
                "r2":
                    r2_score(
                        y[test_mask],
                        prediction,
                    ),
            }
        )

lc = pd.DataFrame(
    lc_rows
)

learning_curve_path = (
    RESULTS_DIR
    /
    "03_learning_curve.csv"
)

lc.to_csv(
    learning_curve_path,
    index=False,
)

print(
    "\nGrouped holdout of %d studies."
    % LC_HOLDOUT_STUDIES
)

print(
    lc.groupby(
        "train_studies"
    )
    .r2
    .agg(
        [
            "median",
            "size",
        ]
    )
    .round(3)
    .to_string()
)


# ============================================================
# 12. MIXED-EFFECTS META-REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("3. MIXED-EFFECTS META-REGRESSION")
print("=" * 70)

md = (
    d
    .dropna(
        subset=[
            "Tn",
            "pHn",
            "sub_cat",
            "mode",
            "ref",
        ]
    )
    .reset_index(
        drop=True
    )
)

formula = (
    'ly ~ C(sub_cat, '
    'Treatment(reference="%s")) + '
    'C(mode) + Tn + pHn'
    % REFERENCE_SUBSTRATE
)

mm = smf.mixedlm(
    formula,
    md,
    groups=md["ref"].values,
).fit(
    reml=True
)

res = pd.DataFrame(
    {
        "term":
            mm.params.index,
        "coef":
            mm.params.values,
        "se":
            mm.bse.values,
        "p":
            mm.pvalues.values,
    }
)

res["term"] = (
    res["term"]
    .str.replace(
        'C(sub_cat, Treatment(reference="%s"))[T.'
        % REFERENCE_SUBSTRATE,
        "substrate: ",
        regex=False,
    )
    .str.replace(
        "C(mode)[T.",
        "mode: ",
        regex=False,
    )
    .str.replace(
        "]",
        "",
        regex=False,
    )
)

res["ci_lo"] = (
    res["coef"]
    -
    1.96 * res["se"]
)

res["ci_hi"] = (
    res["coef"]
    +
    1.96 * res["se"]
)

meta_path = (
    RESULTS_DIR
    /
    "03_metaregression.csv"
)

res.to_csv(
    meta_path,
    index=False,
)

print(
    "\nFormula:"
)

print(
    formula
)

print(
    "\nObservations:",
    len(md),
)

print(
    "\nMeta-regression results:"
)

print(
    res.round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. TEMPERATURE EFFECT
# ============================================================

temperature_row = (
    res[
        res.term == "Tn"
    ]
)

if temperature_row.empty:
    raise RuntimeError(
        "Temperature coefficient (Tn) was not found "
        "in the meta-regression results."
    )

t = temperature_row.iloc[0]

y0 = baseline_yield(
    d
)

span = (
    T_THERMO
    -
    T_MESO
)


def implied_difference(
    beta
):
    """
    Back-transform a temperature coefficient from the log1p scale.

    The coefficient is evaluated over the mesophilic-to-thermophilic
    temperature interval at the median modelling-set yield.
    """

    return float(
        np.expm1(
            np.log1p(y0)
            +
            beta * span
        )
        -
        y0
    )


dy = implied_difference(
    t.coef
)

dy_lo = implied_difference(
    t.ci_lo
)

dy_hi = implied_difference(
    t.ci_hi
)

temperature_effect = pd.DataFrame(
    [
        {
            "baseline_yield":
                y0,
            "span_C":
                span,
            "coef":
                t.coef,
            "implied_difference":
                dy,
            "ci_lo":
                dy_lo,
            "ci_hi":
                dy_hi,
        }
    ]
)

temperature_path = (
    RESULTS_DIR
    /
    "03_temperature_effect.csv"
)

temperature_effect.to_csv(
    temperature_path,
    index=False,
)

print(
    "\nTemperature coefficient:"
)

print(
    "  %+.5f per °C on the log1p scale"
    % t.coef
)

print(
    "  95%% CI: %.5f to %.5f"
    % (
        t.ci_lo,
        t.ci_hi,
    )
)

print(
    "  p = %.4f"
    % t.p
)

print(
    "\nModel-implied yield difference:"
)

print(
    "  %.0f -> %.0f °C at median yield %.4f:"
    % (
        T_MESO,
        T_THERMO,
        y0,
    )
)

print(
    "  %+.3f dm3 H2 g-1 VS"
    " (95%% CI %+.3f to %+.3f)"
    % (
        dy,
        dy_lo,
        dy_hi,
    )
)


# ============================================================
# 14. pH EFFECT
# ============================================================

pH_row = (
    res[
        res.term == "pHn"
    ]
)

if not pH_row.empty:

    ph = pH_row.iloc[0]

    print(
        "\nEstimated pH effect:"
    )

    print(
        "  %+.4f per pH unit"
        % ph.coef
    )

    print(
        "  p = %.4f"
        % ph.p
    )


# ============================================================
# 15. RANDOM-EFFECTS POOLED YIELDS
# ============================================================

print("\n" + "=" * 70)
print("4. POOLED YIELD BY SUBSTRATE CLASS")
print("=" * 70)

pooled_rows = []

for category, group_data in d.groupby(
    "sub_cat"
):

    n_studies = (
        group_data.ref.nunique()
    )

    if (
        n_studies
        <
        MIN_STUDIES_FOR_POOLING
    ):

        pooled_rows.append(
            {
                "substrate":
                    category,
                "rows":
                    len(group_data),
                "studies":
                    n_studies,
                "pooled":
                    np.nan,
                "ci_lo":
                    np.nan,
                "ci_hi":
                    np.nan,
            }
        )

        continue

    model = mixed_intercept(
        group_data
    )

    beta = float(
        model.params.iloc[0]
    )

    se = float(
        model.bse.iloc[0]
    )

    pooled_rows.append(
        {
            "substrate":
                category,
            "rows":
                len(group_data),
            "studies":
                n_studies,
            "pooled":
                np.expm1(
                    beta
                ),
            "ci_lo":
                np.expm1(
                    beta
                    -
                    1.96 * se
                ),
            "ci_hi":
                np.expm1(
                    beta
                    +
                    1.96 * se
                ),
        }
    )

pooled = (
    pd.DataFrame(
        pooled_rows
    )
    .sort_values(
        "pooled",
        ascending=False,
        na_position="last",
    )
)

pooled_path = (
    RESULTS_DIR
    /
    "03_pooled_yields.csv"
)

pooled.to_csv(
    pooled_path,
    index=False,
)

print(
    pooled.round(4)
    .to_string(
        index=False
    )
)

print(
    "\nCategories supported by fewer than %d studies "
    "are not pooled."
    % MIN_STUDIES_FOR_POOLING
)


# ============================================================
# 16. STUDY CONCENTRATION
# ============================================================

study_counts = (
    d.ref.value_counts()
)

n_single_observation_studies = int(
    (
        study_counts == 1
    ).sum()
)

six_largest_rows = int(
    study_counts
    .head(6)
    .sum()
)

six_largest_fraction = (
    100
    *
    six_largest_rows
    /
    len(d)
)

print(
    "\n" + "=" * 70
)

print(
    "STUDY CONCENTRATION"
)

print(
    "=" * 70
)

print(
    "\nNumber of studies:",
    len(study_counts),
)

print(
    "Studies contributing one observation:",
    n_single_observation_studies,
)

print(
    "Rows contributed by six largest studies:",
    six_largest_rows,
)

print(
    "Share of all observations:",
    f"{six_largest_fraction:.1f}%",
)


# ============================================================
# 17. FINAL OUTPUT FILES
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
    variance_path,
)

print(
    "2.",
    learning_curve_path,
)

print(
    "3.",
    meta_path,
)

print(
    "4.",
    temperature_path,
)

print(
    "5.",
    pooled_path,
)

print(
    "\nStatistical analyses complete."
)

print(
    "=" * 70
)
