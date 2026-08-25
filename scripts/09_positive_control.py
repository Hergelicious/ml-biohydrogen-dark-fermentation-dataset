"""
09_positive_control.py — can this design detect a transferable signal when one exists?

The central objection to a negative grouped-validation result on a small compiled
dataset is that it may show only that the dataset is uninformative, not that the
validation protocol matters. This script answers that objection directly.

Design
    The real structure is preserved exactly: the same 82 source studies, the same
    number of records per study, the same predictor values and therefore the same
    class imbalance and the same collinearity between predictors and study
    identity. Only the response is replaced, by

        y_ij = f(X_ij) + u_j + e_ij

    where f is a KNOWN linear function of the one-hot encoded predictors held
    fixed across the whole sweep, u_j ~ N(0, sigma_u^2) is a study offset, and
    e_ij ~ N(0, sigma_e^2) is within-study noise. Var(f(X)) and sigma_e^2 are
    fixed; sigma_u^2 is swept so that the empirical intraclass correlation of the
    simulated response runs from near zero to about 0.9.

    The identical grouped protocol from 02_validation_ladder.py is then applied at
    every level.

What the output establishes
    1. At low ICC the pipeline recovers a transferable signal at n = 224 across 82
       studies, with positive grouped R2. Underpowered-not-broken is excluded:
       the design has enough data to detect transferable structure when it is
       present.
    2. The ICC at which grouped R2 crosses zero, and at which it falls below a
       mean predictor, is a design target for the field rather than a property of
       this particular compilation.
    3. An oracle arm, in which the fitted model is replaced by f(X) itself,
       reports the ceiling any model could reach on each simulated response.

Interpretation caveat
    Because the categorical predictors are close to deterministic functions of
    study identity in the real compilation, f(X) itself carries between-study
    variance. The decomposition columns report how much of the between-study
    variance at each level comes from the signal and how much from the injected
    study offset, so the two are never conflated.

Outputs
    results/09_positive_control.csv        the sweep
    results/09_positive_control_summary.csv crossover points and the real-data comparison
    results/09_positive_control.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from common import (FEATURES, GROUP_COL, SEED, TARGET, TEST_SIZE, get_logger,
                    load_modelling_set, summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("09_positive_control")

SIM_RESAMPLES = 20          # fewer than the 40 used on real data; the sweep has 10 levels
SIGNAL_VAR = 1.0            # Var(f(X)), fixed
RESIDUAL_VAR = 1.0          # sigma_e^2, fixed
STUDY_VAR_GRID = [0.0, 0.1, 0.25, 0.5, 0.9, 1.5, 2.5, 4.0, 6.5, 10.0, 16.0, 26.0]


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """One-hot the categoricals and standardise the numerics, exactly as the models see them."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from common import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC_FEATURES),
    ])
    return pre.fit_transform(df[FEATURES])


def empirical_icc(y: np.ndarray, groups: np.ndarray) -> float:
    """Direct decomposition, identical to the estimator used in 03_statistics.py."""
    blocks = [y[groups == g] for g in pd.unique(groups)]
    n_total = int(sum(b.size for b in blocks))
    means = np.array([b.mean() for b in blocks])
    within = float(np.sum([np.sum((b - b.mean()) ** 2) for b in blocks])) / n_total
    between = float(means.var(ddof=0))
    return float(between / (between + within)) if (between + within) > 0 else np.nan


def between_share(values: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of the variance of `values` that lies between studies."""
    blocks = [values[groups == g] for g in pd.unique(groups)]
    grand = values.mean()
    n_i = np.array([b.size for b in blocks], float)
    means = np.array([b.mean() for b in blocks])
    ss_b = float(np.sum(n_i * (means - grand) ** 2))
    ss_t = float(np.sum((values - grand) ** 2))
    return ss_b / ss_t if ss_t > 0 else np.nan


def evaluate(df: pd.DataFrame, y: np.ndarray, signal: np.ndarray) -> dict:
    """Row-wise, grouped, mean-predictor and oracle R2 over the resampling distribution."""
    from importlib import import_module
    mod = import_module("02_validation_ladder")

    groups = df[GROUP_COL].to_numpy()
    X = df[FEATURES]
    out: dict[str, list] = {"row_wise": [], "study_grouped": [], "mean_predictor": [], "oracle": []}

    for seed in range(SIM_RESAMPLES):
        for protocol in ("row_wise", "study_grouped"):
            sp = (GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                  if protocol == "study_grouped"
                  else ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed))
            g = groups if protocol == "study_grouped" else None
            tr, te = next(sp.split(X, y, groups=g))
            model = mod.make_model("RandomForest").fit(X.iloc[tr], y[tr])
            out[protocol].append(r2_score(y[te], model.predict(X.iloc[te])))
            if protocol == "study_grouped":
                out["mean_predictor"].append(r2_score(y[te], np.full(len(te), y[tr].mean())))
                # oracle: the true signal, offset by the training-fold mean discrepancy
                oracle = signal[te] + (y[tr].mean() - signal[tr].mean())
                out["oracle"].append(r2_score(y[te], oracle))
    return {k: summarise(v)["median"] for k, v in out.items()}


def main() -> None:
    df = load_modelling_set().reset_index(drop=True)
    groups = df[GROUP_COL].to_numpy()
    n, k = len(df), df[GROUP_COL].nunique()
    log.info("real structure preserved: %d observations nested within %d studies", n, k)
    log.info("real target ICC = %.3f", empirical_icc(df[TARGET].to_numpy(float), groups))

    # ── fixed, known transferable signal ────────────────────────────────────
    rng = np.random.default_rng(SEED)
    Z = design_matrix(df)
    beta = rng.normal(0.0, 1.0, Z.shape[1])
    raw = Z @ beta
    signal = (raw - raw.mean()) / raw.std() * np.sqrt(SIGNAL_VAR)
    log.info("signal f(X): %d design columns; Var(f(X)) fixed at %.2f", Z.shape[1], SIGNAL_VAR)
    log.info("share of Var(f(X)) lying between studies = %.3f  "
             "(predictors are themselves largely study markers)",
             between_share(signal, groups))

    study_ids = pd.unique(groups)
    # Draw the study offsets and the residuals ONCE at unit variance and rescale at
    # each level. Redrawing per level would confound the effect of raising the study
    # variance with sampling noise in the realised offsets across only 82 studies,
    # which makes the sweep non-monotonic and uninterpretable.
    base_rng = np.random.default_rng(SEED + 1)
    u0 = dict(zip(study_ids, base_rng.normal(0.0, 1.0, len(study_ids))))
    u_unit = np.array([u0[g] for g in groups])
    u_unit = (u_unit - u_unit.mean()) / u_unit.std()
    e_unit = base_rng.normal(0.0, 1.0, n)
    e_unit = (e_unit - e_unit.mean()) / e_unit.std()

    signal_between = between_share(signal, groups)
    rows = []
    for sv in STUDY_VAR_GRID:
        y = signal + np.sqrt(sv) * u_unit + np.sqrt(RESIDUAL_VAR) * e_unit

        icc = empirical_icc(y, groups)
        scores = evaluate(df, y, signal)
        row = {
            "study_variance": sv,
            "empirical_icc": round(icc, 3),
            "between_share_of_response": round(between_share(y, groups), 3),
            "between_share_of_signal": round(signal_between, 3),
            **{f"{key}_r2": round(v, 3) for key, v in scores.items()},
            "leakage_gap": round(scores["row_wise"] - scores["study_grouped"], 3),
            "margin_over_mean": round(scores["study_grouped"] - scores["mean_predictor"], 3),
        }
        rows.append(row)
        log.info("sigma_u^2=%5.2f  ICC=%.3f | grouped R2=%+.3f  row-wise=%+.3f  "
                 "mean=%+.3f  oracle=%+.3f  gap=%+.3f",
                 sv, icc, scores["study_grouped"], scores["row_wise"],
                 scores["mean_predictor"], scores["oracle"],
                 scores["row_wise"] - scores["study_grouped"])

    sweep = pd.DataFrame(rows)
    write_table(sweep, "09_positive_control.csv")

    # ── crossover points by linear interpolation on ICC ─────────────────────
    def crossover(col: str, threshold_col: str | float) -> float:
        x = sweep["empirical_icc"].to_numpy(float)
        d = (sweep[col].to_numpy(float) -
             (sweep[threshold_col].to_numpy(float) if isinstance(threshold_col, str)
              else float(threshold_col)))
        for i in range(len(d) - 1):
            if d[i] > 0 >= d[i + 1]:
                return float(x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]))
        return float("nan")

    icc_zero = crossover("study_grouped_r2", 0.0)
    icc_mean = crossover("study_grouped_r2", "mean_predictor_r2")
    real_icc = empirical_icc(df[TARGET].to_numpy(float), groups)

    summary = pd.DataFrame([
        {"quantity": "grouped R2 at the lowest ICC tested",
         "value": float(sweep["study_grouped_r2"].iloc[0]),
         "note": "positive value shows the design detects transferable signal at this n"},
        {"quantity": "ICC at which grouped R2 crosses zero", "value": icc_zero,
         "note": "above this, a model cannot beat predicting the overall mean"},
        {"quantity": "ICC at which grouped R2 falls below a mean predictor", "value": icc_mean,
         "note": "design target for compiled datasets in this field"},
        {"quantity": "ICC of the real harmonised target", "value": real_icc,
         "note": "where this literature actually sits"},
        {"quantity": "share of Var(f(X)) between studies",
         "value": between_share(signal, groups),
         "note": "predictor-study collinearity carried over from the real compilation"},
    ])
    write_table(summary, "09_positive_control_summary.csv")

    log.info("-" * 72)
    log.info("grouped R2 at lowest ICC tested        = %+.3f",
             sweep["study_grouped_r2"].iloc[0])
    log.info("grouped R2 crosses zero at ICC         = %.3f", icc_zero)
    log.info("grouped R2 falls below mean pred. at   = %.3f", icc_mean)
    log.info("real harmonised target sits at ICC     = %.3f", real_icc)
    if np.isfinite(icc_mean) and real_icc > icc_mean:
        log.info("CONCLUSION: the real compilation lies above the threshold at which a "
                 "transferable signal of this strength becomes undetectable under grouped "
                 "validation. The design is not underpowered; the between-study variance is "
                 "too large relative to the transferable signal.")
    log.info("done")


if __name__ == "__main__":
    main()
