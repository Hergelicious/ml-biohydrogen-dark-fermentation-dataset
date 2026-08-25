"""
03_statistics.py — variance decomposition, learning curve, meta-regression.

Estimates how much of the variance in harmonised yield sits between rather than
within source studies, tests whether the grouped result is a sample-size
artefact, and fits the mixed-effects meta-regression that the energy balance in
04_energy_balance.py draws on.

Sixteen moderator terms are estimated without correction for multiplicity, so
every coefficient here is an exploratory moderator signal rather than
confirmatory evidence.

Outputs
    results/03_variance_components.csv   Table S14
    results/03_learning_curve.csv        Table S15
    results/03_metaregression.csv        Table S16
    results/03_pooled_yields.csv         Table S17
    results/03_temperature_effect.csv
    results/03_statistics.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (CATEGORICAL_FEATURES, FEATURES, GROUP_COL, NUMERIC_FEATURES,
                    SEED, TARGET, T_MESO, T_THERMO, get_logger, load_modelling_set,
                    summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("03_statistics")

LEARNING_SIZES = [10, 20, 30, 40, 50, 60, 65]
LEARNING_REPEATS = 15
HELD_OUT_STUDIES = 16


# ─────────────────────────────────────────────────────────────────────────────
# Variance decomposition
# ─────────────────────────────────────────────────────────────────────────────
def direct_icc(df: pd.DataFrame, col: str) -> dict:
    """
    Direct decomposition of the raw target into within- and between-study parts.

    within  = mean squared deviation of each record from its own study mean
    between = variance of the study means
    ICC     = between / (between + within)

    This is the estimator quoted in the main text. It is deliberately not the ANOVA
    ICC(1), which applies an unbiased n0 correction for unbalanced group sizes and
    returns a substantially lower value on a compilation where 47 of 82 studies
    contribute a single observation. The mixed-model estimate in mixed_icc() is
    reported alongside it precisely because the two frameworks differ.
    """
    blocks = [g[col].dropna().to_numpy(dtype=float) for _, g in df.groupby(GROUP_COL)]
    blocks = [b for b in blocks if b.size > 0]
    n_total = int(sum(b.size for b in blocks))
    means = np.array([b.mean() for b in blocks])
    within = float(np.sum([np.sum((b - b.mean()) ** 2) for b in blocks])) / n_total
    between = float(means.var(ddof=0))
    icc = between / (between + within) if (between + within) > 0 else np.nan
    return {"within_study": within, "between_study": between, "icc": icc}


def mixed_icc(df: pd.DataFrame) -> dict:
    """ICC from a random-intercept model on log(1 + y)."""
    work = df.dropna(subset=[TARGET]).copy()
    work["y"] = np.log1p(work[TARGET])
    fit = smf.mixedlm("y ~ 1", work, groups=work[GROUP_COL]).fit(reml=True)
    var_b = float(fit.cov_re.iloc[0, 0])
    var_w = float(fit.scale)
    return {"within_study": var_w, "between_study": var_b,
            "icc": var_b / (var_b + var_w)}


# ─────────────────────────────────────────────────────────────────────────────
# Learning curve
# ─────────────────────────────────────────────────────────────────────────────
def make_rf() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
    ])
    return Pipeline([("pre", pre),
                     ("est", RandomForestRegressor(random_state=SEED, n_jobs=-1))])


def learning_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Hold out a fixed set of studies, then grow the training set study by study.
    Reports the median over repeats at each size, and a slope fitted to all
    per-repeat values (not to the medians).
    """
    rng = np.random.default_rng(SEED)
    studies = df[GROUP_COL].unique()
    held = rng.choice(studies, size=HELD_OUT_STUDIES, replace=False)
    pool = np.setdiff1d(studies, held)
    test = df[df[GROUP_COL].isin(held)]
    log.info("learning curve: %d studies held out (%d observations), %d available for training",
             len(held), len(test), len(pool))

    per_repeat, rows = [], []
    for size in LEARNING_SIZES:
        if size > len(pool):
            continue
        scores = []
        for rep in range(LEARNING_REPEATS):
            r = np.random.default_rng(1000 * size + rep)
            chosen = r.choice(pool, size=size, replace=False)
            train = df[df[GROUP_COL].isin(chosen)]
            model = make_rf().fit(train[FEATURES], train[TARGET])
            s = r2_score(test[TARGET], model.predict(test[FEATURES]))
            scores.append(s)
            per_repeat.append((size, s))
        rows.append({"training_studies": size, "median_grouped_r2": float(np.median(scores)),
                     "lo": float(np.percentile(scores, 25)),
                     "hi": float(np.percentile(scores, 75)), "n_repeats": len(scores)})
        log.info("  %2d training studies -> median R2 = %+.3f", size, np.median(scores))

    x = np.array([p[0] for p in per_repeat], dtype=float)
    y = np.array([p[1] for p in per_repeat], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])

    boots = []
    rb = np.random.default_rng(SEED)
    for _ in range(2000):
        idx = rb.integers(0, len(x), len(x))
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    log.info("slope = %+.4f per additional study, bootstrap 95%% CI [%+.4f, %+.4f] "
             "(fitted to all %d per-repeat values)", slope, ci[0], ci[1], len(x))
    return pd.DataFrame(rows), {"slope": slope, "ci_lo": ci[0], "ci_hi": ci[1],
                                "n_points": len(x)}


# ─────────────────────────────────────────────────────────────────────────────
# Meta-regression
# ─────────────────────────────────────────────────────────────────────────────
def meta_regression(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    """
    Random-intercept mixed model of log(1 + yield) on substrate class, operating
    mode, temperature and pH, fitted by REML with source study as the grouping
    factor. Complete-case: records missing any moderator are dropped.
    """
    work = df.dropna(subset=[TARGET, "temperature_C", "pH"]).copy()
    work["mode"] = work["reactor_mode"].str.split(" / ").str[-1]
    work = work.dropna(subset=["substrate_class", "mode"])
    work["y"] = np.log1p(work[TARGET])

    n, k = len(work), work[GROUP_COL].nunique()
    log.info("%s meta-regression: n = %d records, %d studies (complete case)", label, n, k)

    fit = smf.mixedlm(
        "y ~ C(substrate_class, Treatment(reference='food and kitchen waste')) "
        "+ C(mode, Treatment(reference='batch')) + temperature_C + pH",
        work, groups=work[GROUP_COL]).fit(reml=True)

    rows = []
    for term in fit.params.index:
        rows.append({"subset": label, "term": term,
                     "coefficient": float(fit.params[term]),
                     "std_error": float(fit.bse.get(term, np.nan)),
                     "ci_lo": float(fit.conf_int().loc[term, 0]) if term in fit.conf_int().index else np.nan,
                     "ci_hi": float(fit.conf_int().loc[term, 1]) if term in fit.conf_int().index else np.nan,
                     "p_value": float(fit.pvalues.get(term, np.nan))})
    rows.append({"subset": label, "term": "study random-effect variance",
                 "coefficient": float(fit.cov_re.iloc[0, 0]), "std_error": np.nan,
                 "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan})

    beta = float(fit.params["temperature_C"])
    se = float(fit.bse["temperature_C"])
    p = float(fit.pvalues["temperature_C"])
    y0 = float(work[TARGET].median())
    dT = T_THERMO - T_MESO

    # Back-transform on log(1 + y): a coefficient on the log scale must be
    # evaluated at a stated baseline yield, not read off linearly.
    def back(b: float) -> float:
        return (1.0 + y0) * (np.exp(b * dT) - 1.0)

    info = {"subset": label, "n_records": n, "n_studies": k,
            "beta_per_C": beta, "se": se, "p_value": p,
            "baseline_median_yield": y0,
            "delta_y_point": back(beta),
            "delta_y_lo": back(beta - 1.96 * se),
            "delta_y_hi": back(beta + 1.96 * se)}
    log.info("  temperature beta = %+.5f per C (SE %.5f, p = %.4f); "
             "back-transformed %g -> %g dm3 H2 g-1 at baseline %.4f",
             beta, se, p, T_MESO, info["delta_y_point"], y0)
    return pd.DataFrame(rows), info


def pooled_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Random-effects pooled yield per substrate class, fitted within class."""
    rows = []
    for cls, block in df.groupby("substrate_class"):
        n, k = len(block), block[GROUP_COL].nunique()
        entry = {"substrate_class": cls, "rows": n, "studies": k}
        if k < 2:
            entry.update({"pooled_yield": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                          "note": "not pooled (single study)"})
        else:
            try:
                f = smf.mixedlm("y ~ 1", block.assign(y=np.log1p(block[TARGET])),
                                groups=block[GROUP_COL]).fit(reml=True)
                m = float(f.params["Intercept"]); s = float(f.bse["Intercept"])
                entry.update({"pooled_yield": np.expm1(m),
                              "ci_lo": np.expm1(m - 1.96 * s),
                              "ci_hi": np.expm1(m + 1.96 * s), "note": ""})
            except Exception as exc:                      # boundary convergence
                entry.update({"pooled_yield": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                              "note": f"did not converge: {type(exc).__name__}"})
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def main() -> None:
    df = load_modelling_set()
    log.info("modelling set: %d observations nested within %d studies",
             len(df), df[GROUP_COL].nunique())

    # ── variance components (Table S14) ─────────────────────────────────────
    direct = direct_icc(df, TARGET)
    mixed = mixed_icc(df)
    log.info("ICC: direct decomposition %.3f, mixed model on log1p %.3f",
             direct["icc"], mixed["icc"])
    write_table(pd.DataFrame([
        {"method": "direct decomposition (raw yield)", **direct},
        {"method": "random-intercept mixed model (log1p yield)", **mixed},
    ]), "03_variance_components.csv")

    # ── learning curve (Table S15) ──────────────────────────────────────────
    lc, slope = learning_curve(df)
    write_table(lc, "03_learning_curve.csv")

    # ── meta-regression, both target bases (Table S16) ──────────────────────
    mixed_tbl, mixed_info = meta_regression(df, "mixed basis")

    native = df[df["Unit Group Recomputed"].eq("A") &
                df["Original Unit"].astype(str).str.lower().str.contains("vs")]
    if len(native) < 30:
        log.warning("native-VS subset has only %d records — check the unit-string filter",
                    len(native))
    native_tbl, native_info = meta_regression(native, "native volatile solids")

    write_table(pd.concat([mixed_tbl, native_tbl], ignore_index=True), "03_metaregression.csv")
    write_table(pd.DataFrame([mixed_info, native_info, {"subset": "learning curve slope",
                                                        **slope}]),
                "03_temperature_effect.csv")

    # ── pooled yields (Table S17) ───────────────────────────────────────────
    write_table(pooled_yields(df), "03_pooled_yields.csv")
    log.info("done")


if __name__ == "__main__":
    main()
