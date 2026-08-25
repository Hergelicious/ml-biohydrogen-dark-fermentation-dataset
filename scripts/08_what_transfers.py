"""
08_what_transfers.py — what survives study-level validation?

A negative R2 shows that absolute point prediction fails, not that nothing
transfers. This script puts three weaker questions to the same grouped protocol
and quantifies how far each predictor is itself a study marker.

DEPENDENCE WARNING
    Each of the 82 studies recurs in a mean of ~8 test partitions across the 40
    resamples, and the same rows are reused throughout. Statistics computed over
    study-resample combinations are therefore NOT independent observations, and
    any nominal p-value is anticonservative. They are reported as descriptive
    summaries of the resampling distribution. A study-level collapse (one value
    per study) is reported alongside for comparison.

Outputs
    results/08_rank_transfer.csv        pooled and within-study rank correlation
    results/08_interval_coverage.csv    calibration of nominal 90% intervals
    results/08_learning_curve_slope.csv slope over training-study size
    results/08_predictor_icc.csv        Table S29
    results/08_transfers.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

from common import (CATEGORICAL_FEATURES, FEATURES, GROUP_COL, NUMERIC_FEATURES,
                    N_RESAMPLES, TARGET, TEST_SIZE, get_logger, load_modelling_set,
                    summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("08_transfers")

NOMINAL_COVERAGE = 0.90
MIN_OBS_FOR_RANK = 4


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical variables."""
    table = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def numeric_icc(df: pd.DataFrame, col: str) -> float:
    groups = [g[col].dropna().to_numpy(float) for _, g in df.groupby(GROUP_COL)]
    groups = [g for g in groups if g.size > 0]
    n_i = np.array([g.size for g in groups], float)
    k = len(groups)
    grand = np.concatenate(groups).mean()
    means = np.array([g.mean() for g in groups])
    ms_b = float(np.sum(n_i * (means - grand) ** 2)) / (k - 1)
    ms_w = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups])) / max(int(n_i.sum()) - k, 1)
    n0 = (n_i.sum() - (n_i ** 2).sum() / n_i.sum()) / (k - 1)
    var_b = max((ms_b - ms_w) / n0, 0.0)
    return float(var_b / (var_b + ms_w)) if (var_b + ms_w) > 0 else np.nan


def main() -> None:
    from importlib import import_module
    mod = import_module("02_validation_ladder")

    df = load_modelling_set()
    log.info("modelling set: %d observations nested within %d studies",
             len(df), df[GROUP_COL].nunique())

    pooled_rho, within_rows, coverage, widths = [], [], [], []

    for seed in range(N_RESAMPLES):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
        Xtr, ytr = df[FEATURES].iloc[tr], df[TARGET].iloc[tr]
        Xte, yte = df[FEATURES].iloc[te], df[TARGET].iloc[te].to_numpy(float)

        model = mod.make_model("RandomForest").fit(Xtr, ytr)
        pred = model.predict(Xte)

        if np.std(pred) > 0:
            pooled_rho.append(stats.spearmanr(pred, yte).statistic)

        # prediction intervals from the spread of the individual trees
        pre = model.named_steps["pre"]
        forest = model.named_steps["est"]
        Xte_t = pre.transform(Xte)
        tree_preds = np.stack([t.predict(Xte_t) for t in forest.estimators_])
        lo = np.percentile(tree_preds, 100 * (1 - NOMINAL_COVERAGE) / 2, axis=0)
        hi = np.percentile(tree_preds, 100 * (1 + NOMINAL_COVERAGE) / 2, axis=0)
        coverage.append(float(np.mean((yte >= lo) & (yte <= hi))))
        widths.append(float(np.median(hi - lo)))

        groups_te = df[GROUP_COL].iloc[te].to_numpy()
        for study in pd.unique(groups_te):
            m = groups_te == study
            if m.sum() >= MIN_OBS_FOR_RANK and np.std(pred[m]) > 0 and np.std(yte[m]) > 0:
                within_rows.append({"seed": seed, "study": study, "n_obs": int(m.sum()),
                                    "spearman_rho": stats.spearmanr(pred[m], yte[m]).statistic})

    within = pd.DataFrame(within_rows)
    rank_rows = [
        {"scope": "pooled across the test partition", "unit": "resample",
         "n_units": len(pooled_rho), **summarise(pooled_rho)},
        {"scope": "within a held-out study", "unit": "study-resample combination",
         "n_units": len(within), **summarise(within["spearman_rho"])},
    ]
    if not within.empty:
        # study-level collapse: one value per study, which IS independent
        per_study = within.groupby("study")["spearman_rho"].median()
        w = stats.wilcoxon(per_study, alternative="greater")
        rank_rows.append({"scope": "within-study, collapsed to one value per study",
                          "unit": "study", "n_units": len(per_study),
                          **summarise(per_study),
                          "wilcoxon_p_independent_units": float(w.pvalue)})
        log.info("within-study rank: median rho = %+.2f over %d study-resample "
                 "combinations; collapsed to %d independent studies the median is %+.2f "
                 "(Wilcoxon p = %.3g)",
                 within["spearman_rho"].median(), len(within), len(per_study),
                 per_study.median(), w.pvalue)
    write_table(pd.DataFrame(rank_rows), "08_rank_transfer.csv")

    cov = summarise(coverage)
    write_table(pd.DataFrame([{"nominal_coverage": NOMINAL_COVERAGE,
                               "empirical_coverage_median": cov["median"],
                               "empirical_coverage_lo": cov["pct_lo"],
                               "empirical_coverage_hi": cov["pct_hi"],
                               "median_interval_width": float(np.median(widths)),
                               "verdict": "anticonservative" if cov["median"] < NOMINAL_COVERAGE
                                          else "conservative"}]),
                "08_interval_coverage.csv")
    log.info("nominal %.0f%% intervals achieve %.2f empirical coverage, median width %.3f",
             NOMINAL_COVERAGE * 100, cov["median"], float(np.median(widths)))

    # predictor confounding with study identity (Table S29)
    rows = [{"predictor": c, "statistic": "intraclass correlation",
             "value": round(numeric_icc(df, c), 3)} for c in NUMERIC_FEATURES]
    rows += [{"predictor": c, "statistic": "Cramer's V against study identity",
              "value": round(cramers_v(df[c], df[GROUP_COL]), 3)} for c in CATEGORICAL_FEATURES]
    write_table(pd.DataFrame(rows), "08_predictor_icc.csv")
    for r in rows:
        log.info("  %-18s %-36s %.3f", r["predictor"], r["statistic"], r["value"])

    # learning-curve slope is computed in 03_statistics.py; mirrored here for convenience
    lc = pd.read_csv("../results/03_temperature_effect.csv") if \
        (pd.io.common.file_exists("../results/03_temperature_effect.csv")) else None
    if lc is not None and (lc["subset"] == "learning curve slope").any():
        write_table(lc[lc["subset"] == "learning curve slope"], "08_learning_curve_slope.csv")

    log.info("done")


if __name__ == "__main__":
    main()
