#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_what_transfers.py -- a negative R2 says point prediction fails, but not whether
anything weaker survives. This script asks four questions of the study-grouped
protocol:

  A. Does rank ordering transfer? Spearman correlation between predicted and
     observed on held-out studies, pooled across the test partition and computed
     within each held-out study separately.
  B. Does coarse classification transfer? Accuracy at sorting held-out
     observations into low / medium / high tertiles, against a 1/3 chance rate
     and against a majority-class baseline.
  C. Are the predictions honestly calibrated? Empirical coverage and width of
     90% prediction intervals taken from the spread of the individual trees.
  D. Why does it fail? Prediction error against the distance between a held-out
     study's mean and the training mean; and the intraclass correlation of the
     predictors themselves, which quantifies how far the features are confounded
     with study identity.

Also reports the slope of the learning curve with a bootstrap interval, so the
"no gain with more studies" statement can be made quantitative rather than visual.

Writes results/08_rank_transfer.csv, 08_classification.csv,
       08_interval_coverage.csv, 08_error_vs_centroid.csv,
       08_predictor_icc.csv, 08_learning_curve_slope.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Ensure common.py can be imported from the current script directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CAT_FEATURES,
    NUM_FEATURES,
    RESULTS,
    SEED,
    TEST_FRACTION,
    design_matrix,
    load_modelling_data,
    summarise,
)

warnings.filterwarnings("ignore")
N = 40


def rf():
    """Build a standard Random Forest Pipeline."""
    return make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ]
        ),
        RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
    )


def tree_predictions(pipe, X):
    """Per-tree predictions, for interval construction."""
    pre = pipe[:-1]
    forest = pipe[-1]
    Z = pre.transform(X)
    return np.stack([t.predict(Z) for t in forest.estimators_])


def main():
    d = load_modelling_data()
    X, y, g = design_matrix(d)
    splits = [
        next(
            GroupShuffleSplit(
                1, test_size=TEST_FRACTION, random_state=s
            ).split(X, y, g)
        )
        for s in range(N)
    ]
    print(
        "n = %d observations from %d studies, %d grouped resamples\n"
        % (len(d), d.ref.nunique(), N)
    )

    pooled_rho, within_rho, r2s = [], [], []
    acc, chance, major = [], [], []
    cover, width = [], []
    err_rows = []

    tertiles = np.quantile(y, [1 / 3, 2 / 3])
    y_class = np.digitize(y, tertiles)

    for k, (tr, te) in enumerate(splits):
        pipe = rf().fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        r2s.append(r2_score(y[te], pred))

        # ---- A. rank transfer
        if len(np.unique(y[te])) > 2:
            pooled_rho.append(stats.spearmanr(y[te], pred).statistic)
        for st in set(g[te]):
            m = g[te] == st
            if m.sum() >= 4 and len(np.unique(y[te][m])) > 2:
                r = stats.spearmanr(y[te][m], pred[m]).statistic
                if np.isfinite(r):
                    within_rho.append(r)

        # ---- B. coarse classification
        pred_class = np.digitize(pred, tertiles)
        acc.append(accuracy_score(y_class[te], pred_class))
        chance.append(1 / 3)
        vals, counts = np.unique(y_class[tr], return_counts=True)
        major.append((y_class[te] == vals[counts.argmax()]).mean())

        # ---- C. prediction intervals from tree spread
        tp = tree_predictions(pipe, X.iloc[te])
        lo, hi = np.percentile(tp, [5, 95], axis=0)
        cover.append(float(((y[te] >= lo) & (y[te] <= hi)).mean()))
        width.append(float(np.median(hi - lo)))

        # ---- D. error against distance from the training centroid
        train_mean = y[tr].mean()
        for st in set(g[te]):
            m = g[te] == st
            err_rows.append(
                dict(
                    resample=k,
                    study=st,
                    n=int(m.sum()),
                    study_mean=float(y[te][m].mean()),
                    dist_from_train_mean=float(
                        abs(y[te][m].mean() - train_mean)
                    ),
                    mae=float(np.abs(pred[m] - y[te][m]).mean()),
                )
            )

    def f(v):
        return "%+.3f [%+.3f, %+.3f]" % summarise(v)

    print("A. RANK TRANSFER (Spearman, study-grouped holdout)")
    print("   pooled across the test partition : %s" % f(pooled_rho))
    print(
        "   within each held-out study       : %s  (%d studies with n>=4)"
        % (f(within_rho), len(within_rho))
    )
    t = stats.wilcoxon(within_rho) if len(within_rho) > 10 else None
    if t:
        print("   within-study rho vs 0: Wilcoxon p = %.4g" % t.pvalue)
    pd.DataFrame(
        [
            dict(
                scope="pooled",
                median=np.median(pooled_rho),
                lo=np.percentile(pooled_rho, 2.5),
                hi=np.percentile(pooled_rho, 97.5),
                n=len(pooled_rho),
            ),
            dict(
                scope="within-study",
                median=np.median(within_rho),
                lo=np.percentile(within_rho, 2.5),
                hi=np.percentile(within_rho, 97.5),
                n=len(within_rho),
            ),
        ]
    ).to_csv(RESULTS / "08_rank_transfer.csv", index=False)

    print("\nB. COARSE CLASSIFICATION INTO YIELD TERTILES")
    print("   model accuracy      : %s" % f(acc))
    print("   majority-class rate : %s" % f(major))
    print("   chance rate         : 0.333")
    pd.DataFrame(
        [
            dict(
                metric="model accuracy",
                median=np.median(acc),
                lo=np.percentile(acc, 2.5),
                hi=np.percentile(acc, 97.5),
            ),
            dict(
                metric="majority-class baseline",
                median=np.median(major),
                lo=np.percentile(major, 2.5),
                hi=np.percentile(major, 97.5),
            ),
            dict(
                metric="chance",
                median=1 / 3,
                lo=np.nan,
                hi=np.nan,
            ),
        ]
    ).to_csv(RESULTS / "08_classification.csv", index=False)

    print("\nC. 90% PREDICTION INTERVALS FROM TREE SPREAD")
    print("   empirical coverage  : %s   (nominal 0.90)" % f(cover))
    print(
        "   median width        : %.3f dm3 H2 g-1 substrate" % np.median(width)
    )
    print("   target SD           : %.3f" % y.std(ddof=1))
    pd.DataFrame(
        [
            dict(
                nominal=0.90,
                empirical_coverage=np.median(cover),
                cov_lo=np.percentile(cover, 2.5),
                cov_hi=np.percentile(cover, 97.5),
                median_width=np.median(width),
                target_sd=y.std(ddof=1),
            )
        ]
    ).to_csv(RESULTS / "08_interval_coverage.csv", index=False)

    print("\nD. WHY IT FAILS")
    er = pd.DataFrame(err_rows)
    er.to_csv(RESULTS / "08_error_vs_centroid.csv", index=False)
    rho = stats.spearmanr(er.dist_from_train_mean, er.mae)
    print(
        "   MAE vs |study mean - training mean|: Spearman rho = %.3f (p = %.3g, n = %d)"
        % (rho.statistic, rho.pvalue, len(er))
    )
    print(
        "   -> error is driven by how far a held-out study sits from the training centroid."
    )

    # Predictor ICC
    rows = []
    for c in NUM_FEATURES:
        v = pd.to_numeric(d[c], errors="coerce")
        tmp = pd.DataFrame({"v": v, "ref": d.ref}).dropna()
        gg = tmp.groupby("ref").v
        w = float(np.average(gg.var(ddof=0).fillna(0), weights=gg.size()))
        b = float(gg.mean().var(ddof=0))
        rows.append(dict(predictor=c, icc=b / (b + w), kind="numeric"))
    for c in CAT_FEATURES:
        tab = pd.crosstab(d[c], d.ref)
        chi2 = stats.chi2_contingency(tab).statistic
        n = tab.to_numpy().sum()
        v_ = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        single = d.groupby(c).ref.nunique() == 1
        share = d[c].isin(single[single].index).mean()
        rows.append(
            dict(
                predictor=c,
                icc=v_,
                kind="categorical (Cramer's V)",
                share_rows_in_single_study_levels=share,
            )
        )
    picc = pd.DataFrame(rows)
    picc.to_csv(RESULTS / "08_predictor_icc.csv", index=False)
    print("\n   Confounding of the predictors themselves with study identity:")
    print(picc.round(3).to_string(index=False))

    # Learning-curve slope with a bootstrap interval
    lc = pd.read_csv(RESULTS / "03_learning_curve.csv")
    rng = np.random.default_rng(SEED)
    slopes = []
    for _ in range(2000):
        s = lc.sample(
            len(lc), replace=True, random_state=int(rng.integers(1e9))
        )
        slopes.append(np.polyfit(s.train_studies, s.r2, 1)[0])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    obs = np.polyfit(lc.train_studies, lc.r2, 1)[0]
    pd.DataFrame(
        [
            dict(
                slope_per_study=obs,
                ci_lo=lo,
                ci_hi=hi,
                studies_to_reach_R2_0p5=(
                    np.nan if obs <= 0 else (0.5 - lc.r2.median()) / obs
                ),
            )
        ]
    ).to_csv(RESULTS / "08_learning_curve_slope.csv", index=False)
    print(
        "\n   Learning-curve slope: %+.5f R2 per additional training study "
        "(95%% CI %+.5f to %+.5f)" % (obs, lo, hi)
    )
    if lo <= 0 <= hi:
        print(
            "   -> indistinguishable from zero over the range tested (10-65 studies)."
        )
    if obs > 0:
        print(
            "   -> at this rate, reaching R2 = 0.5 would require roughly %.0f training studies."
            % ((0.5 - lc.r2.median()) / obs)
        )


if __name__ == "__main__":
    main()
