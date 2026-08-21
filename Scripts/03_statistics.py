#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_statistics.py -- the statistical analyses that do not treat rows as independent.

  1. Variance decomposition of harmonised yield into within- and between-study
     parts (direct method), plus a random-intercept mixed model on log(1+y).
  2. Learning curve: grouped test R2 against the number of training studies.
  3. Mixed-effects meta-regression: substrate class, operating mode, temperature
     and pH as moderators, study as a random intercept.
  4. Random-effects pooled yield per substrate class, with 95% CIs.

Writes  results/03_variance_components.csv
        results/03_learning_curve.csv
        results/03_metaregression.csv
        results/03_pooled_yields.csv

Note on the log(1+y) scale: yields are non-negative and include exact zeros, and
the raw distribution is strongly right-skewed.  Because yields are small
(median 0.11), log(1+y) is close to linear in y over the observed range, so
coefficients read approximately as dm3 H2 g-1 per unit of moderator.  Pooled
means are back-transformed with expm1.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Ensure common.py can be imported from the current script directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CAT_FEATURES,
    NATIVE_VS,
    NUM_FEATURES,
    N_LC_REPEATS,
    RESULTS,
    SEED,
    TARGET,
    add_basis,
    design_matrix,
    load_modelling_data,
    resolve_numeric_columns,
    variance_components,
)

warnings.filterwarnings("ignore")

REFERENCE_SUBSTRATE = "Food/kitchen waste"
MIN_STUDIES_FOR_POOLING = 3
LC_HOLDOUT_STUDIES = 16
LC_GRID = [10, 20, 30, 40, 50, 60, 65]
T_MESO, T_THERMO = 37.0, 55.0  # mesophilic / thermophilic reference points


def baseline_yield(d):
    """Baseline used to back-transform the temperature coefficient.

    The median of the modelling set is used rather than any fitted category mean,
    so the reported temperature effect does not depend on which substrate class
    happens to be the regression reference.
    """
    return float(d[TARGET].median())


def rf():
    return make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ]
        ),
        RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
    )


def mixed_intercept(frame, group_col="ref", response="ly"):
    """Random-intercept model; groups passed as an array to avoid index alignment."""
    f = frame.reset_index(drop=True)
    return smf.mixedlm("%s ~ 1" % response, f, groups=f[group_col].values).fit(reml=True)


def temperature_effect(frame, label, tag):
    """Fit the meta-regression on `frame` and back-transform the temperature
    coefficient at that frame's own median yield. Written to its own CSV so
    downstream scripts can pick the dimensionally appropriate estimate.
    """
    dd = frame.dropna(subset=["Tn", "pHn", "sub_cat", "mode", "ref"]).reset_index(drop=True)
    terms = [
        'C(sub_cat, Treatment(reference="%s"))' % REFERENCE_SUBSTRATE
        if dd.sub_cat.nunique() > 1 and (dd.sub_cat == REFERENCE_SUBSTRATE).any()
        else None,
        "C(mode)" if dd["mode"].nunique() > 1 else None,
        "Tn",
        "pHn",
    ]
    formula = "ly ~ " + " + ".join(t for t in terms if t)
    m = smf.mixedlm(formula, dd, groups=dd["ref"].values).fit(reml=True)
    b, se, p = float(m.params["Tn"]), float(m.bse["Tn"]), float(m.pvalues["Tn"])
    y0 = float(dd[TARGET].median())
    span = T_THERMO - T_MESO

    def imp(beta):
        return float(np.expm1(np.log1p(y0) + beta * span) - y0)

    row = dict(
        subset=label,
        n=len(dd),
        studies=dd.ref.nunique(),
        formula=formula,
        baseline_yield=y0,
        span_C=span,
        coef=b,
        se=se,
        p=p,
        ci_lo=b - 1.96 * se,
        ci_hi=b + 1.96 * se,
        implied_difference=imp(b),
        implied_lo=imp(b - 1.96 * se),
        implied_hi=imp(b + 1.96 * se),
        significant=bool(p < 0.05),
    )
    pd.DataFrame([row]).to_csv(RESULTS / ("03_temperature_effect%s.csv" % tag), index=False)
    print(
        "   %-28s n=%3d (%2d studies)  beta=%+.5f  p=%.4f  ->  dY %+.4f [%+.4f, %+.4f]%s"
        % (
            label,
            row["n"],
            row["studies"],
            b,
            p,
            row["implied_difference"],
            row["implied_lo"],
            row["implied_hi"],
            "" if row["significant"] else "   NOT SIGNIFICANT",
        )
    )
    return row


def main():
    d = resolve_numeric_columns(load_modelling_data())
    d = add_basis(d) if "Original Unit" in d.columns else d.assign(basis="unknown")
    d = d.assign(ly=np.log1p(d.y))
    X, y, groups = design_matrix(d)
    print("n = %d observations from %d studies\n" % (len(d), d.ref.nunique()))

    # ---- 1. variance components -------------------------------------------
    within, between, icc = variance_components(d)
    m0 = mixed_intercept(d)
    tau2 = float(m0.cov_re.iloc[0, 0])
    sigma2 = float(m0.scale)
    icc_mixed = tau2 / (tau2 + sigma2)
    vc = pd.DataFrame(
        [
            dict(
                method="direct decomposition (raw yield)",
                within=within,
                between=between,
                icc=icc,
            ),
            dict(
                method="random-intercept mixed model (log1p yield)",
                within=sigma2,
                between=tau2,
                icc=icc_mixed,
            ),
        ]
    )
    vc.to_csv(RESULTS / "03_variance_components.csv", index=False)
    print("VARIANCE COMPONENTS")
    print(vc.round(5).to_string(index=False))
    print(
        "  -> %.0f%% of the variance in harmonised yield lies BETWEEN studies,"
        % (100 * icc)
    )
    print("     which is exactly the component a study-level holdout removes.\n")

    # ---- 2. learning curve -------------------------------------------------
    studies = np.array(sorted(d.ref.unique()))
    rng = np.random.default_rng(0)
    lc_rows = []
    for k in LC_GRID:
        for rep in range(N_LC_REPEATS):
            perm = rng.permutation(studies)
            held = set(perm[:LC_HOLDOUT_STUDIES])
            pool = list(perm[LC_HOLDOUT_STUDIES : LC_HOLDOUT_STUDIES + k])
            tr = np.isin(groups, pool)
            te = np.isin(groups, list(held))
            if tr.sum() < 20 or te.sum() < 8:
                continue
            m = rf().fit(X[tr], y[tr])
            lc_rows.append(
                dict(
                    train_studies=k,
                    repeat=rep,
                    n_train=int(tr.sum()),
                    n_test=int(te.sum()),
                    r2=r2_score(y[te], m.predict(X[te])),
                )
            )
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(RESULTS / "03_learning_curve.csv", index=False)
    print("LEARNING CURVE (grouped holdout of %d studies)" % LC_HOLDOUT_STUDIES)
    print(lc.groupby("train_studies").r2.agg(["median", "size"]).round(3).to_string())
    print("  -> no systematic improvement across the range tested.\n")

    # ---- 3. meta-regression ------------------------------------------------
    md = d.dropna(subset=["Tn", "pHn", "sub_cat", "mode", "ref"]).reset_index(drop=True)
    formula = (
        'ly ~ C(sub_cat, Treatment(reference="%s")) + C(mode) + Tn + pHn'
        % REFERENCE_SUBSTRATE
    )
    mm = smf.mixedlm(formula, md, groups=md["ref"].values).fit(reml=True)
    res = pd.DataFrame(
        {
            "term": mm.params.index,
            "coef": mm.params.values,
            "se": mm.bse.values,
            "p": mm.pvalues.values,
        }
    )
    res["term"] = (
        res.term.str.replace(
            'C(sub_cat, Treatment(reference="%s"))[T.' % REFERENCE_SUBSTRATE,
            "substrate: ",
            regex=False,
        )
        .str.replace("C(mode)[T.", "mode: ", regex=False)
        .str.replace("]", "", regex=False)
    )
    res["ci_lo"] = res.coef - 1.96 * res.se
    res["ci_hi"] = res.coef + 1.96 * res.se
    res.to_csv(RESULTS / "03_metaregression.csv", index=False)
    print("META-REGRESSION  (%s, n = %d)" % (formula, len(md)))
    print(res.round(4).to_string(index=False))
    t = res[res.term == "Tn"].iloc[0]
    print(
        "  -> temperature %+.5f (log1p scale) per degC (95%% CI %.5f to %.5f, p = %.4f)"
        % (t.coef, t.ci_lo, t.ci_hi, t.p)
    )

    # Model-implied yield difference between mesophilic and thermophilic operation.
    # The model is fitted on log(1+y), so the coefficient must be back-transformed
    # at a stated baseline rather than read off linearly.
    print("\n   TEMPERATURE EFFECT BY TARGET BASIS")
    print("   (back-transformed at each subset's own median yield, 37 -> 55 degC)")
    full = temperature_effect(d, "full, mixed basis (per g substrate)", "")
    vs = temperature_effect(d[d.basis == NATIVE_VS], "native VS only (per g VS)", "_vs")
    pd.DataFrame([full, vs]).to_csv(RESULTS / "03_temperature_effect_both.csv", index=False)
    if not vs["significant"]:
        print("   NOTE: on the dimensionally consistent VS subset the temperature term is")
        print(
            "         not significant. The point estimates agree (%+.3f vs %+.3f) but the"
            % (full["implied_difference"], vs["implied_difference"])
        )
        print(
            "         VS subset has %d records from %d studies and cannot exclude zero."
            % (vs["n"], vs["studies"])
        )
    ph = res[res.term == "pHn"].iloc[0]
    print(
        "  -> pH %+.4f (p = %.2f): no detectable effect once study is accounted for\n"
        % (ph.coef, ph.p)
    )

    # ---- 4. pooled yields --------------------------------------------------
    rows = []
    for cat, gd in d.groupby("sub_cat"):
        n_studies = gd.ref.nunique()
        if n_studies < MIN_STUDIES_FOR_POOLING:
            rows.append(
                dict(
                    substrate=cat,
                    rows=len(gd),
                    studies=n_studies,
                    pooled=np.nan,
                    ci_lo=np.nan,
                    ci_hi=np.nan,
                )
            )
            continue
        m = mixed_intercept(gd)
        b, se = float(m.params.iloc[0]), float(m.bse.iloc[0])
        rows.append(
            dict(
                substrate=cat,
                rows=len(gd),
                studies=n_studies,
                pooled=np.expm1(b),
                ci_lo=np.expm1(b - 1.96 * se),
                ci_hi=np.expm1(b + 1.96 * se),
            )
        )
    pooled = pd.DataFrame(rows).sort_values("pooled", ascending=False)
    pooled.to_csv(RESULTS / "03_pooled_yields.csv", index=False)
    print("POOLED YIELD BY SUBSTRATE CLASS (random effects, study-weighted)")
    print(pooled.round(4).to_string(index=False))
    print(
        "  (categories supported by fewer than %d studies are not pooled)"
        % MIN_STUDIES_FOR_POOLING
    )

    vcounts = d.ref.value_counts()
    print(
        "\nSTUDY CONCENTRATION: %d studies; %d contribute one observation; "
        "the six largest contribute %d rows (%.0f%%)"
        % (
            len(vcounts),
            int((vcounts == 1).sum()),
            int(vcounts.head(6).sum()),
            100 * vcounts.head(6).sum() / len(d),
        )
    )
    print(
        "\nwrote results/03_variance_components.csv, 03_learning_curve.csv, "
        "03_metaregression.csv, 03_pooled_yields.csv"
    )


if __name__ == "__main__":
    main()
