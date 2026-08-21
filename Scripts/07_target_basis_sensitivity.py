#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_target_basis_sensitivity.py -- does the study-grouped collapse survive when the
modelling target is restricted to a single, mathematically comparable mass basis?

The compilation inherits its mass basis from the source literature. This script
classifies every modelled record by basis and repeats the full validation ladder
on four nested subsets of increasing strictness:

    T1  all records                      (224)  mixed basis
    T2  drop COD / dry-biomass / unstated       removes the bases that are not
                                                interconvertible without extra data
    T3  T2 minus hexose-equivalent (Group C)    removes the assumed-denominator records
    T4  volatile solids only                    strictest: only records the source
                                                itself reported per g VS

If the ordering augmented > row-wise > grouped, and the grouped collapse to the
mean-predictor level, survive down to T4, the result cannot be an artefact of
mixing denominators.

Writes results/08_basis_composition.csv
       results/08_basis_sensitivity.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Ensure common.py can be imported from the current script directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    CAT_FEATURES,
    NOISE_SD,
    NUM_FEATURES,
    RESULTS,
    SEED,
    TEST_FRACTION,
    design_matrix,
    load_modelling_data,
    summarise,
    variance_components,
)

warnings.filterwarnings("ignore")
N = 40


def basis_of(unit, group):
    """Mass basis of the denominator, derived from the original unit string."""
    u = str(unit).lower()
    if "cod" in u:
        return "g COD"
    if "dry biomass" in u:
        return "g dry biomass (TS)"
    if "mol" in u:
        return (
            "g hexose-equivalent"
            if str(group).strip().endswith("C")
            else "g named substrate"
        )
    if "vss" in u or "vs" in u:
        return "g volatile solids"
    return "g substrate (unstated)"


TIERS = {
    "T1  all records (mixed basis)": None,
    "T2  drop COD / dry biomass / unstated": {
        "g COD",
        "g dry biomass (TS)",
        "g substrate (unstated)",
    },
    "T3  T2 minus hexose-equivalent": {
        "g COD",
        "g dry biomass (TS)",
        "g substrate (unstated)",
        "g hexose-equivalent",
    },
    "T4  volatile solids only": {
        "g COD",
        "g dry biomass (TS)",
        "g substrate (unstated)",
        "g hexose-equivalent",
        "g named substrate",
    },
}


def rf():
    """Build standard Random Forest Pipeline."""
    return make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
                ("num", SimpleImputer(strategy="median"), NUM_FEATURES),
            ]
        ),
        RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
    )


def ladder(d, n=N):
    """Median [2.5, 97.5] test R2 under the three protocols, plus a grouped mean baseline."""
    X, y, g = design_matrix(d)
    out = {}
    for protocol in ["augmented", "rowwise", "grouped", "grouped_mean"]:
        sc = []
        for s in range(n):
            if protocol == "augmented":
                rng = np.random.default_rng(s)
                Xa = X.copy()
                for c in NUM_FEATURES:
                    Xa[c] = Xa[c] + NOISE_SD * rng.normal(size=len(Xa))
                Xr = pd.concat([X, Xa], ignore_index=True)
                yr = np.concatenate([y, y + NOISE_SD * rng.normal(size=len(y))])
                tr, te = train_test_split(
                    np.arange(len(Xr)), test_size=TEST_FRACTION, random_state=s
                )
                sc.append(
                    r2_score(yr[te], rf().fit(Xr.iloc[tr], yr[tr]).predict(Xr.iloc[te]))
                )
            elif protocol == "rowwise":
                tr, te = next(
                    ShuffleSplit(1, test_size=TEST_FRACTION, random_state=s).split(X)
                )
                sc.append(
                    r2_score(y[te], rf().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                )
            else:
                tr, te = next(
                    GroupShuffleSplit(
                        1, test_size=TEST_FRACTION, random_state=s
                    ).split(X, y, g)
                )
                model = (
                    DummyRegressor(strategy="mean")
                    if protocol == "grouped_mean"
                    else rf()
                )
                sc.append(
                    r2_score(y[te], model.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                )
        out[protocol] = sc
    return out


def main():
    d = load_modelling_data()
    d["basis"] = [
        basis_of(u, g) for u, g in zip(d["Original Unit"], d["Unit Group"])
    ]

    comp = (
        d.groupby("basis")
        .agg(
            records=("y", "size"),
            studies=("ref", "nunique"),
            mean_yield=("y", "mean"),
        )
        .sort_values("records", ascending=False)
        .reset_index()
    )
    comp.to_csv(RESULTS / "08_basis_composition.csv", index=False)
    print("MASS BASIS OF THE MODELLING TARGET")
    print(comp.round(4).to_string(index=False))

    rows = []
    print("\nVALIDATION LADDER BY TARGET-BASIS TIER")
    print(
        "%-40s %5s %7s | %-22s %-22s %-22s %-22s"
        % (
            "tier",
            "n",
            "studies",
            "augmented",
            "row-wise",
            "GROUPED (RF)",
            "grouped mean pred.",
        )
    )
    for label, drop in TIERS.items():
        sub = d if drop is None else d[~d.basis.isin(drop)]
        sub = sub.reset_index(drop=True)
        res = ladder(sub)
        icc = variance_components(sub)[2]

        def f(k):
            return "%+.3f [%+.2f,%+.2f]" % summarise(res[k])

        print(
            "%-40s %5d %7d | %-22s %-22s %-22s %-22s"
            % (
                label,
                len(sub),
                sub.ref.nunique(),
                f("augmented"),
                f("rowwise"),
                f("grouped"),
                f("grouped_mean"),
            )
        )
        row = dict(tier=label, n=len(sub), studies=sub.ref.nunique(), icc=icc)
        for k in ["augmented", "rowwise", "grouped", "grouped_mean"]:
            m, lo, hi = summarise(res[k])
            row["%s_median" % k], row["%s_lo" % k], row["%s_hi" % k] = m, lo, hi
        row["gap_rowwise_minus_grouped"] = (
            row["rowwise_median"] - row["grouped_median"]
        )
        row["rf_minus_mean_grouped"] = (
            row["grouped_median"] - row["grouped_mean_median"]
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "08_basis_sensitivity.csv", index=False)

    print("\nSUMMARY")
    for _, r in out.iterrows():
        print(
            "  %-40s ICC %.2f | leakage gap %+.3f | RF minus mean predictor %+.3f"
            % (
                r.tier,
                r.icc,
                r.gap_rowwise_minus_grouped,
                r.rf_minus_mean_grouped,
            )
        )
    print("\nwrote results/08_basis_composition.csv, 08_basis_sensitivity.csv")


if __name__ == "__main__":
    main()
