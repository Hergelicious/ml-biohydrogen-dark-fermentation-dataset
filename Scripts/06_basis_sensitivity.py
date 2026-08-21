#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_basis_sensitivity.py -- does the grouped-validation collapse survive when the
target is restricted to a single mass basis?

The compiled target inherits its mass basis from each source. This script
classifies every modelled record by that basis and re-runs the full three-protocol
ladder on progressively stricter subsets:

    B0  all records                          (n = 224)
    B1  drop COD / dry-biomass / unstated    (n = 212)
    B2  VS records + chemically defined substrates (Group B), where the
        substrate-molecule basis is close to a VS basis                (n ~ 178)
    B3  strictly VS-reported records only                              (n ~ 102)

If the row-wise / grouped gap persists at B3, the collapse cannot be attributed
to heterogeneity of the target denominator.

Writes results/06_basis_composition.csv and results/06_basis_ladder.csv
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
    AUDITED_CSV,
    CAT_FEATURES,
    NOISE_SD,
    NUM_FEATURES,
    RESULTS,
    SEED,
    TEST_FRACTION,
    add_features,
    design_matrix,
    summarise,
    variance_components,
)

warnings.filterwarnings("ignore")
N = 40


def basis(unit):
    """Categorize original unit string into a canonical mass basis."""
    t = str(unit).lower()
    if "cod" in t:
        return "g COD"
    if "dry biomass" in t:
        return "g dry biomass (TS)"
    if "vss" in t:
        return "g VS"
    if "vs" in t and "mol" not in t:
        return "g VS"
    if "mol" in t:
        return (
            "g hexose equivalent"
            if ("hexose" in t or "glucose eq" in t or "c6" in t or "sugar" in t)
            else "g named substrate"
        )
    return "g substrate (unstated)"


def model():
    """Build a Standard Random Forest Pipeline."""
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
    """Run cross-validation protocols across noise-augmented, row-wise, and grouped splits."""
    X, y, g = design_matrix(d)
    out = {}
    for proto in ["augmented", "rowwise", "grouped"]:
        sc = []
        for s in range(n):
            if proto == "augmented":
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
                    r2_score(yr[te], model().fit(Xr.iloc[tr], yr[tr]).predict(Xr.iloc[te]))
                )
            else:
                if proto == "grouped":
                    tr, te = next(
                        GroupShuffleSplit(
                            1, test_size=TEST_FRACTION, random_state=s
                        ).split(X, y, g)
                    )
                else:
                    tr, te = next(
                        ShuffleSplit(1, test_size=TEST_FRACTION, random_state=s).split(X)
                    )
                sc.append(
                    r2_score(y[te], model().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                )
        out[proto] = sc

    # Mean-predictor reference baseline under study grouping
    sc = []
    for s in range(n):
        tr, te = next(
            GroupShuffleSplit(1, test_size=TEST_FRACTION, random_state=s).split(X, y, g)
        )
        sc.append(
            r2_score(
                y[te],
                DummyRegressor(strategy="mean")
                .fit(X.iloc[tr], y[tr])
                .predict(X.iloc[te]),
            )
        )
    out["grouped_mean_baseline"] = sc
    return out


def main():
    a = pd.read_csv(AUDITED_CSV)
    d = add_features(a[a.recommended_for_modelling & a.dm3_H2_per_g.notna()].copy())
    d["basis"] = d["Original Unit"].map(basis)

    comp = (
        d.groupby("basis")
        .agg(
            records=("y", "size"),
            studies=("ref", "nunique"),
            mean_yield=("y", "mean"),
            median_yield=("y", "median"),
        )
        .sort_values("records", ascending=False)
        .reset_index()
    )
    comp.to_csv(RESULTS / "08_basis_composition.csv", index=False)
    print("TARGET BASIS COMPOSITION")
    print(comp.round(4).to_string(index=False))

    SUBSETS = {
        "B0  all records": d,
        "B1  drop COD / TS / unstated": d[
            ~d.basis.isin(
                ["g COD", "g dry biomass (TS)", "g substrate (unstated)"]
            )
        ],
        "B2  VS + defined substrates": d[
            d.basis.isin(["g VS", "g named substrate"])
        ],
        "B3  VS-reported only": d[d.basis == "g VS"],
    }

    rows = []
    print("\nRANDOM FOREST TEST R2 BY TARGET BASIS AND PROTOCOL")
    print(
        "%-30s %6s %7s  %-22s %-22s %-22s %-22s"
        % (
            "subset",
            "n",
            "studies",
            "noise-aug + random",
            "row-wise",
            "STUDY-GROUPED",
            "grouped mean baseline",
        )
    )
    for label, sub in SUBSETS.items():
        sub = sub.reset_index(drop=True)
        r = ladder(sub)
        w, b, icc = variance_components(sub)

        def f(k):
            return "%+.3f [%+.2f,%+.2f]" % summarise(r[k])

        print(
            "%-30s %6d %7d  %-22s %-22s %-22s %-22s"
            % (
                label,
                len(sub),
                sub.ref.nunique(),
                f("augmented"),
                f("rowwise"),
                f("grouped"),
                f("grouped_mean_baseline"),
            )
        )
        for k, v in r.items():
            m, lo, hi = summarise(v)
            rows.append(
                dict(
                    subset=label,
                    n=len(sub),
                    studies=sub.ref.nunique(),
                    icc=icc,
                    protocol=k,
                    median_r2=m,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
        rows[-4]["gap_rowwise_minus_grouped"] = np.median(r["rowwise"]) - np.median(
            r["grouped"]
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "08_basis_ladder.csv", index=False)

    print("\nLEAKAGE GAP AND HETEROGENEITY BY SUBSET")
    print(
        "%-30s %6s %8s %10s %10s %8s"
        % ("subset", "n", "studies", "row-wise", "grouped", "ICC")
    )
    for label, sub in SUBSETS.items():
        s = out[out.subset == label]
        rw = float(s[s.protocol == "rowwise"].median_r2.iloc[0])
        gp = float(s[s.protocol == "grouped"].median_r2.iloc[0])
        print(
            "%-30s %6d %8d %10.3f %10.3f %8.2f  gap %+.3f"
            % (
                label,
                s.n.iloc[0],
                s.studies.iloc[0],
                rw,
                gp,
                s.icc.iloc[0],
                rw - gp,
            )
        )
    print("\nwrote results/06_basis_composition.csv, 06_basis_ladder.csv")


if __name__ == "__main__":
    main()
