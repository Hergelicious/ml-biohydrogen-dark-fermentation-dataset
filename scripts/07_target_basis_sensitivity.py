"""
07_target_basis_sensitivity.py — is the collapse an artefact of the mixed target basis?

The most obvious objection to a grouped-validation failure on a compiled dataset
is that the harmonised target mixes mass denominators, so the failure might
reflect denominator heterogeneity rather than genuine study effects. This script
repeats the validation ladder on four nested subsets of increasing strictness.

The collapse deepens as the basis is purified, so denominator heterogeneity is
not the cause. This script also writes the tertile-classification and
error-vs-centroid analyses (note the 07_ filename prefix; 08_what_transfers.py
writes rank transfer, interval coverage and predictor ICC).

Outputs
    results/07_basis_composition.csv    record counts by original mass basis
    results/07_basis_sensitivity.csv    Table S26
    results/07_classification.csv       tertile classification under grouping
    results/07_error_vs_centroid.csv    error against distance from training centroid
    results/07_basis.log
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from common import (FEATURES, GROUP_COL, N_RESAMPLES, TARGET, TEST_SIZE,
                    get_logger, load_modelling_set, summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("07_basis")


def classify_basis(unit: str) -> str:
    """Assign each record to the mass basis its source actually reported."""
    u = str(unit).lower()
    if "cod" in u:
        return "chemical oxygen demand"
    if "vs" in u or "volatile" in u:
        return "volatile solids"
    if "ts" in u or "dry" in u or "tvs" in u:
        return "dry biomass"
    if "mol" in u:
        return "molar (per substrate molecule)"
    return "unstated"


def ladder(df: pd.DataFrame, label: str) -> dict:
    from importlib import import_module
    mod = import_module("02_validation_ladder")

    out = {"subset": label, "n": len(df), "studies": df[GROUP_COL].nunique()}
    for protocol, aug in [("row_wise", True), ("row_wise", False), ("study_grouped", False)]:
        key = "noise_aug_random" if aug else protocol
        scores = []
        for seed in range(N_RESAMPLES):
            rng = np.random.default_rng(seed)
            work = mod.augment(df, rng) if aug else df
            sp = (GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                  if protocol == "study_grouped"
                  else ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed))
            groups = work[GROUP_COL] if protocol == "study_grouped" else None
            tr, te = next(sp.split(work[FEATURES], work[TARGET], groups=groups))
            model = mod.make_model("RandomForest")
            model.fit(work[FEATURES].iloc[tr], work[TARGET].iloc[tr])
            scores.append(r2_score(work[TARGET].iloc[te],
                                   model.predict(work[FEATURES].iloc[te])))
        s = summarise(scores)
        out[key] = s["median"]
        out[f"{key}_lo"] = s["pct_lo"]
        out[f"{key}_hi"] = s["pct_hi"]

    # mean-predictor baseline under grouping
    dummy = []
    for seed in range(N_RESAMPLES):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
        pred = np.full(len(te), df[TARGET].iloc[tr].mean())
        dummy.append(r2_score(df[TARGET].iloc[te], pred))
    out["mean_predictor"] = float(np.median(dummy))
    out["leakage_gap"] = out["row_wise"] - out["study_grouped"]
    out["margin_over_mean"] = out["study_grouped"] - out["mean_predictor"]

    # intraclass correlation of the target within this subset
    groups = [g[TARGET].to_numpy(float) for _, g in df.groupby(GROUP_COL)]
    n_total = int(sum(g.size for g in groups))
    means = np.array([g.mean() for g in groups])
    within = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups])) / n_total
    between = float(means.var(ddof=0))
    out["icc"] = between / (between + within) if (between + within) > 0 else np.nan
    return out


def classification_and_centroid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tertile classification, and error against distance from the training centroid."""
    from importlib import import_module
    mod = import_module("02_validation_ladder")

    acc, centroid_rows = [], []
    for seed in range(N_RESAMPLES):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        tr, te = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))
        ytr, yte = df[TARGET].iloc[tr], df[TARGET].iloc[te]

        model = mod.make_model("RandomForest").fit(df[FEATURES].iloc[tr], ytr)
        pred = model.predict(df[FEATURES].iloc[te])

        # tertile edges are defined on the training fold only
        edges = np.quantile(ytr, [1 / 3, 2 / 3])
        acc.append(accuracy_score(np.digitize(yte, edges), np.digitize(pred, edges)))

        train_mean = float(ytr.mean())
        test_groups = df[GROUP_COL].iloc[te]
        for study in pd.unique(test_groups):
            mask = (test_groups == study).to_numpy()
            if mask.sum() < 2:
                continue
            centroid_rows.append({
                "seed": seed, "study": study, "n_obs": int(mask.sum()),
                "distance_from_training_centroid": abs(float(yte[mask].mean()) - train_mean),
                "mae": mean_absolute_error(yte[mask], pred[mask]),
            })

    cls = pd.DataFrame([{"metric": "tertile classification accuracy",
                         "chance_rate": 1 / 3, **summarise(acc)}])
    cen = pd.DataFrame(centroid_rows)
    if not cen.empty:
        rho, p = stats.spearmanr(cen["distance_from_training_centroid"], cen["mae"])
        log.info("error vs training centroid: Spearman rho = %.2f over %d "
                 "study-resample combinations (not independent; each study recurs across "
                 "resamples, so the nominal p = %.2e is anticonservative)", rho, len(cen), p)
        cen.attrs["spearman_rho"] = rho
    return cls, cen


def main() -> None:
    df = load_modelling_set()
    df["mass_basis"] = df["Original Unit"].map(classify_basis)

    comp = (df.groupby("mass_basis")
              .agg(records=("mass_basis", "size"), studies=(GROUP_COL, "nunique"))
              .reset_index().sort_values("records", ascending=False))
    write_table(comp, "07_basis_composition.csv")
    log.info("mass-basis composition:\n%s", comp.to_string(index=False))

    # four nested subsets of increasing strictness
    strict_molar = df[df["Unit Group Recomputed"].eq("B")]
    tiers = [
        (df, "all harmonised records"),
        (df[~df["mass_basis"].isin(["chemical oxygen demand", "dry biomass", "unstated"])],
         "excluding COD, dry-biomass and unstated bases"),
        (df[~df["mass_basis"].isin(["chemical oxygen demand", "dry biomass", "unstated"]) &
            ~df["Unit Group Recomputed"].eq("C")],
         "excluding the Group C hexose-equivalent convention also"),
        (df[df["mass_basis"].eq("volatile solids")], "native volatile solids only"),
    ]
    rows = []
    for subset, label in tiers:
        if subset[GROUP_COL].nunique() < 8:
            log.warning("%s: only %d studies — skipped", label, subset[GROUP_COL].nunique())
            continue
        r = ladder(subset, label)
        rows.append(r)
        log.info("%-52s n=%3d k=%2d  grouped R2 = %+.3f  gap = %+.3f  ICC = %.3f",
                 label, r["n"], r["studies"], r["study_grouped"],
                 r["leakage_gap"], r["icc"])
    write_table(pd.DataFrame(rows), "07_basis_sensitivity.csv")

    cls, cen = classification_and_centroid(df)
    write_table(cls, "07_classification.csv")
    write_table(cen, "07_error_vs_centroid.csv")
    log.info("done")


if __name__ == "__main__":
    main()
