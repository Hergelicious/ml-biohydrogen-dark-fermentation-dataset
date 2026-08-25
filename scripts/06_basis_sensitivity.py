"""
06_basis_sensitivity.py — behaviour across original unit conventions.

Describes how the compilation distributes over the three conversion routes and
repeats the validation ladder within each, as a descriptive precursor to the
strict target-basis test in 07_target_basis_sensitivity.py.

Outputs
    results/06_basis_ladder.csv
    results/06_basis.log
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from common import (FEATURES, GROUP_COL, N_RESAMPLES, TARGET, TEST_SIZE,
                    get_logger, load_modelling_set, summarise, write_table)

warnings.filterwarnings("ignore")
log = get_logger("06_basis")

MIN_STUDIES = 8  # a grouped split is meaningless below this


def ladder(df: pd.DataFrame) -> dict:
    """Row-wise and study-grouped medians for one subset."""
    from importlib import import_module
    ladder_mod = import_module("02_validation_ladder")
    out = {}
    for protocol in ("row_wise", "study_grouped"):
        scores = []
        for seed in range(N_RESAMPLES):
            sp = (GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
                  if protocol == "study_grouped"
                  else ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed))
            groups = df[GROUP_COL] if protocol == "study_grouped" else None
            tr, te = next(sp.split(df[FEATURES], df[TARGET], groups=groups))
            model = ladder_mod.make_model("RandomForest")
            model.fit(df[FEATURES].iloc[tr], df[TARGET].iloc[tr])
            scores.append(r2_score(df[TARGET].iloc[te], model.predict(df[FEATURES].iloc[te])))
        s = summarise(scores)
        out[protocol] = s["median"]
        out[f"{protocol}_lo"] = s["pct_lo"]
        out[f"{protocol}_hi"] = s["pct_hi"]
    out["leakage_gap"] = out["row_wise"] - out["study_grouped"]
    return out


def main() -> None:
    df = load_modelling_set()
    rows = [{"unit_group": "all routes", "n": len(df),
             "studies": df[GROUP_COL].nunique(), **ladder(df)}]
    log.info("all routes: n = %d, %d studies", len(df), df[GROUP_COL].nunique())

    for group, block in df.groupby("Unit Group Recomputed"):
        k = block[GROUP_COL].nunique()
        log.info("route %s: n = %d, %d studies", group, len(block), k)
        if k < MIN_STUDIES:
            log.info("  skipped — fewer than %d studies", MIN_STUDIES)
            rows.append({"unit_group": f"route {group}", "n": len(block), "studies": k})
            continue
        rows.append({"unit_group": f"route {group}", "n": len(block), "studies": k,
                     **ladder(block)})

    write_table(pd.DataFrame(rows), "06_basis_ladder.csv")
    log.info("done")


if __name__ == "__main__":
    main()
