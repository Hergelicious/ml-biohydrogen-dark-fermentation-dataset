"""
10_reconcile.py — diff every published number against what the pipeline actually returns.

Why this exists
    The numbers in the manuscript were produced by the authors' original analysis
    code. The pipeline in this repository is a clean reimplementation of the
    documented methodology. Until the two are reconciled, publishing the pipeline
    as "the code behind these results" is an unverified claim — and an unverified
    reproducibility claim in a paper about validation rigour is the worst possible
    place to have one.

    This script reads data/manuscript_claims.csv, which lists every value the paper
    asserts together with where it appears, pulls the corresponding value out of
    results/, and reports a status for each.

Tolerance types
    exact     Counts and sample sizes. Any difference is a real disagreement.
    abs       Deterministic statistics (ICC, regression coefficients, derived
              quantities). Small tolerance for floating-point and REML convergence.
    resample  Medians over a resampling distribution. These are seed-dependent:
              identical code with different seeds moves an R2 median by roughly
              +/-0.02-0.05. A difference inside tolerance is NOT a discrepancy.

How to act on the output
    PASS       Nothing to do.
    TOLERANCE  Within the seed-noise band. Note it; do not chase it.
    MISMATCH   Investigate. The likely_cause column names the most probable source.
    MISSING    The pipeline did not produce that output. Run the upstream script.

    Where a MISMATCH is real, the authors' original code is the source of truth for
    the published numbers. Either adjust this pipeline until it reproduces them, or
    — if the pipeline turns out to be right and the original wrong — correct the
    manuscript before submission. Do not publish a repository that silently
    disagrees with the paper.

Outputs
    results/10_reconciliation.csv
    results/10_reconcile.log
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, RESULTS, get_logger, write_table

log = get_logger("10_reconcile")

CLAIMS = DATA / "manuscript_claims.csv"


def fetch(row) -> tuple[float | None, str]:
    """Pull the pipeline value for one claim. Returns (value, note)."""
    path = RESULTS / str(row["output_file"])
    if not path.exists():
        return None, f"{path.name} not found"
    df = pd.read_csv(path)

    filt = str(row["row_filter"]).strip()
    if filt and filt.lower() != "nan":
        try:
            sub = df.query(filt) if not any(t in filt for t in (".str.", "contains")) \
                else df[df.eval(filt, engine="python")]
        except Exception:
            try:
                sub = df[df.eval(filt, engine="python")]
            except Exception as exc:
                return None, f"filter failed: {type(exc).__name__}"
    else:
        sub = df

    if len(sub) == 0:
        return None, "filter matched no rows"
    col = str(row["column"])
    if col not in sub.columns:
        return None, f"column {col!r} absent (have: {list(sub.columns)[:6]})"
    val = pd.to_numeric(sub[col], errors="coerce").dropna()
    if val.empty:
        return None, "value not numeric"
    if len(val) > 1:
        return float(val.iloc[0]), f"filter matched {len(val)} rows; took the first"
    return float(val.iloc[0]), ""


def classify(claimed: float, actual: float | None, ttype: str, tol: float) -> str:
    if actual is None:
        return "MISSING"
    diff = abs(actual - claimed)
    if ttype == "exact":
        return "PASS" if diff == 0 else "MISMATCH"
    if diff <= tol:
        return "PASS" if ttype == "abs" else "TOLERANCE"
    return "MISMATCH"


def main() -> None:
    if not CLAIMS.exists():
        raise FileNotFoundError(f"{CLAIMS} not found")
    claims = pd.read_csv(CLAIMS)
    log.info("reconciling %d published values against results/", len(claims))

    out = []
    for _, row in claims.iterrows():
        actual, note = fetch(row)
        claimed = float(row["claimed"])
        status = classify(claimed, actual, str(row["tolerance_type"]), float(row["tolerance"]))
        out.append({
            "quantity": row["quantity"],
            "source_in_paper": row["source_in_paper"],
            "manuscript": claimed,
            "pipeline": (round(actual, 5) if actual is not None else None),
            "difference": (round(actual - claimed, 5) if actual is not None else None),
            "tolerance_type": row["tolerance_type"],
            "tolerance": row["tolerance"],
            "status": status,
            "likely_cause": (row["likely_cause_if_mismatch"]
                             if status == "MISMATCH" else ""),
            "note": note,
        })

    rec = pd.DataFrame(out)
    write_table(rec, "10_reconciliation.csv")

    counts = rec["status"].value_counts().to_dict()
    log.info("-" * 78)
    for s in ("PASS", "TOLERANCE", "MISMATCH", "MISSING"):
        log.info("%-10s %d", s, counts.get(s, 0))
    log.info("-" * 78)

    bad = rec[rec.status == "MISMATCH"]
    if not bad.empty:
        log.warning("%d value(s) disagree beyond tolerance:", len(bad))
        for _, r in bad.iterrows():
            log.warning("  %-34s paper %-10s pipeline %-10s | %s",
                        str(r["quantity"])[:34], r["manuscript"], r["pipeline"],
                        r["likely_cause"])
    missing = rec[rec.status == "MISSING"]
    if not missing.empty:
        log.warning("%d value(s) could not be located:", len(missing))
        for _, r in missing.iterrows():
            log.warning("  %-34s %s", str(r["quantity"])[:34], r["note"])

    if bad.empty and missing.empty:
        log.info("RECONCILED. Every published value is reproduced within tolerance. "
                 "The repository can be cited as the code behind the paper.")
    else:
        log.info("NOT YET RECONCILED. Resolve the items above before publishing the "
                 "repository alongside the manuscript.")


if __name__ == "__main__":
    main()
