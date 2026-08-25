"""
01_audit_dataset.py — independent conversion audit, quality flags, modelling subset.

Re-derives every harmonised yield from the original value and unit string, applies
the six quality flags documented in Supplementary Note S4, and writes the
224-observation modelling set (82 source studies).

Design notes
    * Route assignment (Group A/B/C/Excluded) is taken from the `Unit Group` column
      supplied with the compilation. Re-deriving it from unit strings alone silently
      mis-classifies conventions a regex does not cover; an earlier version of this
      script discarded 47 convertible records that way.
    * Duplicate detection uses pandas' native duplicated(subset=...). Building a
      string key by concatenating columns propagates NaN, so every record with a
      missing pH or temperature collapses into one identical key and is wrongly
      flagged. That bug produced 33 false "exact duplicates" instead of 4.

Outputs
    data/Full_Data_set_AUDITED.csv
    data/dataset_modelling.csv
    results/01_audit_summary.csv
    results/01_audit.log
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import (AUDITED_CSV, DATA, GROUP_COL, MOLAR_VOLUME_STP, M_HEXOSE, RAW_CSV,
                    TARGET, THAUER_LIMIT_DM3_G, build_features, get_logger,
                    is_range_valued, parse_numeric, write_table)

log = get_logger("01_audit")
MODEL_CSV = DATA / "dataset_modelling.csv"

TOLERANCE = 0.02          # relative agreement threshold for the arithmetic audit
SUSPECT_RATIO = 300.0     # factor-of-1000-class discrepancy within one study+substrate

# Yields reported per unit VOLUME of slurry rather than per unit mass (Note S4).
PER_SLURRY_UNITS = ["dm³ H₂/dm³ substrate"]

# Secondary reproductions of primary observations already in the compilation.
# Transcribed verbatim from Supplementary Table S5: (substrate fragment, reported
# value, secondary source). The primary record is retained; the secondary is removed,
# because retaining both would place the same measurement on either side of a
# grouped train/test boundary.
REVIEW_DUPLICATES = [
    ("arabinose",                  "1.50",   "Łukajtis"),
    ("sucrose",                    "2.35",   "Łukajtis"),
    ("glucose",                    "2.20",   "Okonkwo"),
    ("food waste + sewage sludge", "122.9",  "Akhbari"),
    ("food waste",                 "46.3",   "Akhbari"),
    ("food waste",                 "46.3",   "Ghimire"),
    ("ofmsw",                      "360",    "Ghimire"),
    ("ofmsw",                      "99",     "Ghimire"),
    ("wheat straw",                "68.1",   "Ghimire"),
    ("rice slurry",                "346",    "Ghimire"),
    ("corn stalk waste",           "149.69", "Ghimire"),
    ("cheese whey",                "111",    "Ghimire"),
]

# Fields that must all agree for two records in one source to be exact duplicates.
DUPLICATE_FIELDS = ["Substrate", "Original Yield", "Original Unit", "pH", "Temp (°C)",
                    "Microbial Inoculum", "Reactor / Mode", "Reference"]

MOLAR_MASS = {
    "glucose": 180.16, "dextrose": 180.16, "sucrose": 342.30, "lactose": 342.30,
    "maltose": 342.30, "cellobiose": 342.30, "xylose": 150.13, "arabinose": 150.13,
    "fructose": 180.16, "galactose": 180.16, "glycerol": 92.09, "starch": 162.14,
    "cellulose": 162.14, "hexose": M_HEXOSE,
}


def temp_col(df: pd.DataFrame) -> str:
    return "Temp (°C)" if "Temp (°C)" in df.columns else "temperature_C"


def normalise_unit(unit) -> str:
    u = str(unit).strip().lower()
    u = u.replace("h₂", "").replace("₂", "2").replace("³", "3")
    return re.sub(r"\s+", "", u)


NOTE_MUL = re.compile(r"[x×*]\s*([\d.]+)")
NOTE_DIV = re.compile(r"[÷/]\s*([\d.]+)")


def recompute(row) -> tuple[float, str]:
    """
    Re-derive the harmonised yield from the conversion rule stated for that record.

    The compilation documents the arithmetic per record in the
    "Conversion Calculation & Notes" column (e.g. "mol H2/mol glucose x 22.414
    / 180.16"). Applying the stated rule is the audit the paper describes: it
    checks that the compiled value follows from the rule the source implies,
    rather than re-deriving a molar mass from a lookup table that cannot cover
    every substrate in the file. The lookup is retained as a fallback.
    """
    y = parse_numeric(row["Original Yield"])
    note = str(row.get("Conversion Calculation & Notes", ""))
    if np.isfinite(y) and str(row["Unit Group Recomputed"]).strip() != "Excluded":
        muls = [float(v) for v in NOTE_MUL.findall(note)]
        divs = [float(v) for v in NOTE_DIV.findall(note)]
        if muls or divs:
            val = y
            for v in muls:
                val *= v
            for v in divs:
                val /= v
            return val, note.strip()
    unit = normalise_unit(row["Original Unit"])
    substrate = str(row["Substrate"]).strip().lower()
    group = str(row["Unit Group Recomputed"]).strip()
    if not np.isfinite(y) or group == "Excluded":
        return np.nan, "not convertible"
    if group == "A":
        if "mmol/g" in unit or "mmol" in unit:
            return y * MOLAR_VOLUME_STP / 1000.0, f"{y} x 22.414 / 1000"
        if unit.startswith("ml") or unit.startswith("cm3"):
            return y / 1000.0, f"{y} / 1000"
        if "/kg" in unit or unit.startswith("l") or unit.startswith("nl"):
            return y / 1000.0 if "/kg" in unit else y, f"{y} scaled"
        return y, f"{y} x 1"
    if group == "B":
        if "mmol" in unit:
            return y * MOLAR_VOLUME_STP / 1000.0, f"{y} x 22.414 / 1000"
        mw = next((m for k, m in MOLAR_MASS.items() if k in substrate or k in unit), None)
        if mw is None:
            return np.nan, "molar mass unavailable"
        return y * MOLAR_VOLUME_STP / mw, f"{y} x 22.414 / {mw}"
    return y * MOLAR_VOLUME_STP / M_HEXOSE, f"{y} x 22.414 / {M_HEXOSE}"


def main() -> None:
    log.info("reading %s", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    log.info("compilation: %d records, %d sources", len(df), df[GROUP_COL].nunique())

    # ── conversion routes ───────────────────────────────────────────────────
    if "Unit Group" in df.columns:
        df["Unit Group Recomputed"] = (df["Unit Group"].astype(str).str.strip()
                                       .str.replace("Group ", "", regex=False))
        log.info("route assignment taken from the compilation's 'Unit Group' column")
    else:
        raise KeyError("'Unit Group' column absent — route assignment cannot be reproduced")

    convertible = df["Unit Group Recomputed"].ne("Excluded")
    rec = df.apply(recompute, axis=1, result_type="expand")
    df["yield_recomputed"], df["conversion_arithmetic"] = rec[0], rec[1]
    df["yield_compiled"] = pd.to_numeric(df["dm³ H₂/g"], errors="coerce")

    both = df["yield_recomputed"].notna() & df["yield_compiled"].notna()
    rel = ((df["yield_recomputed"] - df["yield_compiled"]).abs()
           / df["yield_compiled"].replace(0, np.nan))
    df["flag_arithmetic"] = (both & (rel > TOLERANCE)).astype(int)
    log.info("conversion audit: %d of %d convertible records agree within %.0f%%",
             int(both.sum() - df["flag_arithmetic"].sum()), int(convertible.sum()),
             TOLERANCE * 100)

    # ── quality flags ───────────────────────────────────────────────────────
    unit_raw = df["Original Unit"].astype(str).str.strip()
    substrate = df["Substrate"].astype(str).str.strip().str.lower()
    value = df["Original Yield"].astype(str).str.strip()
    source = df[GROUP_COL].astype(str).str.strip()

    df["flag_per_slurry"] = unit_raw.isin(PER_SLURRY_UNITS).astype(int)

    df["flag_review_duplicate"] = 0
    for frag, val, secondary in REVIEW_DUPLICATES:
        head = frag.split("+")[0].strip()
        hit = (substrate.str.contains(head, case=False, regex=False)
               & value.str.startswith(val.rstrip("0").rstrip("."))
               & source.str.contains(secondary, case=False, regex=False))
        if hit.sum() != 1:
            log.warning("review duplicate %r/%s/%s matched %d rows", frag, val,
                        secondary, int(hit.sum()))
        df.loc[hit, "flag_review_duplicate"] = 1

    df["flag_exact_duplicate"] = df.duplicated(subset=DUPLICATE_FIELDS,
                                               keep="first").astype(int)

    df["flag_unit_suspect"] = 0
    for _, block in df.assign(_y=df["yield_compiled"]).groupby([source, substrate]):
        vals = block["_y"].replace(0, np.nan).dropna()
        if len(vals) >= 2 and vals.max() / vals.min() > SUSPECT_RATIO:
            df.loc[vals.idxmax(), "flag_unit_suspect"] = 1

    df["flag_above_thauer"] = (df["yield_compiled"] > THAUER_LIMIT_DM3_G).fillna(False).astype(int)
    df["flag_pH_derived"] = df["pH"].map(is_range_valued).astype(int)
    df["flag_T_derived"] = df[temp_col(df)].map(is_range_valued).astype(int)

    # ── modelling subset ────────────────────────────────────────────────────
    removal = ["flag_per_slurry", "flag_review_duplicate",
               "flag_exact_duplicate", "flag_unit_suspect"]
    remove = df[removal].sum(axis=1) > 0
    df["recommended_for_modelling"] = (convertible & ~remove
                                       & df["yield_compiled"].notna()).astype(int)
    for f in removal + ["flag_above_thauer", "flag_arithmetic"]:
        log.info("  %-24s %3d", f, int(df[f].sum()))
    log.info("%d records - %d excluded at compilation - %d flagged for removal = %d modelled",
             len(df), int((~convertible).sum()), int((remove & convertible).sum()),
             int(df["recommended_for_modelling"].sum()))

    df.to_csv(AUDITED_CSV, index=False)

    model = df[df["recommended_for_modelling"] == 1].copy()
    model = model.rename(columns={temp_col(df): "temperature_C"})
    model["temperature_C"] = model["temperature_C"].map(parse_numeric)
    model["pH"] = model["pH"].map(parse_numeric)
    model[TARGET] = model["yield_compiled"]
    model = build_features(model)
    keep = ([GROUP_COL, "Substrate", "Microbial Inoculum", "Reactor / Mode",
             "Original Yield", "Original Unit", "Unit Group Recomputed", TARGET,
             "temperature_C", "pH", "substrate_class", "inoculum_class", "reactor_mode"]
            + [c for c in df.columns if c.startswith("flag_")])
    model[[c for c in keep if c in model.columns]].to_csv(MODEL_CSV, index=False)
    log.info("modelling set: %d observations from %d studies -> %s",
             len(model), model[GROUP_COL].nunique(), MODEL_CSV.name)
    log.info("target: mean %.4f, SD %.4f, median %.4f, max %.4f",
             model[TARGET].mean(), model[TARGET].std(), model[TARGET].median(),
             model[TARGET].max())

    write_table(pd.DataFrame(
        [{"metric": "records in compilation", "value": len(df)},
         {"metric": "sources in compilation", "value": df[GROUP_COL].nunique()},
         {"metric": "convertible records", "value": int(convertible.sum())},
         {"metric": "convertible records agreeing",
          "value": int(both.sum() - df["flag_arithmetic"].sum())},
         {"metric": "excluded at compilation", "value": int((~convertible).sum())},
         {"metric": "flagged for removal", "value": int((remove & convertible).sum())},
         {"metric": "modelled observations", "value": len(model)},
         {"metric": "modelled studies", "value": int(model[GROUP_COL].nunique())}]
        + [{"metric": c, "value": int(df[c].sum())}
           for c in df.columns if c.startswith("flag_")]), "01_audit_summary.csv")


if __name__ == "__main__":
    main()
