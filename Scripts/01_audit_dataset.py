#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_audit_dataset.py
===================

Audit and harmonize the dark-fermentation hydrogen-yield dataset.

This script:

1. Reads the original compilation from data/Full_Data_set.csv.
2. Independently recomputes hydrogen-yield unit conversions.
3. Identifies discrepancies between reported and recomputed values.
4. Applies predefined data-quality flags.
5. Parses pH and temperature values.
6. Identifies records recommended for machine-learning modelling.
7. Constructs modelling features using functions defined in common.py.
8. Writes the audited dataset, modelling dataset, and audit summary.

Outputs
-------
data/Full_Data_set_AUDITED.csv
    Complete audited dataset with conversion and quality-control flags.

data/dataset_modelling.csv
    Records retained for machine-learning modelling.

results/01_audit_summary.csv
    Summary of all audit flags and the number of affected records.

Execution order
---------------
Run this script before the downstream modelling and analysis scripts.

Example
-------
From the repository root:

    python scripts/01_audit_dataset.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    MOLAR_VOLUME,
    add_features,
    parse_numeric,
)


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

# Repository root
ROOT = Path(__file__).resolve().parents[1]

# Input/output directories
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Input file
RAW_CSV = DATA_DIR / "Full_Data_set.csv"

# Output files
AUDITED_CSV = DATA_DIR / "Full_Data_set_AUDITED.csv"
MODEL_CSV = DATA_DIR / "dataset_modelling.csv"
SUMMARY_CSV = RESULTS_DIR / "01_audit_summary.csv"

# Create output directories if necessary
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. INPUT VALIDATION
# ============================================================================

print("=" * 70)
print("DARK-FERMENTATION DATASET AUDIT")
print("=" * 70)

print("\nRepository root:")
print(ROOT)

print("\nInput dataset:")
print(RAW_CSV)

if not RAW_CSV.exists():
    raise FileNotFoundError(
        "\nFull_Data_set.csv was not found.\n"
        f"Expected location:\n{RAW_CSV}\n\n"
        "Please place the raw dataset in the repository data/ directory."
    )

print("\nInput file found successfully.")


# ============================================================================
# 3. EXPECTED DATASET COLUMNS
# ============================================================================

COLS = [
    "sub",
    "inoc",
    "reactor",
    "pH",
    "T",
    "y_orig",
    "unit",
    "y_dm3",
    "note",
    "group",
    "ref",
]


# ============================================================================
# 4. MOLECULAR WEIGHTS
# ============================================================================

MW = {
    "glucose": 180.16,
    "hexose": 180.16,
    "c6": 180.16,
    "sucrose": 342.30,
    "lactose": 342.30,
    "cellobiose": 342.30,
    "maltose": 342.30,
    "glycerol": 92.09,
    "c5": 150.13,
    "pentose": 150.13,
    "xylose": 150.13,
}


# ============================================================================
# 5. INDEPENDENT UNIT-CONVERSION AUDIT
# ============================================================================

def expected_conversion(row):
    """
    Independently recompute the harmonized hydrogen yield.

    Parameters
    ----------
    row : pandas.Series
        Dataset record containing the original yield, original unit,
        substrate information, and conversion notes.

    Returns
    -------
    float
        Recomputed yield in dm3 H2 g-1 on the harmonized basis.
        Returns NaN when the conversion cannot be determined.
    """

    unit = str(row.unit).lower()
    y = row.y_orig

    # Missing original yield
    if not np.isfinite(y):
        return np.nan

    # ------------------------------------------------------------------------
    # Group A: volumetric hydrogen yield per unit mass
    # ------------------------------------------------------------------------

    # Already expressed as dm3 H2 g-1
    if re.search(r"dm³ h₂/g|dm3 h2/g", unit):
        return y

    # mL H2 g-1 or cm3 H2 g-1
    if "ml" in unit or "cm³" in unit or "cm3" in unit:
        return y / 1000.0

    # L H2 kg-1
    if "l h₂/kg" in unit or "l h2/kg" in unit:
        return y / 1000.0

    # mmol H2 g-1
    if "mmol" in unit:
        return y * MOLAR_VOLUME / 1000.0

    # dm3 H2 dm-3 slurry
    if "dm³ h₂/dm³" in unit:
        return y / 1000.0

    # ------------------------------------------------------------------------
    # Groups B/C: molar hydrogen yields
    # ------------------------------------------------------------------------

    if "mol" in unit:

        # First use an explicit molecular-weight divisor recorded
        # in the conversion note.
        match = re.search(
            r"÷\s*([\d.]+)",
            str(row.note),
        )

        mw = float(match.group(1)) if match else None

        # If no divisor is explicitly stated, infer the substrate basis.
        if mw is None:
            for substrate, molecular_weight in MW.items():
                if substrate in unit:
                    mw = molecular_weight
                    break

        # Default to hexose-equivalent molecular weight.
        if mw is None:
            mw = 180.16

        return y * MOLAR_VOLUME / mw

    # ------------------------------------------------------------------------
    # Unknown unit
    # ------------------------------------------------------------------------

    return np.nan


# ============================================================================
# 6. READ DATASET
# ============================================================================

print("\n" + "=" * 70)
print("READING DATASET")
print("=" * 70)

d = pd.read_csv(
    RAW_CSV,
    encoding="utf-8",
)

print("\nRows loaded:", len(d))
print("Columns found:", len(d.columns))

print("\nOriginal columns:")
print(list(d.columns))


# ============================================================================
# 7. VERIFY DATASET STRUCTURE
# ============================================================================

if len(d.columns) != len(COLS):
    raise ValueError(
        "\nUnexpected dataset structure.\n"
        f"Expected {len(COLS)} columns, "
        f"but found {len(d.columns)}.\n\n"
        f"Columns found:\n{list(d.columns)}"
    )

d.columns = COLS


# ============================================================================
# 8. CLEAN TEXT COLUMNS
# ============================================================================

TEXT_COLUMNS = [
    "sub",
    "inoc",
    "reactor",
    "unit",
    "note",
    "group",
    "ref",
]

for column in TEXT_COLUMNS:
    d[column] = (
        d[column]
        .astype(str)
        .str.strip()
    )


# ============================================================================
# 9. READ ORIGINAL HARMONIZED YIELD
# ============================================================================

d["y"] = pd.to_numeric(
    d["y_dm3"]
    .astype(str)
    .str.strip(),
    errors="coerce",
)


# ============================================================================
# 10. IDENTIFY EXPLICITLY EXCLUDED RECORDS
# ============================================================================

d["explicitly_excluded"] = (
    d["y_dm3"]
    .astype(str)
    .str.upper()
    .str.contains("EXCLUD")
    |
    d["group"].str.contains(
        "Exclud",
        case=False,
        na=False,
    )
)


# ============================================================================
# 11. INDEPENDENT CONVERSION AUDIT
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1 — CONVERSION AUDIT")
print("=" * 70)

print("\nRecomputing harmonized yield for every record...")

d["y_recomputed"] = d.apply(
    expected_conversion,
    axis=1,
)

# Records where both values are available
ok = (
    d["y"].notna()
    &
    d["y_recomputed"].notna()
)

# Relative difference
relative_difference = (
    (d["y"] - d["y_recomputed"]).abs()
    /
    d["y_recomputed"].replace(0, np.nan)
)

# Flag discrepancies greater than 2%
d["flag_arithmetic"] = (
    ok
    &
    (relative_difference > 0.02)
)

print(
    "\nArithmetic discrepancies (>2%):",
    int(d["flag_arithmetic"].sum()),
)

# Replace discrepant harmonized values with independently
# recomputed values.
d.loc[d["flag_arithmetic"], "y"] = (
    d.loc[d["flag_arithmetic"], "y_recomputed"]
)


# ============================================================================
# 12. DATA-QUALITY FLAGS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2 — QUALITY FLAGS")
print("=" * 70)


# ----------------------------------------------------------------------------
# Flag 1: suspicious unit
# ----------------------------------------------------------------------------

d["flag_unit_suspect"] = (
    d["ref"].str.contains(
        "Chen et al., 2012",
        na=False,
    )
    &
    (d["y_orig"] == 0.0248)
)

print(
    "flag_unit_suspect:",
    int(d["flag_unit_suspect"].sum()),
)


# ----------------------------------------------------------------------------
# Flag 2: yield reported per slurry volume
# ----------------------------------------------------------------------------

d["flag_per_slurry"] = d["ref"].str.contains(
    "Cárdenas et al., 2019",
    na=False,
)

print(
    "flag_per_slurry:",
    int(d["flag_per_slurry"].sum()),
)


# ----------------------------------------------------------------------------
# Flag 3: records duplicated in review articles
# ----------------------------------------------------------------------------

REVIEWS = [
    "Ghimire et al., 2015",
    "Łukajtis et al., 2018",
    "Akhbari & Ibrahim, 2022",
    "Okonkwo, 2019",
]

DUPLICATED = {
    ("Cheese whey", 111.0),
    ("Corn stalk waste", 149.69),
    ("Food waste + sewage sludge", 122.9),
    ("Food waste", 46.3),
    ("OFMSW", 360.0),
    ("OFMSW", 99.0),
    ("Rice slurry", 346.0),
    ("Wheat straw", 68.1),
    ("L-Arabinose", 1.5),
    ("Sucrose", 2.35),
    ("Glucose", 2.2),
}

d["flag_review_duplicate"] = d.apply(
    lambda row: (
        (row["sub"], round(row["y_orig"], 2)) in DUPLICATED
        and row["ref"] in REVIEWS
    ),
    axis=1,
)

print(
    "flag_review_duplicate:",
    int(d["flag_review_duplicate"].sum()),
)


# ----------------------------------------------------------------------------
# Flag 4: molecular-weight basis
# ----------------------------------------------------------------------------

d["flag_mw_basis"] = (
    d["sub"].str.contains(
        "ylose",
        na=False,
    )
    &
    d["note"].str.contains(
        "180.16",
        na=False,
    )
)

print(
    "flag_mw_basis:",
    int(d["flag_mw_basis"].sum()),
)


# ----------------------------------------------------------------------------
# Flag 5: yield above Thauer theoretical limit
# ----------------------------------------------------------------------------

d["flag_above_thauer"] = (
    d["y"]
    >
    4 * MOLAR_VOLUME / 180.16
)

print(
    "flag_above_thauer:",
    int(d["flag_above_thauer"].sum()),
)


# ----------------------------------------------------------------------------
# Flag 6: exact duplicates
# ----------------------------------------------------------------------------

d["flag_exact_duplicate"] = d.duplicated(
    subset=[
        "sub",
        "inoc",
        "reactor",
        "pH",
        "T",
        "y_orig",
        "ref",
    ],
    keep="first",
)

print(
    "flag_exact_duplicate:",
    int(d["flag_exact_duplicate"].sum()),
)


# ============================================================================
# 13. pH AND TEMPERATURE PROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3 — pH / TEMPERATURE PROCESSING")
print("=" * 70)

d["pH_numeric"] = d["pH"].map(parse_numeric)
d["T_numeric"] = d["T"].map(parse_numeric)

# Identify values requiring parsing
d["flag_pH_derived"] = d["pH"].astype(str).str.contains(
    r"[a-zA-Z~±]|\d\s*[-–]\s*\d",
    regex=True,
    na=False,
)

d["flag_T_derived"] = d["T"].astype(str).str.contains(
    r"[a-zA-Z~±,]|\d\s*[-–]\s*\d",
    regex=True,
    na=False,
)

print(
    "pH derived:",
    int(d["flag_pH_derived"].sum()),
)

print(
    "Temperature derived:",
    int(d["flag_T_derived"].sum()),
)


# ============================================================================
# 14. DETERMINE MODELLING DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4 — MODELLING DATASET")
print("=" * 70)

d["recommended_for_modelling"] = ~(
    d["flag_unit_suspect"]
    |
    d["flag_per_slurry"]
    |
    d["flag_review_duplicate"]
    |
    d["flag_exact_duplicate"]
    |
    d["y"].isna()
)


# ============================================================================
# 15. PREPARE OUTPUT DATASET
# ============================================================================

out = d.rename(
    columns={
        "sub": "Substrate",
        "inoc": "Microbial Inoculum",
        "reactor": "Reactor / Mode",
        "T": "Temp (°C)",
        "y_orig": "Original Yield",
        "unit": "Original Unit",
        "y": "dm3_H2_per_g",
        "note": "Conversion Calculation & Notes",
        "group": "Unit Group",
        "ref": "Reference",
    }
)

# Remove the original harmonized-yield column.
out = out.drop(
    columns=["y_dm3"],
)


# ============================================================================
# 16. SAVE AUDITED DATASET
# ============================================================================

out.to_csv(
    AUDITED_CSV,
    index=False,
)

print("\nAudited dataset saved:")
print(AUDITED_CSV)


# ============================================================================
# 17. CREATE MODELLING DATASET
# ============================================================================

model_input = out[
    out["recommended_for_modelling"]
].copy()

model = add_features(
    model_input,
)

model.to_csv(
    MODEL_CSV,
    index=False,
)

print("\nModelling dataset saved:")
print(MODEL_CSV)


# ============================================================================
# 18. CREATE AUDIT SUMMARY
# ============================================================================

summary = pd.DataFrame(
    [
        (
            "arithmetic",
            "harmonized value disagrees with stated conversion by >2%",
            int(d["flag_arithmetic"].sum()),
        ),
        (
            "flag",
            "value duplicates adjacent record with factor-1000 unit discrepancy",
            int(d["flag_unit_suspect"].sum()),
        ),
        (
            "flag",
            "yield expressed per volume of slurry",
            int(d["flag_per_slurry"].sum()),
        ),
        (
            "flag",
            "observation already present under primary study",
            int(d["flag_review_duplicate"].sum()),
        ),
        (
            "flag",
            "xylose converted on C6 basis",
            int(d["flag_mw_basis"].sum()),
        ),
        (
            "flag",
            "yield exceeds 4 mol H2 per mol hexose equivalent",
            int(d["flag_above_thauer"].sum()),
        ),
        (
            "flag",
            "exact duplicate within same source",
            int(d["flag_exact_duplicate"].sum()),
        ),
        (
            "flag",
            "pH or temperature parsed from range/tolerance/text",
            int(
                (
                    d["flag_pH_derived"]
                    |
                    d["flag_T_derived"]
                ).sum()
            ),
        ),
    ],
    columns=[
        "type",
        "issue",
        "records",
    ],
)

summary.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================================
# 19. FINAL RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("FINAL AUDIT RESULTS")
print("=" * 70)

print(
    "\nTotal records:",
    len(d),
)

print(
    "Explicitly excluded:",
    int(d["explicitly_excluded"].sum()),
)

print(
    "Convertible records:",
    int(d["y"].notna().sum()),
)

print(
    "Conversions independently verified:",
    int(
        ok.sum()
        -
        d["flag_arithmetic"].sum()
    ),
    "of",
    int(ok.sum()),
)

print(
    "\nRecommended for modelling:",
    len(model),
)

print(
    "Number of source studies:",
    model["Reference"].nunique(),
)


# ============================================================================
# 20. PRINT SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 70)
print("AUDIT SUMMARY TABLE")
print("=" * 70)

print(
    summary.to_string(index=False),
)


# ============================================================================
# 21. OUTPUT FILES
# ============================================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("\n1.", AUDITED_CSV)
print("2.", MODEL_CSV)
print("3.", SUMMARY_CSV)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
