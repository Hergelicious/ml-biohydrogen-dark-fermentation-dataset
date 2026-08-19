#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
common.py
=========

Shared configuration, constants, categorical mappings, parsing utilities,
and feature-construction functions used by the dark-fermentation analysis
pipeline.

"""

import re
from pathlib import Path

import numpy as np


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================

# Repository root: parent directory of src/
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA / "Full_Data_set.csv"
AUDITED_CSV = DATA / "Full_Data_set_AUDITED.csv"
MODEL_CSV = DATA / "dataset_modelling.csv"


# ============================================================================
# 2. REPRODUCIBILITY SETTINGS
# ============================================================================

SEED = 42
N_RESAMPLES = 40
N_LC_REPEATS = 15
TEST_FRACTION = 0.20
NOISE_SD = 0.05


# ============================================================================
# 3. PHYSICAL CONSTANTS
# ============================================================================

# Molar gas volume at STP, L mol-1
MOLAR_VOLUME = 22.414

# Hydrogen density, kg m-3
H2_DENSITY = 0.0899

# Hydrogen lower heating value, MJ kg-1
H2_LHV = 120.0

# Water heat capacity, kJ kg-1 K-1
CP_WATER = 4.18

# Water density, kg m-3
RHO_WATER = 1000.0


# ============================================================================
# 4. MODEL FEATURES
# ============================================================================

FEATURES = [
    "sub_cat",
    "inoc_cat",
    "react_cat",
    "pHn",
    "Tn",
]

CAT_FEATURES = FEATURES[:3]
NUM_FEATURES = FEATURES[3:]

TARGET = "y"
GROUP = "ref"


# ============================================================================
# 5. SUBSTRATE CATEGORIZATION
# ============================================================================

def cat_sub(s):
    """Map the original substrate description to a standardized category."""

    s = str(s).lower()

    for keyword, category in [
        ("coffee mucilage", "Coffee mucilage"),
        ("spent coffee", "Lignocellulosic"),
        ("food", "Food/kitchen waste"),
        ("kitchen", "Food/kitchen waste"),
        ("vegetab", "Food/kitchen waste"),
        ("fruit", "Food/kitchen waste"),
        ("apple", "Food/kitchen waste"),
        ("melon", "Food/kitchen waste"),
        ("grape", "Food/kitchen waste"),
        ("potato", "Food/kitchen waste"),
        ("rice slurry", "Food/kitchen waste"),
        ("ofmsw", "Municipal solid waste"),
        ("municipal solid", "Municipal solid waste"),
        ("straw", "Lignocellulosic"),
        ("stover", "Lignocellulosic"),
        ("stalk", "Lignocellulosic"),
        ("bagasse", "Lignocellulosic"),
        ("wood", "Lignocellulosic"),
        ("cellul", "Lignocellulosic"),
        ("husk", "Lignocellulosic"),
        ("grass", "Lignocellulosic"),
        ("leaves", "Lignocellulosic"),
        ("sorghum", "Lignocellulosic"),
        ("barley", "Lignocellulosic"),
        ("silphium", "Lignocellulosic"),
        ("switchgrass", "Lignocellulosic"),
        ("alga", "Algal biomass"),
        ("glycerol", "Glycerol/biodiesel"),
        ("biodiesel", "Glycerol/biodiesel"),
        ("whey", "Industrial wastewater"),
        ("cassava", "Industrial wastewater"),
        ("citric", "Industrial wastewater"),
        ("beverage", "Industrial wastewater"),
        ("palm oil", "Industrial wastewater"),
        ("vinasse", "Industrial wastewater"),
        ("synthetic wastewater", "Industrial wastewater"),
        ("cattle wastewater", "Manure/slurry"),
        ("manure", "Manure/slurry"),
        ("slurry", "Manure/slurry"),
        ("pig", "Manure/slurry"),
        ("biosolid", "Sewage sludge"),
        ("sewage", "Sewage sludge"),
        ("starch", "Simple sugars/starch"),
        ("glucose", "Simple sugars/starch"),
        ("xylose", "Simple sugars/starch"),
        ("arabinose", "Simple sugars/starch"),
        ("sucrose", "Simple sugars/starch"),
        ("lactose", "Simple sugars/starch"),
        ("maltose", "Simple sugars/starch"),
        ("cellobiose", "Simple sugars/starch"),
        ("molasses", "Simple sugars/starch"),
        ("organic waste", "Mixed organic waste"),
        ("mixed", "Mixed organic waste"),
    ]:
        if keyword in s:
            return category

    return "Other"


# ============================================================================
# 6. INOCULUM CATEGORIZATION
# ============================================================================

def cat_inoc(s):
    """Map the original inoculum description to a standardized category."""

    s = str(s).lower()

    if "lactobacillus" in s:
        return "Lactobacillus co-culture"

    if "granul" in s:
        return "Granular sludge"

    if "compost" in s:
        return "Compost"

    if "digest" in s:
        return "Digester sludge"

    if "sewage" in s or "wwtp" in s or "activated" in s:
        return "Sewage/WWTP sludge"

    if "sludge" in s:
        return "Anaerobic sludge"

    if "caldicell" in s or "thermo" in s:
        return "Thermophilic pure culture"

    if "clostrid" in s or "c. " in s:
        return "Clostridium spp."

    if (
        "enterobacter" in s
        or "klebsiella" in s
        or "citrobacter" in s
    ):
        return "Enterobacteriaceae"

    if "mixed" in s or "culture" in s or "consort" in s:
        return "Mixed culture"

    return "Other"


# ============================================================================
# 7. REACTOR / MODE CATEGORIZATION
# ============================================================================

def cat_react(s):
    """Map reactor descriptions to standardized reactor and operating mode."""

    s = str(s).lower()

    mode = (
        "Batch"
        if "batch" in s
        else "Semi-continuous"
        if "semi" in s
        else "Continuous"
        if "contin" in s
        else "Unspecified"
    )

    for keyword, reactor in [
        ("uasb", "UASB"),
        ("cstr", "CSTR"),
        ("membrane", "Membrane"),
        ("packed", "Packed bed"),
        ("fluidi", "Fluidised bed"),
        ("leach", "Leaching bed"),
        ("stirred", "Stirred tank"),
        ("asbr", "ASBR"),
        ("bioreactor", "Bioreactor"),
        ("fermenter", "Bioreactor"),
        ("fermentor", "Bioreactor"),
        ("flask", "Flask"),
        ("erlenmeyer", "Flask"),
        ("serum", "Flask"),
        ("tank", "Stirred tank"),
    ]:
        if keyword in s:
            return f"{reactor} / {mode}"

    return f"Generic / {mode}"


# ============================================================================
# 8. NUMERIC PARSING
# ============================================================================

def parse_numeric(s):
    """
    Extract a representative numeric value from a scalar, range,
    tolerance, or text-containing field.
    """

    s = str(s).strip()

    if s.lower() in (
        "na",
        "n/a",
        "",
        "nan",
        "uncontrolled",
    ):
        return np.nan

    s = (
        s.replace("~", "")
        .replace("\u00b1", " ")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

    numbers = re.findall(
        r"\d+\.?\d*",
        s,
    )

    if not numbers:
        return np.nan

    values = [float(x) for x in numbers]

    if (
        len(values) > 1
        and re.search(r"\d\s*[-,]\s*\d", s)
    ):
        return float(np.mean(values[:2]))

    return values[0]


# ============================================================================
# 9. FEATURE CONSTRUCTION
# ============================================================================

def add_features(d):
    """
    Add standardized categorical and numerical modelling features.
    """

    d = d.copy()

    d["sub_cat"] = d["Substrate"].map(cat_sub)

    d["inoc_cat"] = d["Microbial Inoculum"].map(cat_inoc)

    d["react_cat"] = d["Reactor / Mode"].map(cat_react)

    d["mode"] = (
        d["react_cat"]
        .str.split(" / ")
        .str[1]
    )

    if "pH_numeric" in d.columns:
        d["pHn"] = d["pH_numeric"]
        d["Tn"] = d["T_numeric"]

    else:
        d["pHn"] = d["pH"].map(parse_numeric)

        d["Tn"] = d["Temp (°C)"].map(parse_numeric)

    d = d.rename(
        columns={
            "dm3_H2_per_g": "y",
            "Reference": "ref",
        }
    )

    return d
