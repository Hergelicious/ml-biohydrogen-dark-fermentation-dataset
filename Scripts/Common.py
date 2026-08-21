#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py -- shared configuration, category mappings and feature construction.

Every script in this repository imports from here, so the category definitions
that produced the published numbers exist in exactly one place.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RAW_CSV = DATA / "Full_Data_set.csv"                 # compilation as submitted
AUDITED_CSV = DATA / "Full_Data_set_AUDITED.csv"     # written by 01_audit_dataset.py
MODEL_CSV = DATA / "dataset_modelling_224.csv"       # written by 01_audit_dataset.py

SEED = 42
N_RESAMPLES = 40          # resamples per validation protocol (Table 1, Fig. 1)
N_LC_REPEATS = 15         # repeats per point on the learning curve
TEST_FRACTION = 0.20
NOISE_SD = 0.05           # sigma of the Gaussian augmentation being audited

MOLAR_VOLUME = 22.414     # dm3 mol-1 at STP (273.15 K, 101.325 kPa)
H2_DENSITY = 0.0899       # kg m-3 at STP
H2_LHV = 120.0            # MJ kg-1
CP_WATER = 4.18           # kJ kg-1 K-1
RHO_WATER = 1000.0        # kg m-3

FEATURES = ["sub_cat", "inoc_cat", "react_cat", "pHn", "Tn"]
CAT_FEATURES = FEATURES[:3]
NUM_FEATURES = FEATURES[3:]
TARGET = "y"              # dm3 H2 per g substrate
GROUP = "ref"             # source study -- the grouping variable


# --------------------------------------------------------------------------
# Category mappings.  First matching keyword wins; order is therefore
# significant (e.g. "spent coffee" is matched before "coffee mucilage").
# --------------------------------------------------------------------------
def cat_sub(s):
    s = str(s).lower()
    for k, v in [
        ('coffee mucilage', 'Coffee mucilage'), ('spent coffee', 'Lignocellulosic'),
        ('food', 'Food/kitchen waste'), ('kitchen', 'Food/kitchen waste'), ('vegetab', 'Food/kitchen waste'),
        ('fruit', 'Food/kitchen waste'), ('apple', 'Food/kitchen waste'), ('melon', 'Food/kitchen waste'),
        ('grape', 'Food/kitchen waste'), ('potato', 'Food/kitchen waste'), ('rice slurry', 'Food/kitchen waste'),
        ('ofmsw', 'Municipal solid waste'), ('municipal solid', 'Municipal solid waste'),
        ('straw', 'Lignocellulosic'), ('stover', 'Lignocellulosic'), ('stalk', 'Lignocellulosic'),
        ('bagasse', 'Lignocellulosic'), ('wood', 'Lignocellulosic'), ('cellul', 'Lignocellulosic'),
        ('husk', 'Lignocellulosic'), ('grass', 'Lignocellulosic'), ('leaves', 'Lignocellulosic'),
        ('sorghum', 'Lignocellulosic'), ('barley', 'Lignocellulosic'), ('silphium', 'Lignocellulosic'),
        ('switchgrass', 'Lignocellulosic'), ('alga', 'Algal biomass'), ('glycerol', 'Glycerol/biodiesel'),
        ('biodiesel', 'Glycerol/biodiesel'), ('whey', 'Industrial wastewater'), ('cassava', 'Industrial wastewater'),
        ('citric', 'Industrial wastewater'), ('beverage', 'Industrial wastewater'), ('palm oil', 'Industrial wastewater'),
        ('vinasse', 'Industrial wastewater'), ('synthetic wastewater', 'Industrial wastewater'),
        ('cattle wastewater', 'Manure/slurry'), ('manure', 'Manure/slurry'), ('slurry', 'Manure/slurry'),
        ('pig', 'Manure/slurry'), ('biosolid', 'Sewage sludge'), ('sewage', 'Sewage sludge'),
        ('starch', 'Simple sugars/starch'), ('glucose', 'Simple sugars/starch'), ('xylose', 'Simple sugars/starch'),
        ('arabinose', 'Simple sugars/starch'), ('sucrose', 'Simple sugars/starch'), ('lactose', 'Simple sugars/starch'),
        ('maltose', 'Simple sugars/starch'), ('cellobiose', 'Simple sugars/starch'), ('molasses', 'Simple sugars/starch'),
        ('organic waste', 'Mixed organic waste'), ('mixed', 'Mixed organic waste')
    ]:
        if k in s:
            return v
    return 'Other'


def cat_inoc(s):
    s = str(s).lower()
    if 'lactobacillus' in s:
        return 'Lactobacillus co-culture'
    if 'granul' in s:
        return 'Granular sludge'
    if 'compost' in s:
        return 'Compost'
    if 'digest' in s:
        return 'Digester sludge'
    if 'sewage' in s or 'wwtp' in s or 'activated' in s:
        return 'Sewage/WWTP sludge'
    if 'sludge' in s:
        return 'Anaerobic sludge'
    if 'caldicell' in s or 'thermo' in s:
        return 'Thermophilic pure culture'
    if 'clostrid' in s or 'c. ' in s:
        return 'Clostridium spp.'
    if 'enterobacter' in s or 'klebsiella' in s or 'citrobacter' in s:
        return 'Enterobacteriaceae'
    if 'mixed' in s or 'culture' in s or 'consort' in s:
        return 'Mixed culture'
    return 'Other'


def cat_react(s):
    s = str(s).lower()
    mode = 'Batch' if 'batch' in s else ('Semi-continuous' if 'semi' in s else ('Continuous' if 'contin' in s else 'Unspecified'))
    for k, v in [
        ('uasb', 'UASB'), ('cstr', 'CSTR'), ('membrane', 'Membrane'), ('packed', 'Packed bed'),
        ('fluidi', 'Fluidised bed'), ('leach', 'Leaching bed'), ('stirred', 'Stirred tank'), ('asbr', 'ASBR'),
        ('bioreactor', 'Bioreactor'), ('fermenter', 'Bioreactor'), ('fermentor', 'Bioreactor'),
        ('flask', 'Flask'), ('erlenmeyer', 'Flask'), ('serum', 'Flask'), ('tank', 'Stirred tank')
    ]:
        if k in s:
            return v + ' / ' + mode
    return 'Generic / ' + mode


# --------------------------------------------------------------------------
# Helper functions for data parsing and matrix generation.
# --------------------------------------------------------------------------
def parse_numeric(s):
    """Parse pH / temperature cells.

    Returns the midpoint for a range ("35-45", "50, 70"), strips tolerance and
    approximation marks ("35 +- 1", "~6.8"), and returns NaN for "NA" or
    "uncontrolled". Records affected by this parsing are flagged in the
    audited dataset (flag_pH_derived / flag_T_derived).
    """
    s = str(s).strip()
    if s.lower() in ("na", "n/a", "", "nan", "uncontrolled"):
        return np.nan
    s = s.replace("~", "").replace("\u00b1", " ").replace("\u2013", "-").replace("\u2014", "-")
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return np.nan
    vals = [float(x) for x in nums]
    if len(vals) > 1 and re.search(r"\d\s*[-,]\s*\d", s):
        return float(np.mean(vals[:2]))
    return vals[0]


def load_modelling_data(path=None):
    """Load the modelling set and attach categories and numeric predictors."""
    path = Path(path) if path else MODEL_CSV
    d = pd.read_csv(path)
    if "sub_cat" not in d.columns:
        d = add_features(d)
    return d


def add_features(d):
    """Attach sub_cat / inoc_cat / react_cat / mode / pHn / Tn to an audited frame."""
    d = d.copy()
    d["sub_cat"] = d["Substrate"].map(cat_sub)
    d["inoc_cat"] = d["Microbial Inoculum"].map(cat_inoc)
    d["react_cat"] = d["Reactor / Mode"].map(cat_react)
    d["mode"] = d["react_cat"].str.split(" / ").str[1]
    if "pH_numeric" in d.columns:
        d["pHn"] = d["pH_numeric"]
        d["Tn"] = d["T_numeric"]
    else:
        d["pHn"] = d["pH"].map(parse_numeric)
        d["Tn"] = d["Temp (\u00b0C)"].map(parse_numeric)
    d = d.rename(columns={"dm3_H2_per_g": "y", "Reference": "ref"})
    return d


def design_matrix(d):
    """Return (X, y, groups). Numeric predictors are median-filled here only
    for the two records with no reported value; inside the modelling scripts
    imputation is refitted within each training fold."""
    X = d[FEATURES].copy()
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(d[c].median())
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    return X.reset_index(drop=True), d[TARGET].to_numpy(), d[GROUP].to_numpy()


def variance_components(d, target=TARGET, group=GROUP):
    """Direct decomposition of the target into within- and between-study parts."""
    g = d.groupby(group)[target]
    within = float(np.average(g.var(ddof=0).fillna(0), weights=g.size()))
    between = float(g.mean().var(ddof=0))
    return within, between, between / (between + within)


def summarise(v):
    """Median and 2.5-97.5 percentile interval of a list of scores."""
    v = np.asarray(v, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def fmt(v):
    m, lo, hi = summarise(v)
    return "%+.3f (%+.3f to %+.3f)" % (m, lo, hi)


# --------------------------------------------------------------------------
# Column-name resolution. The audited file carries "Temp (\u00b0C)" and
# "pH", the modelling file carries the pre-parsed "Tn"/"pHn", and some
# intermediate exports use "T_numeric"/"pH_numeric". Patsy formulas cannot
# take a name containing "\u00b0" or brackets, so every script resolves through
# here and works on the safe aliases Tn / pHn only.
# --------------------------------------------------------------------------
TEMP_ALIASES = ["Tn", "T_numeric", "Temp (\u00b0C)", "Temp (C)", "temperature", "temp", "T"]
PH_ALIASES = ["pHn", "pH_numeric", "pH", "ph"]


def resolve_numeric_columns(d):
    """Guarantee that `d` has numeric `Tn` and `pHn`, whatever the source file.

    Returns a copy; never mutates the caller's frame. Raises a clear error
    naming the columns actually present if neither temperature nor pH can be
    found, instead of a bare KeyError from downstream code.
    """
    d = d.copy()
    for target, aliases in (("Tn", TEMP_ALIASES), ("pHn", PH_ALIASES)):
        if target in d.columns and pd.api.types.is_numeric_dtype(d[target]):
            continue
        for name in aliases:
            if name in d.columns:
                d[target] = (
                    d[name].map(parse_numeric)
                    if not pd.api.types.is_numeric_dtype(d[name])
                    else pd.to_numeric(d[name], errors="coerce")
                )
                break
        else:
            raise KeyError(
                "no column found for %s; tried %s. Columns present: %s"
                % (target, aliases, sorted(d.columns))
            )
    return d


def basis_of(unit, unit_group):
    """Mass basis of the denominator, inferred from the original unit string."""
    u = str(unit).lower()
    if "cod" in u:
        return "g COD"
    if "dry biomass" in u:
        return "g dry biomass (TS)"
    if "mol" in u:
        return "g hexose-equivalent" if str(unit_group).strip().endswith("C") else "g named substrate"
    if "vss" in u or "vs" in u:
        return "g volatile solids"
    return "g substrate (unstated)"


def add_basis(d):
    """Attach a `basis` column classifying each record's denominator."""
    d = d.copy()
    d["basis"] = [basis_of(u, g) for u, g in zip(d["Original Unit"], d["Unit Group"])]
    return d


NATIVE_VS = "g volatile solids"
