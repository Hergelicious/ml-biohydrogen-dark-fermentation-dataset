"""
common.py — shared configuration for the dark-fermentation validation pipeline.

Every constant, category mapping, model specification and split protocol used
anywhere in the analysis is defined here exactly once, so that no two scripts
can silently disagree about a hyperparameter, a seed, or a category boundary.

Reference: Hassan, Moustafa & Abdelkader, "How much of the reported accuracy of
machine-learning models for dark-fermentative hydrogen yield survives
study-level validation? Evidence from 224 harmonised observations."
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RAW_CSV = DATA / "Full_Data_set.csv"
AUDITED_CSV = DATA / "Full_Data_set_AUDITED.csv"
MODEL_CSV = DATA / "dataset_modelling.csv"
REFERENCES_CSV = DATA / "references.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 0

# Seeds 0..N-1. These form a resampling distribution over partitions of one fixed
# dataset; they are not N independent validations. Override for a fast smoke test:
#     DF_N_RESAMPLES=5 ./run_all.sh
# All published values use the default of 40.
N_RESAMPLES = int(os.environ.get("DF_N_RESAMPLES", "40"))
TEST_SIZE = 0.2
PCT_LO, PCT_HI = 2.5, 97.5  # empirical percentile interval reported throughout

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants (energy balance, script 04)
# ─────────────────────────────────────────────────────────────────────────────
MOLAR_VOLUME_STP = 22.414   # dm3 mol-1 at 273.15 K, 101.325 kPa
M_HEXOSE = 180.16           # g mol-1, glucose; Group C hexose-equivalent basis
CP_WATER = 4.18             # kJ kg-1 K-1
RHO_WATER = 1000.0          # kg m-3
RHO_H2 = 0.0899             # kg m-3 at STP
LHV_H2 = 120.0              # MJ kg-1
E_H2_PER_DM3 = RHO_H2 * LHV_H2 / 1000.0 * 1000.0  # -> 10.788 kJ dm-3

THAUER_LIMIT_MOL = 4.0                              # mol H2 per mol hexose
THAUER_LIMIT_DM3_G = THAUER_LIMIT_MOL * MOLAR_VOLUME_STP / M_HEXOSE  # 0.4976 dm3 g-1

T_MESO, T_THERMO = 37.0, 55.0
T_INFLUENT = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# Modelling specification
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "dm3_H2_per_g"
GROUP_COL = "Reference"

CATEGORICAL_FEATURES = ["substrate_class", "inoculum_class", "reactor_mode"]
NUMERIC_FEATURES = ["pH", "temperature_C"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Noise augmentation, reproducing the practice audited in the paper
AUG_NOISE_SD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Category mappings
#
# Keyword rules are applied in order; the first match wins. Order matters:
# more specific patterns must precede more general ones.
# ─────────────────────────────────────────────────────────────────────────────
SUBSTRATE_RULES: list[tuple[str, str]] = [
    (r"coffee|mucilage",                                    "coffee mucilage"),
    (r"glycerol|biodiesel",                                 "glycerol / biodiesel waste"),
    (r"algae|algal|microalga|chlorella|spirulina",          "algal biomass"),
    (r"sewage|waste ?activated sludge|\bwas\b|sludge",      "sewage sludge"),
    (r"manure|slurry|piggery|cattle|dairy|swine|poultry",   "manure / slurry"),
    (r"\bmsw\b|municipal solid",                            "municipal solid waste"),
    (r"straw|stover|bagasse|corncob|lignocell|wood|"
     r"grass|husk|stalk|rice hull|cellulos",                "lignocellulosic biomass"),
    (r"whey|molasses|vinasse|brewery|distillery|winery|"
     r"olive mill|palm oil|pome|starch wastewater|"
     r"citric|textile|dairy wastewater|industrial",         "industrial wastewater"),
    (r"food waste|kitchen|canteen|restaurant|fruit|"
     r"vegetable|potato|banana|apple|cafeteria",            "food and kitchen waste"),
    (r"glucose|sucrose|xylose|starch|lactose|maltose|"
     r"fructose|arabinose|cellobiose|galactose|sugar",      "simple sugars and starches"),
    (r"mixed organic|organic fraction|ofmsw|co-?digest",    "mixed organic waste"),
]

INOCULUM_RULES: list[tuple[str, str]] = [
    (r"clostridium",                                        "Clostridium (pure)"),
    (r"enterobacter",                                       "Enterobacter (pure)"),
    (r"thermoanaerobact|caldicellulosiruptor|thermotoga",   "thermophilic pure culture"),
    (r"escherichia|e\. ?coli",                              "Escherichia coli"),
    (r"pure culture|isolate|strain",                        "other pure culture"),
    (r"heat[- ]?treat|heat[- ]?shock|boil",                 "heat-treated mixed culture"),
    (r"acid[- ]?treat|base[- ]?treat|chemical",             "chemically pretreated mixed culture"),
    (r"anaerobic (sludge|granul|digest)|granular",          "anaerobic digester sludge"),
    (r"compost|soil",                                       "compost / soil"),
    (r"cow dung|rumen|manure",                              "rumen / manure"),
    (r"mixed|consorti|indigenous|native|undefined",         "untreated mixed culture"),
]

REACTOR_RULES: list[tuple[str, str]] = [
    (r"\bcstr\b|continuous stirred",                        "CSTR"),
    (r"\bubf\b|upflow|\buasb\b",                            "upflow / UASB"),
    (r"packed bed|fixed bed|immobilis|immobiliz",           "packed / fixed bed"),
    (r"fluidis|fluidiz",                                    "fluidised bed"),
    (r"membrane|\bmbr\b",                                   "membrane reactor"),
    (r"leach|\blbr\b",                                      "leach bed"),
    (r"serum bottle|batch|flask|bottle|reactor batch",      "batch"),
]

MODE_RULES: list[tuple[str, str]] = [
    (r"semi[- ]?continuous|fed[- ]?batch|repeated batch",   "semi-continuous"),
    (r"continuous",                                         "continuous"),
    (r"batch",                                              "batch"),
]

SUBSTRATE_FALLBACK = "other"
INOCULUM_FALLBACK = "other mixed culture"
REACTOR_FALLBACK = "unspecified reactor"
MODE_FALLBACK = "unspecified"

# Coarse three-class substrate grouping, used in the encoding-sensitivity arm
COARSE_SUBSTRATE = {
    "simple sugars and starches": "soluble carbohydrate",
    "coffee mucilage": "soluble carbohydrate",
    "glycerol / biodiesel waste": "soluble carbohydrate",
    "industrial wastewater": "soluble carbohydrate",
    "food and kitchen waste": "complex biodegradable",
    "mixed organic waste": "complex biodegradable",
    "municipal solid waste": "complex biodegradable",
    "algal biomass": "complex biodegradable",
    "manure / slurry": "recalcitrant / low-yield",
    "sewage sludge": "recalcitrant / low-yield",
    "lignocellulosic biomass": "recalcitrant / low-yield",
    "other": "complex biodegradable",
}


def _apply_rules(value: object, rules: list[tuple[str, str]], fallback: str) -> str:
    """Map a free-text literature string onto a category by ordered keyword rules."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    text = str(value).strip().lower()
    if not text or text in {"na", "n/a", "-", "nan"}:
        return fallback
    for pattern, label in rules:
        if re.search(pattern, text):
            return label
    return fallback


def classify_substrate(v) -> str:
    return _apply_rules(v, SUBSTRATE_RULES, SUBSTRATE_FALLBACK)


def classify_inoculum(v) -> str:
    return _apply_rules(v, INOCULUM_RULES, INOCULUM_FALLBACK)


def classify_reactor(v) -> str:
    return _apply_rules(v, REACTOR_RULES, REACTOR_FALLBACK)


def classify_mode(v) -> str:
    return _apply_rules(v, MODE_RULES, MODE_FALLBACK)


# ─────────────────────────────────────────────────────────────────────────────
# Column-name handling (Supplementary Note S27)
#
# The audited dataset carries 'Temp (°C)' and 'pH'; the modelling file carries
# pre-parsed numeric columns. Formula interfaces cannot accept a column name
# containing a degree sign or brackets, so both conventions are resolved here.
# ─────────────────────────────────────────────────────────────────────────────
_ALIASES = {
    "temperature_C": ["temperature_C", "Tn", "Temp (°C)", "Temp (C)", "Temperature",
                      "temperature", "Temp"],
    "pH": ["pH", "pHn", "pH value", "ph"],
    TARGET: [TARGET, "dm³ H₂/g", "dm3 H2/g", "Harmonised Yield", "harmonised_yield"],
    GROUP_COL: [GROUP_COL, "reference", "Source", "Study", "study"],
}


def resolve_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever alias is present to the canonical name; parse ranges."""
    out = df.copy()
    for canonical, aliases in _ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out = out.rename(columns={alias: canonical})
                break
        else:
            raise KeyError(
                f"None of the accepted column names for {canonical!r} were found. "
                f"Looked for: {aliases}. Present: {list(out.columns)}"
            )
    for col in ("temperature_C", "pH", TARGET):
        out[col] = out[col].map(parse_numeric)
    return out


def parse_numeric(value) -> float:
    """
    Parse a literature-reported numeric entry.

    Handles plain numbers, ranges ('35-37', '5.5 to 6.0'), tolerance notation
    ('37 ± 1'), and inequality-prefixed values ('<0.01'). Ranges are reduced to
    their midpoint and flagged upstream by 01_audit_dataset.py.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text or text.lower() in {"na", "n/a", "-", "nan", "nr", "not reported"}:
        return np.nan
    text = re.sub(r"[<>~≈]", "", text)
    tol = re.match(r"^\s*([\d.]+)\s*(?:±|\+/-)\s*[\d.]+\s*$", text)
    if tol:
        return float(tol.group(1))
    rng = re.match(r"^\s*([\d.]+)\s*(?:-|–|—|to)\s*([\d.]+)\s*$", text)
    if rng:
        return (float(rng.group(1)) + float(rng.group(2))) / 2.0
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[0]) if nums else np.nan


def is_range_valued(value) -> bool:
    """True when a raw entry encoded a range or a tolerance rather than a point value."""
    if value is None or isinstance(value, (int, float, np.number)):
        return False
    text = str(value)
    return bool(re.search(r"(?:-|–|—|to|±|\+/-)", text)) and bool(re.search(r"\d", text))


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, substrate_encoding: str = "mapped") -> pd.DataFrame:
    """
    Attach the five modelling predictors.

    substrate_encoding
        'mapped' : the 12-class keyword mapping used throughout the paper
        'raw'    : the 69 raw literature strings, unmapped
        'coarse' : the three-class grouping
    """
    out = df.copy()
    out["substrate_class"] = out["Substrate"].map(classify_substrate)
    out["inoculum_class"] = out["Microbial Inoculum"].map(classify_inoculum)
    reactor = out["Reactor / Mode"].map(classify_reactor)
    mode = out["Reactor / Mode"].map(classify_mode)
    out["reactor_mode"] = reactor.str.cat(mode, sep=" / ")

    if substrate_encoding == "raw":
        out["substrate_class"] = out["Substrate"].astype(str).str.strip().str.lower()
    elif substrate_encoding == "coarse":
        out["substrate_class"] = out["substrate_class"].map(COARSE_SUBSTRATE).fillna(
            "complex biodegradable")
    elif substrate_encoding != "mapped":
        raise ValueError(f"unknown substrate_encoding: {substrate_encoding!r}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────
def summarise(scores) -> dict:
    """Median and empirical percentile interval of a resampling distribution."""
    arr = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    if arr.size == 0:
        return {"median": np.nan, "pct_lo": np.nan, "pct_hi": np.nan, "n_resamples": 0}
    return {
        "median": float(np.median(arr)),
        "pct_lo": float(np.percentile(arr, PCT_LO)),
        "pct_hi": float(np.percentile(arr, PCT_HI)),
        "n_resamples": int(arr.size),
    }


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    df.to_csv(path, index=False)
    logging.info("wrote %s (%d rows)", path.name, len(df))
    return path


def get_logger(name: str) -> logging.Logger:
    """Log to stdout and to results/<name>.log simultaneously."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(RESULTS / f"{name}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_modelling_set(encoding: str = "mapped") -> pd.DataFrame:
    """Load the 224-observation modelling set with features attached."""
    if not MODEL_CSV.exists():
        raise FileNotFoundError(
            f"{MODEL_CSV} not found — run scripts/01_audit_dataset.py first.")
    df = pd.read_csv(MODEL_CSV)
    df = resolve_numeric_columns(df)
    if "substrate_class" not in df.columns or encoding != "mapped":
        df = build_features(df, substrate_encoding=encoding)
    return df
