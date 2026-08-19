#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_energy_balance.py

Energy-balance and break-even analysis linking the Step 03
mixed-effects temperature association to reactor heating demand.

The Step 03 meta-regression models:

    log(1 + y) = beta_T * T + other moderators + study random effect

where y is hydrogen yield [dm3 H2 g^-1 VS].

Because beta_T is estimated on the log1p(y) scale, it is not
interpreted as a raw hydrogen-yield increase per °C.

For a 37 -> 55 °C temperature shift, the model-implied raw-scale
yield is calculated as:

    y_55 = exp(log(1 + y_37) + beta_T * (55 - 37)) - 1

and:

    Delta_y = y_55 - y_37

The resulting yield difference is a model-derived association,
not a causal experimental temperature effect.

The raw-scale yield difference is evaluated at the median yield
of the modelling dataset. Because the log1p back-transformation
is nonlinear, the resulting uplift is baseline-dependent.

Inputs
------
    results/03_metaregression.csv
    data/dataset_modelling_224.csv

Outputs
-------
    results/04_energy_balance.csv
    results/04_breakeven_uplift.csv
    results/04_breakeven_loading.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
RESULTS = BASE_DIR / "results"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_CSV = DATA_DIR / "dataset_modelling_224.csv"


# ============================================================
# CONSTANTS
# ============================================================

# ------------------------------------------------------------
# Water properties
# ------------------------------------------------------------

# kJ kg^-1 K^-1
CP_WATER = 4.18

# kg m^-3
RHO_WATER = 1000.0


# ------------------------------------------------------------
# Hydrogen properties
# ------------------------------------------------------------

# kg H2 m^-3
H2_DENSITY = 0.0899

# MJ kg^-1 H2
H2_LHV = 120.0

# Hydrogen energy per dm3:
#
# 0.0899 kg H2 m^-3 × 120 MJ kg^-1
# = 10.788 MJ m^-3
#
# Therefore:
#
# 1 dm3 H2 = 0.010788 MJ
#           = 10.788 kJ

KJ_PER_DM3 = (
    H2_DENSITY
    * H2_LHV
)


# ------------------------------------------------------------
# Temperature points
# ------------------------------------------------------------

T_INFLUENT = 20.0
T_MESO = 37.0
T_THERMO = 55.0


# ------------------------------------------------------------
# Heat-recovery scenarios
# ------------------------------------------------------------

ETAS = [
    0.0,
    0.5,
    0.7,
    0.85
]


# ------------------------------------------------------------
# Solids-loading scenarios
#
# g VS L^-1
# ------------------------------------------------------------

LOADINGS = [
    10,
    20,
    30,
    50,
    100,
    150
]


# ------------------------------------------------------------
# Typical chemical energy content of VS
#
# MJ kg^-1 VS
# ------------------------------------------------------------

VS_CALORIFIC = 15.6


# ============================================================
# ENERGY FUNCTIONS
# ============================================================

def h2_energy(Y, S):
    """
    Calculate hydrogen energy produced per m3 of feed.

    Parameters
    ----------
    Y : float
        Hydrogen yield [dm3 H2 g^-1 VS].

    S : float
        Solids loading [g VS L^-1].

    Returns
    -------
    float
        Hydrogen energy [MJ m^-3 feed].

    Unit conversion:

        Y [dm3 H2 g^-1 VS]
        × S [g VS L^-1]
        × 1000 [L m^-3]
        × KJ_PER_DM3 [kJ dm^-3]
        / 1000 [kJ MJ^-1]

    Therefore:

        E_H2 = Y × S × KJ_PER_DM3

    in MJ m^-3.
    """

    return (
        Y
        * S
        * KJ_PER_DM3
    )


def heat_duty(dT, eta):
    """
    Calculate net heating duty per m3 of feed.

    Parameters
    ----------
    dT : float
        Temperature increase [°C].

    eta : float
        Heat-recovery efficiency [0–1].

    Returns
    -------
    float
        Net heating duty [MJ m^-3].
    """

    return (
        CP_WATER
        * RHO_WATER
        * dT
        / 1000.0
        * (1.0 - eta)
    )


def breakeven_uplift(dT, eta, S):
    """
    Calculate the raw-scale hydrogen-yield increase required
    to offset the heating duty.

    Parameters
    ----------
    dT : float
        Temperature increase [°C].

    eta : float
        Heat-recovery efficiency [0–1].

    S : float
        Solids loading [g VS L^-1].

    Returns
    -------
    float
        Required yield increase [dm3 H2 g^-1 VS].
    """

    denominator = (
        S
        * KJ_PER_DM3
    )

    if denominator <= 0:
        return np.nan

    return (
        heat_duty(dT, eta)
        / denominator
    )


def breakeven_loading(dT, eta, dY):
    """
    Calculate the solids loading required for a given
    model-implied raw-scale yield difference to offset
    additional heating duty.

    Parameters
    ----------
    dT : float
        Temperature increase [°C].

    eta : float
        Heat-recovery efficiency [0–1].

    dY : float
        Model-implied yield difference [dm3 H2 g^-1 VS].

    Returns
    -------
    float
        Break-even solids loading [g VS L^-1].
    """

    if (
        not np.isfinite(dY)
        or dY <= 0
    ):
        return np.nan

    denominator = (
        dY
        * KJ_PER_DM3
    )

    if denominator <= 0:
        return np.nan

    return (
        heat_duty(dT, eta)
        / denominator
    )


# ============================================================
# MODEL-IMPLIED TEMPERATURE UPLIFT
# ============================================================

def model_implied_uplift_from_metaregression():
    """
    Retrieve the Step 03 temperature coefficient and convert
    its log1p-scale association into a model-implied raw-yield
    difference for a 37 -> 55 °C temperature shift.

    Step 03 uses:

        ly = log1p(y)

    Therefore:

        log1p(y_55)
            =
        log1p(y_37)
        + beta_T * DeltaT

    and:

        y_55
            =
        expm1(
            log1p(y_37)
            + beta_T * DeltaT
        )

    The raw-scale difference is:

        Delta_y = y_55 - y_37

    The baseline y_37 is taken as the median hydrogen yield
    in dataset_modelling_224.csv.

    Because log1p/expm1 is nonlinear, the resulting raw-scale
    uplift is baseline-dependent.

    Returns
    -------
    tuple
        baseline_y
        dY
        dY_lo
        dY_hi
        beta
        beta_lo
        beta_hi
        y_thermo
        y_thermo_lo
        y_thermo_hi
    """

    # --------------------------------------------------------
    # 1. Read Step 03 results
    # --------------------------------------------------------

    mr_path = RESULTS / "03_metaregression.csv"

    if not mr_path.exists():
        raise FileNotFoundError(
            "\n03_metaregression.csv was not found.\n\n"
            "Run Step 03 first."
        )

    mr = pd.read_csv(mr_path)


    # --------------------------------------------------------
    # 2. Validate output structure
    # --------------------------------------------------------

    required_columns = [
        "term",
        "coef",
        "ci_lo",
        "ci_hi"
    ]

    missing = [
        c
        for c in required_columns
        if c not in mr.columns
    ]

    if missing:
        raise RuntimeError(
            "03_metaregression.csv is missing "
            f"required columns: {missing}"
        )


    # --------------------------------------------------------
    # 3. Locate temperature coefficient
    # --------------------------------------------------------

    temperature_rows = mr[
        mr["term"].astype(str) == "Tn"
    ]

    if temperature_rows.empty:
        raise RuntimeError(
            "Temperature coefficient 'Tn' was not found "
            "in 03_metaregression.csv."
        )

    row = temperature_rows.iloc[0]

    beta = float(row["coef"])
    beta_lo = float(row["ci_lo"])
    beta_hi = float(row["ci_hi"])


    # --------------------------------------------------------
    # 4. Validate coefficients
    # --------------------------------------------------------

    if not all(
        np.isfinite(
            [
                beta,
                beta_lo,
                beta_hi
            ]
        )
    ):
        raise RuntimeError(
            "Temperature coefficient or confidence limits "
            "contain non-finite values."
        )


    # --------------------------------------------------------
    # 5. Read modelling dataset
    # --------------------------------------------------------

    if not MODEL_CSV.exists():
        raise FileNotFoundError(
            "\ndataset_modelling_224.csv was not found at:\n"
            f"{MODEL_CSV}\n\n"
            "Run Step 01 first."
        )

    d = pd.read_csv(MODEL_CSV)


    # --------------------------------------------------------
    # 6. Validate target
    # --------------------------------------------------------

    if "y" not in d.columns:
        raise RuntimeError(
            "Column 'y' was not found in "
            "dataset_modelling_224.csv."
        )

    y = pd.to_numeric(
        d["y"],
        errors="coerce"
    ).dropna()

    if len(y) == 0:
        raise RuntimeError(
            "Column 'y' contains no usable numeric observations."
        )


    # --------------------------------------------------------
    # 7. Validate yield
    # --------------------------------------------------------

    if (y < 0).any():
        raise RuntimeError(
            "Negative hydrogen-yield values were found in "
            "column 'y'."
        )


    # --------------------------------------------------------
    # 8. Baseline yield
    # --------------------------------------------------------

    baseline_y = float(y.median())


    # --------------------------------------------------------
    # 9. Temperature interval
    # --------------------------------------------------------

    span = T_THERMO - T_MESO


    # --------------------------------------------------------
    # 10. Baseline on log1p scale
    # --------------------------------------------------------

    baseline_log = np.log1p(baseline_y)


    # --------------------------------------------------------
    # 11. Temperature association on log1p scale
    # --------------------------------------------------------

    delta_log = beta * span
    delta_log_lo = beta_lo * span
    delta_log_hi = beta_hi * span


    # --------------------------------------------------------
    # 12. Back-transform to raw yield
    # --------------------------------------------------------

    y_thermo = np.expm1(
        baseline_log
        + delta_log
    )

    y_thermo_lo = np.expm1(
        baseline_log
        + delta_log_lo
    )

    y_thermo_hi = np.expm1(
        baseline_log
        + delta_log_hi
    )


    # --------------------------------------------------------
    # 13. Raw-scale yield difference
    # --------------------------------------------------------

    dY = y_thermo - baseline_y
    dY_lo = y_thermo_lo - baseline_y
    dY_hi = y_thermo_hi - baseline_y


    return (
        baseline_y,
        dY,
        dY_lo,
        dY_hi,
        beta,
        beta_lo,
        beta_hi,
        y_thermo,
        y_thermo_lo,
        y_thermo_hi
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("04 — ENERGY BALANCE")
    print("=" * 70)


    # ========================================================
    # Heat-balance assumptions
    # ========================================================

    print("\nHEAT BALANCE ASSUMPTIONS")

    print(
        "  cp %.2f kJ kg-1 K-1"
        % CP_WATER
    )

    print(
        "  rho %.0f kg m-3"
        % RHO_WATER
    )

    print(
        "  influent temperature %.0f °C"
        % T_INFLUENT
    )

    print(
        "  mesophilic temperature %.0f °C"
        % T_MESO
    )

    print(
        "  thermophilic temperature %.0f °C"
        % T_THERMO
    )

    print(
        "  1 dm3 H2 = %.3f kJ"
        % KJ_PER_DM3
    )

    print(
        "  H2 LHV = %.0f MJ kg-1"
        % H2_LHV
    )

    print(
        "  H2 density = %.4f kg m-3"
        % H2_DENSITY
    )

    print(
        "  mixing and pumping excluded"
    )

    print()


    # ========================================================
    # 1. Net energy per m3
    # ========================================================

    rows = []

    for S in [
        10,
        20,
        50,
        100
    ]:

        for T, Y in [
            (37, 0.15),
            (55, 0.25),
            (70, 0.20)
        ]:

            e = h2_energy(
                Y,
                S
            )

            q = heat_duty(
                T - T_INFLUENT,
                0.70
            )

            rows.append(
                {
                    "solids_gVS_L": S,
                    "temperature_C": T,
                    "yield_dm3_g": Y,
                    "h2_energy_MJ_m3": e,
                    "heat_duty_MJ_m3": q,
                    "net_MJ_m3": e - q
                }
            )


    net = pd.DataFrame(rows)

    net.to_csv(
        RESULTS / "04_energy_balance.csv",
        index=False
    )

    print(
        "NET ENERGY PER m3 OF FEED "
        "(70% heat recovery)"
    )

    print()

    print(
        net.round(3)
        .to_string(index=False)
    )

    print()


    # ========================================================
    # 2. Required yield difference grid
    # ========================================================

    grid = pd.DataFrame(
        {
            "solids_gVS_L": LOADINGS
        }
    )

    for eta in ETAS:

        column = "eta_%d" % int(eta * 100)

        grid[column] = [
            breakeven_uplift(
                T_THERMO - T_MESO,
                eta,
                S
            )
            for S in LOADINGS
        ]


    grid.to_csv(
        RESULTS / "04_breakeven_uplift.csv",
        index=False
    )

    print(
        "YIELD DIFFERENCE REQUIRED FOR "
        "37 → 55 °C TO PAY FOR HEATING"
    )

    print("(dm3 H2 g-1 VS)")

    print()

    print(
        grid.round(4)
        .to_string(index=False)
    )

    print()


    # ========================================================
    # 3. Model-implied temperature association
    # ========================================================

    (
        baseline_y,
        dY,
        dY_lo,
        dY_hi,
        beta,
        beta_lo,
        beta_hi,
        y_thermo,
        y_thermo_lo,
        y_thermo_hi
    ) = model_implied_uplift_from_metaregression()


    print(
        "MODEL-IMPLIED TEMPERATURE ASSOCIATION"
    )

    print()

    print(
        "Meta-regression coefficient "
        "on log1p(y): %+.6f per °C"
        % beta
    )

    print(
        "95%% CI: %+.6f to %+.6f"
        % (
            beta_lo,
            beta_hi
        )
    )

    print()

    print(
        "Baseline yield "
        "(dataset median): %.4f dm3 H2 g-1 VS"
        % baseline_y
    )

    print(
        "Temperature interval: "
        "%.0f → %.0f °C"
        % (
            T_MESO,
            T_THERMO
        )
    )

    print()

    print(
        "Model-implied yield at %.0f °C "
        "from the %.0f °C baseline:"
        % (
            T_THERMO,
            T_MESO
        )
    )

    print(
        "  central : %.4f dm3 H2 g-1 VS"
        % y_thermo
    )

    print(
        "  lower CI: %.4f dm3 H2 g-1 VS"
        % y_thermo_lo
    )

    print(
        "  upper CI: %.4f dm3 H2 g-1 VS"
        % y_thermo_hi
    )

    print()

    print(
        "Model-implied yield difference "
        "for %.0f → %.0f °C:"
        % (
            T_MESO,
            T_THERMO
        )
    )

    print(
        "  central : %+.4f dm3 H2 g-1 VS"
        % dY
    )

    print(
        "  lower CI: %+.4f dm3 H2 g-1 VS"
        % dY_lo
    )

    print(
        "  upper CI: %+.4f dm3 H2 g-1 VS"
        % dY_hi
    )

    print()

    print(
        "NOTE: The raw-scale uplift is evaluated "
        "at the dataset median baseline yield and "
        "is therefore baseline-dependent."
    )

    print()


    # ========================================================
    # 4. Break-even solids loading
    # ========================================================

    out = []

    for eta in ETAS[1:]:

        out.append(
            {
                "heat_recovery": eta,

                "breakeven_loading_gVS_L":
                    breakeven_loading(
                        T_THERMO - T_MESO,
                        eta,
                        dY
                    ),

                "loading_at_CI_low":
                    breakeven_loading(
                        T_THERMO - T_MESO,
                        eta,
                        dY_hi
                    ),

                "loading_at_CI_high":
                    breakeven_loading(
                        T_THERMO - T_MESO,
                        eta,
                        dY_lo
                    )
            }
        )


    be = pd.DataFrame(out)

    be.to_csv(
        RESULTS / "04_breakeven_loading.csv",
        index=False
    )

    print(
        "BREAK-EVEN SOLIDS LOADING "
        "GIVEN THE MODEL-IMPLIED YIELD DIFFERENCE"
    )

    print("(g VS L-1)")

    print()

    print(
        be.round(3)
        .to_string(index=False)
    )

    print()


    # ========================================================
    # 5. Hydrogen energy recovery
    # ========================================================

    print(
        "FRACTION OF FEED CHEMICAL ENERGY "
        "RECOVERED AS H2"
    )

    print(
        "(VS calorific value = %.1f MJ kg-1)"
        % VS_CALORIFIC
    )

    print()


    # --------------------------------------------------------
    # Theoretical Thauer limit:
    #
    # 4 mol H2 per mol hexose
    #
    # 4 × 22.414 dm3/mol
    # -------------------
    # 180.16 g/mol
    #
    # = 0.4978 dm3 H2 g^-1
    # --------------------------------------------------------

    thauer_yield = (
        4.0
        * 22.414
        / 180.16
    )


    for label, Y in [

        (
            "dataset median yield",
            baseline_y
        ),

        (
            "Thauer limit, "
            "4 mol H2 per mol hexose",
            thauer_yield
        )

    ]:

        # Y [dm3/g]
        #
        # × 1000 g/kg
        # × kJ/dm3
        # / 1000 kJ/MJ
        #
        # = MJ/kg VS

        mj = (
            Y
            * 1000.0
            * KJ_PER_DM3
            / 1000.0
        )

        fraction = (
            100.0
            * mj
            / VS_CALORIFIC
        )

        print(
            "  %-45s "
            "Y = %.3f -> %6.2f MJ kg-1 VS "
            "(%5.1f%%)"
            % (
                label,
                Y,
                mj,
                fraction
            )
        )


    print()

    print(
        "  The remainder leaves primarily as "
        "non-H2 products such as volatile fatty acids."
    )


    # ========================================================
    # Files
    # ========================================================

    print()

    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(
        RESULTS / "04_energy_balance.csv"
    )

    print(
        RESULTS / "04_breakeven_uplift.csv"
    )

    print(
        RESULTS / "04_breakeven_loading.csv"
    )

    print()

    print("DONE.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
