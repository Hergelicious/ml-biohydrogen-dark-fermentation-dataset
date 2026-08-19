#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_energy_balance.py -- couple the meta-regression temperature coefficient
to a reactor heat balance, to test whether the yield-optimal temperature
is also the energy-optimal one.

Per cubic metre of aqueous feed:

    heating duty
        Q = cp * rho * dT * (1 - eta) / 1000

    hydrogen energy
        E = Y * S * KJ_PER_DM3

where:

    Y   = hydrogen yield (dm3 H2 g-1 VS)
    S   = solids loading (g VS L-1)
    eta = heat-recovery efficiency

and:

    KJ_PER_DM3 = H2_DENSITY * H2_LHV

Setting the incremental hydrogen energy equal to the additional heating
duty gives the break-even yield uplift:

    dY = heat_duty / (S * KJ_PER_DM3)

Mixing and pumping energy are excluded, so the reported break-even
thresholds are optimistic.

Reads:
    results/03_temperature_effect.csv

Writes:
    results/04_energy_balance.csv
    results/04_breakeven_uplift.csv
    results/04_breakeven_loading.csv
"""


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# 2. PROJECT PATH
# ============================================================

# This file is located in:
#
#     repo/src/04_energy_balance.py
#
# Therefore the repository root is one directory above src/.

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 3. IMPORT SHARED DEFINITIONS
# ============================================================

from common import (
    CP_WATER,
    H2_DENSITY,
    H2_LHV,
    RESULTS,
    RHO_WATER,
)


# ============================================================
# 4. CONSTANTS
# ============================================================

KJ_PER_DM3 = (
    H2_DENSITY
    * H2_LHV
)

T_MESO = 37.0
T_THERMO = 55.0
T_INFLUENT = 20.0

ETAS = [
    0.0,
    0.5,
    0.7,
    0.85,
]

LOADINGS = [
    10,
    20,
    30,
    50,
    100,
    150,
]

VS_CALORIFIC = 15.6


# ============================================================
# 5. HYDROGEN ENERGY
# ============================================================

def h2_energy(
    Y,
    S,
):
    """
    Calculate hydrogen energy per m3 of feed.

    Parameters
    ----------
    Y : float
        Hydrogen yield in dm3 H2 g-1 VS.

    S : float
        Solids loading in g VS L-1.

    Returns
    -------
    float
        Hydrogen energy in MJ m-3.
    """

    return (
        Y
        * S
        * KJ_PER_DM3
    )


# ============================================================
# 6. HEAT DUTY
# ============================================================

def heat_duty(
    dT,
    eta,
):
    """
    Calculate heating duty per m3 of feed.

    Parameters
    ----------
    dT : float
        Temperature increase in degC.

    eta : float
        Heat-recovery efficiency.

    Returns
    -------
    float
        Heating duty in MJ m-3.
    """

    return (
        CP_WATER
        * RHO_WATER
        * dT
        / 1000.0
        * (1.0 - eta)
    )


# ============================================================
# 7. BREAK-EVEN YIELD UPLIFT
# ============================================================

def breakeven_uplift(
    dT,
    eta,
    S,
):
    """
    Calculate the hydrogen-yield increase required to offset
    additional heating duty.

    Returns
    -------
    float
        Required yield uplift in dm3 H2 g-1 VS.
    """

    return (
        heat_duty(
            dT,
            eta,
        )
        /
        (
            S
            * KJ_PER_DM3
        )
    )


# ============================================================
# 8. BREAK-EVEN SOLIDS LOADING
# ============================================================

def breakeven_loading(
    dT,
    eta,
    dY,
):
    """
    Calculate the solids loading at which a specified hydrogen-yield
    uplift offsets the additional heating duty.

    Returns
    -------
    float
        Break-even solids loading in g VS L-1.
    """

    return (
        heat_duty(
            dT,
            eta,
        )
        /
        (
            dY
            * KJ_PER_DM3
        )
    )


# ============================================================
# 9. READ MEASURED TEMPERATURE EFFECT
# ============================================================

def measured_uplift():
    """
    Read the model-implied 37 -> 55 degC yield difference generated
    by 03_statistics.py.

    Returns
    -------
    tuple
        Yield difference and its lower and upper confidence limits.
    """

    path = (
        RESULTS
        / "03_temperature_effect.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nRequired file was not found:\n"
            f"{path}\n\n"
            "Run 03_statistics.py before running "
            "04_energy_balance.py."
        )

    result = pd.read_csv(
        path
    ).iloc[0]

    return (
        result["implied_difference"],
        result["ci_lo"],
        result["ci_hi"],
    )


# ============================================================
# 10. MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("ENERGY BALANCE ANALYSIS")
    print("=" * 70)

    print(
        "\nRepository:"
    )

    print(
        ROOT
    )

    print(
        "\nResults directory:"
    )

    print(
        RESULTS
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Heat-balance assumptions
    # --------------------------------------------------------

    dT = (
        T_THERMO
        - T_INFLUENT
    )

    print(
        "\nHEAT BALANCE ASSUMPTIONS"
    )

    print(
        "  cp %.2f kJ kg-1 K-1, "
        "rho %.0f kg m-3, "
        "influent %.0f degC"
        % (
            CP_WATER,
            RHO_WATER,
            T_INFLUENT,
        )
    )

    print(
        "  1 dm3 H2 = %.2f kJ "
        "(LHV %.0f MJ kg-1, density %.4f kg m-3)"
        % (
            KJ_PER_DM3,
            H2_LHV,
            H2_DENSITY,
        )
    )

    print(
        "  mixing and pumping excluded\n"
    )

    # ========================================================
    # 11. NET ENERGY PER m3
    # ========================================================

    rows = []

    for S in [
        10,
        20,
        50,
        100,
    ]:

        for T, Y in [
            (37, 0.15),
            (55, 0.25),
            (70, 0.20),
        ]:

            e = h2_energy(
                Y,
                S,
            )

            q = heat_duty(
                T - T_INFLUENT,
                0.70,
            )

            rows.append(
                {
                    "solids_gVS_L": S,
                    "temperature_C": T,
                    "yield_dm3_g": Y,
                    "h2_energy_MJ_m3": e,
                    "heat_duty_MJ_m3": q,
                    "net_MJ_m3": e - q,
                }
            )

    net = pd.DataFrame(
        rows
    )

    net_path = (
        RESULTS
        / "04_energy_balance.csv"
    )

    net.to_csv(
        net_path,
        index=False,
    )

    print(
        "NET ENERGY PER m3 OF FEED "
        "(70% heat recovery)"
    )

    print(
        net.round(1)
        .to_string(
            index=False
        )
    )

    print(
        "  -> at 10 g VS L-1 the balance is negative "
        "at every temperature tested.\n"
    )

    # ========================================================
    # 12. REQUIRED YIELD UPLIFT
    # ========================================================

    grid = pd.DataFrame(
        {
            "solids_gVS_L": LOADINGS
        }
    )

    temperature_span = (
        T_THERMO
        - T_MESO
    )

    for eta in ETAS:

        grid[
            "eta_%d"
            % int(
                eta * 100
            )
        ] = [
            breakeven_uplift(
                temperature_span,
                eta,
                S,
            )
            for S in LOADINGS
        ]

    uplift_path = (
        RESULTS
        / "04_breakeven_uplift.csv"
    )

    grid.to_csv(
        uplift_path,
        index=False,
    )

    print(
        "YIELD UPLIFT REQUIRED FOR "
        "37 -> 55 degC TO PAY FOR ITS OWN HEAT "
        "(dm3 H2 g-1)"
    )

    print(
        grid.round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 13. MEASURED TEMPERATURE UPLIFT
    # ========================================================

    dY, dY_lo, dY_hi = (
        measured_uplift()
    )

    print(
        "\nMODEL-IMPLIED YIELD DIFFERENCE, "
        "37 -> 55 degC (back-transformed)"
    )

    print(
        "  %+.3f dm3 H2 g-1 "
        "(95%% CI %+.3f to %+.3f)"
        % (
            dY,
            dY_lo,
            dY_hi,
        )
    )

    # ========================================================
    # 14. BREAK-EVEN LOADING
    # ========================================================

    out = []

    for eta in ETAS[1:]:

        out.append(
            {
                "heat_recovery": eta,

                "breakeven_loading_gVS_L":
                    breakeven_loading(
                        temperature_span,
                        eta,
                        dY,
                    ),

                "loading_at_CI_low":
                    breakeven_loading(
                        temperature_span,
                        eta,
                        dY_hi,
                    ),

                "loading_at_CI_high":
                    breakeven_loading(
                        temperature_span,
                        eta,
                        dY_lo,
                    ),
            }
        )

    be = pd.DataFrame(
        out
    )

    loading_path = (
        RESULTS
        / "04_breakeven_loading.csv"
    )

    be.to_csv(
        loading_path,
        index=False,
    )

    print(
        "\nBREAK-EVEN SOLIDS LOADING "
        "GIVEN THE MEASURED UPLIFT "
        "(g VS L-1)"
    )

    print(
        be.round(2)
        .to_string(
            index=False
        )
    )

    print(
        "  dilute industrial wastewater is typically "
        "1-10 g VS L-1;"
    )

    print(
        "  food-waste slurry is typically "
        "40-120 g VS L-1."
    )

    # ========================================================
    # 15. SUBSTRATE ENERGY RECOVERY
    # ========================================================

    print(
        "\nFRACTION OF FEED CHEMICAL ENERGY "
        "RECOVERED AS H2 "
        "(VS at %.1f MJ kg-1)"
        % VS_CALORIFIC
    )

    energy_cases = [
        (
            "dataset median yield",
            0.111,
        ),
        (
            "Thauer limit, "
            "4 mol H2 per mol hexose",
            4 * 22.414 / 180.16,
        ),
    ]

    for label, Y in energy_cases:

        mj = (
            Y
            * 1000
            * KJ_PER_DM3
            / 1000.0
        )

        fraction = (
            100
            * mj
            / VS_CALORIFIC
        )

        print(
            "  %-40s "
            "Y = %.3f -> %5.2f MJ per kg VS "
            "(%4.1f%%)"
            % (
                label,
                Y,
                mj,
                fraction,
            )
        )

    print(
        "  the remainder leaves as volatile fatty acids."
    )

    # ========================================================
    # 16. OUTPUT FILES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES WRITTEN"
    )

    print(
        "=" * 70
    )

    print(
        "\n1.",
        net_path,
    )

    print(
        "2.",
        uplift_path,
    )

    print(
        "3.",
        loading_path,
    )

    print(
        "\nEnergy-balance analysis complete."
    )

    print(
        "=" * 70
    )


# ============================================================
# 17. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
