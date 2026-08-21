#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_energy_balance.py -- couple the meta-regression temperature coefficient to a
reactor heat balance, to test whether the yield-optimal temperature is also the
energy-optimal one.

Per cubic metre of aqueous feed:
    heating duty   Q  = cp * rho * dT * (1 - eta) / 1000     MJ m-3
    hydrogen energy E  = 10.79 * Y * S / 1000 * 1000 = 10.79*Y*S  MJ m-3
where 1 dm3 H2 = H2_DENSITY * H2_LHV = 10.79 kJ (LHV basis), Y is the yield in
dm3 H2 per g VS, S the solids loading in g VS L-1, and eta the heat-recovery
efficiency. Setting E(Y+dY) - E(Y) = Q gives the break-even uplift

    dY = cp * rho * dT * (1 - eta) / (1000 * 10.79 * S)

Mixing and pumping (roughly 2-7 MJ m-3 at 0.5-2 kWh m-3) are excluded, so every
threshold reported here is optimistic.

Reads   results/03_metaregression.csv   (for the measured temperature effect)
Writes  results/04_energy_balance.csv        net energy per m3
        results/04_breakeven_uplift.csv        required uplift grid
        results/04_breakeven_loading.csv       break-even solids loading
"""
import sys
import warnings
from pathlibPath import Path

import numpy as np
import pandas as pd

# Ensure common.py can be imported from the current script directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CP_WATER, H2_DENSITY, H2_LHV, RESULTS, RHO_WATER

warnings.filterwarnings("ignore")

KJ_PER_DM3 = H2_DENSITY * H2_LHV  # 10.79 kJ per dm3 H2, LHV basis
T_MESO, T_THERMO, T_INFLUENT = 37.0, 55.0, 20.0
ETAS = [0.0, 0.5, 0.7, 0.85]
LOADINGS = [10, 20, 30, 50, 100, 150]
VS_CALORIFIC = 15.6  # MJ per kg VS, typical for organic solids


def h2_energy(Y, S):
    """MJ of hydrogen per m3 of feed."""
    return Y * S * KJ_PER_DM3


def heat_duty(dT, eta):
    """MJ per m3 of feed to raise the influent by dT with heat recovery eta."""
    return CP_WATER * RHO_WATER * dT / 1000.0 * (1 - eta)


def breakeven_uplift(dT, eta, S):
    """Yield uplift (dm3 H2 g-1) needed to offset the extra heating duty."""
    return heat_duty(dT, eta) / (S * KJ_PER_DM3)


def breakeven_loading(dT, eta, dY):
    """Solids loading (g VS L-1) at which a given uplift pays for the heating."""
    return heat_duty(dT, eta) / (dY * KJ_PER_DM3)


def measured_uplift(basis="vs"):
    """Model-implied 37 -> 55 degC yield difference from 03_statistics.py.

    The heat balance divides by a solids loading in g VS L-1, so the yield term
    must also be on a volatile-solids basis or the ratio is dimensionally
    inconsistent. The default therefore reads the native-VS estimate; the
    mixed-basis estimate is reported alongside for comparison only.
    """
    tag = "_vs" if basis == "vs" else ""
    path = RESULTS / ("03_temperature_effect%s.csv" % tag)
    if not path.exists():
        raise SystemExit("run 03_statistics.py first (needs %s)" % path.name)
    r = pd.read_csv(path).iloc[0]
    return r


def main():
    dT = T_THERMO - T_INFLUENT
    print("HEAT BALANCE ASSUMPTIONS")
    print(
        "  cp %.2f kJ kg-1 K-1, rho %.0f kg m-3, influent %.0f degC"
        % (CP_WATER, RHO_WATER, T_INFLUENT)
    )
    print(
        "  1 dm3 H2 = %.2f kJ (LHV %.0f MJ kg-1, density %.4f kg m-3)"
        % (KJ_PER_DM3, H2_LHV, H2_DENSITY)
    )
    print("  mixing and pumping excluded\n")

    # ---- net energy per m3 --------------------------------------------------
    rows = []
    for S in [10, 20, 50, 100]:
        for T, Y in [(37, 0.15), (55, 0.25), (70, 0.20)]:
            e = h2_energy(Y, S)
            q = heat_duty(T - T_INFLUENT, 0.70)
            rows.append(
                dict(
                    solids_gVS_L=S,
                    temperature_C=T,
                    yield_dm3_g=Y,
                    h2_energy_MJ_m3=e,
                    heat_duty_MJ_m3=q,
                    net_MJ_m3=e - q,
                )
            )
    net = pd.DataFrame(rows)
    net.to_csv(RESULTS / "04_energy_balance.csv", index=False)
    print("NET ENERGY PER m3 OF FEED (70% heat recovery)")
    print(net.round(1).to_string(index=False))
    print(
        "  -> at 10 g VS L-1 the balance is negative at every temperature tested.\n"
    )

    # ---- required uplift grid ----------------------------------------------
    grid = pd.DataFrame({"solids_gVS_L": LOADINGS})
    for eta in ETAS:
        grid["eta_%d" % int(eta * 100)] = [
            breakeven_uplift(T_THERMO - T_MESO, eta, S) for S in LOADINGS
        ]
    grid.to_csv(RESULTS / "04_breakeven_uplift.csv", index=False)
    print(
        "YIELD UPLIFT REQUIRED FOR 37 -> 55 degC TO PAY FOR ITS OWN HEAT (dm3 H2 g-1)"
    )
    print(grid.round(3).to_string(index=False))

    # ---- measured uplift and the resulting threshold ------------------------
    vs = measured_uplift("vs")
    mixed = measured_uplift("mixed")
    dY, dY_lo, dY_hi = vs.implied_difference, vs.implied_lo, vs.implied_hi
    print("\nMODEL-IMPLIED YIELD DIFFERENCE, 37 -> 55 degC (back-transformed)")
    print(
        "  native VS basis   %+.4f dm3 H2 g-1 VS  (95%% CI %+.4f to %+.4f, p = %.4f, n = %d)"
        % (dY, dY_lo, dY_hi, vs.p, vs.n)
    )
    print(
        "  mixed basis       %+.4f dm3 H2 g-1 substrate (95%% CI %+.4f to %+.4f, p = %.4f, n = %d)"
        % (
            mixed.implied_difference,
            mixed.implied_lo,
            mixed.implied_hi,
            mixed.p,
            mixed.n,
        )
    )
    print(
        "  the heat balance below uses the native-VS estimate, because the loading term"
    )
    print("  S is in g VS L-1 and the two must share a denominator.")
    if vs.p >= 0.05:
        print(
            "  CAUTION: the VS estimate is not significantly different from zero, so the"
        )
        print(
            "           upper confidence bound on the break-even loading is unbounded."
        )
    out = []
    for eta in ETAS[1:]:
        lo_bound = (
            breakeven_loading(T_THERMO - T_MESO, eta, dY_hi)
            if dY_hi > 0
            else float("nan")
        )
        hi_bound = (
            breakeven_loading(T_THERMO - T_MESO, eta, dY_lo)
            if dY_lo > 0
            else float("inf")
        )
        out.append(
            dict(
                heat_recovery=eta,
                breakeven_loading_gVS_L=breakeven_loading(
                    T_THERMO - T_MESO, eta, dY
                ),
                loading_at_CI_low=lo_bound,
                loading_at_CI_high=hi_bound,
            )
        )
    be = pd.DataFrame(out)
    be.to_csv(RESULTS / "04_breakeven_loading.csv", index=False)
    print("\nBREAK-EVEN SOLIDS LOADING GIVEN THE MEASURED UPLIFT (g VS L-1)")
    print(be.round(2).to_string(index=False))
    print("  dilute industrial wastewater is typically 1-10 g VS L-1;")
    print("  food-waste slurry is typically 40-120 g VS L-1.")

    # ---- where the substrate energy goes -----------------------------------
    print(
        "\nFRACTION OF FEED CHEMICAL ENERGY RECOVERED AS H2 (VS at %.1f MJ kg-1)"
        % VS_CALORIFIC
    )
    for label, Y in [
        ("dataset median yield", 0.111),
        (
            "Thauer limit, 4 mol H2 per mol hexose",
            4 * 22.414 / 180.16,
        ),
    ]:
        mj = Y * 1000 * KJ_PER_DM3 / 1000.0
        print(
            "  %-40s Y = %.3f -> %5.2f MJ per kg VS (%4.1f%%)"
            % (label, Y, mj, 100 * mj / VS_CALORIFIC)
        )
    print("  the remainder leaves as volatile fatty acids.")
    print(
        "\nwrote results/04_energy_balance.csv, 04_breakeven_uplift.csv, "
        "04_breakeven_loading.csv"
    )


if __name__ == "__main__":
    main()
