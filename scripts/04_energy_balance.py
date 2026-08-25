"""
04_energy_balance.py — reactor heat balance coupled to the temperature association.

Couples the meta-regression temperature coefficient, estimated on the native
volatile-solids subset so that the statistical estimate and the solids loading
share a mass denominator, to a sanitary-engineering thermal balance.

IMPORTANT — how these numbers should be read
    The native-VS temperature term is not statistically significant (p ~ 0.30)
    and its confidence interval on the yield difference includes zero. Every
    loading produced here is therefore a CONDITIONAL SCENARIO CALCULATION
    evaluated at a point estimate, not an empirically established break-even
    threshold. Because the interval spans zero, no finite break-even loading is
    established by these data and the upper bound is unbounded.

Energy accounting, per cubic metre of aqueous feed
    Heating duty     Q = cp * rho * dT * (1 - eta) / 1000   MJ m-3
                       = 4.18 * dT * (1 - eta)              MJ m-3
    Hydrogen energy  E = 10.79 * Y * S                      MJ m-3
        with Y in dm3 H2 g-1 VS and S in g VS L-1 (= kg VS m-3),
        since Y * S is a volume ratio and 10.79 kJ dm-3 = 10.79 MJ m-3 per unit ratio.
    Break-even       dY = 4.18 * dT * (1 - eta) / (10.79 * S)
    Break-even S     S  = 4.18 * dT * (1 - eta) / (10.79 * dY)

Mixing and pumping are excluded from this screening balance and are quantified
separately below so that the omission is bounded rather than asserted.

Outputs
    results/04_breakeven_uplift.csv    Table S22
    results/04_breakeven_loading.csv   Table S23
    results/04_energy_balance.csv      Table S24
    results/04_energy.log
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (CP_WATER, E_H2_PER_DM3, RESULTS, T_INFLUENT, T_MESO, T_THERMO,
                    get_logger, write_table)

log = get_logger("04_energy")

ETAS = [0.00, 0.50, 0.70, 0.85]
LOADINGS = [10, 20, 30, 50, 100, 150]
DELTA_T = T_THERMO - T_MESO                 # 18 K, the incremental heating step

# Screening ranges for mixing power (W m-3) and hydraulic retention time (days)
MIXING_POWER_W_M3 = (5.0, 8.0)
HRT_DAYS = (1.0, 5.0)


def heating_duty(delta_t: float, eta: float) -> float:
    """Incremental heating duty, MJ per m3 of feed."""
    return CP_WATER * delta_t * (1.0 - eta)


def required_uplift(delta_t: float, eta: float, loading: float) -> float:
    """Yield uplift needed to offset the heating duty, dm3 H2 g-1 VS."""
    return heating_duty(delta_t, eta) / (E_H2_PER_DM3 * loading)


def breakeven_loading(delta_t: float, eta: float, delta_y: float) -> float:
    """Solids loading at which a given yield difference offsets the heating duty."""
    if delta_y <= 0:
        return np.inf
    return heating_duty(delta_t, eta) / (E_H2_PER_DM3 * delta_y)


def load_temperature_effect() -> dict:
    """Read the native-VS temperature effect written by 03_statistics.py."""
    path = RESULTS / "03_temperature_effect.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run 03_statistics.py first.")
    tbl = pd.read_csv(path)
    row = tbl[tbl["subset"] == "native volatile solids"]
    if row.empty:
        raise ValueError("native volatile solids row missing from 03_temperature_effect.csv")
    r = row.iloc[0]
    return {"delta_y": float(r["delta_y_point"]),
            "delta_y_lo": float(r["delta_y_lo"]),
            "delta_y_hi": float(r["delta_y_hi"]),
            "p_value": float(r["p_value"]),
            "n_records": int(r["n_records"]),
            "n_studies": int(r["n_studies"])}


def main() -> None:
    eff = load_temperature_effect()
    log.info("native-VS temperature effect: dY = %+.4f dm3 H2 g-1 VS "
             "(95%% CI %+.4f to %+.4f; p = %.4f; n = %d records, %d studies)",
             eff["delta_y"], eff["delta_y_lo"], eff["delta_y_hi"],
             eff["p_value"], eff["n_records"], eff["n_studies"])
    if eff["p_value"] > 0.05:
        log.warning("temperature term is NOT statistically significant — every loading "
                    "below is a conditional scenario calculation, not a threshold")
    if eff["delta_y_lo"] <= 0.0:
        log.warning("confidence interval on the yield difference includes zero — "
                    "the upper bound on break-even loading is unbounded")

    # ── Table S22: uplift required at each loading and heat-recovery level ──
    rows = []
    for S in LOADINGS:
        entry = {"solids_loading_g_VS_per_L": S}
        for eta in ETAS:
            entry[f"eta_{int(eta * 100)}"] = round(required_uplift(DELTA_T, eta, S), 4)
        rows.append(entry)
    uplift = pd.DataFrame(rows)
    write_table(uplift, "04_breakeven_uplift.csv")

    # ── Table S23: conditional scenario loadings ────────────────────────────
    rows = []
    for eta in [0.85, 0.70, 0.50]:
        point = breakeven_loading(DELTA_T, eta, eff["delta_y"])
        lower = breakeven_loading(DELTA_T, eta, eff["delta_y_hi"])   # optimistic bound
        upper = breakeven_loading(DELTA_T, eta, eff["delta_y_lo"])   # unbounded if CI spans 0
        rows.append({
            "heat_recovery": f"{int(eta * 100)}%",
            "scenario_loading_g_VS_per_L": round(point, 1),
            "lower_bound_g_VS_per_L": round(lower, 1),
            "upper_bound_g_VS_per_L": ("unbounded" if not np.isfinite(upper) else round(upper, 1)),
            "interpretation": "conditional scenario calculation at the point estimate; "
                              "not an established threshold",
        })
    loading = pd.DataFrame(rows)
    write_table(loading, "04_breakeven_loading.csv")
    for r in rows:
        log.info("  eta = %-4s -> scenario loading %.1f g VS L-1 (lower bound %.1f, upper %s)",
                 r["heat_recovery"], r["scenario_loading_g_VS_per_L"],
                 r["lower_bound_g_VS_per_L"], r["upper_bound_g_VS_per_L"])

    # ── Table S24: absolute screening balance from a 20 C influent ──────────
    rows = []
    for T in [T_MESO, T_THERMO, 70.0]:
        for S, Y in [(10, 0.15), (10, 0.25), (20, 0.15), (20, 0.25),
                     (50, 0.25), (100, 0.25)]:
            q = CP_WATER * (T - T_INFLUENT) * (1.0 - 0.70)
            e = E_H2_PER_DM3 * Y * S
            rows.append({
                "temperature_C": T,
                "solids_loading_g_VS_per_L": S,
                "assumed_yield_dm3_per_g_VS": Y,
                "hydrogen_energy_MJ_per_m3": round(e, 1),
                "heating_duty_MJ_per_m3": round(q, 1),
                "net_MJ_per_m3": round(e - q, 1),
            })
    balance = pd.DataFrame(rows)
    write_table(balance, "04_energy_balance.csv")

    # ── bound the excluded mixing and pumping term ──────────────────────────
    log.info("--- excluded terms, bounded rather than asserted ---")
    for p_w in MIXING_POWER_W_M3:
        for hrt in HRT_DAYS:
            mj = p_w * hrt * 86400.0 / 1e6      # W m-3 * s -> MJ m-3
            log.info("  mixing at %.0f W m-3 over %.0f d HRT = %.2f MJ m-3 "
                     "(%.1f%% of the 18 K duty at 70%% recovery)",
                     p_w, hrt, mj, 100.0 * mj / heating_duty(DELTA_T, 0.70))
    log.info("Mixing is therefore a small fraction of the incremental heating duty at "
             "short HRT and becomes comparable only at long HRT; it does not change the "
             "direction of the screening result.")


if __name__ == "__main__":
    main()
