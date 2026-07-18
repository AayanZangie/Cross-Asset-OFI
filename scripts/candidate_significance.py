"""DM/MCS significance checks for the confirmed 300s CCZ-lag candidate family."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tests as grid  # noqa: E402
from cross_sectional import _lasso_wf, _pick_alpha  # noqa: E402
from evaluation import forecast_portfolio, rolling_forecast_vol  # noqa: E402
from leadlag import fwd_return, make_lag_windows, score, walk_forward  # noqa: E402
from significance_tests import diebold_mariano, model_confidence_set, squared_error  # noqa: E402


OUT = ROOT / "output" / "confirmation"
BAR_S = 300
SCHEME = "pca"
LAGS = [1, 2, 3, 5, 10, 20, 30]
HORIZONS = [2, 5, 10, 20]
TRAINS = [3000, 6000]
LEVELS = [10, 20]
REFIT_CAP = 200
COSTS = [0.0, 0.25, 0.5, 1.0]


def load_panel(levels):
    old_levels = grid.LEVELS
    grid.LEVELS = levels
    bars = {a: grid.load_levels_bars(a, BAR_S * 1_000_000_000, levels=levels)
            for a in grid.ASSETS}
    ofi, ret, spr = grid.build_scheme_panel(bars, SCHEME)
    grid.LEVELS = old_levels
    return ofi, ret, spr


def finite(x):
    x = float(x)
    return x if np.isfinite(x) else None


def fit_family(ofi, ret, spr, horizon, train):
    t, n = ofi.shape
    step = max(500, t // REFIT_CAP)
    max_lag = max(LAGS)
    ofi_blocks = [make_lag_windows(ofi[:, j], LAGS) for j in range(n)]
    ret_blocks = [make_lag_windows(ret[:, j], LAGS) for j in range(n)]
    all_ofi = np.hstack(ofi_blocks)
    all_ret = np.hstack(ret_blocks)
    f_own = np.full((t, n), np.nan)
    f_car = np.full((t, n), np.nan)
    f_cross = np.full((t, n), np.nan)
    realized = np.column_stack([fwd_return(ret[:, i], horizon) for i in range(n)])
    score_rows = []
    sig_rows = []

    for i, name in enumerate(grid.NAMES):
        y = realized[:, i]
        base = (np.arange(t) >= max_lag - 1) & ~np.isnan(y)

        own_x = ofi_blocks[i]
        ok_own = base & ~np.isnan(own_x).any(axis=1)
        p_own = walk_forward(own_x[ok_own], y[ok_own], train=train, step=step, purge=horizon)
        f_own[np.where(ok_own)[0], i] = p_own

        ok_car = base & ~np.isnan(all_ret).any(axis=1)
        alpha_car = _pick_alpha(all_ret[ok_car][:train], y[ok_car][:train])
        p_car, _ = _lasso_wf(all_ret[ok_car], y[ok_car], alpha_car, train, step, purge=horizon)
        f_car[np.where(ok_car)[0], i] = p_car

        ok_cross = base & ~np.isnan(all_ofi).any(axis=1)
        alpha_cross = _pick_alpha(all_ofi[ok_cross][:train], y[ok_cross][:train])
        p_cross, _ = _lasso_wf(all_ofi[ok_cross], y[ok_cross], alpha_cross, train, step, purge=horizon)
        f_cross[np.where(ok_cross)[0], i] = p_cross

        sc_own = score(y, f_own[:, i])
        sc_car = score(y, f_car[:, i])
        sc_cross = score(y, f_cross[:, i])
        score_rows.append({
            "asset": name,
            "horizon": horizon,
            "horizon_seconds": horizon * BAR_S,
            "train": train,
            "own_r2": finite(sc_own[0]),
            "car_r2": finite(sc_car[0]),
            "cross_r2": finite(sc_cross[0]),
            "cross_minus_own_r2": finite(sc_cross[0] - sc_own[0]),
            "cross_minus_car_r2": finite(sc_cross[0] - sc_car[0]),
            "own_hit": finite(sc_own[1]),
            "car_hit": finite(sc_car[1]),
            "cross_hit": finite(sc_cross[1]),
            "n": int(sc_cross[2]),
        })

        losses = pd.DataFrame({
            "own": squared_error(f_own[:, i], y),
            "car": squared_error(f_car[:, i], y),
            "cross": squared_error(f_cross[:, i], y),
        }).dropna()
        dm_own, p_own_dm = diebold_mariano(losses["own"], losses["cross"], horizon=horizon)
        dm_car, p_car_dm = diebold_mariano(losses["car"], losses["cross"], horizon=horizon)
        try:
            mcs = model_confidence_set(losses, alpha=0.10, reps=300,
                                       block_size=max(5, horizon), seed=train + horizon + i)
            included = ",".join(mcs["included"])
        except Exception as exc:
            included = f"ERROR:{exc!r}"
        sig_rows.append({
            "asset": name,
            "horizon": horizon,
            "horizon_seconds": horizon * BAR_S,
            "train": train,
            "dm_own_minus_cross": finite(dm_own),
            "p_own_vs_cross": finite(p_own_dm),
            "dm_car_minus_cross": finite(dm_car),
            "p_car_vs_cross": finite(p_car_dm),
            "mcs_included_10pct": included,
        })

    sig = rolling_forecast_vol(f_cross, window=max(30, min(180, t // 50)))
    port_rows = []
    for cost in COSTS:
        res = forecast_portfolio(f_cross, realized, spr, sig, cost_bps=cost)
        port_rows.append({
            "horizon": horizon,
            "horizon_seconds": horizon * BAR_S,
            "train": train,
            "cost_bps": cost,
            "cum_net": finite(res["cum_net"]),
            "cum_gross": finite(res["cum_gross"]),
            "hit_rate": finite(res["hit_rate"]),
            "frac_traded": finite(res["frac_traded"]),
            "avg_turnover": finite(res["avg_turnover"]),
            "periods": int(res["periods"]),
        })

    return score_rows, sig_rows, port_rows


def write_summary(scores, sigs, ports):
    s = pd.DataFrame(scores)
    g = pd.DataFrame(sigs)
    p = pd.DataFrame(ports)
    lines = ["# Candidate Significance", ""]
    lines.append("Candidate family: 300-second bars, PCA-integrated OFI, CCZ lag windows.")
    lines.append("")
    lines.append("## Aggregate Scores")
    agg = (s.groupby(["levels", "train", "horizon", "horizon_seconds"])
             .agg(own_r2=("own_r2", "mean"),
                  car_r2=("car_r2", "mean"),
                  cross_r2=("cross_r2", "mean"),
                  cross_minus_own_r2=("cross_minus_own_r2", "mean"),
                  cross_minus_car_r2=("cross_minus_car_r2", "mean"),
                  cross_hit=("cross_hit", "mean"))
             .reset_index())
    lines.append(agg.to_markdown(index=False))
    lines.append("")
    lines.append("## DM/MCS By Asset")
    lines.append(g.to_markdown(index=False))
    lines.append("")
    lines.append("## Portfolio")
    lines.append(p.to_markdown(index=False))
    lines.append("")
    (OUT / "significance.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_scores, all_sigs, all_ports = [], [], []
    for levels in LEVELS:
        print(f"loading 300s panel levels={levels}", flush=True)
        ofi, ret, spr = load_panel(levels)
        for train in TRAINS:
            for horizon in HORIZONS:
                print(f"significance levels={levels} train={train} h={horizon}", flush=True)
                scores, sigs, ports = fit_family(ofi, ret, spr, horizon, train)
                for row in scores:
                    row["levels"] = levels
                for row in sigs:
                    row["levels"] = levels
                for row in ports:
                    row["levels"] = levels
                all_scores.extend(scores)
                all_sigs.extend(sigs)
                all_ports.extend(ports)
    pd.DataFrame(all_scores).to_csv(OUT / "significance_scores.csv", index=False)
    pd.DataFrame(all_sigs).to_csv(OUT / "significance_tests.csv", index=False)
    pd.DataFrame(all_ports).to_csv(OUT / "portfolio.csv", index=False)
    write_summary(all_scores, all_sigs, all_ports)
    print(OUT / "significance.md")


if __name__ == "__main__":
    main()
