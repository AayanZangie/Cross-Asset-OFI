"""Robustness battery for the 300s/CCZ cross-OFI candidate.

Runs, in order:
1. Day-shifted cross-feature placebo for the 28 broad-grid 300s/CCZ cells.
2. DM/MCS on the same 28 real cells, with p-values corrected for the full
   672-cell discovery grid.
3. Pre-declared horizon extension for the PCA/CCZ family.
4. Pre-declared lag-window/lookback alternatives for the PCA family.
"""

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


OUT = ROOT / "output" / "robustness_300s"
BAR_S = 300
DAY_BARS = 24 * 60 * 60 // BAR_S
SEARCH_M = 672
SCHEMES = ["best", "sum", "distance", "pca"]
CCZ = [1, 2, 3, 5, 10, 20, 30]
GRID_HORIZONS = [1, 2, 3, 5, 10, 20, 30]
EXTENDED_HORIZONS = [40, 60, 90, 120]
LOOKBACKS = {
    "ccz_short": [1, 2, 3, 5],
    "ccz_dense": [1, 2, 3, 5, 10, 15, 20, 25, 30],
    "ccz_long": [1, 2, 3, 5, 10, 20, 30, 45, 60, 90, 120],
}
COSTS = [0.0, 0.25, 0.5, 1.0]


def finite(x):
    try:
        y = float(x)
    except Exception:
        return None
    return y if np.isfinite(y) else None


def shift_back_one_day(x):
    y = np.full_like(x, np.nan, dtype=float)
    y[DAY_BARS:] = x[:-DAY_BARS]
    return y


def load_bars(levels, cache):
    if levels in cache:
        return cache[levels]
    old_levels = grid.LEVELS
    grid.LEVELS = levels
    bars = {a: grid.load_levels_bars(a, BAR_S * 1_000_000_000, levels=levels)
            for a in grid.ASSETS}
    grid.LEVELS = old_levels
    cache[levels] = bars
    return bars


def build_arrays(levels, scheme, cache):
    bars = load_bars(levels, cache)
    old_levels = grid.LEVELS
    grid.LEVELS = levels
    ofi, ret, spr = grid.build_scheme_panel(bars, scheme)
    grid.LEVELS = old_levels
    return ofi, ret, spr


def average_loss(pred, realized):
    loss = squared_error(pred, realized)
    valid = ~np.isnan(loss)
    denom = valid.sum(axis=1)
    total = np.nansum(loss, axis=1)
    out = np.full(loss.shape[0], np.nan)
    np.divide(total, denom, out=out, where=denom > 0)
    return out


def eval_cell(ofi, ret, spr, lagset, horizon, train, refit_cap, placebo=False):
    t, n = ofi.shape
    step = max(500, t // refit_cap)
    max_lag = max(lagset)
    ret_blocks = [make_lag_windows(ret[:, j], lagset) for j in range(n)]
    real_ofi_blocks = [make_lag_windows(ofi[:, j], lagset) for j in range(n)]
    shifted_blocks = [make_lag_windows(shift_back_one_day(ofi[:, j]), lagset) for j in range(n)]
    all_ret = np.hstack(ret_blocks)

    f_own = np.full((t, n), np.nan)
    f_car = np.full((t, n), np.nan)
    f_cross = np.full((t, n), np.nan)
    realized = np.column_stack([fwd_return(ret[:, i], horizon) for i in range(n)])
    asset_rows = []

    for i, name in enumerate(grid.NAMES):
        y = realized[:, i]
        base = (np.arange(t) >= max_lag - 1) & ~np.isnan(y)

        own_x = real_ofi_blocks[i]
        ok_own = base & ~np.isnan(own_x).any(axis=1)
        p_own = walk_forward(own_x[ok_own], y[ok_own], train=train, step=step, purge=horizon)
        f_own[np.where(ok_own)[0], i] = p_own

        ok_car = base & ~np.isnan(all_ret).any(axis=1)
        a_car = _pick_alpha(all_ret[ok_car][:train], y[ok_car][:train])
        p_car, _ = _lasso_wf(all_ret[ok_car], y[ok_car], a_car, train, step, purge=horizon)
        f_car[np.where(ok_car)[0], i] = p_car

        blocks = []
        for j in range(n):
            if j == i:
                blocks.append(real_ofi_blocks[j])
            else:
                blocks.append(shifted_blocks[j] if placebo else real_ofi_blocks[j])
        all_ofi = np.hstack(blocks)
        ok_cross = base & ~np.isnan(all_ofi).any(axis=1)
        a_cross = _pick_alpha(all_ofi[ok_cross][:train], y[ok_cross][:train])
        p_cross, _ = _lasso_wf(all_ofi[ok_cross], y[ok_cross], a_cross, train, step, purge=horizon)
        f_cross[np.where(ok_cross)[0], i] = p_cross

        sc_own = score(y, f_own[:, i])
        sc_car = score(y, f_car[:, i])
        sc_cross = score(y, f_cross[:, i])
        asset_rows.append({
            "asset": name,
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

    sig = rolling_forecast_vol(f_cross, window=max(30, min(180, t // 50)))
    pnl = {c: forecast_portfolio(f_cross, realized, spr, sig, cost_bps=c) for c in COSTS}
    losses = pd.DataFrame({
        "own": average_loss(f_own, realized),
        "car": average_loss(f_car, realized),
        "cross": average_loss(f_cross, realized),
    }).dropna()

    own_r2 = np.nanmean([r["own_r2"] for r in asset_rows])
    car_r2 = np.nanmean([r["car_r2"] for r in asset_rows])
    cross_r2 = np.nanmean([r["cross_r2"] for r in asset_rows])
    own_hit = np.nanmean([r["own_hit"] for r in asset_rows])
    car_hit = np.nanmean([r["car_hit"] for r in asset_rows])
    cross_hit = np.nanmean([r["cross_hit"] for r in asset_rows])
    best = max(asset_rows, key=lambda r: -np.inf if r["cross_minus_own_r2"] is None else r["cross_minus_own_r2"])

    out = {
        "own_r2": finite(own_r2),
        "car_r2": finite(car_r2),
        "cross_r2": finite(cross_r2),
        "cross_minus_own_r2": finite(cross_r2 - own_r2),
        "cross_minus_car_r2": finite(cross_r2 - car_r2),
        "own_hit": finite(own_hit),
        "car_hit": finite(car_hit),
        "cross_hit": finite(cross_hit),
        "pnl_net": finite(pnl[0.5]["cum_net"]),
        "pnl_gross": finite(pnl[0.0]["cum_gross"]),
        "pnl_net_0p25bp": finite(pnl[0.25]["cum_net"]),
        "pnl_net_0p5bp": finite(pnl[0.5]["cum_net"]),
        "pnl_net_1p0bp": finite(pnl[1.0]["cum_net"]),
        "pnl_hit": finite(pnl[0.5]["hit_rate"]),
        "turnover": finite(pnl[0.5]["avg_turnover"]),
        "frac_traded": finite(pnl[0.5]["frac_traded"]),
        "best_asset": best["asset"],
        "best_asset_cross_minus_own_r2": best["cross_minus_own_r2"],
        "asset_metrics_json": json.dumps(asset_rows, separators=(",", ":")),
    }
    return out, losses


def corrected_pvalues(df, pcol, m=SEARCH_M):
    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(float)
    out_bonf = np.minimum(p * m, 1.0)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    holm = np.full(len(p), np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)
        holm[idx] = running
    bh = np.full(len(p), np.nan)
    running = 1.0
    finite_order = [idx for idx in order if np.isfinite(p[idx])]
    for rank_from_end, idx in enumerate(reversed(finite_order), start=1):
        rank = len(finite_order) - rank_from_end + 1
        val = min(p[idx] * m / rank, 1.0)
        running = min(running, val)
        bh[idx] = running
    return out_bonf, holm, bh


def run_placebo_and_significance(cache):
    rows = []
    sig_rows = []
    loss_store = {}
    for scheme in SCHEMES:
        ofi, ret, spr = build_arrays(10, scheme, cache)
        for horizon in GRID_HORIZONS:
            print(f"placebo/significance scheme={scheme} h={horizon}", flush=True)
            real, real_losses = eval_cell(ofi, ret, spr, CCZ, horizon, train=3000, refit_cap=80, placebo=False)
            plac, plac_losses = eval_cell(ofi, ret, spr, CCZ, horizon, train=3000, refit_cap=80, placebo=True)
            base = {
                "bar_s": BAR_S,
                "scheme": scheme,
                "lagset": "ccz",
                "horizon": horizon,
                "horizon_seconds": horizon * BAR_S,
                "train": 3000,
                "levels": 10,
            }
            rows.append({**base, "kind": "real", **real})
            rows.append({**base, "kind": "placebo", **plac})
            rows.append({
                **base,
                "kind": "real_minus_placebo",
                "cross_minus_own_r2": finite(real["cross_minus_own_r2"] - plac["cross_minus_own_r2"]),
                "cross_minus_car_r2": finite(real["cross_minus_car_r2"] - plac["cross_minus_car_r2"]),
                "cross_hit": finite(real["cross_hit"] - plac["cross_hit"]),
                "pnl_net": finite(real["pnl_net"] - plac["pnl_net"]),
            })

            dm_own, p_own = diebold_mariano(real_losses["own"], real_losses["cross"], horizon=horizon)
            dm_car, p_car = diebold_mariano(real_losses["car"], real_losses["cross"], horizon=horizon)
            joined_cross = pd.DataFrame({
                "placebo_cross": plac_losses["cross"],
                "real_cross": real_losses["cross"],
            }).dropna()
            dm_placebo, p_placebo = diebold_mariano(joined_cross["placebo_cross"],
                                                    joined_cross["real_cross"],
                                                    horizon=horizon)
            try:
                mcs = model_confidence_set(real_losses[["own", "car", "cross"]], alpha=0.10,
                                           reps=300, block_size=max(5, horizon),
                                           seed=1000 + horizon + 17 * SCHEMES.index(scheme))
                mcs_included = ",".join(mcs["included"])
            except Exception as exc:
                mcs_included = f"ERROR:{exc!r}"
            sig_rows.append({
                **base,
                "dm_own_minus_cross": finite(dm_own),
                "p_own_vs_cross": finite(p_own),
                "dm_car_minus_cross": finite(dm_car),
                "p_car_vs_cross": finite(p_car),
                "dm_placebo_cross_minus_real_cross": finite(dm_placebo),
                "p_placebo_vs_real_cross": finite(p_placebo),
                "mcs_included_10pct": mcs_included,
            })
            loss_store[(scheme, horizon)] = (real_losses, plac_losses)
    sig = pd.DataFrame(sig_rows)
    for pcol in ["p_own_vs_cross", "p_car_vs_cross", "p_placebo_vs_real_cross"]:
        bonf, holm, bh = corrected_pvalues(sig, pcol)
        sig[f"{pcol}_bonf672"] = bonf
        sig[f"{pcol}_holm672"] = holm
        sig[f"{pcol}_bh672"] = bh
    return pd.DataFrame(rows), sig


def run_horizon_extension(cache):
    rows = []
    for levels in [10, 20]:
        ofi, ret, spr = build_arrays(levels, "pca", cache)
        for train in [3000, 6000]:
            for horizon in EXTENDED_HORIZONS:
                for placebo in [False, True]:
                    print(f"extended levels={levels} train={train} h={horizon} placebo={placebo}", flush=True)
                    res, _ = eval_cell(ofi, ret, spr, CCZ, horizon, train=train,
                                       refit_cap=200, placebo=placebo)
                    rows.append({
                        "bar_s": BAR_S,
                        "scheme": "pca",
                        "lagset": "ccz",
                        "horizon": horizon,
                        "horizon_seconds": horizon * BAR_S,
                        "train": train,
                        "levels": levels,
                        "kind": "placebo" if placebo else "real",
                        **res,
                    })
    return pd.DataFrame(rows)


def run_lookback_extension(cache):
    rows = []
    for lagname, lagset in LOOKBACKS.items():
        ofi, ret, spr = build_arrays(10, "pca", cache)
        for train in [3000, 6000]:
            for horizon in [2, 5, 10, 20]:
                for placebo in [False, True]:
                    print(f"lookback {lagname} train={train} h={horizon} placebo={placebo}", flush=True)
                    res, _ = eval_cell(ofi, ret, spr, lagset, horizon, train=train,
                                       refit_cap=200, placebo=placebo)
                    rows.append({
                        "bar_s": BAR_S,
                        "scheme": "pca",
                        "lagset": lagname,
                        "lag_windows": ",".join(map(str, lagset)),
                        "horizon": horizon,
                        "horizon_seconds": horizon * BAR_S,
                        "train": train,
                        "levels": 10,
                        "kind": "placebo" if placebo else "real",
                        **res,
                    })
    return pd.DataFrame(rows)


def pivot_real_placebo(df):
    metric_cols = ["cross_minus_own_r2", "cross_minus_car_r2", "cross_hit", "pnl_net"]
    keys = [c for c in ["bar_s", "scheme", "lagset", "horizon", "horizon_seconds", "train", "levels"] if c in df.columns]
    real = df[df["kind"] == "real"][keys + metric_cols].copy()
    plac = df[df["kind"] == "placebo"][keys + metric_cols].copy()
    real = real.rename(columns={c: f"real_{c}" for c in metric_cols})
    plac = plac.rename(columns={c: f"placebo_{c}" for c in metric_cols})
    out = real.merge(plac, on=keys, how="inner")
    for c in metric_cols:
        out[f"delta_{c}"] = out[f"real_{c}"] - out[f"placebo_{c}"]
    return out


def write_summary(placebo, sig, extended, lookback):
    lines = ["# Robustness: 300s CCZ Candidate", ""]
    lines.append("Pre-declared checks: day-shifted cross-feature placebo; DM/MCS with p-values corrected for the 672-cell search; horizon extension {40,60,90,120}; lookback alternatives ccz_short, ccz_dense, ccz_long.")
    lines.append("")

    lines.append("## Placebo: 300s/CCZ 28 Cells")
    pp = pivot_real_placebo(placebo)
    cols = ["scheme", "horizon", "horizon_seconds",
            "real_cross_minus_own_r2", "placebo_cross_minus_own_r2", "delta_cross_minus_own_r2",
            "real_cross_hit", "placebo_cross_hit", "delta_cross_hit",
            "real_pnl_net", "placebo_pnl_net", "delta_pnl_net"]
    lines.append(pp[cols].to_markdown(index=False))
    lines.append("")
    killed = pp[(pp["placebo_cross_minus_own_r2"] >= pp["real_cross_minus_own_r2"]) |
                (pp["placebo_pnl_net"] >= pp["real_pnl_net"])]
    lines.append(f"Placebo warning cells where placebo >= real on R2 or PnL: {len(killed)} / {len(pp)}.")
    lines.append("")

    lines.append("## Grid-Corrected Significance")
    sig_cols = ["scheme", "horizon", "horizon_seconds",
                "dm_own_minus_cross", "p_own_vs_cross", "p_own_vs_cross_bonf672", "p_own_vs_cross_bh672",
                "dm_car_minus_cross", "p_car_vs_cross", "p_car_vs_cross_bonf672", "p_car_vs_cross_bh672",
                "dm_placebo_cross_minus_real_cross", "p_placebo_vs_real_cross",
                "p_placebo_vs_real_cross_bonf672", "mcs_included_10pct"]
    lines.append(sig[sig_cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Horizon Extension")
    ep = pivot_real_placebo(extended)
    lines.append(ep[cols].to_markdown(index=False))
    lines.append("")

    lines.append("## Lookback Alternatives")
    lp = pivot_real_placebo(lookback)
    look_cols = ["lagset", "horizon", "horizon_seconds", "train",
                 "real_cross_minus_own_r2", "placebo_cross_minus_own_r2",
                 "real_cross_minus_car_r2", "placebo_cross_minus_car_r2",
                 "real_cross_hit", "placebo_cross_hit", "real_pnl_net", "placebo_pnl_net"]
    lines.append(lp[look_cols].to_markdown(index=False))
    lines.append("")

    OUT.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    placebo, sig = run_placebo_and_significance(cache)
    extended = run_horizon_extension(cache)
    lookback = run_lookback_extension(cache)

    placebo.to_csv(OUT / "placebo.csv", index=False)
    sig.to_csv(OUT / "significance.csv", index=False)
    extended.to_csv(OUT / "extended_horizons.csv", index=False)
    lookback.to_csv(OUT / "lookbacks.csv", index=False)
    write_summary(placebo, sig, extended, lookback)
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
