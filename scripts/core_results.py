"""
Focused evidence pass for the Cross-Asset OFI crypto project.

Produces data coverage, PCA diagnostics, contemporaneous and predictive
cross-impact tables, lead-lag rankings, portfolio diagnostics and extension
checks under output/core/.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_sectional import (  # noqa: E402
    _lasso_wf,
    _pick_alpha,
    ablation,
    compare_estimators,
    fit_cross_impact,
    lead_lag_network,
)
from evaluation import forecast_portfolio, rolling_forecast_vol  # noqa: E402
from leadlag import CCZ_LAG_WINDOWS, fwd_return, make_lag_windows, score, sweep, walk_forward  # noqa: E402
from ofi import bar_l2_ofi, bucket_times, fit_integrated_ofi_weights  # noqa: E402

ASSETS = ["btc", "eth", "sol", "xrp"]
NAMES = [a.upper() for a in ASSETS]
PROCESSED = ROOT / "data" / "processed"
OUTDIR = ROOT / "output" / "core"
LEVELS = 10
BAR_S = 60
BAR_NS = BAR_S * 1_000_000_000
TRAIN = 3000
FOLDCAP = 60
COST_BP = 0.5
HORIZONS = [1, 2, 3, 5, 10]
LAGS = CCZ_LAG_WINDOWS


def clean_float(x):
    if x is None:
        return None
    try:
        y = float(x)
    except Exception:
        return x
    return None if not math.isfinite(y) else y


def metric_tuple(t):
    return {"r2": clean_float(t[0]), "hit": clean_float(t[1]), "n": int(t[2])}


def load_asset_raw(asset):
    files = sorted(PROCESSED.glob(f"{asset}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No processed files for {asset}")
    return files


def load_asset_bars(asset, levels=LEVELS, bar_ns=BAR_NS):
    parts = []
    pca_rows = []
    fitted_w = None
    first_evr = np.nan
    files = load_asset_raw(asset)
    for idx, path in enumerate(files):
        sym = dict(np.load(path))
        b_raw, fitted_w, evr = bar_l2_ofi(sym, levels=levels, bar_ns=bar_ns, weights=fitted_w)
        if idx == 0:
            first_evr = evr
        if levels >= 2:
            level_cols = [f"ofi_L{i}" for i in range(levels)]
            _, evr_diag = fit_integrated_ofi_weights(b_raw[level_cols].to_numpy())
        else:
            evr_diag = np.nan
        pca_rows.append({
            "asset": asset.upper(),
            "file": path.name,
            "pc1_evr_bar_norm": clean_float(evr_diag),
            "snapshots": int(len(sym["ts"])),
            "bars": int(len(b_raw)),
        })

        rel = (sym["ask_p"][:, 0] - sym["bid_p"][:, 0]) / (0.5 * (sym["ask_p"][:, 0] + sym["bid_p"][:, 0]))
        bucket = bucket_times(sym["ts"], bar_ns)
        sp = pd.Series(rel, index=bucket).groupby(level=0).mean().rename("spread")

        b_raw = b_raw.join(sp)
        parts.append(b_raw)
    bars = pd.concat(parts).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    return bars.dropna(subset=["ret", "spread"]), pca_rows, clean_float(first_evr)


def build_panel(bars_by_asset, scheme="integrated"):
    panel = None
    for asset in ASSETS:
        b = bars_by_asset[asset]
        if scheme == "best":
            ofi_col = "ofi_L0"
        elif scheme == "integrated":
            ofi_col = "ofi_I"
        else:
            raise ValueError(scheme)
        sub = b[[ofi_col, "ret", "spread"]].rename(columns={
            ofi_col: f"ofi_{asset}",
            "ret": f"ret_{asset}",
            "spread": f"spread_{asset}",
        })
        panel = sub if panel is None else panel.join(sub, how="inner")
    panel = panel.dropna()
    ofi = np.column_stack([panel[f"ofi_{a}"].to_numpy() for a in ASSETS])
    ret = np.column_stack([panel[f"ret_{a}"].to_numpy() for a in ASSETS])
    spr = np.column_stack([panel[f"spread_{a}"].to_numpy() for a in ASSETS])
    return panel, ofi, ret, spr


def data_coverage(bars_by_asset, aligned_panel):
    rows = []
    for asset in ASSETS:
        b = bars_by_asset[asset]
        rows.append({
            "asset": asset.upper(),
            "bars_60s": int(len(b)),
            "first_bar_utc": pd.to_datetime(int(b.index.min()), unit="ns", utc=True).isoformat(),
            "last_bar_utc": pd.to_datetime(int(b.index.max()), unit="ns", utc=True).isoformat(),
            "mean_spread_bps": clean_float(b["spread"].mean() * 1e4),
            "median_spread_bps": clean_float(b["spread"].median() * 1e4),
        })
    return rows, {
        "aligned_bars_60s": int(len(aligned_panel)),
        "first_aligned_utc": pd.to_datetime(int(aligned_panel.index.min()), unit="ns", utc=True).isoformat(),
        "last_aligned_utc": pd.to_datetime(int(aligned_panel.index.max()), unit="ns", utc=True).isoformat(),
    }


def summarize_fit(result):
    rows = []
    for name in result["names"]:
        pi = metric_tuple(result["pi"][name])
        ci = metric_tuple(result["ci"][name])
        rows.append({
            "asset": name,
            "pi_r2": pi["r2"],
            "ci_r2": ci["r2"],
            "ci_minus_pi_r2": clean_float((ci["r2"] or np.nan) - (pi["r2"] or np.nan)),
            "pi_hit": pi["hit"],
            "ci_hit": ci["hit"],
            "ci_minus_pi_hit": clean_float((ci["hit"] or np.nan) - (pi["hit"] or np.nan)),
            "n": ci["n"],
        })
    return rows


def run_cross_fit(ofi, ret, horizon, alpha_mode="fixed"):
    T = len(ret)
    step = max(500, T // FOLDCAP)
    alpha = 1e-8 if alpha_mode == "fixed" else None
    return fit_cross_impact(ofi, ret, NAMES, lag_windows=LAGS if horizon > 0 else (1,),
                            horizon=horizon, train=TRAIN, step=step, alpha=alpha)


def predictive_manual(ofi, ret, spr):
    T, N = ofi.shape
    step = max(500, T // FOLDCAP)
    rows = []
    forecasts_by_h = {}
    loss_models = {}
    for h in HORIZONS:
        F_cross = np.full((T, N), np.nan)
        F_own = np.full((T, N), np.nan)
        loss_models[h] = {}
        for i, name in enumerate(NAMES):
            y = fwd_return(ret[:, i], h)
            base = (np.arange(T) >= max(LAGS) - 1) & ~np.isnan(y)
            own_X = make_lag_windows(ofi[:, i], LAGS)
            all_X = np.hstack([make_lag_windows(ofi[:, j], LAGS) for j in range(N)])

            ok_own = base & ~np.isnan(own_X).any(axis=1)
            p_own = walk_forward(own_X[ok_own], y[ok_own], train=TRAIN, step=step, purge=h)
            sc_own = score(y[ok_own], p_own)
            idx_own = np.where(ok_own)[0]
            F_own[idx_own, i] = p_own

            ok_all = base & ~np.isnan(all_X).any(axis=1)
            Xc, yc = all_X[ok_all], y[ok_all]
            alpha = _pick_alpha(Xc[:TRAIN], yc[:TRAIN]) if len(yc) >= TRAIN + 10 else 1e-8
            p_cross, coef = _lasso_wf(Xc, yc, alpha=alpha, train=TRAIN, step=step, purge=h)
            sc_cross = score(yc, p_cross)
            idx_all = np.where(ok_all)[0]
            F_cross[idx_all, i] = p_cross

            rows.append({
                "horizon_bars": h,
                "horizon_seconds": h * BAR_S,
                "asset": name,
                "own_r2": clean_float(sc_own[0]),
                "cross_r2": clean_float(sc_cross[0]),
                "cross_minus_own_r2": clean_float(sc_cross[0] - sc_own[0]),
                "own_hit": clean_float(sc_own[1]),
                "cross_hit": clean_float(sc_cross[1]),
                "cross_minus_own_hit": clean_float(sc_cross[1] - sc_own[1]),
                "n": int(sc_cross[2]),
                "alpha": clean_float(alpha),
            })

            y_full = np.full(T, np.nan)
            y_full[:] = y
            loss_models[h][name] = {"own": F_own[:, i], "cross": F_cross[:, i], "realized": y_full}
        forecasts_by_h[h] = {"own": F_own, "cross": F_cross}
    return rows, forecasts_by_h, loss_models


def portfolio_rows(forecasts_by_h, ret, spr):
    rows = []
    for h, fs in forecasts_by_h.items():
        realized = np.column_stack([fwd_return(ret[:, i], h) for i in range(ret.shape[1])])
        sig = rolling_forecast_vol(fs["cross"], window=max(30, min(180, len(ret) // 50)))
        for c in [0.0, COST_BP, 1.0]:
            res = forecast_portfolio(fs["cross"], realized, spr, sig, cost_bps=c,
                                     periods_per_year=365 * 24 * 3600 / (BAR_S * h))
            rows.append({
                "horizon_bars": h,
                "horizon_seconds": h * BAR_S,
                "cost_bps": c,
                "cum_net": clean_float(res["cum_net"]),
                "cum_gross": clean_float(res["cum_gross"]),
                "mean_bps": clean_float(res["mean_bps"]),
                "hit_rate": clean_float(res["hit_rate"]),
                "frac_traded": clean_float(res["frac_traded"]),
                "avg_turnover": clean_float(res["avg_turnover"]),
                "sharpe_annual": clean_float(res.get("sharpe_annual")),
                "periods": int(res["periods"]),
            })
    return rows


def dm_rows(loss_models):
    from significance_tests import diebold_mariano, model_confidence_set, squared_error
    rows = []
    for h, by_asset in loss_models.items():
        for asset, d in by_asset.items():
            own_loss = squared_error(d["own"], d["realized"])
            cross_loss = squared_error(d["cross"], d["realized"])
            stat, p = diebold_mariano(own_loss, cross_loss, horizon=h)
            mcs_included = None
            mcs_error = None
            try:
                losses = pd.DataFrame({"own": own_loss, "cross": cross_loss}).dropna()
                mcs = model_confidence_set(losses, alpha=0.10, reps=300,
                                           block_size=max(5, h), seed=h * 100 + len(asset))
                mcs_included = ",".join(mcs["included"])
            except Exception as exc:
                mcs_error = repr(exc)
            rows.append({
                "horizon_bars": h,
                "asset": asset,
                "dm_own_minus_cross": clean_float(stat),
                "p_value": clean_float(p),
                "mcs_included_10pct": mcs_included,
                "mcs_error": mcs_error,
            })
    return rows


def lead_lag_rows(result):
    L = result["Lambda"]
    rows = []
    for i, target in enumerate(NAMES):
        for j, source in enumerate(NAMES):
            rows.append({"target": target, "source": source, "coef_sum": clean_float(L[i, j])})
    hub = [{"source": k, "out_strength": clean_float(v)} for k, v in lead_lag_network(L, NAMES).items()]
    return rows, hub


def extension_rows(ofi, ret):
    rows = []
    T = len(ret)
    step = max(500, T // FOLDCAP)
    pair_checks = [("BTC", "ETH"), ("BTC", "SOL"), ("BTC", "XRP"), ("ETH", "SOL"), ("ETH", "XRP")]
    for src, tgt in pair_checks:
        si, ti = NAMES.index(src), NAMES.index(tgt)
        res = sweep(ofi[:, si], ofi[:, ti], ret[:, ti], horizons=[1, 2, 5],
                    lag_windows=LAGS, rho=0.6, train=TRAIN, step=step)
        for h in [1, 2, 5]:
            rows.append({
                "extension": "propagator",
                "source": src,
                "target": tgt,
                "horizon_bars": h,
                "own_r2": clean_float(res["own"][h][0]),
                "free_r2": clean_float(res["cross_free"][h][0]),
                "propagator_r2": clean_float(res["cross_prop"][h][0]),
                "free_minus_own_r2": clean_float(res["cross_free"][h][0] - res["own"][h][0]),
                "prop_minus_own_r2": clean_float(res["cross_prop"][h][0] - res["own"][h][0]),
                "free_hit": clean_float(res["cross_free"][h][1]),
                "prop_hit": clean_float(res["cross_prop"][h][1]),
            })
    try:
        xgb = compare_estimators(ofi, ret, NAMES, target="SOL", lag_windows=LAGS,
                                 horizon=1, train=TRAIN, step=max(step, 1200), alpha=None)
        for model, tup in xgb.items():
            rows.append({
                "extension": "xgboost_check",
                "source": "ALL",
                "target": "SOL",
                "horizon_bars": 1,
                "model": model,
                "r2": clean_float(tup[0]),
                "hit": clean_float(tup[1]),
                "n": int(tup[2]),
            })
    except Exception as exc:
        rows.append({"extension": "xgboost_check", "source": "ALL", "target": "SOL",
                     "horizon_bars": 1, "error": repr(exc)})
    return rows


def ablation_rows(ofi, ret):
    rows = []
    T = len(ret)
    step = max(500, T // FOLDCAP)
    for target in NAMES:
        try:
            res = ablation(ofi, ret, NAMES, target=target, lag_windows=LAGS, horizon=1,
                           train=TRAIN, step=step, alpha=None)
            own = res["own"][0]
            for spec, tup in res.items():
                rows.append({
                    "target": target,
                    "spec": spec,
                    "r2": clean_float(tup[0]),
                    "hit": clean_float(tup[1]),
                    "n": int(tup[2]),
                    "r2_minus_own": clean_float(tup[0] - own),
                })
        except Exception as exc:
            rows.append({"target": target, "spec": "error", "error": repr(exc)})
    return rows


def write_outputs(tables, summary):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUTDIR / f"{name}.csv", index=False)
    with open(OUTDIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "tables": tables}, fh, indent=2)

    pred = pd.DataFrame(tables["predictive"])
    cont = pd.DataFrame(tables["contemporaneous"])
    port = pd.DataFrame(tables["portfolio"])
    hub = pd.DataFrame(tables["hub"])
    pca = pd.DataFrame(tables["pca"])
    if len(pca):
        pca_summary = (pca.groupby("asset")["pc1_evr_bar_norm"]
                       .agg(["mean", "min", "max", "count"])
                       .reset_index())
    else:
        pca_summary = pd.DataFrame()
    dm = pd.DataFrame(tables["dm"])
    ext = pd.DataFrame(tables["extensions"])
    lines = []
    lines.append("# Core Results")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()}")
    lines.append(f"Spec: {BAR_S}s bars, assets {', '.join(NAMES)}, top {LEVELS} levels, CCZ lag windows {list(LAGS)}.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"Aligned bars: {summary['aligned']['aligned_bars_60s']} from {summary['aligned']['first_aligned_utc']} to {summary['aligned']['last_aligned_utc']}.")
    lines.append("")
    lines.append("## PCA Depth Diagnostic")
    lines.append(pca_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Contemporaneous")
    lines.append(cont.to_markdown(index=False))
    lines.append("")
    lines.append("## Predictive")
    lines.append(pred.to_markdown(index=False))
    lines.append("")
    lines.append("## Lead-Lag Hub")
    lines.append(hub.to_markdown(index=False))
    lines.append("")
    lines.append("## DM And MCS")
    lines.append(dm.to_markdown(index=False))
    lines.append("")
    lines.append("## Extensions")
    lines.append(ext.to_markdown(index=False))
    lines.append("")
    lines.append("## Portfolio")
    lines.append(port.to_markdown(index=False))
    lines.append("")
    with open(OUTDIR / "results.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main():
    t0 = time.time()
    print("loading bars...", flush=True)
    bars_by_asset = {}
    pca_rows = []
    first_day_evr = []
    for asset in ASSETS:
        b, rows, evr = load_asset_bars(asset)
        bars_by_asset[asset] = b
        pca_rows.extend(rows)
        first_day_evr.append({"asset": asset.upper(), "first_day_raw_pc1_evr": evr})
        print(f"  {asset}: {len(b)} bars", flush=True)

    panel_i, ofi_i, ret_i, spr = build_panel(bars_by_asset, "integrated")
    panel_b, ofi_b, ret_b, _ = build_panel(bars_by_asset, "best")
    coverage, aligned = data_coverage(bars_by_asset, panel_i)
    print(f"aligned bars: {len(panel_i)}", flush=True)

    print("contemporaneous fits...", flush=True)
    cont_rows = []
    for scheme, ofi in [("best", ofi_b), ("integrated", ofi_i)]:
        res = run_cross_fit(ofi, ret_i, horizon=0, alpha_mode="fixed")
        for row in summarize_fit(res):
            row["scheme"] = scheme
            row["horizon_bars"] = 0
            cont_rows.append(row)

    print("predictive fits...", flush=True)
    pred_rows, forecasts_by_h, loss_models = predictive_manual(ofi_i, ret_i, spr)
    print("portfolio and significance...", flush=True)
    port_rows = portfolio_rows(forecasts_by_h, ret_i, spr)
    dm = dm_rows(loss_models)

    print("lead-lag matrix...", flush=True)
    lead_res = run_cross_fit(ofi_i, ret_i, horizon=1, alpha_mode="fixed")
    matrix_rows, hub_rows = lead_lag_rows(lead_res)

    print("ablation and extensions...", flush=True)
    abl_rows = ablation_rows(ofi_i, ret_i)
    ext_rows = extension_rows(ofi_i, ret_i)

    tables = {
        "coverage": coverage,
        "pca": pca_rows,
        "first_day_pca": first_day_evr,
        "contemporaneous": cont_rows,
        "predictive": pred_rows,
        "portfolio": port_rows,
        "dm": dm,
        "matrix": matrix_rows,
        "hub": hub_rows,
        "ablation": abl_rows,
        "extensions": ext_rows,
    }
    summary = {
        "bar_seconds": BAR_S,
        "assets": NAMES,
        "levels": LEVELS,
        "train_bars": TRAIN,
        "lag_windows": list(LAGS),
        "aligned": aligned,
        "elapsed_seconds": clean_float(time.time() - t0),
    }
    write_outputs(tables, summary)
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    print(OUTDIR / "results.md")


if __name__ == "__main__":
    main()
