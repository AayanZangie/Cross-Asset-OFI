"""
tests.py - exhaustive search for predictive power of OFI on future returns.

Searches bar size, lag set, horizon, and OFI definition for own-asset and
cross-asset predictive models. Results are written incrementally so long runs
can resume without rerunning completed configurations.

Run from the repo root:
    python src/tests.py
    python src/tests.py --spec subminute
    python src/tests.py --spec subminute_decay
    python src/tests.py --smoke
"""

import os
import sys
import glob
import time
import traceback
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from ofi import bar_l2_ofi, bucket_times, compute_ofi
from leadlag import (SINGLE_BAR_WINDOWS, fwd_return, make_lag_windows, score,
                     tuned_ridge_predict, walk_forward)
from panel import build_second_panel as build_subminute_second_panel
from cross_sectional import _lasso_wf, _pick_alpha
from evaluation import forecast_portfolio, rolling_forecast_vol, unit_position_pnl_bps

SMOKE = "--smoke" in sys.argv
DENSE_REFIT = "--dense-refit" in sys.argv
DEEP_LEVELS = "--deep-levels" in sys.argv
SUBMINUTE = "--subminute" in sys.argv or (
    "--spec" in sys.argv and sys.argv.index("--spec") + 1 < len(sys.argv)
    and sys.argv[sys.argv.index("--spec") + 1] == "subminute"
)
SUBMINUTE_DECAY = "--subminute-decay" in sys.argv or (
    "--spec" in sys.argv and sys.argv.index("--spec") + 1 < len(sys.argv)
    and sys.argv[sys.argv.index("--spec") + 1] == "subminute_decay"
)
FORCE_SUBMINUTE = "--force-subminute" in sys.argv
FORCE_SUBMINUTE_DECAY = "--force-subminute-decay" in sys.argv

# ------------------------------------------------------------------ config
if SMOKE:
    PROCESSED = "data/ptest"
    ASSETS    = ["btc", "eth"]
    BARS_S    = [15, 60]
    SCHEMES   = ["best", "pca"]
    LAGSETS   = {"single": [1], "high": [1, 2, 3, 5, 10]}
    HORIZONS  = [1, 2, 5]
    RESULTS   = os.path.join("output", "grid", "smoke_results.csv")
else:
    PROCESSED = "data/processed"
    ASSETS    = ["btc", "eth", "sol", "xrp"]
    BARS_S    = [5, 10, 15, 30, 60, 300]                   # scale: 5 s .. 5 min bars
    SCHEMES   = ["best", "sum", "distance", "pca"]         # OFI definition / depth handling
    LAGSETS   = {"single": [1],                            # lag window
                 "low":  [1, 2, 3],
                 "mid":  [1, 2, 3, 5, 10],
                 "ccz":  [1, 2, 3, 5, 10, 20, 30]}
    HORIZONS  = [1, 2, 3, 5, 10, 20, 30]                   # forecast horizon, in bars
    RESULTS   = os.path.join("output", "grid", "results.csv")

NAMES   = [a.upper() for a in ASSETS]
LEVELS  = 20 if DEEP_LEVELS else 10
TRAIN   = 3000               # walk-forward training window (bars)
FOLDCAP = 200 if DENSE_REFIT else 80
COST_BP = 0.5                # cost level for the net-PnL column
COST_GRID_BP = [0.0, 0.25, 0.5, 1.0]
SUBMINUTE_BARS_S = [1, 5, 10]
SUBMINUTE_DECAY_BARS_S = list(range(1, 31))
SUBMINUTE_TRAIN_SEC = 7 * 24 * 60 * 60
SUBMINUTE_TEST_SEC = 24 * 60 * 60
SUBMINUTE_STEP_SEC = 7 * 24 * 60 * 60
SUBMINUTE_RESULTS = os.path.join("output", "subminute", "results.csv")
SUBMINUTE_DECAY_RESULTS = os.path.join("output", "subminute_decay", "results.csv")
RUN_TOTAL_TO_DO = 0
RUN_DONE_THIS_SESSION = 0
RUN_START_TS = None


# ------------------------------------------------------------------ helpers
def make_lags_set(x, lagset):
    """Columns are trailing-window sums ending at t for each k in lagset."""
    return make_lag_windows(x, lagset)


def fmt_duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "estimating"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def paper_key(bar_s, horizon):
    return f"paper|L{LEVELS}|R{FOLDCAP}|bar{bar_s}|pca|paperlags|h{horizon}"


def grid_key(bar_s, scheme, lagname, horizon):
    return f"grid|L{LEVELS}|R{FOLDCAP}|bar{bar_s}|{scheme}|{lagname}|h{horizon}"


def subminute_key(bar_s, scheme):
    return f"subminute|L{LEVELS}|bar{bar_s}|{scheme}|single_bar|h1"


def subminute_decay_key(bar_s, scheme):
    return f"subminute_decay|L{LEVELS}|bar{bar_s}|{scheme}|single_bar|h1"


def load_levels_bars(asset, bar_ns, levels=LEVELS):
    """Per-asset bar table with per-level OFI (ofi_L*), ret and relative spread."""
    files = sorted(glob.glob(os.path.join(PROCESSED, f"{asset}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {asset}_*.npz in {PROCESSED}")
    parts = []
    fitted_w = None
    for f in files:
        sym = dict(np.load(f))
        b, fitted_w, _ = bar_l2_ofi(sym, levels=levels, bar_ns=bar_ns, weights=fitted_w)
        rel = (sym["ask_p"][:, 0] - sym["bid_p"][:, 0]) / \
              (0.5 * (sym["ask_p"][:, 0] + sym["bid_p"][:, 0]))
        bucket = bucket_times(sym["ts"], bar_ns)
        sp = pd.Series(rel, index=bucket).groupby(level=0).mean().rename("spread")
        parts.append(b.join(sp))
    bars = pd.concat(parts).sort_index()
    bars = bars[~bars.index.duplicated()].dropna(subset=["ret"])
    return bars


def scheme_series(bars, scheme, levels=LEVELS):
    """Collapse per-level OFI into one series under the chosen definition."""
    if scheme == "pca" and "ofi_I" in bars.columns:
        return bars["ofi_I"].to_numpy()
    cols = [f"ofi_L{m}" for m in range(levels) if f"ofi_L{m}" in bars.columns]
    if not cols:                                   # levels==1 path stored 'ofi'
        return bars["ofi"].to_numpy()
    M = bars[cols].to_numpy()
    if scheme == "best":
        return M[:, 0]
    if scheme == "sum":                            # equal-weight (== mean for prediction)
        return np.nansum(M, axis=1)
    if scheme == "distance":                       # deeper levels down-weighted
        w = 1.0 / (1.0 + np.arange(M.shape[1]))
        return M @ (w / w.sum())
    if scheme == "pca":                            # fallback for externally supplied bars
        Xc = np.nan_to_num(M - np.nanmean(M, axis=0))
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        w = Vt[0]
        if w.sum() < 0:
            w = -w
        return M @ (w / np.abs(w).sum())
    raise ValueError(scheme)


def build_scheme_panel(bars_by_asset, scheme):
    """Align all assets on the shared bar grid for one OFI scheme."""
    sub = None
    for a in ASSETS:
        b = bars_by_asset[a]
        s = pd.DataFrame({f"ofi_{a}": scheme_series(b, scheme),
                          f"ret_{a}": b["ret"].to_numpy(),
                          f"spread_{a}": b["spread"].to_numpy()}, index=b.index)
        sub = s if sub is None else sub.join(s, how="inner")
    sub = sub.dropna()
    ofi = np.column_stack([sub[f"ofi_{a}"] for a in ASSETS])
    ret = np.column_stack([sub[f"ret_{a}"] for a in ASSETS])
    spr = np.column_stack([sub[f"spread_{a}"] for a in ASSETS])
    return ofi, ret, spr


def finite_or_none(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def finite_diff_or_none(a, b):
    if a is None or b is None:
        return None
    return finite_or_none(float(a) - float(b))


def eval_config(ofi, ret, spr, lagset, horizon):
    """Own/cross OFI plus return-history baselines; aggregate metrics + PnL."""
    N = ofi.shape[1]
    T = len(ret)
    step = max(500, T // FOLDCAP)
    Lmax = max(lagset)
    F = np.full((T, N), np.nan)            # cross forecasts, for the PnL book
    ofi_blocks = [make_lags_set(ofi[:, j], lagset) for j in range(N)]
    ret_blocks = [make_lags_set(ret[:, j], lagset) for j in range(N)]
    all_ofi = np.hstack(ofi_blocks)
    all_ret = np.hstack(ret_blocks)
    own_r2, own_hit, cr_r2, cr_hit = [], [], [], []
    ar_r2, ar_hit, car_r2, car_hit = [], [], [], []
    asset_metrics = []
    for ti in range(N):
        y = fwd_return(ret[:, ti], horizon)
        base = (np.arange(T) >= Lmax - 1) & ~np.isnan(y)
        own_X = ofi_blocks[ti]

        # Own-return features.
        ar_X = ret_blocks[ti]
        oka = base & ~np.isnan(ar_X).any(1)
        pa = walk_forward(ar_X[oka], y[oka], train=TRAIN, step=step, purge=horizon)
        r2a, ha, _ = score(y[oka], pa)

        # Cross-return features.
        okcar = base & ~np.isnan(all_ret).any(1)
        acar = _pick_alpha(all_ret[okcar][:TRAIN], y[okcar][:TRAIN])
        pcar, _ = _lasso_wf(all_ret[okcar], y[okcar], acar, TRAIN, step, purge=horizon)
        r2car, hcar, _ = score(y[okcar], pcar)

        # Own-OFI features.
        oko = base & ~np.isnan(own_X).any(1)
        po = walk_forward(own_X[oko], y[oko], train=TRAIN, step=step, purge=horizon)
        r2o, ho, _ = score(y[oko], po)

        # Cross-OFI features.
        okc = base & ~np.isnan(all_ofi).any(1)
        a = _pick_alpha(all_ofi[okc][:TRAIN], y[okc][:TRAIN])
        pc, _ = _lasso_wf(all_ofi[okc], y[okc], a, TRAIN, step, purge=horizon)
        r2c, hc, _ = score(y[okc], pc)
        ar_r2.append(r2a); ar_hit.append(ha); car_r2.append(r2car); car_hit.append(hcar)
        own_r2.append(r2o); own_hit.append(ho); cr_r2.append(r2c); cr_hit.append(hc)
        fcol = np.full(T, np.nan); fcol[np.where(okc)[0]] = pc; F[:, ti] = fcol
        asset_metrics.append({
            "asset": NAMES[ti],
            "ar_r2": finite_or_none(r2a),
            "car_r2": finite_or_none(r2car),
            "own_r2": finite_or_none(r2o),
            "cross_r2": finite_or_none(r2c),
            "cross_minus_own_r2": finite_or_none(r2c - r2o),
            "ar_hit": finite_or_none(ha),
            "car_hit": finite_or_none(hcar),
            "own_hit": finite_or_none(ho),
            "cross_hit": finite_or_none(hc),
        })

    Rf = np.column_stack([fwd_return(ret[:, i], horizon) for i in range(N)])
    sig = rolling_forecast_vol(F, window=max(30, min(180, T // 50)))
    pnl = {c: forecast_portfolio(F, Rf, spr, sig, cost_bps=c) for c in COST_GRID_BP}
    gross = pnl[0.0]
    net = pnl[COST_BP]

    gains = np.array([m["cross_minus_own_r2"] for m in asset_metrics], dtype=float)
    best_idx = int(np.nanargmax(gains)) if np.isfinite(gains).any() else -1
    out = dict(
        ar_r2=float(np.nanmean(ar_r2)), ar_hit=float(np.nanmean(ar_hit)),
        car_r2=float(np.nanmean(car_r2)), car_hit=float(np.nanmean(car_hit)),
        own_r2=float(np.nanmean(own_r2)), own_hit=float(np.nanmean(own_hit)),
        cross_r2=float(np.nanmean(cr_r2)), cross_hit=float(np.nanmean(cr_hit)),
        cross_minus_own_r2=float(np.nanmean(cr_r2) - np.nanmean(own_r2)),
        cross_minus_ar_r2=float(np.nanmean(cr_r2) - np.nanmean(ar_r2)),
        cross_minus_car_r2=float(np.nanmean(cr_r2) - np.nanmean(car_r2)),
        pnl_gross=float(gross["cum_gross"]), pnl_net=float(net["cum_net"]),
        pnl_hit=float(gross["hit_rate"]), turnover=float(gross["avg_turnover"]),
        n_bars=int(T), step=int(step),
        best_asset=NAMES[best_idx] if best_idx >= 0 else "",
        best_asset_cross_minus_own_r2=float(gains[best_idx]) if best_idx >= 0 else np.nan,
        asset_metrics_json=json.dumps(asset_metrics, separators=(",", ":")))
    for c, res in pnl.items():
        label = str(c).replace(".", "p")
        out[f"pnl_net_{label}bp"] = float(res["cum_net"])
    return out


# ------------------------------------------------------------------ sub-minute parity path
def r2_zero(y, pred):
    m = ~np.isnan(y) & ~np.isnan(pred)
    if m.sum() < 10:
        return np.nan
    den = float(np.sum(y[m] * y[m]))
    return np.nan if den <= 0 else 1.0 - float(np.sum((y[m] - pred[m]) ** 2) / den)


def sign_acc(y, pred):
    m = ~np.isnan(y) & ~np.isnan(pred) & (pred != 0) & (y != 0)
    return float(np.mean(np.sign(y[m]) == np.sign(pred[m]))) if m.any() else np.nan


def subminute_bar_panel(second, width, assets=ASSETS):
    sec = second["sec"]
    bar_id = sec // width
    _, start, counts = np.unique(bar_id, return_index=True, return_counts=True)
    min_count = max(1, int(np.ceil(0.8 * width)))
    keep = counts >= min_count
    last = start + counts - 1
    out = {"bar_end_sec": sec[last[keep]]}
    for asset in assets:
        levels = second[f"level_ofi_{asset}"]
        ofi_i = second[f"ofiI_{asset}"]
        summed_levels = np.add.reduceat(levels, start, axis=0)[keep]
        summed_integrated = np.add.reduceat(ofi_i, start)[keep]
        out[f"level_ofi_{asset}"] = summed_levels.astype(np.float32)
        out[f"ofiI_{asset}"] = summed_integrated.astype(np.float32)
        out[f"log_mid_{asset}"] = second[f"log_mid_{asset}"][last[keep]]
    return out


def subminute_scheme_panel(bar_panel, scheme, assets=ASSETS):
    ofi_cols, ret_cols, log_cols = [], [], []
    for asset in assets:
        levels = np.asarray(bar_panel[f"level_ofi_{asset}"], dtype=np.float32)
        if scheme == "best":
            s = levels[:, 0]
        elif scheme == "sum":
            s = np.nansum(levels, axis=1)
        elif scheme == "distance":
            w = (1.0 / (1.0 + np.arange(levels.shape[1]))).astype(np.float32)
            s = levels @ (w / w.sum())
        elif scheme == "pca":
            s = np.asarray(bar_panel[f"ofiI_{asset}"], dtype=np.float32)
        else:
            raise ValueError(scheme)
        lm = np.asarray(bar_panel[f"log_mid_{asset}"], dtype=float)
        ret_now = np.empty(len(lm), dtype=np.float32)
        ret_now[0] = np.nan
        ret_now[1:] = np.diff(lm).astype(np.float32)
        ofi_cols.append(np.asarray(s, dtype=np.float32))
        ret_cols.append(ret_now)
        log_cols.append(lm)
    return (np.column_stack(ofi_cols),
            np.column_stack(ret_cols),
            log_cols,
            np.asarray(bar_panel["bar_end_sec"], dtype=np.int64))


def exact_time_shift_matrix(times, x, shift_sec):
    desired = times - shift_sec
    pos = np.searchsorted(times, desired)
    exact = np.zeros(len(times), dtype=bool)
    ok = pos < len(times)
    exact[ok] = times[pos[ok]] == desired[ok]
    out = np.full_like(x, np.nan, dtype=float)
    out[exact] = x[pos[exact]]
    return out


def subminute_calendar_windows(times, train_sec=SUBMINUTE_TRAIN_SEC,
                               test_sec=SUBMINUTE_TEST_SEC,
                               step_sec=SUBMINUTE_STEP_SEC):
    """Shared absolute-time walk-forward boundaries for the sub-minute grid.

    Boundaries are derived from the covered bar-end clock only, before any
    model-specific validity filter is applied. Each evaluator then fills these
    same windows with its own valid rows.
    """
    times = np.asarray(times, dtype=np.int64)
    if len(times) == 0:
        return []
    start = int(times[0])
    stop = int(times[-1]) + 1
    windows = []
    cur = start
    while cur + train_sec + test_sec <= stop:
        train_start = cur
        train_end = cur + train_sec
        test_start = train_end
        test_end = train_end + test_sec
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        cur += step_sec
    return windows


def subminute_oos_pred(X, y, times, windows):
    ok = np.isfinite(np.column_stack([y, X])).all(axis=1)
    return subminute_oos_pred_with_mask(X, y, ok, times, windows)


def subminute_oos_pred_with_mask(X, y, ok, times, windows):
    pred = np.full(len(y), np.nan)
    times = np.asarray(times, dtype=np.int64)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(np.column_stack([y, X])).all(axis=1)
    if len(times) != len(y):
        raise ValueError("times and y must have the same length")
    if ok.sum() == 0:
        return pred, []
    meta = []
    min_train = max(20, X.shape[1] + 5)
    for w in windows:
        train_mask = (ok & (times >= w["train_start"]) &
                      (times < w["train_end"]))
        test_mask = (ok & (times >= w["test_start"]) &
                     (times < w["test_end"]))
        train_n = int(train_mask.sum())
        test_n = int(test_mask.sum())
        if train_n < min_train or test_n == 0:
            meta.append({**w, "train_n": train_n, "test_n": test_n,
                         "status": "skipped"})
            continue
        p, info = tuned_ridge_predict(X[train_mask], y[train_mask], X[test_mask])
        pred[test_mask] = p
        meta.append({**w, "train_n": train_n, "test_n": test_n,
                     "status": "ok", "info": info})
    return pred, meta


def average_loss(pred, realized):
    loss = (np.asarray(realized, float) - np.asarray(pred, float)) ** 2
    valid = ~np.isnan(loss)
    denom = valid.sum(axis=1)
    total = np.nansum(loss, axis=1)
    out = np.full(loss.shape[0], np.nan)
    np.divide(total, denom, out=out, where=denom > 0)
    return out


def eval_subminute_cell(second, bar_s, scheme, placebo=False):
    """Evaluate one sub-minute cell from the independently built second panel."""
    bars = subminute_bar_panel(second, bar_s)
    ofi, ret_now, log_mid, times = subminute_scheme_panel(bars, scheme)
    t, n = ofi.shape
    train = max(20, SUBMINUTE_TRAIN_SEC // bar_s)
    test = max(10, SUBMINUTE_TEST_SEC // bar_s)
    step = max(10, SUBMINUTE_STEP_SEC // bar_s)
    windows = subminute_calendar_windows(times)
    shifted = exact_time_shift_matrix(times, ofi, SUBMINUTE_TEST_SEC) if placebo else None

    f_own = np.full((t, n), np.nan)
    f_ar = np.full((t, n), np.nan)
    f_car = np.full((t, n), np.nan)
    f_cross = np.full((t, n), np.nan)
    realized = np.full((t, n), np.nan)
    asset_rows = []

    for i, name in enumerate(NAMES):
        lm = log_mid[i]
        y = np.r_[lm[1:] - lm[:-1], np.nan]
        realized[:, i] = y
        own_x = ofi[:, [i]]
        ar_x = ret_now[:, [i]]
        car_x = ret_now
        cross_x = ofi.copy()
        if placebo:
            for j in range(n):
                if j != i:
                    cross_x[:, j] = shifted[:, j]

        p_own, _ = subminute_oos_pred(own_x, y, times, windows)
        p_ar, _ = subminute_oos_pred(ar_x, y, times, windows)
        p_car, _ = subminute_oos_pred(car_x, y, times, windows)
        p_cross, _ = subminute_oos_pred(cross_x, y, times, windows)
        f_own[:, i] = p_own
        f_ar[:, i] = p_ar
        f_car[:, i] = p_car
        f_cross[:, i] = p_cross

        own_r2 = r2_zero(y, p_own)
        ar_r2 = r2_zero(y, p_ar)
        car_r2 = r2_zero(y, p_car)
        cross_r2 = r2_zero(y, p_cross)
        row = {
            "asset": name,
            "own_r2": finite_or_none(own_r2),
            "ar_r2": finite_or_none(ar_r2),
            "car_r2": finite_or_none(car_r2),
            "cross_r2": finite_or_none(cross_r2),
            "cross_minus_own_r2": finite_or_none(cross_r2 - own_r2),
            "cross_minus_ar_r2": finite_or_none(cross_r2 - ar_r2),
            "cross_minus_car_r2": finite_or_none(cross_r2 - car_r2),
            "own_hit": finite_or_none(sign_acc(y, p_own)),
            "ar_hit": finite_or_none(sign_acc(y, p_ar)),
            "car_hit": finite_or_none(sign_acc(y, p_car)),
            "cross_hit": finite_or_none(sign_acc(y, p_cross)),
            "own_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_own, y)),
            "car_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_car, y)),
            "cross_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_cross, y)),
            "n": int((~np.isnan(p_cross) & ~np.isnan(y)).sum()),
        }
        row["cross_minus_own_unit_pnl_bps"] = finite_diff_or_none(
            row["cross_unit_pnl_bps"], row["own_unit_pnl_bps"])
        asset_rows.append(row)

    losses = pd.DataFrame({
        "own": average_loss(f_own, realized),
        "ar": average_loss(f_ar, realized),
        "car": average_loss(f_car, realized),
        "cross": average_loss(f_cross, realized),
    })

    def avg(metric):
        return float(np.nanmean([r[metric] for r in asset_rows]))

    gains = np.array([r["cross_minus_own_r2"] for r in asset_rows], dtype=float)
    best_idx = int(np.nanargmax(gains)) if np.isfinite(gains).any() else -1
    out = {
        "r2_type": "zero",
        "model": "tuned_ridge_calendar",
        "train": int(train),
        "test": int(test),
        "step": int(step),
        "train_seconds": int(SUBMINUTE_TRAIN_SEC),
        "test_seconds": int(SUBMINUTE_TEST_SEC),
        "step_seconds": int(SUBMINUTE_STEP_SEC),
        "n_windows": int(len(windows)),
        "coverage_min": 0.8,
        "n_bars": int(t),
        "own_r2": avg("own_r2"),
        "ar_r2": avg("ar_r2"),
        "car_r2": avg("car_r2"),
        "cross_r2": avg("cross_r2"),
        "cross_minus_own_r2": avg("cross_minus_own_r2"),
        "cross_minus_ar_r2": avg("cross_minus_ar_r2"),
        "cross_minus_car_r2": avg("cross_minus_car_r2"),
        "own_hit": avg("own_hit"),
        "ar_hit": avg("ar_hit"),
        "car_hit": avg("car_hit"),
        "cross_hit": avg("cross_hit"),
        "own_unit_pnl_bps": avg("own_unit_pnl_bps"),
        "car_unit_pnl_bps": avg("car_unit_pnl_bps"),
        "cross_unit_pnl_bps": avg("cross_unit_pnl_bps"),
        "cross_minus_own_unit_pnl_bps": avg("cross_minus_own_unit_pnl_bps"),
        "best_asset": NAMES[best_idx] if best_idx >= 0 else "",
        "best_asset_cross_minus_own_r2": float(gains[best_idx]) if best_idx >= 0 else np.nan,
        "asset_metrics_json": json.dumps(asset_rows, separators=(",", ":")),
    }
    return out, losses


def eval_subminute_placebo_pair(second, bar_s, scheme):
    """Evaluate real and day-shifted cross OFI on one common valid calendar.

    Real cross and shifted-placebo cross are fit and scored on the same
    train/test blocks, so their loss series are pairable.
    """
    bars = subminute_bar_panel(second, bar_s)
    ofi, ret_now, log_mid, times = subminute_scheme_panel(bars, scheme)
    t, n = ofi.shape
    train = max(20, SUBMINUTE_TRAIN_SEC // bar_s)
    test = max(10, SUBMINUTE_TEST_SEC // bar_s)
    step = max(10, SUBMINUTE_STEP_SEC // bar_s)
    windows = subminute_calendar_windows(times)
    shifted = exact_time_shift_matrix(times, ofi, SUBMINUTE_TEST_SEC)

    forecasts = {
        "real": {
            "own": np.full((t, n), np.nan),
            "ar": np.full((t, n), np.nan),
            "car": np.full((t, n), np.nan),
            "cross": np.full((t, n), np.nan),
            "assets": [],
        },
        "placebo": {
            "own": np.full((t, n), np.nan),
            "ar": np.full((t, n), np.nan),
            "car": np.full((t, n), np.nan),
            "cross": np.full((t, n), np.nan),
            "assets": [],
        },
    }
    realized = np.full((t, n), np.nan)

    for i, name in enumerate(NAMES):
        lm = log_mid[i]
        y = np.r_[lm[1:] - lm[:-1], np.nan]
        realized[:, i] = y
        own_x = ofi[:, [i]]
        ar_x = ret_now[:, [i]]
        car_x = ret_now
        real_x = ofi.copy()
        plac_x = ofi.copy()
        for j in range(n):
            if j != i:
                plac_x[:, j] = shifted[:, j]
        common = np.isfinite(np.column_stack([y, own_x, ar_x, car_x, real_x, plac_x])).all(axis=1)

        p_own, _ = subminute_oos_pred_with_mask(own_x, y, common, times, windows)
        p_ar, _ = subminute_oos_pred_with_mask(ar_x, y, common, times, windows)
        p_car, _ = subminute_oos_pred_with_mask(car_x, y, common, times, windows)
        p_real, _ = subminute_oos_pred_with_mask(real_x, y, common, times, windows)
        p_plac, _ = subminute_oos_pred_with_mask(plac_x, y, common, times, windows)

        for kind, p_cross in [("real", p_real), ("placebo", p_plac)]:
            forecasts[kind]["own"][:, i] = p_own
            forecasts[kind]["ar"][:, i] = p_ar
            forecasts[kind]["car"][:, i] = p_car
            forecasts[kind]["cross"][:, i] = p_cross
            own_r2 = r2_zero(y, p_own)
            ar_r2 = r2_zero(y, p_ar)
            car_r2 = r2_zero(y, p_car)
            cross_r2 = r2_zero(y, p_cross)
            row = {
                "asset": name,
                "own_r2": finite_or_none(own_r2),
                "ar_r2": finite_or_none(ar_r2),
                "car_r2": finite_or_none(car_r2),
                "cross_r2": finite_or_none(cross_r2),
                "cross_minus_own_r2": finite_or_none(cross_r2 - own_r2),
                "cross_minus_ar_r2": finite_or_none(cross_r2 - ar_r2),
                "cross_minus_car_r2": finite_or_none(cross_r2 - car_r2),
                "own_hit": finite_or_none(sign_acc(y, p_own)),
                "ar_hit": finite_or_none(sign_acc(y, p_ar)),
                "car_hit": finite_or_none(sign_acc(y, p_car)),
                "cross_hit": finite_or_none(sign_acc(y, p_cross)),
                "own_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_own, y)),
                "car_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_car, y)),
                "cross_unit_pnl_bps": finite_or_none(unit_position_pnl_bps(p_cross, y)),
                "n": int((~np.isnan(p_cross) & ~np.isnan(y)).sum()),
            }
            row["cross_minus_own_unit_pnl_bps"] = finite_diff_or_none(
                row["cross_unit_pnl_bps"], row["own_unit_pnl_bps"])
            forecasts[kind]["assets"].append(row)

    outs = {}
    loss_out = {}
    for kind in ["real", "placebo"]:
        asset_rows = forecasts[kind]["assets"]

        def avg(metric):
            vals = np.array([r[metric] for r in asset_rows], dtype=float)
            return float(np.nanmean(vals))

        gains = np.array([r["cross_minus_own_r2"] for r in asset_rows], dtype=float)
        best_idx = int(np.nanargmax(gains)) if np.isfinite(gains).any() else -1
        outs[kind] = {
            "r2_type": "zero",
            "model": "tuned_ridge_calendar_common_placebo_mask",
            "train": int(train),
            "test": int(test),
            "step": int(step),
            "train_seconds": int(SUBMINUTE_TRAIN_SEC),
            "test_seconds": int(SUBMINUTE_TEST_SEC),
            "step_seconds": int(SUBMINUTE_STEP_SEC),
            "n_windows": int(len(windows)),
            "coverage_min": 0.8,
            "n_bars": int(t),
            "own_r2": avg("own_r2"),
            "ar_r2": avg("ar_r2"),
            "car_r2": avg("car_r2"),
            "cross_r2": avg("cross_r2"),
            "cross_minus_own_r2": avg("cross_minus_own_r2"),
            "cross_minus_ar_r2": avg("cross_minus_ar_r2"),
            "cross_minus_car_r2": avg("cross_minus_car_r2"),
            "own_hit": avg("own_hit"),
            "ar_hit": avg("ar_hit"),
            "car_hit": avg("car_hit"),
            "cross_hit": avg("cross_hit"),
            "own_unit_pnl_bps": avg("own_unit_pnl_bps"),
            "car_unit_pnl_bps": avg("car_unit_pnl_bps"),
            "cross_unit_pnl_bps": avg("cross_unit_pnl_bps"),
            "cross_minus_own_unit_pnl_bps": avg("cross_minus_own_unit_pnl_bps"),
            "best_asset": NAMES[best_idx] if best_idx >= 0 else "",
            "best_asset_cross_minus_own_r2": float(gains[best_idx]) if best_idx >= 0 else np.nan,
            "asset_metrics_json": json.dumps(asset_rows, separators=(",", ":")),
        }
        loss_out[kind] = pd.DataFrame({
            "own": average_loss(forecasts[kind]["own"], realized),
            "ar": average_loss(forecasts[kind]["ar"], realized),
            "car": average_loss(forecasts[kind]["car"], realized),
            "cross": average_loss(forecasts[kind]["cross"], realized),
        })
    return outs["real"], outs["placebo"], loss_out["real"], loss_out["placebo"]


def run_subminute_spec_grid(results_path, bars_s, spec_name, key_func, force=False):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    done = set()
    records = []
    if os.path.exists(results_path) and not force:
        old = pd.read_csv(results_path)
        if "key" in old.columns:
            ok_old = old[old.get("status", "ok") == "ok"].copy()
            done = set(ok_old["key"].tolist())
            records = ok_old.to_dict("records")

    expected = [key_func(bar_s, scheme) for bar_s in bars_s for scheme in SCHEMES]
    remaining = [k for k in expected if k not in done]
    print(f"{spec_name}: {len(expected)} configs total, {len(remaining)} remaining", flush=True)
    print(f"{spec_name}: building independent main-repo 1-second panel ...", flush=True)
    second = build_subminute_second_panel(PROCESSED, ASSETS, levels=LEVELS)
    print(f"{spec_name}: aligned seconds={len(second['sec']):,}", flush=True)

    t0 = time.time()
    finished = 0
    for bar_s in bars_s:
        for scheme in SCHEMES:
            key = key_func(bar_s, scheme)
            if key in done:
                continue
            c0 = time.time()
            rec = {
                "spec": spec_name,
                "key": key,
                "bar_s": bar_s,
                "scheme": scheme,
                "lagset": "single_bar",
                "lag_windows": ",".join(map(str, SINGLE_BAR_WINDOWS)),
                "horizon": 1,
                "horizon_seconds": bar_s,
                "levels": LEVELS,
            }
            try:
                res, _, _, _ = eval_subminute_placebo_pair(second, bar_s, scheme)
                rec.update(res)
                rec["status"] = "ok"
            except Exception as exc:
                rec["status"] = "error"
                rec["error"] = repr(exc)
                traceback.print_exc()
            rec["secs"] = round(time.time() - c0, 1)
            records.append(rec)
            done.add(key)
            pd.DataFrame(records).to_csv(results_path, index=False)
            finished += 1
            elapsed = time.time() - t0
            avg = elapsed / max(finished, 1)
            eta = avg * max(len(remaining) - finished, 0)
            extra = (f"cross-own={rec.get('cross_minus_own_r2'):+.5f} "
                     f"cross-car={rec.get('cross_minus_car_r2'):+.5f}") if rec["status"] == "ok" else rec.get("error", "")
            print(f"  [{rec['status']}] {key} ({rec['secs']}s) "
                  f"progress={finished}/{len(remaining)} eta={fmt_duration(eta)} {extra}", flush=True)

    print(f"{spec_name} done -> {results_path}", flush=True)
    return pd.DataFrame(records)


def run_subminute_grid():
    return run_subminute_spec_grid(SUBMINUTE_RESULTS, SUBMINUTE_BARS_S, "subminute", subminute_key,
                       force=FORCE_SUBMINUTE)


def run_subminute_decay_grid():
    return run_subminute_spec_grid(SUBMINUTE_DECAY_RESULTS, SUBMINUTE_DECAY_BARS_S, "subminute_decay",
                       subminute_decay_key, force=FORCE_SUBMINUTE_DECAY)


# ------------------------------------------------------------------ checkpoint
def load_done():
    if os.path.exists(RESULTS):
        df = pd.read_csv(RESULTS)
        return set(df["key"].tolist()), df.to_dict("records")
    return set(), []


def save(records):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    pd.DataFrame(records).to_csv(RESULTS, index=False)


def run_and_record(key, meta, records, done, fn):
    global RUN_DONE_THIS_SESSION
    if key in done:
        return
    t0 = time.time()
    rec = dict(meta); rec["key"] = key
    try:
        rec.update(fn()); rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"; rec["error"] = repr(e)
        traceback.print_exc()
    rec["secs"] = round(time.time() - t0, 1)
    records.append(rec); done.add(key); save(records)
    RUN_DONE_THIS_SESSION += 1
    if RUN_TOTAL_TO_DO:
        elapsed = time.time() - RUN_START_TS
        avg = elapsed / max(RUN_DONE_THIS_SESSION, 1)
        remaining = max(RUN_TOTAL_TO_DO - RUN_DONE_THIS_SESSION, 0)
        eta = avg * remaining
        progress = (f"progress={RUN_DONE_THIS_SESSION}/{RUN_TOTAL_TO_DO} "
                    f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)} "
                    f"avg={fmt_duration(avg)}/cfg")
    else:
        progress = "progress=unknown"
    tag = rec.get("status")
    extra = (f"own_hit={rec.get('own_hit'):.3f} cross_hit={rec.get('cross_hit'):.3f} "
             f"net={rec.get('pnl_net'):+.3f}") if tag == "ok" else rec.get("error", "")
    print(f"  [{tag}] {key}  ({rec['secs']}s)  {progress}  {extra}", flush=True)


# ------------------------------------------------------------------ sanity gate
def sanity_gate():
    print("sanity: checking src/ modules import and compute...", flush=True)
    sym = {"ts": np.array([0, 1]), "bid_p": np.array([[100.0], [100.01]]),
           "bid_q": np.array([[5.0], [5.0]]), "ask_p": np.array([[100.02], [100.02]]),
           "ask_q": np.array([[5.0], [5.0]])}
    assert np.isclose(compute_ofi(sym, 1)["ofi"].iloc[0], 5.0), "compute_ofi broken"
    x = np.random.default_rng(0).standard_normal(6000)
    y = 0.5 * x + np.random.default_rng(1).standard_normal(6000)
    p = walk_forward(make_lags_set(x, [1])[1:], y[1:], train=2000, step=1000, purge=1)
    assert np.isfinite(score(y[1:], p)[0]), "walk_forward broken"
    print("sanity: OK\n", flush=True)


# ------------------------------------------------------------------ main
def main():
    global RUN_TOTAL_TO_DO, RUN_START_TS
    sanity_gate()
    if SUBMINUTE:
        run_subminute_grid()
        return
    if SUBMINUTE_DECAY:
        run_subminute_decay_grid()
        return
    done, records = load_done()
    paper_bar = 60
    paper_horizons = [1, 4] if SMOKE else [1, 2, 3, 5, 10]
    expected = [paper_key(paper_bar, h) for h in paper_horizons]
    expected.extend(grid_key(bar_s, sc, ln, h)
                    for bar_s in BARS_S
                    for sc in SCHEMES
                    for ln in LAGSETS
                    for h in HORIZONS)
    RUN_TOTAL_TO_DO = sum(k not in done for k in expected)
    RUN_START_TS = time.time()
    print(f"resuming: {len(done)} configs already done in {RESULTS}", flush=True)
    print(f"planned: {len(expected)} configs total, {RUN_TOTAL_TO_DO} remaining; "
          f"levels={LEVELS}, refit_cap={FOLDCAP}", flush=True)
    print("ETA will stabilise after the first few completed configs.\n", flush=True)

    # ---- 1) 60s reference specification ----
    print("=== paper spec (CCZ): 60 s bars, PCA-integrated, lags {1,2,3,5,10,20,30} ===", flush=True)
    bba = {a: load_levels_bars(a, paper_bar * 1_000_000_000) for a in ASSETS}
    ofi, ret, spr = build_scheme_panel(bba, "pca")
    for h in paper_horizons:
        key = paper_key(paper_bar, h)
        meta = dict(spec="paper", bar_s=paper_bar, scheme="pca",
                    lagset="paperlags", horizon=h, levels=LEVELS, refit_cap=FOLDCAP)
        run_and_record(key, meta, records, done,
                       lambda h=h: eval_config(ofi, ret, spr, [1, 2, 3, 5, 10, 20, 30], h))
    del bba, ofi, ret, spr

    # ---- 2) broad grid ----
    print("\n=== grid sweep ===", flush=True)
    for bar_s in BARS_S:
        # Per-asset bars for this scale.
        bar_keys = [grid_key(bar_s, sc, ln, h)
                    for sc in SCHEMES for ln in LAGSETS for h in HORIZONS]
        if all(k in done for k in bar_keys):
            print(f"bar={bar_s}s: all done, skipping", flush=True)
            continue
        print(f"bar={bar_s}s: loading {len(ASSETS)} assets ...", flush=True)
        bba = {a: load_levels_bars(a, bar_s * 1_000_000_000) for a in ASSETS}
        for scheme in SCHEMES:
            ofi, ret, spr = build_scheme_panel(bba, scheme)
            for lagname, lagset in LAGSETS.items():
                for h in HORIZONS:
                    key = grid_key(bar_s, scheme, lagname, h)
                    meta = dict(spec="grid", bar_s=bar_s, scheme=scheme,
                                lagset=lagname, horizon=h, levels=LEVELS,
                                refit_cap=FOLDCAP)
                    run_and_record(key, meta, records, done,
                                   lambda o=ofi, r=ret, s=spr, ls=lagset, hh=h:
                                       eval_config(o, r, s, ls, hh))
            del ofi, ret, spr
        del bba

    print(f"\nDONE. {len(records)} rows in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()

