"""
leadlag.py - pairwise predictive cross-impact models and walk-forward evaluation.

Fits AR, own-OFI, cross_free, and cross_prop models for ordered asset pairs
across horizons, then returns out-of-sample forecasts and scores.
"""

import numpy as np


CCZ_LAG_WINDOWS = (1, 2, 3, 5, 10, 20, 30)
SINGLE_BAR_WINDOWS = (1,)


# --------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------
def make_lags(a, L):
    """Individual bar lags. Row t, col k = a[t-k], k=0..L-1."""
    N = len(a)
    X = np.full((N, L), np.nan)
    for k in range(L):
        X[k:, k] = a[:N - k]
    return X


def resolve_lag_windows(lags=None, lag_windows=None):
    """Return positive trailing-window lengths in bars.

    CCZ use features over trailing horizons {1,2,3,5,10,20,30} buckets, not
    separate one-bucket lags. Passing an integer keeps the old shorthand by
    expanding it to 1..lags.
    """
    if lag_windows is not None:
        out = np.asarray(tuple(lag_windows), dtype=int)
    elif lags is None:
        out = np.asarray(CCZ_LAG_WINDOWS, dtype=int)
    elif np.isscalar(lags):
        out = np.arange(1, int(lags) + 1, dtype=int)
    else:
        out = np.asarray(tuple(lags), dtype=int)
    if out.ndim != 1 or len(out) == 0 or np.any(out <= 0):
        raise ValueError("lag windows must be positive integers")
    return np.unique(out)


def make_lag_windows(a, windows=CCZ_LAG_WINDOWS):
    """Trailing-window sums ending at t.

    Row t, col j is sum(a[t-window+1:t+1]). This matches CCZ's
    ofi^(kh)_{i,t} and r^(kh)_{i,t} features for k in the lag set.
    """
    x = np.asarray(a, dtype=float)
    windows = resolve_lag_windows(lag_windows=windows)
    N = len(x)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    valid = np.concatenate([[0], np.cumsum(~np.isnan(x))])
    X = np.full((N, len(windows)), np.nan)
    idx = np.arange(N)
    for j, w in enumerate(windows):
        ok = idx >= w - 1
        total = cs[idx[ok] + 1] - cs[idx[ok] + 1 - w]
        n_valid = valid[idx[ok] + 1] - valid[idx[ok] + 1 - w]
        X[idx[ok], j] = np.where(n_valid == w, total, np.nan)
    return X


def fwd_return(ret, h):
    """y_t = sum of ret over the next h bars (t+1..t+h); NaN where unavailable."""
    N = len(ret)
    c = np.concatenate([[0.0], np.cumsum(ret)])
    y = np.full(N, np.nan)
    idx = np.arange(N)
    ok = idx + 1 + h <= N
    y[ok] = c[idx[ok] + 1 + h] - c[idx[ok] + 1]
    return y


def propagator(cross_lags, rho):
    """Collapse free cross lags into one rho^k-decayed feature (weights sum 1)."""
    L = cross_lags.shape[1]
    w = rho ** np.arange(L)
    w = w / w.sum()
    return (cross_lags @ w).reshape(-1, 1)


# --------------------------------------------------------------------------
# weighted OLS + walk-forward harness
# --------------------------------------------------------------------------
def _wls(X, y, w=None, ridge=1e-8):
    A = np.column_stack([np.ones(len(X)), X])
    if w is None:
        w = np.ones(len(X))
    AtA = A.T @ (w[:, None] * A) + ridge * np.eye(A.shape[1])
    return np.linalg.solve(AtA, A.T @ (w * y))


def _predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def walk_forward(X, y, train=3000, step=500, mode="hard", halflife=1500, purge=0):
    """Rolling-origin OOS predictions. Refits on a trailing `train` window
    (minus a `purge` embargo) every `step` bars. Returns an array of OOS
    predictions aligned to y (NaN before the first test block)."""
    N = len(y)
    pred = np.full(N, np.nan)
    start = train
    while start < N:
        tr0, tr1 = max(0, start - train), start - purge
        if tr1 - tr0 < X.shape[1] + 5:
            start += step
            continue
        Xtr, ytr = X[tr0:tr1], y[tr0:tr1]
        if mode == "expo":
            age = np.arange(len(Xtr))[::-1]          # most recent -> age 0
            w = 0.5 ** (age / halflife)
        else:
            w = None
        beta = _wls(Xtr, ytr, w)
        te1 = min(start + step, N)
        pred[start:te1] = _predict(beta, X[start:te1])
        start += step
    return pred


def zfit(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.clip((x - mu) / sd, -12.0, 12.0), mu, sd


def zapply(x, mu, sd):
    return np.clip((x - mu) / sd, -12.0, 12.0)


def ridge_predict(xtr, ytr, xte, lam=0.0):
    """Standardized ridge prediction used by the subminute parity sweep."""
    xz, mu, sd = zfit(xtr)
    xt = zapply(xte, mu, sd)
    xz = np.column_stack([np.ones(len(xz)), xz])
    xt = np.column_stack([np.ones(len(xt)), xt])
    pen = np.eye(xz.shape[1]) * float(lam)
    pen[0, 0] = 0.0
    lhs = xz.T @ xz + pen
    rhs = xz.T @ ytr
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return xt @ beta


def tuned_ridge_predict(xtr, ytr, xte, lambdas=(0.0, 10.0, 100.0)):
    """Choose a ridge penalty on the last 25% of the training block."""
    split = int(len(ytr) * 0.75)
    best_lam = float(lambdas[0])
    best_mse = np.inf
    for lam in lambdas:
        pred = ridge_predict(xtr[:split], ytr[:split], xtr[split:], lam)
        mse = float(np.mean((ytr[split:] - pred) ** 2))
        if mse < best_mse:
            best_lam = float(lam)
            best_mse = mse
    return ridge_predict(xtr, ytr, xte, best_lam), best_lam


def walk_forward_blocks(X, y, train, test, step, predict_func=tuned_ridge_predict):
    """Calendar-block walk-forward used for sub-minute parity.

    Unlike walk_forward(), `step` controls where the next train/test block
    starts; each fit only predicts `test` rows.
    """
    n = len(y)
    pred = np.full(n, np.nan)
    meta = []
    for start in range(0, n - train - test + 1, step):
        tr = slice(start, start + train)
        te = slice(start + train, start + train + test)
        p, info = predict_func(X[tr], y[tr], X[te])
        pred[te] = p
        meta.append({"start": start, "test_start": start + train,
                     "test_end": start + train + test - 1, "info": info})
    return pred, meta


def score(y, p):
    """Out-of-sample R2 (vs OOS mean) and directional hit rate."""
    m = ~np.isnan(p) & ~np.isnan(y)
    yt, pt = y[m], p[m]
    if len(yt) < 10:
        return np.nan, np.nan, int(m.sum())
    ss_res = np.sum((yt - pt) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    nz = pt != 0
    hit = np.mean(np.sign(pt[nz]) == np.sign(yt[nz])) if nz.any() else np.nan
    return r2, hit, int(m.sum())


# --------------------------------------------------------------------------
# the per-pair horizon sweep
# --------------------------------------------------------------------------
def sweep(ofi_X, ofi_Y, ret_Y, horizons, L=None, lag_windows=None, rho=0.6,
          mode="hard", train=3000, step=500):
    """Run the four-rung ladder across horizons for the ordered pair X -> Y.

    L is kept as a backward-compatible shorthand. For CCZ replication, leave it
    unset or pass lag_windows=(1,2,3,5,10,20,30).

    Returns dict: model -> {h: (r2_oos, hit_rate, n_oos)}."""
    windows = resolve_lag_windows(L, lag_windows)
    max_lag = int(windows.max())
    own = make_lag_windows(ofi_Y, windows)
    cross = make_lag_windows(ofi_X, windows)
    ar = make_lag_windows(ret_Y, windows)
    prop = propagator(make_lags(ofi_X, max_lag), rho)
    specs = {
        "AR": ar,
        "own": own,
        "cross_free": np.hstack([own, cross]),
        "cross_prop": np.hstack([own, prop]),
    }
    results = {k: {} for k in specs}
    N = len(ret_Y)
    for h in horizons:
        y = fwd_return(ret_Y, h)
        base_ok = (np.arange(N) >= max_lag - 1) & ~np.isnan(y)
        for name, feats in specs.items():
            ok = base_ok & ~np.isnan(feats).any(axis=1)
            Xc, yc = feats[ok], y[ok]
            pred = walk_forward(Xc, yc, train=train, step=step,
                                mode=mode, purge=h)
            results[name][h] = score(yc, pred)
    return results


def _print_table(results, horizons, title):
    print(f"\n{title}")
    print("model        " + "".join(f"  h={h:<3d}      " for h in horizons))
    for name, byh in results.items():
        cells = []
        for h in horizons:
            r2, hit, _ = byh[h]
            cells.append(f"R2={r2:+.3f} hit={hit:.2f}")
        print(f"{name:<12}" + "  ".join(cells))


# --------------------------------------------------------------------------
# synthetic lead-lag self-test
# --------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(1)
    T = 20000
    ofi_X = rng.standard_normal(T)
    ofi_Y = rng.standard_normal(T)
    # Synthetic target return with a one-bar cross-asset lead.
    ret_Y = np.empty(T)
    ret_Y[0] = 0.1 * ofi_Y[0] + 0.5 * rng.standard_normal()
    ret_Y[1:] = 0.3 * ofi_X[:-1] + 0.1 * ofi_Y[1:] + 0.5 * rng.standard_normal(T - 1)

    res = sweep(ofi_X, ofi_Y, ret_Y, horizons=[1, 6], L=5,
                train=4000, step=1000)
    cross1 = res["cross_free"][1][0]
    ar1 = res["AR"][1][0]
    own1 = res["own"][1][0]
    cross_hit1 = res["cross_free"][1][1]
    cross6 = res["cross_free"][6][0]
    prop1 = res["cross_prop"][1][0]

    assert cross1 > 0.10, f"cross model should have clear OOS R2 at h=1, got {cross1:.3f}"
    assert cross1 > ar1 and cross1 > own1, "cross flow should beat AR and own at h=1"
    assert cross_hit1 > 0.55, f"directional hit should exceed 0.55, got {cross_hit1:.3f}"
    assert cross6 < cross1, "predictability should decay to longer horizon"
    assert prop1 > 0.05, f"propagator should also capture the lead, got {prop1:.3f}"

    # Exponential-weighting smoke path.
    res_e = sweep(ofi_X, ofi_Y, ret_Y, horizons=[1], L=5, mode="expo",
                  train=4000, step=1000)
    assert res_e["cross_free"][1][0] > 0.05

    print("leadlag.py self-test passed:")
    print(f"  h=1  cross R2={cross1:+.3f} (AR={ar1:+.3f}, own={own1:+.3f})  "
          f"hit={cross_hit1:.2f}  prop R2={prop1:+.3f}")
    print(f"  decay: cross R2  h=1 {cross1:+.3f} -> h=6 {cross6:+.3f}")
    print(f"  expo mode cross R2={res_e['cross_free'][1][0]:+.3f}")


# --------------------------------------------------------------------------
def _real_demo():
    import os
    from ofi import bar_l2_ofi
    if not (os.path.exists("processed/btcusdt.npz") and
            os.path.exists("processed/ethusdt.npz")):
        return
    bars = {}
    for tag in ["btcusdt", "ethusdt"]:
        sym = dict(np.load(f"processed/{tag}.npz"))
        bars[tag], _, _ = bar_l2_ofi(sym, levels=10, bar_ns=10_000_000_000)
        bars[tag] = bars[tag].dropna(subset=["ret"])
    j = bars["btcusdt"].join(bars["ethusdt"], lsuffix="_btc", rsuffix="_eth",
                             how="inner").dropna()
    H = [1, 2, 3, 6]            # 10s bars -> 10/20/30/60s ahead
    print(f"\n=== REAL DATA (1 day, {len(j)} 10s bars) ===")
    r = sweep(j["ofi_I_btc"].to_numpy(), j["ofi_I_eth"].to_numpy(),
              j["ret_eth"].to_numpy(), horizons=H, L=10, train=3000, step=500)
    _print_table(r, H, "BTC OFI  ->  ETH future return")
    r2 = sweep(j["ofi_I_eth"].to_numpy(), j["ofi_I_btc"].to_numpy(),
               j["ret_btc"].to_numpy(), horizons=H, L=10, train=3000, step=500)
    _print_table(r2, H, "ETH OFI  ->  BTC future return")


if __name__ == "__main__":
    _self_test()
    _real_demo()

