"""
cross_sectional.py - N-asset cross-impact matrix estimation.

Fits price-impact and cross-impact LASSO models for each target asset using the
walk-forward conventions from src/leadlag.py. Writes out-of-sample forecast
comparisons and stores a directed Lambda matrix where Lambda[i, j] is the
standardised effect of asset j's OFI on asset i's return.

horizon=0 gives contemporaneous cross-impact; horizon>0 gives forward-looking
predictive cross-impact.
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoCV

from leadlag import fwd_return, make_lag_windows, resolve_lag_windows, score, walk_forward


def _std_fit(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def _pick_alpha(X, y):
    """One-off LASSO alpha via small CV on a (standardised) training block."""
    mu, sd = _std_fit(X)
    cv = LassoCV(cv=3, max_iter=20000)
    cv.fit((X - mu) / sd, y)
    return cv.alpha_


def _lasso_wf(X, y, alpha, train, step, purge):
    """Rolling-origin LASSO. Returns (OOS preds aligned to y, mean std-space coef).

    If alpha is None, choose it by CV inside each training window. This is
    slower but matches the paper's per-regression penalty selection more closely.
    """
    N = len(y)
    pred = np.full(N, np.nan)
    coefs = []
    start = train
    while start < N:
        tr0, tr1 = max(0, start - train), start - purge
        if tr1 - tr0 < max(30, X.shape[1] + 5):
            start += step
            continue
        Xtr, ytr = X[tr0:tr1], y[tr0:tr1]
        mu, sd = _std_fit(Xtr)
        a = _pick_alpha(Xtr, ytr) if alpha is None else alpha
        m = Lasso(alpha=a, max_iter=20000).fit((Xtr - mu) / sd, ytr)
        te1 = min(start + step, N)
        pred[start:te1] = m.predict((X[start:te1] - mu) / sd)
        coefs.append(m.coef_)
        start += step
    coef = np.mean(coefs, axis=0) if coefs else np.zeros(X.shape[1])
    return pred, coef


def fit_cross_impact(ofi, ret, names, lags=None, lag_windows=None, horizon=1,
                     train=3000, step=500, alpha=None):
    """Estimate the cross-impact matrix and PI-vs-CI comparison for N assets.

    ofi, ret: (T, N) arrays of (integrated) OFI and bar returns, aligned.
    Returns dict with:
      'Lambda'    (N, N) signed cross-impact matrix, row=target i, col=source j
      'pi' / 'ci' per-target (r2_oos, hit, n) for own-only and all-asset models
      'forecasts' (T, N) OOS forecasts from the CI model (for evaluation.py)
      'names', 'horizon'
    """
    if horizon == 0 and lags is None and lag_windows is None:
        windows = resolve_lag_windows(lag_windows=(1,))
    else:
        windows = resolve_lag_windows(lags, lag_windows)
    max_lag = int(windows.max())
    T, N = ofi.shape
    blocks = [make_lag_windows(ofi[:, j], windows) for j in range(N)]
    all_feat = np.hstack(blocks)
    Lambda = np.zeros((N, N))
    pi, ci = {}, {}
    forecasts = np.full((T, N), np.nan)

    for i in range(N):
        y = ret[:, i] if horizon == 0 else fwd_return(ret[:, i], horizon)
        base = (np.arange(T) >= max_lag - 1) & ~np.isnan(y)

        ok_all = base & ~np.isnan(all_feat).any(1)

        own = blocks[i]
        ok = base & ~np.isnan(own).any(1)
        p = walk_forward(own[ok], y[ok], train=train, step=step, purge=horizon)
        pi[names[i]] = score(y[ok], p)

        p, coef = _lasso_wf(all_feat[ok_all], y[ok_all], alpha, train, step, purge=horizon)
        cc = np.full(T, np.nan); cc[ok_all] = p
        ci[names[i]] = score(y[ok_all], p)
        forecasts[:, i] = cc

        for j in range(N):
            Lambda[i, j] = coef[j * len(windows):(j + 1) * len(windows)].sum()

    return {"Lambda": Lambda, "pi": pi, "ci": ci, "forecasts": forecasts,
            "names": names, "horizon": horizon}


def lead_lag_network(Lambda, names):
    """Off-diagonal of Lambda as a directed network. Returns out-strength per
    asset (how much it predicts OTHERS) -> the hub ranking."""
    N = len(names)
    out_strength = {names[j]: sum(abs(Lambda[i, j]) for i in range(N) if i != j)
                    for j in range(N)}
    return dict(sorted(out_strength.items(), key=lambda kv: -kv[1]))


def ablation(ofi, ret, names, target, lags=None, lag_windows=None, horizon=1,
             train=3000, step=500, alpha=None):
    """Localise where cross-asset signal lives for one target asset.

    Compares, out-of-sample, feature sets that add cross-asset OFI in different
    combinations:
      own              - target's own OFI lags only (the baseline)
      own+<X>          - own plus EACH single other asset (does X specifically help?)
      all              - own plus all others
      cross_only       - all others, EXCLUDING the target's own OFI
    Returns dict: spec -> (r2_oos, hit, n). Own-only uses OLS; cross specs use LASSO.
    """
    if horizon == 0 and lags is None and lag_windows is None:
        windows = resolve_lag_windows(lag_windows=(1,))
    else:
        windows = resolve_lag_windows(lags, lag_windows)
    max_lag = int(windows.max())
    N = ofi.shape[1]
    ti = names.index(target)
    blocks = [make_lag_windows(ofi[:, j], windows) for j in range(N)]
    y = ret[:, ti] if horizon == 0 else fwd_return(ret[:, ti], horizon)
    base = (np.arange(len(y)) >= max_lag - 1) & ~np.isnan(y)

    specs = {"own": blocks[ti]}
    for j in range(N):
        if j != ti:
            specs[f"own+{names[j]}"] = np.hstack([blocks[ti], blocks[j]])
    specs["all"] = np.hstack(blocks)
    specs["cross_only"] = np.hstack([blocks[j] for j in range(N) if j != ti])

    out = {}
    for nm, F in specs.items():
        ok = base & ~np.isnan(F).any(1)
        if nm == "own":
            p = walk_forward(F[ok], y[ok], train=train, step=step, purge=horizon)
        else:
            p, _ = _lasso_wf(F[ok], y[ok], alpha, train, step, purge=horizon)
        out[nm] = score(y[ok], p)
    return out


def _xgb_wf(X, y, train, step, purge, **kw):
    """Walk-forward XGBoost (the nonlinear rung). Returns OOS predictions."""
    from xgboost import XGBRegressor
    params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                  subsample=0.8, n_jobs=2, verbosity=0)
    params.update(kw)
    N = len(y)
    pred = np.full(N, np.nan)
    start = train
    while start < N:
        tr0, tr1 = max(0, start - train), start - purge
        if tr1 - tr0 < X.shape[1] + 5:
            start += step
            continue
        m = XGBRegressor(**params).fit(X[tr0:tr1], y[tr0:tr1])
        te1 = min(start + step, N)
        pred[start:te1] = m.predict(X[start:te1])
        start += step
    return pred


def compare_estimators(ofi, ret, names, target, lags=None, lag_windows=None, horizon=1,
                       train=3000, step=500, alpha=None):
    """LASSO vs XGBoost on the all-asset cross features for one target.
    Returns dict estimator -> (r2_oos, hit, n)."""
    if horizon == 0 and lags is None and lag_windows is None:
        windows = resolve_lag_windows(lag_windows=(1,))
    else:
        windows = resolve_lag_windows(lags, lag_windows)
    max_lag = int(windows.max())
    N = ofi.shape[1]
    ti = names.index(target)
    F = np.hstack([make_lag_windows(ofi[:, j], windows) for j in range(N)])
    y = ret[:, ti] if horizon == 0 else fwd_return(ret[:, ti], horizon)
    ok = (np.arange(len(y)) >= max_lag - 1) & ~np.isnan(y) & ~np.isnan(F).any(1)
    Xc, yc = F[ok], y[ok]
    p_l, _ = _lasso_wf(Xc, yc, alpha, train, step, purge=horizon)
    p_x = _xgb_wf(Xc, yc, train, step, purge=horizon)
    return {"LASSO": score(yc, p_l), "XGBoost": score(yc, p_x)}


# --------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(3)
    T, N = 16000, 3
    ofi = rng.standard_normal((T, N))
    ret = np.empty((T, N))
    # Synthetic three-asset fixture with one cross-asset lead.
    ret[:, 0] = 0.1 * ofi[:, 0] + 0.5 * rng.standard_normal(T)
    ret[:, 2] = 0.1 * ofi[:, 2] + 0.5 * rng.standard_normal(T)
    ret[0, 1] = 0.5 * rng.standard_normal()
    ret[1:, 1] = 0.4 * ofi[:-1, 0] + 0.1 * ofi[1:, 1] + 0.5 * rng.standard_normal(T - 1)

    names = ["X0", "X1", "X2"]
    r = fit_cross_impact(ofi, ret, names=names,
                         lags=5, horizon=1, train=4000, step=1000)
    L = r["Lambda"]
    ci1, pi1 = r["ci"]["X1"][0], r["pi"]["X1"][0]
    ci0, pi0 = r["ci"]["X0"][0], r["pi"]["X0"][0]

    # Led-asset cross-impact assertion.
    assert ci1 - pi1 > 0.05, f"CI should beat PI for led asset, got {ci1-pi1:.3f}"
    # Directed Lambda assertion.
    offdiag = [(abs(L[i, j]), i, j) for i in range(N) for j in range(N) if i != j]
    _, bi, bj = max(offdiag)
    assert (bi, bj) == (1, 0), f"largest cross term should be X0->X1, got ({bi},{bj})"
    # Independent-asset assertion.
    assert ci0 - pi0 < 0.03, f"no cross gain expected for X0, got {ci0-pi0:.3f}"

    net = lead_lag_network(L, names)
    assert list(net.keys())[0] == "X0", f"X0 should be the hub, got {net}"

    # Led-asset ablation.
    ab = ablation(ofi, ret, names, "X1", lags=5, horizon=1,
                  train=4000, step=1000)
    assert ab["own+X0"][0] > ab["own"][0], "adding the leader X0 should help X1"
    assert ab["cross_only"][0] > ab["own"][0], "cross-only should beat own for led X1"
    assert ab["own+X2"][0] <= ab["own+X0"][0] + 1e-6, "X0 should help more than X2"

    print("cross_sectional.py self-test passed:")
    print(f"  led asset X1:   PI R2={pi1:+.3f} -> CI R2={ci1:+.3f}  (cross helps)")
    print(f"  indep asset X0: PI R2={pi0:+.3f} -> CI R2={ci0:+.3f}  (no gain)")
    print(f"  largest cross term: X0 -> X1  (Lambda[1,0]={L[1,0]:+.3f})")
    print(f"  hub out-strength ranking: {{ {', '.join(f'{k}:{v:.2f}' for k,v in net.items())} }}")
    print(f"  ablation X1: own R2={ab['own'][0]:+.3f}, own+X0={ab['own+X0'][0]:+.3f}, "
          f"cross_only={ab['cross_only'][0]:+.3f}")


def _real_demo():
    import os
    if not (os.path.exists("processed/btcusdt.npz") and
            os.path.exists("processed/ethusdt.npz")):
        return
    from ofi import bar_l2_ofi
    bars = {}
    for tag in ["btcusdt", "ethusdt"]:
        sym = dict(np.load(f"processed/{tag}.npz"))
        bars[tag], _, _ = bar_l2_ofi(sym, levels=10, bar_ns=10_000_000_000)
        bars[tag] = bars[tag].dropna(subset=["ret"])
    j = bars["btcusdt"].join(bars["ethusdt"], lsuffix="_b", rsuffix="_e",
                             how="inner").dropna()
    ofi = np.column_stack([j["ofi_I_b"], j["ofi_I_e"]])
    ret = np.column_stack([j["ret_b"], j["ret_e"]])
    names = ["BTC", "ETH"]
    print(f"\n=== REAL DATA cross-impact (1 day, {len(j)} 10s bars) ===")
    r = fit_cross_impact(ofi, ret, names, lags=10, horizon=1, train=3000, step=500)
    print("  PI vs CI (forward, h=1):")
    for nm in names:
        print(f"    {nm}: PI R2={r['pi'][nm][0]:+.4f} hit={r['pi'][nm][1]:.2f}  |  "
              f"CI R2={r['ci'][nm][0]:+.4f} hit={r['ci'][nm][1]:.2f}")
    print("  Lambda (row=target, col=source OFI), standardised:")
    L = r["Lambda"]
    print("            " + "  ".join(f"{n:>8}" for n in names))
    for i, nm in enumerate(names):
        print(f"    {nm:>6} " + "  ".join(f"{L[i,j]:+8.3f}" for j in range(len(names))))
    net = lead_lag_network(L, names)
    print(f"  hub ranking: {{ {', '.join(f'{k}:{v:.2e}' for k, v in net.items())} }}")
    print("  (one day, 2 assets -> expect CI ~ PI and tiny terms; not a result.)")


if __name__ == "__main__":
    _self_test()
    _real_demo()
