"""
evaluation.py - economic evaluation of return forecasts.

Builds forecast-implied self-financed portfolio weights from out-of-sample
forecasts, applies transaction costs, and reports net PnL, hit rate, and
annualised Sharpe.
"""

import numpy as np


def directional_hit_rate(forecast, realized):
    """Fraction of nonzero forecasts whose sign matches the realised return."""
    f, r = np.asarray(forecast, float).ravel(), np.asarray(realized, float).ravel()
    m = ~np.isnan(f) & ~np.isnan(r) & (f != 0)
    return float(np.mean(np.sign(f[m]) == np.sign(r[m]))) if m.any() else np.nan


def unit_position_pnl_bps(forecast, realized):
    """Mean sign-forecast unit-position PnL in bps per bar.

    This is the cost-free unit convention used by the sub-minute parity files.
    The main economic evaluation remains forecast_portfolio().
    """
    f, r = np.asarray(forecast, float).ravel(), np.asarray(realized, float).ravel()
    m = ~np.isnan(f) & ~np.isnan(r) & (f != 0)
    return float(np.mean(np.sign(f[m]) * r[m]) * 1e4) if m.any() else np.nan


def rolling_forecast_vol(forecasts, window):
    """Per-asset rolling std of the forecast series (CCZ's sigma_{i,t}),
    using only past values. forecasts: (T, N). Returns (T, N)."""
    f = np.asarray(forecasts, float)
    T, N = f.shape
    out = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - window)
        seg = f[lo:t]
        for j in range(N):
            vals = seg[:, j]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 5:
                out[t, j] = np.std(vals)
    return out


def forecast_portfolio(forecasts, realized, spreads, fcast_vol,
                       cost_bps=0.0, periods_per_year=None):
    """Backtest the CCZ forecast-implied portfolio, net of costs.

    All inputs shape (T, N) = (time, assets):
      forecasts  - one-step-ahead return forecast known at t
      realized   - the return actually realised over (t, t+1]
      spreads    - relative bid-ask spread at t (gate threshold)
      fcast_vol  - sigma_{i,t}, e.g. from rolling_forecast_vol
    cost_bps     - round-trip-ish cost per unit turnover, in basis points
    periods_per_year - for annualising Sharpe (e.g. 6.3e6 for 10s bars, 24/7)

    Returns a dict of summary stats plus the per-period net PnL series.
    """
    f = np.asarray(forecasts, float)
    r = np.asarray(realized, float)
    s = np.asarray(spreads, float)
    sig = np.asarray(fcast_vol, float)

    trade = (np.abs(f) > s).astype(float)                 # gate vs spread
    raw = trade * f / np.where(sig > 0, sig, np.nan)       # signed, vol-scaled
    raw = np.nan_to_num(raw)
    denom = np.nansum(np.abs(raw), axis=1, keepdims=True)  # self-finance
    w = np.zeros_like(raw)
    np.divide(raw, denom, out=w, where=denom > 0)

    gross = np.nansum(w * np.nan_to_num(r), axis=1)        # per-period gross
    turnover = np.nansum(np.abs(np.diff(w, axis=0, prepend=0.0)), axis=1)
    cost = (cost_bps * 1e-4) * turnover
    net = gross - cost

    valid = (~np.isnan(f).all(axis=1) & ~np.isnan(r).all(axis=1) &
             ~np.isnan(s).all(axis=1) & ~np.isnan(sig).all(axis=1))
    netv = net[valid]
    if len(netv) == 0:
        mean, sd = np.nan, np.nan
    else:
        mean, sd = netv.mean(), netv.std()
    traded = np.abs(w).sum(axis=1) > 0
    out = {
        "periods": int(valid.sum()),
        "frac_traded": float(traded[valid].mean()) if valid.any() else np.nan,
        "mean_bps": mean * 1e4 if not np.isnan(mean) else np.nan,
        "cum_net": float(netv.sum()) if len(netv) else np.nan,
        "cum_gross": float(gross[valid].sum()) if valid.any() else np.nan,
        "avg_turnover": float(turnover[valid].mean()) if valid.any() else np.nan,
        "sharpe_per_period": float(mean / sd) if sd > 0 else np.nan,
        "hit_rate": directional_hit_rate(np.where(trade > 0, f, np.nan), r),
        "net_series": np.where(valid, net, np.nan),
        "gross_series": np.where(valid, gross, np.nan),
    }
    if periods_per_year is not None and sd > 0:
        out["sharpe_annual"] = float(mean / sd * np.sqrt(periods_per_year))
    return out


# --------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(2)
    T, N = 12000, 3
    true_fut = 0.0008 * rng.standard_normal((T, N))
    realized = true_fut + 0.0008 * rng.standard_normal((T, N))
    spreads = np.full((T, N), 0.0002)

    # Informative forecast fixture.
    fc = true_fut + 0.0010 * rng.standard_normal((T, N))
    sig = rolling_forecast_vol(fc, window=300)
    res = forecast_portfolio(fc, realized, spreads, sig, cost_bps=0.0)
    res_cost = forecast_portfolio(fc, realized, spreads, sig, cost_bps=1.0)
    assert res["cum_net"] > 0, "informative forecast should be profitable gross"
    assert res["hit_rate"] > 0.52, f"hit rate should beat 0.5, got {res['hit_rate']:.3f}"
    assert res["sharpe_per_period"] > 0
    assert res_cost["cum_net"] < res["cum_net"], "costs must reduce PnL"

    # Gross-exposure normalisation check.
    f1 = true_fut + 0.001 * rng.standard_normal((T, N))
    trade = (np.abs(f1) > spreads)
    raw = np.nan_to_num(trade * f1 / rolling_forecast_vol(f1, 300))
    den = np.nansum(np.abs(raw), axis=1, keepdims=True)
    w = np.zeros_like(raw)
    np.divide(raw, den, out=w, where=den > 0)
    gross_exp = np.abs(w).sum(axis=1)
    traded = gross_exp > 0
    assert np.allclose(gross_exp[traded], 1.0), "weights should self-finance to 1"

    # Independent-noise forecast fixture.
    noise = 0.001 * rng.standard_normal((T, N))
    rn = forecast_portfolio(noise, realized, spreads,
                            rolling_forecast_vol(noise, 300), cost_bps=0.0)
    assert abs(rn["hit_rate"] - 0.5) < 0.03, f"noise hit should be ~0.5, got {rn['hit_rate']:.3f}"

    print("evaluation.py self-test passed:")
    print(f"  informative: cum_net={res['cum_net']:.4f} hit={res['hit_rate']:.3f} "
          f"sharpe/period={res['sharpe_per_period']:.3f}")
    print(f"  with 1bp cost: cum_net={res_cost['cum_net']:.4f} (< gross)")
    print(f"  noise control: hit={rn['hit_rate']:.3f} cum_net={rn['cum_net']:.4f}")


def _real_demo():
    import os
    if not (os.path.exists("processed/btcusdt.npz") and
            os.path.exists("processed/ethusdt.npz")):
        return
    from ofi import bar_l2_ofi, bucket_times
    from leadlag import fwd_return, make_lag_windows, walk_forward
    import numpy as np

    BAR = 10_000_000_000
    bars, spr = {}, {}
    for tag in ["btcusdt", "ethusdt"]:
        sym = dict(np.load(f"processed/{tag}.npz"))
        b, _, _ = bar_l2_ofi(sym, levels=10, bar_ns=BAR)
        b = b.dropna(subset=["ret"])
        bars[tag] = b
        rel = (sym["ask_p"][:, 0] - sym["bid_p"][:, 0]) / \
              (0.5 * (sym["ask_p"][:, 0] + sym["bid_p"][:, 0]))
        bucket = bucket_times(sym["ts"], BAR)
        import pandas as pd
        spr[tag] = pd.Series(rel, index=bucket).groupby(level=0).mean()

    j = bars["btcusdt"].join(bars["ethusdt"], lsuffix="_btc", rsuffix="_eth",
                             how="inner").dropna()
    L, h = 10, 1
    windows = np.arange(1, L + 1)

    def oos_forecast(ofi_self, ofi_other, ret_self):
        own = make_lag_windows(ofi_self, windows)
        cross = make_lag_windows(ofi_other, windows)
        y = fwd_return(ret_self, h)
        X = np.hstack([own, cross])
        ok = (np.arange(len(y)) >= windows.max() - 1) & ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        pred = np.full(len(y), np.nan)
        p = walk_forward(X[ok], y[ok], train=3000, step=500, purge=h)
        pred[ok] = p
        return pred, y

    f_eth, r_eth = oos_forecast(j["ofi_I_eth"].to_numpy(),
                                j["ofi_I_btc"].to_numpy(), j["ret_eth"].to_numpy())
    f_btc, r_btc = oos_forecast(j["ofi_I_btc"].to_numpy(),
                                j["ofi_I_eth"].to_numpy(), j["ret_btc"].to_numpy())

    F = np.column_stack([f_btc, f_eth])
    R = np.column_stack([r_btc, r_eth])
    S = np.column_stack([spr["btcusdt"].reindex(j.index).to_numpy(),
                         spr["ethusdt"].reindex(j.index).to_numpy()])
    sig = rolling_forecast_vol(F, window=180)
    ppy = 365 * 24 * 3600 / 10                      # 10s bars, 24/7

    print(f"\n=== REAL DATA PnL demo (1 day, {len(j)} 10s bars, BTC+ETH) ===")
    for c in [0.0, 0.5, 1.0]:
        res = forecast_portfolio(F, R, S, sig, cost_bps=c, periods_per_year=ppy)
        print(f"  cost={c:>3}bp | cum_net={res['cum_net']:+.4f} "
              f"hit={res['hit_rate']:.3f} frac_traded={res['frac_traded']:.2f} "
              f"sharpe(ann)={res.get('sharpe_annual', float('nan')):+.2f}")
    print("  (one day, no signal expected -> PnL ~0/negative; this validates the")
    print("   machinery, not a result. Real test needs the 6-week, 4-asset data.)")


if __name__ == "__main__":
    _self_test()
    _real_demo()

