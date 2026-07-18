"""
significance_tests.py - Diebold-Mariano and Model Confidence Set helpers.
"""

import numpy as np
import pandas as pd
from scipy import stats


def squared_error(forecast, realized):
    return (np.asarray(realized, float) - np.asarray(forecast, float)) ** 2


def diebold_mariano(loss_a, loss_b, horizon=1):
    """Return (DM_stat, p_value) for H0: equal expected loss.

    Positive DM means model A has higher loss than model B.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    L = max(horizon - 1, 0)
    lrv = np.mean(dc ** 2)
    for k in range(1, L + 1):
        cov = np.mean(dc[k:] * dc[:-k])
        lrv += 2.0 * (1.0 - k / (L + 1)) * cov
    var_dbar = lrv / T
    if var_dbar <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_dbar)
    h = horizon
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    dm *= hln
    p = 2.0 * stats.t.sf(abs(dm), df=T - 1)
    return float(dm), float(p)


def model_confidence_set(losses, alpha=0.10, reps=1000, block_size=None,
                         method="R", seed=0):
    """Run Hansen-Lunde-Nason's Model Confidence Set via arch."""
    from arch.bootstrap import MCS
    if not isinstance(losses, pd.DataFrame):
        losses = pd.DataFrame(losses,
                              columns=[f"m{i}" for i in range(np.shape(losses)[1])])
    losses = losses.dropna()
    rs = np.random.RandomState(seed)
    kw = dict(size=alpha, reps=reps, method=method, seed=rs)
    if block_size is not None:
        kw["block_size"] = block_size
    mcs = MCS(losses, **kw)
    mcs.compute()
    pv = mcs.pvalues.iloc[:, 0].to_dict()
    return {"included": list(mcs.included),
            "excluded": list(mcs.excluded),
            "pvalues": pv}


def compare_models(forecasts, realized, horizon=1, alpha=0.10, reps=1000):
    """Build squared-error losses and compare models by MCS and DM tests."""
    names = list(forecasts)
    real = np.asarray(realized, float)
    loss = {n: squared_error(forecasts[n], real) for n in names}
    L = pd.DataFrame(loss)
    mean_loss = L.mean()
    best = mean_loss.idxmin()
    mcs = model_confidence_set(L, alpha=alpha, reps=reps)
    dm = {}
    for n in names:
        if n == best:
            continue
        stat, p = diebold_mariano(loss[n], loss[best], horizon=horizon)
        dm[n] = {"vs_best": best, "dm_stat": stat, "p_value": p}
    return {"best": best,
            "mean_loss": mean_loss.to_dict(),
            "mcs_included": mcs["included"],
            "mcs_pvalues": mcs["pvalues"],
            "dm_vs_best": dm}


def _self_test():
    rng = np.random.default_rng(7)
    T = 3000
    signal = rng.standard_normal(T)
    realized = signal + 0.5 * rng.standard_normal(T)
    f_best = signal + 0.20 * rng.standard_normal(T)
    f_good = signal + 0.20 * rng.standard_normal(T)
    f_ok = signal + 0.60 * rng.standard_normal(T)
    f_bad = rng.standard_normal(T)
    _, p_bad = diebold_mariano(squared_error(f_best, realized),
                               squared_error(f_bad, realized))
    _, p_good = diebold_mariano(squared_error(f_best, realized),
                                squared_error(f_good, realized))
    assert p_bad < 0.01
    assert p_good > 0.01
    res = compare_models({"best": f_best, "good": f_good, "ok": f_ok, "bad": f_bad},
                         realized, horizon=1, alpha=0.10, reps=500)
    assert res["best"] in ("best", "good")
    assert "bad" not in res["mcs_included"]
    print("significance_tests.py self-test passed")


if __name__ == "__main__":
    _self_test()
