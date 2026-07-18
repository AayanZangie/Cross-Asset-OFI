"""
ofi.py - order-flow imbalance feature construction.

Reads reconstructed order-book snapshots and builds best-level OFI, multi-level
OFI, PCA-integrated OFI, mid price, and bucketed log returns.
"""

import gzip
import heapq
import json
from collections import defaultdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_l2(path, levels=10):
    """Parse gzipped JSONL depth snapshots into per-symbol numpy arrays.

    Returns dict: symbol -> {'ts','bid_p','bid_q','ask_p','ask_q'} where the
    price/size arrays are (T, M), level 0 = best. Snapshots with fewer than
    `levels` levels on either side are skipped.
    """
    rows = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            symbol = (rec.get("stream") or "").split("@", 1)[0]
            data = rec.get("data") or {}
            bids, asks = data.get("bids", []), data.get("asks", [])
            if len(bids) < levels or len(asks) < levels:
                continue
            rows[symbol].append((rec["recv_ns"], bids[:levels], asks[:levels]))

    out = {}
    for symbol, recs in rows.items():
        recs.sort(key=lambda r: r[0])
        out[symbol] = {
            "ts": np.array([r[0] for r in recs], dtype=np.int64),
            "bid_p": np.array([[float(p) for p, _ in r[1]] for r in recs]),
            "bid_q": np.array([[float(q) for _, q in r[1]] for r in recs]),
            "ask_p": np.array([[float(p) for p, _ in r[2]] for r in recs]),
            "ask_q": np.array([[float(q) for _, q in r[2]] for r in recs]),
        }
    return out


def load_l2_delta(path, levels=10, ts_unit_ns=1_000_000):
    """Reconstruct the top-`levels` book from a snapshot+update (delta) stream.

    Handles OKX-style L2 .data files: each line is
        {"instId","action":"snapshot"|"update","ts",
         "bids":[[price,size,norders],...],"asks":[...]}
    Starting from a snapshot, updates are applied incrementally; a level whose
    size is 0 is a deletion. Returns the SAME dict structure as load_l2 (with ts
    in nanoseconds), so compute_ofi / add_integrated_ofi / bar_ofi all work
    unchanged.

    ts_unit_ns scales the source timestamp to ns (OKX ts is in ms -> 1_000_000).

    NOTE: reconstruction is O(rows x levels) and not fast (~90s for a 24h, ~86k
    row file). Treat it as a one-time preprocessing step and cache the result:
        sym = load_l2_delta("data/raw/BTC-...-2026-06-20.data")
        np.savez_compressed("data/processed/btcusdt.npz", **sym)
    then `dict(np.load("data/processed/btcusdt.npz"))` loads instantly later.
    """
    bids, asks = {}, {}
    ts_out, bp, bq, ap, aq = [], [], [], [], []

    def _apply(side, rows):
        for r in rows:
            p, s = float(r[0]), float(r[1])
            if s == 0.0:
                side.pop(p, None)
            else:
                side[p] = s

    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("action") == "snapshot":
                bids, asks = {}, {}
            _apply(bids, r.get("bids", []))
            _apply(asks, r.get("asks", []))
            if len(bids) < levels or len(asks) < levels:
                continue
            tb = heapq.nlargest(levels, bids.items())    # highest bids first
            ta = heapq.nsmallest(levels, asks.items())   # lowest asks first
            ts_out.append(int(r["ts"]) * ts_unit_ns)
            bp.append([p for p, _ in tb]); bq.append([q for _, q in tb])
            ap.append([p for p, _ in ta]); aq.append([q for _, q in ta])

    return {"ts": np.array(ts_out, dtype=np.int64),
            "bid_p": np.array(bp), "bid_q": np.array(bq),
            "ask_p": np.array(ap), "ask_q": np.array(aq)}


# --------------------------------------------------------------------------
# OFI core
# --------------------------------------------------------------------------
def _side_flow(p, q, side):
    """Per-level signed order flow between consecutive snapshots (CKS).
    p, q: (T, M). Returns (T-1, M)."""
    p_prev, p_cur = p[:-1], p[1:]
    q_prev, q_cur = q[:-1], q[1:]
    if side == "bid":
        add = np.where(p_cur >= p_prev, q_cur, 0.0)
        rem = np.where(p_cur <= p_prev, q_prev, 0.0)
    else:  # ask aggression is a price DECREASE
        add = np.where(p_cur <= p_prev, q_cur, 0.0)
        rem = np.where(p_cur >= p_prev, q_prev, 0.0)
    return add - rem


def compute_ofi(sym, levels=1, normalize=False):
    """Per-snapshot OFI increments for one symbol.

    levels=1  -> best-level OFI (column 'ofi').
    levels>1  -> per-level columns 'ofi_L0'.. plus summed 'ofi'.
    normalize -> optional per-step scale normalisation. The main research panel
                 uses bar_l2_ofi, which applies CCZ interval-level depth
                 normalisation after aggregating raw OFI into bars.

    Returns DataFrame: ts, mid_prev, mid_cur, ret (log mid return), ofi (+levels).
    """
    levels = min(levels, sym["bid_p"].shape[1])
    bp, bq = sym["bid_p"][:, :levels], sym["bid_q"][:, :levels]
    ap, aq = sym["ask_p"][:, :levels], sym["ask_q"][:, :levels]

    ofi_lvl = _side_flow(bp, bq, "bid") - _side_flow(ap, aq, "ask")  # +=buying

    if normalize:
        depth = 0.5 * (bq + aq)                       # (T, M) per-level half-depth
        avg_depth = 0.5 * (depth[:-1] + depth[1:])    # (T-1, M)
        Q = np.nanmean(avg_depth, axis=1, keepdims=True)   # one scalar per step
        ofi_lvl = ofi_lvl / np.where(Q > 0, Q, np.nan)

    mid = 0.5 * (sym["bid_p"][:, 0] + sym["ask_p"][:, 0])
    df = pd.DataFrame({
        "ts": sym["ts"][1:],
        "mid_prev": mid[:-1],
        "mid_cur": mid[1:],
        "ret": np.log(mid[1:] / mid[:-1]),
    })
    if levels == 1:
        df["ofi"] = ofi_lvl[:, 0]
    else:
        for m in range(levels):
            df[f"ofi_L{m}"] = ofi_lvl[:, m]
        df["ofi"] = np.nansum(ofi_lvl, axis=1)
    return df


def fit_integrated_ofi_weights(level_ofi):
    """Fit CCZ eq. 4 PCA weights on the supplied historical rows only."""
    X = np.asarray(level_ofi, dtype=float)
    mask = ~np.isnan(X).any(axis=1)
    if mask.sum() < 2:
        raise ValueError("need at least two complete rows to fit integrated OFI")
    Xc = X[mask] - X[mask].mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if np.sum(S ** 2) <= 0:
        raise ValueError("cannot fit integrated OFI on zero-variance data")
    w1 = Vt[0]
    if w1.sum() < 0:
        w1 = -w1
    w1n = w1 / np.sum(np.abs(w1))
    evr = (S[0] ** 2) / np.sum(S ** 2)
    return w1n, evr


def integrated_ofi(level_ofi, weights=None):
    """Collapse multi-level OFI into one integrated OFI via PCA (CCZ eq. 4).

    level_ofi: (T, M) array of depth-normalised per-level OFIs.
    weights: optional PC weights fitted on a historical sample.
    Returns (integrated (T,), l1-normalised weights (M,), PC1 variance ratio).
    """
    X = np.asarray(level_ofi, dtype=float)
    if weights is None:
        w, evr = fit_integrated_ofi_weights(X)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != X.shape[1]:
            raise ValueError("weights must match the number of OFI levels")
        denom = np.sum(np.abs(w))
        if denom <= 0:
            raise ValueError("weights must not all be zero")
        w = w / denom
        evr = np.nan
    return X @ w, w, evr


def add_integrated_ofi(df, weights=None, fit_mask=None):
    """Add an 'ofi_I' column from the ofi_L* columns. Returns (df, weights, evr).

    Pass weights, or pass fit_mask to fit PCA weights on a historical subset and
    apply them to the full frame.
    """
    cols = sorted((c for c in df.columns if c.startswith("ofi_L")),
                  key=lambda c: int(c.split("L")[1]))
    X = df[cols].to_numpy()
    if weights is None and fit_mask is not None:
        weights, evr = fit_integrated_ofi_weights(X[np.asarray(fit_mask)])
        integ, w, _ = integrated_ofi(X, weights=weights)
    else:
        integ, w, evr = integrated_ofi(X, weights=weights)
    df = df.copy()
    df["ofi_I"] = integ
    return df, w, evr


def bucket_times(ts, bar_ns):
    """Assign timestamps to right-closed buckets (start, start+bar]."""
    x = np.asarray(ts, dtype=np.int64)
    return ((np.maximum(x, 1) - 1) // bar_ns) * bar_ns


def bar_ofi(df, bar_ns, min_coverage=None, expected_updates_per_second=1.0):
    """Aggregate per-snapshot increments into fixed right-closed time buckets.
    OFI sums within the bucket; return is the log mid move across it.

    min_coverage is an optional bar-validity filter for second-level parity
    checks. For example, min_coverage=0.8 on a 5s bar keeps only bars with at
    least ceil(0.8 * 5) observed update seconds. It defaults off so existing
    grids are byte-for-byte compatible unless they opt in.
    """
    bucket = bucket_times(df["ts"].to_numpy(), bar_ns)
    g = df.groupby(bucket)
    ofi_cols = [c for c in df.columns if c.startswith("ofi")]
    out = g[ofi_cols].sum()
    out["ret"] = np.log(g["mid_cur"].last() / g["mid_prev"].first())
    if min_coverage is not None:
        expected = max(1.0, (bar_ns / 1_000_000_000.0) * expected_updates_per_second)
        min_count = int(np.ceil(float(min_coverage) * expected))
        counts = g.size()
        out = out[counts.reindex(out.index).to_numpy() >= min_count]
    out.index.name = "bar_start_ns"
    return out


def bar_l2_ofi(sym, levels=10, bar_ns=10_000_000_000, weights=None,
               min_coverage=None, expected_updates_per_second=1.0):
    """Build CCZ-style bar OFI from a reconstructed L2 book.

    The depth normalization is applied after aggregating raw OFI into the bar.
    Each level's interval OFI is divided by one average top-M depth scalar for
    that interval.

    Returns (bars, weights, evr), where bars contains normalized ofi_L* columns,
    'ofi' as their sum, 'ofi_I' as PCA-integrated OFI, and 'ret'.
    """
    df = compute_ofi(sym, levels=levels, normalize=False)
    b = bar_ofi(df, bar_ns=bar_ns, min_coverage=min_coverage,
                expected_updates_per_second=expected_updates_per_second)

    levels = min(levels, sym["bid_p"].shape[1])
    depth = 0.5 * (sym["bid_q"][:, :levels] + sym["ask_q"][:, :levels])
    bucket = bucket_times(sym["ts"], bar_ns)
    q = pd.Series(np.nanmean(depth, axis=1), index=bucket).groupby(level=0).mean()
    qv = q.reindex(b.index).to_numpy()
    qv = np.where(qv > 0, qv, np.nan)

    if levels == 1:
        b["ofi"] = b["ofi"] / qv
        b["ofi_I"] = b["ofi"]
        return b, weights, np.nan

    cols = [f"ofi_L{m}" for m in range(levels)]
    b[cols] = b[cols].div(qv, axis=0)
    b["ofi"] = b[cols].sum(axis=1)
    b, weights, evr = add_integrated_ofi(b, weights=weights)
    return b, weights, evr


# --------------------------------------------------------------------------
# Synthetic self-test
# --------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)

    # Test 1: price-move branches signed correctly
    sym = {"ts": np.array([0, 1]), "bid_p": np.array([[100.0], [100.01]]),
           "bid_q": np.array([[5.0], [5.0]]), "ask_p": np.array([[100.02], [100.02]]),
           "ask_q": np.array([[5.0], [5.0]])}
    assert np.isclose(compute_ofi(sym, 1)["ofi"].iloc[0], 5.0)
    sym["bid_p"] = np.array([[100.0], [99.99]])
    assert np.isclose(compute_ofi(sym, 1)["ofi"].iloc[0], -5.0)

    T, base = 4000, 50.0
    # Test 2a: exact recovery under constant prices
    g = rng.standard_normal(T)
    sym3 = {"ts": np.arange(T), "bid_p": np.full((T, 1), 99.99),
            "bid_q": (base + np.cumsum(np.maximum(g, 0.0))).reshape(-1, 1),
            "ask_p": np.full((T, 1), 100.01),
            "ask_q": (base + np.cumsum(np.maximum(-g, 0.0))).reshape(-1, 1)}
    df3 = compute_ofi(sym3, 1)
    assert np.allclose(df3["ofi"].to_numpy(), g[1:])

    # Test 2b: OFI -> return link under a coherent moving-price sim
    f = rng.standard_normal(T)
    mid = 100.0 + np.cumsum(0.005 * f + 0.002 * rng.standard_normal(T))
    sym_b = {"ts": np.arange(T), "bid_p": (mid - 0.01).reshape(-1, 1),
             "ask_p": (mid + 0.01).reshape(-1, 1),
             "bid_q": (base + 3 * np.maximum(f, 0) + rng.standard_normal(T)).reshape(-1, 1),
             "ask_q": (base + 3 * np.maximum(-f, 0) + rng.standard_normal(T)).reshape(-1, 1)}
    dfb = compute_ofi(sym_b, 1)
    beta = np.polyfit(dfb["ofi"], dfb["ret"], 1)[0]
    corr_ret = np.corrcoef(dfb["ofi"], dfb["ret"])[0, 1]
    assert beta > 0 and corr_ret > 0.8

    # Test 3: integrated OFI via PCA recovers a common factor
    M = 10
    common = rng.standard_normal(T)
    loadings = rng.uniform(0.5, 1.5, M)
    X = np.outer(common, loadings) + 0.3 * rng.standard_normal((T, M))
    integ, w, evr = integrated_ofi(X)
    assert evr > 0.7, f"PC1 should dominate, evr={evr:.3f}"
    assert abs(np.corrcoef(integ, common)[0, 1]) > 0.95
    assert np.isclose(np.sum(np.abs(w)), 1.0)

    # Test 4: full multi-level path + integration + bars on the sim
    Mb = 5
    sym4 = {"ts": np.arange(T),
            "bid_p": np.tile((mid - 0.01).reshape(-1, 1), (1, Mb)),
            "bid_q": np.tile(sym_b["bid_q"], (1, Mb)) + rng.standard_normal((T, Mb)),
            "ask_p": np.tile((mid + 0.01).reshape(-1, 1), (1, Mb)),
            "ask_q": np.tile(sym_b["ask_q"], (1, Mb)) + rng.standard_normal((T, Mb))}
    dfm = compute_ofi(sym4, levels=Mb, normalize=True)
    dfm, w4, evr4 = add_integrated_ofi(dfm)
    assert "ofi_I" in dfm.columns
    bars = bar_ofi(dfm, bar_ns=100)
    assert "ret" in bars.columns and len(bars) > 0

    print("ofi.py self-test passed:")
    print(f"  exact recovery (const prices) corr = "
          f"{np.corrcoef(df3['ofi'], g[1:])[0,1]:.4f}")
    print(f"  OFI -> return                 beta = {beta:.2e} (>0), corr = {corr_ret:.3f}")
    print(f"  integrated OFI (PCA)          evr  = {evr:.3f}, corr w/ factor = "
          f"{abs(np.corrcoef(integ, common)[0,1]):.3f}")
    print(f"  multi-level + bars            {len(bars)} bars, PC1 evr = {evr4:.3f}")


if __name__ == "__main__":
    _self_test()
