"""
panel.py - aligned multi-asset modelling panel builder.

Builds a time-aligned panel of OFI features and returns from the per-asset daily
caches produced by scripts/reconstruct.py.
"""

import glob
import os

import numpy as np
import pandas as pd

from ofi import bar_l2_ofi, bucket_times, compute_ofi, fit_integrated_ofi_weights


def load_asset_bars(processed_dir, asset, levels=10, bar_ns=10_000_000_000):
    """Concatenate all cached days for one asset into a bar table with
    columns including 'ofi_I' (integrated OFI) and 'ret'."""
    files = sorted(glob.glob(os.path.join(processed_dir, f"{asset}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {asset}_*.npz in {processed_dir}")
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
    bars = bars[~bars.index.duplicated(keep="first")]
    return bars.dropna(subset=["ret"])


def _collapse_duplicate_seconds(sec, level_ofi, log_mid):
    order = np.argsort(sec)
    sec = np.asarray(sec, dtype=np.int64)[order]
    level_ofi = np.asarray(level_ofi, dtype=np.float32)[order]
    log_mid = np.asarray(log_mid, dtype=np.float64)[order]
    uniq, start = np.unique(sec, return_index=True)
    if len(uniq) == len(sec):
        return sec, level_ofi, log_mid
    level_sum = np.add.reduceat(level_ofi, start, axis=0).astype(np.float32)
    last = np.r_[start[1:] - 1, len(sec) - 1]
    return uniq, level_sum, log_mid[last]


def load_asset_second_panel(processed_dir, asset, levels=10):
    """Build a per-asset 1-second panel from processed books.

    Quote-update OFI is normalized at the transition level, duplicate seconds
    are summed, and the last log mid in the second is kept. PCA weights are fit
    on the first cached day only.
    """
    files = sorted(glob.glob(os.path.join(processed_dir, f"{asset}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {asset}_*.npz in {processed_dir}")

    secs, levels_out, mids = [], [], []
    fitted_w = None
    evr = np.nan
    for f in files:
        sym = dict(np.load(f))
        use_levels = min(levels, sym["bid_p"].shape[1])
        df = compute_ofi(sym, levels=use_levels, normalize=False)
        cols = [f"ofi_L{m}" for m in range(use_levels)]
        raw = df[cols].to_numpy(float)

        depth = 0.5 * np.mean(
            sym["bid_q"][1:, :use_levels] + sym["ask_q"][1:, :use_levels],
            axis=1,
        )
        depth = np.where(depth > 0, depth, np.nan)
        scaled = raw / depth[:, None]
        if fitted_w is None:
            fitted_w, evr = fit_integrated_ofi_weights(scaled)

        mid = 0.5 * (sym["bid_p"][1:, 0] + sym["ask_p"][1:, 0])
        secs.append((sym["ts"][1:] // 1_000_000_000).astype(np.int64))
        levels_out.append(scaled.astype(np.float32))
        mids.append(np.log(mid).astype(np.float64))

    sec = np.concatenate(secs)
    level_ofi = np.concatenate(levels_out, axis=0)
    log_mid = np.concatenate(mids)
    sec, level_ofi, log_mid = _collapse_duplicate_seconds(sec, level_ofi, log_mid)
    return {
        "sec": sec,
        "level_ofi": level_ofi,
        "ofi1": level_ofi[:, 0].astype(np.float32),
        "ofiI": (level_ofi @ fitted_w).astype(np.float32),
        "log_mid": log_mid,
        "pca_weights": fitted_w.astype(float),
        "pc1_evr": float(evr),
    }


def build_second_panel(processed_dir, assets, levels=10):
    """Inner-join per-asset 1-second panels on common seconds.

    The returned structure carries common seconds, per-level OFI, integrated
    OFI, log mids and PCA metadata for each asset.
    """
    per = {}
    common = None
    for asset in assets:
        p = load_asset_second_panel(processed_dir, asset, levels=levels)
        per[asset] = p
        common = p["sec"] if common is None else np.intersect1d(common, p["sec"], assume_unique=True)

    out = {"sec": common}
    meta = {}
    for asset in assets:
        idx = np.searchsorted(per[asset]["sec"], common)
        out[f"level_ofi_{asset}"] = per[asset]["level_ofi"][idx]
        out[f"ofi1_{asset}"] = per[asset]["ofi1"][idx]
        out[f"ofiI_{asset}"] = per[asset]["ofiI"][idx]
        out[f"log_mid_{asset}"] = per[asset]["log_mid"][idx]
        meta[asset] = {
            "pc1_evr": per[asset]["pc1_evr"],
            "pca_weights": per[asset]["pca_weights"].tolist(),
        }
    out["meta"] = meta
    return out


def build_panel(processed_dir, assets, levels=10, bar_ns=10_000_000_000):
    """Inner-join the assets' bars on the shared time grid. Returns a DataFrame
    with columns ofi_I_<asset> and ret_<asset> for each asset."""
    panel = None
    for a in assets:
        b = load_asset_bars(processed_dir, a, levels, bar_ns)
        sub = b[["ofi_I", "ret", "spread"]].rename(
            columns={"ofi_I": f"ofi_I_{a}", "ret": f"ret_{a}", "spread": f"spread_{a}"})
        panel = sub if panel is None else panel.join(sub, how="inner")
    return panel.dropna()


def panel_arrays(panel, assets):
    """Extract (ofi (T,N), ret (T,N), names) from a built panel."""
    ofi = np.column_stack([panel[f"ofi_I_{a}"].to_numpy() for a in assets])
    ret = np.column_stack([panel[f"ret_{a}"].to_numpy() for a in assets])
    return ofi, ret, list(assets)


if __name__ == "__main__":
    # Cached-data smoke test.
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    assets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["btc", "eth"]
    p = build_panel(d, assets, levels=10)
    ofi, ret, names = panel_arrays(p, assets)
    print(f"panel: {len(p)} aligned bars x {len(names)} assets {names}")
    print(f"ofi matrix {ofi.shape}, ret matrix {ret.shape}")
    print(p.head(3).to_string())

