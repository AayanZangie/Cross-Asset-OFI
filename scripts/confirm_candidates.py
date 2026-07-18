"""Targeted confirmation pass for the best-looking broad-grid candidates."""

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tests as grid  # noqa: E402


OUT = ROOT / "output" / "confirmation"

LAGSETS = {
    "single": [1],
    "low": [1, 2, 3],
    "mid": [1, 2, 3, 5, 10],
    "ccz": [1, 2, 3, 5, 10, 20, 30],
}

CANDIDATES = [
    # 300s h=2 candidate family.
    {"bar_s": 300, "scheme": "pca", "lagset": "ccz", "horizon": 2, "tag": "300s_pca_ccz_h2"},
    {"bar_s": 300, "scheme": "best", "lagset": "ccz", "horizon": 2, "tag": "300s_best_ccz_h2"},
    {"bar_s": 300, "scheme": "sum", "lagset": "ccz", "horizon": 2, "tag": "300s_sum_ccz_h2"},
    {"bar_s": 300, "scheme": "distance", "lagset": "ccz", "horizon": 2, "tag": "300s_distance_ccz_h2"},
    # Longer 300s horizon variants.
    {"bar_s": 300, "scheme": "pca", "lagset": "ccz", "horizon": 5, "tag": "300s_pca_ccz_h5"},
    {"bar_s": 300, "scheme": "pca", "lagset": "ccz", "horizon": 10, "tag": "300s_pca_ccz_h10"},
    {"bar_s": 300, "scheme": "pca", "lagset": "ccz", "horizon": 20, "tag": "300s_pca_ccz_h20"},
    # 60s reference variants.
    {"bar_s": 60, "scheme": "pca", "lagset": "ccz", "horizon": 5, "tag": "60s_pca_ccz_h5"},
    {"bar_s": 60, "scheme": "best", "lagset": "low", "horizon": 5, "tag": "60s_best_low_h5"},
    # 5s reference variant.
    {"bar_s": 5, "scheme": "pca", "lagset": "ccz", "horizon": 1, "tag": "5s_pca_ccz_h1"},
]


def load_panel(bar_s, levels, cache):
    key = (bar_s, levels)
    if key in cache:
        return cache[key]
    old_levels = grid.LEVELS
    grid.LEVELS = levels
    bars = {a: grid.load_levels_bars(a, bar_s * 1_000_000_000, levels=levels)
            for a in grid.ASSETS}
    cache[key] = bars
    grid.LEVELS = old_levels
    return bars


def run_one(spec, train, refit_cap, levels, cache):
    old_train, old_refit, old_levels = grid.TRAIN, grid.FOLDCAP, grid.LEVELS
    grid.TRAIN = train
    grid.FOLDCAP = refit_cap
    grid.LEVELS = levels
    bars = load_panel(spec["bar_s"], levels, cache)
    ofi, ret, spr = grid.build_scheme_panel(bars, spec["scheme"])
    res = grid.eval_config(ofi, ret, spr, LAGSETS[spec["lagset"]], spec["horizon"])
    grid.TRAIN, grid.FOLDCAP, grid.LEVELS = old_train, old_refit, old_levels
    out = dict(spec)
    out.update(res)
    out["train"] = train
    out["refit_cap"] = refit_cap
    out["levels"] = levels
    out["horizon_seconds"] = spec["bar_s"] * spec["horizon"]
    return out


def write_summary(df):
    cols = [
        "tag", "bar_s", "scheme", "lagset", "horizon", "horizon_seconds",
        "train", "levels", "own_r2", "cross_r2", "cross_minus_own_r2",
        "cross_minus_car_r2", "own_hit", "cross_hit", "pnl_net",
        "pnl_net_0p25bp", "pnl_net_0p5bp", "pnl_net_1p0bp",
        "turnover", "best_asset", "best_asset_cross_minus_own_r2",
    ]
    lines = ["# Candidate Confirmation", ""]
    lines.append("Dense confirmation uses refit_cap=200 and two training windows.")
    lines.append("")
    lines.append("## Ranked By Cross vs Own R2")
    lines.append(df.sort_values("cross_minus_own_r2", ascending=False)[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Ranked By Net PnL At 0.5bp")
    lines.append(df.sort_values("pnl_net", ascending=False)[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Per-Asset Metrics For Top Rows")
    asset_rows = []
    for _, row in df.sort_values("cross_minus_own_r2", ascending=False).head(10).iterrows():
        for metric in json.loads(row["asset_metrics_json"]):
            item = {k: row[k] for k in ["tag", "bar_s", "scheme", "lagset", "horizon", "train", "levels", "pnl_net"]}
            item.update(metric)
            asset_rows.append(item)
    lines.append(pd.DataFrame(asset_rows).to_markdown(index=False))
    (OUT / "confirmation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cache = {}
    for levels in [10, 20]:
        for train in [3000, 6000]:
            for spec in CANDIDATES:
                print(f"confirm {spec['tag']} train={train} levels={levels}", flush=True)
                rows.append(run_one(spec, train=train, refit_cap=200, levels=levels, cache=cache))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "confirmation.csv", index=False)
    write_summary(df)
    print(OUT / "confirmation.md")


if __name__ == "__main__":
    main()
