"""Summarise the broad tests.py grid into ranked result artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "grid"
RESULTS = OUT / "results.csv"


NUMERIC = [
    "bar_s", "horizon", "levels", "refit_cap", "secs",
    "ar_r2", "car_r2", "own_r2", "cross_r2",
    "cross_minus_own_r2", "cross_minus_ar_r2", "cross_minus_car_r2",
    "ar_hit", "car_hit", "own_hit", "cross_hit",
    "pnl_gross", "pnl_net", "pnl_hit", "turnover",
    "pnl_net_0p0bp", "pnl_net_0p25bp", "pnl_net_0p5bp", "pnl_net_1p0bp",
    "best_asset_cross_minus_own_r2",
]


def load_results():
    df = pd.read_csv(RESULTS)
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["horizon_seconds"] = df["bar_s"] * df["horizon"]
    return df


def asset_frame(df):
    rows = []
    for _, row in df.iterrows():
        try:
            metrics = json.loads(row.get("asset_metrics_json", "[]"))
        except Exception:
            metrics = []
        for metric in metrics:
            out = {
                "key": row["key"],
                "bar_s": row["bar_s"],
                "scheme": row["scheme"],
                "lagset": row["lagset"],
                "horizon": row["horizon"],
                "horizon_seconds": row["horizon_seconds"],
                "pnl_net": row["pnl_net"],
            }
            out.update(metric)
            rows.append(out)
    adf = pd.DataFrame(rows)
    for col in ["bar_s", "horizon", "horizon_seconds", "pnl_net",
                "ar_r2", "car_r2", "own_r2", "cross_r2",
                "cross_minus_own_r2", "ar_hit", "car_hit", "own_hit", "cross_hit"]:
        if col in adf:
            adf[col] = pd.to_numeric(adf[col], errors="coerce")
    return adf


def candidate_score(df):
    hit_edge = df["cross_hit"].fillna(0.5) - 0.5
    pnl_scaled = np.tanh(df["pnl_net"].fillna(0.0) / 2.0)
    df = df.copy()
    df["candidate_score"] = (
        1000.0 * df["cross_minus_own_r2"].fillna(-1.0)
        + 1000.0 * df["cross_minus_car_r2"].fillna(-1.0)
        + 2.0 * hit_edge
        + 0.25 * pnl_scaled
    )
    df["passes_short_filter"] = (
        (df["horizon_seconds"] <= 600)
        & (df["cross_minus_own_r2"] > 0)
        & (df["cross_minus_car_r2"] > 0)
        & (df["cross_hit"] > 0.505)
        & (df["pnl_net"] > 0)
    )
    return df


def write_markdown(df, adf):
    short = df[df["horizon_seconds"] <= 600].copy()
    ranked_cols = [
        "key", "bar_s", "scheme", "lagset", "horizon", "horizon_seconds",
        "own_r2", "cross_r2", "cross_minus_own_r2", "cross_minus_car_r2",
        "own_hit", "cross_hit", "pnl_net", "turnover",
        "best_asset", "best_asset_cross_minus_own_r2", "candidate_score",
    ]
    lines = []
    lines.append("# Broad Grid Summary")
    lines.append("")
    lines.append(f"Rows: {len(df)}. Status counts: {df['status'].value_counts().to_dict()}.")
    lines.append("")
    lines.append("## Main Read")
    lines.append("")
    lines.append("- The strongest aggregate cross-OFI gains are concentrated in 300-second bars, especially CCZ lag windows.")
    lines.append("- The best short-horizon candidates are 300s x 2 bars, i.e. 10-minute forecasts, not the original 60s paper-style run.")
    lines.append("- 5-second candidates can show R2 gains, but net PnL is strongly negative because turnover is too high.")
    lines.append("- Many top PnL rows do not beat own OFI or CAR in R2, so PnL alone is not a reliable selection rule.")
    lines.append("")
    lines.append("## Best Short Candidates")
    filt = df[df["passes_short_filter"]].sort_values("candidate_score", ascending=False)
    lines.append(filt[ranked_cols].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Top Aggregate Cross R2 Improvement")
    lines.append(df.sort_values("cross_minus_own_r2", ascending=False)[ranked_cols].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Top Net PnL")
    lines.append(df.sort_values("pnl_net", ascending=False)[ranked_cols].head(20).to_markdown(index=False))
    lines.append("")
    lines.append("## Top Per-Asset Short R2 Improvement")
    asset_cols = [
        "key", "asset", "bar_s", "scheme", "lagset", "horizon", "horizon_seconds",
        "own_r2", "cross_r2", "cross_minus_own_r2", "own_hit", "cross_hit", "pnl_net",
    ]
    short_assets = adf[adf["horizon_seconds"] <= 600].sort_values("cross_minus_own_r2", ascending=False)
    lines.append(short_assets[asset_cols].head(30).to_markdown(index=False))
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = candidate_score(load_results())
    adf = asset_frame(df)
    df.sort_values("candidate_score", ascending=False).to_csv(OUT / "ranked.csv", index=False)
    adf.sort_values("cross_minus_own_r2", ascending=False).to_csv(OUT / "asset_ranked.csv", index=False)
    write_markdown(df, adf)
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
