"""Robustness battery for the sub-minute parity sweep.

Outputs are written to output/subminute/ and do not overwrite the broad grid or
the 300s/CCZ robustness files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tests as grid  # noqa: E402
from significance_tests import diebold_mariano, model_confidence_set  # noqa: E402


OUT = ROOT / "output" / "subminute"
SEARCH_M = 684


def finite(x):
    try:
        y = float(x)
    except Exception:
        return None
    return y if np.isfinite(y) else None


def corrected_pvalues(df, pcol, m=SEARCH_M):
    p = pd.to_numeric(df[pcol], errors="coerce").to_numpy(float)
    bonf = np.minimum(p * m, 1.0)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    holm = np.full(len(p), np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)
        holm[idx] = running
    bh = np.full(len(p), np.nan)
    finite_order = [idx for idx in order if np.isfinite(p[idx])]
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(finite_order), start=1):
        rank = len(finite_order) - rank_from_end + 1
        val = min(p[idx] * m / rank, 1.0)
        running = min(running, val)
        bh[idx] = running
    return bonf, holm, bh


def run_cells(second):
    rows = []
    sig_rows = []
    for bar_s in grid.SUBMINUTE_BARS_S:
        for scheme in grid.SCHEMES:
            print(f"subminute robustness bar={bar_s}s scheme={scheme}", flush=True)
            real, plac, real_losses, plac_losses = grid.eval_subminute_placebo_pair(second, bar_s, scheme)
            base = {
                "spec": "subminute",
                "bar_s": bar_s,
                "scheme": scheme,
                "lagset": "single_bar",
                "horizon": 1,
                "horizon_seconds": bar_s,
                "levels": grid.LEVELS,
                "train": real["train"],
                "test": real["test"],
                "step": real["step"],
            }
            rows.append({**base, "kind": "real", **real})
            rows.append({**base, "kind": "placebo", **plac})
            rows.append({
                **base,
                "kind": "real_minus_placebo",
                "cross_minus_own_r2": finite(real["cross_minus_own_r2"] - plac["cross_minus_own_r2"]),
                "cross_minus_car_r2": finite(real["cross_minus_car_r2"] - plac["cross_minus_car_r2"]),
                "cross_hit": finite(real["cross_hit"] - plac["cross_hit"]),
                "cross_unit_pnl_bps": finite(real["cross_unit_pnl_bps"] - plac["cross_unit_pnl_bps"]),
            })

            dm_own, p_own = diebold_mariano(real_losses["own"], real_losses["cross"], horizon=1)
            dm_car, p_car = diebold_mariano(real_losses["car"], real_losses["cross"], horizon=1)
            joined = pd.DataFrame({
                "placebo_cross": plac_losses["cross"],
                "real_cross": real_losses["cross"],
            }).dropna()
            dm_placebo, p_placebo = diebold_mariano(joined["placebo_cross"], joined["real_cross"], horizon=1)
            try:
                losses = real_losses[["own", "car", "cross"]].dropna()
                mcs = model_confidence_set(losses, alpha=0.10, reps=100,
                                           block_size=max(5, 60 // bar_s),
                                           seed=9000 + 17 * bar_s + grid.SCHEMES.index(scheme))
                mcs_included = ",".join(mcs["included"])
            except Exception as exc:
                mcs_included = f"ERROR:{exc!r}"

            sig_rows.append({
                **base,
                "dm_own_minus_cross": finite(dm_own),
                "p_own_vs_cross": finite(p_own),
                "dm_car_minus_cross": finite(dm_car),
                "p_car_vs_cross": finite(p_car),
                "dm_placebo_cross_minus_real_cross": finite(dm_placebo),
                "p_placebo_vs_real_cross": finite(p_placebo),
                "mcs_included_10pct": mcs_included,
            })

    sig = pd.DataFrame(sig_rows)
    for pcol in ["p_own_vs_cross", "p_car_vs_cross", "p_placebo_vs_real_cross"]:
        bonf, holm, bh = corrected_pvalues(sig, pcol, SEARCH_M)
        sig[f"{pcol}_bonf684"] = bonf
        sig[f"{pcol}_holm684"] = holm
        sig[f"{pcol}_bh684"] = bh
    return pd.DataFrame(rows), sig


def pivot_real_placebo(df):
    keys = ["bar_s", "scheme", "lagset", "horizon", "horizon_seconds", "levels"]
    metrics = ["cross_minus_own_r2", "cross_minus_car_r2", "cross_hit", "cross_unit_pnl_bps"]
    real = df[df["kind"] == "real"][keys + metrics].rename(columns={c: f"real_{c}" for c in metrics})
    plac = df[df["kind"] == "placebo"][keys + metrics].rename(columns={c: f"placebo_{c}" for c in metrics})
    out = real.merge(plac, on=keys, how="inner")
    for c in metrics:
        out[f"delta_{c}"] = out[f"real_{c}"] - out[f"placebo_{c}"]
    return out


def write_report(results, placebo, sig):
    lines = [
        "# Subminute Robustness Report",
        "",
        "This report runs the sub-minute parity grid inside the main repo: bar sizes {1s, 5s, 10s}, schemes {best, sum, distance, pca}, single current-bar OFI, horizon=1 bar, 7-day train, 1-day test, 7-day step, tuned ridge estimator.",
        "",
        "Supersession note: these results replace the earlier `subminute` outputs. The earlier run chose walk-forward split points by ordinal row position after filtering an irregular bar clock, so the headline and paired-placebo paths could test different calendar periods. This run fixes train/test boundaries in absolute UTC seconds first, then fills each window with valid rows. The headline table is now sourced from the paired-real evaluator, so headline and placebo-comparison numbers use one boundary schedule and one evaluated population.",
        "",
        "Multiple-comparison correction uses Bonferroni/Holm/BH with m=684: the original 672-cell search plus the 12 new subminute cells.",
        "",
        "## Headline Cells",
        "",
    ]
    head_cols = ["bar_s", "scheme", "cross_minus_own_r2", "cross_minus_car_r2",
                 "cross_hit", "cross_unit_pnl_bps", "status"]
    lines.append(results[head_cols].to_markdown(index=False))
    lines.append("")

    pp = pivot_real_placebo(placebo)
    lines += [
        "## Placebo Comparison",
        "",
    ]
    pcols = ["bar_s", "scheme", "real_cross_minus_own_r2", "placebo_cross_minus_own_r2",
             "delta_cross_minus_own_r2", "real_cross_minus_car_r2",
             "placebo_cross_minus_car_r2", "delta_cross_minus_car_r2",
             "real_cross_unit_pnl_bps", "placebo_cross_unit_pnl_bps",
             "delta_cross_unit_pnl_bps"]
    lines.append(pp[pcols].to_markdown(index=False))
    lines.append("")

    sig_cols = ["bar_s", "scheme", "dm_own_minus_cross", "p_own_vs_cross_bonf684",
                "dm_car_minus_cross", "p_car_vs_cross_bonf684",
                "dm_placebo_cross_minus_real_cross",
                "p_placebo_vs_real_cross_bonf684", "mcs_included_10pct"]
    lines += [
        "## Corrected Significance",
        "",
        sig[sig_cols].to_markdown(index=False),
        "",
        "| Test | Passing cells after Bonferroni(684) |",
        "|---|---:|",
        f"| Cross beats own | {int((sig['p_own_vs_cross_bonf684'] < 0.05).sum())} / {len(sig)} |",
        f"| Cross beats CAR | {int((sig['p_car_vs_cross_bonf684'] < 0.05).sum())} / {len(sig)} |",
        f"| Real cross beats shifted placebo | {int((sig['p_placebo_vs_real_cross_bonf684'] < 0.05).sum())} / {len(sig)} |",
        "",
    ]

    lines += ["## Interpretation", ""]
    own_pass = int((sig["p_own_vs_cross_bonf684"] < 0.05).sum())
    car_pass = int((sig["p_car_vs_cross_bonf684"] < 0.05).sum())
    plac_pass = int((sig["p_placebo_vs_real_cross_bonf684"] < 0.05).sum())
    if own_pass and car_pass and plac_pass:
        lines.append("The main repo subminute path supports a positive sub-minute cross-OFI result under the strict controls.")
    elif own_pass:
        lines.append("The main repo subminute path finds cross-over-own evidence, but the stricter CAR/placebo controls prevent a conclusive positive.")
    else:
        lines.append("The main repo subminute path does not find corrected cross-over-own evidence in this sweep.")
    lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "results.csv"
    if not result_path.exists():
        print("output/subminute/results.csv not found; running subminute sweep first", flush=True)
        grid.run_subminute_grid()
    results = pd.read_csv(result_path)
    second = grid.build_subminute_second_panel(grid.PROCESSED, grid.ASSETS, levels=grid.LEVELS)
    placebo, sig = run_cells(second)
    placebo.to_csv(OUT / "placebo.csv", index=False)
    sig.to_csv(OUT / "significance.csv", index=False)
    write_report(results, placebo, sig)
    print(OUT / "report.md")


if __name__ == "__main__":
    main()

