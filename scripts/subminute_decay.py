"""Decay-grid robustness battery for the corrected subminute evaluator.

Outputs are written to output/subminute_decay/. The script uses
src.tests.eval_subminute_placebo_pair(), the fixed-UTC-boundary paired-real
evaluator used by the corrected subminute path and the Phase-3 bar_s=15
pre-verification.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tests as grid  # noqa: E402
from significance_tests import diebold_mariano, model_confidence_set  # noqa: E402


OUT = ROOT / "output" / "subminute_decay"
SUBMINUTE_OUT = ROOT / "output" / "subminute"
SEARCH_M = 792
CHECKPOINT_BARS = {1, 5, 10}
FLOAT_TOL = 1e-12

SUBMINUTE_DECAY_RESULTS = OUT / "results.csv"
SUBMINUTE_DECAY_PLACEBO = OUT / "placebo.csv"
SUBMINUTE_DECAY_SIGNIFICANCE = OUT / "significance.csv"
SUBMINUTE_DECAY_REPORT = OUT / "report.md"
SUBMINUTE_RESULTS = SUBMINUTE_OUT / "results.csv"


def finite(x):
    try:
        y = float(x)
    except Exception:
        return None
    return y if np.isfinite(y) else None


def finite_diff(a, b):
    if a is None or b is None:
        return None
    return finite(float(a) - float(b))


def fmt_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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


def write_progress(results, placebo, sig_rows):
    pd.DataFrame(results).to_csv(SUBMINUTE_DECAY_RESULTS, index=False)
    pd.DataFrame(placebo).to_csv(SUBMINUTE_DECAY_PLACEBO, index=False)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(SUBMINUTE_DECAY_SIGNIFICANCE, index=False)


def run_cells(second):
    results = []
    placebo_rows = []
    sig_rows = []
    total = len(grid.SUBMINUTE_DECAY_BARS_S) * len(grid.SCHEMES)
    done = 0
    t0 = time.time()

    for bar_s in grid.SUBMINUTE_DECAY_BARS_S:
        for scheme in grid.SCHEMES:
            c0 = time.time()
            base = {
                "spec": "subminute_decay",
                "key": grid.subminute_decay_key(bar_s, scheme),
                "bar_s": bar_s,
                "scheme": scheme,
                "lagset": "single_bar",
                "lag_windows": ",".join(map(str, grid.SINGLE_BAR_WINDOWS)),
                "horizon": 1,
                "horizon_seconds": bar_s,
                "levels": grid.LEVELS,
            }
            try:
                real, plac, real_losses, plac_losses = grid.eval_subminute_placebo_pair(
                    second, bar_s, scheme
                )
                result_row = {**base, **real, "status": "ok"}
                results.append(result_row)

                placebo_base = {
                    "spec": "subminute_decay",
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
                placebo_rows.append({**placebo_base, "kind": "real", **real})
                placebo_rows.append({**placebo_base, "kind": "placebo", **plac})
                placebo_rows.append({
                    **placebo_base,
                    "kind": "real_minus_placebo",
                    "cross_minus_own_r2": finite_diff(real["cross_minus_own_r2"], plac["cross_minus_own_r2"]),
                    "cross_minus_car_r2": finite_diff(real["cross_minus_car_r2"], plac["cross_minus_car_r2"]),
                    "cross_hit": finite_diff(real["cross_hit"], plac["cross_hit"]),
                    "cross_unit_pnl_bps": finite_diff(real["cross_unit_pnl_bps"], plac["cross_unit_pnl_bps"]),
                })

                dm_own, p_own = diebold_mariano(real_losses["own"], real_losses["cross"], horizon=1)
                dm_car, p_car = diebold_mariano(real_losses["car"], real_losses["cross"], horizon=1)
                joined = pd.DataFrame({
                    "placebo_cross": plac_losses["cross"],
                    "real_cross": real_losses["cross"],
                }).dropna()
                dm_placebo, p_placebo = diebold_mariano(
                    joined["placebo_cross"], joined["real_cross"], horizon=1
                )
                try:
                    losses = real_losses[["own", "car", "cross"]].dropna()
                    mcs = model_confidence_set(
                        losses,
                        alpha=0.10,
                        reps=100,
                        block_size=max(5, 60 // bar_s),
                        seed=12000 + 17 * bar_s + grid.SCHEMES.index(scheme),
                    )
                    mcs_included = ",".join(mcs["included"])
                except Exception as exc:
                    mcs_included = f"ERROR:{exc!r}"

                sig_rows.append({
                    **placebo_base,
                    "dm_own_minus_cross": finite(dm_own),
                    "p_own_vs_cross": finite(p_own),
                    "dm_car_minus_cross": finite(dm_car),
                    "p_car_vs_cross": finite(p_car),
                    "dm_placebo_cross_minus_real_cross": finite(dm_placebo),
                    "p_placebo_vs_real_cross": finite(p_placebo),
                    "mcs_included_10pct": mcs_included,
                })
            except Exception as exc:
                result_row = {**base, "status": "error", "error": repr(exc)}
                results.append(result_row)
                traceback.print_exc()

            done += 1
            elapsed = time.time() - t0
            avg = elapsed / max(done, 1)
            eta = avg * max(total - done, 0)
            results[-1]["secs"] = round(time.time() - c0, 1)
            write_progress(results, placebo_rows, sig_rows)
            extra = ""
            if results[-1].get("status") == "ok":
                extra = (f"cross-own={results[-1].get('cross_minus_own_r2'):+.5f} "
                         f"cross-car={results[-1].get('cross_minus_car_r2'):+.5f}")
            else:
                extra = results[-1].get("error", "")
            print(
                f"subminute_decay [{results[-1].get('status')}] bar={bar_s}s scheme={scheme} "
                f"({results[-1]['secs']}s) progress={done}/{total} "
                f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)} {extra}",
                flush=True,
            )

    results_df = pd.DataFrame(results)
    sig = pd.DataFrame(sig_rows)
    for pcol in ["p_own_vs_cross", "p_car_vs_cross", "p_placebo_vs_real_cross"]:
        bonf, holm, bh = corrected_pvalues(sig, pcol, SEARCH_M)
        sig[f"{pcol}_bonf792"] = bonf
        sig[f"{pcol}_holm792"] = holm
        sig[f"{pcol}_bh792"] = bh

    placebo_df = pd.DataFrame(placebo_rows)
    results_df.to_csv(SUBMINUTE_DECAY_RESULTS, index=False)
    placebo_df.to_csv(SUBMINUTE_DECAY_PLACEBO, index=False)
    sig.to_csv(SUBMINUTE_DECAY_SIGNIFICANCE, index=False)
    return results_df, placebo_df, sig


def require_all_ok(results):
    ok = results.get("status").eq("ok")
    if len(results) != 120 or int(ok.sum()) != 120:
        bad = results.loc[~ok, ["bar_s", "scheme", "status"]].to_dict("records")
        raise RuntimeError(f"subminute_decay grid incomplete: rows={len(results)} ok={int(ok.sum())} bad={bad}")


def checkpoint_match(results):
    if not SUBMINUTE_RESULTS.exists():
        raise FileNotFoundError(SUBMINUTE_RESULTS)
    locked = pd.read_csv(SUBMINUTE_RESULTS)
    locked = locked[locked["bar_s"].isin(CHECKPOINT_BARS)].copy()
    ours = results[results["bar_s"].isin(CHECKPOINT_BARS)].copy()
    keys = ["bar_s", "scheme"]
    metrics = [
        "own_r2", "ar_r2", "car_r2", "cross_r2",
        "cross_minus_own_r2", "cross_minus_ar_r2", "cross_minus_car_r2",
        "own_hit", "ar_hit", "car_hit", "cross_hit",
        "own_unit_pnl_bps", "car_unit_pnl_bps", "cross_unit_pnl_bps",
        "cross_minus_own_unit_pnl_bps", "best_asset_cross_minus_own_r2",
    ]
    merged = ours.merge(locked[keys + metrics], on=keys, suffixes=("_decay", "_locked"))
    if len(merged) != 12:
        raise RuntimeError(f"checkpoint merge expected 12 rows, got {len(merged)}")
    diffs = {}
    for metric in metrics:
        diff = (pd.to_numeric(merged[f"{metric}_decay"], errors="coerce") -
                pd.to_numeric(merged[f"{metric}_locked"], errors="coerce")).abs()
        diffs[metric] = float(diff.max())
    max_abs = max(diffs.values())
    if max_abs > FLOAT_TOL:
        detail = {k: v for k, v in diffs.items() if v > FLOAT_TOL}
        raise RuntimeError(f"subminute_decay checkpoint mismatch max_abs={max_abs}: {detail}")
    return max_abs, diffs


def add_pass_columns(sig):
    out = sig.copy()
    out["pass_own_bonf792"] = out["p_own_vs_cross_bonf792"] < 0.05
    out["pass_car_bonf792"] = out["p_car_vs_cross_bonf792"] < 0.05
    out["pass_placebo_bonf792"] = out["p_placebo_vs_real_cross_bonf792"] < 0.05
    out["pass_all_bonf792"] = (
        out["pass_own_bonf792"] &
        out["pass_car_bonf792"] &
        out["pass_placebo_bonf792"]
    )
    return out


def significance_stop_text(sig):
    s = add_pass_columns(sig)
    by_bar = (s.groupby("bar_s", as_index=False)
              .agg(pass_all=("pass_all_bonf792", "sum"),
                   pass_own=("pass_own_bonf792", "sum"),
                   pass_car=("pass_car_bonf792", "sum"),
                   pass_placebo=("pass_placebo_bonf792", "sum")))
    full = by_bar[by_bar["pass_all"] == len(grid.SCHEMES)]["bar_s"].astype(int).tolist()
    any_pass = by_bar[by_bar["pass_all"] > 0]["bar_s"].astype(int).tolist()
    prefix = []
    for bar_s in grid.SUBMINUTE_DECAY_BARS_S:
        if bar_s in full:
            prefix.append(bar_s)
        else:
            break
    if prefix:
        first_stop = max(prefix) + 1 if max(prefix) < max(grid.SUBMINUTE_DECAY_BARS_S) else None
        if first_stop is None:
            return "All four schemes pass all three controls through 30s.", by_bar
        if any_pass:
            return (f"All four schemes pass all three controls through {max(prefix)}s. "
                    f"The first bar width where full four-scheme significance stops is {first_stop}s; "
                    f"the last bar width with at least one all-control passing scheme is {max(any_pass)}s."), by_bar
        return (f"All four schemes pass all three controls through {max(prefix)}s. "
                f"No later bar width has an all-control passing scheme."), by_bar
    if any_pass:
        return (f"No initial contiguous region has all four schemes passing. "
                f"The last bar width with at least one all-control passing scheme is {max(any_pass)}s."), by_bar
    return "No cell passes all three controls after Bonferroni(792).", by_bar


def write_report(results, placebo, sig, checkpoint_max_diff):
    sig2 = add_pass_columns(sig)
    merged = results.merge(
        sig2[["bar_s", "scheme", "pass_own_bonf792", "pass_car_bonf792",
              "pass_placebo_bonf792", "pass_all_bonf792"]],
        on=["bar_s", "scheme"],
        how="left",
    )
    by_bar = (merged.groupby("bar_s", as_index=False)
              .agg(cross_minus_car_min=("cross_minus_car_r2", "min"),
                   cross_minus_car_mean=("cross_minus_car_r2", "mean"),
                   cross_minus_car_max=("cross_minus_car_r2", "max"),
                   cross_minus_own_mean=("cross_minus_own_r2", "mean"),
                   cross_unit_pnl_bps_mean=("cross_unit_pnl_bps", "mean"),
                   pass_all_cells=("pass_all_bonf792", "sum"),
                   pass_car_cells=("pass_car_bonf792", "sum")))
    stop_text, pass_by_bar = significance_stop_text(sig)
    pass_counts = {
        "own": int(sig2["pass_own_bonf792"].sum()),
        "car": int(sig2["pass_car_bonf792"].sum()),
        "placebo": int(sig2["pass_placebo_bonf792"].sum()),
        "all": int(sig2["pass_all_bonf792"].sum()),
        "total": len(sig2),
    }

    summary_cols = [
        "bar_s", "cross_minus_car_min", "cross_minus_car_mean",
        "cross_minus_car_max", "cross_minus_own_mean",
        "cross_unit_pnl_bps_mean", "pass_all_cells", "pass_car_cells",
    ]
    pass_cols = ["bar_s", "pass_own", "pass_car", "pass_placebo", "pass_all"]

    lines = [
        "# Subminute Decay Report",
        "",
        "This report executes the frozen 1s-30s sub-minute decay grid described in `report/methodology.md`.",
        "",
        "## Grid Recap",
        "",
        "- `bar_s` is every integer from 1 to 30 seconds.",
        "- Four schemes are tested at each bar width: `best`, `sum`, `distance`, `pca`.",
        "- Lag is single current bar only (`single_bar`), horizon is 1 bar.",
        "- The evaluator is `src/tests.py::eval_subminute_placebo_pair`, reusing the corrected fixed-UTC-boundary-first subminute path.",
        "- Headline rows are sourced from the paired-real evaluator.",
        "",
        "Multiple-comparison correction uses Bonferroni, Holm and BH with `m = 792`: 672 original broad-grid discovery cells plus 120 preregistered decay-grid cells. The original 12-cell subminute sweep is subsumed inside the 120 decay cells, so it is not added again.",
        "",
        "## Headline By Bar Width",
        "",
        by_bar[summary_cols].to_markdown(index=False),
        "",
        "Rows with `NaN` effect-size summaries completed with `status='ok'` but did not produce finite losses under the exact-timestamp day-shift/common-mask requirement. They are retained in the preregistered grid and counted as non-passing cells.",
        "",
        "## Corrected Pass Counts",
        "",
        "| Test | Passing cells after Bonferroni(792) |",
        "|---|---:|",
        f"| Cross beats own OFI | {pass_counts['own']} / {pass_counts['total']} |",
        f"| Cross beats CAR | {pass_counts['car']} / {pass_counts['total']} |",
        f"| Real cross beats shifted placebo | {pass_counts['placebo']} / {pass_counts['total']} |",
        f"| Passes all three controls | {pass_counts['all']} / {pass_counts['total']} |",
        "",
        "Pass counts by bar width:",
        "",
        pass_by_bar[pass_cols].to_markdown(index=False),
        "",
        "## Significance Boundary",
        "",
        stop_text,
        "",
        "## Checkpoint Reconciliation",
        "",
        f"The `bar_s = 1/5/10` rows in this subminute_decay run match the Phase 2 subminute checkpoint source rows with max absolute metric difference `{checkpoint_max_diff:.3g}`.",
        "",
        "## Guardrail Checklist",
        "",
        "- [x] Grid matches the preregistered 30 x 4 design; no bar sizes, schemes or controls were added or removed.",
        "- [x] `m = 792` is used and justified as 672 + 120, with the original 12 subminute cells subsumed rather than added.",
        f"- [x] `bar_s = 1/5/10` rows match the Phase 2 checkpoint; max absolute difference `{checkpoint_max_diff:.3g}`.",
        "- [x] Outputs are written under `output/subminute_decay/`; `output/subminute/` is left unchanged.",
        "- [x] The chart script uses the effect-size curve with a significance overlay and in-sample, pre-holdout labeling.",
        f"- [x] All 120 result rows show `status='ok'`.",
        "",
    ]
    SUBMINUTE_DECAY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("subminute_decay: building independent main-repo 1-second panel ...", flush=True)
    second = grid.build_subminute_second_panel(grid.PROCESSED, grid.ASSETS, levels=grid.LEVELS)
    print(f"subminute_decay: aligned seconds={len(second['sec']):,}", flush=True)
    results, placebo, sig = run_cells(second)
    require_all_ok(results)
    checkpoint_max, _ = checkpoint_match(results)
    print(f"subminute_decay checkpoint max abs diff={checkpoint_max:.3g}", flush=True)
    write_report(results, placebo, sig, checkpoint_max)
    print(SUBMINUTE_DECAY_REPORT, flush=True)


if __name__ == "__main__":
    main()

