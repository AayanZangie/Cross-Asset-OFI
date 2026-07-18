"""Temporal stability check for the frozen sub-minute 1s-6s specification.

This is not a holdout validation: the 1s-6s region was selected after seeing
the full 8-week sample. The script uses one fixed 6-calendar-week training
window and one final 2-calendar-week evaluation window to check whether the
already-selected in-sample pattern is temporally stable.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tests as grid  # noqa: E402
from evaluation import unit_position_pnl_bps  # noqa: E402
from significance_tests import diebold_mariano  # noqa: E402


OUT = ROOT / "output" / "subminute_decay"
RESULTS = OUT / "temporal_stability.csv"
REPORT = OUT / "temporal_stability_report.md"

STABILITY_BARS = list(range(1, 7))
SCHEMES = ["best", "sum", "distance", "pca"]
TRAIN_SEC = 6 * 7 * 24 * 60 * 60
TEST_SEC = 2 * 7 * 24 * 60 * 60


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


def utc(sec):
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def single_window(sample_start_sec):
    train_start = int(sample_start_sec)
    train_end = train_start + TRAIN_SEC
    test_start = train_end
    test_end = test_start + TEST_SEC
    return [{
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }]


def average_loss(pred, realized):
    return grid.average_loss(pred, realized)


def mean_metric(rows, metric):
    vals = np.array([r[metric] for r in rows], dtype=float)
    return finite(np.nanmean(vals))


def eval_cell(second, bar_s, scheme, windows):
    bars = grid.subminute_bar_panel(second, bar_s)
    ofi, ret_now, log_mid, times = grid.subminute_scheme_panel(bars, scheme)
    t, n = ofi.shape
    shifted = grid.exact_time_shift_matrix(times, ofi, grid.SUBMINUTE_TEST_SEC)

    forecasts = {
        "own": np.full((t, n), np.nan),
        "ar": np.full((t, n), np.nan),
        "car": np.full((t, n), np.nan),
        "real": np.full((t, n), np.nan),
        "placebo": np.full((t, n), np.nan),
    }
    realized = np.full((t, n), np.nan)
    asset_rows = []
    meta_rows = []

    for i, name in enumerate(grid.NAMES):
        lm = log_mid[i]
        y = np.r_[lm[1:] - lm[:-1], np.nan]
        realized[:, i] = y
        own_x = ofi[:, [i]]
        ar_x = ret_now[:, [i]]
        car_x = ret_now
        real_x = ofi.copy()
        plac_x = ofi.copy()
        for j in range(n):
            if j != i:
                plac_x[:, j] = shifted[:, j]

        common = np.isfinite(np.column_stack([
            y, own_x, ar_x, car_x, real_x, plac_x
        ])).all(axis=1)

        p_own, m_own = grid.subminute_oos_pred_with_mask(own_x, y, common, times, windows)
        p_ar, m_ar = grid.subminute_oos_pred_with_mask(ar_x, y, common, times, windows)
        p_car, m_car = grid.subminute_oos_pred_with_mask(car_x, y, common, times, windows)
        p_real, m_real = grid.subminute_oos_pred_with_mask(real_x, y, common, times, windows)
        p_plac, m_plac = grid.subminute_oos_pred_with_mask(plac_x, y, common, times, windows)

        forecasts["own"][:, i] = p_own
        forecasts["ar"][:, i] = p_ar
        forecasts["car"][:, i] = p_car
        forecasts["real"][:, i] = p_real
        forecasts["placebo"][:, i] = p_plac

        own_r2 = grid.r2_zero(y, p_own)
        ar_r2 = grid.r2_zero(y, p_ar)
        car_r2 = grid.r2_zero(y, p_car)
        real_r2 = grid.r2_zero(y, p_real)
        plac_r2 = grid.r2_zero(y, p_plac)
        row = {
            "asset": name,
            "own_r2": finite(own_r2),
            "ar_r2": finite(ar_r2),
            "car_r2": finite(car_r2),
            "cross_r2": finite(real_r2),
            "placebo_cross_r2": finite(plac_r2),
            "cross_minus_own_r2": finite(real_r2 - own_r2),
            "cross_minus_car_r2": finite(real_r2 - car_r2),
            "cross_minus_placebo_r2": finite(real_r2 - plac_r2),
            "cross_hit": finite(grid.sign_acc(y, p_real)),
            "own_hit": finite(grid.sign_acc(y, p_own)),
            "car_hit": finite(grid.sign_acc(y, p_car)),
            "placebo_cross_hit": finite(grid.sign_acc(y, p_plac)),
            "cross_unit_pnl_bps": finite(unit_position_pnl_bps(p_real, y)),
            "own_unit_pnl_bps": finite(unit_position_pnl_bps(p_own, y)),
            "car_unit_pnl_bps": finite(unit_position_pnl_bps(p_car, y)),
            "placebo_cross_unit_pnl_bps": finite(unit_position_pnl_bps(p_plac, y)),
            "n": int((~np.isnan(p_real) & ~np.isnan(y)).sum()),
        }
        row["cross_minus_own_unit_pnl_bps"] = finite_diff(
            row["cross_unit_pnl_bps"], row["own_unit_pnl_bps"]
        )
        asset_rows.append(row)

        for model, meta in [
            ("own", m_own), ("ar", m_ar), ("car", m_car),
            ("cross", m_real), ("placebo", m_plac),
        ]:
            if meta:
                item = dict(meta[0])
                item["asset"] = name
                item["model"] = model
                meta_rows.append(item)

    losses = pd.DataFrame({
        "own": average_loss(forecasts["own"], realized),
        "ar": average_loss(forecasts["ar"], realized),
        "car": average_loss(forecasts["car"], realized),
        "cross": average_loss(forecasts["real"], realized),
        "placebo": average_loss(forecasts["placebo"], realized),
    })

    dm_own, p_own = diebold_mariano(losses["own"], losses["cross"], horizon=1)
    dm_car, p_car = diebold_mariano(losses["car"], losses["cross"], horizon=1)
    joined = losses[["placebo", "cross"]].dropna()
    dm_placebo, p_placebo = diebold_mariano(joined["placebo"], joined["cross"], horizon=1)

    meta = pd.DataFrame(meta_rows)
    train_n_min = int(meta["train_n"].min()) if not meta.empty and "train_n" in meta else 0
    train_n_max = int(meta["train_n"].max()) if not meta.empty and "train_n" in meta else 0
    test_n_min = int(meta["test_n"].min()) if not meta.empty and "test_n" in meta else 0
    test_n_max = int(meta["test_n"].max()) if not meta.empty and "test_n" in meta else 0
    skipped = int((meta["status"] != "ok").sum()) if not meta.empty and "status" in meta else 0

    out = {
        "spec": "subminute_temporal_stability",
        "label": "temporal stability check on the frozen in-sample spec",
        "bar_s": int(bar_s),
        "scheme": scheme,
        "lagset": "single_bar",
        "horizon": 1,
        "horizon_seconds": int(bar_s),
        "train_start_utc": utc(windows[0]["train_start"]),
        "train_end_utc": utc(windows[0]["train_end"]),
        "test_start_utc": utc(windows[0]["test_start"]),
        "test_end_utc": utc(windows[0]["test_end"]),
        "train_seconds": TRAIN_SEC,
        "test_seconds": TEST_SEC,
        "n_windows": 1,
        "n_bars": int(t),
        "train_n_min": train_n_min,
        "train_n_max": train_n_max,
        "test_n_min": test_n_min,
        "test_n_max": test_n_max,
        "skipped_model_asset_fits": skipped,
        "own_r2": mean_metric(asset_rows, "own_r2"),
        "ar_r2": mean_metric(asset_rows, "ar_r2"),
        "car_r2": mean_metric(asset_rows, "car_r2"),
        "cross_r2": mean_metric(asset_rows, "cross_r2"),
        "placebo_cross_r2": mean_metric(asset_rows, "placebo_cross_r2"),
        "cross_minus_own_r2": mean_metric(asset_rows, "cross_minus_own_r2"),
        "cross_minus_car_r2": mean_metric(asset_rows, "cross_minus_car_r2"),
        "cross_minus_placebo_r2": mean_metric(asset_rows, "cross_minus_placebo_r2"),
        "cross_hit": mean_metric(asset_rows, "cross_hit"),
        "own_hit": mean_metric(asset_rows, "own_hit"),
        "car_hit": mean_metric(asset_rows, "car_hit"),
        "placebo_cross_hit": mean_metric(asset_rows, "placebo_cross_hit"),
        "cross_unit_pnl_bps": mean_metric(asset_rows, "cross_unit_pnl_bps"),
        "own_unit_pnl_bps": mean_metric(asset_rows, "own_unit_pnl_bps"),
        "car_unit_pnl_bps": mean_metric(asset_rows, "car_unit_pnl_bps"),
        "placebo_cross_unit_pnl_bps": mean_metric(asset_rows, "placebo_cross_unit_pnl_bps"),
        "dm_own_minus_cross": finite(dm_own),
        "p_own_vs_cross_raw": finite(p_own),
        "dm_car_minus_cross": finite(dm_car),
        "p_car_vs_cross_raw": finite(p_car),
        "dm_placebo_cross_minus_real_cross": finite(dm_placebo),
        "p_placebo_vs_real_cross_raw": finite(p_placebo),
        "control_cross_beats_own_raw_5pct": bool(np.isfinite(p_own) and p_own < 0.05),
        "control_cross_beats_car_raw_5pct": bool(np.isfinite(p_car) and p_car < 0.05),
        "control_cross_beats_placebo_raw_5pct": bool(np.isfinite(p_placebo) and p_placebo < 0.05),
        "all_three_controls_raw_5pct": bool(
            np.isfinite(p_own) and p_own < 0.05 and
            np.isfinite(p_car) and p_car < 0.05 and
            np.isfinite(p_placebo) and p_placebo < 0.05
        ),
        "asset_metrics_json": json.dumps(asset_rows, separators=(",", ":")),
    }
    return out


def write_report(df):
    display = df[[
        "bar_s", "scheme", "cross_minus_own_r2", "cross_minus_car_r2",
        "cross_hit", "cross_unit_pnl_bps", "p_own_vs_cross_raw",
        "p_car_vs_cross_raw", "p_placebo_vs_real_cross_raw",
        "all_three_controls_raw_5pct",
    ]].copy()
    by_bar = (df.groupby("bar_s", as_index=False)
              .agg(cross_minus_car_mean=("cross_minus_car_r2", "mean"),
                   cross_minus_car_min=("cross_minus_car_r2", "min"),
                   cross_minus_car_max=("cross_minus_car_r2", "max"),
                   cross_minus_own_mean=("cross_minus_own_r2", "mean"),
                   cross_unit_pnl_bps_mean=("cross_unit_pnl_bps", "mean"),
                   raw_all_controls_count=("all_three_controls_raw_5pct", "sum"),
                   raw_car_control_count=("control_cross_beats_car_raw_5pct", "sum")))
    pass_counts = {
        "own": int(df["control_cross_beats_own_raw_5pct"].sum()),
        "car": int(df["control_cross_beats_car_raw_5pct"].sum()),
        "placebo": int(df["control_cross_beats_placebo_raw_5pct"].sum()),
        "all": int(df["all_three_controls_raw_5pct"].sum()),
    }
    first = df.iloc[0]
    lines = [
        "# Sub-Minute Temporal Stability Check on the Frozen In-Sample Spec",
        "",
        "This is a temporal stability check on the frozen in-sample spec, not a holdout validation, out-of-sample test or fresh-data test.",
        "",
        "Selection context: the 1s-6s bar-width region was selected after searching the full 8-week sample, and that full sample includes the final two-week period evaluated here. This split checks whether the already-selected in-sample finding is temporally stable inside the same sample; it does not establish generalisation to unseen data.",
        "",
        "## Design",
        "",
        "- Frozen cells: `bar_s in {1,2,3,4,5,6}` x `{best,sum,distance,pca}` = 24 cells.",
        "- Feature: single current-bar OFI (`single_bar`).",
        "- Horizon: 1 bar.",
        "- One fixed calendar partition: first 6 calendar weeks for training, final 2 calendar weeks for evaluation.",
        f"- Train window: {first['train_start_utc']} to {first['train_end_utc']}.",
        f"- Evaluation window: {first['test_start_utc']} to {first['test_end_utc']}.",
        "- Estimator: same tuned ridge machinery and lambda grid `{0,10,100}`, with the same 75/25 internal split inside the 6-week training window.",
        "- Boundary handling: same fixed-UTC-first helper path as the sub-minute decay grid, with one boundary schedule instead of the 7-window rolling schedule.",
        "",
        "No new multiple-comparison correction is applied. These 24 cells were already selected by the Bonferroni(792)-corrected decay-grid analysis; this is a confirmatory temporal-stability re-check of that selected spec, not a new search over bar widths, schemes or controls. Raw DM p-values are reported only as reference diagnostics.",
        "",
        "This single two-week evaluation period has much less data than the original 7-window rolling evaluation. A control failing here should be read as lower-power evidence on this split, not as proof that the effect disappeared.",
        "",
        "## Summary By Bar Width",
        "",
        by_bar.to_markdown(index=False),
        "",
        "## Raw Control Counts",
        "",
        "| Raw reference control | Passing cells at p < 0.05 |",
        "|---|---:|",
        f"| Cross beats own OFI | {pass_counts['own']} / {len(df)} |",
        f"| Cross beats CAR | {pass_counts['car']} / {len(df)} |",
        f"| Real cross beats shifted placebo | {pass_counts['placebo']} / {len(df)} |",
        f"| Passes all three controls | {pass_counts['all']} / {len(df)} |",
        "",
        "## Cell-Level Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if pass_counts["all"] == len(df):
        lines.append("All 24 frozen cells pass the three raw reference controls on this temporal split.")
    elif pass_counts["all"] > 0:
        lines.append(f"{pass_counts['all']} of 24 frozen cells pass all three raw reference controls on this temporal split. The result should be read alongside the lower statistical power of a single two-week evaluation window.")
    else:
        lines.append("No frozen cell passes all three raw reference controls on this single temporal split. Given the much smaller evaluation window, this is not equivalent to refuting the rolling in-sample result.")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("temporal stability: building independent main-repo 1-second panel ...", flush=True)
    second = grid.build_subminute_second_panel(grid.PROCESSED, grid.ASSETS, levels=grid.LEVELS)
    windows = single_window(second["sec"][0])
    print(f"temporal stability: aligned seconds={len(second['sec']):,}", flush=True)
    print(f"temporal stability: train {utc(windows[0]['train_start'])} -> {utc(windows[0]['train_end'])}", flush=True)
    print(f"temporal stability: eval  {utc(windows[0]['test_start'])} -> {utc(windows[0]['test_end'])}", flush=True)

    rows = []
    total = len(STABILITY_BARS) * len(SCHEMES)
    for bar_s in STABILITY_BARS:
        for scheme in SCHEMES:
            c0 = time.time()
            row = eval_cell(second, bar_s, scheme, windows)
            row["secs"] = round(time.time() - c0, 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
            print(
                f"temporal stability [{len(rows)}/{total}] bar={bar_s}s scheme={scheme} "
                f"({row['secs']}s) cross-own={row['cross_minus_own_r2']:+.5f} "
                f"cross-car={row['cross_minus_car_r2']:+.5f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS, index=False)
    write_report(df)
    print(f"temporal stability done in {time.time() - t0:.1f}s", flush=True)
    print(RESULTS, flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()


