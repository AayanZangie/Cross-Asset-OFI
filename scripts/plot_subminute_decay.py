"""Plot the sub-minute decay effect-size curve."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "subminute_decay"
RESULTS = OUT / "results.csv"
SIG = OUT / "significance.csv"
CURVE_DATA = OUT / "curve_data.csv"
CURVE_PNG = OUT / "curve.png"


def load_curve_data():
    if not RESULTS.exists() or not SIG.exists():
        raise FileNotFoundError("Run scripts/subminute_decay.py first.")
    results = pd.read_csv(RESULTS)
    sig = pd.read_csv(SIG)
    sig["pass_own_bonf792"] = sig["p_own_vs_cross_bonf792"] < 0.05
    sig["pass_car_bonf792"] = sig["p_car_vs_cross_bonf792"] < 0.05
    sig["pass_placebo_bonf792"] = sig["p_placebo_vs_real_cross_bonf792"] < 0.05
    sig["pass_all_bonf792"] = (
        sig["pass_own_bonf792"] &
        sig["pass_car_bonf792"] &
        sig["pass_placebo_bonf792"]
    )
    data = results.merge(
        sig[["bar_s", "scheme", "pass_own_bonf792", "pass_car_bonf792",
             "pass_placebo_bonf792", "pass_all_bonf792"]],
        on=["bar_s", "scheme"],
        how="inner",
    )
    summary = (data.groupby("bar_s", as_index=False)
               .agg(mean_cross_minus_car_r2=("cross_minus_car_r2", "mean"),
                    min_cross_minus_car_r2=("cross_minus_car_r2", "min"),
                    max_cross_minus_car_r2=("cross_minus_car_r2", "max"),
                    pass_all_cells=("pass_all_bonf792", "sum")))
    return data.merge(summary, on="bar_s", how="left")


def plot(data):
    summary = data[["bar_s", "mean_cross_minus_car_r2",
                    "min_cross_minus_car_r2", "max_cross_minus_car_r2"]].drop_duplicates()
    summary = summary.sort_values("bar_s")

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)
    ax.fill_between(
        summary["bar_s"],
        summary["min_cross_minus_car_r2"],
        summary["max_cross_minus_car_r2"],
        color="#8fb9aa",
        alpha=0.25,
        label="Scheme min-max band",
    )
    ax.plot(
        summary["bar_s"],
        summary["mean_cross_minus_car_r2"],
        color="#1f6f63",
        linewidth=2.4,
        label="Mean across schemes",
    )

    colors = {
        "best": "#34495e",
        "sum": "#8e44ad",
        "distance": "#d35400",
        "pca": "#2471a3",
    }
    offsets = {"best": -0.18, "sum": -0.06, "distance": 0.06, "pca": 0.18}
    for scheme, group in data.sort_values(["scheme", "bar_s"]).groupby("scheme"):
        color = colors.get(scheme, "#555555")
        x = group["bar_s"] + offsets.get(scheme, 0.0)
        passed = group["pass_all_bonf792"].astype(bool)
        ax.scatter(
            x[passed],
            group.loc[passed, "cross_minus_car_r2"],
            s=42,
            marker="o",
            facecolors=color,
            edgecolors=color,
            linewidths=0.9,
            alpha=0.95,
        )
        ax.scatter(
            x[~passed],
            group.loc[~passed, "cross_minus_car_r2"],
            s=34,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=0.9,
            alpha=0.35,
        )

    ax.axhline(0.0, color="#6f6f6f", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title("Cross-asset OFI predictive edge vs. bar width (in-sample, pre-holdout)")
    ax.set_xlabel("Bar width (seconds)")
    ax.set_ylabel("Cross - CAR R2")
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(range(1, 31, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_items = [
        Line2D([0], [0], color="#1f6f63", linewidth=2.4, label="Mean across schemes"),
        Patch(facecolor="#8fb9aa", alpha=0.25, label="Scheme min-max band"),
        Line2D([0], [0], marker="o", linestyle="None", color="#34495e",
               markerfacecolor="#34495e", label="Cell passes all controls"),
        Line2D([0], [0], marker="o", linestyle="None", color="#34495e",
               markerfacecolor="none", alpha=0.45, label="Cell does not pass all controls"),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(CURVE_PNG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_curve_data()
    data.sort_values(["bar_s", "scheme"]).to_csv(CURVE_DATA, index=False)
    plot(data)
    print(CURVE_DATA)
    print(CURVE_PNG)


if __name__ == "__main__":
    sys.exit(main())


