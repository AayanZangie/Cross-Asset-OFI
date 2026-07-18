"""Generate publication-ready figures from the committed result artifacts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FIG = OUT / "figures"
DECAY = OUT / "subminute_decay" / "curve_data.csv"
DECAY_SIG = OUT / "subminute_decay" / "significance.csv"
SUB_SIG = OUT / "subminute" / "significance.csv"
ROBUST_SIG = OUT / "robustness_300s" / "significance.csv"
CORE_MATRIX = OUT / "core" / "matrix.csv"
CORE_PCA = OUT / "core" / "pca.csv"
CORE_COVERAGE = OUT / "core" / "coverage.csv"

ASSETS = ["BTC", "ETH", "SOL", "XRP"]
SCHEMES = ["best", "sum", "distance", "pca"]
COLORS = {
    "cross": "#0f766e",
    "own": "#2563eb",
    "car": "#9333ea",
    "ar": "#ea580c",
    "grid": "#e5e7eb",
    "neutral": "#475569",
    "pass": "#166534",
    "fail": "#cbd5e1",
}


def setup():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#334155",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def read_decay():
    df = pd.read_csv(DECAY)
    for c in [
        "pass_own_bonf792",
        "pass_car_bonf792",
        "pass_placebo_bonf792",
        "pass_all_bonf792",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def pass_columns(sig, suffix):
    out = sig.copy()
    out["pass_own"] = out[f"p_own_vs_cross_bonf{suffix}"] < 0.05
    out["pass_car"] = out[f"p_car_vs_cross_bonf{suffix}"] < 0.05
    out["pass_placebo"] = out[f"p_placebo_vs_real_cross_bonf{suffix}"] < 0.05
    out["pass_all"] = out[["pass_own", "pass_car", "pass_placebo"]].all(axis=1)
    out["controls_passed"] = out[["pass_own", "pass_car", "pass_placebo"]].sum(axis=1)
    return out


def figure_predictive_power(df):
    g = (df.groupby("bar_s", as_index=False)
         .agg(cross_r2=("cross_r2", "mean"),
              own_r2=("own_r2", "mean"),
              car_r2=("car_r2", "mean"),
              ar_r2=("ar_r2", "mean")))

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axvspan(0.5, 5.5, color="#dcfce7", alpha=0.45, label="Supported 1s-5s region")
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
    ax.plot(g["bar_s"], g["cross_r2"], color=COLORS["cross"], linewidth=2.6,
            marker="o", label="Cross-asset OFI")
    ax.plot(g["bar_s"], g["own_r2"], color=COLORS["own"], linewidth=1.9,
            marker="o", label="Own OFI baseline")
    ax.plot(g["bar_s"], g["car_r2"], color=COLORS["car"], linewidth=1.9,
            marker="o", label="Cross-return baseline")
    ax.plot(g["bar_s"], g["ar_r2"], color=COLORS["ar"], linewidth=1.4,
            marker="o", alpha=0.75, label="Own-return baseline")
    ax.set_title("Predictive power decays as bar width increases")
    ax.set_xlabel("Bar width (seconds)")
    ax.set_ylabel("Mean out-of-sample R2 across OFI schemes")
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(range(1, 31, 1))
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "01_predictive_power_vs_baselines.png")


def figure_edge_decay(df):
    g = (df.groupby("bar_s", as_index=False)
         .agg(cross_minus_own_r2=("cross_minus_own_r2", "mean"),
              cross_minus_car_r2=("cross_minus_car_r2", "mean"),
              cross_minus_ar_r2=("cross_minus_ar_r2", "mean"),
              pass_all_cells=("pass_all_bonf792", "sum")))

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.axvspan(0.5, 5.5, color="#dcfce7", alpha=0.45)
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
    ax.plot(g["bar_s"], g["cross_minus_own_r2"], color=COLORS["own"], linewidth=2.2,
            marker="o", label="Cross OFI - own OFI")
    ax.plot(g["bar_s"], g["cross_minus_car_r2"], color=COLORS["cross"], linewidth=2.6,
            marker="o", label="Cross OFI - cross-return baseline")
    ax.plot(g["bar_s"], g["cross_minus_ar_r2"], color=COLORS["ar"], linewidth=1.8,
            marker="o", alpha=0.75, label="Cross OFI - own-return baseline")
    ax.scatter(g["bar_s"], g["cross_minus_car_r2"],
               s=22 + 18 * g["pass_all_cells"].to_numpy(),
               facecolor="white", edgecolor=COLORS["cross"], linewidth=1.2,
               label="Marker size = schemes passing all controls")
    ax.set_title("Cross-asset OFI edge over baselines fades after the first few seconds")
    ax.set_xlabel("Bar width (seconds)")
    ax.set_ylabel("Mean R2 improvement")
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(range(1, 31, 1))
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "02_cross_edge_decay.png")


def figure_pass_heatmap(sig):
    pivot = (sig.pivot(index="scheme", columns="bar_s", values="controls_passed")
             .reindex(index=SCHEMES, columns=range(1, 31)))
    data = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    cmap = ListedColormap(["#f1f5f9", "#c7d2fe", "#86efac", "#166534"])
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_title("Corrected control pass map by bar width and OFI scheme")
    ax.set_xlabel("Bar width (seconds)")
    ax.set_ylabel("OFI scheme")
    ax.set_xticks(np.arange(30))
    ax.set_xticklabels(range(1, 31))
    ax.set_yticks(np.arange(len(SCHEMES)))
    ax.set_yticklabels(SCHEMES)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, int(val), ha="center", va="center",
                        color="white" if val >= 3 else "#0f172a", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Controls passed after Bonferroni(792)")
    return save(fig, "03_corrected_pass_heatmap.png")


def figure_pass_counts(sig):
    by_bar = sig.groupby("bar_s", as_index=False)["pass_all"].sum()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(by_bar["bar_s"], by_bar["pass_all"],
                  color=[COLORS["pass"] if x == 4 else "#94a3b8" for x in by_bar["pass_all"]])
    ax.axvspan(0.5, 5.5, color="#dcfce7", alpha=0.45)
    ax.set_title("Number of OFI schemes passing all corrected controls")
    ax.set_xlabel("Bar width (seconds)")
    ax.set_ylabel("Schemes passing all three controls (out of 4)")
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(range(1, 31, 1))
    ax.set_ylim(0, 4.5)
    for rect, val in zip(bars, by_bar["pass_all"]):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.08,
                str(int(val)), ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "04_all_control_pass_counts.png")


def figure_robustness_summary():
    rows = []
    specs = [
        ("300s candidate", ROBUST_SIG, 672),
        ("1s/5s/10s grid", SUB_SIG, 684),
        ("1s-30s decay grid", DECAY_SIG, 792),
    ]
    for label, path, suffix in specs:
        sig = pass_columns(pd.read_csv(path), suffix)
        total = len(sig)
        rows.extend([
            (label, "Beats own OFI", int(sig["pass_own"].sum()), total),
            (label, "Beats cross returns", int(sig["pass_car"].sum()), total),
            (label, "Beats shifted placebo", int(sig["pass_placebo"].sum()), total),
        ])
    df = pd.DataFrame(rows, columns=["battery", "test", "passes", "total"])
    df["share"] = df["passes"] / df["total"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    tests = ["Beats own OFI", "Beats cross returns", "Beats shifted placebo"]
    batteries = ["300s candidate", "1s/5s/10s grid", "1s-30s decay grid"]
    x = np.arange(len(batteries))
    width = 0.24
    palette = [COLORS["own"], COLORS["car"], COLORS["cross"]]
    for k, test in enumerate(tests):
        sub = df[df["test"] == test].set_index("battery").loc[batteries]
        ax.bar(x + (k - 1) * width, sub["share"], width=width, color=palette[k], label=test)
        for xi, (_, row) in zip(x + (k - 1) * width, sub.iterrows()):
            ax.text(xi, row["share"] + 0.025, f"{row['passes']}/{row['total']}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Robustness controls distinguish the accepted sub-minute result")
    ax.set_ylabel("Share of cells passing corrected test")
    ax.set_xticks(x)
    ax.set_xticklabels(batteries)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save(fig, "05_robustness_pass_summary.png")


def figure_cross_matrix():
    matrix = pd.read_csv(CORE_MATRIX)
    mat = (matrix.pivot(index="target", columns="source", values="coef_sum")
           .reindex(index=ASSETS, columns=ASSETS))
    scaled = mat * 1e5
    vmax = np.nanmax(np.abs(scaled.to_numpy()))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(scaled, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("60s cross-impact coefficient matrix")
    ax.set_xlabel("Source OFI")
    ax.set_ylabel("Target return")
    ax.set_xticks(np.arange(len(ASSETS)))
    ax.set_xticklabels(ASSETS)
    ax.set_yticks(np.arange(len(ASSETS)))
    ax.set_yticklabels(ASSETS)
    ax.grid(False)
    for i, target in enumerate(ASSETS):
        for j, source in enumerate(ASSETS):
            val = scaled.loc[target, source]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color="white" if abs(val) > vmax * 0.55 else "#0f172a", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Coefficient sum x 1e5")
    return save(fig, "06_cross_impact_matrix.png")


def figure_data_depth_diagnostics():
    pca = pd.read_csv(CORE_PCA)
    cov = pd.read_csv(CORE_COVERAGE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    cov = cov.set_index("asset").reindex(ASSETS)
    ax.bar(cov.index, cov["median_spread_bps"], color="#64748b", label="Median spread")
    ax.scatter(cov.index, cov["mean_spread_bps"], color=COLORS["cross"], zorder=3,
               label="Mean spread")
    ax.set_title("Observed spread by asset")
    ax.set_ylabel("Relative spread (bps)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    grouped = [pca[pca["asset"] == asset]["pc1_evr_bar_norm"].dropna().to_numpy()
               for asset in ASSETS]
    ax.boxplot(grouped, labels=ASSETS, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color="#475569"),
               medianprops=dict(color=COLORS["cross"], linewidth=2),
               whiskerprops=dict(color="#475569"),
               capprops=dict(color="#475569"))
    ax.set_title("PCA depth factor explains most normalised OFI variation")
    ax.set_ylabel("PC1 explained variance ratio")
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Data and depth diagnostics", y=1.03, fontsize=14)
    return save(fig, "07_data_depth_diagnostics.png")


def figure_workflow():
    labels = [
        "Raw OKX L2\n.data files",
        "Daily book\nreconstruction",
        "OFI features\nand returns",
        "Common-time\npanels",
        "Walk-forward\nmodels",
        "Controls and\nsignificance",
        "Frozen\ninterpretation",
    ]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_axis_off()
    x_positions = np.linspace(0.08, 0.92, len(labels))
    y = 0.5
    for idx, (x, label) in enumerate(zip(x_positions, labels)):
        box = FancyBboxPatch(
            (x - 0.047, y - 0.16), 0.094, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#475569",
            facecolor="#f8fafc",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                color="#0f172a", transform=ax.transAxes)
        if idx < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + 0.058, y), (x_positions[idx + 1] - 0.058, y),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.5,
                color=COLORS["cross"],
                transform=ax.transAxes,
            )
            ax.add_patch(arrow)
    ax.text(0.5, 0.88, "Reproducible research workflow", ha="center",
            va="center", fontsize=14, color="#0f172a", transform=ax.transAxes)
    return save(fig, "08_research_workflow.png")


def main():
    setup()
    decay = read_decay()
    decay_sig = pass_columns(pd.read_csv(DECAY_SIG), 792)
    paths = [
        figure_predictive_power(decay),
        figure_edge_decay(decay),
        figure_pass_heatmap(decay_sig),
        figure_pass_counts(decay_sig),
        figure_robustness_summary(),
        figure_cross_matrix(),
        figure_data_depth_diagnostics(),
        figure_workflow(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
