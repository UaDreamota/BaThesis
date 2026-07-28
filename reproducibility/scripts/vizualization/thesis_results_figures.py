"""Build publication-ready figures from the verified empirical audit.

No model is fitted here.  Every estimate, country-CR1 standard error,
confidence interval, and Webb wild-bootstrap p-value is read from the frozen
machine-readable audit.  The profile panels are algebraic displays of the
reported linear contrasts; adjacent interval panels show their uncertainty.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "outputs/test_speeches/empirical_regression_audit"
    / "all_contrasts.csv"
)
OUTPUT = ROOT / "outputs/thesis_results/figures"

BLUE = "#2C6E9B"
ORANGE = "#C96B32"
GREEN = "#39856B"
PURPLE = "#76558F"
INK = "#263238"
MUTED = "#66747B"
GRID = "#D9DEE2"
PALE_BLUE = "#EAF2F7"
PALE_ORANGE = "#F8EDE6"


def audit_rows(model_id: str) -> pd.DataFrame:
    data = pd.read_csv(AUDIT)
    rows = data.loc[data["model_id"].eq(model_id)].copy()
    if rows.empty:
        raise ValueError(f"Verified model not found in audit: {model_id}")
    return rows.set_index("contrast")


def p_label(value: float) -> str:
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".removeprefix("0")


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def interval_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    contrasts: list[str],
    labels: list[str],
    colors: list[str],
    *,
    scale: float,
    xlabel: str,
    show_p: bool = True,
) -> None:
    y = np.arange(len(contrasts))[::-1]
    display_labels = []
    for ypos, contrast, label, color in zip(y, contrasts, labels, colors):
        row = rows.loc[contrast]
        estimate = row["estimate"] * scale
        low = row["country_ci_low"] * scale
        high = row["country_ci_high"] * scale
        ax.errorbar(
            estimate,
            ypos,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            markersize=5.3,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=1.4,
            capsize=2.7,
            zorder=2,
        )
        display_labels.append(
            f"{label}\n"
            rf"$p_{{WB}}={p_label(float(row['wild_p']))}$"
            if show_p
            else label
        )
    ax.axvline(0, color=INK, linewidth=0.9, zorder=1)
    ax.set_yticks(y, display_labels)
    ax.tick_params(axis="y", length=0, labelsize=7.6)
    ax.set_xlabel(xlabel, fontsize=8.2)
    clean_axis(ax, grid_axis="x")


def build_plda_responsiveness_figure() -> None:
    rows = audit_rows("plda_alignment_m1_natural")
    mid = 0.0
    opposition_slope = float(rows.loc["opposition_slope", "estimate"])
    government_slope = float(rows.loc["government_slope", "estimate"])
    gaps = rows.loc[
        ["government_gap_p0", "government_gap_p05", "government_gap_p1"],
        "estimate",
    ].to_numpy()
    cycle = np.array([0.0, 0.5, 1.0])
    opposition = mid + opposition_slope * (cycle - 0.5)
    government = opposition + gaps

    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.12], wspace=0.58)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        cycle,
        opposition,
        marker="o",
        linewidth=2.0,
        markersize=5.4,
        color=BLUE,
        label="Opposition",
    )
    ax.plot(
        cycle,
        government,
        marker="o",
        linewidth=2.0,
        markersize=5.4,
        color=ORANGE,
        label="Government",
    )
    ax.fill_between(cycle, opposition, government, color=GRID, alpha=0.32)
    ax.set_xticks(cycle, ["Post-election\n0", "Mid-cycle\n0.5", "Pre-election\n1"])
    ax.set_ylabel("Alignment relative to opposition at mid-cycle", fontsize=8.4)
    ax.set_title("A. Estimated alignment profile", loc="left", fontsize=9.5, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    interval_panel(
        ax,
        rows,
        ["opposition_slope", "government_slope"],
        ["Opposition", "Government"],
        [BLUE, ORANGE],
        scale=1.0,
        xlabel="Full-cycle alignment change",
    )
    ax.set_title("B. Change over the cycle", loc="left", fontsize=9.5, weight="bold")

    ax = fig.add_subplot(gs[0, 2])
    interval_panel(
        ax,
        rows,
        ["government_gap_p0", "government_gap_p05", "government_gap_p1"],
        ["Post-election", "Mid-cycle", "Pre-election"],
        [PURPLE, PURPLE, PURPLE],
        scale=1.0,
        xlabel="Government minus opposition",
    )
    ax.set_title("C. Status differences", loc="left", fontsize=9.5, weight="bold")

    fig.suptitle(
        "Harmonized Jensen–Shannon alignment across the electoral cycle",
        fontsize=12,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Panel A is an algebraic profile of the verified linear contrasts. "
        "Bars in B–C are country-CR1 95% confidence intervals; pWB uses "
        "999 Webb wild-bootstrap replications by country.",
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.22)
    fig.savefig(OUTPUT / "plda_alignment_cycle.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / "plda_alignment_cycle.svg", bbox_inches="tight")
    # Retain the historical filename so existing thesis includes resolve to the
    # corrected headline figure rather than the superseded component model.
    fig.savefig(OUTPUT / "plda_responsiveness_cycle.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / "plda_responsiveness_cycle.svg", bbox_inches="tight")
    plt.close(fig)


def build_auxiliary_salience_figure() -> None:
    specifications = [
        (
            "plda_alignment_m1_natural",
            "A. Headline harmonized JS alignment",
            GREEN,
        ),
        (
            "spearman_m1_natural",
            "B. Spearman topic-rank alignment",
            PURPLE,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), gridspec_kw={"wspace": 0.58})
    for ax, (model_id, title, color) in zip(axes, specifications):
        rows = audit_rows(model_id)
        interval_panel(
            ax,
            rows,
            ["opposition_slope", "government_slope", "government_gap_p05"],
            ["Opposition slope", "Government slope", "Mid-cycle status gap"],
            [BLUE, ORANGE, color],
            scale=1.0,
            xlabel="Change or gap (measure units)",
        )
        ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
    fig.suptitle(
        "Headline salience alignment and rank-order robustness",
        fontsize=11.5,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Panels use separate numerical scales. Bars are country-CR1 95% "
        "confidence intervals; pWB uses country-level Webb wild bootstrap.",
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.78, bottom=0.25)
    fig.savefig(OUTPUT / "salience_auxiliary_contrasts.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / "salience_auxiliary_contrasts.svg", bbox_inches="tight")
    plt.close(fig)


def build_inconsistency_figure() -> None:
    rows = audit_rows("new_macro_m1_natural")
    cycle = np.array([0.0, 0.5, 1.0])
    opposition_slope = float(rows.loc["opposition_slope", "estimate"]) * 100
    government_slope = float(rows.loc["government_slope", "estimate"]) * 100
    gap_p0 = float(rows.loc["government_gap_p0", "estimate"]) * 100
    opposition = opposition_slope * cycle
    government = gap_p0 + government_slope * cycle

    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.12], wspace=0.6)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(
        cycle,
        opposition,
        marker="o",
        linewidth=2.0,
        markersize=5.4,
        color=BLUE,
        label="Opposition",
    )
    ax.plot(
        cycle,
        government,
        marker="o",
        linewidth=2.0,
        markersize=5.4,
        color=ORANGE,
        label="Government",
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(cycle, ["Post-election\n0", "Mid-cycle\n0.5", "Pre-election\n1"])
    ax.set_ylabel("Change relative to opposition at 0 (pp)", fontsize=8.2)
    ax.set_title("A. Linear profile, normalized", loc="left", fontsize=9.5, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    clean_axis(ax)

    ax = fig.add_subplot(gs[0, 1])
    interval_panel(
        ax,
        rows,
        ["opposition_slope", "government_slope"],
        ["Opposition", "Government"],
        [BLUE, ORANGE],
        scale=100.0,
        xlabel="Full-cycle change (percentage points)",
    )
    ax.set_title("B. Electoral-cycle slopes", loc="left", fontsize=9.5, weight="bold")

    ax = fig.add_subplot(gs[0, 2])
    interval_panel(
        ax,
        rows,
        ["government_gap_p0", "government_gap_p05", "government_gap_p1"],
        ["Post-election", "Mid-cycle", "Pre-election"],
        [PURPLE, PURPLE, PURPLE],
        scale=100.0,
        xlabel="Government minus opposition (pp)",
    )
    ax.set_title("C. Status differences", loc="left", fontsize=9.5, weight="bold")

    fig.suptitle(
        "Measured Macroeconomic inconsistency across the electoral cycle",
        fontsize=12,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Panel A shows point estimates normalized to opposition immediately "
        "after an election. Bars in B–C are country-CR1 95% confidence "
        "intervals; pWB uses 999 Webb wild-bootstrap replications by country.",
        ha="center",
        fontsize=7.6,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.84, bottom=0.22)
    fig.savefig(OUTPUT / "inconsistency_cycle_summary.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / "inconsistency_cycle_summary.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    build_plda_responsiveness_figure()
    build_auxiliary_salience_figure()
    build_inconsistency_figure()


if __name__ == "__main__":
    main()
