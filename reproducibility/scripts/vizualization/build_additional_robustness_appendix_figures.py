from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS = ROOT / "outputs/test_speeches/additional_robustness_checks"
OUTPUT = ROOT / "substantive graphs/robustness_appendix"

BLUE = "#2C6E9B"
ORANGE = "#C96B32"
GREEN = "#39856B"
PURPLE = "#76558F"
INK = "#263238"
GRID = "#D9DEE2"
LIGHT = "#C6CDD2"


def clean_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def build_complete_cycle_figure() -> None:
    salience = pd.read_csv(ROBUSTNESS / "salience_complete_cycle.csv")
    conditional = pd.read_csv(ROBUSTNESS / "conditional_h_complete_cycle.csv")

    salience_rows = [
        ("Salience: cycle change", "h1a_average_cycle_change", BLUE),
        ("Salience: gov-opposition gap", "h2a_average_government_gap", ORANGE),
    ]
    conditional_rows = [
        ("Macro H: cycle change", "new_macro", "Cycle change", BLUE),
        ("Macro H: gov-opposition gap", "new_macro", "Government-opposition gap", ORANGE),
        ("GAL-TAN H: cycle change", "new_galtan", "Cycle change", GREEN),
        ("GAL-TAN H: gov-opposition gap", "new_galtan", "Government-opposition gap", PURPLE),
        ("Pooled H: cycle change", "new_combined", "Cycle change", BLUE),
        ("Pooled H: gov-opposition gap", "new_combined", "Government-opposition gap", ORANGE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6))

    ax = axes[0]
    y = np.arange(len(salience_rows))[::-1]
    for ypos, (label, short_name, color) in zip(y, salience_rows):
        row = salience.loc[salience["short_name"].eq(short_name)].iloc[0]
        estimate = row["estimate"]
        low = row["wild_ci_95_low"]
        high = row["wild_ci_95_high"]
        ax.errorbar(
            estimate,
            ypos,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=LIGHT,
            elinewidth=1.3,
            capsize=2.5,
            markersize=5.0,
        )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [label for label, _, _ in salience_rows])
    ax.set_title("A. Complete-cycle salience checks", fontsize=9, weight="bold")
    ax.set_xlabel("Estimate", fontsize=8)
    clean_axis(ax)

    ax = axes[1]
    y = np.arange(len(conditional_rows))[::-1]
    for ypos, (label, family, contrast, color) in zip(y, conditional_rows):
        row = conditional.loc[
            conditional["family"].eq(family) & conditional["contrast"].eq(contrast)
        ].iloc[0]
        estimate = row["estimate"] * 100
        low = row["wild_ci_95_low"] * 100
        high = row["wild_ci_95_high"] * 100
        ax.errorbar(
            estimate,
            ypos,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=LIGHT,
            elinewidth=1.3,
            capsize=2.5,
            markersize=5.0,
        )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [label for label, _, _, _ in conditional_rows])
    ax.set_title("B. Complete-cycle conditional H checks", fontsize=9, weight="bold")
    ax.set_xlabel("Estimate (pp)", fontsize=8)
    clean_axis(ax)

    fig.subplots_adjust(left=0.27, right=0.98, top=0.9, bottom=0.12, wspace=0.38)
    save(fig, "complete_cycle_robustness")


def build_leave_one_out_figure() -> None:
    data = pd.read_csv(ROBUSTNESS / "conditional_h_leave_one_country_out.csv")
    full = pd.read_csv(ROBUSTNESS / "conditional_h_complete_cycle.csv")

    specs = [
        ("new_macro", "Cycle change", "A. Macro cycle change", BLUE),
        ("new_macro", "Government-opposition gap", "B. Macro status gap", ORANGE),
        ("new_galtan", "Cycle change", "C. GAL-TAN cycle change", GREEN),
        ("new_galtan", "Government-opposition gap", "D. GAL-TAN status gap", PURPLE),
        ("new_combined", "Cycle change", "E. Pooled cycle change", BLUE),
        ("new_combined", "Government-opposition gap", "F. Pooled status gap", ORANGE),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.2, 10.2), sharey=False)
    for ax, (family, contrast, title, color) in zip(axes.flat, specs):
        rows = data.loc[data["family"].eq(family) & data["contrast"].eq(contrast)].copy()
        rows = rows.sort_values("estimate")
        y = np.arange(len(rows))
        estimate = rows["estimate"].to_numpy() * 100
        low = rows["wild_ci_95_low"].to_numpy() * 100
        high = rows["wild_ci_95_high"].to_numpy() * 100
        err = np.vstack([estimate - low, high - estimate])
        ax.errorbar(
            estimate,
            y,
            xerr=err,
            fmt="o",
            color=color,
            ecolor=LIGHT,
            elinewidth=1.1,
            capsize=2.0,
            markersize=4.0,
        )
        full_row = full.loc[full["family"].eq(family) & full["contrast"].eq(contrast)].iloc[0]
        ax.axvline(full_row["estimate"] * 100, color=INK, linewidth=1.0, linestyle="--")
        ax.axvline(0, color=LIGHT, linewidth=0.9)
        ax.set_yticks(y, rows["omitted_country"])
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_xlabel("Estimate after omitting country (pp)", fontsize=8)
        clean_axis(ax)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.06, hspace=0.38, wspace=0.28)
    save(fig, "conditional_h_leave_one_out")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    build_complete_cycle_figure()
    build_leave_one_out_figure()


if __name__ == "__main__":
    main()