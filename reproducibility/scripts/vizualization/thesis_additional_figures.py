"""Build additional publication-ready figures for the bachelor thesis.

The regression figures read frozen audit outputs. Descriptive figures use the
same analytical panel files as the audited models and do not refit any model.
"""

from __future__ import annotations

from pathlib import Path
import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "outputs/test_speeches/empirical_regression_audit"
AUDIT = AUDIT_DIR / "all_contrasts.csv"
FUNCTIONAL = AUDIT_DIR / "functional_form_effects.csv"
INFLUENCE = AUDIT_DIR / "influence_estimates.csv"
PLDA_PANEL = (
    ROOT
    / "outputs/test_speeches/plda_salience/substantive/data/topic_salience_panel.csv"
)
MACRO_PANEL_DIR = ROOT / "outputs/test_speeches/nli_consensus_regression_panel"
OUTPUT = ROOT / "outputs/thesis_results/figures"

BLUE = "#2C6E9B"
ORANGE = "#C96B32"
GREEN = "#39856B"
PURPLE = "#76558F"
INK = "#263238"
MUTED = "#66747B"
GRID = "#D9DEE2"
LIGHT = "#C6CDD2"


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=360, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT)


def build_plda_correspondence() -> None:
    data = pd.read_csv(PLDA_PANEL, low_memory=False)
    data = data.loc[data["prior_manifesto"].fillna(False).astype(bool)].copy()
    keep = [
        "manifesto_salience",
        "speech_salience",
        "electoral_proximity",
        "party_in_government",
    ]
    data = data.dropna(subset=keep)
    data["party_in_government"] = data["party_in_government"].astype(bool)
    data["cycle_phase"] = pd.cut(
        data["electoral_proximity"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["Post-election", "Mid-cycle", "Pre-election"],
    )
    data["manifesto_bin"] = pd.qcut(
        data["manifesto_salience"], q=12, labels=False, duplicates="drop"
    )
    binned = (
        data.groupby(
            ["cycle_phase", "party_in_government", "manifesto_bin"],
            observed=True,
        )[["manifesto_salience", "speech_salience"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), sharex=True, sharey=True)
    phases = ["Post-election", "Mid-cycle", "Pre-election"]
    for ax, phase in zip(axes, phases):
        part = binned.loc[binned["cycle_phase"].eq(phase)]
        for government, label, color, marker in [
            (False, "Opposition", BLUE, "o"),
            (True, "Government", ORANGE, "s"),
        ]:
            rows = part.loc[part["party_in_government"].eq(government)]
            ax.plot(
                rows["manifesto_salience"],
                rows["speech_salience"],
                color=color,
                marker=marker,
                markersize=3.6,
                linewidth=1.35,
                label=label,
                zorder=2,
            )
        ax.plot([0, 0.5], [0, 0.5], color=LIGHT, linewidth=0.9, zorder=1)
        ax.set_title(phase, fontsize=9, weight="bold")
        ax.set_xlabel("Manifesto topic share", fontsize=8)
        clean_axis(ax)
    axes[0].set_ylabel("Mean speech topic share", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.22, wspace=0.13)
    save(fig, "plda_manifesto_speech_correspondence")


def build_topic_composition() -> None:
    data = pd.read_csv(PLDA_PANEL, low_memory=False)
    data = data.loc[data["prior_manifesto"].fillna(False).astype(bool)].copy()
    means = (
        data.groupby("topic")[["manifesto_salience", "speech_salience"]]
        .mean()
        .sort_values("manifesto_salience")
    )
    labels = {
        "Economics": "Economic policy",
        "Environment_Land_Energy": "Environment & energy",
        "Foreign_Security": "Foreign & security",
        "Gal_Tan": "GAL–TAN",
        "Infrastructure_Technology": "Infrastructure & technology",
        "Institutions_Rights_Law": "Institutions, rights & law",
        "Macroeconomics": "Macroeconomics",
        "Welfare_Human_Development": "Welfare & human development",
    }
    y = np.arange(len(means))
    manifesto = means["manifesto_salience"].to_numpy() * 100
    speech = means["speech_salience"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    for ypos, left, right in zip(y, manifesto, speech):
        ax.plot([left, right], [ypos, ypos], color=LIGHT, linewidth=2.0, zorder=1)
    ax.scatter(manifesto, y, color=PURPLE, marker="D", s=28, zorder=2)
    ax.scatter(speech, y, color=GREEN, marker="o", s=31, zorder=3)
    ax.set_yticks(y, [labels.get(topic, topic) for topic in means.index])
    ax.set_xlabel("Mean share of the eight-topic composition (%)", fontsize=8.5)
    ax.legend(
        handles=[
            Line2D([], [], marker="D", linestyle="none", color=PURPLE, label="Manifesto"),
            Line2D([], [], marker="o", linestyle="none", color=GREEN, label="Speech"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid_axis="x")
    fig.subplots_adjust(left=0.31, right=0.98, top=0.96, bottom=0.18)
    save(fig, "plda_topic_composition")


def load_macro_panel() -> pd.DataFrame:
    paths = sorted(
        glob.glob(
            str(
                MACRO_PANEL_DIR
                / "*"
                / "*_Macroeconomics_nli_consensus_deberta_regression_panel_model.csv"
            )
        )
    )
    if not paths:
        raise FileNotFoundError("No consensus Macroeconomics panels found.")
    return pd.concat(
        [pd.read_csv(path, low_memory=False) for path in paths],
        ignore_index=True,
        sort=False,
    )


def build_classifier_decomposition() -> None:
    data = load_macro_panel()
    data = data.dropna(
        subset=[
            "electoral_cycle_progress",
            "party_in_government",
            "classifier_share_consistent",
            "classifier_share_unrelated",
            "classifier_share_inconsistent",
        ]
    ).copy()
    data["party_in_government"] = data["party_in_government"].astype(bool)
    data["cycle_phase"] = pd.cut(
        data["electoral_cycle_progress"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["Post-election", "Mid-cycle", "Pre-election"],
    )
    cols = [
        "classifier_share_unrelated",
        "classifier_share_consistent",
        "classifier_share_inconsistent",
    ]
    grouped = (
        data.groupby(["cycle_phase", "party_in_government"], observed=True)[cols]
        .mean()
        .reset_index()
    )

    phases = ["Post-election", "Mid-cycle", "Pre-election"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1), sharey=True)
    for ax, government, title in zip(
        axes, [False, True], ["Opposition", "Government"]
    ):
        part = (
            grouped.loc[grouped["party_in_government"].eq(government)]
            .set_index("cycle_phase")
            .reindex(phases)
        )
        bottom = np.zeros(len(phases))
        for col, label, color in [
            ("classifier_share_unrelated", "Unrelated", LIGHT),
            ("classifier_share_consistent", "Consistent", BLUE),
            ("classifier_share_inconsistent", "Inconsistent", ORANGE),
        ]:
            values = part[col].to_numpy() * 100
            ax.bar(
                phases,
                values,
                bottom=bottom,
                color=color,
                width=0.68,
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )
            if col == "classifier_share_inconsistent":
                for xpos, base, value in zip(range(len(phases)), bottom, values):
                    ax.text(
                        xpos,
                        base + value / 2,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=INK,
                    )
            bottom += values
        ax.set_title(title, fontsize=9.2, weight="bold")
        ax.set_xlabel("Position in electoral cycle", fontsize=8)
        clean_axis(ax)
    axes[0].set_ylabel("Mean classifier share within party-month (%)", fontsize=8)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(left=0.1, right=0.99, top=0.83, bottom=0.21, wspace=0.18)
    save(fig, "classifier_label_decomposition")


def estimate_range(
    audit: pd.DataFrame, model_ids: list[str], contrast: str, scale: float = 1.0
) -> tuple[float, float]:
    values = audit.loc[
        audit["model_id"].isin(model_ids) & audit["contrast"].eq(contrast),
        "estimate",
    ].to_numpy()
    if not len(values):
        return np.nan, np.nan
    return float(values.min() * scale), float(values.max() * scale)


def point(
    audit: pd.DataFrame, model_id: str, contrast: str, scale: float = 1.0
) -> float:
    rows = audit.loc[
        audit["model_id"].eq(model_id) & audit["contrast"].eq(contrast),
        "estimate",
    ]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0] * scale)


def stability_panel(
    ax: plt.Axes,
    audit: pd.DataFrame,
    rows: list[tuple[str, str]],
    primary: str,
    transition: str,
    weighted: list[str],
    *,
    scale: float,
    xlabel: str,
) -> None:
    y = np.arange(len(rows))[::-1]
    for ypos, (_, contrast) in zip(y, rows):
        low, high = estimate_range(audit, weighted, contrast, scale)
        ax.plot([low, high], [ypos, ypos], color=LIGHT, linewidth=5.0, zorder=1)
        ax.scatter(
            point(audit, primary, contrast, scale),
            ypos,
            color=BLUE,
            marker="o",
            s=34,
            zorder=3,
        )
        ax.scatter(
            point(audit, transition, contrast, scale),
            ypos,
            color=ORANGE,
            marker="D",
            s=29,
            zorder=3,
        )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid_axis="x")


def build_salience_stability() -> None:
    audit = load_audit()
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 3.25))
    stability_panel(
        axes[0],
        audit,
        [
            ("Opposition", "opposition_slope"),
            ("Government", "government_slope"),
            ("Mid-cycle gap", "government_gap_p05"),
        ],
        "plda_topic_m1_equal",
        "plda_topic_m3_equal",
        [
            "plda_topic_m5_word_weight",
            "plda_topic_m5_segment_weight",
            "plda_topic_m5_minimum_volume",
        ],
        scale=1.0,
        xlabel="Responsiveness change",
    )
    axes[0].set_title("A. PLDA responsiveness", fontsize=9, weight="bold")

    stability_panel(
        axes[1],
        audit,
        [
            ("Opposition", "opposition_slope"),
            ("Government", "government_slope"),
        ],
        "plda_alignment_m1_natural",
        "plda_alignment_m3_m1_natural",
        [
            "plda_alignment_m5_volume_weight",
            "plda_alignment_m5_sqrt_volume_weight",
            "plda_alignment_m5_capped_volume_weight",
            "plda_alignment_m5_minimum_volume",
        ],
        scale=1.0,
        xlabel="Alignment-score change",
    )
    axes[1].set_title("B. PLDA alignment score", fontsize=9, weight="bold")

    stability_panel(
        axes[2],
        audit,
        [
            ("Opposition", "opposition_slope"),
            ("Government", "government_slope"),
        ],
        "spearman_m1_natural",
        "spearman_m3_m1_natural",
        [
            "spearman_m5_volume_weight",
            "spearman_m5_sqrt_volume_weight",
            "spearman_m5_capped_volume_weight",
            "spearman_m5_minimum_volume",
        ],
        scale=1.0,
        xlabel="Spearman change",
    )
    axes[2].set_title("C. Topic-rank alignment", fontsize=9, weight="bold")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", color=BLUE, label="M1"),
            Line2D([], [], marker="D", linestyle="none", color=ORANGE, label="M3"),
            Line2D([], [], linewidth=5, color=LIGHT, label="M5 estimate range"),
        ],
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.79, bottom=0.21, wspace=0.62)
    save(fig, "salience_specification_stability")


def build_macro_stability() -> None:
    audit = load_audit()
    rows = [
        ("Opposition slope", "opposition_slope"),
        ("Government slope", "government_slope"),
        ("Mid-cycle status gap", "government_gap_p05"),
    ]
    y = np.arange(len(rows))[::-1]
    weighted = [
        "new_macro_m5_volume_weight",
        "new_macro_m5_sqrt_volume_weight",
        "new_macro_m5_capped_volume_weight",
        "new_macro_m5_minimum_volume",
        "new_macro_m5_pair_weight",
    ]
    models = [
        ("new_macro_m1_natural", "M1", BLUE, "o", -0.12),
        ("new_macro_m2_natural", "M2", GREEN, "s", -0.04),
        ("new_macro_m3_m1_natural", "M3–M1", ORANGE, "D", 0.04),
        ("new_macro_m3_m2_natural", "M3–M2", PURPLE, "^", 0.12),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    for ypos, (_, contrast) in zip(y, rows):
        low, high = estimate_range(audit, weighted, contrast, 100)
        ax.plot([low, high], [ypos, ypos], color=LIGHT, linewidth=6.5, zorder=1)
        for model_id, _, color, marker, offset in models:
            ax.scatter(
                point(audit, model_id, contrast, 100),
                ypos + offset,
                color=color,
                marker=marker,
                s=35,
                zorder=3,
            )
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, [label for label, _ in rows])
    ax.set_xlabel("Change or gap (percentage points)", fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid_axis="x")
    handles = [
        Line2D([], [], marker=marker, linestyle="none", color=color, label=label)
        for _, label, color, marker, _ in models
    ]
    handles.append(Line2D([], [], linewidth=6, color=LIGHT, label="M5 estimate range"))
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.18)
    save(fig, "macro_specification_stability")


def spline_panel(
    ax_profile: plt.Axes,
    ax_gap: plt.Axes,
    effects: pd.DataFrame,
    family: str,
    *,
    scale: float,
    ylabel: str,
) -> None:
    rows = effects.loc[
        effects["family"].eq(family)
        & effects["functional_form"].eq("restricted cubic spline")
    ]
    for trajectory, label, color in [
        ("opposition_relative_midcycle", "Opposition", BLUE),
        ("government_relative_midcycle", "Government", ORANGE),
    ]:
        part = rows.loc[rows["trajectory"].eq(trajectory)].sort_values("proximity")
        x = part["proximity"].to_numpy()
        estimate = part["estimate"].to_numpy() * scale
        low = part["ci_95_low"].to_numpy() * scale
        high = part["ci_95_high"].to_numpy() * scale
        ax_profile.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        ax_profile.plot(x, estimate, color=color, linewidth=1.6, label=label)
    gap = rows.loc[
        rows["trajectory"].eq("government_minus_opposition")
    ].sort_values("proximity")
    x = gap["proximity"].to_numpy()
    estimate = gap["estimate"].to_numpy() * scale
    low = gap["ci_95_low"].to_numpy() * scale
    high = gap["ci_95_high"].to_numpy() * scale
    ax_gap.fill_between(x, low, high, color=PURPLE, alpha=0.14, linewidth=0)
    ax_gap.plot(x, estimate, color=PURPLE, linewidth=1.6)
    for ax in [ax_profile, ax_gap]:
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["Post-election", "Mid-cycle", "Pre-election"])
        clean_axis(ax)
    ax_profile.set_ylabel(ylabel, fontsize=8)


def build_nonlinear_profiles() -> None:
    effects = pd.read_csv(FUNCTIONAL)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0))
    spline_panel(
        axes[0, 0],
        axes[0, 1],
        effects,
        "plda_alignment",
        scale=1.0,
        ylabel="Change relative to mid-cycle",
    )
    spline_panel(
        axes[1, 0],
        axes[1, 1],
        effects,
        "new_macro",
        scale=100,
        ylabel="Change relative to mid-cycle (pp)",
    )
    axes[0, 0].set_title("A. PLDA: fitted profiles", fontsize=9, weight="bold")
    axes[0, 1].set_title("B. PLDA: status gap", fontsize=9, weight="bold")
    axes[1, 0].set_title("C. Macroeconomics: fitted profiles", fontsize=9, weight="bold")
    axes[1, 1].set_title("D. Macroeconomics: status gap", fontsize=9, weight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    axes[1, 0].set_xlabel("Electoral proximity", fontsize=8)
    axes[1, 1].set_xlabel("Electoral proximity", fontsize=8)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.93, bottom=0.12, wspace=0.28, hspace=0.48)
    save(fig, "nonlinear_cycle_profiles")


def build_macro_country_influence() -> None:
    influence = pd.read_csv(INFLUENCE)
    audit = load_audit()
    specs = [
        ("opposition_slope", "A. Opposition cycle slope"),
        ("government_gap_p05", "B. Mid-cycle government gap"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.0), sharey=True)
    for ax, (contrast, title) in zip(axes, specs):
        rows = influence.loc[
            influence["family"].eq("new_macro")
            & influence["influence_type"].eq("leave_one_country_out")
            & influence["contrast"].eq(contrast)
        ].copy()
        rows = rows.sort_values("estimate")
        y = np.arange(len(rows))
        estimate = rows["estimate"].to_numpy() * 100
        error = rows["country_se"].to_numpy() * 1.96 * 100
        ax.errorbar(
            estimate,
            y,
            xerr=error,
            fmt="o",
            color=BLUE if contrast == "opposition_slope" else PURPLE,
            ecolor=LIGHT,
            elinewidth=1.1,
            capsize=2.0,
            markersize=4.2,
        )
        full = point(audit, "new_macro_m1_natural", contrast, 100)
        ax.axvline(full, color=ORANGE, linewidth=1.2, label="Full-sample estimate")
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(y, rows["omitted_unit"])
        ax.set_title(title, fontsize=9, weight="bold")
        ax.set_xlabel("Estimate after omitting country (pp)", fontsize=8)
        clean_axis(ax, grid_axis="x")
    axes[0].set_ylabel("Country omitted", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7.7, loc="lower right")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.92, bottom=0.12, wspace=0.2)
    save(fig, "macro_leave_one_country_out")


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
    build_plda_correspondence()
    build_topic_composition()
    build_classifier_decomposition()
    build_salience_stability()
    build_macro_stability()
    build_nonlinear_profiles()
    build_macro_country_influence()


if __name__ == "__main__":
    main()
