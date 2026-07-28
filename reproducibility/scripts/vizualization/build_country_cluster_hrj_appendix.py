from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
CONSENSUS_PANEL_DIR = BASE_DIR / "outputs" / "test_speeches" / "nli_consensus_regression_panel"
SALIENCE_LOO_PATH = BASE_DIR / "outputs" / "test_speeches" / "plda_salience_priority_robustness" / "leave_one_country_out.csv"
OUT_DATA_DIR = BASE_DIR / "outputs" / "test_speeches" / "appendix_numerical_evidence"
OUT_FIG_DIR = BASE_DIR / "substantive graphs" / "robustness_appendix"
OUT_TYP_PATH = BASE_DIR / "appendix_country_cluster_hrj.typ"

CLUSTERS = {
    "AT": "West Europe",
    "BE": "West Europe",
    "NL": "West Europe",
    "CZ": "Central-East Europe",
    "EE": "Central-East Europe",
    "LV": "Central-East Europe",
    "PL": "Central-East Europe",
    "DK": "Nordics",
    "FI": "Nordics",
    "NO": "Nordics",
    "SE": "Nordics",
    "ES": "South Europe",
    "GR": "South Europe",
    "IT": "South Europe",
    "PT": "South Europe",
    "GB": "Great Britain",
}
CLUSTER_ORDER = ["Central-East Europe", "Nordics", "West Europe", "South Europe", "Great Britain"]
CLUSTER_SHORT = {
    "Central-East Europe": "CEE",
    "Nordics": "Nordics",
    "West Europe": "West",
    "South Europe": "South",
    "Great Britain": "GB",
}
COLORS = {
    "H": "#c44e52",
    "R": "#4c72b0",
    "J": "#55a868",
    "consistent": "#4c72b0",
    "unrelated": "#9aa1a6",
    "inconsistent": "#c44e52",
    "Central-East Europe": "#2f6db3",
    "Nordics": "#1f9d8a",
    "West Europe": "#c17c2f",
    "South Europe": "#c44e52",
    "Great Britain": "#7f7f7f",
}


def load_consensus_panels() -> pd.DataFrame:
    usecols = [
        "country_code",
        "nli_topic",
        "electoral_cycle_progress",
        "classifier_count_consistent",
        "classifier_count_unrelated",
        "classifier_count_inconsistent",
        "classifier_share_consistent",
        "classifier_share_unrelated",
        "classifier_share_inconsistent",
        "n_pairs",
    ]
    frames = []
    for path in sorted(CONSENSUS_PANEL_DIR.glob("*/*_regression_panel_model.csv")):
        frames.append(pd.read_csv(path, usecols=usecols))
    data = pd.concat(frames, ignore_index=True)
    data["cluster"] = data["country_code"].map(CLUSTERS)
    data["topic"] = data["nli_topic"].replace({"Gal_Tan": "GAL-TAN"})
    related = data["classifier_count_consistent"] + data["classifier_count_inconsistent"]
    data["H"] = np.where(related.gt(0), data["classifier_count_inconsistent"] / related, np.nan)
    data["R"] = (data["classifier_count_consistent"] + data["classifier_count_inconsistent"]) / data["n_pairs"]
    data["J"] = data["classifier_count_inconsistent"] / data["n_pairs"]
    data["phase"] = pd.cut(
        data["electoral_cycle_progress"],
        bins=[-1e-9, 1 / 3, 2 / 3, 1 + 1e-9],
        labels=["Post-election", "Mid-cycle", "Pre-election"],
    )
    data["cycle_bin"] = pd.cut(
        data["electoral_cycle_progress"],
        bins=np.linspace(0, 1, 11),
        labels=[f"{0.05 + 0.1 * i:.2f}" for i in range(10)],
        include_lowest=True,
    )
    data["cycle_midpoint"] = data["cycle_bin"].astype(float)
    return data


def summarize_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = (
        data.groupby("cycle_midpoint", as_index=False)
        .agg(
            H=("H", "mean"),
            R=("R", "mean"),
            J=("J", "mean"),
            consistent=("classifier_share_consistent", "mean"),
            unrelated=("classifier_share_unrelated", "mean"),
            inconsistent=("classifier_share_inconsistent", "mean"),
        )
        .sort_values("cycle_midpoint")
    )
    cluster = (
        data.groupby(["cluster", "cycle_midpoint"], as_index=False)
        .agg(H=("H", "mean"), R=("R", "mean"), J=("J", "mean"))
        .sort_values(["cluster", "cycle_midpoint"])
    )
    return pooled, cluster


def summarize_country_and_cluster_shifts(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_summary = (
        data.groupby(["country_code", "cluster", "phase"], as_index=False)
        .agg(H=("H", "mean"), R=("R", "mean"), J=("J", "mean"), cells=("n_pairs", "size"), undefined_H=("H", lambda v: int(v.isna().sum())))
    )
    cluster_phase_summary = (
        data.groupby(["cluster", "phase"], as_index=False)
        .agg(H=("H", "mean"), R=("R", "mean"), J=("J", "mean"), cells=("n_pairs", "size"), undefined_H=("H", lambda v: int(v.isna().sum())))
    )

    def pivot_shift(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        wide = frame.pivot(index=keys, columns="phase", values=["H", "R", "J", "cells", "undefined_H"])
        wide.columns = [f"{metric}_{phase}" for metric, phase in wide.columns]
        out = wide.reset_index()
        out["delta_H_pp"] = 100 * (out["H_Pre-election"] - out["H_Post-election"])
        out["delta_R_pp"] = 100 * (out["R_Pre-election"] - out["R_Post-election"])
        out["delta_J_pp"] = 100 * (out["J_Pre-election"] - out["J_Post-election"])
        total_undefined = out["undefined_H_Post-election"].fillna(0) + out["undefined_H_Mid-cycle"].fillna(0) + out["undefined_H_Pre-election"].fillna(0)
        total_cells = out["cells_Post-election"].fillna(0) + out["cells_Mid-cycle"].fillna(0) + out["cells_Pre-election"].fillna(0)
        out["undefined_H_share"] = np.where(total_cells.gt(0), total_undefined / total_cells, np.nan)
        return out

    country_shift = pivot_shift(phase_summary, ["country_code", "cluster"])
    country_shift["cluster_order"] = country_shift["cluster"].map({name: i for i, name in enumerate(CLUSTER_ORDER)})
    country_shift = country_shift.sort_values(["cluster_order", "country_code"]).drop(columns="cluster_order")

    cluster_shift = pivot_shift(cluster_phase_summary, ["cluster"])
    cluster_shift["cluster_order"] = cluster_shift["cluster"].map({name: i for i, name in enumerate(CLUSTER_ORDER)})
    cluster_shift = cluster_shift.sort_values("cluster_order").drop(columns="cluster_order")
    return country_shift, cluster_shift


def summarize_salience_clusters() -> pd.DataFrame:
    salience = pd.read_csv(SALIENCE_LOO_PATH)
    salience["cluster"] = salience["omitted_country"].map(CLUSTERS)
    rows = []
    for cluster in CLUSTER_ORDER:
        subset = salience.loc[salience["cluster"].eq(cluster)]
        if subset.empty:
            continue
        row = {"cluster": cluster, "countries": ", ".join(sorted(subset["omitted_country"].unique()))}
        for hypothesis in ["H1a", "H2a"]:
            part = subset.loc[subset["hypothesis"].eq(hypothesis)]
            est = 100 * part["estimate"]
            row[f"{hypothesis}_range"] = f"{est.min():+.2f} to {est.max():+.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_cycle_diagnostics_figure(pooled: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9.2, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

    ax_top = fig.add_subplot(gs[0, :])
    x = pooled["cycle_midpoint"].to_numpy()
    consistent = 100 * pooled["consistent"].to_numpy()
    unrelated = 100 * pooled["unrelated"].to_numpy()
    inconsistent = 100 * pooled["inconsistent"].to_numpy()
    ax_top.bar(x, consistent, width=0.082, color=COLORS["consistent"], label="Consistent")
    ax_top.bar(x, unrelated, width=0.082, bottom=consistent, color=COLORS["unrelated"], label="Unrelated")
    ax_top.bar(x, inconsistent, width=0.082, bottom=consistent + unrelated, color=COLORS["inconsistent"], label="Inconsistent")
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_ylim(0, 100)
    ax_top.set_xticks([0.0, 0.5, 1.0], ["Post-election", "Mid-cycle", "Pre-election"])
    ax_top.set_ylabel("Average pair share (%)")
    ax_top.set_title("A. Pair composition over the electoral cycle", loc="left", fontsize=10, weight="bold")
    ax_top.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax_top.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)

    titles = {"H": "B. Conditional H", "R": "C. Relation rate R", "J": "D. Unconditional J"}
    for idx, measure in enumerate(["H", "R", "J"]):
        ax = fig.add_subplot(gs[1, idx])
        ax.plot(x, 100 * pooled[measure].to_numpy(), color=COLORS[measure], linewidth=2.0)
        ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_title(titles[measure], loc="left", fontsize=10, weight="bold")
        if idx == 0:
            ax.set_ylabel("Equal-cell mean (%)")
        ax.set_xlabel("Electoral proximity")
    fig.subplots_adjust(top=0.9, left=0.08, right=0.98, bottom=0.08)
    fig.savefig(OUT_FIG_DIR / "hrj_cycle_diagnostics.png", dpi=220)
    plt.close(fig)


def save_cluster_profiles_figure(cluster: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), sharex=True)
    titles = ["A. H by cluster", "B. R by cluster", "C. J by cluster"]
    for ax, measure, title in zip(axes, ["H", "R", "J"], titles):
        for cluster_name in CLUSTER_ORDER:
            rows = cluster.loc[cluster["cluster"].eq(cluster_name)].sort_values("cycle_midpoint")
            if rows.empty:
                continue
            ax.plot(rows["cycle_midpoint"], 100 * rows[measure], linewidth=1.8, color=COLORS[cluster_name], label=CLUSTER_SHORT[cluster_name])
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0], ["Post-election", "Mid-cycle", "Pre-election"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=10, weight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Equal-cell mean (%)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.subplots_adjust(left=0.08, right=0.84, top=0.9, bottom=0.18, wspace=0.28)
    fig.savefig(OUT_FIG_DIR / "hrj_cluster_profiles.png", dpi=220)
    plt.close(fig)


def format_pp(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:+.2f}"


def make_country_table(country_shift: pd.DataFrame) -> str:
    lines = [
        "#figure(",
        "  kind: table,",
        "  caption: [Descriptive pre-election minus post-election shifts in H, R, and J by country],",
        "  table(",
        "    columns: (0.8fr, 1.4fr, 0.85fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr),",
        "    inset: 3.5pt,",
        "    align: (left, left, right, right, right, right, right),",
        "    stroke: none,",
        "    table.header([*Country*], [*Cluster*], [*Delta H*], [*Delta R*], [*Delta J*], [*Cells*], [*Undefined H (%)*]),",
        "    table.hline(stroke: 0.7pt),",
    ]
    for _, row in country_shift.iterrows():
        total_cells = int(row["cells_Post-election"] + row["cells_Mid-cycle"] + row["cells_Pre-election"])
        undefined_share = 100 * row["undefined_H_share"]
        lines.append(
            f"    [{row['country_code']}], [{row['cluster']}], [{format_pp(row['delta_H_pp'])}], [{format_pp(row['delta_R_pp'])}], [{format_pp(row['delta_J_pp'])}], [{total_cells:,}], [{undefined_share:.1f}],"
        )
    lines.extend(["    table.hline(stroke: 0.7pt),", "  )", ")", ""])
    return "\n".join(lines)


def make_cluster_table(cluster_shift: pd.DataFrame, salience_cluster: pd.DataFrame) -> str:
    merged = cluster_shift.merge(salience_cluster, on="cluster", how="left")
    lines = [
        "#figure(",
        "  kind: table,",
        "  caption: [Descriptive cluster-level shifts in H, R, and J, with salience leave-one-country-out ranges],",
        "  table(",
        "    columns: (1.35fr, 1.15fr, 0.85fr, 0.85fr, 0.85fr, 1.2fr, 1.2fr),",
        "    inset: 3.5pt,",
        "    align: (left, left, right, right, right, center, center),",
        "    stroke: none,",
        "    table.header([*Cluster*], [*Countries*], [*Delta H*], [*Delta R*], [*Delta J*], [*H1a LOO range*], [*H2a LOO range*]),",
        "    table.hline(stroke: 0.7pt),",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"    [{row['cluster']}], [{row.get('countries', '--')}], [{format_pp(row['delta_H_pp'])}], [{format_pp(row['delta_R_pp'])}], [{format_pp(row['delta_J_pp'])}], [{row.get('H1a_range', '--')}], [{row.get('H2a_range', '--')}],"
        )
    lines.extend(["    table.hline(stroke: 0.7pt),", "  )", ")", ""])
    return "\n".join(lines)


def write_typ_fragment(country_shift: pd.DataFrame, cluster_shift: pd.DataFrame, salience_cluster: pd.DataFrame) -> None:
    text = f"""== Country and Cluster Profiles

The next appendix block adds descriptive country and cluster summaries for the consensus-classifier decomposition. These are equal-cell summaries over the observed party-topic-month panels. They are intended to make the denominator logic visually transparent rather than to replace the primary Webb-inference tables.

#figure(
  image("substantive graphs/robustness_appendix/hrj_cycle_diagnostics.png", width: 100%),
  caption: [How the underlying pair composition drives the H/R/J decomposition over the electoral cycle],
)

#figure(
  image("substantive graphs/robustness_appendix/hrj_cluster_profiles.png", width: 100%),
  caption: [Cluster-level descriptive profiles for H, R, and J],
)

{make_country_table(country_shift)}
#text(size: 9pt)[
Notes: `Delta H`, `Delta R`, and `Delta J` are pre-election minus post-election descriptive differences in percentage points. The summaries pool both topics and average equally across observed country-party-topic-month cells. `Undefined H (%)` reports the share of cells with no pair classified as either consistent or inconsistent.
]

{make_cluster_table(cluster_shift, salience_cluster)}
#text(size: 9pt)[
Notes: The cluster columns report the same pre-election minus post-election descriptive differences in percentage points. The salience leave-one-country-out ranges come from the completed Webb-bootstrap salience archive and summarize the omitted-country H1a and H2a point estimates within each regional cluster. Exact conditional-$H$ leave-one-country-out bootstrap estimates are still not available in the evidence archive, so the inconsistency cluster summaries remain descriptive.
]
"""
    OUT_TYP_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_consensus_panels()
    pooled, cluster = summarize_profiles(data)
    country_shift, cluster_shift = summarize_country_and_cluster_shifts(data)
    salience_cluster = summarize_salience_clusters()
    pooled.to_csv(OUT_DATA_DIR / "hrj_cycle_bin_summary.csv", index=False)
    cluster.to_csv(OUT_DATA_DIR / "hrj_cluster_cycle_summary.csv", index=False)
    country_shift.to_csv(OUT_DATA_DIR / "hrj_country_shift_summary.csv", index=False)
    cluster_shift.to_csv(OUT_DATA_DIR / "hrj_cluster_shift_summary.csv", index=False)
    salience_cluster.to_csv(OUT_DATA_DIR / "salience_cluster_leave_one_country_ranges.csv", index=False)
    save_cycle_diagnostics_figure(pooled)
    save_cluster_profiles_figure(cluster)
    write_typ_fragment(country_shift, cluster_shift, salience_cluster)


if __name__ == "__main__":
    main()
