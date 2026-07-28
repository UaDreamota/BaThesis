from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "thesis_plda_outputs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from causality import plda_linear_regression as ols
from utils import HYBRID_TOPIC_MAP, load_country, merge_topics

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


CLUSTERS = {
    "cee": ("Central-East Europe", ["CZ", "EE", "LV", "PL"]),
    "nordics": ("Nordics", ["DK", "FI", "NO", "SE"]),
    "west": ("West Europe", ["AT", "BE", "NL"]),
    "south": ("South Europe", ["ES", "IT", "PT", "GR"]),
}

TOPIC_ORDER = [
    "Macroeconomics",
    "Gal_Tan",
    "Economics",
    "Welfare_Human_Development",
    "Institutions_Rights_Law",
    "Foreign_Security",
    "Environment_Land_Energy",
    "Infrastructure_Technology",
    "Residual",
]

TOPIC_LABELS = {
    "Macroeconomics": "Macro",
    "Gal_Tan": "GAL-TAN",
    "Economics": "Economics",
    "Welfare_Human_Development": "Welfare",
    "Institutions_Rights_Law": "Institutions",
    "Foreign_Security": "Foreign/security",
    "Environment_Land_Energy": "Environment",
    "Infrastructure_Technology": "Infrastructure",
    "Residual": "Residual",
}

TOPIC_COLORS = {
    "Macroeconomics": "#0072B2",
    "Gal_Tan": "#D55E00",
    "Economics": "#009E73",
    "Welfare_Human_Development": "#CC79A7",
    "Institutions_Rights_Law": "#F0E442",
    "Foreign_Security": "#56B4E9",
    "Environment_Land_Energy": "#6A3D9A",
    "Infrastructure_Technology": "#999999",
    "Residual": "#333333",
}

REGRESSION_TERMS = [
    "electoral_cycle_progress",
    "party_in_government",
    "party_prime_minister",
    "party_seat_share",
    "log1p_speech_words",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]

TERM_LABELS = {
    "electoral_cycle_progress": "Electoral cycle progress",
    "party_in_government": "Government party",
    "party_prime_minister": "Prime minister party",
    "party_seat_share": "Party seat share",
    "log1p_speech_words": "Speech volume, log words",
    "cabinet_is_coalition": "Coalition cabinet",
    "cabinet_has_absolute_majority": "Cabinet majority",
    "cabinet_caretaker": "Caretaker cabinet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis-ready PLDA figures and cluster regressions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-countries", nargs="*", default=["CZ", "DK", "NL", "PT"])
    parser.add_argument("--top-parties", type=int, default=4)
    return parser.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {name: base / name for name in ["figures", "data", "regression"]}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def topic_cols(data: pd.DataFrame) -> list[str]:
    cols = [col for col in data.columns if re.fullmatch(r"topic_\d+", str(col))]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def best_grid(country: str) -> pd.Series:
    path = TEST_OUTPUT_DIR / f"plda_grid_search_log_{country}.csv"
    data = pd.read_csv(path)
    if "is_best" in data.columns:
        mask = data["is_best"]
        if mask.dtype == object:
            mask = mask.astype(str).str.lower().eq("true")
        if mask.any():
            return data.loc[mask].iloc[0]
    return data.iloc[0]


def simple_tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)


def base_topic_label(label: str) -> str:
    return re.sub(r"\s+\d+$", "", str(label)).strip()


def topic_mapping(country: str, distribution: pd.DataFrame) -> dict[str, str]:
    labels_path = (
        TEST_OUTPUT_DIR
        / "plda_topic_distributions"
        / country
        / f"{country}_plda_topic_labels.csv"
    )
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        required = {"topic_column", "topic_label"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"{labels_path} is missing columns: {sorted(missing)}")
        out = {}
        for _, row in labels.iterrows():
            base = base_topic_label(str(row["topic_label"]))
            out[str(row["topic_column"])] = HYBRID_TOPIC_MAP.get(base, base)
        return out

    grid = best_grid(country)
    latent_topics = int(grid.get("n_latent_topics", 0))
    topics_per_label = int(grid.get("topics_per_label", 1))
    visible_cols = topic_cols(distribution)[:-latent_topics] if latent_topics else topic_cols(distribution)
    expected_labels = len(visible_cols) // topics_per_label

    individual_path = TEST_OUTPUT_DIR / f"plda_individual_speeches_{country}.csv"
    if individual_path.exists():
        individual_labels = (
            pd.read_csv(individual_path, usecols=["topic_label"], low_memory=False)["topic_label"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )
        if len(individual_labels) == expected_labels:
            expanded_individual = []
            for label in individual_labels:
                expanded_individual.extend([label] * topics_per_label)
            return {
                topic_col: HYBRID_TOPIC_MAP.get(base_topic_label(label), base_topic_label(label))
                for topic_col, label in zip(visible_cols, expanded_individual, strict=True)
            }

    metadata = merge_topics(load_country(country)).dropna(subset=["topic_label"]).copy()
    metadata = metadata.loc[metadata["text"].map(lambda text: len(simple_tokenizer(text)) > 0)]
    metadata = metadata.reset_index(drop=True)

    label_col = None
    for candidate in ["original_topic_label", "topic_label", "broad_topic"]:
        if candidate in metadata.columns and metadata[candidate].dropna().nunique() == expected_labels:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(f"Could not infer PLDA topic labels for {country}: expected {expected_labels}.")

    labels = metadata[label_col].dropna().drop_duplicates().astype(str).tolist()
    expanded: list[str] = []
    for label in labels:
        expanded.extend([label] * topics_per_label)
    if len(expanded) != len(visible_cols):
        raise ValueError(f"PLDA topic layout mismatch for {country}.")
    return {
        topic_col: HYBRID_TOPIC_MAP.get(base_topic_label(label), base_topic_label(label))
        for topic_col, label in zip(visible_cols, expanded, strict=True)
    }


def country_cluster(country: str) -> tuple[str, str]:
    for cluster_id, (label, countries) in CLUSTERS.items():
        if country in countries:
            return cluster_id, label
    return "other", "Other"


def load_distribution(country: str) -> pd.DataFrame:
    path = TEST_OUTPUT_DIR / "plda_topic_distributions" / country / f"{country}_plda_topic_distribution_log_word_count.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def aggregate_country(country: str) -> pd.DataFrame:
    distribution = load_distribution(country)
    mapping = topic_mapping(country, distribution)
    weight_source = distribution["speech_word_count_sum"] if "speech_word_count_sum" in distribution.columns else distribution.get("speech_rows", 1.0)
    weights = pd.to_numeric(weight_source, errors="coerce") if isinstance(weight_source, pd.Series) else pd.Series(1.0, index=distribution.index)
    weights = weights.fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=distribution.index)

    rows = []
    for topic in TOPIC_ORDER:
        cols = [col for col, mapped in mapping.items() if mapped == topic]
        values = distribution[cols].sum(axis=1) if cols else pd.Series(0.0, index=distribution.index)
        rows.append({"country": country, "topic": topic, "share": float(np.average(values, weights=weights))})
    out = pd.DataFrame(rows)
    total = out["share"].sum()
    if total > 0:
        out["share"] = out["share"] / total
    cluster_id, cluster_label = country_cluster(country)
    out["cluster"] = cluster_id
    out["cluster_label"] = cluster_label
    return out


def topic_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    country_summary = pd.concat(
        [aggregate_country(country) for _, countries in CLUSTERS.values() for country in countries],
        ignore_index=True,
    )
    cluster_summary = (
        country_summary.groupby(["cluster", "cluster_label", "topic"], as_index=False)
        .agg(share=("share", "mean"))
    )
    cluster_summary["share"] = cluster_summary.groupby("cluster")["share"].transform(lambda values: values / values.sum())
    return country_summary, cluster_summary


def plot_cluster_topics(cluster_summary: pd.DataFrame, path: Path) -> None:
    cluster_ids = list(CLUSTERS)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    left = np.zeros(len(cluster_ids))
    y = np.arange(len(cluster_ids))
    for topic in TOPIC_ORDER:
        values = [
            float(cluster_summary.loc[(cluster_summary["cluster"] == cluster_id) & (cluster_summary["topic"] == topic), "share"].sum())
            for cluster_id in cluster_ids
        ]
        ax.barh(y, values, left=left, color=TOPIC_COLORS[topic], edgecolor="white", linewidth=0.7, label=TOPIC_LABELS[topic])
        left += np.asarray(values)
    ax.set_yticks(y)
    ax.set_yticklabels([CLUSTERS[cluster_id][0] for cluster_id in cluster_ids])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Share of PLDA speech-topic mass")
    ax.set_title("Parliamentary speech topic distributions by country cluster")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_country_heatmap(country_summary: pd.DataFrame, path: Path) -> None:
    countries = [country for _, cluster_countries in CLUSTERS.values() for country in cluster_countries]
    labels = [f"{country} ({country_cluster(country)[1]})" for country in countries]
    matrix = (
        country_summary.pivot_table(index="country", columns="topic", values="share", fill_value=0)
        .reindex(index=countries, columns=TOPIC_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_yticks(np.arange(len(countries)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(TOPIC_ORDER)))
    ax.set_xticklabels([TOPIC_LABELS[topic] for topic in TOPIC_ORDER], rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.0%}", ha="center", va="center", fontsize=6)
    ax.set_title("Country-level PLDA topic distribution")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Topic share")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_path(country: str) -> Path:
    return TEST_OUTPUT_DIR / "plda_regression_panel" / country / f"{country}_plda_regression_panel_model.csv"


def load_panel(country: str) -> pd.DataFrame:
    data = pd.read_csv(panel_path(country), low_memory=False)
    data["month_start"] = pd.to_datetime(data["month_start"], errors="coerce")
    for column in ["last_election_date", "next_election_date"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def recent_window(data: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]:
    min_date = data["month_start"].min()
    max_date = data["month_start"].max()
    election_cols = [col for col in ["last_election_date", "next_election_date"] if col in data.columns]
    elections = pd.concat([data[col] for col in election_cols], ignore_index=True).dropna().drop_duplicates().sort_values()
    elections = elections[(elections >= min_date) & (elections <= max_date)]
    if len(elections) >= 2:
        start = elections.iloc[-2] - pd.DateOffset(months=6)
    else:
        start = max_date - pd.DateOffset(years=5)
    start = max(pd.Timestamp(start), pd.Timestamp(min_date))
    return start, pd.Timestamp(max_date), [pd.Timestamp(date) for date in elections if start <= date <= max_date]


def plot_timeseries(country: str, output_dir: Path, top_parties: int) -> Path:
    data = load_panel(country)
    start, end, elections = recent_window(data)
    window = data[(data["month_start"] >= start) & (data["month_start"] <= end)].copy()
    weight_col = "speech_volume_words" if "speech_volume_words" in window.columns else "speech_rows"
    top_parties_index = window.groupby("speech_party")[weight_col].sum().sort_values(ascending=False).head(top_parties).index
    plot_df = window[window["speech_party"].isin(top_parties_index)].sort_values(["speech_party", "month_start"])
    plot_df["js_rolling"] = plot_df.groupby("speech_party")["js_distance"].transform(lambda series: series.rolling(3, min_periods=1).mean())

    fig, ax = plt.subplots(figsize=(11, 5.6))
    for party, party_df in plot_df.groupby("speech_party", sort=True):
        ax.plot(party_df["month_start"], party_df["js_rolling"], marker="o", markersize=3, linewidth=1.8, label=str(party))
    for date in elections:
        ax.axvline(date, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(date, 0.98, date.strftime("%Y-%m"), rotation=90, va="top", ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlim(start - pd.DateOffset(months=1), end + pd.DateOffset(months=1))
    ax.set_ylabel("Jensen-Shannon distance")
    ax.set_xlabel("Month")
    ax.set_title(f"{country}: manifesto-speech distance, recent electoral cycle window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(top_parties, 4), fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = output_dir / f"{country}_js_distance_recent_election_window.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def pstars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def regression_cell(result: ols.RegressionResult, term: str) -> str:
    row = result.coefficients[result.coefficients["term"] == term]
    if row.empty:
        return ""
    values = row.iloc[0]
    return f"{values['estimate']:.3f}{pstars(values['p_value'])} ({values['std_error']:.3f})"


def regression_table(results: dict[str, ols.RegressionResult]) -> str:
    labels = list(results)
    lines = ["term\t" + "\t".join(labels)]
    for term in REGRESSION_TERMS:
        lines.append(TERM_LABELS[term] + "\t" + "\t".join(regression_cell(result, term) for result in results.values()))
    lines.append("Observations\t" + "\t".join(str(result.nobs) for result in results.values()))
    lines.append("Adjusted R2\t" + "\t".join(f"{result.adj_r_squared:.3f}" for result in results.values()))
    lines.append("")
    lines.append("Notes: HC1 robust standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01.")
    lines.append("All multi-country models include speech-party and country fixed effects.")
    return "\n".join(lines) + "\n"


def run_regressions(output_dir: Path, outcome: str) -> None:
    specs = {cluster_id: countries for cluster_id, (_, countries) in CLUSTERS.items()}
    specs["pooled_clusters"] = [country for _, countries in CLUSTERS.values() for country in countries]
    results: dict[str, ols.RegressionResult] = {}
    for label, countries in specs.items():
        data, inputs = ols.load_country_panels(countries)
        fixed_effects = list(ols.DEFAULT_FIXED_EFFECTS)
        if len(countries) > 1 and "country_code" not in fixed_effects:
            fixed_effects.append("country_code")
        result = ols.fit_ols(
            data=data,
            outcome=outcome,
            predictors=list(ols.DEFAULT_PREDICTORS),
            fixed_effects=fixed_effects,
            se_type="robust",
            drop_missing=True,
            model_label=label,
        )
        results[label] = result
        (output_dir / f"{label}_{outcome}_regression_table.txt").write_text(
            ols.summary_text(result, ", ".join(str(path) for path in inputs)),
            encoding="utf-8",
        )
        result.coefficients.to_csv(output_dir / f"{label}_{outcome}_coefficients.csv", index=False)
    (output_dir / f"cluster_regression_table_{outcome}.txt").write_text(regression_table(results), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.output_dir)

    country_summary, cluster_summary = topic_summaries()
    country_summary.to_csv(paths["data"] / "plda_country_topic_distribution_summary.csv", index=False)
    cluster_summary.to_csv(paths["data"] / "plda_cluster_topic_distribution_summary.csv", index=False)
    plot_cluster_topics(cluster_summary, paths["figures"] / "plda_cluster_topic_distribution_stacked.png")
    plot_country_heatmap(country_summary, paths["figures"] / "plda_country_topic_distribution_heatmap.png")
    timeseries_paths = [
        plot_timeseries(country.upper(), paths["figures"], args.top_parties)
        for country in args.plot_countries
    ]
    for outcome in ["js_distance", "alignment_score"]:
        run_regressions(paths["regression"], outcome)

    manifest_lines = [
        "Thesis PLDA outputs",
        "",
        "Figures:",
        str(paths["figures"] / "plda_cluster_topic_distribution_stacked.png"),
        str(paths["figures"] / "plda_country_topic_distribution_heatmap.png"),
        *[str(path) for path in timeseries_paths],
        "",
        "Regression tables:",
        str(paths["regression"] / "cluster_regression_table_js_distance.txt"),
        str(paths["regression"] / "cluster_regression_table_alignment_score.txt"),
    ]
    (args.output_dir / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
