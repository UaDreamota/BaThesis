from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import load_country, merge_topics


DEFAULT_DISTRIBUTION_INPUT = OUTPUT_DIR / "plda_distribution.csv"
DEFAULT_GRID_LOG_INPUT = OUTPUT_DIR / "plda_grid_search_log.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "plda_topic_distributions"
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--c",
    default=None,
    type=str,
    help="Country abbreviation. If omitted, infer it from the PLDA grid log.",
)
parser.add_argument(
    "--distribution-input",
    default=DEFAULT_DISTRIBUTION_INPUT,
    type=Path,
    help="CSV containing one PLDA topic distribution row per modeled speech.",
)
parser.add_argument(
    "--grid-log-input",
    default=DEFAULT_GRID_LOG_INPUT,
    type=Path,
    help="PLDA grid-search log used to infer --c when --c is omitted.",
)
parser.add_argument(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    type=Path,
    help="Directory where PLDA monthly distribution plots and CSVs are written.",
)
parser.add_argument(
    "--latent-topics",
    "--latent_topics",
    dest="latent_topics",
    default=None,
    type=int,
    help=(
        "Number of PLDA latent topics. Latent topics are assumed to be the last "
        "topic columns and are excluded from plots. If omitted, infer from the "
        "best PLDA grid-search row."
    ),
)
parser.add_argument(
    "--topics-per-label",
    "--topics_per_label",
    dest="topics_per_label",
    default=None,
    type=int,
    help=(
        "PLDA topics_per_label value. If omitted, infer from the best PLDA "
        "grid-search row."
    ),
)
parser.add_argument(
    "--topic-label-column",
    default=None,
    type=str,
    help=(
        "Metadata column used as PLDA label. If omitted, infer from columns "
        "whose unique value count matches the PLDA label count."
    ),
)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def infer_country(grid_log_path: Path) -> str:
    if not grid_log_path.exists():
        raise FileNotFoundError(
            f"Could not infer country because grid log is missing: {grid_log_path}"
        )

    grid_log = pd.read_csv(grid_log_path)
    if "country" not in grid_log.columns:
        raise ValueError(f"Grid log has no country column: {grid_log_path}")

    if "is_best" in grid_log.columns:
        is_best = grid_log["is_best"]
        if is_best.dtype == object:
            best_mask = is_best.astype(str).str.lower().eq("true")
        else:
            best_mask = is_best.fillna(False).astype(bool)
    else:
        best_mask = pd.Series(False, index=grid_log.index)

    if best_mask.any():
        countries = grid_log.loc[best_mask, "country"].dropna().unique()
    else:
        countries = grid_log["country"].dropna().unique()

    if len(countries) != 1:
        raise ValueError(
            "Could not infer a single country from the PLDA grid log. "
            "Pass --c explicitly."
        )
    return str(countries[0]).upper()


def infer_latent_topics(grid_log_path: Path) -> int:
    if not grid_log_path.exists():
        raise FileNotFoundError(
            f"Could not infer latent topics because grid log is missing: {grid_log_path}"
        )

    grid_log = pd.read_csv(grid_log_path)
    if "n_latent_topics" not in grid_log.columns:
        raise ValueError(
            f"Grid log has no n_latent_topics column: {grid_log_path}. "
            "Pass --latent-topics explicitly."
        )

    if "is_best" in grid_log.columns:
        is_best = grid_log["is_best"]
        if is_best.dtype == object:
            best_mask = is_best.astype(str).str.lower().eq("true")
        else:
            best_mask = is_best.fillna(False).astype(bool)
    else:
        best_mask = pd.Series(False, index=grid_log.index)

    if best_mask.any():
        latent_topic_values = grid_log.loc[best_mask, "n_latent_topics"].dropna().unique()
    else:
        latent_topic_values = grid_log["n_latent_topics"].dropna().unique()

    if len(latent_topic_values) != 1:
        raise ValueError(
            "Could not infer a single latent topic count from the PLDA grid log. "
            "Pass --latent-topics explicitly."
        )
    return int(latent_topic_values[0])


def infer_topics_per_label(grid_log_path: Path) -> int:
    if not grid_log_path.exists():
        raise FileNotFoundError(
            f"Could not infer topics per label because grid log is missing: {grid_log_path}"
        )

    grid_log = pd.read_csv(grid_log_path)
    if "topics_per_label" not in grid_log.columns:
        raise ValueError(
            f"Grid log has no topics_per_label column: {grid_log_path}. "
            "Pass --topics-per-label explicitly."
        )

    if "is_best" in grid_log.columns:
        is_best = grid_log["is_best"]
        if is_best.dtype == object:
            best_mask = is_best.astype(str).str.lower().eq("true")
        else:
            best_mask = is_best.fillna(False).astype(bool)
    else:
        best_mask = pd.Series(False, index=grid_log.index)

    if best_mask.any():
        values = grid_log.loc[best_mask, "topics_per_label"].dropna().unique()
    else:
        values = grid_log["topics_per_label"].dropna().unique()

    if len(values) != 1:
        raise ValueError(
            "Could not infer a single topics_per_label value from the PLDA grid log. "
            "Pass --topics-per-label explicitly."
        )
    return int(values[0])


def topic_columns(data: pd.DataFrame) -> list[str]:
    columns = [
        col
        for col in data.columns
        if TOPIC_COLUMN_RE.match(str(col))
    ]
    if not columns:
        raise ValueError("No topic_* columns found in PLDA distribution CSV.")
    return sorted(columns, key=lambda col: int(TOPIC_COLUMN_RE.match(col).group(1)))


def visible_topic_columns(topic_cols: list[str], latent_topics: int) -> list[str]:
    if latent_topics < 0:
        raise ValueError("--latent-topics must be >= 0.")
    if latent_topics == 0:
        return topic_cols
    if latent_topics >= len(topic_cols):
        raise ValueError(
            f"--latent-topics={latent_topics} would leave no topics to plot "
            f"from {len(topic_cols)} total topic columns."
        )
    return topic_cols[:-latent_topics]


def normalize_topic_columns(data: pd.DataFrame, topic_cols: list[str]) -> pd.DataFrame:
    normalized = data.copy()
    row_totals = normalized[topic_cols].sum(axis=1)
    if row_totals.le(0).any():
        raise ValueError("Cannot renormalize rows with zero visible topic mass.")
    normalized.loc[:, topic_cols] = normalized[topic_cols].div(row_totals, axis=0)
    return normalized


def infer_topic_label_column(
    metadata: pd.DataFrame,
    expected_label_count: int,
    explicit_label_column: str | None,
) -> str:
    if explicit_label_column is not None:
        if explicit_label_column not in metadata.columns:
            raise ValueError(f"Metadata has no column: {explicit_label_column}")
        actual_label_count = metadata[explicit_label_column].dropna().nunique()
        if actual_label_count != expected_label_count:
            raise ValueError(
                f"{explicit_label_column} has {actual_label_count} unique values, "
                f"but PLDA topic layout expects {expected_label_count} labels."
            )
        return explicit_label_column

    candidates = [
        column
        for column in ["topic_label", "broad_topic"]
        if column in metadata.columns
        and metadata[column].dropna().nunique() == expected_label_count
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer PLDA label column. "
            f"Expected {expected_label_count} unique labels; candidates={candidates}. "
            "Pass --topic-label-column explicitly."
        )
    return candidates[0]


def topic_display_labels(
    metadata: pd.DataFrame,
    topic_cols: list[str],
    topics_per_label: int,
    label_column: str,
) -> list[str]:
    if topics_per_label <= 0:
        raise ValueError("--topics-per-label must be > 0.")
    if len(topic_cols) % topics_per_label != 0:
        raise ValueError(
            f"{len(topic_cols)} plotted topics is not divisible by "
            f"--topics-per-label={topics_per_label}."
        )

    topic_labels = metadata[label_column].dropna().drop_duplicates().astype(str).tolist()
    expected_label_count = len(topic_cols) // topics_per_label
    if len(topic_labels) != expected_label_count:
        raise ValueError(
            f"{label_column} has {len(topic_labels)} first-seen labels, "
            f"but PLDA topic layout expects {expected_label_count}."
        )

    display_labels = []
    for topic_label in topic_labels:
        if topics_per_label == 1:
            display_labels.append(topic_label)
        else:
            display_labels.extend(
                f"{topic_label} {topic_i}"
                for topic_i in range(1, topics_per_label + 1)
            )
    return display_labels


def load_plda_distribution(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find PLDA distribution CSV: {path}")

    data = pd.read_csv(path)
    columns = topic_columns(data)
    return data[columns].copy()


def load_plda_metadata(country_code: str) -> pd.DataFrame:
    data = load_country(country_code)
    data = merge_topics(data)
    data = data.dropna(subset=["topic_label"]).copy()
    return data[["party", "month", "month_start", "topic_label", "broad_topic"]].reset_index(drop=True)


def build_monthly_distribution(topic_df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    metadata = load_plda_metadata(country_code)
    if len(topic_df) != len(metadata):
        raise ValueError(
            "PLDA distribution rows do not match the row-aligned speech metadata: "
            f"{len(topic_df):,} topic rows vs {len(metadata):,} metadata rows."
        )

    combined = pd.concat(
        [metadata.reset_index(drop=True), topic_df.reset_index(drop=True)],
        axis=1,
    )
    columns = topic_columns(topic_df)

    distribution = (
        combined.groupby(["party", "month", "month_start"], as_index=False)[columns]
        .mean()
        .sort_values(["month_start", "party"])
        .reset_index(drop=True)
    )
    return distribution


def generate_topic_colors(n_topics: int) -> np.ndarray:
    return plt.cm.gist_ncar(np.linspace(0, 1, n_topics, endpoint=False))


def plot_stacked_topic_bars(
    ax: plt.Axes,
    party_df: pd.DataFrame,
    topic_cols: list[str],
    topic_labels: list[str],
    colors: np.ndarray,
) -> None:
    x_positions = list(range(len(party_df)))
    month_labels = party_df["month"].to_numpy()
    bottom = np.zeros(len(party_df), dtype=float)

    for color, topic, topic_label in zip(colors, topic_cols, topic_labels, strict=True):
        values = party_df[topic].to_numpy()
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=topic_label,
            color=color,
            width=0.9,
        )
        bottom += values

    ax.set_ylim(0, 1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Topic proportion")

    tick_step = max(1, len(month_labels) // 12)
    tick_positions = x_positions[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(month_labels[::tick_step], rotation=45, ha="right")


def save_topic_distribution_plot(
    distribution_df: pd.DataFrame,
    country_code: str,
    latent_topics: int,
    topics_per_label: int,
    topic_labels: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    all_topic_cols = topic_columns(distribution_df)
    topic_cols = visible_topic_columns(all_topic_cols, latent_topics)
    if len(topic_labels) != len(topic_cols):
        raise ValueError(
            f"Got {len(topic_labels)} topic labels for {len(topic_cols)} plotted topics."
        )
    colors = generate_topic_colors(len(all_topic_cols))[: len(topic_cols)]
    saved_paths: list[Path] = []

    saved_paths.extend(
        save_plot_set(
            distribution_df=distribution_df,
            country_code=country_code,
            latent_topics=latent_topics,
            topic_cols=topic_cols,
            topic_labels=topic_labels,
            colors=colors,
            output_dir=output_dir,
            suffix="",
            title_note="original scale",
        )
    )

    normalized_df = normalize_topic_columns(distribution_df, topic_cols)
    saved_paths.extend(
        save_plot_set(
            distribution_df=normalized_df,
            country_code=country_code,
            latent_topics=latent_topics,
            topic_cols=topic_cols,
            topic_labels=topic_labels,
            colors=colors,
            output_dir=output_dir,
            suffix="_renormalized",
            title_note="renormalized visible topics",
        )
    )

    country_dir = output_dir / country_code.upper()
    csv_path = country_dir / f"{country_code.upper()}_plda_topic_distribution.csv"
    distribution_df.to_csv(csv_path, index=False)
    saved_paths.append(csv_path)

    mapping_path = country_dir / f"{country_code.upper()}_plda_topic_labels.csv"
    pd.DataFrame(
        {
            "topic_column": topic_cols,
            "topic_label": topic_labels,
            "topics_per_label": topics_per_label,
            "latent_topics_excluded": latent_topics,
        }
    ).to_csv(mapping_path, index=False)
    saved_paths.append(mapping_path)

    return saved_paths


def save_plot_set(
    distribution_df: pd.DataFrame,
    country_code: str,
    latent_topics: int,
    topic_cols: list[str],
    topic_labels: list[str],
    colors: np.ndarray,
    output_dir: Path,
    suffix: str,
    title_note: str,
) -> list[Path]:
    country_dir = output_dir / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    parties = sorted(distribution_df["party"].dropna().unique())
    saved_paths: list[Path] = []

    ncols = 2
    nrows = int((len(parties) + ncols - 1) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, max(4 * nrows, 6)),
        sharey=True,
    )
    axes_array = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, party in zip(axes_array, parties, strict=False):
        party_df = distribution_df[distribution_df["party"] == party].sort_values("month_start")
        plot_stacked_topic_bars(ax, party_df, topic_cols, topic_labels, colors)
        ax.set_title(str(party))

    for ax in axes_array[len(parties):]:
        ax.remove()

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(
        handles,
        topic_labels,
        loc="upper center",
        ncol=min(10, len(topic_labels)),
        fontsize=6,
    )
    fig.suptitle(
        (
            f"{country_code.upper()} PLDA topic shares by party over time "
            f"({len(topic_cols)} plotted topics, {latent_topics} latent excluded, "
            f"{title_note})"
        ),
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    master_plot_path = country_dir / (
        f"{country_code.upper()}_all_parties_plda_topic_distribution{suffix}.png"
    )
    fig.savefig(master_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(master_plot_path)

    for party in parties:
        party_df = distribution_df[distribution_df["party"] == party].sort_values("month_start")

        fig, ax = plt.subplots(figsize=(18, 7))
        plot_stacked_topic_bars(ax, party_df, topic_cols, topic_labels, colors)
        ax.set_title(
            f"{country_code.upper()} {party} PLDA topic distribution per month "
            f"({len(topic_cols)} plotted topics, {latent_topics} latent excluded, "
            f"{title_note})"
        )
        ax.legend(loc="upper center", ncol=min(10, len(topic_labels)), fontsize=6)
        fig.tight_layout()

        plot_path = country_dir / (
            f"{country_code.upper()}_{sanitize_filename(party)}"
            f"_plda_topic_distribution{suffix}.png"
        )
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def main(args: argparse.Namespace) -> None:
    country_code = args.c.upper() if args.c else infer_country(args.grid_log_input)
    latent_topics = (
        args.latent_topics
        if args.latent_topics is not None
        else infer_latent_topics(args.grid_log_input)
    )
    topics_per_label = (
        args.topics_per_label
        if args.topics_per_label is not None
        else infer_topics_per_label(args.grid_log_input)
    )
    topic_df = load_plda_distribution(args.distribution_input)
    all_topic_cols = topic_columns(topic_df)
    visible_topic_cols = visible_topic_columns(all_topic_cols, latent_topics)
    metadata = load_plda_metadata(country_code)
    label_column = infer_topic_label_column(
        metadata,
        len(visible_topic_cols) // topics_per_label,
        args.topic_label_column,
    )
    topic_labels = topic_display_labels(
        metadata,
        visible_topic_cols,
        topics_per_label,
        label_column,
    )
    distribution_df = build_monthly_distribution(topic_df, country_code)
    saved_paths = save_topic_distribution_plot(
        distribution_df,
        country_code,
        latent_topics,
        topics_per_label,
        topic_labels,
        args.output_dir,
    )

    print(f"Loaded {len(topic_df):,} PLDA document-topic rows for {country_code}.")
    print(
        f"Built {len(distribution_df):,} party-month rows across "
        f"{len(topic_columns(topic_df))} topics."
    )
    print(
        f"Plotted {len(visible_topic_cols)} "
        f"topics and excluded {latent_topics} latent topic(s)."
    )
    print(f"Topic legend labels inferred from {label_column}.")
    print(f"Saved PLDA topic outputs to: {args.output_dir / country_code}")
    print(saved_paths[0])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
