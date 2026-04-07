from pathlib import Path
import argparse
import os
import re
import pandas as pd 
import numpy as np

import datetime
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer


parser = argparse.ArgumentParser()

parser.add_argument("--c", default=None, type=str, help="Country's abbreveation")
parser.add_argument("--top-parties", default=3, type=int, help="Number of biggest parties to keep")




BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "parlam" 
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
PERPLEXITY_DIR = OUTPUT_DIR / "perplexities"
TOPIC_DISTRIBUTION_DIR = OUTPUT_DIR / "topic_distributions"
STOPWORDS_DIR = Path(__file__).resolve().parent / "stopwords"


def load_stopwords(country_code: str) -> list[str]:
    path = STOPWORDS_DIR / f"{country_code.upper()}.txt"
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_country(country_code: str) -> pd.DataFrame:
    path = DATA_DIR / f"ParlaMint-{country_code}_extracted.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find ParlaMint file: {path}")

    df = pd.read_csv(
        path,
        usecols=["country", "date", "party", "content_kind", "text", "speaker_type"],
    )
    df = df[df["content_kind"] == "speech"].copy()
    df = df[df["speaker_type"] == "regular"].copy()
    df = df.dropna(subset=["party", "text", "date"]).copy()

    df["party"] = df["party"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["party"] != ""].copy()
    df = df[df["text"] != ""].copy()
    df = df.dropna(subset=["date"]).copy()

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True)

def keep_top_parties(data: pd.DataFrame, n_parties: int = 3) -> pd.DataFrame:
    top_parties = data["party"].value_counts().head(n_parties).index
    return data[data["party"].isin(top_parties)].copy()

def aggregate_party_month(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["country", "party", "month", "month_start"], as_index=False)
        .agg(text=("text", " ".join))
        .sort_values(["month_start", "party"])
        .reset_index(drop=True)
    )

def transform_data(data: pd.DataFrame, country_code: str):
    Vectorizer = CountVectorizer(stop_words=load_stopwords(country_code))
    X = Vectorizer.fit_transform(data["text"])
    return X

def perplexity_magic(X):
    # n_topics_range = list(range(30, 41))
    n_topics_range = [45, 55, 57, 58, 59, 60, 65]
    results = {}
    for n_topics in n_topics_range:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method="batch",
        )
        lda.fit(X)
        results[n_topics] = lda.perplexity(X)
    return results

def build_topic_distribution(X, n_topics: int) -> pd.DataFrame:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )
    return pd.DataFrame(lda.fit_transform(X))


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def generate_topic_colors(topic_count: int):
    if topic_count <= 0:
        return []

    # Sample evenly across a continuous colormap so topic colors stay distinct
    # even when the number of topics exceeds the size of discrete palettes.
    return plt.cm.gist_ncar(np.linspace(0, 1, topic_count, endpoint=False))


def prepare_topic_distribution(
    aggregated_df: pd.DataFrame,
    X,
    best_n_topics: int,
) -> pd.DataFrame:
    topic_distribution = build_topic_distribution(X, best_n_topics)
    topic_distribution.columns = [f"topic_{idx}" for idx in topic_distribution.columns]

    distribution_df = pd.concat(
        [aggregated_df[["party", "month", "month_start"]].reset_index(drop=True), topic_distribution],
        axis=1,
    ).sort_values(["month_start", "party"])
    return distribution_df


def build_perplexity_profiles(
    speeches_df: pd.DataFrame,
    country_code: str,
    top_party_count: int,
) -> dict[str, dict[int, float]]:
    profiles: dict[str, dict[int, float]] = {}

    all_aggregated = aggregate_party_month(speeches_df)
    profiles["all_parties"] = perplexity_magic(transform_data(all_aggregated, country_code))

    top_one_aggregated = aggregate_party_month(keep_top_parties(speeches_df, n_parties=1))
    profiles["top_1_party"] = perplexity_magic(transform_data(top_one_aggregated, country_code))

    top_n_aggregated = aggregate_party_month(
        keep_top_parties(speeches_df, n_parties=top_party_count),
    )
    profiles[f"top_{top_party_count}_parties"] = perplexity_magic(
        transform_data(top_n_aggregated, country_code),
    )

    return profiles


def choose_topic_count(results: dict[str, dict[int, float]], top_party_count: int) -> int:
    selection_key = f"top_{top_party_count}_parties"
    return min(results[selection_key], key=results[selection_key].get)


def save_perplexity_plot(
    results: dict[str, dict[int, float]],
    country_code: str,
    selected_topics: int,
) -> Path:
    country_dir = PERPLEXITY_DIR / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "all_parties": "All parties",
        "top_1_party": "Top 1 party",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, profile in results.items():
        topics = list(profile.keys())
        perplexities = list(profile.values())
        ax.plot(topics, perplexities, marker="o", label=label_map.get(key, key.replace("_", " ")))

    ax.axvline(selected_topics, color="black", linestyle="--", linewidth=1, label=f"Selected topics: {selected_topics}")
    ax.set_title(f"{country_code.upper()} perplexity by topic count")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel("Perplexity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = country_dir / f"{country_code.upper()}_perplexity.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_topic_distribution_plot(
    distribution_df: pd.DataFrame,
    best_n_topics: int,
    country_code: str,
) -> list[Path]:
    country_dir = TOPIC_DISTRIBUTION_DIR / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    topic_columns = [col for col in distribution_df.columns if col.startswith("topic_")]
    parties = sorted(distribution_df["party"].unique())
    colors = generate_topic_colors(len(topic_columns))
    saved_paths: list[Path] = []

    ncols = 2
    nrows = int((len(parties) + ncols - 1) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, max(4 * nrows, 6)),
        sharey=True,
    )
    axes_array = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, party in zip(axes_array, parties, strict=False):
        party_df = distribution_df[distribution_df["party"] == party].sort_values("month_start")
        x_positions = list(range(len(party_df)))
        month_labels = party_df["month"].to_numpy()
        bottom = [0.0] * len(party_df)

        for color, topic in zip(colors, topic_columns, strict=False):
            values = party_df[topic].to_numpy()
            ax.bar(x_positions, values, bottom=bottom, label=topic, color=color, width=0.9)
            bottom = [current + value for current, value in zip(bottom, values, strict=False)]

        ax.set_title(party)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic proportion")
        tick_step = max(1, len(month_labels) // 12)
        tick_positions = x_positions[::tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(month_labels[::tick_step], rotation=45, ha="right")

    for ax in axes_array[len(parties):]:
        ax.remove()

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles, topic_columns, loc="upper center", ncol=min(5, len(topic_columns)))
    fig.suptitle(
        f"{country_code.upper()} topic shares by party over time ({best_n_topics} topics)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    master_plot_path = country_dir / f"{country_code.upper()}_all_parties_topic_distribution.png"
    fig.savefig(master_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(master_plot_path)

    for party in parties:
        party_df = distribution_df[distribution_df["party"] == party].sort_values("month_start")
        x_positions = list(range(len(party_df)))
        month_labels = party_df["month"].to_numpy()
        bottom = [0.0] * len(party_df)

        fig, ax = plt.subplots(figsize=(16, 7))
        for color, topic in zip(colors, topic_columns, strict=False):
            values = party_df[topic].to_numpy()
            ax.bar(x_positions, values, bottom=bottom, label=topic, color=color, width=0.9)
            bottom = [current + value for current, value in zip(bottom, values, strict=False)]

        ax.set_title(f"{country_code.upper()} {party} topic distribution per month ({best_n_topics} topics)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic proportion")
        ax.set_ylim(0, 1)
        tick_step = max(1, len(month_labels) // 12)
        tick_positions = x_positions[::tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(month_labels[::tick_step], rotation=45, ha="right")
        ax.legend(loc="upper center", ncol=min(5, len(topic_columns)))
        fig.tight_layout()

        plot_path = country_dir / f"{country_code.upper()}_{sanitize_filename(party)}_topic_distribution.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def main(args: argparse.Namespace):
    df = load_country(args.c)
    aggregated_df = aggregate_party_month(df)
    print(aggregated_df.info())
    X = transform_data(aggregated_df, args.c)
    start = perf_counter()
    results = build_perplexity_profiles(df, args.c, top_party_count=args.top_parties)
    end = perf_counter()
    best_n_topics = choose_topic_count(results, top_party_count=args.top_parties)
    perplexity_plot_path = save_perplexity_plot(results, args.c, selected_topics=best_n_topics)
    distribution_df = prepare_topic_distribution(aggregated_df, X, best_n_topics)
    save_topic_distribution_plot(distribution_df, best_n_topics, args.c)
    print(f"Time taken for clustering:{end-start}")
    print(results)
    print(f"Selected topic count from top {args.top_parties} parties: {best_n_topics}")
    print(f"Saved perplexity plot to: {perplexity_plot_path}")
    print(f"Saved topic distribution plots to: {TOPIC_DISTRIBUTION_DIR / args.c.upper()}")
    return None


if __name__ == ("__main__"):
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
