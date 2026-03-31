from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "parlam"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "topic_distributions"


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


def build_party_month_corpus(df: pd.DataFrame) -> pd.DataFrame:
    corpus = (
        df.groupby(["country", "party", "month", "month_start"], as_index=False)
        .agg(
            text=("text", " ".join),
            speech_count=("text", "size"),
        )
        .sort_values(["month_start", "party"])
        .reset_index(drop=True)
    )
    corpus["word_count"] = corpus["text"].str.split().str.len()
    return corpus


def topic_keywords(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n_words: int,
) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    rows: list[dict[str, str | int]] = []

    for topic_idx, weights in enumerate(model.components_):
        top_indices = np.argsort(weights)[::-1][:top_n_words]
        rows.append(
            {
                "topic": topic_idx,
                "top_words": ", ".join(feature_names[top_indices]),
            }
        )

    return pd.DataFrame(rows)


def fit_topic_model(
    corpus: pd.DataFrame,
    n_topics: int,
    max_features: int,
    top_n_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if corpus.empty:
        raise ValueError("No party-month documents are available after filtering.")

    topic_count = min(n_topics, len(corpus))
    if topic_count < 1:
        raise ValueError("Topic model needs at least one party-month document.")

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(corpus["text"])

    lda = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=42,
        learning_method="batch",
        max_iter=40,
    )
    lda.fit(matrix)
    normalized = lda.transform(matrix)

    distribution = corpus[
        ["country", "party", "month", "month_start", "speech_count", "word_count"]
    ].copy()
    for topic_idx in range(topic_count):
        distribution[f"topic_{topic_idx}"] = normalized[:, topic_idx]

    topic_columns = [col for col in distribution.columns if col.startswith("topic_")]
    distribution["dominant_topic"] = distribution[topic_columns].idxmax(axis=1)

    keywords = topic_keywords(lda, vectorizer, top_n_words=top_n_words)
    return distribution, keywords


def plot_topic_heatmap(distribution: pd.DataFrame, country_code: str) -> Path:
    topic_columns = [col for col in distribution.columns if col.startswith("topic_")]
    plot_df = distribution.sort_values(["month_start", "party"]).reset_index(drop=True)
    labels = [
        f"{party} {month}"
        for party, month in zip(plot_df["party"], plot_df["month"], strict=False)
    ]
    matrix = plot_df[topic_columns].to_numpy()

    fig_height = max(6, len(plot_df) * 0.28)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=matrix.max())
    ax.set_title(f"{country_code}: topic shares by party-month")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Party-month")
    ax.set_xticks(range(len(topic_columns)))
    ax.set_xticklabels(topic_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(image, ax=ax, label="Topic share")
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{country_code}_topic_distribution_heatmap.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_party_topic_shares(distribution: pd.DataFrame, country_code: str) -> Path:
    topic_columns = [col for col in distribution.columns if col.startswith("topic_")]
    parties = sorted(distribution["party"].unique())
    n_parties = len(parties)
    ncols = 2
    nrows = int(np.ceil(n_parties / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(16, max(4 * nrows, 6)),
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab20(np.linspace(0, 1, len(topic_columns)))

    for ax, party in zip(axes_array, parties, strict=False):
        party_df = distribution[distribution["party"] == party].sort_values("month_start")
        x_positions = np.arange(len(party_df))
        month_labels = party_df["month"].to_numpy()
        bottom = np.zeros(len(party_df))

        for color, topic in zip(colors, topic_columns, strict=False):
            values = party_df[topic].to_numpy()
            ax.bar(x_positions, values, bottom=bottom, label=topic, color=color, width=0.9)
            bottom += values

        ax.set_title(party)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic share")
        tick_step = max(1, len(month_labels) // 12)
        tick_positions = x_positions[::tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(month_labels[::tick_step], rotation=45, ha="right")

    for ax in axes_array[n_parties:]:
        ax.remove()

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles, topic_columns, loc="upper center", ncol=min(5, len(topic_columns)))
    fig.suptitle(f"{country_code}: topic shares by party over time", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = OUTPUT_DIR / f"{country_code}_party_topic_shares.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_outputs(
    distribution: pd.DataFrame,
    keywords: pd.DataFrame,
    country_code: str,
) -> tuple[Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    distribution_path = OUTPUT_DIR / f"{country_code}_party_month_topic_distribution.csv"
    keywords_path = OUTPUT_DIR / f"{country_code}_topic_keywords.csv"

    distribution.to_csv(distribution_path, index=False)
    keywords.to_csv(keywords_path, index=False)
    heatmap_path = plot_topic_heatmap(distribution, country_code)
    party_plot_path = plot_party_topic_shares(distribution, country_code)

    return distribution_path, keywords_path, heatmap_path, party_plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate topic distributions per month and party from ParlaMint speeches.",
    )
    parser.add_argument("--country", default="CZ", help="ParlaMint country code, e.g. CZ.")
    parser.add_argument("--topics", type=int, default=15, help="Number of LDA topics.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of count-vectorizer features.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=10,
        help="Number of top words to export for each topic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    country_code = args.country.upper()

    speeches = load_country(country_code)
    corpus = build_party_month_corpus(speeches)
    distribution, keywords = fit_topic_model(
        corpus=corpus,
        n_topics=args.topics,
        max_features=args.max_features,
        top_n_words=args.top_words,
    )
    distribution_path, keywords_path, heatmap_path, party_plot_path = save_outputs(
        distribution,
        keywords,
        country_code,
    )

    print(f"Loaded {len(speeches):,} speeches for {country_code}.")
    print(f"Built {len(corpus):,} party-month documents.")
    print("Model: LatentDirichletAllocation")
    print(f"Saved topic distributions to: {distribution_path}")
    print(f"Saved topic keywords to: {keywords_path}")
    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved party plots to: {party_plot_path}")
    print()
    print(distribution.head().to_string(index=False))
    print()
    print(keywords.to_string(index=False))


if __name__ == "__main__":
    main()
