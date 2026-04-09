from pathlib import Path
import argparse
import os
import re
import sys
import pandas as pd 
import numpy as np

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

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from path_config import get_parlam_csv_path
from ml_algos.scores_lda.held_out_perplexity import (
    PerplexitySummary,
    compute_perplexity_profile,
)
from ml_algos.scores_lda.coherence_scores import (
    CoherenceSummary,
    compute_coherence_profile,
)


parser = argparse.ArgumentParser()

parser.add_argument("--c", default=None, type=str, help="Country's abbreveation")
parser.add_argument("--top-parties", default=3, type=int, help="Number of biggest parties to keep")
parser.add_argument(
    "--n-runs",
    default=0,
    type=int,
    help="If > 0, compute held-out perplexity over N random train/test runs; also used for coherence runs when enabled.",
)
parser.add_argument(
    "--coherence",
    action="store_true",
    help="Compute average topic NPMI and c_v using the shared --n-runs value.",
)
parser.add_argument(
    "--coherence-top-n",
    default=10,
    type=int,
    help="Number of top words per topic to use for NPMI and c_v coherence.",
)
parser.add_argument(
    "--doc-topic-prior",
    default=None,
    type=float,
    help="LDA doc_topic_prior (alpha). If omitted, sklearn uses its default.",
)
parser.add_argument(
    "--topic-word-prior",
    default=None,
    type=float,
    help="LDA topic_word_prior (eta). If omitted, sklearn uses its default.",
)

BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
PERPLEXITY_DIR = OUTPUT_DIR / "perplexities"
COHERENCE_DIR = OUTPUT_DIR / "coherences"
TOPIC_DISTRIBUTION_DIR = OUTPUT_DIR / "topic_distributions"
STOPWORDS_DIR = Path(__file__).resolve().parent / "stopwords"
DEFAULT_RANDOM_SEED = 42
HELD_OUT_TEST_SIZE = 0.2
TOPIC_COUNTS = [44]


def load_stopwords(country_code: str) -> list[str]:
    path = STOPWORDS_DIR / f"{country_code.upper()}.txt"
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_country(country_code: str) -> pd.DataFrame:
    path = get_parlam_csv_path(country_code)
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


def build_vectorizer(country_code: str) -> CountVectorizer:
    return CountVectorizer(stop_words=load_stopwords(country_code))


def transform_data(data: pd.DataFrame, country_code: str):
    vectorizer = build_vectorizer(country_code)
    X = vectorizer.fit_transform(data["text"])
    return X


def build_lda_model(
    n_topics: int,
    random_seed: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_seed,
        learning_method="batch",
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )


def build_topic_distribution(
    X,
    n_topics: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> pd.DataFrame:
    lda = build_lda_model(
        n_topics,
        DEFAULT_RANDOM_SEED,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
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
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> pd.DataFrame:
    topic_distribution = build_topic_distribution(
        X,
        best_n_topics,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )
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
    n_runs: int = 0,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[str, dict[int, PerplexitySummary]]:
    profiles: dict[str, dict[int, PerplexitySummary]] = {}
    stop_words = load_stopwords(country_code)

    def evaluate_profile(aggregated_df: pd.DataFrame) -> dict[int, PerplexitySummary]:
        return compute_perplexity_profile(
            texts=aggregated_df["text"].tolist(),
            stop_words=stop_words,
            topic_counts=TOPIC_COUNTS,
            held_out_runs=n_runs,
            base_random_seed=DEFAULT_RANDOM_SEED,
            test_size=HELD_OUT_TEST_SIZE,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )

    all_aggregated = aggregate_party_month(speeches_df)
    profiles["all_parties"] = evaluate_profile(all_aggregated)

    top_one_aggregated = aggregate_party_month(keep_top_parties(speeches_df, n_parties=1))
    profiles["top_1_party"] = evaluate_profile(top_one_aggregated)

    top_n_aggregated = aggregate_party_month(
        keep_top_parties(speeches_df, n_parties=top_party_count),
    )
    profiles[f"top_{top_party_count}_parties"] = evaluate_profile(top_n_aggregated)

    return profiles


def build_coherence_profiles(
    speeches_df: pd.DataFrame,
    country_code: str,
    top_party_count: int,
    n_runs: int,
    coherence_top_n: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[str, dict[int, CoherenceSummary]]:
    profiles: dict[str, dict[int, CoherenceSummary]] = {}
    stop_words = load_stopwords(country_code)

    def evaluate_profile(aggregated_df: pd.DataFrame) -> dict[int, CoherenceSummary]:
        return compute_coherence_profile(
            texts=aggregated_df["text"].tolist(),
            stop_words=stop_words,
            topic_counts=TOPIC_COUNTS,
            n_runs=n_runs,
            base_random_seed=DEFAULT_RANDOM_SEED,
            top_n_words=coherence_top_n,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )

    all_aggregated = aggregate_party_month(speeches_df)
    profiles["all_parties"] = evaluate_profile(all_aggregated)

    top_one_aggregated = aggregate_party_month(keep_top_parties(speeches_df, n_parties=1))
    profiles["top_1_party"] = evaluate_profile(top_one_aggregated)

    top_n_aggregated = aggregate_party_month(
        keep_top_parties(speeches_df, n_parties=top_party_count),
    )
    profiles[f"top_{top_party_count}_parties"] = evaluate_profile(top_n_aggregated)

    return profiles


def choose_topic_count(results: dict[str, dict[int, PerplexitySummary]], top_party_count: int) -> int:
    selection_key = f"top_{top_party_count}_parties"
    return min(results[selection_key], key=lambda n_topics: results[selection_key][n_topics].mean)


def save_perplexity_plot(
    results: dict[str, dict[int, PerplexitySummary]],
    country_code: str,
    selected_topics: int,
    held_out_runs: int,
) -> Path:
    country_dir = PERPLEXITY_DIR / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    label_map = {
        "all_parties": "All parties",
        "top_1_party": "Top 1 party",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, profile in results.items():
        topics = sorted(profile.keys())
        mean_perplexities = [profile[n_topics].mean for n_topics in topics]
        std_perplexities = [profile[n_topics].std for n_topics in topics]
        label = label_map.get(key, key.replace("_", " "))
        if held_out_runs > 0:
            line, = ax.plot(
                topics,
                mean_perplexities,
                marker="o",
                linewidth=2,
                label=label,
            )
            color = line.get_color()
            lower = [mean - std for mean, std in zip(mean_perplexities, std_perplexities, strict=False)]
            upper = [mean + std for mean, std in zip(mean_perplexities, std_perplexities, strict=False)]
            ax.fill_between(topics, lower, upper, color=color, alpha=0.12)
            ax.errorbar(
                topics,
                mean_perplexities,
                yerr=std_perplexities,
                fmt="none",
                ecolor=color,
                elinewidth=1.25,
                capsize=4,
                alpha=0.9,
            )
        else:
            ax.plot(topics, mean_perplexities, marker="o", linewidth=2, label=label)

    ax.axvline(selected_topics, color="black", linestyle="--", linewidth=1, label=f"Selected topics: {selected_topics}")
    metric_name = "Held-out perplexity" if held_out_runs > 0 else "Perplexity"
    ax.set_title(f"{country_code.upper()} {metric_name.lower()} by topic count")
    ax.set_xlabel("Number of topics")
    ax.set_ylabel(f"{metric_name} (mean +/- SD)" if held_out_runs > 0 else metric_name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = country_dir / f"{country_code.upper()}_perplexity.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def format_perplexity_results(
    results: dict[str, dict[int, PerplexitySummary]],
) -> dict[str, dict[int, dict[str, float | int]]]:
    return {
        profile_name: {
            n_topics: {
                "mean": round(summary.mean, 4),
                "std": round(summary.std, 4),
                "runs": summary.runs,
            }
            for n_topics, summary in profile.items()
        }
        for profile_name, profile in results.items()
    }


def format_coherence_results(
    results: dict[str, dict[int, CoherenceSummary]],
) -> dict[str, dict[int, dict[str, float | int]]]:
    rows = build_coherence_rows(results, round_digits=4)
    formatted: dict[str, dict[int, dict[str, float | int]]] = {}
    for row in rows:
        profile_rows = formatted.setdefault(str(row["profile"]), {})
        profile_rows[int(row["n_topics"])] = {
            "npmi_mean": float(row["npmi_mean"]),
            "npmi_std": float(row["npmi_std"]),
            "c_v_mean": float(row["c_v_mean"]),
            "c_v_std": float(row["c_v_std"]),
            "runs": int(row["runs"]),
        }
    return formatted


def build_coherence_rows(
    results: dict[str, dict[int, CoherenceSummary]],
    round_digits: int | None = None,
) -> list[dict[str, float | int | str]]:
    def maybe_round(value: float) -> float:
        return round(value, round_digits) if round_digits is not None else value

    rows: list[dict[str, float | int | str]] = []
    for profile_name, profile in results.items():
        for n_topics, summary in profile.items():
            if summary.npmi.runs != summary.c_v.runs:
                raise ValueError(
                    f"Mismatched coherence run counts for {profile_name} at {n_topics} topics: "
                    f"npmi={summary.npmi.runs}, c_v={summary.c_v.runs}"
                )

            rows.append(
                {
                    "profile": profile_name,
                    "n_topics": n_topics,
                    "npmi_mean": maybe_round(summary.npmi.mean),
                    "npmi_std": maybe_round(summary.npmi.std),
                    "c_v_mean": maybe_round(summary.c_v.mean),
                    "c_v_std": maybe_round(summary.c_v.std),
                    "runs": summary.npmi.runs,
                }
            )
    return rows


def save_coherence_results(
    results: dict[str, dict[int, CoherenceSummary]],
    country_code: str,
) -> Path:
    country_dir = COHERENCE_DIR / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    rows = build_coherence_rows(results, round_digits=4)
    csv_path = country_dir / f"{country_code.upper()}_coherence.csv"
    pd.DataFrame(rows).sort_values(["profile", "n_topics"]).to_csv(csv_path, index=False)
    return csv_path


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
    if args.n_runs < 0:
        raise ValueError("--n-runs must be >= 0.")
    if args.coherence_top_n < 2:
        raise ValueError("--coherence-top-n must be >= 2.")
    if args.coherence and args.n_runs <= 0:
        raise ValueError("--coherence requires --n-runs > 0.")
    if args.doc_topic_prior is not None and args.doc_topic_prior <= 0:
        raise ValueError("--doc-topic-prior must be > 0.")
    if args.topic_word_prior is not None and args.topic_word_prior <= 0:
        raise ValueError("--topic-word-prior must be > 0.")

    df = load_country(args.c)
    aggregated_df = aggregate_party_month(df)
    print(aggregated_df.info())
    X = transform_data(aggregated_df, args.c)
    perplexity_start = perf_counter()
    perplexity_results = build_perplexity_profiles(
        df,
        args.c,
        top_party_count=args.top_parties,
        n_runs=args.n_runs,
        doc_topic_prior=args.doc_topic_prior,
        topic_word_prior=args.topic_word_prior,
    )
    perplexity_end = perf_counter()
    best_n_topics = choose_topic_count(perplexity_results, top_party_count=args.top_parties)
    perplexity_plot_path = save_perplexity_plot(
        perplexity_results,
        args.c,
        selected_topics=best_n_topics,
        held_out_runs=args.n_runs,
    )

    coherence_results: dict[str, dict[int, CoherenceSummary]] | None = None
    coherence_csv_path: Path | None = None
    coherence_elapsed: float | None = None
    if args.coherence:
        coherence_start = perf_counter()
        coherence_results = build_coherence_profiles(
            df,
            args.c,
            top_party_count=args.top_parties,
            n_runs=args.n_runs,
            coherence_top_n=args.coherence_top_n,
            doc_topic_prior=args.doc_topic_prior,
            topic_word_prior=args.topic_word_prior,
        )
        coherence_elapsed = perf_counter() - coherence_start
        coherence_csv_path = save_coherence_results(coherence_results, args.c)

    distribution_df = prepare_topic_distribution(
        aggregated_df,
        X,
        best_n_topics,
        doc_topic_prior=args.doc_topic_prior,
        topic_word_prior=args.topic_word_prior,
    )
    save_topic_distribution_plot(distribution_df, best_n_topics, args.c)
    print(f"Time taken for perplexity evaluation: {perplexity_end - perplexity_start}")
    print(format_perplexity_results(perplexity_results))
    if args.n_runs > 0:
        print(f"Used held-out perplexity over {args.n_runs} runs with test size {HELD_OUT_TEST_SIZE}.")
    if coherence_results is not None and coherence_elapsed is not None and coherence_csv_path is not None:
        print(f"Time taken for coherence evaluation: {coherence_elapsed}")
        print(format_coherence_results(coherence_results))
        print(
            f"Used coherence metrics over {args.n_runs} runs with top {args.coherence_top_n} words per topic."
        )
        print(f"Saved coherence results to: {coherence_csv_path}")
    print(
        "LDA priors: "
        f"doc_topic_prior={args.doc_topic_prior if args.doc_topic_prior is not None else 'default'}, "
        f"topic_word_prior={args.topic_word_prior if args.topic_word_prior is not None else 'default'}"
    )
    print(f"Selected topic count from top {args.top_parties} parties: {best_n_topics}")
    print(f"Saved perplexity plot to: {perplexity_plot_path}")
    print(f"Saved topic distribution plots to: {TOPIC_DISTRIBUTION_DIR / args.c.upper()}")
    return None


if __name__ == ("__main__"):
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
