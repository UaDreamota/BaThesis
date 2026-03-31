from pathlib import Path
import argparse
import os
import re

import pandas as pd


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


parser = argparse.ArgumentParser()
parser.add_argument("--c", required=True, type=str, help="Country's abbreveation")
parser.add_argument(
    "--top-parties",
    default=None,
    type=int,
    help="Optional: keep only the N parties with the most speeches for plotting",
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "parlam"
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches" / "annotated_topic_distributions"


def load_country(country_code: str) -> pd.DataFrame:
    path = DATA_DIR / f"ParlaMint-{country_code}_extracted.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find ParlaMint file: {path}")

    df = pd.read_csv(
        path,
        usecols=[
            "country",
            "date",
            "party",
            "content_kind",
            "speaker_type",
            "topic_code",
            "topic_label",
        ],
    )
    df = df[df["content_kind"] == "speech"].copy()
    df = df[df["speaker_type"] == "regular"].copy()
    df = df.dropna(subset=["party", "date", "topic_label"]).copy()

    df["party"] = df["party"].astype(str).str.strip()
    df["topic_label"] = df["topic_label"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["party"] != ""].copy()
    df = df[df["topic_label"] != ""].copy()
    df = df.dropna(subset=["date"]).copy()

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True)


def keep_top_parties(data: pd.DataFrame, n_parties: int | None) -> pd.DataFrame:
    if n_parties is None:
        return data
    top_parties = data["party"].value_counts().head(n_parties).index
    return data[data["party"].isin(top_parties)].copy()


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def build_annotated_distribution(data: pd.DataFrame) -> pd.DataFrame:
    counts = (
        data.groupby(["party", "month", "month_start", "topic_label"], as_index=False)
        .size()
        .rename(columns={"size": "speech_count"})
    )
    counts["total_speeches"] = counts.groupby(["party", "month"])["speech_count"].transform("sum")
    counts["topic_proportion"] = counts["speech_count"] / counts["total_speeches"]

    distribution = (
        counts.pivot_table(
            index=["party", "month", "month_start"],
            columns="topic_label",
            values="topic_proportion",
            fill_value=0,
        )
        .reset_index()
        .sort_values(["month_start", "party"])
    )
    distribution.columns.name = None
    return distribution


def save_topic_distribution_plot(
    distribution_df: pd.DataFrame,
    country_code: str,
) -> list[Path]:
    country_dir = OUTPUT_DIR / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)

    topic_columns = [col for col in distribution_df.columns if col not in {"party", "month", "month_start"}]
    parties = sorted(distribution_df["party"].unique())
    colors = plt.cm.tab20(range(len(topic_columns)))
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
    fig.legend(handles, topic_columns, loc="upper center", ncol=min(4, len(topic_columns)))
    fig.suptitle(f"{country_code.upper()} annotated topic shares by party over time", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    master_plot_path = country_dir / f"{country_code.upper()}_all_parties_annotated_topic_distribution.png"
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

        ax.set_title(f"{country_code.upper()} {party} annotated topic distribution per month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic proportion")
        ax.set_ylim(0, 1)
        tick_step = max(1, len(month_labels) // 12)
        tick_positions = x_positions[::tick_step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(month_labels[::tick_step], rotation=45, ha="right")
        ax.legend(loc="upper center", ncol=min(4, len(topic_columns)))
        fig.tight_layout()

        plot_path = country_dir / f"{country_code.upper()}_{sanitize_filename(party)}_annotated_topic_distribution.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    csv_path = country_dir / f"{country_code.upper()}_annotated_topic_distribution.csv"
    distribution_df.to_csv(csv_path, index=False)
    saved_paths.append(csv_path)

    return saved_paths


def main(args: argparse.Namespace) -> None:
    df = load_country(args.c)
    df = keep_top_parties(df, args.top_parties)
    distribution_df = build_annotated_distribution(df)
    saved_paths = save_topic_distribution_plot(distribution_df, args.c)

    print(f"Loaded {len(df):,} annotated speeches for {args.c.upper()}.")
    print(f"Built {len(distribution_df):,} party-month rows.")
    print(f"Saved annotated topic outputs to: {OUTPUT_DIR / args.c.upper()}")
    print(saved_paths[0])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
