from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_INPUT_DIR = TEST_OUTPUT_DIR / "plda_alignment"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_alignment_timeseries"
MPL_CONFIG_DIR = SCRIPT_DIR / ".mplconfig"
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DEFAULT_CZ_ELECTION_DATES = ["2013-10-25", "2017-10-20", "2021-10-08"]
COMPARISON_METRICS = ["js_distance", "cosine_similarity", "hellinger_distance"]
METRIC_LABELS = {
    "alignment_score": "Alignment score (1 - Jensen-Shannon distance)",
    "js_distance": "Jensen-Shannon distance",
    "cosine_similarity": "Cosine similarity",
    "hellinger_distance": "Hellinger distance",
}
METRIC_COLORS = {
    "alignment_score": "#1f77b4",
    "js_distance": "#1f77b4",
    "cosine_similarity": "#2ca02c",
    "hellinger_distance": "#d62728",
}


parser = argparse.ArgumentParser(
    description="Plot PLDA manifesto-alignment time series by party."
)
parser.add_argument("--c", default="CZ", type=str, help="Country abbreviation.")
parser.add_argument(
    "--alignment-input",
    default=None,
    type=Path,
    help=(
        "PLDA manifesto-alignment CSV. If omitted, uses "
        "outputs/test_speeches/plda_alignment/<COUNTRY>/"
        "<COUNTRY>_plda_manifesto_alignment.csv."
    ),
)
parser.add_argument(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    type=Path,
    help="Directory where alignment time-series plots are written.",
)
parser.add_argument(
    "--score-column",
    default="alignment_score",
    choices=[
        "alignment_score",
        "js_distance",
        "cosine_similarity",
        "hellinger_distance",
        "all",
    ],
    help=(
        "Metric to plot. Use all to plot Jensen-Shannon distance, cosine "
        "similarity, and Hellinger distance together in one plot set."
    ),
)
parser.add_argument(
    "--vertical-lines",
    "--election-dates",
    dest="vertical_lines",
    nargs="*",
    default=None,
    help=(
        "Dates to draw as vertical reference lines. Use YYYY-MM-DD. If omitted, "
        "CZ defaults to 2013-10-25, 2017-10-20, and 2021-10-08. Pass the flag "
        "with no dates to draw no lines."
    ),
)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def default_alignment_input(country_code: str) -> Path:
    return (
        DEFAULT_INPUT_DIR
        / country_code
        / f"{country_code}_plda_manifesto_alignment.csv"
    )


def resolve_vertical_lines(country_code: str, raw_dates: list[str] | None) -> list[pd.Timestamp]:
    if raw_dates is None:
        raw_dates = DEFAULT_CZ_ELECTION_DATES if country_code == "CZ" else []

    dates = pd.to_datetime(raw_dates, errors="coerce")
    if dates.isna().any():
        bad_dates = [
            raw_date
            for raw_date, parsed_date in zip(raw_dates, dates, strict=False)
            if pd.isna(parsed_date)
        ]
        raise ValueError(f"Could not parse vertical line date(s): {bad_dates}")
    return [pd.Timestamp(date) for date in dates]


def load_alignment_data(path: Path, score_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find PLDA alignment CSV: {path}")

    data = pd.read_csv(path, low_memory=False)
    required_columns = {"speech_party", "month", *score_columns}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Alignment CSV is missing required columns: {missing_text}")

    data = data.copy()
    if "month_start" in data.columns:
        data["plot_date"] = pd.to_datetime(data["month_start"], errors="coerce")
    else:
        data["plot_date"] = pd.to_datetime(data["month"], errors="coerce")

    for score_column in score_columns:
        data[score_column] = pd.to_numeric(data[score_column], errors="coerce")

    data = data.dropna(subset=["speech_party", "plot_date", *score_columns]).copy()
    if data.empty:
        raise ValueError(f"No plottable alignment rows found in {path}")

    data["speech_party"] = data["speech_party"].astype(str)
    return data.sort_values(["speech_party", "plot_date"]).reset_index(drop=True)


def y_axis_label(score_column: str) -> str:
    if score_column == "all":
        return "Metric value"
    if score_column in METRIC_LABELS:
        return METRIC_LABELS[score_column]
    raise ValueError(f"Unsupported score column: {score_column}")


def metric_filename_token(score_column: str) -> str:
    if score_column == "alignment_score":
        return "alignment"
    if score_column == "all":
        return "all_metrics"
    return score_column


def configure_time_axis(
    ax: plt.Axes,
    dates: pd.Series,
    vertical_lines: list[pd.Timestamp],
) -> None:
    axis_dates = list(dates.dropna()) + vertical_lines
    min_date = min(axis_dates) if axis_dates else pd.NaT
    max_date = max(axis_dates) if axis_dates else pd.NaT
    if pd.notna(min_date) and pd.notna(max_date):
        padding = pd.Timedelta(days=30)
        ax.set_xlim(min_date - padding, max_date + padding)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.tick_params(axis="x", rotation=45)


def add_vertical_lines(ax: plt.Axes, vertical_lines: list[pd.Timestamp]) -> None:
    for line_date in vertical_lines:
        ax.axvline(
            line_date,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.55,
        )


def plot_party_alignment(
    ax: plt.Axes,
    party_df: pd.DataFrame,
    score_columns: list[str],
    vertical_lines: list[pd.Timestamp],
    show_legend: bool,
) -> None:
    for score_column in score_columns:
        ax.plot(
            party_df["plot_date"],
            party_df[score_column],
            color=METRIC_COLORS[score_column],
            label=METRIC_LABELS[score_column],
            marker="o",
            markersize=3,
            linewidth=1.8,
        )
    add_vertical_lines(ax, vertical_lines)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Month")
    ax.set_ylabel(
        y_axis_label("all") if len(score_columns) > 1 else y_axis_label(score_columns[0])
    )
    ax.grid(True, axis="y", alpha=0.25)
    configure_time_axis(ax, party_df["plot_date"], vertical_lines)
    if show_legend and len(score_columns) > 1:
        ax.legend(loc="best", fontsize=8)


def save_alignment_timeseries_plots(
    alignment_df: pd.DataFrame,
    country_code: str,
    score_columns: list[str],
    vertical_lines: list[pd.Timestamp],
    output_dir: Path,
) -> list[Path]:
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)

    parties = sorted(alignment_df["speech_party"].dropna().unique())
    if not parties:
        raise ValueError("No parties found in alignment data.")

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
        party_df = alignment_df[alignment_df["speech_party"] == party]
        plot_party_alignment(
            ax,
            party_df,
            score_columns,
            vertical_lines,
            show_legend=False,
        )
        ax.set_title(str(party))

    for ax in axes_array[len(parties):]:
        ax.remove()

    score_column = "all" if len(score_columns) > 1 else score_columns[0]
    title_metric = (
        "Jensen-Shannon distance, cosine similarity, and Hellinger distance"
        if score_column == "all"
        else y_axis_label(score_column)
    )
    fig.suptitle(f"{country_code} PLDA manifesto alignment by party over time: {title_metric}", y=0.995)
    if len(score_columns) > 1:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=METRIC_COLORS[score_column],
                marker="o",
                linewidth=1.8,
                markersize=3,
            )
            for score_column in score_columns
        ]
        labels = [METRIC_LABELS[score_column] for score_column in score_columns]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(score_columns),
            bbox_to_anchor=(0.5, 0.975),
        )
    top_margin = 0.93 if len(score_columns) > 1 else 0.97
    fig.tight_layout(rect=(0, 0, 1, top_margin))

    metric_token = metric_filename_token(score_column)
    master_plot_path = country_dir / (
        f"{country_code}_all_parties_plda_{metric_token}_timeseries.png"
    )
    fig.savefig(master_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(master_plot_path)

    for party in parties:
        party_df = alignment_df[alignment_df["speech_party"] == party]
        fig, ax = plt.subplots(figsize=(16, 7))
        plot_party_alignment(
            ax,
            party_df,
            score_columns,
            vertical_lines,
            show_legend=True,
        )
        ax.set_title(f"{country_code} {party} PLDA manifesto alignment over time: {title_metric}")
        fig.tight_layout()

        plot_path = country_dir / (
            f"{country_code}_{sanitize_filename(party)}_plda_{metric_token}_timeseries.png"
        )
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


def main(args: argparse.Namespace) -> None:
    if not args.c.strip():
        raise ValueError("--c must not be empty.")

    country_code = args.c.strip().upper()
    alignment_input = args.alignment_input or default_alignment_input(country_code)
    vertical_lines = resolve_vertical_lines(country_code, args.vertical_lines)
    score_columns = COMPARISON_METRICS if args.score_column == "all" else [args.score_column]
    alignment_df = load_alignment_data(alignment_input, score_columns)
    saved_paths = save_alignment_timeseries_plots(
        alignment_df=alignment_df,
        country_code=country_code,
        score_columns=score_columns,
        vertical_lines=vertical_lines,
        output_dir=args.output_dir,
    )

    print(f"Loaded {len(alignment_df):,} alignment rows for {country_code}.")
    print(f"Plotted {alignment_df['speech_party'].nunique():,} parties.")
    print(f"Metrics plotted: {', '.join(score_columns)}")
    print(f"Saved PLDA alignment time-series plots to: {args.output_dir / country_code}")
    print(saved_paths[0])


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
