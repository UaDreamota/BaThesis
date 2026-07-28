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
DEFAULT_TOPIC_INPUT = TEST_OUTPUT_DIR / "plda_distribution.csv"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_topic_distributions"
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import load_country, merge_topics


WEIGHTING_COLUMN_BY_NAME = {
    "word_count": "speech_word_count",
    "log_word_count": "speech_log_word_count",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build length-weighted PLDA party-month speech topic distributions. "
            "This keeps the original unweighted topic distribution intact."
        )
    )
    parser.add_argument("--country", "--c", dest="country", default="CZ", type=str)
    parser.add_argument(
        "--topic-input",
        type=Path,
        default=DEFAULT_TOPIC_INPUT,
        help="CSV with one PLDA topic distribution row per modeled speech.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory. Files are written under <output-dir>/<COUNTRY>/.",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=sorted(WEIGHTING_COLUMN_BY_NAME),
        default=sorted(WEIGHTING_COLUMN_BY_NAME),
        help="Weighting schemes to write.",
    )
    return parser


def country_suffixed_path(path: Path, country_code: str) -> Path:
    suffix = f"_{country_code.upper()}"
    if path.stem.upper().endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def resolve_topic_input(path: Path, country_code: str) -> Path:
    suffixed = country_suffixed_path(path, country_code)
    if suffixed.exists():
        return suffixed
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find PLDA topic input: {path}")


def topic_columns(data: pd.DataFrame) -> list[str]:
    columns = [col for col in data.columns if TOPIC_COLUMN_RE.match(str(col))]
    if not columns:
        raise ValueError("No topic_* columns found.")
    return sorted(columns, key=lambda col: int(TOPIC_COLUMN_RE.match(col).group(1)))


def simple_tokenizer(raw_text: str) -> list[str]:
    raw_text = raw_text.lower()
    return re.findall(r"\b\w+\b", raw_text, flags=re.UNICODE)


def drop_empty_token_rows(data: pd.DataFrame) -> pd.DataFrame:
    token_mask = data["text"].map(lambda text: len(simple_tokenizer(str(text))) > 0)
    dropped_rows = int((~token_mask).sum())
    if dropped_rows:
        print(
            f"Dropped {dropped_rows:,} speech row(s) with no PLDA tokens "
            "after tokenization."
        )
    return data.loc[token_mask].reset_index(drop=True)


def load_row_aligned_speech_topics(
    country_code: str,
    topic_input: Path,
) -> tuple[pd.DataFrame, list[str]]:
    topic_df = pd.read_csv(topic_input, low_memory=False)
    topics = topic_columns(topic_df)
    topic_df = topic_df[topics].copy()

    metadata = load_country(country_code)
    metadata = merge_topics(metadata)
    metadata = metadata.dropna(subset=["topic_label"]).copy()
    metadata = drop_empty_token_rows(metadata)
    metadata = metadata[
        ["party", "month", "month_start", "text", "topic_label", "broad_topic"]
    ].reset_index(drop=True)

    if len(topic_df) != len(metadata):
        raise ValueError(
            "PLDA topic rows do not match row-aligned speech metadata: "
            f"{len(topic_df):,} topic rows vs {len(metadata):,} metadata rows."
        )

    metadata["speech_word_count"] = (
        metadata["text"].fillna("").astype(str).str.split().str.len()
    )
    metadata["speech_log_word_count"] = np.log1p(metadata["speech_word_count"])
    metadata.loc[metadata["speech_word_count"] <= 0, "speech_word_count"] = 1
    metadata.loc[metadata["speech_log_word_count"] <= 0, "speech_log_word_count"] = 1.0

    combined = pd.concat(
        [metadata.reset_index(drop=True), topic_df.reset_index(drop=True)],
        axis=1,
    )
    return combined, topics


def weighted_average(group: pd.DataFrame, topic_cols: list[str], weight_col: str) -> pd.Series:
    weights = group[weight_col].to_numpy(dtype=float)
    if np.any(weights < 0):
        raise ValueError(f"Negative weights found in {weight_col}.")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError(f"Zero total weight for group using {weight_col}.")
    values = group[topic_cols].to_numpy(dtype=float)
    weighted_topics = (values * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_topics, index=topic_cols)


def build_weighted_distribution(
    combined: pd.DataFrame,
    topic_cols: list[str],
    weighting: str,
) -> pd.DataFrame:
    weight_col = WEIGHTING_COLUMN_BY_NAME[weighting]
    distribution = (
        combined.groupby(["party", "month", "month_start"], group_keys=False)
        .apply(
            weighted_average,
            topic_cols=topic_cols,
            weight_col=weight_col,
            include_groups=False,
        )
        .reset_index()
    )
    weight_stats = (
        combined.groupby(["party", "month", "month_start"], as_index=False)
        .agg(
            speech_rows=("text", "size"),
            speech_word_count_sum=("speech_word_count", "sum"),
            speech_word_count_mean=("speech_word_count", "mean"),
            speech_log_word_count_sum=("speech_log_word_count", "sum"),
        )
    )
    distribution = distribution.merge(
        weight_stats,
        on=["party", "month", "month_start"],
        how="left",
    )
    distribution.insert(3, "speech_topic_weighting", weighting)
    return distribution.sort_values(["month_start", "party"]).reset_index(drop=True)


def output_path(output_dir: Path, country_code: str, weighting: str) -> Path:
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    return country_dir / f"{country_code}_plda_topic_distribution_{weighting}.csv"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    country_code = args.country.strip().upper()
    topic_input = resolve_topic_input(args.topic_input, country_code)
    combined, topics = load_row_aligned_speech_topics(country_code, topic_input)

    saved_paths = []
    for weighting in args.weightings:
        distribution = build_weighted_distribution(combined, topics, weighting)
        path = output_path(args.output_dir, country_code, weighting)
        distribution.to_csv(path, index=False)
        saved_paths.append(path)

    print(
        f"Loaded {len(combined):,} speech-topic rows and built "
        f"{len(saved_paths):,} weighted party-month distribution file(s)."
    )
    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
