from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MANIFESTO_INPUT_DIR = BASE_DIR / "outputs" / "manifesto_quasi_sentences"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_alignment"
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")
DEFAULT_SPEECH_TOPIC_WEIGHTING = "log_word_count"


parser = argparse.ArgumentParser(
    description=(
        "Join PLDA speech party-month topics to linked manifesto topics and "
        "compute Jensen-Shannon, cosine, and Hellinger alignment metrics."
    )
)
parser.add_argument("--c", default="CZ", type=str, help="Country abbreviation.")
parser.add_argument(
    "--speech-topic-input",
    default=None,
    type=Path,
    help=(
        "Speech party-month PLDA topic CSV. If omitted, uses "
        "outputs/test_speeches/plda_topic_distributions/<COUNTRY>/"
        "<COUNTRY>_plda_topic_distribution_log_word_count.csv."
    ),
)
parser.add_argument(
    "--manifesto-topic-input",
    default=None,
    type=Path,
    help=(
        "Manifesto-document PLDA topic CSV from plda_inference.py. If omitted, uses "
        "outputs/test_speeches/plda_manifesto_inference/<COUNTRY>/"
        "<COUNTRY>_plda_manifesto_topic_distribution.csv."
    ),
)
parser.add_argument(
    "--bridge-input",
    default=None,
    type=Path,
    help=(
        "Speech month to manifesto bridge CSV. If omitted, uses "
        "outputs/manifesto_quasi_sentences/<COUNTRY>/"
        "<COUNTRY>_speech_month_to_manifesto_bridge.csv."
    ),
)
parser.add_argument(
    "--output",
    default=None,
    type=Path,
    help=(
        "Alignment CSV output. If omitted, writes to "
        "outputs/test_speeches/plda_alignment/<COUNTRY>/."
    ),
)
parser.add_argument(
    "--speech-topic-weighting",
    default=DEFAULT_SPEECH_TOPIC_WEIGHTING,
    type=str,
    help="Label recorded in the output to identify the speech-topic weighting scheme.",
)


def validate_args(args: argparse.Namespace) -> None:
    if not args.c.strip():
        raise ValueError("--c must not be empty.")


def default_speech_topic_input(country_code: str) -> Path:
    return (
        TEST_OUTPUT_DIR
        / "plda_topic_distributions"
        / country_code
        / f"{country_code}_plda_topic_distribution_{DEFAULT_SPEECH_TOPIC_WEIGHTING}.csv"
    )


def default_manifesto_topic_input(country_code: str) -> Path:
    return (
        TEST_OUTPUT_DIR
        / "plda_manifesto_inference"
        / country_code
        / f"{country_code}_plda_manifesto_topic_distribution.csv"
    )


def default_bridge_input(country_code: str) -> Path:
    return (
        MANIFESTO_INPUT_DIR
        / country_code
        / f"{country_code}_speech_month_to_manifesto_bridge.csv"
    )


def default_output(country_code: str) -> Path:
    return (
        DEFAULT_OUTPUT_DIR
        / country_code
        / f"{country_code}_plda_manifesto_alignment.csv"
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    country_code = args.c.strip().upper()
    speech_topic_input = args.speech_topic_input or default_speech_topic_input(country_code)
    manifesto_topic_input = (
        args.manifesto_topic_input or default_manifesto_topic_input(country_code)
    )
    bridge_input = args.bridge_input or default_bridge_input(country_code)
    output = args.output or default_output(country_code)
    return speech_topic_input, manifesto_topic_input, bridge_input, output


def topic_columns(data: pd.DataFrame) -> list[str]:
    columns = [
        col
        for col in data.columns
        if TOPIC_COLUMN_RE.match(str(col))
    ]
    if not columns:
        raise ValueError("No topic_* columns found.")
    return sorted(columns, key=lambda col: int(TOPIC_COLUMN_RE.match(col).group(1)))


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find input CSV: {path}. "
            "If this is a weighted speech-topic file, run "
            "scripts/metrics/plda_weighted_speech_topics.py first."
        )

    data = pd.read_csv(path, low_memory=False)
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing_text}")
    return data


def normalize_distribution(values: pd.Series) -> np.ndarray:
    vector = values.to_numpy(dtype=float)
    vector = np.clip(vector, 0.0, None)
    total = vector.sum()
    if total <= 0:
        raise ValueError("Cannot compare a topic vector with zero total mass.")
    return vector / total


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")
    return float(np.dot(left, right) / denominator)


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.sqrt(left) - np.sqrt(right)) / np.sqrt(2.0))


def select_one_bridge_row_per_party_month(bridge_df: pd.DataFrame) -> pd.DataFrame:
    if bridge_df.empty:
        return bridge_df

    dedupe_columns = ["speech_party", "month"]
    duplicate_mask = bridge_df.duplicated(dedupe_columns, keep=False)
    if not duplicate_mask.any():
        return bridge_df

    sortable = bridge_df.copy()
    if "speech_rows" in sortable.columns:
        sortable["speech_rows"] = pd.to_numeric(
            sortable["speech_rows"],
            errors="coerce",
        ).fillna(0)
    else:
        sortable["speech_rows"] = 0
    if "speech_dates" in sortable.columns:
        sortable["speech_dates"] = pd.to_numeric(
            sortable["speech_dates"],
            errors="coerce",
        ).fillna(0)
    else:
        sortable["speech_dates"] = 0
    sortable["_manifesto_effective_date_sort"] = pd.to_datetime(
        sortable["manifesto_effective_date"]
        if "manifesto_effective_date" in sortable.columns
        else pd.Series(pd.NaT, index=sortable.index),
        errors="coerce",
    )
    sortable = sortable.sort_values(
        [
            "speech_party",
            "month",
            "speech_rows",
            "speech_dates",
            "_manifesto_effective_date_sort",
            "doc_key",
        ],
        ascending=[True, True, False, False, False, True],
    )
    deduped = (
        sortable.drop_duplicates(dedupe_columns, keep="first")
        .drop(columns=["_manifesto_effective_date_sort"])
        .reset_index(drop=True)
    )
    dropped_rows = len(bridge_df) - len(deduped)
    print(
        f"Dropped {dropped_rows:,} duplicate bridge row(s) after selecting "
        "one manifesto per speech-party month."
    )
    return deduped


def build_alignment_df(
    speech_topic_df: pd.DataFrame,
    manifesto_topic_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    speech_topic_weighting: str = "unweighted",
) -> pd.DataFrame:
    speech_topic_cols = topic_columns(speech_topic_df)
    manifesto_topic_cols = topic_columns(manifesto_topic_df)
    if speech_topic_cols != manifesto_topic_cols:
        raise ValueError(
            "Speech and manifesto topic columns do not match exactly. "
            f"speech={speech_topic_cols}, manifesto={manifesto_topic_cols}"
        )

    bridge_cols = [
        col
        for col in [
            "speech_party",
            "month",
            "month_start",
            "doc_key",
            "mpds_party_id",
            "mpds_partyname",
            "mpds_partyabbrev",
            "manifesto_partyname",
            "manifesto_partyabbrev",
            "manifesto_date",
            "manifesto_effective_date",
            "selection_method",
            "speech_rows",
            "speech_dates",
            "speech_start_date",
            "speech_end_date",
        ]
        if col in bridge_df.columns
    ]

    bridge_df = select_one_bridge_row_per_party_month(bridge_df)
    speech_cols = ["party", "month"] + speech_topic_cols
    manifesto_cols = ["doc_key"] + manifesto_topic_cols

    merged = (
        bridge_df[bridge_cols]
        .merge(
            speech_topic_df[speech_cols],
            left_on=["speech_party", "month"],
            right_on=["party", "month"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            manifesto_topic_df[manifesto_cols],
            on="doc_key",
            how="inner",
            suffixes=("_speech", "_manifesto"),
            validate="many_to_one",
        )
        .drop(columns=["party"])
    )

    if merged.empty:
        raise RuntimeError(
            "No rows remained after joining bridge, speech topics, and manifesto topics."
        )

    speech_cols_suffixed = [f"{col}_speech" for col in speech_topic_cols]
    manifesto_cols_suffixed = [f"{col}_manifesto" for col in manifesto_topic_cols]

    js_distances = []
    cosine_similarities = []
    hellinger_distances = []
    for _, row in merged.iterrows():
        speech_vector = normalize_distribution(row[speech_cols_suffixed])
        manifesto_vector = normalize_distribution(row[manifesto_cols_suffixed])
        js_distances.append(
            float(jensenshannon(speech_vector, manifesto_vector, base=2.0))
        )
        cosine_similarities.append(cosine_similarity(speech_vector, manifesto_vector))
        hellinger_distances.append(hellinger_distance(speech_vector, manifesto_vector))

    metrics_df = pd.DataFrame(
        {
            "js_distance": js_distances,
            "alignment_score": [1.0 - value for value in js_distances],
            "cosine_similarity": cosine_similarities,
            "hellinger_distance": hellinger_distances,
        },
        index=merged.index,
    )
    merged = pd.concat([merged.copy(), metrics_df], axis=1)
    merged["speech_topic_weighting"] = speech_topic_weighting

    leading_cols = [
        col
        for col in [
            "speech_party",
            "month",
            "month_start",
            "doc_key",
            "manifesto_date",
            "manifesto_effective_date",
            "js_distance",
            "alignment_score",
            "cosine_similarity",
            "hellinger_distance",
            "speech_topic_weighting",
        ]
        if col in merged.columns
    ]
    remaining_cols = [col for col in merged.columns if col not in leading_cols]
    return merged[leading_cols + remaining_cols].sort_values(
        ["speech_party", "month"]
    ).reset_index(drop=True)


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    country_code = args.c.strip().upper()
    speech_topic_input, manifesto_topic_input, bridge_input, output = resolve_paths(args)

    speech_topic_df = load_csv(speech_topic_input, {"party", "month"})
    manifesto_topic_df = load_csv(manifesto_topic_input, {"doc_key"})
    bridge_df = load_csv(bridge_input, {"speech_party", "month", "doc_key"})

    alignment_df = build_alignment_df(
        speech_topic_df=speech_topic_df,
        manifesto_topic_df=manifesto_topic_df,
        bridge_df=bridge_df,
        speech_topic_weighting=args.speech_topic_weighting,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    alignment_df.to_csv(output, index=False)
    print(
        f"Built {len(alignment_df):,} PLDA manifesto-alignment rows for {country_code} "
        "with Jensen-Shannon, cosine similarity, and Hellinger distance."
    )
    print(f"Saved alignment CSV to: {output}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
