from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tomotopy as tp
from rich import print


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MANIFESTO_INPUT_DIR = BASE_DIR / "outputs" / "manifesto_quasi_sentences"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_manifesto_inference"
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plda_test import simple_tokenizer


parser = argparse.ArgumentParser(
    description="Infer saved PLDA topic distributions for manifesto quasi-sentences."
)
parser.add_argument("--c", default="CZ", type=str, help="Country abbreviation.")
parser.add_argument(
    "--model-input",
    default=None,
    type=Path,
    help=(
        "Saved PLDA model path. If omitted, uses "
        "outputs/test_speeches/plda_model_<COUNTRY>.bin."
    ),
)
parser.add_argument(
    "--manifesto-input",
    default=None,
    type=Path,
    help=(
        "Manifesto quasi-sentence CSV. If omitted, uses "
        "outputs/manifesto_quasi_sentences/<COUNTRY>/<COUNTRY>_manifesto_quasi_sentences.csv."
    ),
)
parser.add_argument(
    "--quasi-output",
    default=None,
    type=Path,
    help=(
        "CSV for quasi-sentence topic distributions. If omitted, writes to "
        "outputs/test_speeches/plda_manifesto_inference/<COUNTRY>/."
    ),
)
parser.add_argument(
    "--document-output",
    default=None,
    type=Path,
    help=(
        "CSV for manifesto-document topic distributions. If omitted, writes to "
        "outputs/test_speeches/plda_manifesto_inference/<COUNTRY>/."
    ),
)
parser.add_argument(
    "--iterations",
    default=100,
    type=int,
    help="Tomotopy inference iterations per quasi-sentence.",
)
parser.add_argument(
    "--workers",
    default=0,
    type=int,
    help="Workers passed to tomotopy infer().",
)
parser.add_argument(
    "--text-column",
    default="text",
    type=str,
    help="Column containing quasi-sentence text.",
)


def validate_args(args: argparse.Namespace) -> None:
    if not args.c.strip():
        raise ValueError("--c must not be empty.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0.")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0.")


def default_model_input(country_code: str) -> Path:
    return TEST_OUTPUT_DIR / f"plda_model_{country_code}.bin"


def default_manifesto_input(country_code: str) -> Path:
    return (
        MANIFESTO_INPUT_DIR
        / country_code
        / f"{country_code}_manifesto_quasi_sentences.csv"
    )


def default_quasi_output(country_code: str) -> Path:
    return (
        DEFAULT_OUTPUT_DIR
        / country_code
        / f"{country_code}_plda_manifesto_quasi_sentence_topics.csv"
    )


def default_document_output(country_code: str) -> Path:
    return (
        DEFAULT_OUTPUT_DIR
        / country_code
        / f"{country_code}_plda_manifesto_topic_distribution.csv"
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    country_code = args.c.strip().upper()
    model_input = args.model_input or default_model_input(country_code)
    manifesto_input = args.manifesto_input or default_manifesto_input(country_code)
    quasi_output = args.quasi_output or default_quasi_output(country_code)
    document_output = args.document_output or default_document_output(country_code)
    return model_input, manifesto_input, quasi_output, document_output


def load_manifesto_quasi_sentences(path: Path, text_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find manifesto quasi-sentence CSV: {path}")

    data = pd.read_csv(path, low_memory=False)
    required_columns = {"doc_key", text_column}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifesto quasi-sentence CSV is missing: {missing_text}")

    data = data.dropna(subset=["doc_key", text_column]).copy()
    data[text_column] = data[text_column].astype(str).str.strip()
    data = data[data[text_column] != ""].copy()
    if data.empty:
        raise ValueError(f"No non-empty manifesto quasi-sentences found in {path}")

    return data.reset_index(drop=True)


def topic_columns(model: tp.PLDAModel) -> list[str]:
    return [f"topic_{topic_idx}" for topic_idx in range(model.k)]


def infer_quasi_sentence_topics(
    model: tp.PLDAModel,
    quasi_df: pd.DataFrame,
    text_column: str,
    iterations: int,
    workers: int,
) -> pd.DataFrame:
    topic_cols = topic_columns(model)
    topic_rows: list[np.ndarray] = []
    lls: list[float] = []

    for text in quasi_df[text_column]:
        doc = model.make_doc(simple_tokenizer(text), labels=[])
        topic_dist, ll = model.infer(doc, iterations=iterations, workers=workers)
        topic_rows.append(np.asarray(topic_dist, dtype=float))
        if isinstance(ll, (list, tuple, np.ndarray)):
            lls.append(float(ll[0]))
        else:
            lls.append(float(ll))

    topic_df = pd.DataFrame(topic_rows, columns=topic_cols)
    quasi_topic_df = pd.concat(
        [quasi_df.reset_index(drop=True), topic_df],
        axis=1,
    )
    quasi_topic_df["predicted_topic"] = topic_df.idxmax(axis=1)
    quasi_topic_df["infer_ll"] = lls
    return quasi_topic_df


def build_manifesto_document_topics(quasi_topic_df: pd.DataFrame) -> pd.DataFrame:
    topic_cols = [
        col
        for col in quasi_topic_df.columns
        if TOPIC_COLUMN_RE.match(str(col))
    ]
    if not topic_cols:
        raise ValueError("No topic_* columns found in quasi-sentence inference output.")
    topic_cols = sorted(
        topic_cols,
        key=lambda col: int(TOPIC_COLUMN_RE.match(str(col)).group(1)),
    )

    doc_topic_df = (
        quasi_topic_df.groupby("doc_key", as_index=False)[topic_cols]
        .mean()
        .reset_index(drop=True)
    )
    doc_topic_df["predicted_topic"] = doc_topic_df[topic_cols].idxmax(axis=1)

    metadata_candidates = [
        "manifesto_id",
        "mpds_party_id",
        "manifesto_date",
    ]
    metadata_cols = [
        col for col in metadata_candidates if col in quasi_topic_df.columns
    ]
    if metadata_cols:
        metadata = (
            quasi_topic_df.groupby("doc_key", as_index=False)[metadata_cols]
            .first()
            .reset_index(drop=True)
        )
        doc_topic_df = metadata.merge(doc_topic_df, on="doc_key", how="right")

    quasi_counts = (
        quasi_topic_df.groupby("doc_key", as_index=False)
        .size()
        .rename(columns={"size": "quasi_sentence_count"})
    )
    doc_topic_df = doc_topic_df.merge(quasi_counts, on="doc_key", how="left")

    leading_cols = [
        col
        for col in [
            "doc_key",
            "manifesto_id",
            "mpds_party_id",
            "manifesto_date",
            "quasi_sentence_count",
            "predicted_topic",
        ]
        if col in doc_topic_df.columns
    ]
    remaining_cols = [col for col in doc_topic_df.columns if col not in leading_cols]
    return doc_topic_df[leading_cols + remaining_cols]


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    country_code = args.c.strip().upper()
    model_input, manifesto_input, quasi_output, document_output = resolve_paths(args)

    if not model_input.exists():
        raise FileNotFoundError(f"Could not find saved PLDA model: {model_input}")

    print(f"Loading PLDA model from: {model_input}")
    model = tp.PLDAModel.load(str(model_input))
    quasi_df = load_manifesto_quasi_sentences(manifesto_input, args.text_column)
    print(f"Loaded {len(quasi_df):,} manifesto quasi-sentences for {country_code}.")

    quasi_topic_df = infer_quasi_sentence_topics(
        model=model,
        quasi_df=quasi_df,
        text_column=args.text_column,
        iterations=args.iterations,
        workers=args.workers,
    )
    document_topic_df = build_manifesto_document_topics(quasi_topic_df)

    quasi_output.parent.mkdir(parents=True, exist_ok=True)
    document_output.parent.mkdir(parents=True, exist_ok=True)
    quasi_topic_df.to_csv(quasi_output, index=False)
    document_topic_df.to_csv(document_output, index=False)

    print(f"Inferred {model.k} PLDA topics for manifesto quasi-sentences.")
    print(f"Saved quasi-sentence topic distributions to: {quasi_output}")
    print(f"Saved manifesto-document topic distributions to: {document_output}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
