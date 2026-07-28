from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sqlite3
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from scripts.inconsistency.artifact_provenance import (
        PROVENANCE_VERSION,
        atomic_write_json,
        canonical_json_fingerprint,
        dataframe_fingerprint,
        sampled_file_content_fingerprint,
    )
except ModuleNotFoundError:
    from artifact_provenance import (  # type: ignore
        PROVENANCE_VERSION,
        atomic_write_json,
        canonical_json_fingerprint,
        dataframe_fingerprint,
        sampled_file_content_fingerprint,
    )


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MANIFESTO_INPUT_DIR = BASE_DIR / "outputs" / "manifesto_quasi_sentences"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "nli_inconsistency"
DEFAULT_EMBEDDING_CACHE_DB = TEST_OUTPUT_DIR / "embedding_cache" / "embeddings.sqlite"
DEFAULT_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
MAX_EMBEDDING_INPUT_WORDS = 3000
MAX_EMBEDDING_INPUT_CHARS = 24000
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
CHAIR_ADDRESS_RE = (
    r"(?:pane|paní)\s+"
    r"(?:předsedající|předsedo|předsedkyně|místopředsedo|místopředsedkyně)"
)

PROCEDURAL_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"(?:děkuji|dekuji|díky)"
    r"(?:\s+(?:vám|pěkně|za\s+(?:slovo|možnost\s+vystoupit)))?"
    rf"(?:,\s*{CHAIR_ADDRESS_RE})?"
    rf"|vážen[ýá]\s+{CHAIR_ADDRESS_RE}"
    r"|vážené\s+kolegyně,\s+vážení\s+kolegové"
    r")\s*[\.,:;!\-]*\s*",
    flags=re.IGNORECASE,
)
PROCEDURAL_CLOSING_RE = re.compile(
    r"\s*(?:"
    r"(?:děkuji|dekuji)(?:\s+(?:vám|všem))?"
    r"(?:\s+za\s+(?:pozornost|podporu(?:\s+mého\s+návrhu)?|odpověď))?"
    r"|prosím\s+o\s+podporu(?:\s+(?:tohoto|mého)\s+návrhu)?"
    r")\s*[\.,:;!]*\s*$",
    flags=re.IGNORECASE,
)
EN_PROCEDURAL_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"(?:thank\s+you|many\s+thanks)"
    r"(?:\s*,?\s+(?:mr|madam|mrs|ms)?\s*"
    r"(?:speaker|deputy\s+speaker|chair|chairman|chairwoman|chairperson|minister))?"
    r"|i\s+(?:thank|am\s+grateful\s+to)\s+"
    r"(?:the\s+)?(?:right\s+hon(?:ourable)?|hon(?:ourable)?)?\s*"
    r"(?:member|gentleman|lady|friend|minister)"
    r"(?:\s+for\s+(?:his|her|their|that)\s+(?:question|intervention|comments?|remarks?))?"
    r"|it\s+is\s+(?:a\s+)?(?:pleasure|privilege)\s+to\s+"
    r"(?:serve\s+under\s+your\s+chairmanship|speak\s+in\s+this\s+debate)"
    r"|i\s+beg\s+to\s+move"
    r"|i\s+rise\s+to\s+(?:speak|support|oppose)"
    r")\s*[\.,:;!\-]*\s*",
    flags=re.IGNORECASE,
)
EN_PROCEDURAL_CLOSING_RE = re.compile(
    r"\s*(?:"
    r"(?:thank\s+you|many\s+thanks)"
    r"(?:\s*,?\s+(?:mr|madam|mrs|ms)?\s*"
    r"(?:speaker|deputy\s+speaker|chair|chairman|chairwoman|chairperson|minister))?"
    r"|i\s+(?:commend|support)\s+(?:the\s+)?(?:bill|motion|amendment)\s+"
    r"to\s+the\s+house"
    r"|i\s+urge\s+(?:hon(?:ourable)?\s+)?members\s+to\s+support\s+"
    r"(?:the\s+)?(?:bill|motion|amendment)"
    r")\s*[\.,:;!]*\s*$",
    flags=re.IGNORECASE,
)
EN_SHORT_PROCEDURAL_RE = re.compile(
    r"(?i)"
    r"\b(?:thank\s+you|many\s+thanks|i\s+thank|i\s+beg\s+to\s+move|"
    r"point\s+of\s+order|will\s+the\s+(?:hon|right\s+hon)|"
    r"give\s+way|speaker|deputy\s+speaker|chair)\b"
)
NAME_ONLY_RE = re.compile(r"^\s*[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž-]+(?:\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][\wÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž-]+){1,2}\s*$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])")


@dataclass(frozen=True)
class RetrievalCheckpointArtifact:
    part_paths: tuple[Path, ...]
    selected_rows: int
    candidate_pairs_scored: int
    processed_units: int
    signature: str


def sanitize_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def nli_model_token(model_name: str) -> str:
    return sanitize_filename_token(model_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample speeches from a selected PLDA topic, compare them to linked "
            "manifesto quasi-sentences with multilingual NLI, and save pair-level "
            "classifications plus aggregate distributions."
        )
    )
    parser.add_argument("--country", "--c", dest="country", default="CZ", type=str)
    parser.add_argument(
        "--topic",
        default="Gal_Tan",
        type=str,
        help="Topic label to run, or 'all' to run every topic_label in the speech input.",
    )
    parser.add_argument(
        "--sample-size",
        default=200,
        type=int,
        help="Maximum number of speeches sampled after topic filtering.",
    )
    parser.add_argument(
        "--sample-per-party",
        default=None,
        type=int,
        help="Optional maximum number of speeches sampled per party.",
    )
    parser.add_argument(
        "--pairs-per-speech",
        default=3,
        type=int,
        help=(
            "Maximum manifesto quasi-sentence comparisons per sampled speech. "
            "Use 0 to keep all candidate pairs after filtering/retrieval."
        ),
    )
    parser.add_argument(
        "--speech-unit",
        choices=["speech", "segment"],
        default="speech",
        help=(
            "Use whole speeches for retrieval/NLI, or split sampled speeches into "
            "overlapping sentence-window segments before pairing."
        ),
    )
    parser.add_argument(
        "--segment-window-size",
        default=2,
        type=int,
        help="Number of sentences per segment when --speech-unit segment is used.",
    )
    parser.add_argument(
        "--segment-stride",
        default=1,
        type=int,
        help="Sentence stride for overlapping segments when --speech-unit segment is used.",
    )
    parser.add_argument(
        "--min-segment-words",
        default=12,
        type=int,
        help="Minimum word count for segment rows when --speech-unit segment is used.",
    )
    parser.add_argument(
        "--max-segment-words",
        default=90,
        type=int,
        help="Maximum word count for segment rows when --speech-unit segment is used.",
    )
    parser.add_argument(
        "--pair-selection",
        choices=["hybrid_embedding_bm25", "embedding_topk", "openai_embedding_topk", "tfidf_topk", "random"],
        default="tfidf_topk",
        help=(
            "How to select manifesto quasi-sentences after party/month/doc_key "
            "joining. embedding_topk uses the selected embedding provider; "
            "openai_embedding_topk is a backwards-compatible alias; "
            "hybrid_embedding_bm25 unions embedding and BM25 candidates before reranking; "
            "tfidf_topk ranks candidates by lexical TF-IDF cosine similarity; "
            "random preserves the original random sampling behavior."
        ),
    )
    parser.add_argument(
        "--min-retrieval-score",
        default=0.0,
        type=float,
        help=(
            "Minimum retrieval cosine similarity for keeping a speech-manifesto pair "
            "when a top-k retrieval mode is used. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "local"],
        default="openai",
        help=(
            "Embedding backend for embedding retrieval modes. Use openai for "
            "text-embedding-3-small as an external benchmark, or local for Hugging Face models."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Embedding model. OpenAI benchmark: text-embedding-3-small. "
            "Local options include Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, "
            "and intfloat/multilingual-e5-large."
        ),
    )
    parser.add_argument(
        "--embedding-batch-size",
        default=96,
        type=int,
        help="Number of texts per embedding batch.",
    )
    parser.add_argument(
        "--local-embedding-device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Torch device for local embedding models.",
    )
    parser.add_argument(
        "--embedding-cache-db",
        default=DEFAULT_EMBEDDING_CACHE_DB,
        type=Path,
        help="SQLite cache for OpenAI/local embeddings. Use with --no-embedding-disk-cache to disable.",
    )
    parser.add_argument(
        "--embedding-disk-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist and reuse embeddings across script runs. Enabled by default.",
    )
    parser.add_argument(
        "--hybrid-embedding-top-k",
        default=50,
        type=int,
        help="Number of embedding-retrieved manifesto candidates per speech unit in hybrid mode.",
    )
    parser.add_argument(
        "--hybrid-bm25-top-k",
        default=50,
        type=int,
        help="Number of BM25-retrieved manifesto candidates per speech unit in hybrid mode.",
    )
    parser.add_argument(
        "--hybrid-rerank-top-k",
        default=30,
        type=int,
        help=(
            "Final number of unioned hybrid candidates kept after reranking. "
            "Use 0 to fall back to --pairs-per-speech, or keep all if that is also 0."
        ),
    )
    parser.add_argument(
        "--hybrid-embedding-weight",
        default=0.5,
        type=float,
        help=(
            "Weight on normalized embedding score in hybrid reranking. "
            "The remaining weight is assigned to normalized BM25."
        ),
    )
    parser.add_argument(
        "--hybrid-min-embedding-score",
        default=0.0,
        type=float,
        help=(
            "Minimum raw embedding cosine score for retaining a hybrid candidate. "
            "Useful as an absolute comparability filter; 0 disables it."
        ),
    )
    parser.add_argument(
        "--hybrid-min-bm25-score",
        default=0.0,
        type=float,
        help=(
            "Minimum raw BM25 score for retaining a hybrid candidate. "
            "Useful as a lexical-overlap filter; 0 disables it."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help=(
            "Environment variable containing the OpenAI API key. If this is "
            "OPENAI_API_KEY and it is unset, the script also checks OPEN_API_KEY."
        ),
    )
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument(
        "--nli-max-length",
        default=512,
        type=int,
        help="Maximum tokenized sequence length for NLI classification.",
    )
    parser.add_argument(
        "--nli-checkpoint-interval",
        default=50000,
        type=int,
        help=(
            "Write resumable NLI classification checkpoints every N classified pairs. "
            "Use 0 to disable checkpointing."
        ),
    )
    parser.add_argument(
        "--retrieval-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory for resumable hybrid-retrieval parquet parts. "
            "Defaults beside the pair output."
        ),
    )
    parser.add_argument(
        "--retrieval-checkpoint-interval",
        type=int,
        default=10000,
        help=(
            "Persist hybrid selections every N retrieval units. "
            "Use 0 to disable retrieval checkpointing."
        ),
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Stop after atomically assembling retrieved pairs. This skips the "
            "generic NLI model so a downstream classifier can consume the "
            "retrieval in chunks."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, type=str)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help=(
            "Torch device for transformer NLI. auto uses CUDA when available, "
            "then Apple MPS when available, otherwise CPU."
        ),
    )
    parser.add_argument(
        "--min-speech-words",
        default=30,
        type=int,
        help=(
            "Minimum word count after optional procedural phrase stripping. "
            "Use 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--min-speech-topic-mass",
        default=0.2,
        type=float,
        help=(
            "Minimum summed PLDA probability across the selected topic columns "
            "for a speech row to be sampled. Use 0 to disable this filter."
        ),
    )
    parser.add_argument(
        "--keep-procedural-speeches",
        action="store_true",
        help="Keep rows that look purely procedural, such as thanks or name-only fragments.",
    )
    parser.add_argument(
        "--no-strip-procedural-phrases",
        action="store_true",
        help="Do not remove common procedural openings/closings before NLI.",
    )
    parser.add_argument(
        "--speech-input",
        default=None,
        type=Path,
        help=(
            "PLDA individual speech CSV. Defaults to "
            "outputs/test_speeches/plda_individual_speeches_<COUNTRY>.csv."
        ),
    )
    parser.add_argument(
        "--manifesto-quasi-input",
        default=None,
        type=Path,
        help=(
            "PLDA manifesto quasi-sentence topic CSV. Defaults to "
            "outputs/test_speeches/plda_manifesto_inference/<COUNTRY>/"
            "<COUNTRY>_plda_manifesto_quasi_sentence_topics.csv."
        ),
    )
    parser.add_argument(
        "--bridge-input",
        default=None,
        type=Path,
        help=(
            "Speech-date to manifesto bridge CSV. Rows without a manifesto on or "
            "before the speech date are excluded. Defaults to "
            "outputs/manifesto_quasi_sentences/<COUNTRY>/"
            "<COUNTRY>_speech_date_to_manifesto_bridge.csv."
        ),
    )
    parser.add_argument(
        "--topic-labels-input",
        default=None,
        type=Path,
        help=(
            "Optional topic-column label CSV from plda_distribution.py. If omitted, "
            "the script infers topic columns from the speech topic distributions."
        ),
    )
    parser.add_argument(
        "--manifesto-topic-filter",
        choices=["same_topic", "all"],
        default="same_topic",
        help=(
            "Use only linked manifesto quasi-sentences predicted as the selected "
            "PLDA topic, or compare against all quasi-sentences in the linked manifesto."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Base output directory. Files are written under <output-dir>/<COUNTRY>/.",
    )
    return parser


def default_speech_input(country_code: str) -> Path:
    return TEST_OUTPUT_DIR / f"plda_individual_speeches_{country_code}.csv"


def default_manifesto_quasi_input(country_code: str) -> Path:
    return (
        TEST_OUTPUT_DIR
        / "plda_manifesto_inference"
        / country_code
        / f"{country_code}_plda_manifesto_quasi_sentence_topics.csv"
    )


def default_bridge_input(country_code: str) -> Path:
    return (
        MANIFESTO_INPUT_DIR
        / country_code
        / f"{country_code}_speech_date_to_manifesto_bridge.csv"
    )


def default_topic_labels_input(country_code: str) -> Path:
    return (
        TEST_OUTPUT_DIR
        / "plda_topic_distributions"
        / country_code
        / f"{country_code}_plda_topic_labels.csv"
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    country_code = args.country.strip().upper()
    speech_input = args.speech_input or default_speech_input(country_code)
    manifesto_quasi_input = (
        args.manifesto_quasi_input or default_manifesto_quasi_input(country_code)
    )
    bridge_input = args.bridge_input or default_bridge_input(country_code)
    topic_labels_input = args.topic_labels_input or default_topic_labels_input(country_code)
    if not topic_labels_input.exists():
        topic_labels_input = None
    return speech_input, manifesto_quasi_input, bridge_input, topic_labels_input


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {path}")
    data = pd.read_csv(path, low_memory=False)
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return data


def topic_columns(data: pd.DataFrame) -> list[str]:
    columns = [col for col in data.columns if TOPIC_COLUMN_RE.match(str(col))]
    return sorted(columns, key=lambda col: int(TOPIC_COLUMN_RE.match(col).group(1)))


def base_topic_label(label: object) -> str:
    return re.sub(r"\s+\d+$", "", str(label).strip())


def topic_columns_from_label_file(
    path: Path | None,
    topic: str,
    expected_topic_count: int,
) -> list[str]:
    if path is None:
        return []
    labels = load_csv(path, {"topic_column", "topic_label"})
    if len(labels) != expected_topic_count:
        print(
            f"Ignoring stale topic-label file {path}: "
            f"{len(labels)} labels for {expected_topic_count} topic columns."
        )
        return []
    mask = labels["topic_label"].map(base_topic_label) == topic
    return labels.loc[mask, "topic_column"].astype(str).tolist()


def infer_topic_columns_from_speeches(
    speech_df: pd.DataFrame,
    topic: str,
    threshold: float = 1e-9,
) -> list[str]:
    topics = topic_columns(speech_df)
    if not topics:
        raise ValueError("Speech input has no topic_* columns.")

    inside = speech_df[speech_df["topic_label"].astype(str) == topic]
    outside = speech_df[speech_df["topic_label"].astype(str) != topic]
    if inside.empty:
        raise ValueError(f"No speech rows found with topic_label == {topic!r}.")

    inside_mean = inside[topics].mean(numeric_only=True)
    outside_mean = outside[topics].mean(numeric_only=True) if not outside.empty else 0
    selected = [
        col
        for col in topics
        if float(inside_mean[col]) > threshold
        and (not isinstance(outside_mean, pd.Series) or float(outside_mean[col]) <= threshold)
    ]
    if selected:
        return selected

    return [col for col in topics if float(inside_mean[col]) > threshold]


def word_count(text: object) -> int:
    return len(WORD_RE.findall(str(text)))


def procedural_patterns(country_code: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    if country_code.upper() in {"GB", "UK", "IE", "US", "CA", "AU", "NZ"}:
        return EN_PROCEDURAL_OPENING_RE, EN_PROCEDURAL_CLOSING_RE, EN_SHORT_PROCEDURAL_RE
    short_re = re.compile(
        r"(?i)d[ěe]kuji|díky|za\s+pozornost|za\s+slovo|prosím\s+o\s+podporu"
    )
    return PROCEDURAL_OPENING_RE, PROCEDURAL_CLOSING_RE, short_re


def strip_procedural_phrases(text: object, country_code: str) -> str:
    opening_re, closing_re, _ = procedural_patterns(country_code)
    stripped = str(text).strip()
    previous = None
    while stripped != previous:
        previous = stripped
        stripped = opening_re.sub("", stripped).strip()
        stripped = closing_re.sub("", stripped).strip()
    return stripped


def looks_procedural(
    text: object,
    stripped_text: object,
    stripped_word_count: int,
    country_code: str,
) -> bool:
    raw = str(text).strip()
    _, _, short_procedural_re = procedural_patterns(country_code)
    if not raw:
        return True
    if NAME_ONLY_RE.match(raw):
        return True
    if stripped_word_count == 0:
        return True
    if stripped_word_count <= 8 and short_procedural_re.search(raw):
        return True
    return False


def diagnose_speeches(
    speech_df: pd.DataFrame,
    country_code: str,
    topic: str,
    selected_topic_cols: list[str],
    min_words: int,
    min_topic_mass: float,
    keep_procedural: bool,
    strip_procedural: bool,
) -> pd.DataFrame:
    topic_speeches = speech_df[speech_df["topic_label"].astype(str) == topic].copy()
    topic_speeches["speech_filter_reason"] = ""

    required_cols = ["party", "month", "text"]
    missing_required_mask = topic_speeches[required_cols].isna().any(axis=1)
    topic_speeches.loc[missing_required_mask, "speech_filter_reason"] = "missing_required_metadata"

    topic_speeches["text"] = topic_speeches["text"].astype(str).str.strip()
    empty_text_mask = topic_speeches["text"] == ""
    topic_speeches.loc[
        (topic_speeches["speech_filter_reason"] == "") & empty_text_mask,
        "speech_filter_reason",
    ] = "empty_text"

    selected_cols = [col for col in selected_topic_cols if col in topic_speeches.columns]
    missing_topic_cols = sorted(set(selected_topic_cols) - set(selected_cols))
    if missing_topic_cols:
        raise ValueError(f"Speech input is missing selected topic columns: {missing_topic_cols}")

    if selected_cols:
        topic_speeches["speech_selected_topic_mass"] = (
            topic_speeches[selected_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )
    else:
        topic_speeches["speech_selected_topic_mass"] = 0.0

    topic_speeches["speech_text_original"] = topic_speeches["text"]
    topic_speeches["speech_text_for_nli"] = (
        topic_speeches["text"].map(lambda text: strip_procedural_phrases(text, country_code))
        if strip_procedural
        else topic_speeches["text"]
    )
    topic_speeches["speech_word_count_original"] = topic_speeches["text"].map(word_count)
    topic_speeches["speech_word_count_for_nli"] = topic_speeches["speech_text_for_nli"].map(word_count)
    topic_speeches["speech_is_procedural"] = topic_speeches.apply(
        lambda row: looks_procedural(
            row["speech_text_original"],
            row["speech_text_for_nli"],
            int(row["speech_word_count_for_nli"]),
            country_code,
        ),
        axis=1,
    )

    if min_topic_mass > 0:
        low_topic_mask = topic_speeches["speech_selected_topic_mass"] < min_topic_mass
        topic_speeches.loc[
            (topic_speeches["speech_filter_reason"] == "") & low_topic_mask,
            "speech_filter_reason",
        ] = "low_selected_topic_mass"

    if not keep_procedural:
        procedural_mask = topic_speeches["speech_is_procedural"]
        topic_speeches.loc[
            (topic_speeches["speech_filter_reason"] == "") & procedural_mask,
            "speech_filter_reason",
        ] = "procedural_or_name_only"

    if min_words > 0:
        short_mask = topic_speeches["speech_word_count_for_nli"] < min_words
        topic_speeches.loc[
            (topic_speeches["speech_filter_reason"] == "") & short_mask,
            "speech_filter_reason",
        ] = "too_short_after_procedural_strip"

    topic_speeches["speech_filter_kept"] = topic_speeches["speech_filter_reason"] == ""
    topic_speeches.loc[topic_speeches["speech_filter_kept"], "speech_filter_reason"] = "kept"
    return topic_speeches.reset_index(drop=True)


def compact_speech_rows(speeches: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "plda_doc_id",
        "country",
        "date",
        "party",
        "month",
        "month_start",
        "topic_label",
        "text",
        "speech_text_for_nli",
        "speech_text_original",
        "speech_selected_topic_mass",
        "speech_word_count_for_nli",
        "speech_is_procedural",
        "speech_filter_reason",
        "speech_filter_kept",
    ]
    return speeches[[col for col in keep_cols if col in speeches.columns]].copy()


def sample_speeches(
    speech_df: pd.DataFrame,
    topic: str,
    sample_size: int,
    sample_per_party: int | None,
    random_state: int,
) -> pd.DataFrame:
    topic_speeches = speech_df[speech_df["speech_filter_kept"]].copy()
    topic_speeches["text"] = topic_speeches["speech_text_for_nli"].astype(str).str.strip()
    if topic_speeches.empty:
        reason_counts = speech_df["speech_filter_reason"].value_counts(dropna=False).to_dict()
        raise ValueError(
            f"No speeches remained for topic {topic!r} after quality filters. "
            f"Filter reason counts: {reason_counts}"
        )

    if sample_per_party is not None:
        sampled_parts = [
            group.sample(
                n=min(len(group), sample_per_party),
                random_state=random_state,
            )
            for _, group in topic_speeches.groupby("party", sort=False)
        ]
        sampled = pd.concat(sampled_parts, ignore_index=True)
    else:
        sampled = topic_speeches

    if sample_size > 0 and len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)

    return compact_speech_rows(sampled).reset_index(drop=True)


def split_sentences(text: object) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(cleaned) if part.strip()]
    return sentences or [cleaned]


def build_sentence_windows(
    sentences: list[str],
    window_size: int,
    stride: int,
) -> list[tuple[int, int, str]]:
    if not sentences:
        return []
    effective_window = min(window_size, len(sentences))
    windows = []
    for start in range(0, len(sentences), stride):
        end = min(start + effective_window, len(sentences))
        if end <= start:
            continue
        windows.append((start, end - 1, " ".join(sentences[start:end]).strip()))
        if end == len(sentences):
            break
    return windows


def build_speech_units(
    speeches: pd.DataFrame,
    speech_unit: str,
    window_size: int,
    stride: int,
    min_segment_words: int,
    max_segment_words: int,
) -> pd.DataFrame:
    if speech_unit == "speech":
        units = speeches.copy()
        units["speech_unit"] = "speech"
        units["speech_segment_id"] = pd.NA
        units["retrieval_unit_id"] = units["plda_doc_id"].astype(str)
        units["speech_segment_index"] = pd.NA
        units["segment_start_sentence"] = pd.NA
        units["segment_end_sentence"] = pd.NA
        units["speech_text_original_full"] = units["speech_text_original"]
        return units.reset_index(drop=True)

    segment_rows: list[dict[str, Any]] = []
    for speech in speeches.itertuples(index=False):
        row = speech._asdict()
        full_text = str(row.get("text", "")).strip()
        sentences = split_sentences(full_text)
        segment_index = 0
        for start, end, segment_text in build_sentence_windows(sentences, window_size, stride):
            segment_words = word_count(segment_text)
            if min_segment_words > 0 and segment_words < min_segment_words:
                continue
            if max_segment_words > 0 and segment_words > max_segment_words:
                continue
            segment_id = f"{row['plda_doc_id']}:{segment_index}"
            segment_row = dict(row)
            segment_row.update(
                {
                    "speech_unit": "segment",
                    "speech_segment_id": segment_id,
                    "retrieval_unit_id": segment_id,
                    "speech_segment_index": segment_index,
                    "segment_start_sentence": start,
                    "segment_end_sentence": end,
                    "speech_text_original_full": row.get("speech_text_original", full_text),
                    "speech_segment_text": segment_text,
                    "text": segment_text,
                    "speech_text_for_nli": segment_text,
                    "speech_word_count_for_nli": segment_words,
                }
            )
            segment_rows.append(segment_row)
            segment_index += 1

    if not segment_rows:
        raise ValueError(
            "No speech segments remained after segmentation filters. "
            "Lower --min-segment-words, raise --max-segment-words, or use --speech-unit speech."
        )
    return pd.DataFrame(segment_rows).reset_index(drop=True)


def retrieval_group_column(pairs: pd.DataFrame) -> str:
    if "retrieval_unit_id" in pairs.columns:
        return "retrieval_unit_id"
    return "plda_doc_id"


def prepare_manifesto_quasi(
    manifesto_df: pd.DataFrame,
    topic_cols: list[str],
    topic_filter: str,
) -> pd.DataFrame:
    manifesto_df = manifesto_df.dropna(subset=["doc_key", "text"]).copy()
    manifesto_df["text"] = manifesto_df["text"].astype(str).str.strip()
    manifesto_df = manifesto_df[manifesto_df["text"] != ""].copy()
    if topic_filter == "same_topic":
        if "predicted_topic" not in manifesto_df.columns:
            raise ValueError(
                "Manifesto topic filtering needs a predicted_topic column. "
                "Pass --manifesto-topic-filter all to disable it."
            )
        manifesto_df = manifesto_df[
            manifesto_df["predicted_topic"].astype(str).isin(topic_cols)
        ].copy()
    if manifesto_df.empty:
        raise ValueError("No manifesto quasi-sentences remained after filtering.")
    return manifesto_df.reset_index(drop=True)


def select_random_pairs(
    pairs: pd.DataFrame,
    pairs_per_speech: int,
    random_state: int,
) -> pd.DataFrame:
    group_col = retrieval_group_column(pairs)
    if pairs_per_speech > 0:
        sampled_parts = [
            group.sample(
                n=min(len(group), pairs_per_speech),
                random_state=random_state,
            )
            for _, group in pairs.groupby(group_col, sort=False)
        ]
        pairs = pd.concat(sampled_parts, ignore_index=True)
    else:
        pairs = pairs.copy()

    pairs["retrieval_score"] = pd.NA
    pairs["retrieval_rank"] = pairs.groupby(group_col, sort=False).cumcount() + 1
    pairs["pair_selection_method"] = "random"
    return pairs


def tfidf_similarity_scores(speech_text: str, manifesto_texts: list[str]) -> np.ndarray:
    if not manifesto_texts:
        return np.array([], dtype=float)
    texts = [speech_text] + manifesto_texts
    try:
        matrix = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        ).fit_transform(texts)
    except ValueError:
        return np.zeros(len(manifesto_texts), dtype=float)
    return cosine_similarity(matrix[0], matrix[1:]).ravel()


def select_tfidf_topk_pairs(
    pairs: pd.DataFrame,
    pairs_per_speech: int,
    min_retrieval_score: float,
) -> pd.DataFrame:
    selected_parts = []
    group_col = retrieval_group_column(pairs)
    for _, group in pairs.groupby(group_col, sort=False):
        group = group.copy()
        speech_text = str(group["text"].iloc[0])
        manifesto_texts = group["manifesto_text"].fillna("").astype(str).tolist()
        scores = tfidf_similarity_scores(speech_text, manifesto_texts)
        order = np.argsort(-scores, kind="stable")
        ranked = group.iloc[order].copy()
        ranked["retrieval_score"] = scores[order]
        ranked["retrieval_rank"] = range(1, len(ranked) + 1)
        ranked["pair_selection_method"] = "tfidf_topk"
        if min_retrieval_score > 0:
            ranked = ranked[ranked["retrieval_score"] >= min_retrieval_score].copy()
        if pairs_per_speech > 0:
            ranked = ranked.head(pairs_per_speech).copy()
        if not ranked.empty:
            selected_parts.append(ranked)

    if not selected_parts:
        raise ValueError(
            "No speech-manifesto pairs remained after TF-IDF top-k retrieval. "
            "Lower --min-retrieval-score or use --pair-selection random."
        )
    return pd.concat(selected_parts, ignore_index=True)


def select_tfidf_topk_pairs_from_joined(
    merged: pd.DataFrame,
    manifesto: pd.DataFrame,
    pairs_per_speech: int,
    min_retrieval_score: float,
) -> tuple[pd.DataFrame, int]:
    manifesto_by_doc = {
        doc_key: group.reset_index(drop=True)
        for doc_key, group in manifesto.groupby("doc_key", sort=False)
    }
    group_col = retrieval_group_column(merged)
    selected_rows: list[dict[str, Any]] = []
    candidate_pair_count = 0
    total_units = merged[group_col].nunique()
    started = time.perf_counter()

    for unit_i, (_, unit_rows) in enumerate(merged.groupby(group_col, sort=False), start=1):
        candidate_parts = []
        bridge_rows_by_doc: dict[Any, dict[str, Any]] = {}
        for bridge_row in unit_rows.to_dict("records"):
            doc_key = bridge_row.get("doc_key")
            doc_manifesto = manifesto_by_doc.get(doc_key)
            if doc_manifesto is None or doc_manifesto.empty:
                continue
            candidate_parts.append(doc_manifesto)
            bridge_rows_by_doc[doc_key] = bridge_row

        if not candidate_parts:
            continue

        candidates = pd.concat(candidate_parts, ignore_index=True)
        candidate_pair_count += len(candidates)
        speech_text = str(unit_rows["text"].iloc[0])
        manifesto_texts = candidates["manifesto_text"].fillna("").astype(str).tolist()
        scores = tfidf_similarity_scores(speech_text, manifesto_texts)
        order = np.argsort(-scores, kind="stable")
        if min_retrieval_score > 0:
            order = np.array([idx for idx in order if scores[idx] >= min_retrieval_score], dtype=int)
        if pairs_per_speech > 0:
            order = order[:pairs_per_speech]

        for rank, candidate_idx in enumerate(order, start=1):
            manifesto_row = candidates.iloc[int(candidate_idx)].to_dict()
            bridge_row = bridge_rows_by_doc.get(manifesto_row.get("doc_key"), unit_rows.iloc[0].to_dict())
            selected_row = dict(bridge_row)
            selected_row.update(manifesto_row)
            selected_row["retrieval_score"] = float(scores[int(candidate_idx)])
            selected_row["retrieval_rank"] = rank
            selected_row["pair_selection_method"] = "tfidf_topk"
            selected_rows.append(selected_row)

        if unit_i == 1 or unit_i % 1000 == 0 or unit_i == total_units:
            elapsed = time.perf_counter() - started
            print(
                f"TF-IDF retrieval {unit_i:,}/{total_units:,} units: "
                f"selected={len(selected_rows):,}, candidates_scored={candidate_pair_count:,}, "
                f"elapsed={elapsed:.1f}s"
            )

    if not selected_rows:
        raise ValueError(
            "No speech-manifesto pairs remained after TF-IDF top-k retrieval. "
            "Lower --min-retrieval-score or use --pair-selection random."
        )
    return pd.DataFrame(selected_rows), candidate_pair_count


def openai_api_key(env_name: str) -> str:
    api_key = os.getenv(env_name)
    if not api_key and env_name == "OPENAI_API_KEY":
        api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing OpenAI API key in {env_name}. "
            "Set it in the environment/.env, or pass --api-key-env."
        )
    return api_key


def embed_text_batch(
    texts: list[str],
    model: str,
    api_key: str,
    max_retries: int = 4,
) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENAI_EMBEDDINGS_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2**attempt, 30)
                time.sleep(wait)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                error_text = response.text.strip()
                if error_text:
                    raise RuntimeError(f"{exc}: {error_text}") from exc
                raise
            data = response.json()["data"]
            data = sorted(data, key=lambda item: item["index"])
            return [item["embedding"] for item in data]
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI embeddings request failed: {last_error}")


_EMBEDDING_CACHE: dict[tuple[str, str, str, str], np.ndarray] = {}
_LOCAL_EMBEDDING_COMPONENT_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}


def embedding_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_embedding_cache_db(cache_db: Path) -> None:
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                role TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, model, role, text_hash)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_lookup "
            "ON embeddings(provider, model, role, text_hash)"
        )


def vector_to_blob(vector: np.ndarray) -> tuple[bytes, int, str]:
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes(), int(arr.size), str(arr.dtype)


def vector_from_blob(blob: bytes, dtype: str, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.dtype(dtype), count=dim)
    return (
        vector
        if vector.dtype == np.dtype(np.float32)
        else vector.astype(np.float32)
    )


def load_embeddings_from_disk_cache(
    cache_db: Path,
    provider: str,
    model: str,
    role: str,
    texts: list[str],
    query_batch_size: int = 500,
) -> dict[str, np.ndarray]:
    """Fetch many cached vectors with one connection and bounded IN queries."""
    if not cache_db.exists() or not texts:
        return {}
    hashes = [embedding_text_hash(text) for text in texts]
    vectors: dict[str, np.ndarray] = {}
    with sqlite3.connect(cache_db) as conn:
        for start in range(0, len(hashes), query_batch_size):
            batch = hashes[start : start + query_batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"""
                SELECT text_hash, vector, dtype, dim
                FROM embeddings
                WHERE provider = ? AND model = ? AND role = ?
                  AND text_hash IN ({placeholders})
                """,
                (provider, model, role, *batch),
            ).fetchall()
            for text_hash, blob, dtype, dim in rows:
                vectors[str(text_hash)] = vector_from_blob(
                    blob=blob, dtype=dtype, dim=int(dim)
                )
    return vectors


def save_embeddings_to_disk_cache(
    cache_db: Path,
    provider: str,
    model: str,
    role: str,
    items: list[tuple[str, np.ndarray]],
) -> None:
    if not items:
        return
    init_embedding_cache_db(cache_db)
    rows = []
    for text, vector in items:
        blob, dim, dtype = vector_to_blob(vector)
        rows.append((provider, model, role, embedding_text_hash(text), text, dim, dtype, blob))
    with sqlite3.connect(cache_db) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
                (provider, model, role, text_hash, text, dim, dtype, vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def truncate_embedding_text(text: str) -> str:
    stripped = text.strip()
    if len(stripped) > MAX_EMBEDDING_INPUT_CHARS:
        stripped = stripped[:MAX_EMBEDDING_INPUT_CHARS]
    words = stripped.split()
    if len(words) > MAX_EMBEDDING_INPUT_WORDS:
        stripped = " ".join(words[:MAX_EMBEDDING_INPUT_WORDS])
    return stripped


def embedding_input_text(text: str, model: str, role: str) -> str:
    stripped = truncate_embedding_text(text)
    model_lower = model.lower()
    if "multilingual-e5" in model_lower:
        prefix = "query" if role == "query" else "passage"
        return f"{prefix}: {stripped}"
    return stripped


def mean_pool_embeddings(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def load_local_embedding_components(model: str, device_name: str) -> tuple[Any, Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_torch_device(torch, device_name)
    cache_key = (model, str(device))
    cached = _LOCAL_EMBEDDING_COMPONENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    print_torch_device_summary(torch, device)
    print(f"Loading local embedding tokenizer/model on {device}: {model}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
    hf_model.eval()
    components = (torch, tokenizer, hf_model, device)
    _LOCAL_EMBEDDING_COMPONENT_CACHE[cache_key] = components
    print(f"Loaded local embedding model in {time.perf_counter() - load_start:.1f}s.")
    return components


def embed_local_text_batch(
    texts: list[str],
    model: str,
    device_name: str,
) -> list[np.ndarray]:
    torch, tokenizer, hf_model, device = load_local_embedding_components(model, device_name)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = hf_model(**encoded)
        pooled = mean_pool_embeddings(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
    return [np.array(row, dtype=float) for row in pooled]


def embed_unique_texts(
    texts: list[str],
    model: str,
    provider: str,
    api_key_env: str,
    batch_size: int,
    role: str,
    local_device: str,
    cache_db: Path,
    use_disk_cache: bool,
) -> dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(text.strip() for text in texts if text.strip()))
    embeddings: dict[str, np.ndarray] = {}
    uncached_texts: list[str] = []
    disk_candidates: list[str] = []
    memory_hits = 0
    disk_hits = 0

    for text in unique_texts:
        cache_key = (provider, model, role, text)
        cached = _EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            embeddings[text] = cached
            memory_hits += 1
            continue
        if use_disk_cache:
            disk_candidates.append(text)
        else:
            uncached_texts.append(text)

    if disk_candidates:
        disk_vectors = load_embeddings_from_disk_cache(
            cache_db=cache_db,
            provider=provider,
            model=model,
            role=role,
            texts=disk_candidates,
        )
        for text in disk_candidates:
            disk_cached = disk_vectors.get(embedding_text_hash(text))
            if disk_cached is not None:
                embeddings[text] = disk_cached
                _EMBEDDING_CACHE[
                    (provider, model, role, text)
                ] = disk_cached
                disk_hits += 1
            else:
                uncached_texts.append(text)

    if memory_hits or disk_hits:
        print(
            f"Embedding cache hits ({provider}, {role}): "
            f"memory={memory_hits:,}, disk={disk_hits:,}, misses={len(uncached_texts):,}."
        )

    total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
    api_key = openai_api_key(api_key_env) if provider == "openai" and uncached_texts else ""
    for batch_i, start in enumerate(range(0, len(uncached_texts), batch_size), start=1):
        batch = uncached_texts[start : start + batch_size]
        model_inputs = [embedding_input_text(text, model, role) for text in batch]
        if provider == "openai":
            vectors = [
                np.array(vector, dtype=np.float32)
                for vector in embed_text_batch(model_inputs, model=model, api_key=api_key)
            ]
        elif provider == "local":
            vectors = embed_local_text_batch(model_inputs, model=model, device_name=local_device)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        cache_items: list[tuple[str, np.ndarray]] = []
        for text, vector in zip(batch, vectors, strict=True):
            norm = np.linalg.norm(vector)
            normalized = vector / norm if norm else vector
            embeddings[text] = normalized
            _EMBEDDING_CACHE[(provider, model, role, text)] = normalized
            cache_items.append((text, normalized))
        if use_disk_cache:
            save_embeddings_to_disk_cache(cache_db, provider, model, role, cache_items)
        print(
            f"Embedding batch {batch_i:,}/{total_batches:,} ({provider}, {role}): "
            f"{min(start + len(batch), len(uncached_texts)):,}/{len(uncached_texts):,} uncached texts."
        )
    if unique_texts and not uncached_texts:
        print(f"Reused {len(unique_texts):,} cached embedding text(s) ({provider}, {role}).")
    return embeddings

def embed_query_and_passage_texts(
    query_texts: list[str],
    passage_texts: list[str],
    model: str,
    provider: str,
    api_key_env: str,
    batch_size: int,
    local_device: str,
    cache_db: Path,
    use_disk_cache: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    query_embeddings = embed_unique_texts(
        texts=query_texts,
        model=model,
        provider=provider,
        api_key_env=api_key_env,
        batch_size=batch_size,
        role="query",
        local_device=local_device,
        cache_db=cache_db,
        use_disk_cache=use_disk_cache,
    )
    passage_embeddings = embed_unique_texts(
        texts=passage_texts,
        model=model,
        provider=provider,
        api_key_env=api_key_env,
        batch_size=batch_size,
        role="passage",
        local_device=local_device,
        cache_db=cache_db,
        use_disk_cache=use_disk_cache,
    )
    return query_embeddings, passage_embeddings

def select_embedding_topk_pairs(
    pairs: pd.DataFrame,
    pairs_per_speech: int,
    min_retrieval_score: float,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
    local_embedding_device: str,
    embedding_cache_db: Path,
    embedding_disk_cache: bool,
) -> pd.DataFrame:
    query_texts = pairs["text"].fillna("").astype(str).str.strip().tolist()
    passage_texts = pairs["manifesto_text"].fillna("").astype(str).str.strip().tolist()
    query_embeddings, passage_embeddings = embed_query_and_passage_texts(
        query_texts=query_texts,
        passage_texts=passage_texts,
        model=embedding_model,
        provider=embedding_provider,
        api_key_env=api_key_env,
        batch_size=embedding_batch_size,
        local_device=local_embedding_device,
        cache_db=embedding_cache_db,
        use_disk_cache=embedding_disk_cache,
    )

    selected_parts = []
    group_col = retrieval_group_column(pairs)
    for _, group in pairs.groupby(group_col, sort=False):
        group = group.copy()
        speech_text = str(group["text"].iloc[0]).strip()
        speech_vector = query_embeddings.get(speech_text)
        if speech_vector is None:
            scores = np.zeros(len(group), dtype=float)
        else:
            manifesto_vectors = [
                passage_embeddings.get(text.strip())
                for text in group["manifesto_text"].fillna("").astype(str)
            ]
            scores = np.array(
                [
                    float(np.dot(speech_vector, vector)) if vector is not None else 0.0
                    for vector in manifesto_vectors
                ],
                dtype=float,
            )
        order = np.argsort(-scores, kind="stable")
        ranked = group.iloc[order].copy()
        ranked["retrieval_score"] = scores[order]
        ranked["retrieval_rank"] = range(1, len(ranked) + 1)
        ranked["pair_selection_method"] = "embedding_topk"
        ranked["embedding_provider"] = embedding_provider
        ranked["embedding_model"] = embedding_model
        if min_retrieval_score > 0:
            ranked = ranked[ranked["retrieval_score"] >= min_retrieval_score].copy()
        if pairs_per_speech > 0:
            ranked = ranked.head(pairs_per_speech).copy()
        if not ranked.empty:
            selected_parts.append(ranked)

    if not selected_parts:
        raise ValueError(
            "No speech-manifesto pairs remained after embedding retrieval. "
            "Lower --min-retrieval-score or use another --pair-selection mode."
        )
    return pd.concat(selected_parts, ignore_index=True)

def retrieval_tokens(text: object) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_terms = retrieval_tokens(query)
    if not query_terms or not documents:
        return np.zeros(len(documents), dtype=float)

    tokenized_docs = [retrieval_tokens(doc) for doc in documents]
    doc_count = len(tokenized_docs)
    doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
    avg_doc_len = float(doc_lengths.mean()) if doc_count else 0.0
    if avg_doc_len <= 0:
        return np.zeros(len(documents), dtype=float)

    dfs: dict[str, int] = {}
    for tokens in tokenized_docs:
        for term in set(tokens):
            dfs[term] = dfs.get(term, 0) + 1

    scores = np.zeros(doc_count, dtype=float)
    query_vocab = set(query_terms)
    for doc_i, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        term_counts: dict[str, int] = {}
        for term in tokens:
            term_counts[term] = term_counts.get(term, 0) + 1
        length_norm = k1 * (1.0 - b + b * doc_lengths[doc_i] / avg_doc_len)
        for term in query_vocab:
            tf = term_counts.get(term, 0)
            if not tf:
                continue
            df = dfs.get(term, 0)
            idf = np.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            scores[doc_i] += idf * (tf * (k1 + 1.0)) / (tf + length_norm)
    return scores


def normalized_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        return np.ones_like(scores, dtype=float) if max_score > 0 else np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    if top_k > 0:
        order = order[:top_k]
    return [int(index) for index in order]


def hybrid_retrieval_signature(
    merged: pd.DataFrame,
    manifesto: pd.DataFrame,
    group_col: str,
    config: dict[str, Any],
) -> str:
    speech_columns = [
        column
        for column in (
            group_col,
            "doc_key",
            "date",
            "party",
            "plda_doc_id",
            "speech_segment_id",
            "retrieval_unit_id",
            "text",
            "speech_text_for_nli",
        )
        if column in merged.columns
    ]
    manifesto_columns = [
        column
        for column in (
            "doc_key",
            "quasi_sentence_id",
            "manifesto_text",
            "text",
            "predicted_topic",
            "cmp_code",
            "eu_code",
        )
        if column in manifesto.columns
    ]
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "speech_columns": speech_columns,
        "speech_data_fingerprint": dataframe_fingerprint(
            merged, speech_columns
        ),
        "manifesto_columns": manifesto_columns,
        "manifesto_data_fingerprint": dataframe_fingerprint(
            manifesto, manifesto_columns
        ),
        "speech_rows": len(merged),
        "manifesto_rows": len(manifesto),
        "units": int(merged[group_col].nunique(dropna=False)),
        "config": config,
        "embedding_input_limits": {
            "max_words": MAX_EMBEDDING_INPUT_WORDS,
            "max_characters": MAX_EMBEDDING_INPUT_CHARS,
        },
    }
    return canonical_json_fingerprint(payload)


def load_hybrid_retrieval_checkpoint(
    checkpoint_dir: Path,
    signature: str,
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        stray = list(checkpoint_dir.glob("part-*.parquet"))
        if stray:
            raise RuntimeError(
                f"Retrieval checkpoint parts exist without a manifest: "
                f"{checkpoint_dir}"
            )
        return (
            {
                "version": PROVENANCE_VERSION,
                "signature": signature,
                "processed_units": 0,
                "candidate_pairs_scored": 0,
                "selected_rows": 0,
                "complete": False,
                "parts": [],
            },
            [],
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != PROVENANCE_VERSION:
        raise RuntimeError(
            "Refusing to resume a legacy retrieval checkpoint whose text/data "
            f"identity was not fingerprinted: {checkpoint_dir}"
        )
    if manifest.get("signature") != signature:
        raise RuntimeError(
            "Refusing to resume hybrid retrieval: checkpoint signature "
            "does not match the strict bridge/data/configuration"
        )
    part_paths: list[Path] = []
    expected_start = 1
    selected_rows_from_parts = 0
    candidate_pairs_from_parts = 0
    for part in manifest.get("parts", []):
        if int(part["unit_start"]) != expected_start:
            raise RuntimeError(
                "Hybrid retrieval checkpoint unit ranges are not contiguous"
            )
        expected_start = int(part["unit_end"]) + 1
        selected_rows_from_parts += int(part.get("selected_rows", 0))
        candidate_pairs_from_parts += int(
            part.get("candidate_pairs_scored", 0)
        )
        relative = part.get("path")
        if relative:
            path = checkpoint_dir / str(relative)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing hybrid retrieval checkpoint part: {path}"
                )
            if int(part.get("bytes", -1)) != path.stat().st_size:
                raise RuntimeError(
                    f"Hybrid retrieval checkpoint part size changed: {path}"
                )
            expected_fingerprint = part.get("sampled_content_sha256")
            if (
                not expected_fingerprint
                or sampled_file_content_fingerprint(path)
                != expected_fingerprint
            ):
                raise RuntimeError(
                    f"Hybrid retrieval checkpoint part content changed: {path}"
                )
            part_paths.append(path)
    if int(manifest.get("processed_units", 0)) != expected_start - 1:
        raise RuntimeError(
            "Hybrid retrieval checkpoint manifest has an inconsistent unit count"
        )
    if int(manifest.get("selected_rows", -1)) != selected_rows_from_parts:
        raise RuntimeError(
            "Hybrid retrieval checkpoint manifest has an inconsistent selected-row count"
        )
    if (
        int(manifest.get("candidate_pairs_scored", -1))
        != candidate_pairs_from_parts
    ):
        raise RuntimeError(
            "Hybrid retrieval checkpoint manifest has an inconsistent candidate-pair count"
        )
    return manifest, part_paths


def save_hybrid_retrieval_checkpoint(
    checkpoint_dir: Path,
    manifest: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    unit_start: int,
    unit_end: int,
    candidate_pairs: int,
) -> Path | None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    part_path: Path | None = None
    if selected_rows:
        part_path = checkpoint_dir / (
            f"part-{unit_start:09d}-{unit_end:09d}.parquet"
        )
        temporary_part = part_path.with_suffix(".parquet.tmp")
        pd.DataFrame(selected_rows).to_parquet(
            temporary_part, index=False
        )
        os.replace(temporary_part, part_path)
    manifest["parts"].append(
        {
            "unit_start": unit_start,
            "unit_end": unit_end,
            "candidate_pairs_scored": candidate_pairs,
            "selected_rows": len(selected_rows),
            "path": part_path.name if part_path is not None else None,
            "bytes": part_path.stat().st_size if part_path is not None else 0,
            "sampled_content_sha256": (
                sampled_file_content_fingerprint(part_path)
                if part_path is not None
                else None
            ),
        }
    )
    manifest["processed_units"] = unit_end
    manifest["candidate_pairs_scored"] = int(
        manifest.get("candidate_pairs_scored", 0)
    ) + candidate_pairs
    manifest["selected_rows"] = int(
        manifest.get("selected_rows", 0)
    ) + len(selected_rows)
    manifest_path = checkpoint_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return part_path


def retrieval_checkpoint_artifact(
    manifest: dict[str, Any],
    part_paths: list[Path],
) -> RetrievalCheckpointArtifact:
    artifact = RetrievalCheckpointArtifact(
        part_paths=tuple(part_paths),
        selected_rows=int(manifest.get("selected_rows", 0)),
        candidate_pairs_scored=int(
            manifest.get("candidate_pairs_scored", 0)
        ),
        processed_units=int(manifest.get("processed_units", 0)),
        signature=str(manifest.get("signature", "")),
    )
    if artifact.selected_rows <= 0 or not artifact.part_paths:
        raise ValueError(
            "No speech-manifesto pairs remained after hybrid retrieval."
        )
    return artifact


def select_hybrid_embedding_bm25_pairs_from_joined(
    merged: pd.DataFrame,
    manifesto: pd.DataFrame,
    pairs_per_speech: int,
    min_retrieval_score: float,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
    local_embedding_device: str,
    embedding_cache_db: Path,
    embedding_disk_cache: bool,
    embedding_top_k: int,
    bm25_top_k: int,
    rerank_top_k: int,
    embedding_weight: float,
    min_embedding_score: float,
    min_bm25_score: float,
    checkpoint_dir: Path | None,
    checkpoint_interval: int,
    materialize_checkpoint: bool,
) -> tuple[pd.DataFrame | RetrievalCheckpointArtifact, int]:
    manifesto_by_doc = {
        doc_key: group.reset_index(drop=True)
        for doc_key, group in manifesto.groupby("doc_key", sort=False)
    }
    group_col = retrieval_group_column(merged)
    total_units = merged[group_col].nunique()
    started = time.perf_counter()
    retrieval_config = {
        "pairs_per_speech": pairs_per_speech,
        "min_retrieval_score": min_retrieval_score,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_top_k": embedding_top_k,
        "bm25_top_k": bm25_top_k,
        "rerank_top_k": rerank_top_k,
        "embedding_weight": embedding_weight,
        "min_embedding_score": min_embedding_score,
        "min_bm25_score": min_bm25_score,
        "embedding_batch_size": embedding_batch_size,
    }
    signature = hybrid_retrieval_signature(
        merged, manifesto, group_col, retrieval_config
    )
    if checkpoint_dir is not None and checkpoint_interval > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint, checkpoint_parts = load_hybrid_retrieval_checkpoint(
            checkpoint_dir, signature
        )
    else:
        checkpoint = {
            "version": PROVENANCE_VERSION,
            "signature": signature,
            "processed_units": 0,
            "candidate_pairs_scored": 0,
            "selected_rows": 0,
            "complete": False,
            "parts": [],
        }
        checkpoint_parts = []
    checkpoint["total_units"] = total_units
    resume_after = int(checkpoint["processed_units"])
    if resume_after > total_units:
        raise RuntimeError(
            "Hybrid retrieval checkpoint contains more units than the "
            "current strict-linkage input"
        )
    if resume_after:
        print(
            "Resuming hybrid retrieval after "
            f"{resume_after:,}/{total_units:,} unit(s), with "
            f"{int(checkpoint['selected_rows']):,} selected row(s) saved."
        )
    if bool(checkpoint.get("complete")):
        if resume_after != total_units:
            raise RuntimeError(
                "Hybrid retrieval checkpoint is marked complete but does not "
                "cover every current retrieval unit"
            )
        artifact = retrieval_checkpoint_artifact(
            checkpoint, checkpoint_parts
        )
        if not materialize_checkpoint:
            return artifact, artifact.candidate_pairs_scored
        selected = pd.concat(
            [
                pd.read_parquet(path)
                for path in artifact.part_paths
            ],
            ignore_index=True,
        )
        return selected, artifact.candidate_pairs_scored
    selected_rows: list[dict[str, Any]] = []
    candidate_pair_count = 0
    checkpoint_unit_start = resume_after + 1

    relevant_doc_keys = set(merged["doc_key"].dropna().tolist())
    relevant_manifesto = manifesto[manifesto["doc_key"].isin(relevant_doc_keys)].copy()
    query_texts = merged.drop_duplicates(group_col)["text"].fillna("").astype(str).str.strip().tolist()
    passage_texts = relevant_manifesto["manifesto_text"].fillna("").astype(str).str.strip().tolist()
    print(
        "Embedding hybrid retrieval texts: "
        f"{len(dict.fromkeys(text for text in query_texts if text)):,} unique query text(s), "
        f"{len(dict.fromkeys(text for text in passage_texts if text)):,} unique passage text(s)."
    )
    query_embeddings, passage_embeddings = embed_query_and_passage_texts(
        query_texts=query_texts,
        passage_texts=passage_texts,
        model=embedding_model,
        provider=embedding_provider,
        api_key_env=api_key_env,
        batch_size=embedding_batch_size,
        local_device=local_embedding_device,
        cache_db=embedding_cache_db,
        use_disk_cache=embedding_disk_cache,
    )

    for unit_i, (_, unit_rows) in enumerate(merged.groupby(group_col, sort=False), start=1):
        if unit_i <= resume_after:
            continue
        candidate_parts = []
        bridge_rows_by_doc: dict[Any, dict[str, Any]] = {}
        for bridge_row in unit_rows.to_dict("records"):
            doc_key = bridge_row.get("doc_key")
            doc_manifesto = manifesto_by_doc.get(doc_key)
            if doc_manifesto is None or doc_manifesto.empty:
                continue
            candidate_parts.append(doc_manifesto)
            bridge_rows_by_doc[doc_key] = bridge_row

        if candidate_parts:
            candidates = pd.concat(candidate_parts, ignore_index=True)
            candidate_pair_count += len(candidates)
            speech_text = str(unit_rows["text"].iloc[0]).strip()
            manifesto_texts = candidates["manifesto_text"].fillna("").astype(str).str.strip().tolist()

            speech_vector = query_embeddings.get(speech_text)
            if speech_vector is None:
                embedding_scores = np.zeros(len(candidates), dtype=float)
            else:
                embedding_scores = np.array(
                    [
                        float(np.dot(speech_vector, passage_embeddings[text]))
                        if text in passage_embeddings
                        else 0.0
                        for text in manifesto_texts
                    ],
                    dtype=float,
                )
            bm25_raw_scores = bm25_scores(speech_text, manifesto_texts)

            embedding_indices = top_indices(embedding_scores, embedding_top_k)
            bm25_indices = top_indices(bm25_raw_scores, bm25_top_k)
            union_indices = list(dict.fromkeys(embedding_indices + bm25_indices))
            if union_indices:
                embedding_norm = normalized_scores(embedding_scores)
                bm25_norm = normalized_scores(bm25_raw_scores)
                combined_scores = (
                    embedding_weight * embedding_norm
                    + (1.0 - embedding_weight) * bm25_norm
                )
                final_top_k = (
                    rerank_top_k if rerank_top_k > 0 else pairs_per_speech
                )
                ranked_indices = sorted(
                    union_indices,
                    key=lambda idx: (-float(combined_scores[idx]), idx),
                )
                if min_retrieval_score > 0:
                    ranked_indices = [
                        idx for idx in ranked_indices
                        if float(combined_scores[idx]) >= min_retrieval_score
                    ]
                if min_embedding_score > 0:
                    ranked_indices = [
                        idx for idx in ranked_indices
                        if float(embedding_scores[idx]) >= min_embedding_score
                    ]
                if min_bm25_score > 0:
                    ranked_indices = [
                        idx for idx in ranked_indices
                        if float(bm25_raw_scores[idx]) >= min_bm25_score
                    ]
                if final_top_k > 0:
                    ranked_indices = ranked_indices[:final_top_k]

                embedding_rank_by_index = {
                    idx: rank
                    for rank, idx in enumerate(embedding_indices, start=1)
                }
                bm25_rank_by_index = {
                    idx: rank
                    for rank, idx in enumerate(bm25_indices, start=1)
                }
                for rank, candidate_idx in enumerate(
                    ranked_indices, start=1
                ):
                    manifesto_row = candidates.iloc[
                        int(candidate_idx)
                    ].to_dict()
                    bridge_row = bridge_rows_by_doc.get(
                        manifesto_row.get("doc_key"),
                        unit_rows.iloc[0].to_dict(),
                    )
                    selected_row = dict(bridge_row)
                    selected_row.update(manifesto_row)
                    selected_row["retrieval_score"] = float(
                        combined_scores[candidate_idx]
                    )
                    selected_row["retrieval_rank"] = rank
                    selected_row["pair_selection_method"] = (
                        "hybrid_embedding_bm25"
                    )
                    selected_row["embedding_provider"] = embedding_provider
                    selected_row["embedding_model"] = embedding_model
                    selected_row["embedding_score"] = float(
                        embedding_scores[candidate_idx]
                    )
                    selected_row["embedding_score_normalized"] = float(
                        embedding_norm[candidate_idx]
                    )
                    selected_row["bm25_score"] = float(
                        bm25_raw_scores[candidate_idx]
                    )
                    selected_row["bm25_score_normalized"] = float(
                        bm25_norm[candidate_idx]
                    )
                    selected_row["hybrid_union_size"] = len(union_indices)
                    selected_row["hybrid_embedding_rank"] = (
                        embedding_rank_by_index.get(candidate_idx, pd.NA)
                    )
                    selected_row["hybrid_bm25_rank"] = (
                        bm25_rank_by_index.get(candidate_idx, pd.NA)
                    )
                    selected_rows.append(selected_row)

        should_checkpoint = (
            checkpoint_dir is not None
            and checkpoint_interval > 0
            and (
                unit_i % checkpoint_interval == 0
                or unit_i == total_units
            )
        )
        if should_checkpoint:
            part = save_hybrid_retrieval_checkpoint(
                checkpoint_dir=checkpoint_dir,
                manifest=checkpoint,
                selected_rows=selected_rows,
                unit_start=checkpoint_unit_start,
                unit_end=unit_i,
                candidate_pairs=candidate_pair_count,
            )
            if part is not None:
                checkpoint_parts.append(part)
            selected_rows = []
            candidate_pair_count = 0
            checkpoint_unit_start = unit_i + 1

        if unit_i == 1 or unit_i % 1000 == 0 or unit_i == total_units:
            elapsed = time.perf_counter() - started
            print(
                f"Hybrid retrieval {unit_i:,}/{total_units:,} units: "
                f"selected={int(checkpoint['selected_rows']) + len(selected_rows):,}, "
                f"candidates_scored={int(checkpoint['candidate_pairs_scored']) + candidate_pair_count:,}, "
                f"elapsed={elapsed:.1f}s"
            )

    if checkpoint_dir is not None and checkpoint_interval > 0:
        checkpoint["complete"] = True
        manifest_path = checkpoint_dir / "manifest.json"
        atomic_write_json(manifest_path, checkpoint)
        artifact = retrieval_checkpoint_artifact(
            checkpoint, checkpoint_parts
        )
        total_candidate_pairs = artifact.candidate_pairs_scored
        query_embeddings.clear()
        passage_embeddings.clear()
        _EMBEDDING_CACHE.clear()
        gc.collect()
        if not materialize_checkpoint:
            return artifact, total_candidate_pairs
        selected = pd.concat(
            [pd.read_parquet(path) for path in artifact.part_paths],
            ignore_index=True,
        )
    else:
        selected = pd.DataFrame(selected_rows)
        total_candidate_pairs = candidate_pair_count
    if selected.empty:
        raise ValueError(
            "No speech-manifesto pairs remained after hybrid embedding/BM25 retrieval. "
            "Lower --min-retrieval-score or rerank limits, or use another --pair-selection mode."
        )
    return selected, total_candidate_pairs

def select_pairs(
    pairs: pd.DataFrame,
    pair_selection: str,
    pairs_per_speech: int,
    random_state: int,
    min_retrieval_score: float,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
    local_embedding_device: str,
) -> pd.DataFrame:
    if pair_selection == "random":
        return select_random_pairs(pairs, pairs_per_speech, random_state)
    if pair_selection == "tfidf_topk":
        return select_tfidf_topk_pairs(pairs, pairs_per_speech, min_retrieval_score)
    if pair_selection in {"embedding_topk", "openai_embedding_topk"}:
        return select_embedding_topk_pairs(
            pairs=pairs,
            pairs_per_speech=pairs_per_speech,
            min_retrieval_score=min_retrieval_score,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            api_key_env=api_key_env,
            local_embedding_device=local_embedding_device,
        )
    raise ValueError(f"Unsupported pair selection method: {pair_selection}")

def numeric_summary(series: pd.Series, prefix: str) -> dict[str, float | int | pd.NA]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": pd.NA,
            f"{prefix}_std": pd.NA,
            f"{prefix}_min": pd.NA,
            f"{prefix}_p10": pd.NA,
            f"{prefix}_p25": pd.NA,
            f"{prefix}_median": pd.NA,
            f"{prefix}_p75": pd.NA,
            f"{prefix}_p90": pd.NA,
            f"{prefix}_max": pd.NA,
        }
    quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()) if values.size > 1 else 0.0,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p10": float(quantiles.loc[0.10]),
        f"{prefix}_p25": float(quantiles.loc[0.25]),
        f"{prefix}_median": float(quantiles.loc[0.50]),
        f"{prefix}_p75": float(quantiles.loc[0.75]),
        f"{prefix}_p90": float(quantiles.loc[0.90]),
        f"{prefix}_max": float(values.max()),
    }


def count_pairs_by_party(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["party", f"{prefix}_pairs", f"{prefix}_speeches"])
    return (
        data.groupby("party", as_index=False)
        .agg(
            **{
                f"{prefix}_pairs": ("plda_doc_id", "size"),
                f"{prefix}_speeches": ("plda_doc_id", "nunique"),
                f"{prefix}_months": ("month", "nunique"),
                f"{prefix}_manifesto_docs": ("doc_key", "nunique"),
            }
        )
        .reset_index(drop=True)
    )


def build_pair_quality_diagnostics(
    sampled_speeches: pd.DataFrame,
    bridge_matches: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    pair_selection: str,
    pairs_per_speech: int,
    min_retrieval_score: float,
) -> dict[str, pd.DataFrame]:
    group_col = retrieval_group_column(candidate_pairs)
    sampled_speech_ids = set(sampled_speeches["plda_doc_id"].dropna().tolist())
    bridge_speech_ids = set(bridge_matches["plda_doc_id"].dropna().tolist())
    candidate_speech_ids = set(candidate_pairs["plda_doc_id"].dropna().tolist())
    selected_speech_ids = set(selected_pairs["plda_doc_id"].dropna().tolist())
    sampled_unit_ids = set(sampled_speeches[group_col].dropna().tolist()) if group_col in sampled_speeches.columns else sampled_speech_ids
    bridge_unit_ids = set(bridge_matches[group_col].dropna().tolist()) if group_col in bridge_matches.columns else bridge_speech_ids
    candidate_unit_ids = set(candidate_pairs[group_col].dropna().tolist())
    selected_unit_ids = set(selected_pairs[group_col].dropna().tolist())

    retrieval_summary = (
        numeric_summary(selected_pairs["retrieval_score"], "retrieval_score")
        if "retrieval_score" in selected_pairs.columns
        else numeric_summary(pd.Series(dtype=float), "retrieval_score")
    )
    overall = pd.DataFrame(
        [
            {
                "pair_selection_method": pair_selection,
                "pairs_per_speech": pairs_per_speech,
                "min_retrieval_score": min_retrieval_score,
                "sampled_speeches": len(sampled_speech_ids),
                "sampled_units": len(sampled_unit_ids),
                "sampled_parties": sampled_speeches["party"].nunique(),
                "sampled_months": sampled_speeches["month"].nunique(),
                "bridge_matched_speeches": len(bridge_speech_ids),
                "bridge_matched_units": len(bridge_unit_ids),
                "bridge_matched_parties": bridge_matches["party"].nunique(),
                "bridge_matched_months": bridge_matches["month"].nunique(),
                "candidate_speeches": len(candidate_speech_ids),
                "candidate_units": len(candidate_unit_ids),
                "candidate_pairs": len(candidate_pairs),
                "candidate_parties": candidate_pairs["party"].nunique(),
                "candidate_months": candidate_pairs["month"].nunique(),
                "candidate_manifesto_docs": candidate_pairs["doc_key"].nunique(),
                "candidate_manifesto_quasi": candidate_pairs["quasi_sentence_id"].nunique(),
                "selected_speeches": len(selected_speech_ids),
                "selected_units": len(selected_unit_ids),
                "selected_pairs": len(selected_pairs),
                "selected_parties": selected_pairs["party"].nunique(),
                "selected_months": selected_pairs["month"].nunique(),
                "selected_manifesto_docs": selected_pairs["doc_key"].nunique(),
                "selected_manifesto_quasi": selected_pairs["quasi_sentence_id"].nunique(),
                "sampled_to_bridge_speech_retention": len(bridge_speech_ids)
                / max(len(sampled_speech_ids), 1),
                "bridge_to_candidate_speech_retention": len(candidate_speech_ids)
                / max(len(bridge_speech_ids), 1),
                "candidate_to_selected_speech_retention": len(selected_speech_ids)
                / max(len(candidate_speech_ids), 1),
                "candidate_to_selected_unit_retention": len(selected_unit_ids)
                / max(len(candidate_unit_ids), 1),
                "candidate_to_selected_pair_retention": len(selected_pairs)
                / max(len(candidate_pairs), 1),
                **retrieval_summary,
            }
        ]
    )

    sampled_by_party = (
        sampled_speeches.groupby("party", as_index=False)
        .agg(
            sampled_speeches=("plda_doc_id", "nunique"),
            sampled_months=("month", "nunique"),
        )
        .reset_index(drop=True)
    )
    bridge_by_party = count_pairs_by_party(bridge_matches, "bridge")
    candidate_by_party = count_pairs_by_party(candidate_pairs, "candidate")
    selected_by_party = count_pairs_by_party(selected_pairs, "selected")
    by_party = (
        sampled_by_party.merge(bridge_by_party, on="party", how="left")
        .merge(candidate_by_party, on="party", how="left")
        .merge(selected_by_party, on="party", how="left")
    )
    count_cols = [col for col in by_party.columns if col != "party"]
    by_party[count_cols] = by_party[count_cols].fillna(0)
    by_party["candidate_to_selected_speech_retention"] = (
        by_party["selected_speeches"] / by_party["candidate_speeches"].replace(0, pd.NA)
    )
    by_party["candidate_to_selected_pair_retention"] = (
        by_party["selected_pairs"] / by_party["candidate_pairs"].replace(0, pd.NA)
    )

    candidate_by_speech = (
        candidate_pairs.groupby(group_col, as_index=False)
        .agg(
            candidate_pairs=(group_col, "size"),
            candidate_manifesto_docs=("doc_key", "nunique"),
            candidate_manifesto_quasi=("quasi_sentence_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    selected_for_speech = selected_pairs.copy()
    selected_aggs: dict[str, tuple[str, str]] = {
        "selected_pairs": ("plda_doc_id", "size"),
        "selected_manifesto_docs": ("doc_key", "nunique"),
        "selected_manifesto_quasi": ("quasi_sentence_id", "nunique"),
    }
    if "retrieval_score" in selected_pairs.columns:
        selected_for_speech["retrieval_score_numeric"] = pd.to_numeric(
            selected_for_speech["retrieval_score"],
            errors="coerce",
        )
        selected_aggs.update(
            {
                "retrieval_score_max": ("retrieval_score_numeric", "max"),
                "retrieval_score_mean": ("retrieval_score_numeric", "mean"),
                "retrieval_score_min": ("retrieval_score_numeric", "min"),
            }
        )
    if "retrieval_rank" in selected_pairs.columns:
        selected_aggs["best_retrieval_rank"] = ("retrieval_rank", "min")
    selected_by_speech = (
        selected_for_speech.groupby(group_col, as_index=False)
        .agg(**selected_aggs)
        .reset_index(drop=True)
    )
    speech_cols = ["plda_doc_id", "party", "month"]
    optional_speech_cols = [
        "speech_unit",
        "speech_segment_id",
        "retrieval_unit_id",
        "speech_segment_index",
        "segment_start_sentence",
        "segment_end_sentence",
        "speech_selected_topic_mass",
        "speech_word_count_for_nli",
        "speech_is_procedural",
    ]
    speech_cols.extend([col for col in optional_speech_cols if col in sampled_speeches.columns])
    by_speech = (
        sampled_speeches[speech_cols]
        .drop_duplicates(group_col if group_col in sampled_speeches.columns else "plda_doc_id")
        .merge(candidate_by_speech, on=group_col, how="left")
        .merge(selected_by_speech, on=group_col, how="left")
    )
    count_cols = [
        col
        for col in [
            "candidate_pairs",
            "candidate_manifesto_docs",
            "candidate_manifesto_quasi",
            "selected_pairs",
            "selected_manifesto_docs",
            "selected_manifesto_quasi",
        ]
        if col in by_speech.columns
    ]
    by_speech[count_cols] = by_speech[count_cols].fillna(0)
    by_speech["has_candidate_pair"] = by_speech["candidate_pairs"] > 0
    by_speech["has_selected_pair"] = by_speech["selected_pairs"] > 0
    by_speech["candidate_to_selected_pair_retention"] = (
        by_speech["selected_pairs"] / by_speech["candidate_pairs"].replace(0, pd.NA)
    )

    return {
        "pair_quality_overall": overall,
        "pair_quality_by_party": by_party,
        "pair_quality_by_speech": by_speech,
    }


def build_pairs(
    speeches: pd.DataFrame,
    manifesto_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    pairs_per_speech: int,
    random_state: int,
    pair_selection: str,
    min_retrieval_score: float,
    embedding_provider: str,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
    local_embedding_device: str,
    embedding_cache_db: Path,
    embedding_disk_cache: bool,
    hybrid_embedding_top_k: int,
    hybrid_bm25_top_k: int,
    hybrid_rerank_top_k: int,
    hybrid_embedding_weight: float,
    hybrid_min_embedding_score: float,
    hybrid_min_bm25_score: float,
    retrieval_checkpoint_dir: Path | None,
    retrieval_checkpoint_interval: int,
    retrieval_only: bool,
) -> tuple[
    pd.DataFrame | RetrievalCheckpointArtifact,
    dict[str, pd.DataFrame],
]:
    bridge_cols = [
        "speech_party",
        "date",
        "doc_key",
        "manifesto_partyname",
        "manifesto_partyabbrev",
        "manifesto_date",
        "manifesto_effective_date",
        "selection_method",
    ]
    bridge = bridge_df[[col for col in bridge_cols if col in bridge_df.columns]].copy()
    required_bridge = {
        "speech_party", "date", "doc_key", "manifesto_effective_date",
        "selection_method",
    }
    missing_bridge = sorted(required_bridge - set(bridge.columns))
    if missing_bridge:
        raise ValueError(
            "Strict temporal retrieval requires a speech-date bridge; "
            f"missing columns: {missing_bridge}"
        )
    bridge["speech_party"] = bridge["speech_party"].astype(str).str.strip()
    bridge["date"] = pd.to_datetime(
        bridge["date"], errors="coerce"
    ).dt.normalize()
    bridge["manifesto_effective_date"] = pd.to_datetime(
        bridge["manifesto_effective_date"], errors="coerce"
    ).dt.normalize()
    speeches = speeches.copy()
    speeches["date"] = pd.to_datetime(
        speeches["date"], errors="coerce"
    ).dt.normalize()
    invalid_date = bridge["date"].isna() | bridge[
        "manifesto_effective_date"
    ].isna()
    if invalid_date.any():
        raise ValueError(
            "Strict temporal bridge contains "
            f"{int(invalid_date.sum())} invalid speech/manifesto date rows"
        )
    duplicate = bridge.duplicated(["speech_party", "date"], keep=False)
    if duplicate.any():
        conflict = (
            bridge.loc[duplicate]
            .groupby(["speech_party", "date"])["doc_key"]
            .nunique()
            .gt(1)
        )
        if conflict.any():
            raise ValueError(
                "Strict temporal bridge maps a party-date to multiple manifestos"
            )
        bridge = bridge.drop_duplicates(
            ["speech_party", "date"], keep="first"
        )
    strict_method = bridge["selection_method"].eq(
        "latest_manifesto_on_or_before_speech_date"
    )
    no_future = bridge["manifesto_effective_date"].le(bridge["date"])
    strict_bridge = bridge.loc[strict_method & no_future].copy()
    rejected = len(bridge) - len(strict_bridge)
    if strict_bridge.empty:
        raise ValueError(
            "No latest-preceding manifesto rows remain in the strict date bridge"
        )
    merged = speeches.merge(
        strict_bridge,
        left_on=["party", "date"],
        right_on=["speech_party", "date"],
        how="inner",
        validate="m:1",
    )
    print(
        "Strict latest-preceding bridge "
        f"excluded {rejected:,} fallback/future bridge row(s) and retained "
        f"{merged['plda_doc_id'].nunique():,}/"
        f"{speeches['plda_doc_id'].nunique():,} sampled speeches "
        f"and {merged[retrieval_group_column(merged)].nunique():,}/"
        f"{speeches[retrieval_group_column(speeches)].nunique():,} retrieval units "
        f"across {merged['party'].nunique():,}/{speeches['party'].nunique():,} parties."
    )
    if merged.empty:
        raise ValueError(
            "No sampled speeches matched a strict latest-preceding manifesto"
        )
    if (
        merged["manifesto_effective_date"]
        .gt(merged["date"])
        .any()
    ):
        raise AssertionError(
            "Future manifesto survived strict speech-date linkage"
        )
    if not merged["selection_method"].eq(
        "latest_manifesto_on_or_before_speech_date"
    ).all():
        raise AssertionError(
            "Non-latest-preceding manifesto survived strict linkage"
        )

    manifesto_cols = [
        "doc_key",
        "quasi_sentence_id",
        "cmp_code",
        "eu_code",
        "text",
        "predicted_topic",
    ]
    manifesto = manifesto_df[
        [col for col in manifesto_cols if col in manifesto_df.columns]
    ].rename(columns={"text": "manifesto_text"})

    streamed_candidate_pair_count: int | None = None
    if pair_selection == "hybrid_embedding_bm25":
        pairs, streamed_candidate_pair_count = select_hybrid_embedding_bm25_pairs_from_joined(
            merged=merged,
            manifesto=manifesto,
            pairs_per_speech=pairs_per_speech,
            min_retrieval_score=min_retrieval_score,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            api_key_env=api_key_env,
            local_embedding_device=local_embedding_device,
            embedding_cache_db=embedding_cache_db,
            embedding_disk_cache=embedding_disk_cache,
            embedding_top_k=hybrid_embedding_top_k,
            bm25_top_k=hybrid_bm25_top_k,
            rerank_top_k=hybrid_rerank_top_k,
            embedding_weight=hybrid_embedding_weight,
            min_embedding_score=hybrid_min_embedding_score,
            min_bm25_score=hybrid_min_bm25_score,
            checkpoint_dir=retrieval_checkpoint_dir,
            checkpoint_interval=retrieval_checkpoint_interval,
            materialize_checkpoint=not retrieval_only,
        )
        if isinstance(pairs, RetrievalCheckpointArtifact):
            print(
                "Manifesto claim hybrid retrieval checkpoint retained "
                f"{pairs.selected_rows:,} pairs across "
                f"{pairs.processed_units:,} strict retrieval units after "
                f"scoring {pairs.candidate_pairs_scored:,} candidate pairs."
            )
            return pairs, {}
        print(
            f"Manifesto claim hybrid retrieval retained {pairs['plda_doc_id'].nunique():,}/"
            f"{merged['plda_doc_id'].nunique():,} bridge-matched speeches "
            f"using {pairs['doc_key'].nunique():,}/{merged['doc_key'].nunique():,} "
            f"linked manifesto documents after scoring {streamed_candidate_pair_count:,} "
            "candidate pairs."
        )
        candidate_pairs = pairs.copy()
    elif pair_selection == "tfidf_topk":
        pairs, streamed_candidate_pair_count = select_tfidf_topk_pairs_from_joined(
            merged=merged,
            manifesto=manifesto,
            pairs_per_speech=pairs_per_speech,
            min_retrieval_score=min_retrieval_score,
        )
        print(
            f"Manifesto claim streaming retained {pairs['plda_doc_id'].nunique():,}/"
            f"{merged['plda_doc_id'].nunique():,} bridge-matched speeches "
            f"using {pairs['doc_key'].nunique():,}/{merged['doc_key'].nunique():,} "
            f"linked manifesto documents after scoring {streamed_candidate_pair_count:,} "
            "candidate pairs."
        )
        candidate_pairs = pairs.copy()
    else:
        pairs = merged.merge(manifesto, on="doc_key", how="inner")
        print(
            f"Manifesto claim join retained {pairs['plda_doc_id'].nunique():,}/"
            f"{merged['plda_doc_id'].nunique():,} bridge-matched speeches "
            f"using {pairs['doc_key'].nunique():,}/{merged['doc_key'].nunique():,} "
            "linked manifesto documents."
        )
        if pairs.empty:
            raise ValueError("No speech-manifesto pairs remained after joining doc_key.")

        candidate_pairs = pairs.copy()
        pairs = select_pairs(
            pairs=pairs,
            pair_selection=pair_selection,
            pairs_per_speech=pairs_per_speech,
            random_state=random_state,
            min_retrieval_score=min_retrieval_score,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            api_key_env=api_key_env,
            local_embedding_device=local_embedding_device,
        )
    print(
        f"Selected {len(pairs):,} pair(s) from "
        f"{pairs['plda_doc_id'].nunique():,} speech row(s) and "
        f"{pairs[retrieval_group_column(pairs)].nunique():,} retrieval unit(s) "
        f"using {pair_selection!r}."
    )
    pair_quality = build_pair_quality_diagnostics(
        sampled_speeches=speeches,
        bridge_matches=merged,
        candidate_pairs=candidate_pairs,
        selected_pairs=pairs,
        pair_selection=pair_selection,
        pairs_per_speech=pairs_per_speech,
        min_retrieval_score=min_retrieval_score,
    )
    if streamed_candidate_pair_count is not None:
        overall = pair_quality["pair_quality_overall"]
        overall.loc[0, "candidate_pairs"] = streamed_candidate_pair_count
        overall.loc[0, "candidate_to_selected_pair_retention"] = len(pairs) / max(
            streamed_candidate_pair_count,
            1,
        )
        overall.loc[0, "hybrid_embedding_weight"] = hybrid_embedding_weight
        overall.loc[0, "hybrid_min_embedding_score"] = hybrid_min_embedding_score
        overall.loc[0, "hybrid_min_bm25_score"] = hybrid_min_bm25_score
        overall.loc[
            0,
            "candidate_pair_diagnostics_note",
        ] = "candidate_pairs counted during streaming; detailed candidate diagnostics are based on selected pairs"

    pairs = pairs.rename(
        columns={
            "text": "speech_text",
            "topic_label": "speech_topic_label",
            "predicted_topic": "manifesto_predicted_topic",
        }
    )
    pairs.insert(0, "nli_pair_id", range(len(pairs)))
    return pairs.reset_index(drop=True), pair_quality


def normalize_label(label: object) -> str:
    normalized = str(label).strip().lower()
    if normalized.startswith("label_"):
        return normalized
    return normalized


def nli_label_order(model: Any) -> list[str]:
    id2label = getattr(model.config, "id2label", None) or {}
    labels = [normalize_label(id2label.get(i, i)) for i in range(model.config.num_labels)]
    aliases = {
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
    }
    return [aliases.get(label, label) for label in labels]


def torch_mps_available(torch_module: Any) -> bool:
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def resolve_torch_device(torch_module: Any, device_name: str) -> Any:
    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = torch_mps_available(torch_module)
    requested = device_name.lower()

    if requested == "cuda":
        if cuda_available:
            return torch_module.device("cuda")
        print("Requested --device cuda, but CUDA is not available. Falling back to CPU.")
        return torch_module.device("cpu")
    if requested == "mps":
        if mps_available:
            return torch_module.device("mps")
        print("Requested --device mps, but Apple MPS is not available. Falling back to CPU.")
        return torch_module.device("cpu")
    if requested == "cpu":
        return torch_module.device("cpu")
    if cuda_available:
        return torch_module.device("cuda")
    if mps_available:
        return torch_module.device("mps")
    return torch_module.device("cpu")


def print_torch_device_summary(torch_module: Any, device: Any) -> None:
    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = torch_mps_available(torch_module)
    selected_device = str(device)
    style = "bold green" if selected_device.startswith("cuda") else "bold yellow"
    status = "CUDA ENABLED" if selected_device.startswith("cuda") else "CUDA NOT USED"

    device_detail = ""
    if cuda_available:
        try:
            device_detail = f" | GPU={torch_module.cuda.get_device_name(0)}"
        except Exception:
            device_detail = " | GPU=<name unavailable>"

    message = (
        f"{status}: selected_device={selected_device} | "
        f"cuda_available={cuda_available} | mps_available={mps_available}"
        f"{device_detail}"
    )
    try:
        from rich.console import Console

        Console().print(message, style=style)
    except Exception:
        print(f"[{status}] {message}")


_NLI_COMPONENT_CACHE: dict[tuple[str, str], tuple[Any, Any, Any, Any, list[str]]] = {}


def load_nli_components(model_name: str, device_name: str) -> tuple[Any, Any, Any, Any, list[str]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = resolve_torch_device(torch, device_name)
    cache_key = (model_name, str(device))
    cached = _NLI_COMPONENT_CACHE.get(cache_key)
    if cached is not None:
        print(f"Reusing NLI tokenizer/model on {device}: {model_name}")
        return cached

    print_torch_device_summary(torch, device)
    print(f"Loading NLI tokenizer/model on {device}: {model_name}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    labels = nli_label_order(model)
    components = (torch, tokenizer, model, device, labels)
    _NLI_COMPONENT_CACHE[cache_key] = components
    print(
        f"Loaded NLI model in {time.perf_counter() - load_start:.1f}s. "
        f"Labels: {', '.join(labels)}"
    )
    return components


def nli_pair_fingerprints(pairs: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in (
            "nli_pair_id",
            "retrieval_unit_id",
            "speech_segment_id",
            "plda_doc_id",
            "doc_key",
            "manifesto_text",
            "speech_text",
        )
        if column in pairs.columns
    ]
    if "manifesto_text" not in columns or "speech_text" not in columns:
        raise ValueError("NLI pair fingerprints require manifesto_text and speech_text")
    hashes = pd.util.hash_pandas_object(
        pairs[columns], index=True
    ).to_numpy(dtype=np.uint64, copy=False)
    return [f"{int(value):016x}" for value in hashes]


def nli_checkpoint_state_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".state.json")


def nli_run_spec(
    pairs: pd.DataFrame,
    model_name: str,
    max_length: int,
    labels: list[str],
) -> dict[str, Any]:
    model_path = Path(model_name)
    model_identity: dict[str, Any] = {"name_or_path": model_name}
    if model_path.exists():
        model_identity["artifact_fingerprint"] = (
            sampled_file_content_fingerprint(model_path)
            if model_path.is_file()
            else canonical_json_fingerprint(
                {
                    child.relative_to(model_path).as_posix():
                    sampled_file_content_fingerprint(child)
                    for child in sorted(model_path.rglob("*"))
                    if child.is_file()
                }
            )
        )
    fingerprint_columns = [
        column
        for column in (
            "nli_pair_id",
            "retrieval_unit_id",
            "speech_segment_id",
            "plda_doc_id",
            "doc_key",
            "manifesto_text",
            "speech_text",
        )
        if column in pairs.columns
    ]
    return {
        "provenance_version": PROVENANCE_VERSION,
        "model": model_identity,
        "max_length": int(max_length),
        "labels": labels,
        "pair_order": ["manifesto_text", "speech_text"],
        "pair_rows": len(pairs),
        "pair_data_fingerprint": dataframe_fingerprint(
            pairs, fingerprint_columns
        ),
    }


def load_nli_checkpoint(
    path: Path,
    expected_labels: list[str],
    expected_pair_fingerprints: list[str],
    run_spec: dict[str, Any],
) -> tuple[list[dict[str, float | str]], dict[str, Any]]:
    state_path = nli_checkpoint_state_path(path)
    expected_signature = canonical_json_fingerprint(run_spec)
    if not path.exists() and not state_path.exists():
        return [], {
            "provenance_version": PROVENANCE_VERSION,
            "run_signature": expected_signature,
            "run_spec": run_spec,
            "processed_rows": 0,
            "output_bytes": 0,
            "pending_chunk": None,
            "complete": False,
        }
    if not path.exists() or not state_path.exists():
        raise RuntimeError(
            f"Incomplete NLI checkpoint/state pair: {path}. Remove both files "
            "or choose a new output path."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("provenance_version") != PROVENANCE_VERSION
        or state.get("run_signature") != expected_signature
        or state.get("run_spec") != run_spec
    ):
        raise RuntimeError(
            f"NLI checkpoint model/data/configuration mismatch: {path}. "
            "Remove it or choose a new output path."
        )
    pending = state.get("pending_chunk")
    if pending:
        before = int(pending["output_bytes_before"])
        actual = path.stat().st_size
        if actual < before:
            raise RuntimeError(f"NLI checkpoint is shorter than durable state: {path}")
        if actual != before:
            with path.open("r+b") as handle:
                handle.truncate(before)
                handle.flush()
                os.fsync(handle.fileno())
        state.update(
            {
                "processed_rows": int(pending["row_start"]),
                "output_bytes": before,
                "pending_chunk": None,
                "complete": False,
            }
        )
        atomic_write_json(state_path, state)
    if int(state.get("output_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"NLI checkpoint byte size differs from state: {path}")
    if state.get("complete"):
        if int(state.get("processed_rows", -1)) != len(
            expected_pair_fingerprints
        ):
            raise RuntimeError(
                f"Completed NLI checkpoint does not cover all current pairs: {path}"
            )
        if (
            state.get("output_sampled_content_sha256")
            != sampled_file_content_fingerprint(path)
        ):
            raise RuntimeError(
                f"Completed NLI checkpoint content fingerprint changed: {path}"
            )
    checkpoint = pd.read_csv(path, low_memory=False)
    expected_columns = (
        ["pair_fingerprint"]
        + [f"nli_prob_{label}" for label in expected_labels]
        + ["nli_label"]
    )
    missing = [column for column in expected_columns if column not in checkpoint.columns]
    if missing:
        raise ValueError(
            f"NLI checkpoint has incompatible schema: {path}. Missing columns: {missing}"
        )
    checkpoint = checkpoint[expected_columns].copy()
    if len(checkpoint) != int(state.get("processed_rows", -1)):
        raise RuntimeError(
            f"NLI checkpoint row count differs from durable state: {path}"
        )
    observed_fingerprints = checkpoint["pair_fingerprint"].astype(str).tolist()
    if observed_fingerprints != expected_pair_fingerprints[: len(checkpoint)]:
        raise RuntimeError(
            f"NLI checkpoint pair identities/order differ from current pairs: {path}"
        )
    rows = checkpoint.drop(columns="pair_fingerprint").to_dict("records")
    print(f"Loaded {len(rows):,} NLI checkpoint row(s) from: {path}")
    return rows, state


def append_nli_checkpoint(
    path: Path,
    rows: list[dict[str, float | str]],
    pair_fingerprints: list[str],
    state: dict[str, Any],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state_path = nli_checkpoint_state_path(path)
    output_bytes_before = path.stat().st_size if path.exists() else 0
    row_start = int(state.get("processed_rows", 0))
    state.update(
        {
            "pending_chunk": {
                "row_start": row_start,
                "row_end": row_start + len(rows),
                "output_bytes_before": output_bytes_before,
            },
            "complete": False,
        }
    )
    atomic_write_json(state_path, state)
    checkpoint_df = pd.DataFrame(rows)
    checkpoint_df.insert(0, "pair_fingerprint", pair_fingerprints)
    checkpoint_df.to_csv(path, mode="a", header=not path.exists(), index=False)
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    state.update(
        {
            "processed_rows": row_start + len(rows),
            "output_bytes": path.stat().st_size,
            "pending_chunk": None,
            "complete": False,
        }
    )
    atomic_write_json(state_path, state)


def classify_pairs(
    pairs: pd.DataFrame,
    model_name: str,
    batch_size: int,
    device_name: str,
    max_length: int,
    checkpoint_path: Path | None,
    checkpoint_interval: int,
) -> pd.DataFrame:
    torch, tokenizer, model, device, labels = load_nli_components(model_name, device_name)

    rows: list[dict[str, float | str]] = []
    checkpoint_buffer: list[dict[str, float | str]] = []
    pair_fingerprints = nli_pair_fingerprints(pairs)
    checkpoint_state: dict[str, Any] | None = None
    if checkpoint_path is not None:
        run_spec = nli_run_spec(pairs, model_name, max_length, labels)
        rows, checkpoint_state = load_nli_checkpoint(
            checkpoint_path,
            labels,
            pair_fingerprints,
            run_spec,
        )
        if len(rows) > len(pairs):
            raise ValueError(
                f"NLI checkpoint has more rows than current pairs: {checkpoint_path} "
                f"({len(rows):,} > {len(pairs):,}). Remove it or use a different output path."
            )

    total_batches = (len(pairs) + batch_size - 1) // batch_size
    completed_pairs = len(rows)
    if completed_pairs:
        completed_batches = completed_pairs // batch_size
        print(
            f"Resuming NLI classification after {completed_pairs:,}/{len(pairs):,} "
            f"pairs ({completed_batches:,}/{total_batches:,} full batches)."
        )
    infer_start = time.perf_counter()
    last_checkpoint_at = completed_pairs
    with torch.no_grad():
        for batch_i, start in enumerate(range(0, len(pairs), batch_size), start=1):
            if start < completed_pairs:
                continue
            batch = pairs.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["manifesto_text"].astype(str).tolist(),
                batch["speech_text"].astype(str).tolist(),
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu()
            for prob in probs:
                row = {f"nli_prob_{label}": float(value) for label, value in zip(labels, prob)}
                row["nli_label"] = max(row, key=row.get).replace("nli_prob_", "")
                rows.append(row)
                checkpoint_buffer.append(row)

            pairs_done = len(rows)
            if (
                checkpoint_path is not None
                and checkpoint_interval > 0
                and pairs_done - last_checkpoint_at >= checkpoint_interval
            ):
                buffer_start = pairs_done - len(checkpoint_buffer)
                append_nli_checkpoint(
                    checkpoint_path,
                    checkpoint_buffer,
                    pair_fingerprints[buffer_start:pairs_done],
                    checkpoint_state,
                )
                checkpoint_buffer.clear()
                last_checkpoint_at = pairs_done
                print(f"Saved NLI checkpoint at {pairs_done:,}/{len(pairs):,}: {checkpoint_path}")

            if batch_i == 1 or batch_i % 10 == 0 or batch_i == total_batches:
                elapsed = time.perf_counter() - infer_start
                seconds_per_pair = elapsed / max(pairs_done - completed_pairs, 1)
                remaining_pairs = len(pairs) - pairs_done
                eta = remaining_pairs * seconds_per_pair
                print(
                    f"NLI batch {batch_i:,}/{total_batches:,}: "
                    f"{pairs_done:,}/{len(pairs):,} pairs done, "
                    f"elapsed={elapsed:.1f}s, eta={eta:.1f}s"
                )

    if checkpoint_path is not None and checkpoint_buffer:
        buffer_start = len(rows) - len(checkpoint_buffer)
        append_nli_checkpoint(
            checkpoint_path,
            checkpoint_buffer,
            pair_fingerprints[buffer_start:len(rows)],
            checkpoint_state,
        )
        print(f"Saved final NLI checkpoint at {len(rows):,}/{len(pairs):,}: {checkpoint_path}")
    if checkpoint_path is not None and checkpoint_state is not None:
        checkpoint_state["complete"] = len(rows) == len(pairs)
        if checkpoint_path.exists():
            checkpoint_state["output_bytes"] = checkpoint_path.stat().st_size
            checkpoint_state["output_sampled_content_sha256"] = (
                sampled_file_content_fingerprint(checkpoint_path)
            )
        atomic_write_json(
            nli_checkpoint_state_path(checkpoint_path),
            checkpoint_state,
        )

    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summarize_distribution(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    prob_cols = [col for col in data.columns if col.startswith("nli_prob_")]
    label_counts = (
        data.groupby(group_cols + ["nli_label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        if group_cols
        else data.groupby("nli_label").size().to_frame().T
    )
    if not group_cols:
        label_counts.index = pd.Index([0])

    summary = (
        data.groupby(group_cols, dropna=False)
        .agg(
            n_pairs=("nli_pair_id", "size"),
            n_speeches=("plda_doc_id", "nunique"),
            n_manifesto_quasi=("quasi_sentence_id", "nunique"),
        )
        if group_cols
        else pd.DataFrame(
            {
                "n_pairs": [len(data)],
                "n_speeches": [data["plda_doc_id"].nunique()],
                "n_manifesto_quasi": [data["quasi_sentence_id"].nunique()],
            }
        )
    )
    mean_probs = (
        data.groupby(group_cols, dropna=False)[prob_cols].mean()
        if group_cols
        else pd.DataFrame(data[prob_cols].mean()).T
    )
    if not group_cols:
        mean_probs.index = pd.Index([0])

    result = summary.join(mean_probs).join(label_counts.add_prefix("nli_count_"))
    for column in [col for col in result.columns if col.startswith("nli_count_")]:
        label = column.replace("nli_count_", "")
        result[f"nli_share_{label}"] = result[column] / result["n_pairs"]
    return result.reset_index(drop=not group_cols)


def output_paths(output_dir: Path, country_code: str, topic: str, model_name: str) -> dict[str, Path]:
    topic_token = sanitize_filename_token(topic)
    model_token = nli_model_token(model_name)
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{country_code}_{topic_token}_nli_{model_token}"
    return {
        "pairs": country_dir / f"{stem}_pairs.csv",
        "overall": country_dir / f"{stem}_summary_overall.csv",
        "party": country_dir / f"{stem}_summary_by_party.csv",
        "party_month": country_dir / f"{stem}_summary_by_party_month.csv",
        "diagnostics": country_dir / f"{stem}_speech_diagnostics.csv",
        "pair_quality_overall": country_dir / f"{stem}_pair_quality_overall.csv",
        "pair_quality_by_party": country_dir / f"{stem}_pair_quality_by_party.csv",
        "pair_quality_by_speech": country_dir / f"{stem}_pair_quality_by_speech.csv",
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.min_speech_words < 0:
        raise ValueError("--min-speech-words must be >= 0.")
    if args.min_speech_topic_mass < 0:
        raise ValueError("--min-speech-topic-mass must be >= 0.")
    if args.min_speech_topic_mass > 1:
        raise ValueError("--min-speech-topic-mass must be <= 1.")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0.")
    if args.pairs_per_speech < 0:
        raise ValueError("--pairs-per-speech must be >= 0.")
    if args.segment_window_size <= 0:
        raise ValueError("--segment-window-size must be > 0.")
    if args.segment_stride <= 0:
        raise ValueError("--segment-stride must be > 0.")
    if args.min_segment_words < 0:
        raise ValueError("--min-segment-words must be >= 0.")
    if args.max_segment_words < 0:
        raise ValueError("--max-segment-words must be >= 0.")
    if args.max_segment_words and args.max_segment_words < args.min_segment_words:
        raise ValueError("--max-segment-words must be >= --min-segment-words, or 0.")
    if args.min_retrieval_score < 0:
        raise ValueError("--min-retrieval-score must be >= 0.")
    if args.min_retrieval_score > 1:
        raise ValueError("--min-retrieval-score must be <= 1.")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0.")
    if args.hybrid_embedding_top_k < 0:
        raise ValueError("--hybrid-embedding-top-k must be >= 0.")
    if args.hybrid_bm25_top_k < 0:
        raise ValueError("--hybrid-bm25-top-k must be >= 0.")
    if args.hybrid_rerank_top_k < 0:
        raise ValueError("--hybrid-rerank-top-k must be >= 0.")
    if args.hybrid_embedding_weight < 0 or args.hybrid_embedding_weight > 1:
        raise ValueError("--hybrid-embedding-weight must be between 0 and 1.")
    if args.hybrid_min_embedding_score < 0:
        raise ValueError("--hybrid-min-embedding-score must be >= 0.")
    if args.hybrid_min_bm25_score < 0:
        raise ValueError("--hybrid-min-bm25-score must be >= 0.")
    if args.embedding_provider == "local" and args.embedding_model == DEFAULT_EMBEDDING_MODEL:
        raise ValueError(
            "--embedding-provider local requires a Hugging Face --embedding-model, such as "
            "Qwen/Qwen3-Embedding-0.6B, BAAI/bge-m3, or intfloat/multilingual-e5-large."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.nli_max_length <= 0:
        raise ValueError("--nli-max-length must be > 0.")
    if args.nli_checkpoint_interval < 0:
        raise ValueError("--nli-checkpoint-interval must be >= 0.")
    if args.retrieval_checkpoint_interval < 0:
        raise ValueError("--retrieval-checkpoint-interval must be >= 0.")


def print_filter_summary(diagnostics: pd.DataFrame) -> None:
    total = len(diagnostics)
    kept = int(diagnostics["speech_filter_kept"].sum())
    print(
        f"Speech quality filters retained {kept:,}/{total:,} "
        f"candidate speech rows ({kept / max(total, 1):.1%})."
    )
    reason_counts = diagnostics["speech_filter_reason"].value_counts(dropna=False)
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count:,}")


def available_topic_labels(speech_df: pd.DataFrame) -> list[str]:
    labels = (
        speech_df["topic_label"]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    labels = labels[labels != ""]
    return sorted(labels.unique().tolist())


def stream_retrieval_artifact_to_csv(
    artifact: RetrievalCheckpointArtifact,
    output_path: Path,
) -> dict[str, Any]:
    """Assemble checkpoint parts without materializing the full retrieval."""
    state_path = output_path.with_name(
        f"{output_path.stem}_retrieval_state.json"
    )
    if state_path.exists() and output_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("complete")
            and state.get("signature") == artifact.signature
            and int(state.get("rows", -1)) == artifact.selected_rows
            and int(state.get("output_bytes", -1))
            == output_path.stat().st_size
            and state.get("output_sampled_content_sha256")
            == sampled_file_content_fingerprint(output_path)
        ):
            print(f"Retrieved-pair CSV already assembled: {output_path}")
            return state

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    rows_written = 0
    speeches: set[str] = set()
    parties: set[str] = set()
    manifestos: set[str] = set()
    retrieval_units: set[str] = set()
    for part_index, part_path in enumerate(
        artifact.part_paths, start=1
    ):
        frame = pd.read_parquet(part_path)
        frame = frame.rename(
            columns={
                "text": "speech_text",
                "topic_label": "speech_topic_label",
                "predicted_topic": "manifesto_predicted_topic",
            }
        )
        frame.insert(
            0,
            "nli_pair_id",
            np.arange(rows_written, rows_written + len(frame)),
        )
        frame.to_csv(
            temporary,
            mode="a",
            header=rows_written == 0,
            index=False,
        )
        rows_written += len(frame)
        if "plda_doc_id" in frame:
            speeches.update(
                frame["plda_doc_id"].dropna().astype(str)
            )
        if "party" in frame:
            parties.update(frame["party"].dropna().astype(str))
        if "doc_key" in frame:
            manifestos.update(frame["doc_key"].dropna().astype(str))
        unit_column = retrieval_group_column(frame)
        retrieval_units.update(
            frame[unit_column].dropna().astype(str)
        )
        print(
            f"Assembled retrieval part {part_index:,}/"
            f"{len(artifact.part_paths):,}: {rows_written:,}/"
            f"{artifact.selected_rows:,} rows",
            flush=True,
        )
    if rows_written != artifact.selected_rows:
        raise RuntimeError(
            "Assembled retrieved-pair row count does not match the "
            "checkpoint manifest"
        )
    os.replace(temporary, output_path)
    state = {
        "provenance_version": PROVENANCE_VERSION,
        "complete": True,
        "signature": artifact.signature,
        "temporal_linkage": (
            "strict latest MPDS-election-dated manifesto on or before each "
            "speech date; fallback rows excluded before retrieval"
        ),
        "rows": rows_written,
        "retrieval_units": len(retrieval_units),
        "speeches": len(speeches),
        "parties": len(parties),
        "manifesto_documents": len(manifestos),
        "candidate_pairs_scored": artifact.candidate_pairs_scored,
        "checkpoint_parts": len(artifact.part_paths),
        "output": str(output_path.resolve()),
        "output_bytes": output_path.stat().st_size,
        "output_sampled_content_sha256": sampled_file_content_fingerprint(
            output_path
        ),
    }
    atomic_write_json(state_path, state)
    return state


def run_topic(
    args: argparse.Namespace,
    country_code: str,
    topic: str,
    speech_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    manifesto_df: pd.DataFrame,
    topic_labels_input: Path | None,
) -> None:
    topic_start = time.perf_counter()
    print("=" * 80)
    print(f"Running topic: {topic}")

    selected_topic_cols = topic_columns_from_label_file(
        topic_labels_input,
        topic,
        expected_topic_count=len(topic_columns(speech_df)),
    )
    if not selected_topic_cols:
        selected_topic_cols = infer_topic_columns_from_speeches(speech_df, topic)
    print(f"Selected topic columns for {topic}: {', '.join(selected_topic_cols)}")

    paths = output_paths(args.output_dir, country_code, topic, args.model_name)
    retrieval_checkpoint_dir = (
        args.retrieval_checkpoint_dir
        or paths["pairs"].with_name(
            f"{paths['pairs'].stem}_retrieval_checkpoint"
        )
    )
    print("Diagnosing and filtering speech candidates...")
    diagnostics = diagnose_speeches(
        speech_df=speech_df,
        country_code=country_code,
        topic=topic,
        selected_topic_cols=selected_topic_cols,
        min_words=args.min_speech_words,
        min_topic_mass=args.min_speech_topic_mass,
        keep_procedural=args.keep_procedural_speeches,
        strip_procedural=not args.no_strip_procedural_phrases,
    )
    diagnostics.to_csv(paths["diagnostics"], index=False)
    print_filter_summary(diagnostics)
    print(f"Saved speech diagnostics to: {paths['diagnostics']}")

    print("Sampling speeches...")
    speeches = sample_speeches(
        speech_df=diagnostics,
        topic=topic,
        sample_size=args.sample_size,
        sample_per_party=args.sample_per_party,
        random_state=args.random_state,
    )
    print(
        f"Sampled {len(speeches):,} speech rows from "
        f"{speeches['party'].nunique():,} parties."
    )
    speeches = build_speech_units(
        speeches=speeches,
        speech_unit=args.speech_unit,
        window_size=args.segment_window_size,
        stride=args.segment_stride,
        min_segment_words=args.min_segment_words,
        max_segment_words=args.max_segment_words,
    )
    if args.speech_unit == "segment":
        print(
            f"Built {len(speeches):,} speech segment unit(s) from "
            f"{speeches['plda_doc_id'].nunique():,} sampled speeches."
        )

    print("Filtering manifesto quasi-sentences...")
    topic_manifesto_df = prepare_manifesto_quasi(
        manifesto_df=manifesto_df,
        topic_cols=selected_topic_cols,
        topic_filter=args.manifesto_topic_filter,
    )
    print(
        f"Retained {len(topic_manifesto_df):,} manifesto quasi-sentences after "
        f"{args.manifesto_topic_filter!r} filtering."
    )

    print("Building speech-manifesto pairs...")
    pairs, pair_quality = build_pairs(
        speeches=speeches,
        manifesto_df=topic_manifesto_df,
        bridge_df=bridge_df,
        pairs_per_speech=args.pairs_per_speech,
        random_state=args.random_state,
        pair_selection=args.pair_selection,
        min_retrieval_score=args.min_retrieval_score,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        api_key_env=args.api_key_env,
        local_embedding_device=args.local_embedding_device,
        embedding_cache_db=args.embedding_cache_db,
        embedding_disk_cache=args.embedding_disk_cache,
        hybrid_embedding_top_k=args.hybrid_embedding_top_k,
        hybrid_bm25_top_k=args.hybrid_bm25_top_k,
        hybrid_rerank_top_k=args.hybrid_rerank_top_k,
        hybrid_embedding_weight=args.hybrid_embedding_weight,
        hybrid_min_embedding_score=args.hybrid_min_embedding_score,
        hybrid_min_bm25_score=args.hybrid_min_bm25_score,
        retrieval_checkpoint_dir=retrieval_checkpoint_dir,
        retrieval_checkpoint_interval=args.retrieval_checkpoint_interval,
        retrieval_only=args.retrieval_only,
    )
    if isinstance(pairs, RetrievalCheckpointArtifact):
        state = stream_retrieval_artifact_to_csv(
            pairs, paths["pairs"]
        )
        print(
            f"Retrieval-only output contains {state['rows']:,} pairs "
            f"from {state['speeches']:,} speeches, "
            f"{state['retrieval_units']:,} retrieval units, and "
            f"{state['manifesto_documents']:,} manifesto documents."
        )
        print(f"Saved retrieved pairs to: {paths['pairs']}")
        print(f"Topic runtime: {time.perf_counter() - topic_start:.1f}s")
        return
    for name, table in pair_quality.items():
        table.to_csv(paths[name], index=False)
        print(f"Saved {name.replace('_', ' ')} to: {paths[name]}")
    print(
        f"Built {len(pairs):,} NLI pairs from {pairs['plda_doc_id'].nunique():,} "
        f"speeches, {pairs['party'].nunique():,} parties, "
        f"and {pairs['doc_key'].nunique():,} manifesto documents."
    )
    if args.retrieval_only:
        temporary_pairs = paths["pairs"].with_suffix(
            paths["pairs"].suffix + ".tmp"
        )
        pairs.to_csv(temporary_pairs, index=False)
        os.replace(temporary_pairs, paths["pairs"])
        print(f"Saved retrieval-only pairs to: {paths['pairs']}")
        print(f"Topic runtime: {time.perf_counter() - topic_start:.1f}s")
        return

    nli_checkpoint_path = paths["pairs"].with_name(f"{paths['pairs'].stem}_checkpoint.csv")
    classified = classify_pairs(
        pairs=pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device_name=args.device,
        max_length=args.nli_max_length,
        checkpoint_path=nli_checkpoint_path if args.nli_checkpoint_interval else None,
        checkpoint_interval=args.nli_checkpoint_interval,
    )

    classified.to_csv(paths["pairs"], index=False)
    summarize_distribution(classified, []).to_csv(paths["overall"], index=False)
    summarize_distribution(classified, ["party"]).to_csv(paths["party"], index=False)
    summarize_distribution(classified, ["party", "month"]).to_csv(
        paths["party_month"],
        index=False,
    )

    print(f"Classified {len(classified):,} speech-manifesto pair(s).")
    print(f"Saved pair-level classifications to: {paths['pairs']}")
    print(f"Saved overall summary to: {paths['overall']}")
    print(f"Saved party summary to: {paths['party']}")
    print(f"Saved party-month summary to: {paths['party_month']}")
    print(f"Saved speech diagnostics to: {paths['diagnostics']}")
    print(f"Topic runtime: {time.perf_counter() - topic_start:.1f}s")


def main(argv: list[str] | None = None) -> int:
    run_start = time.perf_counter()
    args = build_parser().parse_args(argv)
    validate_args(args)
    load_dotenv(BASE_DIR / ".env")
    country_code = args.country.strip().upper()
    speech_input, manifesto_input, bridge_input, topic_labels_input = resolve_paths(args)

    print(f"Country: {country_code}")
    print(f"Topic: {args.topic}")
    print(f"Speech input: {speech_input}")
    print(f"Manifesto quasi input: {manifesto_input}")
    print(f"Bridge input: {bridge_input}")
    print("Loading input CSV files...")
    load_start = time.perf_counter()
    speech_df = load_csv(
        speech_input,
        {"plda_doc_id", "date", "party", "month", "topic_label", "text"},
    )
    bridge_df = load_csv(
        bridge_input,
        {
            "speech_party", "date", "doc_key", "manifesto_effective_date",
            "selection_method",
        },
    )
    manifesto_df = load_csv(manifesto_input, {"doc_key", "text"})
    print(
        f"Loaded files in {time.perf_counter() - load_start:.1f}s: "
        f"{len(speech_df):,} speeches, {len(bridge_df):,} bridge rows, "
        f"{len(manifesto_df):,} manifesto quasi-sentences."
    )

    requested_topic = args.topic.strip()
    if requested_topic.lower() == "all":
        topics = available_topic_labels(speech_df)
        if not topics:
            raise ValueError("No topic labels found in speech input.")
        print(f"Running all topics: {len(topics):,} topic label(s).")
    else:
        topics = [requested_topic]

    failures: list[tuple[str, str]] = []
    for index, topic in enumerate(topics, start=1):
        if len(topics) > 1:
            print(f"Topic {index:,}/{len(topics):,}")
        try:
            run_topic(
                args=args,
                country_code=country_code,
                topic=topic,
                speech_df=speech_df,
                bridge_df=bridge_df,
                manifesto_df=manifesto_df,
                topic_labels_input=topic_labels_input,
            )
        except Exception as exc:
            if len(topics) == 1:
                raise
            failures.append((topic, str(exc)))
            print(f"Topic {topic!r} failed: {exc}")

    if failures:
        print("=" * 80)
        print(f"Completed with {len(failures):,}/{len(topics):,} topic failure(s):")
        for topic, message in failures:
            print(f"  {topic}: {message}")
        print(f"Total runtime: {time.perf_counter() - run_start:.1f}s")
        return 1

    print("=" * 80)
    print(f"Completed {len(topics):,} topic run(s) successfully.")
    print(f"Total runtime: {time.perf_counter() - run_start:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
