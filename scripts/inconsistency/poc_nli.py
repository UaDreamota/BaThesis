from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MANIFESTO_INPUT_DIR = BASE_DIR / "outputs" / "manifesto_quasi_sentences"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "nli_inconsistency"
DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample speeches from a selected PLDA topic, compare them to linked "
            "manifesto quasi-sentences with multilingual NLI, and save pair-level "
            "classifications plus aggregate distributions."
        )
    )
    parser.add_argument("--country", "--c", dest="country", default="CZ", type=str)
    parser.add_argument("--topic", default="Gal_Tan", type=str)
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
        "--pair-selection",
        choices=["openai_embedding_topk", "tfidf_topk", "random"],
        default="tfidf_topk",
        help=(
            "How to select manifesto quasi-sentences after party/month/doc_key "
            "joining. openai_embedding_topk uses OpenAI embeddings for semantic "
            "retrieval; tfidf_topk ranks candidates by lexical TF-IDF cosine "
            "similarity; random preserves the original random sampling behavior."
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
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model used by --pair-selection openai_embedding_topk.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        default=96,
        type=int,
        help="Number of texts per OpenAI embeddings request.",
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
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, type=str)
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
            "Speech month to manifesto bridge CSV. Defaults to "
            "outputs/manifesto_quasi_sentences/<COUNTRY>/"
            "<COUNTRY>_speech_month_to_manifesto_bridge.csv."
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
        / f"{country_code}_speech_month_to_manifesto_bridge.csv"
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

    return sampled.reset_index(drop=True)


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
    if pairs_per_speech > 0:
        sampled_parts = [
            group.sample(
                n=min(len(group), pairs_per_speech),
                random_state=random_state,
            )
            for _, group in pairs.groupby("plda_doc_id", sort=False)
        ]
        pairs = pd.concat(sampled_parts, ignore_index=True)
    else:
        pairs = pairs.copy()

    pairs["retrieval_score"] = pd.NA
    pairs["retrieval_rank"] = pairs.groupby("plda_doc_id", sort=False).cumcount() + 1
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
    for _, group in pairs.groupby("plda_doc_id", sort=False):
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


def embed_unique_texts(
    texts: list[str],
    model: str,
    api_key: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(text.strip() for text in texts if text.strip()))
    embeddings: dict[str, np.ndarray] = {}
    total_batches = (len(unique_texts) + batch_size - 1) // batch_size
    for batch_i, start in enumerate(range(0, len(unique_texts), batch_size), start=1):
        batch = unique_texts[start : start + batch_size]
        vectors = embed_text_batch(batch, model=model, api_key=api_key)
        for text, vector in zip(batch, vectors, strict=True):
            arr = np.array(vector, dtype=float)
            norm = np.linalg.norm(arr)
            embeddings[text] = arr / norm if norm else arr
        print(
            f"Embedding batch {batch_i:,}/{total_batches:,}: "
            f"{min(start + len(batch), len(unique_texts)):,}/{len(unique_texts):,} texts."
        )
    return embeddings


def select_openai_embedding_topk_pairs(
    pairs: pd.DataFrame,
    pairs_per_speech: int,
    min_retrieval_score: float,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
) -> pd.DataFrame:
    texts = (
        pairs["text"].fillna("").astype(str).str.strip().tolist()
        + pairs["manifesto_text"].fillna("").astype(str).str.strip().tolist()
    )
    embeddings = embed_unique_texts(
        texts=texts,
        model=embedding_model,
        api_key=openai_api_key(api_key_env),
        batch_size=embedding_batch_size,
    )

    selected_parts = []
    for _, group in pairs.groupby("plda_doc_id", sort=False):
        group = group.copy()
        speech_text = str(group["text"].iloc[0]).strip()
        speech_vector = embeddings.get(speech_text)
        if speech_vector is None:
            scores = np.zeros(len(group), dtype=float)
        else:
            manifesto_vectors = [
                embeddings.get(text.strip())
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
        ranked["pair_selection_method"] = "openai_embedding_topk"
        ranked["embedding_model"] = embedding_model
        if min_retrieval_score > 0:
            ranked = ranked[ranked["retrieval_score"] >= min_retrieval_score].copy()
        if pairs_per_speech > 0:
            ranked = ranked.head(pairs_per_speech).copy()
        if not ranked.empty:
            selected_parts.append(ranked)

    if not selected_parts:
        raise ValueError(
            "No speech-manifesto pairs remained after OpenAI embedding retrieval. "
            "Lower --min-retrieval-score or use another --pair-selection mode."
        )
    return pd.concat(selected_parts, ignore_index=True)


def select_pairs(
    pairs: pd.DataFrame,
    pair_selection: str,
    pairs_per_speech: int,
    random_state: int,
    min_retrieval_score: float,
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
) -> pd.DataFrame:
    if pair_selection == "random":
        return select_random_pairs(pairs, pairs_per_speech, random_state)
    if pair_selection == "tfidf_topk":
        return select_tfidf_topk_pairs(pairs, pairs_per_speech, min_retrieval_score)
    if pair_selection == "openai_embedding_topk":
        return select_openai_embedding_topk_pairs(
            pairs=pairs,
            pairs_per_speech=pairs_per_speech,
            min_retrieval_score=min_retrieval_score,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            api_key_env=api_key_env,
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
    sampled_speech_ids = set(sampled_speeches["plda_doc_id"].dropna().tolist())
    bridge_speech_ids = set(bridge_matches["plda_doc_id"].dropna().tolist())
    candidate_speech_ids = set(candidate_pairs["plda_doc_id"].dropna().tolist())
    selected_speech_ids = set(selected_pairs["plda_doc_id"].dropna().tolist())

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
                "sampled_parties": sampled_speeches["party"].nunique(),
                "sampled_months": sampled_speeches["month"].nunique(),
                "bridge_matched_speeches": len(bridge_speech_ids),
                "bridge_matched_parties": bridge_matches["party"].nunique(),
                "bridge_matched_months": bridge_matches["month"].nunique(),
                "candidate_speeches": len(candidate_speech_ids),
                "candidate_pairs": len(candidate_pairs),
                "candidate_parties": candidate_pairs["party"].nunique(),
                "candidate_months": candidate_pairs["month"].nunique(),
                "candidate_manifesto_docs": candidate_pairs["doc_key"].nunique(),
                "candidate_manifesto_quasi": candidate_pairs["quasi_sentence_id"].nunique(),
                "selected_speeches": len(selected_speech_ids),
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
        candidate_pairs.groupby("plda_doc_id", as_index=False)
        .agg(
            candidate_pairs=("plda_doc_id", "size"),
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
        selected_for_speech.groupby("plda_doc_id", as_index=False)
        .agg(**selected_aggs)
        .reset_index(drop=True)
    )
    speech_cols = ["plda_doc_id", "party", "month"]
    optional_speech_cols = [
        "speech_selected_topic_mass",
        "speech_word_count_for_nli",
        "speech_is_procedural",
    ]
    speech_cols.extend([col for col in optional_speech_cols if col in sampled_speeches.columns])
    by_speech = (
        sampled_speeches[speech_cols]
        .drop_duplicates("plda_doc_id")
        .merge(candidate_by_speech, on="plda_doc_id", how="left")
        .merge(selected_by_speech, on="plda_doc_id", how="left")
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
    embedding_model: str,
    embedding_batch_size: int,
    api_key_env: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bridge_cols = [
        "speech_party",
        "month",
        "doc_key",
        "manifesto_partyname",
        "manifesto_partyabbrev",
        "manifesto_date",
        "manifesto_effective_date",
    ]
    bridge = bridge_df[[col for col in bridge_cols if col in bridge_df.columns]].copy()
    merged = speeches.merge(
        bridge,
        left_on=["party", "month"],
        right_on=["speech_party", "month"],
        how="inner",
    )
    print(
        f"Bridge join retained {merged['plda_doc_id'].nunique():,}/"
        f"{speeches['plda_doc_id'].nunique():,} sampled speeches "
        f"across {merged['party'].nunique():,}/{speeches['party'].nunique():,} parties."
    )
    if merged.empty:
        raise ValueError("No sampled speeches matched the speech-month manifesto bridge.")

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
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        api_key_env=api_key_env,
    )
    print(
        f"Selected {len(pairs):,} pair(s) from "
        f"{pairs['plda_doc_id'].nunique():,} speech rows using {pair_selection!r}."
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


def classify_pairs(
    pairs: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Loading NLI tokenizer/model on {device}: {model_name}")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    labels = nli_label_order(model)
    print(
        f"Loaded NLI model in {time.perf_counter() - load_start:.1f}s. "
        f"Labels: {', '.join(labels)}"
    )

    rows = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    infer_start = time.perf_counter()
    with torch.no_grad():
        for batch_i, start in enumerate(range(0, len(pairs), batch_size), start=1):
            batch = pairs.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["manifesto_text"].astype(str).tolist(),
                batch["speech_text"].astype(str).tolist(),
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu()
            for prob in probs:
                row = {f"nli_prob_{label}": float(value) for label, value in zip(labels, prob)}
                row["nli_label"] = max(row, key=row.get).replace("nli_prob_", "")
                rows.append(row)
            if batch_i == 1 or batch_i % 10 == 0 or batch_i == total_batches:
                elapsed = time.perf_counter() - infer_start
                pairs_done = min(start + len(batch), len(pairs))
                seconds_per_pair = elapsed / max(pairs_done, 1)
                remaining_pairs = len(pairs) - pairs_done
                eta = remaining_pairs * seconds_per_pair
                print(
                    f"NLI batch {batch_i:,}/{total_batches:,}: "
                    f"{pairs_done:,}/{len(pairs):,} pairs done, "
                    f"elapsed={elapsed:.1f}s, eta={eta:.1f}s"
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


def output_paths(output_dir: Path, country_code: str, topic: str) -> dict[str, Path]:
    topic_token = re.sub(r"[^A-Za-z0-9_-]+", "_", topic).strip("_")
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{country_code}_{topic_token}_nli"
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
    if args.min_retrieval_score < 0:
        raise ValueError("--min-retrieval-score must be >= 0.")
    if args.min_retrieval_score > 1:
        raise ValueError("--min-retrieval-score must be <= 1.")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")


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
    speech_df = load_csv(speech_input, {"plda_doc_id", "party", "month", "topic_label", "text"})
    bridge_df = load_csv(bridge_input, {"speech_party", "month", "doc_key"})
    manifesto_df = load_csv(manifesto_input, {"doc_key", "text"})
    print(
        f"Loaded files in {time.perf_counter() - load_start:.1f}s: "
        f"{len(speech_df):,} speeches, {len(bridge_df):,} bridge rows, "
        f"{len(manifesto_df):,} manifesto quasi-sentences."
    )

    selected_topic_cols = topic_columns_from_label_file(
        topic_labels_input,
        args.topic,
        expected_topic_count=len(topic_columns(speech_df)),
    )
    if not selected_topic_cols:
        selected_topic_cols = infer_topic_columns_from_speeches(speech_df, args.topic)
    print(f"Selected topic columns for {args.topic}: {', '.join(selected_topic_cols)}")

    paths = output_paths(args.output_dir, country_code, args.topic)
    print("Diagnosing and filtering speech candidates...")
    diagnostics = diagnose_speeches(
        speech_df=speech_df,
        country_code=country_code,
        topic=args.topic,
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
        topic=args.topic,
        sample_size=args.sample_size,
        sample_per_party=args.sample_per_party,
        random_state=args.random_state,
    )
    print(
        f"Sampled {len(speeches):,} speech rows from "
        f"{speeches['party'].nunique():,} parties."
    )
    print("Filtering manifesto quasi-sentences...")
    manifesto_df = prepare_manifesto_quasi(
        manifesto_df=manifesto_df,
        topic_cols=selected_topic_cols,
        topic_filter=args.manifesto_topic_filter,
    )
    print(
        f"Retained {len(manifesto_df):,} manifesto quasi-sentences after "
        f"{args.manifesto_topic_filter!r} filtering."
    )
    print("Building speech-manifesto pairs...")
    pairs, pair_quality = build_pairs(
        speeches=speeches,
        manifesto_df=manifesto_df,
        bridge_df=bridge_df,
        pairs_per_speech=args.pairs_per_speech,
        random_state=args.random_state,
        pair_selection=args.pair_selection,
        min_retrieval_score=args.min_retrieval_score,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        api_key_env=args.api_key_env,
    )
    for name, table in pair_quality.items():
        table.to_csv(paths[name], index=False)
        print(f"Saved {name.replace('_', ' ')} to: {paths[name]}")
    print(
        f"Built {len(pairs):,} NLI pairs from {pairs['plda_doc_id'].nunique():,} "
        f"speeches, {pairs['party'].nunique():,} parties, "
        f"and {pairs['doc_key'].nunique():,} manifesto documents."
    )
    classified = classify_pairs(
        pairs=pairs,
        model_name=args.model_name,
        batch_size=args.batch_size,
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
    print(f"Total runtime: {time.perf_counter() - run_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
