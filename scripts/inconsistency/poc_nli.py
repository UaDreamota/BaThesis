from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
MANIFESTO_INPUT_DIR = BASE_DIR / "outputs" / "manifesto_quasi_sentences"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "nli_inconsistency"
DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)$")


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
        help="Maximum manifesto quasi-sentence comparisons per sampled speech.",
    )
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, type=str)
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


def sample_speeches(
    speech_df: pd.DataFrame,
    topic: str,
    sample_size: int,
    sample_per_party: int | None,
    random_state: int,
) -> pd.DataFrame:
    topic_speeches = speech_df[speech_df["topic_label"].astype(str) == topic].copy()
    topic_speeches = topic_speeches.dropna(subset=["party", "month", "text"])
    topic_speeches["text"] = topic_speeches["text"].astype(str).str.strip()
    topic_speeches = topic_speeches[topic_speeches["text"] != ""].copy()
    if topic_speeches.empty:
        raise ValueError(f"No non-empty speeches found for topic {topic!r}.")

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


def build_pairs(
    speeches: pd.DataFrame,
    manifesto_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    pairs_per_speech: int,
    random_state: int,
) -> pd.DataFrame:
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

    if pairs_per_speech > 0:
        sampled_parts = [
            group.sample(
                n=min(len(group), pairs_per_speech),
                random_state=random_state,
            )
            for _, group in pairs.groupby("plda_doc_id", sort=False)
        ]
        pairs = pd.concat(sampled_parts, ignore_index=True)

    pairs = pairs.rename(
        columns={
            "text": "speech_text",
            "topic_label": "speech_topic_label",
            "predicted_topic": "manifesto_predicted_topic",
        }
    )
    pairs.insert(0, "nli_pair_id", range(len(pairs)))
    return pairs.reset_index(drop=True)


def normalize_label(label: object) -> str:
    normalized = str(label).strip().lower()
    if normalized.startswith("label_"):
        return normalized
    return normalized


def nli_label_order(model: AutoModelForSequenceClassification) -> list[str]:
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
    }


def main(argv: list[str] | None = None) -> int:
    run_start = time.perf_counter()
    args = build_parser().parse_args(argv)
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

    print("Sampling speeches...")
    speeches = sample_speeches(
        speech_df=speech_df,
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
    pairs = build_pairs(
        speeches=speeches,
        manifesto_df=manifesto_df,
        bridge_df=bridge_df,
        pairs_per_speech=args.pairs_per_speech,
        random_state=args.random_state,
    )
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

    paths = output_paths(args.output_dir, country_code, args.topic)
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
    print(f"Total runtime: {time.perf_counter() - run_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
