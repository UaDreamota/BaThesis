from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.inconsistency.artifact_provenance import (
        PROVENANCE_VERSION,
        artifact_fingerprint,
        atomic_write_json,
        canonical_json_fingerprint,
        file_identity,
        sampled_file_content_fingerprint,
    )
except ModuleNotFoundError:
    from artifact_provenance import (  # type: ignore
        PROVENANCE_VERSION,
        artifact_fingerprint,
        atomic_write_json,
        canonical_json_fingerprint,
        file_identity,
        sampled_file_content_fingerprint,
    )


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = BASE_DIR / "outputs" / "test_speeches" / "nli_inconsistency"
DEFAULT_OUTPUT_ROOT = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "nli_consensus_classifier_manifesto_grouped"
)
DEFAULT_MODEL = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "methodology_freeze_robustness"
    / "manifesto_grouped_deberta"
    / "model"
)
DEFAULT_SOURCE_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CLUSTERS = {
    "cee": ["CZ", "EE", "LV", "PL"],
    "nordics": ["DK", "FI", "NO", "SE"],
    "west": ["AT", "BE", "NL"],
    "south": ["ES", "IT", "PT", "GR"],
    "gb": ["GB"],
}
LABELS = ("consistent", "unrelated", "inconsistent")


def token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the LLM-consensus-fine-tuned DeBERTa model to existing NLI pair files. "
            "Existing pairs are reused, so retrieval candidates, rates, ranks, and scores "
            "are identical to the preceding NLI contradiction run."
        )
    )
    parser.add_argument("--cluster", choices=[*CLUSTERS, "all"], default=None)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--topic", choices=["Macroeconomics", "Gal_Tan", "all"], default="Macroeconomics")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source-model-name", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def selected_countries(args: argparse.Namespace) -> list[str]:
    if args.countries:
        return list(dict.fromkeys(country.strip().upper() for country in args.countries))
    if args.cluster == "all":
        return [country for countries in CLUSTERS.values() for country in countries]
    if args.cluster:
        return CLUSTERS[args.cluster]
    raise ValueError("Specify --cluster or --countries.")


def selected_topics(topic: str) -> list[str]:
    return ["Macroeconomics", "Gal_Tan"] if topic == "all" else [topic]


def resolve_input(root: Path, country: str, topic: str, source_model: str) -> Path:
    country_dir = root / country
    topic_name = token(topic)
    model_name = token(source_model)
    preferred = country_dir / f"{country}_{topic_name}_nli_{model_name}_pairs.csv"
    if preferred.exists():
        return preferred
    plain = country_dir / f"{country}_{topic_name}_nli_pairs.csv"
    if plain.exists():
        return plain
    candidates = sorted(
        country_dir.glob(f"{country}_{topic_name}_nli_*_pairs.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No existing retrieved pair file for {country} {topic} in {country_dir}")


def output_paths(root: Path, country: str, topic: str) -> dict[str, Path]:
    directory = root / country
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{country}_{token(topic)}_consensus_deberta"
    return {
        "pairs": directory / f"{stem}_pairs.csv",
        "state": directory / f"{stem}_state.json",
        "overall": directory / f"{stem}_summary_overall.csv",
        "party": directory / f"{stem}_summary_by_party.csv",
        "party_month": directory / f"{stem}_summary_by_party_month.csv",
    }


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_rows": 0, "complete": False}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_json(path, state)


def classifier_run_spec(
    input_path: Path,
    model_path: Path,
    max_length: int,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    return {
        "provenance_version": PROVENANCE_VERSION,
        "input": file_identity(input_path),
        "model": {
            "path": str(model_path),
            "artifact_fingerprint": artifact_fingerprint(model_path),
        },
        "max_length": int(max_length),
        "labels": list(LABELS),
        "pair_order": ["manifesto_text", "speech_text"],
    }


def initialized_state(run_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "provenance_version": PROVENANCE_VERSION,
        "run_signature": canonical_json_fingerprint(run_spec),
        "run_spec": run_spec,
        "processed_rows": 0,
        "output_bytes": 0,
        "pending_chunk": None,
        "complete": False,
    }


def validate_or_initialize_state(
    state_path: Path,
    pair_path: Path,
    run_spec: dict[str, Any],
) -> dict[str, Any]:
    expected_signature = canonical_json_fingerprint(run_spec)
    if not state_path.exists():
        if pair_path.exists():
            raise RuntimeError(
                f"Classifier output exists without a versioned state: {pair_path}. "
                "Use --overwrite or a new output root; it cannot be resumed safely."
            )
        return initialized_state(run_spec)

    state = read_state(state_path)
    if (
        state.get("provenance_version") != PROVENANCE_VERSION
        or not state.get("run_signature")
        or not state.get("run_spec")
    ):
        raise RuntimeError(
            f"Legacy classifier checkpoint has no model-bound provenance: {state_path}. "
            "Use --overwrite or a new output root."
        )
    if state["run_signature"] != expected_signature or state["run_spec"] != run_spec:
        raise RuntimeError(
            f"Classifier checkpoint model/data/configuration mismatch: {state_path}. "
            "Use --overwrite or a new output root."
        )

    pending = state.get("pending_chunk")
    if pending:
        output_bytes_before = int(pending["output_bytes_before"])
        if pair_path.exists():
            actual = pair_path.stat().st_size
            if actual < output_bytes_before:
                raise RuntimeError(
                    f"Classifier output is shorter than its durable checkpoint: {pair_path}"
                )
            if actual != output_bytes_before:
                with pair_path.open("r+b") as handle:
                    handle.truncate(output_bytes_before)
                    handle.flush()
                    os.fsync(handle.fileno())
        elif output_bytes_before:
            raise RuntimeError(
                f"Classifier output disappeared after durable rows were saved: {pair_path}"
            )
        state["processed_rows"] = int(pending["row_start"])
        state["output_bytes"] = output_bytes_before
        state["pending_chunk"] = None
        state["complete"] = False
        write_state(state_path, state)

    processed = int(state.get("processed_rows", 0))
    expected_bytes = int(state.get("output_bytes", 0))
    if processed:
        if not pair_path.exists():
            raise RuntimeError(
                f"Classifier state records {processed:,} rows but output is missing: "
                f"{pair_path}"
            )
        if pair_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Classifier output byte size differs from its durable state: {pair_path}"
            )
    elif pair_path.exists() and pair_path.stat().st_size:
        raise RuntimeError(
            f"Classifier output has data but state records zero rows: {pair_path}"
        )
    return state


def best_speech_column(columns: Iterable[str]) -> str:
    available = set(columns)
    for column in (
        "speech_text_for_nli", "speech_segment_text", "speech_text", "speech_text_original",
    ):
        if column in available:
            return column
    raise ValueError("Pair input has no usable speech text column.")


def classify_frame(
    frame: Any,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
    batch_size: int,
    max_length: int,
) -> Any:
    import numpy as np

    if "manifesto_text" not in frame.columns:
        raise ValueError("Pair input is missing manifesto_text.")
    speech_column = best_speech_column(frame.columns)
    probabilities: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["manifesto_text"].fillna("").astype(str).tolist(),
                batch[speech_column].fillna("").astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(**encoded).logits
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    probs = np.concatenate(probabilities, axis=0)
    id2label = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    expected = set(LABELS)
    if set(id2label.values()) != expected:
        raise ValueError(f"Unexpected model labels: {id2label}; expected {sorted(expected)}")
    for class_index, label in id2label.items():
        frame[f"classifier_prob_{label}"] = probs[:, class_index]
    predicted = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    frame["classifier_label"] = [id2label[int(index)] for index in predicted]
    frame["classifier_confidence"] = probs.max(axis=1)
    frame["classifier_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    return frame


def summarize_pairs(pair_path: Path, paths: dict[str, Path], chunk_size: int) -> None:
    import pandas as pd

    probability_columns = [f"classifier_prob_{label}" for label in LABELS]
    required = [
        "classifier_label", *probability_columns, "party", "month", "plda_doc_id", "quasi_sentence_id",
    ]
    accumulators: dict[tuple[str, ...], dict[tuple[str, ...], dict[str, Any]]] = {
        (): defaultdict(lambda: {"n_pairs": 0, "prob_sums": defaultdict(float), "labels": defaultdict(int), "speeches": set(), "quasi": set()}),
        ("party",): defaultdict(lambda: {"n_pairs": 0, "prob_sums": defaultdict(float), "labels": defaultdict(int), "speeches": set(), "quasi": set()}),
        ("party", "month"): defaultdict(lambda: {"n_pairs": 0, "prob_sums": defaultdict(float), "labels": defaultdict(int), "speeches": set(), "quasi": set()}),
    }
    for chunk in pd.read_csv(pair_path, usecols=lambda name: name in required, chunksize=chunk_size, low_memory=False):
        for group_columns, groups in accumulators.items():
            iterator = [((), chunk)] if not group_columns else chunk.groupby(list(group_columns), dropna=False)
            for raw_key, group in iterator:
                key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                entry = groups[key]
                entry["n_pairs"] += len(group)
                for column in probability_columns:
                    entry["prob_sums"][column] += float(group[column].sum())
                for label, count in group["classifier_label"].value_counts().items():
                    entry["labels"][str(label)] += int(count)
                entry["speeches"].update(group["plda_doc_id"].dropna().astype(str))
                entry["quasi"].update(group["quasi_sentence_id"].dropna().astype(str))

    for group_columns, output_key in (((), "overall"), (("party",), "party"), (("party", "month"), "party_month")):
        records = []
        for key, entry in accumulators[group_columns].items():
            record = dict(zip(group_columns, key))
            n_pairs = entry["n_pairs"]
            record.update({
                "n_pairs": n_pairs,
                "n_speeches": len(entry["speeches"]),
                "n_manifesto_quasi": len(entry["quasi"]),
            })
            for label in LABELS:
                count = entry["labels"].get(label, 0)
                record[f"classifier_count_{label}"] = count
                record[f"classifier_share_{label}"] = count / n_pairs if n_pairs else 0.0
                record[f"classifier_prob_{label}"] = entry["prob_sums"][f"classifier_prob_{label}"] / n_pairs if n_pairs else 0.0
            records.append(record)
        pd.DataFrame(records).to_csv(paths[output_key], index=False)


def classify_file(
    input_path: Path,
    paths: dict[str, Path],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
) -> None:
    import pandas as pd

    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    run_spec = classifier_run_spec(input_path, args.model, args.max_length)
    state = validate_or_initialize_state(paths["state"], paths["pairs"], run_spec)
    if state.get("complete") and paths["pairs"].exists():
        expected_fingerprint = state.get("output_sampled_content_sha256")
        if (
            not expected_fingerprint
            or sampled_file_content_fingerprint(paths["pairs"])
            != expected_fingerprint
        ):
            raise RuntimeError(
                f"Completed classifier output failed its content check: {paths['pairs']}"
            )
        missing_summaries = [
            paths[name]
            for name in ("overall", "party", "party_month")
            if not paths[name].exists()
        ]
        if missing_summaries:
            raise RuntimeError(
                "Completed classifier state is missing summaries: "
                + ", ".join(str(path) for path in missing_summaries)
            )
        print(f"Already complete, skipping: {paths['pairs']}")
        return

    processed = int(state.get("processed_rows", 0))
    seen = 0
    wrote_any = paths["pairs"].exists() and processed > 0
    for chunk in pd.read_csv(input_path, chunksize=args.chunk_size, low_memory=False):
        chunk_end = seen + len(chunk)
        if chunk_end <= processed:
            seen = chunk_end
            continue
        if processed > seen:
            chunk = chunk.iloc[processed - seen :].copy()
        else:
            chunk = chunk.copy()
        classified = classify_frame(
            chunk, tokenizer, model, torch, device, args.batch_size, args.max_length
        )
        output_bytes_before = (
            paths["pairs"].stat().st_size if paths["pairs"].exists() else 0
        )
        state.update(
            {
                "pending_chunk": {
                    "row_start": processed,
                    "row_end": processed + len(classified),
                    "output_bytes_before": output_bytes_before,
                },
                "complete": False,
            }
        )
        write_state(paths["state"], state)
        classified.to_csv(paths["pairs"], mode="a", header=not wrote_any, index=False)
        with paths["pairs"].open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        wrote_any = True
        processed += len(classified)
        seen = chunk_end
        state.update(
            {
                "processed_rows": processed,
                "output_bytes": paths["pairs"].stat().st_size,
                "pending_chunk": None,
                "complete": False,
            }
        )
        write_state(paths["state"], state)
        print(f"  classified {processed:,} rows", flush=True)

    summarize_pairs(paths["pairs"], paths, max(args.chunk_size, 10_000))
    state.update(
        {
            "processed_rows": processed,
            "output_bytes": paths["pairs"].stat().st_size,
            "output_sampled_content_sha256": sampled_file_content_fingerprint(
                paths["pairs"]
            ),
            "pending_chunk": None,
            "complete": True,
        }
    )
    write_state(paths["state"], state)
    print(f"Saved pair classifications: {paths['pairs']}")
    print(f"Saved overall summary: {paths['overall']}")
    print(f"Saved party-month summary: {paths['party_month']}")


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0 or args.max_length <= 0:
        raise ValueError("Batch size, chunk size, and max length must be positive.")
    countries = selected_countries(args)
    topics = selected_topics(args.topic)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    print(f"Loaded classifier once on {device}: {args.model}")

    failures: list[str] = []
    for topic in topics:
        for country in countries:
            print("=" * 80)
            print(f"Classifying existing retrieval: {country} {topic}")
            try:
                input_path = resolve_input(args.input_root, country, topic, args.source_model_name)
                print(f"Input pairs: {input_path}")
                classify_file(
                    input_path, output_paths(args.output_root, country, topic),
                    tokenizer, model, torch, device, args,
                )
            except (FileNotFoundError, ValueError) as exc:
                failures.append(f"{country} {topic}: {exc}")
                print(f"SKIPPED: {exc}")
    if failures:
        raise SystemExit("Some cells failed:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
