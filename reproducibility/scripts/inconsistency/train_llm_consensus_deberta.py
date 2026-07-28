from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.inconsistency.artifact_provenance import (
        PROVENANCE_VERSION,
        atomic_write_json,
        canonical_json_fingerprint,
        records_fingerprint,
    )
except ModuleNotFoundError:
    from artifact_provenance import (  # type: ignore
        PROVENANCE_VERSION,
        atomic_write_json,
        canonical_json_fingerprint,
        records_fingerprint,
    )


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER_DIR = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "nli_inconsistency"
    / "llm_consensus"
    / "provider_labels_contradiction80_emb040_final"
)
DEFAULT_OUTPUT_DIR = (
    BASE_DIR / "outputs" / "test_speeches" / "nli_inconsistency" / "deberta_llm_consensus"
)
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
PROVIDERS = ("anthropic", "gemini", "openai")
VALID_LABELS = ("consistent", "unrelated", "inconsistent")
# Preserve the base NLI head's entailment/neutral/contradiction ordering.
LABEL2ID = {"consistent": 0, "unrelated": 1, "inconsistent": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the existing multilingual DeBERTa NLI model on labels supported "
            "by at least two of the three LLM annotators. Ambiguous and three-way "
            "disagreement cases are held out from supervised training."
        )
    )
    parser.add_argument("--provider-dir", type=Path, default=DEFAULT_PROVIDER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate grouped dataset splits without loading or training the model.",
    )
    parser.add_argument(
        "--unweighted-loss",
        action="store_true",
        help="Disable inverse-frequency class weights in the training loss.",
    )
    return parser


def provider_name(path: Path) -> str:
    name = path.name.lower()
    matches = [provider for provider in PROVIDERS if provider in name]
    if len(matches) != 1:
        raise ValueError(f"Could not identify exactly one provider from filename: {path}")
    return matches[0]


def read_provider_file(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Return one final nonblank row per sample ID and retry diagnostics."""
    attempts: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "llm_label", "speech_text_for_nli", "manifesto_text"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            attempts[row["sample_id"]].append(row)

    final: dict[str, dict[str, str]] = {}
    conflicting = 0
    for sample_id, rows in attempts.items():
        nonblank = [row for row in rows if row.get("llm_label", "").strip()]
        labels = {row["llm_label"].strip().lower() for row in nonblank}
        if len(labels) > 1:
            conflicting += 1
        # Appended provider files contain checkpoint/retry rows. The last successful
        # response is authoritative; retain a blank only when every attempt is blank.
        final[sample_id] = nonblank[-1] if nonblank else rows[-1]

    diagnostics = {
        "raw_rows": sum(len(rows) for rows in attempts.values()),
        "unique_ids": len(attempts),
        "duplicate_ids": sum(len(rows) > 1 for rows in attempts.values()),
        "conflicting_nonblank_ids": conflicting,
        "blank_final_labels": sum(not row.get("llm_label", "").strip() for row in final.values()),
    }
    return final, diagnostics


def discover_provider_files(provider_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(provider_dir.glob("*.csv")):
        provider = provider_name(path)
        if provider in files:
            raise ValueError(f"Multiple {provider} CSV files found in {provider_dir}")
        files[provider] = path
    missing = set(PROVIDERS).difference(files)
    if missing:
        raise FileNotFoundError(f"Missing provider files for: {sorted(missing)}")
    return files


def first_text(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def manifesto_group_id(row: dict[str, str]) -> str:
    """Keep every pair from one manifesto document in exactly one split."""
    country = row.get("sample_country") or row.get("country") or "unknown"
    doc_key = str(row.get("doc_key", "")).strip()
    if not doc_key:
        raise ValueError(
            f"Sample {row.get('sample_id', '<unknown>')} has no manifesto doc_key; "
            "manifesto-document grouping is mandatory."
        )
    return f"{country}|party_manifesto|{doc_key}"


def build_examples(provider_rows: dict[str, dict[str, dict[str, str]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common_ids = set.intersection(*(set(rows) for rows in provider_rows.values()))
    union_ids = set.union(*(set(rows) for rows in provider_rows.values()))
    examples: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()

    for sample_id in sorted(common_ids):
        base = provider_rows["openai"][sample_id]
        base_identity = {
            "doc_key": str(base.get("doc_key", "")).strip(),
            "country": str(
                base.get("sample_country") or base.get("country", "")
            ).strip(),
            "manifesto_text": first_text(base, ("manifesto_text",)),
            "speech_text": first_text(
                base,
                (
                    "speech_text_for_nli", "speech_segment_text",
                    "speech_text", "speech_text_original",
                ),
            ),
        }
        for provider in PROVIDERS:
            candidate = provider_rows[provider][sample_id]
            candidate_identity = {
                "doc_key": str(candidate.get("doc_key", "")).strip(),
                "country": str(
                    candidate.get("sample_country")
                    or candidate.get("country", "")
                ).strip(),
                "manifesto_text": first_text(
                    candidate, ("manifesto_text",)
                ),
                "speech_text": first_text(
                    candidate,
                    (
                        "speech_text_for_nli", "speech_segment_text",
                        "speech_text", "speech_text_original",
                    ),
                ),
            }
            for field, base_value in base_identity.items():
                if base_value != candidate_identity[field]:
                    raise ValueError(
                        f"Provider input mismatch for {sample_id}, field "
                        f"{field}: openai vs {provider}"
                    )
        labels = [provider_rows[provider][sample_id].get("llm_label", "").strip().lower() for provider in PROVIDERS]
        nonblank = [label for label in labels if label]
        counts = Counter(nonblank)
        agreement = max(counts.values(), default=0)
        majority_label = counts.most_common(1)[0][0] if agreement >= 2 else ""

        if agreement < 2:
            outcomes["no_majority"] += 1
            continue
        if majority_label == "ambiguous":
            outcomes["ambiguous_majority"] += 1
            continue
        if majority_label not in VALID_LABELS:
            outcomes["invalid_majority"] += 1
            continue

        manifesto = first_text(base, ("manifesto_text",))
        speech = first_text(
            base,
            ("speech_text_for_nli", "speech_segment_text", "speech_text", "speech_text_original"),
        )
        if not manifesto or not speech:
            outcomes["missing_text"] += 1
            continue

        examples.append(
            {
                "sample_id": sample_id,
                "group_id": manifesto_group_id(base),
                "manifesto_doc_key": str(base["doc_key"]).strip(),
                "country": base.get("sample_country") or base.get("country", ""),
                "topic": base.get("sample_topic") or base.get("speech_topic_label", ""),
                "manifesto_text": manifesto,
                "speech_text": speech,
                "label_name": majority_label,
                "labels": LABEL2ID[majority_label],
                "agreement_count": agreement,
            }
        )
        outcomes[f"kept_{majority_label}"] += 1

    diagnostics = {
        "provider_union_ids": len(union_ids),
        "provider_common_ids": len(common_ids),
        "outcomes": dict(outcomes),
    }
    return examples, diagnostics


def split_fingerprint(rows: list[dict[str, Any]]) -> str:
    fields = (
        "sample_id",
        "group_id",
        "manifesto_doc_key",
        "country",
        "topic",
        "manifesto_text",
        "speech_text",
        "label_name",
        "labels",
        "agreement_count",
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if pd.isna(value):
                item[field] = None
            elif field in {"labels", "agreement_count"}:
                item[field] = int(value)
            else:
                item[field] = str(value)
        normalized.append(item)
    return records_fingerprint(normalized, fields)


def assert_split_integrity(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Fail fast on row, group, or manifesto-document leakage."""
    names = ("train", "validation", "test")
    sample_sets = {
        name: {str(row["sample_id"]) for row in splits[name]}
        for name in names
    }
    group_sets = {
        name: {str(row["group_id"]) for row in splits[name]}
        for name in names
    }
    document_sets = {
        name: {str(row["manifesto_doc_key"]) for row in splits[name]}
        for name in names
    }
    pairwise: dict[str, dict[str, int]] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        row_overlap = sample_sets[left] & sample_sets[right]
        group_overlap = group_sets[left] & group_sets[right]
        document_overlap = document_sets[left] & document_sets[right]
        pairwise[f"{left}_vs_{right}"] = {
            "sample_overlap": len(row_overlap),
            "group_overlap": len(group_overlap),
            "manifesto_document_overlap": len(document_overlap),
        }
        if row_overlap or group_overlap or document_overlap:
            raise RuntimeError(
                f"Classifier split leakage for {left} vs {right}: "
                f"samples={len(row_overlap)}, groups={len(group_overlap)}, "
                f"manifestos={len(document_overlap)}"
            )
    if sum(len(rows) for rows in splits.values()) != len(
        set().union(*sample_sets.values())
    ):
        raise RuntimeError("Classifier split rows are not a disjoint partition")
    return {
        "status": "passed",
        "grouping_unit": "country + manifesto doc_key",
        "splits": {
            name: {
                "rows": len(splits[name]),
                "groups": len(group_sets[name]),
                "manifesto_documents": len(document_sets[name]),
                "fingerprint": split_fingerprint(splits[name]),
            }
            for name in names
        },
        "pairwise_overlap": pairwise,
    }


def assert_resume_split_compatible(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Refuse a checkpoint whose saved row partition differs from this run."""
    checkpoints = [
        path
        for path in (output_dir / "checkpoints").glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        return {"checkpoint_found": False, "status": "not_applicable"}
    checkpoint = max(
        checkpoints,
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    comparisons: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        path = output_dir / f"{name}.csv"
        if not path.exists():
            raise RuntimeError(
                f"Cannot resume {checkpoint}: missing saved {name}.csv"
            )
        prior_frame = pd.read_csv(path, low_memory=False)
        prior_rows = prior_frame.to_dict("records")
        prior_ids = prior_frame["sample_id"].astype(str).tolist()
        current_ids = [str(row["sample_id"]) for row in splits[name]]
        if prior_ids != current_ids:
            raise RuntimeError(
                f"Refusing to resume {checkpoint}: the {name} split differs "
                "from the checkpoint's saved dataset"
            )
        prior_fingerprint = split_fingerprint(prior_rows)
        current_fingerprint = split_fingerprint(splits[name])
        if prior_fingerprint != current_fingerprint:
            raise RuntimeError(
                f"Refusing to resume {checkpoint}: the {name} texts, labels, "
                "groups, or metadata differ from the checkpoint's saved dataset"
            )
        comparisons[name] = {
            "rows": len(current_ids),
            "fingerprint": current_fingerprint,
        }
    return {
        "checkpoint_found": True,
        "checkpoint": str(checkpoint),
        "status": "passed",
        "splits": comparisons,
    }


def grouped_splits(examples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Dataset preparation requires scikit-learn.") from exc

    labels = [
        f"{row['country']}::{row['label_name']}"
        for row in examples
    ]
    groups = [row["group_id"] for row in examples]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(examples, labels, groups))
    train_val = [examples[index] for index in train_val_idx]
    test = [examples[index] for index in test_idx]

    inner_labels = [
        f"{row['country']}::{row['label_name']}"
        for row in train_val
    ]
    inner_groups = [row["group_id"] for row in train_val]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_idx, validation_idx = next(inner.split(train_val, inner_labels, inner_groups))
    splits = {
        "train": [train_val[index] for index in train_idx],
        "validation": [train_val[index] for index in validation_idx],
        "test": test,
    }

    assert_split_integrity(splits)
    return splits


def write_prepared_data(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id", "group_id", "manifesto_doc_key", "country", "topic",
        "manifesto_text", "speech_text", "label_name", "labels",
        "agreement_count",
    ]
    for name, rows in splits.items():
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "manifesto_documents": len(
            {row["manifesto_doc_key"] for row in rows}
        ),
        "labels": dict(Counter(row["label_name"] for row in rows)),
        "agreement": dict(Counter(str(row["agreement_count"]) for row in rows)),
        "countries": dict(Counter(row["country"] for row in rows)),
        "topics": dict(Counter(row["topic"] for row in rows)),
    }


def training_run_spec(
    args: argparse.Namespace,
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    payload = {
        "provenance_version": PROVENANCE_VERSION,
        "base_model": str(args.model),
        "seed": int(args.seed),
        "max_length": int(args.max_length),
        "epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "fp16": bool(args.fp16),
        "bf16": bool(args.bf16),
        "unweighted_loss": bool(args.unweighted_loss),
        "label2id": LABEL2ID,
        "splits": {
            name: {
                "rows": len(rows),
                "fingerprint": split_fingerprint(rows),
            }
            for name, rows in splits.items()
        },
    }
    return {
        "run_signature": canonical_json_fingerprint(payload),
        "run_spec": payload,
    }


def training_artifacts_complete(output_dir: Path) -> bool:
    """Return true only for a fully saved model and held-out evaluation."""
    required = [
        output_dir / "train.csv",
        output_dir / "validation.csv",
        output_dir / "test.csv",
        output_dir / "split_integrity_assertions.json",
        output_dir / "test_results.json",
        output_dir / "model" / "config.json",
        output_dir / "model" / "tokenizer_config.json",
    ]
    weights = [
        output_dir / "model" / "model.safetensors",
        output_dir / "model" / "pytorch_model.bin",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and any(
        path.is_file() and path.stat().st_size > 0 for path in weights
    )


def assert_training_run_compatible(
    args: argparse.Namespace,
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    expected = training_run_spec(args, splits)
    path = args.output_dir / "training_run_provenance.json"
    checkpoints = list((args.output_dir / "checkpoints").glob("checkpoint-*"))
    model_path = args.output_dir / "model"
    if not path.exists():
        if checkpoints or model_path.exists():
            raise RuntimeError(
                "Existing training checkpoints/model lack full data and "
                f"hyperparameter provenance: {args.output_dir}. Use a new output "
                "directory instead of resuming unsafely."
            )
        atomic_write_json(path, expected)
        return {"status": "initialized", **expected}
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior != expected:
        raise RuntimeError(
            "Training checkpoint data/model/hyperparameters differ from the "
            f"recorded run: {args.output_dir}. Use a new output directory."
        )
    return {"status": "passed", **expected}


def train(args: argparse.Namespace, splits: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training requires torch, transformers, datasets, accelerate, numpy, and scikit-learn."
        ) from exc

    integrity = assert_split_integrity(splits)
    resume_compatibility = assert_resume_split_compatible(
        args.output_dir, splits
    )
    training_compatibility = assert_training_run_compatible(args, splits)
    (args.output_dir / "split_integrity_assertions.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    (args.output_dir / "resume_split_assertions.json").write_text(
        json.dumps(resume_compatibility, indent=2), encoding="utf-8"
    )
    atomic_write_json(
        args.output_dir / "training_resume_assertions.json",
        training_compatibility,
    )
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    # AMP requires FP32 trainable parameters and casts only the forward pass.
    model = model.float()
    if args.gradient_checkpointing:
        model.config.use_cache = False

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        # Match poc_nli.py inference order: manifesto premise, speech hypothesis.
        return tokenizer(
            batch["manifesto_text"],
            batch["speech_text"],
            truncation=True,
            max_length=args.max_length,
        )

    datasets = {
        name: Dataset.from_list(rows).map(tokenize, batched=True, remove_columns=[
            "sample_id", "group_id", "manifesto_doc_key", "country", "topic",
            "manifesto_text", "speech_text", "label_name", "agreement_count",
        ])
        for name, rows in splits.items()
    }

    train_counts = Counter(row["labels"] for row in splits["train"])
    class_weights = torch.tensor(
        [len(splits["train"]) / (len(LABEL2ID) * train_counts[index]) for index in range(len(LABEL2ID))],
        dtype=torch.float,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = None if args.unweighted_loss else class_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    checkpoint_steps = int(getattr(args, "checkpoint_steps", 50))

    class FrequentCheckpointCallback(TrainerCallback):
        """Request resumable checkpoints at the configured step interval."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step > 0 and state.global_step % checkpoint_steps == 0:
                control.should_save = True
            return control

    def compute_metrics(prediction: Any) -> dict[str, float]:
        predictions = np.argmax(prediction.predictions, axis=-1)
        labels = prediction.label_ids
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
            "inconsistent_f1": float(f1_score(labels == LABEL2ID["inconsistent"], predictions == LABEL2ID["inconsistent"])),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        train_sampling_strategy=(
            "group_by_length"
            if bool(getattr(args, "group_by_length", False))
            else "random"
        ),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            FrequentCheckpointCallback(),
        ],
    )
    from transformers.trainer_utils import get_last_checkpoint

    resume_checkpoint = get_last_checkpoint(str(args.output_dir / "checkpoints"))
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output_dir / "model"))
    tokenizer.save_pretrained(str(args.output_dir / "model"))

    prediction = trainer.predict(datasets["test"])
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    true_ids = prediction.label_ids
    report = classification_report(
        true_ids,
        predicted_ids,
        labels=sorted(ID2LABEL),
        target_names=[ID2LABEL[index] for index in sorted(ID2LABEL)],
        output_dict=True,
        zero_division=0,
    )
    result = {
        "metrics": {key: float(value) for key, value in prediction.metrics.items()},
        "classification_report": report,
        "confusion_matrix_label_order": [ID2LABEL[index] for index in sorted(ID2LABEL)],
        "confusion_matrix": confusion_matrix(true_ids, predicted_ids, labels=sorted(ID2LABEL)).tolist(),
        "class_weights": class_weights.tolist(),
    }
    (args.output_dir / "test_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    files = discover_provider_files(args.provider_dir)
    provider_rows: dict[str, dict[str, dict[str, str]]] = {}
    provider_diagnostics: dict[str, Any] = {}
    for provider, path in files.items():
        rows, diagnostics = read_provider_file(path)
        provider_rows[provider] = rows
        provider_diagnostics[provider] = {"path": str(path), **diagnostics}
        if diagnostics["conflicting_nonblank_ids"]:
            raise ValueError(f"Conflicting retry labels found for {provider}: {diagnostics}")

    examples, build_diagnostics = build_examples(provider_rows)
    splits = grouped_splits(examples, args.seed)
    integrity = assert_split_integrity(splits)
    resume_compatibility = assert_resume_split_compatible(
        args.output_dir, splits
    )
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "label2id": LABEL2ID,
        "provider_diagnostics": provider_diagnostics,
        "build_diagnostics": build_diagnostics,
        "splits": {name: split_summary(rows) for name, rows in splits.items()},
        "split_policy": (
            "StratifiedGroupKFold using country-by-label strata and grouped by "
            "country + manifesto doc_key "
            "(80/20, then 75/25 train/validation)"
        ),
        "split_integrity_assertions": integrity,
        "resume_split_compatibility": resume_compatibility,
        "pair_order": ["manifesto_text", "speech_text"],
    }
    write_prepared_data(args.output_dir, splits, metadata)
    (args.output_dir / "split_integrity_assertions.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if not args.prepare_only:
        if training_artifacts_complete(args.output_dir):
            print(
                f"Training already complete in {args.output_dir}; "
                "kept the saved model and held-out results."
            )
        else:
            train(args, splits)


if __name__ == "__main__":
    main()
