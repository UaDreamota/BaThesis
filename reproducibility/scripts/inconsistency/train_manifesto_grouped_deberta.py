from __future__ import annotations

"""Retrain the consensus classifier with party-manifesto document groups."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.inconsistency import train_llm_consensus_deberta as original


TEST_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_INPUT = TEST_DIR / "nli_inconsistency" / "deberta_llm_consensus"
DEFAULT_SOURCE = (
    TEST_DIR
    / "nli_inconsistency"
    / "llm_consensus"
    / "llm_contradiction80_emb040_final_sample.csv"
)
DEFAULT_OUTPUT = TEST_DIR / "methodology_freeze_robustness" / "manifesto_grouped_deberta"
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    return parser


def manifesto_grouped_splits(
    records: list[dict[str, object]],
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    """Balance countries and labels while keeping whole manifestos together."""
    strata = [
        f"{row['country']}::{row['label_name']}"
        for row in records
    ]
    groups = [str(row["group_id"]) for row in records]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation_index, test_index = next(
        outer.split(records, strata, groups)
    )
    train_validation = [records[index] for index in train_validation_index]
    test = [records[index] for index in test_index]
    inner_strata = [
        f"{row['country']}::{row['label_name']}"
        for row in train_validation
    ]
    inner_groups = [str(row["group_id"]) for row in train_validation]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_index, validation_index = next(
        inner.split(train_validation, inner_strata, inner_groups)
    )
    return {
        "train": [train_validation[index] for index in train_index],
        "validation": [train_validation[index] for index in validation_index],
        "test": test,
    }


def assert_resume_split_compatible(
    output_dir: Path,
    splits: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    return original.assert_resume_split_compatible(output_dir, splits)


def main() -> int:
    args = build_parser().parse_args()
    frames = [
        pd.read_csv(args.input_dir / f"{split}.csv", low_memory=False)
        for split in ("train", "validation", "test")
    ]
    examples = pd.concat(frames, ignore_index=True)
    source = pd.read_csv(
        args.sample_source,
        usecols=[
            "sample_id", "doc_key", "country", "party", "plda_doc_id",
            "retrieval_unit_id", "speech_segment_id",
        ],
        low_memory=False,
    )
    examples = examples.merge(source, on="sample_id", how="left", validate="1:1")
    if examples["doc_key"].isna().any():
        raise ValueError("Missing manifesto document keys after source merge")
    examples["group_id"] = (
        examples["country_x"].fillna(examples["country_y"]).astype(str)
        + "|party_manifesto|"
        + examples["doc_key"].astype(str)
    )
    examples["manifesto_doc_key"] = examples["doc_key"].astype(str)
    examples["country"] = examples["country_x"].fillna(examples["country_y"])
    keep = [
        "sample_id", "group_id", "manifesto_doc_key", "country", "topic",
        "manifesto_text", "speech_text", "label_name", "labels",
        "agreement_count",
    ]
    records = examples[keep].to_dict("records")
    splits = manifesto_grouped_splits(records, args.seed)
    integrity = original.assert_split_integrity(splits)
    resume_compatibility = assert_resume_split_compatible(
        args.output_dir, splits
    )
    document_sets = {
        name: {
            row["group_id"]
            for row in rows
        }
        for name, rows in splits.items()
    }
    source_identity = examples.set_index("sample_id")[
        [
            "country", "plda_doc_id", "retrieval_unit_id",
            "speech_segment_id",
        ]
    ].to_dict("index")
    speech_identity_overlap: dict[str, dict[str, int]] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = document_sets[left] & document_sets[right]
        if overlap:
            raise RuntimeError(
                f"Manifesto leakage between {left} and {right}: {len(overlap)}"
            )
        identity_counts: dict[str, int] = {}
        for column in (
            "plda_doc_id", "retrieval_unit_id", "speech_segment_id"
        ):
            identity_sets: dict[str, set[str]] = {}
            for split_name in (left, right):
                values: set[str] = set()
                for row in splits[split_name]:
                    identity = source_identity[str(row["sample_id"])]
                    value = identity.get(column)
                    if pd.notna(value) and str(value).strip():
                        values.add(
                            f"{identity['country']}|{str(value).strip()}"
                        )
                identity_sets[split_name] = values
            identity_counts[column] = len(
                identity_sets[left] & identity_sets[right]
            )
        speech_identity_overlap[f"{left}_vs_{right}"] = identity_counts
        if any(identity_counts.values()):
            raise RuntimeError(
                f"Speech/retrieval leakage between {left} and {right}: "
                f"{identity_counts}"
            )
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "label2id": original.LABEL2ID,
        "split_policy": (
            "StratifiedGroupKFold using country-by-label strata and grouping by "
            "country + party-manifesto doc_key "
            "(80/20, then 75/25 train/validation)"
        ),
        "splits": {
            name: {
                **original.split_summary(rows),
                "manifesto_groups": len(document_sets[name]),
            }
            for name, rows in splits.items()
        },
        "manifesto_group_overlap": {
            f"{left}_vs_{right}": len(document_sets[left] & document_sets[right])
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
        "speech_identity_overlap": speech_identity_overlap,
        "split_integrity_assertions": integrity,
        "resume_split_compatibility": resume_compatibility,
    }
    original.write_prepared_data(args.output_dir, splits, metadata)
    (args.output_dir / "manifesto_split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output_dir / "split_integrity_assertions.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    (args.output_dir / "resume_split_assertions.json").write_text(
        json.dumps(resume_compatibility, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    if args.prepare_only:
        return 0
    if original.training_artifacts_complete(args.output_dir):
        print(
            f"Training already complete in {args.output_dir}; "
            "kept the saved model and held-out results."
        )
        return 0
    train_args = SimpleNamespace(
        seed=args.seed,
        model=args.model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        group_by_length=True,
        checkpoint_steps=200,
        fp16=args.fp16,
        bf16=args.bf16,
        unweighted_loss=False,
    )
    original.train(train_args, splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
