from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.inconsistency import classify_llm_consensus_deberta as classifier
from scripts.inconsistency import methodology_freeze_robustness as freeze
from scripts.inconsistency import poc_nli
from scripts.inconsistency import train_llm_consensus_deberta as training
from scripts.inconsistency.artifact_provenance import atomic_write_json


class ClassifierStateTests(unittest.TestCase):
    def test_model_change_invalidates_completed_or_partial_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"version": 1}', encoding="utf-8")
            source = root / "input.csv"
            source.write_text("manifesto_text,speech_text\nm,s\n", encoding="utf-8")
            output = root / "pairs.csv"
            state_path = root / "state.json"
            run_spec = classifier.classifier_run_spec(source, model, 512)
            atomic_write_json(
                state_path,
                classifier.initialized_state(run_spec),
            )

            (model / "config.json").write_text('{"version": 2}', encoding="utf-8")
            changed = classifier.classifier_run_spec(source, model, 512)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                classifier.validate_or_initialize_state(
                    state_path, output, changed
                )

    def test_pending_append_is_truncated_to_durable_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("model", encoding="utf-8")
            source = root / "input.csv"
            source.write_text("manifesto_text,speech_text\nm,s\n", encoding="utf-8")
            output = root / "pairs.csv"
            durable = b"header\nrow1\n"
            output.write_bytes(durable + b"partial-or-duplicated-row\n")
            run_spec = classifier.classifier_run_spec(source, model, 512)
            state = classifier.initialized_state(run_spec)
            state.update(
                {
                    "processed_rows": 1,
                    "output_bytes": len(durable),
                    "pending_chunk": {
                        "row_start": 1,
                        "row_end": 2,
                        "output_bytes_before": len(durable),
                    },
                }
            )
            state_path = root / "state.json"
            atomic_write_json(state_path, state)

            recovered = classifier.validate_or_initialize_state(
                state_path, output, run_spec
            )
            self.assertEqual(output.read_bytes(), durable)
            self.assertEqual(recovered["processed_rows"], 1)
            self.assertIsNone(recovered["pending_chunk"])

    def test_legacy_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("model", encoding="utf-8")
            source = root / "input.csv"
            source.write_text("manifesto_text,speech_text\nm,s\n", encoding="utf-8")
            output = root / "pairs.csv"
            output.write_text("old\n", encoding="utf-8")
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"processed_rows": 1, "complete": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Legacy"):
                classifier.validate_or_initialize_state(
                    state_path,
                    output,
                    classifier.classifier_run_spec(source, model, 512),
                )


class RetrievalAndNliCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.speeches = pd.DataFrame(
            {
                "retrieval_unit_id": ["speech:0"],
                "doc_key": ["party_202001"],
                "date": [pd.Timestamp("2020-02-01")],
                "party": ["P"],
                "plda_doc_id": ["speech"],
                "text": ["speech text"],
            }
        )
        self.manifesto = pd.DataFrame(
            {
                "doc_key": ["party_202001"],
                "quasi_sentence_id": ["q1"],
                "manifesto_text": ["manifesto text"],
                "predicted_topic": ["topic_1"],
            }
        )

    def test_retrieval_signature_covers_both_texts(self) -> None:
        config = {"model": "embedding", "top_k": 3}
        base = poc_nli.hybrid_retrieval_signature(
            self.speeches,
            self.manifesto,
            "retrieval_unit_id",
            config,
        )
        changed_speech = self.speeches.copy()
        changed_speech.loc[0, "text"] = "different speech"
        changed_manifesto = self.manifesto.copy()
        changed_manifesto.loc[0, "manifesto_text"] = "different manifesto"
        self.assertNotEqual(
            base,
            poc_nli.hybrid_retrieval_signature(
                changed_speech,
                self.manifesto,
                "retrieval_unit_id",
                config,
            ),
        )
        self.assertNotEqual(
            base,
            poc_nli.hybrid_retrieval_signature(
                self.speeches,
                changed_manifesto,
                "retrieval_unit_id",
                config,
            ),
        )

    def test_nli_checkpoint_binds_pair_identity_and_order(self) -> None:
        pairs = pd.DataFrame(
            {
                "nli_pair_id": [0, 1],
                "doc_key": ["d1", "d2"],
                "manifesto_text": ["m1", "m2"],
                "speech_text": ["s1", "s2"],
            }
        )
        labels = ["entailment", "neutral", "contradiction"]
        fingerprints = poc_nli.nli_pair_fingerprints(pairs)
        run_spec = poc_nli.nli_run_spec(pairs, "model-name", 512, labels)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.csv"
            _, state = poc_nli.load_nli_checkpoint(
                checkpoint, labels, fingerprints, run_spec
            )
            row = {
                "nli_prob_entailment": 0.1,
                "nli_prob_neutral": 0.2,
                "nli_prob_contradiction": 0.7,
                "nli_label": "contradiction",
            }
            poc_nli.append_nli_checkpoint(
                checkpoint, [row], fingerprints[:1], state
            )
            loaded, _ = poc_nli.load_nli_checkpoint(
                checkpoint, labels, fingerprints, run_spec
            )
            self.assertEqual(len(loaded), 1)

            changed_pairs = pairs.copy()
            changed_pairs.loc[0, "speech_text"] = "changed"
            changed_fingerprints = poc_nli.nli_pair_fingerprints(changed_pairs)
            changed_spec = poc_nli.nli_run_spec(
                changed_pairs, "model-name", 512, labels
            )
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                poc_nli.load_nli_checkpoint(
                    checkpoint,
                    labels,
                    changed_fingerprints,
                    changed_spec,
                )


class SplitAndTemporalTests(unittest.TestCase):
    def test_complete_training_artifacts_prevent_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model = output / "model"
            model.mkdir()
            for relative in (
                "train.csv",
                "validation.csv",
                "test.csv",
                "split_integrity_assertions.json",
                "test_results.json",
                "model/config.json",
                "model/tokenizer_config.json",
                "model/model.safetensors",
            ):
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("complete", encoding="utf-8")
            self.assertTrue(training.training_artifacts_complete(output))
            (output / "test_results.json").unlink()
            self.assertFalse(training.training_artifacts_complete(output))

    def test_split_fingerprint_covers_text_and_label(self) -> None:
        row = {
            "sample_id": "1",
            "group_id": "g",
            "manifesto_doc_key": "d",
            "country": "AT",
            "topic": "Macro",
            "manifesto_text": "manifesto",
            "speech_text": "speech",
            "label_name": "consistent",
            "labels": 0,
            "agreement_count": 3,
        }
        changed = dict(row)
        changed["speech_text"] = "different speech"
        self.assertNotEqual(
            training.split_fingerprint([row]),
            training.split_fingerprint([changed]),
        )

    def test_matching_doc_key_cannot_override_future_date(self) -> None:
        batch = pd.DataFrame(
            {
                "nli_pair_id": ["1"],
                "date": ["2020-01-15"],
                "party": ["P"],
                "month": ["2020-01"],
                "plda_doc_id": ["speech"],
                "doc_key": ["d"],
                "manifesto_effective_date": ["2020-02-01"],
                "retrieval_rank": [1],
                "embedding_score": [0.5],
                "classifier_prob_inconsistent": [0.8],
                "classifier_confidence": [0.8],
                "classifier_label": ["inconsistent"],
                "speech_text": ["speech"],
            }
        )
        bridge = pd.DataFrame(
            {
                "speech_party": ["P"],
                "date": [pd.Timestamp("2020-01-15")],
                "expected_doc_key": ["d"],
                "expected_selection_method": [
                    "latest_manifesto_on_or_before_speech_date"
                ],
                "expected_manifesto_effective_date": [
                    pd.Timestamp("2020-02-01")
                ],
            }
        )
        normalized = freeze.normalize_pair_batch(
            batch, "AT", "Macroeconomics", bridge
        )
        self.assertEqual(
            normalized.loc[0, "temporal_status"],
            "future_manifesto_different_month",
        )
        self.assertFalse(bool(normalized.loc[0, "strict_date_valid"]))


if __name__ == "__main__":
    unittest.main()
