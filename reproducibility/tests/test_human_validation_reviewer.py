from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.inconsistency.human_validation_reviewer import (
    CodingStore,
    PRIMARY_SCHEMA,
    SECOND_SCHEMA,
    detect_schema,
    text_sha256,
    validate_judgment,
)


PRIMARY_FIELDS = [
    "validation_id",
    "country_code",
    "manifesto_text",
    "speech_text",
    "human_label",
    "human_comparable",
    "human_confidence",
    "human_notes",
]


def write_primary(path: Path) -> None:
    rows = [
        {
            "validation_id": "MHV001",
            "country_code": "LV",
            "manifesto_text": "Ārpus, komats",
            "speech_text": "Runa\nsecond line",
            "human_label": "",
            "human_comparable": "",
            "human_confidence": "",
            "human_notes": "",
        },
        {
            "validation_id": "MHV002",
            "country_code": "DK",
            "manifesto_text": "Earlier",
            "speech_text": "Later",
            "human_label": "unrelated",
            "human_comparable": "no",
            "human_confidence": "high",
            "human_notes": "different propositions",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_translations(path: Path, *, bad_manifesto_hash: bool = False) -> None:
    fields = [
        "validation_id",
        "country_code",
        "manifesto_text_sha256",
        "speech_text_sha256",
        "manifesto_text_en",
        "speech_text_en",
        "translation_provider",
        "translation_model",
    ]
    row = {
        "validation_id": "MHV001",
        "country_code": "LV",
        "manifesto_text_sha256": (
            "0" * 64 if bad_manifesto_hash else text_sha256("Ārpus, komats")
        ),
        "speech_text_sha256": text_sha256("Runa\nsecond line"),
        "manifesto_text_en": "Outside, comma",
        "speech_text_en": "Speech\nsecond line",
        "translation_provider": "openai",
        "translation_model": "translation-test-model",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_suggestions(
    path: Path,
    *,
    bad_translation_hash: bool = False,
) -> None:
    fields = [
        "validation_id",
        "country_code",
        "manifesto_text_sha256",
        "speech_text_sha256",
        "manifesto_translation_sha256",
        "speech_translation_sha256",
        "suggested_label",
        "suggested_comparable",
        "suggested_confidence",
        "rationale",
        "translation_warning",
        "review_focus",
        "assessment_provider",
        "assessment_model",
    ]
    row = {
        "validation_id": "MHV001",
        "country_code": "LV",
        "manifesto_text_sha256": text_sha256("Ārpus, komats"),
        "speech_text_sha256": text_sha256("Runa\nsecond line"),
        "manifesto_translation_sha256": (
            "0" * 64 if bad_translation_hash else text_sha256("Outside, comma")
        ),
        "speech_translation_sha256": text_sha256("Speech\nsecond line"),
        "suggested_label": "unrelated",
        "suggested_comparable": "no",
        "suggested_confidence": "high",
        "rationale": "The two supplied propositions concern different issues.",
        "translation_warning": "false",
        "review_focus": "Confirm that no shared substantive proposition is present.",
        "assessment_provider": "openai",
        "assessment_model": "assessment-test-model",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def test_store_updates_atomically_and_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "coding.csv"
    write_primary(path)
    store = CodingStore(path)
    assert store.progress() == {
        "complete": 1,
        "total": 2,
        "remaining": 1,
        "first_incomplete": 0,
    }

    saved = store.update(
        {
            "validation_id": "MHV001",
            "label": "inconsistent",
            "comparable": "yes",
            "confidence": "medium",
            "notes": "Latviešu piezīme",
        }
    )

    assert saved["complete"] is True
    assert store.backup_path.exists()
    reloaded = CodingStore(path)
    assert reloaded.progress()["complete"] == 2
    assert reloaded.public_row(0)["manifesto_text"] == "Ārpus, komats"
    assert reloaded.public_row(0)["speech_text"] == "Runa\nsecond line"
    assert reloaded.public_row(0)["notes"] == "Latviešu piezīme"


def test_invalid_label_comparability_combinations_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires comparable=yes"):
        validate_judgment("consistent", "no", "high", require_complete=True)
    with pytest.raises(ValueError, match="requires comparable=no"):
        validate_judgment("unrelated", "yes", "low", require_complete=True)
    validate_judgment("ambiguous", "no", "medium", require_complete=True)


def test_detects_primary_and_second_coder_schemas() -> None:
    assert detect_schema(PRIMARY_FIELDS) == PRIMARY_SCHEMA
    second = [
        "validation_id",
        "country_code",
        "manifesto_text",
        "speech_text",
        "second_coder_label",
        "second_coder_comparable",
        "second_coder_confidence",
        "second_coder_notes",
    ]
    assert detect_schema(second) == SECOND_SCHEMA


def test_store_loads_hash_matched_english_translations(tmp_path: Path) -> None:
    coding = tmp_path / "coding.csv"
    translations = tmp_path / "translations.csv"
    write_primary(coding)
    write_translations(translations)

    row = CodingStore(coding, translations).public_row(0)

    assert row["translations_available"] is True
    assert row["manifesto_text_en"] == "Outside, comma"
    assert row["speech_text_en"] == "Speech\nsecond line"
    assert row["translation_provider"] == "openai"


def test_store_rejects_stale_translation_hashes(tmp_path: Path) -> None:
    coding = tmp_path / "coding.csv"
    translations = tmp_path / "translations.csv"
    write_primary(coding)
    write_translations(translations, bad_manifesto_hash=True)

    with pytest.raises(ValueError, match="source-text hash mismatch"):
        CodingStore(coding, translations)


def test_store_loads_reasoned_suggestions_without_overwriting_labels(
    tmp_path: Path,
) -> None:
    coding = tmp_path / "coding.csv"
    translations = tmp_path / "translations.csv"
    suggestions = tmp_path / "suggestions.csv"
    write_primary(coding)
    write_translations(translations)
    write_suggestions(suggestions)

    row = CodingStore(coding, translations, suggestions).public_row(0)

    assert row["suggestion_available"] is True
    assert row["suggested_label"] == "unrelated"
    assert row["suggested_comparable"] == "no"
    assert row["suggestion_rationale"].startswith("The two supplied")
    assert row["label"] == ""
    assert row["comparable"] == ""


def test_store_rejects_suggestions_for_stale_translations(tmp_path: Path) -> None:
    coding = tmp_path / "coding.csv"
    translations = tmp_path / "translations.csv"
    suggestions = tmp_path / "suggestions.csv"
    write_primary(coding)
    write_translations(translations)
    write_suggestions(suggestions, bad_translation_hash=True)

    with pytest.raises(ValueError, match="translation hash mismatch"):
        CodingStore(coding, translations, suggestions)
