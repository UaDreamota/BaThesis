from __future__ import annotations

"""Generate resumable, blinded AI suggestions for human-validation review.

The script reads only the blinded coding packet and its English-translation
sidecar. It never sends existing human judgments, classifier predictions,
probabilities, private keys, or regression results to the assessment model.
Suggestions are written to a separate file and never overwrite human labels.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_DIR = BASE_DIR / "outputs" / "test_speeches" / "macro_human_validation"
DEFAULT_CODING = DEFAULT_PACKET_DIR / "BLINDED_PRIMARY_CODING.csv"
DEFAULT_TRANSLATIONS = DEFAULT_PACKET_DIR / "BLINDED_ENGLISH_TRANSLATIONS.csv"
DEFAULT_OUTPUT = DEFAULT_PACKET_DIR / "BLINDED_AI_REVIEW_SUGGESTIONS.csv"
OPENAI_URL = "https://api.openai.com/v1/responses"
LABELS = ("consistent", "inconsistent", "unrelated", "ambiguous")
CONFIDENCE = ("low", "medium", "high")
OUTPUT_FIELDS = [
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
    "response_id",
    "assessed_at_utc",
]
SYSTEM_PROMPT = """You are an exacting research adjudicator assessing a blinded pair of political texts.
The manifesto statement is earlier and the parliamentary speech is later. Judge only the supplied texts.

Required sequence:
1. Decide whether both texts address the same substantive proposition closely enough for a directional consistency judgment.
2. Assign exactly one label:
   - consistent: comparable statements that support, repeat, or are mutually compatible on the proposition;
   - inconsistent: comparable statements that advocate clearly incompatible positions or directions;
   - unrelated: not sufficiently comparable, including merely sharing a broad economic topic;
   - ambiguous: a relationship may exist, but wording or missing context prevents a defensible judgment.
3. Assign low, medium, or high confidence based on textual evidence clarity.

Consistent and inconsistent require comparable=yes. Unrelated requires comparable=no. Ambiguous may be yes or no.
Do not infer party ideology, intentions, or facts outside the texts. Different actors, periods, jurisdictions, levels of government,
specificity, or emphasis are not automatically conflicts. A factual difference is inconsistent only when the two supplied claims
are genuinely incompatible rather than context-dependent. Translate neither text anew unless checking a possible translation issue.

Give a concise rationale that identifies each text's proposition and explains the relationship. Give one short review_focus telling
the human what matters most to verify. Set translation_warning=true only when the English rendering appears uncertain or potentially
meaning-changing compared with the original. Return only the required JSON object."""
OPENAI_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "blinded_human_validation_suggestion",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "suggested_label": {"type": "string", "enum": list(LABELS)},
            "suggested_comparable": {"type": "string", "enum": ["yes", "no"]},
            "suggested_confidence": {"type": "string", "enum": list(CONFIDENCE)},
            "rationale": {"type": "string"},
            "translation_warning": {"type": "boolean"},
            "review_focus": {"type": "string"},
        },
        "required": [
            "suggested_label",
            "suggested_comparable",
            "suggested_confidence",
            "rationale",
            "translation_warning",
            "review_focus",
        ],
    },
}


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--coding-file", type=Path, default=DEFAULT_CODING)
    out.add_argument("--translations-file", type=Path, default=DEFAULT_TRANSLATIONS)
    out.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--model", default="gpt-5.4")
    out.add_argument("--workers", type=int, default=6)
    out.add_argument("--max-retries", type=int, default=5)
    out.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Assess at most this many currently missing rows; zero means all.",
    )
    out.add_argument("--check", action="store_true")
    out.add_argument("--force", action="store_true")
    return out


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def load_inputs(
    coding_path: Path,
    translations_path: Path,
) -> list[dict[str, str]]:
    coding_rows, coding_fields = read_csv(coding_path)
    required_coding = {"validation_id", "country_code", "manifesto_text", "speech_text"}
    missing = sorted(required_coding - set(coding_fields))
    if missing:
        raise ValueError(f"Coding file is missing columns: {missing}")
    coding_ids = [row["validation_id"] for row in coding_rows]
    if not coding_ids or len(coding_ids) != len(set(coding_ids)):
        raise ValueError("Coding validation_id values must be non-empty and unique")

    translated_rows, translated_fields = read_csv(translations_path)
    required_translated = {
        "validation_id",
        "country_code",
        "manifesto_text_sha256",
        "speech_text_sha256",
        "manifesto_text_en",
        "speech_text_en",
    }
    missing = sorted(required_translated - set(translated_fields))
    if missing:
        raise ValueError(f"Translation sidecar is missing columns: {missing}")
    translated_ids = [row["validation_id"] for row in translated_rows]
    if len(translated_ids) != len(set(translated_ids)):
        raise ValueError("Translation validation_id values must be unique")
    translated = {row["validation_id"]: row for row in translated_rows}

    inputs: list[dict[str, str]] = []
    for source in coding_rows:
        validation_id = source["validation_id"]
        english = translated.get(validation_id)
        if english is None:
            raise ValueError(f"Missing English translation for {validation_id}")
        if english["country_code"] != source["country_code"]:
            raise ValueError(f"Translation country mismatch for {validation_id}")
        if english["manifesto_text_sha256"] != text_sha256(
            source["manifesto_text"]
        ) or english["speech_text_sha256"] != text_sha256(source["speech_text"]):
            raise ValueError(f"Translation source-text hash mismatch for {validation_id}")
        if not english["manifesto_text_en"].strip() or not english["speech_text_en"].strip():
            raise ValueError(f"Empty English translation for {validation_id}")
        inputs.append(
            {
                "validation_id": validation_id,
                "country_code": source["country_code"],
                "manifesto_text": source["manifesto_text"],
                "speech_text": source["speech_text"],
                "manifesto_text_en": english["manifesto_text_en"],
                "speech_text_en": english["speech_text_en"],
            }
        )
    return inputs


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows, fields = read_csv(path)
    missing = sorted(set(OUTPUT_FIELDS) - set(fields))
    if missing:
        raise ValueError(f"Suggestion sidecar is missing columns: {missing}")
    ids = [row["validation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Suggestion sidecar contains duplicate validation_id values")
    return {row["validation_id"]: row for row in rows}


def cache_valid(source: dict[str, str], cached: dict[str, str] | None) -> bool:
    if not cached:
        return False
    label = cached.get("suggested_label", "")
    comparable = cached.get("suggested_comparable", "")
    confidence = cached.get("suggested_confidence", "")
    consistent_combination = (
        label in {"consistent", "inconsistent"} and comparable == "yes"
    ) or (label == "unrelated" and comparable == "no") or (
        label == "ambiguous" and comparable in {"yes", "no"}
    )
    return (
        cached.get("country_code") == source["country_code"]
        and cached.get("manifesto_text_sha256") == text_sha256(source["manifesto_text"])
        and cached.get("speech_text_sha256") == text_sha256(source["speech_text"])
        and cached.get("manifesto_translation_sha256")
        == text_sha256(source["manifesto_text_en"])
        and cached.get("speech_translation_sha256") == text_sha256(source["speech_text_en"])
        and label in LABELS
        and confidence in CONFIDENCE
        and consistent_combination
        and bool(cached.get("rationale", "").strip())
        and bool(cached.get("review_focus", "").strip())
    )


def write_atomic(
    path: Path,
    source_rows: list[dict[str, str]],
    assessed: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="raise")
            writer.writeheader()
            for source in source_rows:
                row = assessed.get(source["validation_id"])
                if row:
                    writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def openai_key() -> str:
    for name in ("OPENAI_API_KEY", "OPEN_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError("Missing OpenAI API key; tried OPENAI_API_KEY and OPEN_API_KEY")


def post_json(
    headers: dict[str, str],
    payload: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    last_error = "unknown request error"
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=240,
            )
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(2**attempt, 45)
                    time.sleep(delay)
                    continue
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                raise RuntimeError(f"{error}: {response.text.strip()}") from error
            return response.json()
        except Exception as error:
            last_error = str(error)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 45))
    raise RuntimeError(last_error)


def response_text(response: dict[str, Any]) -> str:
    values: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                values.append(str(content.get("text", "")))
    return "\n".join(values).strip()


def assess_one(
    source: dict[str, str],
    *,
    api_key: str,
    model: str,
    max_retries: int,
) -> dict[str, str]:
    prompt = json.dumps(
        {
            "validation_id": source["validation_id"],
            "country_code_language_hint": source["country_code"],
            "earlier_manifesto_english": source["manifesto_text_en"],
            "later_speech_english": source["speech_text_en"],
            "earlier_manifesto_original": source["manifesto_text"],
            "later_speech_original": source["speech_text"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "text": {"format": OPENAI_FORMAT},
        "max_output_tokens": 2_000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    project_id = os.getenv("OPENAI_PROJECT") or os.getenv("OPEN_API_PROJECT_KEY")
    if project_id:
        headers["OpenAI-Project"] = project_id
    response = post_json(headers, payload, max_retries)
    result = json.loads(response_text(response))
    if not isinstance(result, dict):
        raise ValueError(f"Expected an assessment object for {source['validation_id']}")

    label = str(result.get("suggested_label", "")).strip().lower()
    comparable = str(result.get("suggested_comparable", "")).strip().lower()
    confidence = str(result.get("suggested_confidence", "")).strip().lower()
    if label not in LABELS or confidence not in CONFIDENCE:
        raise ValueError(f"Invalid assessment values for {source['validation_id']}")
    if label in {"consistent", "inconsistent"} and comparable != "yes":
        raise ValueError(f"{label} requires comparable=yes for {source['validation_id']}")
    if label == "unrelated" and comparable != "no":
        raise ValueError(f"unrelated requires comparable=no for {source['validation_id']}")
    if label == "ambiguous" and comparable not in {"yes", "no"}:
        raise ValueError(f"ambiguous requires yes/no comparability for {source['validation_id']}")
    rationale = str(result.get("rationale", "")).strip()
    review_focus = str(result.get("review_focus", "")).strip()
    if not rationale or not review_focus:
        raise ValueError(f"Missing assessment explanation for {source['validation_id']}")

    return {
        "validation_id": source["validation_id"],
        "country_code": source["country_code"],
        "manifesto_text_sha256": text_sha256(source["manifesto_text"]),
        "speech_text_sha256": text_sha256(source["speech_text"]),
        "manifesto_translation_sha256": text_sha256(source["manifesto_text_en"]),
        "speech_translation_sha256": text_sha256(source["speech_text_en"]),
        "suggested_label": label,
        "suggested_comparable": comparable,
        "suggested_confidence": confidence,
        "rationale": rationale,
        "translation_warning": "true" if result.get("translation_warning") is True else "false",
        "review_focus": review_focus,
        "assessment_provider": "openai",
        "assessment_model": str(response.get("model", model)),
        "response_id": str(response.get("id", "")),
        "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    source_rows = load_inputs(args.coding_file.resolve(), args.translations_file.resolve())
    existing = read_existing(args.output.resolve())
    valid = {
        row["validation_id"]: existing[row["validation_id"]]
        for row in source_rows
        if not args.force and cache_valid(row, existing.get(row["validation_id"]))
    }
    missing = [row for row in source_rows if row["validation_id"] not in valid]

    if args.check:
        if missing:
            raise ValueError(
                f"Suggestion sidecar is incomplete or stale: {len(missing)} of "
                f"{len(source_rows)} rows require assessment"
            )
        print(f"Validated {len(valid)} blinded reasoned suggestions in {args.output.resolve()}")
        return

    if args.limit:
        missing = missing[: args.limit]
    print(
        f"Suggestion status: {len(valid)}/{len(source_rows)} cached; "
        f"assessing {len(missing)} rows."
    )
    if not missing:
        print(f"Reasoned suggestions are complete: {args.output.resolve()}")
        return

    load_dotenv(BASE_DIR / ".env")
    api_key = openai_key()
    completed = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(missing))) as pool:
        futures = {
            pool.submit(
                assess_one,
                source,
                api_key=api_key,
                model=args.model,
                max_retries=args.max_retries,
            ): source
            for source in missing
        }
        for future in as_completed(futures):
            source = futures[future]
            valid[source["validation_id"]] = future.result()
            completed += 1
            write_atomic(args.output.resolve(), source_rows, valid)
            print(
                f"Assessed {completed}/{len(missing)}: "
                f"{source['validation_id']} ({source['country_code']})"
            )

    remaining = [row for row in source_rows if not cache_valid(row, valid.get(row["validation_id"]))]
    print(
        f"Saved {len(valid)}/{len(source_rows)} suggestions to {args.output.resolve()}; "
        f"{len(remaining)} remain."
    )


if __name__ == "__main__":
    main()
