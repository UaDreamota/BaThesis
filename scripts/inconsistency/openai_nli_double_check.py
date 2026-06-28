from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "nli_inconsistency"
    / "CZ"
    / "CZ_Gal_Tan_nli_pairs.csv"
)
DEFAULT_MODEL = "gpt-4.1-mini"
API_URL = "https://api.openai.com/v1/responses"
VALID_LABELS = {"entailment", "inconsistent", "neutral"}


SYSTEM_PROMPT = """You are checking political-text NLI labels for a research dataset.

Classify the relationship between a party speech excerpt and one manifesto claim.

Definitions:
- entailment: the speech clearly supports, implies, restates, or is consistent with the manifesto claim.
- inconsistent: the speech clearly conflicts with, undermines, deviates from, or takes a position difficult to reconcile with the manifesto claim.
- neutral: the relationship is unclear, too indirect, about a different issue, merely procedural, or has insufficient evidence.

Important rules:
- The manifesto claim is the hypothesis to check.
- The party speech is the evidence/premise.
- Use inconsistent only for a real substantive mismatch, not just a missing topic match.
- If either text is too vague, procedural, or not about the same policy issue, choose neutral.
- Do not infer party ideology beyond the two supplied texts.
- Return only the requested JSON object."""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "manifesto_speech_nli_check",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {
                "type": "string",
                "enum": ["entailment", "inconsistent", "neutral"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "rationale": {
                "type": "string",
                "description": "One short sentence explaining the decision.",
            },
        },
        "required": ["label", "confidence", "rationale"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use OpenAI to double-check NLI classifications between manifesto "
            "claims and party speech excerpts from a pairs CSV."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output CSV. Defaults to <input stem>_openai_double_check.csv.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--project-env",
        default="OPENAI_PROJECT",
        help=(
            "Optional environment variable containing the OpenAI project id. "
            "If unset, OPEN_API_PROJECT_KEY is also tried for this repo."
        ),
    )
    parser.add_argument("--limit", default=0, type=int, help="Maximum rows to process; 0 means all.")
    parser.add_argument("--sleep", default=0.2, type=float, help="Seconds to sleep between requests.")
    parser.add_argument("--max-retries", default=4, type=int)
    parser.add_argument(
        "--nli-labels",
        nargs="*",
        default=None,
        help=(
            "Optional existing nli_label values to check, e.g. "
            "--nli-labels contradiction. Omit to check all rows."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output and reprocess rows from scratch.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_openai_double_check{input_path.suffix}")


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"speech_text", "manifesto_text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    return rows


def load_completed_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "nli_pair_id" not in (reader.fieldnames or []):
            return set()
        return {
            str(row.get("nli_pair_id", "")).strip()
            for row in reader
            if str(row.get("openai_label", "")).strip()
        }


def should_keep(row: dict[str, str], labels: set[str] | None) -> bool:
    if labels is None:
        return True
    return str(row.get("nli_label", "")).strip().lower() in labels


def row_id(row: dict[str, str], fallback_index: int) -> str:
    value = str(row.get("nli_pair_id", "")).strip()
    return value if value else str(fallback_index)


def comparable_label(label: object) -> str:
    value = str(label).strip().lower()
    if value == "contradiction":
        return "inconsistent"
    return value


def build_user_prompt(row: dict[str, str]) -> str:
    party = row.get("party", "")
    date = row.get("date", "")
    manifesto_party = row.get("manifesto_partyabbrev") or row.get("manifesto_partyname", "")
    existing = row.get("nli_label", "")
    return (
        "Classify the relation between the speech and manifesto claim.\n\n"
        f"Party in speech: {party}\n"
        f"Speech date: {date}\n"
        f"Manifesto party: {manifesto_party}\n"
        f"Existing automated NLI label, if any: {existing}\n\n"
        "Speech evidence/premise:\n"
        f"{row.get('speech_text', '').strip()}\n\n"
        "Manifesto claim/hypothesis:\n"
        f"{row.get('manifesto_text', '').strip()}"
    )


def extract_output_text(response_json: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(str(content.get("text", "")))
    return "\n".join(texts).strip()


def classify_with_openai(
    row: dict[str, str],
    api_key: str,
    project_id: str | None,
    model: str,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": build_user_prompt(row),
        "text": {"format": RESPONSE_SCHEMA},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if project_id:
        headers["OpenAI-Project"] = project_id

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
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
            response_json = response.json()
            output_text = extract_output_text(response_json)
            parsed = json.loads(output_text)
            label = str(parsed.get("label", "")).strip().lower()
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label from OpenAI: {label!r}")
            return {
                "openai_label": label,
                "openai_confidence": parsed.get("confidence", ""),
                "openai_rationale": str(parsed.get("rationale", "")).strip(),
                "openai_model": response_json.get("model", model),
                "openai_response_id": response_json.get("id", ""),
                "openai_error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))

    return {
        "openai_label": "",
        "openai_confidence": "",
        "openai_rationale": "",
        "openai_model": model,
        "openai_response_id": "",
        "openai_error": last_error or "unknown error",
    }


def output_fieldnames(input_fieldnames: list[str]) -> list[str]:
    extra = [
        "openai_label",
        "openai_confidence",
        "openai_rationale",
        "openai_model",
        "openai_response_id",
        "openai_error",
        "openai_agrees_with_nli",
    ]
    return input_fieldnames + [field for field in extra if field not in input_fieldnames]


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(BASE_DIR / ".env")

    if args.limit < 0:
        raise ValueError("--limit must be >= 0.")
    if args.sleep < 0:
        raise ValueError("--sleep must be >= 0.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")

    api_key = os.getenv(args.api_key_env) or os.getenv("OPEN_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Missing API key in environment variable {args.api_key_env} "
            "or OPEN_API_KEY."
        )
    project_id = os.getenv(args.project_env) or os.getenv("OPEN_API_PROJECT_KEY")

    output_path = args.output or default_output_path(args.input)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    rows = load_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    fieldnames = output_fieldnames(list(rows[0].keys()))

    label_filter = (
        {label.strip().lower() for label in args.nli_labels if label.strip()}
        if args.nli_labels is not None
        else None
    )
    completed = set() if args.overwrite else load_completed_pair_ids(output_path)

    processed = 0
    skipped = 0
    buffer: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        pair_id = row_id(row, index)
        if pair_id in completed or not should_keep(row, label_filter):
            skipped += 1
            continue
        if args.limit and processed >= args.limit:
            break

        result = classify_with_openai(
            row,
            api_key,
            project_id,
            args.model,
            args.max_retries,
        )
        existing_label = comparable_label(row.get("nli_label", ""))
        result["openai_agrees_with_nli"] = (
            comparable_label(result["openai_label"]) == existing_label
            if existing_label and result["openai_label"]
            else ""
        )
        buffer.append({**row, **result})
        processed += 1

        if len(buffer) >= 10:
            append_rows(output_path, fieldnames, buffer)
            buffer.clear()
        if args.sleep:
            time.sleep(args.sleep)

        if processed == 1 or processed % 25 == 0:
            print(
                f"Processed {processed:,} row(s); skipped {skipped:,}; output={output_path}",
                flush=True,
            )

    if buffer:
        append_rows(output_path, fieldnames, buffer)

    print(f"Done. Processed {processed:,} row(s), skipped {skipped:,}.", flush=True)
    print(f"Saved OpenAI double-check results to: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
