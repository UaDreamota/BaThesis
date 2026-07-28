from __future__ import annotations

"""Create a resumable, blinded English-translation sidecar for validation pairs.

Only the blinded coding CSV is read. Classifier predictions, probabilities,
private keys, and regression outputs are never loaded. Existing translations
are reused when their source-text hashes still match the coding packet.
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
DEFAULT_PACKET = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "macro_human_validation"
    / "BLINDED_PRIMARY_CODING.csv"
)
DEFAULT_OUTPUT_NAME = "BLINDED_ENGLISH_TRANSLATIONS.csv"
OPENAI_URL = "https://api.openai.com/v1/responses"
OUTPUT_FIELDS = [
    "validation_id",
    "country_code",
    "manifesto_text_sha256",
    "speech_text_sha256",
    "manifesto_text_en",
    "speech_text_en",
    "translation_provider",
    "translation_model",
    "response_id",
    "translated_at_utc",
]
SYSTEM_PROMPT = """You are a professional translator preparing political texts for blinded human validation.
Translate the manifesto statement and parliamentary speech independently into faithful, natural English.
Preserve all numbers, negation, modality, uncertainty, policy direction, named entities, and institutional terms.
Do not summarize, explain, harmonize the two texts, infer missing context, or make their positions more similar.
If a text is already English, reproduce it faithfully. Return only the required JSON object."""
OPENAI_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "blinded_pair_english_translation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "manifesto_text_en": {"type": "string"},
            "speech_text_en": {"type": "string"},
        },
        "required": ["manifesto_text_en", "speech_text_en"],
    },
}


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--input", type=Path, default=DEFAULT_PACKET)
    out.add_argument("--output", type=Path)
    out.add_argument("--model", default="gpt-5.4-mini")
    out.add_argument("--workers", type=int, default=6)
    out.add_argument("--max-retries", type=int, default=5)
    out.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Translate at most this many currently missing rows; zero means all.",
    )
    out.add_argument(
        "--check",
        action="store_true",
        help="Validate an existing sidecar against the coding packet without API calls.",
    )
    out.add_argument("--force", action="store_true", help="Regenerate all translations.")
    return out


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_coding_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"validation_id", "country_code", "manifesto_text", "speech_text"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Coding file is missing columns: {missing}")
        rows = [dict(row) for row in reader]
    ids = [row["validation_id"].strip() for row in rows]
    if not ids or any(not value for value in ids):
        raise ValueError("Coding file must contain non-empty validation_id values")
    if len(ids) != len(set(ids)):
        raise ValueError("validation_id values must be unique")
    if any(not row["manifesto_text"].strip() or not row["speech_text"].strip() for row in rows):
        raise ValueError("Every row must contain both source texts")
    return rows


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(OUTPUT_FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Translation sidecar is missing columns: {missing}")
        rows = [dict(row) for row in reader]
    ids = [row["validation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Translation sidecar contains duplicate validation_id values")
    return {row["validation_id"]: row for row in rows}


def cache_valid(source: dict[str, str], cached: dict[str, str] | None) -> bool:
    if not cached:
        return False
    return (
        cached.get("country_code") == source["country_code"]
        and cached.get("manifesto_text_sha256") == text_sha256(source["manifesto_text"])
        and cached.get("speech_text_sha256") == text_sha256(source["speech_text"])
        and bool(cached.get("manifesto_text_en", "").strip())
        and bool(cached.get("speech_text_en", "").strip())
    )


def write_atomic(
    path: Path,
    source_rows: list[dict[str, str]],
    translated: dict[str, dict[str, str]],
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
                row = translated.get(source["validation_id"])
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
                timeout=180,
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


def translated_row(
    source: dict[str, str],
    manifesto_text_en: str,
    speech_text_en: str,
    *,
    provider: str,
    model: str,
    response_id: str,
) -> dict[str, str]:
    manifesto_text_en = manifesto_text_en.strip()
    speech_text_en = speech_text_en.strip()
    if not manifesto_text_en or not speech_text_en:
        raise ValueError(f"Empty translation returned for {source['validation_id']}")
    return {
        "validation_id": source["validation_id"],
        "country_code": source["country_code"],
        "manifesto_text_sha256": text_sha256(source["manifesto_text"]),
        "speech_text_sha256": text_sha256(source["speech_text"]),
        "manifesto_text_en": manifesto_text_en,
        "speech_text_en": speech_text_en,
        "translation_provider": provider,
        "translation_model": model,
        "response_id": response_id,
        "translated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def translate_openai(
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
            "manifesto_text": source["manifesto_text"],
            "speech_text": source["speech_text"],
        },
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": prompt,
        "text": {"format": OPENAI_FORMAT},
        "max_output_tokens": 6_000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    project_id = os.getenv("OPENAI_PROJECT") or os.getenv("OPEN_API_PROJECT_KEY")
    if project_id:
        headers["OpenAI-Project"] = project_id
    response = post_json(headers, payload, max_retries)
    result = json.loads(response_text(response))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a translation object for {source['validation_id']}")
    return translated_row(
        source,
        str(result.get("manifesto_text_en", "")),
        str(result.get("speech_text_en", "")),
        provider="openai",
        model=str(response.get("model", model)),
        response_id=str(response.get("id", "")),
    )


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    source_path = args.input.resolve()
    output_path = (args.output or source_path.with_name(DEFAULT_OUTPUT_NAME)).resolve()
    source_rows = read_coding_rows(source_path)
    existing = read_existing(output_path)
    valid = {
        row["validation_id"]: existing[row["validation_id"]]
        for row in source_rows
        if not args.force and cache_valid(row, existing.get(row["validation_id"]))
    }
    missing = [row for row in source_rows if row["validation_id"] not in valid]

    if args.check:
        if missing:
            raise ValueError(
                f"Translation sidecar is incomplete or stale: {len(missing)} of "
                f"{len(source_rows)} rows require translation"
            )
        print(f"Validated {len(valid)} English-translated blinded rows in {output_path}")
        return

    if args.limit:
        missing = missing[: args.limit]
    print(
        f"Translation status: {len(valid)}/{len(source_rows)} cached; "
        f"processing {len(missing)} rows."
    )
    if not missing:
        print(f"English translations are complete: {output_path}")
        return

    for source in [row for row in missing if row["country_code"].upper() == "GB"]:
        valid[source["validation_id"]] = translated_row(
            source,
            source["manifesto_text"],
            source["speech_text"],
            provider="local",
            model="verbatim-english-copy",
            response_id="",
        )
        write_atomic(output_path, source_rows, valid)

    remote = [row for row in missing if row["country_code"].upper() != "GB"]
    if remote:
        load_dotenv(BASE_DIR / ".env")
        api_key = openai_key()
        completed = 0
        with ThreadPoolExecutor(max_workers=min(args.workers, len(remote))) as pool:
            futures = {
                pool.submit(
                    translate_openai,
                    source,
                    api_key=api_key,
                    model=args.model,
                    max_retries=args.max_retries,
                ): source
                for source in remote
            }
            for future in as_completed(futures):
                source = futures[future]
                valid[source["validation_id"]] = future.result()
                completed += 1
                write_atomic(output_path, source_rows, valid)
                print(
                    f"Translated {completed}/{len(remote)} API rows: "
                    f"{source['validation_id']} ({source['country_code']})"
                )

    remaining = [row for row in source_rows if not cache_valid(row, valid.get(row["validation_id"]))]
    print(
        f"Saved {len(valid)}/{len(source_rows)} English translations to {output_path}; "
        f"{len(remaining)} remain."
    )


if __name__ == "__main__":
    main()
