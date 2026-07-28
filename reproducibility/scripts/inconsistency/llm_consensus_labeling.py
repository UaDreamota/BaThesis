from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {chr(34), chr(39)}:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2_147_483_647)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = BASE_DIR / 'outputs' / 'test_speeches' / 'nli_inconsistency'
DEFAULT_SAMPLE = DEFAULT_ROOT / 'llm_consensus' / 'llm_consensus_sample.csv'
DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
DEFAULT_COUNTRIES = 'AT BE CZ DK EE ES FI GB GR IT LV NL NO PL PT SE'.split()
DEFAULT_TOPICS = ['Macroeconomics', 'Gal_Tan']
VALID_LABELS = {'unrelated', 'consistent', 'inconsistent', 'ambiguous'}
VALID_TYPES = {'none', 'surface_contradiction', 'factual_inconsistency', 'indirect_value_inconsistency', 'ambiguous'}
OPENAI_URL = 'https://api.openai.com/v1/responses'
ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

SYSTEM_PROMPT = '''You are labeling political-text pairs for a research dataset.
Classify the relationship between a parliamentary speech excerpt and a manifesto claim.
Labels:
- unrelated: not the same substantive policy issue, merely procedural, or too vague to compare.
- consistent: same substantive policy issue and reconcilable, including support, restatement, compatible detail, or no meaningful conflict.
- inconsistent: same substantive policy issue and positions are difficult to reconcile.
- ambiguous: possibly comparable, but evidence is insufficient or too unclear.
Inconsistency types: none, surface_contradiction, factual_inconsistency, indirect_value_inconsistency, ambiguous.
Rules: do not infer party ideology beyond the supplied texts; if texts discuss different issues choose unrelated; texts may be non-English; return only one JSON object.'''

OPENAI_SCHEMA: dict[str, Any] = {
    'type': 'json_schema',
    'name': 'manifesto_speech_inconsistency_label',
    'strict': True,
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'label': {'type': 'string', 'enum': sorted(VALID_LABELS)},
            'comparable': {'type': 'boolean'},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'inconsistency_type': {'type': 'string', 'enum': sorted(VALID_TYPES)},
            'rationale': {'type': 'string'},
        },
        'required': ['label', 'comparable', 'confidence', 'inconsistency_type', 'rationale'],
    },
}


def token(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('_')


def as_float(value: object) -> float | None:
    text = str(value or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: object) -> int | None:
    number = as_float(value)
    return None if number is None else int(number)


def as_bool(value: object) -> bool | None:
    text = str(value or '').strip().lower()
    if text in {'true', '1', 'yes', 'y'}:
        return True
    if text in {'false', '0', 'no', 'n'}:
        return False
    return None


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_label_shares(value: str) -> dict[str, float]:
    shares: dict[str, float] = {}
    for item in value.split(','):
        if not item.strip():
            continue
        label, share = item.split('=', 1)
        shares[label.strip().lower()] = float(share)
    total = sum(shares.values())
    if total <= 0:
        raise ValueError('label shares must sum to a positive value')
    return {label: share / total for label, share in shares.items()}


def pairs_path(root: Path, country: str, topic: str, model_name: str) -> Path | None:
    country = country.upper()
    topic_token = token(topic)
    model_token = token(model_name)
    country_dir = root / country
    candidates = [
        country_dir / f'{country}_{topic_token}_nli_{model_token}_pairs.csv',
        country_dir / f'{country}_{topic_token}_nli_pairs.csv',
    ]
    candidates.extend(sorted(country_dir.glob(f'{country}_{topic_token}_nli_*_pairs.csv')))
    for path in candidates:
        if path.exists():
            return path
    return None


def best_speech(row: dict[str, str]) -> str:
    for column in ['speech_text_for_nli', 'speech_segment_text', 'speech_text', 'speech_text_original', 'speech_text_original_full']:
        value = str(row.get(column, '')).strip()
        if value:
            return value
    return ''


def best_manifesto(row: dict[str, str]) -> str:
    return str(row.get('manifesto_text', '')).strip()


def word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text, flags=re.UNICODE))


def eligible(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not best_speech(row) or not best_manifesto(row):
        return False
    if as_bool(row.get('speech_filter_kept')) is False:
        return False
    if args.max_retrieval_rank > 0:
        rank = as_int(row.get('retrieval_rank'))
        if rank is not None and rank > args.max_retrieval_rank:
            return False
    if args.min_embedding_score is not None:
        score = as_float(row.get('embedding_score'))
        if score is None or score < args.min_embedding_score:
            return False
    if args.min_speech_words > 0:
        words = as_int(row.get('speech_word_count_for_nli'))
        if words is None:
            words = word_count(best_speech(row))
        if words < args.min_speech_words:
            return False
    return True


def quotas(per_cell: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {label: per_cell * share for label, share in shares.items()}
    out = {label: int(value) for label, value in raw.items()}
    missing = per_cell - sum(out.values())
    labels = sorted(raw, key=lambda label: raw[label] - out[label], reverse=True)
    for label in labels[:missing]:
        out[label] += 1
    return out


def reservoir_add(rows: list[dict[str, str]], row: dict[str, str], seen: int, capacity: int, rng: random.Random) -> None:
    if len(rows) < capacity:
        rows.append(row)
        return
    index = rng.randrange(seen)
    if index < capacity:
        rows[index] = row


def sample_cell(path: Path, country: str, topic: str, target: int, label_quotas: dict[str, int], args: argparse.Namespace, rng: random.Random) -> tuple[list[dict[str, str]], dict[str, Any], list[str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        capacity = max(target, max(label_quotas.values(), default=0))
        buckets = {label: [] for label in label_quotas}
        seen: Counter[str] = Counter()
        eligible_n = 0
        scanned_n = 0
        for index, row in enumerate(reader):
            if args.max_scan_rows_per_file > 0 and index >= args.max_scan_rows_per_file:
                break
            scanned_n += 1
            nli_label = str(row.get('nli_label', '')).strip().lower()
            if nli_label not in label_quotas or not eligible(row, args):
                continue
            eligible_n += 1
            seen[nli_label] += 1
            row = dict(row)
            row['source_pairs_file'] = path.as_posix()
            row['sample_country'] = country
            row['sample_topic'] = topic
            row['sample_stratum'] = f'{country}|{topic}|{nli_label}'
            pair_id = row.get('nli_pair_id') or str(index)
            row['_key'] = path.as_posix() + '::' + str(pair_id)
            reservoir_add(buckets[nli_label], row, seen[nli_label], capacity, rng)
    selected: list[dict[str, str]] = []
    keys: set[str] = set()
    for label, quota in label_quotas.items():
        candidates = list(buckets[label])
        rng.shuffle(candidates)
        for row in candidates[:quota]:
            if row['_key'] not in keys:
                selected.append(row)
                keys.add(row['_key'])
    if len(selected) < target:
        leftovers = [row for rows in buckets.values() for row in rows if row['_key'] not in keys]
        rng.shuffle(leftovers)
        for row in leftovers:
            if len(selected) >= target:
                break
            selected.append(row)
            keys.add(row['_key'])
    summary = {
        'country': country,
        'topic': topic,
        'path': path.as_posix(),
        'target': target,
        'eligible': eligible_n,
        'scanned': scanned_n,
        'selected': len(selected),
        'seen_by_nli_label': dict(seen),
        'selected_by_nli_label': dict(Counter(row.get('nli_label', '') for row in selected)),
    }
    return selected, summary, fieldnames


def write_sample(rows: list[dict[str, str]], input_fields: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    leading = ['sample_id', 'sample_country', 'sample_topic', 'sample_stratum', 'source_pairs_file']
    fields = leading + [field for field in input_fields if field not in leading]
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            clean = {key: value for key, value in row.items() if not key.startswith('_')}
            clean['sample_id'] = f'llm_pair_{index:06d}'
            writer.writerow(clean)


def write_metadata(output: Path, args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    metadata = {
        'created_at_unix': time.time(),
        'sample_csv': output.as_posix(),
        'target_size': args.target_size,
        'countries': args.countries,
        'topics': args.topics,
        'label_shares': args.label_shares,
        'max_retrieval_rank': args.max_retrieval_rank,
        'min_embedding_score': args.min_embedding_score,
        'min_speech_words': args.min_speech_words,
        'max_scan_rows_per_file': args.max_scan_rows_per_file,
        'cell_summaries': summaries,
    }
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')


def run_sample(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Sample already exists: {args.output}. Use --overwrite to replace it.')
    args.label_shares = parse_label_shares(args.label_shares)
    paths: list[tuple[str, str, Path]] = []
    for country in args.countries:
        for topic in args.topics:
            path = pairs_path(args.root, country, topic, args.nli_model_name)
            if path is None:
                print(f'Missing pairs file for {country} {topic}; skipping.', file=sys.stderr)
                continue
            paths.append((country.upper(), topic, path))
    if not paths:
        raise FileNotFoundError('No pairs files found for requested countries/topics.')
    if args.per_country_topic > 0:
        targets = [args.per_country_topic] * len(paths)
    else:
        base = args.target_size // len(paths)
        remainder = args.target_size % len(paths)
        targets = [base + (1 if index < remainder else 0) for index in range(len(paths))]
    rng = random.Random(args.random_state)
    all_rows: list[dict[str, str]] = []
    all_fields: list[str] = []
    summaries: list[dict[str, Any]] = []
    for (country, topic, path), target in zip(paths, targets):
        selected, summary, fields = sample_cell(path, country, topic, target, quotas(target, args.label_shares), args, rng)
        all_rows.extend(selected)
        summaries.append(summary)
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)
        eligible_count = summary.get('eligible', 0)
        print(f'{country} {topic}: selected {len(selected):,}/{target:,} from {eligible_count:,} eligible rows', flush=True)
    rng.shuffle(all_rows)
    if args.target_size > 0 and args.per_country_topic <= 0 and len(all_rows) > args.target_size:
        all_rows = all_rows[:args.target_size]
    write_sample(all_rows, all_fields, args.output)
    write_metadata(args.output, args, summaries)
    metadata_path = args.output.with_suffix('.metadata.json')
    print(f'Saved frozen sample with {len(all_rows):,} rows to: {args.output}', flush=True)
    print(f'Saved metadata to: {metadata_path}', flush=True)
    return 0


def build_prompt(row: dict[str, str]) -> str:
    meta = {
        'sample_id': row.get('sample_id', ''),
        'country': row.get('country') or row.get('sample_country', ''),
        'topic': row.get('sample_topic') or row.get('speech_topic_label', ''),
        'party': row.get('party') or row.get('speech_party', ''),
        'date': row.get('date', ''),
        'existing_nli_label': row.get('nli_label', ''),
        'retrieval_rank': row.get('retrieval_rank', ''),
        'retrieval_score': row.get('retrieval_score', ''),
        'embedding_score': row.get('embedding_score', ''),
    }
    meta_lines = '\n'.join(f'{key}: {value}' for key, value in meta.items() if str(value).strip())
    return (
        'Classify this speech-manifesto pair. Return only JSON.\n\n'
        + 'Metadata:\n' + meta_lines + '\n\n'
        + 'Parliamentary speech excerpt:\n' + best_speech(row) + '\n\n'
        + 'Manifesto claim:\n' + best_manifesto(row)
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize(parsed: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed.get('classification'), dict):
        parsed = {**parsed, **parsed['classification']}
    label_value = ''
    for key in ['label', 'nli_label', 'classification', 'relationship', 'relation', 'verdict', 'answer']:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            label_value = value
            break
    label = str(label_value).strip().lower().replace('-', '_').replace(' ', '_')
    label_aliases = {
        'contradiction': 'inconsistent',
        'contradictory': 'inconsistent',
        'conflict': 'inconsistent',
        'conflicting': 'inconsistent',
        'inconsistency': 'inconsistent',
        'compatible': 'consistent',
        'related_consistent': 'consistent',
        'same_issue_consistent': 'consistent',
        'not_related': 'unrelated',
        'different_issue': 'unrelated',
        'irrelevant': 'unrelated',
        'unclear': 'ambiguous',
        'unknown': 'ambiguous',
    }
    label = label_aliases.get(label, label)
    if label not in VALID_LABELS:
        preview = json.dumps(parsed, ensure_ascii=False)[:500]
        raise ValueError(f'Invalid label: {label!r}; parsed={preview}')
    inconsistency_type = str(parsed.get('inconsistency_type', 'none')).strip().lower().replace('-', '_').replace(' ', '_')
    if inconsistency_type not in VALID_TYPES:
        inconsistency_type = 'none' if label != 'inconsistent' else 'ambiguous'
    confidence = as_float(parsed.get('confidence'))
    confidence = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
    comparable = parsed.get('comparable', label in {'consistent', 'inconsistent', 'ambiguous'})
    if isinstance(comparable, str):
        comparable = as_bool(comparable)
    if comparable is None:
        comparable = label in {'consistent', 'inconsistent', 'ambiguous'}
    return {
        'llm_label': label,
        'llm_comparable': comparable,
        'llm_confidence': confidence,
        'llm_inconsistency_type': inconsistency_type,
        'llm_rationale': str(parsed.get('rationale', parsed.get('reason', ''))).strip(),
    }


class ProviderAbortError(RuntimeError):
    pass


def is_provider_abort_error(provider: str, error: str) -> bool:
    text = error.lower()
    if any(marker in text for marker in ['404 client error', 'not_found']):
        return True
    if provider == 'gemini':
        return any(marker in text for marker in ['429 client error', 'resource_exhausted', 'quota', 'too many requests'])
    if provider == 'anthropic':
        return any(marker in text for marker in ['credit balance is too low', 'insufficient_quota', 'quota exceeded'])
    if provider == 'openai':
        return any(marker in text for marker in ['insufficient_quota', 'billing hard limit', 'quota exceeded'])
    return False

def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], max_retries: int) -> dict[str, Any]:
    last_error = 'unknown request error'
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = response.headers.get('Retry-After')
                wait = float(retry_after) if retry_after else min(2 ** attempt, 45)
                time.sleep(wait)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                body = response.text.strip()
                raise RuntimeError(f'{exc}: {body}') from exc
            return response.json()
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 45))
    raise RuntimeError(last_error)


def openai_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get('output', []):
        for content in item.get('content', []):
            if content.get('type') == 'output_text':
                texts.append(str(content.get('text', '')))
    return '\n'.join(texts).strip()


def call_openai(row: dict[str, str], api_key: str, model: str, max_retries: int) -> tuple[dict[str, Any], str, str]:
    payload = {'model': model, 'instructions': SYSTEM_PROMPT, 'input': build_prompt(row), 'text': {'format': OPENAI_SCHEMA}}
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    project_id = os.getenv('OPENAI_PROJECT') or os.getenv('OPEN_API_PROJECT_KEY')
    if project_id:
        headers['OpenAI-Project'] = project_id
    response = post_json(OPENAI_URL, headers, payload, max_retries)
    return normalize(extract_json(openai_text(response))), response.get('model', model), response.get('id', '')


def anthropic_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get('content', []):
        if item.get('type') == 'text':
            texts.append(str(item.get('text', '')))
    return '\n'.join(texts).strip()


def call_anthropic(row: dict[str, str], api_key: str, model: str, max_retries: int) -> tuple[dict[str, Any], str, str]:
    payload = {
        'model': model,
        'max_tokens': 350,
        'temperature': 0,
        'system': SYSTEM_PROMPT,
        'messages': [{'role': 'user', 'content': build_prompt(row)}],
    }
    headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
    response = post_json(ANTHROPIC_URL, headers, payload, max_retries)
    return normalize(extract_json(anthropic_text(response))), response.get('model', model), response.get('id', '')


def gemini_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get('candidates', []):
        for part in candidate.get('content', {}).get('parts', []):
            if 'text' in part:
                texts.append(str(part.get('text', '')))
    return '\n'.join(texts).strip()


def call_gemini(row: dict[str, str], api_key: str, model: str, max_retries: int) -> tuple[dict[str, Any], str, str]:
    payload = {
        'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
        'contents': [{'role': 'user', 'parts': [{'text': build_prompt(row)}]}],
        'generationConfig': {
            'temperature': 0,
            'responseMimeType': 'application/json',
            'responseSchema': {
                'type': 'OBJECT',
                'properties': {
                    'label': {'type': 'STRING', 'enum': sorted(VALID_LABELS)},
                    'comparable': {'type': 'BOOLEAN'},
                    'confidence': {'type': 'NUMBER'},
                    'inconsistency_type': {'type': 'STRING', 'enum': sorted(VALID_TYPES)},
                    'rationale': {'type': 'STRING'},
                },
                'required': ['label', 'comparable', 'confidence', 'inconsistency_type', 'rationale'],
            },
        },
    }
    headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
    response = post_json(GEMINI_URL.format(model=model), headers, payload, max_retries)
    return normalize(extract_json(gemini_text(response))), model, ''


def api_key_for(provider: str) -> str:
    names = {
        'openai': ['OPENAI_API_KEY', 'OPEN_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY', 'CLAUDE_KEY'],
        'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_KEY'],
    }[provider]
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    names_text = ', '.join(names)
    raise RuntimeError(f'Missing API key for {provider}; tried {names_text}')


def model_for(args: argparse.Namespace, provider: str) -> str:
    return {'openai': args.openai_model, 'anthropic': args.anthropic_model, 'gemini': args.gemini_model}[provider]


def call_provider(provider: str, row: dict[str, str], api_key: str, model: str, max_retries: int) -> tuple[dict[str, Any], str, str]:
    if provider == 'openai':
        return call_openai(row, api_key, model, max_retries)
    if provider == 'anthropic':
        return call_anthropic(row, api_key, model, max_retries)
    if provider == 'gemini':
        return call_gemini(row, api_key, model, max_retries)
    raise ValueError(f'Unknown provider: {provider}')


def provider_output(args: argparse.Namespace, provider: str, model: str) -> Path:
    return args.output_dir / f'{args.input.stem}_{provider}_{token(model)}_labels.csv'


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows, _fields = read_rows(path)
    return {row.get('sample_id', '') for row in rows if row.get('sample_id') and row.get('llm_label')}


def label_one(provider: str, row: dict[str, str], api_key: str, model: str, args: argparse.Namespace) -> dict[str, Any]:
    actual_model = model
    response_id = ''
    error = ''
    try:
        result, actual_model, response_id = call_provider(provider, row, api_key, model, args.max_retries)
    except Exception as exc:
        error = str(exc)
        if is_provider_abort_error(provider, error):
            raise ProviderAbortError(error) from exc
        result = {'llm_label': '', 'llm_comparable': '', 'llm_confidence': '', 'llm_inconsistency_type': '', 'llm_rationale': ''}
    return {**row, **result, 'provider': provider, 'provider_model': actual_model, 'provider_response_id': response_id, 'llm_error': error}


def run_label_provider(args: argparse.Namespace, provider: str) -> Path:
    rows, fields = read_rows(args.input)
    model = model_for(args, provider)
    output = provider_output(args, provider, model)
    if args.overwrite and output.exists():
        output.unlink()
    done = set() if args.overwrite else completed_ids(output)
    api_key = api_key_for(provider)
    out_fields = fields + [
        'provider',
        'provider_model',
        'provider_response_id',
        'llm_label',
        'llm_comparable',
        'llm_confidence',
        'llm_inconsistency_type',
        'llm_rationale',
        'llm_error',
    ]
    todo: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        sample_id = row.get('sample_id', '')
        if sample_id in done:
            skipped += 1
            continue
        if args.limit and len(todo) >= args.limit:
            break
        todo.append(row)
    print(f'{provider}: queued {len(todo):,} row(s); skipped {skipped:,}; workers={args.workers}', flush=True)

    processed = 0
    buffer: list[dict[str, Any]] = []

    def flush_if_needed(force: bool = False) -> None:
        nonlocal buffer
        if buffer and (force or len(buffer) >= args.flush_every):
            append_rows(output, out_fields, buffer)
            buffer = []

    if args.workers <= 1:
        for row in todo:
            try:
                buffer.append(label_one(provider, row, api_key, model, args))
            except ProviderAbortError as exc:
                print(f'{provider}: aborting after fatal provider error (quota/rate-limit or 404): {exc}', file=sys.stderr, flush=True)
                break
            processed += 1
            flush_if_needed()
            if args.sleep:
                time.sleep(args.sleep)
            if processed == 1 or processed % 25 == 0:
                print(f'{provider}: processed {processed:,}/{len(todo):,}; output={output}', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(label_one, provider, row, api_key, model, args) for row in todo]
            for future in as_completed(futures):
                try:
                    buffer.append(future.result())
                except ProviderAbortError as exc:
                    for pending in futures:
                        pending.cancel()
                    print(f'{provider}: aborting after fatal provider error (quota/rate-limit or 404): {exc}', file=sys.stderr, flush=True)
                    break
                processed += 1
                flush_if_needed()
                if processed == 1 or processed % 25 == 0:
                    print(f'{provider}: processed {processed:,}/{len(todo):,}; output={output}', flush=True)
    flush_if_needed(force=True)
    print(f'{provider}: saved labels to {output}', flush=True)
    return output


def run_label(args: argparse.Namespace) -> int:
    load_dotenv(BASE_DIR / '.env')
    providers = args.providers
    if 'all' in providers:
        providers = ['openai', 'anthropic', 'gemini']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for provider in providers:
        if provider not in {'openai', 'anthropic', 'gemini'}:
            raise ValueError(f'Unsupported provider: {provider}')
        run_label_provider(args, provider)
    return 0


def infer_provider(path: Path, rows: list[dict[str, str]]) -> str:
    for row in rows:
        provider = str(row.get('provider', '')).strip()
        if provider:
            return provider
    name = path.name.lower()
    for provider in ['openai', 'anthropic', 'gemini']:
        if provider in name:
            return provider
    return token(path.stem)


def run_consensus(args: argparse.Namespace) -> int:
    sample_rows, sample_fields = read_rows(args.input)
    label_files = args.label_files or sorted(args.output_dir.glob(f'{args.input.stem}_*_labels.csv'))
    if not label_files:
        raise FileNotFoundError(f'No provider label files found in {args.output_dir}')
    provider_maps: dict[str, dict[str, dict[str, str]]] = {}
    for path in label_files:
        rows, _fields = read_rows(path)
        provider = infer_provider(path, rows)
        provider_maps[provider] = {row.get('sample_id', ''): row for row in rows if row.get('sample_id')}
        print(f'Loaded {len(provider_maps[provider]):,} labels from {provider}: {path}', flush=True)
    providers = sorted(provider_maps)
    extra: list[str] = []
    for provider in providers:
        extra.extend([f'{provider}_label', f'{provider}_confidence', f'{provider}_comparable', f'{provider}_type', f'{provider}_error'])
    extra.extend(['provider_count', 'agreement_count', 'consensus_label', 'consensus_unanimous', 'training_keep'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields + [field for field in extra if field not in sample_fields], extrasaction='ignore')
        writer.writeheader()
        kept = 0
        for row in sample_rows:
            sample_id = row.get('sample_id', '')
            out = dict(row)
            labels: list[str] = []
            for provider in providers:
                provider_row = provider_maps[provider].get(sample_id, {})
                label = str(provider_row.get('llm_label', '')).strip().lower()
                error = str(provider_row.get('llm_error', '')).strip()
                out[f'{provider}_label'] = label
                out[f'{provider}_confidence'] = provider_row.get('llm_confidence', '')
                out[f'{provider}_comparable'] = provider_row.get('llm_comparable', '')
                out[f'{provider}_type'] = provider_row.get('llm_inconsistency_type', '')
                out[f'{provider}_error'] = error
                if label in VALID_LABELS and not error:
                    labels.append(label)
            counts = Counter(labels)
            consensus_label = ''
            agreement = 0
            if counts:
                consensus_label, agreement = counts.most_common(1)[0]
            has_consensus = agreement >= args.min_agreement
            out['provider_count'] = len(labels)
            out['agreement_count'] = agreement
            out['consensus_label'] = consensus_label if has_consensus else ''
            out['consensus_unanimous'] = bool(has_consensus and agreement == len(labels) and labels)
            out['training_keep'] = bool(has_consensus and consensus_label != 'ambiguous')
            if out['training_keep']:
                kept += 1
            writer.writerow(out)
    print(f'Saved consensus file to: {args.output}', flush=True)
    print(f'Rows kept for training by default: {kept:,}/{len(sample_rows):,}', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Freeze a shared NLI-pair sample, label it with multiple LLM providers, and build consensus labels.')
    sub = parser.add_subparsers(dest='command', required=True)

    sample = sub.add_parser('sample', help='Create a frozen stratified sample CSV.')
    sample.add_argument('--root', default=DEFAULT_ROOT, type=Path)
    sample.add_argument('--output', default=DEFAULT_SAMPLE, type=Path)
    sample.add_argument('--countries', nargs='*', default=DEFAULT_COUNTRIES)
    sample.add_argument('--topics', nargs='*', default=DEFAULT_TOPICS)
    sample.add_argument('--target-size', default=8000, type=int)
    sample.add_argument('--per-country-topic', default=0, type=int)
    sample.add_argument('--label-shares', default='contradiction=0.4,entailment=0.2,neutral=0.4')
    sample.add_argument('--nli-model-name', default=DEFAULT_NLI_MODEL)
    sample.add_argument('--max-retrieval-rank', default=3, type=int, help='Use 0 to disable rank filtering.')
    sample.add_argument('--min-embedding-score', default=None, type=float)
    sample.add_argument('--min-speech-words', default=12, type=int)
    sample.add_argument('--max-scan-rows-per-file', default=250000, type=int, help='Use 0 to scan full files.')
    sample.add_argument('--random-state', default=1711, type=int)
    sample.add_argument('--overwrite', action='store_true')
    sample.set_defaults(func=run_sample)

    label = sub.add_parser('label', help='Label a frozen sample with one or more providers.')
    label.add_argument('--input', default=DEFAULT_SAMPLE, type=Path)
    label.add_argument('--output-dir', default=DEFAULT_SAMPLE.parent / 'provider_labels', type=Path)
    label.add_argument('--providers', nargs='*', default=['all'], help='openai anthropic gemini or all')
    label.add_argument('--openai-model', default='gpt-5.4-mini')
    label.add_argument('--anthropic-model', default='claude-haiku-4-5')
    label.add_argument('--gemini-model', default='gemini-2.5-flash')
    label.add_argument('--limit', default=0, type=int, help='Maximum rows per provider; 0 means all.')
    label.add_argument('--sleep', default=0.1, type=float)
    label.add_argument('--workers', default=1, type=int, help='Concurrent API requests per provider. Use 1 for Gemini quota-sensitive resumes.')
    label.add_argument('--flush-every', default=10, type=int)
    label.add_argument('--max-retries', default=4, type=int)
    label.add_argument('--overwrite', action='store_true')
    label.set_defaults(func=run_label)

    consensus = sub.add_parser('consensus', help='Merge provider label files into a majority-vote CSV.')
    consensus.add_argument('--input', default=DEFAULT_SAMPLE, type=Path)
    consensus.add_argument('--output-dir', default=DEFAULT_SAMPLE.parent / 'provider_labels', type=Path)
    consensus.add_argument('--label-files', nargs='*', default=[], type=Path)
    consensus.add_argument('--output', default=DEFAULT_SAMPLE.parent / 'llm_consensus_labels.csv', type=Path)
    consensus.add_argument('--min-agreement', default=2, type=int)
    consensus.set_defaults(func=run_consensus)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        raise SystemExit(130)
