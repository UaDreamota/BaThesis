from __future__ import annotations

"""Fast, blinded browser reviewer for manifesto--speech validation pairs.

The application uses only Python's standard library. It reads one blinded
coding CSV, serves a localhost-only browser interface, and atomically writes
human judgments back to that same CSV. It never reads the private classifier
key or any model predictions.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CODING_FILE = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "macro_human_validation"
    / "BLINDED_PRIMARY_CODING.csv"
)
LABELS = ("consistent", "inconsistent", "unrelated", "ambiguous")
COMPARABILITY = ("yes", "no")
CONFIDENCE = ("low", "medium", "high")
TRANSLATIONS_FILENAME = "BLINDED_ENGLISH_TRANSLATIONS.csv"
SUGGESTIONS_FILENAME = "BLINDED_AI_REVIEW_SUGGESTIONS.csv"


@dataclass(frozen=True)
class CodingSchema:
    label: str
    comparable: str
    confidence: str
    notes: str


PRIMARY_SCHEMA = CodingSchema(
    "human_label", "human_comparable", "human_confidence", "human_notes"
)
SECOND_SCHEMA = CodingSchema(
    "second_coder_label",
    "second_coder_comparable",
    "second_coder_confidence",
    "second_coder_notes",
)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--coding-file", type=Path, default=DEFAULT_CODING_FILE)
    out.add_argument(
        "--translations-file",
        type=Path,
        help=(
            "English-translation sidecar. By default, use "
            "BLINDED_ENGLISH_TRANSLATIONS.csv beside the coding file when present."
        ),
    )
    out.add_argument(
        "--suggestions-file",
        type=Path,
        help=(
            "Reasoned-suggestion sidecar for primary coding. By default, use "
            "BLINDED_AI_REVIEW_SUGGESTIONS.csv beside the coding file when present."
        ),
    )
    out.add_argument("--host", default="127.0.0.1")
    out.add_argument("--port", type=int, default=8765)
    out.add_argument("--no-browser", action="store_true")
    out.add_argument(
        "--check",
        action="store_true",
        help="Validate the coding CSV and print progress without starting the UI.",
    )
    return out


def normalize(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_schema(fieldnames: list[str]) -> CodingSchema:
    available = set(fieldnames)
    for schema in (PRIMARY_SCHEMA, SECOND_SCHEMA):
        required = {schema.label, schema.comparable, schema.confidence, schema.notes}
        if required <= available:
            return schema
    raise ValueError(
        "Coding file must contain either the primary human_* columns or the "
        "second_coder_* columns."
    )


def validate_judgment(
    label: str,
    comparable: str,
    confidence: str,
    *,
    require_complete: bool,
) -> None:
    if label and label not in LABELS:
        raise ValueError(f"Invalid label: {label!r}")
    if comparable and comparable not in COMPARABILITY:
        raise ValueError(f"Invalid comparability value: {comparable!r}")
    if confidence and confidence not in CONFIDENCE:
        raise ValueError(f"Invalid confidence value: {confidence!r}")
    if label in {"consistent", "inconsistent"} and comparable not in {"", "yes"}:
        raise ValueError(f"{label} requires comparable=yes")
    if label == "unrelated" and comparable not in {"", "no"}:
        raise ValueError("unrelated requires comparable=no")
    if require_complete and not (label and comparable and confidence):
        raise ValueError("A complete judgment requires label, comparability, and confidence.")


def judgment_complete(row: dict[str, str], schema: CodingSchema) -> bool:
    label = normalize(row.get(schema.label))
    comparable = normalize(row.get(schema.comparable))
    confidence = normalize(row.get(schema.confidence))
    try:
        validate_judgment(label, comparable, confidence, require_complete=True)
    except ValueError:
        return False
    return True


class CodingStore:
    def __init__(
        self,
        path: Path,
        translations_path: Path | None = None,
        suggestions_path: Path | None = None,
    ):
        self.path = path.resolve()
        self.lock = threading.RLock()
        self.backup_path = self.path.with_name(
            f"{self.path.stem}.pre-review-backup{self.path.suffix}"
        )
        self.fieldnames: list[str] = []
        self.rows: list[dict[str, str]] = []
        self.schema = PRIMARY_SCHEMA
        self.translations_path = (
            translations_path.resolve()
            if translations_path is not None
            else self.path.with_name(TRANSLATIONS_FILENAME)
        )
        self.translations: dict[str, dict[str, str]] = {}
        self.suggestions_path = (
            suggestions_path.resolve()
            if suggestions_path is not None
            else self.path.with_name(SUGGESTIONS_FILENAME)
        )
        self.suggestions: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"No CSV header in {self.path}")
            self.fieldnames = list(reader.fieldnames)
            self.schema = detect_schema(self.fieldnames)
            self.rows = [dict(row) for row in reader]
        required = {"validation_id", "country_code", "manifesto_text", "speech_text"}
        missing = sorted(required - set(self.fieldnames))
        if missing:
            raise ValueError(f"Coding file is missing columns: {missing}")
        ids = [row["validation_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("validation_id values must be unique")
        for row in self.rows:
            validate_judgment(
                normalize(row.get(self.schema.label)),
                normalize(row.get(self.schema.comparable)),
                normalize(row.get(self.schema.confidence)),
                require_complete=False,
            )
        self._load_translations()
        self._load_suggestions()

    def _load_translations(self) -> None:
        self.translations = {}
        if not self.translations_path.exists():
            return
        with self.translations_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            required = {
                "validation_id",
                "country_code",
                "manifesto_text_sha256",
                "speech_text_sha256",
                "manifesto_text_en",
                "speech_text_en",
                "translation_provider",
                "translation_model",
            }
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Translation sidecar is missing columns: {missing}")
            translated_rows = [dict(row) for row in reader]
        ids = [row["validation_id"] for row in translated_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Translation sidecar contains duplicate validation_id values")
        available = {row["validation_id"]: row for row in translated_rows}
        for source in self.rows:
            validation_id = source["validation_id"]
            translated = available.get(validation_id)
            if translated is None:
                continue
            if translated["country_code"] != source["country_code"]:
                raise ValueError(
                    f"Translation country mismatch for {validation_id}: "
                    f"{translated['country_code']!r} != {source['country_code']!r}"
                )
            if translated["manifesto_text_sha256"] != text_sha256(
                source["manifesto_text"]
            ) or translated["speech_text_sha256"] != text_sha256(source["speech_text"]):
                raise ValueError(f"Translation source-text hash mismatch for {validation_id}")
            if not translated["manifesto_text_en"].strip() or not translated[
                "speech_text_en"
            ].strip():
                raise ValueError(f"Empty English translation for {validation_id}")
            self.translations[validation_id] = translated

    def _load_suggestions(self) -> None:
        self.suggestions = {}
        if self.schema != PRIMARY_SCHEMA or not self.suggestions_path.exists():
            return
        with self.suggestions_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            required = {
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
            }
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Suggestion sidecar is missing columns: {missing}")
            suggested_rows = [dict(row) for row in reader]
        ids = [row["validation_id"] for row in suggested_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Suggestion sidecar contains duplicate validation_id values")
        available = {row["validation_id"]: row for row in suggested_rows}
        for source in self.rows:
            validation_id = source["validation_id"]
            suggested = available.get(validation_id)
            if suggested is None:
                continue
            translated = self.translations.get(validation_id)
            if translated is None:
                raise ValueError(f"Suggestion requires an English translation for {validation_id}")
            if suggested["country_code"] != source["country_code"]:
                raise ValueError(f"Suggestion country mismatch for {validation_id}")
            if suggested["manifesto_text_sha256"] != text_sha256(
                source["manifesto_text"]
            ) or suggested["speech_text_sha256"] != text_sha256(source["speech_text"]):
                raise ValueError(f"Suggestion source-text hash mismatch for {validation_id}")
            if suggested["manifesto_translation_sha256"] != text_sha256(
                translated["manifesto_text_en"]
            ) or suggested["speech_translation_sha256"] != text_sha256(
                translated["speech_text_en"]
            ):
                raise ValueError(f"Suggestion translation hash mismatch for {validation_id}")
            label = normalize(suggested["suggested_label"])
            comparable = normalize(suggested["suggested_comparable"])
            confidence = normalize(suggested["suggested_confidence"])
            validate_judgment(label, comparable, confidence, require_complete=True)
            if not suggested["rationale"].strip() or not suggested["review_focus"].strip():
                raise ValueError(f"Suggestion explanation is empty for {validation_id}")
            suggested["suggested_label"] = label
            suggested["suggested_comparable"] = comparable
            suggested["suggested_confidence"] = confidence
            self.suggestions[validation_id] = suggested

    def progress(self) -> dict[str, int | None]:
        complete = [judgment_complete(row, self.schema) for row in self.rows]
        first_incomplete = next((i for i, done in enumerate(complete) if not done), None)
        return {
            "complete": sum(complete),
            "total": len(self.rows),
            "remaining": len(self.rows) - sum(complete),
            "first_incomplete": first_incomplete,
        }

    def public_row(self, index: int) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("Coding file has no rows")
        index = max(0, min(index, len(self.rows) - 1))
        row = self.rows[index]
        schema = self.schema
        translated = self.translations.get(row["validation_id"], {})
        suggested = self.suggestions.get(row["validation_id"], {})
        return {
            "index": index,
            "validation_id": row["validation_id"],
            "country_code": row["country_code"],
            "manifesto_text": row["manifesto_text"],
            "speech_text": row["speech_text"],
            "translations_available": bool(translated),
            "manifesto_text_en": translated.get("manifesto_text_en", ""),
            "speech_text_en": translated.get("speech_text_en", ""),
            "translation_provider": translated.get("translation_provider", ""),
            "translation_model": translated.get("translation_model", ""),
            "suggestion_available": bool(suggested),
            "suggested_label": suggested.get("suggested_label", ""),
            "suggested_comparable": suggested.get("suggested_comparable", ""),
            "suggested_confidence": suggested.get("suggested_confidence", ""),
            "suggestion_rationale": suggested.get("rationale", ""),
            "suggestion_translation_warning": normalize(
                suggested.get("translation_warning", "")
            )
            == "true",
            "suggestion_review_focus": suggested.get("review_focus", ""),
            "assessment_provider": suggested.get("assessment_provider", ""),
            "assessment_model": suggested.get("assessment_model", ""),
            "label": normalize(row.get(schema.label)),
            "comparable": normalize(row.get(schema.comparable)),
            "confidence": normalize(row.get(schema.confidence)),
            "notes": row.get(schema.notes, ""),
            "complete": judgment_complete(row, schema),
            "progress": self.progress(),
        }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation_id = str(payload.get("validation_id", "")).strip()
        label = normalize(payload.get("label"))
        comparable = normalize(payload.get("comparable"))
        confidence = normalize(payload.get("confidence"))
        notes = str(payload.get("notes", "")).strip()
        validate_judgment(label, comparable, confidence, require_complete=False)
        if len(notes) > 10_000:
            raise ValueError("Notes exceed the 10,000-character limit")
        with self.lock:
            matches = [i for i, row in enumerate(self.rows) if row["validation_id"] == validation_id]
            if len(matches) != 1:
                raise ValueError(f"Unknown validation_id: {validation_id!r}")
            index = matches[0]
            row = self.rows[index]
            row[self.schema.label] = label
            row[self.schema.comparable] = comparable
            row[self.schema.confidence] = confidence
            row[self.schema.notes] = notes
            self._write_atomic()
            return self.public_row(index)

    def _write_atomic(self) -> None:
        if not self.backup_path.exists():
            shutil.copy2(self.path, self.backup_path)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = Path(handle.name)
        try:
            with handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="raise")
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded pair reviewer</title>
<style>
:root { color-scheme: light dark; --bg:#f4f1e9; --card:#fffdf8; --ink:#24221e;
  --muted:#6d685d; --line:#d8d1c3; --accent:#315f8c; --good:#2f7352;
  --warn:#9b5b28; --selected:#dce9f5; }
@media (prefers-color-scheme: dark) { :root { --bg:#171815; --card:#22231f; --ink:#eeeae0;
  --muted:#aaa397; --line:#45463f; --accent:#75aadd; --good:#70c69a; --warn:#e3a66e;
  --selected:#263e53; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,sans-serif; }
header { position:sticky; top:0; z-index:4; background:var(--card); border-bottom:1px solid var(--line);
  padding:10px 18px; display:flex; align-items:center; gap:16px; }
header strong { white-space:nowrap; } .progress { height:10px; background:var(--line); border-radius:8px;
  flex:1; overflow:hidden; } .progress > div { height:100%; background:var(--good); width:0; }
.muted { color:var(--muted); } main { max-width:1500px; margin:auto; padding:18px; }
.meta { display:flex; justify-content:space-between; gap:12px; margin-bottom:12px; }
.texts { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
.card h2 { margin:0 0 12px; font-size:15px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.text { white-space:pre-wrap; font:18px/1.6 Georgia,serif; overflow-wrap:anywhere; min-height:180px; }
.original { margin-top:14px; border-top:1px solid var(--line); padding-top:10px; }
.original summary { cursor:pointer; color:var(--muted); font-weight:650; }
.original .text { min-height:0; margin-top:10px; font-size:15px; color:var(--muted); }
.translation-notice { margin:-4px 0 12px; color:var(--muted); font-size:14px; }
.suggestion { margin-top:16px; border-left:5px solid var(--accent); }
.suggestion h2 { color:var(--accent); }
.suggestion-grid { display:flex; flex-wrap:wrap; gap:8px 18px; margin:8px 0; }
.suggestion-grid span { white-space:nowrap; }
.suggestion p { margin:8px 0; }
.suggestion .warning { color:var(--warn); font-weight:650; }
.controls { margin-top:16px; } .group { margin:12px 0; } .group-title { font-weight:650; margin-bottom:7px; }
.buttons { display:flex; flex-wrap:wrap; gap:8px; }
button { border:1px solid var(--line); background:var(--card); color:var(--ink); padding:10px 14px;
  border-radius:8px; cursor:pointer; font-weight:600; }
button:hover { border-color:var(--accent); } button.selected { background:var(--selected); border-color:var(--accent); }
button.primary { background:var(--accent); color:white; border-color:var(--accent); }
kbd { border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:1px 5px; font-size:12px; }
textarea { width:100%; min-height:90px; resize:vertical; border:1px solid var(--line); border-radius:8px;
  padding:10px; background:var(--card); color:var(--ink); font:inherit; }
.footer { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:12px; }
.status { min-height:24px; color:var(--good); } .status.error { color:#bd3d3d; }
.help { margin-top:16px; font-size:14px; color:var(--muted); }
@media (max-width:850px) { .texts { grid-template-columns:1fr; } header { flex-wrap:wrap; } }
</style>
</head>
<body>
<header><strong>Blinded validation</strong><div class="progress"><div id="bar"></div></div><span id="progressText"></span></header>
<main>
  <div class="meta"><div><strong id="id"></strong> · <span id="country"></span></div><div id="position"></div></div>
  <div id="translationNotice" class="translation-notice"></div>
  <section class="texts">
    <article class="card"><h2 id="manifestoHeading">Earlier manifesto statement</h2><div id="manifestoMain" class="text"></div>
      <details id="manifestoOriginalWrap" class="original"><summary>Show original manifesto text</summary><div id="manifestoOriginal" class="text"></div></details></article>
    <article class="card"><h2 id="speechHeading">Later parliamentary speech</h2><div id="speechMain" class="text"></div>
      <details id="speechOriginalWrap" class="original"><summary>Show original speech text</summary><div id="speechOriginal" class="text"></div></details></article>
  </section>
  <section id="suggestionPanel" class="card suggestion" hidden>
    <h2>Reasoned suggestion — validate before accepting</h2>
    <div class="suggestion-grid">
      <span>Classification: <strong id="suggestedLabel"></strong></span>
      <span>Comparable: <strong id="suggestedComparable"></strong></span>
      <span>Confidence: <strong id="suggestedConfidence"></strong></span>
    </div>
    <p id="suggestionRationale"></p>
    <p><strong>What to check:</strong> <span id="suggestionFocus"></span></p>
    <p id="suggestionTranslationWarning" class="warning" hidden>Translation may affect this judgment; compare the original if possible.</p>
    <div class="buttons"><button id="acceptSuggestion" class="primary">Accept suggestion <kbd>A</kbd></button></div>
    <p class="muted">This is an AI-assisted recommendation, not an independent human label. Change any field you disagree with.</p>
  </section>
  <section class="card controls">
    <div class="group"><div class="group-title">1. Classification</div><div class="buttons" id="labels">
      <button data-value="consistent"><kbd>1</kbd> Consistent</button>
      <button data-value="inconsistent"><kbd>2</kbd> Inconsistent</button>
      <button data-value="unrelated"><kbd>3</kbd> Unrelated</button>
      <button data-value="ambiguous"><kbd>4</kbd> Ambiguous</button>
    </div></div>
    <div class="group"><div class="group-title">2. Substantively comparable?</div><div class="buttons" id="comparable">
      <button data-value="yes"><kbd>Y</kbd> Yes</button><button data-value="no"><kbd>N</kbd> No</button>
    </div></div>
    <div class="group"><div class="group-title">3. Confidence</div><div class="buttons" id="confidence">
      <button data-value="low"><kbd>L</kbd> Low</button><button data-value="medium"><kbd>M</kbd> Medium</button>
      <button data-value="high"><kbd>H</kbd> High</button>
    </div></div>
    <div class="group"><label class="group-title" for="notes">Notes (optional; explain difficult cases)</label>
      <textarea id="notes" placeholder="Reason for ambiguity, translation procedure, or difficult distinction…"></textarea></div>
    <div class="footer"><div class="buttons"><button id="prev">← Previous</button><button id="nextIncomplete">Next uncoded</button></div>
      <div id="status" class="status"></div><button id="saveNext" class="primary">Save & next <kbd>Enter</kbd></button></div>
  </section>
  <div class="help">Judge only the supplied text. Different actors, periods, jurisdictions, specificity, or emphasis are not automatically conflicts.
    Shortcuts are disabled while typing notes; use <kbd>Ctrl</kbd>+<kbd>Enter</kbd> there to save and advance.</div>
</main>
<script>
let row=null, dirty=false, saveTimer=null;
const $=id=>document.getElementById(id);
function selected(group,value){ document.querySelectorAll(`#${group} button`).forEach(b=>b.classList.toggle('selected',b.dataset.value===value)); }
function current(){ return {validation_id:row.validation_id,label:row.label,comparable:row.comparable,confidence:row.confidence,notes:$('notes').value}; }
function isComplete(){ return row.label && row.comparable && row.confidence &&
  (!['consistent','inconsistent'].includes(row.label)||row.comparable==='yes') &&
  (row.label!=='unrelated'||row.comparable==='no'); }
function render(){
  $('id').textContent=row.validation_id; $('country').textContent=row.country_code;
  const translated=row.translations_available;
  $('manifestoHeading').textContent=translated?'Earlier manifesto statement — English':'Earlier manifesto statement';
  $('speechHeading').textContent=translated?'Later parliamentary speech — English':'Later parliamentary speech';
  $('manifestoMain').textContent=translated?row.manifesto_text_en:row.manifesto_text;
  $('speechMain').textContent=translated?row.speech_text_en:row.speech_text;
  $('manifestoOriginal').textContent=row.manifesto_text; $('speechOriginal').textContent=row.speech_text;
  $('manifestoOriginalWrap').hidden=!translated||row.manifesto_text_en.trim()===row.manifesto_text.trim();
  $('speechOriginalWrap').hidden=!translated||row.speech_text_en.trim()===row.speech_text.trim();
  $('translationNotice').textContent=translated
    ?`English machine translation (${row.translation_provider}/${row.translation_model}); expand the original text to audit it.`
    :'English translation unavailable for this row; showing the original text.';
  $('suggestionPanel').hidden=!row.suggestion_available;
  if(row.suggestion_available){
    $('suggestedLabel').textContent=row.suggested_label;
    $('suggestedComparable').textContent=row.suggested_comparable;
    $('suggestedConfidence').textContent=row.suggested_confidence;
    $('suggestionRationale').textContent=row.suggestion_rationale;
    $('suggestionFocus').textContent=row.suggestion_review_focus;
    $('suggestionTranslationWarning').hidden=!row.suggestion_translation_warning;
  }
  $('position').textContent=`${row.index+1} / ${row.progress.total}`; $('notes').value=row.notes||'';
  selected('labels',row.label); selected('comparable',row.comparable); selected('confidence',row.confidence);
  const p=row.progress; $('progressText').textContent=`${p.complete}/${p.total} complete · ${p.remaining} left`;
  $('bar').style.width=`${100*p.complete/p.total}%`; dirty=false; status(row.complete?'Complete':'Not yet complete'); window.scrollTo({top:0,behavior:'instant'});
}
function status(message,error=false){ $('status').textContent=message; $('status').classList.toggle('error',error); }
async function load(index){ const r=await fetch(`/api/row?index=${index}`); if(!r.ok) return status(await r.text(),true); row=await r.json(); render(); }
async function save(advance=false){
  clearTimeout(saveTimer); if(!dirty && !advance){return;}
  status('Saving…'); const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(current())});
  if(!r.ok){status(await r.text(),true);return false;} row=await r.json(); dirty=false; status(row.complete?'Saved ✓':'Draft saved');
  const p=row.progress; $('progressText').textContent=`${p.complete}/${p.total} complete · ${p.remaining} left`; $('bar').style.width=`${100*p.complete/p.total}%`;
  if(advance){ if(!isComplete()){status('Choose label, comparability, and confidence first.',true);return false;} await load(Math.min(row.index+1,p.total-1)); }
  return true;
}
function scheduleSave(){ dirty=true; clearTimeout(saveTimer); saveTimer=setTimeout(()=>save(false),500); }
function choose(group,value){
  row[group]=value;
  if(group==='label' && ['consistent','inconsistent'].includes(value)) row.comparable='yes';
  if(group==='label' && value==='unrelated') row.comparable='no';
  if(group==='label' && value==='ambiguous' && !['yes','no'].includes(row.comparable)) row.comparable='';
  selected('labels',row.label); selected('comparable',row.comparable); selected('confidence',row.confidence); scheduleSave();
}
function acceptSuggestion(){
  if(!row.suggestion_available)return;
  row.label=row.suggested_label; row.comparable=row.suggested_comparable; row.confidence=row.suggested_confidence;
  selected('labels',row.label); selected('comparable',row.comparable); selected('confidence',row.confidence); scheduleSave();
  status('Suggestion accepted — review or adjust, then continue.');
}
document.querySelectorAll('#labels button').forEach(b=>b.onclick=()=>choose('label',b.dataset.value));
document.querySelectorAll('#comparable button').forEach(b=>b.onclick=()=>choose('comparable',b.dataset.value));
document.querySelectorAll('#confidence button').forEach(b=>b.onclick=()=>choose('confidence',b.dataset.value));
$('notes').oninput=scheduleSave; $('saveNext').onclick=()=>save(true); $('prev').onclick=async()=>{await save(false);load(Math.max(0,row.index-1));};
$('acceptSuggestion').onclick=acceptSuggestion;
$('nextIncomplete').onclick=async()=>{await save(false); const i=row.progress.first_incomplete; load(i===null?row.index:i);};
document.addEventListener('keydown',async e=>{
  const typing=e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT';
  if(typing){if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();save(true);}return;}
  const labels={'1':'consistent','2':'inconsistent','3':'unrelated','4':'ambiguous'};
  const confidence={'l':'low','m':'medium','h':'high'};
  if(labels[e.key]) choose('label',labels[e.key]); else if(e.key.toLowerCase()==='y') choose('comparable','yes');
  else if(e.key.toLowerCase()==='n') choose('comparable','no'); else if(confidence[e.key.toLowerCase()]) choose('confidence',confidence[e.key.toLowerCase()]);
  else if(e.key.toLowerCase()==='a') acceptSuggestion();
  else if(e.key==='Enter'){e.preventDefault();save(true);} else if(e.key==='ArrowLeft'){await save(false);load(row.index-1);}
  else if(e.key==='ArrowRight'){await save(false);load(row.index+1);}
});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
load(0).then(()=>{if(row.progress.first_incomplete!==null)load(row.progress.first_incomplete);});
</script>
</body></html>"""


class ReviewerHandler(BaseHTTPRequestHandler):
    store: CodingStore

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value: Any, status: int = 200) -> None:
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/row":
            try:
                raw = parse_qs(parsed.query).get("index", ["0"])[0]
                self.send_json(self.store.public_row(int(raw)))
            except (ValueError, TypeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/save":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 65_536:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected one JSON object")
            self.send_json(self.store.update(payload))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def open_reviewer(url: str) -> None:
    if os.getenv("WSL_DISTRO_NAME"):
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            print(f"Could not open the Windows browser automatically: {error}")
        return
    if not webbrowser.open(url):
        print(f"Could not open a browser automatically; visit {url}")


def main() -> None:
    args = parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("For blinding and safety, --host must be a loopback address.")
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535")
    store = CodingStore(
        args.coding_file,
        args.translations_file,
        args.suggestions_file,
    )
    progress = store.progress()
    print(
        f"Validated {progress['total']} blinded rows: {progress['complete']} complete, "
        f"{progress['remaining']} remaining."
    )
    if args.check:
        return
    ReviewerHandler.store = store
    server = ThreadingHTTPServer((args.host, args.port), ReviewerHandler)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"Reviewer: {url}")
    print(f"Coding file: {store.path}")
    if store.translations_path.exists():
        print(
            f"English translations: {store.translations_path} "
            f"({len(store.translations)}/{len(store.rows)} rows available)"
        )
    else:
        print("English translations: not found; original text will be shown.")
    if store.schema == PRIMARY_SCHEMA and store.suggestions_path.exists():
        print(
            f"Reasoned suggestions: {store.suggestions_path} "
            f"({len(store.suggestions)}/{len(store.rows)} rows available)"
        )
    elif store.schema == PRIMARY_SCHEMA:
        print("Reasoned suggestions: not found; manual coding controls only.")
    print(f"First-save backup: {store.backup_path}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: open_reviewer(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReviewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
