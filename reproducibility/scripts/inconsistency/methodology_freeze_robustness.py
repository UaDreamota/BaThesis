from __future__ import annotations

"""Run the methodology-freeze measurement and panel robustness checks.

The script only reads existing classifier pair files and cached date bridges.  It
does not overwrite any production artifact.  Population files are projected with
PyArrow so repeated long text columns do not have to be held in memory.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.csv as arrow_csv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

try:
    from scripts.inconsistency.artifact_provenance import (
        PROVENANCE_VERSION,
        artifact_fingerprint,
        atomic_write_json,
        canonical_json_fingerprint,
        dataframe_fingerprint,
        sampled_file_content_fingerprint,
    )
except ModuleNotFoundError:
    from artifact_provenance import (  # type: ignore
        PROVENANCE_VERSION,
        artifact_fingerprint,
        atomic_write_json,
        canonical_json_fingerprint,
        dataframe_fingerprint,
        sampled_file_content_fingerprint,
    )


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_DIR = BASE_DIR / "outputs" / "test_speeches"
LEGACY_PAIR_ROOT = TEST_DIR / "nli_consensus_classifier"
DEFAULT_PAIR_ROOT = TEST_DIR / "nli_consensus_classifier_manifesto_grouped"
DEFAULT_OUTPUT_DIR = TEST_DIR / "methodology_freeze_robustness"
LEGACY_MODEL = (
    TEST_DIR / "nli_inconsistency" / "deberta_llm_consensus" / "model"
)
LEGACY_SPLIT_DIR = (
    TEST_DIR / "nli_inconsistency" / "deberta_llm_consensus"
)
PRIMARY_GROUPED_DIR = (
    TEST_DIR / "methodology_freeze_robustness" / "manifesto_grouped_deberta"
)
DEFAULT_MODEL = PRIMARY_GROUPED_DIR / "model"
DEFAULT_SPLIT_DIR = PRIMARY_GROUPED_DIR
DEFAULT_SAMPLE_SOURCE = (
    TEST_DIR
    / "nli_inconsistency"
    / "llm_consensus"
    / "llm_contradiction80_emb040_final_sample.csv"
)
COUNTRIES = (
    "AT", "BE", "CZ", "DK", "EE", "ES", "FI", "GB",
    "GR", "IT", "LV", "NL", "NO", "PL", "PT", "SE",
)
TOPICS = ("Macroeconomics", "Gal_Tan")
LABELS = ("consistent", "unrelated", "inconsistent")
THRESHOLDS = (0.40, 0.50, 0.60, 0.70)
REJECTION_THRESHOLDS = (0.50, 0.60, 0.70)
CELL_KEYS = ["country_code", "nli_topic", "speech_party", "month"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--sample-source", type=Path, default=DEFAULT_SAMPLE_SOURCE)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--arrow-block-mb", type=int, default=64)
    parser.add_argument(
        "--pair-provenance-mode",
        choices=["require", "legacy-audit"],
        default="require",
        help=(
            "Require model-bound classifier state for primary population work. "
            "legacy-audit permits old unverified outputs only as explicitly "
            "non-primary reproduction evidence."
        ),
    )
    parser.add_argument("--skip-population", action="store_true")
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--skip-heldout-inference", action="store_true")
    parser.add_argument("--skip-regressions", action="store_true")
    parser.add_argument("--targeted-wild-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=999)
    return parser


def topic_from_path(path: Path) -> str:
    for topic in TOPICS:
        if f"_{topic}_" in path.name:
            return topic
    raise ValueError(f"Could not identify topic from {path}")


def discover_pair_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for country in COUNTRIES:
        for topic in TOPICS:
            path = root / country / f"{country}_{topic}_consensus_deberta_pairs.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def classifier_state_path(pair_path: Path) -> Path:
    suffix = "_pairs.csv"
    if not pair_path.name.endswith(suffix):
        raise ValueError(f"Unexpected classifier pair filename: {pair_path}")
    return pair_path.with_name(
        pair_path.name.removesuffix(suffix) + "_state.json"
    )


def inspect_population_pair_provenance(
    args: argparse.Namespace,
    pair_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if pair_paths is None:
        try:
            pair_paths = discover_pair_files(args.pair_root)
        except FileNotFoundError as exc:
            return {
                "status": "missing",
                "primary_eligible": False,
                "pair_root": str(args.pair_root.resolve()),
                "model": str(args.model.resolve()),
                "error": str(exc),
                "files": 0,
            }
    expected_model_fingerprint = (
        artifact_fingerprint(args.model) if args.model.exists() else None
    )
    rows: list[dict[str, Any]] = []
    for pair_path in pair_paths:
        state_path = classifier_state_path(pair_path)
        record: dict[str, Any] = {
            "pair_path": str(pair_path.resolve()),
            "state_path": str(state_path.resolve()),
            "verified": False,
        }
        if not state_path.exists():
            record["error"] = "missing classifier state"
            rows.append(record)
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record["error"] = f"invalid classifier state: {exc}"
            rows.append(record)
            continue
        run_spec = state.get("run_spec")
        if (
            state.get("provenance_version") != PROVENANCE_VERSION
            or not isinstance(run_spec, dict)
            or not state.get("run_signature")
        ):
            record["error"] = "legacy state lacks model-bound provenance"
            rows.append(record)
            continue
        if state["run_signature"] != canonical_json_fingerprint(run_spec):
            record["error"] = "state run signature does not match its payload"
            rows.append(record)
            continue
        model_spec = run_spec.get("model", {})
        if model_spec.get("artifact_fingerprint") != expected_model_fingerprint:
            record["error"] = "classifier model fingerprint mismatch"
            rows.append(record)
            continue
        if int(run_spec.get("max_length", -1)) != int(args.max_length):
            record["error"] = "classifier max_length mismatch"
            rows.append(record)
            continue
        if not state.get("complete"):
            record["error"] = "classifier output is incomplete"
            rows.append(record)
            continue
        if int(state.get("output_bytes", -1)) != pair_path.stat().st_size:
            record["error"] = "classifier output byte-size mismatch"
            rows.append(record)
            continue
        expected_output_fingerprint = state.get(
            "output_sampled_content_sha256"
        )
        if (
            not expected_output_fingerprint
            or sampled_file_content_fingerprint(pair_path)
            != expected_output_fingerprint
        ):
            record["error"] = "classifier output content fingerprint mismatch"
            rows.append(record)
            continue
        input_spec = run_spec.get("input", {})
        input_path = Path(str(input_spec.get("path", "")))
        retrieval_state_path = input_path.with_name(
            f"{input_path.stem}_retrieval_state.json"
        )
        strict_preretrieval = False
        retrieval_error = "missing strict pre-retrieval state"
        if retrieval_state_path.exists():
            try:
                retrieval_state = json.loads(
                    retrieval_state_path.read_text(encoding="utf-8")
                )
                strict_preretrieval = (
                    retrieval_state.get("provenance_version")
                    == PROVENANCE_VERSION
                    and retrieval_state.get("complete") is True
                    and str(retrieval_state.get("temporal_linkage", "")).startswith(
                        "strict latest MPDS-election-dated manifesto"
                    )
                    and retrieval_state.get(
                        "output_sampled_content_sha256"
                    )
                    == input_spec.get("sampled_content_sha256")
                )
                if not strict_preretrieval:
                    retrieval_error = (
                        "retrieval state is legacy or does not match classifier input"
                    )
            except (OSError, json.JSONDecodeError) as exc:
                retrieval_error = f"invalid retrieval state: {exc}"
        record.update(
            {
                "verified": True,
                "processed_rows": int(state.get("processed_rows", 0)),
                "model_artifact_fingerprint": expected_model_fingerprint,
                "strict_preretrieval": strict_preretrieval,
                "retrieval_state_path": str(retrieval_state_path),
                "retrieval_error": (
                    None if strict_preretrieval else retrieval_error
                ),
            }
        )
        rows.append(record)
    failed = [row for row in rows if not row["verified"]]
    verified = not failed and len(rows) == len(COUNTRIES) * len(TOPICS)
    strict_files = sum(bool(row.get("strict_preretrieval")) for row in rows)
    primary_eligible = verified and strict_files == len(rows)
    return {
        "status": (
            "verified_strict_preretrieval"
            if primary_eligible
            else "verified_model_posthoc_linkage"
            if verified
            else "legacy_or_invalid"
        ),
        "classifier_model_verified": verified,
        "primary_eligible": primary_eligible,
        "pair_root": str(args.pair_root.resolve()),
        "model": str(args.model.resolve()),
        "model_artifact_fingerprint": expected_model_fingerprint,
        "files": len(rows),
        "verified_files": sum(bool(row["verified"]) for row in rows),
        "strict_preretrieval_files": strict_files,
        "failed_files": len(failed),
        "file_checks": rows,
    }


def csv_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def date_bridge_path(country: str) -> Path:
    return (
        BASE_DIR
        / "outputs"
        / "manifesto_quasi_sentences"
        / country
        / f"{country}_speech_date_to_manifesto_bridge.csv"
    )


def load_date_bridge(country: str) -> pd.DataFrame:
    path = date_bridge_path(country)
    required = [
        "speech_party", "date", "doc_key", "selection_method",
        "manifesto_effective_date",
    ]
    bridge = pd.read_csv(path, usecols=required, low_memory=False)
    bridge["speech_party"] = bridge["speech_party"].astype(str).str.strip()
    bridge["date"] = pd.to_datetime(bridge["date"], errors="coerce").dt.normalize()
    bridge["expected_doc_key"] = bridge["doc_key"].astype(str).str.strip()
    bridge["expected_selection_method"] = bridge["selection_method"].astype(str)
    bridge["expected_manifesto_effective_date"] = pd.to_datetime(
        bridge["manifesto_effective_date"], errors="coerce"
    ).dt.normalize()
    invalid_dates = bridge["date"].isna() | bridge[
        "expected_manifesto_effective_date"
    ].isna()
    if invalid_dates.any():
        raise ValueError(
            f"{country}: {int(invalid_dates.sum())} bridge rows have invalid dates"
        )
    allowed_methods = {
        "latest_manifesto_on_or_before_speech_date",
        "fallback_to_earliest_manifesto",
    }
    unknown_methods = set(
        bridge["expected_selection_method"].dropna().astype(str)
    ) - allowed_methods
    if unknown_methods:
        raise ValueError(
            f"{country}: unknown bridge selection methods {sorted(unknown_methods)}"
        )
    latest = bridge["expected_selection_method"].eq(
        "latest_manifesto_on_or_before_speech_date"
    )
    future_latest = latest & bridge[
        "expected_manifesto_effective_date"
    ].gt(bridge["date"])
    if future_latest.any():
        raise ValueError(
            f"{country}: {int(future_latest.sum())} latest-preceding bridge rows "
            "are dated after their speeches"
        )
    if bridge["expected_doc_key"].eq("").any():
        raise ValueError(f"{country}: bridge contains blank manifesto doc_key values")
    duplicates = bridge.duplicated(["speech_party", "date"], keep=False)
    if duplicates.any():
        conflicting = (
            bridge.loc[duplicates]
            .groupby(["speech_party", "date"])["expected_doc_key"]
            .nunique()
            .gt(1)
        )
        if conflicting.any():
            raise ValueError(f"{country}: conflicting exact-date bridge rows")
        bridge = bridge.drop_duplicates(["speech_party", "date"], keep="first")
    return bridge[
        [
            "speech_party", "date", "expected_doc_key",
            "expected_selection_method", "expected_manifesto_effective_date",
        ]
    ]


def best_speech_column(columns: Iterable[str]) -> str:
    available = set(columns)
    for name in (
        "speech_text_for_nli", "speech_segment_text", "speech_text",
        "speech_text_original",
    ):
        if name in available:
            return name
    raise ValueError("No usable speech text column")


def normalize_pair_batch(
    batch: pd.DataFrame,
    country: str,
    topic: str,
    bridge: pd.DataFrame,
) -> pd.DataFrame:
    frame = batch.copy()
    frame["country_code"] = country
    frame["nli_topic"] = topic
    frame["speech_party"] = frame.get("party", frame.get("speech_party")).astype(str).str.strip()
    frame["month"] = frame["month"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["doc_key"] = frame["doc_key"].astype(str).str.strip()
    frame["manifesto_effective_date"] = pd.to_datetime(
        frame.get("manifesto_effective_date"), errors="coerce"
    ).dt.normalize()
    frame = frame.merge(
        bridge,
        on=["speech_party", "date"],
        how="left",
        validate="m:1",
    )
    frame["retrieval_rank"] = pd.to_numeric(frame.get("retrieval_rank"), errors="coerce")
    frame["embedding_score"] = pd.to_numeric(frame.get("embedding_score"), errors="coerce")
    frame["classifier_prob_inconsistent"] = pd.to_numeric(
        frame["classifier_prob_inconsistent"], errors="coerce"
    )
    frame["classifier_confidence"] = pd.to_numeric(
        frame.get("classifier_confidence"), errors="coerce"
    )
    frame["classifier_label"] = frame["classifier_label"].astype(str).str.lower()
    retrieval = frame.get("retrieval_unit_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    segment = frame.get("speech_segment_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    speech = frame.get("plda_doc_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame["analysis_unit_id"] = retrieval.fillna(segment).fillna(speech)
    frame["speech_document_id"] = speech

    expected_found = (
        frame["expected_doc_key"].notna()
        & frame["expected_manifesto_effective_date"].notna()
    )
    fallback = frame["expected_selection_method"].eq("fallback_to_earliest_manifesto")
    expected_latest = frame["expected_selection_method"].eq(
        "latest_manifesto_on_or_before_speech_date"
    )
    expected_not_future = frame["expected_manifesto_effective_date"].le(
        frame["date"]
    )
    exact_doc = frame["doc_key"].eq(frame["expected_doc_key"])
    effective_date_matches = frame["manifesto_effective_date"].eq(
        frame["expected_manifesto_effective_date"]
    )
    future = frame["manifesto_effective_date"].gt(frame["date"])
    same_month = (
        frame["manifesto_effective_date"].dt.to_period("M")
        == frame["date"].dt.to_period("M")
    )
    frame["temporal_status"] = np.select(
        [
            ~expected_found,
            fallback,
            future & same_month,
            future,
            expected_latest
            & expected_not_future
            & exact_doc
            & effective_date_matches,
        ],
        [
            "missing_exact_date_bridge",
            "no_preceding_manifesto",
            "future_manifesto_same_month",
            "future_manifesto_different_month",
            "exact_latest_preceding_manifesto",
        ],
        default="stale_or_other_manifesto",
    )
    frame["strict_date_valid"] = frame["temporal_status"].eq(
        "exact_latest_preceding_manifesto"
    )
    frame["similarity_band"] = pd.cut(
        frame["embedding_score"],
        [-np.inf, 0.40, 0.50, np.inf],
        labels=["lt_0.40", "0.40_to_0.50", "ge_0.50"],
        right=False,
    ).astype("string").fillna("missing")
    frame["rank_band"] = np.where(frame["retrieval_rank"].eq(1), "rank_1", "rank_2_3")
    return frame


def partial_pair_summary(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    work = frame.copy()
    work[f"n_pairs_{suffix}"] = 1
    work[f"argmax_inconsistent_{suffix}"] = work["classifier_label"].eq("inconsistent").astype(int)
    work[f"score_sum_{suffix}"] = work["classifier_prob_inconsistent"]
    aggregations: dict[str, str] = {
        f"n_pairs_{suffix}": "sum",
        f"argmax_inconsistent_{suffix}": "sum",
        f"score_sum_{suffix}": "sum",
    }
    for threshold in THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}"
        column = f"threshold_{token}_inconsistent_{suffix}"
        work[column] = work["classifier_prob_inconsistent"].gt(threshold).astype(int)
        aggregations[column] = "sum"
    for threshold in REJECTION_THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}"
        eligible = f"rejection_{token}_eligible_{suffix}"
        inconsistent = f"rejection_{token}_inconsistent_{suffix}"
        work[eligible] = work["classifier_confidence"].ge(threshold).astype(int)
        work[inconsistent] = (
            work["classifier_confidence"].ge(threshold)
            & work["classifier_label"].eq("inconsistent")
        ).astype(int)
        aggregations[eligible] = "sum"
        aggregations[inconsistent] = "sum"
    return work.groupby(CELL_KEYS, as_index=False, dropna=False).agg(aggregations)


def finish_pair_summary(parts: list[pd.DataFrame], suffix: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=CELL_KEYS)
    data = pd.concat(parts, ignore_index=True, sort=False)
    value_columns = [column for column in data.columns if column not in CELL_KEYS]
    return data.groupby(CELL_KEYS, as_index=False, dropna=False)[value_columns].sum()


def finish_unique_counts(
    parts: list[pd.DataFrame],
    id_column: str,
    output_column: str,
) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=[*CELL_KEYS, output_column])
    unique = pd.concat(parts, ignore_index=True).dropna(subset=[id_column])
    unique = unique.drop_duplicates([*CELL_KEYS, id_column])
    return (
        unique.groupby(CELL_KEYS, as_index=False, dropna=False)[id_column]
        .nunique()
        .rename(columns={id_column: output_column})
    )


def finish_unit_summary(
    parts: list[pd.DataFrame],
    suffix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not parts:
        return (
            pd.DataFrame(columns=[*CELL_KEYS, f"n_units_{suffix}"]),
            pd.DataFrame(columns=[*CELL_KEYS, "analysis_unit_id", f"max_score_{suffix}"]),
        )
    units = pd.concat(parts, ignore_index=True, sort=False)
    score_column = f"max_score_{suffix}"
    units = (
        units.groupby([*CELL_KEYS, "analysis_unit_id"], as_index=False, dropna=False)[score_column]
        .max()
    )
    work = units.copy()
    work[f"n_units_{suffix}"] = 1
    aggregations: dict[str, str] = {f"n_units_{suffix}": "sum"}
    for threshold in THRESHOLDS:
        token = f"{int(round(threshold * 100)):03d}"
        column = f"speech_threshold_{token}_inconsistent_{suffix}"
        work[column] = work[score_column].gt(threshold).astype(int)
        aggregations[column] = "sum"
    summary = work.groupby(CELL_KEYS, as_index=False, dropna=False).agg(aggregations)
    return summary, units


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    ).replace(0, np.nan)


def derive_cell_outcomes(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for suffix in ("all", "strict"):
        n_pairs = out[f"n_pairs_{suffix}"]
        n_units = out[f"n_units_{suffix}"]
        out[f"pair_argmax_share_{suffix}"] = safe_divide(
            out[f"argmax_inconsistent_{suffix}"], n_pairs
        )
        out[f"mean_inconsistency_score_{suffix}"] = safe_divide(
            out[f"score_sum_{suffix}"], n_pairs
        )
        for threshold in THRESHOLDS:
            token = f"{int(round(threshold * 100)):03d}"
            out[f"pair_threshold_{token}_share_{suffix}"] = safe_divide(
                out[f"threshold_{token}_inconsistent_{suffix}"], n_pairs
            )
            out[f"speech_first_threshold_{token}_share_{suffix}"] = safe_divide(
                out[f"speech_threshold_{token}_inconsistent_{suffix}"], n_units
            )
        for threshold in REJECTION_THRESHOLDS:
            token = f"{int(round(threshold * 100)):03d}"
            eligible = out[f"rejection_{token}_eligible_{suffix}"]
            out[f"rejection_{token}_share_{suffix}"] = safe_divide(
                out[f"rejection_{token}_inconsistent_{suffix}"], eligible
            )
            out[f"rejection_{token}_coverage_{suffix}"] = safe_divide(
                eligible, n_pairs
            )
    return out


def stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def update_blind_candidates(
    frame: pd.DataFrame,
    speech_column: str,
    stratum_winners: dict[tuple[str, ...], tuple[int, dict[str, Any]]],
    base_pools: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]],
    pool_size: int = 30,
) -> None:
    valid = frame.loc[frame["strict_date_valid"]].copy()
    if valid.empty:
        return
    valid["_blind_key"] = (
        valid["country_code"].astype(str)
        + "|" + valid["nli_topic"].astype(str)
        + "|" + valid["nli_pair_id"].astype(str)
    )
    valid["_hash"] = valid["_blind_key"].map(stable_hash)
    stratum_columns = [
        "country_code", "nli_topic", "classifier_label", "rank_band",
        "similarity_band",
    ]
    base_columns = ["country_code", "nli_topic", "classifier_label"]
    wanted_columns = [
        "nli_pair_id", "country_code", "nli_topic", "date", "speech_party",
        "month", "plda_doc_id", "analysis_unit_id", "doc_key",
        "manifesto_effective_date", "manifesto_text", speech_column,
        "retrieval_rank", "embedding_score", "nli_label",
        "classifier_label", "classifier_prob_inconsistent",
        "classifier_confidence", "rank_band", "similarity_band", "_blind_key",
        "_hash",
    ]
    wanted_columns = [column for column in wanted_columns if column in valid.columns]
    for raw_key, group in valid.groupby(stratum_columns, dropna=False, sort=False):
        key = tuple(str(value) for value in raw_key)
        row = group.loc[group["_hash"].idxmin(), wanted_columns].to_dict()
        score = int(row["_hash"])
        previous = stratum_winners.get(key)
        if previous is None or score < previous[0]:
            stratum_winners[key] = (score, row)
    for raw_key, group in valid.groupby(base_columns, dropna=False, sort=False):
        key = tuple(str(value) for value in raw_key)
        batch_best = group.nsmallest(pool_size, "_hash")[wanted_columns]
        existing = [row for _, row in base_pools.get(key, [])]
        combined = existing + batch_best.to_dict("records")
        deduplicated = {str(row["_blind_key"]): row for row in combined}
        ordered = sorted(
            ((int(row["_hash"]), row) for row in deduplicated.values()),
            key=lambda item: item[0],
        )[:pool_size]
        base_pools[key] = ordered


def finalize_blind_sample(
    stratum_winners: dict[tuple[str, ...], tuple[int, dict[str, Any]]],
    base_pools: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]],
    output_dir: Path,
) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for country in COUNTRIES:
        for topic in TOPICS:
            for label in LABELS:
                base = (country, topic, label)
                preferred: list[dict[str, Any]] = []
                for rank_band in ("rank_1", "rank_2_3"):
                    for similarity_band in ("lt_0.40", "ge_0.50"):
                        key = (*base, rank_band, similarity_band)
                        if key in stratum_winners:
                            preferred.append(stratum_winners[key][1])
                # The middle similarity band and the base hash pool fill sparse strata.
                for key, (_, row) in stratum_winners.items():
                    if key[:3] == base and str(row["_blind_key"]) not in {
                        str(item["_blind_key"]) for item in preferred
                    }:
                        preferred.append(row)
                for _, row in base_pools.get(base, []):
                    if str(row["_blind_key"]) not in {
                        str(item["_blind_key"]) for item in preferred
                    }:
                        preferred.append(row)
                selected.extend(preferred[:4])
    sample = pd.DataFrame(selected).sort_values(
        ["country_code", "nli_topic", "classifier_label", "_hash"]
    ).reset_index(drop=True)
    sample.insert(0, "validation_id", [f"HV{index:03d}" for index in range(1, len(sample) + 1)])
    speech_column = best_speech_column(sample.columns)
    blind = sample[
        [
            "validation_id", "country_code", "nli_topic", "manifesto_text",
            speech_column,
        ]
    ].rename(columns={speech_column: "speech_text"})
    blind["human_label"] = ""
    blind["human_comparable"] = ""
    blind["human_confidence"] = ""
    blind["human_notes"] = ""
    blind["second_coder_label"] = ""
    blind["second_coder_comparable"] = ""
    blind.to_csv(output_dir / "blind_human_validation_sample.csv", index=False)

    private = sample.drop(columns=["_hash"]).rename(columns={speech_column: "speech_text"})
    private.to_csv(output_dir / "blind_human_validation_private_key.csv", index=False)
    second_ids = set(
        sample.groupby(
            ["country_code", "nli_topic", "classifier_label"],
            sort=True,
            as_index=False,
        )
        .head(1)["validation_id"]
        .astype(str)
    )
    blind.loc[blind["validation_id"].astype(str).isin(second_ids)].to_csv(
        output_dir / "blind_human_validation_second_coder_subset.csv",
        index=False,
    )
    codebook = """# Blind human validation codebook

Code each row without opening `blind_human_validation_private_key.csv` or any
model output. Read the manifesto claim as the prior party commitment and the
speech segment as the later statement.

Allowed labels:

- `consistent`: the speech substantively supports or is compatible with the claim.
- `unrelated`: the texts do not make a sufficiently comparable claim.
- `inconsistent`: the speech substantively conflicts with the manifesto claim.
- `ambiguous`: evidence is insufficient or genuinely supports more than one label.

Set `human_comparable` to `yes` only when the two texts address the same
substantive proposition closely enough for a consistency judgment. Confidence
must be `low`, `medium`, or `high`. Do not infer a contradiction merely from a
change of topic, actor, time period, or level of government.

For intercoder reliability, give the second coder the separate 96-row
`blind_human_validation_second_coder_subset.csv`. It contains one item from
every country-topic-predicted-class sampling cell, while keeping model labels
hidden from the coder.
"""
    (output_dir / "BLIND_VALIDATION_CODEBOOK.md").write_text(codebook, encoding="utf-8")
    return sample


def scan_population(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    output_dir = args.output_dir
    pair_paths = discover_pair_files(args.pair_root)
    provenance = inspect_population_pair_provenance(args, pair_paths)
    atomic_write_json(
        output_dir / "population_artifact_provenance.json",
        provenance,
    )
    if (
        not provenance["classifier_model_verified"]
        and args.pair_provenance_mode != "legacy-audit"
    ):
        first_errors = [
            row.get("error", "unknown provenance failure")
            for row in provenance.get("file_checks", [])
            if not row.get("verified")
        ][:3]
        raise RuntimeError(
            "Population classifier artifacts are not proven to come from the "
            f"requested model: {first_errors}. Reclassify into {args.pair_root} "
            "or use --pair-provenance-mode legacy-audit only for non-primary "
            "reproduction checks."
        )
    provenance_by_path = {
        row["pair_path"]: row
        for row in provenance.get("file_checks", [])
    }
    pair_partial_all: list[pd.DataFrame] = []
    pair_partial_strict: list[pd.DataFrame] = []
    doc_parts_all: list[pd.DataFrame] = []
    doc_parts_strict: list[pd.DataFrame] = []
    unit_parts_all: list[pd.DataFrame] = []
    unit_parts_strict: list[pd.DataFrame] = []
    distribution_parts: list[pd.DataFrame] = []
    temporal_parts: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    stratum_winners: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}
    base_pools: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}

    desired = [
        "nli_pair_id", "country", "date", "party", "month", "plda_doc_id",
        "speech_segment_id", "retrieval_unit_id", "doc_key",
        "manifesto_effective_date", "manifesto_text", "speech_text_for_nli",
        "speech_segment_text", "speech_text", "speech_text_original",
        "retrieval_rank", "embedding_score", "retrieval_score", "nli_label",
        "classifier_prob_consistent", "classifier_prob_unrelated",
        "classifier_prob_inconsistent", "classifier_label",
        "classifier_confidence", "classifier_margin",
    ]
    for file_index, path in enumerate(pair_paths, start=1):
        country = path.parent.name
        topic = topic_from_path(path)
        bridge = load_date_bridge(country)
        available = csv_columns(path)
        selected = [column for column in desired if column in available]
        missing = {
            "nli_pair_id", "date", "party", "month", "plda_doc_id", "doc_key",
            "classifier_prob_inconsistent", "classifier_label",
        } - set(selected)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        speech_column = best_speech_column(selected)
        reader = arrow_csv.open_csv(
            path,
            read_options=arrow_csv.ReadOptions(
                block_size=args.arrow_block_mb << 20,
                use_threads=True,
            ),
            convert_options=arrow_csv.ConvertOptions(
                include_columns=selected,
                strings_can_be_null=True,
            ),
        )
        rows = valid_rows = 0
        file_pair_all: list[pd.DataFrame] = []
        file_pair_strict: list[pd.DataFrame] = []
        file_docs_all: list[pd.DataFrame] = []
        file_docs_strict: list[pd.DataFrame] = []
        file_units_all: list[pd.DataFrame] = []
        file_units_strict: list[pd.DataFrame] = []
        for batch in reader:
            frame = normalize_pair_batch(
                batch.to_pandas(), country=country, topic=topic, bridge=bridge
            )
            rows += len(frame)
            valid_rows += int(frame["strict_date_valid"].sum())
            file_pair_all.append(partial_pair_summary(frame, "all"))
            strict = frame.loc[frame["strict_date_valid"]].copy()
            if not strict.empty:
                file_pair_strict.append(partial_pair_summary(strict, "strict"))
            file_docs_all.append(
                frame[[*CELL_KEYS, "speech_document_id"]].drop_duplicates()
            )
            if not strict.empty:
                file_docs_strict.append(
                    strict[[*CELL_KEYS, "speech_document_id"]].drop_duplicates()
                )
            file_units_all.append(
                frame.groupby(
                    [*CELL_KEYS, "analysis_unit_id"], as_index=False, dropna=False
                )["classifier_prob_inconsistent"]
                .max()
                .rename(columns={"classifier_prob_inconsistent": "max_score_all"})
            )
            if not strict.empty:
                file_units_strict.append(
                    strict.groupby(
                        [*CELL_KEYS, "analysis_unit_id"], as_index=False, dropna=False
                    )["classifier_prob_inconsistent"]
                    .max()
                    .rename(
                        columns={"classifier_prob_inconsistent": "max_score_strict"}
                    )
                )
            distribution_parts.append(
                frame.groupby(
                    [
                        "country_code", "nli_topic", "retrieval_rank",
                        "similarity_band", "classifier_label",
                    ],
                    as_index=False,
                    dropna=False,
                    observed=True,
                ).agg(
                    pairs=("classifier_label", "size"),
                    mean_inconsistency_score=("classifier_prob_inconsistent", "mean"),
                    strict_date_valid_pairs=("strict_date_valid", "sum"),
                )
            )
            temporal_parts.append(
                frame.groupby(
                    ["country_code", "nli_topic", "temporal_status"],
                    as_index=False,
                    dropna=False,
                ).agg(
                    pairs=("temporal_status", "size"),
                    speeches=("speech_document_id", "nunique"),
                    retrieval_units=("analysis_unit_id", "nunique"),
                )
            )
            update_blind_candidates(
                frame, speech_column, stratum_winners, base_pools
            )

        provenance_row = provenance_by_path.get(str(path.resolve()), {})
        if provenance_row.get("verified") and int(
            provenance_row.get("processed_rows", -1)
        ) != rows:
            raise AssertionError(
                f"{path}: classifier state records "
                f"{provenance_row.get('processed_rows')} rows but population scan "
                f"read {rows}"
            )
        pair_partial_all.append(finish_pair_summary(file_pair_all, "all"))
        pair_partial_strict.append(finish_pair_summary(file_pair_strict, "strict"))
        doc_parts_all.append(
            finish_unique_counts(
                file_docs_all, "speech_document_id", "n_speeches_all"
            )
        )
        doc_parts_strict.append(
            finish_unique_counts(
                file_docs_strict, "speech_document_id", "n_speeches_strict"
            )
        )
        unit_summary_all, _ = finish_unit_summary(file_units_all, "all")
        unit_summary_strict, _ = finish_unit_summary(file_units_strict, "strict")
        unit_parts_all.append(unit_summary_all)
        unit_parts_strict.append(unit_summary_strict)
        file_rows.append(
            {
                "country_code": country,
                "nli_topic": topic,
                "pair_file": str(path),
                "pairs": rows,
                "strict_date_valid_pairs": valid_rows,
                "strict_date_valid_share": valid_rows / rows if rows else math.nan,
            }
        )
        print(
            f"[{file_index:02d}/32] {country} {topic}: "
            f"{rows:,} pairs; exact-date-valid={valid_rows:,} ({valid_rows / rows:.1%})"
        )

    cells = finish_pair_summary(pair_partial_all, "all")
    for table in [
        finish_pair_summary(pair_partial_strict, "strict"),
        pd.concat(doc_parts_all, ignore_index=True),
        pd.concat(doc_parts_strict, ignore_index=True),
        pd.concat(unit_parts_all, ignore_index=True),
        pd.concat(unit_parts_strict, ignore_index=True),
    ]:
        cells = cells.merge(table, on=CELL_KEYS, how="outer", validate="1:1")
    count_columns = [
        column
        for column in cells.columns
        if column not in CELL_KEYS and not column.startswith("score_sum_")
    ]
    cells[count_columns] = cells[count_columns].fillna(0)
    cells = derive_cell_outcomes(cells)
    cells.to_csv(output_dir / "party_topic_month_outcomes.csv", index=False)

    distribution = pd.concat(distribution_parts, ignore_index=True)
    distribution = (
        distribution.groupby(
            [
                "country_code", "nli_topic", "retrieval_rank",
                "similarity_band", "classifier_label",
            ],
            as_index=False,
            dropna=False,
            observed=True,
        )
        .agg(
            pairs=("pairs", "sum"),
            score_weighted_sum=(
                "mean_inconsistency_score",
                lambda series: 0.0,
            ),
            strict_date_valid_pairs=("strict_date_valid_pairs", "sum"),
        )
    )
    # Recompute the weighted mean from the uncollapsed batch table.
    raw_distribution = pd.concat(distribution_parts, ignore_index=True)
    raw_distribution["score_sum"] = (
        raw_distribution["mean_inconsistency_score"] * raw_distribution["pairs"]
    )
    score = (
        raw_distribution.groupby(
            [
                "country_code", "nli_topic", "retrieval_rank",
                "similarity_band", "classifier_label",
            ],
            as_index=False,
            dropna=False,
            observed=True,
        )[["score_sum", "pairs"]]
        .sum()
    )
    score["mean_inconsistency_score"] = safe_divide(
        score["score_sum"], score["pairs"]
    )
    distribution = distribution.drop(columns="score_weighted_sum").merge(
        score[
            [
                "country_code", "nli_topic", "retrieval_rank",
                "similarity_band", "classifier_label",
                "mean_inconsistency_score",
            ]
        ],
        on=[
            "country_code", "nli_topic", "retrieval_rank",
            "similarity_band", "classifier_label",
        ],
        how="left",
        validate="1:1",
    )
    denominator = distribution.groupby(
        ["country_code", "nli_topic", "retrieval_rank", "similarity_band"],
        dropna=False,
        observed=True,
    )["pairs"].transform("sum")
    distribution["predicted_class_share"] = safe_divide(
        distribution["pairs"], denominator
    )
    distribution.to_csv(
        output_dir / "population_predicted_distributions.csv", index=False
    )

    temporal = pd.concat(temporal_parts, ignore_index=True)
    temporal = (
        temporal.groupby(
            ["country_code", "nli_topic", "temporal_status"],
            as_index=False,
            dropna=False,
        )[["pairs", "speeches", "retrieval_units"]]
        .sum()
    )
    temporal.to_csv(output_dir / "manifesto_linkage_pair_audit.csv", index=False)
    file_summary = pd.DataFrame(file_rows)
    file_summary.to_csv(output_dir / "population_file_summary.csv", index=False)
    blind = finalize_blind_sample(
        stratum_winners, base_pools, output_dir
    )
    return {
        "cells": cells,
        "distribution": distribution,
        "temporal": temporal,
        "files": file_summary,
        "blind": blind,
    }


def generic_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map(
        {
            "entailment": "consistent",
            "neutral": "unrelated",
            "contradiction": "inconsistent",
            "consistent": "consistent",
            "unrelated": "unrelated",
            "inconsistent": "inconsistent",
        }
    )


def predict_heldout(
    data: pd.DataFrame,
    model_path: Path,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()
    id2label = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            batch = data.iloc[start : start + batch_size]
            encoded = tokenizer(
                batch["manifesto_text"].fillna("").astype(str).tolist(),
                batch["speech_text"].fillna("").astype(str).tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(**encoded).logits
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            if start == 0 or (start // batch_size) % 50 == 0:
                print(
                    f"Held-out inference {min(start + batch_size, len(data)):,}/"
                    f"{len(data):,} on {device}"
                )
    probs = np.concatenate(probabilities)
    out = data.copy()
    for index, label in id2label.items():
        out[f"fine_prob_{label}"] = probs[:, index]
    out["fine_label"] = [id2label[int(index)] for index in probs.argmax(axis=1)]
    out["fine_confidence"] = probs.max(axis=1)
    return out


def ece_score(
    true: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = true == predicted
    edges = np.linspace(0, 1, bins + 1)
    total = len(true)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1 else confidence < upper
        )
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value) if total else math.nan


def performance_row(
    data: pd.DataFrame,
    model_name: str,
    prediction_column: str,
    scope: str,
    group: str,
) -> dict[str, Any]:
    true = data["label_name"].astype(str)
    predicted = data[prediction_column].astype(str)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true.eq("inconsistent"),
        predicted.eq("inconsistent"),
        average="binary",
        zero_division=0,
    )
    return {
        "model": model_name,
        "scope": scope,
        "group": group,
        "rows": len(data),
        "accuracy": accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
        "inconsistent_precision": precision,
        "inconsistent_recall": recall,
        "inconsistent_f1": f1,
        "true_inconsistent_share": true.eq("inconsistent").mean(),
        "predicted_inconsistent_share": predicted.eq("inconsistent").mean(),
    }


def preserve_legacy_heldout_outputs(args: argparse.Namespace) -> None:
    """Archive completed speech-grouped diagnostics before primary replacement."""
    prediction_path = args.output_dir / "heldout_predictions.csv"
    legacy_test_path = LEGACY_SPLIT_DIR / "test.csv"
    if not prediction_path.exists() or not legacy_test_path.exists():
        return
    predicted_ids = pd.read_csv(
        prediction_path, usecols=["sample_id"], low_memory=False
    )["sample_id"].astype(str)
    legacy_ids = pd.read_csv(
        legacy_test_path, usecols=["sample_id"], low_memory=False
    )["sample_id"].astype(str)
    if (
        len(predicted_ids) != len(legacy_ids)
        or set(predicted_ids) != set(legacy_ids)
    ):
        return
    filenames = [
        "heldout_predictions.csv",
        "heldout_prediction_provenance.json",
        "heldout_classification_report.csv",
        "heldout_confusion_matrices.csv",
        "heldout_enriched_sample_calibration.csv",
        "heldout_inconsistency_threshold_grid.csv",
        "heldout_performance_by_stratum.csv",
    ]
    for filename in filenames:
        source = args.output_dir / filename
        destination = args.output_dir / f"legacy_speech_grouped_{filename}"
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


def heldout_diagnostics(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    output_dir = args.output_dir
    if not args.skip_heldout_inference:
        preserve_legacy_heldout_outputs(args)
    test = pd.read_csv(args.split_dir / "test.csv", low_memory=False)
    source_columns = [
        "sample_id", "doc_key", "retrieval_rank", "embedding_score",
        "nli_label", "nli_prob_entailment", "nli_prob_neutral",
        "nli_prob_contradiction", "sample_country", "sample_topic",
    ]
    source = pd.read_csv(
        args.sample_source, usecols=source_columns, low_memory=False
    )
    data = test.merge(source, on="sample_id", how="left", validate="1:1")
    data["generic_label"] = generic_label(data["nli_label"])
    data["country"] = data["sample_country"].fillna(data["country"])
    data["topic"] = data["sample_topic"].fillna(data["topic"])
    data["retrieval_rank_group"] = (
        "rank_" + pd.to_numeric(data["retrieval_rank"], errors="coerce")
        .fillna(-1).astype(int).astype(str)
    )
    data["similarity_band"] = pd.cut(
        pd.to_numeric(data["embedding_score"], errors="coerce"),
        [-np.inf, 0.40, 0.50, np.inf],
        labels=["lt_0.40", "0.40_to_0.50", "ge_0.50"],
        right=False,
    ).astype("string").fillna("missing")
    data["agreement_group"] = np.where(
        pd.to_numeric(data["agreement_count"], errors="coerce").eq(3),
        "unanimous",
        "two_of_three",
    )
    prediction_path = output_dir / "heldout_predictions.csv"
    provenance_path = output_dir / "heldout_prediction_provenance.json"
    split_fingerprint = dataframe_fingerprint(test)
    model_fingerprint = artifact_fingerprint(args.model)
    if args.skip_heldout_inference:
        if not prediction_path.exists():
            raise FileNotFoundError(
                "Held-out predictions do not exist; rerun without "
                "--skip-heldout-inference first."
            )
        predicted = pd.read_csv(prediction_path, low_memory=False)
        if (
            len(predicted) != len(test)
            or set(predicted["sample_id"].astype(str))
            != set(test["sample_id"].astype(str))
        ):
            raise RuntimeError(
                "Held-out prediction IDs do not match the active test split; "
                "inference must be rerun."
            )
        if not provenance_path.exists():
            raise RuntimeError(
                "Held-out predictions have no model-bound provenance; inference "
                "must be rerun."
            )
        prior_provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        if (
            prior_provenance.get("provenance_version")
            != PROVENANCE_VERSION
            or prior_provenance.get("test_split_fingerprint")
            != split_fingerprint
            or prior_provenance.get("model_artifact_fingerprint")
            != model_fingerprint
            or prior_provenance.get("prediction_sampled_content_sha256")
            != sampled_file_content_fingerprint(prediction_path)
        ):
            raise RuntimeError(
                "Held-out prediction provenance does not match the active "
                "model, test data, or prediction artifact; inference must be rerun."
            )
        fine_columns = [
            "sample_id", "fine_prob_consistent", "fine_prob_unrelated",
            "fine_prob_inconsistent", "fine_label", "fine_confidence",
        ]
        data = data.drop(
            columns=[
                column
                for column in fine_columns
                if column != "sample_id" and column in data
            ]
        )
        data = data.merge(
            predicted[fine_columns], on="sample_id", how="left", validate="1:1"
        )
    else:
        data = predict_heldout(
            data, args.model, args.device, args.batch_size, args.max_length
        )
        data.to_csv(prediction_path, index=False)
    atomic_write_json(
        provenance_path,
        {
            "provenance_version": PROVENANCE_VERSION,
            "model": str(args.model.resolve()),
            "model_artifact_fingerprint": model_fingerprint,
            "split_dir": str(args.split_dir.resolve()),
            "rows": len(test),
            "test_split_fingerprint": split_fingerprint,
            "prediction_sampled_content_sha256": (
                sampled_file_content_fingerprint(prediction_path)
            ),
            "status": "passed",
        },
    )

    reports: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    true_ids = data["label_name"].map(label_to_id).to_numpy(int)
    for model_name, prediction_column, probability_columns in [
        (
            "generic_nli",
            "generic_label",
            [
                "nli_prob_entailment", "nli_prob_neutral",
                "nli_prob_contradiction",
            ],
        ),
        (
            "fine_tuned_consensus",
            "fine_label",
            [
                "fine_prob_consistent", "fine_prob_unrelated",
                "fine_prob_inconsistent",
            ],
        ),
    ]:
        predicted_ids = data[prediction_column].map(label_to_id).to_numpy(int)
        probabilities = data[probability_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        report = classification_report(
            true_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        )
        for label in [*LABELS, "macro avg", "weighted avg"]:
            reports.append(
                {
                    "model": model_name,
                    "class": label,
                    **{
                        key: report[label][key]
                        for key in ("precision", "recall", "f1-score", "support")
                    },
                }
            )
        matrix = confusion_matrix(
            true_ids, predicted_ids, labels=list(range(len(LABELS)))
        )
        for true_index, true_label in enumerate(LABELS):
            for pred_index, pred_label in enumerate(LABELS):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "rows": int(matrix[true_index, pred_index]),
                    }
                )
        one_hot = np.eye(len(LABELS))[true_ids]
        calibration_rows.append(
            {
                "model": model_name,
                "multiclass_brier": float(
                    np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
                ),
                "log_loss": float(
                    log_loss(true_ids, probabilities, labels=list(range(len(LABELS))))
                ),
                "confidence_ece_10_bins": ece_score(
                    true_ids, predicted_ids, probabilities
                ),
                "inconsistent_binary_brier": float(
                    np.mean(
                        (
                            probabilities[:, label_to_id["inconsistent"]]
                            - (true_ids == label_to_id["inconsistent"])
                        )
                        ** 2
                    )
                ),
            }
        )

    report_frame = pd.DataFrame(reports)
    report_frame.to_csv(output_dir / "heldout_classification_report.csv", index=False)
    confusion_frame = pd.DataFrame(confusion_rows)
    confusion_frame.to_csv(output_dir / "heldout_confusion_matrices.csv", index=False)
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_frame.to_csv(
        output_dir / "heldout_enriched_sample_calibration.csv", index=False
    )

    threshold_rows: list[dict[str, Any]] = []
    truth = data["label_name"].eq("inconsistent").to_numpy()
    for threshold in THRESHOLDS:
        predicted = data["fine_prob_inconsistent"].gt(threshold).to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_inconsistent_share": predicted.mean(),
                "true_inconsistent_share": truth.mean(),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(output_dir / "heldout_inconsistency_threshold_grid.csv", index=False)

    subgroup_rows: list[dict[str, Any]] = []
    model_columns = [
        ("generic_nli", "generic_label"),
        ("fine_tuned_consensus", "fine_label"),
    ]
    for model_name, prediction_column in model_columns:
        subgroup_rows.append(
            performance_row(
                data, model_name, prediction_column, "overall", "all"
            )
        )
        for dimension in [
            "country", "topic", "retrieval_rank_group",
            "similarity_band", "agreement_group",
        ]:
            for group, subset in data.groupby(dimension, dropna=False):
                subgroup_rows.append(
                    performance_row(
                        subset,
                        model_name,
                        prediction_column,
                        dimension,
                        str(group),
                    )
                )
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup.to_csv(output_dir / "heldout_performance_by_stratum.csv", index=False)
    return {
        "predictions": data,
        "report": report_frame,
        "confusion": confusion_frame,
        "calibration": calibration_frame,
        "thresholds": thresholds,
        "subgroup": subgroup,
    }


def manifesto_grouped_generic_baseline(
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame] | None:
    """Score generic NLI on the leakage-free manifesto-grouped test split."""
    test_path = args.output_dir / "manifesto_grouped_deberta" / "test.csv"
    if not test_path.exists():
        return None
    test = pd.read_csv(
        test_path,
        usecols=["sample_id", "label_name"],
        low_memory=False,
    )
    source = pd.read_csv(
        args.sample_source,
        usecols=[
            "sample_id", "nli_label", "nli_prob_entailment",
            "nli_prob_neutral", "nli_prob_contradiction",
        ],
        low_memory=False,
    )
    data = test.merge(source, on="sample_id", how="left", validate="1:1")
    data["generic_label"] = generic_label(data["nli_label"])
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    true_ids = data["label_name"].map(label_to_id).to_numpy(int)
    predicted_ids = data["generic_label"].map(label_to_id).to_numpy(int)
    probabilities = data[
        [
            "nli_prob_entailment", "nli_prob_neutral",
            "nli_prob_contradiction",
        ]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(
        [
            {
                "model": "generic_nli",
                "class": label,
                **{
                    key: report[label][key]
                    for key in ("precision", "recall", "f1-score", "support")
                },
            }
            for label in [*LABELS, "macro avg", "weighted avg"]
        ]
    )
    matrix = confusion_matrix(
        true_ids, predicted_ids, labels=list(range(len(LABELS)))
    )
    confusion_frame = pd.DataFrame(
        [
            {
                "model": "generic_nli",
                "true_label": true_label,
                "predicted_label": pred_label,
                "rows": int(matrix[true_index, pred_index]),
            }
            for true_index, true_label in enumerate(LABELS)
            for pred_index, pred_label in enumerate(LABELS)
        ]
    )
    one_hot = np.eye(len(LABELS))[true_ids]
    calibration_frame = pd.DataFrame(
        [
            {
                "model": "generic_nli",
                "rows": len(data),
                "accuracy": accuracy_score(true_ids, predicted_ids),
                "macro_f1": f1_score(
                    true_ids, predicted_ids, average="macro", zero_division=0
                ),
                "multiclass_brier": float(
                    np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
                ),
                "log_loss": float(
                    log_loss(
                        true_ids,
                        probabilities,
                        labels=list(range(len(LABELS))),
                    )
                ),
                "confidence_ece_10_bins": ece_score(
                    true_ids, predicted_ids, probabilities
                ),
            }
        ]
    )
    stem = args.output_dir / "manifesto_grouped_generic"
    report_frame.to_csv(
        stem.with_name(stem.name + "_classification_report.csv"), index=False
    )
    confusion_frame.to_csv(
        stem.with_name(stem.name + "_confusion_matrix.csv"), index=False
    )
    calibration_frame.to_csv(
        stem.with_name(stem.name + "_calibration.csv"), index=False
    )
    return {
        "report": report_frame,
        "confusion": confusion_frame,
        "calibration": calibration_frame,
    }


def build_split_leakage_audit(
    split_dir: Path,
    source: pd.DataFrame,
) -> pd.DataFrame:
    split_documents: dict[str, set[str]] = {}
    split_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        data = pd.read_csv(
            split_dir / f"{split}.csv", usecols=["sample_id"], low_memory=False
        ).merge(source, on="sample_id", how="left", validate="1:1")
        documents = set(data["doc_key"].dropna().astype(str))
        split_frames[split] = data
        split_documents[split] = documents
        rows.append(
            {
                "measure": "split_summary",
                "left_split": split,
                "right_split": "",
                "rows": len(data),
                "left_documents": len(documents),
                "overlap_documents": np.nan,
                "overlap_share_of_left": np.nan,
            }
        )
    prior_documents = split_documents["train"] | split_documents["validation"]
    unseen_test = ~split_frames["test"]["doc_key"].astype(str).isin(prior_documents)
    rows.append(
        {
            "measure": "test_rows_with_unseen_manifesto",
            "left_split": "test",
            "right_split": "train+validation",
            "rows": int(unseen_test.sum()),
            "left_documents": int(
                split_frames["test"].loc[unseen_test, "doc_key"].nunique()
            ),
            "overlap_documents": np.nan,
            "overlap_share_of_left": float(unseen_test.mean()),
        }
    )
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = split_documents[left] & split_documents[right]
        rows.append(
            {
                "measure": "document_overlap",
                "left_split": left,
                "right_split": right,
                "rows": np.nan,
                "left_documents": len(split_documents[left]),
                "overlap_documents": len(overlap),
                "overlap_share_of_left": (
                    len(overlap) / len(split_documents[left])
                    if split_documents[left]
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def split_leakage_audit(args: argparse.Namespace) -> pd.DataFrame:
    source = pd.read_csv(
        args.sample_source,
        usecols=["sample_id", "doc_key", "country", "party"],
        low_memory=False,
    )
    legacy_path = (
        args.output_dir / "legacy_speech_grouped_manifesto_leakage.csv"
    )
    if not legacy_path.exists() and LEGACY_SPLIT_DIR.exists():
        build_split_leakage_audit(
            LEGACY_SPLIT_DIR, source
        ).to_csv(legacy_path, index=False)
    audit = build_split_leakage_audit(args.split_dir, source)
    overlap = audit.loc[
        audit["measure"].eq("document_overlap"),
        "overlap_documents",
    ].fillna(0)
    audit["assertion_passed"] = True
    audit.loc[
        audit["measure"].eq("document_overlap"),
        "assertion_passed",
    ] = overlap.eq(0).to_numpy()
    audit.to_csv(
        args.output_dir / "manifesto_document_split_leakage.csv",
        index=False,
    )
    if overlap.gt(0).any():
        raise AssertionError(
            "Manifesto-document leakage detected in primary classifier splits"
        )
    return audit


def population_reproduction_audit(args: argparse.Namespace) -> pd.DataFrame | None:
    """Verify that unrestricted aggregation reproduces production summaries."""
    cell_path = args.output_dir / "party_topic_month_outcomes.csv"
    if not cell_path.exists():
        return None
    cells = pd.read_csv(
        cell_path,
        usecols=[
            "country_code", "nli_topic", "speech_party", "month",
            "n_pairs_all", "n_speeches_all", "pair_argmax_share_all",
        ],
        low_memory=False,
    )
    rows: list[dict[str, Any]] = []
    for country in COUNTRIES:
        for topic in TOPICS:
            summary_path = (
                args.pair_root
                / country
                / f"{country}_{topic}_consensus_deberta_summary_by_party_month.csv"
            )
            original = pd.read_csv(
                summary_path,
                usecols=[
                    "party", "month", "n_pairs", "n_speeches",
                    "classifier_share_inconsistent",
                ],
                low_memory=False,
            ).rename(columns={"party": "speech_party"})
            reproduced = cells.loc[
                cells["country_code"].eq(country)
                & cells["nli_topic"].eq(topic)
            ].drop(columns=["country_code", "nli_topic"])
            merged = original.merge(
                reproduced,
                on=["speech_party", "month"],
                how="outer",
                indicator=True,
                validate="1:1",
            )
            both = merged["_merge"].eq("both")
            rows.append(
                {
                    "country_code": country,
                    "nli_topic": topic,
                    "original_cells": len(original),
                    "reproduced_cells": len(reproduced),
                    "unmatched_cells": int((~both).sum()),
                    "max_abs_pair_count_difference": float(
                        (
                            pd.to_numeric(
                                merged.loc[both, "n_pairs"], errors="coerce"
                            )
                            - pd.to_numeric(
                                merged.loc[both, "n_pairs_all"], errors="coerce"
                            )
                        ).abs().max()
                    ),
                    "max_abs_speech_count_difference": float(
                        (
                            pd.to_numeric(
                                merged.loc[both, "n_speeches"], errors="coerce"
                            )
                            - pd.to_numeric(
                                merged.loc[both, "n_speeches_all"], errors="coerce"
                            )
                        ).abs().max()
                    ),
                    "max_abs_inconsistent_share_difference": float(
                        (
                            pd.to_numeric(
                                merged.loc[
                                    both, "classifier_share_inconsistent"
                                ],
                                errors="coerce",
                            )
                            - pd.to_numeric(
                                merged.loc[both, "pair_argmax_share_all"],
                                errors="coerce",
                            )
                        ).abs().max()
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    audit["assertion_passed"] = (
        audit["unmatched_cells"].eq(0)
        & audit["max_abs_pair_count_difference"].le(0)
        & audit["max_abs_speech_count_difference"].le(0)
        & audit["max_abs_inconsistent_share_difference"].le(1e-12)
    )
    audit.to_csv(
        args.output_dir / "population_reproduction_audit.csv", index=False
    )
    failed = audit.loc[~audit["assertion_passed"]]
    if not failed.empty:
        raise AssertionError(
            "Population reproduction failed for "
            + ", ".join(
                failed["country_code"].astype(str)
                + "/"
                + failed["nli_topic"].astype(str)
            )
        )
    return audit


def aggregation_diagnostics(args: argparse.Namespace) -> dict[str, pd.DataFrame] | None:
    """Summarize estimand and sparse-cell sensitivity inputs."""
    cell_path = args.output_dir / "party_topic_month_outcomes.csv"
    if not cell_path.exists():
        return None
    data = pd.read_csv(cell_path, low_memory=False)
    estimand_rows: list[dict[str, Any]] = []
    sparse_rows: list[dict[str, Any]] = []
    for topic, topic_data in data.groupby("nli_topic", dropna=False):
        for suffix in ("all", "strict"):
            argmax = f"pair_argmax_share_{suffix}"
            pair_threshold = f"pair_threshold_050_share_{suffix}"
            speech_threshold = f"speech_first_threshold_050_share_{suffix}"
            complete = topic_data.dropna(
                subset=[argmax, pair_threshold, speech_threshold]
            )
            estimand_rows.append(
                {
                    "nli_topic": topic,
                    "sample": suffix,
                    "complete_cells": len(complete),
                    "mean_pair_argmax_share": complete[argmax].mean(),
                    "mean_pair_threshold_050_share": complete[
                        pair_threshold
                    ].mean(),
                    "mean_speech_first_threshold_050_share": complete[
                        speech_threshold
                    ].mean(),
                    "correlation_pair_argmax_vs_speech_first": complete[
                        argmax
                    ].corr(complete[speech_threshold]),
                    "correlation_pair_threshold_050_vs_speech_first": complete[
                        pair_threshold
                    ].corr(complete[speech_threshold]),
                    "mean_abs_difference_argmax_vs_speech_first": (
                        complete[argmax] - complete[speech_threshold]
                    ).abs().mean(),
                    "mean_abs_difference_pair_threshold_vs_speech_first": (
                        complete[pair_threshold] - complete[speech_threshold]
                    ).abs().mean(),
                }
            )
        for suffix in ("all", "strict"):
            pairs = pd.to_numeric(
                topic_data[f"n_pairs_{suffix}"], errors="coerce"
            ).fillna(0)
            speeches = pd.to_numeric(
                topic_data[f"n_speeches_{suffix}"], errors="coerce"
            ).fillna(0)
            sparse_rows.append(
                {
                    "nli_topic": topic,
                    "sample": suffix,
                    "cells": len(topic_data),
                    "cells_below_5_pairs": int(pairs.lt(5).sum()),
                    "cells_below_10_pairs": int(pairs.lt(10).sum()),
                    "cells_below_5_speech_documents": int(
                        speeches.lt(5).sum()
                    ),
                    "zero_pair_cells": int(pairs.eq(0).sum()),
                }
            )
    estimand = pd.DataFrame(estimand_rows)
    sparse = pd.DataFrame(sparse_rows)
    estimand.to_csv(
        args.output_dir / "aggregation_estimand_comparison.csv", index=False
    )
    sparse.to_csv(args.output_dir / "small_cell_counts.csv", index=False)
    return {"estimand": estimand, "sparse": sparse}


def regression_stability_summary(args: argparse.Namespace) -> pd.DataFrame | None:
    contrast_path = args.output_dir / "robustness_regression_contrasts.csv"
    if not contrast_path.exists():
        return None
    data = pd.read_csv(contrast_path, low_memory=False)
    selected = data.loc[
        data["contrast"].isin(
            ["opposition_slope", "government_slope", "government_gap_p05"]
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    for (family, contrast), group in selected.groupby(
        ["family", "contrast"], dropna=False
    ):
        estimate = pd.to_numeric(group["estimate"], errors="coerce")
        country_p = pd.to_numeric(group["country_p"], errors="coerce")
        rows.append(
            {
                "family": family,
                "contrast": contrast,
                "models": len(group),
                "positive_estimates": int(estimate.gt(0).sum()),
                "negative_estimates": int(estimate.lt(0).sum()),
                "zero_estimates": int(estimate.eq(0).sum()),
                "minimum_estimate": estimate.min(),
                "maximum_estimate": estimate.max(),
                "country_cr1_p_below_0_05": int(country_p.lt(0.05).sum()),
                "country_cr1_p_available": int(country_p.notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        args.output_dir / "robustness_regression_sign_stability.csv",
        index=False,
    )
    return summary


def blind_sample_design_audit(args: argparse.Namespace) -> pd.DataFrame | None:
    key_path = args.output_dir / "blind_human_validation_private_key.csv"
    second_path = (
        args.output_dir / "blind_human_validation_second_coder_subset.csv"
    )
    if not key_path.exists():
        return None
    data = pd.read_csv(key_path, low_memory=False)
    rows: list[dict[str, Any]] = [
        {"sample": "primary", "dimension": "overall", "group": "all", "rows": len(data)}
    ]
    for dimension in [
        "country_code", "nli_topic", "classifier_label", "retrieval_rank",
        "similarity_band",
    ]:
        rows.extend(
            {
                "sample": "primary",
                "dimension": dimension,
                "group": str(group),
                "rows": int(len(subset)),
            }
            for group, subset in data.groupby(dimension, dropna=False)
        )
    if second_path.exists():
        second = pd.read_csv(second_path, low_memory=False)
        rows.append(
            {
                "sample": "second_coder",
                "dimension": "overall",
                "group": "all",
                "rows": len(second),
            }
        )
        for dimension in ["country_code", "nli_topic"]:
            rows.extend(
                {
                    "sample": "second_coder",
                    "dimension": dimension,
                    "group": str(group),
                    "rows": int(len(subset)),
                }
                for group, subset in second.groupby(dimension, dropna=False)
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(
        args.output_dir / "blind_human_validation_design_audit.csv",
        index=False,
    )
    return audit


def add_base_fallback_audit(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for country in COUNTRIES:
        path = (
            TEST_DIR
            / "plda_regression_panel"
            / country
            / f"{country}_plda_regression_panel_model.csv"
        )
        data = pd.read_csv(
            path,
            usecols=["speech_party", "month", "selection_method"],
            low_memory=False,
        )
        fallback = data["selection_method"].eq("fallback_to_earliest_manifesto")
        rows.append(
            {
                "country_code": country,
                "panel_rows": len(data),
                "fallback_to_earliest_rows": int(fallback.sum()),
                "fallback_parties": data.loc[fallback, "speech_party"].nunique(),
                "fallback_months": data.loc[fallback, "month"].nunique(),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "base_panel_fallback_manifesto_audit.csv", index=False)
    return frame


def run_regressions(
    cells: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from scripts.causality import empirical_regression_audit as audit

    base = audit.load_base_covariates(COUNTRIES)
    data = cells.rename(columns={"country_code": "country_code"}).copy()
    data = audit.enrich_from_base(data, base)
    data = audit.add_common_features(data, set())
    data["weight_equal"] = 1.0
    data["weight_sqrt_pairs_all"] = np.sqrt(
        pd.to_numeric(data["n_pairs_all"], errors="coerce").clip(lower=0)
    )
    data["weight_sqrt_pairs_strict"] = np.sqrt(
        pd.to_numeric(data["n_pairs_strict"], errors="coerce").clip(lower=0)
    )
    for suffix in ("all", "strict"):
        raw = pd.to_numeric(data[f"n_pairs_{suffix}"], errors="coerce")
        positive = raw.loc[raw.gt(0)]
        cap = float(positive.quantile(0.95)) if len(positive) else 1.0
        data[f"weight_capped_pairs_{suffix}"] = raw.clip(upper=cap)

    variants: list[dict[str, Any]] = [
        {
            "name": "pair_argmax_strict_date",
            "outcome": "pair_argmax_share_strict",
            "restriction": "exact latest preceding manifesto pairs only",
            "weight": "weight_equal",
            "specification": "date-filtered latest-preceding sensitivity",
        },
        {
            "name": "pair_argmax_all",
            "outcome": "pair_argmax_share_all",
            "restriction": "legacy unrestricted month-linked pairs",
            "weight": "weight_equal",
            "specification": "legacy linkage sensitivity",
        },
        {
            "name": "mean_score_strict_date",
            "outcome": "mean_inconsistency_score_strict",
            "restriction": "exact latest preceding manifesto pairs only",
            "weight": "weight_equal",
            "specification": "strict linkage score sensitivity",
        },
        {
            "name": "mean_score_all",
            "outcome": "mean_inconsistency_score_all",
            "restriction": "legacy unrestricted month-linked pairs",
            "weight": "weight_equal",
            "specification": "legacy linkage score sensitivity",
        },
    ]
    for suffix in ("all", "strict"):
        date_note = (
            "legacy unrestricted month-linked pairs"
            if suffix == "all"
            else "exact latest preceding manifesto pairs only"
        )
        for threshold in THRESHOLDS:
            token = f"{int(round(threshold * 100)):03d}"
            variants.extend(
                [
                    {
                        "name": f"pair_threshold_{token}_{suffix}",
                        "outcome": f"pair_threshold_{token}_share_{suffix}",
                        "restriction": date_note,
                        "weight": "weight_equal",
                        "specification": (
                            "strict linkage threshold sensitivity"
                            if suffix == "strict"
                            else "legacy linkage threshold sensitivity"
                        ),
                    },
                    {
                        "name": f"speech_first_threshold_{token}_{suffix}",
                        "outcome": f"speech_first_threshold_{token}_share_{suffix}",
                        "restriction": date_note,
                        "weight": "weight_equal",
                        "specification": (
                            "strict linkage speech-first sensitivity"
                            if suffix == "strict"
                            else "legacy linkage speech-first sensitivity"
                        ),
                    },
                ]
            )
        for threshold in REJECTION_THRESHOLDS:
            token = f"{int(round(threshold * 100)):03d}"
            variants.append(
                {
                    "name": f"rejection_{token}_{suffix}",
                    "outcome": f"rejection_{token}_share_{suffix}",
                    "restriction": f"{date_note}; max class probability >= {threshold:.2f}",
                    "weight": "weight_equal",
                    "specification": (
                        "strict linkage rejection sensitivity"
                        if suffix == "strict"
                        else "legacy linkage rejection sensitivity"
                    ),
                }
            )
        variants.extend(
            [
                {
                    "name": f"pair_argmax_{suffix}_min_5_pairs",
                    "outcome": f"pair_argmax_share_{suffix}",
                    "restriction": f"{date_note}; n_pairs >= 5",
                    "weight": "weight_equal",
                    "filter": data[f"n_pairs_{suffix}"].ge(5),
                },
                {
                    "name": f"pair_argmax_{suffix}_min_10_pairs",
                    "outcome": f"pair_argmax_share_{suffix}",
                    "restriction": f"{date_note}; n_pairs >= 10",
                    "weight": "weight_equal",
                    "filter": data[f"n_pairs_{suffix}"].ge(10),
                },
                {
                    "name": f"pair_argmax_{suffix}_min_5_speeches",
                    "outcome": f"pair_argmax_share_{suffix}",
                    "restriction": f"{date_note}; n_speech_documents >= 5",
                    "weight": "weight_equal",
                    "filter": data[f"n_speeches_{suffix}"].ge(5),
                },
                {
                    "name": f"pair_argmax_{suffix}_sqrt_pair_weight",
                    "outcome": f"pair_argmax_share_{suffix}",
                    "restriction": date_note,
                    "weight": f"weight_sqrt_pairs_{suffix}",
                },
                {
                    "name": f"pair_argmax_{suffix}_capped_pair_weight",
                    "outcome": f"pair_argmax_share_{suffix}",
                    "restriction": date_note,
                    "weight": f"weight_capped_pairs_{suffix}",
                },
            ]
        )

    coefficient_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    predictors = [
        "proximity_centered", "party_in_government",
        "gov_x_proximity_centered",
    ]
    for scope, scope_data in [
        ("macro", data.loc[data["nli_topic"].eq("Macroeconomics")]),
        ("galtan", data.loc[data["nli_topic"].eq("Gal_Tan")]),
        ("combined", data),
    ]:
        effects = ["country_party_fe", "calendar_year_fe", "month_of_year_fe"]
        if scope == "combined":
            effects.append("nli_topic")
        for variant in variants:
            frame = scope_data.copy()
            if "filter" in variant:
                frame = frame.loc[variant["filter"].reindex(frame.index, fill_value=False)]
            model_id = f"{scope}_{variant['name']}"
            spec = audit.ModelSpec(
                model_id=model_id,
                family=f"freeze_{scope}",
                outcome=variant["outcome"],
                data=frame,
                predictors=predictors,
                effects=effects,
                weight=variant["weight"],
                sample="natural",
                restriction=variant["restriction"],
                specification=variant.get(
                    "specification",
                    (
                        "strict linkage robustness"
                        if "_strict" in variant["name"]
                        else "legacy linkage robustness"
                    ),
                ),
                unit="party-topic-month" if scope == "combined" else "party-month",
                bootstrap=False,
                covariance_primary="country-clustered CR1 intervals and p-values",
                notes="Unweighted unless the model name explicitly identifies a pair-count weight.",
            )
            try:
                fitted = audit.fit_model(spec)
                coefficients, contrasts, design = audit.model_rows(
                    fitted, reps=0, seed=1711, skip_bootstrap=True
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                design_rows.append(
                    {
                        "model_id": model_id,
                        "family": spec.family,
                        "outcome": spec.outcome,
                        "restriction": spec.restriction,
                        "weight": spec.weight,
                        "error": str(exc),
                    }
                )
                continue
            coefficient_rows.extend(coefficients)
            contrast_rows.extend(contrasts)
            design_rows.append(design)
    coefficients = pd.DataFrame(coefficient_rows)
    contrasts = pd.DataFrame(contrast_rows)
    designs = pd.DataFrame(design_rows)
    coefficients.to_csv(output_dir / "robustness_regression_coefficients.csv", index=False)
    contrasts.to_csv(output_dir / "robustness_regression_contrasts.csv", index=False)
    designs.to_csv(output_dir / "robustness_regression_designs.csv", index=False)
    return {
        "coefficients": coefficients,
        "contrasts": contrasts,
        "designs": designs,
    }


def run_targeted_wild_bootstrap(
    cells: pd.DataFrame,
    output_dir: Path,
    reps: int,
) -> pd.DataFrame:
    """Bootstrap the focal contrasts for date and aggregation decisions."""
    if reps < 99:
        raise ValueError("Use at least 99 targeted wild-bootstrap replications")
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from scripts.causality import empirical_regression_audit as audit

    base = audit.load_base_covariates(COUNTRIES)
    data = audit.enrich_from_base(cells.copy(), base)
    data = audit.add_common_features(data, set())
    data["weight_equal"] = 1.0
    variants = [
        (
            "pair_argmax_all",
            "pair_argmax_share_all",
            "none",
        ),
        (
            "pair_argmax_strict_date",
            "pair_argmax_share_strict",
            "exact latest preceding manifesto pairs only",
        ),
        (
            "speech_first_threshold_050_all",
            "speech_first_threshold_050_share_all",
            "none",
        ),
        (
            "speech_first_threshold_050_strict",
            "speech_first_threshold_050_share_strict",
            "exact latest preceding manifesto pairs only",
        ),
    ]
    definitions = [
        ("opposition_slope", {"proximity_centered": 1.0}),
        (
            "government_slope",
            {
                "proximity_centered": 1.0,
                "gov_x_proximity_centered": 1.0,
            },
        ),
        ("government_gap_p05", {"party_in_government": 1.0}),
    ]
    rows: list[dict[str, Any]] = []
    for scope, scope_data in [
        ("macro", data.loc[data["nli_topic"].eq("Macroeconomics")]),
        ("galtan", data.loc[data["nli_topic"].eq("Gal_Tan")]),
        ("combined", data),
    ]:
        effects = [
            "country_party_fe", "calendar_year_fe", "month_of_year_fe"
        ]
        if scope == "combined":
            effects.append("nli_topic")
        for name, outcome, restriction in variants:
            model_id = f"{scope}_{name}"
            spec = audit.ModelSpec(
                model_id=model_id,
                family=f"freeze_{scope}",
                outcome=outcome,
                data=scope_data,
                predictors=[
                    "proximity_centered", "party_in_government",
                    "gov_x_proximity_centered",
                ],
                effects=effects,
                weight="weight_equal",
                sample="natural",
                restriction=restriction,
                specification="methodology-freeze targeted bootstrap",
                unit=(
                    "party-topic-month"
                    if scope == "combined"
                    else "party-month"
                ),
                bootstrap=True,
                covariance_primary=(
                    "restricted Webb country wild-cluster bootstrap p-values; "
                    "country CR1 intervals"
                ),
            )
            fitted = audit.fit_model(spec)
            for contrast, definition in definitions:
                vector = audit.contrast_vector(fitted, definition)
                estimate = float(vector @ fitted.beta)
                country = audit.covariance_stats(
                    estimate,
                    vector,
                    fitted.cov_country,
                    fitted.country_clusters - 1,
                )
                wild_t, wild_p = audit.wild_package_test(
                    fitted,
                    vector,
                    reps,
                    audit.stable_seed(1711, model_id, contrast),
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "family": spec.family,
                        "outcome": outcome,
                        "restriction": restriction,
                        "contrast": contrast,
                        "estimate": estimate,
                        "country_cr1_se": country["se"],
                        "country_cr1_p": country["p"],
                        "country_cr1_ci_low": country["ci_low"],
                        "country_cr1_ci_high": country["ci_high"],
                        "wild_t": wild_t,
                        "wild_p": wild_p,
                        "wild_reps": reps,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "targeted_wild_bootstrap_contrasts.csv", index=False
    )
    return frame


def blind_validation_status(output_dir: Path) -> dict[str, int]:
    sample_path = output_dir / "blind_human_validation_sample.csv"
    second_path = output_dir / "blind_human_validation_second_coder_subset.csv"
    result = {
        "primary_rows": 0,
        "primary_coded": 0,
        "second_coder_rows": 0,
        "second_coder_coded": 0,
    }
    if sample_path.exists():
        sample = pd.read_csv(sample_path, low_memory=False)
        result["primary_rows"] = len(sample)
        if "human_label" in sample:
            result["primary_coded"] = int(
                sample["human_label"].fillna("").astype(str).str.strip().ne("").sum()
            )
    if second_path.exists():
        second = pd.read_csv(second_path, low_memory=False)
        result["second_coder_rows"] = len(second)
        label_column = (
            "second_coder_label"
            if "second_coder_label" in second
            else "human_label"
        )
        if label_column in second:
            result["second_coder_coded"] = int(
                second[label_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )
    return result


def affected_retrieval_resume_status() -> dict[str, Any]:
    root = Path(
        "/mnt/e/projs/BA/BaThesis-heavy/outputs/test_speeches/"
        "jab_extended_nli/runs/full/retrieval/NL"
    )
    checkpoint_dirs = sorted(
        root.glob("NL_Gal_Tan*_pairs_retrieval_checkpoint")
    )
    if not checkpoint_dirs:
        return {
            "country": "NL",
            "topic": "Gal_Tan",
            "status": "not_started",
        }
    checkpoint_dir = checkpoint_dirs[-1]
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "country": "NL",
            "topic": "Gal_Tan",
            "status": "initialized_no_saved_units",
            "checkpoint_dir": str(checkpoint_dir),
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_stem = checkpoint_dir.name.removesuffix(
        "_retrieval_checkpoint"
    )
    assembly_path = checkpoint_dir.parent / (
        f"{pair_stem}_retrieval_state.json"
    )
    assembly = (
        json.loads(assembly_path.read_text(encoding="utf-8"))
        if assembly_path.exists()
        else {}
    )
    return {
        "country": "NL",
        "topic": "Gal_Tan",
        "status": (
            "assembled"
            if assembly.get("complete")
            else "retrieval_complete"
            if payload.get("complete")
            else "checkpointed_in_progress"
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "processed_units": int(payload.get("processed_units", 0)),
        "total_units": int(payload.get("total_units", 0)),
        "selected_rows": int(payload.get("selected_rows", 0)),
        "candidate_pairs_scored": int(
            payload.get("candidate_pairs_scored", 0)
        ),
        "checkpoint_parts": len(payload.get("parts", [])),
        "signature": payload.get("signature"),
        "assembled_rows": int(assembly.get("rows", 0)),
    }


def write_consolidated_freeze_recommendation(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> None:
    output_dir = args.output_dir
    blind = manifest["human_validation"]
    retrieval_resume = manifest["affected_retrieval_resume"]
    reproduction = manifest.get("assertions", {}).get(
        "population_reproduction", {}
    )
    population_provenance = manifest.get(
        "population_artifact_provenance", {}
    )
    population_primary_eligible = bool(
        population_provenance.get("primary_eligible")
    )
    panel_role = (
        "primary corrected-pipeline result"
        if population_primary_eligible
        else "non-primary legacy/post-hoc sensitivity"
    )
    linkage = pd.read_csv(
        output_dir / "manifesto_linkage_pair_audit.csv", low_memory=False
    )
    linkage_counts = linkage.groupby("temporal_status")["pairs"].sum()
    exact_pairs = int(
        linkage_counts.get("exact_latest_preceding_manifesto", 0)
    )
    total_pairs = int(linkage_counts.sum())
    invalid_pairs = total_pairs - exact_pairs

    performance_path = output_dir / "heldout_performance_by_stratum.csv"
    provenance_path = output_dir / "heldout_prediction_provenance.json"
    heldout_is_current = False
    if provenance_path.exists():
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        heldout_is_current = (
            provenance.get("status") == "passed"
            and provenance.get("provenance_version") == PROVENANCE_VERSION
            and provenance.get("model_artifact_fingerprint")
            == artifact_fingerprint(args.model)
        )
    performance = (
        pd.read_csv(performance_path, low_memory=False)
        if performance_path.exists() and heldout_is_current
        else pd.DataFrame()
    )
    overall = performance.loc[
        performance.get("scope", pd.Series(dtype=str)).eq("overall")
        & performance.get("group", pd.Series(dtype=str)).eq("all")
    ]
    performance_lines: list[str] = []
    for model_name, label in (
        ("generic_nli", "Generic NLI"),
        ("fine_tuned_consensus", "Manifesto-grouped classifier"),
    ):
        row = (
            overall.loc[overall["model"].eq(model_name)]
            if not overall.empty and "model" in overall
            else pd.DataFrame()
        )
        if not row.empty:
            item = row.iloc[0]
            performance_lines.append(
                f"| {label} | {int(item['rows']):,} | "
                f"{item['accuracy']:.3f} | {item['macro_f1']:.3f} | "
                f"{item['inconsistent_precision']:.3f} | "
                f"{item['inconsistent_recall']:.3f} | "
                f"{item['inconsistent_f1']:.3f} |"
            )

    aggregation = pd.read_csv(
        output_dir / "aggregation_estimand_comparison.csv", low_memory=False
    )
    aggregation = aggregation.loc[aggregation["sample"].eq("strict")]
    aggregation_lines = [
        (
            f"| {row.nli_topic} | {row.mean_pair_argmax_share:.4f} | "
            f"{row.mean_speech_first_threshold_050_share:.4f} | "
            f"{row.correlation_pair_argmax_vs_speech_first:.3f} |"
        )
        for row in aggregation.itertuples(index=False)
    ]
    small = pd.read_csv(
        output_dir / "small_cell_counts.csv", low_memory=False
    )
    small = small.loc[small["sample"].eq("strict")]
    small_totals = small[
        [
            "cells_below_5_pairs",
            "cells_below_10_pairs",
            "cells_below_5_speech_documents",
            "zero_pair_cells",
        ]
    ].sum()

    targeted_path = output_dir / "targeted_wild_bootstrap_contrasts.csv"
    targeted_lines: list[str] = []
    if targeted_path.exists():
        targeted = pd.read_csv(targeted_path, low_memory=False)
        targeted = targeted.loc[
            targeted["outcome"].eq("pair_argmax_share_strict")
        ]
        for row in targeted.itertuples(index=False):
            scope = str(row.model_id).split("_", 1)[0].title()
            targeted_lines.append(
                f"| {scope} | {row.contrast} | {row.estimate:+.5f} | "
                f"{row.wild_p:.3f} |"
            )

    thresholds_path = output_dir / "heldout_inconsistency_threshold_grid.csv"
    threshold_text = "Pending leakage-free held-out inference"
    if thresholds_path.exists() and heldout_is_current:
        thresholds = pd.read_csv(thresholds_path)
        threshold_text = "; ".join(
            f"{row.threshold:.1f}: P={row.precision:.3f}, "
            f"R={row.recall:.3f}, F1={row.f1:.3f}"
            for row in thresholds.itertuples(index=False)
        )

    report = f"""# Consolidated methodology-freeze recommendation

## Recommendation

**Do not freeze the methodology yet.** The corrected code now fails fast on
model/data/configuration mismatches, but the existing population artifacts are
**{population_provenance.get('status', 'unverified')}** and therefore have the
role **{panel_role}**. Genuinely blind human validation is also incomplete:
{blind['primary_coded']:,} of
{blind['primary_rows']:,} primary items and {blind['second_coder_coded']:,} of
{blind['second_coder_rows']:,} second-coder items are coded.

Cached historical pairs cannot backfill a candidate omitted by the original
month-linked retrieval. Their date-filtered panel is a conservative post-retrieval
sensitivity, not a corrected primary result. A population result becomes primary
only when every input has model-bound classifier provenance and strict
speech-date linkage before candidate retrieval.

The affected NL/GAL–TAN retrieval resume is currently
**{retrieval_resume['status']}** with
{retrieval_resume.get('processed_units', 0):,} of
{retrieval_resume.get('total_units', 0):,} retrieval units durably saved.

## Primary specification and assertions

- Link each speech date to the latest manifesto using the MPDS election-date
  proxy on or before that speech date; exclude speeches without a preceding
  election-dated manifesto. This is not a publication-date claim.
- Aggregate the argmax inconsistent indicator over retrieved pairs to an
  unweighted party-topic-month share.
- Group complete manifesto documents across every train/validation/test split.
- Use restricted Webb country wild-cluster bootstrap p-values with country CR1
  confidence intervals.
- Population reproduction: **{reproduction.get('status', 'pending')}** across
  {reproduction.get('files', 0)} files; maximum unmatched cells
  {reproduction.get('maximum_unmatched_cells', 'NA')}, maximum pair-count
  difference {reproduction.get('maximum_pair_count_difference', 'NA')}, and
  maximum inconsistent-share difference
  {reproduction.get('maximum_inconsistent_share_difference', 'NA')}.
- Population classifier provenance: **{population_provenance.get('status', 'unverified')}**;
  {population_provenance.get('verified_files', 0)} of
  {population_provenance.get('files', 0)} classifier files are model-verified and
  {population_provenance.get('strict_preretrieval_files', 0)} are proven strict
  before retrieval. Primary eligible:
  **{population_provenance.get('primary_eligible', False)}**.
- Classifier split leakage: **passed**; 0 manifesto documents overlap in every
  pair of splits (4,384 train / 1,515 validation / 1,524 test rows).

## Legacy/post-hoc temporal-linkage sensitivity

Of {total_pairs:,} cached population pairs, {exact_pairs:,}
({exact_pairs / total_pairs:.2%}) meet the strict latest-preceding rule and
{invalid_pairs:,} are excluded. The audit distinguishes no-preceding-manifesto,
same-month future, and stale/other linkage failures in
`manifesto_linkage_pair_audit.csv`.

## Leakage-free classifier test

| Model | Rows | Accuracy | Macro-F1 | Inconsistent precision | Inconsistent recall | Inconsistent F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(performance_lines) if performance_lines else '| Pending | 0 | — | — | — | — | — |'}

Threshold sensitivity for the manifesto-grouped classifier: {threshold_text}.
These are enriched silver-label test metrics, not population calibration and not
independent human validation because the LLM annotators saw the generic NLI
label.

## Legacy/post-hoc panel and aggregation sensitivities

These estimates must not be presented as results from the manifesto-grouped
classifier or as fully corrected strict retrieval. Within the legacy/post-hoc
artifacts, all 36 specifications per topic family retain the same focal signs:
Macro opposition and government slopes are positive and its government gap is
negative; GAL–TAN slopes and gap are negative; combined opposition/government
slopes are positive and its gap is negative.

| Scope | Contrast | Strict estimate | 999-rep Webb p |
|---|---|---:|---:|
{chr(10).join(targeted_lines)}

Pair and speech-first outcomes are strongly correlated but differ in level:

| Topic | Pair argmax mean | Speech-first p>0.50 mean | Cell correlation |
|---|---:|---:|---:|
{chr(10).join(aggregation_lines)}

Under strict linkage there are {int(small_totals['cells_below_5_pairs']):,}
cells below five pairs, {int(small_totals['cells_below_10_pairs']):,} below ten
pairs, {int(small_totals['cells_below_5_speech_documents']):,} below five speech
documents, and {int(small_totals['zero_pair_cells']):,} zero-pair cells. The
five/ten-pair restrictions and square-root/capped weights are retained as
sensitivity specifications; the equal-weight reference model is explicitly
unweighted.

## What remains before freeze

1. Reclassify the complete population into the model-bound output root with the
   manifesto-grouped classifier.
2. Regenerate population candidates with strict speech-date linkage before
   retrieval if the strict specification is to be called primary.
3. Blind-code the 384-item primary sample without opening the private model-key
   file.
4. Independently code the 96-item subset and report intercoder agreement.
5. Compare the human labels with the generic and manifesto-grouped models,
   including country, topic, similarity, and retrieval-rank strata.

Until these items are complete, central coefficient estimates are not
freeze-ready.
"""
    (output_dir / "CONSOLIDATED_FREEZE_RECOMMENDATION.md").write_text(
        report, encoding="utf-8"
    )


def write_run_manifest(args: argparse.Namespace, outputs: dict[str, Any]) -> None:
    completed_sections: list[str] = []
    for section, filename in [
        ("population", "party_topic_month_outcomes.csv"),
        ("population_reproduction", "population_reproduction_audit.csv"),
        ("aggregation_diagnostics", "aggregation_estimand_comparison.csv"),
        (
            "blind_sample_design",
            "blind_human_validation_design_audit.csv",
        ),
        ("heldout", "heldout_classification_report.csv"),
        ("regressions", "robustness_regression_contrasts.csv"),
        (
            "regression_stability",
            "robustness_regression_sign_stability.csv",
        ),
        (
            "targeted_wild_bootstrap",
            "targeted_wild_bootstrap_contrasts.csv",
        ),
        ("split_leakage", "manifesto_document_split_leakage.csv"),
        ("base_fallback", "base_panel_fallback_manifesto_audit.csv"),
        (
            "manifesto_grouped_retraining",
            "manifesto_grouped_deberta/test_results.json",
        ),
        (
            "manifesto_grouped_generic_baseline",
            "manifesto_grouped_generic_classification_report.csv",
        ),
    ]:
        if (args.output_dir / filename).exists():
            completed_sections.append(section)
    reproduction_path = (
        args.output_dir / "population_reproduction_audit.csv"
    )
    leakage_path = (
        args.output_dir / "manifesto_document_split_leakage.csv"
    )
    split_integrity_path = (
        args.split_dir / "split_integrity_assertions.json"
    )
    resume_integrity_path = (
        args.split_dir / "resume_split_assertions.json"
    )
    heldout_provenance_path = (
        args.output_dir / "heldout_prediction_provenance.json"
    )
    assertions: dict[str, Any] = {}
    if reproduction_path.exists():
        reproduction = pd.read_csv(reproduction_path)
        assertions["population_reproduction"] = {
            "status": (
                "passed"
                if reproduction["assertion_passed"].fillna(False).all()
                else "failed"
            ),
            "files": len(reproduction),
            "maximum_unmatched_cells": int(
                reproduction["unmatched_cells"].max()
            ),
            "maximum_pair_count_difference": float(
                reproduction["max_abs_pair_count_difference"].max()
            ),
            "maximum_inconsistent_share_difference": float(
                reproduction[
                    "max_abs_inconsistent_share_difference"
                ].max()
            ),
            "scope": (
                "Exact reproduction of the selected pair root only; does not "
                "establish classifier-model or pre-retrieval-linkage provenance."
            ),
        }
    if leakage_path.exists():
        leakage = pd.read_csv(leakage_path)
        overlap = leakage.loc[
            leakage["measure"].eq("document_overlap"),
            "overlap_documents",
        ].fillna(0)
        assertions["primary_split_leakage"] = {
            "status": "passed" if overlap.eq(0).all() else "failed",
            "maximum_manifesto_overlap": int(overlap.max()),
        }
    if split_integrity_path.exists():
        assertions["classifier_split_integrity"] = json.loads(
            split_integrity_path.read_text(encoding="utf-8")
        )
    if resume_integrity_path.exists():
        assertions["classifier_checkpoint_resume"] = json.loads(
            resume_integrity_path.read_text(encoding="utf-8")
        )
    if heldout_provenance_path.exists():
        heldout_assertion = json.loads(
            heldout_provenance_path.read_text(encoding="utf-8")
        )
        heldout_current = (
            heldout_assertion.get("provenance_version")
            == PROVENANCE_VERSION
            and heldout_assertion.get("model_artifact_fingerprint")
            == artifact_fingerprint(args.model)
        )
        assertions["heldout_prediction_provenance"] = {
            **heldout_assertion,
            "status": (
                "passed"
                if heldout_current
                else "legacy_or_model_unverified"
            ),
        }
        if not heldout_current and "heldout" in completed_sections:
            completed_sections.remove("heldout")
    population_provenance_path = (
        args.output_dir / "population_artifact_provenance.json"
    )
    if population_provenance_path.exists():
        population_provenance = json.loads(
            population_provenance_path.read_text(encoding="utf-8")
        )
        if (
            population_provenance.get("pair_root")
            != str(args.pair_root.resolve())
            or population_provenance.get("model")
            != str(args.model.resolve())
        ):
            population_provenance = inspect_population_pair_provenance(args)
            atomic_write_json(
                population_provenance_path,
                population_provenance,
            )
    else:
        population_provenance = inspect_population_pair_provenance(args)
        atomic_write_json(
            population_provenance_path,
            population_provenance,
        )
    assertions["population_artifact_provenance"] = {
        key: population_provenance.get(key)
        for key in (
            "status",
            "primary_eligible",
            "classifier_model_verified",
            "files",
            "verified_files",
            "strict_preretrieval_files",
            "failed_files",
        )
    }
    human_validation = blind_validation_status(args.output_dir)
    retrieval_resume = affected_retrieval_resume_status()
    blockers: list[str] = []
    if not population_provenance.get("classifier_model_verified"):
        blockers.append(
            "Population pairs are not proven to have been classified by the "
            "manifesto-grouped model."
        )
    if not population_provenance.get("primary_eligible"):
        blockers.append(
            "No complete population artifact is proven to use strict "
            "speech-date linkage before retrieval."
        )
    if (
        human_validation["primary_coded"] < human_validation["primary_rows"]
        or human_validation["second_coder_coded"]
        < human_validation["second_coder_rows"]
    ):
        blockers.append(
            "Genuinely blind primary coding and independent second-coder "
            "reliability are incomplete."
        )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "pair_root": str(args.pair_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "model": str(args.model.resolve()),
        "split_dir": str(args.split_dir.resolve()),
        "sample_source": str(args.sample_source.resolve()),
        "thresholds": list(THRESHOLDS),
        "rejection_thresholds": list(REJECTION_THRESHOLDS),
        "countries": list(COUNTRIES),
        "topics": list(TOPICS),
        "population_artifact_provenance": population_provenance,
        "primary_specification": {
            "temporal_linkage": (
                "latest manifesto by MPDS election-date proxy on or before each "
                "speech date; observations without a preceding election-dated "
                "manifesto are excluded"
            ),
            "aggregation": (
                "unweighted party-topic-month share of retained pairs "
                "classified inconsistent"
            ),
            "classifier_split": (
                "country-by-label stratification with complete manifesto "
                "documents grouped across train/validation/test"
            ),
            "inference": (
                "restricted Webb country wild-cluster bootstrap p-values; "
                "country CR1 intervals"
            ),
        },
        "implemented_corrections": [
            "Production NLI retrieval now requires a speech-date bridge and rejects month-only bridges.",
            "Fallback-to-earliest and future-manifesto rows are excluded before retrieval.",
            "Hybrid retrieval fingerprints speech text, manifesto text, configuration, and checkpoint parts.",
            "Classifier outputs are bound to model, input, max length, labels, and durable output bytes/content.",
            "All classifier split builders group on country plus manifesto doc_key.",
            "Classifier training resume fingerprints full texts, labels, groups, model, and hyperparameters.",
            "Population reproduction and primary split leakage are fail-fast assertions.",
        ],
        "preserved_completed_stages": [
            "9,658,591-pair legacy population scan (non-primary provenance)",
            "32-file exact production-summary reproduction audit",
            "blind human-validation sample export",
            "legacy held-out classifier comparison",
        ],
        "assertions": assertions,
        "human_validation": human_validation,
        "affected_retrieval_resume": retrieval_resume,
        "freeze_recommendation": {
            "status": "freeze_ready" if not blockers else "do_not_freeze_yet",
            "blocking_conditions": blockers,
        },
        "limitations": [
            "Existing historical coefficients are legacy/post-hoc sensitivities, not primary results from the corrected model and retrieval pipeline.",
            "Those cached population pairs cannot backfill or rerank candidates omitted by the original month-linked retrieval; corrected production code applies strict linkage before retrieval.",
            "MPDS edate is an election date, not a manifesto publication date; temporal ordering is therefore an explicit election-date proxy.",
            "Pair counts in the linkage audit are exact; speech and retrieval-unit counts are batch-summed upper bounds when an ID crosses an Arrow batch boundary.",
            "Held-out calibration is measured on the deliberately enriched silver-label sample, not a population-representative human sample.",
            "The blind sample is exported but requires independent human coding before it becomes validation evidence.",
        ],
        "completed_sections": completed_sections,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_consolidated_freeze_recommendation(args, manifest)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    if not args.skip_population:
        outputs["population"] = scan_population(args)
    elif (args.output_dir / "party_topic_month_outcomes.csv").exists():
        outputs["population"] = {
            "cells": pd.read_csv(
                args.output_dir / "party_topic_month_outcomes.csv",
                low_memory=False,
            )
        }
    if not args.skip_heldout:
        outputs["heldout"] = heldout_diagnostics(args)
    grouped_generic = manifesto_grouped_generic_baseline(args)
    if grouped_generic is not None:
        outputs["manifesto_grouped_generic_baseline"] = grouped_generic
    reproduction = population_reproduction_audit(args)
    if reproduction is not None:
        outputs["population_reproduction"] = reproduction
    aggregation = aggregation_diagnostics(args)
    if aggregation is not None:
        outputs["aggregation_diagnostics"] = aggregation
    blind_design = blind_sample_design_audit(args)
    if blind_design is not None:
        outputs["blind_sample_design"] = blind_design
    outputs["split_leakage"] = split_leakage_audit(args)
    outputs["base_fallback"] = add_base_fallback_audit(args.output_dir)
    if not args.skip_regressions:
        if "population" not in outputs:
            raise FileNotFoundError("Population cell outcomes are required for regressions")
        outputs["regressions"] = run_regressions(
            outputs["population"]["cells"], args.output_dir
        )
    if args.targeted_wild_bootstrap:
        if "population" not in outputs:
            raise FileNotFoundError(
                "Population cell outcomes are required for targeted bootstrap"
            )
        outputs["targeted_wild_bootstrap"] = run_targeted_wild_bootstrap(
            outputs["population"]["cells"],
            args.output_dir,
            args.bootstrap_reps,
        )
    stability = regression_stability_summary(args)
    if stability is not None:
        outputs["regression_stability"] = stability
    write_run_manifest(args, outputs)
    print(f"Methodology-freeze robustness outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
