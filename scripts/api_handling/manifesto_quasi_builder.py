from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MPDS_PATH = REPO_ROOT / "data" / "MPDataset_MPDS2025a.csv"
DEFAULT_PARTY_MAPPING_DIR = SCRIPTS_DIR / "party_mappings"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "manifesto_quasi_sentences"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from api_client import ManifestoApiClient
from path_config import get_parlam_csv_path


load_dotenv(REPO_ROOT / ".env")
MPDS_KEY = "MPDS2025a"
CORPUS_VER = "2025-1"

COUNTRY_NAME_BY_CODE: dict[str, str] = {
    "CZ": "Czech Republic",
    "EE": "Estonia",
    "GB": "United Kingdom",
    "HU": "Hungary",
    "LV": "Latvia",
    "LT": "Lithuania",
    "PL": "Poland",
    "SI": "Slovenia",
    "UA": "Ukraine",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build manifesto quasi-sentence datasets only for mapped parties that "
            "actually occur in the parliamentary speech dataset."
        )
    )
    parser.add_argument(
        "--country",
        required=True,
        type=str,
        help="Country code, for example: CZ GB EE LV LT UA HU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for output CSV files.",
    )
    parser.add_argument(
        "--party-mapping-dir",
        type=Path,
        default=DEFAULT_PARTY_MAPPING_DIR,
        help=(
            "Directory containing <COUNTRY>_party_mapping_speech_to_mpds.csv files. "
            "Defaults to scripts/party_mappings under the repository root."
        ),
    )
    parser.add_argument(
        "--translation",
        default=None,
        type=str,
        help="Optional manifesto translation passed to the API, for example 'en'.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Build only the local bridge tables and skip API manifesto downloads.",
    )
    return parser


def create_manifesto_client() -> ManifestoApiClient:
    manifesto_base = os.getenv("BASE_URL_MANIFESTO")
    manifesto_key = os.getenv("MANIFESTO_PROJECT_API_KEY")

    if not manifesto_base or not manifesto_key:
        raise RuntimeError("Missing manifesto API configuration in .env")

    return ManifestoApiClient(
        manifesto_base,
        manifesto_key,
        MPDS_KEY,
        CORPUS_VER,
    )


def load_speech_rows(country_code: str) -> pd.DataFrame:
    path = get_parlam_csv_path(country_code)
    if not path.exists():
        raise FileNotFoundError(f"Could not find ParlaMint file: {path}")

    df = pd.read_csv(
        path,
        usecols=["date", "party", "content_kind", "speaker_type", "text"],
        low_memory=False,
    )
    df = df[df["content_kind"] == "speech"].copy()
    df = df[df["speaker_type"] == "regular"].copy()
    df = df.dropna(subset=["date", "party", "text"]).copy()

    df["party"] = df["party"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["party"] != ""].copy()
    df = df[df["text"] != ""].copy()
    df = df.dropna(subset=["date"]).copy()

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True)


def load_party_mapping(country_code: str, party_mapping_dir: Path) -> pd.DataFrame:
    path = party_mapping_dir / f"{country_code.upper()}_party_mapping_speech_to_mpds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing party mapping file: {path}")

    mapping_df = pd.read_csv(path, low_memory=False)
    mapping_df["speech_party"] = mapping_df["speech_party"].astype(str).str.strip()
    return mapping_df


def build_eligible_party_index(
    speeches_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    party_stats = (
        speeches_df.groupby("party", as_index=False)
        .agg(
            speech_rows=("party", "size"),
            speech_start_date=("date", "min"),
            speech_end_date=("date", "max"),
        )
        .rename(columns={"party": "speech_party"})
    )

    mapped_df = mapping_df.loc[mapping_df["mapping_status"] == "mapped"].copy()
    eligible_df = party_stats.merge(mapped_df, on="speech_party", how="inner")
    eligible_df["mpds_party_id"] = eligible_df["mpds_party_id"].astype("Int64")
    eligible_df = eligible_df.dropna(subset=["mpds_party_id"]).copy()
    eligible_df["mpds_party_id"] = eligible_df["mpds_party_id"].astype(int)
    return eligible_df.sort_values(["speech_party"]).reset_index(drop=True)


def load_mpds_manifestos(country_code: str, party_ids: list[int]) -> pd.DataFrame:
    country_name = COUNTRY_NAME_BY_CODE[country_code]
    mpds = pd.read_csv(
        MPDS_PATH,
        usecols=["countryname", "party", "partyname", "partyabbrev", "date", "edate"],
        low_memory=False,
    )
    manifestos_df = mpds.loc[
        (mpds["countryname"] == country_name) & (mpds["party"].isin(party_ids))
    ].copy()
    manifestos_df["mpds_party_id"] = manifestos_df["party"].astype(int)
    manifestos_df["manifesto_date"] = manifestos_df["date"].astype(int)
    manifestos_df["manifesto_month_start"] = pd.to_datetime(
        manifestos_df["manifesto_date"].astype(str),
        format="%Y%m",
        errors="coerce",
    )
    manifestos_df["manifesto_effective_date"] = pd.to_datetime(
        manifestos_df["edate"],
        format="%d/%m/%Y",
        errors="coerce",
        dayfirst=True,
    )
    missing_effective_mask = manifestos_df["manifesto_effective_date"].isna()
    manifestos_df.loc[missing_effective_mask, "manifesto_effective_date"] = manifestos_df.loc[
        missing_effective_mask, "manifesto_month_start"
    ]
    manifestos_df["doc_key"] = (
        manifestos_df["mpds_party_id"].astype(str)
        + "_"
        + manifestos_df["manifesto_date"].astype(str)
    )
    manifestos_df = manifestos_df.dropna(
        subset=["manifesto_month_start", "manifesto_effective_date"]
    ).copy()
    return (
        manifestos_df[
            [
                "countryname",
                "mpds_party_id",
                "partyname",
                "partyabbrev",
                "manifesto_date",
                "edate",
                "manifesto_month_start",
                "manifesto_effective_date",
                "doc_key",
            ]
        ]
        .drop_duplicates()
        .sort_values(["mpds_party_id", "manifesto_effective_date", "manifesto_date"])
        .reset_index(drop=True)
    )


def build_speech_date_index(
    speeches_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
) -> pd.DataFrame:
    speech_dates_df = (
        speeches_df.groupby(["party", "date", "month", "month_start"], as_index=False)
        .agg(speech_rows=("party", "size"))
        .rename(columns={"party": "speech_party"})
    )
    joined_df = speech_dates_df.merge(
        eligible_df[
            [
                "speech_party",
                "speech_label_type",
                "mpds_party_id",
                "mpds_partyname",
                "mpds_partyabbrev",
            ]
        ],
        on="speech_party",
        how="inner",
    )
    return joined_df.sort_values(["mpds_party_id", "date", "speech_party"]).reset_index(
        drop=True
    )


def build_temporal_manifesto_bridge(
    speech_dates_df: pd.DataFrame,
    manifestos_df: pd.DataFrame,
) -> pd.DataFrame:
    bridge_frames: list[pd.DataFrame] = []

    for party_id, speech_group in speech_dates_df.groupby("mpds_party_id", sort=True):
        manifesto_group = (
            manifestos_df.loc[manifestos_df["mpds_party_id"] == party_id]
            .sort_values("manifesto_effective_date")
            .reset_index(drop=True)
        )
        if manifesto_group.empty:
            continue

        manifesto_group = manifesto_group[
            [
                "countryname",
                "partyname",
                "partyabbrev",
                "manifesto_date",
                "edate",
                "manifesto_month_start",
                "manifesto_effective_date",
                "doc_key",
            ]
        ].copy()

        merged = pd.merge_asof(
            speech_group.sort_values("date").reset_index(drop=True),
            manifesto_group,
            left_on="date",
            right_on="manifesto_effective_date",
            direction="backward",
        )

        missing_mask = merged["doc_key"].isna()
        if missing_mask.any():
            earliest = manifesto_group.iloc[0]
            for column in [
                "countryname",
                "partyname",
                "partyabbrev",
                "manifesto_date",
                "edate",
                "manifesto_month_start",
                "manifesto_effective_date",
                "doc_key",
            ]:
                merged.loc[missing_mask, column] = earliest[column]

        merged["selection_method"] = "latest_manifesto_on_or_before_speech_date"
        merged.loc[missing_mask, "selection_method"] = "fallback_to_earliest_manifesto"
        bridge_frames.append(merged)

    if not bridge_frames:
        return pd.DataFrame()

    bridge_df = pd.concat(bridge_frames, ignore_index=True)
    bridge_df = bridge_df.rename(
        columns={
            "countryname": "manifesto_countryname",
            "partyname": "manifesto_partyname",
            "partyabbrev": "manifesto_partyabbrev",
        }
    )
    return bridge_df.sort_values(["date", "speech_party"]).reset_index(drop=True)


def build_month_bridge_summary(date_bridge_df: pd.DataFrame) -> pd.DataFrame:
    if date_bridge_df.empty:
        return pd.DataFrame()

    return (
        date_bridge_df.groupby(
            [
                "speech_party",
                "month",
                "month_start",
                "speech_label_type",
                "mpds_party_id",
                "mpds_partyname",
                "mpds_partyabbrev",
                "manifesto_countryname",
                "manifesto_partyname",
                "manifesto_partyabbrev",
                "manifesto_date",
                "edate",
                "manifesto_month_start",
                "manifesto_effective_date",
                "doc_key",
                "selection_method",
            ],
            as_index=False,
        )
        .agg(
            speech_rows=("speech_rows", "sum"),
            speech_dates=("date", "nunique"),
            speech_start_date=("date", "min"),
            speech_end_date=("date", "max"),
        )
        .sort_values(["month_start", "speech_party", "manifesto_effective_date", "doc_key"])
        .reset_index(drop=True)
    )


def build_selected_manifesto_documents(bridge_df: pd.DataFrame) -> pd.DataFrame:
    if bridge_df.empty:
        return pd.DataFrame(
            columns=[
                "doc_key",
                "mpds_party_id",
                "manifesto_partyname",
                "manifesto_partyabbrev",
                "manifesto_date",
                "manifesto_month_start",
                "edate",
                "linked_speech_parties",
                "linked_party_months",
            ]
        )

    party_counts = (
        bridge_df.groupby("doc_key", as_index=False)
        .agg(
            linked_speech_parties=("speech_party", "nunique"),
            linked_party_months=("month", "size"),
        )
    )
    documents_df = (
        bridge_df[
            [
                "doc_key",
                "mpds_party_id",
                "manifesto_partyname",
                "manifesto_partyabbrev",
                "manifesto_date",
                "manifesto_month_start",
                "edate",
            ]
        ]
        .drop_duplicates()
        .merge(party_counts, on="doc_key", how="left")
        .sort_values(["mpds_party_id", "manifesto_date"])
        .reset_index(drop=True)
    )
    return documents_df


def fetch_manifesto_documents(
    documents_df: pd.DataFrame,
    translation: str | None,
    client: ManifestoApiClient,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    document_rows: list[dict[str, Any]] = []
    quasi_rows: list[dict[str, Any]] = []

    for row in documents_df.itertuples(index=False):
        try:
            payload = client.get_text_for_party_date(
                int(row.mpds_party_id),
                int(row.manifesto_date),
                translation=translation,
            )
            metadata_item = payload["metadata_item"]
            text_item = payload["text_item"]
            items = text_item.get("items", [])

            document_rows.append(
                {
                    "doc_key": row.doc_key,
                    "mpds_party_id": int(row.mpds_party_id),
                    "manifesto_partyname": row.manifesto_partyname,
                    "manifesto_partyabbrev": row.manifesto_partyabbrev,
                    "manifesto_date": int(row.manifesto_date),
                    "manifesto_month_start": row.manifesto_month_start,
                    "edate": row.edate,
                    "manifesto_id": payload["manifesto_id"],
                    "language": metadata_item.get("language"),
                    "title": metadata_item.get("title"),
                    "source": metadata_item.get("source"),
                    "translation": translation or "",
                    "annotations": metadata_item.get("annotations"),
                    "n_quasi_sentences": len(items),
                    "linked_speech_parties": int(row.linked_speech_parties),
                    "linked_party_months": int(row.linked_party_months),
                    "fetch_status": "ok",
                    "fetch_error": "",
                }
            )

            for idx, item in enumerate(items):
                quasi_rows.append(
                    {
                        "doc_key": row.doc_key,
                        "manifesto_id": payload["manifesto_id"],
                        "mpds_party_id": int(row.mpds_party_id),
                        "manifesto_date": int(row.manifesto_date),
                        "quasi_sentence_id": idx,
                        "cmp_code": item.get("cmp_code"),
                        "eu_code": item.get("eu_code"),
                        "text": item.get("text", ""),
                        "translation": translation or "",
                    }
                )
        except Exception as exc:  # pragma: no cover - network/API dependent
            document_rows.append(
                {
                    "doc_key": row.doc_key,
                    "mpds_party_id": int(row.mpds_party_id),
                    "manifesto_partyname": row.manifesto_partyname,
                    "manifesto_partyabbrev": row.manifesto_partyabbrev,
                    "manifesto_date": int(row.manifesto_date),
                    "manifesto_month_start": row.manifesto_month_start,
                    "edate": row.edate,
                    "manifesto_id": "",
                    "language": "",
                    "title": "",
                    "source": "",
                    "translation": translation or "",
                    "annotations": "",
                    "n_quasi_sentences": 0,
                    "linked_speech_parties": int(row.linked_speech_parties),
                    "linked_party_months": int(row.linked_party_months),
                    "fetch_status": "error",
                    "fetch_error": str(exc),
                }
            )

    return pd.DataFrame(document_rows), pd.DataFrame(quasi_rows)


def output_paths(output_dir: Path, country_code: str) -> dict[str, Path]:
    country_dir = output_dir / country_code.upper()
    country_dir.mkdir(parents=True, exist_ok=True)
    prefix = country_code.upper()
    return {
        "eligible_parties": country_dir / f"{prefix}_eligible_mapped_speech_parties.csv",
        "date_bridge": country_dir / f"{prefix}_speech_date_to_manifesto_bridge.csv",
        "bridge": country_dir / f"{prefix}_speech_month_to_manifesto_bridge.csv",
        "documents": country_dir / f"{prefix}_manifesto_documents.csv",
        "quasi_sentences": country_dir / f"{prefix}_manifesto_quasi_sentences.csv",
    }


def run(
    country_code: str,
    output_dir: Path,
    party_mapping_dir: Path,
    translation: str | None,
    skip_download: bool,
) -> dict[str, Any]:
    normalized_code = country_code.strip().upper()
    if normalized_code not in COUNTRY_NAME_BY_CODE:
        raise ValueError(
            f"Unsupported country code {country_code!r}. Expected one of: "
            f"{', '.join(sorted(COUNTRY_NAME_BY_CODE))}"
        )

    speeches_df = load_speech_rows(normalized_code)
    mapping_df = load_party_mapping(normalized_code, party_mapping_dir)
    eligible_df = build_eligible_party_index(speeches_df, mapping_df)
    if eligible_df.empty:
        raise RuntimeError(f"No mapped speech parties found in the {normalized_code} dataset.")

    manifestos_df = load_mpds_manifestos(
        normalized_code,
        sorted(eligible_df["mpds_party_id"].unique().tolist()),
    )
    speech_dates_df = build_speech_date_index(speeches_df, eligible_df)
    date_bridge_df = build_temporal_manifesto_bridge(speech_dates_df, manifestos_df)
    bridge_df = build_month_bridge_summary(date_bridge_df)
    documents_df = build_selected_manifesto_documents(date_bridge_df)

    paths = output_paths(output_dir, normalized_code)
    eligible_df.to_csv(paths["eligible_parties"], index=False)
    date_bridge_df.to_csv(paths["date_bridge"], index=False)
    bridge_df.to_csv(paths["bridge"], index=False)

    if skip_download:
        documents_df.assign(
            manifesto_id="",
            language="",
            title="",
            source="",
            translation=translation or "",
            annotations="",
            n_quasi_sentences=0,
            fetch_status="skipped",
            fetch_error="",
        ).to_csv(paths["documents"], index=False)
        pd.DataFrame(
            columns=[
                "doc_key",
                "manifesto_id",
                "mpds_party_id",
                "manifesto_date",
                "quasi_sentence_id",
                "cmp_code",
                "eu_code",
                "text",
                "translation",
            ]
        ).to_csv(paths["quasi_sentences"], index=False)
        return {
            "country_code": normalized_code,
            "speech_rows": int(len(speeches_df)),
            "eligible_parties": int(len(eligible_df)),
            "speech_dates": int(len(date_bridge_df)),
            "speech_months": int(len(bridge_df)),
            "selected_manifestos": int(len(documents_df)),
            "quasi_sentences": 0,
            "paths": paths,
            "download_status": "skipped",
        }

    client = create_manifesto_client()
    fetched_documents_df, quasi_sentences_df = fetch_manifesto_documents(
        documents_df,
        translation=translation,
        client=client,
    )
    successful_docs = int((fetched_documents_df["fetch_status"] == "ok").sum())
    failed_docs = int((fetched_documents_df["fetch_status"] == "error").sum())
    if failed_docs == 0:
        download_status = "ok"
    elif successful_docs == 0:
        download_status = "error"
    else:
        download_status = "partial"
    fetched_documents_df.to_csv(paths["documents"], index=False)
    quasi_sentences_df.to_csv(paths["quasi_sentences"], index=False)

    return {
        "country_code": normalized_code,
        "speech_rows": int(len(speeches_df)),
        "eligible_parties": int(len(eligible_df)),
        "speech_dates": int(len(date_bridge_df)),
        "speech_months": int(len(bridge_df)),
        "selected_manifestos": int(len(documents_df)),
        "quasi_sentences": int(len(quasi_sentences_df)),
        "successful_manifestos": successful_docs,
        "failed_manifestos": failed_docs,
        "paths": paths,
        "download_status": download_status,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        country_code=args.country,
        output_dir=args.output_dir,
        party_mapping_dir=args.party_mapping_dir,
        translation=args.translation,
        skip_download=args.skip_download,
    )

    print(f"[{result['country_code']}] eligible mapped speech parties: {result['eligible_parties']}")
    print(f"[{result['country_code']}] speech-party-date rows: {result['speech_dates']}")
    print(f"[{result['country_code']}] speech-party-month rows: {result['speech_months']}")
    print(f"[{result['country_code']}] selected manifesto documents: {result['selected_manifestos']}")
    print(f"[{result['country_code']}] manifesto quasi sentences: {result['quasi_sentences']}")
    print(f"[{result['country_code']}] download status: {result['download_status']}")
    if "successful_manifestos" in result and "failed_manifestos" in result:
        print(
            f"[{result['country_code']}] manifesto fetches ok={result['successful_manifestos']}, "
            f"error={result['failed_manifestos']}"
        )
    for label, path in result["paths"].items():
        print(f"[{result['country_code']}] {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
