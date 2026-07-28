from __future__ import annotations

"""Prepare a blinded, Macroeconomics-only human-validation packet.

The script draws a uniform reservoir sample within each country × predicted
class stratum directly from the completed frozen Macroeconomics pair files. It
does not read methodology-freeze outputs, human labels, GAL--TAN rows, or
robustness-regression results.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = (
    BASE_DIR / "outputs" / "test_speeches" / "nli_consensus_classifier"
)
DEFAULT_OUTPUT = (
    BASE_DIR / "outputs" / "test_speeches" / "macro_human_validation"
)
COUNTRIES = (
    "AT",
    "BE",
    "CZ",
    "DK",
    "EE",
    "ES",
    "FI",
    "GB",
    "GR",
    "IT",
    "LV",
    "NL",
    "NO",
    "PL",
    "PT",
    "SE",
)
LABELS = ("consistent", "unrelated", "inconsistent")
SAMPLE_PER_STRATUM = 4


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--seed", type=int, default=260718)
    return out


def population_counts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for country in COUNTRIES:
        path = (
            CLASSIFIER_DIR
            / country
            / f"{country}_Macroeconomics_consensus_deberta_summary_overall.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        summary = pd.read_csv(path)
        if len(summary) != 1:
            raise ValueError(f"Expected one summary row in {path}")
        row = summary.iloc[0]
        for label in LABELS:
            rows.append(
                {
                    "country_code": country,
                    "classifier_label": label,
                    "population_stratum_n": int(
                        row[f"classifier_count_{label}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def pair_path(country: str) -> Path:
    return (
        CLASSIFIER_DIR
        / country
        / f"{country}_Macroeconomics_consensus_deberta_pairs.csv"
    )


def sample_country(country: str, seed: int) -> list[dict[str, object]]:
    """Uniform reservoir sample within the three predicted-class strata.

    The completed CSVs contain one physical record per line. Splitting only the
    six final, numeric/classifier fields avoids parsing tens of gigabytes of
    repeated text while retaining exact CSV parsing for the selected records.
    """

    path = pair_path(country)
    if not path.exists():
        raise FileNotFoundError(path)
    reservoirs: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: dict[str, int] = {label: 0 for label in LABELS}
    rng = random.Random(seed + int.from_bytes(country.encode("ascii"), "big"))
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if "\n" not in header:
            raise ValueError(f"Malformed header in {path}")
        for line_number, line in enumerate(handle, start=2):
            tail = line.rstrip("\r\n").rsplit(",", 6)
            if len(tail) != 7:
                raise ValueError(
                    f"{path}:{line_number} does not have the expected classifier tail."
                )
            label = tail[4].strip().lower()
            if label not in reservoirs:
                raise ValueError(
                    f"{path}:{line_number} has unexpected classifier label {label!r}."
                )
            seen[label] += 1
            if len(reservoirs[label]) < SAMPLE_PER_STRATUM:
                reservoirs[label].append(line)
            else:
                draw = rng.randrange(seen[label])
                if draw < SAMPLE_PER_STRATUM:
                    reservoirs[label][draw] = line

    rows: list[dict[str, object]] = []
    for label in LABELS:
        if len(reservoirs[label]) != SAMPLE_PER_STRATUM:
            raise ValueError(
                f"{country}/{label}: expected {SAMPLE_PER_STRATUM} rows, "
                f"found {len(reservoirs[label])}."
            )
        for line in reservoirs[label]:
            reader = csv.DictReader(io.StringIO(header + line))
            row = next(reader)
            if str(row["classifier_label"]).strip().lower() != label:
                raise ValueError("Selected row label changed during CSV parsing.")
            rows.append(row)
    return rows


def sample_completed_pairs(seed: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for country in COUNTRIES:
        records.extend(sample_country(country, seed))
        print(f"Sampled {country}: {len(records):,}/192 rows", flush=True)
    sample = pd.DataFrame(records)
    speech_column = next(
        (
            column
            for column in (
                "speech_text_for_nli",
                "speech_segment_text",
                "speech_text",
                "speech_text_original",
            )
            if column in sample
        ),
        None,
    )
    if speech_column is None:
        raise ValueError("No speech-text column is available in completed pair files.")
    sample["country_code"] = sample["country"].astype(str).str.upper()
    sample["speech_party"] = sample.get("speech_party", sample.get("party", ""))
    sample["speech_text"] = sample[speech_column]
    sample["nli_topic"] = "Macroeconomics"
    sample["retrieval_rank"] = pd.to_numeric(
        sample["retrieval_rank"], errors="coerce"
    )
    sample["embedding_score"] = pd.to_numeric(
        sample["embedding_score"], errors="coerce"
    )
    sample["rank_band"] = np.where(
        sample["retrieval_rank"].eq(1), "rank_1", "rank_2_3"
    )
    sample["similarity_band"] = pd.cut(
        sample["embedding_score"],
        bins=[-np.inf, 0.4, 0.5, np.inf],
        labels=["lt_0.40", "0.40_to_0.50", "ge_0.50"],
        right=False,
    ).astype(str)
    sample["_blind_key"] = (
        sample["country_code"]
        + "|Macroeconomics|"
        + sample["nli_pair_id"].astype(str)
    )
    return sample


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = args.output_dir / "BLINDED_PRIMARY_CODING.csv"
    if existing.exists():
        existing_frame = pd.read_csv(existing, keep_default_na=False)
        if (
            "human_label" in existing_frame
            and existing_frame["human_label"].astype(str).str.strip().ne("").any()
        ):
            raise ValueError(
                f"{existing} contains human judgments; refusing to overwrite them."
            )

    key = sample_completed_pairs(args.seed)
    if len(key) != 192:
        raise ValueError(f"Expected 192 sampled rows, found {len(key)}.")
    rng = np.random.default_rng(args.seed)
    key["_order"] = rng.permutation(len(key))
    key = key.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    key.insert(
        0,
        "validation_id",
        [f"MHV{index:03d}" for index in range(1, len(key) + 1)],
    )
    blind = key[
        ["validation_id", "country_code", "manifesto_text", "speech_text"]
    ].copy()
    blind["human_label"] = ""
    blind["human_comparable"] = ""
    blind["human_confidence"] = ""
    blind["human_notes"] = ""
    key["classifier_label"] = (
        key["classifier_label"].astype(str).str.strip().str.lower()
    )
    key = key.merge(
        population_counts(),
        on=["country_code", "classifier_label"],
        how="left",
        validate="m:1",
    )
    sample_counts = (
        key.groupby(["country_code", "classifier_label"], as_index=False)
        .size()
        .rename(columns={"size": "sample_stratum_n"})
    )
    key = key.merge(
        sample_counts,
        on=["country_code", "classifier_label"],
        how="left",
        validate="m:1",
    )
    key["poststrat_weight"] = (
        key["population_stratum_n"] / key["sample_stratum_n"]
    )
    key = key.sort_values("validation_id").reset_index(drop=True)

    second_ids = (
        key.sort_values(["country_code", "classifier_label", "validation_id"])
        .groupby(["country_code", "classifier_label"], as_index=False)
        .head(1)["validation_id"]
    )
    second = blind.loc[blind["validation_id"].isin(set(second_ids))].copy()
    second = second.rename(
        columns={
            "human_label": "second_coder_label",
            "human_comparable": "second_coder_comparable",
            "human_confidence": "second_coder_confidence",
            "human_notes": "second_coder_notes",
        }
    )

    blind.to_csv(args.output_dir / "BLINDED_PRIMARY_CODING.csv", index=False)
    second.to_csv(args.output_dir / "BLINDED_SECOND_CODER.csv", index=False)
    key.to_csv(args.output_dir / "PRIVATE_CLASSIFIER_KEY.csv", index=False)

    design_rows: list[dict[str, object]] = []
    for sample_name, frame in (("primary", key), ("second_coder", key.loc[key["validation_id"].isin(set(second_ids))])):
        design_rows.append(
            {
                "sample": sample_name,
                "dimension": "overall",
                "group": "all",
                "rows": len(frame),
            }
        )
        for dimension in ("country_code", "classifier_label"):
            for group, count in frame[dimension].value_counts().sort_index().items():
                design_rows.append(
                    {
                        "sample": sample_name,
                        "dimension": dimension,
                        "group": group,
                        "rows": int(count),
                    }
                )
    pd.DataFrame(design_rows).to_csv(
        args.output_dir / "DESIGN_AUDIT.csv", index=False
    )

    codebook = """# Blinded Macroeconomics validation codebook

## Blinding

Code `BLINDED_PRIMARY_CODING.csv` without opening `PRIVATE_CLASSIFIER_KEY.csv`,
classifier outputs, or regression results. Give the second coder only
`BLINDED_SECOND_CODER.csv`. Do not discuss individual labels until both files
are locked.

Read the manifesto text as an earlier party statement and the speech text as a
later party statement. Judge only the two supplied texts; do not research the
party or infer its likely position.

## Required coding sequence

1. Set `human_comparable` to `yes` only if both texts address the same
   substantive proposition closely enough for a directional consistency
   judgment. Otherwise set it to `no`.
2. Assign one `human_label`:
   - `consistent`: comparable statements that support, repeat, or are mutually
     compatible on the substantive proposition.
   - `inconsistent`: comparable statements that advocate incompatible
     positions or directions on the substantive proposition.
   - `unrelated`: the statements are not sufficiently comparable.
   - `ambiguous`: the wording is genuinely insufficient for a defensible
     judgment even after applying the comparability rule.
3. Set confidence to `low`, `medium`, or `high`; briefly explain difficult
   decisions in the notes field.

Do not code a contradiction merely because the texts concern different actors,
time periods, jurisdictions, levels of government, or degrees of specificity.
Differences in emphasis are not conflicts unless the advocated positions are
substantively incompatible. Translation tools may be used, but the same tool
and procedure should be used consistently and noted in the validation report.

After coding is locked, run:

```bash
.venv/bin/python scripts/inconsistency/analyze_macro_human_validation.py
```
"""
    (args.output_dir / "CODING_CODEBOOK.md").write_text(
        codebook, encoding="utf-8"
    )

    manifest = {
        "status": "packet_prepared_human_coding_pending",
        "topic": "Macroeconomics",
        "primary_rows": int(len(blind)),
        "second_coder_rows": int(len(second)),
        "countries": list(COUNTRIES),
        "classifier_labels_hidden": list(LABELS),
        "rows_per_country_classifier_cell": 4,
        "second_rows_per_country_classifier_cell": 1,
        "shuffle_seed": args.seed,
        "source_pair_outputs": str(CLASSIFIER_DIR.relative_to(BASE_DIR)),
        "sampling_method": (
            "uniform reservoir sample of four records within every "
            "country-by-predicted-class stratum"
        ),
        "methodology_freeze_experimental_results_used": False,
        "human_judgments_supplied": False,
        "notes": (
            "Rows are freshly sampled from completed frozen Macroeconomics pair "
            "outputs. GAL--TAN rows and every methodology-freeze artifact are "
            "excluded. Post-stratification weights restore the completed pair "
            "population distribution across the sampled country-class strata."
        ),
    }
    (args.output_dir / "PACKET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {len(blind)} blinded primary rows and {len(second)} "
        f"second-coder rows in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
