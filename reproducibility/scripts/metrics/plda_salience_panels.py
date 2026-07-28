from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_PANEL_DIR = TEST_OUTPUT_DIR / "plda_regression_panel"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_salience" / "substantive"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from causality.plda_linear_regression import coerce_bool_series
from vizualization.thesis_plda_cluster_outputs import (
    HYBRID_TOPIC_MAP,
    TOPIC_ORDER,
    base_topic_label,
    load_distribution,
    topic_mapping,
)


CLUSTERS = {
    "cee": ["CZ", "EE", "LV", "PL"],
    "nordics": ["DK", "FI", "NO", "SE"],
    "west": ["AT", "BE", "NL"],
    "south": ["ES", "IT", "PT", "GR"],
}
DEFAULT_COUNTRIES = [country for countries in CLUSTERS.values() for country in countries]
TOPIC_COLUMN_RE = re.compile(r"^topic_(\d+)_(speech|manifesto)$")
PRIOR_MANIFESTO_METHOD = "latest_manifesto_on_or_before_speech_date"
METADATA_COLUMNS = [
    "speech_party",
    "month",
    "month_start",
    "analysis_date",
    "doc_key",
    "manifesto_date",
    "manifesto_effective_date",
    "selection_method",
    "speech_topic_weighting",
    "speech_rows",
    "speech_dates",
    "speech_start_date",
    "speech_end_date",
    "speech_volume_segments",
    "speech_volume_words",
    "log1p_speech_segments",
    "log1p_speech_words",
    "party_in_government",
    "party_prime_minister",
    "party_seat_share",
    "party_vote_share",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
    "electoral_cycle_progress",
    "cycle_boundary_source",
    "last_election_early",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate country-specific PLDA components to shared substantive topics, "
            "then build topic-level salience and party-month Spearman panels."
        )
    )
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--panel-override",
        action="append",
        default=[],
        metavar="COUNTRY=PATH",
        help="Use an explicit regression-panel CSV for one country; repeat as needed.",
    )
    parser.add_argument(
        "--include-residual",
        action="store_true",
        help="Include the Residual family, yielding nine rather than eight topics.",
    )
    return parser


def normalize_countries(raw_countries: list[str]) -> list[str]:
    countries: list[str] = []
    for raw in raw_countries:
        for part in raw.split(","):
            country = part.strip().upper()
            if country and country not in countries:
                countries.append(country)
    if not countries:
        raise ValueError("No countries were provided.")
    return countries


def parse_panel_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --panel-override {value!r}; expected COUNTRY=PATH.")
        country, raw_path = value.split("=", 1)
        country = country.strip().upper()
        path = Path(raw_path.strip())
        if not country or not raw_path.strip():
            raise ValueError(f"Invalid --panel-override {value!r}; expected COUNTRY=PATH.")
        overrides[country] = path
    return overrides


def panel_path(panel_dir: Path, country: str) -> Path:
    return panel_dir / country / f"{country}_plda_regression_panel.csv"


def topic_bases(columns: pd.Index, suffix: str) -> list[str]:
    bases: list[str] = []
    for column in columns:
        match = TOPIC_COLUMN_RE.fullmatch(str(column))
        if match and match.group(2) == suffix:
            bases.append(f"topic_{match.group(1)}")
    return sorted(bases, key=lambda value: int(value.split("_")[1]))


def current_grid_topic_count(
    country: str, observed_topic_count: int | None = None
) -> float:
    if observed_topic_count is not None:
        return float(observed_topic_count)
    path = TEST_OUTPUT_DIR / f"plda_grid_search_log_{country}.csv"
    if not path.exists():
        return math.nan
    grid = pd.read_csv(path)
    if grid.empty:
        return math.nan
    if "is_best" in grid.columns:
        mask = grid["is_best"]
        if mask.dtype == object:
            mask = mask.astype(str).str.lower().eq("true")
        row = grid.loc[mask].iloc[0] if mask.any() else grid.iloc[0]
    else:
        row = grid.iloc[0]
    latent = int(pd.to_numeric(row.get("n_latent_topics", 0), errors="coerce") or 0)
    per_label = int(pd.to_numeric(row.get("topics_per_label", 1), errors="coerce") or 1)
    label_count = 0
    individual_path = TEST_OUTPUT_DIR / f"plda_individual_speeches_{country}.csv"
    if individual_path.exists():
        labels = pd.read_csv(individual_path, usecols=["topic_label"], low_memory=False)
        label_count = labels["topic_label"].dropna().astype(str).str.strip().nunique()
    if not label_count:
        return math.nan
    return float(latent + per_label * label_count)


def fitted_model_topic_mapping(
    country: str, data: pd.DataFrame
) -> dict[str, str] | None:
    """Use labels emitted by the retained PLDA distribution stage."""
    labels_path = (
        TEST_OUTPUT_DIR
        / "plda_topic_distributions"
        / country
        / f"{country}_plda_topic_labels.csv"
    )
    if not labels_path.exists():
        return None
    topics = pd.read_csv(labels_path, low_memory=False)
    required = {"topic_column", "topic_label"}
    if not required.issubset(topics.columns):
        return None
    speech_bases = topic_bases(data.columns, "speech")
    metadata_columns = topics["topic_column"].astype(str).tolist()
    if metadata_columns != speech_bases[: len(metadata_columns)]:
        return None
    mapping: dict[str, str] = {}
    for topic in topics.itertuples(index=False):
        label = base_topic_label(str(topic.topic_label))
        mapping[str(topic.topic_column)] = HYBRID_TOPIC_MAP.get(label, label)
    return mapping


def validate_mapping(
    country: str,
    data: pd.DataFrame,
    mapping: dict[str, str],
) -> tuple[list[str], list[str]]:
    speech_bases = topic_bases(data.columns, "speech")
    manifesto_bases = topic_bases(data.columns, "manifesto")
    if not speech_bases or speech_bases != manifesto_bases:
        raise ValueError(
            f"{country}: speech and manifesto topic layouts differ: "
            f"speech={len(speech_bases)}, manifesto={len(manifesto_bases)}."
        )
    extra = sorted(set(mapping) - set(speech_bases))
    unmapped = sorted(
        set(speech_bases) - set(mapping),
        key=lambda value: int(value.split("_")[1]),
    )
    if extra:
        raise ValueError(f"{country}: topic mapping contains unavailable columns: {extra}")
    if len(unmapped) != 1 or unmapped[0] != speech_bases[-1]:
        raise ValueError(
            f"{country}: expected exactly the final shared-latent topic to be unmapped; "
            f"found {unmapped}."
        )
    unexpected_labels = sorted(set(mapping.values()) - set(TOPIC_ORDER))
    if unexpected_labels:
        raise ValueError(f"{country}: unexpected substantive labels: {unexpected_labels}")
    return speech_bases, unmapped


def broad_topic_frame(
    data: pd.DataFrame,
    mapping: dict[str, str],
    suffix: str,
    topics: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    broad = pd.DataFrame(index=data.index)
    for topic in topics:
        columns = [f"{base}_{suffix}" for base, label in mapping.items() if label == topic]
        if not columns:
            raise ValueError(f"No PLDA components map to {topic!r} for suffix={suffix}.")
        broad[topic] = (
            data[columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    selected_mass = broad.sum(axis=1)
    return broad.div(selected_mass, axis=0), selected_mass


def rank_alignment(
    manifesto_values: np.ndarray,
    speech_values: np.ndarray,
) -> dict[str, object]:
    finite = np.isfinite(manifesto_values) & np.isfinite(speech_values)
    n_topics = int(finite.sum())
    result: dict[str, object] = {
        "rank_topics": n_topics,
        "manifesto_tie_groups": 0,
        "speech_tie_groups": 0,
        "rank_valid": False,
        "rank_invalid_reason": "",
        "spearman_rho": math.nan,
        "spearman_p_value": math.nan,
        "spearman_fisher_z": math.nan,
    }
    if n_topics < 3:
        result["rank_invalid_reason"] = "fewer_than_three_finite_topics"
        return result
    manifesto = manifesto_values[finite]
    speech = speech_values[finite]
    result["manifesto_tie_groups"] = int(
        pd.Series(manifesto).value_counts().gt(1).sum()
    )
    result["speech_tie_groups"] = int(pd.Series(speech).value_counts().gt(1).sum())
    if np.ptp(manifesto) <= 1e-12:
        result["rank_invalid_reason"] = "constant_manifesto_vector"
        return result
    if np.ptp(speech) <= 1e-12:
        result["rank_invalid_reason"] = "constant_speech_vector"
        return result

    manifesto_rank = stats.rankdata(manifesto, method="average")
    speech_rank = stats.rankdata(speech, method="average")
    rho = float(np.corrcoef(manifesto_rank, speech_rank)[0, 1])
    if not np.isfinite(rho):
        result["rank_invalid_reason"] = "nonfinite_rank_correlation"
        return result
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n_topics - 2) / max(1.0 - rho**2, 1e-15))
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n_topics - 2))
    result.update(
        {
            "rank_valid": True,
            "spearman_rho": rho,
            "spearman_p_value": p_value,
            "spearman_fisher_z": float(np.arctanh(np.clip(rho, -0.999999, 0.999999))),
        }
    )
    return result


def distribution_alignment(
    manifesto_values: np.ndarray,
    speech_values: np.ndarray,
) -> dict[str, object]:
    """Compute 1 minus Jensen--Shannon distance on a shared topic basis."""
    manifesto = np.array(manifesto_values, dtype=float, copy=True)
    speech = np.array(speech_values, dtype=float, copy=True)
    result: dict[str, object] = {
        "alignment_topics": int(manifesto.size),
        "alignment_valid": False,
        "alignment_invalid_reason": "",
        "js_distance": math.nan,
        "alignment_score": math.nan,
        "alignment_topic_basis": "harmonized_substantive_8",
    }
    if manifesto.shape != speech.shape or manifesto.ndim != 1 or manifesto.size < 2:
        result["alignment_invalid_reason"] = "incompatible_topic_vectors"
        return result
    if (not np.isfinite(manifesto).any()) or (not np.isfinite(speech).any()):
        result["alignment_invalid_reason"] = "zero_substantive_topic_mass"
        return result
    if not (np.isfinite(manifesto).all() and np.isfinite(speech).all()):
        result["alignment_invalid_reason"] = "nonfinite_topic_share"
        return result
    if (manifesto < 0).any() or (speech < 0).any():
        result["alignment_invalid_reason"] = "negative_topic_share"
        return result
    manifesto_total = float(manifesto.sum())
    speech_total = float(speech.sum())
    if manifesto_total <= 0 or speech_total <= 0:
        result["alignment_invalid_reason"] = "zero_substantive_topic_mass"
        return result
    manifesto /= manifesto_total
    speech /= speech_total
    distance = float(jensenshannon(manifesto, speech, base=2.0))
    if not np.isfinite(distance):
        result["alignment_invalid_reason"] = "nonfinite_js_distance"
        return result
    result.update(
        {
            "alignment_valid": True,
            "js_distance": distance,
            "alignment_score": 1.0 - distance,
        }
    )
    return result


def prepare_metadata(data: pd.DataFrame, country: str) -> pd.DataFrame:
    required = {
        "speech_party",
        "month",
        "doc_key",
        "electoral_cycle_progress",
        "party_in_government",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{country}: source panel is missing columns: {missing}")
    columns = [column for column in METADATA_COLUMNS if column in data.columns]
    metadata = data[columns].copy()
    metadata.insert(0, "country_code", country)
    metadata["speech_party"] = metadata["speech_party"].astype(str).str.strip()
    metadata["month"] = metadata["month"].astype(str).str.strip()
    metadata["country_party_fe"] = country + "::" + metadata["speech_party"]
    metadata["country_month_fe"] = country + "::" + metadata["month"]
    metadata["party_month_fe"] = metadata["country_party_fe"] + "::" + metadata["month"]
    metadata["electoral_proximity"] = pd.to_numeric(
        metadata["electoral_cycle_progress"], errors="coerce"
    )
    metadata["proximity_centered"] = metadata["electoral_proximity"] - 0.5
    metadata["party_in_government"] = coerce_bool_series(metadata["party_in_government"])
    if "party_prime_minister" in metadata.columns:
        metadata["party_prime_minister"] = coerce_bool_series(metadata["party_prime_minister"])
    for column in [
        "cabinet_is_coalition",
        "cabinet_has_absolute_majority",
        "cabinet_caretaker",
        "last_election_early",
    ]:
        if column in metadata.columns:
            metadata[column] = coerce_bool_series(metadata[column])
    metadata["prior_manifesto"] = metadata.get(
        "selection_method", pd.Series("", index=metadata.index)
    ).eq(PRIOR_MANIFESTO_METHOD)
    metadata["observed_next_election"] = metadata.get(
        "cycle_boundary_source", pd.Series("", index=metadata.index)
    ).eq("observed_next_election")
    if metadata.duplicated(["country_code", "speech_party", "month"]).any():
        raise ValueError(f"{country}: duplicate country-party-month rows in source panel.")
    return metadata


def build_country_panels(
    country: str,
    panel_dir: Path,
    include_residual: bool,
    panel_override: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source_path = panel_override or panel_path(panel_dir, country)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    data = pd.read_csv(source_path, low_memory=False)
    mapping = fitted_model_topic_mapping(country, data)
    if mapping is None:
        distribution = load_distribution(country)
        mapping = topic_mapping(country, distribution)
    topic_columns, latent_columns = validate_mapping(country, data, mapping)
    topics = list(TOPIC_ORDER if include_residual else [x for x in TOPIC_ORDER if x != "Residual"])

    metadata = prepare_metadata(data, country)
    speech, speech_selected_mass = broad_topic_frame(data, mapping, "speech", topics)
    manifesto, manifesto_selected_mass = broad_topic_frame(data, mapping, "manifesto", topics)
    latent_speech = pd.to_numeric(data[f"{latent_columns[0]}_speech"], errors="coerce")
    latent_manifesto = pd.to_numeric(data[f"{latent_columns[0]}_manifesto"], errors="coerce")

    metadata["speech_selected_topic_mass"] = speech_selected_mass
    metadata["manifesto_selected_topic_mass"] = manifesto_selected_mass
    metadata["speech_latent_topic_mass"] = latent_speech
    metadata["manifesto_latent_topic_mass"] = latent_manifesto
    grid_topics = current_grid_topic_count(country, observed_topic_count=len(topic_columns))
    metadata["source_topic_count"] = len(topic_columns)
    metadata["mapped_component_count"] = len(mapping)
    metadata["topic_basis_matches_current_grid"] = bool(
        np.isfinite(grid_topics) and int(grid_topics) == len(topic_columns)
    )

    rank_rows = []
    for index in range(len(metadata)):
        manifesto_values = manifesto.iloc[index].to_numpy(dtype=float)
        speech_values = speech.iloc[index].to_numpy(dtype=float)
        rank_rows.append(
            rank_alignment(manifesto_values, speech_values)
            | distribution_alignment(manifesto_values, speech_values)
        )
    rank_panel = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(rank_rows)], axis=1
    )
    rank_panel["gov_x_proximity"] = (
        rank_panel["party_in_government"] * rank_panel["electoral_proximity"]
    )
    rank_panel["gov_x_proximity_centered"] = (
        rank_panel["party_in_government"] * rank_panel["proximity_centered"]
    )

    long_frames: list[pd.DataFrame] = []
    for topic in topics:
        topic_frame = metadata.copy()
        topic_frame["topic"] = topic
        topic_frame["speech_salience"] = speech[topic].to_numpy(dtype=float)
        topic_frame["manifesto_salience"] = manifesto[topic].to_numpy(dtype=float)
        long_frames.append(topic_frame)
    topic_panel = pd.concat(long_frames, ignore_index=True, sort=False)
    topic_panel["country_topic_fe"] = topic_panel["country_code"] + "::" + topic_panel["topic"]
    topic_panel["country_month_topic_fe"] = (
        topic_panel["country_month_fe"] + "::" + topic_panel["topic"]
    )
    topic_panel["manifesto_x_proximity"] = (
        topic_panel["manifesto_salience"] * topic_panel["electoral_proximity"]
    )
    topic_panel["manifesto_x_proximity_centered"] = (
        topic_panel["manifesto_salience"] * topic_panel["proximity_centered"]
    )
    topic_panel["manifesto_x_government"] = (
        topic_panel["manifesto_salience"] * topic_panel["party_in_government"]
    )
    topic_panel["proximity_x_government"] = (
        topic_panel["electoral_proximity"] * topic_panel["party_in_government"]
    )
    topic_panel["manifesto_x_proximity_x_government"] = (
        topic_panel["manifesto_salience"]
        * topic_panel["electoral_proximity"]
        * topic_panel["party_in_government"]
    )
    topic_panel["manifesto_x_proximity_centered_x_government"] = (
        topic_panel["manifesto_salience"]
        * topic_panel["proximity_centered"]
        * topic_panel["party_in_government"]
    )

    label_path = (
        TEST_OUTPUT_DIR
        / "plda_topic_distributions"
        / country
        / f"{country}_plda_topic_labels.csv"
    )
    audit = {
        "country_code": country,
        "party_month_rows": len(rank_panel),
        "country_party_clusters": rank_panel["country_party_fe"].nunique(),
        "topic_rows": len(topic_panel),
        "selected_topics": len(topics),
        "source_topic_count": len(topic_columns),
        "mapped_component_count": len(mapping),
        "latent_component_count": len(latent_columns),
        "mapping_source": "legacy_label_map" if label_path.exists() else "grid_and_training_labels",
        "current_grid_topic_count": grid_topics,
        "topic_basis_matches_current_grid": bool(
            np.isfinite(grid_topics) and int(grid_topics) == len(topic_columns)
        ),
        "prior_manifesto_rows": int(rank_panel["prior_manifesto"].sum()),
        "observed_next_election_rows": int(rank_panel["observed_next_election"].sum()),
        "valid_rank_rows": int(rank_panel["rank_valid"].sum()),
        "valid_alignment_rows": int(rank_panel["alignment_valid"].sum()),
        "mean_spearman_rho": rank_panel["spearman_rho"].mean(),
        "mean_alignment_score": rank_panel["alignment_score"].mean(),
        "mean_speech_selected_topic_mass": speech_selected_mass.mean(),
        "mean_manifesto_selected_topic_mass": manifesto_selected_mass.mean(),
        "mean_speech_latent_topic_mass": latent_speech.mean(),
        "mean_manifesto_latent_topic_mass": latent_manifesto.mean(),
        "source_panel": str(source_path),
    }
    return topic_panel, rank_panel, audit


def write_outputs(
    topic_panels: list[pd.DataFrame],
    rank_panels: list[pd.DataFrame],
    audits: list[dict[str, object]],
    output_dir: Path,
    include_residual: bool,
) -> None:
    data_dir = output_dir / "data"
    country_dir = data_dir / "countries"
    country_dir.mkdir(parents=True, exist_ok=True)
    topic_panel = pd.concat(topic_panels, ignore_index=True, sort=False)
    rank_panel = pd.concat(rank_panels, ignore_index=True, sort=False)
    for country in rank_panel["country_code"].drop_duplicates():
        topic_panel.loc[topic_panel["country_code"].eq(country)].to_csv(
            country_dir / f"{country}_topic_salience_panel.csv", index=False
        )
        rank_panel.loc[rank_panel["country_code"].eq(country)].to_csv(
            country_dir / f"{country}_rank_salience_panel.csv", index=False
        )
    topic_path = data_dir / "topic_salience_panel.csv"
    rank_path = data_dir / "rank_salience_panel.csv"
    audit_path = data_dir / "topic_basis_audit.csv"
    topic_panel.to_csv(topic_path, index=False)
    rank_panel.to_csv(rank_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)

    topics = TOPIC_ORDER if include_residual else [x for x in TOPIC_ORDER if x != "Residual"]
    manifest = [
        "PLDA topic-level and rank-order salience panels",
        "",
        f"Countries: {', '.join(rank_panel['country_code'].drop_duplicates())}",
        f"Topic basis ({len(topics)}): {', '.join(topics)}",
        "Shared latent topic excluded: true",
        f"Residual topic included: {include_residual}",
        "Topic shares renormalized after exclusions: true",
        "Jensen-Shannon alignment basis: harmonized substantive topics above",
        "Alignment outcome: 1 minus Jensen-Shannon distance (base-2)",
        f"Valid alignment rows: {int(rank_panel['alignment_valid'].sum())}",
        f"Party-month rows: {len(rank_panel)}",
        f"Topic-level rows: {len(topic_panel)}",
        f"Country-party clusters: {rank_panel['country_party_fe'].nunique()}",
        "",
        "Key files:",
        str(topic_path),
        str(rank_path),
        str(audit_path),
    ]
    (output_dir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    countries = normalize_countries(args.countries)
    panel_overrides = parse_panel_overrides(args.panel_override)
    topic_panels: list[pd.DataFrame] = []
    rank_panels: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for country in countries:
        topic_panel, rank_panel, audit = build_country_panels(
            country=country,
            panel_dir=args.panel_dir,
            include_residual=args.include_residual,
            panel_override=panel_overrides.get(country),
        )
        topic_panels.append(topic_panel)
        rank_panels.append(rank_panel)
        audits.append(audit)
        print(
            f"{country}: {len(rank_panel):,} party-months, "
            f"{len(topic_panel):,} topic rows, "
            f"mean alignment={rank_panel['alignment_score'].mean():.3f}, "
            f"mean rho={rank_panel['spearman_rho'].mean():.3f}"
        )
    write_outputs(
        topic_panels=topic_panels,
        rank_panels=rank_panels,
        audits=audits,
        output_dir=args.output_dir,
        include_residual=args.include_residual,
    )
    print(f"Wrote salience panels to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
