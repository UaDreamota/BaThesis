from __future__ import annotations

"""Production audit pipeline for manifesto--speech coherence regressions.

This module deliberately consumes cached regression panels and classified outputs.
It never runs topic models, embeddings, NLI, LLM annotation, or classifier inference.
All outputs are written to a new audit directory.
"""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups
from wildboottest.wildboottest import WildboottestCL


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.causality import baseline_panel_fe_regression as baseline
from scripts.causality import plda_salience_regression as salience
from scripts.metrics import plda_regression_panel as panel_builder


TEST_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_OUTPUT = TEST_DIR / "empirical_regression_audit"
SALience_COUNTRIES = list(salience.DEFAULT_COUNTRIES)
MACRO_COUNTRIES = [
    "AT", "BE", "CZ", "DK", "EE", "ES", "FI", "GB",
    "GR", "IT", "LV", "NL", "NO", "PL", "PT", "SE",
]
OLD_COUNTRIES = [country for countries in baseline.CLUSTERS.values() for country in countries]
PRIMARY_CONTRASTS = (
    "opposition_slope",
    "government_slope",
    "government_gap_p0",
    "government_gap_p05",
    "government_gap_p1",
    "government_x_proximity",
)


@dataclass
class ModelSpec:
    model_id: str
    family: str
    outcome: str
    data: pd.DataFrame
    predictors: list[str]
    effects: list[str]
    weight: str = "weight_equal"
    sample: str = "natural"
    restriction: str = "none"
    specification: str = "M1"
    unit: str = "party-month"
    covariance_primary: str = "wild country-cluster bootstrap; country CR1 intervals"
    bootstrap: bool = False
    expected_h1: str = "positive"
    expected_h2: str = "negative"
    notes: str = ""


@dataclass
class FitResult:
    spec: ModelSpec
    used: pd.DataFrame
    terms: list[str]
    beta: np.ndarray
    residual: np.ndarray
    y_within: np.ndarray
    x_within: np.ndarray
    weights: np.ndarray
    ols_result: Any
    cov_country: np.ndarray
    cov_party: np.ndarray
    cov_two_way: np.ndarray
    country_clusters: int
    party_clusters: int
    country_month_clusters: int
    dropped_terms: list[str]
    absorption_iterations: int


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument("--bootstrap-reps", type=int, default=999)
    out.add_argument("--seed", type=int, default=1711)
    out.add_argument("--skip-bootstrap", action="store_true")
    out.add_argument(
        "--render-text-report-only",
        action="store_true",
        help="Render the root plain-text model compendium from existing audit CSVs without refitting.",
    )
    out.add_argument(
        "--text-report-path",
        type=Path,
        default=BASE_DIR / "EMPIRICAL_REGRESSION_MODELS_AND_RESULTS.txt",
    )
    out.add_argument(
        "--skip-sample-flow",
        action="store_true",
        help="Skip scanning cached speech diagnostics for the sample-flow table.",
    )
    out.add_argument("--quick", action="store_true", help="Run smoke verification with fewer robustness fits.")
    return out


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def bool_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    numeric = numeric.where(~text.isin({"true", "yes"}), 1.0)
    numeric = numeric.where(~text.isin({"false", "no"}), 0.0)
    return numeric


def read_panel(path: Path, country: str | None = None, topic: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if country is not None and "country_code" not in frame:
        frame.insert(0, "country_code", country)
    if topic is not None and "nli_topic" not in frame:
        frame.insert(1, "nli_topic", topic)
    return frame


def base_panel_path(country: str) -> Path:
    return TEST_DIR / "plda_regression_panel" / country / f"{country}_plda_regression_panel_model.csv"


def load_base_covariates(countries: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for country in countries:
        path = base_panel_path(country)
        if not path.exists():
            continue
        frame = read_panel(path, country)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No PLDA/ParlGov base panels were found.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["speech_party"] = data["speech_party"].astype(str).str.strip()
    data["month"] = data["month"].astype(str).str.strip()
    data = data.sort_values(["country_code", "speech_party", "month"])
    data = data.drop_duplicates(["country_code", "speech_party", "month"], keep="first")

    # Preserve ParlGov as the preferred source, but repair genuinely missing
    # election-result shares from the linked Manifesto Project document.
    data["party_seat_share_parlgov"] = pd.to_numeric(data.get("party_seat_share"), errors="coerce")
    data["party_vote_share_parlgov"] = pd.to_numeric(data.get("party_vote_share"), errors="coerce")
    tokens = data["doc_key"].astype(str).str.extract(r"^(?P<mpds_party_id>\d+)_(?P<mpds_date>\d+)$")
    data["mpds_party_id_audit"] = pd.to_numeric(tokens["mpds_party_id"], errors="coerce")
    data["mpds_date_audit"] = pd.to_numeric(tokens["mpds_date"], errors="coerce")
    mp = pd.read_csv(
        BASE_DIR / "data" / "MPDataset_MPDS2025a.csv",
        usecols=["party", "date", "pervote", "absseat", "totseats"], low_memory=False,
    ).rename(columns={"party": "mpds_party_id_audit", "date": "mpds_date_audit"})
    mp["party_seat_share_manifesto_project"] = (
        pd.to_numeric(mp["absseat"], errors="coerce") / pd.to_numeric(mp["totseats"], errors="coerce")
    )
    mp["party_vote_share_manifesto_project"] = pd.to_numeric(mp["pervote"], errors="coerce")
    data = data.merge(
        mp[[
            "mpds_party_id_audit", "mpds_date_audit",
            "party_seat_share_manifesto_project", "party_vote_share_manifesto_project",
        ]], on=["mpds_party_id_audit", "mpds_date_audit"], how="left", validate="m:1",
    )
    data["party_seat_share"] = data["party_seat_share_parlgov"].combine_first(
        data["party_seat_share_manifesto_project"]
    )
    data["party_vote_share"] = data["party_vote_share_parlgov"].combine_first(
        data["party_vote_share_manifesto_project"]
    )
    data["party_seat_share_source_audit"] = np.select(
        [
            data["party_seat_share_parlgov"].notna(),
            data["party_seat_share_manifesto_project"].notna(),
        ], ["parlgov", "manifesto_project_fallback"], default="missing",
    )
    return data


def enrich_from_base(data: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["country_code"] = out["country_code"].astype(str).str.upper().str.strip()
    out["speech_party"] = out["speech_party"].astype(str).str.strip()
    out["month"] = out["month"].astype(str).str.strip()
    keys = ["country_code", "speech_party", "month"]
    base_columns = [column for column in base.columns if column not in keys]
    renamed = {column: f"__base_{column}" for column in base_columns}
    out = out.merge(base[keys + base_columns].rename(columns=renamed), on=keys, how="left", validate="m:1")
    for column in base_columns:
        source = f"__base_{column}"
        if column in out:
            out[column] = out[column].combine_first(out[source])
        else:
            out[column] = out[source]
        out = out.drop(columns=source)
    return out


def cabinet_transition_months(countries: Iterable[str]) -> pd.DataFrame:
    db_path = panel_builder.DEFAULT_PARLGOV_DB
    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as connection:
        for country in countries:
            pg = panel_builder.PARLGOV_COUNTRY_SHORT_BY_CODE.get(country)
            if not pg:
                continue
            query = """
                SELECT DISTINCT cabinet_id, start_date, cabinet_name, caretaker
                FROM view_cabinet
                WHERE country_name_short = ?
                ORDER BY start_date
            """
            cabinets = pd.read_sql_query(query, connection, params=(pg,))
            cabinets["start_date"] = pd.to_datetime(cabinets["start_date"], errors="coerce")
            cabinets = cabinets.dropna(subset=["start_date"])
            for row in cabinets.itertuples(index=False):
                rows.append(
                    {
                        "country_code": country,
                        "transition_month": row.start_date.strftime("%Y-%m"),
                        "cabinet_start_date": row.start_date.date().isoformat(),
                        "cabinet_id": row.cabinet_id,
                        "cabinet_name": row.cabinet_name,
                        "cabinet_caretaker_at_start": row.caretaker,
                    }
                )
    return pd.DataFrame(rows)


def add_common_features(data: pd.DataFrame, transition_keys: set[str]) -> pd.DataFrame:
    out = data.copy()
    out["country_code"] = out["country_code"].astype(str).str.upper().str.strip()
    out["speech_party"] = out["speech_party"].astype(str).str.strip()
    out["month"] = out["month"].astype(str).str.strip()
    out["month_date"] = pd.to_datetime(out["month"] + "-01", errors="coerce")
    out["calendar_year_fe"] = out["month_date"].dt.year.astype("Int64").astype(str)
    out.loc[out["month_date"].isna(), "calendar_year_fe"] = pd.NA
    out["month_of_year_fe"] = out["month_date"].dt.month.astype("Int64").astype(str)
    out.loc[out["month_date"].isna(), "month_of_year_fe"] = pd.NA
    out["calendar_month_fe"] = out["month"]
    out["country_year_fe"] = out["country_code"] + "::" + out["calendar_year_fe"]
    out["country_party_fe"] = out["country_code"] + "::" + out["speech_party"]
    out["country_month_fe"] = out["country_code"] + "::" + out["month"]
    election_id = out.get("last_election_id", pd.Series(pd.NA, index=out.index)).astype(str)
    election_date = out.get("last_election_date", pd.Series(pd.NA, index=out.index)).astype(str)
    election_token = election_id.where(~election_id.isin({"<NA>", "nan", "None"}), election_date)
    out["election_cycle_fe"] = out["country_code"] + "::" + election_token
    out["party_cycle_fe"] = out["country_party_fe"] + "::" + election_token
    proximity_source = "electoral_proximity" if "electoral_proximity" in out else "electoral_cycle_progress"
    out["electoral_proximity"] = pd.to_numeric(out[proximity_source], errors="coerce")
    out["proximity_centered"] = out["electoral_proximity"] - 0.5
    out["party_in_government"] = bool_numeric(out["party_in_government"])
    out["gov_x_proximity_centered"] = out["party_in_government"] * out["proximity_centered"]
    for column in [
        "cabinet_caretaker", "cabinet_is_coalition", "cabinet_has_absolute_majority",
        "last_election_early", "party_prime_minister",
    ]:
        if column not in out:
            out[column] = np.nan
        out[column] = bool_numeric(out[column])
    out["transition_key"] = out["country_code"] + "::" + out["month"]
    out["cabinet_transition_month"] = out["transition_key"].isin(transition_keys)
    out["ordinary_cabinet_month"] = (~out["cabinet_transition_month"]) & out["cabinet_caretaker"].fillna(0).eq(0)
    out["weight_equal"] = 1.0
    if "speech_volume_segments" in out:
        out["analysis_volume"] = pd.to_numeric(out["speech_volume_segments"], errors="coerce")
    elif "n_speeches" in out:
        out["analysis_volume"] = pd.to_numeric(out["n_speeches"], errors="coerce")
    else:
        out["analysis_volume"] = 1.0
    out["weight_volume"] = out["analysis_volume"]
    out["weight_sqrt_volume"] = np.sqrt(out["analysis_volume"].clip(lower=0))
    positive = out.loc[out["analysis_volume"].gt(0), "analysis_volume"]
    cap = float(positive.quantile(0.95)) if len(positive) else 1.0
    out["weight_capped_volume"] = out["analysis_volume"].clip(upper=cap)
    if "n_pairs" in out:
        out["weight_pairs"] = pd.to_numeric(out["n_pairs"], errors="coerce")
    out["log_analysis_volume"] = np.log1p(out["analysis_volume"])
    out["minimum_volume"] = out["analysis_volume"].ge(5)
    global_origin = out["month_date"].min()
    out["linear_time"] = (
        (out["month_date"].dt.year - global_origin.year) * 12
        + out["month_date"].dt.month - global_origin.month
    ).astype(float)
    for country in sorted(out["country_code"].dropna().unique()):
        out[f"trend_{country}"] = out["linear_time"] * out["country_code"].eq(country).astype(float)

    coalition = out["cabinet_is_coalition"].fillna(0).eq(1)
    majority = out["cabinet_has_absolute_majority"].fillna(0).eq(1)
    caretaker = out["cabinet_caretaker"].fillna(0).eq(1)
    government = out["party_in_government"].eq(1)
    out["cabinet_party_context"] = "opposition"
    out.loc[government & ~coalition & majority, "cabinet_party_context"] = "single_majority"
    out.loc[government & ~coalition & ~majority, "cabinet_party_context"] = "single_minority"
    out.loc[government & coalition & majority, "cabinet_party_context"] = "coalition_majority"
    out.loc[government & coalition & ~majority, "cabinet_party_context"] = "coalition_minority"
    out.loc[government & caretaker, "cabinet_party_context"] = "caretaker_government"
    return out


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    all_countries = sorted(set(SALience_COUNTRIES + MACRO_COUNTRIES + OLD_COUNTRIES))
    base = load_base_covariates(all_countries)
    transitions = cabinet_transition_months(all_countries)
    transition_keys = set(transitions["country_code"] + "::" + transitions["transition_month"])

    topic = pd.read_csv(TEST_DIR / "plda_salience/substantive/data/topic_salience_panel.csv", low_memory=False)
    rank = pd.read_csv(TEST_DIR / "plda_salience/substantive/data/rank_salience_panel.csv", low_memory=False)
    topic = topic.loc[topic["prior_manifesto"].fillna(False).astype(bool)].copy()
    rank = rank.loc[rank["prior_manifesto"].fillna(False).astype(bool)].copy()
    topic = enrich_from_base(topic, base)
    rank = enrich_from_base(rank, base)

    alignment_columns = {
        "alignment_score",
        "alignment_valid",
        "alignment_topic_basis",
        "js_distance",
    }
    missing_alignment = sorted(alignment_columns - set(rank.columns))
    if missing_alignment:
        raise ValueError(
            "Harmonized Jensen-Shannon alignment columns are missing from the "
            f"salience panel: {missing_alignment}. Rebuild plda_salience_panels.py first."
        )
    valid_alignment = rank["alignment_valid"].fillna(False).astype(bool)
    alignment = rank.loc[valid_alignment].copy()
    bases = set(alignment["alignment_topic_basis"].dropna().astype(str).unique())
    if bases != {"harmonized_substantive_8"}:
        raise ValueError(f"Unexpected Jensen-Shannon topic basis: {sorted(bases)}")
    if not alignment["alignment_score"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Harmonized Jensen-Shannon alignment scores must lie in [0, 1].")
    old_raw, _ = baseline.load_nli_panels(
        TEST_DIR / "nli_regression_panel", OLD_COUNTRIES,
        ["Macroeconomics", "Gal_Tan"], baseline.DEFAULT_NLI_MODEL_NAME,
    )
    old = enrich_from_base(old_raw, base)
    new_macro_raw, _ = baseline.load_nli_panels(
        TEST_DIR / "nli_consensus_regression_panel", MACRO_COUNTRIES,
        ["Macroeconomics"], "consensus_deberta",
    )
    new_galtan_raw, _ = baseline.load_nli_panels(
        TEST_DIR / "nli_consensus_regression_panel", MACRO_COUNTRIES,
        ["Gal_Tan"], "consensus_deberta",
    )
    new_macro = enrich_from_base(new_macro_raw, base)
    new_galtan = enrich_from_base(new_galtan_raw, base)
    new_combined = pd.concat([new_macro, new_galtan], ignore_index=True, sort=False)

    frames = {
        "plda_topic": add_common_features(topic, transition_keys),
        "plda_alignment": add_common_features(alignment, transition_keys),
        "spearman": add_common_features(rank, transition_keys),
        "old_nli": add_common_features(old, transition_keys),
        "new_macro": add_common_features(new_macro, transition_keys),
        "new_galtan": add_common_features(new_galtan, transition_keys),
        "new_combined": add_common_features(new_combined, transition_keys),
    }
    return frames, base, transitions


def factor_codes(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series.astype(str), sort=False)
    if np.any(codes < 0):
        raise ValueError("Missing fixed-effect codes after complete-case filtering.")
    return codes


def fit_model(spec: ModelSpec) -> FitResult:
    required = [
        spec.outcome, *spec.predictors, *spec.effects, spec.weight,
        "country_code", "country_party_fe", "country_month_fe", "month",
    ]
    required = list(dict.fromkeys(required))
    missing = sorted(set(required) - set(spec.data.columns))
    if missing:
        raise ValueError(f"{spec.model_id}: missing columns {missing}")
    data = spec.data.dropna(subset=required).copy()
    numeric_cols = [spec.outcome, *spec.predictors, spec.weight]
    numeric = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1) & numeric[spec.weight].gt(0)
    used = data.loc[mask].copy().reset_index(drop=True)
    numeric = numeric.loc[mask].reset_index(drop=True)
    if len(used) < 50:
        raise ValueError(f"{spec.model_id}: only {len(used)} complete observations")
    weights = numeric[spec.weight].to_numpy(float, copy=True)
    weights /= weights.mean()
    values = numeric[[spec.outcome, *spec.predictors]].to_numpy(float)
    effects = [factor_codes(used[column]) for column in spec.effects]
    within, iterations = salience.absorb_effects(values, effects, weights, 1e-9, 5000)
    y = within[:, 0]
    x = within[:, 1:]
    sqrt_w = np.sqrt(weights)
    kept, dropped = salience.sequential_full_rank_columns(x * sqrt_w[:, None], spec.predictors)
    terms = [spec.predictors[index] for index in kept]
    x = x[:, kept]
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    ols_result = sm.OLS(yw, xw).fit()
    beta = np.asarray(ols_result.params)
    residual = y - x @ beta
    country = pd.factorize(used["country_code"].astype(str), sort=False)[0]
    party = pd.factorize(used["country_party_fe"].astype(str), sort=False)[0]
    country_month = pd.factorize(used["country_month_fe"].astype(str), sort=False)[0]
    cov_c = np.asarray(cov_cluster(ols_result, country, use_correction=True))
    cov_p = np.asarray(cov_cluster(ols_result, party, use_correction=True))
    cov_tw, _, _ = cov_cluster_2groups(ols_result, party, country_month, use_correction=True)
    return FitResult(
        spec=spec, used=used, terms=terms, beta=beta, residual=residual,
        y_within=y, x_within=x, weights=weights, ols_result=ols_result,
        cov_country=np.asarray(cov_tw * 0 + cov_c), cov_party=cov_p,
        cov_two_way=np.asarray(cov_tw), country_clusters=len(np.unique(country)),
        party_clusters=len(np.unique(party)), country_month_clusters=len(np.unique(country_month)),
        dropped_terms=dropped, absorption_iterations=iterations,
    )


def contrast_vector(result: FitResult, terms: dict[str, float]) -> np.ndarray:
    indexes = {term: index for index, term in enumerate(result.terms)}
    missing = sorted(set(terms) - set(indexes))
    if missing:
        raise ValueError(f"{result.spec.model_id}: contrast terms not identified: {missing}")
    vector = np.zeros(len(result.terms))
    for term, coefficient in terms.items():
        vector[indexes[term]] = coefficient
    return vector


def wild_package_test(result: FitResult, vector: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    sqrt_w = np.sqrt(result.weights)
    xw = result.x_within * sqrt_w[:, None]
    yw = result.y_within * sqrt_w
    clusters = pd.factorize(result.used["country_code"].astype(str), sort=False)[0]
    bootstrap = WildboottestCL(
        X=xw, Y=yw, cluster=clusters, R=vector, B=reps,
        seed=seed, parallel=False,
    )
    bootstrap.get_scores(bootstrap_type="11", impose_null=True, adj=True, cluster_adj=True)
    bootstrap.get_weights(weights_type="webb")
    bootstrap.get_numer()
    bootstrap.get_denom()
    bootstrap.get_tboot()
    bootstrap.get_vcov()
    bootstrap.get_tstat()
    bootstrap.get_pvalue(pval_type="two-tailed")
    return float(bootstrap.t_stat), float(bootstrap.pvalue)


def contrast_definitions(family: str) -> list[tuple[str, dict[str, float]]]:
    base = [
        ("opposition_slope", {"proximity_centered": 1.0}),
        ("government_slope", {"proximity_centered": 1.0, "gov_x_proximity_centered": 1.0}),
        ("government_gap_p0", {"party_in_government": 1.0, "gov_x_proximity_centered": -0.5}),
        ("government_gap_p05", {"party_in_government": 1.0}),
        ("government_gap_p1", {"party_in_government": 1.0, "gov_x_proximity_centered": 0.5}),
        ("government_x_proximity", {"gov_x_proximity_centered": 1.0}),
    ]
    if family != "plda_topic":
        return base
    return [
        ("manifesto_linkage_midcycle_opposition", {"manifesto_salience": 1.0}),
        ("opposition_slope", {"manifesto_x_proximity_centered": 1.0}),
        (
            "government_slope",
            {"manifesto_x_proximity_centered": 1.0, "manifesto_x_proximity_centered_x_government": 1.0},
        ),
        (
            "government_gap_p0",
            {"manifesto_x_government": 1.0, "manifesto_x_proximity_centered_x_government": -0.5},
        ),
        ("government_gap_p05", {"manifesto_x_government": 1.0}),
        (
            "government_gap_p1",
            {"manifesto_x_government": 1.0, "manifesto_x_proximity_centered_x_government": 0.5},
        ),
        (
            "government_x_proximity",
            {"manifesto_x_proximity_centered_x_government": 1.0},
        ),
    ]


def covariance_stats(estimate: float, vector: np.ndarray, covariance: np.ndarray, df: int) -> dict[str, float]:
    variance = max(float(vector @ covariance @ vector), 0.0)
    se = math.sqrt(variance)
    statistic = estimate / se if se > 0 else math.nan
    critical = stats.t.ppf(0.975, max(df, 1))
    return {
        "se": se,
        "t": statistic,
        "p": float(2 * stats.t.sf(abs(statistic), max(df, 1))) if np.isfinite(statistic) else math.nan,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
    }


def model_rows(result: FitResult, reps: int, seed: int, skip_bootstrap: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    coefficients: list[dict[str, Any]] = []
    for index, term in enumerate(result.terms):
        vector = np.zeros(len(result.terms)); vector[index] = 1.0
        estimate = float(result.beta[index])
        c = covariance_stats(estimate, vector, result.cov_country, result.country_clusters - 1)
        p = covariance_stats(estimate, vector, result.cov_party, result.party_clusters - 1)
        tw = covariance_stats(
            estimate, vector, result.cov_two_way,
            min(result.party_clusters, result.country_month_clusters) - 1,
        )
        coefficients.append({
            "model_id": result.spec.model_id, "family": result.spec.family, "term": term,
            "estimate": estimate, "country_se": c["se"], "country_p": c["p"],
            "party_se": p["se"], "party_p": p["p"], "two_way_se": tw["se"], "two_way_p": tw["p"],
        })

    contrasts: list[dict[str, Any]] = []
    definitions = contrast_definitions(result.spec.family)
    if result.spec.specification == "placebo":
        definitions = [
            ("opposition_slope", {"placebo_proximity_centered": 1.0}),
            ("government_slope", {"placebo_proximity_centered": 1.0, "placebo_gov_x_proximity": 1.0}),
            ("government_gap_p0", {"party_in_government": 1.0, "placebo_gov_x_proximity": -0.5}),
            ("government_gap_p05", {"party_in_government": 1.0}),
            ("government_gap_p1", {"party_in_government": 1.0, "placebo_gov_x_proximity": 0.5}),
            ("government_x_proximity", {"placebo_gov_x_proximity": 1.0}),
        ]
    for name, definition in definitions:
        try:
            vector = contrast_vector(result, definition)
        except ValueError:
            continue
        estimate = float(vector @ result.beta)
        c = covariance_stats(estimate, vector, result.cov_country, result.country_clusters - 1)
        p = covariance_stats(estimate, vector, result.cov_party, result.party_clusters - 1)
        tw = covariance_stats(
            estimate, vector, result.cov_two_way,
            min(result.party_clusters, result.country_month_clusters) - 1,
        )
        wild_t = wild_p = math.nan
        if result.spec.bootstrap and not skip_bootstrap:
            wild_t, wild_p = wild_package_test(
                result, vector, reps, stable_seed(seed, result.spec.model_id, name)
            )
        contrasts.append({
            "model_id": result.spec.model_id, "family": result.spec.family,
            "specification": result.spec.specification, "sample": result.spec.sample,
            "restriction": result.spec.restriction, "contrast": name,
            "estimate": estimate,
            "country_se": c["se"], "country_p": c["p"],
            "country_ci_low": c["ci_low"], "country_ci_high": c["ci_high"],
            "party_se": p["se"], "party_p": p["p"],
            "two_way_se": tw["se"], "two_way_p": tw["p"],
            "wild_t": wild_t, "wild_p": wild_p,
            "wild_reps": 0 if skip_bootstrap or not result.spec.bootstrap else reps,
            "contrast_terms": json.dumps(definition, sort_keys=True),
        })

    used = result.used
    switcher_units = (
        used[["country_party_fe", "month", "party_in_government"]].drop_duplicates()
        .groupby("country_party_fe")["party_in_government"].nunique().gt(1)
    )
    outcome = pd.to_numeric(used[result.spec.outcome], errors="coerce")
    raw_weight = pd.to_numeric(used[result.spec.weight], errors="coerce")
    diagnostics = {
        "model_id": result.spec.model_id, "family": result.spec.family,
        "specification": result.spec.specification, "sample": result.spec.sample,
        "restriction": result.spec.restriction, "unit": result.spec.unit,
        "outcome": result.spec.outcome, "predictors": ";".join(result.terms),
        "fixed_effects": ";".join(result.spec.effects), "weight": result.spec.weight,
        "covariance_primary": result.spec.covariance_primary,
        "observations": len(used), "countries": used["country_code"].nunique(),
        "parties": used["country_party_fe"].nunique(),
        "election_cycles": used["election_cycle_fe"].nunique(),
        "country_clusters": result.country_clusters, "party_clusters": result.party_clusters,
        "country_month_clusters": result.country_month_clusters,
        "government_switchers": int(switcher_units.sum()),
        "government_identifying_observations": int(used["country_party_fe"].isin(switcher_units[switcher_units].index).sum()),
        "unweighted_outcome_mean": float(outcome.mean()),
        "weighted_outcome_mean": float(np.average(outcome, weights=raw_weight)),
        "dropped_terms": ";".join(result.dropped_terms),
        "absorption_iterations": result.absorption_iterations,
        "notes": result.spec.notes,
    }
    return coefficients, contrasts, diagnostics


def add_context_design(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = data.copy()
    predictors = ["proximity_centered"]
    categories = [
        "single_majority", "single_minority", "coalition_majority",
        "coalition_minority", "caretaker_government",
    ]
    for category in categories:
        dummy = f"context_{category}"
        interaction = f"{dummy}_x_proximity"
        out[dummy] = out["cabinet_party_context"].eq(category).astype(float)
        out[interaction] = out[dummy] * out["proximity_centered"]
        predictors.extend([dummy, interaction])
    return out, predictors


def linear_family_specs(
    family: str,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    expected_h1: str,
    expected_h2: str,
    quick: bool,
) -> list[ModelSpec]:
    effects = ["country_party_fe", "calendar_year_fe", "month_of_year_fe"]
    if family in {"old_nli", "new_combined"}:
        effects.append("nli_topic")
    base_terms = ["proximity_centered", "party_in_government", "gov_x_proximity_centered"]
    common = data.dropna(subset=[outcome, *base_terms, *effects, "party_seat_share"]).copy()
    ordinary = data.loc[data["ordinary_cabinet_month"]].copy()
    ordinary_common = common.loc[common["ordinary_cabinet_month"]].copy()

    def make(
        suffix: str,
        frame: pd.DataFrame,
        predictors: list[str],
        specification: str,
        sample: str,
        restriction: str = "none",
        weight: str = "weight_equal",
        bootstrap: bool = False,
        fixed_effects: list[str] | None = None,
        notes: str = "",
    ) -> ModelSpec:
        return ModelSpec(
            model_id=f"{family}_{suffix}", family=family, outcome=outcome,
            data=frame, predictors=predictors, effects=fixed_effects or effects,
            weight=weight, sample=sample, restriction=restriction,
            specification=specification, unit=unit, bootstrap=bootstrap,
            expected_h1=expected_h1, expected_h2=expected_h2, notes=notes,
        )

    specs = [
        make("m1_natural", data, base_terms, "M1", "natural", bootstrap=True),
        make("m2_natural", data, base_terms + ["party_seat_share"], "M2", "natural", bootstrap=True),
        make("m1_common", common, base_terms, "M1", "M1-M2 common", bootstrap=True),
        make("m2_common", common, base_terms + ["party_seat_share"], "M2", "M1-M2 common", bootstrap=True),
        make(
            "m3_m1_natural", ordinary, base_terms, "M3-M1", "natural",
            "exclude caretaker and cabinet-transition months", bootstrap=True,
        ),
        make(
            "m3_m2_natural", ordinary, base_terms + ["party_seat_share"], "M3-M2", "natural",
            "exclude caretaker and cabinet-transition months", bootstrap=True,
        ),
        make(
            "m3_m1_common", ordinary_common, base_terms, "M3-M1", "M1-M2 common",
            "exclude caretaker and cabinet-transition months", bootstrap=True,
        ),
        make(
            "m3_m2_common", ordinary_common, base_terms + ["party_seat_share"], "M3-M2", "M1-M2 common",
            "exclude caretaker and cabinet-transition months", bootstrap=True,
        ),
    ]
    if quick:
        return specs[:4]

    context_data, context_terms = add_context_design(data)
    specs.append(
        make(
            "m4_context", context_data, context_terms, "M4", "natural", notes=(
                "Mutually exclusive party-context categories relative to opposition; "
                "this estimates cabinet-context heterogeneity, not the M1 government contrast."
            )
        )
    )
    specs.extend(
        [
            make("m5_volume_weight", data, base_terms, "M5", "natural", weight="weight_volume"),
            make("m5_sqrt_volume_weight", data, base_terms, "M5", "natural", weight="weight_sqrt_volume"),
            make("m5_capped_volume_weight", data, base_terms, "M5", "natural", weight="weight_capped_volume"),
            make(
                "m5_minimum_volume", data.loc[data["minimum_volume"]].copy(), base_terms,
                "M5", "minimum volume >= 5 analysis units",
            ),
            make(
                "conditional_log_volume", data, base_terms + ["log_analysis_volume"],
                "conditional-volume", "natural", notes="Conditions on contemporaneous volume; not a primary causal control.",
            ),
            make(
                "country_year", data, base_terms, "FE robustness", "natural",
                fixed_effects=["country_party_fe", "country_year_fe", "month_of_year_fe"],
            ),
            make(
                "party_cycle", data, base_terms, "FE robustness", "natural",
                fixed_effects=["party_cycle_fe", "calendar_year_fe", "month_of_year_fe"] + (["nli_topic"] if family == "old_nli" else []),
                notes="Government level is identified only by within-party-cycle status changes.",
            ),
        ]
    )
    party_variation = (
        data[["country_party_fe", "month", "party_in_government"]].drop_duplicates()
        .groupby("country_party_fe")["party_in_government"].nunique()
    )
    switchers = set(party_variation[party_variation.gt(1)].index)
    specs.append(
        make(
            "switchers_only", data.loc[data["country_party_fe"].isin(switchers)].copy(),
            base_terms, "switcher robustness", "government-status switchers only",
            notes="Government contrasts are identified only by parties observed in both statuses.",
        )
    )
    trend_terms = [f"trend_{country}" for country in sorted(data["country_code"].dropna().unique())]
    specs.append(
        make(
            "country_trends", data, base_terms + trend_terms, "trend robustness", "natural",
            notes="Country-specific trends can absorb genuine slow electoral-cycle variation.",
        )
    )
    early_known = data["last_election_early"].notna()
    if early_known.any():
        specs.append(
            make(
                "exclude_early_cycles", data.loc[early_known & data["last_election_early"].eq(0)].copy(),
                base_terms, "election timing robustness", "non-early preceding elections",
            )
        )
    if {"days_since_last_election", "cycle_length_days"}.issubset(data.columns):
        for months in (6, 12):
            shifted = data.copy()
            days = pd.to_numeric(shifted["days_since_last_election"], errors="coerce")
            length = pd.to_numeric(shifted["cycle_length_days"], errors="coerce")
            shifted["placebo_proximity"] = ((days + months * 30.4375) / length) % 1.0
            shifted["placebo_proximity_centered"] = shifted["placebo_proximity"] - 0.5
            shifted["placebo_gov_x_proximity"] = shifted["party_in_government"] * shifted["placebo_proximity_centered"]
            specs.append(
                make(
                    f"placebo_shift_{months}m", shifted,
                    ["placebo_proximity_centered", "party_in_government", "placebo_gov_x_proximity"],
                    "placebo", "natural", notes=f"Circular {months}-month shift of the actual electoral clock.",
                )
            )
    if "n_pairs" in data:
        specs.append(make("m5_pair_weight", data, base_terms, "M5", "natural", weight="weight_pairs"))
    return specs


def plda_topic_specs(data: pd.DataFrame, quick: bool) -> list[ModelSpec]:
    terms = [
        "manifesto_salience", "manifesto_x_proximity_centered",
        "manifesto_x_government", "manifesto_x_proximity_centered_x_government",
    ]
    effects = ["party_month_fe", "topic"]
    specs = [
        ModelSpec(
            "plda_topic_m1_equal", "plda_topic", "speech_salience", data, terms, effects,
            specification="M1", unit="party-topic-month", bootstrap=True,
            notes="Party-month FE absorb government, proximity, seat share, and all other party-month controls.",
        ),
        ModelSpec(
            "plda_topic_m3_equal", "plda_topic", "speech_salience",
            data.loc[data["ordinary_cabinet_month"]].copy(), terms, effects,
            specification="M3", restriction="exclude caretaker and cabinet-transition months",
            unit="party-topic-month", bootstrap=True,
        ),
    ]
    if quick:
        return specs
    specs.extend(
        [
            ModelSpec(
                "plda_topic_m5_word_weight", "plda_topic", "speech_salience", data, terms, effects,
                weight="speech_volume_words", specification="M5", unit="party-topic-month",
                notes="Average-word/precision-weighted robustness estimand.",
            ),
            ModelSpec(
                "plda_topic_m5_segment_weight", "plda_topic", "speech_salience", data, terms, effects,
                weight="speech_volume_segments", specification="M5", unit="party-topic-month",
                notes="Approximate average-speech/segment robustness estimand.",
            ),
            ModelSpec(
                "plda_topic_m5_minimum_volume", "plda_topic", "speech_salience",
                data.loc[data["minimum_volume"]].copy(), terms, effects,
                specification="M5", sample="minimum volume >= 5 analysis units", unit="party-topic-month",
            ),
        ]
    )
    return specs


def build_specs(frames: dict[str, pd.DataFrame], quick: bool) -> list[ModelSpec]:
    specs = plda_topic_specs(frames["plda_topic"], quick)
    specs.extend(
        linear_family_specs(
            "plda_alignment", frames["plda_alignment"], "alignment_score",
            "party-month", "positive", "negative", quick,
        )
    )
    specs.extend(
        linear_family_specs(
            "spearman", frames["spearman"].loc[frames["spearman"]["rank_valid"].fillna(False)].copy(),
            "spearman_rho", "party-month", "positive", "negative", quick,
        )
    )
    specs.extend(
        linear_family_specs(
            "old_nli", frames["old_nli"], "inconsistency_share",
            "party-topic-month", "negative", "positive", quick,
        )
    )
    specs.extend(
        linear_family_specs(
            "new_macro", frames["new_macro"], "inconsistency_share",
            "party-month", "negative", "positive", quick,
        )
    )
    specs.extend(
        linear_family_specs(
            "new_galtan", frames["new_galtan"], "inconsistency_share",
            "party-month", "negative", "positive", quick,
        )
    )
    specs.extend(
        linear_family_specs(
            "new_combined", frames["new_combined"], "inconsistency_share",
            "party-topic-month", "negative", "positive", quick,
        )
    )
    return specs


def add_qvalues(contrasts: pd.DataFrame) -> pd.DataFrame:
    out = contrasts.copy()
    if out.empty:
        return out
    out["test_family"] = np.select(
        [
            out["specification"].isin(["M1", "M2", "M3-M1", "M3-M2"]),
            out["specification"].eq("M5"),
            out["specification"].str.contains("robustness|trend", case=False, regex=True),
            out["specification"].eq("placebo"),
        ],
        ["nested_primary", "weighting", "fixed_effects", "placebo"],
        default="other_exploratory",
    )
    out["p_for_adjustment"] = out["wild_p"].where(out["wild_p"].notna(), out["country_p"])
    out["bh_q"] = np.nan
    for _, index in out.groupby(["family", "test_family"], dropna=False).groups.items():
        valid = out.loc[index, "p_for_adjustment"].notna()
        valid_index = out.loc[index].index[valid]
        if len(valid_index):
            out.loc[valid_index, "bh_q"] = multipletests(
                out.loc[valid_index, "p_for_adjustment"].to_numpy(float), method="fdr_bh"
            )[1]
    return out


def prediction_row(
    result: FitResult,
    vector: np.ndarray,
    family: str,
    form: str,
    trajectory: str,
    proximity: float,
) -> dict[str, Any]:
    estimate = float(vector @ result.beta)
    c = covariance_stats(estimate, vector, result.cov_country, result.country_clusters - 1)
    return {
        "family": family, "model_id": result.spec.model_id, "functional_form": form,
        "trajectory": trajectory, "proximity": proximity, "estimate": estimate,
        "country_se": c["se"], "country_p": c["p"],
        "ci_95_low": c["ci_low"], "ci_95_high": c["ci_high"],
        "countries": result.country_clusters, "observations": len(result.used),
    }


def functional_form_models(
    family: str,
    data: pd.DataFrame,
    outcome: str,
) -> tuple[list[FitResult], list[dict[str, Any]]]:
    effects = ["country_party_fe", "calendar_year_fe", "month_of_year_fe"]
    if family == "old_nli":
        effects.append("nli_topic")
    fitted: list[FitResult] = []
    rows: list[dict[str, Any]] = []

    bins = data.copy()
    quarter = pd.cut(
        bins["electoral_proximity"], [-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["q1", "q2", "q3", "q4"], include_lowest=True,
    )
    bin_terms: list[str] = ["party_in_government"]
    for label in ["q2", "q3", "q4"]:
        term = f"cycle_{label}"
        interaction = f"gov_x_{label}"
        bins[term] = quarter.eq(label).astype(float)
        bins[interaction] = bins[term] * bins["party_in_government"]
        bin_terms.extend([term, interaction])
    bin_spec = ModelSpec(
        f"{family}_cycle_quarters", family, outcome, bins, bin_terms, effects,
        specification="functional form", unit="party-month",
        notes="Quarter 1 is the omitted post-election category.",
    )
    bin_fit = fit_model(bin_spec); fitted.append(bin_fit)
    for index, proximity in enumerate([0.125, 0.375, 0.625, 0.875], start=1):
        for status, government in [("opposition_relative_q1", 0), ("government_relative_q1", 1)]:
            vector = np.zeros(len(bin_fit.terms))
            lookup = {term: pos for pos, term in enumerate(bin_fit.terms)}
            if index > 1:
                vector[lookup[f"cycle_q{index}"]] = 1.0
                if government:
                    vector[lookup[f"gov_x_q{index}"]] = 1.0
            rows.append(prediction_row(bin_fit, vector, family, "quarter bins", status, proximity))
        lookup = {term: pos for pos, term in enumerate(bin_fit.terms)}
        gap = np.zeros(len(bin_fit.terms)); gap[lookup["party_in_government"]] = 1.0
        if index > 1:
            gap[lookup[f"gov_x_q{index}"]] = 1.0
        rows.append(prediction_row(bin_fit, gap, family, "quarter bins", "government_minus_opposition", proximity))

    spline = data.dropna(subset=["electoral_proximity"]).copy()
    basis = patsy.dmatrix(
        "cr(x, df=4, constraints='center') - 1",
        {"x": spline["electoral_proximity"].to_numpy(float)}, return_type="dataframe",
    )
    design_info = basis.design_info
    spline_terms: list[str] = []
    for index in range(basis.shape[1]):
        term = f"spline_{index}"
        interaction = f"gov_x_spline_{index}"
        spline[term] = basis.iloc[:, index].to_numpy(float)
        spline[interaction] = spline[term] * spline["party_in_government"]
        spline_terms.extend([term, interaction])
    spline_terms.append("party_in_government")
    spline_spec = ModelSpec(
        f"{family}_restricted_cubic_spline", family, outcome, spline,
        spline_terms, effects, specification="functional form", unit="party-month",
        notes="Natural cubic regression spline with four centered degrees of freedom.",
    )
    spline_fit = fit_model(spline_spec); fitted.append(spline_fit)
    grid = np.linspace(0, 1, 41)
    grid_basis = np.asarray(
        patsy.build_design_matrices([design_info], {"x": grid})[0], dtype=float
    )
    mid_basis = np.asarray(
        patsy.build_design_matrices([design_info], {"x": np.array([0.5])})[0], dtype=float
    )[0]
    lookup = {term: pos for pos, term in enumerate(spline_fit.terms)}
    for row_index, proximity in enumerate(grid):
        delta = grid_basis[row_index] - mid_basis
        for status, government in [("opposition_relative_midcycle", 0), ("government_relative_midcycle", 1)]:
            vector = np.zeros(len(spline_fit.terms))
            for basis_index, value in enumerate(delta):
                vector[lookup[f"spline_{basis_index}"]] = value
                if government:
                    vector[lookup[f"gov_x_spline_{basis_index}"]] = value
            rows.append(prediction_row(spline_fit, vector, family, "restricted cubic spline", status, float(proximity)))
        gap = np.zeros(len(spline_fit.terms)); gap[lookup["party_in_government"]] = 1.0
        for basis_index, value in enumerate(grid_basis[row_index]):
            gap[lookup[f"gov_x_spline_{basis_index}"]] = value
        rows.append(prediction_row(spline_fit, gap, family, "restricted cubic spline", "government_minus_opposition", float(proximity)))
    return fitted, rows


def plot_functional_effects(effects: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots" / "functional_form"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (family, form), group in effects.groupby(["family", "functional_form"]):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for trajectory, color in [
            ("opposition_relative_midcycle", "#3566a8"),
            ("government_relative_midcycle", "#c14f3f"),
            ("government_minus_opposition", "#4a8b57"),
            ("opposition_relative_q1", "#3566a8"),
            ("government_relative_q1", "#c14f3f"),
        ]:
            part = group.loc[group["trajectory"].eq(trajectory)].sort_values("proximity")
            if part.empty:
                continue
            ax.plot(part["proximity"], part["estimate"], marker="o" if "quarter" in form else None, label=trajectory.replace("_", " "), color=color)
            ax.fill_between(part["proximity"], part["ci_95_low"], part["ci_95_high"], color=color, alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Electoral proximity (0 post-election, 1 pre-election)", ylabel="Within-party fitted contrast", title=f"{family}: {form}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{family}_{form.replace(' ', '_')}.png", dpi=220)
        plt.close(fig)


def contrast_estimates_without_bootstrap(result: FitResult) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, definition in contrast_definitions(result.spec.family):
        try:
            vector = contrast_vector(result, definition)
        except ValueError:
            continue
        estimate = float(vector @ result.beta)
        c = covariance_stats(estimate, vector, result.cov_country, result.country_clusters - 1)
        rows.append({"contrast": name, "estimate": estimate, "country_se": c["se"], "country_p": c["p"]})
    return rows


def influence_analysis(primary_fits: dict[str, FitResult], quick: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, result in primary_fits.items():
        countries = sorted(result.used["country_code"].unique())
        for country in countries:
            spec = result.spec
            subset_spec = ModelSpec(
                model_id=f"{spec.model_id}_drop_{country}", family=spec.family,
                outcome=spec.outcome, data=spec.data.loc[spec.data["country_code"].ne(country)].copy(),
                predictors=spec.predictors, effects=spec.effects, weight=spec.weight,
                sample=spec.sample, restriction=f"leave out {country}", specification="leave-one-country-out",
                unit=spec.unit,
            )
            try:
                fit = fit_model(subset_spec)
            except (ValueError, np.linalg.LinAlgError):
                continue
            for item in contrast_estimates_without_bootstrap(fit):
                rows.append({
                    "family": family, "influence_type": "leave_one_country_out",
                    "omitted_unit": country, **item,
                })
        if quick or family not in {"spearman", "new_macro", "new_galtan", "new_combined", "plda_alignment", "old_nli"}:
            continue
        cycles = sorted(result.used["election_cycle_fe"].dropna().unique())
        for cycle in cycles:
            spec = result.spec
            subset_spec = ModelSpec(
                model_id=f"{spec.model_id}_drop_cycle", family=spec.family,
                outcome=spec.outcome, data=spec.data.loc[spec.data["election_cycle_fe"].ne(cycle)].copy(),
                predictors=spec.predictors, effects=spec.effects, weight=spec.weight,
                sample=spec.sample, restriction=f"leave out cycle {cycle}", specification="leave-one-cycle-out",
                unit=spec.unit,
            )
            try:
                fit = fit_model(subset_spec)
            except (ValueError, np.linalg.LinAlgError):
                continue
            for item in contrast_estimates_without_bootstrap(fit):
                rows.append({
                    "family": family, "influence_type": "leave_one_cycle_out",
                    "omitted_unit": cycle, **item,
                })
    return pd.DataFrame(rows)


def plot_influence(influence: pd.DataFrame, output_dir: Path) -> None:
    if influence.empty:
        return
    plot_dir = output_dir / "plots" / "influence"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (family, kind, contrast), group in influence.groupby(["family", "influence_type", "contrast"]):
        ordered = group.sort_values("estimate").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.16 * len(ordered))))
        y = np.arange(len(ordered))
        ax.errorbar(ordered["estimate"], y, xerr=1.96 * ordered["country_se"], fmt="o", markersize=3, color="#355c8a", ecolor="#9ab0ca")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ordered["omitted_unit"], fontsize=6)
        ax.set(xlabel="Estimate after omission", title=f"{family}: {contrast} ({kind})")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{family}_{kind}_{contrast}.png", dpi=200)
        plt.close(fig)


def month_number(series: pd.Series) -> pd.Series:
    date = pd.to_datetime(series.astype(str) + "-01", errors="coerce")
    return date.dt.year * 12 + date.dt.month


def assign_nearest_event_time(
    data: pd.DataFrame,
    direction: str,
    window: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        data[["country_code", "country_party_fe", "speech_party", "month", "party_in_government"]]
        .drop_duplicates().sort_values(["country_party_fe", "month"])
    )
    monthly["month_number"] = month_number(monthly["month"])
    monthly["previous_status"] = monthly.groupby("country_party_fe")["party_in_government"].shift()
    if direction == "entry":
        event_mask = monthly["previous_status"].eq(0) & monthly["party_in_government"].eq(1)
    elif direction == "exit":
        event_mask = monthly["previous_status"].eq(1) & monthly["party_in_government"].eq(0)
    else:
        raise ValueError(direction)
    events = monthly.loc[event_mask, ["country_code", "country_party_fe", "month", "month_number"]].copy()
    events = events.rename(columns={"month": "event_month", "month_number": "event_month_number"})
    events["event_id"] = events["country_party_fe"] + "::" + events["event_month"]
    assigned: list[pd.DataFrame] = []
    for party, group in monthly.groupby("country_party_fe", sort=False):
        party_events = events.loc[events["country_party_fe"].eq(party)]
        if party_events.empty:
            continue
        values = group["month_number"].to_numpy()[:, None] - party_events["event_month_number"].to_numpy()[None, :]
        nearest = np.abs(values).argmin(axis=1)
        event_time = values[np.arange(len(group)), nearest]
        part = group.copy()
        part["event_time"] = event_time
        part["event_id"] = party_events.iloc[nearest]["event_id"].to_numpy()
        part = part.loc[part["event_time"].between(-window, window)]
        assigned.append(part[["country_party_fe", "month", "event_time", "event_id"]])
    mapping = pd.concat(assigned, ignore_index=True) if assigned else pd.DataFrame(
        columns=["country_party_fe", "month", "event_time", "event_id"]
    )
    out = data.merge(mapping, on=["country_party_fe", "month"], how="inner", validate="m:1")
    return out, events


def event_study(
    family: str,
    data: pd.DataFrame,
    outcome: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    topic_effect = ["nli_topic"] if family == "old_nli" else []
    for direction in ["entry", "exit"]:
        event_data, events = assign_nearest_event_time(data, direction)
        if events.empty or event_data.empty:
            continue
        terms: list[str] = []
        for event_time in list(range(-12, -1)) + list(range(0, 13)):
            term = f"event_{'m' + str(abs(event_time)) if event_time < 0 else 'p' + str(event_time)}"
            event_data[term] = event_data["event_time"].eq(event_time).astype(float)
            terms.append(term)
        spec = ModelSpec(
            model_id=f"{family}_{direction}_event_study", family=family, outcome=outcome,
            data=event_data, predictors=terms,
            effects=["country_party_fe", "calendar_month_fe", *topic_effect],
            specification="transition event study", sample=f"nearest {direction} transition +/-12 months",
            unit="party-month", notes="Event month -1 is omitted; descriptive within-party estimates only.",
        )
        try:
            fit = fit_model(spec)
        except (ValueError, np.linalg.LinAlgError):
            continue
        lookup = {term: pos for pos, term in enumerate(fit.terms)}
        for event_time in list(range(-12, -1)) + list(range(0, 13)):
            term = f"event_{'m' + str(abs(event_time)) if event_time < 0 else 'p' + str(event_time)}"
            if term not in lookup:
                continue
            vector = np.zeros(len(fit.terms)); vector[lookup[term]] = 1.0
            estimate = float(vector @ fit.beta)
            c = covariance_stats(estimate, vector, fit.cov_country, fit.country_clusters - 1)
            support = fit.used.loc[fit.used["event_time"].eq(event_time)]
            all_rows.append({
                "family": family, "transition": direction, "event_time": event_time,
                "estimate": estimate, "country_se": c["se"], "country_p": c["p"],
                "ci_95_low": c["ci_low"], "ci_95_high": c["ci_high"],
                "observations_at_event_time": len(support),
                "transitions_at_event_time": support["event_id"].nunique(),
                "countries_at_event_time": support["country_code"].nunique(),
                "total_transitions": events["event_id"].nunique(),
                "total_transition_countries": events["country_code"].nunique(),
            })
        lead_terms = [term for term in fit.terms if term.startswith("event_m") and term != "event_m1"]
        if lead_terms:
            indexes = [lookup[term] for term in lead_terms]
            beta = fit.beta[indexes]
            covariance = fit.cov_country[np.ix_(indexes, indexes)]
            statistic = float(beta @ np.linalg.pinv(covariance) @ beta / len(indexes))
            p_value = float(stats.f.sf(statistic, len(indexes), max(fit.country_clusters - 1, 1)))
            tests.append({
                "family": family, "transition": direction,
                "joint_pretrend_f": statistic, "joint_pretrend_df_num": len(indexes),
                "joint_pretrend_df_denom": fit.country_clusters - 1,
                "joint_pretrend_p": p_value, "transitions": events["event_id"].nunique(),
                "countries": events["country_code"].nunique(),
            })
    rows = pd.DataFrame(all_rows); test_frame = pd.DataFrame(tests)
    if not rows.empty:
        plot_dir = output_dir / "plots" / "event_study"; plot_dir.mkdir(parents=True, exist_ok=True)
        for transition, group in rows.groupby("transition"):
            group = group.sort_values("event_time")
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            ax.errorbar(group["event_time"], group["estimate"], yerr=1.96 * group["country_se"], fmt="o-", color="#5a4c9c", ecolor="#aaa2ce", markersize=3)
            ax.axhline(0, color="black", linewidth=0.8); ax.axvline(-1, color="grey", linestyle="--", linewidth=0.8)
            ax.set(xlabel=f"Months from government {transition} (-1 omitted)", ylabel="Within-party outcome contrast", title=f"{family}: government {transition}")
            fig.tight_layout(); fig.savefig(plot_dir / f"{family}_{transition}.png", dpi=220); plt.close(fig)
    return rows, test_frame


def government_identification(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_map = {
        "plda_alignment": frames["plda_alignment"],
        "spearman": frames["spearman"],
        "old_nli": frames["old_nli"],
        "new_macro": frames["new_macro"],
        "new_galtan": frames["new_galtan"],
        "new_combined": frames["new_combined"],
    }
    for family, data in family_map.items():
        monthly = data[["country_code", "speech_party", "country_party_fe", "month", "party_in_government"]].drop_duplicates()
        monthly = monthly.sort_values(["country_party_fe", "month"])
        monthly["previous"] = monthly.groupby("country_party_fe")["party_in_government"].shift()
        for party, group in monthly.groupby("country_party_fe"):
            statuses = group["party_in_government"].dropna().nunique()
            transitions = int((group["party_in_government"].ne(group["previous"]) & group["previous"].notna()).sum())
            rows.append({
                "family": family, "country_code": group["country_code"].iloc[0],
                "country_party_fe": party, "speech_party": group["speech_party"].iloc[0],
                "party_months": len(group), "opposition_months": int(group["party_in_government"].eq(0).sum()),
                "government_months": int(group["party_in_government"].eq(1).sum()),
                "government_switcher": statuses > 1, "observed_status_transitions": transitions,
                "identifies_government_level_under_party_fe": statuses > 1,
            })
    return pd.DataFrame(rows)


def missingness_audit(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variables = [
        "party_seat_share", "party_vote_share", "cabinet_is_coalition",
        "cabinet_has_absolute_majority", "cabinet_caretaker",
        "electoral_proximity", "analysis_volume",
    ]
    for family, data in frames.items():
        if family == "plda_topic":
            data = data.drop_duplicates(["country_code", "speech_party", "month"])
        for (country, status), group in data.groupby(["country_code", "party_in_government"], dropna=False):
            row: dict[str, Any] = {
                "family": family, "country_code": country,
                "party_in_government": status, "rows": len(group),
            }
            for variable in variables:
                row[f"missing_{variable}"] = int(group[variable].isna().sum()) if variable in group else len(group)
                row[f"missing_share_{variable}"] = float(group[variable].isna().mean()) if variable in group else 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def manifesto_project_crosscheck(rank: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "country_code", "speech_party", "doc_key", "manifesto_date",
        "party_seat_share", "party_vote_share", "party_seat_share_parlgov",
        "party_vote_share_parlgov", "party_seat_share_source_audit",
    ]
    current = rank[[column for column in columns if column in rank]].drop_duplicates("doc_key").copy()
    tokens = current["doc_key"].astype(str).str.extract(r"^(?P<mpds_party_id>\d+)_(?P<mpds_date>\d+)$")
    current["mpds_party_id"] = pd.to_numeric(tokens["mpds_party_id"], errors="coerce")
    current["mpds_date"] = pd.to_numeric(tokens["mpds_date"], errors="coerce")
    mp = pd.read_csv(
        BASE_DIR / "data" / "MPDataset_MPDS2025a.csv",
        usecols=["party", "date", "partyname", "partyabbrev", "pervote", "absseat", "totseats"],
        low_memory=False,
    )
    mp = mp.rename(columns={"party": "mpds_party_id", "date": "mpds_date"})
    mp["mp_seat_share"] = pd.to_numeric(mp["absseat"], errors="coerce") / pd.to_numeric(mp["totseats"], errors="coerce")
    mp["mp_vote_share_percent"] = pd.to_numeric(mp["pervote"], errors="coerce")
    out = current.merge(mp, on=["mpds_party_id", "mpds_date"], how="left", validate="m:1")
    out["seat_share_difference_parlgov_minus_mp"] = out["party_seat_share_parlgov"] - out["mp_seat_share"]
    out["vote_share_difference_parlgov_minus_mp"] = out["party_vote_share_parlgov"] - out["mp_vote_share_percent"]
    return out


def diagnostic_path(country: str, topic: str = "Macroeconomics") -> Path | None:
    candidates = list((TEST_DIR / "nli_inconsistency" / country).glob(f"{country}_{topic}*speech_diagnostics.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("MoritzLaurer" not in path.name, len(path.name)))
    return candidates[0]


def sample_flow(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    macro = frames["new_macro"].copy()
    status_map = macro[["country_code", "speech_party", "month", "party_in_government"]].drop_duplicates()
    flow_rows: list[pd.DataFrame] = []
    for country in MACRO_COUNTRIES:
        path = diagnostic_path(country)
        diagnostic_groups: list[pd.DataFrame] = []
        if path is not None:
            use = ["plda_doc_id", "party", "month", "speech_filter_kept"]
            for chunk in pd.read_csv(path, usecols=lambda column: column in use, chunksize=100_000, low_memory=False):
                chunk["speech_filter_kept"] = bool_numeric(chunk["speech_filter_kept"]).fillna(0)
                grouped = chunk.groupby(["party", "month"], as_index=False).agg(
                    speeches_with_topic_assignments=("plda_doc_id", "nunique"),
                    speeches_passing_text_filters=("speech_filter_kept", "sum"),
                )
                diagnostic_groups.append(grouped)
        if diagnostic_groups:
            diagnostic = pd.concat(diagnostic_groups).groupby(["party", "month"], as_index=False).sum()
            diagnostic = diagnostic.rename(columns={"party": "speech_party"})
        else:
            diagnostic = pd.DataFrame(columns=["speech_party", "month", "speeches_with_topic_assignments", "speeches_passing_text_filters"])
        panel = macro.loc[macro["country_code"].eq(country)].copy()
        keep = [
            "speech_party", "month", "n_pairs", "n_speeches", "inconsistency_share",
            "party_in_government", "party_seat_share",
        ]
        panel = panel[[column for column in keep if column in panel]]
        merged = diagnostic.merge(panel, on=["speech_party", "month"], how="outer")
        merged.insert(0, "country_code", country)
        if "party_in_government" not in merged:
            merged = merged.merge(status_map, on=["country_code", "speech_party", "month"], how="left")
        merged["candidate_nli_pairs"] = np.nan
        merged["selected_retrieval_pairs"] = pd.to_numeric(merged.get("n_pairs"), errors="coerce")
        merged["classified_pairs"] = merged["selected_retrieval_pairs"]
        merged["number_relevant_speeches"] = pd.to_numeric(merged.get("n_speeches"), errors="coerce")
        merged["number_unique_speech_sentences"] = np.nan
        merged["party_month_cells"] = merged["inconsistency_share"].notna().astype(int)
        merged["rows_with_outcomes"] = merged["inconsistency_share"].notna().astype(int)
        merged["rows_complete_m1"] = merged[["inconsistency_share", "party_in_government"]].notna().all(axis=1).astype(int)
        merged["rows_complete_m2"] = merged[["inconsistency_share", "party_in_government", "party_seat_share"]].notna().all(axis=1).astype(int)
        flow_rows.append(merged)
    detail = pd.concat(flow_rows, ignore_index=True, sort=False)
    aggregations = {
        "speeches_with_topic_assignments": "sum",
        "speeches_passing_text_filters": "sum",
        "candidate_nli_pairs": "sum",
        "selected_retrieval_pairs": "sum",
        "classified_pairs": "sum",
        "number_relevant_speeches": "sum",
        "number_unique_speech_sentences": "sum",
        "party_month_cells": "sum", "rows_with_outcomes": "sum",
        "rows_complete_m1": "sum", "rows_complete_m2": "sum",
    }
    summary = detail.groupby(["country_code", "party_in_government"], dropna=False).agg(aggregations).reset_index()
    summary["candidate_nli_pairs"] = np.nan
    summary["number_unique_speech_sentences"] = np.nan
    summary.insert(2, "raw_speeches", np.nan)
    summary["raw_speeches_note"] = "Pre-topic raw speech counts are not retained consistently in the cached diagnostics."
    summary["candidate_pairs_note"] = "Pre-top-k candidate pair counts are unavailable; selected retrieval pairs are reported separately."
    summary["unique_sentence_note"] = "Requires a 36+ GB classified-pair scan; not inferred from n_speeches."
    return summary


def stability_table(contrasts: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    merged = contrasts.merge(
        diagnostics[[
            "model_id", "observations", "countries", "parties", "election_cycles",
            "country_clusters", "party_clusters", "fixed_effects", "weight",
            "unit", "unweighted_outcome_mean", "weighted_outcome_mean", "notes",
        ]], on="model_id", how="left",
    )
    merged["estimate_percentage_points"] = np.where(
        merged["family"].isin(["old_nli", "new_macro", "new_galtan", "new_combined"]), 100 * merged["estimate"], np.nan
    )
    merged["display_government_main_is_midcycle_gap"] = merged["contrast"].eq("government_gap_p05")
    merged["covariance_sensitivity"] = (
        merged[["country_p", "party_p", "two_way_p"]].lt(0.05).nunique(axis=1).gt(1)
    )
    return merged


def result_classification(stability: pd.DataFrame, influence: pd.DataFrame, functional: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_ids = {
        "plda_topic": "plda_topic_m1_equal",
        "plda_alignment": "plda_alignment_m1_natural",
        "spearman": "spearman_m1_natural",
        "old_nli": "old_nli_m1_natural",
        "new_macro": "new_macro_m1_natural",
        "new_galtan": "new_galtan_m1_natural",
        "new_combined": "new_combined_m1_natural",
    }
    core_specs = stability.loc[
        stability["specification"].isin(["M1", "M2", "M3-M1", "M3-M2", "M5", "FE robustness", "trend robustness", "switcher robustness"])
    ]
    for family, primary_id in primary_ids.items():
        for contrast in ["opposition_slope", "government_slope", "government_gap_p0", "government_gap_p05", "government_gap_p1"]:
            primary = stability.loc[(stability["model_id"].eq(primary_id)) & stability["contrast"].eq(contrast)]
            if primary.empty:
                continue
            primary = primary.iloc[0]
            comparison = core_specs.loc[(core_specs["family"].eq(family)) & core_specs["contrast"].eq(contrast)]
            sign = np.sign(primary["estimate"])
            same_share = float(np.sign(comparison["estimate"]).eq(sign).mean()) if len(comparison) else math.nan
            infl = influence.loc[(influence["family"].eq(family)) & influence["contrast"].eq(contrast)] if not influence.empty else pd.DataFrame()
            influence_sign_stable = bool(np.sign(infl["estimate"]).eq(sign).all()) if len(infl) else None
            family_functional = functional.loc[functional["family"].eq(family)] if not functional.empty else pd.DataFrame()
            functional_signs: list[float] = []
            if not family_functional.empty:
                if contrast in {"opposition_slope", "government_slope"}:
                    trajectory = (
                        "opposition_relative_midcycle" if contrast == "opposition_slope"
                        else "government_relative_midcycle"
                    )
                    spline = family_functional.loc[
                        family_functional["functional_form"].eq("restricted cubic spline")
                        & family_functional["trajectory"].eq(trajectory)
                        & family_functional["proximity"].isin([0.0, 1.0])
                    ].sort_values("proximity")
                    if len(spline) == 2:
                        functional_signs.append(float(np.sign(spline["estimate"].iloc[1] - spline["estimate"].iloc[0])))
                    quarter_trajectory = (
                        "opposition_relative_q1" if contrast == "opposition_slope"
                        else "government_relative_q1"
                    )
                    quarter = family_functional.loc[
                        family_functional["functional_form"].eq("quarter bins")
                        & family_functional["trajectory"].eq(quarter_trajectory)
                        & family_functional["proximity"].eq(0.875)
                    ]
                    if len(quarter):
                        functional_signs.append(float(np.sign(quarter["estimate"].iloc[0])))
                else:
                    target = {"government_gap_p0": 0.0, "government_gap_p05": 0.5, "government_gap_p1": 1.0}.get(contrast)
                    if target is not None:
                        spline = family_functional.loc[
                            family_functional["functional_form"].eq("restricted cubic spline")
                            & family_functional["trajectory"].eq("government_minus_opposition")
                            & family_functional["proximity"].eq(target)
                        ]
                        if len(spline):
                            functional_signs.append(float(np.sign(spline["estimate"].iloc[0])))
                        quarter_target = {0.0: 0.125, 0.5: 0.625, 1.0: 0.875}[target]
                        quarter = family_functional.loc[
                            family_functional["functional_form"].eq("quarter bins")
                            & family_functional["trajectory"].eq("government_minus_opposition")
                            & family_functional["proximity"].eq(quarter_target)
                        ]
                        if len(quarter):
                            functional_signs.append(float(np.sign(quarter["estimate"].iloc[0])))
            functional_sensitive = (
                bool(any(value != sign for value in functional_signs if value != 0))
                if functional_signs else None
            )
            rows.append({
                "family": family, "contrast": contrast,
                "primary_estimate": primary["estimate"],
                "sign_robustness_share": same_share,
                "sign_robust": bool(same_share >= 0.8) if np.isfinite(same_share) else False,
                "conventionally_significant_country": bool(primary["country_p"] < 0.05),
                "conventionally_significant_wild": bool(primary["wild_p"] < 0.05) if pd.notna(primary["wild_p"]) else False,
                "covariance_sensitive": bool(primary["covariance_sensitivity"]),
                "sample_sensitive": bool(same_share < 1.0) if np.isfinite(same_share) else True,
                "leave_out_sign_stable": influence_sign_stable,
                "functional_form_checked": bool(functional_signs),
                "functional_form_sensitive": functional_sensitive,
                "causal_credibility": "low: associative within-party panel design",
            })
    return pd.DataFrame(rows)


def format_primary_result(stability: pd.DataFrame, model_id: str, contrast: str, percentage: bool = False) -> str:
    row = stability.loc[(stability["model_id"].eq(model_id)) & stability["contrast"].eq(contrast)]
    if row.empty:
        return "not estimated"
    row = row.iloc[0]
    scale = 100 if percentage else 1
    unit = " percentage points" if percentage else ""
    wild = f", wild p={row['wild_p']:.3f}" if pd.notna(row["wild_p"]) else ""
    return (
        f"{scale * row['estimate']:.3f}{unit} "
        f"(country-cluster 95% CI {scale * row['country_ci_low']:.3f} to "
        f"{scale * row['country_ci_high']:.3f}{wild})"
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    out = frame[columns].copy()
    if max_rows is not None:
        out = out.head(max_rows)
    return out.to_markdown(index=False, floatfmt=".4f")


def render_report(
    stability: pd.DataFrame,
    diagnostics: pd.DataFrame,
    classification: pd.DataFrame,
    identification: pd.DataFrame,
    transitions: pd.DataFrame,
    missingness: pd.DataFrame,
    crosscheck: pd.DataFrame,
    event_tests: pd.DataFrame,
    failures: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    switch_summary = (
        identification.groupby("family", as_index=False)
        .agg(parties=("country_party_fe", "nunique"), switchers=("government_switcher", "sum"),
             identifying_party_months=("party_months", lambda values: int(values[identification.loc[values.index, "government_switcher"]].sum())))
    )
    missing_focus = missingness.loc[missingness["country_code"].isin(["AT", "BE", "GR"])]
    cross_matched = crosscheck["mp_seat_share"].notna().sum()
    seat_corr = crosscheck[["party_seat_share_parlgov", "mp_seat_share"]].dropna().corr().iloc[0, 1] if cross_matched > 1 else math.nan
    lines = [
        "# Empirical regression pipeline audit",
        "",
        f"Run output: `{args.output_dir}`  ",
        f"Wild country-cluster replications per primary contrast: {0 if args.skip_bootstrap else args.bootstrap_reps}.",
        "",
        "## Immediate answer",
        "",
        "For the current consensus Macroeconomics outcome, the equal-party-month parsimonious model estimates:",
        "",
        f"- opposition cycle slope: {format_primary_result(stability, 'new_macro_m1_natural', 'opposition_slope', True)};",
        f"- government cycle slope: {format_primary_result(stability, 'new_macro_m1_natural', 'government_slope', True)};",
        f"- government-minus-opposition gap immediately after the election: {format_primary_result(stability, 'new_macro_m1_natural', 'government_gap_p0', True)};",
        f"- the same gap at mid-cycle: {format_primary_result(stability, 'new_macro_m1_natural', 'government_gap_p05', True)};",
        f"- the same gap immediately before the election: {format_primary_result(stability, 'new_macro_m1_natural', 'government_gap_p1', True)}.",
        "",
        "For the completed consensus GAL-TAN outcome, the equivalent model estimates:",
        "",
        f"- opposition cycle slope: {format_primary_result(stability, 'new_galtan_m1_natural', 'opposition_slope', True)};",
        f"- government cycle slope: {format_primary_result(stability, 'new_galtan_m1_natural', 'government_slope', True)};",
        f"- government-minus-opposition gap immediately after the election: {format_primary_result(stability, 'new_galtan_m1_natural', 'government_gap_p0', True)};",
        f"- the same gap at mid-cycle: {format_primary_result(stability, 'new_galtan_m1_natural', 'government_gap_p05', True)};",
        f"- the same gap immediately before the election: {format_primary_result(stability, 'new_galtan_m1_natural', 'government_gap_p1', True)}.",
        "",
        "For the combined two-topic consensus outcome with topic fixed effects, the model estimates:",
        "",
        f"- opposition cycle slope: {format_primary_result(stability, 'new_combined_m1_natural', 'opposition_slope', True)};",
        f"- government cycle slope: {format_primary_result(stability, 'new_combined_m1_natural', 'government_slope', True)};",
        f"- government-minus-opposition gap immediately after the election: {format_primary_result(stability, 'new_combined_m1_natural', 'government_gap_p0', True)};",
        f"- the same gap at mid-cycle: {format_primary_result(stability, 'new_combined_m1_natural', 'government_gap_p05', True)};",
        f"- the same gap immediately before the election: {format_primary_result(stability, 'new_combined_m1_natural', 'government_gap_p1', True)}.",
        "",
        "These are within-party associations conditional on the listed fixed effects. The government main coefficient in displayed centered models is the government-opposition difference at P=0.5, not at P=0.",
        "",
        "## Exact outcomes and units",
        "",
        "| Family | Current outcome | Unit | Aggregation and current weighting |",
        "|---|---|---|---|",
        "| PLDA topic | speech topic salience, related to manifesto salience through interactions | country-party-month-topic | Component-level robustness model; equal weight to each cell with party-month and topic FE. |",
        "| PLDA Jensen-Shannon alignment | `1 - d_JS` on the same normalized eight-topic manifesto and speech vectors | country-party-month | Headline salience model; one harmonized party-month score with equal party-month weighting. |",
        "| Spearman | rank correlation of manifesto and speech topic salience | country-party-month | One correlation per party-month; equal weighting primary, segment/word-volume alternatives. |",
        "| Original NLI | contradiction share or mean contradiction probability | country-party-topic-month | Pair-level labels/probabilities are averaged before monthly regression; old benchmark only. |",
        "| Consensus Macroeconomics | inconsistent-label share or mean probability | country-party-month | Pair-level labels/probabilities are averaged. `n_speeches` is a weight, not a volume control and not a speech-first outcome. |",
        "| Consensus GAL-TAN | inconsistent-label share or mean probability | country-party-month | Pair-level labels/probabilities are averaged separately from Macroeconomics. `n_speeches` is a weight, not a volume control and not a speech-first outcome. |",
        "| Combined consensus topics | inconsistent-label share or mean probability | country-party-topic-month | Macroeconomics and GAL-TAN cells are pooled with topic fixed effects; separate-topic models remain available to show heterogeneity. |",
        "",
        "The classified pair schema contains speech, retrieval-unit, and sentence-span identifiers, but reconstructing a speech-first outcome requires scanning more than 36 GB of repeated-text CSVs. That expensive measurement rebuild was deliberately deferred; it was not silently substituted for the repository's current primary outcome.",
        "",
        "## Variable construction and timing",
        "",
        "- `electoral_proximity` is actual-cycle progress: days since the preceding parliamentary election divided by the interval to the actual next election. The terminal unobserved election is imputed using the country's median observed cycle length. It is displayed as `P - 0.5`.",
        "- Government, caretaker, coalition, and majority status come from ParlGov's cabinet active on `analysis_date`. Upstream `analysis_date` is the midpoint between the earliest and latest speech dates in a party-month, not each individual speech date.",
        f"- The audit identifies {len(transitions):,} cabinet starts across the covered ParlGov histories and flags every sample month containing one. M3 excludes those months and caretaker months.",
        "- ParlGov party seat share is based on the first cabinet-formation parliament composition linked to the preceding election, with the ParlGov election result used as its internal fallback. Where ParlGov remains missing, the audit uses the linked Manifesto Project `absseat/totseats` value and records `party_seat_share_source_audit`; the original ParlGov value is preserved. Vote share is not included jointly with seat share.",
        f"- The Manifesto Project cross-check matched {cross_matched:,} manifesto-party documents; the available seat-share correlation is {seat_corr:.3f}. Differences remain in the machine-readable cross-check and are not zero-coded.",
        "- Coalition, cabinet majority, and caretaker status are not generic M1/M2 controls. M4 uses mutually exclusive party-context categories and therefore changes the estimand.",
        "- Speech volume, relevant speeches, and pair counts are reported separately. Regression weighting is never described as controlling for volume; the log-volume model is labelled conditional.",
        "",
        "## Existing versus revised architecture",
        "",
        "The earlier baseline used country-party and calendar-year fixed effects, automatic seat-share/volume/cabinet controls, volume WLS, and HC1 for inconsistency. The earlier robust script removed bad controls and hand-coded a Webb wild bootstrap, but did not provide CR2, transition treatment, natural/common samples, M1-M5, nonlinear proximity, event studies, full influence checks, sample flow, or multiplicity adjustment.",
        "",
        "The revised pipeline uses statsmodels CR1 covariance for continuity and party/country/two-way sensitivity, and the maintained `wildboottest` package for restricted Webb country-cluster bootstrap tests. Country wild-bootstrap p-values are primary because the electoral clock is shared within only about 15 countries; country CR1 intervals and party/two-way results are supporting.",
        "",
        "## Government-status identification",
        "",
        markdown_table(switch_summary, ["family", "parties", "switchers", "identifying_party_months"]),
        "",
        "Under party fixed effects, the government level coefficient is identified by party-month observations belonging to switchers. Always-government and always-opposition parties do not identify that level contrast, although they can contribute to cycle slopes.",
        "",
        "## Coefficient stability classification",
        "",
        markdown_table(
            classification,
            ["family", "contrast", "primary_estimate", "sign_robustness_share", "sign_robust", "conventionally_significant_wild", "covariance_sensitive", "sample_sensitive", "functional_form_sensitive", "leave_out_sign_stable", "causal_credibility"],
        ),
        "",
        "Sign robustness, conventional significance, covariance sensitivity, sample sensitivity, functional-form sensitivity, and causal credibility are separate claims. No specification is selected because it produces a preferred sign.",
        "",
        "## Missingness and sample flow",
        "",
        "Austria, Belgium, and Greece are shown explicitly below; the complete table covers every country and government status:",
        "",
        markdown_table(
            missing_focus,
            ["family", "country_code", "party_in_government", "rows", "missing_share_party_seat_share", "missing_share_party_vote_share", "missing_share_cabinet_caretaker"],
            max_rows=40,
        ),
        "",
        "`sample_flow.csv` distinguishes topic-assigned speeches, speeches passing text filters, selected retrieval pairs, classified pairs, cells, outcome rows, and M1/M2 complete cases. The cached artifacts do not consistently retain pre-topic raw-speech counts, pre-top-k candidate-pair counts, or unique sentence counts; those fields are present as missing with an explanation rather than invented or silently dropped.",
        "",
        "## Functional form, influence, and transitions",
        "",
        "Quarter-of-cycle and centered natural cubic spline estimates are stored with country-cluster intervals and plotted. Country-year FE, month-of-year FE, party-cycle FE, country-specific trends, non-early cycles, 6/12-month shifted-clock placebos, switcher-only models, leave-one-country-out, and leave-one-cycle-out estimates are separate robustness families.",
        "",
        "Government entry and exit are estimated separately with leads/lags and country-cluster intervals. They are descriptive: a significant joint lead test is evidence against a transition-causal interpretation. Transition support is reported for every event-time coefficient.",
    ]
    if not event_tests.empty:
        lines.extend(["", markdown_table(event_tests, list(event_tests.columns))])
    lines.extend(
        [
            "",
            "## Primary-outcome and multiple-testing status",
            "",
            "The corrected headline salience outcome is one minus Jensen-Shannon distance on the harmonized eight-topic manifesto and speech distributions. PLDA topic responsiveness and Spearman correlation are component-level and rank-order robustness checks. Consensus Macroeconomics and GAL-TAN hard-label shares remain separate inconsistency outcomes, and old NLI remains a benchmark. This hierarchy is an ex-post correction rather than evidence of preregistration; the two consensus topics are not silently pooled.",
            "",
            "Unadjusted estimates and p-values are retained. Benjamini-Hochberg q-values are calculated within outcome family and clearly labelled nested-primary, weighting, fixed-effect, placebo, and other exploratory families.",
            "",
            "## Blocked or deliberately deferred",
            "",
            "- CR2: no maintained CR2 implementation is installed in Python and R/`clubSandwich` is absent. CR2 is not hand-coded. Country wild-cluster inference is the small-G primary method.",
            "- Scheduled-election proximity: the cached panels contain actual elections and an early-election indicator, but not country-specific constitutionally due dates or maximum-term rules. Non-early-cycle results are reported; scheduled timing requires validated institutional data.",
            "- Speech-first classifier aggregation: the classified pair CSVs exceed 36 GB because they repeat long texts. The optional streaming rebuild is measurement work and was not run by default.",
            "- Future-government placebo: government status is highly persistent, so a simple lead is not a defensible negative control and is omitted.",
            "- Blind annotation, classifier calibration, embeddings, PLDA, NLI, OpenAI, and classifier inference are deliberately not rerun.",
            "- Generated-outcome uncertainty is not propagated; doing so requires expensive first-stage resampling.",
            "",
            "## Reproduction",
            "",
            "See `REPRODUCE.md` and `run_manifest.json`. The production entry point is `scripts/causality/empirical_regression_audit.py`; old empirical directories are untouched.",
        ]
    )
    if failures:
        lines.extend(["", "## Models that could not be estimated", "", pd.DataFrame(failures).to_markdown(index=False)])
    return "\n".join(lines) + "\n"


def write_publication_table(stability: pd.DataFrame, output: Path) -> None:
    keep = stability.loc[
        stability["specification"].isin(["M1", "M2", "M3-M1", "M3-M2", "M5", "FE robustness", "trend robustness", "switcher robustness"])
        & stability["contrast"].isin(PRIMARY_CONTRASTS)
    ].copy()
    columns = [
        "family", "model_id", "specification", "sample", "restriction", "contrast",
        "estimate", "country_se", "country_ci_low", "country_ci_high", "country_p", "wild_p",
        "party_p", "two_way_p", "bh_q", "observations", "countries", "parties",
        "election_cycles", "country_clusters", "fixed_effects", "weight", "unit",
    ]
    output.write_text(
        "# Coefficient stability table\n\n" + keep[columns].to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )


def _text_number(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    if number != 0 and abs(number) < 10 ** (-digits):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _text_integer(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return str(int(number)) if np.isfinite(number) else "NA"


def write_root_text_compendium(output_dir: Path, destination: Path) -> None:
    """Render every audited specification and result into one searchable text file."""
    files = {
        "design": "model_samples_and_design.csv",
        "coefficients": "all_coefficients.csv",
        "contrasts": "all_contrasts.csv",
        "functional": "functional_form_effects.csv",
        "influence": "influence_estimates.csv",
        "events": "government_transition_event_studies.csv",
        "pretrends": "government_transition_pretrend_tests.csv",
        "classification": "result_robustness_classification.csv",
    }
    missing = [name for name in files.values() if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot render the text compendium; missing audit files: " + ", ".join(missing)
        )
    tables = {key: pd.read_csv(output_dir / name) for key, name in files.items()}
    design = tables["design"]
    coefficients = tables["coefficients"]
    contrasts = tables["contrasts"]
    functional = tables["functional"]
    influence = tables["influence"]
    events = tables["events"]
    pretrends = tables["pretrends"]
    classification = tables["classification"]

    def get_contrast(model_id: str, name: str) -> pd.Series | None:
        match = contrasts.loc[
            contrasts["model_id"].eq(model_id) & contrasts["contrast"].eq(name)
        ]
        return None if match.empty else match.iloc[0]

    macro_rows = {
        name: get_contrast("new_macro_m1_natural", name)
        for name in [
            "opposition_slope", "government_slope", "government_gap_p0",
            "government_gap_p05", "government_gap_p1",
        ]
    }
    galtan_rows = {
        name: get_contrast("new_galtan_m1_natural", name)
        for name in [
            "opposition_slope", "government_slope", "government_gap_p0",
            "government_gap_p05", "government_gap_p1",
        ]
    }
    combined_rows = {
        name: get_contrast("new_combined_m1_natural", name)
        for name in [
            "opposition_slope", "government_slope", "government_gap_p0",
            "government_gap_p05", "government_gap_p1",
        ]
    }

    def outcome_summary(rows: dict[str, pd.Series | None], name: str, label: str) -> str:
        row = rows[name]
        if row is None:
            return f"- {label}: unavailable"
        return (
            f"- {label}: {_text_number(100 * row['estimate'], 3)} percentage points "
            f"(country p={_text_number(row['country_p'], 3)}, "
            f"wild p={_text_number(row['wild_p'], 3)})."
        )

    lines = [
        "EMPIRICAL REGRESSION MODELS AND RESULTS",
        "Manifesto-parliamentary-speech coherence audit",
        "=" * 78,
        "",
        f"Source audit directory: {output_dir}",
        f"Production script: {BASE_DIR / 'scripts/causality/empirical_regression_audit.py'}",
        "",
        "PURPOSE AND INTERPRETATION",
        "-" * 78,
        "This file is a plain-text compendium of every linear specification, raw",
        "coefficient, requested linear contrast, flexible-form estimate, influence",
        "diagnostic, and government-transition event-study result in the verified audit.",
        "All language is associative: estimates are within-party associations conditional",
        "on the stated fixed effects. They are not causal estimates.",
        "",
        "Electoral proximity P runs from 0 immediately after an election to 1 immediately",
        "before an election. Linear regressions display P_centered = P - 0.5. Therefore the",
        "displayed government main coefficient is the government-opposition difference at",
        "mid-cycle. For share outcomes, multiplying a coefficient by 100 gives percentage",
        "points. Weighting a regression is not equivalent to controlling for speech volume.",
        "",
        "SHORT SUMMARY OF THE VERIFIED RESULTS",
        "-" * 78,
        "1. New consensus Macroeconomics inconsistency (primary current classifier):",
        outcome_summary(macro_rows, "opposition_slope", "Opposition cycle slope"),
        outcome_summary(macro_rows, "government_slope", "Government cycle slope"),
        outcome_summary(macro_rows, "government_gap_p0", "Government-minus-opposition gap at P=0"),
        outcome_summary(macro_rows, "government_gap_p05", "Government-minus-opposition gap at P=0.5"),
        outcome_summary(macro_rows, "government_gap_p1", "Government-minus-opposition gap at P=1"),
        "The linear sign of the government gap is negative across M1/M2, ordinary-period,",
        "weighting, fixed-effect, trend, switcher, and leave-out checks: governing-party",
        "months are generally associated with less measured inconsistency. The primary wild",
        "country-cluster p-values are not below 0.05, and the spline gap changes sign near",
        "P=1, so the government advantage is directional rather than conclusive. Measured",
        "inconsistency rises toward elections, especially for opposition parties, opposite",
        "to the original directional inconsistency hypothesis.",
        "",
        "2. New consensus GAL-TAN inconsistency:",
        outcome_summary(galtan_rows, "opposition_slope", "Opposition cycle slope"),
        outcome_summary(galtan_rows, "government_slope", "Government cycle slope"),
        outcome_summary(galtan_rows, "government_gap_p0", "Government-minus-opposition gap at P=0"),
        outcome_summary(galtan_rows, "government_gap_p05", "Government-minus-opposition gap at P=0.5"),
        outcome_summary(galtan_rows, "government_gap_p1", "Government-minus-opposition gap at P=1"),
        "GAL-TAN is estimated separately from Macroeconomics. Direction and robustness",
        "are reported without selecting the topic that gives the preferred result.",
        "",
        "3. Combined Macroeconomics + GAL-TAN consensus outcome:",
        outcome_summary(combined_rows, "opposition_slope", "Opposition cycle slope"),
        outcome_summary(combined_rows, "government_slope", "Government cycle slope"),
        outcome_summary(combined_rows, "government_gap_p0", "Government-minus-opposition gap at P=0"),
        outcome_summary(combined_rows, "government_gap_p05", "Government-minus-opposition gap at P=0.5"),
        outcome_summary(combined_rows, "government_gap_p1", "Government-minus-opposition gap at P=1"),
        "This pooled model includes topic fixed effects; topic-specific estimates remain",
        "necessary because Macroeconomics and GAL-TAN show different cycle patterns.",
        "",
        "4. Manifesto-speech salience alignment:",
        "Corrected harmonized Jensen-Shannon alignment is the headline salience outcome.",
        "Topic responsiveness and Spearman alignment remain robustness checks. Magnitudes",
        "or significance can weaken under volume weights and ordinary-period restrictions. Government-",
        "opposition alignment gaps are small, measure-dependent, and mostly insignificant.",
        "The audit therefore does not establish that governing parties are more aligned.",
        "",
        "5. Legacy NLI benchmark:",
        "The old NLI outcome shows a stable negative government gap, but it is retained only",
        "as a benchmark because the consensus classifier is the current measurement model.",
        "",
        "6. Inference and causal credibility:",
        "The primary p-values use the maintained wildboottest implementation with Webb",
        "weights clustered by country; country CR1 intervals are supporting. Party and",
        "two-way covariance estimates are sensitivity checks. Entry and exit event studies",
        "have jointly significant pre-transition leads, so their coefficients should not be",
        "given a causal interpretation. CR2 remains unavailable because no maintained Python",
        "implementation or R clubSandwich installation is present.",
        "",
        "SPECIFICATION GLOSSARY",
        "-" * 78,
        "M1: parsimonious centered proximity, government, and interaction model with the",
        "    family-appropriate fixed effects; no automatic contemporaneous volume, coalition,",
        "    majority, or caretaker controls.",
        "M2: M1 plus predetermined seat share. ParlGov is preferred; linked Manifesto Project",
        "    absseat/totseats fills remaining missing values and is source-flagged.",
        "M3-M1/M3-M2: M1/M2 after excluding caretaker and cabinet-transition months.",
        "M4: mutually exclusive cabinet-context categories; this changes the estimand.",
        "M5: equal, volume, square-root, capped, pair-count, minimum-volume, and separately",
        "    labelled conditional log-volume variants.",
        "FE/form robustness: month seasonality, country-year FE, party-cycle FE, country",
        "    trends, early-cycle exclusion, quarter bins, and natural cubic splines.",
        "Placebos/diagnostics: 6- and 12-month shifted clocks, switcher-only models, leave-one-",
        "    country/cycle-out estimates, and separate government-entry/government-exit studies.",
        "",
        f"LINEAR MODEL INVENTORY ({len(design)} SPECIFICATIONS)",
        "=" * 78,
    ]

    for number, (_, row) in enumerate(design.iterrows(), start=1):
        model_id = str(row["model_id"])
        lines.extend([
            "",
            f"MODEL {number:03d}/{len(design):03d}: {model_id}",
            "-" * 78,
            (
                f"Family={row.get('family', 'NA')} | specification={row.get('specification', 'NA')} "
                f"| sample={row.get('sample', 'NA')}"
            ),
            f"Restriction: {row.get('restriction', 'NA')}",
            f"Unit: {row.get('unit', 'NA')} | outcome: {row.get('outcome', 'NA')}",
            f"Predictors: {row.get('predictors', 'NA')}",
            f"Fixed effects: {row.get('fixed_effects', 'NA')}",
            f"Weight: {row.get('weight', 'NA')}",
            f"Primary covariance: {row.get('covariance_primary', 'NA')}",
            (
                f"N={_text_integer(row.get('observations'))}; countries={_text_integer(row.get('countries'))}; "
                f"parties={_text_integer(row.get('parties'))}; cycles={_text_integer(row.get('election_cycles'))}; "
                f"country clusters={_text_integer(row.get('country_clusters'))}; "
                f"party clusters={_text_integer(row.get('party_clusters'))}"
            ),
            (
                f"Government switchers={_text_integer(row.get('government_switchers'))}; "
                f"identifying observations={_text_integer(row.get('government_identifying_observations'))}"
            ),
            (
                f"Outcome mean: unweighted={_text_number(row.get('unweighted_outcome_mean'))}; "
                f"weighted={_text_number(row.get('weighted_outcome_mean'))}"
            ),
        ])
        dropped = row.get("dropped_terms")
        if pd.notna(dropped) and str(dropped).strip():
            lines.append(f"Dropped/collinear terms: {dropped}")
        notes = row.get("notes")
        if pd.notna(notes) and str(notes).strip():
            lines.append(f"Notes: {notes}")

        model_coefficients = coefficients.loc[coefficients["model_id"].eq(model_id)]
        lines.append("Raw coefficients (estimate; country SE/p; party p; two-way p):")
        if model_coefficients.empty:
            lines.append("  [none recorded]")
        for _, result in model_coefficients.iterrows():
            lines.append(
                f"  {result['term']}: b={_text_number(result['estimate'])}; "
                f"SE_country={_text_number(result['country_se'])}; "
                f"p_country={_text_number(result['country_p'])}; "
                f"p_party={_text_number(result['party_p'])}; "
                f"p_two_way={_text_number(result['two_way_p'])}"
            )

        model_contrasts = contrasts.loc[contrasts["model_id"].eq(model_id)]
        lines.append("Reported contrasts (estimate; country 95% CI/p; wild p; BH q):")
        if model_contrasts.empty:
            lines.append("  [none recorded]")
        for _, result in model_contrasts.iterrows():
            lines.append(
                f"  {result['contrast']}: b={_text_number(result['estimate'])}; "
                f"CI=[{_text_number(result['country_ci_low'])}, "
                f"{_text_number(result['country_ci_high'])}]; "
                f"p_country={_text_number(result['country_p'])}; "
                f"p_wild={_text_number(result['wild_p'])}; "
                f"p_party={_text_number(result['party_p'])}; "
                f"p_two_way={_text_number(result['two_way_p'])}; "
                f"BH_q={_text_number(result['bh_q'])}"
            )

    lines.extend([
        "",
        f"FLEXIBLE FUNCTIONAL-FORM RESULTS ({functional['model_id'].nunique()} MODELS)",
        "=" * 78,
        "Each line reports a fitted trajectory or government-opposition gap at the stated P.",
    ])
    for model_id, group in functional.groupby("model_id", sort=False):
        lines.extend(["", f"{model_id} ({group['functional_form'].iloc[0]}):"])
        for _, row in group.iterrows():
            lines.append(
                f"  {row['trajectory']} at P={_text_number(row['proximity'], 3)}: "
                f"estimate={_text_number(row['estimate'])}; "
                f"SE_country={_text_number(row['country_se'])}; "
                f"CI=[{_text_number(row['ci_95_low'])}, {_text_number(row['ci_95_high'])}]; "
                f"p_country={_text_number(row['country_p'])}; N={_text_integer(row['observations'])}; "
                f"countries={_text_integer(row['countries'])}"
            )

    lines.extend([
        "",
        "LEAVE-ONE-OUT INFLUENCE RESULTS",
        "=" * 78,
        "These are complete leave-one-country-out and leave-one-election-cycle-out results.",
    ])
    for _, row in influence.iterrows():
        lines.append(
            f"{row['family']} | {row['influence_type']} | omit={row['omitted_unit']} | "
            f"{row['contrast']}: b={_text_number(row['estimate'])}; "
            f"SE_country={_text_number(row['country_se'])}; p_country={_text_number(row['country_p'])}"
        )

    lines.extend(["", "GOVERNMENT-TRANSITION EVENT STUDIES", "=" * 78])
    for _, row in events.iterrows():
        lines.append(
            f"{row['family']} | {row['transition']} | event month={_text_integer(row['event_time'])}: "
            f"b={_text_number(row['estimate'])}; SE_country={_text_number(row['country_se'])}; "
            f"CI=[{_text_number(row['ci_95_low'])}, {_text_number(row['ci_95_high'])}]; "
            f"p_country={_text_number(row['country_p'])}; "
            f"transitions={_text_integer(row['transitions_at_event_time'])}; "
            f"countries={_text_integer(row['countries_at_event_time'])}"
        )
    lines.extend(["", "Joint tests of pre-transition leads:"])
    for _, row in pretrends.iterrows():
        lines.append(
            f"  {row['family']} | {row['transition']}: F={_text_number(row['joint_pretrend_f'], 3)}; "
            f"df=({_text_integer(row['joint_pretrend_df_num'])}, "
            f"{_text_integer(row['joint_pretrend_df_denom'])}); "
            f"p={_text_number(row['joint_pretrend_p'])}; transitions={_text_integer(row['transitions'])}; "
            f"countries={_text_integer(row['countries'])}"
        )

    lines.extend([
        "",
        "ROBUSTNESS CLASSIFICATION OF PRIMARY CONTRASTS",
        "=" * 78,
    ])
    for _, row in classification.iterrows():
        lines.append(
            f"{row['family']} | {row['contrast']}: primary={_text_number(row['primary_estimate'])}; "
            f"sign robustness={_text_number(row['sign_robustness_share'], 3)}; "
            f"sign robust={row['sign_robust']}; wild significant={row['conventionally_significant_wild']}; "
            f"covariance sensitive={row['covariance_sensitive']}; sample sensitive={row['sample_sensitive']}; "
            f"functional-form sensitive={row['functional_form_sensitive']}; "
            f"leave-out sign stable={row['leave_out_sign_stable']}; "
            f"causal credibility={row['causal_credibility']}"
        )

    lines.extend([
        "",
        "KNOWN LIMITATIONS AND DELIBERATE DEFERRALS",
        "=" * 78,
        "- Upstream cabinet status was assigned at the midpoint of the observed party-month",
        "  speech-date range. M3 excludes every cabinet-transition and caretaker month; exact",
        "  speech-date reassignment requires rebuilding the upstream speech-level panel.",
        "- The current inconsistency outcome averages classified retrieval pairs. A speech-first",
        "  reconstruction requires scanning more than 36 GB of repeated-text pair files.",
        "- Constitutionally scheduled election dates are not stored in the cached panel. The",
        "  original actual-election clock and a non-early-cycle restriction are reported.",
        "- CR2, generated-outcome uncertainty, future-government placebo, and pending blind",
        "  classifier validation are not silently approximated. See AUDIT_REPORT.md for details.",
        "",
        "Machine-readable source tables and plots:",
        str(output_dir),
        "",
    ])
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def package_versions() -> dict[str, str]:
    from importlib.metadata import version
    names = ["numpy", "pandas", "scipy", "statsmodels", "wildboottest", "patsy", "matplotlib"]
    return {name: version(name) for name in names}


def main() -> int:
    args = parser().parse_args()
    if args.bootstrap_reps < 99 and not args.skip_bootstrap:
        raise ValueError("Use at least 99 bootstrap replications; 999+ is recommended.")
    args.output_dir = args.output_dir.resolve()
    args.text_report_path = args.text_report_path.resolve()
    old_roots = {
        (TEST_DIR / "baseline_panel_fe_regression").resolve(),
        (TEST_DIR / "baseline_panel_fe_regression_consensus_macro").resolve(),
        (TEST_DIR / "robust_econometric_models").resolve(),
    }
    if args.output_dir in old_roots:
        raise ValueError("Refusing to overwrite an existing empirical output root.")
    if args.render_text_report_only:
        write_root_text_compendium(args.output_dir, args.text_report_path)
        print(f"Text compendium written: {args.text_report_path}", flush=True)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached panels and ParlGov timing ...", flush=True)
    frames, base, transitions = load_data()
    transitions.to_csv(args.output_dir / "cabinet_transition_months.csv", index=False)
    missing = missingness_audit(frames); missing.to_csv(args.output_dir / "missingness_by_country_status.csv", index=False)
    identification = government_identification(frames); identification.to_csv(args.output_dir / "government_identification.csv", index=False)
    crosscheck = manifesto_project_crosscheck(frames["spearman"]); crosscheck.to_csv(args.output_dir / "seat_vote_share_crosscheck.csv", index=False)

    specifications = build_specs(frames, args.quick)
    all_coefficients: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    fits: dict[str, FitResult] = {}
    failures: list[dict[str, str]] = []
    for number, spec in enumerate(specifications, start=1):
        print(f"[{number}/{len(specifications)}] {spec.model_id}", flush=True)
        try:
            fit = fit_model(spec)
            coefficients, contrasts, diagnostic = model_rows(
                fit, args.bootstrap_reps, args.seed, args.skip_bootstrap
            )
            fits[spec.model_id] = fit
            all_coefficients.extend(coefficients); all_contrasts.extend(contrasts); all_diagnostics.append(diagnostic)
        except Exception as error:
            failures.append({"model_id": spec.model_id, "error": f"{type(error).__name__}: {error}"})
            print(f"  skipped: {error}", flush=True)

    functional_rows: list[dict[str, Any]] = []
    if not args.quick:
        for family, outcome in [
            ("plda_alignment", "alignment_score"), ("spearman", "spearman_rho"),
            ("old_nli", "inconsistency_share"), ("new_macro", "inconsistency_share"),
            ("new_galtan", "inconsistency_share"),
            ("new_combined", "inconsistency_share"),
        ]:
            data = frames[family]
            if family == "spearman":
                data = data.loc[data["rank_valid"].fillna(False)].copy()
            try:
                functional_fits, rows = functional_form_models(family, data, outcome)
                functional_rows.extend(rows)
                for fit in functional_fits:
                    coefficients, _, diagnostic = model_rows(fit, args.bootstrap_reps, args.seed, True)
                    all_coefficients.extend(coefficients); all_diagnostics.append(diagnostic); fits[fit.spec.model_id] = fit
            except Exception as error:
                failures.append({"model_id": f"{family}_functional_forms", "error": f"{type(error).__name__}: {error}"})

    coefficients = pd.DataFrame(all_coefficients)
    contrasts = add_qvalues(pd.DataFrame(all_contrasts))
    diagnostics = pd.DataFrame(all_diagnostics)
    functional = pd.DataFrame(functional_rows)
    stability = stability_table(contrasts, diagnostics)
    coefficients.to_csv(args.output_dir / "all_coefficients.csv", index=False)
    contrasts.to_csv(args.output_dir / "all_contrasts.csv", index=False)
    diagnostics.to_csv(args.output_dir / "model_samples_and_design.csv", index=False)
    functional.to_csv(args.output_dir / "functional_form_effects.csv", index=False)
    stability.to_csv(args.output_dir / "coefficient_stability.csv", index=False)
    if not functional.empty:
        plot_functional_effects(functional, args.output_dir)

    primary_ids = {
        "plda_topic": "plda_topic_m1_equal", "plda_alignment": "plda_alignment_m1_natural",
        "spearman": "spearman_m1_natural", "old_nli": "old_nli_m1_natural",
        "new_macro": "new_macro_m1_natural",
        "new_galtan": "new_galtan_m1_natural",
        "new_combined": "new_combined_m1_natural",
    }
    primary_fits = {family: fits[model_id] for family, model_id in primary_ids.items() if model_id in fits}
    print("Running influence diagnostics ...", flush=True)
    influence = influence_analysis(primary_fits, args.quick)
    influence.to_csv(args.output_dir / "influence_estimates.csv", index=False)
    plot_influence(influence, args.output_dir)

    event_rows: list[pd.DataFrame] = []; event_test_rows: list[pd.DataFrame] = []
    if not args.quick:
        for family, outcome in [
            ("spearman", "spearman_rho"),
            ("new_macro", "inconsistency_share"),
            ("new_galtan", "inconsistency_share"),
        ]:
            data = frames[family]
            if family == "spearman":
                data = data.loc[data["rank_valid"].fillna(False)].copy()
            rows, tests = event_study(family, data, outcome, args.output_dir)
            event_rows.append(rows); event_test_rows.append(tests)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    event_tests = pd.concat(event_test_rows, ignore_index=True) if event_test_rows else pd.DataFrame()
    events.to_csv(args.output_dir / "government_transition_event_studies.csv", index=False)
    event_tests.to_csv(args.output_dir / "government_transition_pretrend_tests.csv", index=False)

    if not args.skip_sample_flow:
        print("Scanning cached speech diagnostics for sample flow ...", flush=True)
        flow = sample_flow(frames)
    else:
        flow = pd.DataFrame()
    flow.to_csv(args.output_dir / "sample_flow.csv", index=False)

    classification = result_classification(stability, influence, functional)
    classification.to_csv(args.output_dir / "result_robustness_classification.csv", index=False)
    write_publication_table(stability, args.output_dir / "COEFFICIENT_STABILITY_TABLE.md")
    (args.output_dir / "AUDIT_REPORT.md").write_text(
        render_report(
            stability, diagnostics, classification, identification, transitions,
            missing, crosscheck, event_tests, failures, args,
        ), encoding="utf-8",
    )
    (args.output_dir / "REPRODUCE.md").write_text(
        "# Reproduce the revised empirical audit\n\n"
        "From the repository root:\n\n"
        "```bash\n"
        "make empirical_regression_audit AUDIT_BOOTSTRAP_REPS=999\n"
        "```\n\n"
        "This exactly reproduces the verified audit. For publication-strength bootstrap precision, "
        "rerun with `AUDIT_BOOTSTRAP_REPS=4999`; bootstrap p-values will then differ slightly.\n\n"
        "Fast smoke verification:\n\n"
        "```bash\n"
        "make empirical_regression_audit_quick\n"
        "```\n\n"
        "The pipeline reads cached PLDA/NLI/classifier panels and does not invoke NLP or APIs.\n",
        encoding="utf-8",
    )
    manifest = {
        "command": "python scripts/causality/empirical_regression_audit.py",
        "output_dir": str(args.output_dir), "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_skipped": args.skip_bootstrap, "bootstrap_package": "wildboottest",
        "bootstrap_weight": "Webb six-point", "bootstrap_type": "11, null imposed",
        "primary_covariance": "country wild-cluster bootstrap p-values; country CR1 t(G-1) intervals",
        "supporting_covariance": "country-party CR1; two-way country-party/country-month CR1",
        "cr2_status": "blocked: maintained Python package unavailable and R clubSandwich absent",
        "package_versions": package_versions(), "models_completed": sorted(fits),
        "models_failed": failures, "source_roots": [
            str(TEST_DIR / "plda_salience/substantive"), str(TEST_DIR / "nli_regression_panel"),
            str(TEST_DIR / "nli_consensus_regression_panel"),
        ],
        "expensive_measurement_stages_rerun": False,
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.quick:
        write_root_text_compendium(args.output_dir, args.text_report_path)

    # Verification invariants.
    if not contrasts.empty:
        observed = contrasts["wild_p"].dropna()
        if not observed.between(0, 1).all():
            raise AssertionError("Wild-bootstrap p-values outside [0,1].")
    for family in ["plda_alignment", "spearman", "old_nli", "new_macro", "new_galtan", "new_combined"]:
        m1 = diagnostics.loc[diagnostics["model_id"].eq(f"{family}_m1_common"), "observations"]
        m2 = diagnostics.loc[diagnostics["model_id"].eq(f"{family}_m2_common"), "observations"]
        if len(m1) and len(m2) and int(m1.iloc[0]) != int(m2.iloc[0]):
            raise AssertionError(f"{family}: M1/M2 common samples differ.")
    print(f"Audit complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
