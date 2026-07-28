from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_PANEL_ROOT = TEST_OUTPUT_DIR / "plda_salience" / "substantive"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_salience_regression" / "substantive"
DEFAULT_COUNTRIES = [
    "CZ",
    "EE",
    "LV",
    "PL",
    "DK",
    "FI",
    "NO",
    "SE",
    "AT",
    "BE",
    "NL",
    "ES",
    "IT",
    "PT",
    "GR",
]
RANK_CONTROLS = [
    "party_seat_share",
    "log1p_speech_words",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]


@dataclass(frozen=True)
class AbsorbedRegressionResult:
    model: str
    outcome: str
    predictors: list[str]
    absorbed_effects: list[str]
    coefficients: pd.DataFrame
    nobs: int
    n_clusters: int
    n_parameters: int
    within_r_squared: float
    adjusted_within_r_squared: float
    weighted_rmse: float
    weight_column: str
    dropped_terms: list[str]
    absorption_iterations: int
    covariance: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate topic-level responsiveness and Spearman alignment models "
            "with absorbed fixed effects and country-party clustered inference."
        )
    )
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--sample",
        choices=["prior", "all"],
        default="prior",
        help="The primary 'prior' sample excludes future-manifesto fallback links.",
    )
    parser.add_argument(
        "--observed-election-only",
        action="store_true",
        help="Keep rows whose next-election boundary is observed rather than imputed.",
    )
    parser.add_argument(
        "--current-basis-only",
        action="store_true",
        help="Exclude legacy topic panels that do not match the current grid metadata.",
    )
    parser.add_argument("--min-speech-words", type=float, default=0.0)
    parser.add_argument(
        "--weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use party-month speech-word precision weights. Enabled by default.",
    )
    parser.add_argument("--absorb-tolerance", type=float, default=1e-10)
    parser.add_argument("--absorb-max-iterations", type=int, default=1000)
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


def load_panels(panel_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_path = panel_root / "data" / "topic_salience_panel.csv"
    rank_path = panel_root / "data" / "rank_salience_panel.csv"
    if not topic_path.exists() or not rank_path.exists():
        raise FileNotFoundError(
            "Salience panels are missing. Run scripts/metrics/plda_salience_panels.py first. "
            f"Expected {topic_path} and {rank_path}."
        )
    return (
        pd.read_csv(topic_path, low_memory=False),
        pd.read_csv(rank_path, low_memory=False),
    )


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def filter_sample(
    data: pd.DataFrame,
    countries: list[str],
    sample: str,
    observed_election_only: bool,
    current_basis_only: bool,
    min_speech_words: float,
) -> pd.DataFrame:
    out = data.loc[data["country_code"].astype(str).str.upper().isin(countries)].copy()
    if sample == "prior":
        out = out.loc[bool_mask(out["prior_manifesto"])].copy()
    if observed_election_only:
        out = out.loc[bool_mask(out["observed_next_election"])].copy()
    if current_basis_only:
        out = out.loc[bool_mask(out["topic_basis_matches_current_grid"])].copy()
    words = pd.to_numeric(out.get("speech_volume_words", 0), errors="coerce").fillna(0)
    out = out.loc[words.ge(min_speech_words)].copy()
    if "rank_valid" in out.columns:
        out = out.loc[bool_mask(out["rank_valid"])].copy()
    if {"party_in_government", "proximity_centered"}.issubset(out.columns):
        out["gov_x_proximity_centered"] = (
            pd.to_numeric(out["party_in_government"], errors="coerce")
            * pd.to_numeric(out["proximity_centered"], errors="coerce")
        )
    out["calendar_year_fe"] = out["month"].astype(str).str.slice(0, 4)
    return out.reset_index(drop=True)


def regression_weights(data: pd.DataFrame, weighted: bool) -> pd.Series:
    if not weighted:
        return pd.Series(1.0, index=data.index, name="unit_weight")
    weights = pd.to_numeric(data["speech_volume_words"], errors="coerce")
    weights = weights.where(weights.gt(0))
    weights.name = "speech_volume_words"
    return weights


def factor_codes(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series.astype(str), sort=False)
    if (codes < 0).any():
        raise ValueError("Fixed-effect or cluster identifiers contain missing values.")
    return codes.astype(np.int64, copy=False)


def demean_one_effect(
    values: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
) -> None:
    group_count = int(codes.max()) + 1
    denominator = np.bincount(codes, weights=weights, minlength=group_count)
    for column in range(values.shape[1]):
        numerator = np.bincount(
            codes,
            weights=weights * values[:, column],
            minlength=group_count,
        )
        means = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )
        values[:, column] -= means[codes]


def absorb_effects(
    values: np.ndarray,
    effect_codes: list[np.ndarray],
    weights: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int]:
    absorbed = np.asarray(values, dtype=float).copy()
    if not effect_codes:
        return absorbed, 0
    scale = max(float(np.max(np.abs(absorbed))), 1.0)
    for iteration in range(1, max_iterations + 1):
        before = absorbed.copy()
        for codes in effect_codes:
            demean_one_effect(absorbed, codes, weights)
        change = float(np.max(np.abs(absorbed - before)))
        if change <= tolerance * scale:
            return absorbed, iteration
    raise RuntimeError(
        f"Fixed-effect absorption did not converge after {max_iterations} iterations "
        f"(tolerance={tolerance})."
    )


def sequential_full_rank_columns(
    x_weighted: np.ndarray,
    terms: list[str],
    tolerance: float = 1e-10,
) -> tuple[list[int], list[str]]:
    kept: list[int] = []
    dropped: list[str] = []
    for column, term in enumerate(terms):
        candidate = x_weighted[:, column]
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= tolerance:
            dropped.append(term)
            continue
        if kept:
            basis = x_weighted[:, kept]
            projection, *_ = np.linalg.lstsq(basis, candidate, rcond=None)
            residual_norm = float(np.linalg.norm(candidate - basis @ projection))
            if residual_norm <= tolerance * max(candidate_norm, 1.0):
                dropped.append(term)
                continue
        kept.append(column)
    return kept, dropped


def fit_absorbed_wls(
    data: pd.DataFrame,
    model: str,
    outcome: str,
    predictors: list[str],
    absorbed_effects: list[str],
    weights: pd.Series,
    cluster_column: str,
    tolerance: float,
    max_iterations: int,
) -> AbsorbedRegressionResult:
    required = [outcome, *predictors, *absorbed_effects, cluster_column]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"{model}: missing model columns: {missing}")

    numeric = data[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1) & weights.notna() & weights.gt(0)
    for column in [*absorbed_effects, cluster_column]:
        mask &= data[column].notna()
    used = data.loc[mask].copy()
    numeric = numeric.loc[mask]
    weight_array = weights.loc[mask].to_numpy(dtype=float, copy=True)
    if len(used) <= len(predictors) + 1:
        raise ValueError(f"{model}: not enough complete observations ({len(used)}).")
    weight_array /= float(np.mean(weight_array))

    y = numeric[outcome].to_numpy(dtype=float)
    x = numeric[predictors].to_numpy(dtype=float)
    combined = np.column_stack([y, x])
    effect_codes = [factor_codes(used[column]) for column in absorbed_effects]
    within, iterations = absorb_effects(
        combined,
        effect_codes=effect_codes,
        weights=weight_array,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    y_within = within[:, 0]
    x_within = within[:, 1:]
    sqrt_weight = np.sqrt(weight_array)
    y_weighted = y_within * sqrt_weight
    x_weighted = x_within * sqrt_weight[:, None]
    kept, dropped = sequential_full_rank_columns(x_weighted, predictors)
    if not kept:
        raise ValueError(f"{model}: every predictor was absorbed or collinear.")
    terms = [predictors[index] for index in kept]
    x_within = x_within[:, kept]
    x_weighted = x_weighted[:, kept]

    beta, *_ = np.linalg.lstsq(x_weighted, y_weighted, rcond=None)
    residual = y_within - x_within @ beta
    nobs = len(y_within)
    n_parameters = len(terms)
    cluster_codes = factor_codes(used[cluster_column])
    n_clusters = int(cluster_codes.max()) + 1
    if n_clusters < 2:
        raise ValueError(f"{model}: clustered inference needs at least two clusters.")

    bread = np.linalg.pinv(x_weighted.T @ x_weighted)
    observation_scores = x_within * (weight_array * residual)[:, None]
    cluster_scores = np.zeros((n_clusters, n_parameters), dtype=float)
    np.add.at(cluster_scores, cluster_codes, observation_scores)
    meat = cluster_scores.T @ cluster_scores
    correction = (n_clusters / (n_clusters - 1)) * ((nobs - 1) / (nobs - n_parameters))
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_stat = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan),
        where=standard_error > 0,
    )
    cluster_df = n_clusters - 1
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=cluster_df)
    critical = stats.t.ppf(0.975, df=cluster_df)
    coefficients = pd.DataFrame(
        {
            "model": model,
            "term": terms,
            "estimate": beta,
            "std_error": standard_error,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_95_low": beta - critical * standard_error,
            "ci_95_high": beta + critical * standard_error,
            "cluster_df": cluster_df,
        }
    )

    weighted_sse = float(np.sum(weight_array * residual**2))
    weighted_sst = float(np.sum(weight_array * y_within**2))
    within_r_squared = 1.0 - weighted_sse / weighted_sst if weighted_sst > 0 else math.nan
    adjusted = (
        1.0 - (1.0 - within_r_squared) * (nobs - 1) / (nobs - n_parameters)
        if nobs > n_parameters and np.isfinite(within_r_squared)
        else math.nan
    )
    rmse = math.sqrt(weighted_sse / float(np.sum(weight_array)))
    return AbsorbedRegressionResult(
        model=model,
        outcome=outcome,
        predictors=terms,
        absorbed_effects=absorbed_effects,
        coefficients=coefficients,
        nobs=nobs,
        n_clusters=n_clusters,
        n_parameters=n_parameters,
        within_r_squared=within_r_squared,
        adjusted_within_r_squared=adjusted,
        weighted_rmse=rmse,
        weight_column=str(weights.name),
        dropped_terms=dropped,
        absorption_iterations=iterations,
        covariance=covariance,
    )


def model_specs() -> list[dict[str, object]]:
    return [
        {
            "model": "topic_requested_h1",
            "panel": "topic",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "electoral_proximity",
                "manifesto_x_proximity",
            ],
            "effects": ["country_party_fe", "topic"],
        },
        {
            "model": "topic_requested_joint",
            "panel": "topic",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "proximity_centered",
                "party_in_government",
                "manifesto_x_proximity_centered",
                "manifesto_x_government",
                "gov_x_proximity_centered",
                "manifesto_x_proximity_centered_x_government",
            ],
            "effects": ["country_party_fe", "topic"],
        },
        {
            "model": "topic_compositional",
            "panel": "topic",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "manifesto_x_proximity_centered",
                "manifesto_x_government",
                "manifesto_x_proximity_centered_x_government",
            ],
            "effects": ["party_month_fe", "topic"],
        },
        {
            "model": "topic_common_shocks",
            "panel": "topic",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "manifesto_x_proximity_centered",
                "manifesto_x_government",
                "manifesto_x_proximity_centered_x_government",
            ],
            "effects": ["party_month_fe", "country_month_topic_fe"],
        },
        {
            "model": "rank_requested_h1",
            "panel": "rank",
            "outcome": "spearman_rho",
            "predictors": ["electoral_proximity"],
            "effects": ["country_party_fe"],
        },
        {
            "model": "rank_requested_joint",
            "panel": "rank",
            "outcome": "spearman_rho",
            "predictors": [
                "proximity_centered",
                "party_in_government",
                "gov_x_proximity_centered",
            ],
            "effects": ["country_party_fe"],
        },
        {
            "model": "rank_adjusted",
            "panel": "rank",
            "outcome": "spearman_rho",
            "predictors": [
                "proximity_centered",
                "party_in_government",
                "gov_x_proximity_centered",
                *RANK_CONTROLS,
            ],
            "effects": ["country_party_fe", "calendar_year_fe"],
        },
    ]


def contrast(
    result: AbsorbedRegressionResult,
    name: str,
    hypothesis: str,
    expected_sign: str,
    weights: dict[str, float],
) -> dict[str, object]:
    terms = result.coefficients["term"].tolist()
    vector = np.array([weights.get(term, 0.0) for term in terms], dtype=float)
    if not np.any(vector):
        return {
            "model": result.model,
            "contrast": name,
            "hypothesis": hypothesis,
            "expected_sign": expected_sign,
            "estimate": math.nan,
            "std_error": math.nan,
            "t_stat": math.nan,
            "p_value_two_sided": math.nan,
            "p_value_one_sided": math.nan,
            "supported_5pct_one_sided": False,
            "terms": str(weights),
        }
    estimates = result.coefficients["estimate"].to_numpy(dtype=float)
    estimate = float(vector @ estimates)
    variance = float(vector @ result.covariance @ vector)
    standard_error = math.sqrt(max(variance, 0.0))
    t_stat = estimate / standard_error if standard_error > 0 else math.nan
    df = result.n_clusters - 1
    two_sided = float(2 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else math.nan
    if expected_sign == "positive":
        one_sided = float(stats.t.sf(t_stat, df=df))
    elif expected_sign == "negative":
        one_sided = float(stats.t.cdf(t_stat, df=df))
    else:
        raise ValueError(f"Unsupported expected sign: {expected_sign}")
    return {
        "model": result.model,
        "contrast": name,
        "hypothesis": hypothesis,
        "expected_sign": expected_sign,
        "estimate": estimate,
        "std_error": standard_error,
        "t_stat": t_stat,
        "p_value_two_sided": two_sided,
        "p_value_one_sided": one_sided,
        "supported_5pct_one_sided": bool(one_sided < 0.05),
        "terms": str(weights),
    }


def hypothesis_contrasts(result: AbsorbedRegressionResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if result.model == "topic_requested_h1":
        rows.append(
            contrast(
                result,
                "manifesto_linkage",
                "General manifesto-speech topic linkage",
                "positive",
                {"manifesto_salience": 1.0},
            )
        )
        rows.append(
            contrast(
                result,
                "topic_h1_cycle",
                "H1a: manifesto responsiveness strengthens near elections",
                "positive",
                {"manifesto_x_proximity": 1.0},
            )
        )
    if result.model in {"topic_requested_joint", "topic_compositional", "topic_common_shocks"}:
        rows.extend(
            [
                contrast(
                    result,
                    "topic_h1_opposition",
                    "H1a cycle slope for opposition parties",
                    "positive",
                    {"manifesto_x_proximity_centered": 1.0},
                ),
                contrast(
                    result,
                    "topic_h1_government",
                    "H1a cycle slope for governing parties",
                    "positive",
                    {
                        "manifesto_x_proximity_centered": 1.0,
                        "manifesto_x_proximity_centered_x_government": 1.0,
                    },
                ),
                contrast(
                    result,
                    "topic_h2_government_p0",
                    "H2a government-minus-opposition responsiveness at proximity 0",
                    "negative",
                    {
                        "manifesto_x_government": 1.0,
                        "manifesto_x_proximity_centered_x_government": -0.5,
                    },
                ),
                contrast(
                    result,
                    "topic_h2_government_p05",
                    "H2a government-minus-opposition responsiveness at proximity 0.5",
                    "negative",
                    {"manifesto_x_government": 1.0},
                ),
                contrast(
                    result,
                    "topic_h2_government_p1",
                    "H2a government-minus-opposition responsiveness at proximity 1",
                    "negative",
                    {
                        "manifesto_x_government": 1.0,
                        "manifesto_x_proximity_centered_x_government": 0.5,
                    },
                ),
            ]
        )
    if result.model == "rank_requested_h1":
        rows.append(
            contrast(
                result,
                "rank_h1_cycle",
                "H1a: Spearman alignment increases near elections",
                "positive",
                {"electoral_proximity": 1.0},
            )
        )
    if result.model in {"rank_requested_joint", "rank_adjusted"}:
        rows.extend(
            [
                contrast(
                    result,
                    "rank_h1_opposition",
                    "H1a cycle slope for opposition parties",
                    "positive",
                    {"proximity_centered": 1.0},
                ),
                contrast(
                    result,
                    "rank_h1_government",
                    "H1a cycle slope for governing parties",
                    "positive",
                    {"proximity_centered": 1.0, "gov_x_proximity_centered": 1.0},
                ),
                contrast(
                    result,
                    "rank_h2_government_p0",
                    "H2a government-minus-opposition rank alignment at proximity 0",
                    "negative",
                    {"party_in_government": 1.0, "gov_x_proximity_centered": -0.5},
                ),
                contrast(
                    result,
                    "rank_h2_government_p05",
                    "H2a government-minus-opposition rank alignment at proximity 0.5",
                    "negative",
                    {"party_in_government": 1.0},
                ),
                contrast(
                    result,
                    "rank_h2_government_p1",
                    "H2a government-minus-opposition rank alignment at proximity 1",
                    "negative",
                    {"party_in_government": 1.0, "gov_x_proximity_centered": 0.5},
                ),
            ]
        )
    return rows


def pstars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def compact_table(results: list[AbsorbedRegressionResult]) -> str:
    lines: list[str] = []
    for result in results:
        lines.extend(
            [
                result.model,
                f"Outcome: {result.outcome}",
                f"Absorbed FE: {', '.join(result.absorbed_effects)}",
                "term\testimate\tclustered_se\tp_value",
            ]
        )
        for row in result.coefficients.itertuples(index=False):
            lines.append(
                f"{row.term}\t{row.estimate:.6f}{pstars(row.p_value)}\t"
                f"{row.std_error:.6f}\t{row.p_value:.4g}"
            )
        lines.extend(
            [
                f"Observations\t{result.nobs}",
                f"Country-party clusters\t{result.n_clusters}",
                f"Within R2\t{result.within_r_squared:.4f}",
                f"Weighted RMSE\t{result.weighted_rmse:.4f}",
                f"Dropped/absorbed terms\t{', '.join(result.dropped_terms) or 'none'}",
                "",
            ]
        )
    lines.append(
        "Notes: CR1 country-party clustered standard errors; t reference distribution uses G-1 df."
    )
    lines.append("* p<0.10, ** p<0.05, *** p<0.01 (two-sided).")
    return "\n".join(lines) + "\n"


def sample_summary(rank_data: pd.DataFrame) -> pd.DataFrame:
    return (
        rank_data.groupby("country_code", as_index=False)
        .agg(
            party_months=("month", "size"),
            parties=("speech_party", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            mean_spearman_rho=("spearman_rho", "mean"),
            mean_electoral_proximity=("electoral_proximity", "mean"),
            government_share=("party_in_government", "mean"),
            mean_speech_words=("speech_volume_words", "mean"),
        )
    )


def model_metadata(result: AbsorbedRegressionResult) -> dict[str, object]:
    return {
        "model": result.model,
        "outcome": result.outcome,
        "predictors": " + ".join(result.predictors),
        "absorbed_effects": " + ".join(result.absorbed_effects),
        "nobs": result.nobs,
        "country_party_clusters": result.n_clusters,
        "n_parameters": result.n_parameters,
        "within_r_squared": result.within_r_squared,
        "adjusted_within_r_squared": result.adjusted_within_r_squared,
        "weighted_rmse": result.weighted_rmse,
        "weight_column": result.weight_column,
        "dropped_terms": "; ".join(result.dropped_terms),
        "absorption_iterations": result.absorption_iterations,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_speech_words < 0:
        raise ValueError("--min-speech-words must be nonnegative.")
    countries = normalize_countries(args.countries)
    topic_raw, rank_raw = load_panels(args.panel_root)
    topic_data = filter_sample(
        topic_raw,
        countries=countries,
        sample=args.sample,
        observed_election_only=args.observed_election_only,
        current_basis_only=args.current_basis_only,
        min_speech_words=args.min_speech_words,
    )
    rank_data = filter_sample(
        rank_raw,
        countries=countries,
        sample=args.sample,
        observed_election_only=args.observed_election_only,
        current_basis_only=args.current_basis_only,
        min_speech_words=args.min_speech_words,
    )
    if topic_data.empty or rank_data.empty:
        raise ValueError("The selected empirical sample is empty.")

    results: list[AbsorbedRegressionResult] = []
    for specification in model_specs():
        data = topic_data if specification["panel"] == "topic" else rank_data
        weights = regression_weights(data, args.weighted)
        result = fit_absorbed_wls(
            data=data,
            model=str(specification["model"]),
            outcome=str(specification["outcome"]),
            predictors=list(specification["predictors"]),
            absorbed_effects=list(specification["effects"]),
            weights=weights,
            cluster_column="country_party_fe",
            tolerance=args.absorb_tolerance,
            max_iterations=args.absorb_max_iterations,
        )
        results.append(result)
        print(
            f"{result.model}: n={result.nobs:,}, clusters={result.n_clusters}, "
            f"within R2={result.within_r_squared:.3f}"
        )

    output_dir = args.output_dir
    coefficient_dir = output_dir / "coefficients"
    coefficient_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        result.coefficients.to_csv(coefficient_dir / f"{result.model}.csv", index=False)
    pd.DataFrame([model_metadata(result) for result in results]).to_csv(
        output_dir / "model_summary.csv", index=False
    )
    contrasts = pd.DataFrame(
        [row for result in results for row in hypothesis_contrasts(result)]
    )
    contrasts.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    sample_summary(rank_data).to_csv(output_dir / "sample_country_summary.csv", index=False)
    (output_dir / "regression_table.txt").write_text(
        compact_table(results), encoding="utf-8"
    )

    government_switchers = int(
        rank_data.groupby("country_party_fe")["party_in_government"].nunique().gt(1).sum()
    )
    manifest = [
        "PLDA topic-level and rank-order salience regressions",
        "",
        f"Panel root: {args.panel_root}",
        f"Countries: {', '.join(sorted(rank_data['country_code'].unique()))}",
        f"Sample: {args.sample}",
        f"Observed next-election only: {args.observed_election_only}",
        f"Current topic-basis only: {args.current_basis_only}",
        f"Minimum speech words: {args.min_speech_words:g}",
        f"Speech-volume weighted: {args.weighted}",
        f"Party-month observations: {len(rank_data)}",
        f"Topic-level observations before model-specific missingness: {len(topic_data)}",
        f"Country-party clusters: {rank_data['country_party_fe'].nunique()}",
        f"Government-status switching clusters: {government_switchers}",
        "Covariance: CR1 clustered by country-party; t df = clusters - 1",
        "",
        "Key files:",
        str(output_dir / "regression_table.txt"),
        str(output_dir / "hypothesis_tests.csv"),
        str(output_dir / "model_summary.csv"),
        str(output_dir / "sample_country_summary.csv"),
    ]
    (output_dir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Wrote regression outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
