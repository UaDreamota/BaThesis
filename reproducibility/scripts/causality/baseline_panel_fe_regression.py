from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_PLDA_PANEL_DIR = TEST_OUTPUT_DIR / "plda_regression_panel"
DEFAULT_NLI_PANEL_DIR = TEST_OUTPUT_DIR / "nli_regression_panel"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "baseline_panel_fe_regression"
DEFAULT_NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from causality import plda_linear_regression as ols


CLUSTERS = {
    "cee": ["CZ", "EE", "LV", "PL"],
    "nordics": ["DK", "FI", "NO", "SE"],
    "west": ["AT", "BE", "NL"],
    "south": ["ES", "IT", "PT", "GR"],
}
DEFAULT_NLI_TOPICS = ["Macroeconomics", "Gal_Tan"]
MAIN_TERMS = [
    "electoral_proximity",
    "party_in_government",
    "gov_x_electoral_proximity",
]
PLDA_CONTROLS = [
    "party_seat_share",
    "log1p_speech_words",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]
NLI_CONTROLS = [
    "party_seat_share",
    "log1p_nli_speeches",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]
TERM_LABELS = {
    "electoral_proximity": "Electoral proximity",
    "party_in_government": "Government party",
    "gov_x_electoral_proximity": "Government party x electoral proximity",
    "party_seat_share": "Party seat share",
    "log1p_speech_words": "Speech volume, log words",
    "log1p_nli_speeches": "Relevant NLI speech units, log",
    "cabinet_is_coalition": "Coalition cabinet",
    "cabinet_has_absolute_majority": "Cabinet majority",
    "cabinet_caretaker": "Caretaker cabinet",
}


@dataclass(frozen=True)
class WeightedRegressionResult:
    coefficients: pd.DataFrame
    fitted: pd.DataFrame
    nobs: int
    n_parameters: int
    residual_df: int
    r_squared: float
    adj_r_squared: float
    rmse: float
    outcome: str
    predictors: list[str]
    fixed_effects: list[str]
    weight_column: str
    se_type: str
    model_label: str
    dropped_terms: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline panel fixed-effects specification for salience "
            "alignment and NLI contradiction-rate outcomes."
        )
    )
    parser.add_argument("--plda-panel-dir", type=Path, default=DEFAULT_PLDA_PANEL_DIR)
    parser.add_argument("--nli-panel-dir", type=Path, default=DEFAULT_NLI_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nli-model-name", default=DEFAULT_NLI_MODEL_NAME, type=str)
    parser.add_argument("--nli-topics", nargs="+", default=DEFAULT_NLI_TOPICS)
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Optional custom pooled country list. Defaults to thesis clusters plus pooled clusters.",
    )
    parser.add_argument("--se", choices=["classic", "robust"], default="robust")
    parser.add_argument(
        "--weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use speech-volume weights. Enabled by default.",
    )
    return parser


def sanitize_filename_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def normalize_countries(raw_countries: list[str]) -> list[str]:
    countries: list[str] = []
    for raw in raw_countries:
        for part in raw.split(","):
            country = part.strip().upper()
            if country and country not in countries:
                countries.append(country)
    if not countries:
        raise ValueError("No country codes provided.")
    return countries


def model_specs(custom_countries: list[str] | None) -> dict[str, list[str]]:
    if custom_countries:
        countries = normalize_countries(custom_countries)
        return {"pooled_custom": countries}
    specs = {label: list(countries) for label, countries in CLUSTERS.items()}
    specs["pooled_clusters"] = [country for countries in CLUSTERS.values() for country in countries]
    return specs


def available(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in columns]


def plda_panel_path(panel_dir: Path, country: str) -> Path:
    return panel_dir / country / f"{country}_plda_regression_panel_model.csv"


def nli_panel_path(panel_dir: Path, country: str, topic: str, model_name: str) -> Path:
    topic_token = sanitize_filename_token(topic)
    model_token = sanitize_filename_token(model_name)
    candidates = [
        panel_dir
        / country
        / f"{country}_{topic_token}_nli_{model_token}_regression_panel_model.csv",
        panel_dir / country / f"{country}_{topic_token}_nli_regression_panel_model.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find NLI regression panel for "
        f"country={country}, topic={topic}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_plda_panels(panel_dir: Path, countries: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for country in countries:
        path = plda_panel_path(panel_dir, country)
        panel = pd.read_csv(path, low_memory=False)
        if "country_code" not in panel.columns:
            panel.insert(0, "country_code", country)
        frames.append(panel)
        paths.append(path)
    return pd.concat(frames, ignore_index=True, sort=False), paths


def load_nli_panels(
    panel_dir: Path,
    countries: list[str],
    topics: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for country in countries:
        for topic in topics:
            path = nli_panel_path(panel_dir, country, topic, model_name)
            panel = pd.read_csv(path, low_memory=False)
            if "country_code" not in panel.columns:
                panel.insert(0, "country_code", country)
            if "nli_topic" not in panel.columns:
                panel.insert(1, "nli_topic", topic)
            frames.append(panel)
            paths.append(path)
    return pd.concat(frames, ignore_index=True, sort=False), paths


def add_common_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    required = {
        "country_code",
        "speech_party",
        "month",
        "party_in_government",
        "electoral_cycle_progress",
    }
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Panel is missing required baseline columns: {missing}")

    out["country_code"] = out["country_code"].astype(str).str.upper().str.strip()
    out["speech_party"] = out["speech_party"].astype(str).str.strip()
    out["month"] = out["month"].astype(str).str.strip()
    month_start = pd.to_datetime(out["month"] + "-01", errors="coerce")
    out["year_fe"] = month_start.dt.year.astype("Int64").astype(str)
    out.loc[month_start.isna(), "year_fe"] = pd.NA
    out["country_party_fe"] = out["country_code"] + "::" + out["speech_party"]
    out["electoral_proximity"] = pd.to_numeric(
        out["electoral_cycle_progress"], errors="coerce"
    )
    out["party_in_government"] = ols.coerce_bool_series(out["party_in_government"])
    out["gov_x_electoral_proximity"] = (
        out["party_in_government"] * out["electoral_proximity"]
    )
    return out


def prepare_plda_data(data: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    out = add_common_features(data)
    if weighted:
        if "speech_volume_words" in out.columns:
            out["regression_weight"] = pd.to_numeric(out["speech_volume_words"], errors="coerce")
        elif "speech_volume_segments" in out.columns:
            out["regression_weight"] = pd.to_numeric(out["speech_volume_segments"], errors="coerce")
        else:
            out["regression_weight"] = 1.0
    else:
        out["regression_weight"] = 1.0
    return out


def prepare_nli_data(data: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    out = add_common_features(data)
    out["nli_topic"] = out["nli_topic"].astype(str).str.strip()
    if weighted:
        if "n_speeches" in out.columns:
            out["regression_weight"] = pd.to_numeric(out["n_speeches"], errors="coerce")
        elif "n_pairs" in out.columns:
            out["regression_weight"] = pd.to_numeric(out["n_pairs"], errors="coerce")
        else:
            out["regression_weight"] = 1.0
    else:
        out["regression_weight"] = 1.0
    return out


def build_model_matrices(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
    weight_column: str,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    required = [outcome, *predictors, *fixed_effects, weight_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    model_data = data.dropna(subset=required).copy()
    y = pd.to_numeric(model_data[outcome], errors="coerce")
    weights = pd.to_numeric(model_data[weight_column], errors="coerce")
    x = pd.concat(
        [
            pd.DataFrame({"const": np.ones(len(model_data), dtype=float)}, index=model_data.index),
            ols.predictor_matrix(model_data, predictors),
            ols.fixed_effect_matrix(model_data, fixed_effects),
        ],
        axis=1,
    )
    mask = y.notna() & weights.notna() & weights.gt(0) & x.notna().all(axis=1)
    y = y.loc[mask].astype(float)
    x = x.loc[mask].astype(float)
    weights = weights.loc[mask].astype(float)
    used = model_data.loc[mask].copy()
    if len(y) <= len(predictors) + 1:
        raise ValueError(f"Not enough rows after preprocessing: n={len(y)}.")
    x, dropped_terms = ols.drop_rank_redundant_columns(x)
    used.attrs["dropped_terms"] = dropped_terms
    return y, x, weights, used


def fit_wls(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
    weight_column: str,
    se_type: str,
    model_label: str,
) -> WeightedRegressionResult:
    y, x, weights, used = build_model_matrices(
        data=data,
        outcome=outcome,
        predictors=predictors,
        fixed_effects=fixed_effects,
        weight_column=weight_column,
    )
    dropped_terms = list(used.attrs.get("dropped_terms", []))
    sqrt_w = np.sqrt(weights.to_numpy(dtype=float))
    y_array = y.to_numpy(dtype=float)
    x_array = x.to_numpy(dtype=float)
    y_w = y_array * sqrt_w
    x_w = x_array * sqrt_w[:, None]

    beta, *_ = np.linalg.lstsq(x_w, y_w, rcond=None)
    fitted_values = x_array @ beta
    residuals = y_array - fitted_values
    nobs = len(y_array)
    n_parameters = x_array.shape[1]
    residual_df = nobs - n_parameters
    if residual_df <= 0:
        raise ValueError(f"Non-positive residual df: n={nobs}, p={n_parameters}.")

    xtwx_inv = np.linalg.pinv(x_w.T @ x_w)
    weighted_residuals = residuals * sqrt_w
    if se_type == "classic":
        sigma2 = float(weighted_residuals @ weighted_residuals / residual_df)
        covariance = sigma2 * xtwx_inv
    elif se_type == "robust":
        meat = x_w.T @ ((weighted_residuals ** 2)[:, None] * x_w)
        scale = nobs / residual_df
        covariance = scale * xtwx_inv @ meat @ xtwx_inv
    else:
        raise ValueError(f"Unsupported standard-error type: {se_type}")

    std_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_stat = np.divide(
        beta,
        std_error,
        out=np.full_like(beta, np.nan, dtype=float),
        where=std_error > 0,
    )
    p_value = 2 * stats.t.sf(np.abs(t_stat), df=residual_df)
    ci_multiplier = stats.t.ppf(0.975, df=residual_df)
    coefficients = pd.DataFrame(
        {
            "term": x.columns,
            "estimate": beta,
            "std_error": std_error,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_95_low": beta - ci_multiplier * std_error,
            "ci_95_high": beta + ci_multiplier * std_error,
        }
    )

    weighted_mean = float(np.average(y_array, weights=weights.to_numpy(dtype=float)))
    sse = float(weighted_residuals @ weighted_residuals)
    sst = float(np.sum(weights.to_numpy(dtype=float) * (y_array - weighted_mean) ** 2))
    r_squared = 1.0 - sse / sst if sst > 0 else math.nan
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (nobs - 1) / residual_df
        if residual_df > 0 and not math.isnan(r_squared)
        else math.nan
    )
    rmse = math.sqrt(sse / residual_df)

    fitted = used.copy()
    fitted["wls_weight"] = weights.to_numpy(dtype=float)
    fitted["wls_fitted"] = fitted_values
    fitted["wls_residual"] = residuals

    return WeightedRegressionResult(
        coefficients=coefficients,
        fitted=fitted,
        nobs=nobs,
        n_parameters=n_parameters,
        residual_df=residual_df,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        rmse=rmse,
        outcome=outcome,
        predictors=predictors,
        fixed_effects=fixed_effects,
        weight_column=weight_column,
        se_type=se_type,
        model_label=model_label,
        dropped_terms=dropped_terms,
    )


def pstars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def non_fixed_coefficients(result: WeightedRegressionResult) -> pd.DataFrame:
    return result.coefficients[
        ~result.coefficients["term"].astype(str).str.startswith("FE_")
    ].copy()


def regression_cell(result: WeightedRegressionResult, term: str) -> str:
    row = result.coefficients[result.coefficients["term"].eq(term)]
    if row.empty:
        return "dropped"
    values = row.iloc[0]
    return f"{values['estimate']:.4f}{pstars(values['p_value'])} ({values['std_error']:.4f})"


def compact_table(results: dict[str, WeightedRegressionResult], terms: list[str]) -> str:
    labels = list(results)
    lines = ["term\t" + "\t".join(labels)]
    for term in terms:
        label = TERM_LABELS.get(term, term)
        lines.append(
            label + "\t" + "\t".join(regression_cell(result, term) for result in results.values())
        )
    lines.append("Observations\t" + "\t".join(str(result.nobs) for result in results.values()))
    lines.append("Adjusted R2\t" + "\t".join(f"{result.adj_r_squared:.3f}" for result in results.values()))
    lines.append("Weighted outcome mean\t" + "\t".join(
        f"{np.average(result.fitted[result.outcome], weights=result.fitted['wls_weight']):.4f}"
        for result in results.values()
    ))
    lines.append("")
    lines.append("Notes: weighted least squares with HC1 robust standard errors in parentheses.")
    lines.append("All models include country-party and calendar-year fixed effects.")
    lines.append("The pooled NLI contradiction-rate model also includes topic fixed effects.")
    return "\n".join(lines) + "\n"


def summary_text(
    result: WeightedRegressionResult,
    input_paths: list[Path],
    outcome_label: str,
) -> str:
    lines = [
        "Baseline Panel Fixed-Effects WLS",
        "",
        f"Outcome: {result.outcome}",
        f"Outcome label: {outcome_label}",
        f"Model label: {result.model_label}",
        f"Inputs: {', '.join(str(path) for path in input_paths)}",
        f"Predictors: {', '.join(result.predictors)}",
        f"Fixed effects: {', '.join(result.fixed_effects)}",
        f"Weight column: {result.weight_column}",
        f"Standard errors: {result.se_type}",
        "",
        f"Observations: {result.nobs}",
        f"Parameters: {result.n_parameters}",
        f"Residual df: {result.residual_df}",
        f"Weighted R-squared: {result.r_squared:.6f}",
        f"Adjusted weighted R-squared: {result.adj_r_squared:.6f}",
        f"Weighted RMSE: {result.rmse:.6f}",
        "",
    ]
    if result.dropped_terms:
        lines.extend(["Dropped rank-redundant terms:", ", ".join(result.dropped_terms), ""])
    lines.append("Non-fixed-effect coefficients:")
    lines.append(non_fixed_coefficients(result).to_string(index=False))
    return "\n".join(lines) + "\n"


def fit_and_write(
    data: pd.DataFrame,
    input_paths: list[Path],
    outcome: str,
    outcome_label: str,
    predictors: list[str],
    fixed_effects: list[str],
    model_label: str,
    output_dir: Path,
    se_type: str,
) -> WeightedRegressionResult:
    result = fit_wls(
        data=data,
        outcome=outcome,
        predictors=predictors,
        fixed_effects=fixed_effects,
        weight_column="regression_weight",
        se_type=se_type,
        model_label=model_label,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_label}_{sanitize_filename_token(outcome_label)}"
    non_fixed_coefficients(result).to_csv(output_dir / f"{stem}_main_coefficients.csv", index=False)
    result.coefficients.to_csv(output_dir / f"{stem}_all_coefficients.csv", index=False)
    result.fitted.to_csv(output_dir / f"{stem}_fitted.csv", index=False)
    (output_dir / f"{stem}_summary.txt").write_text(
        summary_text(result, input_paths, outcome_label), encoding="utf-8"
    )
    return result


def run_plda_regressions(args: argparse.Namespace, specs: dict[str, list[str]]) -> Path:
    output_dir = args.output_dir / "salience_alignment"
    results: dict[str, WeightedRegressionResult] = {}
    for label, countries in specs.items():
        data, paths = load_plda_panels(args.plda_panel_dir, countries)
        prepared = prepare_plda_data(data, weighted=args.weighted)
        controls = available(prepared.columns, PLDA_CONTROLS)
        results[label] = fit_and_write(
            data=prepared,
            input_paths=paths,
            outcome="alignment_score",
            outcome_label="salience_alignment_score",
            predictors=MAIN_TERMS + controls,
            fixed_effects=["country_party_fe", "year_fe"],
            model_label=label,
            output_dir=output_dir,
            se_type=args.se,
        )
    table_path = output_dir / "baseline_salience_alignment_table.txt"
    terms = MAIN_TERMS + available(next(iter(results.values())).predictors, PLDA_CONTROLS)
    table_path.write_text(compact_table(results, terms), encoding="utf-8")
    return table_path


def run_nli_regressions(args: argparse.Namespace, specs: dict[str, list[str]]) -> Path:
    output_dir = args.output_dir / "substantive_contradiction_rate"
    results: dict[str, WeightedRegressionResult] = {}
    for label, countries in specs.items():
        data, paths = load_nli_panels(
            args.nli_panel_dir,
            countries,
            topics=args.nli_topics,
            model_name=args.nli_model_name,
        )
        prepared = prepare_nli_data(data, weighted=args.weighted)
        controls = available(prepared.columns, NLI_CONTROLS)
        results[label] = fit_and_write(
            data=prepared,
            input_paths=paths,
            outcome="inconsistency_share",
            outcome_label="substantive_contradiction_rate",
            predictors=MAIN_TERMS + controls,
            fixed_effects=["country_party_fe", "year_fe", "nli_topic"],
            model_label=label,
            output_dir=output_dir,
            se_type=args.se,
        )
    table_path = output_dir / "baseline_substantive_contradiction_rate_table.txt"
    terms = MAIN_TERMS + available(next(iter(results.values())).predictors, NLI_CONTROLS)
    table_path.write_text(compact_table(results, terms), encoding="utf-8")
    return table_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(args.countries)
    table_paths = [run_plda_regressions(args, specs), run_nli_regressions(args, specs)]

    manifest_lines = [
        "Baseline panel fixed-effects regression outputs",
        "",
        "Specification:",
        "Y_cpt(k) = alpha_cp + lambda_y + delta_k + beta1 Proximity_ct + beta2 Gov_cpt + beta3 Proximity_ct x Gov_cpt + gamma X_cpt(k) + epsilon_cpt(k)",
        "alpha_cp: country-party fixed effects",
        "lambda_y: calendar-year fixed effects",
        "delta_k: topic fixed effects only in pooled NLI contradiction-rate models",
        "Proximity_ct: electoral_cycle_progress, where 0 is just after the previous election and 1 is just before the next election",
        f"Weighted models: {args.weighted}",
        "",
        "Tables:",
        *[str(path) for path in table_paths],
    ]
    manifest_path = args.output_dir / "MANIFEST.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())