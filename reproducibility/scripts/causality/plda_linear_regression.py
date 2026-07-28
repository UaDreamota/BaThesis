from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_INPUT_DIR = TEST_OUTPUT_DIR / "plda_regression_panel"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_linear_regression"

DEFAULT_PREDICTORS = [
    "electoral_cycle_progress",
    "party_in_government",
    "party_prime_minister",
    "party_seat_share",
    "log1p_speech_words",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]
DEFAULT_FIXED_EFFECTS = ["speech_party"]


@dataclass(frozen=True)
class RegressionResult:
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
    se_type: str
    model_label: str
    dropped_terms: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OLS regression on the PLDA manifesto-alignment regression panel."
        )
    )
    parser.add_argument("--country", default="CZ", type=str)
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help=(
            "Run one pooled OLS over multiple country panels. Accepts space- "
            "or comma-separated country codes, for example: CZ LV PL or CZ,LV,PL."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Regression panel CSV. Defaults to "
            "outputs/test_speeches/plda_regression_panel/<COUNTRY>/"
            "<COUNTRY>_plda_regression_panel_model.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where coefficient, fitted-value, and summary outputs are saved.",
    )
    parser.add_argument(
        "--outcome",
        default="alignment_score",
        type=str,
        help="Dependent variable column.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=DEFAULT_PREDICTORS,
        help=(
            "Predictor columns. Numeric and boolean columns are used directly; "
            "string/category columns are expanded to dummies."
        ),
    )
    parser.add_argument(
        "--fixed-effects",
        nargs="*",
        default=DEFAULT_FIXED_EFFECTS,
        help=(
            "Categorical fixed-effect columns expanded to dummies. "
            "Use --fixed-effects with no values to disable."
        ),
    )
    parser.add_argument(
        "--country-fixed-effects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include country_code fixed effects. Defaults to enabled for pooled "
            "multi-country regressions and disabled for single-country regressions."
        ),
    )
    parser.add_argument(
        "--se",
        choices=["classic", "robust"],
        default="robust",
        help="Standard error estimator. robust is HC1 heteroskedasticity-robust.",
    )
    parser.add_argument(
        "--drop-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop rows with missing outcome/predictor/fixed-effect values.",
    )
    return parser


def default_input(country_code: str) -> Path:
    return (
        DEFAULT_INPUT_DIR
        / country_code
        / f"{country_code}_plda_regression_panel_model.csv"
    )


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


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find regression panel: {path}")
    return pd.read_csv(path, low_memory=False)


def load_country_panels(country_codes: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for country_code in country_codes:
        path = default_input(country_code)
        panel = load_panel(path)
        panel.insert(0, "country_code", country_code)
        frames.append(panel)
        paths.append(path)
    return pd.concat(frames, ignore_index=True, sort=False), paths


def validate_columns(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
) -> None:
    required = [outcome] + predictors + fixed_effects
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        bool_values = {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
            "yes": 1.0,
            "no": 0.0,
        }
        if normalized.dropna().isin(bool_values).all():
            return normalized.map(bool_values).astype(float)
    return series


def predictor_matrix(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for column in predictors:
        series = coerce_bool_series(data[column])
        if pd.api.types.is_numeric_dtype(series):
            parts.append(pd.DataFrame({column: pd.to_numeric(series, errors="coerce")}))
        else:
            dummies = pd.get_dummies(
                series.astype("category"),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            parts.append(dummies)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=data.index)


def fixed_effect_matrix(data: pd.DataFrame, fixed_effects: list[str]) -> pd.DataFrame:
    parts = [
        pd.get_dummies(
            data[column].astype("category"),
            prefix=f"FE_{column}",
            drop_first=True,
            dtype=float,
        )
        for column in fixed_effects
    ]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=data.index)


def prepare_model_data(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
    drop_missing: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    validate_columns(data, outcome, predictors, fixed_effects)
    model_columns = [outcome] + predictors + fixed_effects
    model_data = data.copy()

    if drop_missing:
        model_data = model_data.dropna(subset=model_columns).copy()

    y = pd.to_numeric(model_data[outcome], errors="coerce")
    x_parts = [
        pd.DataFrame({"const": np.ones(len(model_data), dtype=float)}, index=model_data.index),
        predictor_matrix(model_data, predictors),
        fixed_effect_matrix(model_data, fixed_effects),
    ]
    x = pd.concat(x_parts, axis=1)

    numeric_mask = y.notna() & x.notna().all(axis=1)
    y = y.loc[numeric_mask].astype(float)
    x = x.loc[numeric_mask].astype(float)
    used_data = model_data.loc[numeric_mask].copy()

    if len(y) <= x.shape[1]:
        raise ValueError(
            f"Not enough rows for OLS after preprocessing: n={len(y)}, p={x.shape[1]}."
        )
    return y, x, used_data


def drop_rank_redundant_columns(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    selected_columns: list[str] = []
    dropped_columns: list[str] = []
    selected_matrix = np.empty((len(x), 0), dtype=float)

    for column in x.columns:
        candidate_column = x[[column]].to_numpy(dtype=float)
        candidate_matrix = np.column_stack([selected_matrix, candidate_column])
        current_rank = 0 if selected_matrix.shape[1] == 0 else np.linalg.matrix_rank(selected_matrix)
        candidate_rank = np.linalg.matrix_rank(candidate_matrix)

        if candidate_rank > current_rank:
            selected_columns.append(column)
            selected_matrix = candidate_matrix
        else:
            dropped_columns.append(column)

    return x.loc[:, selected_columns], dropped_columns


def classic_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    residual_df: int,
) -> np.ndarray:
    sigma2 = float((residuals @ residuals) / residual_df)
    return sigma2 * xtx_inv


def hc1_covariance(
    x: np.ndarray,
    residuals: np.ndarray,
    xtx_inv: np.ndarray,
    nobs: int,
    n_parameters: int,
) -> np.ndarray:
    meat = x.T @ ((residuals ** 2)[:, None] * x)
    scale = nobs / (nobs - n_parameters)
    return scale * xtx_inv @ meat @ xtx_inv


def fit_ols(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    fixed_effects: list[str],
    se_type: str,
    drop_missing: bool,
    model_label: str,
) -> RegressionResult:
    y, x, used_data = prepare_model_data(
        data=data,
        outcome=outcome,
        predictors=predictors,
        fixed_effects=fixed_effects,
        drop_missing=drop_missing,
    )
    x, dropped_terms = drop_rank_redundant_columns(x)
    y_array = y.to_numpy(dtype=float)
    x_array = x.to_numpy(dtype=float)
    nobs, n_parameters = x_array.shape
    residual_df = nobs - n_parameters

    beta, *_ = np.linalg.lstsq(x_array, y_array, rcond=None)
    fitted_values = x_array @ beta
    residuals = y_array - fitted_values
    xtx_inv = np.linalg.pinv(x_array.T @ x_array)

    if se_type == "classic":
        covariance = classic_covariance(x_array, residuals, xtx_inv, residual_df)
    elif se_type == "robust":
        covariance = hc1_covariance(x_array, residuals, xtx_inv, nobs, n_parameters)
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

    sse = float(residuals @ residuals)
    centered_y = y_array - y_array.mean()
    sst = float(centered_y @ centered_y)
    r_squared = 1.0 - sse / sst if sst > 0 else math.nan
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (nobs - 1) / residual_df
        if residual_df > 0 and not math.isnan(r_squared)
        else math.nan
    )
    rmse = math.sqrt(sse / residual_df)

    fitted = used_data.copy()
    fitted["ols_fitted"] = fitted_values
    fitted["ols_residual"] = residuals

    return RegressionResult(
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
        se_type=se_type,
        model_label=model_label,
        dropped_terms=dropped_terms,
    )


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def infer_model_label(data: pd.DataFrame) -> str:
    if "speech_topic_weighting" not in data.columns:
        return "unlabeled"
    values = data["speech_topic_weighting"].dropna().astype(str).unique()
    if len(values) == 1:
        return values[0]
    if len(values) == 0:
        return "unlabeled"
    return "mixed_weighting"


def output_paths(
    output_dir: Path,
    country_code: str,
    outcome: str,
    model_label: str,
) -> dict[str, Path]:
    country_dir = output_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{country_code}_plda_ols_{outcome}_{sanitize_filename(model_label)}"
    return {
        "coefficients": country_dir / f"{stem}_coefficients.csv",
        "fitted": country_dir / f"{stem}_fitted.csv",
        "summary": country_dir / f"{stem}_summary.txt",
    }


def summary_text(result: RegressionResult, input_path: object) -> str:
    lines = [
        "PLDA Manifesto Alignment OLS",
        "",
        f"Input: {input_path}",
        f"Outcome: {result.outcome}",
        f"Model label: {result.model_label}",
        f"Predictors: {', '.join(result.predictors)}",
        f"Fixed effects: {', '.join(result.fixed_effects) if result.fixed_effects else 'none'}",
        f"Standard errors: {result.se_type}",
        "",
        f"Observations: {result.nobs}",
        f"Parameters: {result.n_parameters}",
        f"Residual df: {result.residual_df}",
        f"R-squared: {result.r_squared:.6f}",
        f"Adjusted R-squared: {result.adj_r_squared:.6f}",
        f"RMSE: {result.rmse:.6f}",
        "",
    ]

    if result.dropped_terms:
        lines.extend(
            [
                "Dropped rank-redundant terms:",
                ", ".join(result.dropped_terms),
                "",
            ]
        )

    lines.append("Non-fixed-effect coefficients:")

    display = result.coefficients[
        ~result.coefficients["term"].str.startswith("FE_")
    ].copy()
    lines.append(
        display.to_string(
            index=False,
            columns=[
                "term",
                "estimate",
                "std_error",
                "t_stat",
                "p_value",
                "ci_95_low",
                "ci_95_high",
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.countries and args.input is not None:
        raise ValueError("--input cannot be combined with --countries.")

    if args.countries:
        country_codes = normalize_countries(args.countries)
        data, input_paths = load_country_panels(country_codes)
        country_code = "pooled_" + "_".join(country_codes)
        input_path: object = ", ".join(str(path) for path in input_paths)
    else:
        country_code = args.country.strip().upper()
        input_path = args.input or default_input(country_code)
        data = load_panel(input_path)

    fixed_effects = list(args.fixed_effects)
    use_country_fixed_effects = (
        args.country_fixed_effects
        if args.country_fixed_effects is not None
        else bool(args.countries)
    )
    if use_country_fixed_effects and "country_code" not in fixed_effects:
        fixed_effects.append("country_code")

    model_label = infer_model_label(data)
    result = fit_ols(
        data=data,
        outcome=args.outcome,
        predictors=args.predictors,
        fixed_effects=fixed_effects,
        se_type=args.se,
        drop_missing=args.drop_missing,
        model_label=model_label,
    )

    paths = output_paths(args.output_dir, country_code, args.outcome, model_label)
    result.coefficients.to_csv(paths["coefficients"], index=False)
    result.fitted.to_csv(paths["fitted"], index=False)
    paths["summary"].write_text(summary_text(result, input_path), encoding="utf-8")

    print(f"Ran OLS for {country_code}: {args.outcome}")
    print(f"Observations={result.nobs}, parameters={result.n_parameters}")
    print(f"R-squared={result.r_squared:.4f}, adjusted R-squared={result.adj_r_squared:.4f}")
    if result.dropped_terms:
        print(f"Dropped rank-redundant terms: {', '.join(result.dropped_terms)}")
    print(f"Saved coefficients to: {paths['coefficients']}")
    print(f"Saved fitted values to: {paths['fitted']}")
    print(f"Saved summary to: {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
