from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_NLI_DIR = TEST_OUTPUT_DIR / "nli_inconsistency"
DEFAULT_PLDA_PANEL_DIR = TEST_OUTPUT_DIR / "plda_regression_panel"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "nli_regression_panel"

COVARIATE_COLUMNS = [
    "electoral_cycle_progress",
    "party_in_government",
    "party_prime_minister",
    "party_seat_share",
    "log1p_speech_words",
    "cabinet_is_coalition",
    "cabinet_has_absolute_majority",
    "cabinet_caretaker",
]


def sanitize_topic(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", topic).strip("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a party-month regression panel from NLI manifesto-inconsistency "
            "outputs, optionally enriched with the existing PLDA/Parlgov covariates."
        )
    )
    parser.add_argument("--country", default="GB", type=str)
    parser.add_argument("--topic", default="Macroeconomics", type=str)
    parser.add_argument("--nli-input", type=Path, default=None)
    parser.add_argument(
        "--covariate-panel",
        type=Path,
        default=None,
        help=(
            "Optional PLDA regression panel model CSV. If omitted, the script tries "
            "outputs/test_speeches/plda_regression_panel/<COUNTRY>/"
            "<COUNTRY>_plda_regression_panel_model.csv and continues without it "
            "if absent."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def default_nli_input(country_code: str, topic: str) -> Path:
    topic_token = sanitize_topic(topic)
    return (
        DEFAULT_NLI_DIR
        / country_code
        / f"{country_code}_{topic_token}_nli_summary_by_party_month.csv"
    )


def default_covariate_panel(country_code: str) -> Path:
    return (
        DEFAULT_PLDA_PANEL_DIR
        / country_code
        / f"{country_code}_plda_regression_panel_model.csv"
    )


def default_output(country_code: str, topic: str) -> Path:
    topic_token = sanitize_topic(topic)
    return (
        DEFAULT_OUTPUT_DIR
        / country_code
        / f"{country_code}_{topic_token}_nli_regression_panel_model.csv"
    )


def load_nli_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find NLI party-month summary: {path}")
    df = pd.read_csv(path, low_memory=False).copy()
    required = {
        "party",
        "month",
        "n_pairs",
        "n_speeches",
        "n_manifesto_quasi",
        "nli_share_contradiction",
        "nli_share_entailment",
        "nli_share_neutral",
        "nli_prob_contradiction",
        "nli_prob_entailment",
        "nli_prob_neutral",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"NLI summary is missing required columns: {missing}")
    df["party"] = df["party"].astype(str).str.strip()
    df["month"] = df["month"].astype(str).str.strip()
    df["month_start"] = pd.to_datetime(df["month"] + "-01", errors="coerce")
    return df.dropna(subset=["party", "month_start"]).copy()


def add_nli_features(df: pd.DataFrame, country_code: str, topic: str) -> pd.DataFrame:
    panel = df.copy()
    panel.insert(0, "country_code", country_code)
    panel.insert(1, "nli_topic", topic)
    panel["speech_party"] = panel["party"]
    panel["inconsistency_share"] = panel["nli_share_contradiction"]
    panel["inconsistency_prob"] = panel["nli_prob_contradiction"]
    panel["manifesto_support_share"] = panel["nli_share_entailment"]
    panel["net_inconsistency_share"] = (
        panel["nli_share_contradiction"] - panel["nli_share_entailment"]
    )
    panel["log1p_nli_pairs"] = np.log1p(panel["n_pairs"])
    panel["log1p_nli_speeches"] = np.log1p(panel["n_speeches"])
    panel["log1p_nli_manifesto_quasi"] = np.log1p(panel["n_manifesto_quasi"])
    panel["nli_contradiction_rate_per_100_pairs"] = 100 * panel["nli_share_contradiction"]
    return panel


def load_covariates(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    cov = pd.read_csv(path, low_memory=False).copy()
    party_col = "speech_party" if "speech_party" in cov.columns else "party"
    if party_col not in cov.columns or "month" not in cov.columns:
        raise ValueError(
            f"Covariate panel must contain speech_party/party and month columns: {path}"
        )
    cov["speech_party"] = cov[party_col].astype(str).str.strip()
    cov["month"] = cov["month"].astype(str).str.strip()
    keep = ["speech_party", "month"] + [col for col in COVARIATE_COLUMNS if col in cov.columns]
    if "speech_words" in cov.columns and "speech_words" not in keep:
        keep.append("speech_words")
    cov = cov[keep].drop_duplicates(["speech_party", "month"])
    return cov


def build_panel(
    nli_input: Path,
    covariate_panel: Path,
    country_code: str,
    topic: str,
) -> tuple[pd.DataFrame, bool]:
    nli = add_nli_features(load_nli_summary(nli_input), country_code, topic)
    cov = load_covariates(covariate_panel)
    if cov is None:
        return nli, False
    merged = nli.merge(cov, on=["speech_party", "month"], how="left", validate="m:1")
    return merged, True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    country_code = args.country.strip().upper()
    topic = args.topic.strip()
    nli_input = args.nli_input or default_nli_input(country_code, topic)
    covariate_panel = args.covariate_panel or default_covariate_panel(country_code)
    output = args.output or default_output(country_code, topic)

    panel, used_covariates = build_panel(
        nli_input=nli_input,
        covariate_panel=covariate_panel,
        country_code=country_code,
        topic=topic,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)

    print(f"Built {len(panel):,} NLI regression-panel rows for {country_code} {topic}.")
    print(f"Input NLI summary: {nli_input}")
    if used_covariates:
        print(f"Merged covariates from: {covariate_panel}")
    else:
        print(f"No covariate panel found at {covariate_panel}; wrote NLI-only panel.")
    print(f"Saved NLI regression panel to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
