from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.causality import plda_salience_regression as salience
from scripts.metrics import plda_salience_panels as panels


DEFAULT_PANEL_DIR = BASE_DIR / "outputs" / "test_speeches" / "plda_regression_panel"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches" / "plda_raw_topic_regression"
DEFAULT_COUNTRIES = list(salience.DEFAULT_COUNTRIES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate exploratory salience regressions on raw country-specific PLDA "
            "components rather than the harmonized eight-topic basis."
        )
    )
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--sample",
        choices=["prior", "all"],
        default="prior",
        help="The exploratory 'prior' sample keeps latest-prior-manifesto links only.",
    )
    parser.add_argument(
        "--weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use speech-volume weights. Enabled by default.",
    )
    parser.add_argument("--absorb-tolerance", type=float, default=1e-10)
    parser.add_argument("--absorb-max-iterations", type=int, default=1000)
    return parser


def panel_path(panel_dir: Path, country: str) -> Path:
    return panel_dir / country / f"{country}_plda_regression_panel.csv"


def load_country_panel(panel_dir: Path, country: str) -> pd.DataFrame:
    path = panel_path(panel_dir, country)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw PLDA regression panel: {path}")
    return pd.read_csv(path, low_memory=False)


def filter_sample(data: pd.DataFrame, sample: str) -> pd.DataFrame:
    out = data.copy()
    if sample == "prior":
        if "selection_method" not in out.columns:
            raise ValueError("Raw panel is missing selection_method.")
        out = out.loc[
            out["selection_method"].astype(str).eq(panels.PRIOR_MANIFESTO_METHOD)
        ].copy()
    return out.reset_index(drop=True)


def build_raw_topic_panel(data: pd.DataFrame, country: str) -> pd.DataFrame:
    metadata = panels.prepare_metadata(data, country)
    speech_bases = panels.topic_bases(data.columns, "speech")
    manifesto_bases = panels.topic_bases(data.columns, "manifesto")
    if speech_bases != manifesto_bases:
        raise ValueError(
            f"{country}: speech and manifesto topic layouts differ in the raw panel."
        )

    frames: list[pd.DataFrame] = []
    for base in speech_bases:
        stem = f"{country}::{base}"
        frame = metadata.copy()
        frame["topic"] = stem
        frame["raw_topic_fe"] = stem
        frame["country_month_raw_topic_fe"] = (
            frame["country_month_fe"].astype(str) + "::" + base
        )
        frame["speech_salience"] = pd.to_numeric(
            data[f"{base}_speech"], errors="coerce"
        )
        frame["manifesto_salience"] = pd.to_numeric(
            data[f"{base}_manifesto"], errors="coerce"
        )
        frame["manifesto_x_proximity"] = (
            frame["manifesto_salience"] * frame["electoral_proximity"]
        )
        frame["manifesto_x_proximity_centered"] = (
            frame["manifesto_salience"] * frame["proximity_centered"]
        )
        frame["manifesto_x_government"] = (
            frame["manifesto_salience"] * frame["party_in_government"]
        )
        frame["gov_x_proximity_centered"] = (
            frame["party_in_government"] * frame["proximity_centered"]
        )
        frame["manifesto_x_proximity_centered_x_government"] = (
            frame["manifesto_salience"]
            * frame["proximity_centered"]
            * frame["party_in_government"]
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def model_specs() -> list[dict[str, object]]:
    return [
        {
            "model": "raw_topic_requested_h1",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "electoral_proximity",
                "manifesto_x_proximity",
            ],
            "effects": ["country_party_fe", "raw_topic_fe"],
        },
        {
            "model": "raw_topic_requested_joint",
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
            "effects": ["country_party_fe", "raw_topic_fe"],
        },
        {
            "model": "raw_topic_compositional",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "manifesto_x_proximity_centered",
                "manifesto_x_government",
                "manifesto_x_proximity_centered_x_government",
            ],
            "effects": ["party_month_fe", "raw_topic_fe"],
        },
        {
            "model": "raw_topic_common_shocks",
            "outcome": "speech_salience",
            "predictors": [
                "manifesto_salience",
                "manifesto_x_proximity_centered",
                "manifesto_x_government",
                "manifesto_x_proximity_centered_x_government",
            ],
            "effects": ["party_month_fe", "country_month_raw_topic_fe"],
        },
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    countries = salience.normalize_countries(args.countries)
    topic_frames: list[pd.DataFrame] = []
    country_rows: list[dict[str, object]] = []

    for country in countries:
        panel = load_country_panel(args.panel_dir, country)
        panel = filter_sample(panel, args.sample)
        topic_panel = build_raw_topic_panel(panel, country)
        topic_frames.append(topic_panel)
        country_rows.append(
            {
                "country_code": country,
                "party_months": int(panel.shape[0]),
                "raw_topics": int(
                    len(panels.topic_bases(panel.columns, "speech"))
                ),
                "topic_rows": int(topic_panel.shape[0]),
                "parties": int(panel["speech_party"].astype(str).nunique()),
                "first_month": str(panel["month"].min()),
                "last_month": str(panel["month"].max()),
            }
        )

    topic_data = pd.concat(topic_frames, ignore_index=True, sort=False)
    results: list[salience.AbsorbedRegressionResult] = []
    for specification in model_specs():
        weights = salience.regression_weights(topic_data, args.weighted)
        result = salience.fit_absorbed_wls(
            data=topic_data,
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

    output_dir = args.output_dir / args.sample / ("weighted" if args.weighted else "unweighted")
    coefficient_dir = output_dir / "coefficients"
    coefficient_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        result.coefficients.to_csv(coefficient_dir / f"{result.model}.csv", index=False)
    pd.DataFrame([salience.model_metadata(result) for result in results]).to_csv(
        output_dir / "model_summary.csv", index=False
    )
    contrasts = pd.DataFrame(
        [row for result in results for row in salience.hypothesis_contrasts(result)]
    )
    contrasts.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    pd.DataFrame(country_rows).to_csv(output_dir / "sample_country_summary.csv", index=False)
    (output_dir / "regression_table.txt").write_text(
        salience.compact_table(results), encoding="utf-8"
    )

    manifest = [
        "Exploratory raw-PLDA-topic salience regressions",
        "",
        f"Panel root: {args.panel_dir}",
        f"Countries: {', '.join(countries)}",
        f"Sample: {args.sample}",
        f"Speech-volume weighted: {args.weighted}",
        f"Party-month observations: {topic_data['party_month_fe'].nunique()}",
        f"Topic-level observations: {len(topic_data)}",
        f"Country-party clusters: {topic_data['country_party_fe'].nunique()}",
        "Raw topic identity is country-specific.",
        "Topic fixed effects are defined on country::topic_j, not on harmonized substantive topics.",
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
