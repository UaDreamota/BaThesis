from __future__ import annotations

"""Calculate the four-test BH family aligned with hypotheses H1a--H2b."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.causality.empirical_regression_audit import (  # noqa: E402
    ModelSpec,
    add_common_features,
    contrast_vector,
    covariance_stats,
    fit_model,
)
from scripts.causality.recalculate_salience_responsiveness import (  # noqa: E402
    bootstrap_statistics,
)


MACRO_COUNTRIES = [
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
]
DEFAULT_MACRO_ROOT = (
    BASE_DIR / "outputs/test_speeches/nli_consensus_regression_panel"
)
DEFAULT_SALIENCE_RESULTS = (
    BASE_DIR
    / "outputs/test_speeches/salience_responsiveness_recalculation/all_contrasts.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR / "outputs/test_speeches/primary_hypothesis_bh_recalculation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-root", type=Path, default=DEFAULT_MACRO_ROOT)
    parser.add_argument(
        "--salience-results", type=Path, default=DEFAULT_SALIENCE_RESULTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=4_999)
    parser.add_argument("--seed", type=int, default=1_711)
    return parser.parse_args()


def stable_seed(seed: int, hypothesis: str) -> int:
    digest = hashlib.sha256(
        f"primary_bh_family||{hypothesis}".encode("utf-8")
    ).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def load_macro_panel(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for country in MACRO_COUNTRIES:
        path = (
            root
            / country
            / f"{country}_Macroeconomics_nli_consensus_deberta_regression_panel_model.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing primary Macroeconomics panel: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    return add_common_features(pd.concat(frames, ignore_index=True), set())


def fit_primary_macro(data: pd.DataFrame) -> Any:
    return fit_model(
        ModelSpec(
            model_id="new_macro_m1_natural_primary_bh",
            family="new_macro",
            outcome="inconsistency_share",
            data=data,
            predictors=[
                "proximity_centered",
                "party_in_government",
                "gov_x_proximity_centered",
            ],
            effects=[
                "country_party_fe",
                "calendar_year_fe",
                "month_of_year_fe",
            ],
            weight="weight_equal",
            specification="M1",
            unit="party-month",
            bootstrap=True,
            notes="Primary unconditional Macroeconomics inconsistency model.",
        )
    )


def macro_hypothesis_rows(
    fit: Any, reps: int, seed: int
) -> list[dict[str, Any]]:
    government_share = float(fit.used["party_in_government"].mean())
    centered_proximity_mean = float(fit.used["proximity_centered"].mean())
    definitions = [
        (
            "H1b",
            "Sample-average inconsistency cycle change",
            {
                "proximity_centered": 1.0,
                "gov_x_proximity_centered": government_share,
            },
        ),
        (
            "H2b",
            "Sample-average government-opposition inconsistency gap",
            {
                "party_in_government": 1.0,
                "gov_x_proximity_centered": centered_proximity_mean,
            },
        ),
    ]
    rows: list[dict[str, Any]] = []
    for hypothesis, contrast, terms in definitions:
        print(f"Bootstrapping {hypothesis}: {contrast}", flush=True)
        vector = contrast_vector(fit, terms)
        estimate = float(vector @ fit.beta)
        cr1 = covariance_stats(
            estimate, vector, fit.cov_country, fit.country_clusters - 1
        )
        wild = bootstrap_statistics(
            fit, vector, reps, stable_seed(seed, hypothesis)
        )
        critical = float(wild["wild_abs_t_95_critical"])
        rows.append(
            {
                "hypothesis": hypothesis,
                "outcome_family": "primary_unconditional_inconsistency",
                "contrast": contrast,
                "estimate": estimate,
                "country_cr1_se": cr1["se"],
                "wild_ci_95_low": estimate - critical * cr1["se"],
                "wild_ci_95_high": estimate + critical * cr1["se"],
                "wild_p_two_sided": wild["wild_p_two_sided"],
                "wild_p_mcse": wild["wild_p_mcse"],
                "wild_valid_reps": wild["wild_valid_reps"],
                "contrast_terms": json.dumps(terms, sort_keys=True),
            }
        )
    return rows


def salience_hypothesis_rows(path: Path, reps: int) -> list[dict[str, Any]]:
    results = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for hypothesis in ("H1a", "H2a"):
        selected = results.loc[
            results["hypothesis"].eq(hypothesis) & results["role"].eq("primary")
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one primary {hypothesis} row in {path}; found {len(selected)}."
            )
        row = selected.iloc[0]
        valid_reps = int(row["wild_valid_reps"])
        if valid_reps != reps:
            raise ValueError(
                f"{hypothesis} uses {valid_reps} bootstrap draws, but this run requests {reps}."
            )
        rows.append(
            {
                "hypothesis": hypothesis,
                "outcome_family": "plda_salience_responsiveness",
                "contrast": row["contrast"],
                "estimate": float(row["estimate"]),
                "country_cr1_se": float(row["country_cr1_se"]),
                "wild_ci_95_low": float(row["wild_ci_95_low"]),
                "wild_ci_95_high": float(row["wild_ci_95_high"]),
                "wild_p_two_sided": float(row["wild_p_two_sided"]),
                "wild_p_mcse": float(row["wild_p_mcse"]),
                "wild_valid_reps": valid_reps,
                "contrast_terms": row["contrast_terms"],
            }
        )
    return rows


def apply_bh(rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    order = pd.Categorical(
        results["hypothesis"], categories=["H1a", "H1b", "H2a", "H2b"], ordered=True
    )
    results = results.assign(_order=order).sort_values("_order").drop(columns="_order")
    if results["hypothesis"].tolist() != ["H1a", "H1b", "H2a", "H2b"]:
        raise ValueError("The BH family must contain exactly H1a, H1b, H2a, and H2b.")
    results["bh_q_four_primary"] = multipletests(
        results["wild_p_two_sided"].to_numpy(float), method="fdr_bh"
    )[1]
    results["bh_reject_05"] = results["bh_q_four_primary"].lt(0.05)
    return results


def fmt(value: float) -> str:
    return f"{value:.4f}"


def report(results: pd.DataFrame, fit: Any, reps: int) -> str:
    lines = [
        "# Four-hypothesis primary BH family",
        "",
        f"Benjamini-Hochberg correction is applied once across the sample-average contrasts for H1a, H1b, H2a, and H2b. Inputs are two-sided {reps:,}-draw restricted Webb wild country-cluster bootstrap p-values.",
        "",
        "| Hypothesis | Sample-average contrast | Estimate | Wild p | BH q | q < .05 |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.hypothesis} | {row.contrast} | {fmt(row.estimate)} | "
            f"{fmt(row.wild_p_two_sided)} | {fmt(row.bh_q_four_primary)} | "
            f"{'yes' if row.bh_reject_05 else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"The unconditional inconsistency M1 calculation uses {len(fit.used):,} party-months, {fit.used['country_party_fe'].nunique()} country-parties, and {fit.country_clusters} country clusters. Its observed government share is {fit.used['party_in_government'].mean():.6f}, and its observed mean centered proximity is {fit.used['proximity_centered'].mean():.6f}.",
            "",
            "H1b and H2b estimates run opposite their hypothesized signs: the positive H1b estimate indicates increasing inconsistency toward elections, while the negative H2b estimate indicates lower inconsistency among governing parties. A small q-value for H1b is therefore evidence against, not in support of, H1b.",
            "",
            f"Bootstrap replications per contrast: {reps}.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 99:
        raise ValueError("Use at least 99 bootstrap replications; 4,999 is recommended.")
    macro_root = args.macro_root.resolve()
    salience_results = args.salience_results.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_fit = fit_primary_macro(load_macro_panel(macro_root))
    rows = salience_hypothesis_rows(salience_results, args.bootstrap_reps)
    rows.extend(
        macro_hypothesis_rows(macro_fit, args.bootstrap_reps, args.seed)
    )
    results = apply_bh(rows)
    results.to_csv(output_dir / "primary_bh_family.csv", index=False)
    (output_dir / "PRIMARY_BH_RESULTS.md").write_text(
        report(results, macro_fit, args.bootstrap_reps), encoding="utf-8"
    )
    print(f"Wrote four-test BH results to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
