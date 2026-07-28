from __future__ import annotations

"""Refit the primary salience responsiveness model on calendar-quarter data.

Monthly party-topic cells are collapsed within calendar quarters using speech-word
weights. Manifesto documents and government-status spells remain separate within a
quarter so election or cabinet transitions are not averaged across incompatible
segments. Regression cells receive equal weight after aggregation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.causality.empirical_regression_audit import (
    ModelSpec,
    contrast_vector,
    covariance_stats,
    fit_model,
)
from scripts.causality.recalculate_salience_responsiveness import (
    PREDICTORS,
    bootstrap_statistics,
    contrast_definitions,
    load_primary_panel,
    stable_seed,
)

DEFAULT_PANEL = (
    BASE_DIR
    / "outputs/test_speeches/plda_salience/substantive/data/topic_salience_panel.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR / "outputs/test_speeches/quarterly_salience_responsiveness"
)
MODEL_ID = "plda_topic_m1_equal_calendar_quarter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=4_999)
    parser.add_argument("--seed", type=int, default=1_711)
    return parser.parse_args()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(numeric.mean())
    return float(np.average(numeric.loc[valid], weights=weights.loc[valid]))


def build_quarterly_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    data = monthly.copy()
    month = pd.to_datetime(data["month"].astype(str) + "-01", errors="coerce")
    if month.isna().any():
        raise ValueError("Monthly panel contains invalid month values.")
    data["calendar_quarter"] = month.dt.to_period("Q").astype(str)
    data["speech_weight"] = pd.to_numeric(
        data["speech_volume_words"], errors="coerce"
    ).where(lambda value: value.gt(0))

    keys = [
        "country_code",
        "country_party_fe",
        "speech_party",
        "calendar_quarter",
        "doc_key",
        "party_in_government",
        "topic",
    ]
    rows: list[dict[str, object]] = []
    for key, group in data.groupby(keys, dropna=False, sort=False):
        weights = group["speech_weight"]
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "speech_salience": weighted_mean(group["speech_salience"], weights),
                "manifesto_salience": weighted_mean(
                    group["manifesto_salience"], weights
                ),
                "electoral_proximity": weighted_mean(
                    group["electoral_proximity"], weights
                ),
                "speech_volume_words": float(weights.fillna(0).sum()),
                "monthly_cells": int(len(group)),
            }
        )
        rows.append(row)

    quarterly = pd.DataFrame(rows)
    quarterly["party_in_government"] = pd.to_numeric(
        quarterly["party_in_government"], errors="raise"
    )
    quarterly["proximity_centered"] = quarterly["electoral_proximity"] - 0.5
    quarterly["manifesto_x_proximity_centered"] = (
        quarterly["manifesto_salience"] * quarterly["proximity_centered"]
    )
    quarterly["manifesto_x_government"] = (
        quarterly["manifesto_salience"] * quarterly["party_in_government"]
    )
    quarterly["manifesto_x_proximity_centered_x_government"] = (
        quarterly["manifesto_salience"]
        * quarterly["proximity_centered"]
        * quarterly["party_in_government"]
    )
    segment = (
        quarterly["country_party_fe"].astype(str)
        + "::"
        + quarterly["calendar_quarter"].astype(str)
        + "::"
        + quarterly["doc_key"].astype(str)
        + "::G"
        + quarterly["party_in_government"].astype(int).astype(str)
    )
    quarterly["party_quarter_fe"] = segment
    quarterly["month"] = quarterly["calendar_quarter"]
    quarterly["country_month_fe"] = (
        quarterly["country_code"].astype(str)
        + "::"
        + quarterly["calendar_quarter"].astype(str)
    )
    quarterly["weight_equal"] = 1.0
    return quarterly


def quarterly_model(data: pd.DataFrame) -> ModelSpec:
    return ModelSpec(
        model_id=MODEL_ID,
        family="plda_topic",
        outcome="speech_salience",
        data=data,
        predictors=PREDICTORS,
        effects=["party_quarter_fe", "topic"],
        weight="weight_equal",
        sample="prior manifestos; calendar-quarter segments",
        specification="quarterly aggregation robustness",
        unit="party-topic-calendar-quarter segment",
        bootstrap=True,
        notes=(
            "Speech-word aggregation within calendar quarters; manifesto and "
            "government-status transitions remain separate; equal regression-cell weights."
        ),
    )


def main() -> None:
    args = parse_args()
    monthly = load_primary_panel(args.panel)
    quarterly = build_quarterly_panel(monthly)
    fit = fit_model(quarterly_model(quarterly))

    mean_proximity = float(fit.used["electoral_proximity"].mean())
    government_share = float(fit.used["party_in_government"].mean())
    definitions = [
        definition
        for definition in contrast_definitions(mean_proximity, government_share)
        if definition.role == "primary"
    ]

    rows: list[dict[str, object]] = []
    for definition in definitions:
        vector = contrast_vector(fit, definition.terms)
        estimate = float(vector @ fit.beta)
        cr1 = covariance_stats(
            estimate, vector, fit.cov_country, fit.country_clusters - 1
        )
        bootstrap = bootstrap_statistics(
            fit,
            vector,
            args.bootstrap_reps,
            stable_seed(args.seed, f"calendar_quarter::{definition.short_name}"),
        )
        critical = float(bootstrap["wild_abs_t_95_critical"])
        rows.append(
            {
                "model_id": MODEL_ID,
                "contrast": definition.label,
                "short_name": definition.short_name,
                "hypothesis": definition.hypothesis,
                "estimate": estimate,
                "country_cr1_se": cr1["se"],
                "country_cr1_p_two_sided": cr1["p"],
                "wild_ci_95_low": estimate - critical * cr1["se"],
                "wild_ci_95_high": estimate + critical * cr1["se"],
                "wild_p_two_sided": bootstrap["wild_p_two_sided"],
                "wild_p_mcse": bootstrap["wild_p_mcse"],
                "wild_valid_reps": bootstrap["wild_valid_reps"],
                "contrast_terms": json.dumps(definition.terms, sort_keys=True),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(args.output_dir / "quarterly_topic_salience_panel.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "quarterly_primary_contrasts.csv", index=False)
    metadata = {
        "source_panel": str(args.panel.resolve()),
        "monthly_topic_cells": int(len(monthly)),
        "monthly_party_months": int(monthly["party_month_fe"].nunique()),
        "quarterly_topic_cells": int(len(fit.used)),
        "quarterly_segments": int(fit.used["party_quarter_fe"].nunique()),
        "calendar_quarters": int(fit.used["calendar_quarter"].nunique()),
        "countries": int(fit.used["country_code"].nunique()),
        "government_share": government_share,
        "mean_electoral_proximity": mean_proximity,
        "bootstrap_reps": args.bootstrap_reps,
        "aggregation": (
            "speech-word-weighted within country-party-calendar-quarter-manifesto-"
            "government-status-topic; equal quarterly regression-cell weights"
        ),
    }
    (args.output_dir / "quarterly_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
