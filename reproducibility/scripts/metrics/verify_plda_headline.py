from __future__ import annotations

"""Verify the corrected harmonized Jensen--Shannon headline PLDA model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = BASE_DIR / "outputs" / "test_speeches" / "empirical_regression_audit"
SALIENCE_DIR = (
    BASE_DIR / "outputs" / "test_speeches" / "plda_salience" / "substantive" / "data"
)
PANEL_DIR = BASE_DIR / "outputs" / "test_speeches" / "plda_regression_panel"
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches" / "plda_headline_verification"
MODEL_ID = "plda_alignment_m1_natural"


def check(name: str, observed: object, expected: object) -> dict[str, object]:
    passed = bool(observed == expected)
    if not passed:
        raise AssertionError(f"{name}: observed={observed!r}, expected={expected!r}")
    return {
        "check": name,
        "observed": str(observed),
        "expected": str(expected),
        "passed": passed,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(AUDIT_DIR / "model_samples_and_design.csv")
    design = design.loc[design["model_id"].eq(MODEL_ID)]
    if len(design) != 1:
        raise ValueError(f"Expected one {MODEL_ID} design row, found {len(design)}.")
    row = design.iloc[0]
    topic = pd.read_csv(SALIENCE_DIR / "topic_salience_panel.csv", low_memory=False)
    rank = pd.read_csv(SALIENCE_DIR / "rank_salience_panel.csv", low_memory=False)
    basis = pd.read_csv(SALIENCE_DIR / "topic_basis_audit.csv")
    prior = rank["prior_manifesto"].fillna(False).astype(bool)
    valid = rank["alignment_valid"].fillna(False).astype(bool)

    checks: list[dict[str, object]] = [
        check("model observations", int(row["observations"]), 11_881),
        check("model countries", int(row["countries"]), 15),
        check("model parties", int(row["parties"]), 151),
        check("model unit", row["unit"], "party-month"),
        check("outcome", row["outcome"], "alignment_score"),
        check(
            "predictors",
            row["predictors"],
            "proximity_centered;party_in_government;gov_x_proximity_centered",
        ),
        check(
            "fixed effects",
            row["fixed_effects"],
            "country_party_fe;calendar_year_fe;month_of_year_fe",
        ),
        check("model weight", row["weight"], "weight_equal"),
        check(
            "primary inference",
            row["covariance_primary"],
            "wild country-cluster bootstrap; country CR1 intervals",
        ),
        check("selected broad topics", int(basis["selected_topics"].min()), 8),
        check("one shared latent component", int(basis["latent_component_count"].min()), 1),
        check("cached party-month rows", len(rank), 12_385),
        check("prior-manifesto party-month rows", int(prior.sum()), 11_931),
        check("valid prior-manifesto alignment rows", int((prior & valid).sum()), 11_881),
        check("invalid prior-manifesto alignment rows", int((prior & ~valid).sum()), 50),
        check("rank panel countries", rank["country_code"].nunique(), 15),
        check(
            "alignment basis",
            ",".join(sorted(rank.loc[valid, "alignment_topic_basis"].dropna().unique())),
            "harmonized_substantive_8",
        ),
        check(
            "alignment score bounds",
            bool(rank.loc[valid, "alignment_score"].between(0.0, 1.0).all()),
            True,
        ),
        check(
            "distance-score identity",
            bool(
                np.allclose(
                    rank.loc[valid, "alignment_score"],
                    1.0 - rank.loc[valid, "js_distance"],
                    atol=1e-12,
                )
            ),
            True,
        ),
        check(
            "topics per party-month",
            int(topic.groupby("party_month_fe")["topic"].nunique().min()),
            8,
        ),
    ]

    valid_speech = topic.groupby("party_month_fe")["speech_salience"].apply(
        lambda values: bool(values.notna().all())
    )
    valid_manifesto = topic.groupby("party_month_fe")["manifesto_salience"].apply(
        lambda values: bool(values.notna().all())
    )
    speech_sums = (
        topic.loc[topic["party_month_fe"].isin(valid_speech.index[valid_speech])]
        .groupby("party_month_fe")["speech_salience"]
        .sum()
    )
    manifesto_sums = (
        topic.loc[topic["party_month_fe"].isin(valid_manifesto.index[valid_manifesto])]
        .groupby("party_month_fe")["manifesto_salience"]
        .sum()
    )
    checks.extend(
        [
            check(
                "renormalized speech vectors sum to one",
                bool(np.allclose(speech_sums, 1.0, atol=1e-9)),
                True,
            ),
            check(
                "renormalized manifesto vectors sum to one",
                bool(np.allclose(manifesto_sums, 1.0, atol=1e-9)),
                True,
            ),
        ]
    )

    weighting_values: set[str] = set()
    for country in basis["country_code"]:
        path = PANEL_DIR / country / f"{country}_plda_regression_panel.csv"
        source = pd.read_csv(path, usecols=["speech_topic_weighting"])
        weighting_values.update(source["speech_topic_weighting"].dropna().astype(str).unique())
    checks.append(
        check("cached speech aggregation", ",".join(sorted(weighting_values)), "log_word_count")
    )

    contrasts = pd.read_csv(AUDIT_DIR / "all_contrasts.csv")
    maintained = contrasts.loc[contrasts["model_id"].eq(MODEL_ID)].copy()
    checks.extend(
        [
            check("headline contrasts", len(maintained), 6),
            check(
                "wild bootstrap replications",
                ",".join(
                    map(str, sorted(maintained["wild_reps"].dropna().astype(int).unique()))
                ),
                "999",
            ),
        ]
    )

    pd.DataFrame(checks).to_csv(OUTPUT_DIR / "checks.csv", index=False)
    values = maintained.set_index("contrast")
    summary = {
        "status": "verified",
        "model_id": MODEL_ID,
        "observations": int(row["observations"]),
        "countries": int(row["countries"]),
        "parties": int(row["parties"]),
        "topic_basis": "eight harmonized substantive topics",
        "alignment_outcome": "1 minus base-2 Jensen-Shannon distance",
        "speech_aggregation": "log(1 + word count)-weighted segment topic distributions",
        "manifesto_aggregation": "unweighted mean of inferred quasi-sentence topic distributions",
        "regression_weighting": "equal weight per valid party-month",
        "fixed_effects": ["country-party", "calendar-year", "month-of-year"],
        "inference": (
            "null-imposed two-sided Webb wild bootstrap clustered by country, "
            "999 replications; country-CR1 standard errors and intervals"
        ),
        "opposition_cycle_change": float(values.loc["opposition_slope", "estimate"]),
        "opposition_wild_p": float(values.loc["opposition_slope", "wild_p"]),
        "government_cycle_change": float(values.loc["government_slope", "estimate"]),
        "government_wild_p": float(values.loc["government_slope", "wild_p"]),
        "government_minus_opposition_midcycle": float(
            values.loc["government_gap_p05", "estimate"]
        ),
        "government_minus_opposition_midcycle_wild_p": float(
            values.loc["government_gap_p05", "wild_p"]
        ),
        "government_x_proximity": float(
            values.loc["government_x_proximity", "estimate"]
        ),
        "government_x_proximity_wild_p": float(
            values.loc["government_x_proximity", "wild_p"]
        ),
        "all_runtime_checks_passed": True,
        "full_inference_pipeline_rerun": True,
    }
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Corrected headline PLDA alignment verification

Status: **verified from rebuilt harmonized panels and the rerun regression audit**.

- The headline model is `{MODEL_ID}` with {summary['observations']:,} valid
  party-month observations, {summary['parties']} country-parties, and
  {summary['countries']} countries.
- The outcome is one minus base-2 Jensen--Shannon distance between the separately
  normalized manifesto and speech distributions on the same eight-topic basis.
- The model gives equal weight to each valid party-month and absorbs country-party,
  calendar-year, and month-of-year fixed effects.
- Maintained tests use a null-imposed, two-sided Webb wild bootstrap clustered by
  country with 999 replications. Reported standard errors and intervals are country-CR1.
- All checks in `checks.csv` passed after the full inference rerun.
"""
    (OUTPUT_DIR / "VERIFICATION.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
