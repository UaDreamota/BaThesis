from __future__ import annotations

"""Generate additional thesis robustness checks missing from the appendix bundle.

This script adds three things:
1. complete-cycle-only headline salience responsiveness contrasts,
2. complete-cycle-only headline conditional-H contrasts,
3. leave-one-country-out conditional-H headline contrasts.

It also audits whether a genuine original 23-label salience rerun is feasible from
cached artifacts alone.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.causality.empirical_regression_audit import (
    ModelSpec,
    contrast_vector,
    covariance_stats,
    fit_model,
    load_data,
)
from scripts.causality.recalculate_salience_responsiveness import (
    bootstrap_statistics,
    contrast_definitions as salience_contrast_definitions,
    load_primary_panel,
    model_spec as salience_model_spec,
    stable_seed as salience_stable_seed,
)

DEFAULT_OUTPUT = BASE_DIR / "outputs" / "test_speeches" / "additional_robustness_checks"
DEFAULT_SALIENCE_PANEL = (
    BASE_DIR / "outputs" / "test_speeches" / "plda_salience" / "substantive" / "data" / "topic_salience_panel.csv"
)
PRIMARY_PREDICTORS = ["proximity_centered", "party_in_government", "gov_x_proximity_centered"]
FAMILY_SPECS = {
    "new_macro": {
        "topic_label": "Macroeconomics",
        "effects": ["country_party_fe", "calendar_year_fe", "month_of_year_fe"],
        "unit": "party-month",
    },
    "new_galtan": {
        "topic_label": "GAL-TAN",
        "effects": ["country_party_fe", "calendar_year_fe", "month_of_year_fe"],
        "unit": "party-month",
    },
    "new_combined": {
        "topic_label": "Pooled",
        "effects": ["country_party_fe", "calendar_year_fe", "month_of_year_fe", "nli_topic"],
        "unit": "party-topic-month",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--salience-panel", type=Path, default=DEFAULT_SALIENCE_PANEL)
    parser.add_argument("--complete-cycle-bootstrap-reps", type=int, default=4999)
    parser.add_argument("--loco-bootstrap-reps", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1711)
    return parser.parse_args()


def complete_cycle_salience_rows(panel_path: Path, reps: int, seed: int) -> tuple[pd.DataFrame, dict[str, float | int]]:
    data = load_primary_panel(panel_path)
    observed = data.loc[data["observed_next_election"].fillna(False).astype(bool)].copy()
    fit = fit_model(salience_model_spec(observed))
    mean_proximity = float(fit.used["electoral_proximity"].mean())
    government_share = float(fit.used["party_in_government"].mean())
    rows: list[dict[str, object]] = []
    for definition in salience_contrast_definitions(mean_proximity, government_share):
        if definition.panel != "B_hypothesis_contrasts":
            continue
        vector = contrast_vector(fit, definition.terms)
        estimate = float(vector @ fit.beta)
        cr1 = covariance_stats(estimate, vector, fit.cov_country, fit.country_clusters - 1)
        bootstrap = bootstrap_statistics(
            fit,
            vector,
            reps,
            salience_stable_seed(seed, f"complete_cycle::{definition.short_name}"),
        )
        critical = float(bootstrap["wild_abs_t_95_critical"])
        rows.append(
            {
                "restriction": "complete-cycle only",
                "family": "plda_topic",
                "contrast": definition.label,
                "short_name": definition.short_name,
                "role": definition.role,
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
                "observations": len(fit.used),
                "party_months": int(fit.used["party_month_fe"].nunique()),
                "countries": int(fit.used["country_code"].nunique()),
            }
        )
    summary = {
        "observations": len(fit.used),
        "party_months": int(fit.used["party_month_fe"].nunique()),
        "countries": int(fit.used["country_code"].nunique()),
        "government_share": government_share,
        "mean_proximity": mean_proximity,
        "bootstrap_reps": reps,
    }
    return pd.DataFrame(rows), summary


def fit_conditional_family(data: pd.DataFrame, family: str, sample: str, restriction: str) -> object:
    spec = FAMILY_SPECS[family]
    return fit_model(
        ModelSpec(
            model_id=f"{family}_{sample.replace(' ', '_')}",
            family=family,
            outcome="inconsistency_share",
            data=data,
            predictors=PRIMARY_PREDICTORS,
            effects=spec["effects"],
            weight="weight_equal",
            sample=sample,
            restriction=restriction,
            specification="M1",
            unit=spec["unit"],
            bootstrap=True,
        )
    )


def average_cycle_terms(fit: object) -> dict[str, float]:
    return {
        "proximity_centered": 1.0,
        "gov_x_proximity_centered": float(fit.used["party_in_government"].mean()),
    }


def average_gap_terms(fit: object) -> dict[str, float]:
    return {
        "party_in_government": 1.0,
        "gov_x_proximity_centered": float(fit.used["proximity_centered"].mean()),
    }


def conditional_contrast_row(
    fit: object,
    family: str,
    topic_label: str,
    restriction: str,
    contrast: str,
    terms: dict[str, float],
    reps: int,
    seed: int,
) -> dict[str, object]:
    vector = contrast_vector(fit, terms)
    estimate = float(vector @ fit.beta)
    cr1 = covariance_stats(estimate, vector, fit.cov_country, fit.country_clusters - 1)
    bootstrap = bootstrap_statistics(fit, vector, reps, seed)
    critical = float(bootstrap["wild_abs_t_95_critical"])
    return {
        "restriction": restriction,
        "family": family,
        "topic": topic_label,
        "contrast": contrast,
        "estimate": estimate,
        "country_cr1_se": cr1["se"],
        "country_cr1_p_two_sided": cr1["p"],
        "wild_ci_95_low": estimate - critical * cr1["se"],
        "wild_ci_95_high": estimate + critical * cr1["se"],
        "wild_p_two_sided": bootstrap["wild_p_two_sided"],
        "wild_p_mcse": bootstrap["wild_p_mcse"],
        "wild_valid_reps": bootstrap["wild_valid_reps"],
        "contrast_terms": json.dumps(terms, sort_keys=True),
        "observations": len(fit.used),
        "countries": int(fit.used["country_code"].nunique()),
    }


def complete_cycle_conditional_rows(frames: dict[str, pd.DataFrame], reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, spec in FAMILY_SPECS.items():
        observed = frames[family].loc[frames[family]["cycle_boundary_source"].eq("observed_next_election")].copy()
        fit = fit_conditional_family(observed, family, "observed next election only", "complete-cycle only")
        rows.append(
            conditional_contrast_row(
                fit, family, spec["topic_label"], "complete-cycle only", "Cycle change",
                average_cycle_terms(fit), reps, seed + 11,
            )
        )
        rows.append(
            conditional_contrast_row(
                fit, family, spec["topic_label"], "complete-cycle only", "Government-opposition gap",
                average_gap_terms(fit), reps, seed + 29,
            )
        )
    return pd.DataFrame(rows)


def leave_one_country_out_rows(frames: dict[str, pd.DataFrame], reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, spec in FAMILY_SPECS.items():
        countries = sorted(frames[family]["country_code"].dropna().astype(str).unique())
        family_index = list(FAMILY_SPECS).index(family) + 1
        for country_index, country in enumerate(countries):
            subset = frames[family].loc[frames[family]["country_code"].ne(country)].copy()
            fit = fit_conditional_family(subset, family, "natural", f"leave out {country}")
            for offset, (contrast, terms) in enumerate([
                ("Cycle change", average_cycle_terms(fit)),
                ("Government-opposition gap", average_gap_terms(fit)),
            ], start=1):
                row = conditional_contrast_row(
                    fit,
                    family,
                    spec["topic_label"],
                    "leave-one-country-out",
                    contrast,
                    terms,
                    reps,
                    seed + 1000 * family_index + 37 * country_index + 11 * offset,
                )
                row["omitted_country"] = country
                rows.append(row)
    return pd.DataFrame(rows)


def topic23_feasibility_audit() -> dict[str, object]:
    legacy_dir = BASE_DIR / "outputs" / "test_speeches" / "plda_topic_distributions"
    legacy = {}
    for path in sorted(legacy_dir.glob("*/*plda_topic_labels*.bak")):
        frame = pd.read_csv(path)
        labels = (
            frame["topic_label"]
            .astype(str)
            .str.replace(r"\s+\d+$", "", regex=True)
            .drop_duplicates()
            .tolist()
        )
        legacy[path.parent.name] = {
            "rows": int(len(frame)),
            "unique_base_labels": int(len(labels)),
            "labels": labels,
        }
    current = {}
    for country_dir in sorted((BASE_DIR / "outputs" / "test_speeches" / "plda_topic_distributions").glob("*")):
        if not country_dir.is_dir():
            continue
        path = country_dir / f"{country_dir.name}_plda_topic_distribution_log_word_count.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, nrows=1)
        topic_cols = [column for column in frame.columns if str(column).startswith("topic_")]
        current[country_dir.name] = int(len(topic_cols))
    return {
        "current_distribution_topic_columns": current,
        "legacy_original23_label_artifacts": legacy,
        "conclusion": (
            "A genuine original-23 salience rerun is not recoverable from the current cached full-sample regression panels alone. "
            "The retained broad-basis distributions have 19 columns for 15 countries and 28 for GB, while legacy 23-label "
            "topic-label artifacts survive only for CZ, EE, LV, and PL. A full 23-label check therefore requires refitting "
            "the PLDA models and rerunning inference/alignment."
        ),
    }


def render_report(salience_complete: pd.DataFrame, salience_summary: dict[str, object], conditional_complete: pd.DataFrame, loco: pd.DataFrame, feasibility: dict[str, object], reps_complete: int, reps_loco: int) -> str:
    lines = [
        "# Additional robustness checks",
        "",
        "## Complete-cycle-only headline checks",
        "",
        f"Salience complete-cycle-only sample: {salience_summary['observations']:,} party-topic-month rows, {salience_summary['party_months']:,} party-months, {salience_summary['countries']} countries, {reps_complete:,} Webb replications.",
        "",
        salience_complete.to_markdown(index=False, floatfmt='.4f'),
        "",
        conditional_complete.to_markdown(index=False, floatfmt='.4f'),
        "",
        "## Leave-one-country-out conditional H",
        "",
        f"Bootstrap replications per omitted-country contrast: {reps_loco:,}.",
        "",
        loco.head(24).to_markdown(index=False, floatfmt='.4f'),
        "",
        "## 23-label feasibility audit",
        "",
        feasibility['conclusion'],
        "",
        json.dumps(feasibility, indent=2),
        "",
    ]
    return '\n'.join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    salience_complete, salience_summary = complete_cycle_salience_rows(
        args.salience_panel, args.complete_cycle_bootstrap_reps, args.seed
    )
    frames, _, _ = load_data()
    conditional_complete = complete_cycle_conditional_rows(
        frames, args.complete_cycle_bootstrap_reps, args.seed
    )
    loco = leave_one_country_out_rows(frames, args.loco_bootstrap_reps, args.seed)
    feasibility = topic23_feasibility_audit()

    salience_complete.to_csv(args.output_dir / 'salience_complete_cycle.csv', index=False)
    conditional_complete.to_csv(args.output_dir / 'conditional_h_complete_cycle.csv', index=False)
    loco.to_csv(args.output_dir / 'conditional_h_leave_one_country_out.csv', index=False)
    (args.output_dir / 'topic23_feasibility.json').write_text(json.dumps(feasibility, indent=2), encoding='utf-8')
    report = render_report(
        salience_complete,
        salience_summary,
        conditional_complete,
        loco,
        feasibility,
        args.complete_cycle_bootstrap_reps,
        args.loco_bootstrap_reps,
    )
    (args.output_dir / 'ADDITIONAL_ROBUSTNESS_REPORT.md').write_text(report, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
