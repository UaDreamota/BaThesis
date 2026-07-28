from __future__ import annotations

"""Recalculate thesis-ready contrasts from the primary PLDA salience model.

The model is not re-specified.  This script refits the frozen M1 interaction
model and reports linear combinations already implied by its four coefficients:

    speech_salience ~ manifesto_salience
                      + manifesto_salience x (proximity - 0.5)
                      + manifesto_salience x government
                      + manifesto_salience x (proximity - 0.5) x government

Party-month and topic fixed effects are absorbed.  Country-clustered CR1
statistics are supporting inference.  Primary p-values and confidence intervals
use the same restricted Webb wild-cluster bootstrap-t distribution.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Importing the shared regression implementation imports matplotlib.  Point its
# cache at a writable, non-project directory before that import occurs.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "salience-recalculation-mpl")
)

from scripts.causality.empirical_regression_audit import (  # noqa: E402
    ModelSpec,
    contrast_vector,
    covariance_stats,
    fit_model,
)
from wildboottest.wildboottest import WildboottestCL  # noqa: E402


DEFAULT_PANEL = (
    BASE_DIR
    / "outputs/test_speeches/plda_salience/substantive/data/topic_salience_panel.csv"
)
DEFAULT_OUTPUT = (
    BASE_DIR / "outputs/test_speeches/salience_responsiveness_recalculation"
)
MODEL_ID = "plda_topic_m1_equal_recalculated"
PREDICTORS = [
    "manifesto_salience",
    "manifesto_x_proximity_centered",
    "manifesto_x_government",
    "manifesto_x_proximity_centered_x_government",
]
EFFECTS = ["party_month_fe", "topic"]


@dataclass(frozen=True)
class ContrastDefinition:
    panel: str
    label: str
    short_name: str
    terms: dict[str, float]
    role: str
    hypothesis: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=4_999)
    parser.add_argument("--seed", type=int, default=1_711)
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{MODEL_ID}||{label}".encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def load_primary_panel(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    required = {
        "prior_manifesto",
        "speech_salience",
        "manifesto_salience",
        "manifesto_x_proximity_centered",
        "manifesto_x_government",
        "manifesto_x_proximity_centered_x_government",
        "party_month_fe",
        "topic",
        "country_code",
        "country_party_fe",
        "country_month_fe",
        "month",
        "electoral_proximity",
        "proximity_centered",
        "party_in_government",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Salience panel is missing required columns: {missing}")
    data = data.loc[data["prior_manifesto"].fillna(False).astype(bool)].copy()
    data["weight_equal"] = 1.0
    return data


def model_spec(data: pd.DataFrame) -> ModelSpec:
    return ModelSpec(
        model_id=MODEL_ID,
        family="plda_topic",
        outcome="speech_salience",
        data=data,
        predictors=PREDICTORS,
        effects=EFFECTS,
        weight="weight_equal",
        sample="prior manifestos; complete M1 cells",
        specification="M1",
        unit="party-topic-month",
        bootstrap=True,
        notes=(
            "Equal-cell PLDA responsiveness model; party-month fixed effects "
            "absorb all party-month-level main effects."
        ),
    )


def contrast_definitions(
    mean_proximity: float, government_share: float
) -> list[ContrastDefinition]:
    centered_mean = mean_proximity - 0.5
    definitions: list[ContrastDefinition] = []
    for status, government in (("Opposition", 0.0), ("Government", 1.0)):
        for point, proximity in (
            ("Post-election", 0.0),
            ("Mid-cycle", 0.5),
            ("Pre-election", 1.0),
        ):
            centered = proximity - 0.5
            terms = {
                "manifesto_salience": 1.0,
                "manifesto_x_proximity_centered": centered,
            }
            if government:
                terms.update(
                    {
                        "manifesto_x_government": 1.0,
                        "manifesto_x_proximity_centered_x_government": centered,
                    }
                )
            definitions.append(
                ContrastDefinition(
                    panel="A_linkage",
                    label=f"{status} linkage: {point.lower()}",
                    short_name=f"{status.lower()}_linkage_p{proximity:g}",
                    terms=terms,
                    role="linkage_cell",
                )
            )

    definitions.extend(
        [
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Sample-average post- to pre-election linkage change",
                short_name="h1a_average_cycle_change",
                terms={
                    "manifesto_x_proximity_centered": 1.0,
                    "manifesto_x_proximity_centered_x_government": government_share,
                },
                role="primary",
                hypothesis="H1a",
            ),
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Opposition linkage: post- to pre-election change",
                short_name="opposition_cycle_change",
                terms={"manifesto_x_proximity_centered": 1.0},
                role="decomposition",
            ),
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Government linkage: post- to pre-election change",
                short_name="government_cycle_change",
                terms={
                    "manifesto_x_proximity_centered": 1.0,
                    "manifesto_x_proximity_centered_x_government": 1.0,
                },
                role="decomposition",
            ),
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Sample-average government-opposition linkage difference",
                short_name="h2a_average_government_gap",
                terms={
                    "manifesto_x_government": 1.0,
                    "manifesto_x_proximity_centered_x_government": centered_mean,
                },
                role="primary",
                hypothesis="H2a",
            ),
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Difference in post- to pre-election linkage change",
                short_name="difference_in_cycle_change",
                terms={"manifesto_x_proximity_centered_x_government": 1.0},
                role="decomposition",
            ),
            ContrastDefinition(
                panel="B_hypothesis_contrasts",
                label="Mid-cycle government-opposition linkage difference",
                short_name="midcycle_government_gap",
                terms={"manifesto_x_government": 1.0},
                role="supplementary",
            ),
        ]
    )
    return definitions


def bootstrap_statistics(
    fit: Any,
    vector: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    sqrt_weight = np.sqrt(fit.weights)
    x_weighted = fit.x_within * sqrt_weight[:, None]
    y_weighted = fit.y_within * sqrt_weight
    clusters = pd.factorize(fit.used["country_code"].astype(str), sort=False)[0]
    bootstrap = WildboottestCL(
        X=x_weighted,
        Y=y_weighted,
        cluster=clusters,
        R=vector,
        B=reps,
        seed=seed,
        parallel=False,
    )
    bootstrap.get_scores(
        bootstrap_type="11", impose_null=True, adj=True, cluster_adj=True
    )
    bootstrap.get_weights(weights_type="webb")
    bootstrap.get_numer()
    bootstrap.get_denom()
    bootstrap.get_tboot()
    bootstrap.get_vcov()
    bootstrap.get_tstat()
    bootstrap.get_pvalue(pval_type="two-tailed")

    draws = np.asarray(bootstrap.t_boot, dtype=float)
    valid = draws[np.isfinite(draws)]
    if len(valid) == 0:
        raise RuntimeError("The wild bootstrap produced no valid t draws.")
    # This symmetric bootstrap-t interval uses the same absolute-t rejection
    # rule as the reported two-sided p-value.  It therefore cannot create the
    # zero-in-CI/p<.05 mismatch caused by mixing CR1 intervals and wild p-values.
    critical = float(np.quantile(np.abs(valid), 0.95, method="higher"))
    p_value = float(np.mean(np.abs(valid) > abs(float(bootstrap.t_stat))))
    mcse = math.sqrt(p_value * (1.0 - p_value) / len(valid))
    return {
        "wild_t": float(bootstrap.t_stat),
        "wild_p_two_sided": p_value,
        "wild_p_mcse": mcse,
        "wild_abs_t_95_critical": critical,
        "wild_valid_reps": int(len(valid)),
    }


def coefficient_rows(fit: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(fit.terms):
        vector = np.zeros(len(fit.terms))
        vector[index] = 1.0
        estimate = float(vector @ fit.beta)
        cr1 = covariance_stats(
            estimate, vector, fit.cov_country, fit.country_clusters - 1
        )
        rows.append(
            {
                "model_id": MODEL_ID,
                "term": term,
                "estimate": estimate,
                "country_cr1_se": cr1["se"],
                "country_cr1_p_two_sided": cr1["p"],
                "country_cr1_ci_95_low": cr1["ci_low"],
                "country_cr1_ci_95_high": cr1["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def calculate_contrasts(
    fit: Any,
    definitions: list[ContrastDefinition],
    reps: int,
    seed: int,
    skip_bootstrap: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for number, definition in enumerate(definitions, start=1):
        print(f"[{number}/{len(definitions)}] {definition.label}", flush=True)
        vector = contrast_vector(fit, definition.terms)
        estimate = float(vector @ fit.beta)
        cr1 = covariance_stats(
            estimate, vector, fit.cov_country, fit.country_clusters - 1
        )
        bootstrap: dict[str, Any] = {
            "wild_t": math.nan,
            "wild_p_two_sided": math.nan,
            "wild_p_mcse": math.nan,
            "wild_abs_t_95_critical": math.nan,
            "wild_valid_reps": 0,
        }
        if not skip_bootstrap:
            bootstrap = bootstrap_statistics(
                fit, vector, reps, stable_seed(seed, definition.short_name)
            )
        critical = float(bootstrap["wild_abs_t_95_critical"])
        wild_low = estimate - critical * cr1["se"]
        wild_high = estimate + critical * cr1["se"]
        rows.append(
            {
                "model_id": MODEL_ID,
                "panel": definition.panel,
                "contrast": definition.label,
                "short_name": definition.short_name,
                "role": definition.role,
                "hypothesis": definition.hypothesis,
                "estimate": estimate,
                "country_cr1_se": cr1["se"],
                "country_cr1_p_two_sided": cr1["p"],
                "country_cr1_ci_95_low": cr1["ci_low"],
                "country_cr1_ci_95_high": cr1["ci_high"],
                **bootstrap,
                "wild_ci_95_low": wild_low,
                "wild_ci_95_high": wild_high,
                "contrast_terms": json.dumps(definition.terms, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def sample_metadata(fit: Any, source_panel: Path, reps: int, seed: int) -> dict[str, Any]:
    used = fit.used
    party_months = used.drop_duplicates("party_month_fe")
    manifesto_within_sd = used.groupby("party_month_fe")["manifesto_salience"].std(ddof=0)
    speech_within_sd = used.groupby("party_month_fe")["speech_salience"].std(ddof=0)
    return {
        "model_id": MODEL_ID,
        "source_panel": str(source_panel.resolve()),
        "model_equation": (
            "speech_salience ~ manifesto_salience + "
            "manifesto_salience*(proximity-0.5) + manifesto_salience*government + "
            "manifesto_salience*(proximity-0.5)*government | party_month_fe + topic"
        ),
        "observations_party_topic_months": int(len(used)),
        "party_months": int(used["party_month_fe"].nunique()),
        "country_parties": int(used["country_party_fe"].nunique()),
        "countries_clusters": int(used["country_code"].nunique()),
        "topic_fixed_effect_groups": int(used["topic"].nunique()),
        "party_month_fixed_effect_groups": int(used["party_month_fe"].nunique()),
        "mean_electoral_proximity": float(used["electoral_proximity"].mean()),
        "mean_centered_proximity": float(used["proximity_centered"].mean()),
        "government_share": float(used["party_in_government"].mean()),
        "manifesto_salience_p10": float(used["manifesto_salience"].quantile(0.10)),
        "manifesto_salience_p90": float(used["manifesto_salience"].quantile(0.90)),
        "speech_salience_p10": float(used["speech_salience"].quantile(0.10)),
        "speech_salience_p90": float(used["speech_salience"].quantile(0.90)),
        "mean_within_party_month_sd_manifesto_salience": float(manifesto_within_sd.mean()),
        "median_within_party_month_sd_manifesto_salience": float(manifesto_within_sd.median()),
        "mean_within_party_month_sd_speech_salience": float(speech_within_sd.mean()),
        "median_within_party_month_sd_speech_salience": float(speech_within_sd.median()),
        "government_party_months": int(party_months["party_in_government"].eq(1).sum()),
        "opposition_party_months": int(party_months["party_in_government"].eq(0).sum()),
        "bootstrap_reps_requested": int(reps),
        "bootstrap_seed": int(seed),
        "bootstrap_method": (
            "Restricted WCR11 Webb six-point wild country-cluster bootstrap-t; "
            "symmetric 95% interval uses the 95th percentile of absolute bootstrap t."
        ),
        "supporting_interval": "Country-cluster CR1 with t(G-1) critical value.",
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_p(value: Any, valid_reps: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    reps = int(valid_reps)
    if float(value) == 0.0 and reps > 0:
        return f"<{1.0 / reps:.4f}"
    return f"{float(value):.4f}"


def markdown_report(contrasts: pd.DataFrame, metadata: dict[str, Any]) -> str:
    linkage = contrasts.loc[contrasts["panel"].eq("A_linkage")].set_index("short_name")
    hypothesis = contrasts.loc[contrasts["panel"].eq("B_hypothesis_contrasts")]
    lines = [
        "# Salience responsiveness recalculation",
        "",
        "The primary M1 model is unchanged. Estimates below are linear combinations of its four interaction coefficients. All displayed 95% intervals and p-values use the same restricted Webb wild country-cluster bootstrap-t procedure.",
        "",
        "## Panel A: Estimated manifesto-speech linkage",
        "",
        "| Status | Post-election | Mid-cycle | Pre-election |",
        "|---|---:|---:|---:|",
    ]
    for status in ("opposition", "government"):
        cells = []
        for proximity in (0, 0.5, 1):
            row = linkage.loc[f"{status}_linkage_p{proximity:g}"]
            cells.append(
                f"{fmt(row['estimate'])} [{fmt(row['wild_ci_95_low'])}, {fmt(row['wild_ci_95_high'])}]"
            )
        lines.append(f"| {status.title()} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Panel B: Hypothesis-relevant contrasts",
            "",
            "| Contrast | Role | Estimate | Webb 95% CI | Wild p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in hypothesis.itertuples(index=False):
        lines.append(
            f"| {row.contrast} | {row.role} | {fmt(row.estimate)} | "
            f"[{fmt(row.wild_ci_95_low)}, {fmt(row.wild_ci_95_high)}] | "
            f"{fmt_p(row.wild_p_two_sided, row.wild_valid_reps)} |"
        )

    primary = hypothesis.loc[hypothesis["role"].eq("primary")].set_index("hypothesis")
    h1 = primary.loc["H1a"]
    h2 = primary.loc["H2a"]
    lines.extend(
        [
            "",
            "## Sample and coding",
            "",
            f"- N = {metadata['observations_party_topic_months']:,} complete party-topic-month cells; {metadata['party_months']:,} party-months; {metadata['country_parties']} country-parties; {metadata['countries_clusters']} countries/clusters.",
            f"- Observed mean proximity = {metadata['mean_electoral_proximity']:.6f}; observed government share = {metadata['government_share']:.6f}.",
            "- Since proximity is centered at 0.5, the sample-average government gap is `beta_2 + beta_3 * (mean(P) - 0.5)`.",
            "- The sample-average full-cycle change is `beta_1 + beta_3 * mean(G)`.",
            "",
            "## Primary interpretation",
            "",
            f"The H1a sample-average linkage change is {h1['estimate']:+.4f} (Webb 95% CI [{h1['wild_ci_95_low']:.4f}, {h1['wild_ci_95_high']:.4f}], p = {h1['wild_p_two_sided']:.4f}). The H2a sample-average government-minus-opposition gap is {h2['estimate']:+.4f} (Webb 95% CI [{h2['wild_ci_95_low']:.4f}, {h2['wild_ci_95_high']:.4f}], p = {h2['wild_p_two_sided']:.4f}).",
            "",
            "The four-test thesis BH q-values are reported separately in `primary_hypothesis_bh_recalculation/PRIMARY_BH_RESULTS.md`. They are not calculated from the two salience tests alone because the defined family also contains H1b and H2b.",
            "",
            "A 10-percentage-point difference in manifesto salience corresponds to one tenth of a reported linkage slope in speech-share points.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 99 and not args.skip_bootstrap:
        raise ValueError("Use at least 99 bootstrap replications; 4,999 is recommended.")
    panel_path = args.panel.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {panel_path}", flush=True)
    data = load_primary_panel(panel_path)
    fit = fit_model(model_spec(data))
    mean_proximity = float(fit.used["electoral_proximity"].mean())
    government_share = float(fit.used["party_in_government"].mean())
    definitions = contrast_definitions(mean_proximity, government_share)
    contrasts = calculate_contrasts(
        fit, definitions, args.bootstrap_reps, args.seed, args.skip_bootstrap
    )
    coefficients = coefficient_rows(fit)
    metadata = sample_metadata(
        fit, panel_path, args.bootstrap_reps, args.seed
    )

    coefficients.to_csv(output_dir / "model_coefficients.csv", index=False)
    contrasts.to_csv(output_dir / "all_contrasts.csv", index=False)
    contrasts.loc[contrasts["panel"].eq("A_linkage")].to_csv(
        output_dir / "panel_a_linkage.csv", index=False
    )
    contrasts.loc[contrasts["panel"].eq("B_hypothesis_contrasts")].to_csv(
        output_dir / "panel_b_hypothesis_contrasts.csv", index=False
    )
    (output_dir / "sample_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "SALIENCE_RESPONSIVENESS_RESULTS.md").write_text(
        markdown_report(contrasts, metadata), encoding="utf-8"
    )
    print(f"Wrote recalculation to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
