from __future__ import annotations

"""Analyze the completed blinded Macroeconomics human-validation packet."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = (
    BASE_DIR / "outputs" / "test_speeches" / "macro_human_validation"
)
MODEL_LABELS = ("consistent", "unrelated", "inconsistent")
HUMAN_LABELS = (*MODEL_LABELS, "ambiguous")


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--packet-dir", type=Path, default=DEFAULT_DIR)
    out.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a status file instead of failing when judgments are missing.",
    )
    return out


def normalized(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    valid = first.isin(MODEL_LABELS) & second.isin(MODEL_LABELS)
    first = first.loc[valid]
    second = second.loc[valid]
    if first.empty:
        return float("nan")
    observed = float(first.eq(second).mean())
    first_p = first.value_counts(normalize=True)
    second_p = second.value_counts(normalize=True)
    expected = sum(
        float(first_p.get(label, 0)) * float(second_p.get(label, 0))
        for label in MODEL_LABELS
    )
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def class_metrics(
    human: pd.Series,
    predicted: pd.Series,
    weights: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in MODEL_LABELS:
        true_positive = human.eq(label) & predicted.eq(label)
        predicted_positive = predicted.eq(label)
        actual_positive = human.eq(label)
        tp = float(weights.loc[true_positive].sum())
        pp = float(weights.loc[predicted_positive].sum())
        ap = float(weights.loc[actual_positive].sum())
        precision = tp / pp if pp else float("nan")
        recall = tp / ap if ap else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else float("nan")
        )
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "weighted_support": ap,
            }
        )
    return pd.DataFrame(rows)


def write_incomplete_status(
    packet_dir: Path,
    primary_missing: int,
    second_missing: int,
) -> None:
    text = f"""# Human-validation status

Status: **human coding pending**.

- Primary judgments missing: {primary_missing}
- Second-coder judgments missing: {second_missing}

No classifier-validity statistics have been calculated. Complete the two
blinded coding files according to `CODING_CODEBOOK.md`, lock the judgments,
and rerun the analysis command.
"""
    (packet_dir / "VALIDATION_STATUS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    primary_path = args.packet_dir / "BLINDED_PRIMARY_CODING.csv"
    second_path = args.packet_dir / "BLINDED_SECOND_CODER.csv"
    key_path = args.packet_dir / "PRIVATE_CLASSIFIER_KEY.csv"
    primary = pd.read_csv(primary_path, keep_default_na=False)
    second = pd.read_csv(second_path, keep_default_na=False)
    key = pd.read_csv(key_path, low_memory=False)

    primary["human_label"] = normalized(primary["human_label"])
    primary["human_comparable"] = normalized(primary["human_comparable"])
    second["second_coder_label"] = normalized(second["second_coder_label"])
    second["second_coder_comparable"] = normalized(
        second["second_coder_comparable"]
    )

    invalid_primary = sorted(
        set(primary.loc[primary["human_label"].ne(""), "human_label"])
        - set(HUMAN_LABELS)
    )
    invalid_second = sorted(
        set(second.loc[second["second_coder_label"].ne(""), "second_coder_label"])
        - set(HUMAN_LABELS)
    )
    if invalid_primary or invalid_second:
        raise ValueError(
            f"Invalid labels: primary={invalid_primary}, second={invalid_second}"
        )
    for frame, column in (
        (primary, "human_comparable"),
        (second, "second_coder_comparable"),
    ):
        invalid = sorted(
            set(frame.loc[frame[column].ne(""), column]) - {"yes", "no"}
        )
        if invalid:
            raise ValueError(f"Invalid {column} values: {invalid}")

    primary_missing = int(primary["human_label"].eq("").sum())
    second_missing = int(second["second_coder_label"].eq("").sum())
    if primary_missing or second_missing:
        write_incomplete_status(
            args.packet_dir, primary_missing, second_missing
        )
        message = (
            f"Human coding incomplete: {primary_missing} primary and "
            f"{second_missing} second-coder labels missing."
        )
        if args.allow_incomplete:
            print(message)
            return
        raise SystemExit(message)

    merged = primary.merge(
        key[
            [
                "validation_id",
                "classifier_label",
                "poststrat_weight",
                "population_stratum_n",
                "sample_stratum_n",
            ]
        ],
        on="validation_id",
        how="left",
        validate="1:1",
    )
    merged["classifier_label"] = normalized(merged["classifier_label"])
    if merged["classifier_label"].isna().any():
        raise ValueError("Private key failed to merge.")

    resolved = merged.loc[merged["human_label"].isin(MODEL_LABELS)].copy()
    resolved["correct"] = resolved["human_label"].eq(
        resolved["classifier_label"]
    ).astype(float)
    resolved["unit_weight"] = 1.0

    metric_rows: list[dict[str, object]] = []
    for weighting, weight_column in (
        ("balanced_sample", "unit_weight"),
        ("poststratified_classifier_distribution", "poststrat_weight"),
    ):
        weights = resolved[weight_column].astype(float)
        per_class = class_metrics(
            resolved["human_label"], resolved["classifier_label"], weights
        )
        metric_rows.extend(
            {
                "weighting": weighting,
                "metric": f"{row.label}_{metric}",
                "value": float(getattr(row, metric)),
            }
            for row in per_class.itertuples()
            for metric in ("precision", "recall", "f1")
        )
        metric_rows.extend(
            [
                {
                    "weighting": weighting,
                    "metric": "accuracy",
                    "value": weighted_mean(resolved["correct"], weights),
                },
                {
                    "weighting": weighting,
                    "metric": "macro_f1",
                    "value": float(per_class["f1"].mean()),
                },
                {
                    "weighting": weighting,
                    "metric": "inconsistent_positive_predictive_value",
                    "value": float(
                        per_class.loc[
                            per_class["label"].eq("inconsistent"), "precision"
                        ].iloc[0]
                    ),
                },
            ]
        )

    comparable_model = resolved["classifier_label"].ne("unrelated")
    comparable_human = resolved["human_comparable"].eq("yes")
    comparable_accuracy = float(comparable_model.eq(comparable_human).mean())
    metric_rows.append(
        {
            "weighting": "balanced_sample",
            "metric": "comparability_accuracy",
            "value": comparable_accuracy,
        }
    )

    second_merged = second.merge(
        primary[["validation_id", "human_label", "human_comparable"]],
        on="validation_id",
        how="left",
        validate="1:1",
    )
    kappa = cohen_kappa(
        second_merged["human_label"],
        second_merged["second_coder_label"],
    )
    exact_agreement = float(
        second_merged["human_label"]
        .eq(second_merged["second_coder_label"])
        .mean()
    )
    metric_rows.extend(
        [
            {
                "weighting": "second_coder_subset",
                "metric": "exact_agreement",
                "value": exact_agreement,
            },
            {
                "weighting": "second_coder_subset",
                "metric": "cohen_kappa",
                "value": kappa,
            },
        ]
    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.packet_dir / "VALIDATION_METRICS.csv", index=False)
    confusion = pd.crosstab(
        resolved["human_label"],
        resolved["classifier_label"],
        rownames=["human_label"],
        colnames=["classifier_label"],
        dropna=False,
    ).reindex(index=MODEL_LABELS, columns=MODEL_LABELS, fill_value=0)
    confusion.to_csv(args.packet_dir / "VALIDATION_CONFUSION_MATRIX.csv")
    second_merged.to_csv(
        args.packet_dir / "SECOND_CODER_AGREEMENT_ROWS.csv", index=False
    )

    summary = {
        "status": "complete",
        "primary_rows": int(len(primary)),
        "resolved_primary_rows": int(len(resolved)),
        "ambiguous_primary_rows": int(primary["human_label"].eq("ambiguous").sum()),
        "second_coder_rows": int(len(second)),
        "balanced_accuracy": float(
            metrics.loc[
                metrics["weighting"].eq("balanced_sample")
                & metrics["metric"].eq("accuracy"),
                "value",
            ].iloc[0]
        ),
        "balanced_macro_f1": float(
            metrics.loc[
                metrics["weighting"].eq("balanced_sample")
                & metrics["metric"].eq("macro_f1"),
                "value",
            ].iloc[0]
        ),
        "cohen_kappa": kappa,
        "poststratification_note": (
            "Rows were sampled uniformly within country-by-predicted-class "
            "strata. Weights restore the completed classified-pair population "
            "distribution across those strata."
        ),
    }
    (args.packet_dir / "VALIDATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.packet_dir / "VALIDATION_STATUS.md").write_text(
        "# Human-validation status\n\nStatus: **complete**. See "
        "`VALIDATION_SUMMARY.json`, `VALIDATION_METRICS.csv`, and "
        "`VALIDATION_CONFUSION_MATRIX.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
