from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    BASE_DIR
    / "outputs"
    / "test_speeches"
    / "nli_inconsistency"
    / "GB"
    / "GB_Gal_Tan_nli_pairs.csv"
)

REVIEW_COLUMNS = [
    "human_pair_relevance",
    "human_relation_label",
    "human_confidence",
    "human_notes",
]

PREFERRED_COLUMNS = [
    "nli_pair_id",
    "party",
    "month",
    "date",
    "manifesto_partyname",
    "manifesto_partyabbrev",
    "manifesto_date",
    "doc_key",
    "quasi_sentence_id",
    "speech_topic_label",
    "manifesto_predicted_topic",
    "retrieval_rank",
    "retrieval_score",
    "pair_selection_method",
    "nli_label",
    "nli_prob_entailment",
    "nli_prob_neutral",
    "nli_prob_contradiction",
    "speech_word_count_for_nli",
    "speech_selected_topic_mass",
    "speech_text",
    "manifesto_text",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a manageable manual-review sample from an NLI pairs CSV. "
            "The output preserves key automated fields and appends blank human "
            "review columns."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output CSV. Defaults to <input stem>_validation_sample.csv.",
    )
    parser.add_argument("--sample-size", default=100, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument(
        "--stratify-by",
        default="nli_label",
        choices=["nli_label", "party", "pair_selection_method", "none"],
        help=(
            "Column used for roughly balanced sampling. If the selected column is "
            "missing, the script falls back to unstratified sampling."
        ),
    )
    parser.add_argument(
        "--include-top-retrieval",
        default=0,
        type=int,
        help=(
            "Always include up to this many retrieval-rank-1 rows before random "
            "sampling the remaining rows. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--label-filter",
        nargs="*",
        default=None,
        help="Optional nli_label values to include, for example contradiction neutral.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_validation_sample{input_path.suffix}")


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0.")
    if args.include_top_retrieval < 0:
        raise ValueError("--include-top-retrieval must be >= 0.")


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input pairs CSV not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    required = {"speech_text", "manifesto_text"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return data


def apply_label_filter(data: pd.DataFrame, labels: list[str] | None) -> pd.DataFrame:
    if not labels:
        return data
    if "nli_label" not in data.columns:
        raise ValueError("--label-filter requires an nli_label column in the input.")
    allowed = {label.strip().lower() for label in labels if label.strip()}
    return data[data["nli_label"].astype(str).str.lower().isin(allowed)].copy()


def sample_stratified(
    data: pd.DataFrame,
    sample_size: int,
    stratify_by: str,
    random_state: int,
) -> pd.DataFrame:
    if len(data) <= sample_size:
        return data.copy()
    if stratify_by == "none" or stratify_by not in data.columns:
        return data.sample(n=sample_size, random_state=random_state).copy()

    groups = list(data.groupby(stratify_by, dropna=False, sort=True))
    base_n = max(sample_size // max(len(groups), 1), 1)
    sampled_parts = []
    remaining_parts = []
    used_indices: set[int] = set()

    for group_index, (_, group) in enumerate(groups):
        take_n = min(len(group), base_n)
        part = group.sample(n=take_n, random_state=random_state + group_index)
        sampled_parts.append(part)
        used_indices.update(part.index.tolist())
        remaining_parts.append(group.drop(index=part.index))

    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else data.iloc[0:0]
    remaining_n = sample_size - len(sampled)
    if remaining_n > 0:
        remaining = pd.concat(remaining_parts, ignore_index=False)
        remaining = remaining.drop(index=list(used_indices), errors="ignore")
        if not remaining.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        n=min(remaining_n, len(remaining)),
                        random_state=random_state,
                    ),
                ],
                ignore_index=False,
            )
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def include_top_retrieval_rows(
    data: pd.DataFrame,
    sample_size: int,
    include_n: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if include_n <= 0 or "retrieval_rank" not in data.columns:
        return data.iloc[0:0].copy(), data, sample_size

    retrieval_rank = pd.to_numeric(data["retrieval_rank"], errors="coerce")
    top_candidates = data[retrieval_rank == 1].copy()
    if top_candidates.empty:
        return data.iloc[0:0].copy(), data, sample_size

    top_sample = top_candidates.sample(
        n=min(include_n, sample_size, len(top_candidates)),
        random_state=random_state,
    )
    remainder = data.drop(index=top_sample.index)
    return top_sample, remainder, sample_size - len(top_sample)


def reorder_columns(data: pd.DataFrame) -> pd.DataFrame:
    leading = [col for col in PREFERRED_COLUMNS if col in data.columns]
    review = [col for col in REVIEW_COLUMNS if col in data.columns]
    remaining = [col for col in data.columns if col not in set(leading + review)]
    return data[leading + review + remaining]


def export_validation_sample(args: argparse.Namespace) -> Path:
    pairs = load_pairs(args.input)
    pairs = apply_label_filter(pairs, args.label_filter)
    if pairs.empty:
        raise ValueError("No rows remained after filtering.")

    top_sample, remainder, remaining_n = include_top_retrieval_rows(
        data=pairs,
        sample_size=args.sample_size,
        include_n=args.include_top_retrieval,
        random_state=args.random_state,
    )
    random_sample = (
        sample_stratified(
            data=remainder,
            sample_size=remaining_n,
            stratify_by=args.stratify_by,
            random_state=args.random_state,
        )
        if remaining_n > 0 and not remainder.empty
        else remainder.iloc[0:0].copy()
    )
    sample = pd.concat([top_sample, random_sample], ignore_index=True)
    sample = sample.head(args.sample_size).copy()

    for column in REVIEW_COLUMNS:
        if column not in sample.columns:
            sample[column] = ""

    sample = reorder_columns(sample)
    output_path = args.output or default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    print(
        f"Exported {len(sample):,}/{len(pairs):,} row(s) for manual validation "
        f"to: {output_path}"
    )
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    export_validation_sample(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
