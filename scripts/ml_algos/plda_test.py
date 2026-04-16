import sys
from pathlib import Path

from itertools import product
import re
import time

import numpy as np
import pandas as pd
import tomotopy as tp

import argparse
from rich import print

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils import load_country, merge_topics


parser = argparse.ArgumentParser()

parser.add_argument("--c", default="CZ", type=str, help="Country to run the script on")
parser.add_argument(
    "--n-latent-topics",
    nargs="+",
    default=[1],
    type=int,
    help="One or more PLDA latent_topics values to search.",
)
parser.add_argument(
    "--topics-per-label",
    nargs="+",
    default=[2, 3, 4],
    type=int,
    help="One or more PLDA topics_per_label values to search.",
)
parser.add_argument(
    "--rm-top",
    nargs="+",
    default=[2, 5],
    type=int,
    help="One or more PLDA rm_top values to search.",
)
parser.add_argument(
    "--alpha",
    nargs="+",
    default=[0.1, 0.001],
    type=float,
    help="One or more PLDA alpha values to search.",
)
parser.add_argument(
    "--eta",
    nargs="+",
    default=[0.01],
    type=float,
    help="One or more PLDA eta values to search.",
)
parser.add_argument(
    "--min-cf",
    default=3,
    type=int,
    help="PLDA min_cf value.",
)
parser.add_argument(
    "--iterations",
    default=240,
    type=int,
    help="Number of training iterations for each grid-search candidate.",
)
parser.add_argument(
    "--step",
    default=30,
    type=int,
    help="Training step size used for progress updates.",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="Optional random seed passed to tomotopy.",
)
parser.add_argument(
    "--workers",
    default=0,
    type=int,
    help="Workers passed to tomotopy train(). Use 1 for deterministic seeded runs.",
)
parser.add_argument(
    "--grid-log-output",
    default=OUTPUT_DIR / "plda_grid_search_log.csv",
    type=Path,
    help="CSV path for grid-search run logs.",
)
parser.add_argument(
    "--distribution-output",
    default=OUTPUT_DIR / "plda_distribution.csv",
    type=Path,
    help="CSV path for the best model's document-topic distribution.",
)
parser.add_argument(
    "--speech-output",
    default=OUTPUT_DIR / "plda_individual_speeches.csv",
    type=Path,
    help=(
        "CSV path for individual speech metadata/text with the best model's "
        "document-topic distribution attached. The --c value is appended to "
        "the filename."
    ),
)
parser.add_argument(
    "--model-output",
    default=OUTPUT_DIR / "plda_model.bin",
    type=Path,
    help="Path for the fitted best PLDA model. The --c value is appended to the filename.",
)
parser.add_argument(
    "--print-iteration-logs",
    action="store_true",
    help="Print per-step training logs for every grid-search candidate.",
)


def simple_tokenizer(raw_text):
    raw_text = raw_text.lower()
    return re.findall(r"\b\w+\b", raw_text, flags=re.UNICODE)


def add_documents(model: tp.PLDAModel, main_df: pd.DataFrame) -> None:
    for row in main_df.itertuples(index=False):
        model.add_doc(
            simple_tokenizer(row.text),
            labels=[str(row.topic_label)],
            ignore_empty_words=True,
        )


def drop_empty_token_rows(main_df: pd.DataFrame) -> pd.DataFrame:
    token_mask = main_df["text"].map(lambda text: len(simple_tokenizer(str(text))) > 0)
    dropped_rows = int((~token_mask).sum())
    if dropped_rows:
        print(
            f"Dropped {dropped_rows:,} speech row(s) with no PLDA tokens "
            "after tokenization."
        )
    return main_df.loc[token_mask].reset_index(drop=True)


def validate_args(args: argparse.Namespace) -> None:
    if not args.c.strip():
        raise ValueError("--c must not be empty.")
    if args.min_cf < 0:
        raise ValueError("--min-cf must be >= 0.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0.")
    if args.step <= 0:
        raise ValueError("--step must be > 0.")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0.")
    if any(value < 0 for value in args.n_latent_topics):
        raise ValueError("--n-latent-topics values must be >= 0.")
    if any(value <= 0 for value in args.topics_per_label):
        raise ValueError("--topics-per-label values must be > 0.")
    if any(value < 0 for value in args.rm_top):
        raise ValueError("--rm-top values must be >= 0.")
    if any(value <= 0 for value in args.alpha):
        raise ValueError("--alpha values must be > 0.")
    if any(value <= 0 for value in args.eta):
        raise ValueError("--eta values must be > 0.")


def iter_grid(args: argparse.Namespace):
    yield from product(
        args.n_latent_topics,
        args.topics_per_label,
        args.rm_top,
        args.alpha,
        args.eta,
    )


def build_model(
    args: argparse.Namespace,
    latent_topics: int,
    topics_per_label: int,
    rm_top: int,
    alpha: float,
    eta: float,
) -> tp.PLDAModel:
    return tp.PLDAModel(
        min_cf=args.min_cf,
        topics_per_label=topics_per_label,
        latent_topics=latent_topics,
        rm_top=rm_top,
        alpha=alpha,
        eta=eta,
        seed=args.seed,
    )


def train_model(model: tp.PLDAModel, args: argparse.Namespace) -> None:
    trained_iterations = 0
    while trained_iterations < args.iterations:
        step = min(args.step, args.iterations - trained_iterations)
        model.train(step, workers=args.workers)
        trained_iterations += step
        if args.print_iteration_logs:
            print(
                f"iter={trained_iterations}, "
                f"ll_per_word={model.ll_per_word}, "
                f"perplexity={model.perplexity}"
            )


def get_topic_distribution(model: tp.PLDAModel) -> pd.DataFrame:
    x = np.array([doc.get_topic_dist() for doc in model.docs])
    topic_cols = [f"topic_{k}" for k in range(x.shape[1])]
    return pd.DataFrame(x, columns=topic_cols)


def country_suffixed_path(path: Path, country_code: str) -> Path:
    suffix = f"_{country_code.upper()}"
    if path.stem.upper().endswith(suffix):
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def build_speech_topic_output(
    speech_df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(speech_df) != len(topic_df):
        raise ValueError(
            "Speech rows do not match PLDA document-topic rows: "
            f"{len(speech_df):,} speech rows vs {len(topic_df):,} topic rows."
        )

    speech_output = speech_df.reset_index(drop=True).copy()
    speech_output.insert(0, "plda_doc_id", speech_output.index)
    return pd.concat(
        [speech_output, topic_df.reset_index(drop=True)],
        axis=1,
    )


def main(args: argparse.Namespace):
    validate_args(args)
    country_code = args.c.strip().upper()
    grid_log_output = country_suffixed_path(args.grid_log_output, country_code)
    distribution_output = country_suffixed_path(args.distribution_output, country_code)
    speech_output = country_suffixed_path(args.speech_output, country_code)
    model_output = country_suffixed_path(args.model_output, country_code)

    main_df = load_country(country_code)
    main_df = merge_topics(main_df)
    main_df = main_df.dropna(subset=["topic_label"]).copy()
    main_df = drop_empty_token_rows(main_df)
    main_df.info()

    grid = list(iter_grid(args))
    print(f"Running PLDA grid search with {len(grid)} candidate(s).")

    best_perplexity = float("inf")
    best_topic_df: pd.DataFrame | None = None
    best_model: tp.PLDAModel | None = None
    best_log: dict | None = None
    logs = []

    for candidate_i, (latent_topics, topics_per_label, rm_top, alpha, eta) in enumerate(
        grid, start=1
    ):
        print(
            "[cyan]PLDA candidate "
            f"{candidate_i}/{len(grid)}[/cyan]: "
            f"latent_topics={latent_topics}, "
            f"topics_per_label={topics_per_label}, "
            f"rm_top={rm_top}, "
            f"alpha={alpha}, "
            f"eta={eta}"
        )
        model = build_model(args, latent_topics, topics_per_label, rm_top, alpha, eta)
        add_documents(model, main_df)

        start = time.perf_counter()
        train_model(model, args)
        elapsed = time.perf_counter() - start

        log_row = {
            "country": country_code,
            "candidate": candidate_i,
            "n_latent_topics": latent_topics,
            "topics_per_label": topics_per_label,
            "rm_top": rm_top,
            "alpha": alpha,
            "eta": eta,
            "min_cf": args.min_cf,
            "iterations": args.iterations,
            "seed": args.seed,
            "workers": args.workers,
            "perplexity": float(model.perplexity),
            "ll_per_word": float(model.ll_per_word),
            "n_docs": len(model.docs),
            "n_topics": model.k,
            "fit_seconds": elapsed,
        }
        logs.append(log_row)
        print(
            f"perplexity={log_row['perplexity']}, "
            f"ll_per_word={log_row['ll_per_word']}, "
            f"fit_seconds={log_row['fit_seconds']}"
        )

        if log_row["perplexity"] < best_perplexity:
            best_perplexity = log_row["perplexity"]
            best_topic_df = get_topic_distribution(model)
            best_model = model
            best_log = log_row

    if best_topic_df is None or best_model is None or best_log is None:
        raise RuntimeError("PLDA grid search did not produce a fitted model.")

    grid_log_output.parent.mkdir(parents=True, exist_ok=True)
    distribution_output.parent.mkdir(parents=True, exist_ok=True)
    speech_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    logs_df = pd.DataFrame(logs)
    logs_df["is_best"] = logs_df["candidate"] == best_log["candidate"]
    logs_df.to_csv(grid_log_output, index=False)
    best_topic_df.to_csv(distribution_output, index=False)
    build_speech_topic_output(main_df, best_topic_df).to_csv(speech_output, index=False)
    best_model.save(str(model_output), full=True)

    print("[bright_magenta]Best PLDA candidate[/bright_magenta]")
    print(best_log)
    print(f"Saved grid-search log to: {grid_log_output}")
    print(f"Saved best topic distribution to: {distribution_output}")
    print(f"Saved individual speech topic data to: {speech_output}")
    print(f"Saved best PLDA model to: {model_output}")
    return None

if __name__ == ("__main__"):
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
