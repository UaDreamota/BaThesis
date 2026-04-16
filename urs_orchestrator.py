from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def looks_like_parlam_dir(path: Path) -> bool:
    return any(path.glob("ParlaMint-*_extracted.csv")) or any(path.glob("ParlaMint-*"))


def normalize_env_path(raw_path: str) -> Path:
    windows_drive_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw_path)
    if os.name != "nt" and windows_drive_match:
        drive = windows_drive_match.group(1).lower()
        tail = windows_drive_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{tail}")
    return Path(raw_path).expanduser()


def default_parlam_data_dir() -> Path:
    raw_path = os.getenv("PARLAM_DATA_PATH")
    if not raw_path:
        return REPO_ROOT / "data" / "parlam"

    candidate = normalize_env_path(raw_path)
    if candidate.name.lower() == "parlam":
        return candidate
    if candidate.exists() and looks_like_parlam_dir(candidate):
        return candidate

    parlam_child = candidate / "parlam"
    if parlam_child.exists() or (candidate.exists() and not looks_like_parlam_dir(candidate)):
        return parlam_child
    return candidate


def default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass(frozen=True)
class Stage:
    name: str
    command: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full PLDA salience-alignment pipeline for countries."
    )
    parser.add_argument(
        "countries",
        nargs="+",
        help="Country codes to run, for example: CZ LV GB or CZ,LV,GB.",
    )
    parser.add_argument(
        "--python",
        default=default_python(),
        help="Python executable used to run pipeline scripts.",
    )
    parser.add_argument(
        "--parlam-data-dir",
        type=Path,
        default=default_parlam_data_dir(),
        help="Directory containing ParlaMint country folders and extracted CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining countries after a failed country.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip client_mint.py ParlaMint extraction.",
    )
    parser.add_argument(
        "--skip-party-mapping",
        action="store_true",
        help="Skip topic_modeling_lda.py speech-party to MPDS mapping.",
    )
    parser.add_argument(
        "--skip-manifesto-builder",
        action="store_true",
        help="Skip manifesto_quasi_builder.py manifesto bridge/API stage.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression-panel and linear-regression stages.",
    )
    parser.add_argument(
        "--translation",
        default=None,
        help="Manifesto API translation passed to manifesto_quasi_builder.py.",
    )
    parser.add_argument(
        "--skip-manifesto-download",
        action="store_true",
        help="Pass --skip-download to manifesto_quasi_builder.py.",
    )
    parser.add_argument(
        "--client-unit",
        choices=["auto", "speech", "segment"],
        default=None,
        help="Analysis unit passed to client_mint.py.",
    )
    parser.add_argument(
        "--client-max-files",
        type=int,
        default=None,
        help="Optional max files passed to client_mint.py for test runs.",
    )
    parser.add_argument(
        "--no-notes",
        action="store_true",
        help="Pass --no-notes to client_mint.py.",
    )
    parser.add_argument(
        "--n-latent-topics",
        nargs="+",
        type=int,
        default=None,
        help="Values passed to plda_test.py.",
    )
    parser.add_argument(
        "--topics-per-label",
        nargs="+",
        type=int,
        default=None,
        help="Values passed to plda_test.py.",
    )
    parser.add_argument(
        "--rm-top",
        nargs="+",
        type=int,
        default=None,
        help="Values passed to plda_test.py.",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=None,
        help="Values passed to plda_test.py.",
    )
    parser.add_argument(
        "--eta",
        nargs="+",
        type=float,
        default=None,
        help="Values passed to plda_test.py.",
    )
    parser.add_argument(
        "--min-cf",
        type=int,
        default=None,
        help="Value passed to plda_test.py.",
    )
    parser.add_argument(
        "--plda-iterations",
        type=int,
        default=None,
        help="Training iterations passed to plda_test.py.",
    )
    parser.add_argument(
        "--plda-step",
        type=int,
        default=None,
        help="Training step passed to plda_test.py.",
    )
    parser.add_argument(
        "--plda-workers",
        type=int,
        default=None,
        help="Workers passed to plda_test.py.",
    )
    parser.add_argument(
        "--plda-seed",
        type=int,
        default=None,
        help="Seed passed to plda_test.py.",
    )
    parser.add_argument(
        "--print-iteration-logs",
        action="store_true",
        help="Pass --print-iteration-logs to plda_test.py.",
    )
    parser.add_argument(
        "--inference-iterations",
        type=int,
        default=None,
        help="Iterations passed to plda_inference.py.",
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=None,
        help="Workers passed to plda_inference.py.",
    )
    parser.add_argument(
        "--alignment-score-column",
        choices=[
            "alignment_score",
            "js_distance",
            "cosine_similarity",
            "hellinger_distance",
            "all",
        ],
        default=None,
        help="Score column passed to plda_alignment_timeseries.py.",
    )
    parser.add_argument(
        "--linear-outcome",
        default=None,
        help="Outcome column passed to plda_linear_regression.py.",
    )
    parser.add_argument(
        "--pooled-linear-regression",
        action="store_true",
        help="After country runs, fit one pooled OLS over selected country panels.",
    )
    parser.add_argument(
        "--pooled-linear-countries",
        nargs="+",
        default=None,
        help=(
            "Country list for pooled OLS. Defaults to the orchestrator country list. "
            "Accepts space- or comma-separated codes."
        ),
    )
    parser.add_argument(
        "--parlgov-country-short",
        action="append",
        default=[],
        metavar="COUNTRY=SHORT",
        help="Parlgov short code override, repeatable. Example: CZ=CZE.",
    )
    return parser


def normalize_countries(raw_countries: Iterable[str]) -> list[str]:
    countries: list[str] = []
    for raw in raw_countries:
        for part in raw.split(","):
            country = part.strip().upper()
            if country and country not in countries:
                countries.append(country)
    if not countries:
        raise ValueError("No country codes provided.")
    return countries


def parse_country_short_overrides(raw_overrides: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"Expected COUNTRY=SHORT for --parlgov-country-short, got: {raw}")
        country, short = raw.split("=", 1)
        country = country.strip().upper()
        short = short.strip().upper()
        if not country or not short:
            raise ValueError(f"Expected COUNTRY=SHORT for --parlgov-country-short, got: {raw}")
        overrides[country] = short
    return overrides


def script_path(relative_path: str) -> str:
    return str(REPO_ROOT / relative_path)


def extend_values(command: list[str], flag: str, values: Iterable[object] | None) -> None:
    if values is None:
        return
    command.append(flag)
    command.extend(str(value) for value in values)


def extend_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def country_tei_root(parlam_data_dir: Path, country_code: str) -> Path:
    return (
        parlam_data_dir
        / f"ParlaMint-{country_code}"
        / f"ParlaMint-{country_code}.TEI"
    )


def build_country_stages(
    args: argparse.Namespace,
    country_code: str,
    parlgov_short_overrides: dict[str, str],
) -> list[Stage]:
    python = args.python
    stages: list[Stage] = []

    if not args.skip_extract:
        command = [
            python,
            script_path("scripts/api_handling/client_mint.py"),
            "--tei-root",
            str(country_tei_root(args.parlam_data_dir, country_code)),
            "--output-dir",
            str(args.parlam_data_dir),
            "--per-corpus",
        ]
        extend_optional(command, "--unit", args.client_unit)
        extend_optional(command, "--max-files", args.client_max_files)
        if args.no_notes:
            command.append("--no-notes")
        stages.append(Stage("extract-parlamint", command))

    if not args.skip_party_mapping:
        stages.append(
            Stage(
                "build-party-mapping",
                [
                    python,
                    script_path("scripts/ml_algos/topic_modeling_lda.py"),
                    "--country",
                    country_code,
                ],
            )
        )

    if not args.skip_manifesto_builder:
        command = [
            python,
            script_path("scripts/api_handling/manifesto_quasi_builder.py"),
            "--country",
            country_code,
        ]
        extend_optional(command, "--translation", args.translation)
        if args.skip_manifesto_download:
            command.append("--skip-download")
        stages.append(Stage("build-manifesto-quasi", command))

    command = [
        python,
        script_path("scripts/ml_algos/plda_test.py"),
        "--c",
        country_code,
    ]
    extend_values(command, "--n-latent-topics", args.n_latent_topics)
    extend_values(command, "--topics-per-label", args.topics_per_label)
    extend_values(command, "--rm-top", args.rm_top)
    extend_values(command, "--alpha", args.alpha)
    extend_values(command, "--eta", args.eta)
    extend_optional(command, "--min-cf", args.min_cf)
    extend_optional(command, "--iterations", args.plda_iterations)
    extend_optional(command, "--step", args.plda_step)
    extend_optional(command, "--workers", args.plda_workers)
    extend_optional(command, "--seed", args.plda_seed)
    if args.print_iteration_logs:
        command.append("--print-iteration-logs")
    stages.append(Stage("train-plda", command))

    stages.append(
            Stage(
                "plot-plda-distribution",
                [
                    python,
                    script_path("scripts/vizualization/plda_distribution.py"),
                "--c",
                country_code,
            ],
        )
    )
    stages.append(
        Stage(
            "build-weighted-speech-topics",
            [
                python,
                script_path("scripts/metrics/plda_weighted_speech_topics.py"),
                "--country",
                country_code,
                "--weightings",
                "log_word_count",
            ],
        )
    )

    command = [
        python,
        script_path("scripts/ml_algos/plda_inference.py"),
        "--c",
        country_code,
    ]
    if args.inference_iterations is not None:
        command.extend(["--iterations", str(args.inference_iterations)])
    if args.inference_workers is not None:
        command.extend(["--workers", str(args.inference_workers)])
    stages.append(Stage("infer-manifesto-plda", command))

    stages.append(
        Stage(
            "build-plda-alignment",
            [
                python,
                script_path("scripts/metrics/plda_manifesto_alignment.py"),
                "--c",
                country_code,
            ],
        )
    )

    command = [
        python,
        script_path("scripts/vizualization/plda_alignment_timeseries.py"),
        "--c",
        country_code,
    ]
    extend_optional(command, "--score-column", args.alignment_score_column)
    stages.append(Stage("plot-alignment-timeseries", command))

    if not args.skip_regression:
        command = [
            python,
            script_path("scripts/metrics/plda_regression_panel.py"),
            "--country",
            country_code,
        ]
        extend_optional(
            command,
            "--parlgov-country-short",
            parlgov_short_overrides.get(country_code),
        )
        stages.append(Stage("build-regression-panel", command))

        command = [
            python,
            script_path("scripts/causality/plda_linear_regression.py"),
            "--country",
            country_code,
        ]
        extend_optional(command, "--outcome", args.linear_outcome)
        stages.append(Stage("run-linear-regression", command))

    return stages


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_stage(stage: Stage, dry_run: bool) -> None:
    print(f"\n==> {stage.name}")
    print(format_command(stage.command))
    if dry_run:
        return
    subprocess.run(stage.command, cwd=REPO_ROOT, check=True)


def run_country(
    args: argparse.Namespace,
    country_code: str,
    parlgov_short_overrides: dict[str, str],
) -> None:
    print(f"\n######## {country_code} ########")
    for stage in build_country_stages(args, country_code, parlgov_short_overrides):
        run_stage(stage, dry_run=args.dry_run)


def run_pooled_linear_regression(
    args: argparse.Namespace,
    countries: list[str],
) -> None:
    pooled_countries = (
        normalize_countries(args.pooled_linear_countries)
        if args.pooled_linear_countries
        else countries
    )
    command = [
        args.python,
        script_path("scripts/causality/plda_linear_regression.py"),
        "--countries",
        *pooled_countries,
    ]
    extend_optional(command, "--outcome", args.linear_outcome)
    run_stage(Stage("run-pooled-linear-regression", command), dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.parlam_data_dir = args.parlam_data_dir.expanduser().resolve()
    countries = normalize_countries(args.countries)
    parlgov_short_overrides = parse_country_short_overrides(args.parlgov_country_short)

    failures: list[tuple[str, str]] = []
    for country_code in countries:
        try:
            run_country(args, country_code, parlgov_short_overrides)
        except subprocess.CalledProcessError as exc:
            failures.append((country_code, f"{exc.cmd} exited with {exc.returncode}"))
            if not args.continue_on_error:
                raise
        except Exception as exc:
            failures.append((country_code, str(exc)))
            if not args.continue_on_error:
                raise

    if failures:
        print("\nFailed countries:")
        for country_code, error in failures:
            print(f"- {country_code}: {error}")
        return 1

    if args.pooled_linear_regression:
        run_pooled_linear_regression(args, countries)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
