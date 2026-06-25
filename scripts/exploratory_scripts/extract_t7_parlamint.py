#!/usr/bin/env python3
"""Unpack ParlaMint archives on T7 and extract one CSV per corpus.

The archives are retained.  Each archive is unpacked to
<archive-dir>/ParlaMint-<CODE>/ and the existing client_mint extractor writes
<archive-dir>/ParlaMint-<CODE>_extracted.csv.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DIR = Path("/Volumes/T7/projs/parlam_teis")
ARCHIVE_PATTERN = re.compile(r"^ParlaMint-([A-Z]{2}(?:-[A-Z]{2})?)\.tgz$")
EXTRACTION_MARKER = ".archive_extraction_complete"


def extract_archive(archive: Path, destination: Path) -> None:
    """Unpack a trusted archive from the official ParlaMint repository."""
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(destination)], check=True)
    (destination / EXTRACTION_MARKER).touch()


def corpus_is_complete(corpus_dir: Path, tei_root: Path) -> bool:
    return (
        (corpus_dir / EXTRACTION_MARKER).exists()
        and any(tei_root.glob("*-listPerson.xml"))
        and any(tei_root.glob("*-listOrg.xml"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Optional corpus codes to process, e.g. AT DK NL.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing CSVs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = args.archive_dir.expanduser().resolve()
    archives = sorted(
        archive
        for archive in archive_dir.glob("ParlaMint-*.tgz")
        if ARCHIVE_PATTERN.fullmatch(archive.name)
    )
    if not archives:
        raise FileNotFoundError(f"No ParlaMint-<CODE>.tgz archives under {archive_dir}")

    requested_codes = {code.upper() for code in args.countries} if args.countries else None

    extractor = REPO_ROOT / "scripts" / "api_handling" / "client_mint.py"
    for archive in archives:
        match = ARCHIVE_PATTERN.fullmatch(archive.name)
        assert match is not None
        code = match.group(1)
        if requested_codes is not None and code not in requested_codes:
            continue
        corpus_dir = archive_dir / f"ParlaMint-{code}"
        tei_root = corpus_dir / f"ParlaMint-{code}.TEI"
        csv_path = archive_dir / f"ParlaMint-{code}_extracted.csv"

        if csv_path.exists() and not args.overwrite:
            print(f"[{code}] CSV already exists; skipping: {csv_path}", flush=True)
            continue

        if not corpus_is_complete(corpus_dir, tei_root):
            print(f"[{code}] unpacking {archive.name}", flush=True)
            if not args.dry_run:
                extract_archive(archive, corpus_dir)

        command = [
            sys.executable,
            str(extractor),
            "--tei-root",
            str(tei_root),
            "--output-dir",
            str(archive_dir),
            "--per-corpus",
        ]
        print(f"[{code}] extracting CSV: {csv_path}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
