import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from .api_client import ManifestoApiClient
except ImportError:
    from api_client import ManifestoApiClient


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
MPDS_KEY = "MPDS2025a"
CORPUS_VER = "2025-1"

__all__ = ["party_download", "build_parser", "parse_args", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country",
        default="Ukraine",
        type=str,
        help="Country of the manifesto in question",
    )
    parser.add_argument(
        "--party",
        type=int,
        nargs="+",
        help="Optional party IDs to filter the country party list",
    )
    parser.add_argument(
        "--save-list",
        "--save_list",
        action="store_true",
        dest="save_list",
        help="Save the resulting party list as a CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Defaults to '<country>_party_list' when saving.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def create_manifesto_client() -> ManifestoApiClient:
    manifesto_base = os.getenv("BASE_URL_MANIFESTO")
    manifesto_key = os.getenv("MANIFESTO_PROJECT_API_KEY")

    if not manifesto_base or not manifesto_key:
        raise RuntimeError("Missing manifesto API configuration in .env")

    return ManifestoApiClient(
        manifesto_base,
        manifesto_key,
        MPDS_KEY,
        CORPUS_VER,
    )


def party_download(
    country: str = "Ukraine",
    party_ids: Sequence[int] | None = None,
    save_list: bool = False,
    output_path: str | Path | None = None,
    client: ManifestoApiClient | None = None,
) -> pd.DataFrame:
    client = client or create_manifesto_client()
    mpds = client.fetch_mpds()
    country_list = client.list_parties_by_country(mpds, country)

    if party_ids:
        country_list = (
            country_list.loc[country_list["party"].isin(party_ids)]
            .reset_index(drop=True)
        )

    if save_list:
        destination = Path(output_path) if output_path else Path(f"{country}_party_list")
        country_list.to_csv(destination, index=False)

    return country_list


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    country_list = party_download(
        country=args.country,
        party_ids=args.party,
        save_list=args.save_list,
        output_path=args.output,
    )
    print(country_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
