# This file is for testing
import pandas as pd
from scripts.api_client import ManifestoApiClient
from pathlib import Path
from dotenv import load_dotenv
import os
from dataclasses import dataclass
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--country",
    default="Ukraine",
    type=str,
    help="Country of the Manifesto in question",
)
parser.add_argument(
    "--party", type=int, nargs="+", help="Party to download the manifesto"
)
parser.add_argument(
    "--save_list", default=False, type=bool, help="Flag to save the list of the parties"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
MANIFESTO_BASE = os.getenv("BASE_URL_MANIFESTO")
MANIFESTO_KEY = os.getenv("MANIFESTO_PROJECT_API_KEY")
MPDS_KEY = "MPDS2025a"
CORPUS_VER = "2025-1"


manifesto_client = ManifestoApiClient(
    MANIFESTO_BASE, MANIFESTO_KEY, MPDS_KEY, CORPUS_VER
)


def main(args) -> None:
    mpds = manifesto_client.fetch_mpds()
    # There is a possibility this step might take a lot of time. I can time it
    country_list = manifesto_client.list_parties_by_country(mpds, args.country)
    print(country_list)
    if args.save_list:
        country_list.to_csv(f"{args.country}_party_list")
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
