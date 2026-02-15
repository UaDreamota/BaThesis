import os
import requests
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from io import BytesIO
from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
MANIFESTO_BASE = os.getenv("BASE_URL_MANIFESTO")
MANIFESTO_KEY = os.getenv("MANIFESTO_PROJECT_API_KEY")
MPDS_KEY = "MPDS2025a"
CORPUS_VER = "2025-1"


class ManifestoApiClient:
    def __init__(self, base_url: str, api_key: str, mpds_key: str, corpus_ver: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.mpds_key = mpds_key
        self.corpus_ver = corpus_ver

    def test_api(self) -> dict:
        url = f"{self.base_url}/list_core_versions"
        resp = requests.get(url, params={"api_key": self.api_key}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_mpds(self) -> pd.DataFrame:
        url = f"{self.base_url}/get_core"
        params = {
            "api_key": self.api_key,
            "key": self.mpds_key,
            "kind": "xlsx",
            "raw": "true",
        }
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        return pd.read_excel(BytesIO(resp.content))

    def list_parties_by_country(
        self,
        mpds: pd.DataFrame,
        countryname: str | Sequence[str],
    ) -> pd.DataFrame:
        cols = [
            c
            for c in ["countryname", "party", "partyname", "partyabbrev"]
            if c in mpds.columns
        ]

        if isinstance(countryname, str):
            countries = [countryname]
        else:
            countries = list(countryname)

        countries = [c.strip().strip(",") for c in countries if c and c.strip()]

        out = (
            mpds.loc[mpds["countryname"].isin(countries), cols]
            .drop_duplicates()
            .sort_values(["countryname", "partyname", "party"])
            .reset_index(drop=True)
        )

        return out

    def list_dates_for_party(self, mpds: pd.DataFrame, party_id: int) -> pd.DataFrame:
        cols = [
            c
            for c in [
                "countryname",
                "party",
                "partyname",
                "partyabbrev",
                "date",
                "edate",
            ]
            if c in mpds.columns
        ]
        out = (
            mpds.loc[mpds["party"] == party_id, cols]
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return out

    def metadata(self, keys: list[str]) -> dict:
        url = f"{self.base_url}/metadata"
        params = {"api_key": self.api_key, "version": self.corpus_ver, "keys[]": keys}
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def texts_and_annotations(
        self, keys: list[str], translation: str | None = None
    ) -> dict:
        url = f"{self.base_url}/texts_and_annotations"
        params = {"api_key": self.api_key, "version": self.corpus_ver, "keys[]": keys}
        if translation:
            params["translation"] = translation
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def get_text_for_party_date(
        self, party_id: int, date: int, translation: str | None = "en"
    ) -> dict:
        doc_key = f"{party_id}_{date}"
        meta = self.metadata([doc_key])
        meta_items = meta.get("items", [])
        if not meta_items:
            raise RuntimeError(f"No metadata for {doc_key}")
        manifesto_id = meta_items[0].get("manifesto_id", doc_key)
        txt = self.texts_and_annotations([manifesto_id], translation=translation)
        txt_items = txt.get("items", [])
        if not txt_items:
            raise RuntimeError(f"No text for manifesto_id={manifesto_id}")
        return {
            "doc_key": doc_key,
            "manifesto_id": manifesto_id,
            "metadata_item": meta_items[0],
            "text_item": txt_items[0],
        }


manifesto_client = ManifestoApiClient(
    MANIFESTO_BASE, MANIFESTO_KEY, MPDS_KEY, CORPUS_VER
)


# How to use it:
# The list_parties_by_country - can be usefull to compare the two parties
# The list_dates_for_party - to get the manifesto dates (will be comparing to extract properly)
# The get_text_for_party_date - can be in English translation


def main():
    print(manifesto_client.test_api())
    mpds = manifesto_client.fetch_mpds()
    ua_parties = manifesto_client.list_parties_by_country(mpds, "Ukraine")
    print(ua_parties)
    print("Total unique UA parties:", len(ua_parties))
    party_id = 98952  # Yuschenko - 98615
    dates_df = manifesto_client.list_dates_for_party(mpds, party_id)
    print(dates_df)

    chosen_date = int(dates_df["date"].max())
    result = manifesto_client.get_text_for_party_date(
        party_id, chosen_date, translation=None
    )

    print("doc_key:", result["doc_key"])
    print("manifesto_id:", result["manifesto_id"])
    print("Text item keys:", list(result["text_item"].keys()))

    text_item = result["text_item"]
    text_item_df = pd.DataFrame.from_dict(text_item)
    text_item_df.to_csv("result_text.csv")
    result_df = pd.DataFrame.from_dict(result)
    result_df.to_csv("result.csv")
    return None


if __name__ == "__main__":
    main()
