import os
import requests
from pathlib import Path
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
MANIFESTO_BASE = os.getenv("BASE_URL_MANIFESTO")
MANIFESTO_KEY = os.getenv("MANIFESTO_PROJECT_API_KEY")


class ApiClient:
    def __init__(self, base_url: str, key: str):
        self.base_url = base_url.rstrip("/")
        self.key = key

        self.url_test = f"{self.base_url}/list_core_versions"

    def test_api(self):
        params = {"api_key": self.key}
        resp = requests.get(self.url_test, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()  # usually you want json, not raw resp

    def get_text(self, country, party, year):
        raise NotImplementedError

    def save_full_manifesto(self, country, party, year):
        raise NotImplementedError


manifesto_client = ApiClient(MANIFESTO_BASE, MANIFESTO_KEY)


def main():
    print(manifesto_client.test_api())
    return None


if __name__ == "__main__":
    main()
