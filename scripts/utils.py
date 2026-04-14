import pandas as pd
from path_config import get_parlam_csv_path

from pathlib import Path

STOPWORDS_DIR = Path(__file__).resolve().parent / "stopwords"

def load_stopwords(country_code: str) -> list[str]:
    path = STOPWORDS_DIR / f"{country_code.upper()}.txt"
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_country(country_code: str) -> pd.DataFrame:
    path = get_parlam_csv_path(country_code)
    if not path.exists():
        raise FileNotFoundError(f"Could not find ParlaMint file: {path}")

    df = pd.read_csv(
        path,
        usecols=[
            "country",
            "date",
            "party",
            "content_kind",
            "text",
            "speaker_type",
            "topic_label",
        ],
    )
    df = df[df["content_kind"] == "speech"].copy()
    df = df[df["speaker_type"] == "regular"].copy()
    df = df.dropna(subset=["party", "text", "date"]).copy()

    df["party"] = df["party"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["topic_label"] = df["topic_label"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[df["party"] != ""].copy()
    df = df[df["text"] != ""].copy()
    df = df.dropna(subset=["date"]).copy()

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True)

def merge_topics(df: pd.DataFrame):
    BROAD_TOPIC_MAP = {
    "Agriculture": "Environment_Land_Energy",
    "Civil Rights": "Institutions_Rights_Law",
    "Culture": "Culture_Identity",
    "Defense": "Foreign_Security",
    "Domestic Commerce": "Economics",
    "Education": "Welfare_Human_Development",
    "Energy": "Environment_Land_Energy",
    "Environment": "Environment_Land_Energy",
    "Foreign Trade": "Economics",
    "Government Operations": "Institutions_Rights_Law",
    "Health": "Welfare_Human_Development",
    "Housing": "Welfare_Human_Development",
    "Immigration": "Foreign_Security",
    "International Affairs": "Foreign_Security",
    "Labor": "Economics",
    "Law and Crime": "Institutions_Rights_Law",
    "Macroeconomics": "Economics",
    "Public Lands": "Environment_Land_Energy",
    "Social Welfare": "Welfare_Human_Development",
    "Technology": "Infrastructure_Technology",
    "Transportation": "Infrastructure_Technology",
    "Other": "Residual",
    "Mix": "Residual",
    }
    
    df["broad_topic"] = df["topic_label"].map(BROAD_TOPIC_MAP) 
    return df

