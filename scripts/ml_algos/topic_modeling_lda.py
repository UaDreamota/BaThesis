from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from path_config import get_parlam_csv_path, get_parlam_data_dir


BASE_DIR = Path(__file__).resolve().parents[2]
PARLA_DATA_DIR = get_parlam_data_dir()
MPDS_PATH = BASE_DIR / "data" / "MPDataset_MPDS2025a.csv"
OUTPUT_DIR = BASE_DIR / "scripts" / "party_mappings"

UKRAINE_MAPPING_PATH = (
    Path(__file__).resolve().parent / "Ukraine_party_mapping_speech_to_mpds.csv"
)
UKRAINE_NOT_IN_MPDS_PATH = Path(__file__).resolve().parent / "Ukraine_parties_not_in_mpds.csv"
UKRAINE_NOT_IN_MPDS_PARTIES_PATH = (
    Path(__file__).resolve().parent / "Ukraine_real_parties_not_in_mpds.csv"
)
UKRAINE_NOT_IN_MPDS_PARLIAMENTARY_PATH = (
    Path(__file__).resolve().parent / "Ukraine_parliamentary_labels_not_in_mpds.csv"
)

UKRAINE_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "АПУ06": {"party_id": 98811, "note": "Agrarian Party label in the speech data."},
    "АПУ96": {"party_id": 98811, "note": "Agrarian Party label in the speech data."},
    "БПП": {"party_id": 98631, "preferred_partyname": "Petro Poroshenko Bloc ‘Solidarity'"},
    "Батьківщина": {"party_id": 98617},
    "Голос": {"party_id": 98450},
    "Громада": {"party_id": 98324},
    "КПУ": {"party_id": 98221},
    "КУН": {"party_id": 98711},
    "Ляшко": {"party_id": 98730, "preferred_partyname": "Radical Party of Oleh Lyashko"},
    "НДП": {"party_id": 98426},
    "НСНУ": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
    "НУ": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
    "НФ": {"party_id": 98618, "preferred_partyname": "People's Front"},
    "ОПЗЖ": {"party_id": 98340},
    "Опоблок": {
        "party_id": 98910,
        "note": "MPDS contains two Opposition Bloc rows; this file uses the 2014 entry.",
    },
    "ПП": {"party_id": 98423},
    "ПР": {"party_id": 98952},
    "РУХ": {"party_id": 98611},
    "СДПУ(о)": {"party_id": 98427},
    "СН": {"party_id": 98440},
    "СПУ": {"party_id": 98321},
    "Самопоміч": {"party_id": 98620},
    "Свобода": {"party_id": 98720},
    "УВперед": {"party_id": 98428},
    "УДАР": {"party_id": 98630},
    "ЄС": {"party_id": 98631, "preferred_partyname": "European Solidarity"},
    "бБЮТ": {"party_id": 98616},
    "бЗаЄдУ": {"party_id": 98429},
    "бЛитвин": {"party_id": 98081},
    "бНУ02": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
    "бНУ06": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
    "бНУНС": {
        "party_id": 98615,
        "preferred_partyname": "‘Our Ukraine - People's Self-Defense'",
    },
    "фАПУ": {"party_id": 98811, "note": "Faction label mapped to Agrarian Party of Ukraine."},
    "фБПП": {"party_id": 98631, "preferred_partyname": "Petro Poroshenko Bloc ‘Solidarity'"},
    "фБЮТ": {"party_id": 98616},
    "фБатьківщина": {"party_id": 98617},
    "фВпередУ": {"party_id": 98428},
    "фГолос": {"party_id": 98450},
    "фКПУ": {"party_id": 98221},
    "фНДП": {"party_id": 98426},
    "фНУ02": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
    "фНФ": {"party_id": 98618, "preferred_partyname": "People's Front"},
    "фОПЗЖ": {"party_id": 98340},
    "фОпоблок": {
        "party_id": 98910,
        "note": "Faction label mapped to the 2014 Opposition Bloc MPDS row.",
    },
    "фПР": {"party_id": 98952},
    "фСДПУ(о)": {"party_id": 98427},
    "фСН": {"party_id": 98440},
    "фСПУ": {"party_id": 98321},
    "фСамопоміч": {"party_id": 98620},
    "фУДАР": {"party_id": 98630},
    "фЄС": {"party_id": 98631, "preferred_partyname": "European Solidarity"},
    "фЄдУ": {"party_id": 98429, "note": "Faction label aligned with For a United Ukraine!"},
    "фЄдУп": {"party_id": 98429, "note": "Faction label aligned with For a United Ukraine!"},
    "фрНСНУ05": {
        "party_id": 98615,
        "preferred_partyname": "Viktor Yushchenko Bloc ‘Our Ukraine'",
    },
}

GB_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "A": {"party_id": 51430, "preferred_partyname": "Alliance Party of Northern Ireland"},
    "CON": {"party_id": 51620},
    "GP": {"party_id": 51110, "preferred_partyname": "Green Party of England and Wales"},
    "LAB": {"party_id": 51320},
    "LD": {"party_id": 51421},
}

GB_NON_PARTY_LABELS: dict[str, dict[str, str]] = {
    "BI": {"label_type": "parliamentary_label", "note": "Bishops are not a manifesto party."},
    "CB": {"label_type": "parliamentary_label", "note": "Crossbench is a Lords grouping, not a manifesto party."},
    "I": {"label_type": "independent", "note": "Independent speakers are not a manifesto party."},
    "IL": {"label_type": "independent", "note": "Independent Labour is not an MPDS manifesto party."},
    "L8TA": {"label_type": "independent", "note": "Independent Liberal Democrat is not an MPDS manifesto party."},
    "LI": {"label_type": "independent", "note": "Labour Independent is not an MPDS manifesto party."},
    "QMZZ": {"label_type": "independent", "note": "Independent Ulster Unionist is not an MPDS manifesto party."},
    "ZKPW": {"label_type": "independent", "note": "Independent Social Democrat is not an MPDS manifesto party."},
}

EE_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "EKRE": {"party_id": 83720},
    "ISA": {
        "party_id": 83611,
        "preferred_partyname": "Pro Patria and Res Publica Union",
        "note": "Speech label uses the later Isamaa abbreviation.",
    },
    "KE": {
        "party_id": 83411,
        "preferred_partyname": "Estonian Center Party",
        "note": "Speech label KE refers to Keskerakond / Estonian Center Party, not the older Coalition Party.",
    },
    "RE": {"party_id": 83430, "preferred_partyname": "Estonian Reform Party"},
    "SDE": {"party_id": 83410, "preferred_partyname": "Social Democratic Party"},
    "VABA": {"party_id": 83440, "preferred_partyname": "Free Party"},
}

PL_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "KO": {
        "party_id": 92435,
        "preferred_partyname": "Civic Platform",
        "note": "Civic Coalition speech label aligned with the Civic Platform MPDS manifesto series.",
    },
    "KP-PSL": {
        "party_id": 92050,
        "preferred_partyname": "Polish Coalition",
        "note": "Parliamentary club label aligned with the Polish Coalition / PSL MPDS manifesto series.",
    },
    "Konfederacja": {
        "party_id": 92070,
        "preferred_partyname": "Confederation Liberty and Independence",
    },
    "Kukiz15": {
        "party_id": 92720,
        "preferred_partyname": "Kukiz'15",
    },
    "Lewica": {
        "party_id": 92022,
        "preferred_partyname": "The Left",
        "note": "Speech label aligned with the post-2019 The Left MPDS manifesto series.",
    },
    "PiS": {"party_id": 92436, "preferred_partyname": "Law and Justice"},
    "PrzywrócićPrawo": {
        "party_id": 92720,
        "preferred_partyname": "Kukiz'15",
        "note": "Parliamentary circle formed by MPs elected from Kukiz'15 lists; no separate MPDS manifesto row.",
    },
    "Teraz": {
        "party_id": 92450,
        "preferred_partyname": "Modern",
        "note": "Parliamentary circle split from Modern; no separate MPDS manifesto row.",
    },
    "UPR": {"party_id": 92432, "preferred_partyname": "Union of Real Politics"},
}

LV_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "AP": {"party_id": 87042, "preferred_partyname": "Development/For!"},
    "JV": {"party_id": 87063, "preferred_partyname": "New Unity"},
    "JK": {"party_id": 87640, "preferred_partyname": "New Conservative Party"},
    "KPV-LV": {"party_id": 87730, "preferred_partyname": "Who owns the state?"},
    "LRA": {"party_id": 87901, "preferred_partyname": "Latvian Association of Regions"},
    "NSL": {
        "party_id": 87630,
        "preferred_partyname": "For Latvia from the Heart",
        "note": "Speech label uses the Latvian abbreviation No sirds Latvijai.",
    },
    "SASKAŅA": {
        "party_id": 87340,
        "preferred_partyname": 'Social Democartic Party "Harmony"',
        "note": "Speech label uses the Latvian party name SASKAŅA.",
    },
    "VIENOTĪBA": {
        "party_id": 87062,
        "preferred_partyname": "Unity",
        "note": "Speech label uses the Latvian party name Vienotiba / Unity.",
    },
    "ZZS": {"party_id": 87110, "preferred_partyname": "Greens' and Farmers’ Union"},
}

HU_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "DK": {"party_id": 86221, "preferred_partyname": "Democratic Coalition"},
    "DK-frakció": {
        "party_id": 86221,
        "preferred_partyname": "Democratic Coalition",
        "note": "Faction label mapped to Democratic Coalition.",
    },
    "EGYÜTT": {
        "party_id": 86340,
        "preferred_partyname": "Together 2014 -Dialogue for Hungary Electoral Alliance",
    },
    "Fidesz": {
        "party_id": 86421,
        "preferred_partyname": "Federation of Young Democrats",
    },
    "Fidesz-frakció": {
        "party_id": 86421,
        "preferred_partyname": "Federation of Young Democrats",
        "note": "Faction label mapped to Fidesz.",
    },
    "Huxit": {
        "party_id": 86711,
        "preferred_partyname": "Our Homeland Movement",
        "note": "Volner/Huxit splinter label has no separate MPDS row; aligned with the nearest Our Homeland MPDS series.",
    },
    "JOBBIK": {"party_id": 86710, "preferred_partyname": "Movement for a Better Hungary"},
    "JOBBIK-frakció": {
        "party_id": 86710,
        "preferred_partyname": "Movement for a Better Hungary",
        "note": "Faction label mapped to Jobbik.",
    },
    "KDNP": {
        "party_id": 86421,
        "preferred_partyname": "Alliance of Federation of Young Democrats - Christian Democratic People's Party",
        "note": "Modern KDNP speeches are aligned with the Fidesz-KDNP joint manifesto series.",
    },
    "LMP": {"party_id": 86110, "preferred_partyname": "Politics Can Be Different"},
    "LMP-frakció": {
        "party_id": 86110,
        "preferred_partyname": "Politics Can Be Different",
        "note": "Faction label mapped to LMP.",
    },
    "MLP": {
        "party_id": 86221,
        "preferred_partyname": "Democratic Coalition",
        "note": "Hungarian Liberal Party has no standalone MPDS row; aligned with DK as its closest later parliamentary affiliation.",
    },
    "Mi Hazánk": {"party_id": 86711, "preferred_partyname": "Our Homeland Movement"},
    "Momentum": {
        "party_id": 86001,
        "preferred_partyname": "United for Hungary",
        "note": "Momentum has no standalone MPDS row; post-2022 speech label aligned with the United for Hungary coalition manifesto.",
    },
    "Momentum-frakció": {
        "party_id": 86001,
        "preferred_partyname": "United for Hungary",
        "note": "Momentum faction has no standalone MPDS row; aligned with the United for Hungary coalition manifesto.",
    },
    "MSZP": {"party_id": 86220, "preferred_partyname": "Hungarian Socialist Party"},
    "MSZP-frakció": {
        "party_id": 86220,
        "preferred_partyname": "Hungarian Socialist Party",
        "note": "Faction label mapped to Hungarian Socialist Party.",
    },
    "Párbeszéd": {"party_id": 86111, "preferred_partyname": "Dialogue for Hungary"},
    "Párbeszéd-frakció": {
        "party_id": 86111,
        "preferred_partyname": "Dialogue for Hungary",
        "note": "Faction label mapped to Dialogue for Hungary.",
    },
    "Szolidaritás": {
        "party_id": 86340,
        "preferred_partyname": "Together 2014 -Dialogue for Hungary Electoral Alliance",
        "note": "Hungarian Solidarity Movement has no standalone MPDS row; aligned with the Together 2014 alliance series.",
    },
    "Volner": {
        "party_id": 86711,
        "preferred_partyname": "Our Homeland Movement",
        "note": "Volner Party has no standalone MPDS row; aligned with the nearest Our Homeland MPDS series.",
    },
    "ÚK": {
        "party_id": 86110,
        "preferred_partyname": "Politics Can Be Different",
        "note": "Új Kezdet has no standalone MPDS row; aligned with the LMP electoral-cooperation manifesto series.",
    },
}

HU_NON_PARTY_LABELS: dict[str, dict[str, str]] = {
    "MNOÖ": {
        "label_type": "minority_list",
        "note": "Hungarian German national minority list, not treated as an MPDS manifesto party.",
    },
}

SI_MANUAL_MPDS_MAP: dict[str, dict[str, object]] = {
    "DL": {
        "party_id": 97450,
        "preferred_partyname": "Gregor Virant's Civic List",
        "note": "Speech label DL aligned with the Civic List / Gregor Virant MPDS manifesto series.",
    },
    "DLGV": {
        "party_id": 97450,
        "preferred_partyname": "Gregor Virant's Civic List",
    },
    "DeSUS": {
        "party_id": 97951,
        "preferred_partyname": "Democratic Party of Pensioners of Slovenia",
    },
    "Konkretno": {
        "party_id": 97461,
        "preferred_partyname": "Modern Centre Party",
        "note": "Konkretno speech label aligned with the SMC / Modern Centre Party MPDS manifesto series.",
    },
    "LDS": {"party_id": 97421, "preferred_partyname": "Liberal Democracy of Slovenia"},
    "LMŠ": {"party_id": 97341, "preferred_partyname": "List of Marjan Šarec"},
    "Levica": {
        "party_id": 97230,
        "preferred_partyname": "The Left",
    },
    "Lipa": {
        "party_id": 97710,
        "preferred_partyname": "Slovenian National Party",
        "note": "Lipa was a splinter parliamentary label without a separate MPDS manifesto row; aligned with SNS.",
    },
    "NSi": {
        "party_id": 97522,
        "preferred_partyname": "New Slovenian Christian People’s Party",
    },
    "PS": {
        "party_id": 97340,
        "preferred_partyname": "Zoran Janković's List - Positive Slovenia",
    },
    "SAB": {
        "party_id": 97460,
        "preferred_partyname": "Party of Alenka Bratušek",
    },
    "SD": {"party_id": 97322, "preferred_partyname": "Social Democratic Party"},
    "SDS": {"party_id": 97330, "preferred_partyname": "Slovenian Democratic Party"},
    "SLS": {"party_id": 97521, "preferred_partyname": "Slovenian People's Party"},
    "SLS+SKD": {
        "party_id": 97521,
        "preferred_partyname": "Slovenian People's Party",
        "note": "Joint SLS+SKD parliamentary label aligned with the SLS MPDS manifesto series.",
    },
    "SMC": {
        "party_id": 97461,
        "preferred_partyname": "Modern Centre Party",
    },
    "SMS": {"party_id": 97952, "preferred_partyname": "Party of Slovenian Youth"},
    "SNS": {"party_id": 97710, "preferred_partyname": "Slovenian National Party"},
    "ZL": {"party_id": 97020, "preferred_partyname": "United Left"},
    "ZLSD": {
        "party_id": 97321,
        "preferred_partyname": "Associated List of Social Democrats",
    },
    "ZaAB": {
        "party_id": 97460,
        "preferred_partyname": "Alliance of Alenka Bratušek",
    },
    "Zares": {
        "party_id": 97440,
        "preferred_partyname": "For Real",
    },
}

SI_NON_PARTY_LABELS: dict[str, dict[str, str]] = {
    "NP": {
        "label_type": "unaffiliated",
        "note": "Unaffiliated deputies label, not treated as an MPDS manifesto party.",
    },
    "NeP": {
        "label_type": "unaffiliated",
        "note": "Unaffiliated deputies label, not treated as an MPDS manifesto party.",
    },
}

CZ_NON_PARTY_LABELS: dict[str, dict[str, str]] = {
    "PirSTAN": {
        "label_type": "coalition",
        "note": "PirSTAN is a parliamentary coalition label, not a single MPDS party.",
    },
    "SPOLU": {
        "label_type": "coalition",
        "note": "SPOLU is a parliamentary coalition label, not a single MPDS party.",
    },
}


@dataclass(frozen=True)
class CountryConfig:
    code: str
    manifesto_country: str
    parla_path: Path
    manual_map: dict[str, dict[str, object]] = field(default_factory=dict)
    non_party_labels: dict[str, dict[str, str]] = field(default_factory=dict)
    legacy_output_paths: tuple[Path, Path, Path, Path] | None = None


COUNTRY_CONFIGS: dict[str, CountryConfig] = {
    "UA": CountryConfig(
        code="UA",
        manifesto_country="Ukraine",
        parla_path=get_parlam_csv_path("UA"),
        manual_map=UKRAINE_MANUAL_MPDS_MAP,
    ),
    "GB": CountryConfig(
        code="GB",
        manifesto_country="United Kingdom",
        parla_path=get_parlam_csv_path("GB"),
        manual_map=GB_MANUAL_MPDS_MAP,
        non_party_labels=GB_NON_PARTY_LABELS,
    ),
    "EE": CountryConfig(
        code="EE",
        manifesto_country="Estonia",
        parla_path=get_parlam_csv_path("EE"),
        manual_map=EE_MANUAL_MPDS_MAP,
    ),
    "PL": CountryConfig(
        code="PL",
        manifesto_country="Poland",
        parla_path=get_parlam_csv_path("PL"),
        manual_map=PL_MANUAL_MPDS_MAP,
    ),
    "LV": CountryConfig(
        code="LV",
        manifesto_country="Latvia",
        parla_path=get_parlam_csv_path("LV"),
        manual_map=LV_MANUAL_MPDS_MAP,
    ),
    "CZ": CountryConfig(
        code="CZ",
        manifesto_country="Czech Republic",
        parla_path=get_parlam_csv_path("CZ"),
        non_party_labels=CZ_NON_PARTY_LABELS,
    ),
    "LT": CountryConfig(
        code="LT",
        manifesto_country="Lithuania",
        parla_path=get_parlam_csv_path("LT"),
    ),
    "HU": CountryConfig(
        code="HU",
        manifesto_country="Hungary",
        parla_path=get_parlam_csv_path("HU"),
        manual_map=HU_MANUAL_MPDS_MAP,
        non_party_labels=HU_NON_PARTY_LABELS,
    ),
    "SI": CountryConfig(
        code="SI",
        manifesto_country="Slovenia",
        parla_path=get_parlam_csv_path("SI"),
        manual_map=SI_MANUAL_MPDS_MAP,
        non_party_labels=SI_NON_PARTY_LABELS,
    ),
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def canonicalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return "".join(ch for ch in no_accents.upper() if ch.isalnum())


def load_manifesto_parties(country: str) -> pd.DataFrame:
    mpds = pd.read_csv(MPDS_PATH, low_memory=False)
    return (
        mpds.loc[mpds["countryname"] == country, ["countryname", "party", "partyname", "partyabbrev"]]
        .drop_duplicates()
        .sort_values(["partyname", "party"])
        .reset_index(drop=True)
    )


def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def build_speech_counts(parla_path: Path) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(parla_path, usecols=["party"], chunksize=200_000, low_memory=False):
        labels = chunk["party"].dropna().astype(str).str.strip()
        labels = labels[labels.ne("")]
        counts.update(labels.tolist())

    return (
        pd.DataFrame(sorted(counts.items()), columns=["speech_party", "n_rows"])
        .astype({"speech_party": str, "n_rows": int})
        .reset_index(drop=True)
    )


def compare_party_names(manifesto_df: pd.DataFrame, speech_labels: list[str]) -> dict[str, list[str]]:
    manifesto_work = manifesto_df.copy()
    manifesto_work["partyabbrev_norm"] = normalize_series(manifesto_work["partyabbrev"])
    manifesto_work["partyname_norm"] = normalize_series(manifesto_work["partyname"])

    speech_norm = {normalize_label(value) for value in speech_labels if normalize_label(value)}
    manifesto_abbrev = {value for value in manifesto_work["partyabbrev_norm"].unique() if value}
    manifesto_names = {value for value in manifesto_work["partyname_norm"].unique() if value}

    return {
        "match_abbrev": sorted(speech_norm & manifesto_abbrev),
        "match_name": sorted(speech_norm & manifesto_names),
        "match_any": sorted(speech_norm & (manifesto_abbrev | manifesto_names)),
        "only_in_speech": sorted(speech_norm - (manifesto_abbrev | manifesto_names)),
        "unused_abbrev": sorted(manifesto_abbrev - speech_norm),
        "unused_names": sorted(manifesto_names - speech_norm),
    }


def resolve_mpds_row(
    manifesto_df: pd.DataFrame,
    party_id: int,
    preferred_partyname: str | None = None,
) -> pd.Series:
    matches = manifesto_df.loc[manifesto_df["party"] == party_id]
    if matches.empty:
        raise KeyError(f"Missing MPDS record for party id {party_id}")

    if preferred_partyname:
        preferred = matches.loc[matches["partyname"] == preferred_partyname]
        if not preferred.empty:
            return preferred.iloc[0]

    return matches.iloc[0]


def build_lookup(
    manifesto_df: pd.DataFrame,
    column: str,
    normalizer,
) -> dict[str, list[pd.Series]]:
    lookup: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in manifesto_df.iterrows():
        key = normalizer(row[column])
        if key:
            lookup[key].append(row)
    return dict(lookup)


def resolve_unique_lookup_match(matches: list[pd.Series]) -> tuple[pd.Series | None, str]:
    if not matches:
        return None, ""

    by_party_id: dict[int, list[pd.Series]] = defaultdict(list)
    for row in matches:
        by_party_id[int(row["party"])].append(row)

    if len(by_party_id) == 1:
        rows = next(iter(by_party_id.values()))
        rows = sorted(rows, key=lambda row: (str(row["partyname"]), int(row["party"])))
        return rows[0], ""

    candidate_text = "; ".join(
        f"{party_id}:{rows[0]['partyname']}"
        for party_id, rows in sorted(by_party_id.items(), key=lambda item: item[0])
    )
    return None, f"Ambiguous MPDS match candidates: {candidate_text}"


def build_manual_lookup(
    manual_map: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    raw_lookup = {str(key): value for key, value in manual_map.items()}
    normalized_lookup = {normalize_label(key): value for key, value in manual_map.items()}
    canonical_lookup = {canonicalize_label(key): value for key, value in manual_map.items()}
    return raw_lookup, normalized_lookup, canonical_lookup


def label_type(label: str, config: CountryConfig) -> str:
    if label in config.non_party_labels:
        return config.non_party_labels[label]["label_type"]

    if config.code == "UA":
        lowered = label.lower()
        if lowered.startswith("ф"):
            return "faction"
        if lowered.startswith("г") or lowered.startswith("гр"):
            return "group"
        if lowered.startswith("б"):
            return "bloc"

    if config.code == "HU" and label.endswith("-frakció"):
        return "faction"

    return "party"


def default_unmapped_note(label: str, config: CountryConfig) -> str:
    current_label_type = label_type(label, config)
    if label in config.non_party_labels:
        return config.non_party_labels[label]["note"]
    if current_label_type == "group":
        return "Parliamentary group label with no direct MPDS party match."
    if current_label_type == "faction":
        return "Faction label with no confident direct MPDS party match."
    if current_label_type == "bloc":
        return "Bloc label with no confident direct MPDS party match."
    if current_label_type != "party":
        return "Parliamentary label present in speech data but not treated as a manifesto party."
    return (
        f"Party label present in {config.code} parliamentary speeches but not mapped "
        f"to an MPDS {config.manifesto_country} party."
    )


def country_output_paths(config: CountryConfig) -> tuple[Path, Path, Path, Path]:
    if config.legacy_output_paths is not None:
        return config.legacy_output_paths

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUTPUT_DIR / config.code
    return (
        prefix.with_name(f"{config.code}_party_mapping_speech_to_mpds.csv"),
        prefix.with_name(f"{config.code}_labels_not_in_mpds.csv"),
        prefix.with_name(f"{config.code}_real_parties_not_in_mpds.csv"),
        prefix.with_name(f"{config.code}_parliamentary_labels_not_in_mpds.csv"),
    )


def build_speech_to_mpds_mapping(
    manifesto_df: pd.DataFrame,
    speech_counts_df: pd.DataFrame,
    config: CountryConfig,
) -> pd.DataFrame:
    raw_manual_lookup, normalized_manual_lookup, canonical_manual_lookup = build_manual_lookup(
        config.manual_map
    )
    exact_by_abbrev = build_lookup(manifesto_df, "partyabbrev", normalize_label)
    exact_by_name = build_lookup(manifesto_df, "partyname", normalize_label)
    canonical_by_abbrev = build_lookup(manifesto_df, "partyabbrev", canonicalize_label)
    canonical_by_name = build_lookup(manifesto_df, "partyname", canonicalize_label)

    rows: list[dict[str, object]] = []
    for speech_party, n_rows in speech_counts_df.itertuples(index=False):
        norm_label = normalize_label(speech_party)
        canonical_label = canonicalize_label(speech_party)
        mapping_status = "unmapped"
        match_method = ""
        mapping_note = ""
        mpds_row = None
        ambiguity_note = ""

        manual_config = (
            raw_manual_lookup.get(speech_party)
            or normalized_manual_lookup.get(norm_label)
            or canonical_manual_lookup.get(canonical_label)
        )

        if manual_config is not None:
            mpds_row = resolve_mpds_row(
                manifesto_df,
                int(manual_config["party_id"]),
                manual_config.get("preferred_partyname"),
            )
            mapping_status = "mapped"
            match_method = "manual_alias"
            mapping_note = str(manual_config.get("note", ""))
        else:
            candidate_methods = [
                ("exact_abbrev", exact_by_abbrev.get(norm_label, [])),
                ("exact_name", exact_by_name.get(norm_label, [])),
                ("canonical_abbrev", canonical_by_abbrev.get(canonical_label, [])),
                ("canonical_name", canonical_by_name.get(canonical_label, [])),
            ]

            for method, matches in candidate_methods:
                row, note = resolve_unique_lookup_match(matches)
                if row is not None:
                    mpds_row = row
                    mapping_status = "mapped"
                    match_method = method
                    mapping_note = ""
                    break
                if note and not ambiguity_note:
                    ambiguity_note = note

        if mpds_row is None:
            mapping_note = ambiguity_note or default_unmapped_note(speech_party, config)

        rows.append(
            {
                "speech_party": speech_party,
                "speech_party_norm": norm_label,
                "speech_party_canonical": canonical_label,
                "speech_label_type": label_type(speech_party, config),
                "n_rows": int(n_rows),
                "mapping_status": mapping_status,
                "match_method": match_method,
                "mpds_party_id": None if mpds_row is None else int(mpds_row["party"]),
                "mpds_partyname": None if mpds_row is None else mpds_row["partyname"],
                "mpds_partyabbrev": None if mpds_row is None else mpds_row["partyabbrev"],
                "mapping_note": mapping_note,
            }
        )

    return pd.DataFrame(rows)


def create_country_mapping_csv(
    config: CountryConfig,
    output_path: Path | None = None,
) -> Path:
    manifesto_df = load_manifesto_parties(config.manifesto_country)
    speech_counts_df = build_speech_counts(config.parla_path)
    mapping_df = build_speech_to_mpds_mapping(manifesto_df, speech_counts_df, config)
    resolved_output = output_path or country_output_paths(config)[0]
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(resolved_output, index=False)
    return resolved_output


def create_country_unmapped_csvs(
    config: CountryConfig,
    mapping_path: Path | None = None,
    unmapped_path: Path | None = None,
    parties_path: Path | None = None,
    parliamentary_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    resolved_mapping, resolved_unmapped, resolved_parties, resolved_parliamentary = country_output_paths(
        config
    )

    mapping_file = mapping_path or resolved_mapping
    unmapped_file = unmapped_path or resolved_unmapped
    parties_file = parties_path or resolved_parties
    parliamentary_file = parliamentary_path or resolved_parliamentary

    mapping_df = pd.read_csv(mapping_file, keep_default_na=False)
    unmapped_df = (
        mapping_df.loc[
            mapping_df["mapping_status"] == "unmapped",
            ["speech_party", "speech_label_type", "n_rows", "mapping_note"],
        ]
        .sort_values(["speech_label_type", "speech_party"])
        .reset_index(drop=True)
    )
    parties_df = (
        unmapped_df.loc[unmapped_df["speech_label_type"] == "party"]
        .sort_values("speech_party")
        .reset_index(drop=True)
    )
    parliamentary_df = (
        unmapped_df.loc[unmapped_df["speech_label_type"] != "party"]
        .sort_values(["speech_label_type", "speech_party"])
        .reset_index(drop=True)
    )

    unmapped_file.parent.mkdir(parents=True, exist_ok=True)
    unmapped_df.to_csv(unmapped_file, index=False)
    parties_df.to_csv(parties_file, index=False)
    parliamentary_df.to_csv(parliamentary_file, index=False)
    return unmapped_file, parties_file, parliamentary_file


def create_ukraine_mapping_csv(output_path: Path = UKRAINE_MAPPING_PATH) -> Path:
    return create_country_mapping_csv(COUNTRY_CONFIGS["UA"], output_path=output_path)


def create_ukraine_unmapped_csvs(
    mapping_path: Path = UKRAINE_MAPPING_PATH,
    unmapped_path: Path = UKRAINE_NOT_IN_MPDS_PATH,
    parties_path: Path = UKRAINE_NOT_IN_MPDS_PARTIES_PATH,
    parliamentary_path: Path = UKRAINE_NOT_IN_MPDS_PARLIAMENTARY_PATH,
) -> tuple[Path, Path, Path]:
    return create_country_unmapped_csvs(
        COUNTRY_CONFIGS["UA"],
        mapping_path=mapping_path,
        unmapped_path=unmapped_path,
        parties_path=parties_path,
        parliamentary_path=parliamentary_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map parliamentary speech-side party labels onto MPDS parties."
    )
    parser.add_argument(
        "--country",
        nargs="+",
        default=None,
        help="Country codes to process, for example: UA GB EE LV CZ LT HU PL SI",
    )
    return parser.parse_args()


def resolve_country_codes(raw_codes: list[str] | None) -> list[str]:
    if not raw_codes:
        return list(COUNTRY_CONFIGS)

    resolved: list[str] = []
    for code in raw_codes:
        normalized = code.strip().upper()
        if normalized not in COUNTRY_CONFIGS:
            raise ValueError(
                f"Unsupported country code {code!r}. Expected one of: {', '.join(sorted(COUNTRY_CONFIGS))}"
            )
        resolved.append(normalized)
    return resolved


def run_country(config: CountryConfig) -> dict[str, object]:
    if not config.parla_path.exists():
        return {
            "country_code": config.code,
            "country_name": config.manifesto_country,
            "status": "missing_input",
            "parla_path": str(config.parla_path),
        }

    manifesto_df = load_manifesto_parties(config.manifesto_country)
    speech_counts_df = build_speech_counts(config.parla_path)
    diagnostics = compare_party_names(manifesto_df, speech_counts_df["speech_party"].tolist())

    mapping_path = create_country_mapping_csv(config)
    unmapped_path, parties_path, parliamentary_path = create_country_unmapped_csvs(
        config,
        mapping_path=mapping_path,
    )
    mapping_df = pd.read_csv(mapping_path)

    return {
        "country_code": config.code,
        "country_name": config.manifesto_country,
        "status": "ok",
        "parla_path": str(config.parla_path),
        "mapping_path": str(mapping_path),
        "unmapped_path": str(unmapped_path),
        "parties_path": str(parties_path),
        "parliamentary_path": str(parliamentary_path),
        "speech_labels": int(len(mapping_df)),
        "direct_exact_matches": int(len(diagnostics["match_any"])),
        "mapped_labels": int((mapping_df["mapping_status"] == "mapped").sum()),
        "unmapped_labels": int((mapping_df["mapping_status"] == "unmapped").sum()),
        "unmapped_parties": int((mapping_df["mapping_status"].eq("unmapped") & mapping_df["speech_label_type"].eq("party")).sum()),
        "unmapped_parliamentary_labels": int(
            (mapping_df["mapping_status"].eq("unmapped") & mapping_df["speech_label_type"].ne("party")).sum()
        ),
    }


def main() -> int:
    args = parse_args()
    country_codes = resolve_country_codes(args.country)

    for code in country_codes:
        result = run_country(COUNTRY_CONFIGS[code])
        if result["status"] == "missing_input":
            print(
                f"[{result['country_code']}] Missing speech extract: {result['parla_path']}"
            )
            continue

        print(f"[{result['country_code']}] Saved mapping CSV to {result['mapping_path']}")
        print(f"[{result['country_code']}] Saved unmapped labels CSV to {result['unmapped_path']}")
        print(f"[{result['country_code']}] Saved unmapped party CSV to {result['parties_path']}")
        print(
            f"[{result['country_code']}] Saved unmapped parliamentary-label CSV to "
            f"{result['parliamentary_path']}"
        )
        print(
            f"[{result['country_code']}] Speech labels={result['speech_labels']}, "
            f"mapped={result['mapped_labels']}, unmapped={result['unmapped_labels']}, "
            f"unmapped_parties={result['unmapped_parties']}, "
            f"unmapped_parliamentary_labels={result['unmapped_parliamentary_labels']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
