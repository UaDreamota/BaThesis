from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_DIR = SCRIPTS_DIR.parent
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_speeches"
DEFAULT_PARLGOV_DB = BASE_DIR / "data" / "parlgov" / "parlgov-stable.db"
DEFAULT_OUTPUT_DIR = TEST_OUTPUT_DIR / "plda_regression_panel"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import load_country


PARLGOV_COUNTRY_SHORT_BY_CODE = {
    "CZ": "CZE",
    "EE": "EST",
    "GB": "GBR",
    "HU": "HUN",
    "LV": "LVA",
    "PL": "POL",
    "SI": "SVN",
}

DEFAULT_PARLGOV_PARTY_MAP = {
    "CZ": {
        "ANO2011": ["ANO"],
        "CSSD": ["CSSD"],
        "KDU-ČSL": ["KDU-CSL"],
        "KSCM": ["KSCM"],
        "ODS": ["ODS"],
        "Piráti": ["Pi"],
        "SPD": ["SPD"],
        "STAN": ["STAN"],
        "TOP09": ["TOP09"],
        "Usvit": ["UPD"],
        "SPOLU": ["ODS", "KDU-CSL", "TOP09"],
        "PirSTAN": ["Pi", "STAN"],
    },
    "EE": {
        "EKRE": ["ERa/EKR"],
        "ISA": ["IRL"],
        "KE": ["EK"],
        "RE": ["ERe"],
        "SDE": ["SDE|M"],
        "VABA": ["EV"],
    },
    "HU": {
        "DK": ["DK"],
        "DK-frakció": ["DK"],
        "EGYÜTT": ["Egyutt"],
        "Fidesz": ["Fi-MPSz", "KDNP"],
        "Fidesz-frakció": ["Fi-MPSz", "KDNP"],
        "JOBBIK": ["Jobbik"],
        "JOBBIK-frakció": ["Jobbik"],
        "KDNP": ["KDNP"],
        "LMP": ["LMP"],
        "LMP-frakció": ["LMP"],
        "Mi Hazánk": ["MHM"],
        "MSZP": ["MSZP"],
        "MSZP-frakció": ["MSZP"],
        "Párbeszéd": ["PM"],
        "Párbeszéd-frakció": ["PM"],
    },
    "GB": {
        "A": ["APoNI"],
        "CON": ["Con"],
        "DUP": ["DUP"],
        "GP": ["GP"],
        "LAB": ["Lab"],
        "LD": ["Lib"],
        "PC": ["Plaid"],
        "SDLP": ["SDLP"],
        "SNP": ["SNP"],
        "UKIP": ["UKIP"],
        "UUP": ["UUP"],
    },
    "LV": {
        "AP": ["AP!"],
        "JK": ["JKP"],
        "JV": ["V"],
        "KPV-LV": ["KPV-LV"],
        "LRA": ["LRa"],
        "NSL": ["NsL"],
        "SASKAŅA": ["S"],
        "VIENOTĪBA": ["V"],
        "ZZS": ["ZZS"],
    },
    "PL": {
        "KO": ["PO", "N"],
        "KP-PSL": ["PSL"],
        "Konfederacja": ["KORWIN", "RN"],
        "Kukiz15": ["K"],
        "Lewica": ["SLD", "WIO", "Razem"],
        "PiS": ["PiS"],
        "PrzywrócićPrawo": ["K"],
        "Teraz": ["N"],
        "UPR": ["UPR|KNP"],
    },
    "SI": {
        "DL": ["DL"],
        "DLGV": ["DL"],
        "DeSUS": ["DeSUS"],
        "Konkretno": ["SMC"],
        "LDS": ["LDS"],
        "LMŠ": ["LMS"],
        "Levica": ["L"],
        "Lipa": ["LIPA"],
        "NSi": ["NSI"],
        "PS": ["LZJ-PS"],
        "SAB": ["ZaAB"],
        "SD": ["ZL-SD"],
        "SDS": ["SDS"],
        "SLS": ["SLS"],
        "SLS+SKD": ["SLS"],
        "SMC": ["SMC"],
        "SMS": ["SMS"],
        "SNS": ["SNS"],
        "ZL": ["ZdLe"],
        "ZLSD": ["ZL-SD"],
        "ZaAB": ["ZaAB"],
        "Zares": ["Zares"],
    },
}

EE_CABINET_CONTEXT_OVERRIDES = [
    {
        "cabinet_id": 1664,
        "start_date": "2022-06-03",
        "end_date": "2022-07-15",
        "governing_parties": ["RE"],
        "prime_minister_party": "RE",
        "cabinet_gov_seats": 34,
        "note": (
            "Kallas I minority period after Centre Party ministers were dismissed "
            "on 2022-06-03; applied by analysis_date."
        ),
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich PLDA manifesto-alignment rows with Parlgov election/cabinet "
            "covariates and speech-volume controls for regression analysis."
        )
    )
    parser.add_argument("--country", default="CZ", type=str)
    parser.add_argument(
        "--alignment-input",
        type=Path,
        default=None,
        help=(
            "Input PLDA alignment CSV. Defaults to "
            "outputs/test_speeches/plda_alignment/<COUNTRY>/<COUNTRY>_plda_manifesto_alignment.csv."
        ),
    )
    parser.add_argument(
        "--parlgov-db",
        type=Path,
        default=DEFAULT_PARLGOV_DB,
        help="Path to Parlgov SQLite database.",
    )
    parser.add_argument(
        "--parlgov-country-short",
        default=None,
        type=str,
        help="Parlgov country short code. For Czech Republic this is CZE.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "outputs/test_speeches/plda_regression_panel/<COUNTRY>/<COUNTRY>_plda_regression_panel.csv."
        ),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help=(
            "Slim regression CSV without topic-vector columns. Defaults to the "
            "main output path with _model appended."
        ),
    )
    return parser


def default_alignment_input(country_code: str) -> Path:
    return (
        TEST_OUTPUT_DIR
        / "plda_alignment"
        / country_code
        / f"{country_code}_plda_manifesto_alignment.csv"
    )


def default_output(country_code: str) -> Path:
    return (
        DEFAULT_OUTPUT_DIR
        / country_code
        / f"{country_code}_plda_regression_panel.csv"
    )


def load_sql(db_path: Path, query: str, params: tuple[object, ...]) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Could not find Parlgov database: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def load_alignment(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find PLDA alignment CSV: {path}")
    df = pd.read_csv(path, low_memory=False).copy()
    required = {
        "speech_party",
        "month",
        "month_start",
        "speech_rows",
        "speech_start_date",
        "speech_end_date",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Alignment CSV is missing required columns: {missing}")

    df["month_start"] = pd.to_datetime(df["month_start"], errors="coerce")
    df["speech_start_date"] = pd.to_datetime(df["speech_start_date"], errors="coerce")
    df["speech_end_date"] = pd.to_datetime(df["speech_end_date"], errors="coerce")
    df["analysis_date"] = df["speech_start_date"] + (
        df["speech_end_date"] - df["speech_start_date"]
    ) / 2
    df["analysis_date"] = df["analysis_date"].fillna(df["month_start"])
    return df.dropna(subset=["speech_party", "month_start", "analysis_date"]).copy()


def load_parliament_elections(
    db_path: Path,
    parlgov_country_short: str,
) -> pd.DataFrame:
    elections = load_sql(
        db_path,
        """
        SELECT
          e.id AS election_id,
          e.date AS election_date,
          e.early AS election_early,
          e.seats_total AS election_seats_total
        FROM election e
        JOIN country c ON e.country_id = c.id
        JOIN info_id ii ON e.type_id = ii.id
        WHERE c.name_short = ?
          AND ii.short = 'parliament'
        ORDER BY e.date
        """,
        (parlgov_country_short,),
    )
    elections["election_date"] = pd.to_datetime(elections["election_date"], errors="coerce")
    elections = elections.dropna(subset=["election_date"]).reset_index(drop=True)
    if elections.empty:
        raise ValueError(f"No parliament elections found for {parlgov_country_short}.")
    return elections


def add_electoral_cycle(
    df: pd.DataFrame,
    elections: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    election_dates = elections["election_date"].to_numpy(dtype="datetime64[ns]")
    row_dates = out["analysis_date"].to_numpy(dtype="datetime64[ns]")
    last_positions = np.searchsorted(election_dates, row_dates, side="right") - 1

    valid_last = last_positions >= 0
    out["last_election_id"] = pd.NA
    out["last_election_date"] = pd.NaT
    out["last_election_early"] = pd.NA
    out["election_seats_total"] = pd.NA
    out.loc[valid_last, "last_election_id"] = (
        elections.iloc[last_positions[valid_last]]["election_id"].to_numpy()
    )
    out.loc[valid_last, "last_election_date"] = (
        elections.iloc[last_positions[valid_last]]["election_date"].to_numpy()
    )
    out.loc[valid_last, "last_election_early"] = (
        elections.iloc[last_positions[valid_last]]["election_early"].to_numpy()
    )
    out.loc[valid_last, "election_seats_total"] = (
        elections.iloc[last_positions[valid_last]]["election_seats_total"].to_numpy()
    )

    next_positions = last_positions + 1
    valid_next = valid_last & (next_positions < len(elections))
    out["next_election_id"] = pd.NA
    out["next_election_date"] = pd.NaT
    out.loc[valid_next, "next_election_id"] = (
        elections.iloc[next_positions[valid_next]]["election_id"].to_numpy()
    )
    out.loc[valid_next, "next_election_date"] = (
        elections.iloc[next_positions[valid_next]]["election_date"].to_numpy()
    )

    observed_cycle_days = elections["election_date"].diff().dt.days.dropna()
    imputed_cycle_days = int(round(float(observed_cycle_days.median())))
    out["cycle_next_election_date"] = out["next_election_date"]
    out["cycle_boundary_source"] = "observed_next_election"

    missing_next = out["cycle_next_election_date"].isna() & out["last_election_date"].notna()
    out.loc[missing_next, "cycle_next_election_date"] = (
        out.loc[missing_next, "last_election_date"]
        + pd.to_timedelta(imputed_cycle_days, unit="D")
    )
    out.loc[missing_next, "cycle_boundary_source"] = "imputed_country_median_cycle"

    out["days_since_last_election"] = (
        out["analysis_date"] - out["last_election_date"]
    ).dt.days
    out["days_until_cycle_end"] = (
        out["cycle_next_election_date"] - out["analysis_date"]
    ).dt.days
    out["cycle_length_days"] = (
        out["cycle_next_election_date"] - out["last_election_date"]
    ).dt.days
    out["electoral_cycle_progress"] = (
        out["days_since_last_election"] / out["cycle_length_days"]
    )
    out["electoral_cycle_remaining"] = 1.0 - out["electoral_cycle_progress"]
    out["electoral_cycle_progress"] = out["electoral_cycle_progress"].clip(lower=0, upper=1)
    out["electoral_cycle_remaining"] = out["electoral_cycle_remaining"].clip(lower=0, upper=1)
    return out


def load_party_lookup(db_path: Path, parlgov_country_short: str) -> pd.DataFrame:
    party_lookup = load_sql(
        db_path,
        """
        SELECT DISTINCT
          p.id AS party_id,
          p.name_short AS parlgov_party_short,
          p.name AS parlgov_party_name,
          p.name_english AS parlgov_party_name_english
        FROM party p
        JOIN view_election ve ON p.id = ve.party_id
        WHERE ve.country_name_short = ?
        """,
        (parlgov_country_short,),
    )
    return party_lookup


def party_mapping_frame(country_code: str, party_lookup: pd.DataFrame) -> pd.DataFrame:
    mapping = DEFAULT_PARLGOV_PARTY_MAP.get(country_code)
    if not mapping:
        raise ValueError(
            f"No built-in Parlgov party mapping for {country_code}. "
            "Add one to DEFAULT_PARLGOV_PARTY_MAP."
        )
    rows = [
        {
            "speech_party": speech_party,
            "parlgov_party_short": parlgov_party_short,
            "speech_party_is_coalition": len(parlgov_party_shorts) > 1,
        }
        for speech_party, parlgov_party_shorts in mapping.items()
        for parlgov_party_short in parlgov_party_shorts
    ]
    mapping_df = pd.DataFrame(rows)
    mapping_df = mapping_df.merge(party_lookup, on="parlgov_party_short", how="left")
    missing = mapping_df.loc[mapping_df["party_id"].isna(), "parlgov_party_short"].unique()
    if len(missing):
        raise ValueError(f"Mapped Parlgov party codes not found: {sorted(missing)}")
    mapping_df["party_id"] = mapping_df["party_id"].astype(int)
    return mapping_df


def load_parliament_composition_seats(
    db_path: Path,
    parlgov_country_short: str,
) -> pd.DataFrame:
    composition = load_sql(
        db_path,
        """
        WITH first_composition AS (
          SELECT
            vpc.election_id,
            MIN(vpc.date) AS first_composition_date
          FROM viewcalc_parliament_composition vpc
          JOIN election e ON vpc.election_id = e.id
          JOIN country c ON e.country_id = c.id
          WHERE c.name_short = ?
            AND vpc.cabinet_formation = 1
          GROUP BY vpc.election_id
        )
        SELECT
          vpc.election_id,
          vpc.date AS parliament_composition_date,
          vpc.party_id,
          p.name_short AS parlgov_party_short,
          vpc.seats
        FROM viewcalc_parliament_composition vpc
        JOIN first_composition fc
          ON vpc.election_id = fc.election_id
         AND vpc.date = fc.first_composition_date
        JOIN party p ON vpc.party_id = p.id
        ORDER BY vpc.election_id, p.name_short
        """,
        (parlgov_country_short,),
    )
    composition["parliament_composition_date"] = pd.to_datetime(
        composition["parliament_composition_date"], errors="coerce"
    )
    return composition


def load_election_result_seats(
    db_path: Path,
    parlgov_country_short: str,
) -> pd.DataFrame:
    results = load_sql(
        db_path,
        """
        SELECT
          election_id,
          party_id,
          party_name_short AS parlgov_party_short,
          seats,
          vote_share
        FROM view_election
        WHERE country_name_short = ?
          AND election_type = 'parliament'
        """,
        (parlgov_country_short,),
    )
    return results


def build_party_seat_panel(
    composition: pd.DataFrame,
    election_results: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    mapped_composition = composition.merge(
        mapping_df[["speech_party", "party_id", "speech_party_is_coalition"]],
        on="party_id",
        how="inner",
    )
    seat_panel = (
        mapped_composition.groupby(["election_id", "speech_party"], as_index=False)
        .agg(
            party_seats=("seats", "sum"),
            parliament_composition_date=("parliament_composition_date", "min"),
            speech_party_is_coalition=("speech_party_is_coalition", "max"),
        )
    )
    seat_panel["party_seat_source"] = "parliament_composition"

    mapped_results = election_results.merge(
        mapping_df[["speech_party", "party_id", "speech_party_is_coalition"]],
        on="party_id",
        how="inner",
    )
    fallback = (
        mapped_results.groupby(["election_id", "speech_party"], as_index=False)
        .agg(
            party_seats=("seats", "sum"),
            party_vote_share=("vote_share", "sum"),
            speech_party_is_coalition=("speech_party_is_coalition", "max"),
        )
    )
    fallback["party_seat_source"] = "election_result"

    seat_panel = seat_panel.merge(
        fallback[
            [
                "election_id",
                "speech_party",
                "party_vote_share",
                "party_seats",
                "party_seat_source",
            ]
        ].rename(
            columns={
                "party_seats": "fallback_party_seats",
                "party_seat_source": "fallback_party_seat_source",
            }
        ),
        on=["election_id", "speech_party"],
        how="outer",
    )

    missing_composition = seat_panel["party_seats"].isna()
    seat_panel.loc[missing_composition, "party_seats"] = seat_panel.loc[
        missing_composition, "fallback_party_seats"
    ]
    seat_panel.loc[missing_composition, "party_seat_source"] = seat_panel.loc[
        missing_composition, "fallback_party_seat_source"
    ]
    seat_panel = seat_panel.drop(
        columns=["fallback_party_seats", "fallback_party_seat_source"]
    )
    return seat_panel


def load_cabinets(db_path: Path, parlgov_country_short: str) -> pd.DataFrame:
    cabinets = load_sql(
        db_path,
        """
        SELECT DISTINCT
          cabinet_id,
          start_date AS cabinet_start_date,
          cabinet_name,
          caretaker AS cabinet_caretaker,
          election_id AS cabinet_previous_election_id,
          election_date AS cabinet_previous_election_date,
          election_seats_total AS cabinet_election_seats_total
        FROM view_cabinet
        WHERE country_name_short = ?
        ORDER BY start_date
        """,
        (parlgov_country_short,),
    )
    cabinets["cabinet_start_date"] = pd.to_datetime(
        cabinets["cabinet_start_date"], errors="coerce"
    )
    cabinets["cabinet_previous_election_date"] = pd.to_datetime(
        cabinets["cabinet_previous_election_date"], errors="coerce"
    )
    cabinets = cabinets.dropna(subset=["cabinet_start_date"]).sort_values(
        "cabinet_start_date"
    )
    cabinets["cabinet_next_start_date"] = cabinets["cabinet_start_date"].shift(-1)
    return cabinets


def load_cabinet_parties(db_path: Path, parlgov_country_short: str) -> pd.DataFrame:
    cabinet_parties = load_sql(
        db_path,
        """
        SELECT
          cabinet_id,
          party_id,
          party_name_short AS parlgov_party_short,
          cabinet_party,
          prime_minister,
          seats
        FROM view_cabinet
        WHERE country_name_short = ?
        """,
        (parlgov_country_short,),
    )
    return cabinet_parties


def cabinet_summary(cabinet_parties: pd.DataFrame) -> pd.DataFrame:
    party_rows = cabinet_parties[cabinet_parties["parlgov_party_short"] != "none"].copy()
    gov_rows = party_rows[party_rows["cabinet_party"] == 1].copy()
    summary = (
        gov_rows.groupby("cabinet_id", as_index=False)
        .agg(
            cabinet_party_count=("party_id", "nunique"),
            cabinet_gov_seats=("seats", "sum"),
        )
    )
    all_cabinets = pd.DataFrame({"cabinet_id": cabinet_parties["cabinet_id"].unique()})
    summary = all_cabinets.merge(summary, on="cabinet_id", how="left")
    summary[["cabinet_party_count", "cabinet_gov_seats"]] = summary[
        ["cabinet_party_count", "cabinet_gov_seats"]
    ].fillna(0)
    summary["cabinet_is_coalition"] = summary["cabinet_party_count"] > 1
    return summary


def apply_cabinet_context_overrides(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    out = df.copy()
    out["cabinet_context_source"] = "parlgov"
    out["cabinet_context_note"] = ""

    overrides = EE_CABINET_CONTEXT_OVERRIDES if country_code == "EE" else []
    for override in overrides:
        start_date = pd.Timestamp(override["start_date"])
        end_date = pd.Timestamp(override["end_date"])
        governing_parties = set(override["governing_parties"])
        prime_minister_party = override["prime_minister_party"]
        cabinet_gov_seats = float(override["cabinet_gov_seats"])

        mask = (
            out["cabinet_id"].eq(override["cabinet_id"])
            & out["analysis_date"].ge(start_date)
            & out["analysis_date"].lt(end_date)
        )
        if not mask.any():
            continue

        in_government = out.loc[mask, "speech_party"].isin(governing_parties)
        is_pm_party = out.loc[mask, "speech_party"].eq(prime_minister_party)
        out.loc[mask, "cabinet_party_count"] = len(governing_parties)
        out.loc[mask, "cabinet_gov_seats"] = cabinet_gov_seats
        out.loc[mask, "cabinet_gov_seat_share"] = (
            cabinet_gov_seats / out.loc[mask, "cabinet_election_seats_total"]
        )
        out.loc[mask, "cabinet_is_coalition"] = len(governing_parties) > 1
        out.loc[mask, "cabinet_has_absolute_majority"] = (
            cabinet_gov_seats > out.loc[mask, "cabinet_election_seats_total"] / 2
        )
        out.loc[mask, "party_in_government"] = in_government.astype(int).to_numpy()
        out.loc[mask, "party_prime_minister"] = is_pm_party.astype(int).to_numpy()
        out.loc[mask, "party_cabinet_seats"] = np.where(in_government, cabinet_gov_seats, 0)
        out.loc[mask, "party_constituents_in_government"] = (
            in_government.astype(int).to_numpy()
        )
        out.loc[mask, "cabinet_context_source"] = "parlgov_with_sourced_override"
        out.loc[mask, "cabinet_context_note"] = override["note"]

    return out


def add_party_government_status(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["party_government_status"] = np.select(
        [
            out["party_prime_minister"].eq(1),
            out["party_in_government"].eq(1),
            out["cabinet_caretaker"].eq(1) & out["cabinet_party_count"].fillna(0).eq(0),
        ],
        ["prime_minister_party", "coalition_partner", "nonpartisan_caretaker"],
        default="opposition",
    )
    return out


def add_cabinet_context(
    df: pd.DataFrame,
    cabinets: pd.DataFrame,
    cabinet_parties: pd.DataFrame,
    mapping_df: pd.DataFrame,
    country_code: str,
) -> pd.DataFrame:
    out = df.sort_values("analysis_date").copy()
    cabinets = cabinets.sort_values("cabinet_start_date").copy()
    out = pd.merge_asof(
        out,
        cabinets,
        left_on="analysis_date",
        right_on="cabinet_start_date",
        direction="backward",
    )
    after_cabinet_end = (
        out["cabinet_next_start_date"].notna()
        & (out["analysis_date"] >= out["cabinet_next_start_date"])
    )
    if after_cabinet_end.any():
        out.loc[after_cabinet_end, "cabinet_id"] = pd.NA

    out = out.merge(cabinet_summary(cabinet_parties), on="cabinet_id", how="left")
    out["cabinet_gov_seat_share"] = (
        out["cabinet_gov_seats"] / out["cabinet_election_seats_total"]
    )
    out["cabinet_has_absolute_majority"] = (
        out["cabinet_gov_seats"] > out["cabinet_election_seats_total"] / 2
    )

    mapped_cabinet = cabinet_parties.merge(
        mapping_df[["speech_party", "party_id"]],
        on="party_id",
        how="inner",
    )
    party_cabinet = (
        mapped_cabinet.groupby(["cabinet_id", "speech_party"], as_index=False)
        .agg(
            party_in_government=("cabinet_party", "max"),
            party_prime_minister=("prime_minister", "max"),
            party_cabinet_seats=("seats", "sum"),
            party_constituents_in_government=("cabinet_party", "sum"),
        )
    )
    out = out.merge(party_cabinet, on=["cabinet_id", "speech_party"], how="left")
    for column in [
        "party_in_government",
        "party_prime_minister",
        "party_constituents_in_government",
    ]:
        out[column] = out[column].fillna(0).astype(int)
    out["party_cabinet_seats"] = out["party_cabinet_seats"].fillna(0)

    out = apply_cabinet_context_overrides(out, country_code)
    return add_party_government_status(out)


def add_party_seats(df: pd.DataFrame, party_seat_panel: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["last_election_id"] = pd.to_numeric(out["last_election_id"], errors="coerce")
    party_seat_panel["election_id"] = pd.to_numeric(
        party_seat_panel["election_id"], errors="coerce"
    )
    out = out.merge(
        party_seat_panel.rename(columns={"election_id": "last_election_id"}),
        on=["last_election_id", "speech_party"],
        how="left",
    )
    out["party_seat_share"] = out["party_seats"] / out["election_seats_total"]
    out["party_has_absolute_majority"] = out["party_seats"] > out["election_seats_total"] / 2
    return out


def add_speech_volume(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    out = df.copy()
    speeches = load_country(country_code)
    speeches = speeches[["party", "month", "text"]].copy()
    speeches["speech_volume_words"] = (
        speeches["text"].fillna("").astype(str).str.split().str.len()
    )
    party_month = (
        speeches.groupby(["party", "month"], as_index=False)
        .agg(
            speech_volume_segments=("text", "size"),
            speech_volume_words=("speech_volume_words", "sum"),
        )
        .rename(columns={"party": "speech_party"})
    )
    month_total = (
        party_month.groupby("month", as_index=False)
        .agg(
            month_total_speech_segments=("speech_volume_segments", "sum"),
            month_total_speech_words=("speech_volume_words", "sum"),
        )
    )
    out = out.merge(party_month, on=["speech_party", "month"], how="left")
    out = out.merge(month_total, on="month", how="left")
    out["speech_volume_segments"] = out["speech_volume_segments"].fillna(out["speech_rows"])
    out["speech_volume_words"] = out["speech_volume_words"].fillna(0)
    out["log1p_speech_segments"] = np.log1p(out["speech_volume_segments"])
    out["log1p_speech_words"] = np.log1p(out["speech_volume_words"])
    out["speech_segment_share"] = (
        out["speech_volume_segments"] / out["month_total_speech_segments"]
    )
    out["speech_word_share"] = out["speech_volume_words"] / out["month_total_speech_words"]
    return out


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    leading_columns = [
        "speech_party",
        "month",
        "month_start",
        "analysis_date",
        "alignment_score",
        "js_distance",
        "cosine_similarity",
        "hellinger_distance",
        "speech_topic_weighting",
        "speech_volume_segments",
        "speech_volume_words",
        "log1p_speech_segments",
        "log1p_speech_words",
        "speech_segment_share",
        "speech_word_share",
        "last_election_id",
        "last_election_date",
        "next_election_id",
        "next_election_date",
        "cycle_next_election_date",
        "cycle_boundary_source",
        "days_since_last_election",
        "days_until_cycle_end",
        "cycle_length_days",
        "electoral_cycle_progress",
        "electoral_cycle_remaining",
        "party_seats",
        "party_seat_share",
        "party_has_absolute_majority",
        "party_vote_share",
        "party_seat_source",
        "speech_party_is_coalition",
        "cabinet_id",
        "cabinet_name",
        "cabinet_start_date",
        "cabinet_next_start_date",
        "cabinet_caretaker",
        "cabinet_party_count",
        "cabinet_is_coalition",
        "cabinet_gov_seats",
        "cabinet_gov_seat_share",
        "cabinet_has_absolute_majority",
        "cabinet_context_source",
        "cabinet_context_note",
        "party_in_government",
        "party_prime_minister",
        "party_government_status",
    ]
    existing_leading = [column for column in leading_columns if column in df.columns]
    remaining = [column for column in df.columns if column not in existing_leading]
    return df[existing_leading + remaining]


def model_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if not column.startswith("topic_")]


def build_regression_panel(
    country_code: str,
    alignment_input: Path,
    parlgov_db: Path,
    parlgov_country_short: str,
) -> pd.DataFrame:
    alignment = load_alignment(alignment_input)
    elections = load_parliament_elections(parlgov_db, parlgov_country_short)
    party_lookup = load_party_lookup(parlgov_db, parlgov_country_short)
    mapping_df = party_mapping_frame(country_code, party_lookup)
    composition = load_parliament_composition_seats(parlgov_db, parlgov_country_short)
    election_results = load_election_result_seats(parlgov_db, parlgov_country_short)
    party_seat_panel = build_party_seat_panel(composition, election_results, mapping_df)
    cabinets = load_cabinets(parlgov_db, parlgov_country_short)
    cabinet_parties = load_cabinet_parties(parlgov_db, parlgov_country_short)

    panel = add_electoral_cycle(alignment, elections)
    panel = add_party_seats(panel, party_seat_panel)
    panel = add_cabinet_context(panel, cabinets, cabinet_parties, mapping_df, country_code)
    panel = add_speech_volume(panel, country_code)
    return reorder_columns(panel.sort_values(["month_start", "speech_party"]).reset_index(drop=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    country_code = args.country.strip().upper()
    alignment_input = args.alignment_input or default_alignment_input(country_code)
    output = args.output or default_output(country_code)
    model_output = args.model_output or output.with_name(f"{output.stem}_model.csv")
    parlgov_country_short = (
        args.parlgov_country_short
        or PARLGOV_COUNTRY_SHORT_BY_CODE.get(country_code)
    )
    if not parlgov_country_short:
        raise ValueError(
            f"No Parlgov country short code configured for {country_code}. "
            "Pass --parlgov-country-short."
        )

    panel = build_regression_panel(
        country_code=country_code,
        alignment_input=alignment_input,
        parlgov_db=args.parlgov_db,
        parlgov_country_short=parlgov_country_short,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    panel[model_columns(panel)].to_csv(model_output, index=False)

    print(f"Built {len(panel):,} regression-panel rows for {country_code}.")
    print(f"Parties: {panel['speech_party'].nunique():,}")
    print(
        "Rows with observed next election: "
        f"{int((panel['cycle_boundary_source'] == 'observed_next_election').sum()):,}"
    )
    print(
        "Rows with imputed cycle end: "
        f"{int((panel['cycle_boundary_source'] == 'imputed_country_median_cycle').sum()):,}"
    )
    print(f"Saved regression panel to: {output}")
    print(f"Saved slim model panel to: {model_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
