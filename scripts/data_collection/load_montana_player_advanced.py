"""
File: load_montana_player_advanced.py
Last Modified: 2026-02-12
Purpose: Load current-season advanced season-level Montana player metrics from Sports Reference,
supplementing with Per 100 Possession data and the existing Montana player stats file.

Inputs:
- data/raw/sports_reference/montana_roster.csv
- data/raw/sports_reference/montana_player_stats.csv
- Sports Reference Montana team season page

Outputs:
- data/raw/sports_reference/montana_player_advanced_stats.csv

Sources & Attribution:
- Data Source: Sports Reference - College Basketball
- Source URL Pattern: https://www.sports-reference.com/cbb/schools/montana/men/{season}.html
- Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
- Attribution: “Data courtesy of Sports Reference”
"""

from __future__ import annotations

import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


BASE_URL = "https://www.sports-reference.com"
TEAM_NAME_STANDARD = "Montana"
CONFERENCE_NAME_STANDARD = "Big Sky"

REQUEST_TIMEOUT = 30
REQUEST_SLEEP_SECONDS = 2.0

ROSTER_INPUT_PATH = Path("data/raw/sports_reference/montana_roster.csv")
PLAYER_STATS_INPUT_PATH = Path("data/raw/sports_reference/montana_player_stats.csv")
OUTPUT_PATH = Path("data/raw/sports_reference/montana_player_advanced_stats.csv")

REQUIRED_ROSTER_COLUMNS = [
    "player_name",
    "player_url",
]

REQUIRED_OUTPUT_COLUMNS = [
    "season",
    "team_name",
    "player_name",
    "player_url",
]

OUTPUT_COLUMN_ORDER = [
    "season",
    "team_name",
    "player_name",
    "player_url",
    "games_played",
    "pct_minutes",
    "minutes_total",
    "minutes_per_game",
    "ortg",
    "drtg",
    "pct_possessions",
    "pct_shots",
    "efg_pct",
    "ts_pct",
    "or_pct",
    "dr_pct",
    "trb_pct",
    "assist_rate",
    "turnover_rate",
    "block_pct",
    "steal_pct",
    "fouls_committed_per_40",
    "fouls_drawn_per_40",
    "ft_rate",
    "ftm",
    "fta",
    "ft_pct",
    "two_pm",
    "two_pa",
    "two_pt_pct",
    "three_pm",
    "three_pa",
    "three_pt_pct",
    "usg_pct",
    "per",
    "pprod",
    "orb_pct",
    "drb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_40",
    "obpm",
    "dbpm",
    "bpm",
    "class",
    "position",
    "conference_name",
    "games_started",
    "three_point_attempt_rate",
    "source_team_url",
]

NUMERIC_COLUMNS = [
    "season",
    "games_played",
    "pct_minutes",
    "minutes_total",
    "minutes_per_game",
    "ortg",
    "drtg",
    "pct_possessions",
    "pct_shots",
    "efg_pct",
    "ts_pct",
    "or_pct",
    "dr_pct",
    "trb_pct",
    "assist_rate",
    "turnover_rate",
    "block_pct",
    "steal_pct",
    "fouls_committed_per_40",
    "fouls_drawn_per_40",
    "ft_rate",
    "ftm",
    "fta",
    "ft_pct",
    "two_pm",
    "two_pa",
    "two_pt_pct",
    "three_pm",
    "three_pa",
    "three_pt_pct",
    "usg_pct",
    "per",
    "pprod",
    "orb_pct",
    "drb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_40",
    "obpm",
    "dbpm",
    "bpm",
    "games_started",
    "three_point_attempt_rate",
]

ADVANCED_TABLE_HINT_COLUMNS = {
    "player",
    "g",
    "gs",
    "mp",
    "per",
    "ts_pct",
    "fg3a_rate",
    "ft_rate",
    "pprod",
    "orb_pct",
    "drb_pct",
    "trb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "usg_pct",
    "ows",
    "dws",
    "ws",
    "ws_40",
    "obpm",
    "dbpm",
    "bpm",
}

PER100_TABLE_HINT_COLUMNS = {
    "player",
    "g",
    "gs",
    "mp",
    "fg",
    "fga",
    "fg_pct",
    "fg3",
    "fg3a",
    "fg3_pct",
    "fg2",
    "fg2a",
    "fg2_pct",
    "efg_pct",
    "ft",
    "fta",
    "ft_pct",
    "orb",
    "drb",
    "trb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "pts",
    "ortg",
    "drtg",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_path(relative_path: Path) -> Path:
    return get_project_root() / relative_path


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_name(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_player_url(url: str) -> str:
    url = normalize_text(url)

    if not url:
        return ""

    if url.startswith("http://") or url.startswith("https://"):
        return url

    if not url.startswith("/"):
        url = f"/{url}"

    return f"{BASE_URL}{url}"


def safe_float(value: object) -> Optional[float]:
    text = normalize_text(value)

    if not text:
        return None

    text = text.replace(",", "")
    if text.startswith("."):
        text = f"0{text}"
    if text.startswith("-."):
        text = text.replace("-.", "-0.", 1)
    if text.endswith("%"):
        text = text[:-1]

    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: object) -> Optional[int]:
    numeric_value = safe_float(value)
    if numeric_value is None:
        return None
    return int(round(numeric_value))


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.sports-reference.com/",
        }
    )
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.text


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flattened_columns = []
        for column_tuple in df.columns:
            parts = [normalize_text(part) for part in column_tuple if normalize_text(part)]
            flattened_columns.append("_".join(parts) if parts else "")
        df.columns = flattened_columns
    else:
        df.columns = [normalize_text(column) for column in df.columns]

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)

    rename_map = {
        "rk": "rk",
        "player": "player",
        "team": "team",
        "conf": "conf",
        "class": "class",
        "pos": "pos",
        "g": "g",
        "gs": "gs",
        "mp": "mp",
        "fg": "fg",
        "fga": "fga",
        "fg%": "fg_pct",
        "3p": "fg3",
        "3pa": "fg3a",
        "3p%": "fg3_pct",
        "2p": "fg2",
        "2pa": "fg2a",
        "2p%": "fg2_pct",
        "efg%": "efg_pct",
        "ft": "ft",
        "fta": "fta",
        "ft%": "ft_pct",
        "orb": "orb",
        "drb": "drb",
        "trb": "trb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "tov": "tov",
        "pf": "pf",
        "pts": "pts",
        "ortg": "ortg",
        "drtg": "drtg",
        "per": "per",
        "ts%": "ts_pct",
        "3par": "fg3a_rate",
        "ftr": "ft_rate",
        "pprod": "pprod",
        "orb%": "orb_pct",
        "drb%": "drb_pct",
        "trb%": "trb_pct",
        "ast%": "ast_pct",
        "stl%": "stl_pct",
        "blk%": "blk_pct",
        "tov%": "tov_pct",
        "usg%": "usg_pct",
        "ows": "ows",
        "dws": "dws",
        "ws": "ws",
        "ws/40": "ws_40",
        "obpm": "obpm",
        "dbpm": "dbpm",
        "bpm": "bpm",
        "mp%": "mp_pct",
        "poss%": "poss_pct",
        "shots%": "shots_pct",
        "ast rate": "assist_rate",
        "tov rate": "turnover_rate",
        "pf/40": "pf_per_40",
        "fd/40": "fd_per_40",
    }

    cleaned_columns = []
    for column in df.columns:
        column_key = normalize_text(column).lower()

        if column_key in rename_map:
            cleaned_columns.append(rename_map[column_key])
            continue

        clean_column = column_key.replace("%", "_pct")
        clean_column = clean_column.replace("/", "_per_")
        clean_column = clean_column.replace("&", "and")
        clean_column = clean_column.replace("+", "plus")
        clean_column = re.sub(r"[^\w]+", "_", clean_column)
        clean_column = re.sub(r"_+", "_", clean_column).strip("_")
        cleaned_columns.append(clean_column)

    df.columns = cleaned_columns
    return df


def extract_all_html_tables(html: str) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []

    try:
        tables.extend(pd.read_html(StringIO(html)))
    except ValueError:
        pass

    soup = BeautifulSoup(html, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    for comment in comments:
        comment_text = str(comment)
        if "<table" not in comment_text:
            continue
        try:
            tables.extend(pd.read_html(StringIO(comment_text)))
        except ValueError:
            continue

    return tables


def validate_input_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_roster(path: Path) -> pd.DataFrame:
    validate_input_file(path, "Roster input file")

    roster_df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_ROSTER_COLUMNS if column not in roster_df.columns]
    if missing_columns:
        raise ValueError(f"Roster file is missing required columns: {missing_columns}")

    roster_df = roster_df.copy()
    roster_df["player_name"] = roster_df["player_name"].astype(str).str.strip()
    roster_df["player_url"] = roster_df["player_url"].map(normalize_player_url)
    roster_df["player_name_key"] = roster_df["player_name"].map(normalize_name)

    roster_df = roster_df[
        (roster_df["player_name"] != "")
        & (roster_df["player_url"] != "")
        & (roster_df["player_name_key"] != "")
    ].copy()

    roster_df = roster_df.drop_duplicates(subset=["player_name_key"], keep="first").reset_index(drop=True)

    if roster_df.empty:
        raise ValueError("Roster file contains no valid player rows after cleaning.")

    return roster_df


def load_player_stats_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    stats_df = pd.read_csv(path)
    if "player_name" not in stats_df.columns:
        return pd.DataFrame()

    stats_df = stats_df.copy()
    stats_df["player_name"] = stats_df["player_name"].astype(str).str.strip()
    stats_df["player_name_key"] = stats_df["player_name"].map(normalize_name)

    if "team_name" in stats_df.columns:
        stats_df["team_name"] = stats_df["team_name"].astype(str).str.strip()
        stats_df = stats_df[stats_df["team_name"].str.lower() == TEAM_NAME_STANDARD.lower()].copy()

    stats_df = stats_df.drop_duplicates(subset=["player_name_key"], keep="first").reset_index(drop=True)
    return stats_df


def get_current_season(roster_df: pd.DataFrame, player_stats_df: pd.DataFrame) -> int:
    season_candidates = []

    if "season" in roster_df.columns:
        season_candidates.extend(
            pd.to_numeric(roster_df["season"], errors="coerce").dropna().astype(int).tolist()
        )

    if not player_stats_df.empty and "season" in player_stats_df.columns:
        season_candidates.extend(
            pd.to_numeric(player_stats_df["season"], errors="coerce").dropna().astype(int).tolist()
        )

    if season_candidates:
        return max(season_candidates)

    raise ValueError("Could not determine current season from roster or player stats input.")


def team_page_url(season: int) -> str:
    return f"{BASE_URL}/cbb/schools/montana/men/{season}.html"


def detect_player_column(df: pd.DataFrame) -> Optional[str]:
    for column in ["player", "player_name"]:
        if column in df.columns:
            return column
    return None


def is_non_player_row(player_value: object) -> bool:
    text = normalize_text(player_value).lower()
    return text in {"player", "team totals", "opponent totals", "totals", "school", "career"}


def prepare_player_table(df: pd.DataFrame) -> pd.DataFrame:
    working_df = standardize_column_names(df).copy()

    player_column = detect_player_column(working_df)
    if player_column is None:
        return pd.DataFrame()

    working_df[player_column] = working_df[player_column].astype(str).str.strip()
    working_df = working_df[working_df[player_column] != ""].copy()
    working_df = working_df[~working_df[player_column].map(is_non_player_row)].copy()

    if "rk" in working_df.columns:
        working_df = working_df[working_df["rk"].astype(str).str.lower() != "rk"].copy()

    working_df["player_name"] = working_df[player_column].astype(str).str.strip()
    working_df["player_name_key"] = working_df["player_name"].map(normalize_name)
    working_df = working_df[working_df["player_name_key"] != ""].copy()

    return working_df.reset_index(drop=True)


def score_table(df: pd.DataFrame, hint_columns: set[str]) -> int:
    if df.empty:
        return -1
    return sum(1 for column in hint_columns if column in df.columns)


def find_best_matching_table(tables: list[pd.DataFrame], hint_columns: set[str]) -> pd.DataFrame:
    best_df = pd.DataFrame()
    best_score = -1

    for table in tables:
        candidate_df = prepare_player_table(table)
        score = score_table(candidate_df, hint_columns)

        if score > best_score:
            best_score = score
            best_df = candidate_df.copy()

    return best_df.reset_index(drop=True)


def first_non_null(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if pd.isna(value):
            continue
        return value
    return None


def series_value(series: Optional[pd.Series], column_names: list[str]) -> object:
    if series is None:
        return None

    for column_name in column_names:
        if column_name in series.index:
            value = series.get(column_name)
            if pd.notna(value):
                return value

    return None


def numeric_series_value(series: Optional[pd.Series], column_names: list[str]) -> Optional[float]:
    return safe_float(series_value(series, column_names))


def text_series_value(series: Optional[pd.Series], column_names: list[str]) -> Optional[str]:
    value = series_value(series, column_names)
    text = normalize_text(value)
    return text if text else None


def select_row_by_name(df: pd.DataFrame, player_name_key: str) -> Optional[pd.Series]:
    if df.empty or "player_name_key" not in df.columns:
        return None

    matches = df[df["player_name_key"] == player_name_key].copy()
    if matches.empty:
        return None

    matches["non_null_count"] = matches.notna().sum(axis=1)
    matches = matches.sort_values(by="non_null_count", ascending=False)
    return matches.iloc[0]


def calculate_minutes_per_game(minutes_total: Optional[float], games_played: Optional[int]) -> Optional[float]:
    if minutes_total is None or games_played in (None, 0):
        return None
    return float(minutes_total) / float(games_played)


def calculate_per_40(stat_total: Optional[float], minutes_total: Optional[float]) -> Optional[float]:
    if stat_total is None or minutes_total in (None, 0):
        return None
    return float(stat_total) * 40.0 / float(minutes_total)


def build_player_record(
    roster_row: pd.Series,
    player_stats_row: Optional[pd.Series],
    advanced_row: Optional[pd.Series],
    per100_row: Optional[pd.Series],
    season: int,
    source_team_url: str,
) -> dict:
    player_name = normalize_text(roster_row["player_name"])
    player_url = normalize_text(roster_row["player_url"])

    games_played = first_non_null(
        safe_int(series_value(player_stats_row, ["games_played", "g"])),
        safe_int(series_value(advanced_row, ["g"])),
        safe_int(series_value(per100_row, ["g"])),
    )

    minutes_total = first_non_null(
        numeric_series_value(player_stats_row, ["minutes_total", "mp", "minutes"]),
        numeric_series_value(advanced_row, ["mp"]),
        numeric_series_value(per100_row, ["mp"]),
    )

    minutes_per_game = first_non_null(
        numeric_series_value(player_stats_row, ["minutes_per_game"]),
        calculate_minutes_per_game(minutes_total, games_played),
    )

    fouls_total = first_non_null(
        numeric_series_value(player_stats_row, ["fouls", "pf"]),
        numeric_series_value(per100_row, ["pf"]),
    )

    fta_total = first_non_null(
        numeric_series_value(player_stats_row, ["fta"]),
        numeric_series_value(per100_row, ["fta"]),
    )

    record = {
        "season": first_non_null(
            safe_int(series_value(player_stats_row, ["season"])),
            season,
        ),
        "team_name": TEAM_NAME_STANDARD,
        "player_name": player_name,
        "player_url": player_url,
        "games_played": games_played,
        "pct_minutes": first_non_null(
            numeric_series_value(player_stats_row, ["pct_minutes", "mp_pct"]),
            numeric_series_value(advanced_row, ["mp_pct"]),
        ),
        "minutes_total": minutes_total,
        "minutes_per_game": minutes_per_game,
        "ortg": first_non_null(
            numeric_series_value(player_stats_row, ["ortg"]),
            numeric_series_value(per100_row, ["ortg"]),
        ),
        "drtg": numeric_series_value(per100_row, ["drtg"]),
        "pct_possessions": first_non_null(
            numeric_series_value(player_stats_row, ["pct_possessions", "poss_pct"]),
            numeric_series_value(advanced_row, ["poss_pct"]),
        ),
        "pct_shots": first_non_null(
            numeric_series_value(player_stats_row, ["pct_shots", "shots_pct"]),
            numeric_series_value(advanced_row, ["shots_pct"]),
            numeric_series_value(advanced_row, ["fg3a_rate"]),
        ),
        "efg_pct": first_non_null(
            numeric_series_value(player_stats_row, ["efg_pct"]),
            numeric_series_value(per100_row, ["efg_pct"]),
        ),
        "ts_pct": first_non_null(
            numeric_series_value(player_stats_row, ["ts_pct"]),
            numeric_series_value(advanced_row, ["ts_pct"]),
        ),
        "or_pct": None,
        "dr_pct": None,
        "trb_pct": first_non_null(
            numeric_series_value(player_stats_row, ["trb_pct"]),
            numeric_series_value(advanced_row, ["trb_pct"]),
        ),
        "assist_rate": first_non_null(
            numeric_series_value(player_stats_row, ["assist_rate", "ast_pct"]),
            numeric_series_value(advanced_row, ["ast_pct"]),
        ),
        "turnover_rate": first_non_null(
            numeric_series_value(player_stats_row, ["turnover_rate", "tov_pct"]),
            numeric_series_value(advanced_row, ["tov_pct"]),
        ),
        "block_pct": first_non_null(
            numeric_series_value(player_stats_row, ["block_pct", "blk_pct"]),
            numeric_series_value(advanced_row, ["blk_pct"]),
        ),
        "steal_pct": first_non_null(
            numeric_series_value(player_stats_row, ["steal_pct", "stl_pct"]),
            numeric_series_value(advanced_row, ["stl_pct"]),
        ),
        "fouls_committed_per_40": first_non_null(
            numeric_series_value(player_stats_row, ["fouls_committed_per_40", "pf_per_40"]),
            calculate_per_40(fouls_total, minutes_total),
        ),
        "fouls_drawn_per_40": first_non_null(
            numeric_series_value(player_stats_row, ["fouls_drawn_per_40", "fd_per_40"]),
            calculate_per_40(fta_total, minutes_total),
        ),
        "ft_rate": first_non_null(
            numeric_series_value(player_stats_row, ["ft_rate"]),
            numeric_series_value(advanced_row, ["ft_rate"]),
        ),
        "ftm": first_non_null(
            numeric_series_value(player_stats_row, ["ftm", "ft"]),
            numeric_series_value(per100_row, ["ft"]),
        ),
        "fta": fta_total,
        "ft_pct": first_non_null(
            numeric_series_value(player_stats_row, ["ft_pct"]),
            numeric_series_value(per100_row, ["ft_pct"]),
        ),
        "two_pm": first_non_null(
            numeric_series_value(player_stats_row, ["two_pm", "fg2"]),
            numeric_series_value(per100_row, ["fg2"]),
        ),
        "two_pa": first_non_null(
            numeric_series_value(player_stats_row, ["two_pa", "fg2a"]),
            numeric_series_value(per100_row, ["fg2a"]),
        ),
        "two_pt_pct": first_non_null(
            numeric_series_value(player_stats_row, ["two_pt_pct", "fg2_pct"]),
            numeric_series_value(per100_row, ["fg2_pct"]),
        ),
        "three_pm": first_non_null(
            numeric_series_value(player_stats_row, ["three_pm", "fg3"]),
            numeric_series_value(per100_row, ["fg3"]),
        ),
        "three_pa": first_non_null(
            numeric_series_value(player_stats_row, ["three_pa", "fg3a"]),
            numeric_series_value(per100_row, ["fg3a"]),
        ),
        "three_pt_pct": first_non_null(
            numeric_series_value(player_stats_row, ["three_pt_pct", "fg3_pct"]),
            numeric_series_value(per100_row, ["fg3_pct"]),
        ),
        "usg_pct": first_non_null(
            numeric_series_value(player_stats_row, ["usg_pct"]),
            numeric_series_value(advanced_row, ["usg_pct"]),
        ),
        "per": first_non_null(
            numeric_series_value(player_stats_row, ["per"]),
            numeric_series_value(advanced_row, ["per"]),
        ),
        "pprod": first_non_null(
            numeric_series_value(player_stats_row, ["pprod"]),
            numeric_series_value(advanced_row, ["pprod"]),
        ),
        "orb_pct": first_non_null(
            numeric_series_value(player_stats_row, ["orb_pct"]),
            numeric_series_value(advanced_row, ["orb_pct"]),
        ),
        "drb_pct": first_non_null(
            numeric_series_value(player_stats_row, ["drb_pct"]),
            numeric_series_value(advanced_row, ["drb_pct"]),
        ),
        "ast_pct": first_non_null(
            numeric_series_value(player_stats_row, ["ast_pct"]),
            numeric_series_value(advanced_row, ["ast_pct"]),
        ),
        "stl_pct": first_non_null(
            numeric_series_value(player_stats_row, ["stl_pct"]),
            numeric_series_value(advanced_row, ["stl_pct"]),
        ),
        "blk_pct": first_non_null(
            numeric_series_value(player_stats_row, ["blk_pct"]),
            numeric_series_value(advanced_row, ["blk_pct"]),
        ),
        "tov_pct": first_non_null(
            numeric_series_value(player_stats_row, ["tov_pct"]),
            numeric_series_value(advanced_row, ["tov_pct"]),
        ),
        "ows": first_non_null(
            numeric_series_value(player_stats_row, ["ows"]),
            numeric_series_value(advanced_row, ["ows"]),
        ),
        "dws": first_non_null(
            numeric_series_value(player_stats_row, ["dws"]),
            numeric_series_value(advanced_row, ["dws"]),
        ),
        "ws": first_non_null(
            numeric_series_value(player_stats_row, ["ws"]),
            numeric_series_value(advanced_row, ["ws"]),
        ),
        "ws_per_40": first_non_null(
            numeric_series_value(player_stats_row, ["ws_per_40", "ws_40"]),
            numeric_series_value(advanced_row, ["ws_40"]),
        ),
        "obpm": first_non_null(
            numeric_series_value(player_stats_row, ["obpm"]),
            numeric_series_value(advanced_row, ["obpm"]),
        ),
        "dbpm": first_non_null(
            numeric_series_value(player_stats_row, ["dbpm"]),
            numeric_series_value(advanced_row, ["dbpm"]),
        ),
        "bpm": first_non_null(
            numeric_series_value(player_stats_row, ["bpm"]),
            numeric_series_value(advanced_row, ["bpm"]),
        ),
        "class": first_non_null(
            text_series_value(player_stats_row, ["class"]),
            text_series_value(advanced_row, ["class"]),
            normalize_text(roster_row.get("class")) if "class" in roster_row.index else None,
        ),
        "position": first_non_null(
            text_series_value(player_stats_row, ["position", "position_raw", "pos"]),
            text_series_value(advanced_row, ["pos"]),
            normalize_text(roster_row.get("position_raw")) if "position_raw" in roster_row.index else None,
            normalize_text(roster_row.get("position")) if "position" in roster_row.index else None,
        ),
        "conference_name": first_non_null(
            text_series_value(player_stats_row, ["conference_name", "conf"]),
            text_series_value(advanced_row, ["conf"]),
            CONFERENCE_NAME_STANDARD,
        ),
        "games_started": first_non_null(
            safe_int(series_value(player_stats_row, ["games_started", "gs"])),
            safe_int(series_value(advanced_row, ["gs"])),
            safe_int(series_value(per100_row, ["gs"])),
        ),
        "three_point_attempt_rate": first_non_null(
            numeric_series_value(player_stats_row, ["three_point_attempt_rate", "fg3a_rate"]),
            numeric_series_value(advanced_row, ["fg3a_rate"]),
        ),
        "source_team_url": source_team_url,
    }

    return record


def deduplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df["player_name_key"] = output_df["player_name"].map(normalize_name)
    output_df["non_null_count"] = output_df.notna().sum(axis=1)

    output_df = (
        output_df.sort_values(
            by=["player_name_key", "season", "non_null_count"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["player_name_key", "season"], keep="first")
        .drop(columns=["player_name_key", "non_null_count"])
        .reset_index(drop=True)
    )

    return output_df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")

    return output_df


def ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in OUTPUT_COLUMN_ORDER:
        if column not in output_df.columns:
            output_df[column] = None

    output_df = output_df[OUTPUT_COLUMN_ORDER]

    missing_required = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in output_df.columns]
    if missing_required:
        raise ValueError(f"Output is missing required columns: {missing_required}")

    for column in REQUIRED_OUTPUT_COLUMNS:
        if output_df[column].isna().any():
            raise ValueError(f"Output contains null values in required column: {column}")

    return output_df


def export_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    roster_path = build_path(ROSTER_INPUT_PATH)
    player_stats_path = build_path(PLAYER_STATS_INPUT_PATH)
    output_path = build_path(OUTPUT_PATH)

    roster_df = load_roster(roster_path)
    player_stats_df = load_player_stats_if_available(player_stats_path)
    season = get_current_season(roster_df, player_stats_df)

    team_url = team_page_url(season)
    session = create_session()
    html = fetch_html(session, team_url)
    tables = extract_all_html_tables(html)

    advanced_df = find_best_matching_table(tables, ADVANCED_TABLE_HINT_COLUMNS)
    per100_df = find_best_matching_table(tables, PER100_TABLE_HINT_COLUMNS)

    records = []

    for _, roster_row in roster_df.iterrows():
        player_name_key = roster_row["player_name_key"]

        player_stats_row = select_row_by_name(player_stats_df, player_name_key)
        advanced_row = select_row_by_name(advanced_df, player_name_key)
        per100_row = select_row_by_name(per100_df, player_name_key)

        record = build_player_record(
            roster_row=roster_row,
            player_stats_row=player_stats_row,
            advanced_row=advanced_row,
            per100_row=per100_row,
            season=season,
            source_team_url=team_url,
        )
        records.append(record)

    if not records:
        raise ValueError("No Montana player advanced records were created.")

    output_df = pd.DataFrame(records)
    output_df["team_name"] = TEAM_NAME_STANDARD
    output_df = deduplicate_players(output_df)
    output_df = coerce_numeric_columns(output_df)
    output_df = ensure_output_schema(output_df)

    export_output(output_df, output_path)
    print(f"Montana player advanced stats saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)