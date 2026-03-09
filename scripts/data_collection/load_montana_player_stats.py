"""
File: load_montana_player_stats.py
Purpose: Load current-season season-level Montana player stats from Sports Reference.

Inputs:
- data/raw/sports_reference/montana_roster.csv
- Sports Reference player pages from roster links
- Sports Reference Montana team season page

Output:
- data/raw/sports_reference/montana_player_season_stats.csv
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
TEAM_PAGE_URL_TEMPLATE = "https://www.sports-reference.com/cbb/schools/montana/men/{season}.html"

REQUEST_TIMEOUT = 30
REQUEST_SLEEP_SECONDS = 1.0

ROSTER_INPUT_PATH = Path("data/raw/sports_reference/montana_roster.csv")
OUTPUT_PATH = Path("data/raw/sports_reference/montana_player_season_stats.csv")

REQUIRED_ROSTER_COLUMNS = [
    "player_name",
    "player_url",
]

REQUIRED_OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "player_url",
]

OUTPUT_COLUMN_ORDER = [
    "player_name",
    "team_name",
    "season",
    "player_url",
    "source_team_url",
    "class",
    "position",
    "games",
    "games_started",
    "minutes",
    "minutes_per_game",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_pct",
    "three_points_made",
    "three_points_attempted",
    "three_point_pct",
    "two_points_made",
    "two_points_attempted",
    "two_point_pct",
    "effective_field_goal_pct",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_pct",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "points",
    "offensive_rebounds_per_game",
    "defensive_rebounds_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "fouls_per_game",
    "points_per_game",
]

NUMERIC_COLUMNS = [
    "season",
    "games",
    "games_started",
    "minutes",
    "minutes_per_game",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_pct",
    "three_points_made",
    "three_points_attempted",
    "three_point_pct",
    "two_points_made",
    "two_points_attempted",
    "two_point_pct",
    "effective_field_goal_pct",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_pct",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "points",
    "offensive_rebounds_per_game",
    "defensive_rebounds_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "steals_per_game",
    "blocks_per_game",
    "turnovers_per_game",
    "fouls_per_game",
    "points_per_game",
]

PER_GAME_REQUIRED_COLUMNS = {
    "season",
    "team",
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
}

TOTALS_REQUIRED_COLUMNS = {
    "season",
    "team",
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
}

TEAM_PLAYER_REQUIRED_COLUMNS = {
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
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_path(relative_path: Path) -> Path:
    return get_project_root() / relative_path


def validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Roster input file not found: {path}")


def normalize_text(value: object) -> str:
    if value is None:
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

    if text == "" or text.lower() == "nan":
        return None

    text = text.replace(",", "")
    if text.startswith("."):
        text = f"0{text}"

    try:
        return float(text)
    except ValueError:
        return None


def safe_int(value: object) -> Optional[int]:
    numeric_value = safe_float(value)
    if numeric_value is None:
        return None
    return int(round(numeric_value))


def calculate_per_game(total_value: Optional[float], games: Optional[int]) -> Optional[float]:
    if total_value is None or games in (None, 0):
        return None
    return float(total_value) / float(games)


def calculate_total(per_game_value: Optional[float], games: Optional[int]) -> Optional[float]:
    if per_game_value is None or games in (None, 0):
        return None
    return float(per_game_value) * float(games)


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    time.sleep(REQUEST_SLEEP_SECONDS)
    return response.text


def load_roster(path: Path) -> pd.DataFrame:
    validate_input_file(path)

    roster_df = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_ROSTER_COLUMNS if column not in roster_df.columns]
    if missing_columns:
        raise ValueError(f"Roster file is missing required columns: {missing_columns}")

    roster_df = roster_df.copy()
    roster_df["player_name"] = roster_df["player_name"].astype(str).str.strip()
    roster_df["player_url"] = roster_df["player_url"].astype(str).str.strip()
    roster_df["player_name_key"] = roster_df["player_name"].map(normalize_name)

    roster_df = roster_df[
        (roster_df["player_name"] != "")
        & (roster_df["player_url"] != "")
        & (~roster_df["player_url"].str.lower().eq("nan"))
    ].copy()

    roster_df = roster_df.drop_duplicates(subset=["player_name_key"], keep="first").reset_index(drop=True)

    if roster_df.empty:
        raise ValueError("Roster file contains no valid player rows after cleaning.")

    return roster_df


def get_current_season_from_roster(roster_df: pd.DataFrame) -> int:
    if "season" in roster_df.columns:
        season_values = (
            pd.to_numeric(roster_df["season"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if season_values:
            return max(season_values)

    raise ValueError("Could not determine current season from roster file.")


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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flattened_columns = []
        for column_tuple in df.columns:
            parts = [normalize_text(part) for part in column_tuple if normalize_text(part) != ""]
            flattened_columns.append("_".join(parts) if parts else "")
        df.columns = flattened_columns
    else:
        df.columns = [normalize_text(column) for column in df.columns]

    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df)

    rename_map = {
        "season": "season",
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
        "awards": "awards",
        "player": "player",
        "rk": "rk",
    }

    cleaned_columns = []
    for column in df.columns:
        base_column = normalize_text(column).lower()

        if base_column in rename_map:
            cleaned_columns.append(rename_map[base_column])
            continue

        clean_column = base_column.replace("%", "pct")
        clean_column = clean_column.replace("/", "_per_")
        clean_column = clean_column.replace("&", "and")
        clean_column = clean_column.replace("+", "plus")
        clean_column = re.sub(r"[^\w]+", "_", clean_column)
        clean_column = re.sub(r"_+", "_", clean_column).strip("_")
        cleaned_columns.append(clean_column)

    df.columns = cleaned_columns
    return df


def parse_player_page_season_row_value(value: object) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None

    match = re.search(r"(\d{4})-(\d{2})", text)
    if match:
        start_year = int(match.group(1))
        end_suffix = int(match.group(2))
        century = str(start_year)[:2]
        return int(f"{century}{end_suffix:02d}")

    direct_match = re.search(r"(\d{4})", text)
    if direct_match:
        return int(direct_match.group(1))

    return None


def prepare_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    working_df = standardize_column_names(df).copy()

    if "season" not in working_df.columns or "team" not in working_df.columns:
        return pd.DataFrame()

    working_df["parsed_season"] = working_df["season"].apply(parse_player_page_season_row_value)
    working_df["team_normalized"] = working_df["team"].astype(str).str.strip().str.lower()

    if "g" in working_df.columns:
        working_df["g_numeric"] = pd.to_numeric(working_df["g"], errors="coerce")
    else:
        working_df["g_numeric"] = pd.NA

    if "mp" in working_df.columns:
        working_df["mp_numeric"] = pd.to_numeric(working_df["mp"], errors="coerce")
    else:
        working_df["mp_numeric"] = pd.NA

    return working_df


def select_best_player_season_row(
    tables: list[pd.DataFrame],
    season: int,
    required_columns: set[str],
) -> Optional[pd.Series]:
    candidate_rows: list[pd.Series] = []

    for table in tables:
        working_df = prepare_candidate_table(table)
        if working_df.empty:
            continue

        if not required_columns.issubset(set(working_df.columns)):
            continue

        season_df = working_df[
            (working_df["parsed_season"] == season)
            & (working_df["team_normalized"] == TEAM_NAME_STANDARD.lower())
        ].copy()

        if season_df.empty:
            continue

        season_df["non_null_count"] = season_df.notna().sum(axis=1)
        season_df = season_df.sort_values(
            by=["g_numeric", "non_null_count"],
            ascending=[False, False],
            na_position="last",
        )

        candidate_rows.append(season_df.iloc[0])

    if not candidate_rows:
        return None

    candidate_df = pd.DataFrame(candidate_rows).copy()
    candidate_df["non_null_count"] = candidate_df.notna().sum(axis=1)
    candidate_df = candidate_df.sort_values(
        by=["g_numeric", "non_null_count"],
        ascending=[False, False],
        na_position="last",
    )
    return candidate_df.iloc[0]


def get_player_page_season_rows(
    session: requests.Session,
    player_url: str,
    season: int,
) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    normalized_url = normalize_player_url(player_url)
    if not normalized_url:
        return None, None

    try:
        html = fetch_html(session, normalized_url)
    except requests.RequestException:
        return None, None

    tables = extract_all_html_tables(html)

    per_game_row = select_best_player_season_row(
        tables=tables,
        season=season,
        required_columns=PER_GAME_REQUIRED_COLUMNS,
    )

    totals_row = select_best_player_season_row(
        tables=tables,
        season=season,
        required_columns=TOTALS_REQUIRED_COLUMNS,
    )

    return per_game_row, totals_row


def detect_player_column(df: pd.DataFrame) -> Optional[str]:
    for column in ["player", "player_name"]:
        if column in df.columns:
            return column
    return None


def is_team_totals_row(player_value: object) -> bool:
    text = normalize_text(player_value).lower()
    return text in {"team totals", "opponent totals", "totals"}


def filter_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    player_column = detect_player_column(df)
    if player_column is None:
        return pd.DataFrame()

    filtered_df = df.copy()
    filtered_df[player_column] = filtered_df[player_column].astype(str).str.strip()
    filtered_df = filtered_df[filtered_df[player_column] != ""].copy()
    filtered_df = filtered_df[~filtered_df[player_column].map(is_team_totals_row)].copy()

    if "rk" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["rk"].astype(str).str.lower() != "rk"].copy()

    filtered_df["player_name_key"] = filtered_df[player_column].map(normalize_name)
    filtered_df = filtered_df[filtered_df["player_name_key"] != ""].copy()

    return filtered_df.reset_index(drop=True)


def find_best_team_player_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    best_df = pd.DataFrame()
    best_score = -1

    for table in tables:
        standardized_df = standardize_column_names(table)
        filtered_df = filter_player_rows(standardized_df)

        if filtered_df.empty:
            continue

        score = sum(1 for column in TEAM_PLAYER_REQUIRED_COLUMNS if column in filtered_df.columns)

        if "g" in filtered_df.columns:
            g_value = safe_float(filtered_df.iloc[0].get("g"))
            if g_value is not None and g_value <= 40:
                score += 5

        if "mp" in filtered_df.columns:
            mp_value = safe_float(filtered_df.iloc[0].get("mp"))
            if mp_value is not None and mp_value <= 45:
                score += 5

        if score > best_score:
            best_score = score
            best_df = filtered_df.copy()

    return best_df.reset_index(drop=True)


def select_team_row(df: pd.DataFrame, player_name_key: str) -> Optional[pd.Series]:
    if df.empty:
        return None

    matches = df[df["player_name_key"] == player_name_key].copy()
    if matches.empty:
        return None

    matches["non_null_count"] = matches.notna().sum(axis=1)
    matches = matches.sort_values(by="non_null_count", ascending=False)

    return matches.iloc[0]


def build_stats_from_per_game_row(row: Optional[pd.Series]) -> dict:
    if row is None:
        return {}

    games = safe_int(row.get("g"))
    games_started = safe_int(row.get("gs"))
    minutes_per_game = safe_float(row.get("mp"))
    field_goals_made_per_game = safe_float(row.get("fg"))
    field_goals_attempted_per_game = safe_float(row.get("fga"))
    field_goal_pct = safe_float(row.get("fg_pct"))
    three_points_made_per_game = safe_float(row.get("fg3"))
    three_points_attempted_per_game = safe_float(row.get("fg3a"))
    three_point_pct = safe_float(row.get("fg3_pct"))
    two_points_made_per_game = safe_float(row.get("fg2"))
    two_points_attempted_per_game = safe_float(row.get("fg2a"))
    two_point_pct = safe_float(row.get("fg2_pct"))
    effective_field_goal_pct = safe_float(row.get("efg_pct"))
    free_throws_made_per_game = safe_float(row.get("ft"))
    free_throws_attempted_per_game = safe_float(row.get("fta"))
    free_throw_pct = safe_float(row.get("ft_pct"))
    offensive_rebounds_per_game = safe_float(row.get("orb"))
    defensive_rebounds_per_game = safe_float(row.get("drb"))
    rebounds_per_game = safe_float(row.get("trb"))
    assists_per_game = safe_float(row.get("ast"))
    steals_per_game = safe_float(row.get("stl"))
    blocks_per_game = safe_float(row.get("blk"))
    turnovers_per_game = safe_float(row.get("tov"))
    fouls_per_game = safe_float(row.get("pf"))
    points_per_game = safe_float(row.get("pts"))

    return {
        "games": games,
        "games_started": games_started,
        "minutes": calculate_total(minutes_per_game, games),
        "minutes_per_game": minutes_per_game,
        "field_goals_made": calculate_total(field_goals_made_per_game, games),
        "field_goals_attempted": calculate_total(field_goals_attempted_per_game, games),
        "field_goal_pct": field_goal_pct,
        "three_points_made": calculate_total(three_points_made_per_game, games),
        "three_points_attempted": calculate_total(three_points_attempted_per_game, games),
        "three_point_pct": three_point_pct,
        "two_points_made": calculate_total(two_points_made_per_game, games),
        "two_points_attempted": calculate_total(two_points_attempted_per_game, games),
        "two_point_pct": two_point_pct,
        "effective_field_goal_pct": effective_field_goal_pct,
        "free_throws_made": calculate_total(free_throws_made_per_game, games),
        "free_throws_attempted": calculate_total(free_throws_attempted_per_game, games),
        "free_throw_pct": free_throw_pct,
        "offensive_rebounds": calculate_total(offensive_rebounds_per_game, games),
        "defensive_rebounds": calculate_total(defensive_rebounds_per_game, games),
        "rebounds": calculate_total(rebounds_per_game, games),
        "assists": calculate_total(assists_per_game, games),
        "steals": calculate_total(steals_per_game, games),
        "blocks": calculate_total(blocks_per_game, games),
        "turnovers": calculate_total(turnovers_per_game, games),
        "fouls": calculate_total(fouls_per_game, games),
        "points": calculate_total(points_per_game, games),
        "offensive_rebounds_per_game": offensive_rebounds_per_game,
        "defensive_rebounds_per_game": defensive_rebounds_per_game,
        "rebounds_per_game": rebounds_per_game,
        "assists_per_game": assists_per_game,
        "steals_per_game": steals_per_game,
        "blocks_per_game": blocks_per_game,
        "turnovers_per_game": turnovers_per_game,
        "fouls_per_game": fouls_per_game,
        "points_per_game": points_per_game,
    }


def build_stats_from_totals_row(row: Optional[pd.Series]) -> dict:
    if row is None:
        return {}

    games = safe_int(row.get("g"))
    games_started = safe_int(row.get("gs"))
    minutes = safe_float(row.get("mp"))
    field_goals_made = safe_float(row.get("fg"))
    field_goals_attempted = safe_float(row.get("fga"))
    field_goal_pct = safe_float(row.get("fg_pct"))
    three_points_made = safe_float(row.get("fg3"))
    three_points_attempted = safe_float(row.get("fg3a"))
    three_point_pct = safe_float(row.get("fg3_pct"))
    two_points_made = safe_float(row.get("fg2"))
    two_points_attempted = safe_float(row.get("fg2a"))
    two_point_pct = safe_float(row.get("fg2_pct"))
    effective_field_goal_pct = safe_float(row.get("efg_pct"))
    free_throws_made = safe_float(row.get("ft"))
    free_throws_attempted = safe_float(row.get("fta"))
    free_throw_pct = safe_float(row.get("ft_pct"))
    offensive_rebounds = safe_float(row.get("orb"))
    defensive_rebounds = safe_float(row.get("drb"))
    rebounds = safe_float(row.get("trb"))
    assists = safe_float(row.get("ast"))
    steals = safe_float(row.get("stl"))
    blocks = safe_float(row.get("blk"))
    turnovers = safe_float(row.get("tov"))
    fouls = safe_float(row.get("pf"))
    points = safe_float(row.get("pts"))

    return {
        "games": games,
        "games_started": games_started,
        "minutes": minutes,
        "minutes_per_game": calculate_per_game(minutes, games),
        "field_goals_made": field_goals_made,
        "field_goals_attempted": field_goals_attempted,
        "field_goal_pct": field_goal_pct,
        "three_points_made": three_points_made,
        "three_points_attempted": three_points_attempted,
        "three_point_pct": three_point_pct,
        "two_points_made": two_points_made,
        "two_points_attempted": two_points_attempted,
        "two_point_pct": two_point_pct,
        "effective_field_goal_pct": effective_field_goal_pct,
        "free_throws_made": free_throws_made,
        "free_throws_attempted": free_throws_attempted,
        "free_throw_pct": free_throw_pct,
        "offensive_rebounds": offensive_rebounds,
        "defensive_rebounds": defensive_rebounds,
        "rebounds": rebounds,
        "assists": assists,
        "steals": steals,
        "blocks": blocks,
        "turnovers": turnovers,
        "fouls": fouls,
        "points": points,
        "offensive_rebounds_per_game": calculate_per_game(offensive_rebounds, games),
        "defensive_rebounds_per_game": calculate_per_game(defensive_rebounds, games),
        "rebounds_per_game": calculate_per_game(rebounds, games),
        "assists_per_game": calculate_per_game(assists, games),
        "steals_per_game": calculate_per_game(steals, games),
        "blocks_per_game": calculate_per_game(blocks, games),
        "turnovers_per_game": calculate_per_game(turnovers, games),
        "fouls_per_game": calculate_per_game(fouls, games),
        "points_per_game": calculate_per_game(points, games),
    }


def overlay_prefer_primary(primary_stats: dict, secondary_stats: dict) -> dict:
    combined = secondary_stats.copy()
    combined.update({k: v for k, v in primary_stats.items() if v is not None})
    return combined


def fill_missing_percentages(stats: dict) -> dict:
    stats = stats.copy()

    if stats.get("field_goal_pct") is None:
        fgm = stats.get("field_goals_made")
        fga = stats.get("field_goals_attempted")
        if fgm is not None and fga not in (None, 0):
            stats["field_goal_pct"] = fgm / fga

    if stats.get("three_point_pct") is None:
        fg3 = stats.get("three_points_made")
        fg3a = stats.get("three_points_attempted")
        if fg3 is not None and fg3a not in (None, 0):
            stats["three_point_pct"] = fg3 / fg3a

    if stats.get("two_point_pct") is None:
        fg2 = stats.get("two_points_made")
        fg2a = stats.get("two_points_attempted")
        if fg2 is not None and fg2a not in (None, 0):
            stats["two_point_pct"] = fg2 / fg2a

    if stats.get("free_throw_pct") is None:
        ft = stats.get("free_throws_made")
        fta = stats.get("free_throws_attempted")
        if ft is not None and fta not in (None, 0):
            stats["free_throw_pct"] = ft / fta

    if stats.get("effective_field_goal_pct") is None:
        fgm = stats.get("field_goals_made")
        fg3 = stats.get("three_points_made")
        fga = stats.get("field_goals_attempted")
        if fgm is not None and fg3 is not None and fga not in (None, 0):
            stats["effective_field_goal_pct"] = (fgm + 0.5 * fg3) / fga

    return stats


def get_class_value(roster_row: pd.Series, per_game_row: Optional[pd.Series], totals_row: Optional[pd.Series], team_row: Optional[pd.Series]) -> Optional[str]:
    for source in [per_game_row, totals_row, team_row, roster_row]:
        if source is None:
            continue
        for column in ["class"]:
            if column in source.index:
                value = normalize_text(source.get(column))
                if value and value.lower() != "nan":
                    return value
    return None


def get_position_value(roster_row: pd.Series, per_game_row: Optional[pd.Series], totals_row: Optional[pd.Series], team_row: Optional[pd.Series]) -> Optional[str]:
    for source in [per_game_row, totals_row, team_row]:
        if source is None:
            continue
        for column in ["pos", "position"]:
            if column in source.index:
                value = normalize_text(source.get(column))
                if value and value.lower() != "nan":
                    return value

    for column in ["position_raw", "position"]:
        if column in roster_row.index:
            value = normalize_text(roster_row.get(column))
            if value and value.lower() != "nan":
                return value

    return None


def build_player_record(
    session: requests.Session,
    roster_row: pd.Series,
    season: int,
    source_team_url: str,
    team_player_df: pd.DataFrame,
) -> dict:
    player_name = normalize_text(roster_row["player_name"])
    player_url = normalize_text(roster_row["player_url"])
    player_name_key = normalize_name(player_name)

    per_game_row, totals_row = get_player_page_season_rows(
        session=session,
        player_url=player_url,
        season=season,
    )

    team_row = select_team_row(team_player_df, player_name_key)

    per_game_stats = build_stats_from_per_game_row(per_game_row)
    totals_stats = build_stats_from_totals_row(totals_row)
    team_stats = build_stats_from_per_game_row(team_row)

    combined_stats = overlay_prefer_primary(
        primary_stats=team_stats,
        secondary_stats=per_game_stats,
    )
    combined_stats = overlay_prefer_primary(
        primary_stats=totals_stats,
        secondary_stats=combined_stats,
    )
    combined_stats = fill_missing_percentages(combined_stats)

    record = {
        "player_name": player_name,
        "team_name": TEAM_NAME_STANDARD,
        "season": season,
        "player_url": player_url,
        "source_team_url": source_team_url,
        "class": get_class_value(roster_row, per_game_row, totals_row, team_row),
        "position": get_position_value(roster_row, per_game_row, totals_row, team_row),
    }

    record.update(combined_stats)
    return record


def ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in OUTPUT_COLUMN_ORDER:
        if column not in output_df.columns:
            output_df[column] = None

    output_df = output_df[OUTPUT_COLUMN_ORDER]

    missing_required = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in output_df.columns]
    if missing_required:
        raise ValueError(f"Output is missing required columns: {missing_required}")

    if output_df["player_name"].isna().any():
        raise ValueError("Output contains null player_name values.")

    if output_df["player_url"].isna().any():
        raise ValueError("Output contains null player_url values.")

    return output_df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")

    return output_df


def deduplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    deduped_df = df.copy()
    deduped_df["player_name_key"] = deduped_df["player_name"].map(normalize_name)
    deduped_df["non_null_count"] = deduped_df.notna().sum(axis=1)

    deduped_df = (
        deduped_df.sort_values(
            by=["player_name_key", "season", "non_null_count"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["player_name_key", "season"], keep="first")
        .drop(columns=["player_name_key", "non_null_count"])
        .reset_index(drop=True)
    )

    return deduped_df


def export_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    roster_path = build_path(ROSTER_INPUT_PATH)
    output_path = build_path(OUTPUT_PATH)

    roster_df = load_roster(roster_path)
    season = get_current_season_from_roster(roster_df)
    source_team_url = TEAM_PAGE_URL_TEMPLATE.format(season=season)

    session = create_session()
    team_html = fetch_html(session, source_team_url)
    team_tables = extract_all_html_tables(team_html)
    team_player_df = find_best_team_player_table(team_tables)

    if team_player_df.empty:
        raise ValueError("Could not locate a usable Montana team player stats table.")

    records = []
    for _, roster_row in roster_df.iterrows():
        record = build_player_record(
            session=session,
            roster_row=roster_row,
            season=season,
            source_team_url=source_team_url,
            team_player_df=team_player_df,
        )
        records.append(record)

    if not records:
        raise ValueError("No Montana player records were created.")

    output_df = pd.DataFrame(records)
    output_df["team_name"] = TEAM_NAME_STANDARD
    output_df = coerce_numeric_columns(output_df)
    output_df = deduplicate_players(output_df)
    output_df = ensure_output_schema(output_df)

    export_output(output_df, output_path)
    print(f"Montana player season stats saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)