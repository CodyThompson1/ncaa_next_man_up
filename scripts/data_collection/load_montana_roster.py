"""
File: load_montana_roster.py
Last Modified: 2026-02-20
Purpose: Load the current-season Montana men's basketball roster from Sports Reference.

Inputs:
- Sports Reference Montana team roster page for the current season

Outputs:
- data/raw/sports_reference/montana_roster.csv
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


TEAM_NAME = "Montana"
CONFERENCE_NAME = "Big Sky"
BASE_URL = "https://www.sports-reference.com"
TEAM_URL_PATTERN = "https://www.sports-reference.com/cbb/schools/montana/men/{season}.html"
OUTPUT_PATH = Path("data/raw/sports_reference/montana_roster.csv")
REQUEST_TIMEOUT = 30

REQUIRED_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "player_url",
]

OPTIONAL_OUTPUT_COLUMNS = [
    "jersey_number",
    "conference_name",
    "class",
    "height",
    "weight",
    "hometown",
    "high_school",
    "position_raw",
]


def get_current_season(today: Optional[datetime] = None) -> int:
    current_dt = today or datetime.today()
    return current_dt.year + 1 if current_dt.month >= 11 else current_dt.year


def build_team_url(season: int) -> str:
    return TEAM_URL_PATTERN.format(season=season)


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def get_all_table_candidates(soup: BeautifulSoup) -> list[BeautifulSoup]:
    tables: list[BeautifulSoup] = []

    direct_tables = soup.find_all("table")
    tables.extend(direct_tables)

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if "<table" not in str(comment):
            continue
        comment_soup = BeautifulSoup(str(comment), "html.parser")
        tables.extend(comment_soup.find_all("table"))

    return tables


def find_roster_table(soup: BeautifulSoup) -> BeautifulSoup:
    candidate_tables = get_all_table_candidates(soup)

    for table in candidate_tables:
        if table.get("id", "") == "roster":
            return table

    for table in candidate_tables:
        header_cells = [
            cell.get_text(" ", strip=True).lower()
            for cell in table.find_all(["th", "td"])
        ]
        joined_headers = " ".join(header_cells)
        if (
            "player" in joined_headers
            and "class" in joined_headers
            and ("ht" in joined_headers or "height" in joined_headers)
        ):
            return table

    raise ValueError("Roster table not found on the Montana Sports Reference page.")


def normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def clean_stat_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_row(cells: Iterable[BeautifulSoup]) -> dict[str, Optional[str]]:
    row_data: dict[str, Optional[str]] = {}

    for cell in cells:
        stat_name = cell.get("data-stat")
        if not stat_name:
            continue

        normalized_stat_name = clean_stat_name(stat_name)
        row_data[normalized_stat_name] = normalize_text(cell.get_text(" ", strip=True))

        if stat_name == "player":
            anchor = cell.find("a", href=True)
            if anchor:
                href = anchor["href"].strip()
                row_data["player_url"] = f"{BASE_URL}{href}" if href.startswith("/") else href

    return row_data


def map_output_fields(raw_df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = raw_df.copy()

    rename_map = {
        "player": "player_name",
        "class": "class",
        "ht": "height",
        "height": "height",
        "wt": "weight",
        "weight": "weight",
        "hometown": "hometown",
        "high_school": "high_school",
        "high_school_previous_school": "high_school",
        "pos": "position_raw",
        "position": "position_raw",
        "number": "jersey_number",
        "num": "jersey_number",
    }

    for source_col, target_col in rename_map.items():
        if source_col in df.columns and target_col not in df.columns:
            df[target_col] = df[source_col]

    if "player_name" not in df.columns:
        raise ValueError("Required source field 'player' was not found in the roster table.")

    if "player_url" not in df.columns:
        df["player_url"] = None

    df["team_name"] = TEAM_NAME
    df["season"] = season
    df["conference_name"] = CONFERENCE_NAME

    ordered_columns = REQUIRED_COLUMNS + OPTIONAL_OUTPUT_COLUMNS

    for column in ordered_columns:
        if column not in df.columns:
            df[column] = None

    return df[ordered_columns]


def parse_roster_table(table: BeautifulSoup, season: int) -> pd.DataFrame:
    body = table.find("tbody")
    if body is None:
        raise ValueError("Roster table is present but tbody could not be found.")

    rows = body.find_all("tr")
    parsed_rows: list[dict[str, Optional[str]]] = []

    for row in rows:
        if "class" in row.get("class", []):
            continue

        header_cell = row.find("th", {"data-stat": True})
        data_cells = row.find_all("td", {"data-stat": True})

        all_cells = []
        if header_cell is not None:
            all_cells.append(header_cell)
        all_cells.extend(data_cells)

        row_data = parse_row(all_cells)

        if not row_data:
            continue

        player_name = normalize_text(row_data.get("player"))
        if not player_name:
            continue

        parsed_rows.append(row_data)

    if not parsed_rows:
        raise ValueError("No roster rows were parsed from the roster table.")

    raw_df = pd.DataFrame(parsed_rows)
    return map_output_fields(raw_df, season)


def clean_weight_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.replace("lb", "").replace("lbs", "").strip()
    return cleaned or None


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = df.copy()

    text_columns = [
        "player_name",
        "team_name",
        "player_url",
        "conference_name",
        "class",
        "height",
        "weight",
        "hometown",
        "high_school",
        "position_raw",
        "jersey_number",
    ]

    for column in text_columns:
        if column in standardized_df.columns:
            standardized_df[column] = standardized_df[column].apply(normalize_text)

    standardized_df["team_name"] = TEAM_NAME
    standardized_df["conference_name"] = CONFERENCE_NAME

    if "weight" in standardized_df.columns:
        standardized_df["weight"] = standardized_df["weight"].apply(clean_weight_value)

    standardized_df = standardized_df.drop_duplicates(
        subset=["player_name", "season"],
        keep="first",
    ).reset_index(drop=True)

    return standardized_df


def validate_output(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required output columns: {missing_columns}")

    if df.empty:
        raise ValueError("Roster output is empty. Export aborted.")

    null_required = df[REQUIRED_COLUMNS].isna().any()
    failing_columns = null_required[null_required].index.tolist()

    if failing_columns:
        raise ValueError(
            f"Required output columns contain null values: {failing_columns}"
        )

    if (df["team_name"] != TEAM_NAME).any():
        raise ValueError("team_name standardization failed.")

    if df["season"].isna().any():
        raise ValueError("season contains null values.")


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    ensure_output_directory(output_path)
    df.to_csv(output_path, index=False)


def load_montana_roster() -> pd.DataFrame:
    season = get_current_season()
    url = build_team_url(season)

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    roster_table = find_roster_table(soup)
    roster_df = parse_roster_table(roster_table, season)
    roster_df = standardize_fields(roster_df)

    validate_output(roster_df)
    export_csv(roster_df, OUTPUT_PATH)

    return roster_df


def main() -> None:
    """
    Sources & Attribution:
    - Data Source: Sports Reference College Basketball
    - Source URL Pattern: https://www.sports-reference.com/cbb/schools/montana/men/{season}.html
    - Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
    - Attribution: “Data courtesy of Sports Reference College Basketball”
    """
    try:
        roster_df = load_montana_roster()
        print(f"Montana roster rows saved: {len(roster_df)}")
        print(f"Output written to: {OUTPUT_PATH.resolve()}")
    except Exception as exc:
        print(f"Failed to load Montana roster: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()