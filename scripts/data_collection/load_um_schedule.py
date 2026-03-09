"""
File: load_um_schedule.py
Last Modified: 2026-02-27
Purpose: Load the current Montana men's basketball schedule from Sports Reference,
         keep games played up to today, clean the data, and save raw + processed files.

Inputs:
- Sports Reference Montana team season page

Outputs:
- data/raw/sports_reference/um_schedule_raw.csv
- data/processed/um_schedule_processed.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


# Sources & Attribution:
# - Data Source: Sports Reference / College Basketball at Sports-Reference.com
# - Source URL Pattern: https://www.sports-reference.com/cbb/schools/montana/men/<season>-schedule.html
# - Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
# - Attribution: “Data courtesy of Sports Reference”


BASE_URL = "https://www.sports-reference.com"
TEAM_NAME_STANDARD = "Montana"
TEAM_SLUG = "montana"
SCHEDULE_URL_TEMPLATE = "https://www.sports-reference.com/cbb/schools/{team_slug}/men/{season}-schedule.html"

REQUEST_TIMEOUT = 30
REQUEST_SLEEP_SECONDS = 1.0

RAW_OUTPUT_PATH = Path("data/raw/sports_reference/um_schedule_raw.csv")
PROCESSED_OUTPUT_PATH = Path("data/processed/um_schedule_processed.csv")

OUTPUT_COLUMN_ORDER = [
    "season",
    "team_name",
    "date",
    "game_number",
    "day_of_week",
    "location_type",
    "site_flag",
    "opponent",
    "opponent_conference",
    "team_points",
    "opponent_points",
    "point_differential",
    "result",
    "wins",
    "losses",
    "streak",
    "srs",
    "overtime_text",
    "overtime_flag",
    "game_result",
    "source_url",
]

NUMERIC_COLUMNS = [
    "season",
    "game_number",
    "team_points",
    "opponent_points",
    "wins",
    "losses",
    "srs",
    "point_differential",
]

REQUIRED_OUTPUT_COLUMNS = [
    "season",
    "team_name",
    "date",
    "opponent",
    "location_type",
    "result",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_path(relative_path: Path) -> Path:
    return get_project_root() / relative_path


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_column_name(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace("%", "pct")
    text = text.replace("/", "_per_")
    text = text.replace("&", "and")
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    if isinstance(output_df.columns, pd.MultiIndex):
        flattened_columns = []
        for column_tuple in output_df.columns:
            parts = [normalize_text(part) for part in column_tuple if normalize_text(part) != ""]
            flattened_columns.append("_".join(parts) if parts else "")
        output_df.columns = flattened_columns
    else:
        output_df.columns = [normalize_text(column) for column in output_df.columns]

    return output_df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    output_df = flatten_columns(df)

    output_df.columns = [normalize_column_name(column) for column in output_df.columns]

    rename_map = {
        "g": "game_number",
        "date": "date",
        "time": "tipoff_time",
        "type": "game_type",
        "conf": "opponent_conference",
        "unnamed_4": "site_flag",
        "opp_name": "opponent",
        "opponent": "opponent",
        "opp": "opponent_points",
        "tm": "team_points",
        "ot": "overtime_text",
        "w": "wins",
        "l": "losses",
        "srs": "srs",
        "notes": "notes",
        "unnamed_8": "result_wl",
    }

    for old_name, new_name in rename_map.items():
        if old_name in output_df.columns and new_name not in output_df.columns:
            output_df = output_df.rename(columns={old_name: new_name})

    return output_df


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


def is_duplicate_header_row(row: pd.Series) -> bool:
    row_values = [normalize_text(value).lower() for value in row.tolist()]
    return "date" in row_values and ("opp" in row_values or "opponent" in row_values)


def find_best_schedule_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    best_df = pd.DataFrame()
    best_score = -1

    for table in tables:
        working_df = standardize_column_names(table)

        if working_df.empty:
            continue

        columns = set(working_df.columns)
        score = 0

        for required_column in [
            "date",
            "game_number",
            "opponent",
            "team_points",
            "opponent_points",
            "wins",
            "losses",
        ]:
            if required_column in columns:
                score += 5

        if "site_flag" in columns:
            score += 3
        if "overtime_text" in columns:
            score += 2
        if "srs" in columns:
            score += 1

        if len(working_df) >= 10:
            score += 3

        if score > best_score:
            best_score = score
            best_df = working_df.copy()

    return best_df.reset_index(drop=True)


def infer_current_season(today_value: date) -> int:
    if today_value.month >= 11:
        return today_value.year + 1
    return today_value.year


def build_schedule_url(season: int) -> str:
    return SCHEDULE_URL_TEMPLATE.format(team_slug=TEAM_SLUG, season=season)


def fetch_schedule_table(session: requests.Session, season: int) -> tuple[pd.DataFrame, str]:
    source_url = build_schedule_url(season)
    html = fetch_html(session, source_url)
    tables = extract_all_html_tables(html)
    schedule_df = find_best_schedule_table(tables)

    if schedule_df.empty:
        raise ValueError(f"Could not locate a usable Montana schedule table for season {season}.")

    return schedule_df, source_url


def clean_schedule_table(schedule_df: pd.DataFrame, season: int, source_url: str) -> pd.DataFrame:
    output_df = schedule_df.copy()

    if output_df.empty:
        raise ValueError("Schedule table is empty after extraction.")

    output_df = output_df.loc[:, ~output_df.columns.duplicated()].copy()

    if "game_number" in output_df.columns:
        output_df = output_df[
            ~output_df.apply(is_duplicate_header_row, axis=1)
        ].copy()

    output_df["season"] = season
    output_df["team_name"] = TEAM_NAME_STANDARD
    output_df["source_url"] = source_url

    if "date" in output_df.columns:
        output_df["date"] = pd.to_datetime(output_df["date"], errors="coerce").dt.normalize()

    numeric_candidates = [
        "game_number",
        "team_points",
        "opponent_points",
        "wins",
        "losses",
        "srs",
    ]
    for column in numeric_candidates:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")

    if "site_flag" not in output_df.columns:
        output_df["site_flag"] = ""

    site_values = output_df["site_flag"].astype(str).str.strip().str.upper()
    output_df["location_type"] = "home"
    output_df.loc[site_values == "@", "location_type"] = "away"
    output_df.loc[site_values == "N", "location_type"] = "neutral"

    if "overtime_text" not in output_df.columns:
        output_df["overtime_text"] = ""

    output_df["overtime_flag"] = output_df["overtime_text"].astype(str).str.strip().ne("")

    if "result_wl" not in output_df.columns:
        output_df["result_wl"] = ""

    output_df["result_wl"] = output_df["result_wl"].astype(str).str.strip().str.upper()

    if "team_points" in output_df.columns and "opponent_points" in output_df.columns:
        output_df["point_differential"] = output_df["team_points"] - output_df["opponent_points"]
    else:
        output_df["point_differential"] = pd.NA

    output_df["result"] = pd.NA
    output_df.loc[output_df["result_wl"] == "W", "result"] = "win"
    output_df.loc[output_df["result_wl"] == "L", "result"] = "loss"

    output_df["game_result"] = pd.NA
    has_score_mask = output_df["team_points"].notna() & output_df["opponent_points"].notna()
    output_df.loc[has_score_mask, "game_result"] = (
        output_df.loc[has_score_mask, "result_wl"].astype(str).str.upper()
        + " "
        + output_df.loc[has_score_mask, "team_points"].astype("Int64").astype(str)
        + "-"
        + output_df.loc[has_score_mask, "opponent_points"].astype("Int64").astype(str)
    )

    if "opponent" not in output_df.columns:
        raise ValueError("Schedule table is missing opponent column after cleaning.")

    output_df["opponent"] = output_df["opponent"].astype(str).str.strip()

    if "opponent_conference" not in output_df.columns:
        output_df["opponent_conference"] = pd.NA

    if "streak" not in output_df.columns:
        output_df["streak"] = pd.NA

    played_mask = (
        output_df["date"].notna()
        & output_df["opponent"].ne("")
        & output_df["team_points"].notna()
        & output_df["opponent_points"].notna()
    )
    output_df = output_df[played_mask].copy()

    today_ts = pd.Timestamp.today().normalize()
    output_df = output_df[output_df["date"] <= today_ts].copy()

    output_df = output_df.sort_values(by=["date", "game_number"], ascending=[True, True]).reset_index(drop=True)

    output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")

    return output_df


def ensure_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in OUTPUT_COLUMN_ORDER:
        if column not in output_df.columns:
            output_df[column] = pd.NA

    output_df = output_df[OUTPUT_COLUMN_ORDER]

    missing_required = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in output_df.columns]
    if missing_required:
        raise ValueError(f"Output is missing required columns: {missing_required}")

    for column in REQUIRED_OUTPUT_COLUMNS:
        if output_df[column].isna().all():
            raise ValueError(f"Output column is fully null: {column}")

    return output_df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in output_df.columns:
            output_df[column] = pd.to_numeric(output_df[column], errors="coerce")

    return output_df


def export_output(df: pd.DataFrame, raw_output_path: Path, processed_output_path: Path) -> None:
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(raw_output_path, index=False)
    df.to_csv(processed_output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        required=False,
        help="Optional season end year override. Example: 2026 for the 2025-26 season.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    season = args.season if args.season is not None else infer_current_season(date.today())

    session = create_session()
    raw_schedule_df, source_url = fetch_schedule_table(session=session, season=season)

    cleaned_schedule_df = clean_schedule_table(
        schedule_df=raw_schedule_df,
        season=season,
        source_url=source_url,
    )
    cleaned_schedule_df = coerce_numeric_columns(cleaned_schedule_df)
    cleaned_schedule_df = ensure_output_schema(cleaned_schedule_df)

    raw_output_path = build_path(RAW_OUTPUT_PATH)
    processed_output_path = build_path(PROCESSED_OUTPUT_PATH)

    export_output(
        df=cleaned_schedule_df,
        raw_output_path=raw_output_path,
        processed_output_path=processed_output_path,
    )

    print(f"Montana schedule saved to: {processed_output_path}")
    print(f"Rows exported: {len(cleaned_schedule_df)}")
    print(f"Season used: {season}")
    print(f"Source URL: {source_url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)