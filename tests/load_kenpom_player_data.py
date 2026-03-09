"""
File: load_kenpom_player_data.py
Last Modified: 2026-03-09
Purpose: Pull KenPom player statistics for the current season using the KenPom API and export a validated raw player statistics dataset.

Inputs:
- .env
- KenPom API response for the current season

Outputs:
- data/raw/kenpom/player_stats.csv
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Sources & Attribution:
# - Data Source: KenPom
# - Source URL Pattern: https://kenpom.com/api.php?endpoint=<player_endpoint>&y=<season>
# - Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
# - Attribution: “Data courtesy of KenPom”


REQUEST_TIMEOUT = 60
OUTPUT_FILE_NAME = "player_stats.csv"
PLAYER_ENDPOINT_CANDIDATES = [
    "playerstats",
    "players",
    "player-ratings",
    "player_stats",
]

REQUIRED_COLUMNS = [
    "player",
    "team",
    "minutes",
    "usage",
    "ortg",
    "drtg",
    "ts_pct",
    "efg_pct",
    "assist_rate",
    "turnover_rate",
    "rebound_rate",
    "season",
]

COLUMN_ALIASES = {
    "player": [
        "player",
        "playername",
        "player_name",
        "name",
        "athlete",
    ],
    "team": [
        "team",
        "teamname",
        "team_name",
        "school",
        "school_name",
    ],
    "minutes": [
        "minutes",
        "min",
        "mins",
        "mp",
        "minpct",
        "minutes_pct",
        "min_pct",
    ],
    "usage": [
        "usage",
        "usg",
        "usgpct",
        "usg_pct",
        "usage_rate",
        "usagepct",
    ],
    "ortg": [
        "ortg",
        "orating",
        "off_rating",
        "offensive_rating",
        "o_rating",
        "rating_off",
        "offrtg",
    ],
    "drtg": [
        "drtg",
        "drating",
        "def_rating",
        "defensive_rating",
        "d_rating",
        "rating_def",
        "defrtg",
    ],
    "ts_pct": [
        "ts_pct",
        "tspct",
        "ts",
        "true_shooting_pct",
        "true_shooting",
    ],
    "efg_pct": [
        "efg_pct",
        "efgpct",
        "efg",
        "effective_fg_pct",
        "effective_field_goal_pct",
    ],
    "assist_rate": [
        "assist_rate",
        "assistpct",
        "ast_rate",
        "astpct",
        "assist_pct",
        "assist_percentage",
    ],
    "turnover_rate": [
        "turnover_rate",
        "turnoverpct",
        "tov_rate",
        "tovpct",
        "to_rate",
        "turnover_pct",
        "turnover_percentage",
    ],
    "rebound_rate": [
        "rebound_rate",
        "reboundpct",
        "reb_rate",
        "rebpct",
        "reb_pct",
        "rebound_pct",
        "rebound_percentage",
    ],
    "season": [
        "season",
        "year",
        "y",
    ],
}


def get_repo_root():
    return Path(__file__).resolve().parents[2]


def get_output_path():
    return get_repo_root() / "data" / "raw" / "kenpom" / OUTPUT_FILE_NAME


def get_current_season():
    now = datetime.now()
    if now.month >= 7:
        return now.year + 1
    return now.year


def load_api_key():
    load_dotenv()
    api_key = os.getenv("KENPOM_API_KEY")

    if not api_key:
        raise RuntimeError("KENPOM_API_KEY not found in .env")

    return api_key


def normalize_column_name(column_name):
    normalized = str(column_name).strip().lower()
    normalized = normalized.replace("%", "pct")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def normalize_columns(df):
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def extract_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "results", "players", "playerstats", "stats"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        if payload:
            normalized = pd.json_normalize(payload)
            if not normalized.empty:
                return normalized.to_dict(orient="records")

    raise RuntimeError("KenPom response format not recognized")


def payload_to_dataframe(payload):
    records = extract_records(payload)
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("KenPom API returned no player records")

    return normalize_columns(df)


def endpoint_has_required_shape(df):
    matched_fields = 0

    for candidates in COLUMN_ALIASES.values():
        for candidate in candidates:
            if normalize_column_name(candidate) in df.columns:
                matched_fields += 1
                break

    return matched_fields >= 6


def call_kenpom_api(api_key, season):
    url = "https://kenpom.com/api.php"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "ncaa_next_man_up/1.0",
    }

    failures = []

    for endpoint in PLAYER_ENDPOINT_CANDIDATES:
        params = {
            "endpoint": endpoint,
            "y": season,
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            failures.append(f"{endpoint}: request failed: {exc}")
            continue

        if response.status_code != 200:
            failures.append(f"{endpoint}: HTTP {response.status_code}: {response.text[:300]}")
            continue

        try:
            payload = response.json()
        except ValueError:
            failures.append(f"{endpoint}: response was not valid JSON")
            continue

        try:
            df = payload_to_dataframe(payload)
        except Exception as exc:
            failures.append(f"{endpoint}: payload parse failed: {exc}")
            continue

        if not endpoint_has_required_shape(df):
            failures.append(
                f"{endpoint}: response did not resemble player stats. Available columns: {', '.join(df.columns.tolist())}"
            )
            continue

        return payload, endpoint

    raise RuntimeError(
        "Unable to load KenPom player data from candidate API endpoints. "
        + " | ".join(failures)
    )


def find_source_column(df, candidates):
    normalized_candidates = [normalize_column_name(col) for col in candidates]
    for candidate in normalized_candidates:
        if candidate in df.columns:
            return candidate
    return None


def map_columns(df, season):
    mapped = pd.DataFrame(index=df.index)
    missing_fields = []

    for target_column, candidates in COLUMN_ALIASES.items():
        source_column = find_source_column(df, candidates)

        if source_column is not None:
            mapped[target_column] = df[source_column]
            continue

        if target_column == "season":
            mapped[target_column] = season
            continue

        missing_fields.append(target_column)

    if missing_fields:
        available_columns = ", ".join(df.columns.tolist())
        raise RuntimeError(
            "Missing required field(s) in KenPom response: "
            f"{missing_fields}. Available columns: {available_columns}"
        )

    return mapped


def clean_dataset(df):
    df = df.copy()

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.strip()

    numeric_columns = [
        "minutes",
        "usage",
        "ortg",
        "drtg",
        "ts_pct",
        "efg_pct",
        "assist_rate",
        "turnover_rate",
        "rebound_rate",
        "season",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["season"] = df["season"].fillna(get_current_season())
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    df = df.dropna(
        subset=[
            "player",
            "team",
            "minutes",
            "usage",
            "ortg",
            "drtg",
            "ts_pct",
            "efg_pct",
            "assist_rate",
            "turnover_rate",
            "rebound_rate",
            "season",
        ]
    )

    df = df[(df["player"].astype(str).str.strip() != "")]
    df = df[(df["team"].astype(str).str.strip() != "")]

    df["season"] = df["season"].astype(int)
    df = df.drop_duplicates(subset=["player", "team", "season"]).reset_index(drop=True)

    return df


def validate_schema(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Output schema mismatch. Missing columns: {missing_columns}")

    if df.empty:
        raise RuntimeError("KenPom player dataset is empty")

    if df["player"].isna().any() or (df["player"].astype(str).str.strip() == "").any():
        raise RuntimeError("Player column contains missing values")

    if df["team"].isna().any() or (df["team"].astype(str).str.strip() == "").any():
        raise RuntimeError("Team column contains missing values")


def export_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    season = get_current_season()
    api_key = load_api_key()

    payload, endpoint_used = call_kenpom_api(api_key, season)
    raw_df = payload_to_dataframe(payload)

    output_df = map_columns(raw_df, season)
    output_df = clean_dataset(output_df)
    validate_schema(output_df)

    output_df = output_df[REQUIRED_COLUMNS].sort_values(
        by=["team", "player"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_path = get_output_path()
    export_csv(output_df, output_path)

    print(f"KenPom player stats saved to: {output_path}")
    print(f"KenPom endpoint used: {endpoint_used}")


if __name__ == "__main__":
    main()