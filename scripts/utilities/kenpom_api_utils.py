"""
File: kenpom_api_utils.py
Last Modified: 2026-03-09
Purpose: Shared utility functions for loading current-season KenPom API team-level
data for the NCAA Next Man Up project.

Inputs:
- .env
- KenPom API responses

Outputs:
- Reusable helpers for endpoint loaders
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# Sources & Attribution:
# - Data Source: KenPom
# - Source URL Pattern: https://kenpom.com/
# - Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
# - Attribution: “Data courtesy of KenPom”


REQUEST_TIMEOUT = 60
USER_AGENT = "ncaa_next_man_up/1.0"
KENPOM_API_URL = "https://kenpom.com/api.php"

MONTANA_TEAM_NAME = "Montana"

BIG_SKY_TEAMS = {
    "Eastern Washington",
    "Idaho",
    "Idaho State",
    "Montana",
    "Montana State",
    "Northern Arizona",
    "Northern Colorado",
    "Portland State",
    "Sacramento State",
    "Weber State",
}

TEAM_NAME_STANDARDIZATION_MAP = {
    "Eastern Wash.": "Eastern Washington",
    "E. Washington": "Eastern Washington",
    "Idaho St.": "Idaho State",
    "Montana Grizzlies": "Montana",
    "Montana State Bobcats": "Montana State",
    "Montana St.": "Montana State",
    "Northern Ariz.": "Northern Arizona",
    "Northern Colo.": "Northern Colorado",
    "Portland St.": "Portland State",
    "Sacramento St.": "Sacramento State",
    "Weber St.": "Weber State",
}

CONFERENCE_STANDARDIZATION_MAP = {
    "BSC": "Big Sky",
    "BIG SKY": "Big Sky",
    "BigSky": "Big Sky",
    "big sky": "Big Sky",
}

COMMON_TEAM_NAME_ALIASES = [
    "team_name",
    "teamname",
    "team",
    "school",
    "name",
]

COMMON_CONFERENCE_ALIASES = [
    "conference",
    "conference_name",
    "conf",
    "confshort",
    "conference_abbreviation",
]

COMMON_SEASON_ALIASES = [
    "season",
    "year",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_raw_kenpom_dir() -> Path:
    return get_repo_root() / "data" / "raw" / "kenpom"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_current_season() -> int:
    now = datetime.now()
    return now.year + 1 if now.month >= 7 else now.year


def load_kenpom_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("KENPOM_API_KEY")

    if not api_key:
        raise RuntimeError("KENPOM_API_KEY not found in .env")

    return api_key


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }


def call_kenpom_api(
    endpoint: str,
    season: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    api_key = load_kenpom_api_key()
    params: dict[str, Any] = {"endpoint": endpoint}

    if season is not None:
        params["y"] = season

    if extra_params:
        params.update(extra_params)

    try:
        response = requests.get(
            KENPOM_API_URL,
            headers=build_headers(api_key),
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"KenPom API request failed for endpoint '{endpoint}' with params {params}: {exc}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"KenPom API returned a non-JSON response for endpoint '{endpoint}'"
        ) from exc


def extract_records(payload: Any, endpoint: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        candidate_keys = [
            "data",
            "results",
            "teams",
            "ratings",
            "four_factors",
            "pointdist",
            "height",
            "misc_stats",
            "conferences",
        ]

        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        flattened = pd.json_normalize(payload)
        if not flattened.empty:
            return flattened.to_dict(orient="records")

    raise RuntimeError(f"Unrecognized KenPom response format for endpoint '{endpoint}'")


def normalize_column_name(column_name: str) -> str:
    normalized = str(column_name).strip().lower()

    replacements = {
        "%": "pct",
        "/": "_",
        "-": "_",
        " ": "_",
        ".": "_",
        "(": "",
        ")": "",
        "'": "",
    }

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df.columns = [normalize_column_name(col) for col in output_df.columns]
    return output_df


def payload_to_dataframe(payload: Any, endpoint: str) -> pd.DataFrame:
    records = extract_records(payload, endpoint)
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(f"KenPom endpoint '{endpoint}' returned no rows")

    return normalize_columns(df)


def fetch_endpoint_dataframe(
    endpoint: str,
    season: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    payload = call_kenpom_api(
        endpoint=endpoint,
        season=season,
        extra_params=extra_params,
    )
    return payload_to_dataframe(payload, endpoint)


def find_first_matching_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    normalized_aliases = [normalize_column_name(alias) for alias in aliases]

    for alias in normalized_aliases:
        if alias in df.columns:
            return alias

    return None


def require_column(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    column = find_first_matching_column(df, aliases)

    if column is None:
        raise RuntimeError(
            f"Required column for '{label}' not found. Checked aliases: {aliases}"
        )

    return column


def standardize_team_name(value: Any) -> str | None:
    if pd.isna(value):
        return None

    team_name = str(value).strip()
    if not team_name:
        return None

    return TEAM_NAME_STANDARDIZATION_MAP.get(team_name, team_name)


def standardize_conference_name(value: Any) -> str | None:
    if pd.isna(value):
        return None

    conference_name = str(value).strip()
    if not conference_name:
        return None

    return CONFERENCE_STANDARDIZATION_MAP.get(conference_name, conference_name)


def apply_standardized_team_name_column(
    df: pd.DataFrame,
    source_column: str | None = None,
    output_column: str = "team_name",
) -> pd.DataFrame:
    output_df = df.copy()

    if source_column is None:
        source_column = require_column(output_df, COMMON_TEAM_NAME_ALIASES, "team_name")

    output_df[output_column] = output_df[source_column].apply(standardize_team_name)
    return output_df


def apply_standardized_conference_column(
    df: pd.DataFrame,
    source_column: str | None = None,
    output_column: str = "conference_name",
) -> pd.DataFrame:
    output_df = df.copy()

    if source_column is None:
        source_column = find_first_matching_column(output_df, COMMON_CONFERENCE_ALIASES)

    if source_column is None:
        output_df[output_column] = None
        return output_df

    output_df[output_column] = output_df[source_column].apply(standardize_conference_name)
    return output_df


def apply_season_column(
    df: pd.DataFrame,
    season: int,
    source_column: str | None = None,
    output_column: str = "season",
) -> pd.DataFrame:
    output_df = df.copy()

    if source_column is None:
        source_column = find_first_matching_column(output_df, COMMON_SEASON_ALIASES)

    if source_column is not None:
        output_df[output_column] = pd.to_numeric(
            output_df[source_column],
            errors="coerce",
        ).fillna(season)
    else:
        output_df[output_column] = season

    output_df[output_column] = output_df[output_column].astype("Int64")
    return output_df


def standardize_core_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    output_df = df.copy()
    output_df = apply_standardized_team_name_column(output_df)
    output_df = apply_standardized_conference_column(output_df)
    output_df = apply_season_column(output_df, season=season)
    return output_df


def validate_non_empty_dataframe(df: pd.DataFrame, dataset_name: str) -> None:
    if df.empty:
        raise RuntimeError(f"{dataset_name} is empty")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise RuntimeError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def validate_required_values(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(
                f"{dataset_name} cannot validate missing required column '{column}'"
            )

        if df[column].isna().any():
            raise RuntimeError(
                f"{dataset_name} has missing values in required column '{column}'"
            )

        if df[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(
                f"{dataset_name} has blank values in required column '{column}'"
            )


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    output_df = df.copy()

    if subset:
        subset = [col for col in subset if col in output_df.columns]

    if subset:
        output_df = output_df.drop_duplicates(subset=subset, keep="first")
    else:
        output_df = output_df.drop_duplicates(keep="first")

    return output_df.reset_index(drop=True)


def filter_to_montana(df: pd.DataFrame, team_column: str = "team_name") -> pd.DataFrame:
    if team_column not in df.columns:
        raise RuntimeError(f"Column '{team_column}' not found for Montana filter")

    output_df = df[df[team_column] == MONTANA_TEAM_NAME].copy().reset_index(drop=True)

    if output_df.empty:
        raise RuntimeError("Montana was not found in the dataset")

    return output_df


def filter_to_big_sky(
    df: pd.DataFrame,
    team_column: str = "team_name",
    conference_column: str = "conference_name",
) -> pd.DataFrame:
    output_df = df.copy()

    if team_column not in output_df.columns:
        raise RuntimeError(f"Column '{team_column}' not found for Big Sky filter")

    team_mask = output_df[team_column].isin(BIG_SKY_TEAMS)

    if conference_column in output_df.columns:
        conference_mask = output_df[conference_column].eq("Big Sky")
        mask = team_mask | conference_mask
    else:
        mask = team_mask

    filtered_df = output_df[mask].copy().reset_index(drop=True)

    if filtered_df.empty:
        raise RuntimeError("No Big Sky teams were found in the dataset")

    return filtered_df


def order_columns(
    df: pd.DataFrame,
    preferred_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not preferred_columns:
        return df.copy()

    ordered_existing = [col for col in preferred_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_existing]
    return df[ordered_existing + remaining].copy()


def sort_dataset(
    df: pd.DataFrame,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    output_df = df.copy()

    if not sort_columns:
        return output_df.reset_index(drop=True)

    existing_sort_columns = [col for col in sort_columns if col in output_df.columns]
    if not existing_sort_columns:
        return output_df.reset_index(drop=True)

    return output_df.sort_values(existing_sort_columns).reset_index(drop=True)


def add_source_metadata(
    df: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    output_df = df.copy()
    output_df["source_name"] = "KenPom"
    output_df["source_endpoint"] = endpoint
    output_df["load_timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    return output_df


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)


def export_standard_endpoint_outputs(
    df: pd.DataFrame,
    endpoint_folder_name: str,
    base_file_name: str,
) -> dict[str, Path]:
    base_dir = get_raw_kenpom_dir() / endpoint_folder_name
    ensure_directory(base_dir)

    all_d1_path = base_dir / f"all_d1_{base_file_name}.csv"
    montana_path = base_dir / f"montana_{base_file_name}.csv"
    big_sky_path = base_dir / f"big_sky_{base_file_name}.csv"

    export_csv(df, all_d1_path)
    export_csv(filter_to_montana(df), montana_path)
    export_csv(filter_to_big_sky(df), big_sky_path)

    return {
        "all_d1": all_d1_path,
        "montana": montana_path,
        "big_sky": big_sky_path,
    }


def print_export_summary(
    dataset_name: str,
    all_d1_df: pd.DataFrame,
    montana_df: pd.DataFrame,
    big_sky_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    print(f"{dataset_name} load complete.")
    print(f"All D1 rows: {len(all_d1_df)}")
    print(f"Montana rows: {len(montana_df)}")
    print(f"Big Sky rows: {len(big_sky_df)}")
    print(f"All D1 output: {output_paths['all_d1']}")
    print(f"Montana output: {output_paths['montana']}")
    print(f"Big Sky output: {output_paths['big_sky']}")