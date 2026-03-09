"""
File: load_kenpom_conferences.py
Last Modified: 2026-02-19
Purpose: Load current-season KenPom conferences endpoint data and export the raw conference reference output for downstream team-level merging and validation.

Inputs:
- KenPom API conferences endpoint
- scripts/utilities/kenpom_api_utils.py

Outputs:
- data/raw/kenpom/conferences/all_d1_conferences.csv

Sources & Attribution:
- Data Source: KenPom
- Source URL Pattern: https://kenpom.com/
- Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
- Attribution: “Data courtesy of KenPom”
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.utilities.kenpom_api_utils import (
    add_source_metadata,
    drop_duplicate_rows,
    export_csv,
    fetch_endpoint_dataframe,
    get_current_season,
    get_raw_kenpom_dir,
    standardize_conference_name,
    validate_non_empty_dataframe,
    validate_required_columns,
    validate_required_values,
)


ENDPOINT = "conferences"
ENDPOINT_FOLDER_NAME = "conferences"
REQUIRED_COLUMNS = ["conference_name", "season"]


def build_output_path() -> Path:
    return get_raw_kenpom_dir() / ENDPOINT_FOLDER_NAME / "all_d1_conferences.csv"


def find_conference_source_column(df: pd.DataFrame) -> str | None:
    priority_columns = [
        "conference_name",
        "conference",
        "conf",
        "confshort",
        "conference_abbreviation",
        "confname",
        "name",
        "conference_full_name",
        "conference_short_name",
    ]

    for column in priority_columns:
        if column in df.columns:
            return column

    for column in df.columns:
        if "conference" in column or column.startswith("conf"):
            return column

    return None


def find_season_source_column(df: pd.DataFrame) -> str | None:
    priority_columns = ["season", "year"]

    for column in priority_columns:
        if column in df.columns:
            return column

    return None


def standardize_conference_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    output_df = df.copy()

    conference_source_column = find_conference_source_column(output_df)
    if conference_source_column is None:
        raise RuntimeError(
            "Unable to identify a conference source column in KenPom conferences output."
        )

    output_df["conference_name"] = output_df[conference_source_column].apply(
        standardize_conference_name
    )
    output_df["conference_name"] = output_df["conference_name"].replace("", pd.NA)

    season_source_column = find_season_source_column(output_df)
    if season_source_column is not None:
        output_df["season"] = pd.to_numeric(
            output_df[season_source_column],
            errors="coerce",
        ).fillna(season)
    else:
        output_df["season"] = season

    output_df["season"] = output_df["season"].astype(int)

    return output_df


def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["season", "conference_name"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


def load_all_d1_conferences(season: int) -> pd.DataFrame:
    df = fetch_endpoint_dataframe(endpoint=ENDPOINT, season=season)
    df = standardize_conference_columns(df, season=season)
    df = add_source_metadata(df, endpoint=ENDPOINT)
    df = drop_duplicate_rows(df, subset=["conference_name", "season"])
    df = sort_dataset(df)

    validate_non_empty_dataframe(df, "KenPom all D1 conferences")
    validate_required_columns(df, REQUIRED_COLUMNS, "KenPom all D1 conferences")
    validate_required_values(df, REQUIRED_COLUMNS, "KenPom all D1 conferences")

    return df


def print_conference_export_summary(all_d1_df: pd.DataFrame, output_path: Path) -> None:
    print("KenPom conferences load complete.")
    print(f"All D1 rows: {len(all_d1_df)}")
    print(f"Exported file: {output_path}")


def main() -> None:
    season = get_current_season()
    output_path = build_output_path()

    all_d1_df = load_all_d1_conferences(season=season)
    export_csv(all_d1_df, output_path)
    print_conference_export_summary(all_d1_df=all_d1_df, output_path=output_path)


if __name__ == "__main__":
    main()