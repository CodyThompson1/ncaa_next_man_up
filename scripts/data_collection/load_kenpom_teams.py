"""
File: load_kenpom_teams.py
Last Modified: 2026-02-09
Purpose: Load current-season KenPom teams endpoint data for all D1 teams and export all D1, Montana-only, and Big Sky-only raw outputs.

Inputs:
- KenPom API teams endpoint
- scripts/utilities/kenpom_api_utils.py

Outputs:
- data/raw/kenpom/teams/all_d1_teams.csv
- data/raw/kenpom/teams/montana_teams.csv
- data/raw/kenpom/teams/big_sky_teams.csv

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
    filter_to_big_sky,
    filter_to_montana,
    get_current_season,
    get_raw_kenpom_dir,
    print_export_summary,
    sort_dataset,
    standardize_core_columns,
    validate_non_empty_dataframe,
    validate_required_columns,
    validate_required_values,
)


ENDPOINT = "teams"
ENDPOINT_FOLDER_NAME = "teams"
REQUIRED_COLUMNS = ["team_name", "conference_name", "season"]


def build_output_paths() -> dict[str, Path]:
    base_dir = get_raw_kenpom_dir() / ENDPOINT_FOLDER_NAME

    return {
        "all_d1": base_dir / "all_d1_teams.csv",
        "montana": base_dir / "montana_teams.csv",
        "big_sky": base_dir / "big_sky_teams.csv",
    }


def load_all_d1_teams(season: int) -> pd.DataFrame:
    df = fetch_endpoint_dataframe(endpoint=ENDPOINT, season=season)
    df = standardize_core_columns(df, season=season)
    df = add_source_metadata(df, endpoint=ENDPOINT)
    df = drop_duplicate_rows(df, subset=["team_name", "season"])
    df = sort_dataset(df, sort_columns=["season", "team_name"])

    validate_non_empty_dataframe(df, "KenPom all D1 teams")
    validate_required_columns(df, REQUIRED_COLUMNS, "KenPom all D1 teams")
    validate_required_values(df, REQUIRED_COLUMNS, "KenPom all D1 teams")

    return df


def build_subset_outputs(all_d1_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    montana_df = filter_to_montana(all_d1_df)
    big_sky_df = filter_to_big_sky(all_d1_df)

    montana_df = drop_duplicate_rows(montana_df, subset=["team_name", "season"])
    big_sky_df = drop_duplicate_rows(big_sky_df, subset=["team_name", "season"])

    montana_df = sort_dataset(montana_df, sort_columns=["season", "team_name"])
    big_sky_df = sort_dataset(big_sky_df, sort_columns=["season", "team_name"])

    validate_non_empty_dataframe(montana_df, "KenPom Montana teams")
    validate_required_columns(montana_df, REQUIRED_COLUMNS, "KenPom Montana teams")
    validate_required_values(montana_df, REQUIRED_COLUMNS, "KenPom Montana teams")

    validate_non_empty_dataframe(big_sky_df, "KenPom Big Sky teams")
    validate_required_columns(big_sky_df, REQUIRED_COLUMNS, "KenPom Big Sky teams")
    validate_required_values(big_sky_df, REQUIRED_COLUMNS, "KenPom Big Sky teams")

    return montana_df, big_sky_df


def export_outputs(
    all_d1_df: pd.DataFrame,
    montana_df: pd.DataFrame,
    big_sky_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    export_csv(all_d1_df, output_paths["all_d1"])
    export_csv(montana_df, output_paths["montana"])
    export_csv(big_sky_df, output_paths["big_sky"])


def main() -> None:
    season = get_current_season()
    output_paths = build_output_paths()

    all_d1_df = load_all_d1_teams(season=season)
    montana_df, big_sky_df = build_subset_outputs(all_d1_df=all_d1_df)

    export_outputs(
        all_d1_df=all_d1_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )

    print_export_summary(
        dataset_name="KenPom teams",
        all_d1_df=all_d1_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()