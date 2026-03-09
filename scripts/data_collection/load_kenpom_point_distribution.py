"""
File: load_kenpom_point_distribution.py
Last Modified: 2026-02-16
Purpose: Load current-season KenPom point distribution endpoint data for all D1 teams and export all D1, Montana-only, and Big Sky-only raw outputs.

Inputs:
- KenPom API pointdist endpoint
- KenPom API ratings endpoint
- scripts/utilities/kenpom_api_utils.py

Outputs:
- data/raw/kenpom/point_distribution/all_d1_point_distribution.csv
- data/raw/kenpom/point_distribution/montana_point_distribution.csv
- data/raw/kenpom/point_distribution/big_sky_point_distribution.csv

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


ENDPOINT = "pointdist"
CONFERENCE_BACKFILL_ENDPOINT = "ratings"
ENDPOINT_FOLDER_NAME = "point_distribution"
REQUIRED_COLUMNS = ["team_name", "conference_name", "season"]


def build_output_paths() -> dict[str, Path]:
    base_dir = get_raw_kenpom_dir() / ENDPOINT_FOLDER_NAME

    return {
        "all_d1": base_dir / "all_d1_point_distribution.csv",
        "montana": base_dir / "montana_point_distribution.csv",
        "big_sky": base_dir / "big_sky_point_distribution.csv",
    }


def load_conference_lookup(season: int) -> pd.DataFrame:
    ratings_df = fetch_endpoint_dataframe(
        endpoint=CONFERENCE_BACKFILL_ENDPOINT,
        season=season,
    )
    ratings_df = standardize_core_columns(ratings_df, season=season)

    validate_non_empty_dataframe(ratings_df, "KenPom ratings conference lookup")
    validate_required_columns(
        ratings_df,
        REQUIRED_COLUMNS,
        "KenPom ratings conference lookup",
    )
    validate_required_values(
        ratings_df,
        REQUIRED_COLUMNS,
        "KenPom ratings conference lookup",
    )

    lookup_df = ratings_df[["team_name", "conference_name", "season"]].copy()
    lookup_df = drop_duplicate_rows(lookup_df, subset=["team_name", "season"])

    return lookup_df


def backfill_conference_name(
    df: pd.DataFrame,
    conference_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = df.copy()

    output_df = output_df.merge(
        conference_lookup_df.rename(
            columns={"conference_name": "conference_name_lookup"}
        ),
        on=["team_name", "season"],
        how="left",
    )

    if "conference_name" not in output_df.columns:
        output_df["conference_name"] = output_df["conference_name_lookup"]
    else:
        output_df["conference_name"] = output_df["conference_name"].replace(
            "",
            pd.NA,
        )
        output_df["conference_name"] = output_df["conference_name"].fillna(
            output_df["conference_name_lookup"]
        )

    output_df = output_df.drop(columns=["conference_name_lookup"])

    return output_df


def load_all_d1_point_distribution(season: int) -> pd.DataFrame:
    df = fetch_endpoint_dataframe(endpoint=ENDPOINT, season=season)
    df = standardize_core_columns(df, season=season)

    conference_lookup_df = load_conference_lookup(season=season)
    df = backfill_conference_name(df, conference_lookup_df=conference_lookup_df)

    df = add_source_metadata(df, endpoint=ENDPOINT)
    df = drop_duplicate_rows(df, subset=["team_name", "season"])
    df = sort_dataset(df, sort_columns=["season", "team_name"])

    validate_non_empty_dataframe(df, "KenPom all D1 point distribution")
    validate_required_columns(
        df,
        REQUIRED_COLUMNS,
        "KenPom all D1 point distribution",
    )
    validate_required_values(
        df,
        REQUIRED_COLUMNS,
        "KenPom all D1 point distribution",
    )

    return df


def build_subset_outputs(all_d1_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    montana_df = filter_to_montana(all_d1_df)
    big_sky_df = filter_to_big_sky(all_d1_df)

    montana_df = drop_duplicate_rows(montana_df, subset=["team_name", "season"])
    big_sky_df = drop_duplicate_rows(big_sky_df, subset=["team_name", "season"])

    montana_df = sort_dataset(montana_df, sort_columns=["season", "team_name"])
    big_sky_df = sort_dataset(big_sky_df, sort_columns=["season", "team_name"])

    validate_non_empty_dataframe(montana_df, "KenPom Montana point distribution")
    validate_required_columns(
        montana_df,
        REQUIRED_COLUMNS,
        "KenPom Montana point distribution",
    )
    validate_required_values(
        montana_df,
        REQUIRED_COLUMNS,
        "KenPom Montana point distribution",
    )

    validate_non_empty_dataframe(big_sky_df, "KenPom Big Sky point distribution")
    validate_required_columns(
        big_sky_df,
        REQUIRED_COLUMNS,
        "KenPom Big Sky point distribution",
    )
    validate_required_values(
        big_sky_df,
        REQUIRED_COLUMNS,
        "KenPom Big Sky point distribution",
    )

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

    all_d1_df = load_all_d1_point_distribution(season=season)
    montana_df, big_sky_df = build_subset_outputs(all_d1_df=all_d1_df)

    export_outputs(
        all_d1_df=all_d1_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )

    print_export_summary(
        dataset_name="KenPom point distribution",
        all_d1_df=all_d1_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()