"""
File: clean_kenpom_team_master.py
Last Modified: 2026-03-09
Purpose: Clean and merge raw KenPom team-level endpoint exports into a master team dataset for downstream use in the NCAA Next Man Up project.

Inputs:
- data/raw/kenpom/ratings/all_d1_ratings.csv
- data/raw/kenpom/four_factors/all_d1_four_factors.csv
- data/raw/kenpom/point_distribution/all_d1_point_distribution.csv
- data/raw/kenpom/height/all_d1_height.csv
- data/raw/kenpom/misc_stats/all_d1_misc_stats.csv
- data/raw/kenpom/teams/all_d1_teams.csv
- data/raw/kenpom/conferences/all_d1_conferences.csv

Outputs:
- data/processed/kenpom_team_master.csv
- data/processed/montana_kenpom_team_master.csv
- data/processed/big_sky_kenpom_team_master.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.utilities.kenpom_api_utils import (
    drop_duplicate_rows,
    export_csv,
    filter_to_big_sky,
    filter_to_montana,
    get_raw_kenpom_dir,
    sort_dataset,
    validate_non_empty_dataframe,
    validate_required_columns,
    validate_required_values,
)


KEY_COLUMNS = ["team_name", "season"]
CORE_COLUMNS = ["team_name", "conference_name", "season"]

DATASET_CONFIG = {
    "ratings": {
        "path_parts": ("ratings", "all_d1_ratings.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": True,
    },
    "four_factors": {
        "path_parts": ("four_factors", "all_d1_four_factors.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": False,
    },
    "point_distribution": {
        "path_parts": ("point_distribution", "all_d1_point_distribution.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": False,
    },
    "height": {
        "path_parts": ("height", "all_d1_height.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": False,
    },
    "misc_stats": {
        "path_parts": ("misc_stats", "all_d1_misc_stats.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": False,
    },
    "teams": {
        "path_parts": ("teams", "all_d1_teams.csv"),
        "required_columns": CORE_COLUMNS,
        "use_in_merge": True,
        "is_base": False,
    },
    "conferences": {
        "path_parts": ("conferences", "all_d1_conferences.csv"),
        "required_columns": ["conference_name", "season"],
        "use_in_merge": False,
        "is_base": False,
    },
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_processed_dir() -> Path:
    return get_project_root() / "data" / "processed"


def build_input_paths() -> dict[str, Path]:
    raw_kenpom_dir = get_raw_kenpom_dir()

    input_paths: dict[str, Path] = {}
    for dataset_name, config in DATASET_CONFIG.items():
        input_paths[dataset_name] = raw_kenpom_dir.joinpath(*config["path_parts"])

    return input_paths


def build_output_paths() -> dict[str, Path]:
    processed_dir = get_processed_dir()

    return {
        "all_d1": processed_dir / "kenpom_team_master.csv",
        "montana": processed_dir / "montana_kenpom_team_master.csv",
        "big_sky": processed_dir / "big_sky_kenpom_team_master.csv",
    }


def ensure_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file not found: {file_path}")


def standardize_string_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def standardize_team_dataset(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    output_df["team_name"] = standardize_string_series(output_df["team_name"])
    output_df["conference_name"] = standardize_string_series(output_df["conference_name"])
    output_df["season"] = pd.to_numeric(output_df["season"], errors="coerce").astype("Int64")

    return output_df


def standardize_conference_dataset(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    output_df["conference_name"] = standardize_string_series(output_df["conference_name"])
    output_df["season"] = pd.to_numeric(output_df["season"], errors="coerce").astype("Int64")

    return output_df


def validate_unique_keys(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[duplicate_mask, key_columns].drop_duplicates()
        sample_records = duplicate_rows.head(10).to_dict(orient="records")
        raise RuntimeError(
            f"{dataset_name} has duplicate merge keys for {key_columns}. "
            f"Sample duplicates: {sample_records}"
        )


def load_dataset(dataset_name: str, file_path: Path) -> pd.DataFrame:
    ensure_file_exists(file_path)

    df = pd.read_csv(file_path)
    validate_non_empty_dataframe(df, f"{dataset_name} input")
    validate_required_columns(
        df,
        DATASET_CONFIG[dataset_name]["required_columns"],
        f"{dataset_name} input",
    )

    if dataset_name == "conferences":
        df = standardize_conference_dataset(df)
        validate_required_values(
            df,
            ["conference_name", "season"],
            f"{dataset_name} input",
        )
        df = drop_duplicate_rows(df, subset=["conference_name", "season"])
        validate_unique_keys(df, ["conference_name", "season"], f"{dataset_name} input")
    else:
        df = standardize_team_dataset(df)
        validate_required_values(df, CORE_COLUMNS, f"{dataset_name} input")
        df = drop_duplicate_rows(df, subset=KEY_COLUMNS)
        validate_unique_keys(df, KEY_COLUMNS, f"{dataset_name} input")

    return df


def rename_overlapping_columns(
    df: pd.DataFrame,
    existing_columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    output_df = df.copy()
    rename_map: dict[str, str] = {}

    for column in output_df.columns:
        if column in KEY_COLUMNS:
            continue
        if column in existing_columns:
            rename_map[column] = f"{column}_{dataset_name}"

    if rename_map:
        output_df = output_df.rename(columns=rename_map)

    return output_df


def merge_endpoint_dataset(
    master_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    prepared_df = rename_overlapping_columns(
        df=dataset_df,
        existing_columns=list(master_df.columns),
        dataset_name=dataset_name,
    )

    merged_df = master_df.merge(
        prepared_df,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    return merged_df


def backfill_core_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    output_df = master_df.copy()

    conference_candidates = [
        column
        for column in output_df.columns
        if column == "conference_name" or column.startswith("conference_name_")
    ]

    if "conference_name" not in output_df.columns:
        output_df["conference_name"] = pd.NA

    for column in conference_candidates:
        if column == "conference_name":
            continue
        output_df["conference_name"] = output_df["conference_name"].fillna(output_df[column])

    output_df["team_name"] = standardize_string_series(output_df["team_name"])
    output_df["conference_name"] = standardize_string_series(output_df["conference_name"])
    output_df["season"] = pd.to_numeric(output_df["season"], errors="coerce").astype("Int64")

    return output_df


def validate_merge_row_count(
    base_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if len(base_df) != len(merged_df):
        raise RuntimeError(
            f"Row count changed after merging {dataset_name}. "
            f"Base rows: {len(base_df)}, merged rows: {len(merged_df)}"
        )


def build_master_dataset(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    master_df = datasets["ratings"].copy()

    merge_order = [
        "four_factors",
        "point_distribution",
        "height",
        "misc_stats",
        "teams",
    ]

    for dataset_name in merge_order:
        pre_merge_rows = len(master_df)
        master_df = merge_endpoint_dataset(
            master_df=master_df,
            dataset_df=datasets[dataset_name],
            dataset_name=dataset_name,
        )
        validate_merge_row_count(
            base_df=datasets["ratings"].iloc[:pre_merge_rows].copy(),
            merged_df=master_df,
            dataset_name=dataset_name,
        )

    master_df = backfill_core_columns(master_df)
    master_df = drop_duplicate_rows(master_df, subset=KEY_COLUMNS)
    validate_unique_keys(master_df, KEY_COLUMNS, "KenPom team master")

    validate_non_empty_dataframe(master_df, "KenPom team master")
    validate_required_columns(master_df, CORE_COLUMNS, "KenPom team master")
    validate_required_values(master_df, CORE_COLUMNS, "KenPom team master")

    master_df = sort_master_dataset(master_df)

    return master_df


def sort_master_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return sort_dataset(df, sort_columns=["season", "team_name"])
    except TypeError:
        return (
            df.sort_values(
                by=["season", "team_name"],
                ascending=[True, True],
                na_position="last",
            )
            .reset_index(drop=True)
        )


def validate_conference_reference(
    master_df: pd.DataFrame,
    conference_df: pd.DataFrame,
) -> None:
    reference_keys = set(
        conference_df[["conference_name", "season"]]
        .dropna()
        .itertuples(index=False, name=None)
    )

    unmatched_df = master_df.loc[
        ~master_df[["conference_name", "season"]]
        .apply(tuple, axis=1)
        .isin(reference_keys),
        ["team_name", "conference_name", "season"],
    ].drop_duplicates()

    if not unmatched_df.empty:
        sample_records = unmatched_df.head(10).to_dict(orient="records")
        raise RuntimeError(
            "KenPom team master contains conference_name + season values not found in "
            f"the conference reference dataset. Sample mismatches: {sample_records}"
        )


def filter_outputs(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    montana_df = filter_to_montana(master_df)
    big_sky_df = filter_to_big_sky(master_df)

    validate_non_empty_dataframe(montana_df, "Montana KenPom team master")
    validate_non_empty_dataframe(big_sky_df, "Big Sky KenPom team master")

    montana_df = drop_duplicate_rows(montana_df, subset=KEY_COLUMNS)
    big_sky_df = drop_duplicate_rows(big_sky_df, subset=KEY_COLUMNS)

    validate_unique_keys(montana_df, KEY_COLUMNS, "Montana KenPom team master")
    validate_unique_keys(big_sky_df, KEY_COLUMNS, "Big Sky KenPom team master")

    montana_df = sort_master_dataset(montana_df)
    big_sky_df = sort_master_dataset(big_sky_df)

    validate_required_columns(montana_df, CORE_COLUMNS, "Montana KenPom team master")
    validate_required_columns(big_sky_df, CORE_COLUMNS, "Big Sky KenPom team master")

    validate_required_values(montana_df, CORE_COLUMNS, "Montana KenPom team master")
    validate_required_values(big_sky_df, CORE_COLUMNS, "Big Sky KenPom team master")

    return montana_df, big_sky_df


def export_outputs(
    master_df: pd.DataFrame,
    montana_df: pd.DataFrame,
    big_sky_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    export_csv(master_df, output_paths["all_d1"])
    export_csv(montana_df, output_paths["montana"])
    export_csv(big_sky_df, output_paths["big_sky"])


def print_export_summary(
    master_df: pd.DataFrame,
    montana_df: pd.DataFrame,
    big_sky_df: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    print("KenPom team master cleaning complete.")
    print(f"All D1 rows: {len(master_df)} -> {output_paths['all_d1']}")
    print(f"Montana rows: {len(montana_df)} -> {output_paths['montana']}")
    print(f"Big Sky rows: {len(big_sky_df)} -> {output_paths['big_sky']}")


def main() -> None:
    input_paths = build_input_paths()
    output_paths = build_output_paths()

    datasets = {
        dataset_name: load_dataset(dataset_name, file_path)
        for dataset_name, file_path in input_paths.items()
    }

    master_df = build_master_dataset(datasets)
    validate_conference_reference(
        master_df=master_df,
        conference_df=datasets["conferences"],
    )

    montana_df, big_sky_df = filter_outputs(master_df)

    export_outputs(
        master_df=master_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )

    print_export_summary(
        master_df=master_df,
        montana_df=montana_df,
        big_sky_df=big_sky_df,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()