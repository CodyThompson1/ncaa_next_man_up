"""
File: build_position_groups.py
Last Modified: 2026-03-04
Purpose: Build standardized player position groups for Montana players and the Big Sky comparison pool by converting raw position labels into the project's unified Guard and Forward position groups for use in peer grouping, archetype assignment, and downstream evaluation engine scoring in the NCAA Next Man Up project.

Inputs:
- data/processed/player_data/player_profile_montana.csv
- data/processed/comparison_sets/player_profile_big_sky.csv

Outputs:
- data/features/player_position_groups.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONTANA_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_data" / "player_profile_montana.csv"
)
BIG_SKY_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "comparison_sets" / "player_profile_big_sky.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "player_position_groups.csv"

REQUIRED_COLUMNS = {
    "player_name",
    "team_name",
    "season",
    "position_raw",
}

OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_raw",
    "class",
    "height",
    "weight",
    "position_group",
    "position_group_source",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input file is missing required columns {sorted(missing_columns)}: {path}"
        )

    return df


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["class", "height", "weight"]:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_position_text(position_raw: object) -> str:
    text = _clean_text(position_raw).lower()
    for char in ["/", "-", ",", ";", "|"]:
        text = text.replace(char, " ")
    text = " ".join(text.split())
    return text


def _classify_position_group(position_raw: object) -> tuple[str, str]:
    normalized = _normalize_position_text(position_raw)

    if not normalized:
        return "Forward", "default_missing_position_raw"

    tokens = set(normalized.split())

    guard_keywords = {
        "g",
        "pg",
        "sg",
        "guard",
        "combo",
        "point",
        "shooting",
    }
    forward_keywords = {
        "f",
        "sf",
        "pf",
        "forward",
        "wing",
        "big",
        "c",
        "cf",
        "fc",
        "center",
        "post",
    }

    if normalized in {"g", "pg", "sg"}:
        return "Guard", "position_raw_exact_guard"

    if normalized in {"f", "sf", "pf", "c", "fc", "cf"}:
        return "Forward", "position_raw_exact_forward_or_center"

    if "guard" in tokens or tokens.intersection({"pg", "sg", "g"}):
        return "Guard", "position_raw_contains_guard_keyword"

    if tokens.intersection(forward_keywords):
        return "Forward", "position_raw_contains_forward_or_center_keyword"

    if normalized.startswith("g"):
        return "Guard", "position_raw_prefix_guard"

    if normalized.startswith(("f", "c")):
        return "Forward", "position_raw_prefix_forward_or_center"

    return "Forward", "default_unmapped_to_forward"


def _build_position_groups(df: pd.DataFrame) -> pd.DataFrame:
    classified = df["position_raw"].apply(_classify_position_group)
    df["position_group"] = classified.apply(lambda x: x[0])
    df["position_group_source"] = classified.apply(lambda x: x[1])
    return df


def _standardize_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["player_name", "team_name", "position_raw", "class", "height", "weight"]:
        df[column] = df[column].apply(_clean_text)

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    df["player_name"] = df["player_name"].str.strip()
    df["team_name"] = df["team_name"].str.strip()
    df["position_raw"] = df["position_raw"].replace("", pd.NA)

    return df


def _combine_profiles(montana_df: pd.DataFrame, big_sky_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = pd.concat([montana_df, big_sky_df], ignore_index=True, sort=False)
    combined_df = _ensure_optional_columns(combined_df)
    combined_df = _standardize_core_fields(combined_df)

    combined_df = combined_df.drop_duplicates(
        subset=["player_name", "team_name", "season"],
        keep="first",
    ).reset_index(drop=True)

    return combined_df


def _validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Output dataframe is empty after processing.")

    missing_output_columns = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing_output_columns:
        raise ValueError(
            f"Output dataframe is missing required columns: {missing_output_columns}"
        )

    if df["position_group"].isna().any():
        missing_rows = df[df["position_group"].isna()][
            ["player_name", "team_name", "season", "position_raw"]
        ]
        raise ValueError(
            "Null position_group values remain after classification:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    invalid_groups = sorted(
        set(df["position_group"].dropna().unique()) - {"Guard", "Forward"}
    )
    if invalid_groups:
        raise ValueError(f"Invalid position_group values found: {invalid_groups}")

    if df["season"].isna().any():
        raise ValueError("Null season values remain in the output dataset.")

    if df[["player_name", "team_name"]].isna().any().any():
        raise ValueError("Null player_name or team_name values remain in the output dataset.")


def _finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    output_df = df[OUTPUT_COLUMNS].copy()
    output_df = output_df.sort_values(
        by=["team_name", "player_name", "season"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return output_df


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    montana_df = _read_csv(MONTANA_PROFILE_PATH)
    big_sky_df = _read_csv(BIG_SKY_PROFILE_PATH)

    combined_df = _combine_profiles(montana_df, big_sky_df)
    combined_df = _build_position_groups(combined_df)

    output_df = _finalize_output(combined_df)
    _validate_output(output_df)
    save_output(output_df, OUTPUT_PATH)

    print(f"Player position groups saved: {len(output_df)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()