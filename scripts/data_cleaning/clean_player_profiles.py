"""
File: clean_player_profiles.py
Last Modified: 2026-03-02
Purpose: Clean and standardize raw Montana and Big Sky roster exports from Sports Reference loaders into merge-ready player profile datasets for downstream feature engineering, evaluation engine work, and dashboard use in the NCAA Next Man Up project.

Inputs:
- data/raw/sports_reference/montana_roster.csv
- data/raw/sports_reference/big_sky_rosters.csv

Outputs:
- data/processed/player_data/player_profile_montana.csv
- data/processed/comparison_sets/player_profile_big_sky.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from scripts.utilities import config, file_paths, name_standardization


REQUIRED_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "player_url",
    "jersey_number",
    "conference_name",
    "class",
    "height",
    "weight",
    "hometown",
    "high_school",
    "position_raw",
]

OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "player_url",
    "jersey_number",
    "conference_name",
    "class",
    "height",
    "weight",
    "hometown",
    "high_school",
    "position_raw",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_paths(*names: str) -> list[Path]:
    candidates: list[Path] = []
    for name in names:
        value = getattr(file_paths, name, None)
        if value:
            candidates.append(Path(value))
    return candidates


def _resolve_input_path(default_relative: str, candidate_attr_names: Iterable[str]) -> Path:
    candidates = _candidate_paths(*candidate_attr_names)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return _repo_root() / default_relative


def _resolve_output_path(default_relative: str, candidate_attr_names: Iterable[str]) -> Path:
    candidates = _candidate_paths(*candidate_attr_names)
    if candidates:
        return candidates[0]
    return _repo_root() / default_relative


def _get_standardizer(candidates: list[str]) -> Callable[[object], object]:
    for name in candidates:
        fn = getattr(name_standardization, name, None)
        if callable(fn):
            return fn
    return lambda x: x


STANDARDIZE_PLAYER = _get_standardizer(
    [
        "standardize_player_name",
        "standardize_person_name",
        "normalize_player_name",
        "clean_player_name",
        "clean_name",
    ]
)

STANDARDIZE_TEAM = _get_standardizer(
    [
        "standardize_team_name",
        "normalize_team_name",
        "clean_team_name",
    ]
)

STANDARDIZE_CONFERENCE = _get_standardizer(
    [
        "standardize_conference_name",
        "normalize_conference_name",
        "clean_conference_name",
    ]
)


def _clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return " ".join(text.split())


def _clean_url(value: object) -> object:
    value = _clean_text(value)
    if pd.isna(value):
        return pd.NA
    return str(value)


def _clean_jersey(value: object) -> object:
    value = _clean_text(value)
    if pd.isna(value):
        return pd.NA
    text = str(value).replace("#", "").strip()
    if not text:
        return pd.NA
    return text


def _clean_season(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return pd.NA


def _apply_standardizer(series: pd.Series, standardizer: Callable[[object], object]) -> pd.Series:
    return series.map(lambda x: _clean_text(standardizer(x)) if not pd.isna(x) else pd.NA)


def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _standardize_profile_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    text_columns = [
        "player_name",
        "team_name",
        "player_url",
        "jersey_number",
        "conference_name",
        "class",
        "height",
        "weight",
        "hometown",
        "high_school",
        "position_raw",
    ]

    for col in text_columns:
        df[col] = df[col].map(_clean_text)

    df["player_url"] = df["player_url"].map(_clean_url)
    df["jersey_number"] = df["jersey_number"].map(_clean_jersey)
    df["season"] = df["season"].map(_clean_season)

    df["player_name"] = _apply_standardizer(df["player_name"], STANDARDIZE_PLAYER)
    df["team_name"] = _apply_standardizer(df["team_name"], STANDARDIZE_TEAM)
    df["conference_name"] = _apply_standardizer(df["conference_name"], STANDARDIZE_CONFERENCE)

    df = df[OUTPUT_COLUMNS].copy()

    df = df.dropna(subset=["player_name", "team_name", "season"])
    df = df.drop_duplicates()
    df = df.sort_values(["season", "team_name", "player_name", "player_url"], na_position="last")

    df = df.drop_duplicates(subset=["player_name", "team_name", "season"], keep="first")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("Cleaned dataframe is empty after processing.")

    return df


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    _ = config

    montana_input = _resolve_input_path(
        default_relative="data/raw/sports_reference/montana_roster.csv",
        candidate_attr_names=[
            "MONTANA_ROSTER_FILE",
            "MONTANA_ROSTER_PATH",
            "RAW_MONTANA_ROSTER_FILE",
            "RAW_MONTANA_ROSTER_PATH",
        ],
    )

    big_sky_input = _resolve_input_path(
        default_relative="data/raw/sports_reference/big_sky_rosters.csv",
        candidate_attr_names=[
            "BIG_SKY_ROSTERS_FILE",
            "BIG_SKY_ROSTERS_PATH",
            "RAW_BIG_SKY_ROSTERS_FILE",
            "RAW_BIG_SKY_ROSTERS_PATH",
        ],
    )

    montana_output = _resolve_output_path(
        default_relative="data/processed/player_data/player_profile_montana.csv",
        candidate_attr_names=[
            "PLAYER_PROFILE_MONTANA_FILE",
            "PLAYER_PROFILE_MONTANA_PATH",
            "PROCESSED_PLAYER_PROFILE_MONTANA_FILE",
            "PROCESSED_PLAYER_PROFILE_MONTANA_PATH",
        ],
    )

    big_sky_output = _resolve_output_path(
        default_relative="data/processed/comparison_sets/player_profile_big_sky.csv",
        candidate_attr_names=[
            "PLAYER_PROFILE_BIG_SKY_FILE",
            "PLAYER_PROFILE_BIG_SKY_PATH",
            "PROCESSED_PLAYER_PROFILE_BIG_SKY_FILE",
            "PROCESSED_PLAYER_PROFILE_BIG_SKY_PATH",
        ],
    )

    montana_df = _read_csv(montana_input)
    big_sky_df = _read_csv(big_sky_input)

    _validate_columns(montana_df, montana_input)
    _validate_columns(big_sky_df, big_sky_input)

    montana_clean = _standardize_profile_df(montana_df)
    big_sky_clean = _standardize_profile_df(big_sky_df)

    _write_csv(montana_clean, montana_output)
    _write_csv(big_sky_clean, big_sky_output)

    print(f"Saved Montana player profiles: {len(montana_clean)} rows -> {montana_output}")
    print(f"Saved Big Sky player profiles: {len(big_sky_clean)} rows -> {big_sky_output}")


if __name__ == "__main__":
    main()