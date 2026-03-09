"""
File: clean_player_stats.py
Last Modified: 2026-03-02
Purpose: Clean and merge raw Montana and Big Sky player season stats and advanced stats from Sports Reference loaders into analytics-ready season-level player datasets for downstream feature engineering, evaluation engine work, and dashboard use in the NCAA Next Man Up project.

Inputs:
- data/raw/sports_reference/montana_player_season_stats.csv
- data/raw/sports_reference/big_sky_player_season_stats.csv
- data/raw/sports_reference/montana_player_advanced_stats.csv
- data/raw/sports_reference/big_sky_player_advanced_stats.csv
- data/processed/player_data/player_profile_montana.csv
- data/processed/comparison_sets/player_profile_big_sky.csv

Outputs:
- data/processed/player_data/player_stats_all_games_montana.csv
- data/processed/comparison_sets/conference_player_pool.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from scripts.utilities import config, file_paths, name_standardization


MERGE_KEYS = ["player_name", "team_name", "season"]
DROP_IF_FULLY_EMPTY = ["pct_possessions", "or_pct", "dr_pct"]

REQUIRED_SEASON_COLUMNS = [
    "player_name",
    "team_name",
    "season",
]

REQUIRED_ADVANCED_COLUMNS = [
    "player_name",
    "team_name",
    "season",
]

PROFILE_KEEP_COLUMNS = [
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


def _read_csv(path: Path, optional: bool = False) -> pd.DataFrame:
    if optional and not path.exists():
        return pd.DataFrame()
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty and not optional:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _standardize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].map(_clean_text)
        df["player_name"] = _apply_standardizer(df["player_name"], STANDARDIZE_PLAYER)

    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].map(_clean_text)
        df["team_name"] = _apply_standardizer(df["team_name"], STANDARDIZE_TEAM)

    if "conference_name" in df.columns:
        df["conference_name"] = df["conference_name"].map(_clean_text)
        df["conference_name"] = _apply_standardizer(df["conference_name"], STANDARDIZE_CONFERENCE)

    if "season" in df.columns:
        df["season"] = df["season"].map(_clean_season)

    return df


def _trim_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        df[col] = df[col].map(_clean_text)
    return df


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if not str(col).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def _drop_full_empty_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].isna().all():
            df = df.drop(columns=col)
    return df


def _drop_duplicate_columns_after_merge(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_suffix_cols = [col for col in df.columns if col.endswith("_dup")]
    if duplicate_suffix_cols:
        df = df.drop(columns=duplicate_suffix_cols)
    return df


def _validate_unique_keys(df: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicates = df[df.duplicated(subset=keys, keep=False)]
    if not duplicates.empty:
        example_rows = duplicates[keys].drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate {label} keys found for {keys}: {example_rows}")


def _coalesce_columns(df: pd.DataFrame, left_col: str, right_col: str, out_col: str) -> pd.Series:
    left_series = df[left_col] if left_col in df.columns else pd.Series(pd.NA, index=df.index)
    right_series = df[right_col] if right_col in df.columns else pd.Series(pd.NA, index=df.index)
    return left_series.combine_first(right_series)


def _combine_overlap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    overlap_roots = sorted(
        {
            col[:-2]
            for col in df.columns
            if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
        }
    )

    for root in overlap_roots:
        left_col = f"{root}_x"
        right_col = f"{root}_y"
        df[root] = _coalesce_columns(df, left_col, right_col, root)
        df = df.drop(columns=[left_col, right_col])

    return df


def _infer_numeric_columns(df: pd.DataFrame, protected_text_columns: set[str]) -> list[str]:
    numeric_cols: list[str] = []

    for col in df.columns:
        if col in protected_text_columns:
            continue
        if col in MERGE_KEYS:
            continue

        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if series.dropna().empty:
            continue

        converted = pd.to_numeric(series, errors="coerce")
        non_null_original = series.notna().sum()
        non_null_converted = converted.notna().sum()

        if non_null_original == 0:
            continue

        if non_null_converted == non_null_original:
            numeric_cols.append(col)

    return numeric_cols


def _convert_numeric_columns(df: pd.DataFrame, protected_text_columns: set[str]) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = _infer_numeric_columns(df, protected_text_columns)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def _sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_front = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "player_url",
        "jersey_number",
        "class",
        "height",
        "weight",
        "hometown",
        "high_school",
        "position_raw",
    ]
    front_cols = [col for col in preferred_front if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in front_cols]
    return df[front_cols + remaining_cols].copy()


def _prepare_stats_df(df: pd.DataFrame, required: list[str], path: Path) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    df = _drop_unnamed_columns(df)
    _validate_columns(df, required, path)

    df = _trim_text_columns(df)
    df = _standardize_core_columns(df)
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=MERGE_KEYS).copy()

    if df.empty:
        raise ValueError(f"No valid rows remain after cleaning: {path}")

    _validate_unique_keys(df, MERGE_KEYS, f"source dataframe ({path.name})")
    return df


def _prepare_profile_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = _drop_unnamed_columns(df)
    keep_cols = [col for col in PROFILE_KEEP_COLUMNS if col in df.columns]
    df = df[keep_cols].copy()

    df = _trim_text_columns(df)
    df = _standardize_core_columns(df)
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=MERGE_KEYS).copy()

    if df.empty:
        return df

    _validate_unique_keys(df, MERGE_KEYS, "profile dataframe")
    return df


def _merge_stats_and_advanced(
    season_df: pd.DataFrame,
    advanced_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    merged = season_df.merge(
        advanced_df,
        on=MERGE_KEYS,
        how="outer",
        suffixes=("_x", "_y"),
        validate="one_to_one",
    )

    merged = _combine_overlap_columns(merged)
    merged = _drop_duplicate_columns_after_merge(merged)

    if not profile_df.empty:
        merged = merged.merge(
            profile_df,
            on=MERGE_KEYS,
            how="left",
            suffixes=("", "_dup"),
            validate="one_to_one",
        )
        merged = _combine_overlap_columns(merged)
        merged = _drop_duplicate_columns_after_merge(merged)

    _validate_unique_keys(merged, MERGE_KEYS, f"merged {label} dataframe")

    protected_text_columns = set(PROFILE_KEEP_COLUMNS) | set(MERGE_KEYS)
    merged = _convert_numeric_columns(merged, protected_text_columns)

    for col in DROP_IF_FULLY_EMPTY:
        if col in merged.columns and merged[col].isna().all():
            merged = merged.drop(columns=col)

    merged = _drop_full_empty_columns(merged, DROP_IF_FULLY_EMPTY)

    merged = merged.drop_duplicates().copy()
    merged = merged.sort_values(["season", "team_name", "player_name"]).reset_index(drop=True)

    if merged.empty:
        raise ValueError(f"Merged {label} dataframe is empty.")

    return _sort_columns(merged)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    _ = config

    montana_season_stats_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/montana_player_season_stats.csv",
        candidate_attr_names=[
            "MONTANA_PLAYER_SEASON_STATS_FILE",
            "MONTANA_PLAYER_SEASON_STATS_PATH",
            "RAW_MONTANA_PLAYER_SEASON_STATS_FILE",
            "RAW_MONTANA_PLAYER_SEASON_STATS_PATH",
        ],
    )

    big_sky_season_stats_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/big_sky_player_season_stats.csv",
        candidate_attr_names=[
            "BIG_SKY_PLAYER_SEASON_STATS_FILE",
            "BIG_SKY_PLAYER_SEASON_STATS_PATH",
            "RAW_BIG_SKY_PLAYER_SEASON_STATS_FILE",
            "RAW_BIG_SKY_PLAYER_SEASON_STATS_PATH",
        ],
    )

    montana_advanced_stats_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/montana_player_advanced_stats.csv",
        candidate_attr_names=[
            "MONTANA_PLAYER_ADVANCED_STATS_FILE",
            "MONTANA_PLAYER_ADVANCED_STATS_PATH",
            "RAW_MONTANA_PLAYER_ADVANCED_STATS_FILE",
            "RAW_MONTANA_PLAYER_ADVANCED_STATS_PATH",
        ],
    )

    big_sky_advanced_stats_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/big_sky_player_advanced_stats.csv",
        candidate_attr_names=[
            "BIG_SKY_PLAYER_ADVANCED_STATS_FILE",
            "BIG_SKY_PLAYER_ADVANCED_STATS_PATH",
            "RAW_BIG_SKY_PLAYER_ADVANCED_STATS_FILE",
            "RAW_BIG_SKY_PLAYER_ADVANCED_STATS_PATH",
        ],
    )

    montana_profile_path = _resolve_input_path(
        default_relative="data/processed/player_data/player_profile_montana.csv",
        candidate_attr_names=[
            "PLAYER_PROFILE_MONTANA_FILE",
            "PLAYER_PROFILE_MONTANA_PATH",
            "PROCESSED_PLAYER_PROFILE_MONTANA_FILE",
            "PROCESSED_PLAYER_PROFILE_MONTANA_PATH",
        ],
    )

    big_sky_profile_path = _resolve_input_path(
        default_relative="data/processed/comparison_sets/player_profile_big_sky.csv",
        candidate_attr_names=[
            "PLAYER_PROFILE_BIG_SKY_FILE",
            "PLAYER_PROFILE_BIG_SKY_PATH",
            "PROCESSED_PLAYER_PROFILE_BIG_SKY_FILE",
            "PROCESSED_PLAYER_PROFILE_BIG_SKY_PATH",
        ],
    )

    montana_output_path = _resolve_output_path(
        default_relative="data/processed/player_data/player_stats_all_games_montana.csv",
        candidate_attr_names=[
            "PLAYER_STATS_ALL_GAMES_MONTANA_FILE",
            "PLAYER_STATS_ALL_GAMES_MONTANA_PATH",
            "PROCESSED_PLAYER_STATS_ALL_GAMES_MONTANA_FILE",
            "PROCESSED_PLAYER_STATS_ALL_GAMES_MONTANA_PATH",
        ],
    )

    big_sky_output_path = _resolve_output_path(
        default_relative="data/processed/comparison_sets/conference_player_pool.csv",
        candidate_attr_names=[
            "CONFERENCE_PLAYER_POOL_FILE",
            "CONFERENCE_PLAYER_POOL_PATH",
            "PROCESSED_CONFERENCE_PLAYER_POOL_FILE",
            "PROCESSED_CONFERENCE_PLAYER_POOL_PATH",
        ],
    )

    montana_season_df = _prepare_stats_df(
        _read_csv(montana_season_stats_path),
        REQUIRED_SEASON_COLUMNS,
        montana_season_stats_path,
    )
    big_sky_season_df = _prepare_stats_df(
        _read_csv(big_sky_season_stats_path),
        REQUIRED_SEASON_COLUMNS,
        big_sky_season_stats_path,
    )
    montana_advanced_df = _prepare_stats_df(
        _read_csv(montana_advanced_stats_path),
        REQUIRED_ADVANCED_COLUMNS,
        montana_advanced_stats_path,
    )
    big_sky_advanced_df = _prepare_stats_df(
        _read_csv(big_sky_advanced_stats_path),
        REQUIRED_ADVANCED_COLUMNS,
        big_sky_advanced_stats_path,
    )

    montana_profile_df = _prepare_profile_df(_read_csv(montana_profile_path, optional=True))
    big_sky_profile_df = _prepare_profile_df(_read_csv(big_sky_profile_path, optional=True))

    montana_final = _merge_stats_and_advanced(
        season_df=montana_season_df,
        advanced_df=montana_advanced_df,
        profile_df=montana_profile_df,
        label="Montana",
    )

    big_sky_final = _merge_stats_and_advanced(
        season_df=big_sky_season_df,
        advanced_df=big_sky_advanced_df,
        profile_df=big_sky_profile_df,
        label="Big Sky",
    )

    _write_csv(montana_final, montana_output_path)
    _write_csv(big_sky_final, big_sky_output_path)

    print(f"Saved Montana player stats: {len(montana_final)} rows -> {montana_output_path}")
    print(f"Saved conference player pool: {len(big_sky_final)} rows -> {big_sky_output_path}")


if __name__ == "__main__":
    main()