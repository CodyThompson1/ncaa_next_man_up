"""
File: clean_player_game_logs.py
Last Modified: 2026-02-27
Purpose: Clean and standardize raw player game logs from Sports Reference loaders into analysis-ready game log datasets for Montana and the broader player game log pool for downstream dashboard use and later split-based analysis in the NCAA Next Man Up project.

Inputs:
- data/raw/sports_reference/player_game_logs.csv
- data/raw/sports_reference/player_game_log_failures.csv

Outputs:
- data/processed/player_data/player_game_logs_montana_clean.csv
- data/processed/comparison_sets/player_game_logs_clean.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from scripts.utilities import config, file_paths, name_standardization


COLUMN_ALIASES = {
    "player": "player_name",
    "team": "team_name",
    "opp": "opponent",
    "opponent_name": "opponent",
    "date": "game_date",
    "game_day": "game_date",
}

REQUIRED_COLUMNS = [
    "player_name",
    "team_name",
    "game_date",
]

TEXT_PRIORITY_COLUMNS = [
    "player_name",
    "team_name",
    "conference_name",
    "opponent",
    "game_location",
    "result",
    "player_url",
    "team_url",
    "opponent_url",
]

KEY_COLUMNS = ["player_name", "team_name", "season", "game_date", "opponent"]


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

STANDARDIZE_OPPONENT = _get_standardizer(
    [
        "standardize_team_name",
        "normalize_team_name",
        "clean_team_name",
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


def _rename_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if not str(col).lower().startswith("unnamed:")]
    return df[keep_cols].copy()


def _trim_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_columns:
        df[col] = df[col].map(_clean_text)
    return df


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

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(_clean_text)
        df["opponent"] = _apply_standardizer(df["opponent"], STANDARDIZE_OPPONENT)

    if "season" in df.columns:
        df["season"] = df["season"].map(_clean_season)

    return df


def _parse_game_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


def _derive_season_from_game_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "season" in df.columns and df["season"].notna().any():
        df["season"] = df["season"].map(_clean_season)
        return df

    parsed = pd.to_datetime(df["game_date"], errors="coerce")
    season = parsed.dt.year

    if season.notna().any():
        df["season"] = season.astype("Int64")
    else:
        df["season"] = pd.NA

    return df


def _ensure_opponent_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "opponent" not in df.columns:
        df["opponent"] = pd.NA

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(_clean_text)

    return df


def _fill_key_text_fallbacks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].fillna("Unknown Opponent")

    return df


def _infer_numeric_columns(df: pd.DataFrame, protected_text_columns: set[str]) -> list[str]:
    numeric_cols: list[str] = []

    for col in df.columns:
        if col in protected_text_columns:
            continue
        if col == "game_date":
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

        if non_null_original > 0 and non_null_converted == non_null_original:
            numeric_cols.append(col)

    return numeric_cols


def _convert_numeric_columns(df: pd.DataFrame, protected_text_columns: set[str]) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = _infer_numeric_columns(df, protected_text_columns)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _validate_non_empty_after_clean(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label} is empty after cleaning.")


def _prepare_game_logs(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = _drop_unnamed_columns(df)
    df = _rename_alias_columns(df)
    _validate_columns(df, REQUIRED_COLUMNS, path)

    df = _trim_text_columns(df)
    df = _ensure_opponent_column(df)
    df = _standardize_core_columns(df)

    df["game_date"] = _parse_game_date(df["game_date"])
    df = _derive_season_from_game_date(df)
    df = _fill_key_text_fallbacks(df)

    df = df.dropna(subset=["player_name", "team_name", "season", "game_date"]).copy()

    protected_text_columns = set(TEXT_PRIORITY_COLUMNS) | {"season", "game_date"}
    df = _convert_numeric_columns(df, protected_text_columns)

    df = df.drop_duplicates().copy()
    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    _validate_non_empty_after_clean(df, f"Cleaned game logs from {path.name}")
    return df


def _filter_montana(df: pd.DataFrame) -> pd.DataFrame:
    montana_values = {"Montana", "Montana Grizzlies", "UM", "Montana Griz"}
    montana_df = df[df["team_name"].isin(montana_values)].copy()

    if montana_df.empty:
        montana_df = df[df["team_name"].astype(str).str.contains("Montana", case=False, na=False)].copy()

    _validate_non_empty_after_clean(montana_df, "Montana game logs")
    return montana_df.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    _ = config

    game_logs_input_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/player_game_logs.csv",
        candidate_attr_names=[
            "PLAYER_GAME_LOGS_FILE",
            "PLAYER_GAME_LOGS_PATH",
            "RAW_PLAYER_GAME_LOGS_FILE",
            "RAW_PLAYER_GAME_LOGS_PATH",
        ],
    )

    failures_input_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/player_game_log_failures.csv",
        candidate_attr_names=[
            "PLAYER_GAME_LOG_FAILURES_FILE",
            "PLAYER_GAME_LOG_FAILURES_PATH",
            "RAW_PLAYER_GAME_LOG_FAILURES_FILE",
            "RAW_PLAYER_GAME_LOG_FAILURES_PATH",
        ],
    )

    montana_output_path = _resolve_output_path(
        default_relative="data/processed/player_data/player_game_logs_montana_clean.csv",
        candidate_attr_names=[
            "PLAYER_GAME_LOGS_MONTANA_CLEAN_FILE",
            "PLAYER_GAME_LOGS_MONTANA_CLEAN_PATH",
            "PROCESSED_PLAYER_GAME_LOGS_MONTANA_CLEAN_FILE",
            "PROCESSED_PLAYER_GAME_LOGS_MONTANA_CLEAN_PATH",
        ],
    )

    all_output_path = _resolve_output_path(
        default_relative="data/processed/comparison_sets/player_game_logs_clean.csv",
        candidate_attr_names=[
            "PLAYER_GAME_LOGS_CLEAN_FILE",
            "PLAYER_GAME_LOGS_CLEAN_PATH",
            "PROCESSED_PLAYER_GAME_LOGS_CLEAN_FILE",
            "PROCESSED_PLAYER_GAME_LOGS_CLEAN_PATH",
        ],
    )

    game_logs_df = _read_csv(game_logs_input_path)
    _ = _read_csv(failures_input_path, optional=True)

    cleaned_game_logs = _prepare_game_logs(game_logs_df, game_logs_input_path)
    montana_game_logs = _filter_montana(cleaned_game_logs)

    _write_csv(montana_game_logs, montana_output_path)
    _write_csv(cleaned_game_logs, all_output_path)

    print(f"Saved Montana game logs: {len(montana_game_logs)} rows -> {montana_output_path}")
    print(f"Saved full game log pool: {len(cleaned_game_logs)} rows -> {all_output_path}")


if __name__ == "__main__":
    main()