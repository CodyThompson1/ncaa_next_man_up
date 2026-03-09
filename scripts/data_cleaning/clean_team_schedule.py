"""
File: clean_team_schedule.py
Last Modified: 2026-02-19
Purpose: Refine and standardize the processed Montana team schedule into a final project-ready schedule dataset for downstream dashboard use and later competition-stability analysis in the NCAA Next Man Up project.

Inputs:
- data/processed/um_schedule_processed.csv
- data/raw/sports_reference/um_schedule_raw.csv

Outputs:
- data/processed/team_data/team_schedule.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from scripts.utilities import config, file_paths, name_standardization


REQUIRED_COLUMNS = [
    "team_name",
    "opponent",
    "season",
]

TEXT_PRIORITY_COLUMNS = [
    "team_name",
    "opponent",
    "conference_name",
    "game_type",
    "location",
    "site",
    "result",
    "home_away",
    "overtime",
    "team_rank",
    "opponent_rank",
]

KEY_FALLBACK_COLUMNS = [
    "team_name",
    "season",
    "game_date",
    "opponent",
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

    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].map(_clean_text)
        df["team_name"] = _apply_standardizer(df["team_name"], STANDARDIZE_TEAM)

    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(_clean_text)
        df["opponent"] = _apply_standardizer(df["opponent"], STANDARDIZE_OPPONENT)

    if "conference_name" in df.columns:
        df["conference_name"] = df["conference_name"].map(_clean_text)
        df["conference_name"] = _apply_standardizer(df["conference_name"], STANDARDIZE_CONFERENCE)

    if "season" in df.columns:
        df["season"] = df["season"].map(_clean_season)

    return df


def _standardize_game_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_candidates = ["game_date", "date", "game_day"]
    date_col = next((col for col in date_candidates if col in df.columns), None)

    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        df["game_date"] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)
        if date_col != "game_date":
            df = df.drop(columns=[date_col])

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


def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_present = [col for col in REQUIRED_COLUMNS if col in df.columns]
    if required_present:
        df = df.dropna(subset=required_present).copy()

    if "game_date" in df.columns:
        df = df[df["game_date"].notna()].copy()

    return df


def _sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_front = [
        "team_name",
        "season",
        "game_date",
        "opponent",
        "conference_name",
        "location",
        "site",
        "home_away",
        "result",
        "overtime",
        "game_type",
    ]
    front_cols = [col for col in preferred_front if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in front_cols]
    return df[front_cols + remaining_cols].copy()


def _dedupe_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dedupe_candidates = [
        ["team_name", "season", "game_date", "opponent"],
        ["team_name", "season", "opponent"],
    ]

    for subset in dedupe_candidates:
        if all(col in df.columns for col in subset):
            df = df.drop_duplicates(subset=subset, keep="first").copy()
            return df

    return df.drop_duplicates().copy()


def _prepare_schedule_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = _drop_unnamed_columns(df)
    _validate_columns(df, REQUIRED_COLUMNS, path)

    df = _trim_text_columns(df)
    df = _standardize_core_columns(df)
    df = _standardize_game_date(df)
    df = _drop_empty_rows(df)

    protected_text_columns = set(TEXT_PRIORITY_COLUMNS) | {"season", "game_date"}
    df = _convert_numeric_columns(df, protected_text_columns)

    df = _dedupe_schedule(df)

    sort_cols = [col for col in KEY_FALLBACK_COLUMNS if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Cleaned schedule is empty after processing: {path}")

    return _sort_columns(df)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    _ = config

    processed_input_path = _resolve_input_path(
        default_relative="data/processed/um_schedule_processed.csv",
        candidate_attr_names=[
            "UM_SCHEDULE_PROCESSED_FILE",
            "UM_SCHEDULE_PROCESSED_PATH",
            "PROCESSED_UM_SCHEDULE_FILE",
            "PROCESSED_UM_SCHEDULE_PATH",
        ],
    )

    raw_reference_path = _resolve_input_path(
        default_relative="data/raw/sports_reference/um_schedule_raw.csv",
        candidate_attr_names=[
            "UM_SCHEDULE_RAW_FILE",
            "UM_SCHEDULE_RAW_PATH",
            "RAW_UM_SCHEDULE_FILE",
            "RAW_UM_SCHEDULE_PATH",
        ],
    )

    output_path = _resolve_output_path(
        default_relative="data/processed/team_data/team_schedule.csv",
        candidate_attr_names=[
            "TEAM_SCHEDULE_FILE",
            "TEAM_SCHEDULE_PATH",
            "PROCESSED_TEAM_SCHEDULE_FILE",
            "PROCESSED_TEAM_SCHEDULE_PATH",
        ],
    )

    schedule_df = _read_csv(processed_input_path)
    _ = _read_csv(raw_reference_path, optional=True)

    cleaned_schedule = _prepare_schedule_df(schedule_df, processed_input_path)
    _write_csv(cleaned_schedule, output_path)

    print(f"Saved cleaned team schedule: {len(cleaned_schedule)} rows -> {output_path}")


if __name__ == "__main__":
    main()