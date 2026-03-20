"""
File: build_position_groups.py

Build player position groups for the NCAA Next Man Up project.

Purpose:
- Create a unified player position-group dataset for Montana players and the
  Big Sky comparison pool.
- Standardize all players into only two position groups:
    * Guard
    * Forward
- Preserve the original `position_raw` field.
- Export a reusable engineered dataset for downstream peer grouping,
  archetype assignment, and evaluation scripts.

Inputs:
- data/processed/player_data/player_profile_montana.csv
- data/processed/comparison_sets/player_profile_big_sky.csv

Output:
- data/features/player_position_groups.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONTANA_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_data" / "player_profile_montana.csv"
)

BIG_SKY_PROFILE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "comparison_sets" / "player_profile_big_sky.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_position_groups.csv"
)


# =========================
# REQUIRED COLUMNS
# =========================

REQUIRED_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_raw",
]

OPTIONAL_COLUMNS = [
    "class",
    "height",
    "weight",
]

STANDARD_OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_raw",
    "class",
    "height",
    "weight",
    "position_group",
    "position_group_rule",
    "source_dataset",
]


# =========================
# VALIDATION HELPERS
# =========================

def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], path: Path) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


# =========================
# CLEANING HELPERS
# =========================

def _standardize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


# =========================
# LOADERS
# =========================

def _load_profile(path: Path, source: str) -> pd.DataFrame:
    _validate_file_exists(path)

    df = pd.read_csv(path)
    _validate_required_columns(df, REQUIRED_COLUMNS, path)

    df = _ensure_optional_columns(df)

    text_cols = ["player_name", "team_name", "position_raw", "class", "height", "weight"]
    for col in text_cols:
        df[col] = _standardize_text(df[col])

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["source_dataset"] = source

    return df


# =========================
# POSITION LOGIC
# =========================

def _normalize_position(pos: str) -> str:
    if pd.isna(pos):
        return ""

    pos = str(pos).lower().strip()
    pos = pos.replace("-", "/").replace(",", "/")
    pos = " ".join(pos.split())

    replacements = {
        "point guard": "pg",
        "shooting guard": "sg",
        "small forward": "sf",
        "power forward": "pf",
        "center": "c",
        "wing": "f",
        "big": "c",
    }

    for k, v in replacements.items():
        pos = pos.replace(k, v)

    return pos


def _classify_position_group(position_raw: str) -> tuple[str, str]:
    pos = _normalize_position(position_raw)

    # Guards
    if pos in {"pg", "sg", "g"}:
        return "Guard", "direct_guard"

    # Forwards (includes centers by design)
    if pos in {"sf", "pf", "f", "c"}:
        return "Forward", "forward_or_center"

    # Hybrids
    if "g" in pos and "f" in pos:
        return "Guard", "hybrid_gf"

    if "f" in pos or "c" in pos:
        return "Forward", "forward_contains"

    if "g" in pos:
        return "Guard", "guard_contains"

    # Final fallback (project rule: push ambiguity to Forward)
    return "Forward", "fallback_forward"


# =========================
# CORE TRANSFORM
# =========================

def _apply_position_logic(df: pd.DataFrame) -> pd.DataFrame:
    results = df["position_raw"].apply(_classify_position_group)
    df[["position_group", "position_group_rule"]] = pd.DataFrame(results.tolist(), index=df.index)
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(by=["player_name", "team_name", "season"])
        .drop_duplicates(subset=["player_name", "team_name", "season"], keep="first")
        .copy()
    )


# =========================
# VALIDATION
# =========================

def _validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Output is empty.")

    if df["position_group"].isna().any():
        raise ValueError("Null position_group values found.")

    valid = {"Guard", "Forward"}
    invalid = set(df["position_group"].unique()) - valid

    if invalid:
        raise ValueError(f"Invalid position groups found: {invalid}")


# =========================
# MAIN BUILD
# =========================

def build_player_position_groups() -> pd.DataFrame:

    montana = _load_profile(MONTANA_PROFILE_PATH, "montana")
    big_sky = _load_profile(BIG_SKY_PROFILE_PATH, "big_sky")

    df = pd.concat([montana, big_sky], ignore_index=True)

    df = _deduplicate(df)
    df = _apply_position_logic(df)

    df = df[STANDARD_OUTPUT_COLUMNS].copy()

    df = df.sort_values(
        by=["season", "team_name", "player_name"]
    ).reset_index(drop=True)

    _validate_output(df)

    return df


# =========================
# ENTRY POINT
# =========================

def main() -> None:
    df = build_player_position_groups()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(df["position_group"].value_counts())


if __name__ == "__main__":
    main()