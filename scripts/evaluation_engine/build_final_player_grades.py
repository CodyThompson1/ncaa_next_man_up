"""
File: build_final_player_grades.py
Last Modified: 2026-03-09
Purpose: Combine evaluation category scores into final player grades.

Inputs:
- data/features/shooting_scores.csv
- data/features/playmaking_scores.csv
- data/features/defense_scores.csv
- data/features/rebounding_scores.csv
- data/features/player_archetypes.csv
- data/features/player_peer_groups.csv

Outputs:
- data/outputs/player_final_evaluations.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHOOTING_PATH = PROJECT_ROOT / "data" / "features" / "shooting_scores.csv"
PLAYMAKING_PATH = PROJECT_ROOT / "data" / "features" / "playmaking_scores.csv"
DEFENSE_PATH = PROJECT_ROOT / "data" / "features" / "defense_scores.csv"
REBOUNDING_PATH = PROJECT_ROOT / "data" / "features" / "rebounding_scores.csv"
ARCHETYPES_PATH = PROJECT_ROOT / "data" / "features" / "player_archetypes.csv"
PEER_GROUPS_PATH = PROJECT_ROOT / "data" / "features" / "player_peer_groups.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "player_final_evaluations.csv"

KEY_COLUMNS = ["player", "team"]

DATASET_CONFIG = {
    "shooting": {
        "path": SHOOTING_PATH,
        "score_candidates": ["shooting_score"],
    },
    "playmaking": {
        "path": PLAYMAKING_PATH,
        "score_candidates": ["playmaking_score"],
    },
    "defense": {
        "path": DEFENSE_PATH,
        "score_candidates": ["defense_score", "defensive_score"],
    },
    "rebounding": {
        "path": REBOUNDING_PATH,
        "score_candidates": ["rebounding_score", "rebound_score"],
    },
}

ARCHETYPE_CANDIDATES = ["archetype"]
PEER_GROUP_CANDIDATES = ["peer_group"]

CATEGORY_WEIGHTS = {
    "shooting_score": 0.30,
    "playmaking_score": 0.25,
    "defense_score": 0.25,
    "rebounding_score": 0.20,
}


def load_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{dataset_name} dataset is empty: {path}")

    missing = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} dataset missing required keys: {missing}")

    return df


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in KEY_COLUMNS:
        result[column] = result[column].astype(str).str.strip()

    return result


def prepare_score_dataset(dataset_name: str, config: Dict[str, object]) -> pd.DataFrame:
    df = load_csv(config["path"], dataset_name)
    df = standardize_key_columns(df)

    score_column = find_first_existing_column(df, config["score_candidates"])
    if score_column is None:
        raise ValueError(
            f"{dataset_name} dataset missing score column. "
            f"Expected one of: {config['score_candidates']}"
        )

    result = df[KEY_COLUMNS + [score_column]].copy()
    result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
    result = result.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)

    standardized_score_name = f"{dataset_name}_score"
    result = result.rename(columns={score_column: standardized_score_name})

    return result


def prepare_archetypes_dataset(path: Path) -> pd.DataFrame:
    df = load_csv(path, "archetypes")
    df = standardize_key_columns(df)

    archetype_column = find_first_existing_column(df, ARCHETYPE_CANDIDATES)
    if archetype_column is None:
        raise ValueError(
            f"archetypes dataset missing archetype column. Expected one of: {ARCHETYPE_CANDIDATES}"
        )

    result = df[KEY_COLUMNS + [archetype_column]].copy()
    result[archetype_column] = result[archetype_column].astype(str).str.strip()
    result = result.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    result = result.rename(columns={archetype_column: "archetype"})

    return result


def prepare_peer_groups_dataset(path: Path) -> pd.DataFrame:
    df = load_csv(path, "peer_groups")
    df = standardize_key_columns(df)

    peer_group_column = find_first_existing_column(df, PEER_GROUP_CANDIDATES)
    if peer_group_column is None:
        raise ValueError(
            f"peer_groups dataset missing peer_group column. Expected one of: {PEER_GROUP_CANDIDATES}"
        )

    result = df[KEY_COLUMNS + [peer_group_column]].copy()
    result[peer_group_column] = result[peer_group_column].astype(str).str.strip()
    result = result.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    result = result.rename(columns={peer_group_column: "peer_group"})

    return result


def build_base_player_frame(score_datasets: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df[KEY_COLUMNS].copy() for df in score_datasets]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)

    if combined.empty:
        raise ValueError("No players found across score datasets.")

    return combined


def merge_datasets(
    base_df: pd.DataFrame,
    score_datasets: List[pd.DataFrame],
    archetypes_df: pd.DataFrame,
    peer_groups_df: pd.DataFrame,
) -> pd.DataFrame:
    result = base_df.copy()

    for dataset in score_datasets:
        result = result.merge(dataset, on=KEY_COLUMNS, how="left")

    result = result.merge(archetypes_df, on=KEY_COLUMNS, how="left")
    result = result.merge(peer_groups_df, on=KEY_COLUMNS, how="left")

    return result


def compute_weighted_grade(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    score_columns = list(CATEGORY_WEIGHTS.keys())

    for column in score_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    weight_series = pd.Series(CATEGORY_WEIGHTS, dtype="float64")

    def weighted_average(row: pd.Series) -> float:
        available_scores = row[score_columns].dropna()
        if available_scores.empty:
            return np.nan

        available_weights = weight_series.loc[available_scores.index]
        weight_sum = available_weights.sum()

        if weight_sum == 0:
            return np.nan

        return float((available_scores * available_weights).sum() / weight_sum)

    result["final_player_grade"] = result.apply(weighted_average, axis=1)

    return result


def add_grade_components(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    score_columns = list(CATEGORY_WEIGHTS.keys())
    result["available_score_count"] = result[score_columns].notna().sum(axis=1)
    result["grade_completeness"] = result["available_score_count"] / len(score_columns)

    return result


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "player",
        "team",
        "peer_group",
        "archetype",
        "shooting_score",
        "playmaking_score",
        "defense_score",
        "rebounding_score",
        "available_score_count",
        "grade_completeness",
        "final_player_grade",
    ]

    for column in ordered_columns:
        if column not in df.columns:
            df[column] = np.nan

    result = df[ordered_columns].copy()
    result = result.sort_values(
        by=["final_player_grade", "player", "team"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return result


def validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Final evaluation output is empty.")

    required_columns = [
        "player",
        "team",
        "peer_group",
        "archetype",
        "shooting_score",
        "playmaking_score",
        "defense_score",
        "rebounding_score",
        "final_player_grade",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Output missing required columns: {missing}")

    if df[KEY_COLUMNS].duplicated().any():
        duplicates = df.loc[df[KEY_COLUMNS].duplicated(), KEY_COLUMNS]
        raise ValueError(f"Duplicate player/team rows found in output: {duplicates.to_dict('records')}")


def export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    score_datasets = [
        prepare_score_dataset(dataset_name, config)
        for dataset_name, config in DATASET_CONFIG.items()
    ]

    archetypes_df = prepare_archetypes_dataset(ARCHETYPES_PATH)
    peer_groups_df = prepare_peer_groups_dataset(PEER_GROUPS_PATH)

    base_df = build_base_player_frame(score_datasets)
    merged_df = merge_datasets(base_df, score_datasets, archetypes_df, peer_groups_df)
    graded_df = compute_weighted_grade(merged_df)
    graded_df = add_grade_components(graded_df)
    output_df = finalize_output(graded_df)

    validate_output(output_df)
    export_csv(output_df, OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error in build_final_player_grades.py: {exc}", file=sys.stderr)
        raise