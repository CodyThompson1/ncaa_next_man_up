"""
File: score_shooting.py
Last Modified: 2026-03-02
Purpose: Calculate shooting scores using percentile rankings within peer groups.

Inputs:
- data/features/player_percentiles.csv

Outputs:
- data/features/shooting_scores.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "shooting_scores.csv"

REQUIRED_BASE_COLUMNS = ["player", "team"]
PEER_GROUP_CANDIDATES = [
    "peer_group",
    "position_group",
    "position",
    "archetype",
]

METRIC_CANDIDATES: Dict[str, List[str]] = {
    "three_point_pct": ["three_point_pct", "three_pt_pct", "fg3_pct", "3p_pct", "3pt_pct"],
    "true_shooting_pct": ["true_shooting_pct", "ts_pct", "ts"],
    "effective_field_goal_pct": ["effective_field_goal_pct", "efg_pct", "efg"],
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input player percentiles dataset is empty.")

    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def resolve_peer_group_column(df: pd.DataFrame) -> str:
    peer_group_column = find_first_existing_column(df, PEER_GROUP_CANDIDATES)
    if peer_group_column is None:
        raise ValueError(
            "No peer group column found. Expected one of: "
            f"{PEER_GROUP_CANDIDATES}"
        )
    return peer_group_column


def resolve_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}

    for canonical_name, candidates in METRIC_CANDIDATES.items():
        column = find_first_existing_column(df, candidates)
        if column is None:
            raise ValueError(
                f"Missing shooting metric for {canonical_name}. "
                f"Expected one of: {candidates}"
            )
        resolved[canonical_name] = column

    return resolved


def coerce_numeric(df: pd.DataFrame, metric_columns: Dict[str, str]) -> pd.DataFrame:
    result = df.copy()

    for column in metric_columns.values():
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def percentile_rank_within_group(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    output = pd.Series(np.nan, index=series.index, dtype="float64")

    valid = numeric.notna()
    if valid.sum() == 0:
        return output

    output.loc[valid] = numeric.loc[valid].rank(method="average", pct=True)
    return output


def add_metric_percentiles(
    df: pd.DataFrame,
    peer_group_column: str,
    metric_columns: Dict[str, str],
) -> pd.DataFrame:
    result = df.copy()

    for canonical_name, source_column in metric_columns.items():
        percentile_column = f"{canonical_name}_percentile"
        result[percentile_column] = (
            result.groupby(peer_group_column, group_keys=False)[source_column]
            .apply(percentile_rank_within_group)
            .astype("float64")
        )

    return result


def calculate_shooting_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    percentile_columns = [
        "three_point_pct_percentile",
        "true_shooting_pct_percentile",
        "effective_field_goal_pct_percentile",
    ]

    result["shooting_score"] = result[percentile_columns].mean(axis=1, skipna=True)

    return result


def finalize_output(
    df: pd.DataFrame,
    peer_group_column: str,
    metric_columns: Dict[str, str],
) -> pd.DataFrame:
    ordered_columns = [
        "player",
        "team",
        peer_group_column,
        metric_columns["three_point_pct"],
        metric_columns["true_shooting_pct"],
        metric_columns["effective_field_goal_pct"],
        "three_point_pct_percentile",
        "true_shooting_pct_percentile",
        "effective_field_goal_pct_percentile",
        "shooting_score",
    ]

    ordered_columns = [col for col in ordered_columns if col in df.columns]

    result = df[ordered_columns].copy()

    rename_map = {
        peer_group_column: "peer_group",
        metric_columns["three_point_pct"]: "three_point_pct",
        metric_columns["true_shooting_pct"]: "true_shooting_pct",
        metric_columns["effective_field_goal_pct"]: "effective_field_goal_pct",
    }

    result = result.rename(columns=rename_map)
    result = result.drop_duplicates(subset=["player", "team"], keep="first").reset_index(drop=True)

    return result


def validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Output dataset is empty.")

    required_columns = [
        "player",
        "team",
        "peer_group",
        "three_point_pct",
        "true_shooting_pct",
        "effective_field_goal_pct",
        "three_point_pct_percentile",
        "true_shooting_pct_percentile",
        "effective_field_goal_pct_percentile",
        "shooting_score",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Output missing required columns: {missing}")


def export_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df = load_dataset(INPUT_PATH)
    peer_group_column = resolve_peer_group_column(df)
    metric_columns = resolve_metric_columns(df)

    df = coerce_numeric(df, metric_columns)
    df = add_metric_percentiles(df, peer_group_column, metric_columns)
    df = calculate_shooting_score(df)
    df = finalize_output(df, peer_group_column, metric_columns)

    validate_output(df)
    export_dataset(df, OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error in score_shooting.py: {exc}", file=sys.stderr)
        raise