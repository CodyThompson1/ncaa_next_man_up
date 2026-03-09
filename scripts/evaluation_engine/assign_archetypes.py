"""
File: assign_archetypes.py
Last Modified: 2026-03-01
Purpose: Assign role archetypes to players using rule-based classification logic.

Inputs:
- data/processed/player_stats.csv

Outputs:
- data/features/player_archetypes.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "player_stats.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "player_archetypes.csv"

POSITION_GROUPS = {"Guard", "Wing", "Big"}

REQUIRED_COLUMNS = [
    "player",
    "team",
    "position_group",
]

OPTIONAL_METRIC_CANDIDATES = {
    "usage_rate": ["usage_rate", "usg_pct", "usage", "usg"],
    "assist_rate": ["assist_rate", "ast_rate", "assist_pct", "ast_pct"],
    "assist_to_turnover_ratio": [
        "assist_to_turnover_ratio",
        "ast_to_tov_ratio",
        "ast_tov_ratio",
    ],
    "turnover_rate": ["turnover_rate", "tov_rate", "turnover_pct", "tov_pct"],
    "three_point_attempt_rate": [
        "three_point_attempt_rate",
        "three_pa_rate",
        "3pa_rate",
        "three_point_rate",
    ],
    "three_point_pct": ["three_point_pct", "three_pt_pct", "fg3_pct", "3p_pct"],
    "true_shooting_pct": ["true_shooting_pct", "ts_pct", "ts"],
    "points_per_game": ["points_per_game", "ppg", "points"],
    "steal_rate": ["steal_rate", "stl_rate", "steal_pct", "stl_pct"],
    "block_rate": ["block_rate", "blk_rate", "block_pct", "blk_pct"],
    "defensive_rebound_rate": [
        "defensive_rebound_rate",
        "dreb_rate",
        "def_reb_rate",
        "defensive_rebound_pct",
    ],
    "offensive_rebound_rate": [
        "offensive_rebound_rate",
        "oreb_rate",
        "off_reb_rate",
        "offensive_rebound_pct",
    ],
    "rebound_rate": ["rebound_rate", "trb_rate", "reb_rate", "total_rebound_pct"],
    "free_throw_rate": ["free_throw_rate", "ftr", "ft_rate", "fta_per_fga"],
}


def load_player_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input player stats dataset is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def find_metric_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_metric_map(df: pd.DataFrame) -> Dict[str, str | None]:
    metric_map: Dict[str, str | None] = {}
    for canonical_name, candidates in OPTIONAL_METRIC_CANDIDATES.items():
        metric_map[canonical_name] = find_metric_column(df, candidates)
    return metric_map


def coerce_numeric_columns(df: pd.DataFrame, metric_map: Dict[str, str | None]) -> pd.DataFrame:
    result = df.copy()

    for _, column in metric_map.items():
        if column and column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def normalize_position_group(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    if text in {"guard", "g", "pg", "sg", "combo guard"}:
        return "Guard"
    if text in {"wing", "w", "sf", "gf", "f", "forward", "swingman"}:
        return "Wing"
    if text in {"big", "b", "pf", "c", "fc", "post", "center"}:
        return "Big"

    return str(value).strip().title()


def prepare_dataframe(df: pd.DataFrame, metric_map: Dict[str, str | None]) -> pd.DataFrame:
    result = df.copy()
    result["position_group"] = result["position_group"].apply(normalize_position_group)

    result = result[result["position_group"].isin(POSITION_GROUPS)].copy()

    if result.empty:
        raise ValueError("No valid player rows found after normalizing position_group.")

    result = result.drop_duplicates(subset=["player", "team"], keep="first").reset_index(drop=True)

    for canonical_name, source_col in metric_map.items():
        if source_col and source_col in result.columns:
            result[canonical_name] = result[source_col]
        else:
            result[canonical_name] = np.nan

    return result


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()

    if valid.sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    ranks = numeric[valid].rank(pct=True, method="average")
    output = pd.Series(np.nan, index=series.index, dtype="float64")
    output.loc[valid] = ranks
    return output


def add_group_percentiles(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    result = df.copy()

    for metric in metrics:
        pct_col = f"{metric}_pctile"
        result[pct_col] = (
            result.groupby("position_group", group_keys=False)[metric]
            .apply(percentile_rank)
            .astype("float64")
        )

    return result


def safe_value(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    if pd.isna(value):
        return 0.0
    return float(value)


def score_guard(row: pd.Series) -> Tuple[str, Dict[str, float]]:
    scores = {
        "Primary Creator": 0.0,
        "Secondary Playmaker": 0.0,
        "Off Ball Shooter": 0.0,
    }

    scores["Primary Creator"] = (
        0.30 * safe_value(row, "usage_rate_pctile")
        + 0.30 * safe_value(row, "assist_rate_pctile")
        + 0.20 * safe_value(row, "assist_to_turnover_ratio_pctile")
        + 0.10 * (1 - safe_value(row, "turnover_rate_pctile"))
        + 0.10 * safe_value(row, "points_per_game_pctile")
    )

    scores["Secondary Playmaker"] = (
        0.30 * safe_value(row, "assist_rate_pctile")
        + 0.25 * safe_value(row, "assist_to_turnover_ratio_pctile")
        + 0.20 * (1 - safe_value(row, "turnover_rate_pctile"))
        + 0.15 * safe_value(row, "true_shooting_pct_pctile")
        + 0.10 * safe_value(row, "usage_rate_pctile")
    )

    scores["Off Ball Shooter"] = (
        0.35 * safe_value(row, "three_point_attempt_rate_pctile")
        + 0.35 * safe_value(row, "three_point_pct_pctile")
        + 0.15 * safe_value(row, "true_shooting_pct_pctile")
        + 0.10 * (1 - safe_value(row, "turnover_rate_pctile"))
        + 0.05 * (1 - safe_value(row, "assist_rate_pctile"))
    )

    archetype = max(scores, key=scores.get)
    return archetype, scores


def score_wing(row: pd.Series) -> Tuple[str, Dict[str, float]]:
    scores = {
        "Scoring Wing": 0.0,
        "3 and D Wing": 0.0,
        "Glue Wing": 0.0,
    }

    scores["Scoring Wing"] = (
        0.30 * safe_value(row, "usage_rate_pctile")
        + 0.25 * safe_value(row, "points_per_game_pctile")
        + 0.20 * safe_value(row, "true_shooting_pct_pctile")
        + 0.15 * safe_value(row, "free_throw_rate_pctile")
        + 0.10 * safe_value(row, "three_point_pct_pctile")
    )

    scores["3 and D Wing"] = (
        0.25 * safe_value(row, "three_point_attempt_rate_pctile")
        + 0.25 * safe_value(row, "three_point_pct_pctile")
        + 0.20 * safe_value(row, "steal_rate_pctile")
        + 0.20 * safe_value(row, "block_rate_pctile")
        + 0.10 * safe_value(row, "true_shooting_pct_pctile")
    )

    scores["Glue Wing"] = (
        0.20 * safe_value(row, "assist_rate_pctile")
        + 0.20 * safe_value(row, "rebound_rate_pctile")
        + 0.20 * safe_value(row, "steal_rate_pctile")
        + 0.20 * (1 - safe_value(row, "turnover_rate_pctile"))
        + 0.20 * safe_value(row, "true_shooting_pct_pctile")
    )

    archetype = max(scores, key=scores.get)
    return archetype, scores


def score_big(row: pd.Series) -> Tuple[str, Dict[str, float]]:
    scores = {
        "Interior Big": 0.0,
        "Stretch Big": 0.0,
        "Rebounding Big": 0.0,
        "Defensive Big": 0.0,
    }

    scores["Interior Big"] = (
        0.30 * safe_value(row, "free_throw_rate_pctile")
        + 0.25 * safe_value(row, "true_shooting_pct_pctile")
        + 0.20 * safe_value(row, "offensive_rebound_rate_pctile")
        + 0.15 * safe_value(row, "usage_rate_pctile")
        + 0.10 * safe_value(row, "points_per_game_pctile")
    )

    scores["Stretch Big"] = (
        0.40 * safe_value(row, "three_point_attempt_rate_pctile")
        + 0.30 * safe_value(row, "three_point_pct_pctile")
        + 0.20 * safe_value(row, "true_shooting_pct_pctile")
        + 0.10 * safe_value(row, "usage_rate_pctile")
    )

    scores["Rebounding Big"] = (
        0.45 * safe_value(row, "rebound_rate_pctile")
        + 0.25 * safe_value(row, "offensive_rebound_rate_pctile")
        + 0.20 * safe_value(row, "defensive_rebound_rate_pctile")
        + 0.10 * safe_value(row, "block_rate_pctile")
    )

    scores["Defensive Big"] = (
        0.40 * safe_value(row, "block_rate_pctile")
        + 0.25 * safe_value(row, "defensive_rebound_rate_pctile")
        + 0.20 * safe_value(row, "steal_rate_pctile")
        + 0.15 * safe_value(row, "rebound_rate_pctile")
    )

    archetype = max(scores, key=scores.get)
    return archetype, scores


def assign_archetype(row: pd.Series) -> Tuple[str, Dict[str, float]]:
    position_group = row["position_group"]

    if position_group == "Guard":
        return score_guard(row)
    if position_group == "Wing":
        return score_wing(row)
    if position_group == "Big":
        return score_big(row)

    return "Unassigned", {}


def apply_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    archetypes: List[str] = []
    archetype_scores: List[Dict[str, float]] = []

    for _, row in result.iterrows():
        archetype, scores = assign_archetype(row)
        archetypes.append(archetype)
        archetype_scores.append(scores)

    result["archetype"] = archetypes

    all_score_columns = sorted({key for score_map in archetype_scores for key in score_map.keys()})
    for column in all_score_columns:
        result[f"{column.lower().replace(' ', '_').replace('-', '_')}_score"] = [
            score_map.get(column, np.nan) for score_map in archetype_scores
        ]

    return result


def validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Archetype output is empty.")

    required_output_columns = ["player", "team", "position_group", "archetype"]
    missing = [col for col in required_output_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Output missing required columns: {missing}")

    if df["archetype"].isna().any():
        raise ValueError("One or more players do not have an assigned archetype.")


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "player",
        "team",
        "position_group",
        "archetype",
    ]

    passthrough_metrics = [
        "usage_rate",
        "assist_rate",
        "assist_to_turnover_ratio",
        "turnover_rate",
        "three_point_attempt_rate",
        "three_point_pct",
        "true_shooting_pct",
        "points_per_game",
        "steal_rate",
        "block_rate",
        "defensive_rebound_rate",
        "offensive_rebound_rate",
        "rebound_rate",
        "free_throw_rate",
    ]

    percentile_columns = [f"{metric}_pctile" for metric in passthrough_metrics]
    score_columns = [
        col for col in df.columns if col.endswith("_score")
    ]

    output_columns = preferred_columns + passthrough_metrics + percentile_columns + sorted(score_columns)
    output_columns = [col for col in output_columns if col in df.columns]

    return df[output_columns].copy()


def export_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = load_player_stats(INPUT_PATH)
    metric_map = build_metric_map(df)
    df = coerce_numeric_columns(df, metric_map)
    df = prepare_dataframe(df, metric_map)

    metrics_for_percentiles = [
        "usage_rate",
        "assist_rate",
        "assist_to_turnover_ratio",
        "turnover_rate",
        "three_point_attempt_rate",
        "three_point_pct",
        "true_shooting_pct",
        "points_per_game",
        "steal_rate",
        "block_rate",
        "defensive_rebound_rate",
        "offensive_rebound_rate",
        "rebound_rate",
        "free_throw_rate",
    ]

    df = add_group_percentiles(df, metrics_for_percentiles)
    df = apply_archetypes(df)
    df = select_output_columns(df)
    validate_output(df)
    export_dataframe(df, OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error in assign_archetypes.py: {exc}", file=sys.stderr)
        raise