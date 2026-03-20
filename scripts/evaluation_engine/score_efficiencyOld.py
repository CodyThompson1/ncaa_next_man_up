"""
Build efficiency category scores for the NCAA Next Man Up evaluation engine.

Purpose
-------
This script calculates player-level Efficiency scores using percentile-based
components from the engineered player percentile dataset. The output is designed
to feed later evaluation and dashboard layers.

Primary input
-------------
- data/features/player_percentiles.csv

Supporting input
----------------
- data/features/player_archetype_assignment.csv

Output
------
- data/features/efficiency_scores.csv

Scoring logic
-------------
Efficiency is intended to reflect scoring efficiency and possession-level impact.
Primary metrics:
- ortg
- ts_pct
- efg_pct
- turnover_rate (inverse metric, if available)

The script prefers existing percentile columns where available. If a percentile
column is missing but a raw metric is present, the script computes percentiles
directly from the input data.

Notes
-----
- Higher is better for ortg, ts_pct, and efg_pct.
- Lower is better for turnover_rate, so its percentile is inverted.
- Weights are transparent and easy to adjust in METRIC_CONFIG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
ARCHETYPE_ASSIGNMENT_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "efficiency_scores.csv"

REQUIRED_ID_COLUMNS = ["player_name", "team_name", "season"]
OPTIONAL_ID_COLUMNS = ["conference_name", "position_group"]


@dataclass(frozen=True)
class MetricRule:
    """Configuration for a scoring metric."""

    metric_name: str
    weight: float
    higher_is_better: bool = True


METRIC_CONFIG: list[MetricRule] = [
    MetricRule(metric_name="ortg", weight=0.35, higher_is_better=True),
    MetricRule(metric_name="ts_pct", weight=0.30, higher_is_better=True),
    MetricRule(metric_name="efg_pct", weight=0.25, higher_is_better=True),
    MetricRule(metric_name="turnover_rate", weight=0.10, higher_is_better=False),
]


def _validate_file_exists(path: Path) -> None:
    """Raise an error if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with basic empty-file validation."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    df_name: str,
) -> None:
    """Validate that all required columns are present."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_columns}"
        )


def _standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common merge keys to improve merge reliability while preserving
    the original displayed values as much as possible.
    """
    standardized_df = df.copy()

    if "player_name" in standardized_df.columns:
        standardized_df["player_name"] = (
            standardized_df["player_name"].astype(str).str.strip()
        )

    if "team_name" in standardized_df.columns:
        standardized_df["team_name"] = (
            standardized_df["team_name"].astype(str).str.strip()
        )

    if "season" in standardized_df.columns:
        standardized_df["season"] = pd.to_numeric(
            standardized_df["season"], errors="coerce"
        ).astype("Int64")

    return standardized_df


def _drop_duplicate_players(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """
    Drop duplicate player rows by player/team/season and keep the first record.
    """
    before_count = len(df)
    deduped_df = df.drop_duplicates(subset=REQUIRED_ID_COLUMNS).copy()
    after_count = len(deduped_df)

    if after_count < before_count:
        print(
            f"{df_name}: dropped {before_count - after_count} duplicate rows "
            f"using keys {REQUIRED_ID_COLUMNS}."
        )

    return deduped_df


def _safe_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Convert a raw metric series to a 0-100 percentile scale.

    Missing values remain missing. Ties use average rank. If lower is better,
    the percentile is inverted.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    percentile = numeric_series.rank(pct=True, method="average") * 100.0

    if not higher_is_better:
        percentile = 100.0 - percentile

    return percentile.round(2)


def _candidate_percentile_columns(metric_name: str) -> list[str]:
    """
    Return likely percentile column names for a metric.

    This makes the script more robust to naming differences across upstream files.
    """
    return [
        f"{metric_name}_percentile",
        f"{metric_name}_pctile",
        f"{metric_name}_percent_rank",
        f"{metric_name}_percent_rank_value",
        f"{metric_name}_pr",
    ]


def _find_existing_percentile_column(
    df: pd.DataFrame,
    metric_name: str,
) -> Optional[str]:
    """Find the first matching percentile column for a given metric."""
    candidates = _candidate_percentile_columns(metric_name)
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _resolve_metric_component(
    df: pd.DataFrame,
    metric_rule: MetricRule,
) -> pd.DataFrame:
    """
    Resolve a metric component into raw metric, percentile, and weighted score.

    The function prefers an existing percentile column. If one is not present,
    it computes percentiles from the raw metric column.
    """
    metric_name = metric_rule.metric_name
    percentile_column = _find_existing_percentile_column(df, metric_name)
    raw_column = metric_name if metric_name in df.columns else None

    working_df = df.copy()

    if percentile_column is not None:
        working_df[f"{metric_name}_component_percentile"] = pd.to_numeric(
            working_df[percentile_column], errors="coerce"
        ).round(2)
    elif raw_column is not None:
        working_df[f"{metric_name}_component_percentile"] = _safe_percentile(
            working_df[raw_column],
            higher_is_better=metric_rule.higher_is_better,
        )
    else:
        working_df[f"{metric_name}_component_percentile"] = np.nan

    if raw_column is not None:
        working_df[f"{metric_name}_raw"] = pd.to_numeric(
            working_df[raw_column], errors="coerce"
        )
    else:
        working_df[f"{metric_name}_raw"] = np.nan

    working_df[f"{metric_name}_weight"] = metric_rule.weight
    working_df[f"{metric_name}_weighted_score"] = (
        working_df[f"{metric_name}_component_percentile"] * metric_rule.weight
    ).round(2)

    return working_df


def _attach_archetypes(
    percentiles_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge player archetype assignments onto the percentile base table.
    """
    merge_columns = [col for col in REQUIRED_ID_COLUMNS if col in archetypes_df.columns]
    extra_columns = [
        col
        for col in ["position_group", "archetype"]
        if col in archetypes_df.columns and col not in merge_columns
    ]

    archetype_subset = archetypes_df[merge_columns + extra_columns].copy()
    merged_df = percentiles_df.merge(
        archetype_subset,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )

    return merged_df


def _ensure_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure position_group exists. If both datasets supplied it, preserve an
    archetype-linked version if needed.
    """
    working_df = df.copy()

    if "position_group" not in working_df.columns:
        working_df["position_group"] = np.nan

    return working_df


def _build_efficiency_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build weighted efficiency component scores and an overall Efficiency score.
    """
    working_df = df.copy()

    for metric_rule in METRIC_CONFIG:
        working_df = _resolve_metric_component(working_df, metric_rule)

    component_percentile_columns = [
        f"{metric.metric_name}_component_percentile" for metric in METRIC_CONFIG
    ]
    weighted_score_columns = [
        f"{metric.metric_name}_weighted_score" for metric in METRIC_CONFIG
    ]

    available_weight_sum = np.zeros(len(working_df), dtype=float)

    for metric in METRIC_CONFIG:
        component_col = f"{metric.metric_name}_component_percentile"
        valid_mask = working_df[component_col].notna().to_numpy(dtype=bool)
        available_weight_sum = available_weight_sum + (
            valid_mask.astype(float) * metric.weight
        )

    weighted_sum = (
        working_df[weighted_score_columns]
        .sum(axis=1, min_count=1)
        .astype(float)
    )

    normalized_score = np.where(
        available_weight_sum > 0,
        weighted_sum / available_weight_sum,
        np.nan,
    )

    working_df["efficiency_score"] = np.round(normalized_score, 2)
    working_df["efficiency_metrics_used"] = working_df[component_percentile_columns].notna().sum(axis=1)
    working_df["efficiency_weight_available"] = np.round(available_weight_sum, 2)

    return working_df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order the final output columns.
    """
    ordered_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "archetype",
        "efficiency_score",
        "efficiency_metrics_used",
        "efficiency_weight_available",
    ]

    for metric in METRIC_CONFIG:
        ordered_columns.extend(
            [
                f"{metric.metric_name}_raw",
                f"{metric.metric_name}_component_percentile",
                f"{metric.metric_name}_weight",
                f"{metric.metric_name}_weighted_score",
            ]
        )

    existing_columns = [column for column in ordered_columns if column in df.columns]
    output_df = df[existing_columns].copy()

    output_df = output_df.sort_values(
        by=["season", "team_name", "player_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return output_df


def _validate_output(df: pd.DataFrame) -> None:
    """
    Validate the final output before writing it to disk.
    """
    _validate_required_columns(
        df=df,
        required_columns=[
            "player_name",
            "team_name",
            "season",
            "efficiency_score",
        ],
        df_name="efficiency_scores",
    )

    if df.empty:
        raise ValueError("Efficiency score output is empty.")

    if df["efficiency_score"].notna().sum() == 0:
        raise ValueError("No efficiency scores were computed.")

    duplicated_keys = df.duplicated(subset=REQUIRED_ID_COLUMNS, keep=False)
    if duplicated_keys.any():
        duplicate_rows = df.loc[duplicated_keys, REQUIRED_ID_COLUMNS]
        raise ValueError(
            "Duplicate player-team-season rows detected in output:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def build_efficiency_scores() -> pd.DataFrame:
    """
    Main pipeline for creating Efficiency scores.
    """
    _validate_file_exists(PLAYER_PERCENTILES_PATH)
    _validate_file_exists(ARCHETYPE_ASSIGNMENT_PATH)

    percentiles_df = _read_csv(PLAYER_PERCENTILES_PATH)
    archetypes_df = _read_csv(ARCHETYPE_ASSIGNMENT_PATH)

    _validate_required_columns(
        df=percentiles_df,
        required_columns=REQUIRED_ID_COLUMNS,
        df_name="player_percentiles",
    )
    _validate_required_columns(
        df=archetypes_df,
        required_columns=REQUIRED_ID_COLUMNS,
        df_name="player_archetype_assignment",
    )

    percentiles_df = _standardize_key_columns(percentiles_df)
    archetypes_df = _standardize_key_columns(archetypes_df)

    percentiles_df = _drop_duplicate_players(percentiles_df, "player_percentiles")
    archetypes_df = _drop_duplicate_players(
        archetypes_df, "player_archetype_assignment"
    )

    merged_df = _attach_archetypes(percentiles_df, archetypes_df)
    merged_df = _ensure_position_group(merged_df)

    for optional_column in OPTIONAL_ID_COLUMNS:
        if optional_column not in merged_df.columns:
            merged_df[optional_column] = np.nan

    scored_df = _build_efficiency_scores(merged_df)
    output_df = _select_output_columns(scored_df)

    _validate_output(output_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Efficiency scores saved: {len(output_df):,} rows")
    print(f"Output written to: {OUTPUT_PATH}")

    return output_df


def main() -> None:
    """Run the efficiency scoring pipeline."""
    build_efficiency_scores()


if __name__ == "__main__":
    main()