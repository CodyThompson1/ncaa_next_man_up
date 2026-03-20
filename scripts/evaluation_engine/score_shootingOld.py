"""
Build shooting evaluation scores for NCAA Next Man Up.

Purpose
-------
This script calculates a percentile-based Shooting score for each player using
player peer-group percentile outputs and archetype assignments.

Primary inputs
--------------
- data/features/player_percentiles.csv
- data/features/player_archetype_assignment.csv

Output
------
- data/features/shooting_scores.csv

Scoring logic
-------------
Shooting is intended to capture perimeter shooting efficiency and overall scoring
efficiency within each player's peer context.

Default weighted metrics:
- 3P% percentile
- 3P attempt rate percentile
- eFG% percentile
- TS% percentile

The script is written to be production-ready and resilient to minor schema
variation by attempting to detect likely percentile column names from a list of
accepted aliases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
PLAYER_ARCHETYPE_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "shooting_scores.csv"


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

KEY_COLUMNS = ["player_name", "team_name", "season"]
OPTIONAL_ID_COLUMNS = ["position_group"]
REQUIRED_OUTPUT_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_group",
    "archetype",
    "shooting_score",
]

# Transparent, easy-to-adjust metric weights.
METRIC_WEIGHTS: Dict[str, float] = {
    "three_point_pct": 0.35,
    "three_point_attempt_rate": 0.20,
    "effective_field_goal_pct": 0.25,
    "true_shooting_pct": 0.20,
}

# Candidate percentile columns by conceptual metric.
# The script will use the first matching column found.
PERCENTILE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "three_point_pct": [
        "three_point_pct_percentile",
        "three_point_percentage_percentile",
        "three_pt_pct_percentile",
        "three_pt_percentage_percentile",
        "3p_pct_percentile",
        "3pt_pct_percentile",
        "3pt_percentile",
        "3p_percentile",
        "three_point_pct_pctile",
        "three_point_pct_pr",
        "three_point_pct_perc",
    ],
    "three_point_attempt_rate": [
        "three_point_attempt_rate_percentile",
        "three_point_rate_percentile",
        "three_par_percentile",
        "3par_percentile",
        "three_point_attempt_share_percentile",
        "three_point_attempt_pct_percentile",
        "3pa_rate_percentile",
        "3pt_attempt_rate_percentile",
        "3pt_rate_percentile",
        "three_par_pr",
        "three_par_perc",
    ],
    "effective_field_goal_pct": [
        "effective_field_goal_pct_percentile",
        "efg_pct_percentile",
        "efg_percentile",
        "effective_fg_pct_percentile",
        "effective_fg_percentile",
        "efg_pct_pr",
        "efg_pct_perc",
    ],
    "true_shooting_pct": [
        "true_shooting_pct_percentile",
        "ts_pct_percentile",
        "ts_percentile",
        "true_shooting_percentile",
        "ts_pct_pr",
        "ts_pct_perc",
    ],
}

# Optional raw metric columns to preserve if available for presentation/debugging.
RAW_METRIC_ALIASES: Dict[str, List[str]] = {
    "three_point_pct_raw": [
        "3p_pct",
        "3pt_pct",
        "three_point_pct",
        "three_pt_pct",
        "three_point_percentage",
    ],
    "three_point_attempt_rate_raw": [
        "3par",
        "three_par",
        "three_point_attempt_rate",
        "three_point_rate",
        "3pa_rate",
        "three_point_attempt_share",
    ],
    "effective_field_goal_pct_raw": [
        "efg_pct",
        "effective_field_goal_pct",
        "effective_fg_pct",
    ],
    "true_shooting_pct_raw": [
        "ts_pct",
        "true_shooting_pct",
        "true_shooting_percentage",
    ],
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _validate_file_exists(path: Path) -> None:
    """Raise a clear error if an expected input file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with basic validation."""
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _standardize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Strip whitespace from standard identifier columns.

    This keeps merge keys stable without aggressively changing case or content.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    df_name: str,
) -> None:
    """Ensure a DataFrame contains all required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {missing_columns}"
        )


def _find_first_matching_column(
    df: pd.DataFrame,
    candidate_columns: List[str],
) -> Optional[str]:
    """Return the first candidate column that exists in the DataFrame."""
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _normalize_percentile_series(series: pd.Series) -> pd.Series:
    """
    Normalize percentile values to a 0-100 scale if needed.

    Supported cases:
    - already 0-100
    - 0-1 decimals
    """
    cleaned = pd.to_numeric(series, errors="coerce")

    if cleaned.dropna().empty:
        return cleaned

    series_max = cleaned.max(skipna=True)
    series_min = cleaned.min(skipna=True)

    if series_max <= 1.0 and series_min >= 0.0:
        return cleaned * 100.0

    return cleaned


def _resolve_percentile_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Resolve conceptual shooting metric names to actual percentile column names.

    Raises
    ------
    ValueError
        If any required percentile metric cannot be found.
    """
    resolved_columns: Dict[str, str] = {}

    for metric_name, aliases in PERCENTILE_COLUMN_ALIASES.items():
        matched_column = _find_first_matching_column(df, aliases)
        if matched_column is None:
            raise ValueError(
                f"Could not find a percentile column for metric '{metric_name}'. "
                f"Tried aliases: {aliases}"
            )
        resolved_columns[metric_name] = matched_column

    return resolved_columns


def _attach_optional_raw_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach optional raw metric columns if present.

    This makes the output more explainable and presentation-ready.
    """
    for output_name, aliases in RAW_METRIC_ALIASES.items():
        matched_column = _find_first_matching_column(df, aliases)
        if matched_column is not None:
            df[output_name] = df[matched_column]
        else:
            df[output_name] = pd.NA

    return df


def _validate_weight_config(weights: Dict[str, float]) -> None:
    """Validate metric weights before scoring."""
    if not weights:
        raise ValueError("METRIC_WEIGHTS cannot be empty.")

    negative_weights = {k: v for k, v in weights.items() if v < 0}
    if negative_weights:
        raise ValueError(f"Metric weights cannot be negative: {negative_weights}")

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Sum of metric weights must be greater than zero.")


def _deduplicate_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate percentile rows using core player keys.

    If multiple rows exist for the same player/team/season, keep the last row.
    """
    missing_keys = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Percentiles dataset is missing required key columns: {missing_keys}"
        )

    return df.drop_duplicates(subset=KEY_COLUMNS, keep="last").copy()


def _deduplicate_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate archetype rows using core player keys.

    If multiple rows exist, keep the last row.
    """
    missing_keys = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Archetype dataset is missing required key columns: {missing_keys}"
        )

    return df.drop_duplicates(subset=KEY_COLUMNS, keep="last").copy()


# -----------------------------------------------------------------------------
# Core scoring functions
# -----------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate primary input tables."""
    player_percentiles = _read_csv(PLAYER_PERCENTILES_PATH)
    player_archetypes = _read_csv(PLAYER_ARCHETYPE_PATH)

    _validate_required_columns(
        player_percentiles,
        KEY_COLUMNS,
        "player_percentiles",
    )
    _validate_required_columns(
        player_archetypes,
        KEY_COLUMNS + ["archetype"],
        "player_archetype_assignment",
    )

    player_percentiles = _standardize_text_columns(
        player_percentiles,
        KEY_COLUMNS + OPTIONAL_ID_COLUMNS,
    )
    player_archetypes = _standardize_text_columns(
        player_archetypes,
        KEY_COLUMNS + OPTIONAL_ID_COLUMNS + ["archetype"],
    )

    player_percentiles = _deduplicate_percentiles(player_percentiles)
    player_archetypes = _deduplicate_archetypes(player_archetypes)

    return player_percentiles, player_archetypes


def prepare_scoring_frame(
    player_percentiles: pd.DataFrame,
    player_archetypes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge percentiles with archetype assignments and prepare resolved metric columns.
    """
    archetype_columns = KEY_COLUMNS + [
        column
        for column in ["position_group", "archetype"]
        if column in player_archetypes.columns
    ]
    archetype_frame = player_archetypes[archetype_columns].copy()

    scoring_df = player_percentiles.merge(
        archetype_frame,
        how="left",
        on=KEY_COLUMNS,
        suffixes=("", "_archetype"),
        validate="one_to_one",
    )

    if "position_group" not in scoring_df.columns and "position_group_archetype" in scoring_df.columns:
        scoring_df["position_group"] = scoring_df["position_group_archetype"]
    elif "position_group" in scoring_df.columns and "position_group_archetype" in scoring_df.columns:
        scoring_df["position_group"] = scoring_df["position_group"].fillna(
            scoring_df["position_group_archetype"]
        )

    scoring_df["position_group"] = scoring_df.get("position_group", pd.Series(pd.NA, index=scoring_df.index))
    scoring_df["archetype"] = scoring_df.get("archetype", pd.Series(pd.NA, index=scoring_df.index))

    # Enforce current project position-group structure.
    scoring_df["position_group"] = (
        scoring_df["position_group"]
        .replace({"Big": "Forward", "Center": "Forward", "big": "Forward", "center": "Forward"})
    )

    resolved_percentile_columns = _resolve_percentile_columns(scoring_df)

    percentile_columns = list(resolved_percentile_columns.values())
    scoring_df = _ensure_numeric(scoring_df, percentile_columns)

    for metric_name, column_name in resolved_percentile_columns.items():
        normalized_column = f"{metric_name}_percentile"
        scoring_df[normalized_column] = _normalize_percentile_series(scoring_df[column_name])

    scoring_df = _attach_optional_raw_metric_columns(scoring_df)

    return scoring_df


def calculate_shooting_score(scoring_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted shooting component scores and total shooting score.

    The total score remains on a 0-100 scale when percentiles are on a 0-100 scale.
    """
    _validate_weight_config(METRIC_WEIGHTS)

    working_df = scoring_df.copy()

    component_score_columns: List[str] = []
    weighted_sum = pd.Series(0.0, index=working_df.index)
    total_weight = 0.0

    for metric_name, weight in METRIC_WEIGHTS.items():
        percentile_column = f"{metric_name}_percentile"
        component_column = f"{metric_name}_weighted_score"

        if percentile_column not in working_df.columns:
            raise ValueError(
                f"Expected normalized percentile column not found: {percentile_column}"
            )

        working_df[component_column] = working_df[percentile_column] * weight
        component_score_columns.append(component_column)
        weighted_sum = weighted_sum + working_df[component_column].fillna(0)
        total_weight += weight

    if total_weight <= 0:
        raise ValueError("Total metric weight must be greater than zero.")

    working_df["shooting_score"] = weighted_sum / total_weight

    # Round presentation fields for cleaner downstream use.
    percentile_output_columns = [
        f"{metric_name}_percentile" for metric_name in METRIC_WEIGHTS
    ]
    weighted_output_columns = [
        f"{metric_name}_weighted_score" for metric_name in METRIC_WEIGHTS
    ]

    round_columns = percentile_output_columns + weighted_output_columns + ["shooting_score"]
    for column in round_columns:
        if column in working_df.columns:
            working_df[column] = working_df[column].round(2)

    return working_df


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order final output columns.

    Includes core identifiers, archetype context, raw metrics when available,
    normalized percentiles, weighted subscore columns, and final shooting score.
    """
    ordered_columns: List[str] = [
        "player_name",
        "team_name",
        "season",
        "position_group",
        "archetype",
        "three_point_pct_raw",
        "three_point_attempt_rate_raw",
        "effective_field_goal_pct_raw",
        "true_shooting_pct_raw",
        "three_point_pct_percentile",
        "three_point_attempt_rate_percentile",
        "effective_field_goal_pct_percentile",
        "true_shooting_pct_percentile",
        "three_point_pct_weighted_score",
        "three_point_attempt_rate_weighted_score",
        "effective_field_goal_pct_weighted_score",
        "true_shooting_pct_weighted_score",
        "shooting_score",
    ]

    existing_columns = [column for column in ordered_columns if column in df.columns]
    output_df = df[existing_columns].copy()

    _validate_required_columns(output_df, REQUIRED_OUTPUT_COLUMNS, "shooting_scores output")

    output_df = output_df.sort_values(
        by=["season", "team_name", "player_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    output_df = output_df.drop_duplicates(subset=KEY_COLUMNS, keep="last")

    if output_df.empty:
        raise ValueError("Final shooting_scores output is empty.")

    return output_df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write final scoring output to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_shooting_scores() -> pd.DataFrame:
    """Main pipeline to build shooting evaluation scores."""
    player_percentiles, player_archetypes = load_inputs()
    scoring_frame = prepare_scoring_frame(player_percentiles, player_archetypes)
    scored_df = calculate_shooting_score(scoring_frame)
    output_df = finalize_output(scored_df)
    write_output(output_df, OUTPUT_PATH)
    return output_df


def main() -> None:
    """Execute the shooting scoring pipeline."""
    output_df = build_shooting_scores()
    print(f"Shooting scores rows written: {len(output_df)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()