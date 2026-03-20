"""
Build playmaking scores for the NCAA Next Man Up evaluation engine.

Purpose
-------
This script creates percentile-based Playmaking scores for Montana player
evaluation using the engineered player percentile dataset and archetype
assignment dataset.

Playmaking is intended to reflect:
- ball creation
- passing volume / impact
- decision-making
- turnover control

Primary inputs
--------------
- data/features/player_percentiles.csv
- data/features/player_archetype_assignment.csv

Primary output
--------------
- data/features/playmaking_scores.csv

Project notes
-------------
- Uses percentile-based scoring as the primary scoring method.
- Handles inverse metrics correctly where lower is better, especially for
  turnover-related measures.
- Keeps weights easy to change in the configuration block.
- Preserves player identifier and archetype context columns.
- Designed to be reusable by later evaluation-engine scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
PLAYER_ARCHETYPE_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "playmaking_scores.csv"

ID_COLUMNS = ["player_name", "team_name", "season"]

CONTEXT_COLUMNS_CANDIDATES = [
    "conference_name",
    "position_group",
    "archetype",
    "role_archetype",
    "position_raw",
    "class",
    "games",
    "games_played",
    "minutes",
    "minutes_total",
    "pct_minutes",
    "usg_pct",
    "usage_rate",
    "usage_window_low",
    "usage_window_high",
]

# Metric weights are intentionally easy to edit.
# The keys below represent logical metrics. The script will resolve each metric
# to an available percentile column if possible.
PLAYMAKING_METRIC_WEIGHTS: Dict[str, float] = {
    "ast_pct": 0.45,
    "ast": 0.25,
    "tov_pct": 0.20,
    "creation_proxy": 0.10,
}

# Lower raw values are better for these metrics.
INVERSE_METRICS = {
    "tov_pct",
    "turnover_rate",
    "turnover_pct",
    "tov_per_game",
    "turnovers_per_game",
    "turnovers",
    "tov",
}

# Ordered candidate columns used to find a usable percentile measure
# for each logical metric.
METRIC_CANDIDATES: Dict[str, List[str]] = {
    "ast_pct": [
        "ast_pct_percentile",
        "ast_pct_pctile",
        "ast_pct_percent_rank",
        "ast_pct_pr",
        "assist_rate_percentile",
        "assist_rate_pctile",
        "assist_rate_percent_rank",
        "assist_rate_pr",
        "ast_pct",
        "assist_rate",
    ],
    "ast": [
        "ast_percentile",
        "ast_pctile",
        "ast_percent_rank",
        "ast_pr",
        "assists_percentile",
        "assists_pctile",
        "assists_percent_rank",
        "assists_pr",
        "ast",
        "assists",
        "assists_per_game_percentile",
        "assists_per_game_pctile",
        "assists_per_game_percent_rank",
        "assists_per_game_pr",
        "assists_per_game",
        "ast_per_game",
    ],
    "tov_pct": [
        "tov_pct_percentile",
        "tov_pct_pctile",
        "tov_pct_percent_rank",
        "tov_pct_pr",
        "turnover_rate_percentile",
        "turnover_rate_pctile",
        "turnover_rate_percent_rank",
        "turnover_rate_pr",
        "turnover_pct_percentile",
        "turnover_pct_pctile",
        "turnover_pct_percent_rank",
        "turnover_pct_pr",
        "tov_pct",
        "turnover_rate",
        "turnover_pct",
        "tov_per_game_percentile",
        "tov_per_game_pctile",
        "tov_per_game_percent_rank",
        "tov_per_game_pr",
        "turnovers_per_game_percentile",
        "turnovers_per_game_pctile",
        "turnovers_per_game_percent_rank",
        "turnovers_per_game_pr",
        "tov_per_game",
        "turnovers_per_game",
        "tov",
        "turnovers",
    ],
    "creation_proxy": [
        "creation_load_percentile",
        "creation_load_pctile",
        "creation_load_percent_rank",
        "creation_load_pr",
        "playmaking_load_percentile",
        "playmaking_load_pctile",
        "playmaking_load_percent_rank",
        "playmaking_load_pr",
        "usage_adjusted_creation_percentile",
        "usage_adjusted_creation_pctile",
        "usage_adjusted_creation_percent_rank",
        "usage_adjusted_creation_pr",
        "ast_to_tov_ratio_percentile",
        "ast_to_tov_ratio_pctile",
        "ast_to_tov_ratio_percent_rank",
        "ast_to_tov_ratio_pr",
        "assist_to_turnover_ratio_percentile",
        "assist_to_turnover_ratio_pctile",
        "assist_to_turnover_ratio_percent_rank",
        "assist_to_turnover_ratio_pr",
        "ast_to_tov_ratio",
        "assist_to_turnover_ratio",
        "creation_load",
        "playmaking_load",
        "usage_adjusted_creation",
    ],
}

PERCENTILE_SUFFIXES = [
    "_percentile",
    "_pctile",
    "_percent_rank",
    "_pr",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_parent_dir(path: Path) -> None:
    """Create parent directory if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with basic existence validation."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    """Validate that required columns exist."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}"
        )


def _standardize_key_columns(df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    """
    Standardize join keys for reliable merges.

    String keys are stripped. Season is coerced to numeric where possible.
    """
    df = df.copy()

    for column in key_columns:
        if column not in df.columns:
            continue

        if column == "season":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        else:
            df[column] = df[column].astype(str).str.strip()

    return df


def _drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop exact or subset-based duplicates and return a clean copy."""
    before = len(df)
    df = df.drop_duplicates(subset=subset).copy()
    after = len(df)

    if after == 0:
        raise ValueError("All rows were removed during duplicate handling.")

    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to coerce non-key columns to numeric when possible."""
    df = df.copy()

    for column in df.columns:
        if column in ID_COLUMNS:
            continue

        if df[column].dtype == object:
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().sum() > 0:
                df[column] = converted

    return df


def _is_percentile_column(column_name: str) -> bool:
    """Return True if a column name appears to already be a percentile field."""
    return any(column_name.endswith(suffix) for suffix in PERCENTILE_SUFFIXES)


def _scale_percentile_series(series: pd.Series) -> pd.Series:
    """
    Normalize percentile-like values to a 0-100 scale.

    Supports inputs already on 0-100 or on 0-1 scales.
    """
    series = pd.to_numeric(series, errors="coerce")

    if series.dropna().empty:
        return series

    max_value = series.max(skipna=True)

    if pd.notna(max_value) and max_value <= 1.000001:
        return series * 100.0

    return series


def _reverse_percentile_series(series: pd.Series) -> pd.Series:
    """Convert a percentile to inverse percentile where lower raw is better."""
    return 100.0 - series


def _build_percentile_from_raw(df: pd.DataFrame, raw_column: str, inverse: bool) -> pd.Series:
    """
    Build percentile ranks from a raw metric column on a 0-100 scale.

    Higher is better by default. If inverse=True, lower raw values are better.
    """
    series = pd.to_numeric(df[raw_column], errors="coerce")

    if series.notna().sum() == 0:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")

    ascending = inverse
    percentile = series.rank(method="average", pct=True, ascending=ascending) * 100.0
    return percentile


def _resolve_metric_column(df: pd.DataFrame, logical_metric: str) -> Optional[str]:
    """
    Resolve the best available column for a logical playmaking metric.

    Preference order:
    1. Explicit configured candidate columns
    2. Heuristic percentile columns that contain the metric token
    """
    configured_candidates = METRIC_CANDIDATES.get(logical_metric, [])
    for candidate in configured_candidates:
        if candidate in df.columns:
            return candidate

    # Heuristic fallback
    tokens = {
        "ast_pct": ["ast_pct", "assist_rate"],
        "ast": ["ast", "assists"],
        "tov_pct": ["tov_pct", "turnover_rate", "turnover_pct", "tov", "turnovers"],
        "creation_proxy": [
            "creation",
            "playmaking_load",
            "assist_to_turnover_ratio",
            "ast_to_tov_ratio",
        ],
    }.get(logical_metric, [])

    percentile_matches: List[str] = []
    raw_matches: List[str] = []

    for column in df.columns:
        col_lower = column.lower()
        if any(token in col_lower for token in tokens):
            if _is_percentile_column(column):
                percentile_matches.append(column)
            else:
                raw_matches.append(column)

    if percentile_matches:
        return percentile_matches[0]

    if raw_matches:
        return raw_matches[0]

    return None


def _extract_context_columns(df: pd.DataFrame) -> List[str]:
    """Return context columns that actually exist in the dataframe."""
    return [column for column in CONTEXT_COLUMNS_CANDIDATES if column in df.columns]


def _prepare_archetype_table(archetype_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare archetype assignment table for merging.

    Normalizes archetype column naming and keeps one row per player/team/season.
    """
    archetype_df = _standardize_key_columns(archetype_df, ID_COLUMNS)

    archetype_column_map = {}
    if "role_archetype" in archetype_df.columns and "archetype" not in archetype_df.columns:
        archetype_column_map["role_archetype"] = "archetype"

    if archetype_column_map:
        archetype_df = archetype_df.rename(columns=archetype_column_map)

    keep_columns = [col for col in ID_COLUMNS + ["position_group", "archetype"] if col in archetype_df.columns]
    if not keep_columns:
        raise ValueError("Archetype file does not contain usable merge/context columns.")

    archetype_df = archetype_df[keep_columns].copy()
    archetype_df = _drop_exact_duplicates(archetype_df, subset=ID_COLUMNS)

    return archetype_df


def _build_metric_percentile_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Build resolved percentile fields for each configured playmaking metric.

    Returns
    -------
    df : pd.DataFrame
        Copy of df with standardized scoring columns added.
    resolved_map : Dict[str, str]
        Mapping from logical metric -> source column used.
    """
    df = df.copy()
    resolved_map: Dict[str, str] = {}

    for logical_metric, _weight in PLAYMAKING_METRIC_WEIGHTS.items():
        source_column = _resolve_metric_column(df, logical_metric)
        if source_column is None:
            continue

        output_column = f"{logical_metric}_score_component"
        inverse = logical_metric in INVERSE_METRICS or source_column in INVERSE_METRICS

        if _is_percentile_column(source_column):
            component = _scale_percentile_series(df[source_column])
            if inverse:
                component = _reverse_percentile_series(component)
        else:
            component = _build_percentile_from_raw(df, source_column, inverse=inverse)

        df[output_column] = component.clip(lower=0, upper=100)
        resolved_map[logical_metric] = source_column

    return df, resolved_map


def _compute_weighted_score(df: pd.DataFrame, resolved_map: Dict[str, str]) -> pd.DataFrame:
    """
    Compute weighted playmaking score using only available metric components.

    If some metrics are missing for a player, the score is reweighted over the
    available components for that player.
    """
    df = df.copy()

    component_columns = []
    component_weights = {}

    for logical_metric, weight in PLAYMAKING_METRIC_WEIGHTS.items():
        column = f"{logical_metric}_score_component"
        if logical_metric in resolved_map and column in df.columns:
            component_columns.append(column)
            component_weights[column] = weight

    if not component_columns:
        raise ValueError(
            "No usable playmaking percentile metrics were found in player_percentiles.csv."
        )

    weighted_value_sum = pd.Series(0.0, index=df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for column in component_columns:
        weight = component_weights[column]
        valid_mask = df[column].notna()
        weighted_value_sum = weighted_value_sum.add(df[column].fillna(0) * weight, fill_value=0)
        weight_sum = weight_sum.add(valid_mask.astype(float) * weight, fill_value=0)

    df["playmaking_score"] = (weighted_value_sum / weight_sum).where(weight_sum > 0)
    df["playmaking_score"] = df["playmaking_score"].round(2)
    df["playmaking_metrics_used"] = (weight_sum > 0).astype(int)

    metric_count = pd.Series(0, index=df.index, dtype="int64")
    for column in component_columns:
        metric_count = metric_count + df[column].notna().astype(int)

    df["playmaking_metric_count"] = metric_count
    df["playmaking_weight_sum_used"] = weight_sum.round(4)

    return df


def _add_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional score band labels for downstream dashboard use."""
    df = df.copy()

    def categorize(score: float) -> Optional[str]:
        if pd.isna(score):
            return pd.NA
        if score >= 90:
            return "Elite"
        if score >= 75:
            return "Strong"
        if score >= 60:
            return "Solid"
        if score >= 40:
            return "Developing"
        return "Limited"

    df["playmaking_score_band"] = df["playmaking_score"].apply(categorize)
    return df


def _select_output_columns(df: pd.DataFrame, resolved_map: Dict[str, str]) -> pd.DataFrame:
    """Select final output schema with identifiers, context, and scoring details."""
    context_columns = _extract_context_columns(df)

    output_columns: List[str] = []
    output_columns.extend([col for col in ID_COLUMNS if col in df.columns])
    output_columns.extend([col for col in context_columns if col not in output_columns])

    # Add source-metric traceability columns.
    for logical_metric in PLAYMAKING_METRIC_WEIGHTS:
        source_column = resolved_map.get(logical_metric)
        if source_column is not None:
            source_name_column = f"{logical_metric}_source_metric"
            component_column = f"{logical_metric}_score_component"
            df[source_name_column] = source_column

            if source_name_column not in output_columns:
                output_columns.append(source_name_column)
            if component_column in df.columns and component_column not in output_columns:
                output_columns.append(component_column)

    scoring_columns = [
        "playmaking_metric_count",
        "playmaking_weight_sum_used",
        "playmaking_score",
        "playmaking_score_band",
    ]

    for column in scoring_columns:
        if column in df.columns and column not in output_columns:
            output_columns.append(column)

    final_df = df[output_columns].copy()
    return final_df


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def build_playmaking_scores() -> pd.DataFrame:
    """Build the playmaking score dataset."""
    percentiles_df = _read_csv(PLAYER_PERCENTILES_PATH)
    archetype_df = _read_csv(PLAYER_ARCHETYPE_PATH)

    _validate_required_columns(percentiles_df, ID_COLUMNS, "player_percentiles.csv")
    _validate_required_columns(archetype_df, ID_COLUMNS, "player_archetype_assignment.csv")

    percentiles_df = _standardize_key_columns(percentiles_df, ID_COLUMNS)
    archetype_df = _prepare_archetype_table(archetype_df)

    percentiles_df = _drop_exact_duplicates(percentiles_df, subset=ID_COLUMNS)
    percentiles_df = _coerce_numeric_columns(percentiles_df)

    merged_df = percentiles_df.merge(
        archetype_df,
        on=ID_COLUMNS,
        how="left",
        suffixes=("", "_archetype"),
        validate="one_to_one",
    )

    if merged_df.empty:
        raise ValueError("Merged playmaking base table is empty after combining inputs.")

    merged_df, resolved_map = _build_metric_percentile_columns(merged_df)

    if not resolved_map:
        raise ValueError(
            "No playmaking metrics could be resolved from player_percentiles.csv. "
            "Check available columns and update metric mapping configuration."
        )

    scored_df = _compute_weighted_score(merged_df, resolved_map)
    scored_df = _add_score_bands(scored_df)
    scored_df = _select_output_columns(scored_df, resolved_map)

    scored_df = scored_df.sort_values(
        by=["season", "team_name", "playmaking_score", "player_name"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    if scored_df.empty:
        raise ValueError("Final playmaking score output is empty.")

    return scored_df


def write_playmaking_scores(df: pd.DataFrame, output_path: Path) -> None:
    """Write the final playmaking score file to disk."""
    _ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the playmaking scoring pipeline."""
    playmaking_scores_df = build_playmaking_scores()
    write_playmaking_scores(playmaking_scores_df, OUTPUT_PATH)
    print(f"Playmaking scores written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()