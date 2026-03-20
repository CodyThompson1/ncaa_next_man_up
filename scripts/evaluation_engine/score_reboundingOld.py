"""
Build rebounding evaluation scores for NCAA Next Man Up.

This script creates the `Rebounding` category score for the evaluation engine
using percentile-based rebounding metrics within project peer groups.

Primary input:
- data/features/player_percentiles.csv

Supporting input:
- data/features/player_archetype_assignment.csv

Output:
- data/features/rebounding_scores.csv

Scoring notes:
- Rebounding is based on peer-group percentile performance.
- The core rebounding metrics are:
    - trb_pct
    - orb_pct
    - drb_pct
- Because Bigs are now folded into the Forward group, any rebounding
  interpretation for former big-type players should flow through Forward logic.

The script is written to be resilient to either:
1. Wide percentile tables, where percentile columns already exist, or
2. Long percentile tables, where each row is one metric/player record.

Author: OpenAI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
ARCHETYPE_ASSIGNMENT_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "rebounding_scores.csv"

REQUIRED_ID_COLUMNS = ["player_name", "team_name", "season"]
OPTIONAL_ID_COLUMNS = ["conference_name", "position_group"]

REB_METRICS = ["trb_pct", "orb_pct", "drb_pct"]

# Keep weighting transparent and easy to update.
# These weights are applied to percentile values, not raw stats.
POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Guard": {
        "trb_pct": 0.50,
        "orb_pct": 0.15,
        "drb_pct": 0.35,
    },
    "Forward": {
        "trb_pct": 0.50,
        "orb_pct": 0.25,
        "drb_pct": 0.25,
    },
}

# Default fallback if position_group is missing or unexpected.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "trb_pct": 0.50,
    "orb_pct": 0.25,
    "drb_pct": 0.25,
}

# Supported percentile column naming patterns in wide-form input.
PERCENTILE_SUFFIX_CANDIDATES = [
    "_percentile",
    "_pctile",
    "_percent_rank",
    "_percent_rank_value",
    "_peer_percentile",
    "_peer_pctile",
]


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _validate_file_exists(path: Path) -> None:
    """Raise a helpful error if an expected input file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file with basic validation."""
    _validate_file_exists(path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    logging.info("Loaded %s rows from %s", len(df), path)
    return df


def _normalize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Strip whitespace from selected string columns if they exist."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    df_name: str,
) -> None:
    """Validate that all required columns exist in a DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _drop_exact_duplicates(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Drop exact duplicate rows and log the change."""
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)

    if removed > 0:
        logging.info("Dropped %s exact duplicate rows from %s", removed, df_name)

    return df


def _resolve_archetype_column(df: pd.DataFrame) -> str:
    """Find the archetype column name from a set of common possibilities."""
    candidates = [
        "archetype",
        "role_archetype",
        "player_archetype",
        "assigned_archetype",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find an archetype column in player_archetype_assignment.csv. "
        f"Checked: {candidates}"
    )


def _resolve_position_group_column(df: pd.DataFrame) -> Optional[str]:
    """Find a usable position group column if available."""
    candidates = [
        "position_group",
        "player_position_group",
        "position_group_final",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected columns to numeric where present."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Percentile input handling
# ---------------------------------------------------------------------

def _is_long_percentile_format(df: pd.DataFrame) -> bool:
    """
    Detect whether player_percentiles appears to be long format.

    Expected long-form hints:
    - metric column exists
    - percentile value column exists
    """
    metric_candidates = ["metric", "metric_name", "stat_name"]
    percentile_value_candidates = [
        "percentile",
        "percentile_value",
        "metric_percentile",
        "peer_percentile",
    ]

    has_metric_col = any(col in df.columns for col in metric_candidates)
    has_percentile_col = any(col in df.columns for col in percentile_value_candidates)

    return has_metric_col and has_percentile_col


def _build_wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form percentile input into wide form.

    Expected long-form columns include:
    - player_name
    - team_name
    - season
    - metric
    - percentile
    """
    df = df.copy()

    metric_col = next(
        col for col in ["metric", "metric_name", "stat_name"] if col in df.columns
    )
    percentile_col = next(
        col
        for col in ["percentile", "percentile_value", "metric_percentile", "peer_percentile"]
        if col in df.columns
    )

    id_columns = [col for col in REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS if col in df.columns]

    filtered = df[df[metric_col].isin(REB_METRICS)].copy()
    if filtered.empty:
        raise ValueError(
            "No rebounding metrics found in long-form player_percentiles input. "
            f"Expected metrics: {REB_METRICS}"
        )

    filtered = _coerce_numeric(filtered, [percentile_col])

    wide = (
        filtered.pivot_table(
            index=id_columns,
            columns=metric_col,
            values=percentile_col,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    rename_map = {metric: f"{metric}_percentile" for metric in REB_METRICS if metric in wide.columns}
    wide = wide.rename(columns=rename_map)

    return wide


def _find_percentile_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    """Locate a percentile column for a given metric in wide-form input."""
    direct_candidates = [
        f"{metric_name}_percentile",
        f"{metric_name}_pctile",
        f"{metric_name}_percent_rank",
        f"{metric_name}_peer_percentile",
    ]

    for col in direct_candidates:
        if col in df.columns:
            return col

    for suffix in PERCENTILE_SUFFIX_CANDIDATES:
        candidate = f"{metric_name}{suffix}"
        if candidate in df.columns:
            return candidate

    return None


def _prepare_percentile_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare player_percentiles input into a scoring-ready wide table.

    Supports either:
    - long format with metric rows, or
    - wide format with metric percentile columns.
    """
    df = _drop_exact_duplicates(df, "player_percentiles")
    df = _normalize_string_columns(
        df,
        REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS,
    )

    _validate_required_columns(df, REQUIRED_ID_COLUMNS, "player_percentiles")

    if _is_long_percentile_format(df):
        logging.info("Detected long-form player_percentiles input")
        wide_df = _build_wide_from_long(df)
    else:
        logging.info("Detected wide-form player_percentiles input")
        wide_df = df.copy()

    percentile_column_map = {}
    for metric in REB_METRICS:
        percentile_col = _find_percentile_column(wide_df, metric)
        if percentile_col is None:
            raise ValueError(
                f"Could not find a percentile column for metric '{metric}' in "
                "player_percentiles.csv"
            )
        percentile_column_map[metric] = percentile_col

    selected_columns: List[str] = [
        col for col in REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS if col in wide_df.columns
    ]
    selected_columns.extend(percentile_column_map.values())

    scoring_df = wide_df[selected_columns].copy()

    rename_map = {
        percentile_column_map["trb_pct"]: "trb_pct_percentile",
        percentile_column_map["orb_pct"]: "orb_pct_percentile",
        percentile_column_map["drb_pct"]: "drb_pct_percentile",
    }
    scoring_df = scoring_df.rename(columns=rename_map)

    scoring_df = _coerce_numeric(
        scoring_df,
        ["trb_pct_percentile", "orb_pct_percentile", "drb_pct_percentile"],
    )

    return scoring_df


# ---------------------------------------------------------------------
# Archetype handling
# ---------------------------------------------------------------------

def _prepare_archetype_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare archetype assignment input for merging."""
    df = _drop_exact_duplicates(df, "player_archetype_assignment")
    df = _normalize_string_columns(df, REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS)

    _validate_required_columns(df, REQUIRED_ID_COLUMNS, "player_archetype_assignment")

    archetype_col = _resolve_archetype_column(df)
    position_group_col = _resolve_position_group_column(df)

    keep_columns = REQUIRED_ID_COLUMNS.copy()

    if position_group_col is not None:
        keep_columns.append(position_group_col)

    keep_columns.append(archetype_col)

    archetype_df = df[keep_columns].copy()

    rename_map = {archetype_col: "archetype"}
    if position_group_col is not None and position_group_col != "position_group":
        rename_map[position_group_col] = "position_group"

    archetype_df = archetype_df.rename(columns=rename_map)

    subset_keys = REQUIRED_ID_COLUMNS.copy()
    if "position_group" in archetype_df.columns:
        subset_keys.append("position_group")

    archetype_df = archetype_df.drop_duplicates(subset=REQUIRED_ID_COLUMNS, keep="first")

    return archetype_df


# ---------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------

def _resolve_weights(position_group: Optional[str]) -> Dict[str, float]:
    """Return metric weights for the given position group."""
    if pd.isna(position_group) or position_group is None:
        return DEFAULT_WEIGHTS

    normalized = str(position_group).strip().title()
    return POSITION_WEIGHTS.get(normalized, DEFAULT_WEIGHTS)


def _score_row(row: pd.Series) -> pd.Series:
    """
    Compute weighted rebounding components and total score for one player row.

    The score is based on percentile values, expected on a 0-100 scale.
    """
    weights = _resolve_weights(row.get("position_group"))

    trb_pctile = row.get("trb_pct_percentile")
    orb_pctile = row.get("orb_pct_percentile")
    drb_pctile = row.get("drb_pct_percentile")

    trb_component = (trb_pctile if pd.notna(trb_pctile) else 0.0) * weights["trb_pct"]
    orb_component = (orb_pctile if pd.notna(orb_pctile) else 0.0) * weights["orb_pct"]
    drb_component = (drb_pctile if pd.notna(drb_pctile) else 0.0) * weights["drb_pct"]

    available_metrics = sum(
        [
            pd.notna(trb_pctile),
            pd.notna(orb_pctile),
            pd.notna(drb_pctile),
        ]
    )

    return pd.Series(
        {
            "trb_pct_weight": weights["trb_pct"],
            "orb_pct_weight": weights["orb_pct"],
            "drb_pct_weight": weights["drb_pct"],
            "trb_pct_component_score": trb_component,
            "orb_pct_component_score": orb_component,
            "drb_pct_component_score": drb_component,
            "rebounding_score": trb_component + orb_component + drb_component,
            "rebounding_metrics_available": available_metrics,
        }
    )


def _build_rebounding_scores(
    percentiles_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge sources and build final rebounding score dataset."""
    merged = percentiles_df.merge(
        archetypes_df,
        on=REQUIRED_ID_COLUMNS,
        how="left",
        suffixes=("", "_arch"),
        validate="m:1",
    )

    if "position_group" not in merged.columns and "position_group_arch" in merged.columns:
        merged = merged.rename(columns={"position_group_arch": "position_group"})
    elif "position_group_arch" in merged.columns:
        merged["position_group"] = merged["position_group"].fillna(merged["position_group_arch"])
        merged = merged.drop(columns=["position_group_arch"])

    if "position_group" not in merged.columns:
        raise ValueError(
            "Could not determine position_group after merging player_percentiles "
            "and player_archetype_assignment."
        )

    score_columns = merged.apply(_score_row, axis=1)
    output_df = pd.concat([merged, score_columns], axis=1)

    output_df["evaluation_category"] = "Rebounding"

    preferred_column_order = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "archetype",
        "evaluation_category",
        "trb_pct_percentile",
        "orb_pct_percentile",
        "drb_pct_percentile",
        "trb_pct_weight",
        "orb_pct_weight",
        "drb_pct_weight",
        "trb_pct_component_score",
        "orb_pct_component_score",
        "drb_pct_component_score",
        "rebounding_metrics_available",
        "rebounding_score",
    ]

    existing_columns = [col for col in preferred_column_order if col in output_df.columns]
    remaining_columns = [col for col in output_df.columns if col not in existing_columns]

    output_df = output_df[existing_columns + remaining_columns].copy()

    output_df = output_df.drop_duplicates(
        subset=["player_name", "team_name", "season"],
        keep="first",
    )

    output_df = output_df.sort_values(
        by=["season", "team_name", "player_name"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return output_df


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def _write_output(df: pd.DataFrame, path: Path) -> None:
    """Write output CSV to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Wrote %s rows to %s", len(df), path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the rebounding scoring pipeline."""
    logging.info("Starting rebounding scoring pipeline")

    player_percentiles_df = _load_csv(PLAYER_PERCENTILES_PATH)
    archetype_assignment_df = _load_csv(ARCHETYPE_ASSIGNMENT_PATH)

    prepared_percentiles_df = _prepare_percentile_input(player_percentiles_df)
    prepared_archetypes_df = _prepare_archetype_input(archetype_assignment_df)

    rebounding_scores_df = _build_rebounding_scores(
        percentiles_df=prepared_percentiles_df,
        archetypes_df=prepared_archetypes_df,
    )

    if rebounding_scores_df.empty:
        raise ValueError("Rebounding scoring output is empty. No file was written.")

    _write_output(rebounding_scores_df, OUTPUT_PATH)

    logging.info("Rebounding scoring pipeline completed successfully")


if __name__ == "__main__":
    main()