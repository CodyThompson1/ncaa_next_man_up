"""
Build defense evaluation scores for the NCAA Next Man Up project.

Purpose
-------
This script creates the reusable feature table:

    data/features/defense_scores.csv

The table supports the player evaluation engine by converting peer-group
percentiles into a weighted Defense score. Defense is intended to reflect
disruption, defensive activity, and defensive effectiveness within each
player's comparison context.

Primary input
-------------
- data/features/player_percentiles.csv

Supporting input
----------------
- data/features/player_archetype_assignment.csv

Design notes
------------
- Percentiles are used as the scoring backbone.
- Metrics where lower is better are inverted before scoring.
- Metric weights are intentionally centralized and easy to edit.
- The script is built to be resilient to small schema differences in the
  percentile table by using a metric alias system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
)
PLAYER_ARCHETYPES_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "defense_scores.csv"


# ---------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------

IDENTIFIER_COLUMNS = [
    "player_name",
    "team_name",
    "season",
]

OPTIONAL_ID_COLUMNS = [
    "conference_name",
    "position_group",
    "position_raw",
]

ARCHETYPE_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_group",
    "player_archetype",
]

# Weights are intentionally simple and easy to adjust.
# The "metric_key" is the logical metric used in scoring.
# The "aliases" list allows matching percentile columns that may be named
# slightly differently across pipeline versions.
#
# All scoring is based on percentiles on a 0-100 scale.
# If a column is already a percentile and higher is better:
#     adjusted = percentile
# If lower is better (inverse metric):
#     adjusted = 100 - percentile
DEFENSE_METRIC_CONFIG = {
    "stl_pct": {
        "weight": 0.35,
        "higher_is_better": True,
        "aliases": [
            "stl_pct",
            "steal_pct",
            "stl_rate",
            "steal_rate",
        ],
    },
    "blk_pct": {
        "weight": 0.25,
        "higher_is_better": True,
        "aliases": [
            "blk_pct",
            "block_pct",
            "blk_rate",
            "block_rate",
        ],
    },
    "drtg": {
        "weight": 0.40,
        "higher_is_better": False,
        "aliases": [
            "drtg",
            "def_rtg",
            "defensive_rating",
            "def_rating",
        ],
    },
}

PERCENTILE_SUFFIX_CANDIDATES = [
    "_percentile",
    "_pctile",
    "_perc",
    "_pctl",
    "_percent_rank",
    "_percent_rank_value",
]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _validate_file_exists(path: Path) -> None:
    """Raise a FileNotFoundError if the expected file does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def _normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Strip whitespace from selected text columns when present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : Iterable[str]
        Candidate text columns to clean.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized text fields.
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            df.loc[df[column].isin(["nan", "None"]), column] = pd.NA
    return df


def _normalize_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize position_group values to the project convention.

    Project rule:
    - Guard
    - Forward

    Any legacy Big / Center style values are mapped to Forward.
    """
    df = df.copy()

    if "position_group" not in df.columns:
        return df

    mapping = {
        "guard": "Guard",
        "guards": "Guard",
        "g": "Guard",
        "forward": "Forward",
        "forwards": "Forward",
        "f": "Forward",
        "big": "Forward",
        "bigs": "Forward",
        "center": "Forward",
        "centers": "Forward",
        "c": "Forward",
    }

    normalized = (
        df["position_group"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    df["position_group"] = normalized.fillna(df["position_group"])
    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    df_name: str,
) -> None:
    """Raise a ValueError if required columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}"
        )


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric with coercion."""
    return pd.to_numeric(series, errors="coerce")


def _standardize_percentile_scale(series: pd.Series) -> pd.Series:
    """
    Standardize percentile values to a 0-100 scale.

    Handles both:
    - 0 to 1
    - 0 to 100
    """
    numeric = _safe_numeric(series)

    if numeric.dropna().empty:
        return numeric

    max_value = numeric.dropna().max()
    if max_value <= 1.0:
        return numeric * 100.0

    return numeric


def _build_percentile_column_candidates(metric_alias: str) -> List[str]:
    """
    Build likely percentile column names for a given raw metric alias.
    """
    candidates = [metric_alias]

    for suffix in PERCENTILE_SUFFIX_CANDIDATES:
        candidates.append(f"{metric_alias}{suffix}")

    return candidates


def _find_metric_percentile_column(
    df: pd.DataFrame,
    metric_key: str,
    aliases: List[str],
) -> Optional[str]:
    """
    Find the best matching percentile column for a metric.

    Search order
    ------------
    1. Exact logical metric name if present
    2. Exact alias match if present
    3. Alias + known percentile suffix patterns
    4. Case-insensitive fallback search for names containing the alias and
       a percentile keyword
    """
    columns = list(df.columns)
    lower_map = {col.lower(): col for col in columns}

    direct_candidates = [metric_key] + aliases
    for candidate in direct_candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    generated_candidates: List[str] = []
    for alias in aliases:
        generated_candidates.extend(_build_percentile_column_candidates(alias))
    generated_candidates.extend(_build_percentile_column_candidates(metric_key))

    for candidate in generated_candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    percentile_keywords = [
        "percentile",
        "pctile",
        "pctl",
        "perc",
        "percent_rank",
    ]

    for alias in [metric_key] + aliases:
        alias_lower = alias.lower()
        for col in columns:
            col_lower = col.lower()
            if alias_lower in col_lower and any(
                keyword in col_lower for keyword in percentile_keywords
            ):
                return col

    return None


def _deduplicate_on_keys(
    df: pd.DataFrame,
    subset: List[str],
    df_name: str,
) -> pd.DataFrame:
    """
    Drop duplicate rows on a business key and keep first.

    A warning-style print is used instead of failure because later pipeline
    steps benefit from graceful handling of mild duplicates.
    """
    duplicate_count = df.duplicated(subset=subset).sum()
    if duplicate_count > 0:
        print(
            f"[WARN] {df_name} has {duplicate_count} duplicate rows on {subset}. "
            "Keeping first occurrence."
        )
        df = df.drop_duplicates(subset=subset, keep="first").copy()

    return df


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _load_player_percentiles() -> pd.DataFrame:
    """
    Load and prepare the player percentile table.
    """
    _validate_file_exists(PLAYER_PERCENTILES_PATH)
    df = _read_csv(PLAYER_PERCENTILES_PATH)

    _validate_required_columns(
        df=df,
        required_columns=IDENTIFIER_COLUMNS,
        df_name="player_percentiles",
    )

    df = _normalize_text_columns(
        df,
        IDENTIFIER_COLUMNS + OPTIONAL_ID_COLUMNS,
    )
    df = _normalize_position_group(df)

    if "season" in df.columns:
        df["season"] = _safe_numeric(df["season"])

    df = _deduplicate_on_keys(
        df=df,
        subset=IDENTIFIER_COLUMNS,
        df_name="player_percentiles",
    )

    return df


def _load_player_archetypes() -> pd.DataFrame:
    """
    Load and prepare the player archetype assignment table.
    """
    _validate_file_exists(PLAYER_ARCHETYPES_PATH)
    df = _read_csv(PLAYER_ARCHETYPES_PATH)

    _validate_required_columns(
        df=df,
        required_columns=["player_name", "team_name", "season"],
        df_name="player_archetype_assignment",
    )

    df = _normalize_text_columns(
        df,
        list(set(ARCHETYPE_COLUMNS + ["archetype"])),
    )
    df = _normalize_position_group(df)

    if "season" in df.columns:
        df["season"] = _safe_numeric(df["season"])

    # Support either `player_archetype` or `archetype`.
    if "player_archetype" not in df.columns and "archetype" in df.columns:
        df = df.rename(columns={"archetype": "player_archetype"})

    required_after_rename = [
        "player_name",
        "team_name",
        "season",
        "player_archetype",
    ]
    _validate_required_columns(
        df=df,
        required_columns=required_after_rename,
        df_name="player_archetype_assignment",
    )

    # Keep only necessary columns, but allow position_group to pass through
    # when available.
    keep_columns = [
        col for col in ARCHETYPE_COLUMNS if col in df.columns
    ]
    df = df[keep_columns].copy()

    df = _deduplicate_on_keys(
        df=df,
        subset=["player_name", "team_name", "season"],
        df_name="player_archetype_assignment",
    )

    return df


def _merge_inputs(
    percentiles_df: pd.DataFrame,
    archetypes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge percentile data with archetype assignments.
    """
    merged_df = percentiles_df.merge(
        archetypes_df,
        on=["player_name", "team_name", "season"],
        how="left",
        suffixes=("", "_arch"),
        validate="one_to_one",
    )

    # Prefer percentile table position group first, then archetype table.
    if "position_group_arch" in merged_df.columns:
        if "position_group" not in merged_df.columns:
            merged_df["position_group"] = merged_df["position_group_arch"]
        else:
            merged_df["position_group"] = merged_df["position_group"].fillna(
                merged_df["position_group_arch"]
            )
        merged_df = merged_df.drop(columns=["position_group_arch"])

    merged_df = _normalize_position_group(merged_df)
    return merged_df


# ---------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------

def _resolve_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Resolve the actual DataFrame column used for each defense metric.

    Returns
    -------
    dict
        Mapping from logical metric key to resolved column name.
    """
    resolved_columns: Dict[str, str] = {}

    for metric_key, config in DEFENSE_METRIC_CONFIG.items():
        column_name = _find_metric_percentile_column(
            df=df,
            metric_key=metric_key,
            aliases=config["aliases"],
        )
        if column_name is not None:
            resolved_columns[metric_key] = column_name

    if not resolved_columns:
        raise ValueError(
            "No defense percentile columns could be resolved from "
            "player_percentiles.csv. Expected percentile columns related to "
            "stl_pct / blk_pct / drtg."
        )

    return resolved_columns


def _build_component_scores(
    df: pd.DataFrame,
    resolved_metric_columns: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create adjusted percentile component score columns for each available
    defense metric.

    Returns
    -------
    tuple
        (DataFrame with component columns, list of component column names)
    """
    df = df.copy()
    component_score_columns: List[str] = []

    for metric_key, source_column in resolved_metric_columns.items():
        config = DEFENSE_METRIC_CONFIG[metric_key]
        higher_is_better = config["higher_is_better"]

        adjusted_column = f"{metric_key}_defense_component_score"
        raw_percentile = _standardize_percentile_scale(df[source_column])

        if higher_is_better:
            df[adjusted_column] = raw_percentile
        else:
            df[adjusted_column] = 100.0 - raw_percentile

        df[adjusted_column] = df[adjusted_column].clip(lower=0, upper=100)
        component_score_columns.append(adjusted_column)

    return df, component_score_columns


def _build_weighted_defense_score(
    df: pd.DataFrame,
    resolved_metric_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Create the weighted overall defense score.

    Scoring approach
    ----------------
    - Use only metrics that were successfully found in the percentile file.
    - Re-normalize weights across available metrics so the final score remains
      on a 0-100 scale even when one optional metric is unavailable.
    """
    df = df.copy()

    available_weights = {
        metric_key: DEFENSE_METRIC_CONFIG[metric_key]["weight"]
        for metric_key in resolved_metric_columns
    }

    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        raise ValueError("Total defense metric weight must be greater than zero.")

    normalized_weights = {
        metric_key: weight / total_weight
        for metric_key, weight in available_weights.items()
    }

    weighted_score = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_key, normalized_weight in normalized_weights.items():
        component_column = f"{metric_key}_defense_component_score"
        weighted_score = weighted_score.add(
            df[component_column].fillna(0) * normalized_weight,
            fill_value=0,
        )

    df["defense_score"] = weighted_score.round(2)
    return df


def _add_metadata_columns(
    df: pd.DataFrame,
    resolved_metric_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Add transparent metadata columns showing which source percentile columns
    were used for scoring.
    """
    df = df.copy()

    for metric_key, source_column in resolved_metric_columns.items():
        df[f"{metric_key}_percentile_source_column"] = source_column

    return df


# ---------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------

def _build_output_table(
    scored_df: pd.DataFrame,
    resolved_metric_columns: Dict[str, str],
) -> pd.DataFrame:
    """
    Select and order final output columns for defense_scores.csv.
    """
    output_columns: List[str] = [
        "player_name",
        "team_name",
        "season",
    ]

    optional_front_columns = [
        "conference_name",
        "position_group",
        "position_raw",
        "player_archetype",
    ]
    output_columns.extend(
        [col for col in optional_front_columns if col in scored_df.columns]
    )

    raw_percentile_columns: List[str] = []
    component_columns: List[str] = []
    source_columns: List[str] = []

    for metric_key, source_column in resolved_metric_columns.items():
        raw_percentile_output_name = f"{metric_key}_percentile"
        scored_df[raw_percentile_output_name] = _standardize_percentile_scale(
            scored_df[source_column]
        ).round(2)

        raw_percentile_columns.append(raw_percentile_output_name)
        component_columns.append(f"{metric_key}_defense_component_score")
        source_columns.append(f"{metric_key}_percentile_source_column")

    output_columns.extend(raw_percentile_columns)
    output_columns.extend(component_columns)
    output_columns.append("defense_score")
    output_columns.extend(source_columns)

    # Keep only columns that exist to stay robust across minor schema shifts.
    output_columns = [col for col in output_columns if col in scored_df.columns]

    output_df = scored_df[output_columns].copy()

    output_df = output_df.sort_values(
        by=["season", "team_name", "player_name"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return output_df


def _write_output(df: pd.DataFrame, path: Path) -> None:
    """
    Write the defense score output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def build_defense_scores() -> pd.DataFrame:
    """
    Run the full defense scoring pipeline.

    Returns
    -------
    pd.DataFrame
        Final defense score feature table.
    """
    percentiles_df = _load_player_percentiles()
    archetypes_df = _load_player_archetypes()

    merged_df = _merge_inputs(
        percentiles_df=percentiles_df,
        archetypes_df=archetypes_df,
    )

    resolved_metric_columns = _resolve_metric_columns(merged_df)

    scored_df, _ = _build_component_scores(
        df=merged_df,
        resolved_metric_columns=resolved_metric_columns,
    )

    scored_df = _build_weighted_defense_score(
        df=scored_df,
        resolved_metric_columns=resolved_metric_columns,
    )

    scored_df = _add_metadata_columns(
        df=scored_df,
        resolved_metric_columns=resolved_metric_columns,
    )

    output_df = _build_output_table(
        scored_df=scored_df,
        resolved_metric_columns=resolved_metric_columns,
    )

    _write_output(output_df, OUTPUT_PATH)
    return output_df


def main() -> None:
    """
    Entry point for script execution.
    """
    output_df = build_defense_scores()
    print(
        f"[SUCCESS] Defense scores written to: {OUTPUT_PATH} "
        f"({len(output_df)} rows)"
    )


if __name__ == "__main__":
    main()