"""
Build player percentiles for the NCAA Next Man Up project.

Purpose
-------
Create a reusable player-level percentile dataset by comparing each Montana
target player against the valid Big Sky peer set defined in
data/features/player_peer_groups.csv.

Inputs
------
- data/processed/player_data/player_stats_all_games_montana.csv
- data/processed/comparison_sets/conference_player_pool.csv
- data/features/player_peer_groups.csv
- data/features/player_position_groups.csv
- data/features/player_archetype_assignment.csv

Output
------
- data/features/player_percentiles.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_PLAYER_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_data"
    / "player_stats_all_games_montana.csv"
)

CONFERENCE_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comparison_sets"
    / "conference_player_pool.csv"
)

PLAYER_PEER_GROUPS_PATH = PROJECT_ROOT / "data" / "features" / "player_peer_groups.csv"
PLAYER_POSITION_GROUPS_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_position_groups.csv"
)
PLAYER_ARCHETYPE_ASSIGNMENT_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_PATH = OUTPUT_DIR / "player_percentiles.csv"


# ============================================================================
# Schema config
# ============================================================================

PLAYER_KEYS = ["player_name", "team_name", "season"]

REQUIRED_TARGET_COLUMNS = ["player_name", "team_name", "season"]
REQUIRED_POOL_COLUMNS = ["player_name", "team_name", "season"]
REQUIRED_POSITION_COLUMNS = ["player_name", "team_name", "season", "position_group"]
REQUIRED_ARCHETYPE_BASE_COLUMNS = ["player_name", "team_name", "season"]
REQUIRED_PEER_GROUP_COLUMNS = [
    "target_player_name",
    "target_team_name",
    "peer_player_name",
    "peer_team_name",
]

ARCHETYPE_CANDIDATE_COLUMNS = [
    "archetype",
    "role_archetype",
    "player_archetype",
    "assigned_archetype",
]

SEASON_LIKE_COLUMNS = {"season", "target_season", "peer_season", "peer_season_match"}

# Only metrics that exist in both target and peer tables will be used.
METRIC_CONFIG: Dict[str, Dict[str, str]] = {
    # Shooting
    "points_per_game": {"category": "shooting", "direction": "high"},
    "efg_pct": {"category": "shooting", "direction": "high"},
    "ts_pct": {"category": "shooting", "direction": "high"},
    "three_pt_pct": {"category": "shooting", "direction": "high"},
    "three_pa": {"category": "shooting", "direction": "high"},
    "three_point_attempt_rate": {"category": "shooting", "direction": "high"},
    "ft_rate": {"category": "shooting", "direction": "high"},
    "pprod": {"category": "shooting", "direction": "high"},

    # Playmaking
    "ast_pct": {"category": "playmaking", "direction": "high"},
    "assist_rate": {"category": "playmaking", "direction": "high"},
    "usg_pct": {"category": "playmaking", "direction": "high"},
    "tov_pct": {"category": "playmaking", "direction": "low"},

    # Rebounding
    "trb_pct": {"category": "rebounding", "direction": "high"},
    "orb_pct": {"category": "rebounding", "direction": "high"},
    "drb_pct": {"category": "rebounding", "direction": "high"},

    # Defense
    "stl_pct": {"category": "defense", "direction": "high"},
    "blk_pct": {"category": "defense", "direction": "high"},
    "drtg": {"category": "defense", "direction": "low"},

    # Efficiency
    "ortg": {"category": "efficiency", "direction": "high"},
    "per": {"category": "efficiency", "direction": "high"},
    "bpm": {"category": "efficiency", "direction": "high"},
    "ws": {"category": "efficiency", "direction": "high"},
    "minutes_per_game": {"category": "efficiency", "direction": "high"},
    "games_played": {"category": "efficiency", "direction": "high"},
}


# ============================================================================
# Utility helpers
# ============================================================================

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and validate that it exists and is not empty."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df


def _standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from object columns."""
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip()

    return df


def _coerce_key_columns(df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    """Standardize merge-key column types."""
    df = df.copy()

    for column in key_columns:
        if column not in df.columns:
            continue

        if column in SEASON_LIKE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        else:
            df[column] = df[column].astype(str).str.strip()

    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    df_name: str,
) -> None:
    """Validate that required columns exist."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


def _find_first_existing_column(df: pd.DataFrame, candidate_columns: List[str]) -> str | None:
    """Return the first candidate column found in the DataFrame."""
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def _normalize_percentile(value: float) -> float:
    """Clip percentile into [0, 1] and round."""
    if pd.isna(value):
        return np.nan
    return round(float(np.clip(value, 0.0, 1.0)), 6)


def _percentile_rank(
    peer_values: pd.Series,
    target_value: float,
    higher_is_better: bool,
) -> float:
    """
    Compute percentile rank for one target value against a peer distribution.

    Inclusive percentile with 0.5 weighting on ties:
        (count_less + 0.5 * count_equal) / n

    For inverse metrics, the percentile is flipped:
        1 - percentile
    """
    valid_peer_values = _safe_numeric(peer_values).dropna()

    if valid_peer_values.empty or pd.isna(target_value):
        return np.nan

    count_less = (valid_peer_values < target_value).sum()
    count_equal = (valid_peer_values == target_value).sum()
    denominator = len(valid_peer_values)

    percentile = (count_less + 0.5 * count_equal) / denominator

    if not higher_is_better:
        percentile = 1.0 - percentile

    return _normalize_percentile(percentile)


def _deduplicate_player_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate to one row per player/team/season.

    Preference is given to rows with more minutes if available.
    """
    working_df = df.copy()

    sort_column = None
    for candidate in ["minutes_total", "minutes", "games_played", "games"]:
        if candidate in working_df.columns:
            sort_column = candidate
            working_df[candidate] = _safe_numeric(working_df[candidate])
            break

    if sort_column is not None:
        working_df = working_df.sort_values(
            by=sort_column,
            ascending=False,
            kind="stable",
        )

    working_df = working_df.drop_duplicates(subset=PLAYER_KEYS, keep="first")
    working_df = working_df.reset_index(drop=True)

    return working_df


# ============================================================================
# Input preparation
# ============================================================================

def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required input tables."""
    target_stats_df = _read_csv(TARGET_PLAYER_STATS_PATH)
    conference_pool_df = _read_csv(CONFERENCE_POOL_PATH)
    peer_groups_df = _read_csv(PLAYER_PEER_GROUPS_PATH)
    position_groups_df = _read_csv(PLAYER_POSITION_GROUPS_PATH)
    archetype_df = _read_csv(PLAYER_ARCHETYPE_ASSIGNMENT_PATH)

    target_stats_df = _standardize_text_columns(target_stats_df)
    conference_pool_df = _standardize_text_columns(conference_pool_df)
    peer_groups_df = _standardize_text_columns(peer_groups_df)
    position_groups_df = _standardize_text_columns(position_groups_df)
    archetype_df = _standardize_text_columns(archetype_df)

    target_stats_df = _coerce_key_columns(target_stats_df, PLAYER_KEYS)
    conference_pool_df = _coerce_key_columns(conference_pool_df, PLAYER_KEYS)
    position_groups_df = _coerce_key_columns(position_groups_df, PLAYER_KEYS)
    archetype_df = _coerce_key_columns(archetype_df, PLAYER_KEYS)

    _validate_required_columns(target_stats_df, REQUIRED_TARGET_COLUMNS, "player_stats_all_games_montana")
    _validate_required_columns(conference_pool_df, REQUIRED_POOL_COLUMNS, "conference_player_pool")
    _validate_required_columns(position_groups_df, REQUIRED_POSITION_COLUMNS, "player_position_groups")
    _validate_required_columns(archetype_df, REQUIRED_ARCHETYPE_BASE_COLUMNS, "player_archetype_assignment")

    return (
        target_stats_df,
        conference_pool_df,
        peer_groups_df,
        position_groups_df,
        archetype_df,
    )


def _prepare_archetype_table(archetype_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize archetype naming."""
    working_df = archetype_df.copy()

    archetype_column = _find_first_existing_column(
        working_df,
        ARCHETYPE_CANDIDATE_COLUMNS,
    )

    if archetype_column is None:
        raise ValueError(
            "player_archetype_assignment.csv does not contain a recognized archetype "
            f"column. Expected one of: {ARCHETYPE_CANDIDATE_COLUMNS}"
        )

    if archetype_column != "archetype":
        working_df = working_df.rename(columns={archetype_column: "archetype"})

    working_df = working_df[PLAYER_KEYS + ["archetype"]].drop_duplicates()

    return working_df


def _prepare_peer_groups(peer_groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize peer-group schema and keep only fields needed downstream.

    This avoids suffix clutter from bridge-table columns that overlap with the
    target and peer stat tables.
    """
    working_df = peer_groups_df.copy()

    rename_map = {}

    if "target_season" in working_df.columns and "season" not in working_df.columns:
        rename_map["target_season"] = "season"

    if "peer_season_match" in working_df.columns and "peer_season" not in working_df.columns:
        rename_map["peer_season_match"] = "peer_season"

    if rename_map:
        working_df = working_df.rename(columns=rename_map)

    _validate_required_columns(
        working_df,
        REQUIRED_PEER_GROUP_COLUMNS,
        "player_peer_groups",
    )

    if "season" not in working_df.columns:
        raise ValueError(
            "player_peer_groups.csv must include either 'season' or 'target_season'."
        )

    if "peer_season" not in working_df.columns:
        working_df["peer_season"] = working_df["season"]

    keep_columns = [
        "target_player_name",
        "target_team_name",
        "season",
        "peer_player_name",
        "peer_team_name",
        "peer_season",
    ]

    optional_columns = [
        "position_group",
        "usage_difference",
        "peer_group_rule",
    ]

    for column in optional_columns:
        if column in working_df.columns:
            keep_columns.append(column)

    working_df = working_df.loc[:, [column for column in keep_columns if column in working_df.columns]]

    key_like_columns = [
        "target_player_name",
        "target_team_name",
        "season",
        "peer_player_name",
        "peer_team_name",
        "peer_season",
    ]
    working_df = _coerce_key_columns(working_df, key_like_columns)

    working_df = working_df.dropna(
        subset=[
            "target_player_name",
            "target_team_name",
            "season",
            "peer_player_name",
            "peer_team_name",
            "peer_season",
        ]
    ).copy()

    working_df = working_df.drop_duplicates().reset_index(drop=True)

    return working_df


def _prepare_supporting_metadata(
    position_groups_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a clean player-level metadata table."""
    clean_position_df = (
        position_groups_df[PLAYER_KEYS + ["position_group"]]
        .drop_duplicates()
        .copy()
    )

    clean_archetype_df = _prepare_archetype_table(archetype_df)

    metadata_df = clean_position_df.merge(
        clean_archetype_df,
        on=PLAYER_KEYS,
        how="outer",
        validate="one_to_one",
    )

    return metadata_df


def _resolve_available_metrics(
    target_stats_df: pd.DataFrame,
    conference_pool_df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    """Keep only metrics that exist in both target and peer stat tables."""
    available_metrics = {
        metric_name: metric_info
        for metric_name, metric_info in METRIC_CONFIG.items()
        if metric_name in target_stats_df.columns and metric_name in conference_pool_df.columns
    }

    if not available_metrics:
        raise ValueError(
            "No configured percentile metrics were found in both target and peer tables."
        )

    return available_metrics


# ============================================================================
# Percentile build helpers
# ============================================================================

def _attach_target_and_peer_metrics(
    peer_groups_df: pd.DataFrame,
    target_stats_df: pd.DataFrame,
    conference_pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach target-player metrics and peer-player metrics from the proper source tables."""
    target_pool_df = target_stats_df.add_prefix("target_")
    peer_pool_df = conference_pool_df.add_prefix("peer_")

    merged_df = peer_groups_df.merge(
        target_pool_df,
        left_on=["target_player_name", "target_team_name", "season"],
        right_on=["target_player_name", "target_team_name", "target_season"],
        how="left",
        validate="many_to_one",
    )

    merged_df = merged_df.merge(
        peer_pool_df,
        left_on=["peer_player_name", "peer_team_name", "peer_season"],
        right_on=["peer_player_name", "peer_team_name", "peer_season"],
        how="left",
        validate="many_to_one",
    )

    return merged_df


def _build_target_base_table(
    target_stats_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the base Montana target-player table for final output."""
    base_df = target_stats_df.merge(
        metadata_df,
        on=PLAYER_KEYS,
        how="left",
        validate="one_to_one",
    )

    return base_df


def _compute_target_percentiles(
    peer_metrics_df: pd.DataFrame,
    target_base_df: pd.DataFrame,
    available_metrics: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Compute target-player percentiles across all available metrics."""
    output_rows: List[dict] = []

    grouped = peer_metrics_df.groupby(
        ["target_player_name", "target_team_name", "season"],
        dropna=False,
    )

    for (target_player_name, target_team_name, season), group_df in grouped:
        target_match_df = target_base_df[
            (target_base_df["player_name"] == target_player_name)
            & (target_base_df["team_name"] == target_team_name)
            & (target_base_df["season"] == season)
        ]

        if target_match_df.empty:
            continue

        target_row = target_match_df.iloc[0].to_dict()

        target_row["peer_group_size"] = int(
            group_df[["peer_player_name", "peer_team_name", "peer_season"]]
            .drop_duplicates()
            .shape[0]
        )

        if "usage_difference" in group_df.columns:
            target_row["avg_usage_difference"] = round(
                float(_safe_numeric(group_df["usage_difference"]).dropna().mean()),
                6,
            ) if not _safe_numeric(group_df["usage_difference"]).dropna().empty else np.nan

        for metric_name, metric_info in available_metrics.items():
            target_metric_column = f"target_{metric_name}"
            peer_metric_column = f"peer_{metric_name}"

            if (
                target_metric_column not in group_df.columns
                or peer_metric_column not in group_df.columns
            ):
                target_row[f"{metric_name}_percentile"] = np.nan
                continue

            target_metric_values = _safe_numeric(group_df[target_metric_column]).dropna()
            peer_metric_values = _safe_numeric(group_df[peer_metric_column]).dropna()

            if target_metric_values.empty or peer_metric_values.empty:
                target_row[f"{metric_name}_percentile"] = np.nan
                continue

            target_metric_value = float(target_metric_values.iloc[0])

            target_row[f"{metric_name}_percentile"] = _percentile_rank(
                peer_values=peer_metric_values,
                target_value=target_metric_value,
                higher_is_better=(metric_info["direction"] == "high"),
            )

        output_rows.append(target_row)

    percentiles_df = pd.DataFrame(output_rows)

    if percentiles_df.empty:
        raise ValueError(
            "No percentile rows were created. Check peer-group and stat-table merge keys."
        )

    return percentiles_df


def _add_category_percentiles(
    percentiles_df: pd.DataFrame,
    available_metrics: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Add category-level percentile averages."""
    working_df = percentiles_df.copy()

    category_map: Dict[str, List[str]] = {}

    for metric_name, metric_info in available_metrics.items():
        percentile_column = f"{metric_name}_percentile"
        if percentile_column not in working_df.columns:
            continue

        category = metric_info["category"]
        category_map.setdefault(category, []).append(percentile_column)

    for category, percentile_columns in category_map.items():
        working_df[f"{category}_percentile"] = (
            working_df[percentile_columns]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .round(6)
        )

    return working_df


def _finalize_output(
    percentiles_df: pd.DataFrame,
    available_metrics: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Finalize output ordering and duplicate protection."""
    working_df = percentiles_df.copy()

    working_df = working_df.drop_duplicates(subset=PLAYER_KEYS).reset_index(drop=True)

    identifier_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_raw",
        "position_group",
        "archetype",
    ]

    peer_summary_columns = [
        "peer_group_size",
        "avg_usage_difference",
    ]

    category_columns = [
        "shooting_percentile",
        "playmaking_percentile",
        "rebounding_percentile",
        "defense_percentile",
        "efficiency_percentile",
    ]

    raw_metric_columns = [
        metric_name for metric_name in available_metrics if metric_name in working_df.columns
    ]

    percentile_columns = [
        f"{metric_name}_percentile"
        for metric_name in available_metrics
        if f"{metric_name}_percentile" in working_df.columns
    ]

    remaining_columns = [
        column
        for column in working_df.columns
        if column not in set(
            identifier_columns
            + peer_summary_columns
            + category_columns
            + raw_metric_columns
            + percentile_columns
        )
    ]

    ordered_columns = (
        [column for column in identifier_columns if column in working_df.columns]
        + [column for column in peer_summary_columns if column in working_df.columns]
        + [column for column in category_columns if column in working_df.columns]
        + raw_metric_columns
        + percentile_columns
        + remaining_columns
    )

    working_df = working_df.loc[:, ordered_columns]

    sort_columns = [column for column in ["season", "team_name", "player_name"] if column in working_df.columns]
    if sort_columns:
        working_df = working_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    return working_df


# ============================================================================
# Main build function
# ============================================================================

def build_player_percentiles() -> pd.DataFrame:
    """Build the player percentile feature table."""
    (
        target_stats_df,
        conference_pool_df,
        peer_groups_df,
        position_groups_df,
        archetype_df,
    ) = _load_inputs()

    target_stats_df = _deduplicate_player_table(target_stats_df)
    conference_pool_df = _deduplicate_player_table(conference_pool_df)
    peer_groups_df = _prepare_peer_groups(peer_groups_df)
    metadata_df = _prepare_supporting_metadata(position_groups_df, archetype_df)

    available_metrics = _resolve_available_metrics(
        target_stats_df=target_stats_df,
        conference_pool_df=conference_pool_df,
    )

    peer_metrics_df = _attach_target_and_peer_metrics(
        peer_groups_df=peer_groups_df,
        target_stats_df=target_stats_df,
        conference_pool_df=conference_pool_df,
    )

    target_base_df = _build_target_base_table(
        target_stats_df=target_stats_df,
        metadata_df=metadata_df,
    )

    percentiles_df = _compute_target_percentiles(
        peer_metrics_df=peer_metrics_df,
        target_base_df=target_base_df,
        available_metrics=available_metrics,
    )

    percentiles_df = _add_category_percentiles(
        percentiles_df=percentiles_df,
        available_metrics=available_metrics,
    )

    percentiles_df = _finalize_output(
        percentiles_df=percentiles_df,
        available_metrics=available_metrics,
    )

    return percentiles_df


def main() -> None:
    """Run the percentile build pipeline and save the output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    player_percentiles_df = build_player_percentiles()
    player_percentiles_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Player percentile rows saved: {len(player_percentiles_df)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()