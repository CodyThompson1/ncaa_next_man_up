"""
Assign player archetypes for the NCAA Next Man Up project.

Purpose
-------
Build a reusable player-level archetype assignment dataset using explainable,
rule-based logic. Archetypes are assigned for both Montana players and Big Sky
comparison players, then written to a single master feature table.

Inputs
------
- data/processed/player_data/player_stats_all_games_montana.csv
- data/processed/comparison_sets/conference_player_pool.csv
- data/features/player_position_groups.csv

Output
------
- data/features/player_archetype_assignment.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

MONTANA_PLAYER_STATS_PATH = (
    REPO_ROOT / "data" / "processed" / "player_data" / "player_stats_all_games_montana.csv"
)
CONFERENCE_PLAYER_POOL_PATH = (
    REPO_ROOT / "data" / "processed" / "comparison_sets" / "conference_player_pool.csv"
)
PLAYER_POSITION_GROUPS_PATH = (
    REPO_ROOT / "data" / "features" / "player_position_groups.csv"
)
OUTPUT_PATH = REPO_ROOT / "data" / "features" / "player_archetype_assignment.csv"


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

KEY_COLUMNS = ["player_name", "team_name", "season"]

REQUIRED_POSITION_COLUMNS = KEY_COLUMNS + ["position_group"]

REQUIRED_STAT_COLUMNS = KEY_COLUMNS + [
    "conference_name",
    "position_raw",
]

NUMERIC_COLUMNS = [
    "games",
    "games_played",
    "games_started",
    "minutes",
    "minutes_total",
    "minutes_per_game",
    "pct_minutes",
    "pct_possessions",
    "pct_shots",
    "field_goals_made",
    "field_goals_attempted",
    "field_goal_pct",
    "three_points_made",
    "three_points_attempted",
    "three_point_pct",
    "effective_field_goal_pct",
    "free_throws_made",
    "free_throws_attempted",
    "free_throw_pct",
    "offensive_rebounds",
    "defensive_rebounds",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fouls",
    "points",
    "points_per_game",
    "ortg",
    "drtg",
    "efg_pct",
    "ts_pct",
    "or_pct",
    "dr_pct",
    "trb_pct",
    "assist_rate",
    "turnover_rate",
    "block_pct",
    "steal_pct",
    "fouls_committed_per_40",
    "fouls_drawn_per_40",
    "ft_rate",
    "ftm",
    "fta",
    "ft_pct",
    "two_pm",
    "two_pa",
    "two_pt_pct",
    "three_pm",
    "three_pa",
    "three_pt_pct",
    "usg_pct",
    "per",
    "pprod",
    "orb_pct",
    "drb_pct",
    "ast_pct",
    "stl_pct",
    "blk_pct",
    "tov_pct",
    "ows",
    "dws",
    "ws",
    "ws_per_40",
    "obpm",
    "dbpm",
    "bpm",
    "three_point_attempt_rate",
]

POSITION_GROUP_MAP = {
    "g": "Guard",
    "guard": "Guard",
    "guards": "Guard",
    "pg": "Guard",
    "sg": "Guard",
    "combo guard": "Guard",
    "wing": "Forward",
    "wings": "Forward",
    "f": "Forward",
    "forward": "Forward",
    "forwards": "Forward",
    "pf": "Forward",
    "sf": "Forward",
    "big": "Forward",
    "bigs": "Forward",
    "center": "Forward",
    "centers": "Forward",
    "c": "Forward",
}

VALID_POSITION_GROUPS = {"Guard", "Forward"}

GUARD_ARCHETYPES = [
    "Primary Creator",
    "Secondary Playmaker",
    "Off-Ball Shooter",
]

FORWARD_ARCHETYPES = [
    "Scoring Forward",
    "3-and-D Forward",
    "Interior Forward",
    "Glue Forward",
]

ARCHETYPE_PRIORITY = {
    "Guard": GUARD_ARCHETYPES,
    "Forward": FORWARD_ARCHETYPES,
}

ARCHETYPE_RULE_TEXT = {
    "Primary Creator": (
        "Assigned to high-creation guards driven by assist rate, usage, foul pressure, and offensive involvement."
    ),
    "Secondary Playmaker": (
        "Assigned to guards who provide balanced connective playmaking, efficiency, and secondary creation."
    ),
    "Off-Ball Shooter": (
        "Assigned to guards whose profile is driven by perimeter volume, three-point efficiency, and spacing tendency."
    ),
    "Scoring Forward": (
        "Assigned to forwards whose profile is driven by usage, scoring efficiency, free-throw pressure, and point production."
    ),
    "3-and-D Forward": (
        "Assigned to forwards whose profile is driven by perimeter shooting plus defensive event production."
    ),
    "Interior Forward": (
        "Assigned to forwards whose profile is driven by rebounding, interior activity, and rim-oriented impact."
    ),
    "Glue Forward": (
        "Assigned to forwards with broad all-around contribution across defense, rebounding, passing, and efficiency."
    ),
}


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def _validate_file_exists(path: Path) -> None:
    """Raise an error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file."""
    return pd.read_csv(path)


def _standardize_text(series: pd.Series) -> pd.Series:
    """Normalize text fields used in merge keys."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _standardize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize merge key columns."""
    df = df.copy()

    for column in ["player_name", "team_name"]:
        if column in df.columns:
            df[column] = _standardize_text(df[column])

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df


def _coerce_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric if they exist."""
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_name: str,
) -> None:
    """Validate that all required columns exist."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def _standardize_position_group(value: object) -> object:
    """Map raw position group values to Guard or Forward."""
    if pd.isna(value):
        return np.nan

    cleaned = str(value).strip().lower()
    mapped = POSITION_GROUP_MAP.get(cleaned)

    if mapped is not None:
        return mapped

    return str(value).strip().title()


def _sort_for_deduplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame so the best available row is retained when duplicates exist.
    Preference:
    - higher minutes_total
    - higher minutes
    - higher games_played
    - higher games
    """
    sort_columns = []
    ascending = []

    for column in ["minutes_total", "minutes", "games_played", "games"]:
        if column in df.columns:
            sort_columns.append(column)
            ascending.append(False)

    if sort_columns:
        return df.sort_values(by=sort_columns, ascending=ascending, na_position="last")

    return df


def _drop_duplicate_players(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop duplicate player rows on project keys."""
    df = df.copy()
    df = _sort_for_deduplication(df)

    before_count = len(df)
    df = df.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)
    after_count = len(df)

    if before_count != after_count:
        print(
            f"{dataset_name}: dropped {before_count - after_count} duplicate rows based on {KEY_COLUMNS}."
        )

    return df


def _safe_rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Convert a metric to percentile-style ranks within a position group.

    Missing values are filled with 0.50 so a missing metric does not fully
    zero out a player's archetype score.
    """
    if series.notna().sum() == 0:
        return pd.Series(np.full(len(series), 0.50), index=series.index)

    ranked = series.rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.50)


def _safe_value(row: pd.Series, column: str) -> float:
    """Return a float-safe value from a row, defaulting missing values to NaN."""
    value = row.get(column, np.nan)
    try:
        return float(value) if pd.notna(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _weighted_score(row: pd.Series, weights: Dict[str, float]) -> float:
    """Compute a weighted average score from ranked features."""
    numerator = 0.0
    denominator = 0.0

    for feature_name, weight in weights.items():
        value = row.get(feature_name, np.nan)
        if pd.notna(value):
            numerator += float(value) * weight
            denominator += weight

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _format_metric_value(value: float) -> str:
    """Format a numeric metric for human-readable reasons."""
    if pd.isna(value):
        return "NA"

    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"

    return f"{value:.3f}"


def _ensure_required_columns_exist(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """Create missing required columns as NaN if needed."""
    df = df.copy()

    for column in required_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df


# --------------------------------------------------------------------------------------
# Data loading and preparation
# --------------------------------------------------------------------------------------

def _load_input_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required input tables."""
    _validate_file_exists(MONTANA_PLAYER_STATS_PATH)
    _validate_file_exists(CONFERENCE_PLAYER_POOL_PATH)
    _validate_file_exists(PLAYER_POSITION_GROUPS_PATH)

    montana_df = _read_csv(MONTANA_PLAYER_STATS_PATH)
    conference_pool_df = _read_csv(CONFERENCE_PLAYER_POOL_PATH)
    position_groups_df = _read_csv(PLAYER_POSITION_GROUPS_PATH)

    _validate_required_columns(
        montana_df,
        REQUIRED_STAT_COLUMNS,
        "player_stats_all_games_montana.csv",
    )
    _validate_required_columns(
        conference_pool_df,
        REQUIRED_STAT_COLUMNS,
        "conference_player_pool.csv",
    )
    _validate_required_columns(
        position_groups_df,
        REQUIRED_POSITION_COLUMNS,
        "player_position_groups.csv",
    )

    montana_df = _standardize_merge_keys(montana_df)
    conference_pool_df = _standardize_merge_keys(conference_pool_df)
    position_groups_df = _standardize_merge_keys(position_groups_df)

    montana_df = _coerce_numeric_columns(montana_df, NUMERIC_COLUMNS)
    conference_pool_df = _coerce_numeric_columns(conference_pool_df, NUMERIC_COLUMNS)
    position_groups_df["position_group"] = position_groups_df["position_group"].map(
        _standardize_position_group
    )

    montana_df = _drop_duplicate_players(
        montana_df,
        "player_stats_all_games_montana.csv",
    )
    conference_pool_df = _drop_duplicate_players(
        conference_pool_df,
        "conference_player_pool.csv",
    )
    position_groups_df = _drop_duplicate_players(
        position_groups_df,
        "player_position_groups.csv",
    )

    return montana_df, conference_pool_df, position_groups_df


def _combine_player_pool(
    montana_df: pd.DataFrame,
    conference_pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine Montana season stats with the Big Sky comparison pool.

    The final archetype assignment file should contain:
    - Montana target players
    - Big Sky comparison players
    """
    montana_df = _ensure_required_columns_exist(montana_df, REQUIRED_STAT_COLUMNS + NUMERIC_COLUMNS)
    conference_pool_df = _ensure_required_columns_exist(
        conference_pool_df,
        REQUIRED_STAT_COLUMNS + NUMERIC_COLUMNS,
    )

    combined_df = pd.concat(
        [montana_df, conference_pool_df],
        ignore_index=True,
        sort=False,
    )

    combined_df["player_source_group"] = np.where(
        combined_df["team_name"].eq("Montana"),
        "Montana",
        "Big Sky Comparison",
    )

    combined_df = _drop_duplicate_players(
        combined_df,
        "combined_player_pool",
    )

    if combined_df.empty:
        raise ValueError("Combined player pool is empty after combining Montana and conference data.")

    return combined_df


def _prepare_player_base(
    combined_player_df: pd.DataFrame,
    position_groups_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the combined player pool with the engineered player position groups.

    The position_group from player_position_groups.csv is treated as authoritative.
    """
    player_df = combined_player_df.copy()
    position_subset = position_groups_df[KEY_COLUMNS + ["position_group"]].copy()

    player_df = player_df.merge(
        position_subset,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    missing_position_count = int(player_df["position_group"].isna().sum())
    if missing_position_count > 0:
        missing_examples = (
            player_df.loc[player_df["position_group"].isna(), KEY_COLUMNS]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{missing_position_count} players are missing position_group after merging "
            f"with player_position_groups.csv. Example rows: {missing_examples}"
        )

    player_df["position_group"] = player_df["position_group"].map(_standardize_position_group)

    invalid_position_groups = sorted(
        set(player_df["position_group"].dropna().unique()) - VALID_POSITION_GROUPS
    )
    if invalid_position_groups:
        raise ValueError(
            f"Unexpected position_group values found: {invalid_position_groups}"
        )

    return player_df


def _ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all archetype-relevant metric columns exist."""
    df = df.copy()

    needed_columns = [
        "games",
        "games_played",
        "minutes",
        "minutes_total",
        "minutes_per_game",
        "points_per_game",
        "pct_possessions",
        "usg_pct",
        "ast_pct",
        "assist_rate",
        "three_pa",
        "three_pt_pct",
        "three_point_attempt_rate",
        "ft_rate",
        "trb_pct",
        "orb_pct",
        "drb_pct",
        "blk_pct",
        "stl_pct",
        "ts_pct",
        "efg_pct",
        "ortg",
        "drtg",
        "pprod",
        "tov_pct",
    ]

    for column in needed_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df


def _build_rank_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build percentile-ranked features within each position group.

    Higher values are generally better, except for drtg and tov_pct where lower is better.
    """
    df = _ensure_metric_columns(player_df)

    higher_is_better_metrics = [
        "points_per_game",
        "pct_possessions",
        "usg_pct",
        "ast_pct",
        "assist_rate",
        "three_pa",
        "three_pt_pct",
        "three_point_attempt_rate",
        "ft_rate",
        "trb_pct",
        "orb_pct",
        "drb_pct",
        "blk_pct",
        "stl_pct",
        "ts_pct",
        "efg_pct",
        "ortg",
        "pprod",
    ]

    for metric in higher_is_better_metrics:
        df[f"{metric}_rank"] = (
            df.groupby("position_group", group_keys=False)[metric]
            .apply(_safe_rank_pct, ascending=True)
        )

    df["drtg_rank"] = (
        df.groupby("position_group", group_keys=False)["drtg"]
        .apply(_safe_rank_pct, ascending=False)
    )

    df["tov_pct_rank"] = (
        df.groupby("position_group", group_keys=False)["tov_pct"]
        .apply(_safe_rank_pct, ascending=False)
    )

    return df


# --------------------------------------------------------------------------------------
# Archetype scoring logic
# --------------------------------------------------------------------------------------

def _guard_archetype_scores(row: pd.Series) -> Dict[str, float]:
    """Score the three guard archetypes."""
    return {
        "Primary Creator": _weighted_score(
            row,
            {
                "ast_pct_rank": 0.30,
                "usg_pct_rank": 0.22,
                "assist_rate_rank": 0.18,
                "pct_possessions_rank": 0.12,
                "ft_rate_rank": 0.08,
                "ortg_rank": 0.05,
                "tov_pct_rank": 0.05,
            },
        ),
        "Secondary Playmaker": _weighted_score(
            row,
            {
                "ast_pct_rank": 0.22,
                "assist_rate_rank": 0.18,
                "ts_pct_rank": 0.15,
                "efg_pct_rank": 0.13,
                "stl_pct_rank": 0.10,
                "ortg_rank": 0.10,
                "tov_pct_rank": 0.07,
                "three_point_attempt_rate_rank": 0.05,
            },
        ),
        "Off-Ball Shooter": _weighted_score(
            row,
            {
                "three_pt_pct_rank": 0.34,
                "three_pa_rank": 0.24,
                "three_point_attempt_rate_rank": 0.22,
                "ts_pct_rank": 0.10,
                "efg_pct_rank": 0.06,
                "stl_pct_rank": 0.04,
            },
        ),
    }


def _forward_archetype_scores(row: pd.Series) -> Dict[str, float]:
    """Score the four forward archetypes."""
    return {
        "Scoring Forward": _weighted_score(
            row,
            {
                "usg_pct_rank": 0.22,
                "ts_pct_rank": 0.18,
                "efg_pct_rank": 0.14,
                "ft_rate_rank": 0.12,
                "pprod_rank": 0.12,
                "ortg_rank": 0.10,
                "points_per_game_rank": 0.12,
            },
        ),
        "3-and-D Forward": _weighted_score(
            row,
            {
                "three_pt_pct_rank": 0.24,
                "three_point_attempt_rate_rank": 0.20,
                "three_pa_rank": 0.12,
                "stl_pct_rank": 0.14,
                "blk_pct_rank": 0.14,
                "drtg_rank": 0.10,
                "ts_pct_rank": 0.06,
            },
        ),
        "Interior Forward": _weighted_score(
            row,
            {
                "orb_pct_rank": 0.22,
                "drb_pct_rank": 0.20,
                "trb_pct_rank": 0.20,
                "blk_pct_rank": 0.14,
                "ft_rate_rank": 0.10,
                "ortg_rank": 0.08,
                "efg_pct_rank": 0.06,
            },
        ),
        "Glue Forward": _weighted_score(
            row,
            {
                "ast_pct_rank": 0.12,
                "stl_pct_rank": 0.14,
                "drb_pct_rank": 0.14,
                "trb_pct_rank": 0.14,
                "ts_pct_rank": 0.12,
                "drtg_rank": 0.12,
                "ortg_rank": 0.12,
                "tov_pct_rank": 0.10,
            },
        ),
    }


def _get_archetype_scores(row: pd.Series) -> Dict[str, float]:
    """Return archetype score mapping based on the player's position group."""
    position_group = row["position_group"]

    if position_group == "Guard":
        return _guard_archetype_scores(row)

    if position_group == "Forward":
        return _forward_archetype_scores(row)

    raise ValueError(f"Unsupported position_group encountered: {position_group}")


def _reason_metric_candidates(archetype: str) -> List[str]:
    """Return the most relevant raw metrics for a given archetype."""
    metric_map = {
        "Primary Creator": [
            "ast_pct",
            "usg_pct",
            "assist_rate",
            "pct_possessions",
            "ft_rate",
        ],
        "Secondary Playmaker": [
            "ast_pct",
            "assist_rate",
            "ts_pct",
            "efg_pct",
            "stl_pct",
            "tov_pct",
        ],
        "Off-Ball Shooter": [
            "three_pt_pct",
            "three_pa",
            "three_point_attempt_rate",
            "ts_pct",
            "efg_pct",
        ],
        "Scoring Forward": [
            "usg_pct",
            "ts_pct",
            "efg_pct",
            "ft_rate",
            "pprod",
            "points_per_game",
        ],
        "3-and-D Forward": [
            "three_pt_pct",
            "three_pa",
            "three_point_attempt_rate",
            "stl_pct",
            "blk_pct",
            "drtg",
        ],
        "Interior Forward": [
            "orb_pct",
            "drb_pct",
            "trb_pct",
            "blk_pct",
            "ft_rate",
            "ortg",
        ],
        "Glue Forward": [
            "ast_pct",
            "stl_pct",
            "trb_pct",
            "drb_pct",
            "drtg",
            "ts_pct",
            "tov_pct",
        ],
    }

    return metric_map.get(archetype, [])


def _rank_column_for_metric(metric_name: str) -> str:
    """Return the corresponding ranked feature name for a raw metric."""
    if metric_name == "drtg":
        return "drtg_rank"
    if metric_name == "tov_pct":
        return "tov_pct_rank"
    return f"{metric_name}_rank"


def _build_archetype_reason(row: pd.Series, archetype: str, position_group: str) -> str:
    """Build a concise reason string explaining the assignment."""
    candidates = _reason_metric_candidates(archetype)

    ranked_metrics = []
    for metric in candidates:
        rank_column = _rank_column_for_metric(metric)
        raw_value = _safe_value(row, metric)
        rank_value = row.get(rank_column, np.nan)

        if pd.notna(raw_value) and pd.notna(rank_value):
            ranked_metrics.append((metric, float(rank_value), raw_value))

    ranked_metrics.sort(key=lambda item: item[1], reverse=True)
    top_metrics = ranked_metrics[:3]

    if not top_metrics:
        return f"Assigned as {archetype} using the {position_group} archetype rule set."

    metric_text = ", ".join(
        f"{metric}={_format_metric_value(raw_value)}"
        for metric, _, raw_value in top_metrics
    )

    return (
        f"Assigned as {archetype} because the strongest aligned {position_group.lower()} "
        f"indicators were {metric_text}."
    )


def _score_column_name(archetype_name: str) -> str:
    """Create a safe column name for an archetype score."""
    cleaned = archetype_name.lower().replace("-", "").replace(" ", "_")
    return f"score_{cleaned}"


def _assign_single_archetype(row: pd.Series) -> pd.Series:
    """Assign exactly one archetype to a player row."""
    score_map = _get_archetype_scores(row)
    position_group = row["position_group"]
    archetype_order = ARCHETYPE_PRIORITY[position_group]

    best_archetype = max(
        archetype_order,
        key=lambda archetype_name: score_map[archetype_name],
    )
    best_score = float(score_map[best_archetype])

    archetype_reason = _build_archetype_reason(
        row=row,
        archetype=best_archetype,
        position_group=position_group,
    )
    archetype_rule = ARCHETYPE_RULE_TEXT[best_archetype]

    output = {
        "archetype": best_archetype,
        "archetype_reason": archetype_reason,
        "archetype_rule": archetype_rule,
        "archetype_score": round(best_score, 6),
    }

    for archetype_name, score_value in score_map.items():
        output[_score_column_name(archetype_name)] = round(float(score_value), 6)

    return pd.Series(output)


# --------------------------------------------------------------------------------------
# Final dataset builder
# --------------------------------------------------------------------------------------

def _build_player_archetype_assignment(player_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final player archetype assignment table."""
    ranked_df = _build_rank_features(player_df)
    assignment_df = ranked_df.apply(_assign_single_archetype, axis=1)
    final_df = pd.concat([ranked_df, assignment_df], axis=1)

    preferred_output_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "player_source_group",
        "position_raw",
        "position_group",
        "games_played",
        "minutes_total",
        "minutes_per_game",
        "points_per_game",
        "pct_possessions",
        "usg_pct",
        "ast_pct",
        "assist_rate",
        "three_pa",
        "three_pt_pct",
        "three_point_attempt_rate",
        "ft_rate",
        "trb_pct",
        "orb_pct",
        "drb_pct",
        "blk_pct",
        "stl_pct",
        "ts_pct",
        "efg_pct",
        "ortg",
        "drtg",
        "pprod",
        "tov_pct",
        "archetype",
        "archetype_reason",
        "archetype_rule",
        "archetype_score",
        "score_primary_creator",
        "score_secondary_playmaker",
        "score_offball_shooter",
        "score_scoring_forward",
        "score_3andd_forward",
        "score_interior_forward",
        "score_glue_forward",
    ]

    existing_columns = [column for column in preferred_output_columns if column in final_df.columns]
    final_df = final_df[existing_columns].copy()

    final_df = _drop_duplicate_players(
        final_df,
        "player_archetype_assignment",
    )

    if final_df["archetype"].isna().any():
        missing_count = int(final_df["archetype"].isna().sum())
        raise ValueError(
            f"Final archetype assignment contains {missing_count} missing archetype values."
        )

    if final_df.duplicated(subset=KEY_COLUMNS).any():
        duplicate_count = int(final_df.duplicated(subset=KEY_COLUMNS).sum())
        raise ValueError(
            f"Final archetype assignment contains {duplicate_count} duplicate player rows."
        )

    return final_df


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write the final feature dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the full player archetype assignment pipeline."""
    montana_df, conference_pool_df, position_groups_df = _load_input_tables()
    combined_player_df = _combine_player_pool(
        montana_df=montana_df,
        conference_pool_df=conference_pool_df,
    )
    player_df = _prepare_player_base(
        combined_player_df=combined_player_df,
        position_groups_df=position_groups_df,
    )
    archetype_assignment_df = _build_player_archetype_assignment(player_df)
    _write_output(archetype_assignment_df, OUTPUT_PATH)

    print(f"Player archetype rows saved: {len(archetype_assignment_df)}")
    print(f"Montana player rows: {(archetype_assignment_df['team_name'] == 'Montana').sum()}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()