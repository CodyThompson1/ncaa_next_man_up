"""
File: build_player_peer_groups.py

Build player peer groups for the NCAA Next Man Up project.

Purpose:
- Create a reusable long-format bridge table that maps Montana target players
  to valid Big Sky peer players.
- Use position-group matching and a usage-based similarity window to define
  peer groups.
- Support downstream percentile, archetype, and evaluation scripts.

Important project note:
- `conference_player_pool.csv` is treated as the Big Sky peer pool.
- Montana target players are sourced from
  `data/processed/player_data/player_stats_all_games_montana.csv`.
- Position groups are sourced from `data/features/player_position_groups.csv`.

Inputs:
- data/processed/comparison_sets/conference_player_pool.csv
- data/processed/player_data/player_stats_all_games_montana.csv
- data/features/player_position_groups.csv

Output:
- data/features/player_peer_groups.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFERENCE_PLAYER_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comparison_sets"
    / "conference_player_pool.csv"
)

MONTANA_PLAYER_STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_data"
    / "player_stats_all_games_montana.csv"
)

PLAYER_POSITION_GROUPS_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "player_position_groups.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "player_peer_groups.csv"
)


# =========================
# CONFIGURATION
# =========================

USAGE_WINDOW = 5.0
MIN_MINUTES_PLAYED = 200
MIN_GAMES_PLAYED = 5

REQUIRED_PEER_POOL_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "conference_name",
    "usg_pct",
]

REQUIRED_TARGET_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "usg_pct",
]

REQUIRED_POSITION_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_group",
]

VALID_POSITION_GROUPS = {"Guard", "Forward"}

STANDARD_OUTPUT_COLUMNS = [
    "target_player_name",
    "peer_player_name",
    "target_team_name",
    "peer_team_name",
    "season",
    "position_group",
    "target_usg_pct",
    "peer_usg_pct",
    "usage_difference",
    "target_conference_name",
    "peer_conference_name",
    "target_minutes_played",
    "peer_minutes_played",
    "target_games_played",
    "peer_games_played",
    "peer_group_rule",
]


# =========================
# VALIDATION HELPERS
# =========================

def _validate_file_exists(path: Path) -> None:
    """Validate that a required input file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    path: Path,
) -> None:
    """Validate that the DataFrame contains required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")


def _standardize_text(series: pd.Series) -> pd.Series:
    """Standardize text-like columns."""
    return series.astype("string").str.strip().replace("", pd.NA)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric."""
    return pd.to_numeric(series, errors="coerce")


# =========================
# COLUMN RESOLUTION HELPERS
# =========================

def _get_first_existing_column(df: pd.DataFrame, candidate_columns: list[str]) -> str | None:
    """Return the first candidate column that exists in the DataFrame."""
    for column in candidate_columns:
        if column in df.columns:
            return column
    return None


def _add_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standardized minutes and games columns using the best available source columns.
    """
    minutes_column = _get_first_existing_column(
        df,
        [
            "minutes_played",
            "minutes",
            "mp",
            "min",
            "total_minutes",
            "mins",
        ],
    )

    games_column = _get_first_existing_column(
        df,
        [
            "games_played",
            "games",
            "g",
        ],
    )

    df["minutes_played_std"] = (
        _coerce_numeric(df[minutes_column]) if minutes_column else pd.NA
    )
    df["games_played_std"] = (
        _coerce_numeric(df[games_column]) if games_column else pd.NA
    )

    df["minutes_source_column"] = minutes_column if minutes_column else pd.NA
    df["games_source_column"] = games_column if games_column else pd.NA

    return df


def _apply_sample_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to players with sufficient playing time.

    Rule:
    - Keep players if they meet either the minutes threshold or the games threshold.
    """
    minutes_ok = df["minutes_played_std"].ge(MIN_MINUTES_PLAYED).fillna(False)
    games_ok = df["games_played_std"].ge(MIN_GAMES_PLAYED).fillna(False)

    filtered_df = df.loc[minutes_ok | games_ok].copy()

    if filtered_df.empty:
        raise ValueError(
            "All rows were removed by the minimum sample thresholds. "
            "Check minutes/games fields and threshold settings."
        )

    return filtered_df


# =========================
# LOADERS
# =========================

def _load_conference_player_pool(path: Path) -> pd.DataFrame:
    """
    Load and standardize the Big Sky peer pool input.
    """
    _validate_file_exists(path)

    df = pd.read_csv(path)
    _validate_required_columns(df, REQUIRED_PEER_POOL_COLUMNS, path)

    for column in ["player_name", "team_name", "conference_name"]:
        df[column] = _standardize_text(df[column])

    df["season"] = _coerce_numeric(df["season"]).astype("Int64")
    df["usg_pct"] = _coerce_numeric(df["usg_pct"])

    df = _add_metric_columns(df)

    required_not_null = ["player_name", "team_name", "season", "conference_name", "usg_pct"]
    for column in required_not_null:
        if df[column].isna().any():
            invalid_rows = df.loc[
                df[column].isna(),
                required_not_null,
            ].head(10)
            raise ValueError(
                f"Null values found in required peer-pool field `{column}`.\n"
                f"Example invalid rows:\n{invalid_rows.to_string(index=False)}"
            )

    return df


def _load_montana_target_stats(path: Path) -> pd.DataFrame:
    """
    Load and standardize Montana target player stats.

    This is the Montana target table because conference_player_pool.csv
    may only contain the Big Sky comparison pool.
    """
    _validate_file_exists(path)

    df = pd.read_csv(path)
    _validate_required_columns(df, REQUIRED_TARGET_COLUMNS, path)

    for column in ["player_name", "team_name"]:
        df[column] = _standardize_text(df[column])

    df["season"] = _coerce_numeric(df["season"]).astype("Int64")
    df["usg_pct"] = _coerce_numeric(df["usg_pct"])

    df = _add_metric_columns(df)

    if "conference_name" not in df.columns:
        df["conference_name"] = "Big Sky"
    else:
        df["conference_name"] = _standardize_text(df["conference_name"]).fillna("Big Sky")

    required_not_null = ["player_name", "team_name", "season", "conference_name", "usg_pct"]
    for column in required_not_null:
        if df[column].isna().any():
            invalid_rows = df.loc[
                df[column].isna(),
                required_not_null,
            ].head(10)
            raise ValueError(
                f"Null values found in required Montana target field `{column}`.\n"
                f"Example invalid rows:\n{invalid_rows.to_string(index=False)}"
            )

    return df


def _load_position_groups(path: Path) -> pd.DataFrame:
    """Load and validate the player position groups input."""
    _validate_file_exists(path)

    df = pd.read_csv(path)
    _validate_required_columns(df, REQUIRED_POSITION_COLUMNS, path)

    for column in ["player_name", "team_name", "position_group"]:
        df[column] = _standardize_text(df[column])

    df["season"] = _coerce_numeric(df["season"]).astype("Int64")

    if df["position_group"].isna().any():
        invalid_rows = df.loc[
            df["position_group"].isna(),
            ["player_name", "team_name", "season"],
        ].head(10)
        raise ValueError(
            "Null position_group values found in player_position_groups input.\n"
            f"Example invalid rows:\n{invalid_rows.to_string(index=False)}"
        )

    invalid_groups = set(df["position_group"].dropna().unique()) - VALID_POSITION_GROUPS
    if invalid_groups:
        raise ValueError(
            f"Unexpected position_group values found: {sorted(invalid_groups)}"
        )

    df = (
        df.sort_values(by=["player_name", "team_name", "season"])
        .drop_duplicates(subset=["player_name", "team_name", "season"], keep="first")
        .copy()
    )

    return df


# =========================
# MERGE HELPERS
# =========================

def _merge_position_groups(
    player_df: pd.DataFrame,
    position_groups_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Attach position groups to a player table."""
    merged_df = player_df.merge(
        position_groups_df[
            ["player_name", "team_name", "season", "position_group"]
        ],
        on=["player_name", "team_name", "season"],
        how="left",
        validate="many_to_one",
    )

    if merged_df["position_group"].isna().any():
        missing_df = merged_df.loc[
            merged_df["position_group"].isna(),
            ["player_name", "team_name", "season", "conference_name"],
        ].drop_duplicates()

        raise ValueError(
            f"Some players in {dataset_name} could not be matched to "
            f"player_position_groups.csv.\n"
            f"Example missing matches:\n{missing_df.head(15).to_string(index=False)}"
        )

    return merged_df


def _deduplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate player rows at the player/team/season level.
    """
    df = df.copy()

    df["_usg_present"] = df["usg_pct"].notna().astype(int)
    df["_minutes_present"] = df["minutes_played_std"].notna().astype(int)
    df["_games_present"] = df["games_played_std"].notna().astype(int)

    df = df.sort_values(
        by=[
            "player_name",
            "team_name",
            "season",
            "_usg_present",
            "_minutes_present",
            "_games_present",
        ],
        ascending=[True, True, True, False, False, False],
    )

    df = df.drop_duplicates(
        subset=["player_name", "team_name", "season"],
        keep="first",
    ).copy()

    df = df.drop(columns=["_usg_present", "_minutes_present", "_games_present"])
    return df


# =========================
# TARGET / PEER BUILD
# =========================

def _prepare_targets_and_peers(
    montana_df: pd.DataFrame,
    peer_pool_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the final target and peer tables after thresholds and de-duplication.
    """
    montana_df = _apply_sample_thresholds(montana_df)
    peer_pool_df = _apply_sample_thresholds(peer_pool_df)

    montana_df = _deduplicate_players(montana_df)
    peer_pool_df = _deduplicate_players(peer_pool_df)

    if montana_df.empty:
        raise ValueError("No Montana target players remain after filtering.")

    if peer_pool_df.empty:
        raise ValueError("No Big Sky peer players remain after filtering.")

    return montana_df, peer_pool_df


def _build_peer_bridge(
    targets_df: pd.DataFrame,
    peers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the long-format target-to-peer bridge table.

    Matching rules:
    - same season
    - same position_group
    - peer usage within target usage ± USAGE_WINDOW
    - exclude self-matches
    """
    target_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "usg_pct",
        "minutes_played_std",
        "games_played_std",
    ]
    peer_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "usg_pct",
        "minutes_played_std",
        "games_played_std",
    ]

    targets = targets_df[target_columns].rename(
        columns={
            "player_name": "target_player_name",
            "team_name": "target_team_name",
            "conference_name": "target_conference_name",
            "position_group": "target_position_group",
            "usg_pct": "target_usg_pct",
            "minutes_played_std": "target_minutes_played",
            "games_played_std": "target_games_played",
        }
    )

    peers = peers_df[peer_columns].rename(
        columns={
            "player_name": "peer_player_name",
            "team_name": "peer_team_name",
            "conference_name": "peer_conference_name",
            "position_group": "peer_position_group",
            "usg_pct": "peer_usg_pct",
            "minutes_played_std": "peer_minutes_played",
            "games_played_std": "peer_games_played",
        }
    )

    merged_df = targets.merge(
        peers,
        on="season",
        how="inner",
    )

    merged_df = merged_df.loc[
        merged_df["target_position_group"] == merged_df["peer_position_group"]
    ].copy()

    merged_df["usage_difference"] = (
        merged_df["peer_usg_pct"] - merged_df["target_usg_pct"]
    ).abs()

    merged_df = merged_df.loc[
        merged_df["usage_difference"] <= USAGE_WINDOW
    ].copy()

    merged_df = merged_df.loc[
        ~(
            (merged_df["target_player_name"] == merged_df["peer_player_name"])
            & (merged_df["target_team_name"] == merged_df["peer_team_name"])
        )
    ].copy()

    merged_df["position_group"] = merged_df["target_position_group"]
    merged_df["peer_group_rule"] = (
        f"same_position_group_and_usage_within_{USAGE_WINDOW:g}_pct"
    )

    output_df = merged_df[
        [
            "target_player_name",
            "peer_player_name",
            "target_team_name",
            "peer_team_name",
            "season",
            "position_group",
            "target_usg_pct",
            "peer_usg_pct",
            "usage_difference",
            "target_conference_name",
            "peer_conference_name",
            "target_minutes_played",
            "peer_minutes_played",
            "target_games_played",
            "peer_games_played",
            "peer_group_rule",
        ]
    ].copy()

    output_df = output_df.sort_values(
        by=[
            "season",
            "target_team_name",
            "target_player_name",
            "usage_difference",
            "peer_team_name",
            "peer_player_name",
        ],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)

    return output_df


# =========================
# OUTPUT VALIDATION
# =========================

def _validate_output(df: pd.DataFrame) -> None:
    """Validate the engineered output before writing to disk."""
    if df.empty:
        raise ValueError(
            "Peer-group output is empty. No valid player peer matches were created."
        )

    null_check_columns = [
        "target_player_name",
        "peer_player_name",
        "target_team_name",
        "peer_team_name",
        "season",
        "position_group",
        "target_usg_pct",
        "peer_usg_pct",
        "usage_difference",
    ]

    for column in null_check_columns:
        if df[column].isna().any():
            invalid_rows = df.loc[
                df[column].isna(),
                null_check_columns,
            ].head(10)
            raise ValueError(
                f"Null values found in required output column `{column}`.\n"
                f"Example invalid rows:\n{invalid_rows.to_string(index=False)}"
            )

    invalid_groups = set(df["position_group"].dropna().unique()) - VALID_POSITION_GROUPS
    if invalid_groups:
        raise ValueError(
            f"Invalid position_group values in output: {sorted(invalid_groups)}"
        )

    self_matches = df.loc[
        (df["target_player_name"] == df["peer_player_name"])
        & (df["target_team_name"] == df["peer_team_name"])
    ]
    if not self_matches.empty:
        raise ValueError(
            "Self-matches were found in the output, which should be excluded."
        )

    duplicate_count = df.duplicated(
        subset=[
            "target_player_name",
            "peer_player_name",
            "target_team_name",
            "peer_team_name",
            "season",
        ]
    ).sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Duplicate target-peer rows remain in output: {duplicate_count}"
        )


# =========================
# MAIN BUILD
# =========================

def build_player_peer_groups() -> pd.DataFrame:
    """
    Build the reusable player peer-group bridge table.
    """
    peer_pool_df = _load_conference_player_pool(CONFERENCE_PLAYER_POOL_PATH)
    montana_targets_df = _load_montana_target_stats(MONTANA_PLAYER_STATS_PATH)
    position_groups_df = _load_position_groups(PLAYER_POSITION_GROUPS_PATH)

    peer_pool_df = _merge_position_groups(
        peer_pool_df,
        position_groups_df,
        dataset_name="conference_player_pool.csv",
    )

    montana_targets_df = _merge_position_groups(
        montana_targets_df,
        position_groups_df,
        dataset_name="player_stats_all_games_montana.csv",
    )

    montana_targets_df, peer_pool_df = _prepare_targets_and_peers(
        montana_df=montana_targets_df,
        peer_pool_df=peer_pool_df,
    )

    peer_groups_df = _build_peer_bridge(
        targets_df=montana_targets_df,
        peers_df=peer_pool_df,
    )

    peer_groups_df = peer_groups_df[STANDARD_OUTPUT_COLUMNS].copy()

    _validate_output(peer_groups_df)
    return peer_groups_df


def main() -> None:
    """Run the player peer-group feature engineering pipeline."""
    peer_groups_df = build_player_peer_groups()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    peer_groups_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows written: {len(peer_groups_df):,}")
    print("Top target peer counts:")
    print(
        peer_groups_df.groupby(
            ["target_player_name", "position_group"],
            dropna=False,
        ).size().sort_values(ascending=False).head(15).to_string()
    )


if __name__ == "__main__":
    main()