"""
File: build_player_peer_groups.py
Last Modified: 2026-03-01
Purpose: Build a reusable long-format player peer group bridge table for the NCAA Next Man Up project by matching players within the Big Sky comparison pool based on season, standardized position group, and a usage window of ± 5%, while applying minimum participation thresholds to reduce tiny samples for downstream percentile scoring and evaluation engine workflows.

Inputs:
- data/processed/comparison_sets/conference_player_pool.csv
- data/features/player_position_groups.csv

Outputs:
- data/features/player_peer_groups.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFERENCE_PLAYER_POOL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "comparison_sets" / "conference_player_pool.csv"
)
PLAYER_POSITION_GROUPS_PATH = (
    PROJECT_ROOT / "data" / "features" / "player_position_groups.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "player_peer_groups.csv"

REQUIRED_POOL_COLUMNS = {
    "player_name",
    "team_name",
    "season",
    "conference_name",
}
REQUIRED_POSITION_COLUMNS = {
    "player_name",
    "team_name",
    "season",
    "position_group",
}

USAGE_CANDIDATE_COLUMNS = [
    "usg_pct",
    "usage_pct",
    "usage_rate",
    "usage",
]

MINUTES_CANDIDATE_COLUMNS = [
    "minutes_played",
    "minutes",
    "mp",
    "total_minutes",
]

GAMES_CANDIDATE_COLUMNS = [
    "games_played",
    "games",
    "g",
]

TARGET_CONFERENCE_NAME = "Big Sky"
USAGE_WINDOW = 5.0
MIN_MINUTES_PLAYED = 200.0
MIN_GAMES_PLAYED = 8

OUTPUT_COLUMNS = [
    "target_player_name",
    "target_team_name",
    "peer_player_name",
    "peer_team_name",
    "season",
    "position_group",
    "target_usage",
    "peer_usage",
    "usage_difference",
    "usage_window",
    "conference_name",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Input file is missing required columns {sorted(missing_columns)}: {path}"
        )


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _standardize_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(_clean_text)
            df[column] = df[column].replace("", pd.NA)
    return df


def _find_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find a valid {label} column. Looked for: {candidates}"
    )


def _prepare_conference_player_pool(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_POOL_COLUMNS, CONFERENCE_PLAYER_POOL_PATH)

    df = df.copy()
    df = _standardize_text_columns(
        df,
        ["player_name", "team_name", "conference_name", "position_raw"],
    )

    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    usage_column = _find_first_existing_column(df, USAGE_CANDIDATE_COLUMNS, "usage")
    minutes_column = _find_first_existing_column(df, MINUTES_CANDIDATE_COLUMNS, "minutes")
    games_column = _find_first_existing_column(df, GAMES_CANDIDATE_COLUMNS, "games")

    df["usage_metric"] = pd.to_numeric(df[usage_column], errors="coerce")
    df["minutes_metric"] = pd.to_numeric(df[minutes_column], errors="coerce")
    df["games_metric"] = pd.to_numeric(df[games_column], errors="coerce")

    df = df[df["conference_name"].str.lower() == TARGET_CONFERENCE_NAME.lower()].copy()

    df = df.dropna(subset=["player_name", "team_name", "season", "usage_metric"])
    df = df[
        (df["minutes_metric"].fillna(0) >= MIN_MINUTES_PLAYED)
        | (df["games_metric"].fillna(0) >= MIN_GAMES_PLAYED)
    ].copy()

    df = df.drop_duplicates(
        subset=["player_name", "team_name", "season"],
        keep="first",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Conference player pool is empty after applying conference and participation filters."
        )

    return df


def _prepare_position_groups(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, REQUIRED_POSITION_COLUMNS, PLAYER_POSITION_GROUPS_PATH)

    df = df.copy()
    df = _standardize_text_columns(
        df,
        ["player_name", "team_name", "position_group", "position_raw"],
    )
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["player_name", "team_name", "season", "position_group"])
    df = df.drop_duplicates(
        subset=["player_name", "team_name", "season"],
        keep="first",
    ).reset_index(drop=True)

    allowed_groups = {"Guard", "Forward"}
    invalid_groups = set(df["position_group"].dropna().unique()) - allowed_groups
    if invalid_groups:
        raise ValueError(f"Invalid position_group values found: {sorted(invalid_groups)}")

    if df.empty:
        raise ValueError("Player position groups input is empty after cleaning.")

    return df


def _merge_position_groups(
    conference_pool_df: pd.DataFrame,
    position_groups_df: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["player_name", "team_name", "season"]

    merged_df = conference_pool_df.merge(
        position_groups_df[merge_keys + ["position_group"]],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    missing_position_group_df = merged_df[merged_df["position_group"].isna()][
        ["player_name", "team_name", "season"]
    ].drop_duplicates()

    if not missing_position_group_df.empty:
        raise ValueError(
            "Some players in conference_player_pool.csv could not be matched to "
            "player_position_groups.csv:\n"
            f"{missing_position_group_df.to_string(index=False)}"
        )

    return merged_df


def _build_peer_groups(player_df: pd.DataFrame) -> pd.DataFrame:
    target_df = player_df[
        ["player_name", "team_name", "season", "conference_name", "position_group", "usage_metric"]
    ].copy()
    peer_df = target_df.copy()

    target_df = target_df.rename(
        columns={
            "player_name": "target_player_name",
            "team_name": "target_team_name",
            "usage_metric": "target_usage",
        }
    )
    peer_df = peer_df.rename(
        columns={
            "player_name": "peer_player_name",
            "team_name": "peer_team_name",
            "usage_metric": "peer_usage",
        }
    )

    merged_peer_df = target_df.merge(
        peer_df,
        on=["season", "conference_name", "position_group"],
        how="inner",
        validate="many_to_many",
    )

    merged_peer_df["usage_difference"] = (
        merged_peer_df["target_usage"] - merged_peer_df["peer_usage"]
    ).abs()
    merged_peer_df["usage_window"] = USAGE_WINDOW

    merged_peer_df = merged_peer_df[
        merged_peer_df["usage_difference"] <= USAGE_WINDOW
    ].copy()

    merged_peer_df = merged_peer_df[
        ~(
            (merged_peer_df["target_player_name"] == merged_peer_df["peer_player_name"])
            & (merged_peer_df["target_team_name"] == merged_peer_df["peer_team_name"])
            & (merged_peer_df["season"] == merged_peer_df["season"])
        )
    ].copy()

    if merged_peer_df.empty:
        raise ValueError(
            "No peer groups were created. Review usage values, position groups, or thresholds."
        )

    merged_peer_df = merged_peer_df[OUTPUT_COLUMNS].copy()
    merged_peer_df = merged_peer_df.sort_values(
        by=[
            "target_team_name",
            "target_player_name",
            "season",
            "position_group",
            "usage_difference",
            "peer_team_name",
            "peer_player_name",
        ]
    ).reset_index(drop=True)

    return merged_peer_df


def _validate_output(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Output dataframe is empty.")

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Output dataframe is missing required columns: {missing_columns}")

    for column in [
        "target_player_name",
        "target_team_name",
        "peer_player_name",
        "peer_team_name",
        "season",
        "position_group",
        "target_usage",
        "peer_usage",
        "usage_difference",
    ]:
        if df[column].isna().any():
            raise ValueError(f"Null values found in required output column: {column}")

    invalid_groups = set(df["position_group"].unique()) - {"Guard", "Forward"}
    if invalid_groups:
        raise ValueError(f"Invalid position_group values in output: {sorted(invalid_groups)}")

    if (df["usage_difference"] > USAGE_WINDOW).any():
        raise ValueError("Found usage_difference values outside the allowed usage window.")

    exact_self_matches = df[
        (df["target_player_name"] == df["peer_player_name"])
        & (df["target_team_name"] == df["peer_team_name"])
    ]
    if not exact_self_matches.empty:
        raise ValueError("Output contains self-matches, which should have been excluded.")

    peer_counts = (
        df.groupby(["target_player_name", "target_team_name", "season"])
        .size()
        .reset_index(name="peer_count")
    )
    zero_peer_targets = peer_counts[peer_counts["peer_count"] <= 0]
    if not zero_peer_targets.empty:
        raise ValueError("Some target players have zero peers after filtering.")


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    conference_player_pool_df = _read_csv(CONFERENCE_PLAYER_POOL_PATH)
    player_position_groups_df = _read_csv(PLAYER_POSITION_GROUPS_PATH)

    conference_player_pool_df = _prepare_conference_player_pool(conference_player_pool_df)
    player_position_groups_df = _prepare_position_groups(player_position_groups_df)

    merged_player_df = _merge_position_groups(
        conference_player_pool_df=conference_player_pool_df,
        position_groups_df=player_position_groups_df,
    )

    player_peer_groups_df = _build_peer_groups(merged_player_df)
    _validate_output(player_peer_groups_df)
    save_output(player_peer_groups_df, OUTPUT_PATH)

    print(f"Player peer groups saved: {len(player_peer_groups_df)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()