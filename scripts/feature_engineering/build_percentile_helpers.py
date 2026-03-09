"""
File: build_percentile_helpers.py
Last Modified: 2026-03-02
Purpose: Generate player metric percentiles within peer groups for downstream evaluation use.

Inputs:
- data/processed/player_stats_all_games.csv
- data/features/player_peer_groups.csv

Outputs:
- data/features/player_percentiles.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_KEYS = ["conference", "position_group"]
PLAYER_KEYS = ["player", "team", "conference", "position_group"]


METRIC_CANDIDATES = {
    "shooting": [
        "shooting",
        "shooting_score",
        "shooting_rating",
        "shooting_index",
        "three_point_pct",
        "three_pt_pct",
        "fg3_pct",
        "effective_fg_pct",
        "efg_pct",
        "ts_pct",
    ],
    "playmaking": [
        "playmaking",
        "playmaking_score",
        "playmaking_rating",
        "assist_to_turnover_ratio",
        "ast_to_ratio",
        "assist_rate",
        "assist_pct",
        "ast_pct",
        "ast_per_game",
    ],
    "rebounding": [
        "rebounding",
        "rebounding_score",
        "rebounding_rating",
        "rebound_rate",
        "reb_pct",
        "trb_pct",
        "total_rebound_pct",
        "rebounds_per_game",
        "reb_per_game",
    ],
    "defense": [
        "defense",
        "defense_score",
        "defensive_rating",
        "defensive_activity",
        "steal_block_rate",
        "stocks_rate",
        "stl_blk_rate",
        "steal_rate",
        "stl_pct",
        "block_rate",
        "blk_pct",
    ],
    "efficiency": [
        "efficiency",
        "efficiency_score",
        "efficiency_rating",
        "player_efficiency_rating",
        "per",
        "true_shooting_pct",
        "ts_pct",
        "offensive_rating",
        "ortg",
    ],
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    if df.empty:
        raise ValueError(f"Input file is empty: {file_path}")

    return df


def normalize_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("%", "pct")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    normalized_map = {normalize_text(col): col for col in df.columns}

    for candidate in candidates:
        key = normalize_text(candidate)
        if key in normalized_map:
            return normalized_map[key]

    if required:
        raise ValueError(
            f"Missing required column. Expected one of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def standardize_conference(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace(
            {
                "BIG SKY": "Big Sky",
                "big sky": "Big Sky",
                "BigSky": "Big Sky",
            }
        )
    )


def standardize_position_group(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[_\-]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    mapping = {
        "g": "Guard",
        "guard": "Guard",
        "pg": "Guard",
        "sg": "Guard",
        "combo guard": "Guard",
        "wing": "Wing",
        "w": "Wing",
        "sf": "Wing",
        "f": "Wing",
        "forward": "Wing",
        "big": "Big",
        "b": "Big",
        "c": "Big",
        "center": "Big",
        "pf": "Big",
        "post": "Big",
        "big man": "Big",
    }

    mapped = normalized.map(mapping)
    return mapped.fillna(series.astype(str).str.strip())


def validate_required_base_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "player": find_column(df, ["player", "player_name", "athlete", "name"]),
        "team": find_column(df, ["team", "team_name", "school"]),
        "conference": find_column(df, ["conference", "conf", "league"]),
        "position_group": find_column(
            df,
            ["position_group", "pos_group", "role_group", "position"],
        ),
    }


def resolve_metric_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}

    for metric_name, candidates in METRIC_CANDIDATES.items():
        column_name = find_column(df, candidates, required=False)
        if column_name is not None:
            resolved[metric_name] = column_name

    if len(resolved) < len(METRIC_CANDIDATES):
        missing = sorted(set(METRIC_CANDIDATES.keys()) - set(resolved.keys()))
        raise ValueError(
            f"Missing metric columns for: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return resolved


def prepare_player_level_stats(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = validate_required_base_columns(player_stats_df)
    metric_cols = resolve_metric_columns(player_stats_df)

    df = player_stats_df.copy()
    df[base_cols["conference"]] = standardize_conference(df[base_cols["conference"]])
    df[base_cols["position_group"]] = standardize_position_group(df[base_cols["position_group"]])

    for metric_col in metric_cols.values():
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    df = df.loc[
        df[base_cols["conference"]].eq("Big Sky")
        & df[base_cols["position_group"]].isin(["Guard", "Wing", "Big"])
    ].copy()

    if df.empty:
        raise ValueError("No valid Big Sky player records found in player stats dataset.")

    groupby_cols = [
        base_cols["player"],
        base_cols["team"],
        base_cols["conference"],
        base_cols["position_group"],
    ]

    agg_dict = {metric_col: "mean" for metric_col in metric_cols.values()}

    player_level = (
        df.groupby(groupby_cols, dropna=False, as_index=False)
        .agg(agg_dict)
        .rename(
            columns={
                base_cols["player"]: "player",
                base_cols["team"]: "team",
                base_cols["conference"]: "conference",
                base_cols["position_group"]: "position_group",
                **{v: k for k, v in metric_cols.items()},
            }
        )
    )

    player_level = player_level.drop_duplicates(subset=PLAYER_KEYS).reset_index(drop=True)

    if player_level.empty:
        raise ValueError("Player-level metric dataset is empty after aggregation.")

    return player_level


def validate_peer_group_columns(peer_groups_df: pd.DataFrame) -> dict[str, str]:
    return {
        "player": find_column(peer_groups_df, ["player"]),
        "team": find_column(peer_groups_df, ["team"]),
        "conference": find_column(peer_groups_df, ["conference"]),
        "position_group": find_column(peer_groups_df, ["position_group"]),
        "peer_group_key": find_column(peer_groups_df, ["peer_group_key"]),
        "peer_player": find_column(peer_groups_df, ["peer_player"]),
    }


def prepare_peer_group_membership(peer_groups_df: pd.DataFrame) -> pd.DataFrame:
    peer_cols = validate_peer_group_columns(peer_groups_df)

    df = peer_groups_df.copy()
    df[peer_cols["conference"]] = standardize_conference(df[peer_cols["conference"]])
    df[peer_cols["position_group"]] = standardize_position_group(df[peer_cols["position_group"]])

    if df.empty:
        raise ValueError("Peer group dataset is empty.")

    anchors = df[
        [
            peer_cols["peer_group_key"],
            peer_cols["player"],
            peer_cols["team"],
            peer_cols["conference"],
            peer_cols["position_group"],
        ]
    ].drop_duplicates()

    anchors = anchors.rename(
        columns={
            peer_cols["peer_group_key"]: "peer_group_key",
            peer_cols["player"]: "anchor_player",
            peer_cols["team"]: "anchor_team",
            peer_cols["conference"]: "conference",
            peer_cols["position_group"]: "position_group",
        }
    )

    anchor_members = anchors.rename(
        columns={
            "anchor_player": "player",
            "anchor_team": "team",
        }
    )[["peer_group_key", "conference", "position_group", "player", "team"]]

    peer_members = df[
        [
            peer_cols["peer_group_key"],
            peer_cols["conference"],
            peer_cols["position_group"],
            peer_cols["peer_player"],
        ]
    ].drop_duplicates()

    team_lookup = (
        df[
            [
                peer_cols["peer_player"],
                find_column(df, ["peer_team"], required=False),
            ]
        ]
        if find_column(df, ["peer_team"], required=False) is not None
        else pd.DataFrame(columns=[peer_cols["peer_player"], "peer_team"])
    )

    if not team_lookup.empty:
        team_lookup = team_lookup.drop_duplicates().rename(
            columns={
                peer_cols["peer_player"]: "player",
                team_lookup.columns[1]: "team",
            }
        )
        peer_members = peer_members.rename(
            columns={
                peer_cols["peer_group_key"]: "peer_group_key",
                peer_cols["conference"]: "conference",
                peer_cols["position_group"]: "position_group",
                peer_cols["peer_player"]: "player",
            }
        ).merge(team_lookup, on="player", how="left")
    else:
        peer_members = peer_members.rename(
            columns={
                peer_cols["peer_group_key"]: "peer_group_key",
                peer_cols["conference"]: "conference",
                peer_cols["position_group"]: "position_group",
                peer_cols["peer_player"]: "player",
            }
        )
        peer_members["team"] = np.nan

    membership = pd.concat([anchor_members, peer_members], ignore_index=True)
    membership = membership.drop_duplicates(
        subset=["peer_group_key", "conference", "position_group", "player"]
    ).reset_index(drop=True)

    if membership.empty:
        raise ValueError("No peer group memberships could be constructed.")

    return membership


def percentile_rank(series: pd.Series) -> pd.Series:
    valid = series.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    if valid.sum() == 0:
        return result

    if valid.sum() == 1:
        result.loc[valid] = 1.0
        return result

    result.loc[valid] = series.loc[valid].rank(method="average", pct=True)
    return result


def build_percentiles(
    player_metrics_df: pd.DataFrame,
    peer_membership_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = peer_membership_df.merge(
        player_metrics_df,
        on=["player", "team", "conference", "position_group"],
        how="left",
        validate="m:1",
    )

    missing_metric_rows = merged[["shooting", "playmaking", "rebounding", "defense", "efficiency"]].isna().all(axis=1)
    merged = merged.loc[~missing_metric_rows].copy()

    if merged.empty:
        raise ValueError("No matched player metrics found for peer group memberships.")

    for metric in METRIC_CANDIDATES.keys():
        merged[f"{metric}_percentile"] = (
            merged.groupby("peer_group_key", group_keys=False)[metric]
            .apply(percentile_rank)
            .astype(float)
            .round(6)
        )

    group_sizes = (
        merged.groupby("peer_group_key")["player"]
        .nunique()
        .rename("peer_group_size")
        .reset_index()
    )

    anchor_lookup = (
        peer_membership_df.rename(
            columns={
                "player": "anchor_player_candidate",
                "team": "anchor_team_candidate",
            }
        )
        .merge(
            player_metrics_df[["player", "team", "conference", "position_group"]],
            left_on=["anchor_player_candidate", "anchor_team_candidate", "conference", "position_group"],
            right_on=["player", "team", "conference", "position_group"],
            how="left",
        )
    )

    del anchor_lookup

    result = merged.merge(group_sizes, on="peer_group_key", how="left")

    selected_columns = [
        "peer_group_key",
        "conference",
        "position_group",
        "player",
        "team",
        "shooting",
        "shooting_percentile",
        "playmaking",
        "playmaking_percentile",
        "rebounding",
        "rebounding_percentile",
        "defense",
        "defense_percentile",
        "efficiency",
        "efficiency_percentile",
        "peer_group_size",
    ]

    result = result[selected_columns].drop_duplicates(
        subset=["peer_group_key", "player", "team"]
    )

    result = result.sort_values(
        by=["position_group", "peer_group_key", "player", "team"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return result


def export_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    project_root = get_project_root()

    player_stats_path = project_root / "data" / "processed" / "player_stats_all_games.csv"
    peer_groups_path = project_root / "data" / "features" / "player_peer_groups.csv"
    output_path = project_root / "data" / "features" / "player_percentiles.csv"

    player_stats_df = load_csv(player_stats_path)
    peer_groups_df = load_csv(peer_groups_path)

    player_metrics_df = prepare_player_level_stats(player_stats_df)
    peer_membership_df = prepare_peer_group_membership(peer_groups_df)
    percentile_df = build_percentiles(player_metrics_df, peer_membership_df)

    export_output(percentile_df, output_path)

    print(f"Player percentiles created successfully: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)