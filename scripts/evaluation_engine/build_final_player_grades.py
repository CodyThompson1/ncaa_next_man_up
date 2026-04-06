
"""
File name: build_final_player_grades.py
Last Modified: 2026-04-06

Purpose:
    Build the final Montana player evaluation table for the NCAA Next
    Man Up project.

    This script combines the category-level score files into one final
    player evaluation table. It preserves both the numeric category
    scores and category letter grades, then calculates an archetype-
    adjusted overall score and final letter grade.

Inputs:
    - data/features/shooting_scores.csv
    - data/features/playmaking_scores.csv
    - data/features/rebounding_scores.csv
    - data/features/defense_scores.csv
    - data/features/efficiency_scores.csv
    - data/features/player_archetype_assignment.csv
    - data/features/player_position_groups.csv
    - data/features/player_percentiles.csv

Outputs:
    - data/outputs/player_final_evaluations.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
LOCAL_DIR = Path(__file__).resolve().parent

SHOOTING_PATH = FEATURES_DIR / "shooting_scores.csv"
PLAYMAKING_PATH = FEATURES_DIR / "playmaking_scores.csv"
REBOUNDING_PATH = FEATURES_DIR / "rebounding_scores.csv"
DEFENSE_PATH = FEATURES_DIR / "defense_scores.csv"
EFFICIENCY_PATH = FEATURES_DIR / "efficiency_scores.csv"
ARCHETYPE_PATH = FEATURES_DIR / "player_archetype_assignment.csv"
POSITION_GROUPS_PATH = FEATURES_DIR / "player_position_groups.csv"
PLAYER_PERCENTILES_PATH = FEATURES_DIR / "player_percentiles.csv"
OUTPUT_PATH = OUTPUTS_DIR / "player_final_evaluations.csv"

KEY_COLUMNS = ["player_name", "team_name", "season"]

MONTANA_TEAM_NAMES = {
    "Montana",
    "Montana Grizzlies",
    "Montana Griz",
}

CATEGORY_SCORE_COLUMNS = {
    "shooting_score": ["shooting_score"],
    "playmaking_score": ["playmaking_score"],
    "rebounding_score": ["rebounding_score"],
    "defense_score": ["defense_score"],
    "efficiency_score": ["efficiency_score"],
}

CATEGORY_GRADE_COLUMNS = {
    "shooting_score": "shooting_grade",
    "playmaking_score": "playmaking_grade",
    "rebounding_score": "rebounding_grade",
    "defense_score": "defense_grade",
    "efficiency_score": "efficiency_grade",
}

DEFAULT_CATEGORY_WEIGHTS = {
    "shooting_score": 0.24,
    "playmaking_score": 0.20,
    "rebounding_score": 0.12,
    "defense_score": 0.16,
    "efficiency_score": 0.28,
}

ARCHETYPE_CATEGORY_WEIGHTS = {
    "Primary Creator": {
        "shooting_score": 0.20,
        "playmaking_score": 0.34,
        "rebounding_score": 0.06,
        "defense_score": 0.10,
        "efficiency_score": 0.30,
    },
    "Secondary Playmaker": {
        "shooting_score": 0.22,
        "playmaking_score": 0.30,
        "rebounding_score": 0.08,
        "defense_score": 0.12,
        "efficiency_score": 0.28,
    },
    "Off-Ball Shooter": {
        "shooting_score": 0.34,
        "playmaking_score": 0.08,
        "rebounding_score": 0.10,
        "defense_score": 0.15,
        "efficiency_score": 0.33,
    },
    "Scoring Forward": {
        "shooting_score": 0.26,
        "playmaking_score": 0.10,
        "rebounding_score": 0.18,
        "defense_score": 0.14,
        "efficiency_score": 0.32,
    },
    "3-and-D Forward": {
        "shooting_score": 0.26,
        "playmaking_score": 0.05,
        "rebounding_score": 0.14,
        "defense_score": 0.27,
        "efficiency_score": 0.28,
    },
    "Interior Forward": {
        "shooting_score": 0.10,
        "playmaking_score": 0.06,
        "rebounding_score": 0.29,
        "defense_score": 0.23,
        "efficiency_score": 0.32,
    },
    "Glue Forward": {
        "shooting_score": 0.18,
        "playmaking_score": 0.14,
        "rebounding_score": 0.22,
        "defense_score": 0.18,
        "efficiency_score": 0.28,
    },
}

ROLE_CONTEXT_WEIGHTS = {
    "Primary Creator": {
        "usg_pct_conf_pct": 0.28,
        "ast_pct_conf_pct": 0.24,
        "assist_rate_conf_pct": 0.18,
        "pprod_conf_pct": 0.14,
        "points_per_game_conf_pct": 0.10,
        "minutes_per_game_conf_pct": 0.06,
    },
    "Secondary Playmaker": {
        "ast_pct_conf_pct": 0.28,
        "assist_rate_conf_pct": 0.22,
        "usg_pct_conf_pct": 0.14,
        "pprod_conf_pct": 0.12,
        "minutes_per_game_conf_pct": 0.12,
        "ts_pct_conf_pct": 0.12,
    },
    "Off-Ball Shooter": {
        "three_pt_pct_conf_pct": 0.26,
        "three_point_attempt_rate_conf_pct": 0.24,
        "efg_pct_conf_pct": 0.18,
        "ts_pct_conf_pct": 0.18,
        "minutes_per_game_conf_pct": 0.14,
    },
    "Scoring Forward": {
        "points_per_game_conf_pct": 0.22,
        "pprod_conf_pct": 0.18,
        "ts_pct_conf_pct": 0.18,
        "efg_pct_conf_pct": 0.14,
        "trb_pct_conf_pct": 0.14,
        "minutes_per_game_conf_pct": 0.14,
    },
    "3-and-D Forward": {
        "three_pt_pct_conf_pct": 0.20,
        "three_point_attempt_rate_conf_pct": 0.16,
        "stl_pct_conf_pct": 0.18,
        "blk_pct_conf_pct": 0.14,
        "drtg_conf_pct": 0.16,
        "minutes_per_game_conf_pct": 0.16,
    },
    "Interior Forward": {
        "trb_pct_conf_pct": 0.24,
        "orb_pct_conf_pct": 0.18,
        "drb_pct_conf_pct": 0.16,
        "blk_pct_conf_pct": 0.14,
        "ortg_conf_pct": 0.14,
        "ts_pct_conf_pct": 0.14,
    },
    "Glue Forward": {
        "trb_pct_conf_pct": 0.20,
        "drb_pct_conf_pct": 0.16,
        "stl_pct_conf_pct": 0.14,
        "ast_pct_conf_pct": 0.14,
        "minutes_per_game_conf_pct": 0.18,
        "ts_pct_conf_pct": 0.18,
    },
}

DEFAULT_ROLE_CONTEXT_WEIGHTS = {
    "points_per_game_conf_pct": 0.20,
    "minutes_per_game_conf_pct": 0.20,
    "ts_pct_conf_pct": 0.20,
    "trb_pct_conf_pct": 0.20,
    "ast_pct_conf_pct": 0.20,
}

CATEGORY_GRADE_BINS = [
    (90, "A+"),
    (82, "A"),
    (74, "A-"),
    (66, "B+"),
    (58, "B"),
    (50, "B-"),
    (42, "C+"),
    (34, "C"),
    (26, "C-"),
    (18, "D+"),
    (10, "D"),
    (5, "D-"),
    (0, "F"),
]

FINAL_GRADE_BINS = [
    (92, "A+"),
    (84, "A"),
    (76, "A-"),
    (68, "B+"),
    (60, "B"),
    (52, "B-"),
    (44, "C+"),
    (36, "C"),
    (28, "C-"),
    (20, "D+"),
    (12, "D"),
    (6, "D-"),
    (0, "F"),
]

CONFIDENCE_GRADE_BINS = [
    (90, "Very High"),
    (75, "High"),
    (60, "Moderate"),
    (45, "Low"),
    (0, "Very Low"),
]

CONTEXT_METRIC_COLUMNS = [
    "conference_name",
    "position_group",
    "points_per_game",
    "pprod",
    "usg_pct",
    "ast_pct",
    "assist_rate",
    "minutes_per_game",
    "three_pt_pct",
    "three_point_attempt_rate",
    "trb_pct",
    "orb_pct",
    "drb_pct",
    "stl_pct",
    "blk_pct",
    "ortg",
    "drtg",
    "ts_pct",
    "efg_pct",
    "tov_pct",
    "peer_group_size",
]


def resolve_input_path(preferred_path: Path, fallback_name: str) -> Path:
    """Use project path first, then local uploaded-file fallback."""
    if preferred_path.exists():
        return preferred_path

    fallback_path = LOCAL_DIR / fallback_name
    if fallback_path.exists():
        return fallback_path

    return preferred_path


def validate_file_exists(path: Path) -> None:
    """Raise a clear error when a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    """Read and validate one CSV file."""
    validate_file_exists(path)
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df


def ensure_output_directory(path: Path) -> None:
    """Create the output directory before writing the final file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize player merge keys across all input tables."""
    working_df = df.copy()

    if "player_name" in working_df.columns:
        working_df["player_name"] = (
            working_df["player_name"].astype("string").str.strip()
        )

    if "team_name" in working_df.columns:
        working_df["team_name"] = (
            working_df["team_name"].astype("string").str.strip()
        )

    if "season" in working_df.columns:
        working_df["season"] = pd.to_numeric(
            working_df["season"],
            errors="coerce",
        ).astype("Int64")

    return working_df


def normalize_position_group(series: pd.Series) -> pd.Series:
    """Collapse raw position labels into Guard or Forward."""
    mapping = {
        "g": "Guard",
        "guard": "Guard",
        "guards": "Guard",
        "f": "Forward",
        "forward": "Forward",
        "forwards": "Forward",
        "c": "Forward",
        "center": "Forward",
        "centers": "Forward",
        "big": "Forward",
        "bigs": "Forward",
    }

    original = series.astype("string").str.strip()
    normalized = original.str.lower().map(mapping)
    return normalized.fillna(original)


def normalize_percentile_like(series: pd.Series) -> pd.Series:
    """Convert either 0-1 or 0-100 percentile-like values to 0-100."""
    cleaned = pd.to_numeric(series, errors="coerce")

    if cleaned.dropna().empty:
        return cleaned

    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:
    """Resolve alternate column names without assuming exact casing."""
    lower_map = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        matched_column = lower_map.get(candidate.lower())
        if matched_column is not None:
            return matched_column

    return None


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Ensure each input table contains the fields needed downstream."""
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def drop_duplicate_keys(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """Fail fast when player merge keys are duplicated."""
    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)

    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, KEY_COLUMNS]
            .drop_duplicates()
            .sort_values(KEY_COLUMNS)
        )
        raise ValueError(
            f"{dataset_name} contains duplicate rows for keys {KEY_COLUMNS}.\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return df.copy()


def apply_grade_scale(
    value: Optional[float],
    grade_bins: list[tuple[float, str]],
) -> Optional[str]:
    """Map a numeric score to a letter-style band."""
    if pd.isna(value):
        return pd.NA

    for threshold, grade in grade_bins:
        if float(value) >= threshold:
            return grade

    return "F"


def weighted_mean(
    row: pd.Series,
    metric_weights: dict[str, float],
) -> float:
    """Compute a weighted mean using only the metrics available."""
    weighted_sum = 0.0
    weight_sum = 0.0

    for column_name, weight in metric_weights.items():
        metric_value = row.get(column_name)

        if pd.notna(metric_value):
            weighted_sum += float(metric_value) * weight
            weight_sum += weight

    if weight_sum == 0:
        return np.nan

    return weighted_sum / weight_sum


def prepare_archetype_table() -> pd.DataFrame:
    """Load archetypes and standardize the source-of-truth role field."""
    path = resolve_input_path(
        ARCHETYPE_PATH,
        "player_archetype_assignment.csv",
    )
    archetype_df = read_csv(path)
    archetype_df = standardize_key_columns(archetype_df)
    archetype_df = drop_duplicate_keys(
        archetype_df,
        "player_archetype_assignment.csv",
    )

    archetype_column = find_first_existing_column(
        archetype_df,
        ["player_archetype", "archetype", "role_archetype"],
    )

    if archetype_column is None:
        raise ValueError(
            "Could not resolve an archetype column in "
            "player_archetype_assignment.csv"
        )

    keep_columns = KEY_COLUMNS + [archetype_column]
    if "position_group" in archetype_df.columns:
        keep_columns.append("position_group")

    archetype_df = archetype_df[keep_columns].copy()
    archetype_df = archetype_df.rename(
        columns={archetype_column: "player_archetype"}
    )

    if "position_group" in archetype_df.columns:
        archetype_df["position_group"] = normalize_position_group(
            archetype_df["position_group"]
        )

    return archetype_df


def prepare_position_group_table() -> pd.DataFrame:
    """Load position groups as a backfill table for missing context."""
    path = resolve_input_path(
        POSITION_GROUPS_PATH,
        "player_position_groups.csv",
    )
    position_df = read_csv(path)
    position_df = standardize_key_columns(position_df)
    position_df = drop_duplicate_keys(
        position_df,
        "player_position_groups.csv",
    )
    validate_required_columns(
        position_df,
        KEY_COLUMNS + ["position_group"],
        "player_position_groups.csv",
    )

    position_df = position_df[KEY_COLUMNS + ["position_group"]].copy()
    position_df["position_group"] = normalize_position_group(
        position_df["position_group"]
    )

    return position_df


def prepare_player_percentiles_table() -> pd.DataFrame:
    """Load conference context metrics used for role-level adjustments."""
    path = resolve_input_path(
        PLAYER_PERCENTILES_PATH,
        "player_percentiles.csv",
    )
    percentiles_df = read_csv(path)
    percentiles_df = standardize_key_columns(percentiles_df)
    percentiles_df = drop_duplicate_keys(
        percentiles_df,
        "player_percentiles.csv",
    )

    existing_columns = [
        column for column in KEY_COLUMNS + CONTEXT_METRIC_COLUMNS
        if column in percentiles_df.columns
    ]
    percentiles_df = percentiles_df[existing_columns].copy()

    numeric_columns = [
        column for column in percentiles_df.columns
        if column not in KEY_COLUMNS + ["conference_name", "position_group"]
    ]

    for column in numeric_columns:
        percentiles_df[column] = pd.to_numeric(
            percentiles_df[column],
            errors="coerce",
        )

    if "position_group" in percentiles_df.columns:
        percentiles_df["position_group"] = normalize_position_group(
            percentiles_df["position_group"]
        )

    return percentiles_df


def read_category_score_table(
    path: Path,
    dataset_name: str,
    logical_score_name: str,
) -> pd.DataFrame:
    """Read one category table and standardize the numeric score field."""
    score_df = read_csv(path)
    score_df = standardize_key_columns(score_df)
    score_df = drop_duplicate_keys(score_df, dataset_name)

    score_column = find_first_existing_column(
        score_df,
        CATEGORY_SCORE_COLUMNS[logical_score_name],
    )

    if score_column is None:
        raise ValueError(
            f"Could not resolve {logical_score_name} in {dataset_name}. "
            f"Available columns: {list(score_df.columns)}"
        )

    output_df = score_df[KEY_COLUMNS + [score_column]].copy()
    output_df = output_df.rename(columns={score_column: logical_score_name})
    output_df[logical_score_name] = normalize_percentile_like(
        output_df[logical_score_name]
    ).round(2)

    return output_df


def build_base_player_table(score_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Create the player spine before merging context and scores."""
    return (
        pd.concat(
            [score_table[KEY_COLUMNS] for score_table in score_tables],
            ignore_index=True,
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )


def merge_player_tables(
    base_df: pd.DataFrame,
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge one-player-per-row tables onto the base player spine."""
    merged_df = base_df.copy()

    for table in tables:
        merged_df = merged_df.merge(
            table,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    return merged_df


def backfill_context_columns(final_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill player_archetype and position_group from source tables."""
    working_df = final_df.copy()

    if "position_group_from_positions" in working_df.columns:
        if "position_group" not in working_df.columns:
            working_df["position_group"] = working_df[
                "position_group_from_positions"
            ]
        else:
            working_df["position_group"] = working_df["position_group"].fillna(
                working_df["position_group_from_positions"]
            )
        working_df = working_df.drop(columns=["position_group_from_positions"])

    if "position_group" in working_df.columns:
        working_df["position_group"] = normalize_position_group(
            working_df["position_group"]
        )

    return working_df


def filter_to_montana_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Montana rows for the final player-facing output."""
    montana_df = df.loc[df["team_name"].isin(MONTANA_TEAM_NAMES)].copy()

    if montana_df.empty:
        available_teams = sorted(
            df["team_name"].dropna().astype(str).unique().tolist()
        )
        raise ValueError(
            "No Montana players found after filtering. "
            "Check MONTANA_TEAM_NAMES. "
            f"Available team_name values: {available_teams}"
        )

    return montana_df


def add_conference_context_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw context metrics into conference-relative percentiles."""
    working_df = df.copy()

    ranking_columns = [
        "points_per_game",
        "pprod",
        "usg_pct",
        "ast_pct",
        "assist_rate",
        "minutes_per_game",
        "three_pt_pct",
        "three_point_attempt_rate",
        "trb_pct",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "ortg",
        "drtg",
        "ts_pct",
        "efg_pct",
        "tov_pct",
    ]

    for column in ranking_columns:
        if column not in working_df.columns:
            continue

        percentile_series = (
            working_df[column].rank(method="average", pct=True) * 100
        )

        if column in {"drtg", "tov_pct"}:
            percentile_series = 100 - percentile_series

        working_df[f"{column}_conf_pct"] = percentile_series.clip(
            lower=0,
            upper=100,
        ).round(2)

    return working_df


def get_archetype_category_weights(archetype: Optional[str]) -> dict[str, float]:
    """Return final category weights for the player's archetype."""
    if pd.isna(archetype):
        return DEFAULT_CATEGORY_WEIGHTS

    return ARCHETYPE_CATEGORY_WEIGHTS.get(
        str(archetype),
        DEFAULT_CATEGORY_WEIGHTS,
    )


def get_role_context_weights(archetype: Optional[str]) -> dict[str, float]:
    """Return context-weight metrics for the player's archetype."""
    if pd.isna(archetype):
        return DEFAULT_ROLE_CONTEXT_WEIGHTS

    return ROLE_CONTEXT_WEIGHTS.get(
        str(archetype),
        DEFAULT_ROLE_CONTEXT_WEIGHTS,
    )


def compute_role_context_score(row: pd.Series) -> float:
    """Score role burden and fit using archetype-specific context metrics."""
    return weighted_mean(
        row,
        get_role_context_weights(row.get("player_archetype")),
    )


def add_role_context_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add role-context and peer-reliability support fields."""
    working_df = df.copy()

    working_df["role_context_score"] = working_df.apply(
        compute_role_context_score,
        axis=1,
    ).round(2)

    peer_size = pd.to_numeric(
        working_df.get("peer_group_size"),
        errors="coerce",
    ).fillna(0)

    reliability = 70 + (peer_size.clip(lower=0, upper=25) * 1.2)
    working_df["peer_group_reliability_score"] = reliability.clip(
        lower=70,
        upper=100,
    ).round(2)

    return working_df


def add_category_letter_grades(df: pd.DataFrame) -> pd.DataFrame:
    """Map each category score to its own letter grade."""
    working_df = df.copy()

    for score_column, grade_column in CATEGORY_GRADE_COLUMNS.items():
        working_df[grade_column] = working_df[score_column].apply(
            lambda value: apply_grade_scale(value, CATEGORY_GRADE_BINS)
        )

    return working_df


def compute_weighted_category_score(row: pd.Series) -> float:
    """Compute the archetype-weighted mean of the five category scores."""
    weights = get_archetype_category_weights(row.get("player_archetype"))
    return weighted_mean(row, weights)


def compute_adjusted_overall_score(row: pd.Series) -> float:
    """
    Blend category score, role context, and peer reliability.

    The category score remains the anchor. Role context adds value for
    high-burden roles. Reliability slightly tempers tiny peer-group noise.
    """
    category_score = row.get("base_overall_score")
    role_context = row.get("role_context_score")
    reliability = row.get("peer_group_reliability_score")

    if pd.isna(category_score):
        return np.nan

    reliability_factor = 0.92 + ((float(reliability) - 70) / 30) * 0.08

    adjusted_score = (
        (float(category_score) * 0.82)
        + (float(role_context) * 0.18 if pd.notna(role_context) else 0.0)
    )

    adjusted_score *= reliability_factor
    return float(np.clip(adjusted_score, 0, 100))


def add_overall_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate base overall score, adjusted overall score, and grade."""
    working_df = df.copy()

    working_df["base_overall_score"] = working_df.apply(
        compute_weighted_category_score,
        axis=1,
    ).round(2)

    for score_column in DEFAULT_CATEGORY_WEIGHTS:
        weight_column = f"{score_column}_final_weight"
        working_df[weight_column] = working_df["player_archetype"].apply(
            lambda archetype: get_archetype_category_weights(archetype)[
                score_column
            ]
        )

    working_df["overall_score_100"] = working_df.apply(
        compute_adjusted_overall_score,
        axis=1,
    ).round(2)

    working_df["letter_grade"] = working_df["overall_score_100"].apply(
        lambda value: apply_grade_scale(value, FINAL_GRADE_BINS)
    )

    working_df["grade_confidence_score"] = (
        (
            working_df["peer_group_reliability_score"] * 0.60
            + working_df["role_context_score"].fillna(50) * 0.20
            + working_df["base_overall_score"].fillna(50) * 0.20
        )
        .clip(lower=0, upper=100)
        .round(2)
    )

    working_df["grade_confidence_band"] = (
        working_df["grade_confidence_score"].apply(
            lambda value: apply_grade_scale(value, CONFIDENCE_GRADE_BINS)
        )
    )

    return working_df


def validate_final_context(df: pd.DataFrame) -> None:
    """Ensure Montana players have the key context fields populated."""
    required_context_columns = ["position_group", "player_archetype"]

    for column in required_context_columns:
        missing_rows = df.loc[df[column].isna(), KEY_COLUMNS]

        if not missing_rows.empty:
            raise ValueError(
                f"Montana players are missing {column}.\n"
                f"{missing_rows.to_string(index=False)}"
            )


def order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place the most useful columns first for dashboard use."""
    ordered_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "player_archetype",
        "shooting_score",
        "shooting_grade",
        "playmaking_score",
        "playmaking_grade",
        "rebounding_score",
        "rebounding_grade",
        "defense_score",
        "defense_grade",
        "efficiency_score",
        "efficiency_grade",
        "base_overall_score",
        "role_context_score",
        "peer_group_reliability_score",
        "overall_score_100",
        "letter_grade",
        "grade_confidence_score",
        "grade_confidence_band",
        "shooting_score_final_weight",
        "playmaking_score_final_weight",
        "rebounding_score_final_weight",
        "defense_score_final_weight",
        "efficiency_score_final_weight",
        "peer_group_size",
    ]

    existing_ordered_columns = [
        column for column in ordered_columns if column in df.columns
    ]
    remaining_columns = [
        column for column in df.columns if column not in existing_ordered_columns
    ]

    return df[existing_ordered_columns + remaining_columns].copy()


def validate_final_output(df: pd.DataFrame) -> None:
    """Validate the final one-row-per-player output before writing."""
    required_columns = [
        "player_name",
        "team_name",
        "season",
        "position_group",
        "player_archetype",
        "shooting_score",
        "playmaking_score",
        "rebounding_score",
        "defense_score",
        "efficiency_score",
        "overall_score_100",
        "letter_grade",
    ]
    validate_required_columns(
        df,
        required_columns,
        "final player evaluations",
    )

    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, KEY_COLUMNS]
            .drop_duplicates()
            .sort_values(KEY_COLUMNS)
        )
        raise ValueError(
            "Final output contains duplicate player rows.\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def build_final_player_evaluations() -> pd.DataFrame:
    """Build the final archetype-adjusted Montana player evaluation table."""
    shooting_df = read_category_score_table(
        resolve_input_path(SHOOTING_PATH, "shooting_scores.csv"),
        "shooting_scores.csv",
        "shooting_score",
    )
    playmaking_df = read_category_score_table(
        resolve_input_path(PLAYMAKING_PATH, "playmaking_scores.csv"),
        "playmaking_scores.csv",
        "playmaking_score",
    )
    rebounding_df = read_category_score_table(
        resolve_input_path(REBOUNDING_PATH, "rebounding_scores.csv"),
        "rebounding_scores.csv",
        "rebounding_score",
    )
    defense_df = read_category_score_table(
        resolve_input_path(DEFENSE_PATH, "defense_scores.csv"),
        "defense_scores.csv",
        "defense_score",
    )
    efficiency_df = read_category_score_table(
        resolve_input_path(EFFICIENCY_PATH, "efficiency_scores.csv"),
        "efficiency_scores.csv",
        "efficiency_score",
    )

    archetype_df = prepare_archetype_table()
    position_df = prepare_position_group_table().rename(
        columns={"position_group": "position_group_from_positions"}
    )
    percentiles_df = prepare_player_percentiles_table()

    score_tables = [
        shooting_df,
        playmaking_df,
        rebounding_df,
        defense_df,
        efficiency_df,
    ]

    base_df = build_base_player_table(score_tables)

    final_df = merge_player_tables(
        base_df,
        [
            archetype_df,
            position_df,
            percentiles_df,
            shooting_df,
            playmaking_df,
            rebounding_df,
            defense_df,
            efficiency_df,
        ],
    )

    final_df = backfill_context_columns(final_df)
    final_df = filter_to_montana_players(final_df)
    validate_final_context(final_df)

    final_df = add_conference_context_percentiles(final_df)
    final_df = add_role_context_score(final_df)
    final_df = add_category_letter_grades(final_df)
    final_df = add_overall_scoring(final_df)
    final_df = order_output_columns(final_df)

    final_df = final_df.sort_values(
        by=["season", "overall_score_100", "player_name"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    validate_final_output(final_df)
    return final_df


def main() -> None:
    """Run the build and write the final CSV output."""
    final_df = build_final_player_evaluations()

    output_path = resolve_input_path(OUTPUT_PATH, "player_final_evaluations.csv")
    if output_path == OUTPUT_PATH:
        ensure_output_directory(output_path)

    final_df.to_csv(output_path, index=False)

    print("=" * 80)
    print("Final player evaluations build complete.")
    print(f"Rows written: {len(final_df):,}")
    print(f"Columns written: {len(final_df.columns):,}")
    print(f"Output path: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
