"""
Build the final player evaluation table for the NCAA Next Man Up project.

Purpose:
    Combine the category-level scoring outputs into one final Montana player
    evaluation table for dashboard use.

Important notes:
    - This version uses the corrected category scores directly.
    - It does NOT percentile-rank the category scores again.
    - Each category score is already expected to be on a 0-100 scale.
    - Archetype is pulled from player_archetype_assignment.csv as the source of truth.
    - Final output includes:
        * overall_score_100
        * letter_grade

Inputs:
    - data/features/shooting_scores.csv
    - data/features/playmaking_scores.csv
    - data/features/rebounding_scores.csv
    - data/features/defense_scores.csv
    - data/features/efficiency_scores.csv
    - data/features/player_archetype_assignment.csv
    - data/features/player_position_groups.csv

Output:
    - data/outputs/player_final_evaluations.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"

SHOOTING_SCORES_PATH = FEATURES_DIR / "shooting_scores.csv"
PLAYMAKING_SCORES_PATH = FEATURES_DIR / "playmaking_scores.csv"
REBOUNDING_SCORES_PATH = FEATURES_DIR / "rebounding_scores.csv"
DEFENSE_SCORES_PATH = FEATURES_DIR / "defense_scores.csv"
EFFICIENCY_SCORES_PATH = FEATURES_DIR / "efficiency_scores.csv"
ARCHETYPE_PATH = FEATURES_DIR / "player_archetype_assignment.csv"
POSITION_GROUPS_PATH = FEATURES_DIR / "player_position_groups.csv"

OUTPUT_PATH = OUTPUTS_DIR / "player_final_evaluations.csv"

KEY_COLUMNS = ["player_name", "team_name", "season"]

# Category weights should sum to 1.0.
CATEGORY_WEIGHTS: Dict[str, float] = {
    "shooting_score": 0.25,
    "playmaking_score": 0.20,
    "rebounding_score": 0.15,
    "defense_score": 0.20,
    "efficiency_score": 0.20,
}

# Update if your Montana team_name is stored differently.
MONTANA_TEAM_NAME_CANDIDATES = {
    "Montana",
    "Montana Grizzlies",
    "Montana Griz",
}

LETTER_GRADE_BINS = [
    (97, "A+"),
    (93, "A"),
    (90, "A-"),
    (87, "B+"),
    (83, "B"),
    (80, "B-"),
    (77, "C+"),
    (73, "C"),
    (70, "C-"),
    (67, "D+"),
    (63, "D"),
    (60, "D-"),
    (0, "F"),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _validate_file_exists(path: Path) -> None:
    """Raise an error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _ensure_output_directory(path: Path) -> None:
    """Create the parent output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV after validating it exists."""
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize key columns used for merges."""
    df = df.copy()

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype("string").str.strip()

    if "team_name" in df.columns:
        df["team_name"] = df["team_name"].astype("string").str.strip()

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df


def _normalize_position_group(series: pd.Series) -> pd.Series:
    """Collapse Big/Center values into Forward and standardize labels."""
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

    normalized = series.astype("string").str.strip().str.lower().map(mapping)
    return normalized.fillna(series.astype("string").str.strip())


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Ensure a dataframe contains all required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def _drop_duplicate_keys(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Drop duplicate rows by player key and warn via exception if duplicates remain ambiguous."""
    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, KEY_COLUMNS]
            .sort_values(KEY_COLUMNS)
            .drop_duplicates()
        )
        raise ValueError(
            f"{dataset_name} contains duplicate rows for keys {KEY_COLUMNS}.\n"
            f"{duplicate_rows.to_string(index=False)}"
        )
    return df.copy()


def _validate_weights(weights: Dict[str, float]) -> None:
    """Validate final category weights."""
    if not weights:
        raise ValueError("CATEGORY_WEIGHTS is empty.")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("CATEGORY_WEIGHTS cannot contain negative values.")

    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"CATEGORY_WEIGHTS must sum to 1.0. Current sum: {weight_sum:.6f}"
        )


def _resolve_archetype_column(df: pd.DataFrame) -> str:
    """Find the archetype column in the archetype assignment file."""
    for candidate in ["archetype", "player_archetype", "role_archetype"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No archetype column found in player_archetype_assignment.csv")


def _prepare_context_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare archetype and position-group context tables."""
    archetype_df = _read_csv(ARCHETYPE_PATH)
    archetype_df = _standardize_key_columns(archetype_df)
    _validate_required_columns(archetype_df, KEY_COLUMNS, "player_archetype_assignment.csv")

    archetype_column = _resolve_archetype_column(archetype_df)

    archetype_keep_columns = KEY_COLUMNS + [archetype_column]
    if "position_group" in archetype_df.columns:
        archetype_keep_columns.append("position_group")

    archetype_df = archetype_df[archetype_keep_columns].copy()
    archetype_df = archetype_df.rename(columns={archetype_column: "player_archetype"})

    if "position_group" in archetype_df.columns:
        archetype_df["position_group"] = _normalize_position_group(archetype_df["position_group"])

    archetype_df = _drop_duplicate_keys(archetype_df, "player_archetype_assignment.csv")

    position_group_df = _read_csv(POSITION_GROUPS_PATH)
    position_group_df = _standardize_key_columns(position_group_df)
    _validate_required_columns(
        position_group_df,
        KEY_COLUMNS + ["position_group"],
        "player_position_groups.csv",
    )

    position_group_df = position_group_df[KEY_COLUMNS + ["position_group"]].copy()
    position_group_df["position_group"] = _normalize_position_group(position_group_df["position_group"])
    position_group_df = _drop_duplicate_keys(position_group_df, "player_position_groups.csv")

    return archetype_df, position_group_df


def _read_category_score_file(
    path: Path,
    dataset_name: str,
    score_column: str,
) -> pd.DataFrame:
    """
    Read one category score file and keep only the key fields plus the score.

    Any archetype or position columns inside category score files are ignored here.
    The archetype assignment and position group tables are the source of truth.
    """
    df = _read_csv(path)
    df = _standardize_key_columns(df)

    _validate_required_columns(df, KEY_COLUMNS + [score_column], dataset_name)
    df = _drop_duplicate_keys(df, dataset_name)

    output_df = df[KEY_COLUMNS + [score_column]].copy()
    output_df[score_column] = pd.to_numeric(output_df[score_column], errors="coerce")
    output_df[score_column] = output_df[score_column].clip(lower=0, upper=100).round(2)

    return output_df


def _build_base_player_table(score_tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Build a base player table from all category score tables."""
    base_df = (
        pd.concat([table[KEY_COLUMNS] for table in score_tables], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return base_df


def _merge_player_level_tables(base_df: pd.DataFrame, tables: List[pd.DataFrame]) -> pd.DataFrame:
    """Sequentially merge player-level tables onto the base dataframe."""
    merged_df = base_df.copy()

    for table in tables:
        merged_df = merged_df.merge(
            table,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    return merged_df


def _filter_to_montana_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Montana players in the final output."""
    df = df.copy()

    if "team_name" not in df.columns:
        raise KeyError("Expected 'team_name' column is missing.")

    filtered_df = df.loc[df["team_name"].isin(MONTANA_TEAM_NAME_CANDIDATES)].copy()

    if filtered_df.empty:
        available_teams = df["team_name"].dropna().drop_duplicates().sort_values().tolist()
        raise ValueError(
            "No Montana players found after filtering. "
            "Check MONTANA_TEAM_NAME_CANDIDATES.\n"
            f"Available team_name values: {available_teams}"
        )

    return filtered_df


def _validate_montana_context(df: pd.DataFrame) -> None:
    """Ensure Montana players have archetype and position group context."""
    required_context_columns = ["position_group", "player_archetype"]
    for column in required_context_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required final context column: {column}")

        missing_rows = df.loc[df[column].isna(), KEY_COLUMNS]
        if not missing_rows.empty:
            raise ValueError(
                f"Montana players are missing {column}.\n"
                f"{missing_rows.to_string(index=False)}"
            )


def _compute_weighted_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final overall score out of 100.

    This uses the already-corrected category scores directly and re-normalizes
    weights only across available category scores for each player.
    """
    df = df.copy()

    missing_score_columns = [
        column for column in CATEGORY_WEIGHTS if column not in df.columns
    ]
    if missing_score_columns:
        raise KeyError(
            "Cannot compute overall_score_100 because these category columns are missing: "
            f"{missing_score_columns}"
        )

    weighted_sum = pd.Series(0.0, index=df.index, dtype="float64")
    available_weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for score_column, weight in CATEGORY_WEIGHTS.items():
        valid_mask = df[score_column].notna()
        weighted_sum.loc[valid_mask] += df.loc[valid_mask, score_column] * weight
        available_weight_sum.loc[valid_mask] += weight

    df["overall_score_100"] = weighted_sum.div(
        available_weight_sum.where(available_weight_sum > 0)
    )
    df["overall_score_100"] = df["overall_score_100"].round(2)

    return df


def _assign_letter_grade(score: Optional[float]) -> Optional[str]:
    """Assign a letter grade from the numeric overall score."""
    if pd.isna(score):
        return None

    for threshold, grade in LETTER_GRADE_BINS:
        if score >= threshold:
            return grade

    return "F"


def _add_letter_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Add letter_grade from overall_score_100."""
    df = df.copy()
    df["letter_grade"] = df["overall_score_100"].apply(_assign_letter_grade)
    return df


def _add_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the final category weights used for readability in the output."""
    df = df.copy()
    for column, weight in CATEGORY_WEIGHTS.items():
        df[f"{column}_final_weight"] = weight
    return df


def _order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns for clean downstream dashboard use."""
    preferred_columns = [
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
        "shooting_score_final_weight",
        "playmaking_score_final_weight",
        "rebounding_score_final_weight",
        "defense_score_final_weight",
        "efficiency_score_final_weight",
    ]

    ordered_existing = [column for column in preferred_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_existing]

    return df[ordered_existing + remaining_columns].copy()


def _validate_final_output(df: pd.DataFrame) -> None:
    """Validate final player evaluation output."""
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
    _validate_required_columns(df, required_columns, "final player evaluations")

    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates()
        raise ValueError(
            "Final output contains duplicate player rows:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    non_missing_required = [
        "player_name",
        "team_name",
        "season",
        "position_group",
        "player_archetype",
    ]
    missing_summary = df[non_missing_required].isna().sum()
    problematic = missing_summary[missing_summary > 0]

    if not problematic.empty:
        raise ValueError(
            "Final output contains missing required values:\n"
            f"{problematic.to_string()}"
        )


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_final_player_evaluations() -> pd.DataFrame:
    """
    Build the final Montana player evaluation table.

    Workflow:
        1. Load all corrected category score files.
        2. Load archetype and position-group tables as source-of-truth context.
        3. Merge everything into one player-level table.
        4. Filter to Montana players.
        5. Compute overall score out of 100 from corrected category scores.
        6. Assign letter grade.
        7. Write final CSV.
    """
    _validate_weights(CATEGORY_WEIGHTS)

    shooting_df = _read_category_score_file(
        path=SHOOTING_SCORES_PATH,
        dataset_name="shooting_scores.csv",
        score_column="shooting_score",
    )
    playmaking_df = _read_category_score_file(
        path=PLAYMAKING_SCORES_PATH,
        dataset_name="playmaking_scores.csv",
        score_column="playmaking_score",
    )
    rebounding_df = _read_category_score_file(
        path=REBOUNDING_SCORES_PATH,
        dataset_name="rebounding_scores.csv",
        score_column="rebounding_score",
    )
    defense_df = _read_category_score_file(
        path=DEFENSE_SCORES_PATH,
        dataset_name="defense_scores.csv",
        score_column="defense_score",
    )
    efficiency_df = _read_category_score_file(
        path=EFFICIENCY_SCORES_PATH,
        dataset_name="efficiency_scores.csv",
        score_column="efficiency_score",
    )

    archetype_df, position_group_df = _prepare_context_tables()

    score_tables = [
        shooting_df,
        playmaking_df,
        rebounding_df,
        defense_df,
        efficiency_df,
    ]

    base_df = _build_base_player_table(score_tables)

    final_df = _merge_player_level_tables(
        base_df=base_df,
        tables=[
            position_group_df,
            archetype_df[KEY_COLUMNS + ["player_archetype"]],
            shooting_df,
            playmaking_df,
            rebounding_df,
            defense_df,
            efficiency_df,
        ],
    )

    # If position_group came only from archetype file and not position groups file,
    # backfill it from the archetype table.
    if "position_group" not in final_df.columns or final_df["position_group"].isna().any():
        archetype_position_df = archetype_df[KEY_COLUMNS + ["position_group"]].copy()
        final_df = final_df.merge(
            archetype_position_df,
            on=KEY_COLUMNS,
            how="left",
            suffixes=("", "_arch"),
            validate="one_to_one",
        )

        if "position_group_arch" in final_df.columns:
            if "position_group" not in final_df.columns:
                final_df["position_group"] = final_df["position_group_arch"]
            else:
                final_df["position_group"] = final_df["position_group"].fillna(final_df["position_group_arch"])
            final_df = final_df.drop(columns=["position_group_arch"])

    if "position_group" in final_df.columns:
        final_df["position_group"] = _normalize_position_group(final_df["position_group"])

    final_df = _filter_to_montana_players(final_df)
    _validate_montana_context(final_df)

    final_df = _compute_weighted_overall_score(final_df)
    final_df = _add_letter_grade(final_df)
    final_df = _add_weight_columns(final_df)
    final_df = _order_output_columns(final_df)

    final_df = final_df.sort_values(
        by=["season", "overall_score_100", "team_name", "player_name"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    _validate_final_output(final_df)

    return final_df


def main() -> None:
    """Run the final player evaluation build process and write the output CSV."""
    final_df = build_final_player_evaluations()

    _ensure_output_directory(OUTPUT_PATH)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("Final player evaluations build complete.")
    print(f"Rows written: {len(final_df):,}")
    print(f"Columns written: {len(final_df.columns):,}")
    print(f"Output path: {OUTPUT_PATH}")
    print("Final scoring interpretation:")
    print("- Each category score is already on a 0-100 scale.")
    print("- overall_score_100 is the weighted final score out of 100.")
    print("- letter_grade is based on overall_score_100.")
    print("=" * 80)


if __name__ == "__main__":
    main()