"""
Build shooting scores for the NCAA Next Man Up evaluation engine.

Scoring principles
------------------
- Uses player_percentiles.csv as the scoring source.
- Only uses percentile columns.
- Supports percentile values stored either as 0-1 or 0-100.
- Produces a final shooting_score on a 0-100 scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
PLAYER_ARCHETYPE_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "shooting_scores.csv"

KEY_COLUMNS = ["player_name", "team_name", "season"]

METRIC_WEIGHTS: Dict[str, float] = {
    "three_point_pct": 0.35,
    "three_point_attempt_rate": 0.20,
    "effective_field_goal_pct": 0.25,
    "true_shooting_pct": 0.20,
}

PERCENTILE_COLUMN_ALIASES: Dict[str, List[str]] = {
    "three_point_pct": [
        "three_point_pct_percentile",
        "three_pt_pct_percentile",
        "3p_pct_percentile",
        "3pt_pct_percentile",
        "3pt_percentile",
        "3p_percentile",
        "three_point_pct_pctile",
        "three_point_pct_pr",
        "three_point_pct_perc",
    ],
    "three_point_attempt_rate": [
        "three_point_attempt_rate_percentile",
        "three_point_rate_percentile",
        "three_par_percentile",
        "3par_percentile",
        "three_point_attempt_share_percentile",
        "3pa_rate_percentile",
        "3pt_attempt_rate_percentile",
        "3pt_rate_percentile",
        "three_par_pr",
        "three_par_perc",
    ],
    "effective_field_goal_pct": [
        "effective_field_goal_pct_percentile",
        "efg_pct_percentile",
        "efg_percentile",
        "effective_fg_pct_percentile",
        "effective_fg_percentile",
        "efg_pct_pr",
        "efg_pct_perc",
    ],
    "true_shooting_pct": [
        "true_shooting_pct_percentile",
        "ts_pct_percentile",
        "ts_percentile",
        "true_shooting_percentile",
        "ts_pct_pr",
        "ts_pct_perc",
    ],
}


def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _standardize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            if col == "season":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = df[col].astype("string").str.strip()
    return df


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _deduplicate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    duplicate_mask = df.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_rows = df.loc[duplicate_mask, KEY_COLUMNS].drop_duplicates()
        raise ValueError(
            f"{name} contains duplicate player rows on keys {KEY_COLUMNS}:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )
    return df.copy()


def _normalize_position_group(series: pd.Series) -> pd.Series:
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


def _resolve_archetype_table(archetype_df: pd.DataFrame) -> pd.DataFrame:
    archetype_column = None
    for candidate in ["archetype", "player_archetype", "role_archetype"]:
        if candidate in archetype_df.columns:
            archetype_column = candidate
            break

    if archetype_column is None:
        raise ValueError("No archetype column found in player_archetype_assignment.csv")

    keep_columns = [col for col in KEY_COLUMNS + ["position_group", archetype_column] if col in archetype_df.columns]
    archetype_df = archetype_df[keep_columns].copy()
    archetype_df = archetype_df.rename(columns={archetype_column: "archetype"})

    if "position_group" in archetype_df.columns:
        archetype_df["position_group"] = _normalize_position_group(archetype_df["position_group"])

    return _deduplicate(archetype_df, "player_archetype_assignment")


def _find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _normalize_percentile_series(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")

    if cleaned.dropna().empty:
        return cleaned

    max_value = cleaned.max(skipna=True)
    min_value = cleaned.min(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    player_percentiles = _read_csv(PLAYER_PERCENTILES_PATH)
    player_archetypes = _read_csv(PLAYER_ARCHETYPE_PATH)

    player_percentiles = _standardize_columns(player_percentiles, KEY_COLUMNS + ["position_group"])
    player_archetypes = _standardize_columns(player_archetypes, KEY_COLUMNS + ["position_group", "archetype", "player_archetype", "role_archetype"])

    _validate_required_columns(player_percentiles, KEY_COLUMNS, "player_percentiles")
    _validate_required_columns(player_archetypes, KEY_COLUMNS, "player_archetype_assignment")

    player_percentiles = _deduplicate(player_percentiles, "player_percentiles")
    player_archetypes = _resolve_archetype_table(player_archetypes)

    return player_percentiles, player_archetypes


def prepare_scoring_frame(
    player_percentiles: pd.DataFrame,
    player_archetypes: pd.DataFrame,
) -> pd.DataFrame:
    merged = player_percentiles.merge(
        player_archetypes,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_arch"),
        validate="one_to_one",
    )

    if "position_group_arch" in merged.columns:
        if "position_group" not in merged.columns:
            merged["position_group"] = merged["position_group_arch"]
        else:
            merged["position_group"] = merged["position_group"].fillna(merged["position_group_arch"])
        merged = merged.drop(columns=["position_group_arch"])

    if "position_group" in merged.columns:
        merged["position_group"] = _normalize_position_group(merged["position_group"])

    resolved_columns = {}
    for metric_name, aliases in PERCENTILE_COLUMN_ALIASES.items():
        column_name = _find_first_matching_column(merged, aliases)
        if column_name is None:
            raise ValueError(f"Could not find percentile column for {metric_name}. Tried: {aliases}")
        resolved_columns[metric_name] = column_name

    for metric_name, source_column in resolved_columns.items():
        output_column = f"{metric_name}_percentile"
        merged[output_column] = _normalize_percentile_series(merged[source_column])

    return merged


def calculate_shooting_score(scoring_df: pd.DataFrame) -> pd.DataFrame:
    working_df = scoring_df.copy()

    weighted_sum = pd.Series(0.0, index=working_df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=working_df.index, dtype="float64")

    for metric_name, weight in METRIC_WEIGHTS.items():
        percentile_column = f"{metric_name}_percentile"
        component_column = f"{metric_name}_weighted_score"

        working_df[component_column] = working_df[percentile_column] * weight

        valid_mask = working_df[percentile_column].notna()
        weighted_sum.loc[valid_mask] += working_df.loc[valid_mask, percentile_column] * weight
        weight_sum.loc[valid_mask] += weight

    working_df["shooting_score"] = (weighted_sum / weight_sum.where(weight_sum > 0)).round(2)

    for metric_name in METRIC_WEIGHTS:
        percentile_column = f"{metric_name}_percentile"
        component_column = f"{metric_name}_weighted_score"
        working_df[percentile_column] = working_df[percentile_column].round(2)
        working_df[component_column] = working_df[component_column].round(2)

    return working_df


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "position_raw",
        "archetype",
        "three_point_pct_percentile",
        "three_point_attempt_rate_percentile",
        "effective_field_goal_pct_percentile",
        "true_shooting_pct_percentile",
        "three_point_pct_weighted_score",
        "three_point_attempt_rate_weighted_score",
        "effective_field_goal_pct_weighted_score",
        "true_shooting_pct_weighted_score",
        "shooting_score",
    ]

    output_df = df[[col for col in ordered_columns if col in df.columns]].copy()
    output_df = output_df.sort_values(by=["season", "team_name", "player_name"]).reset_index(drop=True)
    _validate_required_columns(
        output_df,
        ["player_name", "team_name", "season", "position_group", "archetype", "shooting_score"],
        "shooting_scores output",
    )
    return output_df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_shooting_scores() -> pd.DataFrame:
    player_percentiles, player_archetypes = load_inputs()
    scoring_frame = prepare_scoring_frame(player_percentiles, player_archetypes)
    scored_df = calculate_shooting_score(scoring_frame)
    output_df = finalize_output(scored_df)
    write_output(output_df, OUTPUT_PATH)
    return output_df


def main() -> None:
    output_df = build_shooting_scores()
    print(f"Shooting scores rows written: {len(output_df)}")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()