"""
Build playmaking scores for the NCAA Next Man Up evaluation engine.

Scoring principles
------------------
- Uses player_percentiles.csv as the scoring source.
- Only uses percentile columns when available.
- Falls back to raw metric -> percentile conversion only if needed.
- Produces playmaking_score on a 0-100 scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
PLAYER_ARCHETYPE_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "playmaking_scores.csv"

KEY_COLUMNS = ["player_name", "team_name", "season"]

PLAYMAKING_METRIC_WEIGHTS: Dict[str, float] = {
    "ast_pct": 0.45,
    "ast": 0.25,
    "tov_pct": 0.20,
    "creation_proxy": 0.10,
}

INVERSE_METRICS = {"tov_pct"}

METRIC_CANDIDATES: Dict[str, List[str]] = {
    "ast_pct": [
        "ast_pct_percentile",
        "ast_pct_pctile",
        "ast_pct_percent_rank",
        "ast_pct_pr",
        "assist_rate_percentile",
        "assist_rate_pctile",
        "assist_rate_percent_rank",
        "assist_rate_pr",
        "ast_pct",
        "assist_rate",
    ],
    "ast": [
        "ast_percentile",
        "ast_pctile",
        "ast_percent_rank",
        "ast_pr",
        "assists_percentile",
        "assists_pctile",
        "assists_percent_rank",
        "assists_pr",
        "ast",
        "assists",
        "assists_per_game_percentile",
        "assists_per_game_pctile",
        "assists_per_game_percent_rank",
        "assists_per_game_pr",
        "assists_per_game",
        "ast_per_game",
    ],
    "tov_pct": [
        "tov_pct_percentile",
        "tov_pct_pctile",
        "tov_pct_percent_rank",
        "tov_pct_pr",
        "turnover_rate_percentile",
        "turnover_rate_pctile",
        "turnover_rate_percent_rank",
        "turnover_rate_pr",
        "turnover_pct_percentile",
        "turnover_pct_pctile",
        "turnover_pct_percent_rank",
        "turnover_pct_pr",
        "tov_pct",
        "turnover_rate",
        "turnover_pct",
    ],
    "creation_proxy": [
        "creation_load_percentile",
        "creation_load_pctile",
        "creation_load_percent_rank",
        "creation_load_pr",
        "playmaking_load_percentile",
        "playmaking_load_pctile",
        "playmaking_load_percent_rank",
        "playmaking_load_pr",
        "usage_adjusted_creation_percentile",
        "usage_adjusted_creation_pctile",
        "usage_adjusted_creation_percent_rank",
        "usage_adjusted_creation_pr",
        "ast_to_tov_ratio_percentile",
        "ast_to_tov_ratio_pctile",
        "ast_to_tov_ratio_percent_rank",
        "ast_to_tov_ratio_pr",
        "assist_to_turnover_ratio_percentile",
        "assist_to_turnover_ratio_pctile",
        "assist_to_turnover_ratio_percent_rank",
        "assist_to_turnover_ratio_pr",
        "ast_to_tov_ratio",
        "assist_to_turnover_ratio",
        "creation_load",
        "playmaking_load",
        "usage_adjusted_creation",
    ],
}

PERCENTILE_SUFFIXES = ["_percentile", "_pctile", "_percent_rank", "_pr"]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: List[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _standardize_key_columns(df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for column in key_columns:
        if column not in df.columns:
            continue
        if column == "season":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        else:
            df[column] = df[column].astype("string").str.strip()
    return df


def _drop_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset).copy()


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if column in KEY_COLUMNS:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        # Only replace the column if at least one real numeric value was found.
        if converted.notna().any():
            df[column] = converted

    return df


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


def _is_percentile_column(column_name: str) -> bool:
    return any(column_name.endswith(suffix) for suffix in PERCENTILE_SUFFIXES)


def _scale_percentile_series(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")

    if cleaned.dropna().empty:
        return cleaned

    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def _reverse_percentile_series(series: pd.Series) -> pd.Series:
    return (100 - series).clip(lower=0, upper=100)


def _build_percentile_from_raw(df: pd.DataFrame, source_column: str, inverse: bool = False) -> pd.Series:
    raw = pd.to_numeric(df[source_column], errors="coerce")
    percentile = raw.rank(method="average", pct=True) * 100
    if inverse:
        percentile = 100 - percentile
    return percentile.clip(lower=0, upper=100)


def _resolve_metric_column(df: pd.DataFrame, logical_metric: str) -> Optional[str]:
    candidates = METRIC_CANDIDATES[logical_metric]
    lower_map = {col.lower(): col for col in df.columns}

    percentile_matches = []
    raw_matches = []

    for column in candidates:
        if column in df.columns:
            if _is_percentile_column(column):
                percentile_matches.append(column)
            else:
                raw_matches.append(column)
        elif column.lower() in lower_map:
            matched = lower_map[column.lower()]
            if _is_percentile_column(matched):
                percentile_matches.append(matched)
            else:
                raw_matches.append(matched)

    if percentile_matches:
        return percentile_matches[0]
    if raw_matches:
        return raw_matches[0]
    return None


def _prepare_archetype_table(archetype_df: pd.DataFrame) -> pd.DataFrame:
    archetype_df = _standardize_key_columns(archetype_df, KEY_COLUMNS)

    rename_map = {}
    if "role_archetype" in archetype_df.columns and "archetype" not in archetype_df.columns:
        rename_map["role_archetype"] = "archetype"
    if "player_archetype" in archetype_df.columns and "archetype" not in archetype_df.columns:
        rename_map["player_archetype"] = "archetype"

    if rename_map:
        archetype_df = archetype_df.rename(columns=rename_map)

    keep_columns = [col for col in KEY_COLUMNS + ["position_group", "archetype"] if col in archetype_df.columns]
    archetype_df = archetype_df[keep_columns].copy()
    archetype_df = _drop_exact_duplicates(archetype_df, subset=KEY_COLUMNS)

    if "position_group" in archetype_df.columns:
        archetype_df["position_group"] = _normalize_position_group(archetype_df["position_group"])

    return archetype_df


def _build_metric_percentile_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    resolved_map: Dict[str, str] = {}

    for logical_metric in PLAYMAKING_METRIC_WEIGHTS:
        source_column = _resolve_metric_column(df, logical_metric)
        if source_column is None:
            continue

        output_column = f"{logical_metric}_score_component"
        inverse = logical_metric in INVERSE_METRICS

        if _is_percentile_column(source_column):
            component = _scale_percentile_series(df[source_column])
            if inverse:
                component = _reverse_percentile_series(component)
        else:
            component = _build_percentile_from_raw(df, source_column, inverse=inverse)

        df[output_column] = component.round(2)
        resolved_map[logical_metric] = source_column

    return df, resolved_map


def _compute_weighted_score(df: pd.DataFrame, resolved_map: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    component_columns = []
    component_weights = {}

    for logical_metric, weight in PLAYMAKING_METRIC_WEIGHTS.items():
        column = f"{logical_metric}_score_component"
        if logical_metric in resolved_map and column in df.columns:
            component_columns.append(column)
            component_weights[column] = weight

    if not component_columns:
        raise ValueError("No usable playmaking metrics were found in player_percentiles.csv.")

    weighted_value_sum = pd.Series(0.0, index=df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for column in component_columns:
        weight = component_weights[column]
        valid_mask = df[column].notna()
        weighted_value_sum.loc[valid_mask] += df.loc[valid_mask, column] * weight
        weight_sum.loc[valid_mask] += weight

    df["playmaking_score"] = (weighted_value_sum / weight_sum.where(weight_sum > 0)).round(2)

    metric_count = pd.Series(0, index=df.index, dtype="int64")
    for column in component_columns:
        metric_count += df[column].notna().astype(int)

    df["playmaking_metric_count"] = metric_count
    df["playmaking_weight_sum_used"] = weight_sum.round(4)

    return df


def _add_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def categorize(score: float) -> Optional[str]:
        if pd.isna(score):
            return pd.NA
        if score >= 90:
            return "Elite"
        if score >= 75:
            return "Strong"
        if score >= 60:
            return "Solid"
        if score >= 40:
            return "Developing"
        return "Limited"

    df["playmaking_score_band"] = df["playmaking_score"].apply(categorize)
    return df


def _select_output_columns(df: pd.DataFrame, resolved_map: Dict[str, str]) -> pd.DataFrame:
    output_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "position_raw",
        "archetype",
        "usg_pct",
        "usage_window_low",
        "usage_window_high",
    ]

    for logical_metric in PLAYMAKING_METRIC_WEIGHTS:
        component_column = f"{logical_metric}_score_component"
        source_column = f"{logical_metric}_source_metric"
        if logical_metric in resolved_map:
            df[source_column] = resolved_map[logical_metric]
            if source_column not in output_columns:
                output_columns.append(source_column)
            if component_column in df.columns and component_column not in output_columns:
                output_columns.append(component_column)

    output_columns.extend(
        [
            "playmaking_metric_count",
            "playmaking_weight_sum_used",
            "playmaking_score",
            "playmaking_score_band",
        ]
    )

    final_df = df[[col for col in output_columns if col in df.columns]].copy()
    return final_df


def build_playmaking_scores() -> pd.DataFrame:
    percentiles_df = _read_csv(PLAYER_PERCENTILES_PATH)
    archetype_df = _read_csv(PLAYER_ARCHETYPE_PATH)

    _validate_required_columns(percentiles_df, KEY_COLUMNS, "player_percentiles.csv")
    _validate_required_columns(archetype_df, KEY_COLUMNS, "player_archetype_assignment.csv")

    percentiles_df = _standardize_key_columns(percentiles_df, KEY_COLUMNS + ["position_group"])
    archetype_df = _prepare_archetype_table(archetype_df)

    percentiles_df = _drop_exact_duplicates(percentiles_df, subset=KEY_COLUMNS)
    percentiles_df = _coerce_numeric_columns(percentiles_df)

    merged_df = percentiles_df.merge(
        archetype_df,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_archetype"),
        validate="one_to_one",
    )

    if "position_group_archetype" in merged_df.columns:
        if "position_group" not in merged_df.columns:
            merged_df["position_group"] = merged_df["position_group_archetype"]
        else:
            merged_df["position_group"] = merged_df["position_group"].fillna(merged_df["position_group_archetype"])
        merged_df = merged_df.drop(columns=["position_group_archetype"])

    if "position_group" in merged_df.columns:
        merged_df["position_group"] = _normalize_position_group(merged_df["position_group"])

    merged_df, resolved_map = _build_metric_percentile_columns(merged_df)

    if not resolved_map:
        raise ValueError("No playmaking metrics could be resolved from player_percentiles.csv.")

    scored_df = _compute_weighted_score(merged_df, resolved_map)
    scored_df = _add_score_bands(scored_df)
    scored_df = _select_output_columns(scored_df, resolved_map)

    scored_df = scored_df.sort_values(
        by=["season", "team_name", "playmaking_score", "player_name"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return scored_df


def write_playmaking_scores(df: pd.DataFrame, output_path: Path) -> None:
    _ensure_parent_dir(output_path)
    df.to_csv(output_path, index=False)


def main() -> None:
    playmaking_scores_df = build_playmaking_scores()
    write_playmaking_scores(playmaking_scores_df, OUTPUT_PATH)
    print(f"Playmaking scores written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()