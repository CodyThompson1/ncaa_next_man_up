"""
Build efficiency category scores for the NCAA Next Man Up evaluation engine.

Scoring principles
------------------
- Uses player_percentiles.csv as the scoring source.
- Prefers percentile columns.
- If needed, computes percentiles from raw metrics.
- Normalizes every component to 0-100 before weighting.
- Produces efficiency_score on a 0-100 scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
ARCHETYPE_ASSIGNMENT_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "efficiency_scores.csv"

REQUIRED_ID_COLUMNS = ["player_name", "team_name", "season"]
OPTIONAL_ID_COLUMNS = ["conference_name", "position_group"]


@dataclass(frozen=True)
class MetricRule:
    metric_name: str
    weight: float
    higher_is_better: bool = True


METRIC_CONFIG: list[MetricRule] = [
    MetricRule(metric_name="ortg", weight=0.35, higher_is_better=True),
    MetricRule(metric_name="ts_pct", weight=0.30, higher_is_better=True),
    MetricRule(metric_name="efg_pct", weight=0.25, higher_is_better=True),
    MetricRule(metric_name="turnover_rate", weight=0.10, higher_is_better=False),
]


def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")


def _standardize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = df.copy()

    if "player_name" in standardized_df.columns:
        standardized_df["player_name"] = standardized_df["player_name"].astype("string").str.strip()

    if "team_name" in standardized_df.columns:
        standardized_df["team_name"] = standardized_df["team_name"].astype("string").str.strip()

    if "season" in standardized_df.columns:
        standardized_df["season"] = pd.to_numeric(standardized_df["season"], errors="coerce").astype("Int64")

    return standardized_df


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


def _drop_duplicate_players(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    before_count = len(df)
    deduped_df = df.drop_duplicates(subset=REQUIRED_ID_COLUMNS).copy()
    after_count = len(deduped_df)

    if after_count < before_count:
        print(f"{df_name}: dropped {before_count - after_count} duplicate rows on {REQUIRED_ID_COLUMNS}")

    return deduped_df


def _attach_archetypes(percentiles_df: pd.DataFrame, archetypes_df: pd.DataFrame) -> pd.DataFrame:
    archetype_column = None
    for candidate in ["archetype", "player_archetype", "role_archetype"]:
        if candidate in archetypes_df.columns:
            archetype_column = candidate
            break

    if archetype_column is None:
        raise ValueError("No archetype column found in player_archetype_assignment.csv")

    keep_columns = [col for col in REQUIRED_ID_COLUMNS + ["position_group", archetype_column] if col in archetypes_df.columns]
    archetypes_df = archetypes_df[keep_columns].copy().rename(columns={archetype_column: "archetype"})

    merged_df = percentiles_df.merge(
        archetypes_df,
        on=REQUIRED_ID_COLUMNS,
        how="left",
        suffixes=("", "_arch"),
        validate="one_to_one",
    )

    if "position_group_arch" in merged_df.columns:
        if "position_group" not in merged_df.columns:
            merged_df["position_group"] = merged_df["position_group_arch"]
        else:
            merged_df["position_group"] = merged_df["position_group"].fillna(merged_df["position_group_arch"])
        merged_df = merged_df.drop(columns=["position_group_arch"])

    if "position_group" in merged_df.columns:
        merged_df["position_group"] = _normalize_position_group(merged_df["position_group"])

    return merged_df


def _ensure_position_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position_group" not in df.columns:
        df["position_group"] = np.nan
    return df


def _is_percentile_scale(series: pd.Series) -> bool:
    cleaned = pd.to_numeric(series, errors="coerce")
    if cleaned.dropna().empty:
        return False
    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)
    return 0 <= min_value and max_value <= 100


def _normalize_percentile_series(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    if cleaned.dropna().empty:
        return cleaned

    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def _build_percentile_from_raw(series: pd.Series, higher_is_better: bool) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    percentile = cleaned.rank(method="average", pct=True) * 100
    if not higher_is_better:
        percentile = 100 - percentile
    return percentile.clip(lower=0, upper=100)


def _resolve_percentile_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    candidates = [
        f"{metric_name}_percentile",
        f"{metric_name}_pctile",
        f"{metric_name}_percent_rank",
        f"{metric_name}_pr",
    ]
    lower_map = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for column in df.columns:
        lower = column.lower()
        if metric_name.lower() in lower and any(token in lower for token in ["percentile", "pctile", "percent_rank", "pr"]):
            return column

    return None


def _resolve_raw_metric_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    alias_map = {
        "ortg": ["ortg", "offensive_rating"],
        "ts_pct": ["ts_pct", "true_shooting_pct", "true_shooting_percentage"],
        "efg_pct": ["efg_pct", "effective_field_goal_pct", "effective_fg_pct"],
        "turnover_rate": ["turnover_rate", "turnover_pct", "tov_pct"],
    }

    lower_map = {col.lower(): col for col in df.columns}
    for alias in alias_map.get(metric_name, [metric_name]):
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _build_metric_component(df: pd.DataFrame, rule: MetricRule) -> tuple[pd.Series, str]:
    percentile_column = _resolve_percentile_column(df, rule.metric_name)
    if percentile_column is not None:
        component = _normalize_percentile_series(df[percentile_column])
        if not rule.higher_is_better:
            component = 100 - component
        return component.clip(lower=0, upper=100), percentile_column

    raw_column = _resolve_raw_metric_column(df, rule.metric_name)
    if raw_column is not None:
        component = _build_percentile_from_raw(df[raw_column], higher_is_better=rule.higher_is_better)
        return component, raw_column

    return pd.Series(np.nan, index=df.index, dtype="float64"), ""


def _build_efficiency_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    weighted_sum = pd.Series(0.0, index=df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for rule in METRIC_CONFIG:
        component, source_column = _build_metric_component(df, rule)

        percentile_output_column = f"{rule.metric_name}_percentile"
        weighted_output_column = f"{rule.metric_name}_weighted_score"
        source_output_column = f"{rule.metric_name}_source_column"

        df[percentile_output_column] = component.round(2)
        df[weighted_output_column] = (component * rule.weight).round(2)
        df[source_output_column] = source_column if source_column else pd.NA

        valid_mask = component.notna()
        weighted_sum.loc[valid_mask] += component.loc[valid_mask] * rule.weight
        weight_sum.loc[valid_mask] += rule.weight

    df["efficiency_score"] = (weighted_sum / weight_sum.where(weight_sum > 0)).round(2)

    return df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "archetype",
        "ortg_percentile",
        "ts_pct_percentile",
        "efg_pct_percentile",
        "turnover_rate_percentile",
        "ortg_weighted_score",
        "ts_pct_weighted_score",
        "efg_pct_weighted_score",
        "turnover_rate_weighted_score",
        "ortg_source_column",
        "ts_pct_source_column",
        "efg_pct_source_column",
        "turnover_rate_source_column",
        "efficiency_score",
    ]

    output_df = df[[col for col in ordered_columns if col in df.columns]].copy()
    return output_df.sort_values(by=["season", "team_name", "player_name"]).reset_index(drop=True)


def _validate_output(df: pd.DataFrame) -> None:
    _validate_required_columns(
        df=df,
        required_columns=["player_name", "team_name", "season", "position_group", "archetype", "efficiency_score"],
        df_name="efficiency_scores output",
    )

    duplicated_keys = df.duplicated(subset=REQUIRED_ID_COLUMNS, keep=False)
    if duplicated_keys.any():
        duplicate_rows = df.loc[duplicated_keys, REQUIRED_ID_COLUMNS]
        raise ValueError(
            "Duplicate player-team-season rows detected in output:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )


def build_efficiency_scores() -> pd.DataFrame:
    _validate_file_exists(PLAYER_PERCENTILES_PATH)
    _validate_file_exists(ARCHETYPE_ASSIGNMENT_PATH)

    percentiles_df = _read_csv(PLAYER_PERCENTILES_PATH)
    archetypes_df = _read_csv(ARCHETYPE_ASSIGNMENT_PATH)

    _validate_required_columns(percentiles_df, REQUIRED_ID_COLUMNS, "player_percentiles")
    _validate_required_columns(archetypes_df, REQUIRED_ID_COLUMNS, "player_archetype_assignment")

    percentiles_df = _standardize_key_columns(percentiles_df)
    archetypes_df = _standardize_key_columns(archetypes_df)

    percentiles_df = _drop_duplicate_players(percentiles_df, "player_percentiles")
    archetypes_df = _drop_duplicate_players(archetypes_df, "player_archetype_assignment")

    merged_df = _attach_archetypes(percentiles_df, archetypes_df)
    merged_df = _ensure_position_group(merged_df)

    for optional_column in OPTIONAL_ID_COLUMNS:
        if optional_column not in merged_df.columns:
            merged_df[optional_column] = np.nan

    scored_df = _build_efficiency_scores(merged_df)
    output_df = _select_output_columns(scored_df)

    _validate_output(output_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Efficiency scores saved: {len(output_df):,} rows")
    print(f"Output written to: {OUTPUT_PATH}")

    return output_df


def main() -> None:
    build_efficiency_scores()


if __name__ == "__main__":
    main()