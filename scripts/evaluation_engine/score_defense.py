"""
Build defense evaluation scores for the NCAA Next Man Up project.

Scoring principles
------------------
- Uses player_percentiles.csv as the scoring source.
- Only resolves percentile columns, not raw metric columns.
- Inverts drtg percentile because lower drtg is better.
- Produces defense_score on a 0-100 scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
PLAYER_ARCHETYPES_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "defense_scores.csv"

IDENTIFIER_COLUMNS = ["player_name", "team_name", "season"]

OPTIONAL_ID_COLUMNS = [
    "conference_name",
    "position_group",
    "position_raw",
]

ARCHETYPE_COLUMNS = [
    "player_name",
    "team_name",
    "season",
    "position_group",
    "player_archetype",
    "archetype",
]

DEFENSE_METRIC_CONFIG = {
    "stl_pct": {
        "weight": 0.35,
        "higher_is_better": True,
        "aliases": ["stl_pct", "steal_pct", "stl_rate", "steal_rate"],
    },
    "blk_pct": {
        "weight": 0.25,
        "higher_is_better": True,
        "aliases": ["blk_pct", "block_pct", "blk_rate", "block_rate"],
    },
    "drtg": {
        "weight": 0.40,
        "higher_is_better": False,
        "aliases": ["drtg", "def_rtg", "defensive_rating", "def_rating"],
    },
}

PERCENTILE_SUFFIX_CANDIDATES = [
    "_percentile",
    "_pctile",
    "_perc",
    "_pctl",
    "_percent_rank",
    "_percent_rank_value",
]


def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            if column == "season":
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
            else:
                df[column] = df[column].astype("string").str.strip()
                df.loc[df[column].isin(["nan", "None"]), column] = pd.NA
    return df


def _normalize_position_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "position_group" not in df.columns:
        return df

    mapping = {
        "guard": "Guard",
        "guards": "Guard",
        "g": "Guard",
        "forward": "Forward",
        "forwards": "Forward",
        "f": "Forward",
        "big": "Forward",
        "bigs": "Forward",
        "center": "Forward",
        "centers": "Forward",
        "c": "Forward",
    }

    normalized = df["position_group"].astype("string").str.strip().str.lower().map(mapping)
    df["position_group"] = normalized.fillna(df["position_group"])
    return df


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _standardize_percentile_scale(series: pd.Series) -> pd.Series:
    cleaned = _safe_numeric(series)
    if cleaned.dropna().empty:
        return cleaned

    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")


def _build_percentile_column_candidates(metric_alias: str) -> List[str]:
    return [f"{metric_alias}{suffix}" for suffix in PERCENTILE_SUFFIX_CANDIDATES]


def _find_metric_percentile_column(
    df: pd.DataFrame,
    metric_key: str,
    aliases: List[str],
) -> Optional[str]:
    columns = list(df.columns)
    lower_map = {col.lower(): col for col in columns}

    generated_candidates: List[str] = []
    for alias in aliases:
        generated_candidates.extend(_build_percentile_column_candidates(alias))
    generated_candidates.extend(_build_percentile_column_candidates(metric_key))

    for candidate in generated_candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    percentile_keywords = ["percentile", "pctile", "pctl", "perc", "percent_rank"]

    for alias in [metric_key] + aliases:
        alias_lower = alias.lower()
        for col in columns:
            col_lower = col.lower()
            if alias_lower in col_lower and any(keyword in col_lower for keyword in percentile_keywords):
                return col

    return None


def _deduplicate_on_keys(df: pd.DataFrame, subset: List[str], df_name: str) -> pd.DataFrame:
    duplicate_count = df.duplicated(subset=subset).sum()
    if duplicate_count > 0:
        print(f"[WARN] {df_name} has {duplicate_count} duplicate rows on {subset}. Keeping first occurrence.")
        df = df.drop_duplicates(subset=subset, keep="first").copy()
    return df


def _load_player_percentiles() -> pd.DataFrame:
    _validate_file_exists(PLAYER_PERCENTILES_PATH)
    df = _read_csv(PLAYER_PERCENTILES_PATH)

    _validate_required_columns(df=df, required_columns=IDENTIFIER_COLUMNS, df_name="player_percentiles")

    df = _normalize_text_columns(df, IDENTIFIER_COLUMNS + OPTIONAL_ID_COLUMNS)
    df = _normalize_position_group(df)
    df = _deduplicate_on_keys(df=df, subset=IDENTIFIER_COLUMNS, df_name="player_percentiles")

    return df


def _load_player_archetypes() -> pd.DataFrame:
    _validate_file_exists(PLAYER_ARCHETYPES_PATH)
    df = _read_csv(PLAYER_ARCHETYPES_PATH)

    _validate_required_columns(df=df, required_columns=["player_name", "team_name", "season"], df_name="player_archetype_assignment")

    df = _normalize_text_columns(df, list(set(ARCHETYPE_COLUMNS + ["archetype"])))
    df = _normalize_position_group(df)

    if "player_archetype" not in df.columns and "archetype" in df.columns:
        df["player_archetype"] = df["archetype"]
    elif "archetype" not in df.columns and "player_archetype" in df.columns:
        df["archetype"] = df["player_archetype"]

    keep_columns = [col for col in ["player_name", "team_name", "season", "position_group", "player_archetype", "archetype"] if col in df.columns]
    df = df[keep_columns].copy()
    df = _deduplicate_on_keys(df=df, subset=["player_name", "team_name", "season"], df_name="player_archetype_assignment")

    return df


def _merge_inputs(percentiles_df: pd.DataFrame, archetypes_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = percentiles_df.merge(
        archetypes_df,
        on=["player_name", "team_name", "season"],
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

    merged_df = _normalize_position_group(merged_df)
    return merged_df


def _resolve_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved_columns: Dict[str, str] = {}

    for metric_key, config in DEFENSE_METRIC_CONFIG.items():
        column_name = _find_metric_percentile_column(df=df, metric_key=metric_key, aliases=config["aliases"])
        if column_name is not None:
            resolved_columns[metric_key] = column_name

    if not resolved_columns:
        raise ValueError("No defense percentile columns could be resolved from player_percentiles.csv.")

    return resolved_columns


def _build_component_scores(df: pd.DataFrame, resolved_metric_columns: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    component_score_columns: List[str] = []

    for metric_key, source_column in resolved_metric_columns.items():
        config = DEFENSE_METRIC_CONFIG[metric_key]
        higher_is_better = config["higher_is_better"]

        adjusted_column = f"{metric_key}_defense_component_score"
        raw_percentile = _standardize_percentile_scale(df[source_column])

        df[f"{metric_key}_percentile"] = raw_percentile.round(2)

        if higher_is_better:
            df[adjusted_column] = raw_percentile
        else:
            df[adjusted_column] = 100.0 - raw_percentile

        df[adjusted_column] = df[adjusted_column].clip(lower=0, upper=100).round(2)
        component_score_columns.append(adjusted_column)

    return df, component_score_columns


def _build_weighted_defense_score(df: pd.DataFrame, resolved_metric_columns: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    available_weights = {
        metric_key: DEFENSE_METRIC_CONFIG[metric_key]["weight"]
        for metric_key in resolved_metric_columns
    }

    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        raise ValueError("Total defense metric weight must be greater than zero.")

    normalized_weights = {
        metric_key: weight / total_weight
        for metric_key, weight in available_weights.items()
    }

    weighted_sum = pd.Series(0.0, index=df.index, dtype="float64")
    weight_sum = pd.Series(0.0, index=df.index, dtype="float64")

    for metric_key, normalized_weight in normalized_weights.items():
        component_column = f"{metric_key}_defense_component_score"
        valid_mask = df[component_column].notna()
        weighted_sum.loc[valid_mask] += df.loc[valid_mask, component_column] * normalized_weight
        weight_sum.loc[valid_mask] += normalized_weight

    df["defense_score"] = (weighted_sum / weight_sum.where(weight_sum > 0)).round(2)
    return df


def _add_metadata_columns(df: pd.DataFrame, resolved_metric_columns: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for metric_key, source_column in resolved_metric_columns.items():
        df[f"{metric_key}_percentile_source_column"] = source_column
    return df


def _build_output_table(scored_df: pd.DataFrame, resolved_metric_columns: Dict[str, str]) -> pd.DataFrame:
    output_columns: List[str] = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "position_raw",
        "player_archetype",
        "archetype",
    ]

    raw_percentile_columns: List[str] = []
    component_columns: List[str] = []
    source_columns: List[str] = []

    for metric_key in resolved_metric_columns:
        raw_percentile_columns.append(f"{metric_key}_percentile")
        component_columns.append(f"{metric_key}_defense_component_score")
        source_columns.append(f"{metric_key}_percentile_source_column")

    output_columns.extend([col for col in raw_percentile_columns if col in scored_df.columns])
    output_columns.extend([col for col in component_columns if col in scored_df.columns])
    output_columns.append("defense_score")
    output_columns.extend([col for col in source_columns if col in scored_df.columns])

    output_df = scored_df[[col for col in output_columns if col in scored_df.columns]].copy()
    output_df = output_df.sort_values(by=["season", "team_name", "player_name"], kind="stable").reset_index(drop=True)
    return output_df


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_defense_scores() -> pd.DataFrame:
    percentiles_df = _load_player_percentiles()
    archetypes_df = _load_player_archetypes()

    merged_df = _merge_inputs(percentiles_df=percentiles_df, archetypes_df=archetypes_df)
    resolved_metric_columns = _resolve_metric_columns(merged_df)

    scored_df, _ = _build_component_scores(df=merged_df, resolved_metric_columns=resolved_metric_columns)
    scored_df = _build_weighted_defense_score(df=scored_df, resolved_metric_columns=resolved_metric_columns)
    scored_df = _add_metadata_columns(df=scored_df, resolved_metric_columns=resolved_metric_columns)

    output_df = _build_output_table(scored_df=scored_df, resolved_metric_columns=resolved_metric_columns)
    _write_output(output_df, OUTPUT_PATH)
    return output_df


def main() -> None:
    output_df = build_defense_scores()
    print(f"[SUCCESS] Defense scores written to: {OUTPUT_PATH} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()