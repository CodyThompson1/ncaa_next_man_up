"""
Build rebounding evaluation scores for NCAA Next Man Up.

Scoring principles
------------------
- Uses player_percentiles.csv as the scoring source.
- Rebounding percentiles are normalized to 0-100.
- Guards and Forwards use different weight profiles.
- Produces rebounding_score on a 0-100 scale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAYER_PERCENTILES_PATH = PROJECT_ROOT / "data" / "features" / "player_percentiles.csv"
ARCHETYPE_ASSIGNMENT_PATH = PROJECT_ROOT / "data" / "features" / "player_archetype_assignment.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "rebounding_scores.csv"

REQUIRED_ID_COLUMNS = ["player_name", "team_name", "season"]
OPTIONAL_ID_COLUMNS = ["conference_name", "position_group"]

REB_METRICS = ["trb_pct", "orb_pct", "drb_pct"]

POSITION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Guard": {
        "trb_pct": 0.50,
        "orb_pct": 0.15,
        "drb_pct": 0.35,
    },
    "Forward": {
        "trb_pct": 0.50,
        "orb_pct": 0.25,
        "drb_pct": 0.25,
    },
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "trb_pct": 0.50,
    "orb_pct": 0.25,
    "drb_pct": 0.25,
}

PERCENTILE_SUFFIX_CANDIDATES = [
    "_percentile",
    "_pctile",
    "_percent_rank",
    "_percent_rank_value",
    "_peer_percentile",
    "_peer_pctile",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_csv(path: Path) -> pd.DataFrame:
    _validate_file_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    logging.info("Loaded %s rows from %s", len(df), path)
    return df


def _normalize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            if col == "season":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = df[col].astype("string").str.strip()
    return df


def _validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _drop_exact_duplicates(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)
    if removed > 0:
        logging.info("Dropped %s exact duplicate rows from %s", removed, df_name)
    return df


def _resolve_archetype_column(df: pd.DataFrame) -> str:
    candidates = ["archetype", "role_archetype", "player_archetype", "assigned_archetype"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not find an archetype column in player_archetype_assignment.csv.")


def _resolve_position_group_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["position_group", "player_position_group", "position_group_final"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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


def _normalize_percentile_series(series: pd.Series) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    if cleaned.dropna().empty:
        return cleaned

    min_value = cleaned.min(skipna=True)
    max_value = cleaned.max(skipna=True)

    if 0 <= min_value and max_value <= 1:
        cleaned = cleaned * 100

    return cleaned.clip(lower=0, upper=100)


def _is_long_percentile_format(df: pd.DataFrame) -> bool:
    metric_candidates = {"metric", "metric_name", "stat", "stat_name"}
    value_candidates = {"percentile", "percentile_value", "pctile", "percent_rank"}

    has_metric_col = any(col in df.columns for col in metric_candidates)
    has_value_col = any(col in df.columns for col in value_candidates)
    return has_metric_col and has_value_col


def _build_wide_from_long(df: pd.DataFrame) -> pd.DataFrame:
    metric_col = next(col for col in ["metric", "metric_name", "stat", "stat_name"] if col in df.columns)
    percentile_col = next(col for col in ["percentile", "percentile_value", "pctile", "percent_rank"] if col in df.columns)

    filtered = df[df[metric_col].isin(REB_METRICS)].copy()
    if filtered.empty:
        raise ValueError("Long-form player_percentiles did not contain required rebounding metrics.")

    filtered = _coerce_numeric(filtered, [percentile_col])

    id_columns = [col for col in REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS if col in filtered.columns]

    wide = (
        filtered.pivot_table(
            index=id_columns,
            columns=metric_col,
            values=percentile_col,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    rename_map = {metric: f"{metric}_percentile" for metric in REB_METRICS if metric in wide.columns}
    return wide.rename(columns=rename_map)


def _find_percentile_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    direct_candidates = [
        f"{metric_name}_percentile",
        f"{metric_name}_pctile",
        f"{metric_name}_percent_rank",
        f"{metric_name}_peer_percentile",
    ]

    lower_map = {col.lower(): col for col in df.columns}

    for col in direct_candidates:
        if col in df.columns:
            return col
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    for suffix in PERCENTILE_SUFFIX_CANDIDATES:
        candidate = f"{metric_name}{suffix}"
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def _prepare_percentile_input(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_exact_duplicates(df, "player_percentiles")
    df = _normalize_string_columns(df, REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS)
    _validate_required_columns(df, REQUIRED_ID_COLUMNS, "player_percentiles")

    if _is_long_percentile_format(df):
        logging.info("Detected long-form player_percentiles input")
        wide_df = _build_wide_from_long(df)
    else:
        logging.info("Detected wide-form player_percentiles input")
        wide_df = df.copy()

    percentile_column_map = {}
    for metric in REB_METRICS:
        percentile_col = _find_percentile_column(wide_df, metric)
        if percentile_col is None:
            raise ValueError(f"Could not find a percentile column for metric '{metric}' in player_percentiles.csv")
        percentile_column_map[metric] = percentile_col

    selected_columns: List[str] = [col for col in REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS if col in wide_df.columns]
    selected_columns.extend(percentile_column_map.values())

    scoring_df = wide_df[selected_columns].copy()
    scoring_df = scoring_df.rename(
        columns={
            percentile_column_map["trb_pct"]: "trb_pct_percentile",
            percentile_column_map["orb_pct"]: "orb_pct_percentile",
            percentile_column_map["drb_pct"]: "drb_pct_percentile",
        }
    )

    scoring_df = _coerce_numeric(scoring_df, ["trb_pct_percentile", "orb_pct_percentile", "drb_pct_percentile"])

    for column in ["trb_pct_percentile", "orb_pct_percentile", "drb_pct_percentile"]:
        scoring_df[column] = _normalize_percentile_series(scoring_df[column])

    return scoring_df


def _prepare_archetype_input(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_exact_duplicates(df, "player_archetype_assignment")
    df = _normalize_string_columns(df, REQUIRED_ID_COLUMNS + OPTIONAL_ID_COLUMNS)
    _validate_required_columns(df, REQUIRED_ID_COLUMNS, "player_archetype_assignment")

    archetype_col = _resolve_archetype_column(df)
    position_group_col = _resolve_position_group_column(df)

    keep_columns = REQUIRED_ID_COLUMNS.copy()
    if position_group_col is not None:
        keep_columns.append(position_group_col)
    keep_columns.append(archetype_col)

    archetype_df = df[keep_columns].copy()

    rename_map = {archetype_col: "archetype"}
    if position_group_col is not None and position_group_col != "position_group":
        rename_map[position_group_col] = "position_group"

    archetype_df = archetype_df.rename(columns=rename_map)
    archetype_df = archetype_df.drop_duplicates(subset=REQUIRED_ID_COLUMNS, keep="first")

    if "position_group" in archetype_df.columns:
        archetype_df["position_group"] = _normalize_position_group(archetype_df["position_group"])

    return archetype_df


def _resolve_weights(position_group: Optional[str]) -> Dict[str, float]:
    if pd.isna(position_group) or position_group is None:
        return DEFAULT_WEIGHTS
    normalized = str(position_group).strip().title()
    return POSITION_WEIGHTS.get(normalized, DEFAULT_WEIGHTS)


def _score_row(row: pd.Series) -> pd.Series:
    weights = _resolve_weights(row.get("position_group"))

    trb_pctile = row.get("trb_pct_percentile")
    orb_pctile = row.get("orb_pct_percentile")
    drb_pctile = row.get("drb_pct_percentile")

    components = {
        "trb_pct_component_score": trb_pctile if pd.notna(trb_pctile) else pd.NA,
        "orb_pct_component_score": orb_pctile if pd.notna(orb_pctile) else pd.NA,
        "drb_pct_component_score": drb_pctile if pd.notna(drb_pctile) else pd.NA,
    }

    weighted_sum = 0.0
    weight_sum = 0.0

    if pd.notna(trb_pctile):
        weighted_sum += trb_pctile * weights["trb_pct"]
        weight_sum += weights["trb_pct"]

    if pd.notna(orb_pctile):
        weighted_sum += orb_pctile * weights["orb_pct"]
        weight_sum += weights["orb_pct"]

    if pd.notna(drb_pctile):
        weighted_sum += drb_pctile * weights["drb_pct"]
        weight_sum += weights["drb_pct"]

    rebounding_score = weighted_sum / weight_sum if weight_sum > 0 else pd.NA

    return pd.Series(
        {
            "trb_pct_weight": weights["trb_pct"],
            "orb_pct_weight": weights["orb_pct"],
            "drb_pct_weight": weights["drb_pct"],
            **components,
            "rebounding_score": round(rebounding_score, 2) if pd.notna(rebounding_score) else pd.NA,
            "rebounding_metrics_available": sum([pd.notna(trb_pctile), pd.notna(orb_pctile), pd.notna(drb_pctile)]),
        }
    )


def _build_rebounding_scores(percentiles_df: pd.DataFrame, archetypes_df: pd.DataFrame) -> pd.DataFrame:
    merged = percentiles_df.merge(
        archetypes_df,
        on=REQUIRED_ID_COLUMNS,
        how="left",
        suffixes=("", "_arch"),
        validate="m:1",
    )

    if "position_group" not in merged.columns and "position_group_arch" in merged.columns:
        merged = merged.rename(columns={"position_group_arch": "position_group"})
    elif "position_group_arch" in merged.columns:
        merged["position_group"] = merged["position_group"].fillna(merged["position_group_arch"])
        merged = merged.drop(columns=["position_group_arch"])

    if "position_group" not in merged.columns:
        raise ValueError("Could not determine position_group after merging player_percentiles and player_archetype_assignment.")

    merged["position_group"] = _normalize_position_group(merged["position_group"])
    scored_components = merged.apply(_score_row, axis=1)
    output_df = pd.concat([merged, scored_components], axis=1)

    ordered_columns = [
        "player_name",
        "team_name",
        "season",
        "conference_name",
        "position_group",
        "archetype",
        "trb_pct_percentile",
        "orb_pct_percentile",
        "drb_pct_percentile",
        "trb_pct_weight",
        "orb_pct_weight",
        "drb_pct_weight",
        "trb_pct_component_score",
        "orb_pct_component_score",
        "drb_pct_component_score",
        "rebounding_score",
        "rebounding_metrics_available",
    ]

    output_df = output_df[[col for col in ordered_columns if col in output_df.columns]].copy()
    output_df = output_df.sort_values(by=["season", "team_name", "player_name"]).reset_index(drop=True)
    return output_df


def build_rebounding_scores() -> pd.DataFrame:
    player_percentiles_df = _load_csv(PLAYER_PERCENTILES_PATH)
    archetype_df = _load_csv(ARCHETYPE_ASSIGNMENT_PATH)

    prepared_percentiles_df = _prepare_percentile_input(player_percentiles_df)
    prepared_archetype_df = _prepare_archetype_input(archetype_df)

    output_df = _build_rebounding_scores(prepared_percentiles_df, prepared_archetype_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    logging.info("Rebounding scores written to %s", OUTPUT_PATH)
    return output_df


def main() -> None:
    output_df = build_rebounding_scores()
    print(f"Rebounding scores written: {len(output_df)} rows")
    print(f"Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()