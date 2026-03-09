"""
File Name: name_standardization.py
Last Modified: 2026-03-03

Overview:
Standardize player and team names across project data sources.
This module supports both row-level helper functions and dataset-level
standardization for pipeline use.

Inputs:
- DataFrames containing player and/or team name columns
- CSV files passed into the pipeline for name standardization
- Raw player and team names from KenPom, Sports Reference, Team Sites, and exports

Outputs:
- Standardized player names
- Standardized team names
- Standardized CSV outputs when used as a script

Dependencies:
- argparse
- pathlib
- re
- unicodedata
- pandas
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


MULTI_SPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_KEEP_SPACE_PATTERN = re.compile(r"[^a-z0-9\s]")
SUFFIX_PATTERN = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)

PLAYER_COLUMN_CANDIDATES = (
    "player_name",
    "player",
    "athlete_name",
    "name",
    "full_name",
)

TEAM_COLUMN_CANDIDATES = (
    "team_name",
    "team",
    "school",
    "college",
    "team_full_name",
)

OPPONENT_COLUMN_CANDIDATES = (
    "opponent_name",
    "opponent",
    "opp_team",
    "opp",
)

TEAM_NAME_ALIASES = {
    "montana grizzlies": "montana",
    "montana state bobcats": "montana state",
    "northern arizona lumberjacks": "northern arizona",
    "northern colorado bears": "northern colorado",
    "idaho state bengals": "idaho state",
    "portland state vikings": "portland state",
    "sacramento state hornets": "sacramento state",
    "eastern washington eagles": "eastern washington",
    "weber state wildcats": "weber state",
    "idaho vandals": "idaho",
    "montana st": "montana state",
    "weber st": "weber state",
    "portland st": "portland state",
    "sacramento st": "sacramento state",
    "idaho st": "idaho state",
    "n colorado": "northern colorado",
    "n arizona": "northern arizona",
    "e washington": "eastern washington",
    "sac state": "sacramento state",
    "um": "montana",
    "ewu": "eastern washington",
    "unc": "northern colorado",
    "isu": "idaho state",
    "psu": "portland state",
    "msu": "montana state",
}


def _is_missing(value: object) -> bool:
    """Normalize missing text handling before downstream matching."""
    if value is None:
        return True

    if isinstance(value, float) and pd.isna(value):
        return True

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


def _to_string(value: object, field_name: str) -> str:
    """Fail early on unexpected types so bad data is easier to trace."""
    if _is_missing(value):
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)) and not pd.isna(value):
        return str(value)

    raise TypeError(f"{field_name} must be string-like or missing. Received {type(value).__name__}.")


def _strip_accents(value: str) -> str:
    """Accent removal reduces false mismatches across sources."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_text(value: str) -> str:
    """A shared cleaner keeps source-to-source normalization consistent."""
    cleaned = _strip_accents(value)
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("-", " ")
    cleaned = cleaned.replace("/", " ")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace(".", " ")
    cleaned = NON_ALPHANUMERIC_KEEP_SPACE_PATTERN.sub(" ", cleaned)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def _to_title_case(value: str) -> str:
    """Readable outputs help merges, exports, and dashboard labels stay clean."""
    if value == "":
        return ""

    return " ".join(part.capitalize() for part in value.split())


def _resolve_column_name(df: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str | None:
    """Flexible column resolution keeps the utility reusable across sources."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...],
) -> None:
    """Schema checks should fail before transformation starts."""
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")


def standardize_player_name(player_name: object) -> str:
    """
    Standardize a single player name.

    Args:
        player_name: Raw player name from a source dataset.

    Returns:
        Standardized player name.
    """
    value = _to_string(player_name, "player_name")
    if value == "":
        return ""

    cleaned = _normalize_text(value)
    cleaned = SUFFIX_PATTERN.sub("", cleaned)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()

    return _to_title_case(cleaned)


def standardize_team_name(team_name: object) -> str:
    """
    Standardize a single team name.

    Args:
        team_name: Raw team name from a source dataset.

    Returns:
        Standardized team name.
    """
    value = _to_string(team_name, "team_name")
    if value == "":
        return ""

    cleaned = _normalize_text(value)
    cleaned = TEAM_NAME_ALIASES.get(cleaned, cleaned)

    return _to_title_case(cleaned)


def standardize_name_columns(
    df: pd.DataFrame,
    player_column: str | None = None,
    team_column: str | None = None,
    opponent_column: str | None = None,
    add_standardized_columns: bool = True,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """
    Standardize player and team columns in a DataFrame.

    Args:
        df: Input DataFrame.
        player_column: Name of player column. If None, common names are searched.
        team_column: Name of team column. If None, common names are searched.
        opponent_column: Name of opponent column. If None, common names are searched.
        add_standardized_columns: If True, create new standardized columns.
        overwrite_existing: If True, overwrite original source columns.

    Returns:
        Transformed DataFrame.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If no usable target columns are found.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        return df.copy()

    result_df = df.copy()

    resolved_player_column = player_column or _resolve_column_name(
        result_df,
        PLAYER_COLUMN_CANDIDATES,
        "player",
    )
    resolved_team_column = team_column or _resolve_column_name(
        result_df,
        TEAM_COLUMN_CANDIDATES,
        "team",
    )
    resolved_opponent_column = opponent_column or _resolve_column_name(
        result_df,
        OPPONENT_COLUMN_CANDIDATES,
        "opponent",
    )

    if not any([resolved_player_column, resolved_team_column, resolved_opponent_column]):
        raise ValueError(
            "No supported player/team/opponent columns were found. "
            "Pass explicit column names or provide a compatible schema."
        )

    if resolved_player_column and resolved_player_column not in result_df.columns:
        raise ValueError(f"Player column not found: {resolved_player_column}")

    if resolved_team_column and resolved_team_column not in result_df.columns:
        raise ValueError(f"Team column not found: {resolved_team_column}")

    if resolved_opponent_column and resolved_opponent_column not in result_df.columns:
        raise ValueError(f"Opponent column not found: {resolved_opponent_column}")

    if resolved_player_column:
        standardized_player = result_df[resolved_player_column].apply(standardize_player_name)

        if overwrite_existing:
            result_df[resolved_player_column] = standardized_player
        elif add_standardized_columns:
            result_df[f"{resolved_player_column}_standardized"] = standardized_player

    if resolved_team_column:
        standardized_team = result_df[resolved_team_column].apply(standardize_team_name)

        if overwrite_existing:
            result_df[resolved_team_column] = standardized_team
        elif add_standardized_columns:
            result_df[f"{resolved_team_column}_standardized"] = standardized_team

    if resolved_opponent_column:
        standardized_opponent = result_df[resolved_opponent_column].apply(standardize_team_name)

        if overwrite_existing:
            result_df[resolved_opponent_column] = standardized_opponent
        elif add_standardized_columns:
            result_df[f"{resolved_opponent_column}_standardized"] = standardized_opponent

    return result_df


def load_input_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file for standardization.

    Args:
        file_path: Path to input CSV file.

    Returns:
        Loaded DataFrame.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Only CSV input is supported. Received: {path.suffix}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    if df.columns.duplicated().any():
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        duplicate_text = ", ".join(duplicate_columns)
        raise ValueError(f"Duplicate columns found in input file: {duplicate_text}")

    return df


def write_output_file(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Write standardized DataFrame to CSV.

    Args:
        df: Output DataFrame.
        output_path: Destination CSV path.

    Returns:
        Resolved output path.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Refusing to write an empty DataFrame")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    return path.resolve()


def standardize_file(
    input_path: str | Path,
    output_path: str | Path,
    player_column: str | None = None,
    team_column: str | None = None,
    opponent_column: str | None = None,
    add_standardized_columns: bool = True,
    overwrite_existing: bool = False,
) -> Path:
    """
    Load, standardize, and write a dataset.

    Args:
        input_path: Input CSV path.
        output_path: Output CSV path.
        player_column: Explicit player column name if needed.
        team_column: Explicit team column name if needed.
        opponent_column: Explicit opponent column name if needed.
        add_standardized_columns: Whether to create parallel standardized columns.
        overwrite_existing: Whether to replace original values.

    Returns:
        Resolved output path.
    """
    if add_standardized_columns and overwrite_existing:
        raise ValueError(
            "Choose either add_standardized_columns=True or overwrite_existing=True, not both."
        )

    df = load_input_file(input_path)

    standardized_df = standardize_name_columns(
        df=df,
        player_column=player_column,
        team_column=team_column,
        opponent_column=opponent_column,
        add_standardized_columns=add_standardized_columns,
        overwrite_existing=overwrite_existing,
    )

    return write_output_file(standardized_df, output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI support makes the utility usable in pipeline runs and manual checks."""
    parser = argparse.ArgumentParser(
        description="Standardize player and team names in a CSV file."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--player-column", default=None, help="Player name column")
    parser.add_argument("--team-column", default=None, help="Team name column")
    parser.add_argument("--opponent-column", default=None, help="Opponent team column")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite original columns instead of creating standardized columns",
    )
    return parser


def main() -> None:
    """Entry point for file-based standardization runs."""
    parser = build_argument_parser()
    args = parser.parse_args()

    output_path = standardize_file(
        input_path=args.input,
        output_path=args.output,
        player_column=args.player_column,
        team_column=args.team_column,
        opponent_column=args.opponent_column,
        add_standardized_columns=not args.overwrite_existing,
        overwrite_existing=args.overwrite_existing,
    )

    print(f"Standardized file written to: {output_path}")


if __name__ == "__main__":
    main()