"""
File: load_um_roster.py
Last Modified: 2026-02-02

Overview:
Loads Sports-Reference roster CSV exports for University of Montana men's basketball,
cleans column names/types, adds a consistent player_key for future joins, and saves a
processed roster table for analysis.

Required Inputs:
- --season (one or more season end years). Example: 2024 = 2023–24 season.

Expected Raw Files:
- data/raw/um_roster_<season>_raw.csv

Outputs:
- data/processed/um_roster_<season>_processed.csv

Sources & Attribution:
- Data Source: Sports-Reference College Basketball
- Export Method: "Get table as CSV (for Excel)" on the roster table
- Terms of Use:
  Data accessed for non-commercial, educational use in accordance with
  Sports-Reference’s published Terms of Service.
- Attribution:
  “Data courtesy of Sports-Reference.com”
"""

import argparse
import os
import re
from datetime import datetime, timezone

import pandas as pd


def make_player_key(player_name):
    # Create a consistent, join-friendly key from player_name
    if pd.isna(player_name):
        return None

    name = str(player_name).strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)      # remove punctuation
    name = re.sub(r"\s+", "_", name)          # spaces -> underscores
    name = name.replace("-", "_")             # hyphens -> underscores
    name = re.sub(r"_+", "_", name)           # collapse repeated underscores
    return name


def main():
    # Read season(s) from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        nargs="+",
        type=int,
        required=True,
        help="Season end year(s). Example: 2024 (for 2023–24)."
    )
    args = parser.parse_args()
    seasons = args.season

    # Make sure processed folder exists before saving files
    os.makedirs("data/processed", exist_ok=True)

    # Loop through each requested season
    for season_end_year in seasons:
        raw_file = f"data/raw/um_roster_{season_end_year}_raw.csv"

        # Load the raw roster CSV export
        roster_df = pd.read_csv(raw_file)

        # Standardize column names to lowercase with underscores
        roster_df.columns = [str(col).strip().lower().replace(" ", "_") for col in roster_df.columns]

        # Rename common fields to clearer names (only if present)
        rename_map = {
            "pos": "position",
            "wt": "weight",
            "ht": "height",
            "class": "class_year"
        }
        for old_name in rename_map:
            if old_name in roster_df.columns:
                roster_df.rename(columns={old_name: rename_map[old_name]}, inplace=True)

        # Add season field
        roster_df.insert(0, "season", season_end_year)

        # Add player_key if a player name column exists
        if "player" in roster_df.columns:
            roster_df.insert(2, "player_key", roster_df["player"].apply(make_player_key))
        else:
            roster_df["player_key"] = None

        # Convert weight to numeric if present
        if "weight" in roster_df.columns:
            roster_df["weight"] = pd.to_numeric(roster_df["weight"], errors="coerce")

        # Keep original height and also create height_in for numeric analysis
        roster_df["height_in"] = None
        if "height" in roster_df.columns:
            height_text = roster_df["height"].astype(str).str.strip()
            height_in_list = []

            for h in height_text:
                if "-" in h:
                    parts = h.split("-")
                    if len(parts) == 2:
                        feet = pd.to_numeric(parts[0], errors="coerce")
                        inches = pd.to_numeric(parts[1], errors="coerce")
                        if pd.notna(feet) and pd.notna(inches):
                            height_in_list.append(int(feet * 12 + inches))
                        else:
                            height_in_list.append(None)
                    else:
                        height_in_list.append(None)
                else:
                    height_in_list.append(None)

            roster_df["height_in"] = height_in_list

        # Add metadata fields for tracking and reproducibility (UTC + local MT)
        processed_at_utc = datetime.now(timezone.utc)
        processed_at_local = processed_at_utc.astimezone()

        roster_df["source_file"] = raw_file
        roster_df["processed_at_utc"] = processed_at_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        roster_df["processed_at_local"] = processed_at_local.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Save the processed roster
        output_file = f"data/processed/um_roster_{season_end_year}_processed.csv"
        print(f"Writing file (overwrites if exists): {output_file}")
        roster_df.to_csv(output_file, index=False)

        # Print a short summary to validate output
        print("\n--------------------------------------")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(roster_df)}")
        print(f"Columns: {list(roster_df.columns)}")
        print(roster_df.head(10))


if __name__ == "__main__":
    main()
