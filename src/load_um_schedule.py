"""
File: load_um_schedule.py
Last Modified: 2026-02-01

Overview:
Loads Sports-Reference schedule CSV exports for University of Montana men's basketball,
cleans column names/types, and saves a processed schedule table for analysis and dashboards.

Required Inputs:
- --season (one or more season end years). Example: 2024 = 2023–24 season.

Outputs:
- data/processed/um_schedule_<season>_processed.csv

Sources & Attribution:
- Data Source: Sports-Reference College Basketball
- Export Method: "Get table as CSV (for Excel)" on the schedule page
- Terms of Use:
  Data accessed for non-commercial, educational use in accordance with
  Sports-Reference’s published Terms of Service.
- Attribution:
  “Data courtesy of Sports-Reference.com”
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd


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
        raw_file = f"data/raw/um_schedule_{season_end_year}_raw.csv"

        # Load the raw CSV export
        schedule_df = pd.read_csv(raw_file)

        # Standardize column names to lowercase with underscores
        schedule_df.columns = [str(col).strip().lower().replace(" ", "_") for col in schedule_df.columns]

        # Rename columns to clearer names (Sports-Reference includes unnamed columns)
        rename_map = {
            "g": "game_number",
            "tm": "team_points",
            "opp": "opponent_points",
            "conf": "opponent_conference",
            "ot": "overtime_text",
            "unnamed:_4": "site_flag",
            "unnamed:_8": "result_wl"
        }
        for old_name in rename_map:
            if old_name in schedule_df.columns:
                schedule_df.rename(columns={old_name: rename_map[old_name]}, inplace=True)

        # Add season field
        schedule_df.insert(0, "season", season_end_year)

        # Convert date into a consistent YYYY-MM-DD format
        if "date" in schedule_df.columns:
            schedule_df["date"] = pd.to_datetime(schedule_df["date"], errors="coerce").dt.date.astype(str)

        # Convert numeric columns to numeric types where possible
        for col in ["game_number", "srs", "team_points", "opponent_points", "w", "l"]:
            if col in schedule_df.columns:
                schedule_df[col] = pd.to_numeric(schedule_df[col], errors="coerce")

        # Create a clean location_type field based on site_flag
        # '@' = away, 'N' = neutral, blank/NaN = home
        schedule_df["location_type"] = "home"
        if "site_flag" in schedule_df.columns:
            site_vals = schedule_df["site_flag"].astype(str).str.strip().str.upper()
            schedule_df.loc[site_vals == "@", "location_type"] = "away"
            schedule_df.loc[site_vals == "N", "location_type"] = "neutral"

        # Create overtime_flag based on whether overtime_text is present
        schedule_df["overtime_flag"] = False
        if "overtime_text" in schedule_df.columns:
            schedule_df["overtime_flag"] = schedule_df["overtime_text"].notna() & (
                schedule_df["overtime_text"].astype(str).str.strip() != ""
            )

        # Add metadata fields for tracking and reproducibility
        # Store both UTC and local (Montana) time for convenience
        processed_at_utc = datetime.now(timezone.utc)
        processed_at_local = processed_at_utc.astimezone()  # uses the computer's local timezone (MST/MDT)

        schedule_df["source_file"] = raw_file
        schedule_df["processed_at_utc"] = processed_at_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        schedule_df["processed_at_local"] = processed_at_local.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Save the processed schedule
        output_file = f"data/processed/um_schedule_{season_end_year}_processed.csv"
        print(f"Writing file (overwrites if exists): {output_file}")
        schedule_df.to_csv(output_file, index=False)

        # Print a short summary to validate output
        print("\n--------------------------------------")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(schedule_df)}")
        print(f"Columns: {list(schedule_df.columns)}")
        print(schedule_df.head(8))


if __name__ == "__main__":
    main()
