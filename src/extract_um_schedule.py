"""
File: extract_um_schedule.py
Last Modified: 2026-02-01

Overview:
Pulls University of Montana men's basketball schedule and results from Sports-Reference
and saves a clean CSV for downstream analysis and dashboard use. This is currently under construction
and allows users to input season years and to scrape data from sports reference without having to manually
add csv files. For now this is just testing but may be implemented later.

Required Inputs:
- season_end_year (via --season). Example: 2024 = 2023–24 season.

Outputs:
- data/raw/um_schedule_<season>.csv

Sources & Attribution:
- Data Source: Sports-Reference College Basketball
- Source URL Pattern:
  https://www.sports-reference.com/cbb/schools/montana/men/<season>-schedule.html
- Terms of Use:
  Data accessed for non-commercial, educational use in accordance with
  Sports-Reference’s published Terms of Service.
- Attribution:
  “Data courtesy of Sports-Reference.com”

Notes:
- Box score URLs are not captured yet and may be added later.
"""


import argparse
import os
from datetime import datetime

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

    # Make sure the output folder exists before saving files
    os.makedirs("data/raw", exist_ok=True)

    # Loop through each requested season
    for season_end_year in seasons:
        schedule_url = f"https://www.sports-reference.com/cbb/schools/montana/men/{season_end_year}-schedule.html"

        # Pull the schedule table from Sports-Reference
        schedule_table = pd.read_html(schedule_url, attrs={"id": "schedule"})[0]

        # Standardize column names to lowercase with underscores
        schedule_table.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in schedule_table.columns
        ]

        # Remove repeated header rows that sometimes appear inside the table
        if "date" in schedule_table.columns:
            schedule_table = schedule_table[schedule_table["date"].notna()]
            schedule_table = schedule_table[schedule_table["date"].astype(str).str.lower() != "date"]

        # Keep only columns needed for schedule and results analysis
        columns_we_want = []
        for col in ["g", "date", "opponent", "w/l", "result"]:
            if col in schedule_table.columns:
                columns_we_want.append(col)

        clean_table = schedule_table[columns_we_want].copy()

        # Rename columns to clearer, descriptive names
        if "g" in clean_table.columns:
            clean_table.rename(columns={"g": "game_number"}, inplace=True)
        if "w/l" in clean_table.columns:
            clean_table.rename(columns={"w/l": "win_loss"}, inplace=True)

        # Convert the date column to a consistent string format
        clean_table["date"] = pd.to_datetime(
            clean_table["date"], errors="coerce"
        ).dt.date.astype(str)

        # Infer game location based on opponent formatting
        # '@ Opponent' indicates away, 'vs Opponent' indicates neutral
        clean_table["location_type"] = "home"
        clean_table.loc[
            clean_table["opponent"].astype(str).str.startswith("@"),
            "location_type"
        ] = "away"
        clean_table.loc[
            clean_table["opponent"].astype(str).str.startswith("vs"),
            "location_type"
        ] = "neutral"

        # Clean opponent names by removing '@' and 'vs'
        clean_table["opponent"] = (
            clean_table["opponent"].astype(str)
            .str.replace(r"^@\s*", "", regex=True)
            .str.replace(r"^vs\s*", "", regex=True)
            .str.strip()
        )

        # Initialize score-related fields
        clean_table["points_for"] = None
        clean_table["points_against"] = None
        clean_table["overtime_flag"] = False

        # Parse final scores and overtime flags from the result column
        if "result" in clean_table.columns:
            for i in range(len(clean_table)):
                result_text = str(clean_table.iloc[i]["result"])
                clean_table.iloc[i, clean_table.columns.get_loc("overtime_flag")] = "(OT" in result_text

                # Expected formats include: "W 78-72", "L 64-70", "W 82-79 (OT)"
                parts = result_text.split()
                if len(parts) >= 2 and "-" in parts[1]:
                    score_parts = parts[1].split("-")
                    if len(score_parts) == 2:
                        clean_table.iloc[i, clean_table.columns.get_loc("points_for")] = score_parts[0]
                        clean_table.iloc[i, clean_table.columns.get_loc("points_against")] = score_parts[1]

        # Add metadata fields for tracking and reproducibility
        clean_table.insert(0, "season", season_end_year)
        clean_table["source_url"] = schedule_url
        clean_table["scraped_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Save the cleaned schedule to a CSV file
        output_file = f"data/raw/um_schedule_{season_end_year}.csv"
        print(f"Writing file (overwrites if exists): {output_file}")
        clean_table.to_csv(output_file, index=False)

        # Print a short summary to validate the output
        print("\n--------------------------------------")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(clean_table)}")
        print(f"Columns: {list(clean_table.columns)}")
        print(clean_table.head(8))


if __name__ == "__main__":
    main()
