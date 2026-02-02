"""
File: load_kenpom_team_overview.py
Last Modified: 2026-02-02

Overview:
Loads team-level season statistics from the KenPom API (ratings endpoint),
cleans column names/types, and saves a processed team overview table for
analysis and dashboard development.

Required Inputs:
- --season (one or more season end years).
  Example: 2026 = 2025–26 season.

Outputs:
- data/processed/kenpom_team_overview_<season>_processed.csv
- data/processed/um_kenpom_team_overview_<season>_processed.csv

Sources & Attribution:
- Data Source: KenPom (official API, paid access)
- Terms of Use:
  Data accessed for non-commercial, educational use in accordance with
  KenPom’s published Terms of Service and API guidelines.
- Attribution:
  “Data courtesy of KenPom.com”
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


def fetch_kenpom_ratings(api_key, season_end_year):
    """
    Fetch team-level KenPom ratings for a given season end year using:
    Base URL: https://kenpom.com
    Endpoint: /api.php?endpoint=ratings&y=<year>
    Auth: Authorization: Bearer <API_KEY>
    """
    url = "https://kenpom.com/api.php"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "endpoint": "ratings",
        "y": season_end_year
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    # Read season(s) from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        nargs="+",
        type=int,
        required=True,
        help="Season end year(s). Example: 2026 (for 2025–26)."
    )
    args = parser.parse_args()
    seasons = args.season

    api_key = os.getenv("KENPOM_API_KEY")
    if api_key is None:
        raise ValueError("KENPOM_API_KEY not found. Make sure it exists in your .env file.")

    # Make sure processed folder exists before saving files
    os.makedirs("data/processed", exist_ok=True)

    # Loop through each requested season
    for season_end_year in seasons:
        # Fetch raw data from KenPom
        raw_data = fetch_kenpom_ratings(api_key, season_end_year)

        # Convert API response to DataFrame
        # Most KenPom endpoints return a list of dict rows.
        if isinstance(raw_data, list):
            team_df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], list):
            team_df = pd.DataFrame(raw_data["data"])
        else:
            team_df = pd.json_normalize(raw_data)

        # Standardize column names
        team_df.columns = [str(col).strip().lower().replace(" ", "_") for col in team_df.columns]

        # Add metadata fields for tracking and reproducibility (UTC + local)
        processed_at_utc = datetime.now(timezone.utc)
        processed_at_local = processed_at_utc.astimezone()

        team_df["source_name"] = "kenpom_api"
        team_df["endpoint"] = "ratings"
        team_df["processed_at_utc"] = processed_at_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        team_df["processed_at_local"] = processed_at_local.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Save the processed team overview file (ALL TEAMS)
        output_file = f"data/processed/kenpom_team_overview_{season_end_year}_processed.csv"
        print(f"Writing file (overwrites if exists): {output_file}")
        team_df.to_csv(output_file, index=False)

        # Save a University of Montana only file (KenPom team name is usually "Montana")
        um_df = team_df[team_df["teamname"].str.strip().str.lower() == "montana"].copy()

        if len(um_df) > 0:
            um_output_file = f"data/processed/um_kenpom_team_overview_{season_end_year}_processed.csv"
            print(f"Writing file (overwrites if exists): {um_output_file}")
            um_df.to_csv(um_output_file, index=False)
        else:
            print("WARNING: Could not find 'Montana' in teamname column for UM-only output.")

        # Print a short summary to validate output
        print("\n--------------------------------------")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(team_df)}")
        print(f"Columns: {list(team_df.columns)}")
        print(team_df.head(10))


if __name__ == "__main__":
    main()
