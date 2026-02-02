"""
File: load_kenpom_four_factors.py
Last Modified: 2026-02-02

Overview:
Loads KenPom Four Factors statistics from the KenPom API (four-factors endpoint),
cleans column names/types, and saves a processed team Four Factors table for
analysis and dashboard development.

Required Inputs:
- --season (one or more season end years).
  Example: 2026 = 2025–26 season.
- --conf_only (optional). If provided, pulls conference-only stats.

Outputs:
- data/processed/kenpom_four_factors_<season>_processed.csv
- data/processed/kenpom_four_factors_conf_only_<season>_processed.csv (if --conf_only is used)
- data/processed/um_kenpom_four_factors_<season>_processed.csv
- data/processed/um_kenpom_four_factors_conf_only_<season>_processed.csv (if --conf_only is used)

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


def fetch_kenpom_four_factors(api_key, season_end_year, conf_only=False):
    """
    Fetch team Four Factors data using:
    Base URL: https://kenpom.com
    Endpoint: /api.php?endpoint=four-factors&y=<year>&conf_only=true|false
    Auth: Authorization: Bearer <API_KEY>
    """
    url = "https://kenpom.com/api.php"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "endpoint": "four-factors",
        "y": season_end_year
    }

    # Only include conf_only param if requested
    if conf_only:
        params["conf_only"] = "true"

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
    parser.add_argument(
        "--conf_only",
        action="store_true",
        help="If set, pulls conference-only Four Factors stats."
    )
    args = parser.parse_args()
    seasons = args.season
    conf_only_flag = args.conf_only

    api_key = os.getenv("KENPOM_API_KEY")
    if api_key is None:
        raise ValueError("KENPOM_API_KEY not found. Make sure it exists in your .env file.")

    # Make sure processed folder exists before saving files
    os.makedirs("data/processed", exist_ok=True)

    # Loop through each requested season
    for season_end_year in seasons:
        # Fetch raw data from KenPom
        raw_data = fetch_kenpom_four_factors(api_key, season_end_year, conf_only=conf_only_flag)

        # Convert API response to DataFrame
        if isinstance(raw_data, list):
            ff_df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict) and "data" in raw_data and isinstance(raw_data["data"], list):
            ff_df = pd.DataFrame(raw_data["data"])
        else:
            ff_df = pd.json_normalize(raw_data)

        # Standardize column names
        ff_df.columns = [str(col).strip().lower().replace(" ", "_") for col in ff_df.columns]

        # Add metadata fields for tracking and reproducibility (UTC + local)
        processed_at_utc = datetime.now(timezone.utc)
        processed_at_local = processed_at_utc.astimezone()

        ff_df["source_name"] = "kenpom_api"
        ff_df["endpoint"] = "four-factors"
        ff_df["processed_at_utc"] = processed_at_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
        ff_df["processed_at_local"] = processed_at_local.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Save the processed file (ALL TEAMS)
        if conf_only_flag:
            output_file = f"data/processed/kenpom_four_factors_conf_only_{season_end_year}_processed.csv"
        else:
            output_file = f"data/processed/kenpom_four_factors_{season_end_year}_processed.csv"

        print(f"Writing file (overwrites if exists): {output_file}")
        ff_df.to_csv(output_file, index=False)

        # Save a University of Montana only file (KenPom team name is usually "Montana")
        um_df = ff_df[ff_df["teamname"].str.strip().str.lower() == "montana"].copy()

        if len(um_df) > 0:
            if conf_only_flag:
                um_output_file = f"data/processed/um_kenpom_four_factors_conf_only_{season_end_year}_processed.csv"
            else:
                um_output_file = f"data/processed/um_kenpom_four_factors_{season_end_year}_processed.csv"

            print(f"Writing file (overwrites if exists): {um_output_file}")
            um_df.to_csv(um_output_file, index=False)
        else:
            print("WARNING: Could not find 'Montana' in teamname column for UM-only output.")

        # Print a short summary to validate output
        print("\n--------------------------------------")
        print(f"Saved: {output_file}")
        print(f"Rows: {len(ff_df)}")
        print(f"Columns: {list(ff_df.columns)}")
        print(ff_df.head(10))


if __name__ == "__main__":
    main()
