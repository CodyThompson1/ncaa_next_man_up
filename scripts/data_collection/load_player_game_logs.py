"""
File: load_player_game_logs.py
Last Modified: 2026-02-23
Purpose: Collect current-season individual player game logs from Sports Reference for all Big Sky men's basketball teams and export a consolidated player game log dataset.

Inputs:
- Sports Reference team roster pages
- Sports Reference player game log pages

Outputs:
- data/raw/sports_reference/player_game_logs.csv
"""

import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


# Sources & Attribution:
# - Data Source: Sports Reference
# - Source URL Pattern: https://www.sports-reference.com/cbb/schools/<team_slug>/men/<season>.html and linked player pages
# - Terms of Use: Data accessed for non-commercial, educational use in accordance with the source’s published terms or policies.
# - Attribution: “Data courtesy of Sports Reference”


REQUEST_TIMEOUT = 60
REQUEST_PAUSE_SECONDS = 2
BASE_URL = "https://www.sports-reference.com"
OUTPUT_FILE_NAME = "player_game_logs.csv"
FAILURE_FILE_NAME = "player_game_log_failures.csv"

BIG_SKY_TEAM_SLUGS = {
    "Eastern Washington": "eastern-washington",
    "Idaho": "idaho",
    "Idaho State": "idaho-state",
    "Montana": "montana",
    "Montana State": "montana-state",
    "Northern Arizona": "northern-arizona",
    "Northern Colorado": "northern-colorado",
    "Portland State": "portland-state",
    "Sacramento State": "sacramento-state",
    "Weber State": "weber-state",
}

REQUIRED_COLUMNS = [
    "player",
    "team",
    "game_date",
    "minutes",
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "fg",
    "fga",
    "three_pt",
    "three_pt_attempts",
    "ft",
    "fta",
]


def get_repo_root():
    return Path(__file__).resolve().parents[2]


def get_output_path():
    return get_repo_root() / "data" / "raw" / "sports_reference" / OUTPUT_FILE_NAME


def get_failure_output_path():
    return get_repo_root() / "data" / "raw" / "sports_reference" / FAILURE_FILE_NAME


def get_current_season():
    now = datetime.now()
    if now.month >= 7:
        return now.year + 1
    return now.year


def build_team_url(team_slug, season):
    return f"{BASE_URL}/cbb/schools/{team_slug}/men/{season}.html"


def fetch_page_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ncaa_next_man_up/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise RuntimeError(f"Request failed for {url}: {response.status_code}")

    return response.text


def extract_visible_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment_text = str(comment)
        if "<table" not in comment_text:
            continue
        try:
            fragment = BeautifulSoup(comment_text, "html.parser")
            for table in fragment.find_all("table"):
                table_id = table.get("id")
                if table_id and soup.find("table", {"id": table_id}) is None:
                    soup.append(table)
        except Exception:
            continue

    return str(soup)


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(
            r"_+",
            "_",
            str(col)
            .strip()
            .lower()
            .replace("%", "pct")
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_"),
        ).strip("_")
        for col in df.columns
    ]
    return df


def read_html_tables(html):
    try:
        return pd.read_html(StringIO(html))
    except ValueError:
        return []


def clean_player_name(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    value = re.sub(r"\*", "", value).strip()
    return value or None


def parse_player_href(href):
    if not href:
        return None

    href = str(href).strip()
    match = re.search(r"^/cbb/players/([a-z0-9\-]+)\.html$", href)
    if not match:
        return None

    return {"player_slug": match.group(1)}


def extract_player_records_from_roster_table(roster_table, team_name, team_slug):
    records = []
    tbody = roster_table.find("tbody")
    if tbody is None:
        return records

    for row in tbody.find_all("tr"):
        row_classes = row.get("class") or []
        if "thead" in row_classes:
            continue

        player_cell = row.find(["th", "td"], {"data-stat": "player"})
        if player_cell is None:
            continue

        link = player_cell.find("a")
        if link is None:
            continue

        player_name = clean_player_name(player_cell.get_text(" ", strip=True))
        player_href = link.get("href")
        player_parts = parse_player_href(player_href)

        if not player_name or not player_parts:
            continue

        records.append(
            {
                "player": player_name,
                "team": team_name,
                "team_slug": team_slug,
                "player_slug": player_parts["player_slug"],
                "player_url": urljoin(BASE_URL, player_href),
            }
        )

    return records


def extract_player_records_from_page_links(soup, team_name, team_slug):
    records = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link.get("href", "").strip()
        player_parts = parse_player_href(href)
        if not player_parts:
            continue

        player_name = clean_player_name(link.get_text(" ", strip=True))
        if not player_name:
            continue

        key = (player_name, player_parts["player_slug"])
        if key in seen:
            continue

        seen.add(key)
        records.append(
            {
                "player": player_name,
                "team": team_name,
                "team_slug": team_slug,
                "player_slug": player_parts["player_slug"],
                "player_url": urljoin(BASE_URL, href),
            }
        )

    return records


def parse_team_roster_links(team_name, team_slug, season):
    url = build_team_url(team_slug, season)
    html = fetch_page_html(url)
    soup = BeautifulSoup(extract_visible_html(html), "html.parser")

    records = []

    roster_table = soup.find("table", {"id": "roster"})
    if roster_table is not None:
        records.extend(extract_player_records_from_roster_table(roster_table, team_name, team_slug))

    if not records:
        records.extend(extract_player_records_from_page_links(soup, team_name, team_slug))

    if not records:
        raise RuntimeError(f"No player links found for {team_name}.")

    roster_df = pd.DataFrame(records).drop_duplicates(
        subset=["player", "team", "player_slug", "player_url"]
    )

    return roster_df.reset_index(drop=True)


def build_gamelog_url(player_slug, season):
    return f"{BASE_URL}/cbb/players/{player_slug}/gamelog/{season}"


def extract_gamelog_table(html):
    visible_html = extract_visible_html(html)
    tables = read_html_tables(visible_html)

    for table in tables:
        df = normalize_columns(table)

        if "date" not in df.columns:
            continue

        stat_markers = {"pts", "trb", "ast"}
        if stat_markers.issubset(set(df.columns)):
            return df

    raise RuntimeError("Game log table not found.")


def first_matching_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise RuntimeError(f"Missing expected source column. Candidates: {candidates}")
    return None


def parse_minutes_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return round(minutes + (seconds / 60), 2)
            except ValueError:
                return None

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return None

    return float(numeric)


def parse_game_date(value):
    if pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_game_log_rows(df):
    df = df.copy()

    if "date" in df.columns:
        df = df[df["date"].notna()]
        df = df[~df["date"].astype(str).str.contains("Date", na=False)]

    if "opp" in df.columns:
        df = df[~df["opp"].astype(str).str.contains("Opponent", na=False)]

    if "gs" in df.columns:
        df = df[~df["gs"].astype(str).str.contains("GS", na=False)]

    if "mp" in df.columns:
        df = df[~df["mp"].astype(str).str.contains("Did Not|DNP|Not Active|Inactive", case=False, na=False)]

    return df.reset_index(drop=True)


def build_player_game_log_df(player_name, team_name, gamelog_df):
    gamelog_df = clean_game_log_rows(gamelog_df)

    date_col = first_matching_column(gamelog_df, ["date"])
    minutes_col = first_matching_column(gamelog_df, ["mp", "minutes"])
    points_col = first_matching_column(gamelog_df, ["pts", "points"])
    rebounds_col = first_matching_column(gamelog_df, ["trb", "reb", "rebounds"])
    assists_col = first_matching_column(gamelog_df, ["ast", "assists"])
    steals_col = first_matching_column(gamelog_df, ["stl", "steals"])
    blocks_col = first_matching_column(gamelog_df, ["blk", "blocks"])
    turnovers_col = first_matching_column(gamelog_df, ["tov", "turnovers"])
    fg_col = first_matching_column(gamelog_df, ["fg"])
    fga_col = first_matching_column(gamelog_df, ["fga"])
    three_pt_col = first_matching_column(gamelog_df, ["fg3", "3p", "three_pt"])
    three_pt_attempts_col = first_matching_column(gamelog_df, ["fg3a", "3pa", "three_pt_attempts"])
    ft_col = first_matching_column(gamelog_df, ["ft"])
    fta_col = first_matching_column(gamelog_df, ["fta"])

    output_df = pd.DataFrame(
        {
            "player": player_name,
            "team": team_name,
            "game_date": gamelog_df[date_col].map(parse_game_date),
            "minutes": gamelog_df[minutes_col].map(parse_minutes_value),
            "points": coerce_numeric(gamelog_df[points_col]),
            "rebounds": coerce_numeric(gamelog_df[rebounds_col]),
            "assists": coerce_numeric(gamelog_df[assists_col]),
            "steals": coerce_numeric(gamelog_df[steals_col]),
            "blocks": coerce_numeric(gamelog_df[blocks_col]),
            "turnovers": coerce_numeric(gamelog_df[turnovers_col]),
            "fg": coerce_numeric(gamelog_df[fg_col]),
            "fga": coerce_numeric(gamelog_df[fga_col]),
            "three_pt": coerce_numeric(gamelog_df[three_pt_col]),
            "three_pt_attempts": coerce_numeric(gamelog_df[three_pt_attempts_col]),
            "ft": coerce_numeric(gamelog_df[ft_col]),
            "fta": coerce_numeric(gamelog_df[fta_col]),
        }
    )

    output_df = output_df[output_df["game_date"].notna()]
    output_df = output_df.drop_duplicates(subset=["player", "team", "game_date"])

    return output_df[REQUIRED_COLUMNS].reset_index(drop=True)


def validate_player_game_logs(df, player_name, team_name):
    if df.empty:
        raise RuntimeError(f"No game log data returned for {player_name} ({team_name}).")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(
            f"Game log schema mismatch for {player_name} ({team_name}). Missing columns: {missing_columns}"
        )

    if df["game_date"].isna().all():
        raise RuntimeError(f"All game dates missing for {player_name} ({team_name}).")


def collect_player_game_logs(player_record, season):
    player_name = player_record["player"]
    team_name = player_record["team"]
    player_slug = player_record["player_slug"]

    gamelog_url = build_gamelog_url(player_slug, season)
    html = fetch_page_html(gamelog_url)

    try:
        gamelog_df = extract_gamelog_table(html)
    except RuntimeError as exc:
        if "Game log table not found" in str(exc):
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        raise

    output_df = build_player_game_log_df(player_name, team_name, gamelog_df)
    validate_player_game_logs(output_df, player_name, team_name)

    return output_df


def validate_output(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Output schema mismatch. Missing columns: {missing_columns}")

    if df.empty:
        raise RuntimeError("Final player game log dataset is empty.")

    if df["player"].isna().any():
        raise RuntimeError("Final player game log dataset contains null player values.")

    if df["team"].isna().any():
        raise RuntimeError("Final player game log dataset contains null team values.")

    if df["game_date"].isna().any():
        raise RuntimeError("Final player game log dataset contains null game_date values.")


def clean_output(df):
    df = df.copy()

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.strip()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)

    numeric_columns = [
        "minutes",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fg",
        "fga",
        "three_pt",
        "three_pt_attempts",
        "ft",
        "fta",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(subset=["player", "team", "game_date"])
    df = df.sort_values(["team", "player", "game_date"], ascending=[True, True, True]).reset_index(drop=True)

    return df[REQUIRED_COLUMNS]


def export_csv(df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    season = get_current_season()

    roster_frames = []
    roster_failures = []

    for team_name, team_slug in BIG_SKY_TEAM_SLUGS.items():
        try:
            roster_df = parse_team_roster_links(team_name, team_slug, season)
            roster_frames.append(roster_df)
            time.sleep(REQUEST_PAUSE_SECONDS)
        except Exception as exc:
            roster_failures.append(f"{team_name}: {exc}")

    if roster_failures:
        raise RuntimeError("Roster collection failed for one or more teams:\n" + "\n".join(roster_failures))

    players_df = pd.concat(roster_frames, ignore_index=True)
    players_df = players_df.drop_duplicates(
        subset=["player", "team", "player_slug"]
    ).reset_index(drop=True)

    game_log_frames = []
    game_log_failures = []
    skipped_players = []

    for player_record in players_df.to_dict(orient="records"):
        try:
            player_logs_df = collect_player_game_logs(player_record, season)

            if not player_logs_df.empty:
                game_log_frames.append(player_logs_df)
            else:
                skipped_players.append(
                    {
                        "team": player_record["team"],
                        "player": player_record["player"],
                        "reason": "No game log table found",
                    }
                )
                print(f"Skipping {player_record['player']} ({player_record['team']}): no game log table found.")

            time.sleep(REQUEST_PAUSE_SECONDS)

        except Exception as exc:
            game_log_failures.append(f"{player_record['team']} | {player_record['player']}: {exc}")

    if game_log_failures:
        print("The following players failed during game log load:")
        for failure in game_log_failures:
            print(f" - {failure}")

    if skipped_players:
        skipped_df = pd.DataFrame(skipped_players)
        export_csv(skipped_df, get_failure_output_path())

    if not game_log_frames:
        raise RuntimeError("No player game logs were collected from Sports Reference.")

    combined_df = pd.concat(game_log_frames, ignore_index=True)
    combined_df = clean_output(combined_df)
    validate_output(combined_df)

    output_path = get_output_path()
    export_csv(combined_df, output_path)

    print(f"Sports Reference player game logs saved to: {output_path}")


if __name__ == "__main__":
    main()