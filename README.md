# NCAA Next Man Up

A role-based player evaluation system for NCAA men's basketball that assigns archetypes, constructs peer groups from Big Sky Conference data, and scores players across five performance dimensions using percentile-based rankings. All outputs are delivered through a standalone interactive HTML dashboard.

---

## Project Overview

**NCAA Next Man Up** was built as a capstone project for the University of Montana MSBA program, in partnership with the UM Men's Basketball Program (Head Coach Jay Flores). The system provides a structured, data-driven framework for evaluating players not by raw statistics alone, but by how well they perform within their role relative to conference peers.

The project covers the full analytics pipeline — from raw data collection through cleaning, feature engineering, and evaluation — and delivers results through a coach-friendly, interactive HTML dashboard that requires no server or installation to use.

**Current scope:** 2025–26 University of Montana men's basketball roster (13 players), evaluated within the Big Sky Conference player pool.

**Season Summary (2025–26):**
- Record: 16–15 | Home: 11–6 | Away: 5–9
- ORTG: 109.0 | DRTG: 106.5 | Net Rating: +2.5
- Team TS%: 61.0% | EFG%: 50.3% | TOV Rate: 16.8%

---

## What Was Built

### Evaluation System
- Collected player statistics and game logs for all Montana players and Big Sky Conference peers
- Cleaned and standardized all data for consistent comparisons across programs
- Constructed peer groups by usage rate and position to enable fair, role-appropriate comparisons
- Assigned each player one of 7 archetypes based on their normalized metric profile
- Scored players across 5 performance dimensions using percentile rankings within peer groups
- Applied archetype-adjusted weights to generate a final overall score and letter grade

### Dashboard
- Built a fully standalone HTML/CSS/JavaScript dashboard requiring no server, no installation, and no internet connection to run
- Displays player archetypes, dimension scores, peer group comparisons, and final evaluations
- Designed for direct use by coaching staff

---

## Archetype System

Players are evaluated within one of **7 role archetypes**. Archetype assignment is determined by scoring each player's normalized metric profile against all archetype templates, then selecting the best fit. Dimension weights are adjusted per archetype, so each player's final score reflects what matters most for their role.

| Archetype | Role Description |
|---|---|
| **Primary Creator** | High-usage ball-handler responsible for generating offense through scoring and playmaking |
| **Secondary Playmaker** | Off-ball initiator who contributes assists and secondary creation |
| **Off-Ball Shooter** | Catch-and-shoot specialist who spaces the floor and scores off movement |
| **Scoring Forward** | Versatile interior/wing scorer who operates in mid-range and post situations |
| **3-and-D Forward** | Two-way wing valued for perimeter shooting and defensive contributions |
| **Interior Forward** | Rim-running big who contributes through rebounding, finishing, and interior defense |
| **Glue Forward** | High-efficiency, low-usage big who contributes across multiple areas without high volume |

---

## 2025–26 Montana Roster Evaluations

| Player | # | Yr | Pos | Archetype | Grade | Score |
|---|---|---|---|---|---|---|
| Courtney Anderson | 13 | SO | G | Off-Ball Shooter | A- | 78.5 |
| Tyler Thompson | 4 | SO | G | Off-Ball Shooter | B+ | 70.7 |
| Tyler Isaak | 8 | JR | G | Secondary Playmaker | B | 63.3 |
| Brooklyn Hicks | 3 | JR | G | Off-Ball Shooter | B- | 59.7 |
| Kadyn Betts | 7 | JR | F | Scoring Forward | B- | 57.7 |
| Kenyon Aguino | 24 | FR | F | Scoring Forward | B- | 56.4 |
| Money Williams | 0 | JR | G | Primary Creator | B- | 54.8 |
| Chase Henderson | 2 | JR | G | Primary Creator | B- | 54.0 |
| Grant Kepley | 11 | SO | G | Primary Creator | C+ | 51.6 |
| Tejon Sawyer | 32 | SR | F | 3-and-D Forward | C+ | 51.2 |
| Trae Taylor | 1 | SR | F | Scoring Forward | C+ | 50.7 |
| Amari Jedkins | 5 | JR | F | 3-and-D Forward | C+ | 44.0 |
| Connor Dick | 10 | JR | G | Secondary Playmaker | C | 38.0 |

---

## Repository Structure

```
ncaa_next_man_up/
│
├── dashboard/
│   └── Ncaa_Next_Man_Up.html     # Standalone interactive dashboard (open in any browser)
│
├── data/
│   ├── raw/                      # Raw CSVs from Sports Reference (not committed)
│   ├── processed/                # Cleaned, standardized datasets (not committed)
│   ├── features/                 # Engineered features, percentile scores, dimension scores
│   └── outputs/                  # Final player evaluations
│
├── scripts/
│   ├── data_collection/          # Scripts to pull player and team data from Sports Reference
│   ├── data_cleaning/            # Standardization, validation, and merge scripts
│   ├── feature_engineering/      # Peer group construction, percentile scoring, archetypes
│   ├── evaluation_engine/        # Dimension scoring and final grade calculation
│   │   ├── score_shooting.py
│   │   ├── score_playmaking.py
│   │   ├── score_rebounding.py
│   │   ├── score_defense.py
│   │   ├── score_efficiency.py
│   │   └── build_final_player_grades.py
│   └── utilities/                # Shared config and helper functions
│
├── docs/                         # Project documentation
├── .gitignore
├── data_dictionary.md            # Column-level documentation for all data files
├── requirements.txt
└── README.md
```

---

## Pipeline

Data flows through five sequential stages:

**1. Data Collection** → `scripts/data_collection/`
Scripts pull player season stats, advanced stats, game logs, and roster info from Sports Reference for all Montana players and Big Sky Conference peers. Outputs land in `data/raw/`.

**2. Data Cleaning** → `scripts/data_cleaning/`
Raw files are standardized: column names normalized, data types enforced, duplicates removed, and missing values handled. Outputs land in `data/processed/`.

**3. Feature Engineering** → `scripts/feature_engineering/`
Position groups and usage tiers are built to define peer group membership. Players are matched into peer groups, percentile ranks are calculated within each group, and archetype assignment is run using template scoring. Outputs land in `data/features/`.

**4. Evaluation Engine** → `scripts/evaluation_engine/`
Each of the five performance dimensions is scored independently using percentile ranks and archetype-adjusted weights. Dimension scores are combined into a final overall score and letter grade. Outputs land in `data/outputs/`.

**5. Dashboard** → `dashboard/Ncaa_Next_Man_Up.html`
All evaluation outputs are compiled into a single standalone HTML file. No server required — open directly in any browser.

---

## How to Run

### Prerequisites
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Pipeline (in order)

```bash
# Step 1 — Collect raw data
python scripts/data_collection/load_montana_roster.py
python scripts/data_collection/load_montana_player_stats.py
python scripts/data_collection/load_montana_player_advanced.py
python scripts/data_collection/load_player_game_logs.py
python scripts/data_collection/load_big_sky_player_stats.py
python scripts/data_collection/load_big_sky_player_advanced.py

# Step 2 — Clean and standardize
python scripts/data_cleaning/clean_player_stats.py
python scripts/data_cleaning/clean_player_profiles.py
python scripts/data_cleaning/clean_player_game_logs.py

# Step 3 — Feature engineering
python scripts/feature_engineering/build_position_groups.py
python scripts/feature_engineering/build_player_peer_groups.py
python scripts/feature_engineering/build_player_percentiles.py
python scripts/feature_engineering/assign_player_archetypes.py

# Step 4 — Evaluation engine
python scripts/evaluation_engine/score_shooting.py
python scripts/evaluation_engine/score_playmaking.py
python scripts/evaluation_engine/score_rebounding.py
python scripts/evaluation_engine/score_defense.py
python scripts/evaluation_engine/score_efficiency.py
python scripts/evaluation_engine/build_final_player_grades.py
```

### Viewing the Dashboard
Open `dashboard/Ncaa_Next_Man_Up.html` directly in any web browser. No server required.

---

## Key Outputs

**`data/features/`**
- `player_archetype_assignment.csv` — archetype classification per player
- `player_peer_groups.csv` — peer group membership and construction
- `player_percentiles.csv` — percentile rankings within peer groups
- `player_position_groups.csv` — position groupings used for peer matching
- `shooting_scores.csv`, `playmaking_scores.csv`, `rebounding_scores.csv`, `defense_scores.csv`, `efficiency_scores.csv` — dimension scores per player

**`data/outputs/`**
- `player_final_evaluations.csv` — final overall scores, letter grades, and dimension grades per player

**`dashboard/`**
- `Ncaa_Next_Man_Up.html` — fully standalone interactive dashboard

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Collection | Sports Reference (web scraping via `requests`, `BeautifulSoup`, `lxml`) |
| Data Processing | `pandas`, `NumPy` |
| Evaluation Engine | Custom percentile-based scoring with archetype-adjusted weights |
| Output / Dashboard | Standalone HTML, CSS, JavaScript (no framework required) |
| Utilities | `python-dotenv`, `tqdm` |

> No machine learning models are used. This is an analytics and evaluation system, not a prediction model.

---

## Data Sources

- **Player Statistics & Game Logs:** [Sports Reference College Basketball](https://www.sports-reference.com/cbb/) — season stats, advanced metrics, game-by-game box scores
- **Roster Information:** Sports Reference roster pages — jersey numbers, class, height, hometown

Raw data files are excluded from version control (see `.gitignore`). Run the data collection scripts to regenerate them.

---

## Author & Course

**Author:** Cody Thompson  
**Course:** MSBA Capstone — BMKT 699, University of Montana, Spring 2026  
**Client:** University of Montana Men's Basketball Program (Head Coach Jay Flores)  
**Contact:** cody.thompson@umconnect.umt.edu
