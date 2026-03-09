# NCAA Next Man Up

NCAA Next Man Up is an open-source basketball analytics project that transforms publicly available NCAA men’s basketball data into structured datasets and coach-friendly insights. The project builds a full data pipeline that collects, cleans, and analyzes player and team performance data, with the goal of supporting player evaluation and basketball decision-making.

The project is currently focused on the University of Montana men’s basketball program and the Big Sky Conference. The framework is designed so that it could be expanded to other NCAA programs if comparable public data is available.

---

## Project Goals

- Build a reproducible data pipeline for collecting and processing NCAA basketball data.
- Create structured datasets that support player comparison and evaluation.
- Develop an evaluation engine that compares players against conference peers.
- Build a dashboard that presents insights in a clear and coach-friendly format.
- Document the full process so the workflow can be replicated or extended.

---

## Core Idea: "Next Man Up"

The core idea behind this project is evaluating how effectively a player can fill a role within a team structure.

Players are evaluated by comparing them against **conference peers with similar usage and position groups**. This creates realistic peer comparisons instead of comparing players with completely different roles.

The evaluation process follows several steps:

1. Build the conference player pool  
2. Identify player peer groups based on usage and position  
3. Assign role archetypes  
4. Generate percentile-based metrics  
5. Score players across performance categories  
6. Combine category scores into a final evaluation grade  

---

## Project Pipeline

The project follows a structured analytics pipeline:
Data Collection > Data Cleaning > Feature engineering > Evaluation Engine > Dashboard 


Each stage is implemented through reusable Python scripts.

---

## Data Sources (Public)

This project uses publicly available NCAA basketball data from sources such as:

- Sports Reference (player statistics and game logs)
- KenPom (advanced team metrics)
- ESPN (game schedules and results)

These sources provide enough information to build both player-level and team-level evaluation metrics.

See:
data/README_data_sources.md
for details about each source.

---

## Repository Structure
ncaa_next_man_up
│
├── scripts
│   ├── data_collection
│   │   ├── load_montana_roster.py
│   │   ├── load_big_sky_rosters.py
│   │   ├── load_montana_player_stats.py
│   │   ├── load_big_sky_player_stats.py
│   │   ├── load_montana_player_advanced.py
│   │   ├── load_big_sky_player_advanced.py
│   │   ├── load_player_game_logs.py
│   │   └── load_um_schedule.py
│   │
│   ├── data_cleaning
│   │   ├── clean_player_stats.py
│   │   ├── clean_player_profiles.py
│   │   ├── clean_player_game_logs.py
│   │   ├── clean_team_schedule.py
│   │   └── clean_kenpom_team_master.py
│   │
│   ├── feature_engineering
│   │   └── (peer groups, percentiles, and engineered metrics)
│   │
│   └── evaluation_engine
│       └── (player scoring and final grade calculations)
│
├── data
│   ├── raw
│   │   ├── sports_reference
│   │   └── kenpom
│   │
│   └── processed
│
├── notebooks
│   Exploratory analysis and metric testing
│
├── dashboard
│   Dashboard development files
│
├── docs
│   Technical documentation and project references
│
└── reports
    Written reports and figures
    
---

# Current Development Status

The following stages are currently implemented or in progress:

- Data collection scripts for player, team, and game data
- Data cleaning and validation scripts
- KenPom team metric integration
- Player statistics and game log processing

The next major development stages include:

- Feature engineering
- Player peer group construction
- Player archetype assignment
- Player evaluation scoring
- Dashboard development

---

# How to Run

Once the project setup is complete, the pipeline will typically run in the following order:
1. Run data collection scripts
2. Run data cleaning scripts
3. Run feature engineering scripts
4. Run evaluation engine scripts
5. Load outputs into the dashboard

Instructions will be expanded as the project pipeline is finalized.

---

# Contact

**Cody Thompson**  
University of Montana  
MSBA Capstone Project – Spring 2026  

Email:  
cody.thompson@umconnect.umt.edu