# NCAA Next Man Up

NCAA Next Man Up is an open-source analytics project that transforms publicly available NCAA men’s college basketball data into clear, dashboard-driven insights for player and team performance evaluation. The project is currently being developed with a focus on the University of Montana men’s basketball program, with a framework intended to scale to other NCAA programs assuming comparable public data availability.

## Project Goals
- Create a repeatable workflow to collect, clean, and analyze NCAA men’s basketball player and team performance data.
- Build a dashboard that presents key performance insights in a logical and coach-friendly format.
- Document the full process (data sources, pipeline, metrics, and assumptions) to support replication.

## Key Questions (initial)
- Which players are driving efficiency and production, and how consistent is that contribution over time?
- How do player roles and on-court production change across opponents, lineups, and game contexts?
- What team-level factors best explain wins/losses and performance trends?

## Data Sources (public)
This project will use publicly available sources such as Sports Reference, KenPom, ESPN, and other NCAA-related data providers. See `data/README_data_sources.md` for details on sources, refresh cadence, and how data will be acquired.

## Repository Structure
- `src/` : reusable Python scripts for data collection, cleaning, and feature engineering
- `notebooks/` : exploratory analysis and metric validation
- `dashboard/` : dashboard application files and documentation
- `data/` : raw/processed data folders (data not committed unless permitted); includes data source documentation
- `docs/` : technical documentation (architecture, data dictionary, style guide)
- `reports/` : written report drafts and figures

## How to Run (placeholder)
This section will be finalized once the initial pipeline and dashboard framework are implemented.

## Contact
Cody Thompson
cody.thompson@umconnect.umt.edu 
University of Montana MSBA Capstone (Spring 2026)
