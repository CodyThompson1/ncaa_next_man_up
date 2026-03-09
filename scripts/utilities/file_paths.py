"""
File Name: file_paths.py
Last Modified: 2026-02-13

Overview:
Centralized file path definitions for the NCAA Next Man Up project.
This module ensures all scripts reference consistent locations for
data inputs, processed outputs, engineered features, and dashboard exports.

Inputs:
- Project directory structure
- Configuration paths from utilities.config

Outputs:
- Reusable Path objects for project data locations
"""

from pathlib import Path

from scripts.utilities.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    OUTPUTS_DIR,
    DASHBOARD_OUTPUTS_DIR,
)


# -----------------------------------------------------------------------------
# Raw Data Paths
# -----------------------------------------------------------------------------

RAW_KENPOM = RAW_DATA_DIR / "kenpom"
RAW_SPORTS_REFERENCE = RAW_DATA_DIR / "sports_reference"
RAW_TEAM_SITE = RAW_DATA_DIR / "team_site"
RAW_EXPORTS = RAW_DATA_DIR / "exports"


# -----------------------------------------------------------------------------
# Interim / Cleaned Data
# -----------------------------------------------------------------------------

INTERIM_CLEAN_DATA = INTERIM_DATA_DIR / "cleaned"
INTERIM_VALIDATION_DATA = INTERIM_DATA_DIR / "validation"


# -----------------------------------------------------------------------------
# Processed Data
# -----------------------------------------------------------------------------

PROCESSED_PLAYER_DATA = PROCESSED_DATA_DIR / "player_data"
PROCESSED_TEAM_DATA = PROCESSED_DATA_DIR / "team_data"
PROCESSED_COMPARISON_DATA = PROCESSED_DATA_DIR / "comparison_sets"


# -----------------------------------------------------------------------------
# Feature Data
# -----------------------------------------------------------------------------

FEATURE_PLAYER_METRICS = FEATURES_DIR / "player_metrics"
FEATURE_TEAM_METRICS = FEATURES_DIR / "team_metrics"
FEATURE_ROLE_CLASSIFICATIONS = FEATURES_DIR / "role_classifications"
FEATURE_PEER_GROUPS = FEATURES_DIR / "peer_groups"


# -----------------------------------------------------------------------------
# Evaluation Engine Outputs
# -----------------------------------------------------------------------------

OUTPUT_PLAYER_EVALUATIONS = OUTPUTS_DIR / "player_evaluations"
OUTPUT_ROLE_GRADES = OUTPUTS_DIR / "role_grades"
OUTPUT_COMPARISON_REPORTS = OUTPUTS_DIR / "comparison_reports"


# -----------------------------------------------------------------------------
# Dashboard Export Paths
# -----------------------------------------------------------------------------

DASHBOARD_DATA_EXPORTS = DASHBOARD_OUTPUTS_DIR / "data_exports"
DASHBOARD_PLAYER_TABLES = DASHBOARD_OUTPUTS_DIR / "player_tables"
DASHBOARD_TEAM_TABLES = DASHBOARD_OUTPUTS_DIR / "team_tables"
DASHBOARD_EVALUATION_TABLES = DASHBOARD_OUTPUTS_DIR / "evaluation_tables"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def ensure_directory(path: Path) -> None:
    """Create directory if missing to prevent downstream export failures."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_directories() -> None:
    """Ensure all defined output directories exist."""
    directories = [
        RAW_KENPOM,
        RAW_SPORTS_REFERENCE,
        RAW_TEAM_SITE,
        RAW_EXPORTS,
        INTERIM_CLEAN_DATA,
        INTERIM_VALIDATION_DATA,
        PROCESSED_PLAYER_DATA,
        PROCESSED_TEAM_DATA,
        PROCESSED_COMPARISON_DATA,
        FEATURE_PLAYER_METRICS,
        FEATURE_TEAM_METRICS,
        FEATURE_ROLE_CLASSIFICATIONS,
        FEATURE_PEER_GROUPS,
        OUTPUT_PLAYER_EVALUATIONS,
        OUTPUT_ROLE_GRADES,
        OUTPUT_COMPARISON_REPORTS,
        DASHBOARD_DATA_EXPORTS,
        DASHBOARD_PLAYER_TABLES,
        DASHBOARD_TEAM_TABLES,
        DASHBOARD_EVALUATION_TABLES,
    ]

    for directory in directories:
        ensure_directory(directory)


if __name__ == "__main__":
    ensure_project_directories()
    print("All project directories verified.")