"""
File Name: config.py
Last Modified: 2026-02-20

Overview:
Central configuration module for the NCAA Next Man Up project.
Defines project paths and core analytical settings so all scripts
reference a single configuration source.

Inputs:
- Local project directory structure
- Optional environment variable override for project root

Outputs:
- Shared path constants
- Project configuration constants
- Validation helpers
"""

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Project Root
# -----------------------------------------------------------------------------

PROJECT_ROOT_ENV_VAR = "NCAA_NEXT_MAN_UP_ROOT"


def _resolve_project_root() -> Path:
    """Resolve project root. Allows environment override for portability."""
    env_root = os.getenv(PROJECT_ROOT_ENV_VAR)

    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise NotADirectoryError(
                f"{PROJECT_ROOT_ENV_VAR} points to an invalid directory: {root}"
            )
        return root

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _resolve_project_root()


# -----------------------------------------------------------------------------
# Data Directories
# -----------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_KENPOM_DIR = RAW_DATA_DIR / "kenpom"
RAW_SPORTS_REFERENCE_DIR = RAW_DATA_DIR / "sports_reference"
RAW_TEAM_SITE_DIR = RAW_DATA_DIR / "team_site"
RAW_EXPORTS_DIR = RAW_DATA_DIR / "exports"


# -----------------------------------------------------------------------------
# Script Directories
# -----------------------------------------------------------------------------

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DATA_COLLECTION_DIR = SCRIPTS_DIR / "data_collection"
DATA_CLEANING_DIR = SCRIPTS_DIR / "data_cleaning"
FEATURE_ENGINEERING_DIR = SCRIPTS_DIR / "feature_engineering"
EVALUATION_ENGINE_DIR = SCRIPTS_DIR / "evaluation_engine"
UTILITIES_DIR = SCRIPTS_DIR / "utilities"


# -----------------------------------------------------------------------------
# Other Project Directories
# -----------------------------------------------------------------------------

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"


# -----------------------------------------------------------------------------
# Feature / Output Paths
# -----------------------------------------------------------------------------

FEATURES_DIR = PROCESSED_DATA_DIR / "features"
EVALUATION_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation_engine"
DASHBOARD_OUTPUTS_DIR = OUTPUTS_DIR / "dashboard"


# -----------------------------------------------------------------------------
# Season & Team Configuration
# -----------------------------------------------------------------------------

CURRENT_SEASON = "2025-2026"

TARGET_CONFERENCE = "Big Sky"

TARGET_TEAM_NAME = "Montana"
TARGET_TEAM_SHORT_NAME = "UM"
TARGET_TEAM_MASCOT = "Grizzlies"

MONTANA_TEAM_IDENTIFIER = "Montana"


# -----------------------------------------------------------------------------
# Evaluation Engine Configuration
# -----------------------------------------------------------------------------

POSITION_GROUPS = ("Guard", "Wing", "Big")

USAGE_WINDOW_PERCENT = 5.0

# Optional guard against extremely small samples
MINIMUM_MINUTES_THRESHOLD = None


# -----------------------------------------------------------------------------
# Role Archetypes
# -----------------------------------------------------------------------------

ROLE_ARCHETYPES = {
    "Guard": (
        "Primary Creator",
        "Secondary Playmaker",
        "Off-Ball Shooter",
    ),
    "Wing": (
        "Scoring Wing",
        "3-and-D Wing",
        "Glue Wing",
    ),
    "Big": (
        "Interior Big",
        "Stretch Big",
        "Rebounding Big",
        "Defensive Big",
    ),
}


# -----------------------------------------------------------------------------
# Evaluation Categories
# -----------------------------------------------------------------------------

EVALUATION_CATEGORIES = (
    "Scoring Efficiency",
    "Shooting",
    "Playmaking",
    "Ball Security",
    "Rebounding",
    "Defensive Activity",
    "Free Throw Pressure / Foul Drawing",
    "Winning Impact",
    "Role Effectiveness",
    "Competition Stability",
)


# -----------------------------------------------------------------------------
# Validation Utilities
# -----------------------------------------------------------------------------

EXPECTED_PROJECT_DIRECTORIES = (
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    SCRIPTS_DIR,
    NOTEBOOKS_DIR,
    DASHBOARD_DIR,
    OUTPUTS_DIR,
    DOCS_DIR,
    TESTS_DIR,
)


def ensure_directories_exist(paths, create_missing=False):
    """Validate required directories to prevent downstream path failures."""
    missing = []

    for path in paths:
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"Expected directory but found file: {path}")

        if not path.exists():
            if create_missing:
                path.mkdir(parents=True, exist_ok=True)
            else:
                missing.append(path)

    if missing:
        missing_list = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing required project directories:\n{missing_list}"
        )


def get_usage_window(player_usage: float):
    """Return usage comparison bounds."""
    if player_usage is None:
        raise ValueError("player_usage cannot be None")

    if not isinstance(player_usage, (int, float)):
        raise TypeError("player_usage must be numeric")

    return (
        player_usage - USAGE_WINDOW_PERCENT,
        player_usage + USAGE_WINDOW_PERCENT,
    )


if __name__ == "__main__":
    ensure_directories_exist(EXPECTED_PROJECT_DIRECTORIES, create_missing=False)

    print("Configuration loaded successfully.")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Season: {CURRENT_SEASON}")
    print(f"Conference: {TARGET_CONFERENCE}")
    print(f"Usage Window: ±{USAGE_WINDOW_PERCENT}%")