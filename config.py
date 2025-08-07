"""
Global configuration module for dataset paths.

This module loads environment variables from .env file and provides
easy-to-import constants for dataset paths across the repository.
Works from any subdirectory and with all Python environments.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the repository root directory (where this config.py file is located)
# Handle both script and interactive environments
try:
    # Works in Python scripts
    REPO_ROOT = Path(__file__).parent.absolute()
except NameError:
    # Works in Jupyter/Quarto - find repo root by looking for .env file
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / '.env').exists() or (current_path / 'config.py').exists():
            REPO_ROOT = current_path
            break
        current_path = current_path.parent
    else:
        # Fallback to current working directory
        REPO_ROOT = Path.cwd()

# Load environment variables from .env file
dotenv_path = REPO_ROOT / '.env'
load_dotenv(dotenv_path)

# Dataset path configuration
DATA_ROOT = os.getenv('DATA_ROOT')
if not DATA_ROOT:
    raise ValueError(
        "DATA_ROOT not found in environment variables. "
        "Please create a .env file in the repository root with your dataset paths."
    )

DATA_ROOT = Path(DATA_ROOT)

# Chapter-specific dataset paths
MOVIELENS_PATH = Path(os.getenv('MOVIELENS_PATH', DATA_ROOT / 'ch02' / 'ml-1m'))
CLICK_RATE_PATH = Path(os.getenv('CLICK_RATE_PATH', DATA_ROOT / 'ch03'))

# Convenience functions for common dataset files
def get_movielens_file(filename):
    """Get full path to a MovieLens dataset file."""
    return MOVIELENS_PATH / filename

def get_click_rate_file(filename):
    """Get full path to a click rate dataset file."""
    return CLICK_RATE_PATH / filename

def get_data_path(chapter, filename):
    """Get full path to any dataset file by chapter and filename."""
    return DATA_ROOT / f'ch{chapter:02d}' / filename

# Verify paths exist (optional - can be disabled for testing)
def verify_paths():
    """Verify that configured dataset paths exist."""
    if not DATA_ROOT.exists():
        print(f"Warning: DATA_ROOT does not exist: {DATA_ROOT}")
    if not MOVIELENS_PATH.exists():
        print(f"Warning: MOVIELENS_PATH does not exist: {MOVIELENS_PATH}")
    if not CLICK_RATE_PATH.exists():
        print(f"Warning: CLICK_RATE_PATH does not exist: {CLICK_RATE_PATH}")

# Uncomment the line below to verify paths on import (optional)
# verify_paths()