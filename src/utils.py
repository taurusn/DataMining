"""
Utility Functions

Helper functions for logging, paths, and common operations.
"""

import os
import logging
from pathlib import Path
import json


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('project.log'),
            logging.StreamHandler()
        ]
    )


def ensure_dir(directory):
    """Ensure directory exists, create if not."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent