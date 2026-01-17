"""
Centralized path utilities.

Single source of truth for all project paths.
"""
import os

# Project root (one level up from src/utils/)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths
DATA_DIR = os.path.join(ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
CLIENTS_CACHE_DIR = os.path.join(PROCESSED_DATA_DIR, 'clients')

# Config paths
CONFIGS_DIR = os.path.join(ROOT, 'configs')

# Experiment paths
EXPERIMENTS_DIR = os.path.join(ROOT, 'experiments')

# Logs path template
LOGS_DIR_TEMPLATE = os.path.join(EXPERIMENTS_DIR, '{experiment_id}', 'logs')


def get_raw_data_path(filename: str) -> str:
    """Get path to raw data file."""
    return os.path.join(RAW_DATA_DIR, filename)


def get_config_path(config_name: str) -> str:
    """Get path to config file."""
    return os.path.join(CONFIGS_DIR, config_name)


def get_experiment_dir(experiment_id: str) -> str:
    """Get experiment directory path."""
    return os.path.join(EXPERIMENTS_DIR, experiment_id)


def get_logs_dir(experiment_id: str) -> str:
    """Get logs directory for an experiment."""
    return os.path.join(EXPERIMENTS_DIR, experiment_id, 'logs')


def get_client_cache_dir(dataset_name: str, client_id: int) -> str:
    """Get cache directory for a specific client."""
    return os.path.join(CLIENTS_CACHE_DIR, dataset_name, str(client_id))


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path
