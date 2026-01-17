"""
Logging utilities for federated learning experiments.

Implements the MANDATORY logging schema from configs/log_schema.yaml:
- Round-wise logs: CSV with experiment_id, seed, round, client_id, participation_flag,
                   train_loss, val_loss, rmse, mae, num_samples, timestamp
- Non-IID logs: CSV with client_id, num_samples, mean_rul, rul_variance,
                operating_condition_entropy, sensor_variance

Output paths per schema:
- experiments/{experiment_id}/logs/rounds.csv
- experiments/{experiment_id}/logs/non_iid.csv
"""
import os
import csv
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .paths import get_experiment_dir, get_logs_dir, ensure_dir


def generate_experiment_id() -> str:
    """Generate unique experiment ID per experiment_policy."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


class ExperimentLogger:
    """
    Logger implementing configs/log_schema.yaml EXACTLY.
    
    Outputs:
    - rounds.csv: Round-wise metrics per client
    - non_iid.csv: Non-IID quantification metrics per client
    - config.json: Experiment configuration (per experiment_policy)
    """
    
    # Mandatory fields per log_schema.yaml
    ROUND_FIELDS = [
        'experiment_id', 'seed', 'round', 'client_id', 'participation_flag',
        'train_loss', 'val_loss', 'rmse', 'mae', 'num_samples', 'timestamp'
    ]
    
    NON_IID_FIELDS = [
        'client_id', 'num_samples', 'mean_rul', 'rul_variance',
        'operating_condition_entropy', 'sensor_variance'
    ]
    
    def __init__(
        self,
        experiment_id: Optional[str] = None,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
        prevent_overwrite: bool = True
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_id: Unique ID (auto-generated if None)
            seed: Random seed for this experiment
            config: Experiment configuration dict to save
            prevent_overwrite: Raise error if experiment dir exists
        """
        self.experiment_id = experiment_id or generate_experiment_id()
        self.seed = seed
        self.config = config or {}
        
        # Setup directories
        self.experiment_dir = get_experiment_dir(self.experiment_id)
        self.logs_dir = get_logs_dir(self.experiment_id)
        
        if prevent_overwrite and os.path.exists(self.experiment_dir):
            raise FileExistsError(
                f"Experiment {self.experiment_id} already exists. "
                "Use a new experiment_id or set prevent_overwrite=False."
            )
        
        ensure_dir(self.logs_dir)
        
        # CSV file paths per schema
        self.rounds_csv_path = os.path.join(self.logs_dir, 'rounds.csv')
        self.non_iid_csv_path = os.path.join(self.logs_dir, 'non_iid.csv')
        self.config_path = os.path.join(self.experiment_dir, 'config.json')
        
        # Initialize CSV files with headers
        self._init_rounds_csv()
        self._init_non_iid_csv()
        
        # Save config
        self._save_config()
    
    def _init_rounds_csv(self):
        """Initialize rounds.csv with header."""
        with open(self.rounds_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.ROUND_FIELDS)
            writer.writeheader()
    
    def _init_non_iid_csv(self):
        """Initialize non_iid.csv with header."""
        with open(self.non_iid_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.NON_IID_FIELDS)
            writer.writeheader()
    
    def _save_config(self):
        """Save experiment config to JSON."""
        config_data = {
            'experiment_id': self.experiment_id,
            'seed': self.seed,
            'created_at': datetime.now().isoformat(),
            **self.config
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def log_round(
        self,
        round_num: int,
        client_id: int,
        participation_flag: bool,
        train_loss: float,
        val_loss: float,
        rmse: float,
        mae: float,
        num_samples: int
    ):
        """
        Log a single client's round metrics to rounds.csv.
        
        All fields are MANDATORY per log_schema.yaml.
        """
        row = {
            'experiment_id': self.experiment_id,
            'seed': self.seed,
            'round': round_num,
            'client_id': client_id,
            'participation_flag': int(participation_flag),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rmse': rmse,
            'mae': mae,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.rounds_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.ROUND_FIELDS)
            writer.writerow(row)
    
    def log_round_batch(self, round_num: int, client_metrics: List[Dict[str, Any]]):
        """
        Log multiple clients for a round.
        
        Each dict in client_metrics must have:
        client_id, participation_flag, train_loss, val_loss, rmse, mae, num_samples
        """
        with open(self.rounds_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.ROUND_FIELDS)
            for metrics in client_metrics:
                row = {
                    'experiment_id': self.experiment_id,
                    'seed': self.seed,
                    'round': round_num,
                    'client_id': metrics['client_id'],
                    'participation_flag': int(metrics['participation_flag']),
                    'train_loss': metrics['train_loss'],
                    'val_loss': metrics['val_loss'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'num_samples': metrics['num_samples'],
                    'timestamp': datetime.now().isoformat()
                }
                writer.writerow(row)
    
    def log_non_iid(
        self,
        client_id: int,
        num_samples: int,
        mean_rul: float,
        rul_variance: float,
        operating_condition_entropy: float,
        sensor_variance: float
    ):
        """
        Log non-IID quantification metrics for a client.
        
        All fields are MANDATORY per log_schema.yaml.
        """
        row = {
            'client_id': client_id,
            'num_samples': num_samples,
            'mean_rul': mean_rul,
            'rul_variance': rul_variance,
            'operating_condition_entropy': operating_condition_entropy,
            'sensor_variance': sensor_variance
        }
        
        with open(self.non_iid_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.NON_IID_FIELDS)
            writer.writerow(row)
    
    def log_non_iid_batch(self, client_stats: List[Dict[str, Any]]):
        """
        Log non-IID stats for multiple clients.
        
        Each dict must have: client_id, num_samples, mean_rul, rul_variance,
        operating_condition_entropy, sensor_variance
        """
        with open(self.non_iid_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.NON_IID_FIELDS)
            for stats in client_stats:
                writer.writerow({
                    'client_id': stats['client_id'],
                    'num_samples': stats['num_samples'],
                    'mean_rul': stats['mean_rul'],
                    'rul_variance': stats['rul_variance'],
                    'operating_condition_entropy': stats['operating_condition_entropy'],
                    'sensor_variance': stats['sensor_variance']
                })


def validate_log_files(experiment_id: str) -> Dict[str, Any]:
    """
    Validate that an experiment's logs conform to log_schema.yaml.
    
    Returns:
        Dict with 'valid' bool, 'errors' list, and 'warnings' list
    """
    logs_dir = get_logs_dir(experiment_id)
    errors = []
    warnings = []
    
    # Check rounds.csv exists and has required fields
    rounds_path = os.path.join(logs_dir, 'rounds.csv')
    if not os.path.exists(rounds_path):
        errors.append("Missing rounds.csv")
    else:
        with open(rounds_path, 'r') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = set(ExperimentLogger.ROUND_FIELDS) - set(header)
            if missing:
                errors.append(f"rounds.csv missing fields: {missing}")
    
    # Check non_iid.csv exists and has required fields
    non_iid_path = os.path.join(logs_dir, 'non_iid.csv')
    if not os.path.exists(non_iid_path):
        errors.append("Missing non_iid.csv")
    else:
        with open(non_iid_path, 'r') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = set(ExperimentLogger.NON_IID_FIELDS) - set(header)
            if missing:
                errors.append(f"non_iid.csv missing fields: {missing}")
    
    # Check config.json exists
    config_path = os.path.join(get_experiment_dir(experiment_id), 'config.json')
    if not os.path.exists(config_path):
        errors.append("Missing config.json")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
