"""
Create an example log demonstrating the mandatory schema from log_schema.yaml.

This script:
1. Creates experiment directory with unique ID
2. Writes rounds.csv with mandatory fields
3. Writes non_iid.csv with mandatory fields
4. Saves config.json per experiment_policy

Run from project root:
    python scripts/create_example_log.py
"""
import os
import sys
import random

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from utils.logging import ExperimentLogger, validate_log_files


def main():
    """Create an example log file demonstrating the schema."""
    
    # Create logger with example config
    logger = ExperimentLogger(
        experiment_id=None,  # Auto-generate
        seed=42,
        config={
            'algorithm': 'FedAvg',
            'dataset': 'FD001',
            'n_clients': 100,
            'clients_per_round': 10,
            'local_epochs': 1,
            'batch_size': 64,
            'learning_rate': 0.001,
            'max_rounds': 100,
            'window_length': 30,
            'rul_cap': 125,
        },
        prevent_overwrite=True
    )
    
    print(f"Created experiment: {logger.experiment_id}")
    print(f"Logs directory: {logger.logs_dir}")
    
    # Log non-IID stats for example clients (per requirements section 8.2)
    random.seed(42)
    non_iid_stats = []
    for client_id in range(1, 11):  # 10 example clients
        stats = {
            'client_id': client_id,
            'num_samples': random.randint(50, 200),
            'mean_rul': random.uniform(40, 80),
            'rul_variance': random.uniform(500, 1500),
            'operating_condition_entropy': random.uniform(0.5, 2.0),
            'sensor_variance': random.uniform(0.1, 1.0),
        }
        non_iid_stats.append(stats)
    
    logger.log_non_iid_batch(non_iid_stats)
    print(f"Logged non-IID stats for {len(non_iid_stats)} clients")
    
    # Log example rounds (per requirements section 8.1)
    for round_num in range(3):
        participating_clients = random.sample(range(1, 11), 5)
        
        round_metrics = []
        for client_id in range(1, 11):
            metrics = {
                'client_id': client_id,
                'participation_flag': client_id in participating_clients,
                'train_loss': random.uniform(0.1, 0.5) if client_id in participating_clients else 0.0,
                'val_loss': random.uniform(0.15, 0.6),
                'rmse': random.uniform(15, 30),
                'mae': random.uniform(10, 25),
                'num_samples': non_iid_stats[client_id - 1]['num_samples'],
            }
            round_metrics.append(metrics)
        
        logger.log_round_batch(round_num, round_metrics)
    
    print(f"Logged {3} rounds")
    
    # Validate the logs
    result = validate_log_files(logger.experiment_id)
    print(f"\nValidation: {'PASS' if result['valid'] else 'FAIL'}")
    if result['errors']:
        print("Errors:")
        for err in result['errors']:
            print(f"  - {err}")
    
    print(f"\nFiles created:")
    print(f"  - {logger.rounds_csv_path}")
    print(f"  - {logger.non_iid_csv_path}")
    print(f"  - {logger.config_path}")
    
    return logger.experiment_id


if __name__ == '__main__':
    main()
