#!/usr/bin/env python
"""
Federated Learning Training Script (C-3).

Runs federated learning with configurable algorithms:
- FedAvg (default)
- FedProx (mu parameter)
- SCAFFOLD (control variates)
- FedDC (daisy-chain corrections)

Usage:
    python run_federated.py --config configs/federated.yaml
    python run_federated.py --dataset FD001 --algorithm fedavg --rounds 100 --seed 42
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.tcn import create_tcn_model, count_parameters
from data.preprocessing import preprocess_dataset
from data.client_dataset import (
    create_client_datasets, 
    compute_all_clients_non_iid_stats
)
from utils.paths import get_experiment_dir, ensure_dir
from utils.logging import ExperimentLogger, generate_experiment_id
from server import FLServer, set_seed
from client import FLClient, create_clients_from_datasets
from runner import FLRunner, create_fl_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='Federated learning training',
        epilog='Recommended: python run_federated.py --config configs/federated.yaml'
    )
    parser.add_argument('--config', type=str, help='Path to config YAML file (recommended)')
    parser.add_argument('--strict', action='store_true',
                        help='Strict mode: require config file, disallow CLI overrides')
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'])
    parser.add_argument('--algorithm', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox', 'scaffold', 'feddc'])
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--participation', type=float, default=1.0,
                        help='Client participation fraction (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--experiment-id', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--max-clients', type=int, default=None,
                        help='Max clients to use (for quick testing)')
    
    # Algorithm-specific parameters
    parser.add_argument('--mu', type=float, default=0.01,
                        help='FedProx proximal term coefficient')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='FedDC drift update coefficient')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Strict mode: require config file
    if args.strict and not args.config:
        print("ERROR: --strict mode requires --config <path>")
        print("Usage: python run_federated.py --config configs/federated.yaml --strict")
        sys.exit(1)
    
    # Load config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        for key, value in config.items():
            key_attr = key.replace('-', '_')
            if hasattr(args, key_attr) and value is not None:
                setattr(args, key_attr, value)
        if args.strict:
            print(f"[STRICT MODE] All parameters loaded from: {args.config}")
    
    # Set seed
    set_seed(args.seed)
    
    # Generate experiment ID
    experiment_id = args.experiment_id or f"{args.algorithm}_{args.dataset}_{generate_experiment_id()}"
    
    print(f"=" * 60)
    print(f"Federated Learning: {args.algorithm.upper()}")
    print(f"=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Participation: {args.participation * 100:.0f}%")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    if args.algorithm == 'fedprox':
        print(f"FedProx mu: {args.mu}")
    print(f"=" * 60)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    result = preprocess_dataset(args.dataset)
    client_datasets = create_client_datasets(result['splits'])
    n_features = result['metadata']['n_features']
    
    # Limit clients if requested
    if args.max_clients:
        client_ids = list(client_datasets.keys())[:args.max_clients]
        client_datasets = {k: client_datasets[k] for k in client_ids}
    
    print(f"Number of clients: {len(client_datasets)}")
    print(f"Features: {n_features}")
    
    # Compute non-IID statistics
    print("Computing non-IID statistics...")
    raw_df = result['raw_df']
    non_iid_stats = compute_all_clients_non_iid_stats(client_datasets, raw_df)
    
    # Create experiment logger
    config_data = {
        'algorithm': args.algorithm,
        'dataset': args.dataset,
        'rounds': args.rounds,
        'local_epochs': args.local_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'participation': args.participation,
        'n_clients': len(client_datasets),
        'n_features': n_features,
        'mu': args.mu if args.algorithm == 'fedprox' else None
    }
    
    logger = ExperimentLogger(
        experiment_id=experiment_id,
        seed=args.seed,
        config=config_data,
        prevent_overwrite=False
    )
    
    # Log non-IID stats
    for stats in non_iid_stats:
        logger.log_non_iid(
            client_id=stats['client_id'],
            num_samples=stats['num_samples'],
            mean_rul=stats['mean_rul'],
            rul_variance=stats['rul_variance'],
            operating_condition_entropy=stats['operating_condition_entropy'],
            sensor_variance=stats['sensor_variance']
        )
    
    # Create FL components
    print("\nInitializing FL components...")
    
    # Create server
    server = FLServer(
        input_channels=n_features,
        device=args.device,
        seed=args.seed
    )
    server.init_model()
    
    # Create clients based on algorithm
    if args.algorithm == 'fedprox':
        # FedProx clients need mu parameter
        from client import FLClientFedProx
        clients = {}
        for client_id, dataset in client_datasets.items():
            clients[client_id] = FLClientFedProx(
                client_id=client_id,
                dataset=dataset,
                device=args.device,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                mu=args.mu
            )
    elif args.algorithm == 'scaffold':
        # SCAFFOLD clients need control variates
        from client import FLClientSCAFFOLD
        clients = {}
        for client_id, dataset in client_datasets.items():
            clients[client_id] = FLClientSCAFFOLD(
                client_id=client_id,
                dataset=dataset,
                device=args.device,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs
            )
        # Initialize server control variate
        server.init_scaffold_control(n_features)
    elif args.algorithm == 'feddc':
        # FedDC clients need drift variables
        from client import FLClientFedDC
        clients = {}
        for client_id, dataset in client_datasets.items():
            clients[client_id] = FLClientFedDC(
                client_id=client_id,
                dataset=dataset,
                device=args.device,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                alpha=getattr(args, 'alpha', 0.1)  # FedDC drift coefficient
            )
        # Initialize server drift tracking
        server.init_feddc_drift(n_features)
    else:
        # Standard FedAvg clients
        clients = create_clients_from_datasets(
            client_datasets,
            device=args.device,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs
        )
    
    # Create runner
    runner = FLRunner(
        server=server,
        clients=clients,
        logger=logger,
        device=args.device
    )
    
    # Run training
    print(f"\nStarting {args.algorithm.upper()} training...")
    start_time = time.time()
    
    round_metrics = runner.run_training(
        max_rounds=args.rounds,
        participation_fraction=args.participation,
        algorithm=args.algorithm,
        verbose=args.verbose
    )
    
    total_time = time.time() - start_time
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = runner.evaluate_global_model()
    
    # Results
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Final Test RMSE: {test_metrics['test_rmse']:.2f}")
    print(f"Final Test MAE: {test_metrics['test_mae']:.2f}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Average round time: {np.mean(runner.round_times):.2f}s")
    
    # Save final model
    experiment_dir = get_experiment_dir(experiment_id)
    torch.save(server.global_model.state_dict(), 
               os.path.join(experiment_dir, 'final_model.pt'))
    
    # Save summary
    summary = {
        'experiment_id': experiment_id,
        'algorithm': args.algorithm,
        'dataset': args.dataset,
        'test_rmse': test_metrics['test_rmse'],
        'test_mae': test_metrics['test_mae'],
        'total_time': total_time,
        'avg_round_time': float(np.mean(runner.round_times)),
        'final_round_rmse': round_metrics[-1]['global_rmse'] if round_metrics else 0.0
    }
    
    logs_dir = os.path.join(experiment_dir, 'logs')
    with open(os.path.join(logs_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLogs saved to: {logs_dir}")
    print(f"Model saved to: {experiment_dir}/final_model.pt")
    
    return summary


if __name__ == '__main__':
    main()
