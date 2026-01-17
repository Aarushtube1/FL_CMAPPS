#!/usr/bin/env python
"""
Local-Only Baseline Training (C-2).

Trains separate models for each client (engine) without federation.
This serves as a lower-bound baseline showing performance without collaboration.

Usage:
    python run_local.py --config configs/local.yaml
    python run_local.py --dataset FD001 --epochs 100 --seed 42
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.tcn import create_tcn_model, count_parameters
from data.preprocessing import preprocess_dataset
from data.client_dataset import create_client_datasets, RULDataset
from utils.paths import get_experiment_dir, ensure_dir
from utils.logging import generate_experiment_id
from server import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Local-only baseline training',
        epilog='Recommended: python run_local.py --config configs/local.yaml'
    )
    parser.add_argument('--config', type=str, help='Path to config YAML file (recommended)')
    parser.add_argument('--strict', action='store_true',
                        help='Strict mode: require config file, disallow CLI overrides')
    parser.add_argument('--dataset', type=str, default='FD001',
                        choices=['FD001', 'FD002', 'FD003', 'FD004'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--experiment-id', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--max-clients', type=int, default=None,
                        help='Max clients to train (for quick testing)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_single_client(
    client_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_features: int,
    epochs: int,
    lr: float,
    device: str,
    verbose: bool = False
) -> Dict:
    """
    Train a model for a single client.
    
    Returns dict with final metrics.
    """
    # Create fresh model for this client
    model = create_tcn_model(n_features)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_y)
            train_samples += len(batch_y)
        
        train_loss /= train_samples
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                preds = model(batch_X)
                val_preds.extend(preds.squeeze().cpu().numpy())
                val_targets.extend(batch_y.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict().copy()
    
    # Load best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            test_preds.extend(preds.squeeze().cpu().numpy())
            test_targets.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_rmse = np.sqrt(np.mean((test_preds - test_targets) ** 2))
    test_mae = np.mean(np.abs(test_preds - test_targets))
    
    return {
        'client_id': client_id,
        'train_samples': train_samples,
        'val_samples': len(val_targets),
        'test_samples': len(test_targets),
        'best_val_rmse': float(best_val_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae)
    }


def main():
    args = parse_args()
    
    # Strict mode: require config file
    if args.strict and not args.config:
        print("ERROR: --strict mode requires --config <path>")
        print("Usage: python run_local.py --config configs/local.yaml --strict")
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
    experiment_id = args.experiment_id or f"local_{args.dataset}_{generate_experiment_id()}"
    experiment_dir = get_experiment_dir(experiment_id)
    ensure_dir(experiment_dir)
    logs_dir = os.path.join(experiment_dir, 'logs')
    ensure_dir(logs_dir)
    
    print(f"=" * 60)
    print(f"Local-Only Baseline Training")
    print(f"=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs per client: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"=" * 60)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    result = preprocess_dataset(args.dataset)
    clients = create_client_datasets(result['splits'])
    n_features = result['metadata']['n_features']
    
    # Limit clients if requested
    if args.max_clients:
        client_ids = list(clients.keys())[:args.max_clients]
        clients = {k: clients[k] for k in client_ids}
    
    print(f"Number of clients: {len(clients)}")
    print(f"Features: {n_features}")
    
    # Save config
    config_data = {
        'experiment_id': experiment_id,
        'algorithm': 'local',
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'n_clients': len(clients),
        'n_features': n_features,
        'created_at': datetime.now().isoformat()
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Train each client
    print("\nTraining clients...")
    all_results = []
    total_start = time.time()
    
    for i, (client_id, client_data) in enumerate(clients.items()):
        # Create loaders
        train_loader = client_data.get_train_loader(batch_size=args.batch_size, shuffle=True)
        val_loader = client_data.get_val_loader(batch_size=args.batch_size)
        test_loader = client_data.get_test_loader(batch_size=args.batch_size)
        
        if val_loader is None or test_loader is None:
            print(f"  Skipping client {client_id} (missing val/test data)")
            continue
        
        client_start = time.time()
        
        result = train_single_client(
            client_id=client_id,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            n_features=n_features,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            verbose=args.verbose
        )
        
        result['train_time'] = time.time() - client_start
        all_results.append(result)
        
        if args.verbose and (i + 1) % 10 == 0:
            print(f"  Client {i + 1}/{len(clients)}: "
                  f"RMSE={result['test_rmse']:.2f}, MAE={result['test_mae']:.2f}")
    
    total_time = time.time() - total_start
    
    # Aggregate results
    total_test_samples = sum(r['test_samples'] for r in all_results)
    weighted_rmse = sum(r['test_rmse'] * r['test_samples'] for r in all_results) / total_test_samples
    weighted_mae = sum(r['test_mae'] * r['test_samples'] for r in all_results) / total_test_samples
    
    # Also compute unweighted (per-client average)
    avg_rmse = np.mean([r['test_rmse'] for r in all_results])
    avg_mae = np.mean([r['test_mae'] for r in all_results])
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS (Aggregated over {len(all_results)} clients)")
    print(f"{'=' * 60}")
    print(f"Weighted Test RMSE: {weighted_rmse:.2f}")
    print(f"Weighted Test MAE: {weighted_mae:.2f}")
    print(f"Average Test RMSE: {avg_rmse:.2f}")
    print(f"Average Test MAE: {avg_mae:.2f}")
    print(f"Total training time: {total_time:.1f}s")
    
    # Save per-client CSV
    csv_path = os.path.join(logs_dir, 'clients.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['client_id', 'train_samples', 'val_samples', 'test_samples',
                      'best_val_rmse', 'test_rmse', 'test_mae', 'train_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    # Save summary
    summary = {
        'experiment_id': experiment_id,
        'algorithm': 'local',
        'dataset': args.dataset,
        'n_clients': len(all_results),
        'total_test_samples': total_test_samples,
        'weighted_test_rmse': weighted_rmse,
        'weighted_test_mae': weighted_mae,
        'avg_test_rmse': avg_rmse,
        'avg_test_mae': avg_mae,
        'total_time': total_time
    }
    with open(os.path.join(logs_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLogs saved to: {logs_dir}")
    
    return summary


if __name__ == '__main__':
    main()
