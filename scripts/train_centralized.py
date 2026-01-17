#!/usr/bin/env python
"""
Centralized Baseline Training (C-1).

Trains a single TCN model on pooled data from all clients.
This serves as the upper-bound baseline for federated learning performance.

Usage:
    python train_centralized.py --config configs/centralized.yaml
    python train_centralized.py --dataset FD001 --epochs 100 --seed 42
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.tcn import create_tcn_model, count_parameters
from data.preprocessing import preprocess_dataset
from data.client_dataset import RULDataset
from utils.paths import get_experiment_dir, ensure_dir
from utils.logging import generate_experiment_id
from server import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Centralized baseline training',
        epilog='Recommended: python train_centralized.py --config configs/centralized.yaml'
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
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_centralized_dataloaders(
    dataset: str,
    batch_size: int = 64
) -> tuple:
    """
    Create centralized dataloaders by pooling all client data.
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    # Preprocess dataset
    result = preprocess_dataset(dataset)
    splits = result['splits']
    metadata = result['metadata']
    
    # Pool all data (already combined in splits)
    train_X, train_y, _ = splits['train']
    val_X, val_y, _ = splits['val']
    test_X, test_y, _ = splits['test']
    
    # Create datasets
    train_dataset = RULDataset(train_X, train_y)
    val_dataset = RULDataset(val_X, val_y)
    test_dataset = RULDataset(test_X, test_y)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, metadata


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        total_samples += len(batch_y)
    
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """Evaluate model, return loss, RMSE, MAE."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            
            total_loss += loss.item() * len(batch_y)
            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    n_samples = len(all_targets)
    avg_loss = total_loss / n_samples
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return {
        'loss': avg_loss,
        'rmse': float(rmse),
        'mae': float(mae),
        'n_samples': n_samples
    }


def main():
    args = parse_args()
    
    # Strict mode: require config file
    if args.strict and not args.config:
        print("ERROR: --strict mode requires --config <path>")
        print("Usage: python train_centralized.py --config configs/centralized.yaml --strict")
        sys.exit(1)
    
    # Load config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            sys.exit(1)
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            key_attr = key.replace('-', '_')
            if hasattr(args, key_attr) and value is not None:
                setattr(args, key_attr, value)
        if args.strict:
            print(f"[STRICT MODE] All parameters loaded from: {args.config}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Generate experiment ID
    experiment_id = args.experiment_id or f"centralized_{args.dataset}_{generate_experiment_id()}"
    experiment_dir = get_experiment_dir(experiment_id)
    ensure_dir(experiment_dir)
    logs_dir = os.path.join(experiment_dir, 'logs')
    ensure_dir(logs_dir)
    
    print(f"=" * 60)
    print(f"Centralized Baseline Training")
    print(f"=" * 60)
    print(f"Experiment ID: {experiment_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"=" * 60)
    
    # Create data loaders
    print("\nLoading and preprocessing data...")
    train_loader, val_loader, test_loader, metadata = create_centralized_dataloaders(
        args.dataset, args.batch_size
    )
    print(f"Train samples: {metadata['n_train']}")
    print(f"Val samples: {metadata['n_val']}")
    print(f"Test samples: {metadata['n_test']}")
    
    # Create model
    n_features = metadata['n_features']
    model = create_tcn_model(n_features)
    model.to(args.device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'epoch_time': []
    }
    
    # Save config
    config_data = {
        'experiment_id': experiment_id,
        'algorithm': 'centralized',
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'device': args.device,
        'n_features': n_features,
        'n_train': metadata['n_train'],
        'n_val': metadata['n_val'],
        'n_test': metadata['n_test'],
        'created_at': datetime.now().isoformat()
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Training loop
    print("\nStarting training...")
    best_val_rmse = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['epoch_time'].append(epoch_time)
        
        # Track best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(experiment_dir, 'best_model.pt'))
        
        if args.verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val RMSE={val_metrics['rmse']:.2f}, "
                  f"Val MAE={val_metrics['mae']:.2f}, "
                  f"Time={epoch_time:.2f}s")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, criterion, args.device)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Best validation RMSE: {best_val_rmse:.2f} (epoch {best_epoch + 1})")
    print(f"Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"Test MAE: {test_metrics['mae']:.2f}")
    print(f"Total training time: {sum(history['epoch_time']):.1f}s")
    
    # Save history
    history['test_rmse'] = test_metrics['rmse']
    history['test_mae'] = test_metrics['mae']
    history['best_epoch'] = best_epoch
    history['best_val_rmse'] = best_val_rmse
    
    with open(os.path.join(logs_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save epoch-wise CSV log
    import csv
    with open(os.path.join(logs_dir, 'epochs.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_rmse', 'val_mae', 'epoch_time'
        ])
        writer.writeheader()
        for i in range(args.epochs):
            writer.writerow({
                'epoch': i + 1,
                'train_loss': history['train_loss'][i],
                'val_loss': history['val_loss'][i],
                'val_rmse': history['val_rmse'][i],
                'val_mae': history['val_mae'][i],
                'epoch_time': history['epoch_time'][i]
            })
    
    print(f"\nLogs saved to: {logs_dir}")
    print(f"Model saved to: {experiment_dir}/best_model.pt")
    
    return test_metrics


if __name__ == '__main__':
    main()
