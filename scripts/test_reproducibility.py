#!/usr/bin/env python
"""
Reproducibility Test Script (RX-2).

Verifies that a small FD001 experiment can be reproduced from raw data.
This script:
1. Runs a short FedAvg experiment (20 rounds, 5 clients)
2. Saves results
3. Runs the same experiment again with the same seed
4. Verifies results match

Usage:
    python test_reproducibility.py
    python test_reproducibility.py --verbose
"""
import argparse
import os
import sys
import json
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.tcn import create_tcn_model
from data.preprocessing import preprocess_dataset
from data.client_dataset import create_client_datasets, compute_all_clients_non_iid_stats
from utils.paths import get_experiment_dir, EXPERIMENTS_DIR, ensure_dir
from utils.logging import ExperimentLogger, generate_experiment_id
from server import FLServer, set_seed
from client import create_clients_from_datasets
from runner import FLRunner


def run_experiment(
    seed: int = 42,
    dataset: str = 'FD001',
    rounds: int = 20,
    max_clients: int = 5,
    experiment_id: str = None,
    verbose: bool = False
) -> dict:
    """
    Run a short federated experiment.
    
    Returns:
        Dict with final metrics and per-round history
    """
    set_seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: seed={seed}, rounds={rounds}")
        print(f"{'='*60}")
    
    # Preprocess data
    if verbose:
        print("Loading and preprocessing data...")
    result = preprocess_dataset(dataset)
    splits = result['splits']
    n_features = result['metadata']['n_features']
    
    # Create clients
    client_datasets = create_client_datasets(splits)
    
    # Limit clients for testing
    if max_clients:
        selected_ids = list(client_datasets.keys())[:max_clients]
        client_datasets = {k: client_datasets[k] for k in selected_ids}
    
    if verbose:
        print(f"Using {len(client_datasets)} clients")
    
    # Create experiment ID
    if experiment_id is None:
        experiment_id = f"repro_test_{generate_experiment_id()}"
    
    # Config
    config_data = {
        "algorithm": "fedavg",
        "dataset": dataset,
        "rounds": rounds,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.001,
        "participation": 1.0,
        "n_clients": len(client_datasets),
        "n_features": n_features,
        "seed": seed,
    }
    
    # Logger
    logger = ExperimentLogger(
        experiment_id=experiment_id,
        seed=seed,
        config=config_data,
        prevent_overwrite=False,
    )
    
    # Server
    server = FLServer(input_channels=n_features, device="cpu", seed=seed)
    server.init_model()
    
    # Clients
    clients = create_clients_from_datasets(
        client_datasets, device="cpu", learning_rate=0.001, batch_size=64, local_epochs=1
    )
    
    # Runner
    runner = FLRunner(server=server, clients=clients, logger=logger, device="cpu")
    
    # Log non-IID stats
    raw_df = result['raw_df']
    non_iid_stats = compute_all_clients_non_iid_stats(client_datasets, raw_df)
    for stats in non_iid_stats:
        logger.log_non_iid(**stats)
    
    # Training
    if verbose:
        print(f"Training for {rounds} rounds...")
    
    history = []
    for round_num in range(rounds):
        round_result = runner.run_round(round_num + 1, participation_fraction=1.0)
        history.append(round_result)
        
        if verbose and (round_num + 1) % 5 == 0:
            print(f"  Round {round_num + 1}: RMSE={round_result['global_rmse']:.4f}")
    
    # Final evaluation
    test_results = runner.evaluate_global_model()
    
    # Save summary.json manually (logger doesn't have finalize method)
    summary_path = os.path.join(logger.logs_dir, 'summary.json')
    summary_data = {
        'experiment_id': experiment_id,
        'seed': seed,
        'final_test_rmse': test_results['test_rmse'],
        'final_test_mae': test_results['test_mae'],
        'n_test_samples': test_results['n_test_samples'],
        'rounds': rounds,
        'n_clients': len(client_datasets)
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    results = {
        'experiment_id': experiment_id,
        'seed': seed,
        'final_rmse': test_results['test_rmse'],
        'final_mae': test_results['test_mae'],
        'history': [{'round': h['round'], 'global_rmse': h['global_rmse'], 'global_mae': h['global_mae']} for h in history],
        'experiment_dir': logger.experiment_dir
    }
    
    if verbose:
        print(f"\nFinal Test RMSE: {test_results['test_rmse']:.4f}")
        print(f"Final Test MAE: {test_results['test_mae']:.4f}")
        print(f"Saved to: {logger.experiment_dir}")
    
    return results


def verify_reproducibility(results1: dict, results2: dict, tolerance: float = 1e-6) -> dict:
    """
    Verify that two experiment runs produced identical results.
    
    Returns:
        Dict with verification status and details
    """
    checks = {}
    
    # Check final metrics
    rmse_diff = abs(results1['final_rmse'] - results2['final_rmse'])
    mae_diff = abs(results1['final_mae'] - results2['final_mae'])
    
    checks['final_rmse_match'] = rmse_diff < tolerance
    checks['final_mae_match'] = mae_diff < tolerance
    checks['rmse_diff'] = rmse_diff
    checks['mae_diff'] = mae_diff
    
    # Check per-round history
    history_match = True
    history_diffs = []
    
    for i, (h1, h2) in enumerate(zip(results1['history'], results2['history'])):
        round_rmse_diff = abs(h1['global_rmse'] - h2['global_rmse'])
        round_mae_diff = abs(h1['global_mae'] - h2['global_mae'])
        
        if round_rmse_diff >= tolerance or round_mae_diff >= tolerance:
            history_match = False
            history_diffs.append({
                'round': h1['round'],
                'rmse_diff': round_rmse_diff,
                'mae_diff': round_mae_diff
            })
    
    checks['history_match'] = history_match
    checks['history_diffs'] = history_diffs
    
    # Overall
    checks['reproducible'] = checks['final_rmse_match'] and checks['final_mae_match'] and checks['history_match']
    
    return checks


def main():
    parser = argparse.ArgumentParser(description='Reproducibility test for FL-CMAPPS')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--rounds', type=int, default=20, help='Number of rounds')
    parser.add_argument('--clients', type=int, default=5, help='Max clients')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--keep-experiments', action='store_true', 
                        help='Keep experiment directories after test')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REPRODUCIBILITY TEST (RX-2)")
    print("="*70)
    print(f"Seed: {args.seed}, Rounds: {args.rounds}, Clients: {args.clients}")
    print("-"*70)
    
    try:
        # Run first experiment
        print("\n[1/3] Running first experiment...")
        results1 = run_experiment(
            seed=args.seed,
            rounds=args.rounds,
            max_clients=args.clients,
            verbose=args.verbose
        )
        
        # Run second experiment with same seed
        print("\n[2/3] Running second experiment (same seed)...")
        results2 = run_experiment(
            seed=args.seed,
            rounds=args.rounds,
            max_clients=args.clients,
            verbose=args.verbose
        )
        
        # Verify reproducibility
        print("\n[3/3] Verifying reproducibility...")
        verification = verify_reproducibility(results1, results2)
        
        # Report results
        print("\n" + "="*70)
        print("REPRODUCIBILITY TEST RESULTS")
        print("="*70)
        
        print(f"\nRun 1 - Final RMSE: {results1['final_rmse']:.6f}, MAE: {results1['final_mae']:.6f}")
        print(f"Run 2 - Final RMSE: {results2['final_rmse']:.6f}, MAE: {results2['final_mae']:.6f}")
        
        print(f"\nMetric Differences:")
        print(f"  RMSE diff: {verification['rmse_diff']:.10f}")
        print(f"  MAE diff:  {verification['mae_diff']:.10f}")
        
        if verification['reproducible']:
            print("\n✓ REPRODUCIBILITY VERIFIED")
            print("  - Final metrics match exactly")
            print("  - Per-round history matches exactly")
            status = 0
        else:
            print("\n✗ REPRODUCIBILITY FAILED")
            if not verification['final_rmse_match']:
                print(f"  - Final RMSE mismatch: {verification['rmse_diff']}")
            if not verification['final_mae_match']:
                print(f"  - Final MAE mismatch: {verification['mae_diff']}")
            if not verification['history_match']:
                print(f"  - History mismatch in {len(verification['history_diffs'])} rounds")
            status = 1
        
        # Cleanup
        if not args.keep_experiments:
            print("\nCleaning up experiment directories...")
            for results in [results1, results2]:
                exp_dir = results['experiment_dir']
                if os.path.exists(exp_dir):
                    shutil.rmtree(exp_dir)
                    print(f"  Removed: {exp_dir}")
        
        print("\n" + "="*70)
        
        return status
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
