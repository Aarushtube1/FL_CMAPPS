#!/usr/bin/env python
"""
Phase 6B: FedDC Validation & Comparison.

This script implements Phase 6B - Conditional Algorithm Extension (FedDC).
Tests FedDC on FD001, FD002, FD004 with 5 clients and compares against SCAFFOLD.

Per tasklist:
- FedDC must clearly outperform SCAFFOLD on FD002/FD004
- Improvement exceeds noise level
- No regression on FD001
- Overhead justified and documented

Decision:
- If FedDC wins → include in Phase 6C
- If not → document and drop
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.tcn import create_tcn_model, count_parameters
from data.preprocessing import preprocess_dataset
from data.client_dataset import create_client_datasets, compute_all_clients_non_iid_stats
from utils.paths import get_experiment_dir, ensure_dir, EXPERIMENTS_DIR
from utils.logging import ExperimentLogger, generate_experiment_id
from server import FLServer, set_seed
from client import FLClient, FLClientFedDC, FLClientSCAFFOLD, create_clients_from_datasets
from runner import FLRunner


# Phase 6B Configuration
DATASETS = ['FD001', 'FD002', 'FD004']  # FD003 excluded per tasklist
ALGORITHMS = ['scaffold', 'feddc']  # Compare SCAFFOLD vs FedDC
N_CLIENTS = 5
DEFAULT_ROUNDS = 50
DEFAULT_SEEDS = [42]

# Thresholds for validation
IMPROVEMENT_THRESHOLD = 0.02  # FedDC must improve by 2% RMSE to be significant
REGRESSION_THRESHOLD = 0.05  # FedDC must not regress by more than 5%


def run_single_experiment(
    dataset: str,
    algorithm: str,
    seed: int,
    rounds: int = DEFAULT_ROUNDS,
    n_clients: int = N_CLIENTS,
    alpha: float = 0.1,
    experiment_prefix: str = "validation_feddc",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single FL experiment and return results.
    """
    set_seed(seed)
    
    exp_id = f"{experiment_prefix}/{dataset}_{algorithm}_{seed}"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {exp_id}")
        print(f"{'='*60}")
    
    # Check if experiment already completed
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    summary_path = os.path.join(exp_dir, 'logs', 'summary.json')
    if os.path.exists(summary_path):
        if verbose:
            print(f"✓ Experiment already completed, loading results...")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        return {
            'success': True,
            'exp_id': exp_id,
            'summary': summary,
            'final_rmse': summary.get('final_rmse'),
            'test_rmse': summary.get('test_rmse'),
            'total_time': summary.get('total_time'),
            'skipped': True
        }
    
    try:
        # Load and preprocess data
        result = preprocess_dataset(dataset)
        client_datasets = create_client_datasets(result['splits'])
        n_features = result['metadata']['n_features']
        
        # Limit to N_CLIENTS
        client_ids = sorted(list(client_datasets.keys()))[:n_clients]
        client_datasets = {k: client_datasets[k] for k in client_ids}
        
        if verbose:
            print(f"Clients: {len(client_datasets)}, Features: {n_features}")
        
        # Config
        config_data = {
            'algorithm': algorithm,
            'dataset': dataset,
            'rounds': rounds,
            'local_epochs': 1,
            'batch_size': 64,
            'lr': 1e-3,
            'participation': 1.0,
            'n_clients': len(client_datasets),
            'n_features': n_features,
            'alpha': alpha if algorithm == 'feddc' else None,
            'phase': '6B',
            'experiment_type': experiment_prefix
        }
        
        # Create experiment directory and logger
        exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
        ensure_dir(exp_dir)
        
        logger = ExperimentLogger(
            experiment_id=exp_id,
            seed=seed,
            config=config_data,
            prevent_overwrite=False
        )
        
        # Create server
        server = FLServer(
            input_channels=n_features,
            device='cpu',
            seed=seed
        )
        server.init_model()
        
        # Create clients based on algorithm
        if algorithm == 'feddc':
            clients = {}
            for client_id, ds in client_datasets.items():
                clients[client_id] = FLClientFedDC(
                    client_id=client_id,
                    dataset=ds,
                    device='cpu',
                    learning_rate=1e-3,
                    batch_size=64,
                    local_epochs=1,
                    alpha=alpha
                )
            server.init_feddc_drift(n_features)
        else:  # scaffold
            clients = {}
            for client_id, ds in client_datasets.items():
                clients[client_id] = FLClientSCAFFOLD(
                    client_id=client_id,
                    dataset=ds,
                    device='cpu',
                    learning_rate=1e-3,
                    batch_size=64,
                    local_epochs=1
                )
            server.init_scaffold_control(n_features)
        
        # Create runner
        runner = FLRunner(
            server=server,
            clients=clients,
            logger=logger,
            device='cpu'
        )
        
        # Run training
        start_time = time.time()
        
        print(f"Starting FL training: {rounds} rounds, 100% participation, {algorithm}")
        
        round_rmse_history = []
        for round_num in range(rounds):
            round_metrics = runner.run_round(round_num, participation_fraction=1.0, algorithm=algorithm)
            round_rmse_history.append(round_metrics['global_rmse'])
            
            if verbose and (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{rounds}: RMSE={round_metrics['global_rmse']:.2f}, "
                      f"MAE={round_metrics['global_mae']:.2f}, Time={round_metrics['round_time']:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Training complete. Total time: {total_time:.1f}s")
        
        # Final evaluation on test set
        test_metrics = runner.evaluate_global_model()
        
        # Save final model
        model_path = os.path.join(exp_dir, 'final_model.pt')
        torch.save(server.global_model.state_dict(), model_path)
        
        # Calculate metrics
        initial_rmse = round_rmse_history[0]
        final_rmse = round_rmse_history[-1]
        best_rmse = min(round_rmse_history)
        best_round = round_rmse_history.index(best_rmse) + 1
        improvement = (initial_rmse - final_rmse) / initial_rmse if initial_rmse > 0 else 0
        
        # Create summary
        summary = {
            'experiment_id': exp_id,
            'dataset': dataset,
            'algorithm': algorithm,
            'seed': seed,
            'rounds': rounds,
            'n_clients': len(client_datasets),
            'participation': 1.0,
            'alpha': alpha if algorithm == 'feddc' else None,
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'best_rmse': best_rmse,
            'best_round': best_round,
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'total_time': total_time,
            'avg_round_time': total_time / rounds,
            'improvement': improvement,
            'round_rmse_history': round_rmse_history
        }
        
        # Save summary JSON
        summary_path = os.path.join(logger.logs_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"\n✓ COMPLETED")
            print(f"Initial RMSE: {initial_rmse:.2f} → Final: {final_rmse:.2f} (Best: {best_rmse:.2f} @ round {best_round})")
            print(f"Test RMSE: {test_metrics['test_rmse']:.2f}, MAE: {test_metrics['test_mae']:.2f}")
            print(f"Time: {total_time:.1f}s")
        
        return {
            'success': True,
            'exp_id': exp_id,
            'summary': summary,
            'final_rmse': final_rmse,
            'test_rmse': test_metrics['test_rmse'],
            'total_time': total_time,
            'skipped': False
        }
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'dataset': dataset,
            'algorithm': algorithm,
            'seed': seed
        }


def run_phase6b_validation(
    rounds: int = DEFAULT_ROUNDS,
    seeds: List[int] = DEFAULT_SEEDS,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Phase 6B: FedDC vs SCAFFOLD comparison.
    """
    print("\n" + "="*70)
    print("PHASE 6B: FedDC vs SCAFFOLD COMPARISON")
    print("="*70)
    print(f"Datasets: {DATASETS}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Clients: {N_CLIENTS}")
    print(f"Rounds: {rounds}")
    print(f"Seeds: {seeds}")
    print("="*70)
    
    results = {}
    all_summaries = []
    
    for dataset in DATASETS:
        results[dataset] = {}
        for algorithm in ALGORITHMS:
            results[dataset][algorithm] = []
            for seed in seeds:
                result = run_single_experiment(
                    dataset=dataset,
                    algorithm=algorithm,
                    seed=seed,
                    rounds=rounds,
                    n_clients=N_CLIENTS,
                    experiment_prefix="validation_feddc",
                    verbose=verbose
                )
                results[dataset][algorithm].append(result)
                if result['success']:
                    all_summaries.append(result['summary'])
    
    return {
        'results': results,
        'summaries': all_summaries
    }


def analyze_results(results: Dict) -> Dict[str, Any]:
    """
    Analyze Phase 6B results and make FedDC decision.
    """
    analysis = {
        'by_dataset': {},
        'comparison': {},
        'decision': None,
        'reason': None
    }
    
    # Analyze by dataset
    for dataset in DATASETS:
        scaffold_results = [r['summary'] for r in results['results'][dataset].get('scaffold', []) if r.get('success')]
        feddc_results = [r['summary'] for r in results['results'][dataset].get('feddc', []) if r.get('success')]
        
        if scaffold_results and feddc_results:
            scaffold_rmse = np.mean([r['test_rmse'] for r in scaffold_results])
            feddc_rmse = np.mean([r['test_rmse'] for r in feddc_results])
            scaffold_time = np.mean([r['total_time'] for r in scaffold_results])
            feddc_time = np.mean([r['total_time'] for r in feddc_results])
            
            improvement = (scaffold_rmse - feddc_rmse) / scaffold_rmse if scaffold_rmse > 0 else 0
            
            analysis['by_dataset'][dataset] = {
                'scaffold_rmse': scaffold_rmse,
                'feddc_rmse': feddc_rmse,
                'improvement': improvement,
                'scaffold_time': scaffold_time,
                'feddc_time': feddc_time,
                'time_overhead': (feddc_time - scaffold_time) / scaffold_time if scaffold_time > 0 else 0,
                'feddc_better': improvement > IMPROVEMENT_THRESHOLD,
                'no_regression': improvement > -REGRESSION_THRESHOLD
            }
    
    # Overall comparison
    all_scaffold_rmse = []
    all_feddc_rmse = []
    
    for dataset in DATASETS:
        if dataset in analysis['by_dataset']:
            all_scaffold_rmse.append(analysis['by_dataset'][dataset]['scaffold_rmse'])
            all_feddc_rmse.append(analysis['by_dataset'][dataset]['feddc_rmse'])
    
    if all_scaffold_rmse and all_feddc_rmse:
        avg_scaffold = np.mean(all_scaffold_rmse)
        avg_feddc = np.mean(all_feddc_rmse)
        overall_improvement = (avg_scaffold - avg_feddc) / avg_scaffold if avg_scaffold > 0 else 0
        
        analysis['comparison'] = {
            'avg_scaffold_rmse': avg_scaffold,
            'avg_feddc_rmse': avg_feddc,
            'overall_improvement': overall_improvement
        }
        
        # Decision logic per tasklist
        fd002_result = analysis['by_dataset'].get('FD002', {})
        fd004_result = analysis['by_dataset'].get('FD004', {})
        fd001_result = analysis['by_dataset'].get('FD001', {})
        
        # FedDC must outperform SCAFFOLD on FD002/FD004
        fd002_better = fd002_result.get('feddc_better', False)
        fd004_better = fd004_result.get('feddc_better', False)
        
        # No regression on FD001
        fd001_ok = fd001_result.get('no_regression', True)
        
        if fd002_better and fd004_better and fd001_ok:
            analysis['decision'] = 'PROMOTE'
            analysis['reason'] = 'FedDC outperforms SCAFFOLD on FD002/FD004 with no regression on FD001'
        elif (fd002_better or fd004_better) and fd001_ok:
            analysis['decision'] = 'MARGINAL'
            analysis['reason'] = 'FedDC shows mixed results - outperforms on some datasets but not all'
        elif not fd001_ok:
            analysis['decision'] = 'DROP'
            analysis['reason'] = f'FedDC shows regression on FD001 (improvement: {fd001_result.get("improvement", 0)*100:.1f}%)'
        else:
            analysis['decision'] = 'DROP'
            analysis['reason'] = 'FedDC does not significantly outperform SCAFFOLD'
    
    return analysis


def generate_decision_document(analysis: Dict, output_path: str) -> str:
    """
    Generate Phase 6B decision document.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    decision_emoji = '✓' if analysis['decision'] == 'PROMOTE' else '⚠' if analysis['decision'] == 'MARGINAL' else '✗'
    
    doc = f"""# Phase 6B Decision Document

**Generated:** {timestamp}  
**Phase:** 6B - FedDC Conditional Algorithm Extension

---

## 1. Executive Summary

### Decision: {decision_emoji} **{analysis['decision']}**

**Reason:** {analysis['reason']}

---

## 2. Dataset-by-Dataset Comparison

| Dataset | SCAFFOLD RMSE | FedDC RMSE | Improvement | Time Overhead | FedDC Better? |
|---------|---------------|------------|-------------|---------------|---------------|
"""
    
    for dataset in DATASETS:
        if dataset in analysis['by_dataset']:
            d = analysis['by_dataset'][dataset]
            better = '✓' if d['feddc_better'] else '✗'
            doc += f"| {dataset} | {d['scaffold_rmse']:.2f} | {d['feddc_rmse']:.2f} | {d['improvement']*100:+.1f}% | {d['time_overhead']*100:+.1f}% | {better} |\n"
    
    doc += f"""

---

## 3. Overall Comparison

- **Average SCAFFOLD Test RMSE:** {analysis['comparison'].get('avg_scaffold_rmse', 0):.2f}
- **Average FedDC Test RMSE:** {analysis['comparison'].get('avg_feddc_rmse', 0):.2f}
- **Overall Improvement:** {analysis['comparison'].get('overall_improvement', 0)*100:+.1f}%

---

## 4. Acceptance Criteria Check

| Criterion | Status | Details |
|-----------|--------|---------|
"""
    
    # Check criteria
    fd002 = analysis['by_dataset'].get('FD002', {})
    fd004 = analysis['by_dataset'].get('FD004', {})
    fd001 = analysis['by_dataset'].get('FD001', {})
    
    fd002_status = '✓' if fd002.get('feddc_better', False) else '✗'
    fd004_status = '✓' if fd004.get('feddc_better', False) else '✗'
    fd001_status = '✓' if fd001.get('no_regression', True) else '✗'
    
    doc += f"| FedDC outperforms SCAFFOLD on FD002 | {fd002_status} | {fd002.get('improvement', 0)*100:+.1f}% improvement |\n"
    doc += f"| FedDC outperforms SCAFFOLD on FD004 | {fd004_status} | {fd004.get('improvement', 0)*100:+.1f}% improvement |\n"
    doc += f"| No regression on FD001 | {fd001_status} | {fd001.get('improvement', 0)*100:+.1f}% change |\n"
    
    doc += f"""

---

## 5. Recommendation

"""
    
    if analysis['decision'] == 'PROMOTE':
        doc += """### ✓ **PROMOTE FedDC to Phase 6C**

FedDC has demonstrated significant improvement over SCAFFOLD on the challenging FD002 and FD004 datasets
while maintaining performance on FD001. Include FedDC in full-scale experiments.
"""
    elif analysis['decision'] == 'MARGINAL':
        doc += """### ⚠ **MARGINAL - Consider Carefully**

FedDC shows mixed results. Consider:
1. Running additional seeds to reduce variance
2. Tuning the alpha parameter
3. Evaluating the computational overhead vs improvement tradeoff

Recommendation: Proceed with caution or conduct additional experiments.
"""
    else:
        doc += """### ✗ **DROP FedDC**

FedDC does not provide sufficient improvement over SCAFFOLD to justify inclusion.
Proceed to Phase 6C with FedAvg, FedProx, and SCAFFOLD only.
"""
    
    doc += """

---

## 6. Next Steps

"""
    
    if analysis['decision'] == 'PROMOTE':
        doc += """1. Include FedDC in Phase 6C full-scale experiments
2. Test FedDC with full client count (100/248/260)
3. Evaluate partial participation robustness
"""
    else:
        doc += """1. Proceed to Phase 6C with promoted algorithms from Phase 6A
2. Document FedDC evaluation results for future reference
3. Consider FedDC for future work with different hyperparameters
"""
    
    doc += """

---

*Document generated by Phase 6B validation script*
"""
    
    # Write document
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    return output_path


def main():
    """Main entry point for Phase 6B validation."""
    parser = argparse.ArgumentParser(description='Phase 6B: FedDC Validation')
    parser.add_argument('--rounds', type=int, default=30, help='Rounds per experiment')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seeds')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    verbose = not args.quiet
    
    print("\n" + "="*70)
    print("PHASE 6B: FedDC CONDITIONAL ALGORITHM EXTENSION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Run validation
    results = run_phase6b_validation(
        rounds=args.rounds,
        seeds=seeds,
        verbose=verbose
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 6B SUMMARY")
    print("="*70)
    
    for dataset in DATASETS:
        if dataset in analysis['by_dataset']:
            d = analysis['by_dataset'][dataset]
            better = "✓ BETTER" if d['feddc_better'] else "✗ NOT BETTER"
            print(f"  {dataset}: SCAFFOLD={d['scaffold_rmse']:.2f}, FedDC={d['feddc_rmse']:.2f} "
                  f"({d['improvement']*100:+.1f}%) {better}")
    
    print(f"\n  Decision: {analysis['decision']}")
    print(f"  Reason: {analysis['reason']}")
    
    # Generate decision document
    decision_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'phase6b_decision.md')
    generate_decision_document(analysis, decision_path)
    
    print("\n" + "="*70)
    print("PHASE 6B COMPLETE")
    print("="*70)
    print(f"Decision document: {decision_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Save full results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'phase6b_results.json')
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    serializable = {
        'analysis': convert_numpy(analysis),
        'summaries': convert_numpy(results['summaries'])
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Full results: {results_path}")
    
    return analysis


if __name__ == '__main__':
    main()
