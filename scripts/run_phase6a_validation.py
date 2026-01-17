#!/usr/bin/env python
"""
Phase 6A: 5-Client Behavioral Validation & Partial Participation Tests

This script runs the Phase 6A validation experiments:
- S-1A: 5-client behavioral validation (FedAvg, FedProx, SCAFFOLD on FD001, FD002, FD004)
- S-1B: 5-client partial participation stress test (100%, 70%, 50%)

Produces:
- experiments/validation_5clients/{dataset}_{algorithm}_{seed}/
- experiments/validation_5clients_dropout/{dataset}_{algorithm}_{participation}_{seed}/
- docs/phase6a_decision.md

Usage:
    python run_phase6a_validation.py
    python run_phase6a_validation.py --rounds 50 --seeds 42,123,456
    python run_phase6a_validation.py --skip-s1b  # Skip partial participation
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
from client import FLClient, FLClientFedProx, FLClientSCAFFOLD, create_clients_from_datasets
from runner import FLRunner
from analysis.non_iid_audit import run_non_iid_audit, compute_heterogeneity_metrics


# Phase 6A Configuration
DATASETS = ['FD001', 'FD002', 'FD004']  # FD003 excluded per tasklist
ALGORITHMS = ['fedavg', 'fedprox', 'scaffold']
N_CLIENTS = 5
DEFAULT_ROUNDS = 50
DEFAULT_SEEDS = [42, 123, 456]
PARTICIPATION_RATES = [1.0, 0.7, 0.5]

# Thresholds for validation
DIVERGENCE_THRESHOLD = 200.0  # RMSE above this = divergence
CONVERGENCE_RATIO_THRESHOLD = 0.8  # Final RMSE must be < 80% of initial


def run_single_experiment(
    dataset: str,
    algorithm: str,
    seed: int,
    rounds: int = DEFAULT_ROUNDS,
    n_clients: int = N_CLIENTS,
    participation: float = 1.0,
    mu: float = 0.01,
    experiment_prefix: str = "validation_5clients",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single FL experiment and return results.
    
    Returns:
        Dict with experiment results and metrics
    """
    set_seed(seed)
    
    # Generate experiment ID
    if participation < 1.0:
        exp_id = f"{experiment_prefix}/{dataset}_{algorithm}_p{int(participation*100)}_{seed}"
    else:
        exp_id = f"{experiment_prefix}/{dataset}_{algorithm}_{seed}"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {exp_id}")
        print(f"{'='*60}")
    
    # Check if experiment already completed (resume logic)
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
            'converged': summary.get('converged', False),
            'diverged': summary.get('diverged', False),
            'final_rmse': summary.get('final_rmse'),
            'initial_rmse': summary.get('initial_rmse'),
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
        
        # Compute non-IID stats
        raw_df = result['raw_df']
        non_iid_stats = compute_all_clients_non_iid_stats(client_datasets, raw_df)
        
        # Config
        config_data = {
            'algorithm': algorithm,
            'dataset': dataset,
            'rounds': rounds,
            'local_epochs': 1,
            'batch_size': 64,
            'lr': 0.001,
            'participation': participation,
            'n_clients': len(client_datasets),
            'n_features': n_features,
            'seed': seed,
            'mu': mu if algorithm == 'fedprox' else None,
            'phase': '6A',
            'experiment_type': 'validation_5clients' if participation == 1.0 else 'partial_participation'
        }
        
        # Logger
        logger = ExperimentLogger(
            experiment_id=exp_id,
            seed=seed,
            config=config_data,
            prevent_overwrite=False
        )
        
        # Log non-IID stats
        for stats in non_iid_stats:
            logger.log_non_iid(**stats)
        
        # Create server
        server = FLServer(input_channels=n_features, device="cpu", seed=seed)
        server.init_model()
        
        # Create clients based on algorithm
        if algorithm == 'fedprox':
            clients = {}
            for client_id, ds in client_datasets.items():
                clients[client_id] = FLClientFedProx(
                    client_id=client_id, dataset=ds, device="cpu",
                    learning_rate=0.001, batch_size=64, local_epochs=1, mu=mu
                )
        elif algorithm == 'scaffold':
            clients = {}
            for client_id, ds in client_datasets.items():
                clients[client_id] = FLClientSCAFFOLD(
                    client_id=client_id, dataset=ds, device="cpu",
                    learning_rate=0.001, batch_size=64, local_epochs=1
                )
            server.init_scaffold_control(n_features)
        else:
            clients = create_clients_from_datasets(
                client_datasets, device="cpu",
                learning_rate=0.001, batch_size=64, local_epochs=1
            )
        
        # Runner
        runner = FLRunner(server=server, clients=clients, logger=logger, device="cpu")
        
        # Training
        start_time = time.time()
        round_metrics = runner.run_training(
            max_rounds=rounds,
            participation_fraction=participation,
            algorithm=algorithm,
            verbose=verbose
        )
        total_time = time.time() - start_time
        
        # Final evaluation
        test_metrics = runner.evaluate_global_model()
        
        # Extract per-round RMSE for convergence analysis
        round_rmse = [m['global_rmse'] for m in round_metrics]
        initial_rmse = round_rmse[0] if round_rmse else 0.0
        final_rmse = round_rmse[-1] if round_rmse else 0.0
        best_rmse = min(round_rmse) if round_rmse else 0.0
        best_round = round_rmse.index(best_rmse) + 1 if round_rmse else 0
        
        # Check for divergence
        diverged = final_rmse > DIVERGENCE_THRESHOLD or any(r > DIVERGENCE_THRESHOLD for r in round_rmse)
        converged = final_rmse < initial_rmse * CONVERGENCE_RATIO_THRESHOLD if initial_rmse > 0 else False
        
        # Save model
        exp_dir = get_experiment_dir(exp_id)
        torch.save(server.global_model.state_dict(), os.path.join(exp_dir, 'final_model.pt'))
        
        # Save summary
        summary = {
            'experiment_id': exp_id,
            'dataset': dataset,
            'algorithm': algorithm,
            'seed': seed,
            'rounds': rounds,
            'n_clients': len(client_datasets),
            'participation': participation,
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'best_rmse': best_rmse,
            'best_round': best_round,
            'test_rmse': test_metrics['test_rmse'],
            'test_mae': test_metrics['test_mae'],
            'total_time': total_time,
            'avg_round_time': float(np.mean(runner.round_times)),
            'converged': converged,
            'diverged': diverged,
            'improvement': (initial_rmse - final_rmse) / initial_rmse if initial_rmse > 0 else 0,
            'round_rmse_history': round_rmse
        }
        
        logs_dir = os.path.join(exp_dir, 'logs')
        with open(os.path.join(logs_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            status = "✓ CONVERGED" if converged else ("✗ DIVERGED" if diverged else "○ MARGINAL")
            print(f"\n{status}")
            print(f"Initial RMSE: {initial_rmse:.2f} → Final: {final_rmse:.2f} (Best: {best_rmse:.2f} @ round {best_round})")
            print(f"Test RMSE: {test_metrics['test_rmse']:.2f}, MAE: {test_metrics['test_mae']:.2f}")
            print(f"Time: {total_time:.1f}s")
        
        return {
            'success': True,
            'summary': summary,
            'experiment_dir': exp_dir
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


def run_s1a_validation(
    rounds: int = DEFAULT_ROUNDS,
    seeds: List[int] = DEFAULT_SEEDS,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run S-1A: 5-Client Behavioral Validation.
    
    Tests FedAvg, FedProx, SCAFFOLD on FD001, FD002, FD004 with 5 clients.
    """
    print("\n" + "="*70)
    print("PHASE 6A - S-1A: 5-Client Behavioral Validation")
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
                    participation=1.0,
                    experiment_prefix="validation_5clients",
                    verbose=verbose
                )
                results[dataset][algorithm].append(result)
                if result['success']:
                    all_summaries.append(result['summary'])
    
    return {
        'results': results,
        'summaries': all_summaries
    }


def run_s1b_partial_participation(
    rounds: int = DEFAULT_ROUNDS,
    seeds: List[int] = DEFAULT_SEEDS,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run S-1B: 5-Client Partial Participation Stress Test.
    
    Tests 100%, 70%, 50% participation rates.
    """
    print("\n" + "="*70)
    print("PHASE 6A - S-1B: 5-Client Partial Participation Stress Test")
    print("="*70)
    print(f"Datasets: {DATASETS}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Participation rates: {PARTICIPATION_RATES}")
    print(f"Seeds: {seeds[:1]}")  # Use only first seed for dropout tests
    print("="*70)
    
    results = {}
    all_summaries = []
    seed = seeds[0]  # Use first seed for dropout
    
    for dataset in DATASETS:
        results[dataset] = {}
        for algorithm in ALGORITHMS:
            results[dataset][algorithm] = {}
            for participation in PARTICIPATION_RATES:
                result = run_single_experiment(
                    dataset=dataset,
                    algorithm=algorithm,
                    seed=seed,
                    rounds=rounds,
                    n_clients=N_CLIENTS,
                    participation=participation,
                    experiment_prefix="validation_5clients_dropout",
                    verbose=verbose
                )
                results[dataset][algorithm][participation] = result
                if result['success']:
                    all_summaries.append(result['summary'])
    
    return {
        'results': results,
        'summaries': all_summaries
    }


def analyze_s1a_results(s1a_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze S-1A results and determine algorithm performance.
    """
    summaries = s1a_results['summaries']
    
    analysis = {
        'by_dataset': {},
        'by_algorithm': {},
        'convergence_summary': {},
        'recommendations': []
    }
    
    # Analyze by dataset
    for dataset in DATASETS:
        dataset_summaries = [s for s in summaries if s['dataset'] == dataset]
        analysis['by_dataset'][dataset] = {
            'n_experiments': len(dataset_summaries),
            'avg_test_rmse': np.mean([s['test_rmse'] for s in dataset_summaries]),
            'convergence_rate': sum(1 for s in dataset_summaries if s['converged']) / len(dataset_summaries) if dataset_summaries else 0,
            'divergence_rate': sum(1 for s in dataset_summaries if s['diverged']) / len(dataset_summaries) if dataset_summaries else 0
        }
    
    # Analyze by algorithm
    for algorithm in ALGORITHMS:
        algo_summaries = [s for s in summaries if s['algorithm'] == algorithm]
        
        if algo_summaries:
            avg_test_rmse = np.mean([s['test_rmse'] for s in algo_summaries])
            std_test_rmse = np.std([s['test_rmse'] for s in algo_summaries])
            convergence_rate = sum(1 for s in algo_summaries if s['converged']) / len(algo_summaries)
            divergence_rate = sum(1 for s in algo_summaries if s['diverged']) / len(algo_summaries)
            avg_improvement = np.mean([s['improvement'] for s in algo_summaries])
            
            analysis['by_algorithm'][algorithm] = {
                'n_experiments': len(algo_summaries),
                'avg_test_rmse': avg_test_rmse,
                'std_test_rmse': std_test_rmse,
                'convergence_rate': convergence_rate,
                'divergence_rate': divergence_rate,
                'avg_improvement': avg_improvement,
                'passed': convergence_rate >= 0.5 and divergence_rate == 0
            }
    
    # Generate recommendations
    fedavg_stats = analysis['by_algorithm'].get('fedavg', {})
    fedprox_stats = analysis['by_algorithm'].get('fedprox', {})
    scaffold_stats = analysis['by_algorithm'].get('scaffold', {})
    
    # FedAvg is baseline - always passes if it converges
    if fedavg_stats.get('passed', False):
        analysis['recommendations'].append({
            'algorithm': 'fedavg',
            'decision': 'PROMOTE',
            'reason': 'Baseline algorithm with acceptable convergence'
        })
    else:
        analysis['recommendations'].append({
            'algorithm': 'fedavg',
            'decision': 'INVESTIGATE',
            'reason': 'Baseline showing issues - investigate before proceeding'
        })
    
    # FedProx - should stabilize or improve vs FedAvg
    if fedprox_stats.get('passed', False):
        fedprox_better = fedprox_stats.get('avg_test_rmse', 999) <= fedavg_stats.get('avg_test_rmse', 999) * 1.05
        analysis['recommendations'].append({
            'algorithm': 'fedprox',
            'decision': 'PROMOTE' if fedprox_better else 'CONDITIONAL',
            'reason': 'Comparable or better than FedAvg' if fedprox_better else 'Passed but no significant improvement'
        })
    else:
        analysis['recommendations'].append({
            'algorithm': 'fedprox',
            'decision': 'DROP',
            'reason': 'Failed validation criteria'
        })
    
    # SCAFFOLD - understand behavior even if worse
    if scaffold_stats.get('divergence_rate', 1) == 0:
        scaffold_analysis = 'PROMOTE' if scaffold_stats.get('passed', False) else 'CONDITIONAL'
        analysis['recommendations'].append({
            'algorithm': 'scaffold',
            'decision': scaffold_analysis,
            'reason': 'No divergence observed' + (', convergence acceptable' if scaffold_stats.get('passed', False) else ', convergence marginal')
        })
    else:
        analysis['recommendations'].append({
            'algorithm': 'scaffold',
            'decision': 'DROP',
            'reason': 'Divergence observed in some experiments'
        })
    
    return analysis


def analyze_s1b_results(s1b_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze S-1B partial participation results.
    """
    summaries = s1b_results['summaries']
    
    analysis = {
        'by_participation': {},
        'degradation_smooth': True,
        'stable_at_50': True
    }
    
    for participation in PARTICIPATION_RATES:
        part_summaries = [s for s in summaries if s['participation'] == participation]
        if part_summaries:
            analysis['by_participation'][participation] = {
                'n_experiments': len(part_summaries),
                'avg_test_rmse': np.mean([s['test_rmse'] for s in part_summaries]),
                'std_test_rmse': np.std([s['test_rmse'] for s in part_summaries]),
                'convergence_rate': sum(1 for s in part_summaries if s['converged']) / len(part_summaries),
                'divergence_rate': sum(1 for s in part_summaries if s['diverged']) / len(part_summaries)
            }
    
    # Check for smooth degradation
    if 1.0 in analysis['by_participation'] and 0.5 in analysis['by_participation']:
        rmse_100 = analysis['by_participation'][1.0]['avg_test_rmse']
        rmse_50 = analysis['by_participation'][0.5]['avg_test_rmse']
        
        # Degradation should be < 50% worse at 50% participation
        analysis['degradation_smooth'] = rmse_50 < rmse_100 * 1.5
        analysis['stable_at_50'] = analysis['by_participation'][0.5]['divergence_rate'] == 0
    
    return analysis


def generate_decision_document(
    s1a_analysis: Dict[str, Any],
    s1b_analysis: Dict[str, Any],
    output_path: str
) -> str:
    """
    Generate the Phase 6A decision document (docs/phase6a_decision.md).
    """
    
    doc = f"""# Phase 6A Decision Document

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Phase:** 6A - Small-Scale Validation & Algorithm Gating

---

## 1. Executive Summary

### S-1A: 5-Client Behavioral Validation

| Algorithm | Avg Test RMSE | Convergence Rate | Divergence Rate | Decision |
|-----------|---------------|------------------|-----------------|----------|
"""
    
    for algo in ALGORITHMS:
        stats = s1a_analysis['by_algorithm'].get(algo, {})
        rec = next((r for r in s1a_analysis['recommendations'] if r['algorithm'] == algo), {})
        doc += f"| {algo.upper()} | {stats.get('avg_test_rmse', 'N/A'):.2f} | {stats.get('convergence_rate', 0)*100:.0f}% | {stats.get('divergence_rate', 0)*100:.0f}% | **{rec.get('decision', 'N/A')}** |\n"
    
    doc += f"""

### S-1B: Partial Participation Stress Test

| Participation | Avg Test RMSE | Convergence Rate | Divergence Rate |
|--------------|---------------|------------------|-----------------|
"""
    
    for part in PARTICIPATION_RATES:
        stats = s1b_analysis['by_participation'].get(part, {})
        doc += f"| {int(part*100)}% | {stats.get('avg_test_rmse', 'N/A'):.2f} | {stats.get('convergence_rate', 0)*100:.0f}% | {stats.get('divergence_rate', 0)*100:.0f}% |\n"
    
    doc += f"""

---

## 2. Dataset Analysis

"""
    
    for dataset in DATASETS:
        stats = s1a_analysis['by_dataset'].get(dataset, {})
        doc += f"""### {dataset}
- Experiments: {stats.get('n_experiments', 0)}
- Average Test RMSE: {stats.get('avg_test_rmse', 'N/A'):.2f}
- Convergence Rate: {stats.get('convergence_rate', 0)*100:.0f}%
- Divergence Rate: {stats.get('divergence_rate', 0)*100:.0f}%

"""
    
    doc += f"""---

## 3. Algorithm Recommendations

"""
    
    for rec in s1a_analysis['recommendations']:
        doc += f"""### {rec['algorithm'].upper()}
- **Decision:** {rec['decision']}
- **Reason:** {rec['reason']}

"""
    
    doc += f"""---

## 4. Partial Participation Assessment

- **Smooth Degradation:** {"✓ Yes" if s1b_analysis.get('degradation_smooth', False) else "✗ No"}
- **Stable at 50%:** {"✓ Yes" if s1b_analysis.get('stable_at_50', False) else "✗ No"}

---

## 5. Gate 1 Decision

### Algorithms Promoted to Phase 6B/6C:
"""
    
    promoted = [r['algorithm'].upper() for r in s1a_analysis['recommendations'] if r['decision'] in ['PROMOTE', 'CONDITIONAL']]
    dropped = [r['algorithm'].upper() for r in s1a_analysis['recommendations'] if r['decision'] == 'DROP']
    
    for algo in promoted:
        doc += f"- ✓ {algo}\n"
    
    if dropped:
        doc += f"\n### Algorithms Dropped:\n"
        for algo in dropped:
            doc += f"- ✗ {algo}\n"
    
    doc += f"""

---

## 6. Next Steps

1. **If all algorithms promoted:** Proceed to Phase 6C (Full-Scale Experiments)
2. **If SCAFFOLD marginal:** Consider FedDC implementation (Phase 6B)
3. **If divergence detected:** Stop and investigate before scaling

---

*Document generated by Phase 6A validation script*
"""
    
    # Write document
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Phase 6A Validation')
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS, help='Rounds per experiment')
    parser.add_argument('--seeds', type=str, default='42,123,456', help='Comma-separated seeds')
    parser.add_argument('--skip-s1a', action='store_true', help='Skip S-1A validation')
    parser.add_argument('--skip-s1b', action='store_true', help='Skip S-1B partial participation')
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    args = parser.parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    verbose = args.verbose and not args.quiet
    
    print("\n" + "="*70)
    print("PHASE 6A: 5-CLIENT VALIDATION & ALGORITHM GATING")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # S-1A: Behavioral Validation
    if not args.skip_s1a:
        s1a_results = run_s1a_validation(rounds=args.rounds, seeds=seeds, verbose=verbose)
        s1a_analysis = analyze_s1a_results(s1a_results)
        results['s1a'] = {
            'results': s1a_results,
            'analysis': s1a_analysis
        }
        
        print("\n" + "="*70)
        print("S-1A SUMMARY")
        print("="*70)
        for rec in s1a_analysis['recommendations']:
            print(f"  {rec['algorithm'].upper()}: {rec['decision']} - {rec['reason']}")
    else:
        print("\n[SKIP] S-1A validation skipped")
        s1a_analysis = {'recommendations': [], 'by_algorithm': {}, 'by_dataset': {}}
    
    # S-1B: Partial Participation
    if not args.skip_s1b:
        s1b_results = run_s1b_partial_participation(rounds=args.rounds, seeds=seeds, verbose=verbose)
        s1b_analysis = analyze_s1b_results(s1b_results)
        results['s1b'] = {
            'results': s1b_results,
            'analysis': s1b_analysis
        }
        
        print("\n" + "="*70)
        print("S-1B SUMMARY")
        print("="*70)
        print(f"  Smooth Degradation: {'✓' if s1b_analysis.get('degradation_smooth', False) else '✗'}")
        print(f"  Stable at 50%: {'✓' if s1b_analysis.get('stable_at_50', False) else '✗'}")
    else:
        print("\n[SKIP] S-1B partial participation skipped")
        s1b_analysis = {'by_participation': {}, 'degradation_smooth': True, 'stable_at_50': True}
    
    # Generate decision document
    docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    decision_path = os.path.join(docs_dir, 'phase6a_decision.md')
    generate_decision_document(s1a_analysis, s1b_analysis, decision_path)
    
    print("\n" + "="*70)
    print("PHASE 6A COMPLETE")
    print("="*70)
    print(f"Decision document: {decision_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Save full results
    results_path = os.path.join(docs_dir, 'phase6a_results.json')
    
    # Convert numpy types for JSON serialization
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
    
    serializable_results = {
        's1a_analysis': convert_numpy(s1a_analysis),
        's1b_analysis': convert_numpy(s1b_analysis)
    }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Full results: {results_path}")
    
    return results


if __name__ == '__main__':
    main()
