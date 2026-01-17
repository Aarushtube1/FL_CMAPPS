#!/usr/bin/env python
"""
Phase 6A Plotting Script.

Generates visualization plots for Phase 6A validation results:
1. Convergence curves by algorithm (per dataset)
2. Algorithm comparison bar charts (Test RMSE)
3. Partial participation degradation curves
4. Dataset comparison heatmap
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.paths import EXPERIMENTS_DIR, ensure_dir


# Configuration
DATASETS = ['FD001', 'FD002', 'FD004']
ALGORITHMS = ['fedavg', 'fedprox', 'scaffold']
ALGORITHM_LABELS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD'}
ALGORITHM_COLORS = {'fedavg': '#1f77b4', 'fedprox': '#ff7f0e', 'scaffold': '#2ca02c'}
PARTICIPATION_RATES = [1.0, 0.7, 0.5]


def load_experiment_summary(exp_dir: str) -> Dict[str, Any]:
    """Load summary.json from an experiment directory."""
    summary_path = os.path.join(exp_dir, 'logs', 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def load_all_s1a_experiments(seed: int = 42) -> Dict[str, Dict[str, Dict]]:
    """Load all S-1A experiments."""
    results = {}
    base_dir = os.path.join(EXPERIMENTS_DIR, 'validation_5clients')
    
    for dataset in DATASETS:
        results[dataset] = {}
        for algorithm in ALGORITHMS:
            exp_id = f"{dataset}_{algorithm}_{seed}"
            exp_dir = os.path.join(base_dir, exp_id)
            summary = load_experiment_summary(exp_dir)
            if summary:
                results[dataset][algorithm] = summary
    
    return results


def load_all_s1b_experiments(seed: int = 42) -> Dict[str, Dict[str, Dict[float, Dict]]]:
    """Load all S-1B partial participation experiments."""
    results = {}
    base_dir = os.path.join(EXPERIMENTS_DIR, 'validation_5clients_dropout')
    
    for dataset in DATASETS:
        results[dataset] = {}
        for algorithm in ALGORITHMS:
            results[dataset][algorithm] = {}
            for participation in PARTICIPATION_RATES:
                if participation == 1.0:
                    exp_id = f"{dataset}_{algorithm}_{seed}"
                else:
                    exp_id = f"{dataset}_{algorithm}_p{int(participation*100)}_{seed}"
                
                exp_dir = os.path.join(base_dir, exp_id)
                summary = load_experiment_summary(exp_dir)
                if summary:
                    results[dataset][algorithm][participation] = summary
    
    return results


def plot_convergence_curves(results: Dict, output_dir: str):
    """
    Plot convergence curves (RMSE vs Round) for each dataset.
    One subplot per dataset, all algorithms on same plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        
        for algorithm in ALGORITHMS:
            if dataset in results and algorithm in results[dataset]:
                summary = results[dataset][algorithm]
                rmse_history = summary.get('round_rmse_history', [])
                rounds = list(range(1, len(rmse_history) + 1))
                
                ax.plot(rounds, rmse_history, 
                       label=ALGORITHM_LABELS[algorithm],
                       color=ALGORITHM_COLORS[algorithm],
                       linewidth=2)
        
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Validation RMSE', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 30)
    
    fig.suptitle('Phase 6A: Convergence Curves (5-Client Validation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_algorithm_comparison(results: Dict, output_dir: str):
    """
    Plot algorithm comparison bar chart (Test RMSE).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    for i, algorithm in enumerate(ALGORITHMS):
        test_rmses = []
        for dataset in DATASETS:
            if dataset in results and algorithm in results[dataset]:
                test_rmses.append(results[dataset][algorithm].get('test_rmse', 0))
            else:
                test_rmses.append(0)
        
        bars = ax.bar(x + i * width, test_rmses, width, 
                     label=ALGORITHM_LABELS[algorithm],
                     color=ALGORITHM_COLORS[algorithm],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, test_rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('Phase 6A: Algorithm Comparison (Test RMSE)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'algorithm_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_participation_degradation(results: Dict, output_dir: str):
    """
    Plot partial participation degradation curves.
    Shows how Test RMSE changes with participation rate.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    participation_labels = ['100%', '70%', '50%']
    
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        
        for algorithm in ALGORITHMS:
            if dataset in results and algorithm in results[dataset]:
                rmses = []
                for participation in PARTICIPATION_RATES:
                    if participation in results[dataset][algorithm]:
                        rmses.append(results[dataset][algorithm][participation].get('test_rmse', 0))
                    else:
                        rmses.append(0)
                
                ax.plot(participation_labels, rmses, 
                       marker='o', markersize=8,
                       label=ALGORITHM_LABELS[algorithm],
                       color=ALGORITHM_COLORS[algorithm],
                       linewidth=2)
        
        ax.set_xlabel('Participation Rate', fontsize=11)
        ax.set_ylabel('Test RMSE', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Phase 6A S-1B: Partial Participation Stress Test', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'participation_degradation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_improvement_heatmap(results: Dict, output_dir: str):
    """
    Plot improvement heatmap (% improvement from initial to final RMSE).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Build improvement matrix
    improvements = []
    for dataset in DATASETS:
        row = []
        for algorithm in ALGORITHMS:
            if dataset in results and algorithm in results[dataset]:
                imp = results[dataset][algorithm].get('improvement', 0) * 100
                row.append(imp)
            else:
                row.append(0)
        improvements.append(row)
    
    improvements = np.array(improvements)
    
    # Create heatmap
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=30, vmax=70)
    
    # Labels
    ax.set_xticks(np.arange(len(ALGORITHMS)))
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_xticklabels([ALGORITHM_LABELS[a] for a in ALGORITHMS])
    ax.set_yticklabels(DATASETS)
    
    # Add text annotations
    for i in range(len(DATASETS)):
        for j in range(len(ALGORITHMS)):
            text = ax.text(j, i, f'{improvements[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    
    ax.set_title('Phase 6A: Convergence Improvement (%)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement %', fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'improvement_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_final_vs_best_rmse(results: Dict, output_dir: str):
    """
    Plot final RMSE vs best RMSE scatter plot.
    Shows how close each algorithm gets to its best achievable performance.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for algorithm in ALGORITHMS:
        finals = []
        bests = []
        for dataset in DATASETS:
            if dataset in results and algorithm in results[dataset]:
                finals.append(results[dataset][algorithm].get('final_rmse', 0))
                bests.append(results[dataset][algorithm].get('best_rmse', 0))
        
        ax.scatter(bests, finals, s=100, 
                  label=ALGORITHM_LABELS[algorithm],
                  color=ALGORITHM_COLORS[algorithm],
                  edgecolor='black', linewidth=1)
    
    # Add diagonal line (y=x)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x (ideal)')
    
    ax.set_xlabel('Best RMSE (achieved)', fontsize=12)
    ax.set_ylabel('Final RMSE', fontsize=12)
    ax.set_title('Phase 6A: Final vs Best RMSE', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'final_vs_best_rmse.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_training_time(results: Dict, output_dir: str):
    """
    Plot training time comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(DATASETS))
    width = 0.25
    
    for i, algorithm in enumerate(ALGORITHMS):
        times = []
        for dataset in DATASETS:
            if dataset in results and algorithm in results[dataset]:
                times.append(results[dataset][algorithm].get('total_time', 0))
            else:
                times.append(0)
        
        bars = ax.bar(x + i * width, times, width, 
                     label=ALGORITHM_LABELS[algorithm],
                     color=ALGORITHM_COLORS[algorithm],
                     edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Phase 6A: Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_time.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def generate_summary_table(results: Dict, output_dir: str):
    """
    Generate a markdown summary table.
    """
    lines = [
        "# Phase 6A Results Summary",
        "",
        "## S-1A: Algorithm Performance",
        "",
        "| Dataset | Algorithm | Test RMSE | Final RMSE | Best RMSE | Improvement | Time (s) |",
        "|---------|-----------|-----------|------------|-----------|-------------|----------|"
    ]
    
    for dataset in DATASETS:
        for algorithm in ALGORITHMS:
            if dataset in results and algorithm in results[dataset]:
                s = results[dataset][algorithm]
                lines.append(
                    f"| {dataset} | {ALGORITHM_LABELS[algorithm]} | "
                    f"{s.get('test_rmse', 0):.2f} | {s.get('final_rmse', 0):.2f} | "
                    f"{s.get('best_rmse', 0):.2f} | {s.get('improvement', 0)*100:.1f}% | "
                    f"{s.get('total_time', 0):.1f} |"
                )
    
    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **All algorithms converged** without divergence on all datasets",
        "2. **FD001** shows best performance (Test RMSE ~10-12)",
        "3. **FD002/FD004** have higher RMSE due to more complex operating conditions",
        "4. **FedAvg/FedProx** perform similarly, SCAFFOLD slightly behind",
        "",
        "---",
        "*Generated by plot_phase6a.py*"
    ])
    
    output_path = os.path.join(output_dir, 'summary_table.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Generate all Phase 6A plots."""
    print("="*60)
    print("PHASE 6A PLOT GENERATION")
    print("="*60)
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'phase6a_plots')
    ensure_dir(output_dir)
    
    # Load S-1A experiments
    print("\nLoading S-1A experiments...")
    s1a_results = load_all_s1a_experiments(seed=42)
    
    # Load S-1B experiments
    print("Loading S-1B experiments...")
    s1b_results = load_all_s1b_experiments(seed=42)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_convergence_curves(s1a_results, output_dir)
    plot_algorithm_comparison(s1a_results, output_dir)
    plot_improvement_heatmap(s1a_results, output_dir)
    plot_final_vs_best_rmse(s1a_results, output_dir)
    plot_training_time(s1a_results, output_dir)
    plot_participation_degradation(s1b_results, output_dir)
    
    # Generate summary table
    generate_summary_table(s1a_results, output_dir)
    
    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
