"""
Plotting utilities for FL experiment analysis (E-1).

Generates convergence, RMSE/MAE, and variance plots.
Saves outputs as PNG and SVG under experiments/{id}/plots/.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

# Color palette for algorithms
ALGORITHM_COLORS = {
    'fedavg': '#1f77b4',      # Blue
    'fedprox': '#ff7f0e',     # Orange
    'scaffold': '#2ca02c',    # Green
    'feddc': '#d62728',       # Red
    'centralized': '#9467bd', # Purple
    'local': '#8c564b'        # Brown
}


def _resolve_metric(df: pd.DataFrame, metric: str) -> str:
    """Resolve metric name to actual column, handling schema variations."""
    if metric in df.columns:
        return metric
    # Map between different naming conventions
    aliases = {
        'global_rmse': ['rmse', 'val_rmse'],
        'global_mae': ['mae', 'val_mae'],
        'global_loss': ['val_loss', 'loss'],
        'rmse': ['global_rmse', 'val_rmse'],
        'mae': ['global_mae', 'val_mae'],
    }
    for alt in aliases.get(metric, []):
        if alt in df.columns:
            return alt
    return metric  # Return original, will fail with helpful error


def ensure_plots_dir(experiment_dir: Union[str, Path]) -> Path:
    """Ensure plots directory exists and return path."""
    plots_dir = Path(experiment_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def load_rounds_csv(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """Load rounds.csv from experiment logs directory."""
    logs_dir = Path(experiment_dir) / 'logs'
    csv_path = logs_dir / 'rounds.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"rounds.csv not found at {csv_path}")
    
    return pd.read_csv(csv_path)


def load_epochs_csv(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """Load epochs.csv from experiment logs directory (for centralized)."""
    logs_dir = Path(experiment_dir) / 'logs'
    csv_path = logs_dir / 'epochs.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"epochs.csv not found at {csv_path}")
    
    return pd.read_csv(csv_path)


def load_clients_csv(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """Load clients.csv from experiment logs directory (for local baseline)."""
    logs_dir = Path(experiment_dir) / 'logs'
    csv_path = logs_dir / 'clients.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"clients.csv not found at {csv_path}")
    
    return pd.read_csv(csv_path)


def load_non_iid_csv(experiment_dir: Union[str, Path]) -> pd.DataFrame:
    """Load non_iid.csv from experiment logs directory."""
    logs_dir = Path(experiment_dir) / 'logs'
    csv_path = logs_dir / 'non_iid.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"non_iid.csv not found at {csv_path}")
    
    return pd.read_csv(csv_path)


def load_config(experiment_dir: Union[str, Path]) -> Dict:
    """Load config.json from experiment directory."""
    config_path = Path(experiment_dir) / 'config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_convergence(
    experiment_dir: Union[str, Path],
    metric: str = 'global_rmse',
    title: Optional[str] = None,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot training convergence curve (loss or RMSE over rounds/epochs).
    
    Args:
        experiment_dir: Path to experiment directory
        metric: Metric to plot ('global_rmse', 'global_mae', 'global_loss', 'train_loss', 'val_rmse')
        title: Custom plot title
        save: Save plot to files
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    experiment_dir = Path(experiment_dir)
    
    # Determine experiment type and load appropriate data
    logs_dir = experiment_dir / 'logs'
    
    if (logs_dir / 'rounds.csv').exists():
        # Federated experiment
        df = load_rounds_csv(experiment_dir)
        x_col = 'round'
        x_label = 'Communication Round'
        exp_type = 'federated'
        # Aggregate per-round (data is per-client-per-round, take mean across clients)
        if 'client_id' in df.columns:
            df = df.groupby('round').mean(numeric_only=True).reset_index()
    elif (logs_dir / 'epochs.csv').exists():
        # Centralized experiment
        df = load_epochs_csv(experiment_dir)
        x_col = 'epoch'
        x_label = 'Epoch'
        exp_type = 'centralized'
        # Map metric names for centralized
        metric_map = {
            'global_rmse': 'val_rmse',
            'global_mae': 'val_mae',
            'global_loss': 'val_loss'
        }
        metric = metric_map.get(metric, metric)
    else:
        raise FileNotFoundError(f"No rounds.csv or epochs.csv found in {logs_dir}")
    
    # Resolve metric name to actual column
    metric = _resolve_metric(df, metric)
    
    if metric not in df.columns:
        available = [c for c in df.columns if c not in [x_col, 'timestamp']]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot convergence
    color = ALGORITHM_COLORS.get('fedavg', '#1f77b4')
    ax.plot(df[x_col], df[metric], color=color, linewidth=2, marker='o', 
            markersize=3, markevery=max(1, len(df) // 20))
    
    # Formatting
    if title is None:
        config = load_config(experiment_dir) if (experiment_dir / 'config.json').exists() else {}
        algo = config.get('algorithm', exp_type).upper()
        dataset = config.get('dataset', 'Unknown')
        title = f'{algo} Convergence ({dataset})'
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add best value annotation
    best_idx = df[metric].idxmin()
    best_val = df[metric].min()
    best_round = df[x_col].iloc[best_idx]
    ax.axhline(y=best_val, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_val:.2f}')
    ax.scatter([best_round], [best_val], color='red', s=100, zorder=5, marker='*')
    ax.legend()
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir(experiment_dir)
        fig.savefig(plots_dir / f'convergence_{metric}.png')
        fig.savefig(plots_dir / f'convergence_{metric}.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_rmse_mae(
    experiment_dir: Union[str, Path],
    title: Optional[str] = None,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot RMSE and MAE curves on dual y-axis.
    
    Args:
        experiment_dir: Path to experiment directory
        title: Custom plot title
        save: Save plot to files
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    experiment_dir = Path(experiment_dir)
    logs_dir = experiment_dir / 'logs'
    
    # Load data based on experiment type
    if (logs_dir / 'rounds.csv').exists():
        df = load_rounds_csv(experiment_dir)
        x_col = 'round'
        x_label = 'Communication Round'
        # Aggregate per-round if needed
        if 'client_id' in df.columns:
            df = df.groupby('round').mean(numeric_only=True).reset_index()
        rmse_col = _resolve_metric(df, 'global_rmse')
        mae_col = _resolve_metric(df, 'global_mae')
    elif (logs_dir / 'epochs.csv').exists():
        df = load_epochs_csv(experiment_dir)
        x_col = 'epoch'
        x_label = 'Epoch'
        rmse_col, mae_col = 'val_rmse', 'val_mae'
    else:
        raise FileNotFoundError(f"No rounds.csv or epochs.csv found in {logs_dir}")
    
    # Create figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot RMSE
    line1 = ax1.plot(df[x_col], df[rmse_col], color='#1f77b4', linewidth=2, 
                      label='RMSE', marker='o', markersize=3, markevery=max(1, len(df) // 20))
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('RMSE', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Plot MAE
    line2 = ax2.plot(df[x_col], df[mae_col], color='#ff7f0e', linewidth=2, 
                      label='MAE', marker='s', markersize=3, markevery=max(1, len(df) // 20))
    ax2.set_ylabel('MAE', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Title
    if title is None:
        config = load_config(experiment_dir) if (experiment_dir / 'config.json').exists() else {}
        algo = config.get('algorithm', 'Unknown').upper()
        dataset = config.get('dataset', 'Unknown')
        title = f'{algo} RMSE & MAE ({dataset})'
    
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir(experiment_dir)
        fig.savefig(plots_dir / 'rmse_mae.png')
        fig.savefig(plots_dir / 'rmse_mae.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_client_variance(
    experiment_dir: Union[str, Path],
    title: Optional[str] = None,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot client data distribution (non-IID characteristics).
    
    Shows sample count distribution and RUL variance across clients.
    
    Args:
        experiment_dir: Path to experiment directory
        title: Custom plot title
        save: Save plot to files
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    experiment_dir = Path(experiment_dir)
    
    try:
        df = load_non_iid_csv(experiment_dir)
    except FileNotFoundError:
        # For local baseline, use clients.csv
        try:
            df = load_clients_csv(experiment_dir)
            df = df.rename(columns={'train_samples': 'num_samples'})
        except FileNotFoundError:
            raise FileNotFoundError(f"No non_iid.csv or clients.csv found in {experiment_dir}/logs/")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sample count distribution
    ax1 = axes[0]
    ax1.bar(range(len(df)), df['num_samples'].sort_values(ascending=False), 
            color='#1f77b4', alpha=0.7)
    ax1.set_xlabel('Client (sorted by sample count)')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Distribution Across Clients')
    ax1.axhline(y=df['num_samples'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["num_samples"].mean():.0f}')
    ax1.legend()
    
    # Plot 2: RUL variance or test RMSE distribution
    ax2 = axes[1]
    if 'rul_variance' in df.columns:
        values = df['rul_variance'].sort_values(ascending=False)
        ylabel = 'RUL Variance'
        title2 = 'RUL Variance Across Clients'
    elif 'mean_rul' in df.columns:
        values = df['mean_rul'].sort_values(ascending=False)
        ylabel = 'Mean RUL'
        title2 = 'Mean RUL Across Clients'
    elif 'test_rmse' in df.columns:
        values = df['test_rmse'].sort_values(ascending=False)
        ylabel = 'Test RMSE'
        title2 = 'Local Test RMSE Across Clients'
    else:
        values = df['num_samples']
        ylabel = 'Samples'
        title2 = 'Client Data Distribution'
    
    ax2.bar(range(len(values)), values, color='#ff7f0e', alpha=0.7)
    ax2.set_xlabel('Client (sorted)')
    ax2.set_ylabel(ylabel)
    ax2.set_title(title2)
    ax2.axhline(y=values.mean(), color='red', linestyle='--', 
                label=f'Mean: {values.mean():.2f}')
    ax2.legend()
    
    # Main title
    if title is None:
        config = load_config(experiment_dir) if (experiment_dir / 'config.json').exists() else {}
        dataset = config.get('dataset', 'Unknown')
        title = f'Client Data Heterogeneity ({dataset})'
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save:
        plots_dir = ensure_plots_dir(experiment_dir)
        fig.savefig(plots_dir / 'client_variance.png')
        fig.savefig(plots_dir / 'client_variance.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_participation_impact(
    experiment_dirs: Dict[str, Union[str, Path]],
    metric: str = 'global_rmse',
    title: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False
) -> plt.Figure:
    """
    Compare convergence across different participation fractions.
    
    Args:
        experiment_dirs: Dict mapping participation labels to experiment paths
                        e.g., {'100%': 'exp1/', '70%': 'exp2/', '50%': 'exp3/'}
        metric: Metric to compare
        title: Custom plot title
        save_dir: Directory to save plot (uses first experiment's plots/ if None)
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(experiment_dirs)))
    
    for (label, exp_dir), color in zip(experiment_dirs.items(), colors):
        exp_dir = Path(exp_dir)
        try:
            df = load_rounds_csv(exp_dir)
            metric_col = _resolve_metric(df, metric)
            ax.plot(df['round'], df[metric_col], label=label, linewidth=2, color=color)
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or 'Impact of Client Participation Rate')
    ax.legend(title='Participation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'participation_impact.png')
        fig.savefig(save_dir / 'participation_impact.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_algorithm_comparison(
    experiment_dirs: Dict[str, Union[str, Path]],
    metric: str = 'global_rmse',
    title: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False
) -> plt.Figure:
    """
    Compare convergence across different FL algorithms.
    
    Args:
        experiment_dirs: Dict mapping algorithm names to experiment paths
                        e.g., {'FedAvg': 'exp1/', 'FedProx': 'exp2/', 'SCAFFOLD': 'exp3/'}
        metric: Metric to compare
        title: Custom plot title
        save_dir: Directory to save plot
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo, exp_dir in experiment_dirs.items():
        exp_dir = Path(exp_dir)
        color = ALGORITHM_COLORS.get(algo.lower(), None)
        
        try:
            df = load_rounds_csv(exp_dir)
            metric_col = _resolve_metric(df, metric)
            ax.plot(df['round'], df[metric_col], label=algo, linewidth=2, color=color)
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title or 'Algorithm Comparison')
    ax.legend(title='Algorithm')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'algorithm_comparison.png')
        fig.savefig(save_dir / 'algorithm_comparison.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_final_metrics_bar(
    experiment_dirs: Dict[str, Union[str, Path]],
    title: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False
) -> plt.Figure:
    """
    Bar chart comparing final RMSE and MAE across experiments.
    
    Args:
        experiment_dirs: Dict mapping labels to experiment paths
        title: Custom plot title
        save_dir: Directory to save plot
        show: Display plot interactively
        
    Returns:
        matplotlib Figure object
    """
    results = []
    
    for label, exp_dir in experiment_dirs.items():
        exp_dir = Path(exp_dir)
        try:
            # Try loading summary.json first
            summary_path = exp_dir / 'logs' / 'summary.json'
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                rmse = summary.get('test_rmse', summary.get('weighted_test_rmse', 0))
                mae = summary.get('test_mae', summary.get('weighted_test_mae', 0))
            else:
                # Fallback to last row of rounds.csv or epochs.csv
                try:
                    df = load_rounds_csv(exp_dir)
                    rmse_col = _resolve_metric(df, 'global_rmse')
                    mae_col = _resolve_metric(df, 'global_mae')
                    rmse = df[rmse_col].iloc[-1]
                    mae = df[mae_col].iloc[-1]
                except:
                    df = load_epochs_csv(exp_dir)
                    rmse = df['val_rmse'].iloc[-1]
                    mae = df['val_mae'].iloc[-1]
            
            results.append({'label': label, 'RMSE': rmse, 'MAE': mae})
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
    
    if not results:
        raise ValueError("No valid experiment data found")
    
    df = pd.DataFrame(results)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['RMSE'], width, label='RMSE', color='#1f77b4')
    bars2 = ax.bar(x + width/2, df['MAE'], width, label='MAE', color='#ff7f0e')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Error')
    ax.set_title(title or 'Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'final_metrics_bar.png')
        fig.savefig(save_dir / 'final_metrics_bar.svg')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def generate_all_plots(
    experiment_dir: Union[str, Path],
    show: bool = False
) -> List[plt.Figure]:
    """
    Generate all applicable plots for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        show: Display plots interactively
        
    Returns:
        List of generated matplotlib Figure objects
    """
    experiment_dir = Path(experiment_dir)
    figures = []
    
    print(f"Generating plots for: {experiment_dir}")
    
    # Always try convergence plot
    try:
        fig = plot_convergence(experiment_dir, metric='global_rmse', save=True, show=show)
        figures.append(fig)
        print("  ✓ Convergence plot (RMSE)")
    except Exception as e:
        try:
            fig = plot_convergence(experiment_dir, metric='val_rmse', save=True, show=show)
            figures.append(fig)
            print("  ✓ Convergence plot (val_rmse)")
        except Exception as e2:
            print(f"  ✗ Convergence plot failed: {e2}")
    
    # RMSE/MAE dual plot
    try:
        fig = plot_rmse_mae(experiment_dir, save=True, show=show)
        figures.append(fig)
        print("  ✓ RMSE/MAE plot")
    except Exception as e:
        print(f"  ✗ RMSE/MAE plot failed: {e}")
    
    # Client variance plot
    try:
        fig = plot_client_variance(experiment_dir, save=True, show=show)
        figures.append(fig)
        print("  ✓ Client variance plot")
    except Exception as e:
        print(f"  ✗ Client variance plot skipped: {e}")
    
    plots_dir = ensure_plots_dir(experiment_dir)
    print(f"\nPlots saved to: {plots_dir}")
    
    return figures
