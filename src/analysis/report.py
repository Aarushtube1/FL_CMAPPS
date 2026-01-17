"""
Experiment report generator (E-2).

Generates markdown summary reports with links to logs and plots.
Output: experiments/{id}/report/index.md
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd


def generate_experiment_report(
    experiment_dir: Union[str, Path],
    title: Optional[str] = None,
    author: str = "FL Experiment Runner"
) -> str:
    """
    Generate a markdown report for a single experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        title: Custom report title
        author: Author name for report header
        
    Returns:
        Path to generated report file
    """
    experiment_dir = Path(experiment_dir)
    experiment_id = experiment_dir.name
    
    # Create report directory
    report_dir = experiment_dir / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config if available
    config = {}
    config_path = experiment_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Determine experiment type
    logs_dir = experiment_dir / 'logs'
    exp_type = config.get('algorithm', 'unknown')
    
    if exp_type == 'centralized':
        report_content = _generate_centralized_report(
            experiment_dir, config, title, author, experiment_id
        )
    elif exp_type == 'local':
        report_content = _generate_local_report(
            experiment_dir, config, title, author, experiment_id
        )
    else:
        report_content = _generate_federated_report(
            experiment_dir, config, title, author, experiment_id
        )
    
    # Write report
    report_path = report_dir / 'index.md'
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated: {report_path}")
    return str(report_path)


def _generate_federated_report(
    experiment_dir: Path,
    config: Dict,
    title: Optional[str],
    author: str,
    experiment_id: str
) -> str:
    """Generate report for federated learning experiment."""
    
    logs_dir = experiment_dir / 'logs'
    plots_dir = experiment_dir / 'plots'
    
    # Load metrics data
    rounds_df = None
    if (logs_dir / 'rounds.csv').exists():
        rounds_df = pd.read_csv(logs_dir / 'rounds.csv')
        # Aggregate per-round if data is per-client-per-round
        if rounds_df is not None and 'client_id' in rounds_df.columns:
            rounds_df = rounds_df.groupby('round').mean(numeric_only=True).reset_index()
    
    summary = {}
    if (logs_dir / 'summary.json').exists():
        with open(logs_dir / 'summary.json') as f:
            summary = json.load(f)
    
    non_iid_df = None
    if (logs_dir / 'non_iid.csv').exists():
        non_iid_df = pd.read_csv(logs_dir / 'non_iid.csv')
    
    # Build report content
    algo = config.get('algorithm', 'FedAvg').upper()
    dataset = config.get('dataset', 'Unknown')
    
    if title is None:
        title = f"{algo} Experiment Report — {dataset}"
    
    content = f"""# {title}

**Experiment ID:** `{experiment_id}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** {author}

---

## 1. Configuration Summary

| Parameter | Value |
|-----------|-------|
| Algorithm | {config.get('algorithm', 'N/A')} |
| Dataset | {config.get('dataset', 'N/A')} |
| Rounds | {config.get('rounds', 'N/A')} |
| Local Epochs | {config.get('local_epochs', 'N/A')} |
| Batch Size | {config.get('batch_size', 'N/A')} |
| Learning Rate | {config.get('lr', 'N/A')} |
| Participation | {config.get('participation', 1.0) * 100:.0f}% |
| Num Clients | {config.get('n_clients', 'N/A')} |
| Seed | {config.get('seed', 'N/A')} |
"""
    
    if config.get('algorithm') == 'fedprox':
        content += f"| Proximal μ | {config.get('mu', 'N/A')} |\n"
    
    content += "\n---\n\n## 2. Final Results\n\n"
    
    # Resolve metric column names
    rmse_col = 'global_rmse' if rounds_df is not None and 'global_rmse' in rounds_df.columns else 'rmse'
    mae_col = 'global_mae' if rounds_df is not None and 'global_mae' in rounds_df.columns else 'mae'
    
    if summary:
        content += f"""| Metric | Value |
|--------|-------|
| **Test RMSE** | **{summary.get('test_rmse', 'N/A'):.2f}** |
| **Test MAE** | **{summary.get('test_mae', 'N/A'):.2f}** |
| Total Time | {summary.get('total_time', 0):.1f}s |
| Avg Round Time | {summary.get('avg_round_time', 0):.2f}s |
"""
    elif rounds_df is not None and rmse_col in rounds_df.columns:
        final = rounds_df.iloc[-1]
        content += f"""| Metric | Value |
|--------|-------|
| **Final RMSE** | **{final.get(rmse_col, 'N/A'):.2f}** |
| **Final MAE** | **{final.get(mae_col, 'N/A'):.2f}** |
| Best RMSE | {rounds_df[rmse_col].min():.2f} (Round {rounds_df[rmse_col].idxmin() + 1}) |
"""
    
    content += "\n---\n\n## 3. Convergence Analysis\n\n"
    
    if rounds_df is not None and rmse_col in rounds_df.columns:
        content += f"""Training completed **{len(rounds_df)}** rounds.

- **Initial RMSE:** {rounds_df[rmse_col].iloc[0]:.2f}
- **Final RMSE:** {rounds_df[rmse_col].iloc[-1]:.2f}
- **Best RMSE:** {rounds_df[rmse_col].min():.2f} (Round {rounds_df[rmse_col].idxmin() + 1})
- **Improvement:** {rounds_df[rmse_col].iloc[0] - rounds_df[rmse_col].min():.2f} ({(1 - rounds_df[rmse_col].min() / rounds_df[rmse_col].iloc[0]) * 100:.1f}%)

"""
    
    # Add plots if they exist
    content += "### Plots\n\n"
    
    plot_files = list(plots_dir.glob('*.png')) if plots_dir.exists() else []
    if plot_files:
        for plot in sorted(plot_files):
            rel_path = f"../plots/{plot.name}"
            content += f"![{plot.stem}]({rel_path})\n\n"
    else:
        content += "*No plots generated yet. Run `generate_all_plots(experiment_dir)` to create plots.*\n\n"
    
    content += "---\n\n## 4. Data Distribution (Non-IID)\n\n"
    
    if non_iid_df is not None:
        content += f"""| Statistic | Value |
|-----------|-------|
| Num Clients | {len(non_iid_df)} |
| Total Samples | {non_iid_df['num_samples'].sum():,} |
| Min Samples/Client | {non_iid_df['num_samples'].min()} |
| Max Samples/Client | {non_iid_df['num_samples'].max()} |
| Mean Samples/Client | {non_iid_df['num_samples'].mean():.1f} |
| Std Samples/Client | {non_iid_df['num_samples'].std():.1f} |
"""
        if 'mean_rul' in non_iid_df.columns:
            content += f"| Mean RUL Range | [{non_iid_df['mean_rul'].min():.1f}, {non_iid_df['mean_rul'].max():.1f}] |\n"
    else:
        content += "*No non-IID statistics available.*\n"
    
    content += f"""
---

## 5. Files & Artifacts

### Logs
- [rounds.csv](../logs/rounds.csv) — Round-wise metrics
- [non_iid.csv](../logs/non_iid.csv) — Per-client data statistics
- [summary.json](../logs/summary.json) — Final experiment summary

### Model
- [final_model.pt](../final_model.pt) — Final global model checkpoint

### Configuration
- [config.json](../config.json) — Full experiment configuration

---

*Report generated automatically by FL Experiment Runner*
"""
    
    return content


def _generate_centralized_report(
    experiment_dir: Path,
    config: Dict,
    title: Optional[str],
    author: str,
    experiment_id: str
) -> str:
    """Generate report for centralized baseline experiment."""
    
    logs_dir = experiment_dir / 'logs'
    plots_dir = experiment_dir / 'plots'
    
    # Load metrics data
    epochs_df = None
    if (logs_dir / 'epochs.csv').exists():
        epochs_df = pd.read_csv(logs_dir / 'epochs.csv')
    
    history = {}
    if (logs_dir / 'history.json').exists():
        with open(logs_dir / 'history.json') as f:
            history = json.load(f)
    
    dataset = config.get('dataset', 'Unknown')
    
    if title is None:
        title = f"Centralized Baseline Report — {dataset}"
    
    content = f"""# {title}

**Experiment ID:** `{experiment_id}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** {author}

---

## 1. Configuration Summary

| Parameter | Value |
|-----------|-------|
| Algorithm | Centralized (Upper Bound) |
| Dataset | {config.get('dataset', 'N/A')} |
| Epochs | {config.get('epochs', 'N/A')} |
| Batch Size | {config.get('batch_size', 'N/A')} |
| Learning Rate | {config.get('lr', 'N/A')} |
| Seed | {config.get('seed', 'N/A')} |
| Train Samples | {config.get('n_train', 'N/A')} |
| Val Samples | {config.get('n_val', 'N/A')} |
| Test Samples | {config.get('n_test', 'N/A')} |

---

## 2. Final Results

| Metric | Value |
|--------|-------|
| **Test RMSE** | **{history.get('test_rmse', 'N/A'):.2f if isinstance(history.get('test_rmse'), (int, float)) else 'N/A'}** |
| **Test MAE** | **{history.get('test_mae', 'N/A'):.2f if isinstance(history.get('test_mae'), (int, float)) else 'N/A'}** |
| Best Val RMSE | {history.get('best_val_rmse', 'N/A'):.2f if isinstance(history.get('best_val_rmse'), (int, float)) else 'N/A'} |
| Best Epoch | {history.get('best_epoch', 'N/A') + 1 if isinstance(history.get('best_epoch'), int) else 'N/A'} |

---

## 3. Training Progress

"""
    
    if epochs_df is not None:
        content += f"""- **Total Epochs:** {len(epochs_df)}
- **Initial Val RMSE:** {epochs_df['val_rmse'].iloc[0]:.2f}
- **Final Val RMSE:** {epochs_df['val_rmse'].iloc[-1]:.2f}
- **Best Val RMSE:** {epochs_df['val_rmse'].min():.2f}

"""
    
    # Add plots
    content += "### Plots\n\n"
    plot_files = list(plots_dir.glob('*.png')) if plots_dir.exists() else []
    if plot_files:
        for plot in sorted(plot_files):
            rel_path = f"../plots/{plot.name}"
            content += f"![{plot.stem}]({rel_path})\n\n"
    else:
        content += "*No plots generated yet.*\n\n"
    
    content += f"""---

## 4. Files & Artifacts

### Logs
- [epochs.csv](../logs/epochs.csv) — Epoch-wise metrics
- [history.json](../logs/history.json) — Full training history

### Model
- [best_model.pt](../best_model.pt) — Best model checkpoint (by validation RMSE)

### Configuration
- [config.json](../config.json) — Full experiment configuration

---

*Report generated automatically by FL Experiment Runner*
"""
    
    return content


def _generate_local_report(
    experiment_dir: Path,
    config: Dict,
    title: Optional[str],
    author: str,
    experiment_id: str
) -> str:
    """Generate report for local-only baseline experiment."""
    
    logs_dir = experiment_dir / 'logs'
    plots_dir = experiment_dir / 'plots'
    
    # Load metrics data
    clients_df = None
    if (logs_dir / 'clients.csv').exists():
        clients_df = pd.read_csv(logs_dir / 'clients.csv')
    
    summary = {}
    if (logs_dir / 'summary.json').exists():
        with open(logs_dir / 'summary.json') as f:
            summary = json.load(f)
    
    dataset = config.get('dataset', 'Unknown')
    
    if title is None:
        title = f"Local-Only Baseline Report — {dataset}"
    
    content = f"""# {title}

**Experiment ID:** `{experiment_id}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** {author}

---

## 1. Configuration Summary

| Parameter | Value |
|-----------|-------|
| Algorithm | Local-Only (Lower Bound) |
| Dataset | {config.get('dataset', 'N/A')} |
| Epochs per Client | {config.get('epochs', 'N/A')} |
| Batch Size | {config.get('batch_size', 'N/A')} |
| Learning Rate | {config.get('lr', 'N/A')} |
| Seed | {config.get('seed', 'N/A')} |
| Num Clients | {config.get('n_clients', 'N/A')} |

---

## 2. Aggregated Results

| Metric | Value |
|--------|-------|
| **Weighted Test RMSE** | **{summary.get('weighted_test_rmse', 'N/A'):.2f if isinstance(summary.get('weighted_test_rmse'), (int, float)) else 'N/A'}** |
| **Weighted Test MAE** | **{summary.get('weighted_test_mae', 'N/A'):.2f if isinstance(summary.get('weighted_test_mae'), (int, float)) else 'N/A'}** |
| Avg Test RMSE | {summary.get('avg_test_rmse', 'N/A'):.2f if isinstance(summary.get('avg_test_rmse'), (int, float)) else 'N/A'} |
| Avg Test MAE | {summary.get('avg_test_mae', 'N/A'):.2f if isinstance(summary.get('avg_test_mae'), (int, float)) else 'N/A'} |
| Total Time | {summary.get('total_time', 0):.1f}s |

---

## 3. Per-Client Performance

"""
    
    if clients_df is not None:
        content += f"""| Statistic | Test RMSE | Test MAE |
|-----------|-----------|----------|
| Mean | {clients_df['test_rmse'].mean():.2f} | {clients_df['test_mae'].mean():.2f} |
| Std | {clients_df['test_rmse'].std():.2f} | {clients_df['test_mae'].std():.2f} |
| Min | {clients_df['test_rmse'].min():.2f} | {clients_df['test_mae'].min():.2f} |
| Max | {clients_df['test_rmse'].max():.2f} | {clients_df['test_mae'].max():.2f} |

"""
    
    # Add plots
    content += "### Plots\n\n"
    plot_files = list(plots_dir.glob('*.png')) if plots_dir.exists() else []
    if plot_files:
        for plot in sorted(plot_files):
            rel_path = f"../plots/{plot.name}"
            content += f"![{plot.stem}]({rel_path})\n\n"
    else:
        content += "*No plots generated yet.*\n\n"
    
    content += f"""---

## 4. Files & Artifacts

### Logs
- [clients.csv](../logs/clients.csv) — Per-client metrics
- [summary.json](../logs/summary.json) — Aggregated summary

### Configuration
- [config.json](../config.json) — Full experiment configuration

---

*Report generated automatically by FL Experiment Runner*
"""
    
    return content


def generate_comparison_report(
    experiment_dirs: Dict[str, Union[str, Path]],
    output_dir: Union[str, Path],
    title: str = "FL Algorithm Comparison Report",
    author: str = "FL Experiment Runner"
) -> str:
    """
    Generate a comparison report across multiple experiments.
    
    Args:
        experiment_dirs: Dict mapping labels to experiment paths
        output_dir: Directory to save the comparison report
        title: Report title
        author: Author name
        
    Returns:
        Path to generated report file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all experiments
    results = []
    for label, exp_dir in experiment_dirs.items():
        exp_dir = Path(exp_dir)
        
        try:
            # Load config
            config = {}
            if (exp_dir / 'config.json').exists():
                with open(exp_dir / 'config.json') as f:
                    config = json.load(f)
            
            # Load summary or final metrics
            summary_path = exp_dir / 'logs' / 'summary.json'
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                rmse = summary.get('test_rmse', summary.get('weighted_test_rmse', None))
                mae = summary.get('test_mae', summary.get('weighted_test_mae', None))
            else:
                # Fallback
                rmse, mae = None, None
            
            results.append({
                'label': label,
                'algorithm': config.get('algorithm', 'unknown'),
                'dataset': config.get('dataset', 'unknown'),
                'rmse': rmse,
                'mae': mae,
                'rounds': config.get('rounds', config.get('epochs', 'N/A')),
                'path': str(exp_dir)
            })
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")
    
    if not results:
        raise ValueError("No valid experiments found")
    
    # Build comparison report
    content = f"""# {title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Author:** {author}

---

## Results Summary

| Experiment | Algorithm | Dataset | Test RMSE | Test MAE | Rounds/Epochs |
|------------|-----------|---------|-----------|----------|---------------|
"""
    
    for r in results:
        rmse_str = f"{r['rmse']:.2f}" if r['rmse'] else "N/A"
        mae_str = f"{r['mae']:.2f}" if r['mae'] else "N/A"
        content += f"| {r['label']} | {r['algorithm']} | {r['dataset']} | {rmse_str} | {mae_str} | {r['rounds']} |\n"
    
    content += """
---

## Analysis

"""
    
    # Find best performer
    valid_results = [r for r in results if r['rmse'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['rmse'])
        content += f"**Best Performer (by RMSE):** {best['label']} with RMSE = {best['rmse']:.2f}\n\n"
    
    content += """---

## Experiment Links

"""
    for r in results:
        content += f"- [{r['label']}]({r['path']}/report/index.md)\n"
    
    content += """
---

*Comparison report generated automatically by FL Experiment Runner*
"""
    
    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.write(content)
    
    print(f"Comparison report generated: {report_path}")
    return str(report_path)
