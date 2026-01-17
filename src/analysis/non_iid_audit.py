"""
Non-IID Audit & Heterogeneity Strength Analysis (RX-0).

This module provides comprehensive tools to quantify and visualize
the non-IID characteristics of federated learning datasets.

Produces:
- Per-client sample counts and statistics
- Skew metrics for RUL distribution (mean, std, Gini, CV)
- Operating condition entropy distribution
- Sensor variance distribution across clients
- Comprehensive plots and summary tables
- Markdown report with heterogeneity assessment

Output: experiments/{id}/non_iid_audit/
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for measuring inequality in distribution.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = maximum inequality)
    """
    if len(values) == 0 or np.all(values == 0):
        return 0.0
    
    values = np.array(values, dtype=float)
    values = values[values >= 0]  # Remove negative values
    
    if len(values) <= 1:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def compute_coefficient_of_variation(values: np.ndarray) -> float:
    """
    Compute coefficient of variation (CV = std/mean).
    
    Args:
        values: Array of values
        
    Returns:
        Coefficient of variation (higher = more variability)
    """
    if len(values) == 0:
        return 0.0
    
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    return float(np.std(values) / mean)


def compute_heterogeneity_metrics(
    non_iid_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute comprehensive heterogeneity metrics from non_iid.csv data.
    
    Args:
        non_iid_df: DataFrame with columns: client_id, num_samples, mean_rul,
                    rul_variance, operating_condition_entropy, sensor_variance
    
    Returns:
        Dict with all heterogeneity metrics
    """
    metrics = {}
    
    # Sample distribution metrics
    samples = non_iid_df['num_samples'].values
    metrics['sample_distribution'] = {
        'client_count': len(samples),
        'total_samples': int(samples.sum()),
        'min': int(samples.min()),
        'max': int(samples.max()),
        'mean': float(samples.mean()),
        'std': float(samples.std()),
        'cv': compute_coefficient_of_variation(samples),
        'gini': compute_gini_coefficient(samples),
        'imbalance_ratio': float(samples.max() / samples.min()) if samples.min() > 0 else float('inf')
    }
    
    # RUL distribution metrics
    mean_ruls = non_iid_df['mean_rul'].values
    rul_variances = non_iid_df['rul_variance'].values
    
    metrics['rul_distribution'] = {
        'mean_rul_range': [float(mean_ruls.min()), float(mean_ruls.max())],
        'mean_rul_avg': float(mean_ruls.mean()),
        'mean_rul_std': float(mean_ruls.std()),
        'mean_rul_cv': compute_coefficient_of_variation(mean_ruls),
        'mean_rul_gini': compute_gini_coefficient(mean_ruls),
        'variance_avg': float(rul_variances.mean()),
        'variance_std': float(rul_variances.std()),
        'variance_cv': compute_coefficient_of_variation(rul_variances)
    }
    
    # Operating condition entropy metrics
    entropies = non_iid_df['operating_condition_entropy'].values
    metrics['operating_conditions'] = {
        'entropy_min': float(entropies.min()),
        'entropy_max': float(entropies.max()),
        'entropy_mean': float(entropies.mean()),
        'entropy_std': float(entropies.std()),
        'entropy_cv': compute_coefficient_of_variation(entropies)
    }
    
    # Sensor variance metrics
    sensor_vars = non_iid_df['sensor_variance'].values
    metrics['sensor_variance'] = {
        'mean': float(sensor_vars.mean()),
        'std': float(sensor_vars.std()),
        'cv': compute_coefficient_of_variation(sensor_vars),
        'min': float(sensor_vars.min()),
        'max': float(sensor_vars.max())
    }
    
    # Overall heterogeneity score (composite metric)
    # Normalized average of key CV metrics
    cv_metrics = [
        metrics['sample_distribution']['cv'],
        metrics['rul_distribution']['mean_rul_cv'],
        metrics['operating_conditions']['entropy_cv'],
        metrics['sensor_variance']['cv']
    ]
    metrics['heterogeneity_score'] = float(np.mean(cv_metrics))
    
    return metrics


def classify_heterogeneity_strength(metrics: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classify the strength of heterogeneity based on computed metrics.
    
    Returns:
        Tuple of (classification, justification)
    """
    score = metrics['heterogeneity_score']
    sample_cv = metrics['sample_distribution']['cv']
    rul_cv = metrics['rul_distribution']['mean_rul_cv']
    
    if score < 0.1 and sample_cv < 0.15 and rul_cv < 0.1:
        classification = "LOW (Near-IID)"
        justification = (
            "Data distribution is relatively uniform across clients. "
            "Standard FedAvg should perform well. "
            "Advanced heterogeneity-handling algorithms may not be necessary."
        )
    elif score < 0.25 or (sample_cv < 0.3 and rul_cv < 0.2):
        classification = "MODERATE"
        justification = (
            "Noticeable but manageable heterogeneity exists. "
            "FedAvg should work but FedProx may improve stability. "
            "Consider partial participation experiments to test robustness."
        )
    elif score < 0.5:
        classification = "HIGH"
        justification = (
            "Significant heterogeneity detected. "
            "FedProx or SCAFFOLD recommended over FedAvg. "
            "Variance reduction techniques will likely improve convergence."
        )
    else:
        classification = "VERY HIGH"
        justification = (
            "Severe non-IID conditions present. "
            "SCAFFOLD or FedDC strongly recommended. "
            "Consider client clustering or personalization approaches."
        )
    
    return classification, justification


def recommend_algorithms(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recommend FL algorithms based on heterogeneity analysis.
    
    Returns:
        List of algorithm recommendations with justifications
    """
    score = metrics['heterogeneity_score']
    sample_cv = metrics['sample_distribution']['cv']
    
    recommendations = []
    
    # FedAvg - always a baseline
    if score < 0.2:
        fedavg_rec = "Recommended"
        fedavg_note = "Low heterogeneity makes FedAvg a good choice"
    else:
        fedavg_rec = "Baseline only"
        fedavg_note = "Use as baseline but expect suboptimal convergence"
    
    recommendations.append({
        'algorithm': 'FedAvg',
        'recommendation': fedavg_rec,
        'note': fedavg_note,
        'priority': 1 if score < 0.2 else 3
    })
    
    # FedProx
    if 0.1 <= score <= 0.4:
        fedprox_rec = "Recommended"
        fedprox_note = f"Proximal term helps with moderate heterogeneity (CV={score:.2f})"
    elif score > 0.4:
        fedprox_rec = "Consider"
        fedprox_note = "May help but SCAFFOLD might be better for high heterogeneity"
    else:
        fedprox_rec = "Optional"
        fedprox_note = "Low heterogeneity - proximal term may not provide significant benefit"
    
    recommendations.append({
        'algorithm': 'FedProx',
        'recommendation': fedprox_rec,
        'note': fedprox_note,
        'priority': 2 if 0.1 <= score <= 0.4 else 3
    })
    
    # SCAFFOLD
    if score > 0.25:
        scaffold_rec = "Strongly Recommended"
        scaffold_note = "Variance reduction beneficial for high heterogeneity"
    else:
        scaffold_rec = "Optional"
        scaffold_note = "Overhead may not be justified for low heterogeneity"
    
    recommendations.append({
        'algorithm': 'SCAFFOLD',
        'recommendation': scaffold_rec,
        'note': scaffold_note,
        'priority': 1 if score > 0.25 else 3
    })
    
    # FedDC
    if score > 0.35 and sample_cv > 0.3:
        feddc_rec = "Recommended"
        feddc_note = "Daisy-chain correction helps with severe label skew"
    else:
        feddc_rec = "Optional"
        feddc_note = "Consider if SCAFFOLD doesn't meet performance targets"
    
    recommendations.append({
        'algorithm': 'FedDC',
        'recommendation': feddc_rec,
        'note': feddc_note,
        'priority': 2 if score > 0.35 else 4
    })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x['priority'])
    
    return recommendations


def plot_sample_histogram(
    non_iid_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """Plot histogram of samples per client."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    samples = non_iid_df['num_samples'].values
    client_ids = non_iid_df['client_id'].values
    
    # Sort by sample count
    sorted_idx = np.argsort(samples)[::-1]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(samples)))
    
    bars = ax.bar(range(len(samples)), samples[sorted_idx], color=colors)
    
    # Add mean line
    mean_samples = samples.mean()
    ax.axhline(y=mean_samples, color='red', linestyle='--', 
               label=f'Mean: {mean_samples:.1f}', linewidth=2)
    
    ax.set_xlabel('Client (sorted by sample count)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Distribution Across Clients', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add statistics annotation
    cv = compute_coefficient_of_variation(samples)
    gini = compute_gini_coefficient(samples)
    ax.text(0.02, 0.98, f'CV: {cv:.3f}\nGini: {gini:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_mean_rul_histogram(
    non_iid_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """Plot histogram of mean RUL per client."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_ruls = non_iid_df['mean_rul'].values
    
    # Sort by mean RUL
    sorted_idx = np.argsort(mean_ruls)[::-1]
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(mean_ruls)))
    
    ax.bar(range(len(mean_ruls)), mean_ruls[sorted_idx], color=colors)
    
    # Add mean line
    overall_mean = mean_ruls.mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--',
               label=f'Mean: {overall_mean:.1f}', linewidth=2)
    
    ax.set_xlabel('Client (sorted by mean RUL)', fontsize=12)
    ax.set_ylabel('Mean RUL', fontsize=12)
    ax.set_title('Mean RUL Distribution Across Clients', fontsize=14)
    ax.legend(loc='upper right')
    
    # Add statistics
    cv = compute_coefficient_of_variation(mean_ruls)
    ax.text(0.02, 0.98, f'CV: {cv:.3f}\nRange: [{mean_ruls.min():.1f}, {mean_ruls.max():.1f}]',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_sensor_variance_boxplot(
    non_iid_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """Plot boxplot of sensor variance across clients."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sensor_vars = non_iid_df['sensor_variance'].values
    
    # Create box plot
    bp = ax.boxplot(sensor_vars, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Overlay individual points
    x = np.random.normal(1, 0.04, len(sensor_vars))
    ax.scatter(x, sensor_vars, alpha=0.6, color='darkblue', s=50, zorder=3)
    
    ax.set_ylabel('Sensor Variance', fontsize=12)
    ax.set_title('Sensor Variance Distribution Across Clients', fontsize=14)
    ax.set_xticks([1])
    ax.set_xticklabels(['All Clients'])
    
    # Add statistics
    stats_text = (f'Mean: {sensor_vars.mean():.3f}\n'
                  f'Std: {sensor_vars.std():.3f}\n'
                  f'Range: [{sensor_vars.min():.3f}, {sensor_vars.max():.3f}]')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_entropy_vs_samples_scatter(
    non_iid_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """Plot scatter of operating condition entropy vs sample count."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    samples = non_iid_df['num_samples'].values
    entropies = non_iid_df['operating_condition_entropy'].values
    client_ids = non_iid_df['client_id'].values
    
    scatter = ax.scatter(samples, entropies, c=non_iid_df['mean_rul'].values,
                         cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean RUL', fontsize=10)
    
    # Add trend line
    z = np.polyfit(samples, entropies, 1)
    p = np.poly1d(z)
    ax.plot(samples, p(samples), "r--", alpha=0.8, label='Trend')
    
    # Compute correlation
    corr = np.corrcoef(samples, entropies)[0, 1]
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Operating Condition Entropy', fontsize=12)
    ax.set_title('Entropy vs Sample Count per Client', fontsize=14)
    ax.legend(loc='upper right')
    
    ax.text(0.02, 0.02, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_heterogeneity_summary(
    metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """Create a summary dashboard of heterogeneity metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: CV comparison bar chart
    ax1 = axes[0, 0]
    cv_names = ['Samples', 'Mean RUL', 'Entropy', 'Sensor Var']
    cv_values = [
        metrics['sample_distribution']['cv'],
        metrics['rul_distribution']['mean_rul_cv'],
        metrics['operating_conditions']['entropy_cv'],
        metrics['sensor_variance']['cv']
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax1.bar(cv_names, cv_values, color=colors, edgecolor='black')
    ax1.axhline(y=0.25, color='red', linestyle='--', label='High threshold')
    ax1.axhline(y=0.1, color='orange', linestyle='--', label='Low threshold')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.set_title('Heterogeneity by Metric (CV)')
    ax1.legend(loc='upper right')
    
    # Add value labels on bars
    for bar, val in zip(bars, cv_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Gini coefficient gauge
    ax2 = axes[0, 1]
    gini_sample = metrics['sample_distribution']['gini']
    gini_rul = metrics['rul_distribution']['mean_rul_gini']
    
    x_pos = [0, 1]
    gini_values = [gini_sample, gini_rul]
    bar_colors = ['#3498db', '#e74c3c']
    bars2 = ax2.bar(x_pos, gini_values, color=bar_colors, edgecolor='black', width=0.6)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Sample Gini', 'RUL Gini'])
    ax2.set_ylabel('Gini Coefficient')
    ax2.set_title('Inequality Metrics (Gini)')
    ax2.set_ylim(0, 1)
    
    for bar, val in zip(bars2, gini_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Sample distribution summary
    ax3 = axes[1, 0]
    sample_stats = metrics['sample_distribution']
    labels = ['Min', 'Mean', 'Max']
    values = [sample_stats['min'], sample_stats['mean'], sample_stats['max']]
    ax3.bar(labels, values, color=['#e74c3c', '#2ecc71', '#3498db'], edgecolor='black')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title(f'Sample Distribution (n={sample_stats["client_count"]} clients)')
    
    for i, v in enumerate(values):
        ax3.text(i, v + max(values)*0.02, f'{v:.0f}', ha='center', fontsize=10)
    
    # Plot 4: Overall assessment text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    classification, justification = classify_heterogeneity_strength(metrics)
    
    text = f"""
HETEROGENEITY ASSESSMENT

Overall Score: {metrics['heterogeneity_score']:.3f}
Classification: {classification}

Key Findings:
• {sample_stats['client_count']} clients with {sample_stats['total_samples']} total samples
• Sample imbalance ratio: {sample_stats['imbalance_ratio']:.1f}x
• RUL range: [{metrics['rul_distribution']['mean_rul_range'][0]:.1f}, {metrics['rul_distribution']['mean_rul_range'][1]:.1f}]

Recommendation:
{justification}
"""
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Non-IID Heterogeneity Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def generate_non_iid_audit_report(
    metrics: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    dataset_name: str = 'Unknown'
) -> str:
    """
    Generate a markdown report summarizing the non-IID audit.
    
    Returns:
        Markdown string
    """
    classification, justification = classify_heterogeneity_strength(metrics)
    
    report = f"""# Non-IID Audit Report — {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

| Question | Answer |
|----------|--------|
| **Is data non-IID?** | {"Yes" if metrics['heterogeneity_score'] > 0.1 else "Marginally"} |
| **How strong is heterogeneity?** | {classification} (Score: {metrics['heterogeneity_score']:.3f}) |
| **Which algorithms are justified?** | {', '.join([r['algorithm'] for r in recommendations if r['priority'] <= 2])} |

---

## 1. Sample Distribution Analysis

| Metric | Value |
|--------|-------|
| Client Count | {metrics['sample_distribution']['client_count']} |
| Total Samples | {metrics['sample_distribution']['total_samples']} |
| Min Samples | {metrics['sample_distribution']['min']} |
| Max Samples | {metrics['sample_distribution']['max']} |
| Mean Samples | {metrics['sample_distribution']['mean']:.1f} |
| Std Samples | {metrics['sample_distribution']['std']:.1f} |
| Coefficient of Variation | {metrics['sample_distribution']['cv']:.3f} |
| Gini Coefficient | {metrics['sample_distribution']['gini']:.3f} |
| Imbalance Ratio | {metrics['sample_distribution']['imbalance_ratio']:.2f}x |

**Interpretation:** {"High sample imbalance detected. Consider weighted aggregation." if metrics['sample_distribution']['cv'] > 0.3 else "Sample distribution is relatively balanced."}

---

## 2. RUL Distribution Analysis

| Metric | Value |
|--------|-------|
| Mean RUL Range | [{metrics['rul_distribution']['mean_rul_range'][0]:.1f}, {metrics['rul_distribution']['mean_rul_range'][1]:.1f}] |
| Mean RUL Average | {metrics['rul_distribution']['mean_rul_avg']:.1f} |
| Mean RUL Std | {metrics['rul_distribution']['mean_rul_std']:.1f} |
| Mean RUL CV | {metrics['rul_distribution']['mean_rul_cv']:.3f} |
| Mean RUL Gini | {metrics['rul_distribution']['mean_rul_gini']:.3f} |
| Variance Average | {metrics['rul_distribution']['variance_avg']:.1f} |

**Interpretation:** {"Significant label distribution skew. SCAFFOLD or FedProx recommended." if metrics['rul_distribution']['mean_rul_cv'] > 0.2 else "Label distribution is relatively uniform."}

---

## 3. Operating Condition Analysis

| Metric | Value |
|--------|-------|
| Entropy Min | {metrics['operating_conditions']['entropy_min']:.3f} |
| Entropy Max | {metrics['operating_conditions']['entropy_max']:.3f} |
| Entropy Mean | {metrics['operating_conditions']['entropy_mean']:.3f} |
| Entropy CV | {metrics['operating_conditions']['entropy_cv']:.3f} |

**Interpretation:** {"Clients operate under diverse conditions. This is typical for C-MAPSS and contributes to feature distribution heterogeneity." if metrics['operating_conditions']['entropy_cv'] > 0.1 else "Operating conditions are similar across clients."}

---

## 4. Sensor Variance Analysis

| Metric | Value |
|--------|-------|
| Mean | {metrics['sensor_variance']['mean']:.4f} |
| Std | {metrics['sensor_variance']['std']:.4f} |
| CV | {metrics['sensor_variance']['cv']:.3f} |
| Range | [{metrics['sensor_variance']['min']:.4f}, {metrics['sensor_variance']['max']:.4f}] |

---

## 5. Algorithm Recommendations

| Algorithm | Recommendation | Priority | Notes |
|-----------|---------------|----------|-------|
"""
    
    for rec in recommendations:
        report += f"| {rec['algorithm']} | {rec['recommendation']} | {rec['priority']} | {rec['note']} |\n"
    
    report += f"""

---

## 6. Conclusion

### Heterogeneity Classification: **{classification}**

{justification}

### Recommended Experimental Protocol

1. **Baseline:** Run FedAvg to establish baseline performance
2. **Primary:** {"Run SCAFFOLD as the primary heterogeneity-aware algorithm" if metrics['heterogeneity_score'] > 0.25 else "Run FedProx for moderate improvement"}
3. **Comparison:** Test across participation rates (100%, 70%, 50%)
4. **Validation:** Use held-out test data for final evaluation

---

## Plots

![Sample Distribution](./samples_histogram.png)

![Mean RUL Distribution](./mean_rul_histogram.png)

![Sensor Variance](./sensor_variance_boxplot.png)

![Entropy vs Samples](./entropy_vs_samples.png)

![Summary Dashboard](./heterogeneity_summary.png)

---

*Report generated by FL-CMAPSS Non-IID Audit Module (RX-0)*
"""
    
    return report


def run_non_iid_audit(
    experiment_dir: str,
    output_subdir: str = 'non_iid_audit',
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Run complete non-IID audit for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory (with logs/non_iid.csv)
        output_subdir: Subdirectory name for audit outputs
        show_plots: Whether to display plots interactively
    
    Returns:
        Dict with metrics, recommendations, and output paths
    """
    experiment_dir = Path(experiment_dir)
    
    # Load non_iid.csv
    non_iid_path = experiment_dir / 'logs' / 'non_iid.csv'
    if not non_iid_path.exists():
        raise FileNotFoundError(f"Non-IID log not found: {non_iid_path}")
    
    non_iid_df = pd.read_csv(non_iid_path)
    
    # Load config for dataset name
    config_path = experiment_dir / 'config.json'
    dataset_name = 'Unknown'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            dataset_name = config.get('dataset', 'Unknown')
    
    # Create output directory
    output_dir = experiment_dir / output_subdir
    output_dir.mkdir(exist_ok=True)
    
    # Compute metrics
    metrics = compute_heterogeneity_metrics(non_iid_df)
    
    # Get recommendations
    recommendations = recommend_algorithms(metrics)
    
    # Generate plots
    plots_generated = []
    
    try:
        plot_sample_histogram(non_iid_df, str(output_dir / 'samples_histogram.png'), show=show_plots)
        plots_generated.append('samples_histogram.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate samples histogram: {e}")
    
    try:
        plot_mean_rul_histogram(non_iid_df, str(output_dir / 'mean_rul_histogram.png'), show=show_plots)
        plots_generated.append('mean_rul_histogram.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate mean RUL histogram: {e}")
    
    try:
        plot_sensor_variance_boxplot(non_iid_df, str(output_dir / 'sensor_variance_boxplot.png'), show=show_plots)
        plots_generated.append('sensor_variance_boxplot.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate sensor variance boxplot: {e}")
    
    try:
        plot_entropy_vs_samples_scatter(non_iid_df, str(output_dir / 'entropy_vs_samples.png'), show=show_plots)
        plots_generated.append('entropy_vs_samples.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate entropy scatter plot: {e}")
    
    try:
        plot_heterogeneity_summary(metrics, str(output_dir / 'heterogeneity_summary.png'), show=show_plots)
        plots_generated.append('heterogeneity_summary.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate summary dashboard: {e}")
    
    # Generate report
    report_md = generate_non_iid_audit_report(metrics, recommendations, dataset_name)
    report_path = output_dir / 'audit_report.md'
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    # Save metrics as JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save recommendations as JSON
    recommendations_path = output_dir / 'recommendations.json'
    with open(recommendations_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Summary table CSV
    summary_df = pd.DataFrame([{
        'client_count': metrics['sample_distribution']['client_count'],
        'min_samples': metrics['sample_distribution']['min'],
        'max_samples': metrics['sample_distribution']['max'],
        'mean_samples': metrics['sample_distribution']['mean'],
        'std_samples': metrics['sample_distribution']['std'],
        'cv_samples': metrics['sample_distribution']['cv'],
        'gini_samples': metrics['sample_distribution']['gini'],
        'heterogeneity_score': metrics['heterogeneity_score']
    }])
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    print(f"\n{'='*60}")
    print("NON-IID AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Clients: {metrics['sample_distribution']['client_count']}")
    print(f"Heterogeneity Score: {metrics['heterogeneity_score']:.3f}")
    classification, _ = classify_heterogeneity_strength(metrics)
    print(f"Classification: {classification}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - audit_report.md")
    print(f"  - metrics.json")
    print(f"  - recommendations.json")
    print(f"  - summary_table.csv")
    for plot in plots_generated:
        print(f"  - {plot}")
    print(f"{'='*60}\n")
    
    return {
        'metrics': metrics,
        'recommendations': recommendations,
        'output_dir': str(output_dir),
        'plots': plots_generated
    }


# Convenience function for running from command line or scripts
def audit_experiment(experiment_id: str, experiments_root: str = None) -> Dict[str, Any]:
    """
    Run non-IID audit on an experiment by ID.
    
    Args:
        experiment_id: Experiment ID
        experiments_root: Root experiments directory (defaults to project experiments/)
    
    Returns:
        Audit results dict
    """
    if experiments_root is None:
        # Get project root
        this_dir = Path(__file__).resolve().parent
        project_root = this_dir.parent.parent
        experiments_root = project_root / 'experiments'
    
    experiment_dir = Path(experiments_root) / experiment_id
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")
    
    return run_non_iid_audit(str(experiment_dir))


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Non-IID Audit on FL Experiment')
    parser.add_argument('experiment_id', help='Experiment ID to audit')
    parser.add_argument('--experiments-root', type=str, default=None,
                        help='Root experiments directory')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    result = audit_experiment(args.experiment_id, args.experiments_root)
    
    print(f"\nAudit complete. Classification: {result['recommendations'][0]['algorithm']} recommended.")
