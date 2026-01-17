"""
Analysis and plotting utilities for FL experiments.
"""
from .plotting import (
    plot_convergence,
    plot_rmse_mae,
    plot_client_variance,
    plot_participation_impact,
    plot_algorithm_comparison,
    plot_final_metrics_bar,
    generate_all_plots
)
from .report import (
    generate_experiment_report,
    generate_comparison_report
)
from .non_iid_audit import (
    run_non_iid_audit,
    audit_experiment,
    compute_heterogeneity_metrics,
    classify_heterogeneity_strength,
    recommend_algorithms,
    compute_gini_coefficient,
    compute_coefficient_of_variation
)

__all__ = [
    'plot_convergence',
    'plot_rmse_mae',
    'plot_client_variance',
    'plot_participation_impact',
    'plot_algorithm_comparison',
    'plot_final_metrics_bar',
    'generate_all_plots',
    'generate_experiment_report',
    'generate_comparison_report',
    # Non-IID audit (RX-0)
    'run_non_iid_audit',
    'audit_experiment',
    'compute_heterogeneity_metrics',
    'classify_heterogeneity_strength',
    'recommend_algorithms',
    'compute_gini_coefficient',
    'compute_coefficient_of_variation'
]
