"""
Dashboard utilities for RLHF Loop system.

This module provides visualization and data processing utilities for the dashboard interface.
"""

from utils.dashboard.data_loader import data_loader, DashboardDataLoader
from utils.dashboard.visualizations import (
    plot_calibration_curve,
    plot_drift_analysis,
    plot_model_performance_evolution,
    plot_basic_performance_evolution,
    plot_prediction_accuracy_over_time,
    plot_confidence_distribution,
    plot_vote_distribution,
    plot_model_evolution,
    plot_training_progress,
    plot_reflection_timeline,
    plot_performance_heatmap,
    plot_feature_importance,
    plot_error_analysis,
    plot_model_comparison,
    plot_calibration_metrics,
    plot_drift_detection,
    plot_performance_trends,
    plot_system_health
)

from .data_loader import (
    load_vote_data,
    load_prediction_data,
    load_reflection_data,
    load_model_metadata,
    get_data_summary,
    refresh_data_cache,
    validate_data_integrity
)

# Import visualization functions
try:
    from utils.dashboard.visualizations import (
        plot_reliability_diagram,
        plot_pre_post_calibration_comparison,
        plot_ece_history,
        plot_confidence_correctness_heatmap,
            plot_model_performance_evolution,
    plot_basic_performance_evolution,
        # Add new drift cluster visualization functions
        plot_enhanced_drift_clusters,
        plot_cluster_entropy_over_time,
        prepare_cluster_stats_table,
        generate_umap_for_drift_clusters
    )
    
    __all__ = [
        'data_loader', 
        'DashboardDataLoader',
        'plot_reliability_diagram',
        'plot_pre_post_calibration_comparison',
        'plot_ece_history',
        'plot_confidence_correctness_heatmap',
            'plot_model_performance_evolution',
    'plot_basic_performance_evolution',
        # Add new drift cluster visualization functions
        'plot_enhanced_drift_clusters',
        'plot_cluster_entropy_over_time',
        'prepare_cluster_stats_table',
        'generate_umap_for_drift_clusters',
        'plot_calibration_curve',
        'plot_drift_analysis',
        'plot_model_performance_evolution',
        'plot_basic_performance_evolution',
        'plot_prediction_accuracy_over_time',
        'plot_confidence_distribution',
        'plot_vote_distribution',
        'plot_model_evolution',
        'plot_training_progress',
        'plot_reflection_timeline',
        'plot_performance_heatmap',
        'plot_feature_importance',
        'plot_error_analysis',
        'plot_model_comparison',
        'plot_calibration_metrics',
        'plot_drift_detection',
        'plot_performance_trends',
        'plot_system_health',
        'load_vote_data',
        'load_prediction_data',
        'load_reflection_data',
        'load_model_metadata',
        'get_data_summary',
        'refresh_data_cache',
        'validate_data_integrity'
    ]
except ImportError:
    # Visualizations module might not be available yet
    __all__ = ['data_loader', 'DashboardDataLoader'] 