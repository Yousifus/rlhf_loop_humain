"""
Interactive Analysis Tools for RLHF System

This module implements advanced interactive analysis capabilities including:
- 3D performance evolution surfaces and interactive exploration
- Animated model evolution timelines with transition effects
- Comparative analysis dashboards with cross-model insights
- Interactive calibration exploration and drill-down capabilities
- Advanced visualization components for complex data relationships
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.offline import plot
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings


@dataclass
class VisualizationConfig:
    """Configuration for advanced visualizations"""
    theme: str = "plotly_dark"  # plotly, plotly_white, plotly_dark
    color_scheme: str = "viridis"  # Color palette for heatmaps
    animation_duration: int = 1000  # milliseconds
    show_confidence_bands: bool = True
    enable_3d_interaction: bool = True
    export_format: str = "html"  # html, png, svg, pdf
    width: int = 1200
    height: int = 800


@dataclass
class InteractiveDashboard:
    """Container for interactive dashboard components"""
    dashboard_id: str
    title: str
    components: List[go.Figure]
    layout_config: Dict[str, Any]
    export_path: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ComparisonReport:
    """Multi-model comparison analysis"""
    models_compared: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    performance_differences: Dict[str, float]
    statistical_significance: Dict[str, bool]
    winner_analysis: Dict[str, str]
    recommendation: str


class InteractiveAnalyzer:
    """
    Advanced interactive analysis and visualization system.
    
    Creates publication-ready interactive dashboards with 3D visualizations,
    animated timelines, and comprehensive comparative analysis tools.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize the interactive analyzer.
        
        Args:
            config: Visualization configuration settings
        """
        self.config = config or VisualizationConfig()
        self.dashboards = {}
        self.animation_frames = {}
        
    def create_3d_performance_surface(self, 
                                    performance_data: pd.DataFrame,
                                    x_metric: str = 'data_size',
                                    y_metric: str = 'training_time', 
                                    z_metric: str = 'accuracy') -> go.Figure:
        """
        Create interactive 3D performance evolution surface.
        
        Args:
            performance_data: DataFrame with performance metrics over time
            x_metric: Column name for X-axis metric
            y_metric: Column name for Y-axis metric  
            z_metric: Column name for Z-axis metric (performance)
            
        Returns:
            Interactive 3D surface plot
        """
        # Validate inputs
        required_cols = [x_metric, y_metric, z_metric]
        if not all(col in performance_data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # If we have discrete points, create a surface by interpolation
        if len(performance_data) > 10:
            # Create grid for surface
            x_vals = performance_data[x_metric].values
            y_vals = performance_data[y_metric].values
            z_vals = performance_data[z_metric].values
            
            # Create meshgrid for surface
            x_range = np.linspace(x_vals.min(), x_vals.max(), 20)
            y_range = np.linspace(y_vals.min(), y_vals.max(), 20)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Interpolate Z values
            from scipy.interpolate import griddata
            Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='cubic')
            Z = np.nan_to_num(Z, nan=z_vals.mean())
            
            # Add surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=self.config.color_scheme,
                opacity=0.8,
                name="Performance Surface",
                hovertemplate=f"{x_metric}: %{{x}}<br>{y_metric}: %{{y}}<br>{z_metric}: %{{z}:.3f}}<extra></extra>"
            ))
            
            # Add scatter points for actual data
            fig.add_trace(go.Scatter3d(
                x=x_vals, y=y_vals, z=z_vals,
                mode='markers',
                marker=dict(
                    size=8,
                    color=z_vals,
                    colorscale=self.config.color_scheme,
                    showscale=True
                ),
                name="Actual Data Points",
                hovertemplate=f"{x_metric}: %{{x}}<br>{y_metric}: %{{y}}<br>{z_metric}: %{{z}:.3f}}<extra></extra>"
            ))
        else:
            # Just show scatter plot if too few points
            fig.add_trace(go.Scatter3d(
                x=performance_data[x_metric],
                y=performance_data[y_metric], 
                z=performance_data[z_metric],
                mode='markers+lines',
                marker=dict(size=10, color=performance_data[z_metric], colorscale=self.config.color_scheme),
                line=dict(width=4),
                name="Performance Evolution"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"3D Performance Evolution: {z_metric} vs {x_metric} & {y_metric}",
            scene=dict(
                xaxis_title=x_metric.replace('_', ' ').title(),
                yaxis_title=y_metric.replace('_', ' ').title(),
                zaxis_title=z_metric.replace('_', ' ').title(),
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_animated_evolution_timeline(self, 
                                         historical_data: pd.DataFrame,
                                         metrics: List[str] = None) -> go.Figure:
        """
        Create animated timeline showing model evolution over time.
        
        Args:
            historical_data: DataFrame with timestamp and performance metrics
            metrics: List of metric columns to animate
            
        Returns:
            Animated timeline figure
        """
        if 'timestamp' not in historical_data.columns:
            raise ValueError("Historical data must have 'timestamp' column")
        
        # Default metrics if not specified
        if metrics is None:
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if col not in ['timestamp']][:4]
        
        # Ensure timestamp is datetime
        data = historical_data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        # Create animation frames
        frames = []
        traces = []
        
        for i in range(1, len(data) + 1):
            frame_data = data.iloc[:i]
            frame_traces = []
            
            for j, metric in enumerate(metrics):
                if metric in frame_data.columns:
                    trace = go.Scatter(
                        x=frame_data['timestamp'],
                        y=frame_data[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=3),
                        marker=dict(size=8),
                        yaxis=f'y{j+1}' if j > 0 else 'y'
                    )
                    frame_traces.append(trace)
            
            frames.append(go.Frame(
                data=frame_traces,
                name=str(i)
            ))
        
        # Create initial figure with all metrics
        fig = go.Figure()
        
        # Create subplots for multiple metrics
        subplot_titles = [metric.replace('_', ' ').title() for metric in metrics]
        rows = min(2, len(metrics))
        cols = (len(metrics) + rows - 1) // rows
        
        if len(metrics) > 1:
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
        
        # Add traces for each metric
        for j, metric in enumerate(metrics):
            if metric in data.columns:
                row = (j // cols) + 1
                col = (j % cols) + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'][:1],
                        y=data[metric][:1], 
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=row, col=col
                )
        
        # Add frames
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            title="Animated Model Evolution Timeline",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": self.config.animation_duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"}
                        }],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time Step:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"{k}"],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }
                        ],
                        "label": str(k),
                        "method": "animate"
                    }
                    for k in range(1, len(data) + 1)
                ]
            }]
        )
        
        return fig
    
    def create_comparative_analysis_dashboard(self, 
                                            model_data: Dict[str, pd.DataFrame],
                                            comparison_metrics: List[str] = None) -> InteractiveDashboard:
        """
        Create comprehensive comparative analysis dashboard.
        
        Args:
            model_data: Dictionary mapping model names to their performance data
            comparison_metrics: List of metrics to compare across models
            
        Returns:
            Interactive dashboard with multiple comparison views
        """
        if not model_data:
            raise ValueError("Must provide data for at least one model")
        
        # Default comparison metrics
        if comparison_metrics is None:
            all_cols = set()
            for df in model_data.values():
                all_cols.update(df.select_dtypes(include=[np.number]).columns)
            comparison_metrics = list(all_cols)[:6]  # Limit to 6 metrics
        
        components = []
        
        # 1. Performance Radar Chart
        radar_fig = self._create_performance_radar(model_data, comparison_metrics)
        components.append(radar_fig)
        
        # 2. Metric Evolution Comparison
        evolution_fig = self._create_evolution_comparison(model_data, comparison_metrics)
        components.append(evolution_fig)
        
        # 3. Statistical Significance Heatmap
        significance_fig = self._create_significance_heatmap(model_data, comparison_metrics)
        components.append(significance_fig)
        
        # 4. Distribution Comparison
        distribution_fig = self._create_distribution_comparison(model_data, comparison_metrics)
        components.append(distribution_fig)
        
        # Create dashboard layout
        layout_config = {
            "grid_layout": "2x2",
            "shared_xaxis": False,
            "shared_yaxis": False,
            "title": "Comprehensive Model Comparison Dashboard"
        }
        
        dashboard = InteractiveDashboard(
            dashboard_id=f"comparison_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Multi-Model Comparative Analysis",
            components=components,
            layout_config=layout_config
        )
        
        return dashboard
    
    def _create_performance_radar(self, model_data: Dict[str, pd.DataFrame], metrics: List[str]) -> go.Figure:
        """Create radar chart comparing model performance across metrics."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(model_data)]
        
        for i, (model_name, data) in enumerate(model_data.items()):
            # Calculate mean performance for each metric
            mean_values = []
            for metric in metrics:
                if metric in data.columns:
                    mean_val = data[metric].mean()
                    # Normalize to 0-1 scale for radar chart
                    mean_values.append(min(1.0, max(0.0, mean_val)))
                else:
                    mean_values.append(0.0)
            
            fig.add_trace(go.Scatterpolar(
                r=mean_values + [mean_values[0]],  # Close the radar
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Performance Radar Comparison",
            template=self.config.theme,
            showlegend=True
        )
        
        return fig
    
    def _create_evolution_comparison(self, model_data: Dict[str, pd.DataFrame], metrics: List[str]) -> go.Figure:
        """Create evolution comparison chart."""
        fig = make_subplots(
            rows=len(metrics), cols=1,
            shared_xaxes=True,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set1[:len(model_data)]
        
        for row, metric in enumerate(metrics, 1):
            for i, (model_name, data) in enumerate(model_data.items()):
                if metric in data.columns and 'timestamp' in data.columns:
                    # Sort by timestamp
                    sorted_data = data.sort_values('timestamp')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=sorted_data['timestamp'],
                            y=sorted_data[metric],
                            mode='lines+markers',
                            name=f"{model_name}" if row == 1 else None,
                            showlegend=(row == 1),
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=6)
                        ),
                        row=row, col=1
                    )
        
        fig.update_layout(
            title="Model Evolution Comparison Over Time",
            template=self.config.theme,
            height=200 * len(metrics),
            width=self.config.width
        )
        
        return fig
    
    def _create_significance_heatmap(self, model_data: Dict[str, pd.DataFrame], metrics: List[str]) -> go.Figure:
        """Create statistical significance heatmap between models."""
        model_names = list(model_data.keys())
        
        # Calculate p-values between all model pairs
        significance_matrix = np.zeros((len(model_names), len(model_names)))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    # Perform statistical test between models
                    p_values = []
                    for metric in metrics:
                        if (metric in model_data[model1].columns and 
                            metric in model_data[model2].columns):
                            from scipy import stats
                            _, p_val = stats.ttest_ind(
                                model_data[model1][metric].dropna(),
                                model_data[model2][metric].dropna()
                            )
                            p_values.append(p_val)
                    
                    # Average p-value across metrics
                    avg_p_value = np.mean(p_values) if p_values else 1.0
                    significance_matrix[i, j] = 1 - avg_p_value  # Higher = more significant
        
        fig = go.Figure(data=go.Heatmap(
            z=significance_matrix,
            x=model_names,
            y=model_names,
            colorscale='RdYlBu_r',
            text=significance_matrix,
            texttemplate="%{text:.3f}",
            textfont={"size": 12},
            hovertemplate="Model 1: %{y}<br>Model 2: %{x}<br>Significance: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Statistical Significance Heatmap (1 - p-value)",
            template=self.config.theme,
            width=600,
            height=600
        )
        
        return fig
    
    def _create_distribution_comparison(self, model_data: Dict[str, pd.DataFrame], metrics: List[str]) -> go.Figure:
        """Create distribution comparison for key metrics."""
        # Select top 2 metrics for distribution comparison
        top_metrics = metrics[:2]
        
        fig = make_subplots(
            rows=1, cols=len(top_metrics),
            subplot_titles=[metric.replace('_', ' ').title() for metric in top_metrics]
        )
        
        colors = px.colors.qualitative.Pastel[:len(model_data)]
        
        for col, metric in enumerate(top_metrics, 1):
            for i, (model_name, data) in enumerate(model_data.items()):
                if metric in data.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=data[metric].dropna(),
                            name=f"{model_name}" if col == 1 else None,
                            showlegend=(col == 1),
                            opacity=0.7,
                            marker_color=colors[i % len(colors)]
                        ),
                        row=1, col=col
                    )
        
        fig.update_layout(
            title="Performance Distribution Comparison",
            template=self.config.theme,
            barmode='overlay',
            width=self.config.width,
            height=400
        )
        
        return fig
    
    def create_interactive_calibration_explorer(self, 
                                              calibration_data: pd.DataFrame,
                                              confidence_col: str = 'confidence',
                                              accuracy_col: str = 'accuracy') -> go.Figure:
        """
        Create interactive calibration exploration dashboard.
        
        Args:
            calibration_data: DataFrame with confidence and accuracy data
            confidence_col: Column name for confidence values
            accuracy_col: Column name for accuracy values
            
        Returns:
            Interactive calibration exploration figure
        """
        # Validate inputs
        if confidence_col not in calibration_data.columns:
            raise ValueError(f"Confidence column '{confidence_col}' not found")
        if accuracy_col not in calibration_data.columns:
            raise ValueError(f"Accuracy column '{accuracy_col}' not found")
        
        # Create reliability diagram with interactive features
        data = calibration_data.copy()
        
        # Bin the confidence values
        n_bins = 10
        data['confidence_bin'] = pd.cut(data[confidence_col], bins=n_bins, labels=False)
        
        # Calculate bin statistics
        bin_stats = data.groupby('confidence_bin').agg({
            confidence_col: ['mean', 'count'],
            accuracy_col: 'mean'
        }).round(4)
        
        bin_stats.columns = ['avg_confidence', 'count', 'avg_accuracy']
        bin_stats = bin_stats.reset_index()
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Interactive Reliability Diagram",
                "Confidence Distribution",
                "Calibration Error by Bin", 
                "Confidence vs Sample Size"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Reliability diagram
        fig.add_trace(
            go.Scatter(
                x=bin_stats['avg_confidence'],
                y=bin_stats['avg_accuracy'],
                mode='markers+lines',
                name='Observed',
                marker=dict(
                    size=bin_stats['count'] / bin_stats['count'].max() * 20 + 5,
                    color='blue',
                    opacity=0.7
                ),
                line=dict(color='blue', width=2),
                hovertemplate="Confidence: %{x:.3f}<br>Accuracy: %{y:.3f}<br>Samples: %{text}<extra></extra>",
                text=bin_stats['count']
            ),
            row=1, col=1
        )
        
        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=data[confidence_col],
                nbinsx=30,
                name='Confidence Distribution',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Calibration error by bin
        calibration_error = abs(bin_stats['avg_confidence'] - bin_stats['avg_accuracy'])
        fig.add_trace(
            go.Bar(
                x=bin_stats.index,
                y=calibration_error,
                name='Calibration Error',
                marker_color='orange',
                opacity=0.8,
                hovertemplate="Bin: %{x}<br>Error: %{y:.3f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # 4. Confidence vs sample size
        fig.add_trace(
            go.Scatter(
                x=bin_stats['avg_confidence'],
                y=bin_stats['count'],
                mode='markers',
                name='Sample Size',
                marker=dict(
                    size=15,
                    color=calibration_error,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Calibration Error")
                ),
                hovertemplate="Confidence: %{x:.3f}<br>Samples: %{y}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive Calibration Explorer",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Predicted Confidence", row=1, col=1)
        fig.update_yaxes(title_text="Observed Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Confidence", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Confidence Bin", row=2, col=1)
        fig.update_yaxes(title_text="Calibration Error", row=2, col=1)
        fig.update_xaxes(title_text="Average Confidence", row=2, col=2)
        fig.update_yaxes(title_text="Sample Count", row=2, col=2)
        
        return fig
    
    def export_dashboard(self, dashboard: InteractiveDashboard, export_path: str = None) -> str:
        """
        Export interactive dashboard to file.
        
        Args:
            dashboard: Dashboard to export
            export_path: Path to save the dashboard
            
        Returns:
            Path to exported file
        """
        if export_path is None:
            export_dir = Path("exports/dashboards")
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"{dashboard.dashboard_id}.html"
        
        # Combine all components into a single HTML file
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard.title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-title {{ text-align: center; color: #333; margin-bottom: 30px; }}
                .component {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }}
                .grid-2x2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            </style>
        </head>
        <body>
            <h1 class="dashboard-title">{dashboard.title}</h1>
            <div class="dashboard-container">
        """
        
        # Add each component
        for i, component in enumerate(dashboard.components):
            div_id = f"plotly-div-{i}"
            html_content += f'<div id="{div_id}" class="component"></div>\n'
            
            # Get the plotly JSON
            fig_json = component.to_json()
            html_content += f"""
            <script>
                Plotly.newPlot('{div_id}', {fig_json});
            </script>
            """
        
        html_content += """
            </div>
            <footer style="text-align: center; margin-top: 40px; color: #666;">
                Generated by RLHF Interactive Analyzer
            </footer>
        </body>
        </html>
        """
        
        # Write to file
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(export_path)
    
    def create_model_comparison_report(self, 
                                     model_data: Dict[str, pd.DataFrame],
                                     metrics: List[str] = None) -> ComparisonReport:
        """
        Generate comprehensive model comparison report.
        
        Args:
            model_data: Dictionary of model names to performance data
            metrics: List of metrics to compare
            
        Returns:
            Detailed comparison report
        """
        if len(model_data) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Default metrics
        if metrics is None:
            all_cols = set()
            for df in model_data.values():
                all_cols.update(df.select_dtypes(include=[np.number]).columns)
            metrics = list(all_cols)[:5]
        
        model_names = list(model_data.keys())
        comparison_metrics = {}
        performance_differences = {}
        statistical_significance = {}
        
        # Calculate metrics for each model
        for model_name, data in model_data.items():
            comparison_metrics[model_name] = {}
            for metric in metrics:
                if metric in data.columns:
                    comparison_metrics[model_name][metric] = {
                        'mean': float(data[metric].mean()),
                        'std': float(data[metric].std()),
                        'median': float(data[metric].median()),
                        'min': float(data[metric].min()),
                        'max': float(data[metric].max())
                    }
        
        # Calculate pairwise differences and significance
        from scipy import stats
        from itertools import combinations
        
        for model1, model2 in combinations(model_names, 2):
            comparison_key = f"{model1}_vs_{model2}"
            performance_differences[comparison_key] = {}
            statistical_significance[comparison_key] = {}
            
            for metric in metrics:
                if (metric in model_data[model1].columns and 
                    metric in model_data[model2].columns):
                    
                    data1 = model_data[model1][metric].dropna()
                    data2 = model_data[model2][metric].dropna()
                    
                    # Calculate difference
                    mean_diff = data1.mean() - data2.mean()
                    performance_differences[comparison_key][metric] = float(mean_diff)
                    
                    # Statistical significance test
                    if len(data1) > 5 and len(data2) > 5:
                        _, p_value = stats.ttest_ind(data1, data2)
                        statistical_significance[comparison_key][metric] = p_value < 0.05
                    else:
                        statistical_significance[comparison_key][metric] = False
        
        # Winner analysis
        winner_analysis = {}
        for metric in metrics:
            metric_means = {}
            for model_name in model_names:
                if metric in comparison_metrics[model_name]:
                    metric_means[model_name] = comparison_metrics[model_name][metric]['mean']
            
            if metric_means:
                winner = max(metric_means.keys(), key=lambda x: metric_means[x])
                winner_analysis[metric] = winner
        
        # Generate recommendation
        overall_winner = max(
            set(winner_analysis.values()), 
            key=list(winner_analysis.values()).count
        )
        
        recommendation = f"Based on the analysis across {len(metrics)} metrics, " \
                        f"{overall_winner} shows the best overall performance. " \
                        f"It wins in {list(winner_analysis.values()).count(overall_winner)} out of {len(metrics)} metrics."
        
        return ComparisonReport(
            models_compared=model_names,
            comparison_metrics=comparison_metrics,
            performance_differences=performance_differences,
            statistical_significance=statistical_significance,
            winner_analysis=winner_analysis,
            recommendation=recommendation
        )


def create_demo_interactive_analysis() -> Dict[str, Any]:
    """
    Create demonstration of interactive analysis capabilities.
    
    Returns:
        Dictionary with demo results and file paths
    """
    # Create demo data
    np.random.seed(42)
    
    # Model performance data over time
    timestamps = pd.date_range(start='2024-01-01', periods=50, freq='D')
    model_data = {
        'GPT-4': pd.DataFrame({
            'timestamp': timestamps,
            'accuracy': np.random.normal(0.85, 0.05, 50).clip(0.7, 0.95),
            'calibration_error': np.random.normal(0.05, 0.02, 50).clip(0.01, 0.15),
            'confidence': np.random.normal(0.8, 0.1, 50).clip(0.5, 1.0),
            'data_size': np.linspace(1000, 10000, 50),
            'training_time': np.linspace(1, 20, 50)
        }),
        'Claude-3': pd.DataFrame({
            'timestamp': timestamps,
            'accuracy': np.random.normal(0.82, 0.04, 50).clip(0.7, 0.95),
            'calibration_error': np.random.normal(0.06, 0.02, 50).clip(0.01, 0.15),
            'confidence': np.random.normal(0.78, 0.08, 50).clip(0.5, 1.0),
            'data_size': np.linspace(1000, 10000, 50),
            'training_time': np.linspace(1, 20, 50)
        }),
        'Local-Model': pd.DataFrame({
            'timestamp': timestamps,
            'accuracy': np.random.normal(0.78, 0.06, 50).clip(0.7, 0.95),
            'calibration_error': np.random.normal(0.08, 0.03, 50).clip(0.01, 0.15),
            'confidence': np.random.normal(0.75, 0.12, 50).clip(0.5, 1.0),
            'data_size': np.linspace(1000, 10000, 50),
            'training_time': np.linspace(1, 20, 50)
        })
    }
    
    # Initialize analyzer
    analyzer = InteractiveAnalyzer()
    
    results = {}
    
    # 1. Create 3D performance surface
    surface_fig = analyzer.create_3d_performance_surface(
        model_data['GPT-4'], 
        x_metric='data_size',
        y_metric='training_time',
        z_metric='accuracy'
    )
    results['3d_surface'] = surface_fig
    
    # 2. Create animated evolution timeline
    animation_fig = analyzer.create_animated_evolution_timeline(
        model_data['GPT-4'],
        metrics=['accuracy', 'calibration_error', 'confidence']
    )
    results['animation'] = animation_fig
    
    # 3. Create comparative dashboard
    dashboard = analyzer.create_comparative_analysis_dashboard(
        model_data,
        comparison_metrics=['accuracy', 'calibration_error', 'confidence']
    )
    results['dashboard'] = dashboard
    
    # 4. Create calibration explorer
    calibration_data = pd.DataFrame({
        'confidence': np.random.beta(5, 2, 1000),
        'accuracy': np.random.binomial(1, np.random.beta(5, 2, 1000), 1000)
    })
    calibration_fig = analyzer.create_interactive_calibration_explorer(calibration_data)
    results['calibration_explorer'] = calibration_fig
    
    # 5. Generate comparison report
    comparison_report = analyzer.create_model_comparison_report(model_data)
    results['comparison_report'] = comparison_report
    
    # Export dashboard
    export_path = analyzer.export_dashboard(dashboard)
    results['export_path'] = export_path
    
    return results


if __name__ == "__main__":
    print("🎯 Interactive Analysis Tools - Demo Mode")
    print("=" * 50)
    
    # Run demo
    demo_results = create_demo_interactive_analysis()
    
    print(f"✅ Created 3D Performance Surface")
    print(f"✅ Generated Animated Evolution Timeline") 
    print(f"✅ Built Comparative Analysis Dashboard")
    print(f"✅ Designed Interactive Calibration Explorer")
    print(f"✅ Generated Model Comparison Report")
    print(f"✅ Exported Dashboard to: {demo_results['export_path']}")
    
    # Print comparison report summary
    report = demo_results['comparison_report']
    print(f"\n📊 Model Comparison Summary:")
    print(f"Models Compared: {', '.join(report.models_compared)}")
    print(f"Recommendation: {report.recommendation}")
    
    print(f"\n🚀 Interactive Analysis Tools Demo Complete!")
    print(f"Phase 3 Module 1: ✅ READY FOR INTEGRATION") 