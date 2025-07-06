"""
Visualization utilities for the RLHF Attunement Dashboard.

This module contains functions to create visualizations for the dashboard,
particularly focused on calibration and confidence visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import sys
import random
from datetime import datetime, timedelta

def plot_reliability_diagram(df, num_bins=10, pre_calibration=True):
    """
    Plot a reliability diagram showing model calibration.
    
    Args:
        df: DataFrame containing confidence and correctness data
        num_bins: Number of confidence bins to create
        pre_calibration: If True, use raw confidence, otherwise use calibrated
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for calibration plot")
    
    # Select the right confidence column
    conf_col = 'model_prediction_confidence_raw' if pre_calibration else 'model_prediction_confidence_calibrated'
    title_prefix = 'Pre-Calibration' if pre_calibration else 'Post-Calibration'
    
    if conf_col not in df.columns or 'is_prediction_correct' not in df.columns:
        return go.Figure().update_layout(title=f"Missing data for {title_prefix} reliability diagram")
    
    # Create confidence bins
    df_copy = df.copy()
    df_copy['confidence_bin'] = pd.cut(
        df_copy[conf_col],
        bins=num_bins,
        labels=[f"{i/num_bins:.1f}-{(i+1)/num_bins:.1f}" for i in range(num_bins)]
    )
    
    # Calculate accuracy per bin
    bin_stats = df_copy.groupby('confidence_bin', observed=True).agg(
        bin_accuracy=('is_prediction_correct', 'mean'),
        bin_confidence=(conf_col, 'mean'),
        count=('is_prediction_correct', 'count')
    ).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Perfect calibration line (diagonal)
    bin_midpoints = [(i + 0.5) / num_bins for i in range(num_bins)]
    fig.add_trace(
        go.Scatter(
            x=bin_midpoints,
            y=bin_midpoints,
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='green', dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Add bars for bin accuracy
    fig.add_trace(
        go.Bar(
            x=[i/num_bins + 1/(2*num_bins) for i in range(num_bins)],
            y=bin_stats['bin_accuracy'],
            width=1/num_bins,
            name='Accuracy',
            marker_color='rgba(58, 71, 80, 0.6)',
            hovertemplate='Bin: %{x:.1f}<br>Accuracy: %{y:.2f}<br>Count: %{text}<extra></extra>',
            text=bin_stats['count']
        )
    )
    
    # Add scatter for bin confidence
    fig.add_trace(
        go.Scatter(
            x=[i/num_bins + 1/(2*num_bins) for i in range(num_bins)],
            y=bin_stats['bin_confidence'],
            mode='markers+lines',
            name='Avg Confidence',
            marker=dict(size=8, color='rgba(220, 50, 50, 0.7)'),
            line=dict(dash='dot', color='rgba(220, 50, 50, 0.7)')
        )
    )
    
    # Calculate ECE and add as annotation
    ece = np.sum(bin_stats['count'] * np.abs(bin_stats['bin_confidence'] - bin_stats['bin_accuracy'])) / np.sum(bin_stats['count'])
    
    # Update layout
    fig.update_layout(
        title=f"{title_prefix} Reliability Diagram (ECE: {ece:.4f})",
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0, 1], tickformat='.1f'),
        yaxis=dict(range=[0, 1], tickformat='.1f'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        width=600,
        height=500
    )
    
    return fig

def plot_pre_post_calibration_comparison(df, num_bins=10):
    """
    Plot a side-by-side comparison of pre and post calibration reliability diagrams.
    
    Args:
        df: DataFrame containing confidence and correctness data
        num_bins: Number of confidence bins to create
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        return go.Figure().update_layout(title="No data available for calibration comparison")
    
    if 'model_prediction_confidence_raw' not in df.columns or 'model_prediction_confidence_calibrated' not in df.columns:
        return go.Figure().update_layout(title="Missing raw or calibrated confidence data")
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pre-Calibration Reliability", "Post-Calibration Reliability"),
        shared_yaxes=True
    )
    
    # Get pre and post reliability diagrams
    pre_fig = plot_reliability_diagram(df, num_bins=num_bins, pre_calibration=True)
    post_fig = plot_reliability_diagram(df, num_bins=num_bins, pre_calibration=False)
    
    # Add traces to subplots
    for trace in pre_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in post_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Calculate ECE improvement
    if 'model_prediction_confidence_raw' in df.columns and 'model_prediction_confidence_calibrated' in df.columns:
        pre_ece = np.mean(np.abs(df['model_prediction_confidence_raw'] - df['is_prediction_correct'].astype(float)))
        post_ece = np.mean(np.abs(df['model_prediction_confidence_calibrated'] - df['is_prediction_correct'].astype(float)))
        ece_improvement = pre_ece - post_ece
        
        # Add annotation
        fig.add_annotation(
            text=f"ECE Improvement: {ece_improvement:.4f}",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update layout
    fig.update_layout(
        title="Pre vs Post Calibration Reliability Comparison",
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis1=dict(range=[0, 1], tickformat='.1f'),
        xaxis2=dict(range=[0, 1], tickformat='.1f'),
        yaxis=dict(range=[0, 1], tickformat='.1f'),
        height=500,
        width=1000,
        showlegend=False,
        hovermode="closest"
    )
    
    return fig

def plot_ece_history(calibration_history):
    """
    Plot ECE (Expected Calibration Error) over time from calibration history.
    
    Args:
        calibration_history: Calibration history dictionary from data_loader
        
    Returns:
        Plotly figure object
    """
    if not calibration_history or 'history' not in calibration_history:
        return go.Figure().update_layout(title="No calibration history available")
    
    history = calibration_history['history']
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    if 'timestamp' not in df.columns or 'calibration_error' not in df.columns:
        return go.Figure().update_layout(title="Calibration history missing required fields")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Ensure timestamp is in datetime format
    if df['timestamp'].dtype != 'datetime64[ns]':
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logger.warning(f"Error converting timestamps: {e}")
            return go.Figure().update_layout(title="Error processing timestamps")
    
    # Convert timestamps to strings for Plotly
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create figure
    fig = go.Figure()
    
    # Add line for calibration error
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_str'],
            y=df['calibration_error'],
            mode='lines+markers',
            name='Calibration Error (ECE)',
            line=dict(width=3, color='rgba(220, 50, 50, 0.7)'),
            marker=dict(size=10)
        )
    )
    
    # Add confidence before/after as area plot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_str'],
            y=df['avg_confidence_before'],
            mode='lines',
            name='Avg Confidence Before',
            line=dict(width=0),
            fill=None,
            fillcolor='rgba(231, 107, 243, 0.2)'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_str'],
            y=df['avg_confidence_after'],
            mode='lines',
            name='Avg Confidence After',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(231, 107, 243, 0.2)'
        )
    )
    
    # Add accuracy line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_str'],
            y=df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(width=3, color='rgba(44, 160, 44, 0.7)'),
            marker=dict(size=10)
        )
    )
    
    # Add text annotations for each calibration run
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['timestamp_str'],
            y=row['calibration_error'],
            text=f"n={row['sample_count']}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30
        )
    
    # Update layout
    fig.update_layout(
        title="Calibration Error (ECE) Over Time",
        xaxis_title="Date",
        yaxis_title="Error / Confidence / Accuracy",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=500
    )
    
    return fig

def plot_confidence_correctness_heatmap(df, theme_column=None):
    """
    Plot a heatmap of confidence vs correctness, optionally split by theme.
    
    Args:
        df: DataFrame containing confidence and correctness data
        theme_column: Optional column name for grouping by theme
        
    Returns:
        Plotly figure object
    """
    if df.empty or 'model_prediction_confidence_raw' not in df.columns or 'is_prediction_correct' not in df.columns:
        return go.Figure().update_layout(title="No data available for confidence-correctness heatmap")
    
    # Create confidence bins
    df_copy = df.copy()
    confidence_bins = 10
    df_copy['confidence_bin'] = pd.cut(
        df_copy['model_prediction_confidence_raw'],
        bins=confidence_bins,
        labels=[f"{i/confidence_bins:.1f}-{(i+1)/confidence_bins:.1f}" for i in range(confidence_bins)]
    )
    
    if theme_column and theme_column in df_copy.columns:
        # Filter out empty themes
        df_copy = df_copy[df_copy[theme_column].notna() & (df_copy[theme_column] != '')]
        
        # Get unique themes
        themes = df_copy[theme_column].unique()
        
        # Create figure with one heatmap per theme
        fig = make_subplots(
            rows=len(themes), 
            cols=1,
            subplot_titles=[f"Theme: {theme}" for theme in themes],
            vertical_spacing=0.1
        )
        
        for i, theme in enumerate(themes):
            theme_df = df_copy[df_copy[theme_column] == theme]
            
            # Create contingency table
            heatmap_data = pd.crosstab(
                theme_df['confidence_bin'], 
                theme_df['is_prediction_correct'],
                normalize='index'
            ).reset_index()
            
            # Add total count per bin
            bin_counts = theme_df['confidence_bin'].value_counts().to_dict()
            heatmap_data['count'] = heatmap_data['confidence_bin'].map(bin_counts)
            
            # If True column doesn't exist (all false), add it
            if True not in heatmap_data.columns:
                heatmap_data[True] = 0
                
            # If False column doesn't exist (all true), add it
            if False not in heatmap_data.columns:
                heatmap_data[False] = 0
            
            # Add heatmap for correct predictions (True)
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data[True],
                    x=['Correct'],
                    y=heatmap_data['confidence_bin'],
                    colorscale='Greens',
                    showscale=i==0,  # Only show colorscale for first heatmap
                    hovertemplate='Confidence: %{y}<br>Correct: %{z:.2f}<br>Count: %{text}<extra></extra>',
                    text=heatmap_data['count']
                ),
                row=i+1, col=1
            )
            
            # Add heatmap for incorrect predictions (False)
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data[False],
                    x=['Incorrect'],
                    y=heatmap_data['confidence_bin'],
                    colorscale='Reds',
                    showscale=i==0,  # Only show colorscale for first heatmap
                    hovertemplate='Confidence: %{y}<br>Incorrect: %{z:.2f}<br>Count: %{text}<extra></extra>',
                    text=heatmap_data['count']
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Confidence × Correctness Heatmap by Theme",
            height=300 * len(themes),
            width=800,
            hovermode="closest"
        )
        
    else:
        # Create contingency table for the whole dataset
        heatmap_data = pd.crosstab(
            df_copy['confidence_bin'], 
            df_copy['is_prediction_correct'],
            normalize='index'
        ).reset_index()
        
        # Add total count per bin
        bin_counts = df_copy['confidence_bin'].value_counts().to_dict()
        heatmap_data['count'] = heatmap_data['confidence_bin'].map(bin_counts)
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap for correct predictions (True)
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data[True],
                x=['Correct'],
                y=heatmap_data['confidence_bin'],
                colorscale='Greens',
                hovertemplate='Confidence: %{y}<br>Correct: %{z:.2f}<br>Count: %{text}<extra></extra>',
                text=heatmap_data['count']
            )
        )
        
        # Add heatmap for incorrect predictions (False)
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data[False],
                x=['Incorrect'],
                y=heatmap_data['confidence_bin'],
                colorscale='Reds',
                hovertemplate='Confidence: %{y}<br>Incorrect: %{z:.2f}<br>Count: %{text}<extra></extra>',
                text=heatmap_data['count']
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Confidence × Correctness Heatmap",
            yaxis_title="Confidence Bin",
            height=600,
            width=800,
            hovermode="closest"
        )
    
    return fig

def plot_model_performance_evolution(df, calibration_history):
    """
    Plot how model's performance (confidence vs accuracy) evolves over time.
    
    Args:
        df: DataFrame with prediction data
        calibration_history: Historical calibration data
    """
    try:
        print("Starting plot_model_performance_evolution function", file=sys.stderr)
        
        if df.empty:
            return go.Figure().update_layout(title="No data available for performance evolution")
        
        # Check required columns
        required_columns = ['timestamp', 'confidence', 'model_correct']
        if not all(col in df.columns for col in required_columns):
            return go.Figure().update_layout(title="Missing required columns for performance evolution")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate rolling performance metrics
        window_size = min(30, len(df) // 5)  # Adaptive window size
        if window_size < 5:
            window_size = 5
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Calculate rolling metrics
        df_sorted['rolling_confidence'] = df_sorted['confidence'].rolling(window=window_size, min_periods=1).mean()
        df_sorted['rolling_accuracy'] = df_sorted['model_correct'].rolling(window=window_size, min_periods=1).mean()
        df_sorted['calibration_gap'] = df_sorted['rolling_confidence'] - df_sorted['rolling_accuracy']
        
        # Create the plot
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_confidence'],
            mode='lines+markers',
            name='Model Confidence',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_accuracy'],
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_confidence'],  # This will be replaced with perfect line
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=1),
            opacity=0.5
        ))
        
        # Update perfect calibration line to be y=confidence
        fig.data[2].y = df_sorted['rolling_confidence']  # This represents perfect calibration
        
        # Update layout
        fig.update_layout(
            title="Model Performance Evolution",
            xaxis_title="Time",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_model_performance_evolution: {str(e)}", file=sys.stderr)
        return go.Figure().update_layout(title=f"Error generating performance evolution plot: {str(e)}")

def plot_basic_performance_evolution(df):
    """
    Simplified version of performance evolution without calibration history.
    
    Args:
        df: DataFrame with prediction data
    """
    try:
        print("Starting plot_basic_performance_evolution function", file=sys.stderr)
        
        if df.empty:
            return go.Figure().update_layout(title="No data available for performance evolution")
        
        # Check required columns
        required_columns = ['timestamp', 'confidence', 'model_correct']
        if not all(col in df.columns for col in required_columns):
            return go.Figure().update_layout(title="Missing required columns for performance evolution")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate rolling performance metrics
        window_size = min(20, len(df) // 4)  # Adaptive window size
        if window_size < 3:
            window_size = 3
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Calculate rolling metrics
        df_sorted['rolling_confidence'] = df_sorted['confidence'].rolling(window=window_size, min_periods=1).mean()
        df_sorted['rolling_accuracy'] = df_sorted['model_correct'].rolling(window=window_size, min_periods=1).mean()
        
        # Create the plot
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_confidence'],
            mode='lines+markers',
            name='Model Confidence',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=6)
        ))
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['rolling_accuracy'],
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='#FF6347', width=3),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title="Model Performance Evolution",
            xaxis_title="Time",
            yaxis_title="Performance Score",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in plot_basic_performance_evolution: {str(e)}", file=sys.stderr)
        return go.Figure().update_layout(title=f"Error generating performance evolution plot: {str(e)}")

def generate_umap_for_drift_clusters(drift_clusters, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Generate UMAP embeddings for drift clusters.
    
    Args:
        drift_clusters: List of drift cluster dictionaries
        n_components: Number of dimensions for UMAP (2 or 3)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with cluster examples and their UMAP coordinates
    """
    try:
        import umap
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"Error importing UMAP dependencies: {e}")
        return None
    
    if not drift_clusters:
        print("No drift clusters provided")
        return None
    
    # Collect all examples from all clusters
    all_examples = []
    for cluster in drift_clusters:
        if 'examples' in cluster and isinstance(cluster['examples'], list):
            for example in cluster['examples']:
                example = example.copy()  # Make a copy to avoid modifying original
                example['cluster_id'] = cluster.get('cluster_id', 'unknown')
                example['cluster_description'] = cluster.get('description', 'No description')
                all_examples.append(example)
    
    if not all_examples:
        print("No examples found in drift clusters")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_examples)
    
    # Check if we have embedding features
    if 'embedding' not in df.columns:
        print("No embedding features found in drift clusters")
        return None
    
    # Extract embeddings into a numpy array
    embeddings = np.array(df['embedding'].tolist())
    
    # Check for NaN values
    if np.isnan(embeddings).any():
        print("Warning: NaN values found in embeddings, replacing with zeros")
        embeddings = np.nan_to_num(embeddings)
    
    # Generate UMAP embedding
    try:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        embedding = reducer.fit_transform(embeddings)
        
        # Add UMAP coordinates to DataFrame
        if n_components == 2:
            df['umap_x'] = embedding[:, 0]
            df['umap_y'] = embedding[:, 1]
        elif n_components == 3:
            df['umap_x'] = embedding[:, 0]
            df['umap_y'] = embedding[:, 1]
            df['umap_z'] = embedding[:, 2]
        
        return df
    except Exception as e:
        print(f"Error generating UMAP: {e}")
        return None

def plot_enhanced_drift_clusters(drift_clusters, reflection_data, use_umap=True, use_3d=False, color_by='cluster_id'):
    """
    Generate an enhanced visualization of drift clusters.
    
    Args:
        drift_clusters: List of drift cluster dictionaries
        reflection_data: DataFrame with reflection data 
        use_umap: Boolean to use UMAP for dimensionality reduction
        use_3d: Boolean to use 3D visualization
        color_by: Field to color points by ('cluster_id', 'confidence', 'is_prediction_correct')
        
    Returns:
        Plotly figure with drift cluster visualization
    """
    if not drift_clusters:
        return go.Figure().update_layout(title="No drift clusters available for visualization")
    
    # Collect all examples from all clusters
    all_examples = []
    for cluster in drift_clusters:
        if 'examples' in cluster and isinstance(cluster['examples'], list):
            for example in cluster['examples']:
                # Add cluster info to example
                example = example.copy()  # Make a copy to avoid modifying original
                example['cluster_id'] = cluster.get('cluster_id', 'unknown')
                example['cluster_description'] = cluster.get('description', 'No description')
                all_examples.append(example)
    
    if not all_examples:
        return go.Figure().update_layout(title="No examples found in drift clusters")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_examples)
    
    # For UMAP visualization
    if use_umap:
        try:
            # Generate UMAP embedding with appropriate dimensions
            umap_df = generate_umap_for_drift_clusters(
                drift_clusters, 
                n_components=3 if use_3d else 2
            )
            
            if umap_df is not None and 'umap_x' in umap_df.columns and 'umap_y' in umap_df.columns:
                df = umap_df
            else:
                use_umap = False
                print("UMAP generation failed, falling back to PCA")
        except Exception as e:
            use_umap = False
            print(f"Error generating UMAP: {e}")
    
    # If not using UMAP or UMAP failed, use simple PCA as fallback
    if not use_umap or 'umap_x' not in df.columns:
        try:
            from sklearn.decomposition import PCA
            
            # Check if we have embedding data
            if 'embedding' in df.columns:
                # Extract embeddings
                embeddings = np.array(df['embedding'].tolist())
                
                # Handle NaN values
                embeddings = np.nan_to_num(embeddings)
                
                # Apply PCA
                pca = PCA(n_components=3 if use_3d else 2)
                pca_result = pca.fit_transform(embeddings)
                
                # Add PCA coordinates
                df['umap_x'] = pca_result[:, 0]  # Using same column names for simplicity
                df['umap_y'] = pca_result[:, 1]
                if use_3d:
                    df['umap_z'] = pca_result[:, 2]
            else:
                print("No embedding data found for PCA")
                use_3d = False  # Fallback to basic display
        except Exception as e:
            print(f"Error with PCA fallback: {e}")
            use_3d = False
    
    # Determine what to color by
    if color_by == 'cluster_id' and 'cluster_id' in df.columns:
        color_col = 'cluster_id'
        color_title = 'Cluster'
    elif color_by == 'confidence' and 'model_prediction_confidence_raw' in df.columns:
        color_col = 'model_prediction_confidence_raw'
        color_title = 'Confidence'
    elif color_by == 'is_prediction_correct' and 'is_prediction_correct' in df.columns:
        color_col = 'is_prediction_correct'
        color_title = 'Correct Prediction'
    else:
        # Default to cluster_id or first available categorical column
        color_col = 'cluster_id' if 'cluster_id' in df.columns else df.columns[0]
        color_title = 'Cluster'
    
    # Create appropriate figure based on dimensions
    if use_3d and 'umap_z' in df.columns:
        fig = px.scatter_3d(
            df,
            x='umap_x',
            y='umap_y',
            z='umap_z',
            color=color_col,
            hover_name='cluster_description',
            hover_data={
                'umap_x': False,
                'umap_y': False,
                'umap_z': False,
                'model_prediction_confidence_raw': ':.2f' if 'model_prediction_confidence_raw' in df.columns else False,
                'is_prediction_correct': True if 'is_prediction_correct' in df.columns else False,
                'cluster_id': True
            },
            title="3D Drift Cluster Visualization",
            labels={
                'umap_x': 'Dimension 1',
                'umap_y': 'Dimension 2',
                'umap_z': 'Dimension 3'
            },
            color_continuous_scale='Viridis' if color_col == 'model_prediction_confidence_raw' else None,
            size_max=10
        )
        
        # Update marker size
        fig.update_traces(marker=dict(size=5))
        
    else:
        # 2D scatter plot
        fig = px.scatter(
            df,
            x='umap_x' if 'umap_x' in df.columns else 'cluster_id',
            y='umap_y' if 'umap_y' in df.columns else 'is_prediction_correct',
            color=color_col,
            hover_name='cluster_description',
            hover_data={
                'umap_x': False,
                'umap_y': False,
                'model_prediction_confidence_raw': ':.2f' if 'model_prediction_confidence_raw' in df.columns else False,
                'is_prediction_correct': True if 'is_prediction_correct' in df.columns else False,
                'cluster_id': True
            },
            title="Drift Cluster Visualization",
            labels={
                'umap_x': 'Dimension 1',
                'umap_y': 'Dimension 2'
            },
            color_continuous_scale='Viridis' if color_col == 'model_prediction_confidence_raw' else None
        )
        
        # Add cluster centroids
        if 'cluster_id' in df.columns:
            centroids = df.groupby('cluster_id', observed=True)[['umap_x', 'umap_y']].mean().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=centroids['umap_x'],
                    y=centroids['umap_y'],
                    mode='markers+text',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='black',
                        line=dict(width=2)
                    ),
                    text=centroids['cluster_id'],
                    textposition="top center",
                    name='Cluster Centers',
                    hoverinfo='text',
                    hovertext=[f"Cluster {c_id} Center" for c_id in centroids['cluster_id']]
                )
            )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(title=color_title),
        height=600 if use_3d else 500,
        legend=dict(title=color_title)
    )
    
    return fig

def plot_cluster_entropy_over_time(clusters):
    """
    Plot cluster entropy over time
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        Plotly figure object
    """
    # Check if we have data
    if not clusters:
        return go.Figure().update_layout(title="No data available for entropy visualization")
    
    # Find clusters with entropy time series
    entropy_data = None
    for cluster in clusters:
        if cluster.get("entropy_time_series"):
            entropy_data = cluster["entropy_time_series"]
            cluster_id = cluster["cluster_id"]
            break
    
    # If no entropy data found in clusters, generate synthetic data
    if not entropy_data:
        # Generate synthetic data for demonstration
        entropy_data = []
        for i in range(30):
            day = datetime.now() - timedelta(days=i)
            entropy_data.append({
                "date": day.strftime("%Y-%m-%d"),
                "entropy": 0.5 + np.sin(i/5) * 0.2 + np.random.normal(0, 0.05)
            })
        cluster_id = "synthetic_data"
    
    # Convert to dataframe
    df = pd.DataFrame(entropy_data)
    
    # Create the figure
    fig = go.Figure()
    
    # Add the entropy line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["entropy"],
        mode='lines+markers',
        name='Cluster Entropy',
        line=dict(color='#FF5733', width=2),
        marker=dict(size=8)
    ))
    
    # Add a horizontal threshold line
    fig.add_hline(
        y=0.7,
        line=dict(color='rgba(255, 0, 0, 0.5)', width=2, dash='dash'),
        annotation_text="High Fragmentation Threshold",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Cluster Entropy Over Time ({cluster_id})",
        xaxis_title="Date",
        yaxis_title="Entropy",
        yaxis_range=[0, 1],
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    
    fig.update_yaxes(
        tickvals=[0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0],
        ticktext=["0 (Uniform)", "0.2", "0.4", "0.6", "0.7", "0.8", "1.0 (Fragmented)"]
    )
    
    return fig

def prepare_cluster_stats_table(clusters):
    """
    Prepare a table of cluster statistics
    
    Args:
        clusters: List of cluster dictionaries
        
    Returns:
        Pandas DataFrame with cluster statistics
    """
    if not clusters:
        return pd.DataFrame()
    
    # Prepare data for table
    table_data = []
    
    for cluster in clusters:
        # Get main error types (top 2)
        error_types = cluster.get('error_types', {})
        top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:2]
        main_error_types = ", ".join([f"{error_type} ({count})" for error_type, count in top_errors]) if top_errors else "N/A"
        
        # Extract metrics and ensure they're strings
        accuracy = cluster.get('accuracy', 0)
        avg_confidence = cluster.get('avg_confidence', 0)
        confidence_accuracy_gap = avg_confidence - accuracy
        
        # Add row for this cluster
        table_data.append({
            'Cluster ID': str(cluster.get('cluster_id', 'Unknown')),
            'Description': str(cluster.get('description', 'No description')),
            'Example Count': str(cluster.get('example_count', 0)),
            'Accuracy': f"{accuracy:.2f}",
            'Avg Confidence': f"{avg_confidence:.2f}",
            'Confidence-Accuracy Gap': f"{confidence_accuracy_gap:.2f}",
            'Main Error Types': main_error_types
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    return df

def plot_drift_alerts_timeline(drift_analysis, reflection_data):
    """
    Plot drift alert markers on a timeline.
    
    Args:
        drift_analysis: Dictionary containing drift analysis data
        reflection_data: DataFrame containing reflection data with timestamps
    
    Returns:
        Plotly figure with drift alerts visualization
    """
    if not drift_analysis or not isinstance(drift_analysis, dict) or reflection_data.empty:
        return go.Figure().update_layout(title="No data available for drift alert visualization")
    
    # Create a copy of the reflection data and ensure timestamp is datetime
    df = reflection_data.copy()
    if 'timestamp' not in df.columns:
        return go.Figure().update_layout(title="Timestamp data missing")
    
    # Convert timestamp to datetime if needed
    if df['timestamp'].dtype != 'datetime64[ns]':
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            return go.Figure().update_layout(title="Error processing timestamps")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate daily accuracy (group by day)
    df['date'] = df['timestamp'].dt.date
    daily_accuracy = df.groupby('date', observed=True)['is_prediction_correct'].mean().reset_index()
    daily_accuracy['timestamp'] = pd.to_datetime(daily_accuracy['date'])
    
    # Extract drift alerts
    drift_alerts = []
    if 'alerts' in drift_analysis:
        for alert in drift_analysis['alerts']:
            if 'timestamp' in alert:
                try:
                    alert_time = pd.to_datetime(alert['timestamp'])
                    drift_alerts.append({
                        'timestamp': alert_time,
                        'severity': alert.get('severity', 'medium'),
                        'description': alert.get('description', 'Drift detected'),
                        'affected_clusters': alert.get('affected_clusters', [])
                    })
                except Exception as e:
                    print(f"Error processing drift alert timestamp: {e}")
    
    # Create figure
    fig = go.Figure()
    
    # Add line plot for daily accuracy
    fig.add_trace(
        go.Scatter(
            x=daily_accuracy['timestamp'].dt.strftime('%Y-%m-%d'),  # Convert to strings
            y=daily_accuracy['is_prediction_correct'],
            mode='lines+markers',
            name='Daily Accuracy',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        )
    )
    
    # Add drift alert markers
    severity_colors = {
        'low': 'rgba(255, 255, 0, 0.3)',  # Yellow with transparency
        'medium': 'rgba(255, 165, 0, 0.4)',  # Orange with transparency
        'high': 'rgba(255, 0, 0, 0.5)'  # Red with transparency
    }
    
    for alert in drift_alerts:
        # Convert timestamps to strings to avoid pandas Timestamp arithmetic operations
        alert_time_str = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Determine span width (1 day by default)
        span_start = alert['timestamp']
        span_end = span_start + pd.Timedelta(days=1)
        span_start_str = span_start.strftime('%Y-%m-%d %H:%M:%S')
        span_end_str = span_end.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add a vertical span for the alert
        color = severity_colors.get(alert['severity'].lower(), 'rgba(255, 165, 0, 0.4)')
        
        # Add colored area for drift alert
        fig.add_shape(
            type="rect",
            x0=span_start_str,
            x1=span_end_str,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=0
        )
        
        # Add annotation for the alert
        fig.add_annotation(
            x=alert_time_str,
            y=1.05,
            text=f"Drift Alert: {alert['severity'].upper()}<br>{alert['description']}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            bgcolor=color.replace('rgba', 'rgb').replace(', 0.3)', ')').replace(', 0.4)', ')').replace(', 0.5)', ')'),
            opacity=0.8
        )
    
    # Update layout
    fig.update_layout(
        title="Model Accuracy with Drift Alerts",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(range=[-0.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=500
    )
    
    return fig

def plot_model_checkpoint_deltas(checkpoint1, checkpoint2):
    """
    Generate delta plots comparing metrics between two model checkpoints.
    
    Args:
        checkpoint1: Base checkpoint dictionary
        checkpoint2: Comparison checkpoint dictionary
        
    Returns:
        Plotly figure with delta visualizations
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    if not checkpoint1 or not checkpoint2:
        return go.Figure().update_layout(title="Missing checkpoint data for comparison")
    
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            "Accuracy Change", 
            "Calibration Error Change", 
            "Confidence Distribution"
        ),
        vertical_spacing=0.12
    )
    
    # 1. Accuracy delta plot
    if 'accuracy' in checkpoint1 and 'accuracy' in checkpoint2:
        acc1 = checkpoint1.get('accuracy', 0)
        acc2 = checkpoint2.get('accuracy', 0)
        
        # Convert to float if they're strings
        try:
            acc1 = float(acc1)
            acc2 = float(acc2)
            
            # Bar chart showing before/after
            fig.add_trace(
                go.Bar(
                    x=['Base', 'Comparison'],
                    y=[acc1, acc2],
                    marker_color=['#2D87BB', '#FFA500'],
                    text=[f"{acc1:.2%}", f"{acc2:.2%}"],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add delta indicator
            delta = acc2 - acc1
            delta_color = 'green' if delta >= 0 else 'red'
            
            fig.add_trace(
                go.Indicator(
                    mode="delta",
                    value=acc2,
                    delta={'reference': acc1, 'relative': True, 'valueformat': '.1%'},
                    domain={'y': [0, 0.1], 'x': [0.75, 1]},
                    title={"text": "Accuracy Change"},
                    number={'valueformat': '.1%'},
                    gauge={'axis': {'range': [None, 1]}}
                ),
                row=1, col=1
            )
        except (ValueError, TypeError):
            # Handle case where accuracy values aren't numeric
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='text',
                    text=['Accuracy data not available in numeric format'],
                ),
                row=1, col=1
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='text',
                text=['Accuracy data not available'],
            ),
            row=1, col=1
        )
    
    # 2. Calibration error (ECE) delta plot
    if 'calibration_error' in checkpoint1 and 'calibration_error' in checkpoint2:
        ece1 = checkpoint1.get('calibration_error', 0)
        ece2 = checkpoint2.get('calibration_error', 0)
        
        # Convert to float if they're strings
        try:
            ece1 = float(ece1)
            ece2 = float(ece2)
            
            # Bar chart showing before/after
            fig.add_trace(
                go.Bar(
                    x=['Base', 'Comparison'],
                    y=[ece1, ece2],
                    marker_color=['#2D87BB', '#FFA500'],
                    text=[f"{ece1:.2%}", f"{ece2:.2%}"],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # Lower ECE is better, so color accordingly
            delta = ece1 - ece2
            delta_color = 'green' if delta >= 0 else 'red'
            
            fig.add_trace(
                go.Indicator(
                    mode="delta",
                    value=ece2,
                    delta={'reference': ece1, 'relative': True, 'valueformat': '.1%', 'decreasing': {'color': 'green'}, 'increasing': {'color': 'red'}},
                    domain={'y': [0, 0.1], 'x': [0.75, 1]},
                    title={"text": "ECE Change (lower is better)"},
                    number={'valueformat': '.1%'},
                ),
                row=2, col=1
            )
        except (ValueError, TypeError):
            # Handle case where values aren't numeric
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='text',
                    text=['Calibration error data not available in numeric format'],
                ),
                row=2, col=1
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0],
                mode='text',
                text=['Calibration error data not available'],
            ),
            row=2, col=1
        )
    
    # 3. Confidence distribution comparison
    if 'confidence_distribution' in checkpoint1 and 'confidence_distribution' in checkpoint2:
        try:
            # Convert distributions to lists if they're not already
            dist1 = checkpoint1['confidence_distribution']
            dist2 = checkpoint2['confidence_distribution']
            
            if isinstance(dist1, str):
                dist1 = eval(dist1)
            if isinstance(dist2, str):
                dist2 = eval(dist2)
            
            # Create histogram bins
            bins = np.linspace(0, 1, 11)  # 0 to 1 in 0.1 increments
            bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
            
            # Plot distributions
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=dist1,
                    name='Base',
                    marker_color='#2D87BB',
                    opacity=0.6
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=dist2,
                    name='Comparison',
                    marker_color='#FFA500',
                    opacity=0.6
                ),
                row=3, col=1
            )
        except (ValueError, TypeError, SyntaxError):
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='text',
                    text=['Confidence distribution data not available in correct format'],
                ),
                row=3, col=1
            )
    else:
        # If we have confidence_avg values, use those to make a simpler comparison
        if 'confidence_avg' in checkpoint1 and 'confidence_avg' in checkpoint2:
            try:
                conf1 = float(checkpoint1.get('confidence_avg', 0))
                conf2 = float(checkpoint2.get('confidence_avg', 0))
                
                fig.add_trace(
                    go.Bar(
                        x=['Base', 'Comparison'],
                        y=[conf1, conf2],
                        marker_color=['#2D87BB', '#FFA500'],
                        text=[f"{conf1:.2%}", f"{conf2:.2%}"],
                        textposition='auto'
                    ),
                    row=3, col=1
                )
                
                # Add delta indicator
                delta = conf2 - conf1
                
                fig.add_trace(
                    go.Indicator(
                        mode="delta",
                        value=conf2,
                        delta={'reference': conf1, 'relative': True, 'valueformat': '.1%'},
                        domain={'y': [0, 0.1], 'x': [0.75, 1]},
                        title={"text": "Avg Confidence Change"},
                        number={'valueformat': '.1%'},
                    ),
                    row=3, col=1
                )
            except (ValueError, TypeError):
                fig.add_trace(
                    go.Scatter(
                        x=[0], y=[0],
                        mode='text',
                        text=['Confidence data not available in numeric format'],
                    ),
                    row=3, col=1
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0], y=[0],
                    mode='text',
                    text=['Confidence distribution data not available'],
                ),
                row=3, col=1
            )
    
    # Update layout for all subplots
    fig.update_layout(
        height=800,
        title=f"Model Checkpoint Comparison: {checkpoint1.get('version', 'Base')} vs {checkpoint2.get('version', 'Comparison')}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update y-axis for percentage formatting
    fig.update_yaxes(tickformat='.1%', row=1, col=1)
    fig.update_yaxes(tickformat='.1%', row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Checkpoint", row=1, col=1)
    fig.update_xaxes(title_text="Checkpoint", row=2, col=1)
    fig.update_xaxes(title_text="Confidence Bin", row=3, col=1)
    
    return fig

def plot_side_by_side_reliability_diagrams(df1, df2, checkpoint1_name="Base", checkpoint2_name="Comparison", num_bins=10):
    """
    Generate side-by-side reliability diagrams for two model checkpoints.
    
    Args:
        df1: DataFrame with reflection data for the first checkpoint
        df2: DataFrame with reflection data for the second checkpoint
        checkpoint1_name: Name of the first checkpoint for display
        checkpoint2_name: Name of the second checkpoint for display
        num_bins: Number of confidence bins for the reliability diagram
        
    Returns:
        Plotly figure with side-by-side reliability diagrams
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    if df1.empty or df2.empty:
        return go.Figure().update_layout(title="No data available for reliability diagrams")
    
    # Create subplot figure with 1 row and 2 columns
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            f"{checkpoint1_name} Reliability Diagram", 
            f"{checkpoint2_name} Reliability Diagram"
        ),
        horizontal_spacing=0.1
    )
    
    # Process df1
    if 'model_prediction_confidence_raw' in df1.columns and 'is_prediction_correct' in df1.columns:
        # Create bins
        df1['confidence_bin'] = pd.cut(
            df1['model_prediction_confidence_raw'],
            bins=np.linspace(0, 1, num_bins + 1),
            labels=[f"{i/num_bins:.1f}-{(i+1)/num_bins:.1f}" for i in range(num_bins)],
            include_lowest=True
        )
        
        # Calculate accuracy per bin
        grouped_df1 = df1.groupby('confidence_bin', observed=True).agg(
            accuracy=('is_prediction_correct', 'mean'),
            count=('is_prediction_correct', 'count')
        ).reset_index()
        
        # Extract bin centers for x-axis
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(num_bins)]
        grouped_df1['bin_center'] = bin_centers
        
        # Add perfect calibration reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect Calibration',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add actual calibration line
        fig.add_trace(
            go.Scatter(
                x=grouped_df1['bin_center'],
                y=grouped_df1['accuracy'],
                mode='lines+markers',
                name=f"{checkpoint1_name} Calibration",
                line=dict(color='blue'),
                marker=dict(
                    size=grouped_df1['count'] / grouped_df1['count'].max() * 15,
                    sizemode='area',
                    sizeref=2.*grouped_df1['count'].max()/(15**2),
                    sizemin=4
                ),
                text=grouped_df1['count'].apply(lambda x: f"n={x}"),
                hovertemplate='Bin Center: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=['Data missing required columns'],
                textposition='middle center'
            ),
            row=1, col=1
        )
    
    # Process df2
    if 'model_prediction_confidence_raw' in df2.columns and 'is_prediction_correct' in df2.columns:
        # Create bins
        df2['confidence_bin'] = pd.cut(
            df2['model_prediction_confidence_raw'],
            bins=np.linspace(0, 1, num_bins + 1),
            labels=[f"{i/num_bins:.1f}-{(i+1)/num_bins:.1f}" for i in range(num_bins)],
            include_lowest=True
        )
        
        # Calculate accuracy per bin
        grouped_df2 = df2.groupby('confidence_bin', observed=True).agg(
            accuracy=('is_prediction_correct', 'mean'),
            count=('is_prediction_correct', 'count')
        ).reset_index()
        
        # Extract bin centers for x-axis
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(num_bins)]
        grouped_df2['bin_center'] = bin_centers
        
        # Add perfect calibration reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Perfect Calibration',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add actual calibration line
        fig.add_trace(
            go.Scatter(
                x=grouped_df2['bin_center'],
                y=grouped_df2['accuracy'],
                mode='lines+markers',
                name=f"{checkpoint2_name} Calibration",
                line=dict(color='orange'),
                marker=dict(
                    size=grouped_df2['count'] / grouped_df2['count'].max() * 15,
                    sizemode='area',
                    sizeref=2.*grouped_df2['count'].max()/(15**2),
                    sizemin=4
                ),
                text=grouped_df2['count'].apply(lambda x: f"n={x}"),
                hovertemplate='Bin Center: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}'
            ),
            row=1, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=['Data missing required columns'],
                textposition='middle center'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Reliability Diagram Comparison",
        height=500,
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis2_title="Confidence",
        yaxis2_title="Accuracy",
        xaxis=dict(range=[0, 1], tickformat='.1f'),
        yaxis=dict(range=[0, 1], tickformat='.1f'),
        xaxis2=dict(range=[0, 1], tickformat='.1f'),
        yaxis2=dict(range=[0, 1], tickformat='.1f'),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_annotation_wordcloud(vote_logs, column_name='prompt', min_word_length=3, max_words=100):
    """
    Generate a wordcloud visualization from annotation text content.
    
    Args:
        vote_logs: DataFrame containing vote logs data
        column_name: Column name to extract text from (prompt, selected_completion, etc.)
        min_word_length: Minimum word length to include
        max_words: Maximum number of words to include in the wordcloud
        
    Returns:
        Base64-encoded image data for the wordcloud to display in Streamlit
    """
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import numpy as np
        import re
        import io
        import base64
        from PIL import Image
        from collections import Counter
    except ImportError as e:
        print(f"Error importing wordcloud dependencies: {e}")
        return None
    
    if vote_logs.empty or column_name not in vote_logs.columns:
        return None
    
    # Combine all text in the specified column
    all_text = " ".join(vote_logs[column_name].fillna("").astype(str))
    
    # Clean text:
    # 1. Convert to lowercase
    # 2. Remove URLs, email addresses
    # 3. Remove non-letters (keep spaces)
    # 4. Remove common stop words
    all_text = all_text.lower()
    all_text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|\S+\.com\S*', '', all_text)
    all_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text)
    
    # Define common stop words to filter out
    stop_words = set([
        'the', 'and', 'a', 'to', 'of', 'in', 'is', 'it', 'that', 'for', 'on', 'with', 'as', 'at', 'by',
        'this', 'be', 'or', 'an', 'are', 'but', 'was', 'not', 'from', 'have', 'has', 'had', 'i', 'you',
        'he', 'she', 'they', 'we', 'what', 'which', 'who', 'when', 'where', 'how', 'why', 'can', 'could',
        'would', 'should', 'will', 'do', 'does', 'did', 'yes', 'no', 'there', 'their', 'these', 'those',
        'than', 'then', 'some', 'such', 'very', 'just', 'more', 'most', 'other', 'so', 'up', 'down',
        'about', 'out', 'over', 'under', 'again', 'too', 'only', 'also', 'any', 'all', 'now', 'if',
        'after', 'before', 'between', 'while', 'through', 'during', 'each', 'few', 'many', 'much'
    ])
    
    # Split text into words and filter
    words = [word for word in all_text.split() if len(word) >= min_word_length and word not in stop_words]
    
    # Count word frequency
    word_counts = Counter(words)
    
    # Create a colormap - use a sequence that works well for text
    colormap = plt.cm.viridis
    
    # Create a mask circle (optional, for rounded wordcloud)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)
    
    # Generate the wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        max_words=max_words,
        background_color='white',
        colormap=colormap,
        contour_width=1,
        contour_color='steelblue',
        mask=mask
    ).generate_from_frequencies(word_counts)
    
    # Create a figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # Encode the image as base64
    img_str = base64.b64encode(buf.read()).decode()
    
    return img_str

def plot_theme_alignment_shifts(vote_logs, theme_column='theme', time_bins='W'):
    """
    Generate a visualization showing shifts in vote alignment per theme over time.
    
    Args:
        vote_logs: DataFrame containing vote logs data
        theme_column: Column name containing theme/category information
        time_bins: Time bin frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    if vote_logs.empty or 'timestamp' not in vote_logs.columns:
        return go.Figure().update_layout(title="No vote log data available")
    
    # Check if we have necessary columns
    has_human_model_data = all(col in vote_logs.columns for col in ['is_model_vote', 'choice'])
    has_theme_data = theme_column in vote_logs.columns
    
    if not has_human_model_data or not has_theme_data:
        return go.Figure().update_layout(
            title="Missing required columns for theme alignment analysis"
        )
    
    # Make a copy to avoid modifying the original
    df = vote_logs.copy()
    
    # Set timestamp index for resampling
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Filter to relevant columns
    df = df[['is_model_vote', 'choice', theme_column]].copy()
    
    # Create time-binned groups
    df['time_bin'] = df.index.to_period(time_bins)
    
    # Process data to find agreement rates per theme and time bin
    results = []
    
    for time_bin, time_group in df.groupby('time_bin', observed=True):
        # Get unique themes in this time period
        themes = time_group[theme_column].unique()
        
        for theme in themes:
            theme_data = time_group[time_group[theme_column] == theme]
            
            # Get human and model votes for this theme
            human_votes = theme_data[theme_data['is_model_vote'] == False]
            model_votes = theme_data[theme_data['is_model_vote'] == True]
            
            # Count votes
            human_count = len(human_votes)
            model_count = len(model_votes)
            
            if human_count == 0 or model_count == 0:
                continue
            
            # Calculate agreement rate for matched pairs
            matched_pairs = []
            
            # This is a simplified matching - in a real implementation you'd want to 
            # match on prompt_id, pair_id etc. to ensure correct pairing
            for _, human_vote in human_votes.iterrows():
                matching_model_votes = model_votes.loc[model_votes.index <= human_vote.name]
                
                if not matching_model_votes.empty:
                    # Get the most recent model vote before this human vote
                    model_vote = matching_model_votes.iloc[-1]
                    matched_pairs.append((human_vote['choice'], model_vote['choice']))
            
            if matched_pairs:
                # Calculate agreement
                agreement_rate = sum(h == m for h, m in matched_pairs) / len(matched_pairs)
                
                # Add to results
                results.append({
                    'time_bin': time_bin.strftime('%Y-%m-%d'),
                    'theme': theme,
                    'agreement_rate': agreement_rate,
                    'sample_count': len(matched_pairs)
                })
    
    if not results:
        return go.Figure().update_layout(
            title="Insufficient data for theme alignment analysis"
        )
    
    # Convert results to DataFrame
    result_df = pd.DataFrame(results)
    
    # Create visualization
    fig = go.Figure()
    
    # Get unique themes
    unique_themes = result_df['theme'].unique()
    
    # Add traces for each theme
    colors = px.colors.qualitative.Plotly
    
    for i, theme in enumerate(unique_themes):
        theme_data = result_df[result_df['theme'] == theme]
        
        # Sort by time bin
        theme_data = theme_data.sort_values('time_bin')
        
        # Add line for this theme
        fig.add_trace(
            go.Scatter(
                x=theme_data['time_bin'],
                y=theme_data['agreement_rate'],
                mode='lines+markers',
                name=theme,
                line=dict(color=colors[i % len(colors)]),
                marker=dict(
                    size=theme_data['sample_count'] / theme_data['sample_count'].max() * 15,
                    sizemode='area',
                    sizeref=2.*theme_data['sample_count'].max()/(15**2),
                    sizemin=4
                ),
                text=theme_data['sample_count'].apply(lambda x: f"n={x}"),
                hovertemplate='Time: %{x}<br>Agreement: %{y:.2f}<br>%{text}'
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Human-AI Agreement by Theme Over Time",
        xaxis_title="Time Period",
        yaxis_title="Agreement Rate",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        hovermode="x unified",
        legend_title="Theme"
    )
    
    return fig 