"""
Model Drift Analysis

This module analyzes patterns in model prediction errors,
performance changes over time, and areas requiring improvement.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from interface.components.utils import create_time_slider, filter_by_time_range

def display_drift_clusters(vote_df, predictions_df):
    """Display model error patterns and drift analysis"""
    st.header("📉 Model Drift Analysis")
    
    # Check if we have the necessary data
    if vote_df.empty:
        st.warning("Insufficient training data for analysis.")
        return
    
    # Check if we have correctness data
    if 'model_correct' not in vote_df.columns:
        st.warning("Cannot find model performance data. Training data is required for analysis.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Error Pattern Analysis",
        "Performance Temporal Analysis",
        "Prediction Pattern Evolution"
    ])
    
    with tab1:
        display_error_clustering(filtered_df)
    
    with tab2:
        display_temporal_drift(filtered_df, predictions_df)
    
    with tab3:
        display_semantic_change(filtered_df, predictions_df)

def display_error_clustering(df):
    """Display patterns in model prediction errors"""
    st.subheader("Model Error Pattern Clustering")
    
    # Check if we have prompt and correctness data
    if 'prompt' not in df.columns or 'model_correct' not in df.columns:
        st.warning("Cannot find conversation data or model performance metrics.")
        return
    
    # Filter for error cases only
    error_df = df[df['model_correct'] == 0].copy()
    
    # Check if we have enough error cases
    if len(error_df) < 5:
        st.info("Insufficient error samples for pattern analysis. At least 5 prediction errors are required.")
        return
    
    # Allow user to select vectorization method
    vectorization_method = st.radio(
        "Select Text Vectorization Method:",
        ["TF-IDF", "Count Vectorization"],
        horizontal=True
    )
    
    # Allow user to select dimensionality reduction technique
    dim_reduction = st.radio(
        "Select Dimensionality Reduction Technique:",
        ["PCA", "UMAP"] if UMAP_AVAILABLE else ["PCA"],
        horizontal=True
    )
    
    # Slider for number of clusters
    min_clusters = min(3, len(error_df) // 2)
    max_clusters = min(10, len(error_df) // 2)
    
    if max_clusters <= min_clusters:
        n_clusters = min_clusters
    else:
        n_clusters = st.slider("Number of Clusters", min_clusters, max_clusters, min(5, max_clusters))
    
    # Get prompts for clustering
    prompts = error_df['prompt'].fillna("").tolist()
    
    # Create vectorizer based on user selection
    if vectorization_method == "TF-IDF":
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            min_df=1
        )
    else:  # Count Vectorization
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            max_features=500,
            stop_words='english',
            min_df=1
        )
    
    # Transform prompts to feature vectors
    try:
        with st.spinner("Analyzing error patterns..."):
            feature_matrix = vectorizer.fit_transform(prompts)
            
            # Get feature names for interpretation
            feature_names = vectorizer.get_feature_names_out()
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Get cluster centers for interpretation
            cluster_centers = kmeans.cluster_centers_
            
            # Add cluster labels to dataframe
            error_df['cluster'] = cluster_labels
            
            # Apply dimensionality reduction for visualization
            if dim_reduction == "UMAP" and UMAP_AVAILABLE:
                reducer = UMAP(n_components=2, random_state=42)
                embedding = reducer.fit_transform(feature_matrix.toarray())
            else:
                # Default to PCA
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(feature_matrix.toarray())
            
            # Create plot dataframe
            plot_df = pd.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'cluster': cluster_labels,
                'prompt': prompts,
                'timestamp': error_df['timestamp'].values
            })
    except Exception as e:
        st.error(f"Error analyzing prediction patterns: {e}")
        return
    
    # Plot the clusters
    fig = px.scatter(
        plot_df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['prompt', 'timestamp'],
        title=f'Error Pattern Clusters ({dim_reduction} visualization)',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f"{dim_reduction} Dimension 1",
        yaxis_title=f"{dim_reduction} Dimension 2",
        height=600
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster details
    st.subheader("Error Pattern Analysis")
    
    # Extract top terms for each cluster
    top_terms_per_cluster = get_top_terms_per_cluster(cluster_centers, feature_names, 10)
    
    # Display cluster information
    for cluster_id in range(n_clusters):
        with st.expander(f"Error Pattern {cluster_id} - {(cluster_labels == cluster_id).sum()} errors"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Top Terms:**")
                for term, weight in top_terms_per_cluster[cluster_id]:
                    st.write(f"- {term} ({weight:.2f})")
            
            with col2:
                st.write("**Sample Prompts:**")
                cluster_samples = error_df[error_df['cluster'] == cluster_id].head(3)
                for _, row in cluster_samples.iterrows():
                    st.info(row['prompt'][:200] + "..." if len(row['prompt']) > 200 else row['prompt'])

def display_temporal_drift(vote_df, predictions_df):
    """Display model performance changes over time"""
    st.subheader("Model Performance Temporal Analysis")
    
    # Check if we have the necessary data
    if 'model_correct' not in vote_df.columns or 'timestamp' not in vote_df.columns:
        st.warning("No historical data available for analysis.")
        return
    
    # Create a copy
    df = vote_df.copy()
    
    # Convert timestamp to datetime
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get time periods for analysis
    time_periods = get_time_periods(df)
    
    if not time_periods:
        st.info("Insufficient historical data for temporal analysis.")
        return
    
    # Create a dataframe for drift metrics
    drift_metrics = []
    
    # Calculate metrics for each time period
    for i, (period_name, period_df) in enumerate(time_periods.items()):
        if len(period_df) < 5:
            continue
            
        error_rate = 1 - period_df['model_correct'].mean()
        
        # Calculate additional metrics if available
        confidence_metric = None
        if predictions_df is not None and 'confidence' in predictions_df.columns:
            # Get common IDs between the period data and predictions
            merge_key = 'prompt_id' if 'prompt_id' in period_df.columns else 'id'
            merged = pd.merge(
                period_df, 
                predictions_df[[merge_key, 'confidence']], 
                on=merge_key, 
                how='inner'
            )
            
            if not merged.empty:
                confidence_metric = merged['confidence'].mean()
        
        drift_metrics.append({
            'period': period_name,
            'error_rate': error_rate,
            'count': len(period_df),
            'confidence': confidence_metric,
            'period_index': i
        })
    
    # Create dataframe from metrics
    metrics_df = pd.DataFrame(drift_metrics)
    
    # Plot drift metrics over time
    if not metrics_df.empty:
        fig = go.Figure()
        
        # Add error rate trace
        fig.add_trace(go.Scatter(
            x=metrics_df['period'],
            y=metrics_df['error_rate'],
            mode='lines+markers',
            name='Error Rate',
            line=dict(width=3, color='#ff6b6b')
        ))
        
        # Add sample count as bar chart
        fig.add_trace(go.Bar(
            x=metrics_df['period'],
            y=metrics_df['count'],
            name='Sample Count',
            opacity=0.3,
            yaxis='y2'
        ))
        
        # Add confidence if available
        if 'confidence' in metrics_df.columns and not metrics_df['confidence'].isna().all():
            fig.add_trace(go.Scatter(
                x=metrics_df['period'],
                y=metrics_df['confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(dash='dash', color='#4ecdc4')
            ))
        
        # Update layout
        fig.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Time Period',
            yaxis=dict(
                title='Error Rate / Confidence',
                range=[0, 1],
                tickformat='.0%'
            ),
            yaxis2=dict(
                title='Sample Count',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            height=500
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate drift metrics
        if len(metrics_df) > 1:
            first_period = metrics_df.iloc[0]['error_rate']
            last_period = metrics_df.iloc[-1]['error_rate']
            
            drift_ratio = last_period / first_period if first_period > 0 else 1
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Performance Change",
                    f"{last_period - first_period:.2%}",
                    delta=f"{last_period - first_period:.2%}",
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Performance Ratio",
                    f"{drift_ratio:.2f}x",
                    help="Performance change ratio between first and last time periods. Values <1 indicate improvement."
                )
    else:
        st.info("Insufficient historical data for temporal analysis.")

def display_semantic_change(vote_df, predictions_df):
    """Display shifts in model error patterns over time"""
    st.subheader("Model Error Pattern Evolution")
    
    # Check if we have prompt and correctness data
    if 'prompt' not in vote_df.columns or 'model_correct' not in vote_df.columns or 'timestamp' not in vote_df.columns:
        st.warning("No historical data available for analysis.")
        return
    
    # Create a copy
    df = vote_df.copy()
    
    # Convert timestamp to datetime
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for error cases
    error_df = df[df['model_correct'] == 0].copy()
    
    # Check if we have enough error cases
    if len(error_df) < 10:
        st.info("Insufficient error cases for pattern analysis. Need at least 10 prediction errors.")
        return
    
    # Get time periods for analysis
    time_periods = get_time_periods(error_df, n_periods=2)
    
    if len(time_periods) < 2:
        st.info("Need prediction errors across at least 2 different time periods for comparative analysis.")
        return
    
    # Get prompts from each period
    period_names = list(time_periods.keys())
    period1_name, period2_name = period_names[0], period_names[-1]
    
    period1_prompts = time_periods[period1_name]['prompt'].fillna("").tolist()
    period2_prompts = time_periods[period2_name]['prompt'].fillna("").tolist()
    
    # Check if we have enough samples in each period
    if len(period1_prompts) < 5 or len(period2_prompts) < 5:
        st.info("Insufficient error samples in each time period for meaningful analysis.")
        return
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=1
    )
    
    try:
        with st.spinner("Analyzing error pattern evolution..."):
            # Vectorize all prompts
            all_prompts = period1_prompts + period2_prompts
            feature_matrix = vectorizer.fit_transform(all_prompts)
            
            # Split back into separate matrices
            period1_matrix = feature_matrix[:len(period1_prompts)]
            period2_matrix = feature_matrix[len(period1_prompts):]
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate term importance for each period
            period1_importance = np.mean(period1_matrix.toarray(), axis=0)
            period2_importance = np.mean(period2_matrix.toarray(), axis=0)
            
            # Calculate term change
            term_change = period2_importance - period1_importance
            
            # Get top changing terms
            increasing_terms = [(feature_names[i], term_change[i]) 
                               for i in np.argsort(term_change)[::-1][:20]]
            
            decreasing_terms = [(feature_names[i], term_change[i]) 
                               for i in np.argsort(term_change)[:20]]
            
            # Create visualization data
            viz_data = []
            
            # Add top increasing terms
            for term, change in increasing_terms[:10]:
                viz_data.append({
                    'term': term,
                    'change': change,
                    'direction': 'Increasing'
                })
            
            # Add top decreasing terms
            for term, change in decreasing_terms[:10]:
                viz_data.append({
                    'term': term,
                    'change': change,
                    'direction': 'Decreasing'
                })
            
            # Create dataframe
            viz_df = pd.DataFrame(viz_data)
    except Exception as e:
        st.error(f"Error analyzing error pattern evolution: {e}")
        return
    
    # Create visualization
    fig = px.bar(
        viz_df,
        x='change',
        y='term',
        color='direction',
        orientation='h',
        title=f'Error Pattern Changes: {period1_name} vs {period2_name}',
        color_discrete_map={'Increasing': '#ff6b6b', 'Decreasing': '#4ecdc4'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Change in Term Importance',
        yaxis_title='Term',
        height=600
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Provide analysis of semantic changes
    st.subheader("Error Pattern Analysis")
    
    # Calculate semantic similarity between periods
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate average vectors for each period
        period1_avg = np.mean(period1_matrix.toarray(), axis=0).reshape(1, -1)
        period2_avg = np.mean(period2_matrix.toarray(), axis=0).reshape(1, -1)
        
        # Calculate similarity
        similarity = cosine_similarity(period1_avg, period2_avg)[0][0]
        
        # Display semantic similarity
        st.metric(
            "Error Pattern Consistency",
            f"{similarity:.2f}",
            help="Similarity of error patterns between time periods. Lower values indicate shifting error patterns."
        )
    except Exception as e:
        st.warning(f"Error measuring pattern consistency: {e}")
    
    # Display sample errors from each period
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Error Samples from {period1_name}:**")
        for prompt in period1_prompts[:3]:
            st.info(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    with col2:
        st.write(f"**Error Samples from {period2_name}:**")
        for prompt in period2_prompts[:3]:
            st.info(prompt[:200] + "..." if len(prompt) > 200 else prompt)

    # Add future enhancements note
    st.markdown("""
    ### Future Enhancements & Notes
    
    - The current clustering uses TF-IDF features from error prompts. For more semantically meaningful clusters, **prompt and completion embeddings** (e.g., from sentence transformers) should be used.
    - **Temporal analysis** of these clusters could reveal how error patterns evolve over time.
    - Interactive **drilldown** into specific examples within each cluster is planned.
    - Adding **Cluster Entropy Over Time** would provide a quantitative measure of error distribution changes.
    - Consider **DBSCAN** or other clustering algorithms that don't require specifying the number of clusters.
    """)

def get_top_terms_per_cluster(cluster_centers, feature_names, n_terms=10):
    """Get the top terms for each cluster based on cluster centers"""
    top_terms = {}
    
    for cluster_idx, center_vector in enumerate(cluster_centers):
        # Get indices of top terms
        sorted_indices = center_vector.argsort()[::-1]
        
        # Get top terms and weights
        top_n_indices = sorted_indices[:n_terms]
        top_n_terms = [(feature_names[i], center_vector[i]) for i in top_n_indices]
        
        top_terms[cluster_idx] = top_n_terms
    
    return top_terms

def get_time_periods(df, n_periods=4):
    """Split the data into time periods for drift analysis"""
    if df.empty or 'timestamp' not in df.columns:
        return {}
    
    # Ensure timestamp is datetime
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get min and max dates
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
    
    # Check if we have enough data span
    date_span = (max_date - min_date).days
    if date_span < 2:
        # If span is too small, split by hours
        min_hour = min_date
        max_hour = max_date
        hour_span = (max_hour - min_hour).total_seconds() / 3600
        
        if hour_span < n_periods:
            return {
                "All Data": df
            }
        
        # Create time periods by hour
        periods = {}
        for i in range(n_periods):
            start_hour = min_hour + timedelta(hours=i * hour_span / n_periods)
            end_hour = min_hour + timedelta(hours=(i + 1) * hour_span / n_periods)
            
            # Create period name
            period_name = f"Period {i+1}"
            
            # Filter data for this period
            period_df = df[(df['timestamp'] >= start_hour) & (df['timestamp'] < end_hour)]
            
            # Add to periods
            if not period_df.empty:
                periods[period_name] = period_df
        
        return periods
    
    # Create time periods by date
    time_delta = (max_date - min_date) / n_periods
    
    periods = {}
    for i in range(n_periods):
        start_date = min_date + i * time_delta
        end_date = min_date + (i + 1) * time_delta
        
        # Create period name
        if date_span > 60:
            # Format as month
            period_name = start_date.strftime('%b %Y')
        else:
            # Format as day
            period_name = start_date.strftime('%b %d')
        
        # Filter data for this period
        period_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]
        
        # Add to periods
        if not period_df.empty:
            periods[period_name] = period_df
    
    return periods 