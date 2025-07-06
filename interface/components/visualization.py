"""
RLHF Visualization Components

Professional visualization components for the RLHF monitoring system,
providing data visualization and analytics display capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def create_trend_chart(data, x_col, y_col, title=None, color=None, 
                       width=None, height=300, use_container_width=True):
    """
    Create a trend visualization showing model performance over time
    
    Args:
        data: Training data for visualization
        x_col: Time column for x-axis
        y_col: Metric column for y-axis
        title: Chart title
        color: Chart color scheme
        width: Chart width
        height: Chart height
        use_container_width: Whether to use container width
        
    Returns:
        Professional visualization of model training progress
    """
    if data.empty:
        return None  # Insufficient data for visualization
        
    # Create base chart
    chart = alt.Chart(data).mark_line(
        point=True
    ).encode(
        x=alt.X(f'{x_col}:T', title='Time'),
        y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
        tooltip=[
            alt.Tooltip(f'{x_col}:T', title='Time'),
            alt.Tooltip(f'{y_col}:Q', title=y_col.replace('_', ' ').title(), format='.2f')
        ]
    )
    
    if color:
        chart = chart.encode(color=alt.value(color))
    
    # Add chart title if provided
    if title:
        chart = chart.properties(title=title)
    
    # Set dimensions
    chart = chart.properties(height=height)
    if width:
        chart = chart.properties(width=width)
    
    # Display chart
    return st.altair_chart(chart, use_container_width=use_container_width)

def plot_confusion_matrix(df, actual_col, predicted_col, title="Model Performance Analysis"):
    """
    Create a confusion matrix visualization for model performance assessment
    
    Args:
        df: Training data with actual and predicted values
        actual_col: User preference column
        predicted_col: Model prediction column
        title: Chart title
    """
    if df.empty:
        st.warning("Insufficient data for confusion matrix analysis")
        return
        
    # Compute confusion matrix
    # For binary classification
    actual = df[actual_col].astype(int)
    predicted = df[predicted_col].astype(int)
    
    tn = ((actual == 0) & (predicted == 0)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()
    fn = ((actual == 1) & (predicted == 0)).sum()
    tp = ((actual == 1) & (predicted == 1)).sum()
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Blues',
        showscale=False,
        text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        width=400,
        height=400,
        xaxis_title="Model Predictions",
        yaxis_title="Actual Values"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compute metrics from confusion matrix
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
    with col2:
        st.metric("Precision", f"{precision:.2f}")
    with col3:
        st.metric("Recall", f"{recall:.2f}")
    with col4:
        st.metric("F1 Score", f"{f1:.2f}")

def create_stacked_bar_chart(data, x_col, y_cols, colors=None, title=None):
    """
    Create a stacked bar chart for categorical data visualization
    
    Args:
        data: Training data for visualization
        x_col: Category column for x-axis
        y_cols: Value columns for stacking
        colors: Color scheme for different categories
        title: Chart title
    """
    if data.empty:
        st.warning("Insufficient data for stacked bar chart")
        return
        
    # Create figure
    fig = go.Figure()
    
    # Add traces for each category
    for i, col in enumerate(y_cols):
        color = colors[i] if colors and i < len(colors) else None
        fig.add_trace(go.Bar(
            x=data[x_col],
            y=data[col],
            name=col.replace('_', ' ').title(),
            marker_color=color
        ))
    
    # Update layout to stack bars
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title='Count'
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)

def plot_text_clusters(text_data, labels, method='PCA', n_clusters=None):
    """
    Create a visualization of text clustering results
    
    Args:
        text_data: Text data for clustering
        labels: Cluster labels
        method: Dimensionality reduction method ('PCA' or 'UMAP')
        n_clusters: Number of clusters to identify
    """
    if len(text_data) == 0:
        st.warning("Insufficient text data for clustering analysis")
        return
    
    # Prepare text vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=2
    )
    
    # Transform text to TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    # Apply dimensionality reduction
    if method == 'UMAP' and UMAP_AVAILABLE:
        reducer = UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(tfidf_matrix.toarray())
    else:
        # Default to PCA
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(tfidf_matrix.toarray())
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': labels,
        'text': text_data
    })
    
    # Create scatter plot
    fig = px.scatter(
        plot_df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['text'],
        title=f'Text Clustering Results ({method} visualization)'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f"{method} Dimension 1",
        yaxis_title=f"{method} Dimension 2"
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    return plot_df

def plot_confidence_vs_accuracy(df, confidence_col='confidence', accuracy_col='model_correct',
                               title='Model Performance Calibration', bins=10):
    """
    Plot model confidence vs accuracy calibration.
    
    Args:
        df: DataFrame with confidence and accuracy data
        confidence_col: Column name for confidence scores
        accuracy_col: Column name for accuracy values
        title: Plot title
        bins: Number of bins for calibration analysis
    """
    # Create calibration bins
    df_copy = df.copy()
    
    # Create confidence bins
    df_copy['confidence_bin'] = pd.cut(df_copy[confidence_col], bins=bins, include_lowest=True)
    
    # Calculate calibration metrics per bin
    calibration_data = df_copy.groupby('confidence_bin').agg({
        confidence_col: ['mean', 'count'],
        accuracy_col: 'mean'
    }).reset_index()
    
    # Flatten column names
    calibration_data.columns = ['confidence_bin', 'mean_confidence', 'count', 'accuracy']
    
    # Create the plot
    fig = go.Figure()
    
    # Add calibration curve
    fig.add_trace(go.Scatter(
        x=calibration_data['mean_confidence'],
        y=calibration_data['accuracy'],
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2),
        opacity=0.7
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Model Confidence',
        yaxis_title='Actual Accuracy',
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400
    )
    
    return fig 