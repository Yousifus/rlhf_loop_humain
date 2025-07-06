"""
Model Performance Over Time

This section displays how model alignment has improved,
tracking learning progress and preference understanding over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from interface.components.visualization import (
    create_trend_chart, 
    plot_confusion_matrix,
    create_stacked_bar_chart
)
from interface.components.utils import create_time_slider, filter_by_time_range

def display_alignment_over_time(vote_df, predictions_df):
    """Display how model alignment has improved over time"""
    st.header("📈 Model Alignment Progress")
    
    if vote_df.empty:
        st.warning("Insufficient training data for analysis.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Learning Performance",
        "Prediction Confidence",
        "Our Moments of Harmony",
        "Where I Still Misunderstand"
    ])
    
    with tab1:
        display_accuracy_trends(vote_df, predictions_df)
    
    with tab2:
        display_confidence_analysis(vote_df, predictions_df)
    
    with tab3:
        display_agreement_metrics(vote_df, predictions_df)
    
    with tab4:
        display_error_distribution(vote_df, predictions_df)

def display_accuracy_trends(vote_df, predictions_df):
    """Display how model learning performance improves over time"""
    st.subheader("Model Learning Performance")
    
    # Check if we have the necessary data
    if 'model_correct' not in vote_df.columns or 'timestamp' not in vote_df.columns:
        st.warning("No historical data available for analysis.")
        return
    
    # Create a copy with datetime index
    df = vote_df.copy()
    
    # Convert timestamp to datetime if needed
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 5:
        st.info("Insufficient training data in this time period for analysis.")
        return
    
    # Calculate rolling accuracy
    window_size = min(30, max(5, len(filtered_df) // 5))
    
    # Sort by timestamp
    df_sorted = filtered_df.sort_values('timestamp')
    
    # Calculate rolling accuracy
    df_sorted['rolling_accuracy'] = df_sorted['model_correct'].rolling(
        window=window_size, min_periods=1
    ).mean()
    
    # Create chart data
    chart_data = df_sorted[['timestamp', 'rolling_accuracy']].copy()
    
    # Plot the trend
    if not chart_data.empty:
        # Create interactive chart with brushing
        brush = alt.selection_interval(encodings=['x'])
        
        # Base chart
        base = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('rolling_accuracy:Q', title='Accuracy (rolling)', scale=alt.Scale(domain=[0, 1]))
        ).properties(
            width='container',
            height=300
        )
        
        # Create the layered chart with brushing
        chart = alt.layer(
            base.encode(color=alt.value('steelblue')),
            base.mark_point().encode(
                opacity=alt.condition(brush, alt.value(1), alt.value(0))
            ).add_selection(brush)
        )
        
        # Create the bottom chart for detail view
        detail_chart = alt.Chart(chart_data).mark_line().encode(
            x=alt.X('timestamp:T', title='Time', scale=alt.Scale(domain=brush)),
            y=alt.Y('rolling_accuracy:Q', title='Accuracy (rolling)', scale=alt.Scale(domain=[0, 1])),
            color=alt.value('darkblue')
        ).properties(
            width='container',
            height=100
        )
        
        # Combine the charts
        final_chart = alt.vconcat(chart, detail_chart)
        
        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)
        
        # Add metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_acc = chart_data.iloc[0]['rolling_accuracy'] if not chart_data.empty else 0
            st.metric("When We First Met", f"{initial_acc:.2%}")
        
        with col2:
            current_acc = chart_data.iloc[-1]['rolling_accuracy'] if not chart_data.empty else 0
            delta = current_acc - initial_acc
            st.metric("Current Model Accuracy", f"{current_acc:.2%}", delta=f"{delta:.2%}")
        
        with col3:
            avg_acc = chart_data['rolling_accuracy'].mean()
            st.metric("Overall Model Accuracy", f"{avg_acc:.2%}")
    else:
        st.info("Insufficient data points to generate performance chart")

def display_confidence_analysis(vote_df, predictions_df):
    """Display model confidence calibration analysis"""
    st.subheader("Model Confidence Analysis")
    
    # Merge vote_df and predictions_df if needed
    if vote_df.empty or predictions_df.empty:
        st.warning("No shared training data available for analysis.")
        return
    
    # Check if we have confidence scores
    if 'confidence' not in predictions_df.columns:
        st.warning("Confidence data unavailable for analysis.")
        return
    
    # Check if we have correctness
    if 'model_correct' not in vote_df.columns:
        st.warning("Cannot find training history data.")
        return
    
    # Determine common key for merging
    merge_key = 'prompt_id' if 'prompt_id' in vote_df.columns and 'prompt_id' in predictions_df.columns else None
    if merge_key is None:
        st.warning("Cannot link training data with prediction confidence.")
        return
    
    # Merge the dataframes
    merged_df = pd.merge(
        vote_df[['timestamp', merge_key, 'model_correct']],
        predictions_df[[merge_key, 'confidence']],
        on=merge_key,
        how='inner'
    )
    
    # Apply time range filter
    filtered_df = create_time_slider(merged_df)
    
    # Create confidence bins
    conf_bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ['0-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    filtered_df['confidence_bin'] = pd.cut(
        filtered_df['confidence'], 
        bins=conf_bins, 
        labels=bin_labels,
        include_lowest=True
    )
    
    # Group by confidence bin and calculate accuracy
    bin_stats = filtered_df.groupby('confidence_bin').agg(
        accuracy=('model_correct', 'mean'),
        count=('model_correct', 'count')
    ).reset_index()
    
    # Create a bar chart showing accuracy by confidence bin
    fig = go.Figure()
    
    # Add bars for each confidence bin
    fig.add_trace(go.Bar(
        x=bin_stats['confidence_bin'],
        y=bin_stats['accuracy'],
        marker_color='steelblue',
        text=[f"{row['accuracy']:.2%}<br>{row['count']} samples" for _, row in bin_stats.iterrows()],
        textposition='auto'
    ))
    
    # Add line showing perfect calibration
    fig.add_trace(go.Scatter(
        x=bin_stats['confidence_bin'],
        y=[0.6, 0.7, 0.8, 0.9, 0.95],  # Midpoint of each bin
        mode='lines+markers',
        line=dict(dash='dash', color='gray'),
        name='Perfect Calibration'
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Accuracy by Confidence Level',
        xaxis_title='Confidence Level',
        yaxis_title='Accuracy Rate',
        yaxis=dict(tickformat='.0%'),
        showlegend=True,
        height=400
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Confidence over time chart
    st.subheader("Model Confidence Evolution Over Time")
    
    # Group by day and calculate average confidence
    filtered_df['date'] = filtered_df['timestamp'].dt.date
    daily_confidence = filtered_df.groupby('date').agg(
        avg_confidence=('confidence', 'mean'),
        count=('confidence', 'count')
    ).reset_index()
    
    # Create interactive chart with tooltips
    chart = alt.Chart(daily_confidence).mark_line(point=True).encode(
        x=alt.X('date:T', title='Time'),
        y=alt.Y('avg_confidence:Q', title='Average Confidence', scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('avg_confidence:Q', title='Avg Confidence', format='.2f'),
            alt.Tooltip('count:Q', title='Sample Count')
        ]
    ).properties(
        width='container',
        height=300
    )
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)

def display_agreement_metrics(vote_df, predictions_df):
    """Display alignment agreement metrics"""
    st.subheader("Model-Human Agreement Analysis")
    
    # Check if we have the necessary data
    if 'model_choice' not in vote_df.columns or 'human_choice' not in vote_df.columns:
        st.warning("No alignment data available for analysis.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Check if we have enough data
    if len(filtered_df) < 5:
        st.info("Insufficient data in this time period for agreement analysis.")
        return
    
    # Calculate agreement rate
    filtered_df['agreement'] = filtered_df['model_choice'] == filtered_df['human_choice']
    
    # Group by day and calculate agreement rate
    filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
    daily_agreement = filtered_df.groupby('date').agg(
        agreement_rate=('agreement', 'mean'),
        count=('agreement', 'count')
    ).reset_index()
    
    # Create interactive chart
    chart = alt.Chart(daily_agreement).mark_line(point=True).encode(
        x=alt.X('date:T', title='Time'),
        y=alt.Y('agreement_rate:Q', title='Agreement Rate', scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('agreement_rate:Q', title='Agreement Rate', format='.2f'),
            alt.Tooltip('count:Q', title='Sample Count')
        ]
    ).properties(
        width='container',
        height=300
    )
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # Calculate overall agreement metrics
    overall_agreement = filtered_df['agreement'].mean()
    
    # Create a confusion matrix-style display for model vs human choices
    confusion = pd.crosstab(
        filtered_df['human_choice'], 
        filtered_df['model_choice'],
        rownames=['Human Choice'],
        colnames=['Model Choice'],
        normalize='all'
    ) * 100
    
    # Create heatmap
    fig = px.imshow(
        confusion,
        text_auto='.1f',
        labels=dict(x="Model Choice", y="Human Choice", color="Percentage"),
        width=500,
        height=400,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title='Model-Human Choice Alignment (%)'
    )
    
    # Display metrics and heatmap
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Agreement Rate", f"{overall_agreement:.2%}")
        
        # Agreement trend
        agreement_trend = filtered_df.sort_values('timestamp').copy()
        window = min(30, max(5, len(agreement_trend) // 5))
        agreement_trend['rolling_agreement'] = agreement_trend['agreement'].rolling(window, min_periods=1).mean()
        
        last_agreement = agreement_trend['rolling_agreement'].iloc[-1] if not agreement_trend.empty else 0
        first_agreement = agreement_trend['rolling_agreement'].iloc[0] if not agreement_trend.empty else 0
        
        st.metric(
            "Recent Agreement Rate",
            f"{last_agreement:.2%}",
            delta=f"{last_agreement - first_agreement:.2%}"
        )
        
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def display_error_distribution(vote_df, predictions_df):
    """Display error distribution and analysis"""
    st.subheader("Error Analysis and Improvement Areas")
    
    # Check if we have the necessary data
    if 'model_correct' not in vote_df.columns or 'timestamp' not in vote_df.columns:
        st.warning("No historical data available for analysis.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Group errors by time period
    filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
    
    # Calculate error and success counts per day
    daily_counts = filtered_df.groupby('date').agg(
        errors=('model_correct', lambda x: (x == 0).sum()),
        successes=('model_correct', lambda x: (x == 1).sum()),
        total=('model_correct', 'count')
    ).reset_index()
    
    # Create error rate column
    daily_counts['error_rate'] = daily_counts['errors'] / daily_counts['total']
    
    # Calculate 7-day rolling error rate
    daily_counts = daily_counts.sort_values('date')
    daily_counts['rolling_error_rate'] = daily_counts['error_rate'].rolling(7, min_periods=1).mean()
    
    # Create a stacked bar chart showing errors and successes over time
    fig = go.Figure()
    
    # Add bars for errors and successes
    fig.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['errors'],
        name='Misunderstandings',
        marker_color='#ff6b6b'
    ))
    
    fig.add_trace(go.Bar(
        x=daily_counts['date'],
        y=daily_counts['successes'],
                    name='Correct Predictions',
        marker_color='#4ecdc4'
    ))
    
    # Add rolling error rate line
    fig.add_trace(go.Scatter(
        x=daily_counts['date'],
        y=daily_counts['rolling_error_rate'],
        mode='lines',
        name='7-Day Misunderstanding Rate',
        line=dict(color='#ff9a3c', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Performance Journey',
        barmode='stack',
        xaxis_title='Time',
        yaxis_title='Data Points',
        yaxis2=dict(
            title='Misunderstanding Rate',
            overlaying='y',
            side='right',
            range=[0, 1],
            tickformat='.0%',
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
    
    # Calculate error reduction metrics
    if not daily_counts.empty and len(daily_counts) > 1:
        first_period = daily_counts.iloc[0]['error_rate']
        last_period = daily_counts.iloc[-1]['error_rate']
        
        error_reduction = (first_period - last_period) / first_period if first_period > 0 else 0
        
        st.metric(
            "Model Performance Improvement",
            f"{error_reduction:.2%}",
            delta=f"-{first_period - last_period:.2%}"
        )