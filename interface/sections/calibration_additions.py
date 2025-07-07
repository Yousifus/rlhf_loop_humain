"""
This module contains model calibration and performance analysis functionality:
1. Model Understanding Assessment
2. Preference Mapping Analysis
3. Calibration Metrics Overview
4. Performance Trend Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from interface.components.utils import create_time_slider, filter_by_time_range

def display_confidence_correctness_heatmap(df):
    """Display a heatmap showing model calibration performance"""
    st.subheader("Model Calibration Analysis")
    
    # Check if we have the necessary data
    if 'confidence' not in df.columns or 'model_correct' not in df.columns:
        st.warning("Confidence or model correctness data not available.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.info("Not enough data points for heatmap analysis in the selected time range.")
        return
    
    # Remove rows with NaN values
    filtered_df = filtered_df.dropna(subset=['confidence', 'model_correct'])
    if len(filtered_df) < 10:
        st.warning("Insufficient valid data for heatmap analysis.")
        return
    
    # Create confidence bins
    n_bins = 10
    filtered_df['confidence_bin'] = pd.cut(
        filtered_df['confidence'], 
        bins=n_bins,
        labels=[f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}" for i in range(n_bins)]
    )
    
    # Create correctness categories (0 = incorrect, 1 = correct)
    filtered_df['correctness'] = filtered_df['model_correct'].map({0: 'Incorrect', 1: 'Correct'})
    
    # Group by confidence bin and correctness
    heatmap_data = filtered_df.groupby(['confidence_bin', 'correctness']).size().reset_index(name='count')
    
    # Pivot the data for the heatmap
    pivot_data = heatmap_data.pivot(
        index='confidence_bin', 
        columns='correctness', 
        values='count'
    ).fillna(0)
    
    # Normalize by row (confidence bin)
    row_sums = pivot_data.sum(axis=1)
    normalized_data = pivot_data.div(row_sums, axis=0)
    
    # Reset index to make confidence_bin a column again
    normalized_data = normalized_data.reset_index()
    
    # Melt the dataframe for plotly
    melted_data = pd.melt(
        normalized_data, 
        id_vars=['confidence_bin'], 
        value_vars=['Correct', 'Incorrect'], 
        var_name='correctness', 
        value_name='normalized_count'
    )
    
    # Create the heatmap
    fig = px.density_heatmap(
        melted_data,
        x='confidence_bin',
        y='correctness',
        z='normalized_count',
        color_continuous_scale='Viridis',
        title='Confidence vs. Correctness Distribution',
        labels={
            'confidence_bin': 'Confidence Range',
            'correctness': 'Prediction Outcome',
            'normalized_count': 'Proportion'
        }
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Confidence Range',
        yaxis_title='Prediction Outcome',
        coloraxis_colorbar=dict(
            title='Proportion',
            tickformat='.0%'
        ),
        height=500
    )
    
    # Display the heatmap
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    st.write("### Calibration Analysis")
    st.write("""
            This heatmap reveals the model's calibration performance across different contexts:
    
            - **Optimal calibration**: When confidence is highest, accuracy should match
            - **Overconfidence**: Model may be more certain than actual performance warrants
            - **Underconfidence**: Model may doubt itself despite good performance
    
            Each cell in this heatmap represents a calibration measurement, showing how well the model's confidence aligns with actual accuracy.
    """)
    
    # Calculate and display additional metrics
    if not filtered_df.empty:
        # Calculate average confidence for correct and incorrect predictions
        avg_conf_correct = filtered_df[filtered_df['model_correct'] == 1]['confidence'].mean()
        avg_conf_incorrect = filtered_df[filtered_df['model_correct'] == 0]['confidence'].mean()
        
        # Calculate high-confidence error rate (confidence > 0.9)
        high_conf = filtered_df[filtered_df['confidence'] > 0.9]
        high_conf_error_rate = 1 - high_conf['model_correct'].mean() if not high_conf.empty else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Confidence When Correct",
                f"{avg_conf_correct:.2%}"
            )
        
        with col2:
            st.metric(
                "Confidence When Incorrect",
                f"{avg_conf_incorrect:.2%}"
            )
        
        with col3:
            st.metric(
                "High Confidence Error Rate",
                f"{high_conf_error_rate:.2%}",
                help="Error rate when confidence is >90%"
            )

def display_pre_post_calibration_comparison(df):
    """Display calibration performance comparison over time"""
    st.subheader("Performance Comparison Analysis")
    
    # Check if we have the necessary data
    if 'confidence' not in df.columns or 'model_correct' not in df.columns or 'timestamp' not in df.columns:
        st.warning("Confidence, correctness, or timestamp data not available.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 20:
        st.info("Not enough data points for calibration comparison in the selected time range.")
        return
    
    # For demonstration, we'll use the midpoint of the time range as a simulated "calibration event"
    # In a real implementation, you would use actual calibration event timestamps from a log
    
    # Convert timestamp to datetime if needed
    if filtered_df['timestamp'].dtype != 'datetime64[ns]':
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    
    # Sort by timestamp
    filtered_df = filtered_df.sort_values('timestamp')
    
    # Get midpoint timestamp
    midpoint_idx = len(filtered_df) // 2
    midpoint_timestamp = filtered_df.iloc[midpoint_idx]['timestamp']
    
        # Allow user to select a different model checkpoint date
    st.write("### Select Model Checkpoint")
    st.write("Select a reference point for the analysis. By default, the midpoint of the dataset is selected.")
    
    # Create a date input for the model checkpoint
    cal_event_date = st.date_input(
        "Model Checkpoint",
        value=midpoint_timestamp.date(),
        min_value=filtered_df['timestamp'].min().date(),
        max_value=filtered_df['timestamp'].max().date()
    )
    
    # Convert to datetime
    cal_event_datetime = datetime.combine(cal_event_date, datetime.min.time())
    
    # Split data into pre and post calibration
    pre_cal_df = filtered_df[filtered_df['timestamp'] < cal_event_datetime]
    post_cal_df = filtered_df[filtered_df['timestamp'] >= cal_event_datetime]
    
    # Check if we have enough data in both periods
    if len(pre_cal_df) < 10 or len(post_cal_df) < 10:
        st.warning(f"Insufficient data before or after {cal_event_date}. Please select a different reference date.")
        return
    
    # Create bins for calibration curves
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Function to calculate calibration curve data
    def calculate_calibration_data(data):
        # Remove NaN values before binning
        data = data.dropna(subset=['confidence', 'model_correct'])
        if len(data) < 5:
            return pd.DataFrame()  # Return empty dataframe if insufficient data
        
        # Bin the data by confidence
        data['confidence_bin'] = pd.cut(data['confidence'], bins=bin_edges, labels=False)
        
        # Group by bin and calculate mean confidence and accuracy
        cal_data = data.groupby('confidence_bin').agg(
            mean_confidence=('confidence', 'mean'),
            accuracy=('model_correct', 'mean'),
            count=('model_correct', 'count')
        ).reset_index()
        
        # Replace bin index with bin center
        cal_data['confidence_bin'] = cal_data['confidence_bin'].apply(lambda x: bin_centers[int(x)])
        
        return cal_data
    
    # Calculate calibration data for pre and post periods
    pre_cal_data = calculate_calibration_data(pre_cal_df)
    post_cal_data = calculate_calibration_data(post_cal_df)
    
    # Check if we have valid calibration data
    if pre_cal_data.empty or post_cal_data.empty:
        st.warning("Insufficient valid data for calibration curve comparison.")
        return
    
    # Create the calibration curve comparison plot
    fig = go.Figure()
    
    # Add perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray')
    ))
    
    # Add pre-calibration curve
    fig.add_trace(go.Scatter(
        x=pre_cal_data['confidence_bin'],
        y=pre_cal_data['accuracy'],
        mode='lines+markers',
        name='Before Reference Date',
        line=dict(color='#ff6b6b'),
        marker=dict(size=10),
        text=[f"Count: {count}" for count in pre_cal_data['count']],
        hovertemplate='Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}'
    ))
    
    # Add post-calibration curve
    fig.add_trace(go.Scatter(
        x=post_cal_data['confidence_bin'],
        y=post_cal_data['accuracy'],
        mode='lines+markers',
        name='After Reference Date',
        line=dict(color='#4ecdc4'),
        marker=dict(size=10),
        text=[f"Count: {count}" for count in post_cal_data['count']],
        hovertemplate='Confidence: %{x:.2f}<br>Accuracy: %{y:.2f}<br>%{text}'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Model Calibration Comparison (Before vs. After {cal_event_date})',
        xaxis_title='Model Confidence',
        yaxis_title='Actual Accuracy',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display calibration metrics
    st.write("### Performance Comparison")
    
    # Calculate ECE for pre and post calibration
    def calculate_ece(data):
        # Calculate expected calibration error
        ece = np.sum(np.abs(data['mean_confidence'] - data['accuracy']) * (data['count'] / data['count'].sum()))
        return ece
    
    pre_ece = calculate_ece(pre_cal_data)
    post_ece = calculate_ece(post_cal_data)
    
    # Calculate average confidence and accuracy
    pre_avg_conf = pre_cal_df['confidence'].mean()
    post_avg_conf = post_cal_df['confidence'].mean()
    
    pre_accuracy = pre_cal_df['model_correct'].mean()
    post_accuracy = post_cal_df['model_correct'].mean()
    
    # Calculate confidence-accuracy gap
    pre_gap = pre_avg_conf - pre_accuracy
    post_gap = post_avg_conf - post_accuracy
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Calibration Error",
            f"{post_ece:.2%}",
            delta=f"{post_ece - pre_ece:.2%}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Confidence-Accuracy Gap",
            f"{post_gap:.2%}",
            delta=f"{post_gap - pre_gap:.2%}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Model Accuracy",
            f"{post_accuracy:.2%}",
            delta=f"{post_accuracy - pre_accuracy:.2%}"
        )
    
    # Add interpretation
    st.write("### Analysis")
    
    if post_ece < pre_ece:
        st.success("✅ Model calibration improved after the reference date.")
    else:
        st.warning("⚠️ Model calibration has not improved since the reference date.")
    
    if abs(post_gap) < abs(pre_gap):
        st.success("✅ The confidence-accuracy gap has narrowed over time.")
    else:
        st.warning("⚠️ The model needs better confidence calibration.")