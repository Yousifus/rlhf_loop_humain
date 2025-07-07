"""
Model Calibration Metrics

Measures how accurately the model predicts outcomes and aligns with user preferences.
Provides comprehensive calibration analysis and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from interface.components.visualization import plot_confidence_vs_accuracy
from interface.components.utils import create_time_slider, filter_by_time_range

# Import additional calibration visualizations
from interface.sections.calibration_additions import (
    display_confidence_correctness_heatmap,
    display_pre_post_calibration_comparison
)

def display_calibration_diagnostics(vote_df, predictions_df):
    """Display model calibration and performance metrics"""
    st.header("📊 Model Calibration Metrics")
    
    # Check if we have the necessary data
    if vote_df.empty or predictions_df.empty:
        st.warning("Vote or prediction data not available.")
        return
    
    # Determine common key for merging
    merge_key = 'prompt_id' if 'prompt_id' in vote_df.columns and 'prompt_id' in predictions_df.columns else None
    if merge_key is None:
        st.warning("Cannot merge vote and prediction data due to missing common key.")
        return
    
    # Check if we have confidence scores and correctness
    if 'confidence' not in predictions_df.columns:
        st.warning("Confidence scores not available in predictions data.")
        return
        
    if 'model_correct' not in vote_df.columns:
        st.warning("Model correctness data not available.")
        return
    
    # Merge the dataframes
    merged_df = pd.merge(
        vote_df[['timestamp', merge_key, 'model_correct']],
        predictions_df[[merge_key, 'confidence']],
        on=merge_key,
        how='inner'
    )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Calibration Analysis",
        "Confidence Distribution",
        "Prediction Errors",
        "Performance Heatmap",
        "Model Evolution"
    ])
    
    with tab1:
        display_calibration_curves(merged_df)
    
    with tab2:
        display_confidence_distribution(merged_df)
    
    with tab3:
        display_calibration_error(merged_df)
        
    with tab4:
        display_confidence_correctness_heatmap(merged_df)
        
    with tab5:
        display_pre_post_calibration_comparison(merged_df)

def display_calibration_curves(df):
    """Display calibration curve analysis"""
    st.subheader("Model Calibration Analysis")
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.info("Not enough data points for calibration analysis in the selected time range.")
        return
    
    # Check if we have confidence and correctness data
    if 'confidence' not in filtered_df.columns or 'model_correct' not in filtered_df.columns:
        st.warning("Confidence or model correctness data not available.")
        return
    
    # Plot calibration curve
    st.write("### Calibration Curve Analysis")
    fig = plot_confidence_vs_accuracy(
        filtered_df, 
        confidence_col='confidence', 
        accuracy_col='model_correct', 
        bins=10, 
        title="Confidence vs. Accuracy Calibration Curve"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed calibration statistics
    st.write("### Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Calibration Performance")
        
        # Calculate average confidence and accuracy
        avg_confidence = filtered_df['confidence'].mean()
        accuracy = filtered_df['model_correct'].mean()
        
        # Calculate confidence-accuracy gap
        gap = avg_confidence - accuracy
        
        # Display metrics
        st.metric(
            "Mean Confidence", 
            f"{avg_confidence:.2%}"
        )
        
        st.metric(
            "Accuracy", 
            f"{accuracy:.2%}"
        )
        
        st.metric(
            "Confidence-Accuracy Gap", 
            f"{gap:.2%}",
            delta="-" if abs(gap) < 0.05 else None,  # Show delta only if the gap is small
            help="Gap between mean confidence and accuracy. Values close to 0 indicate good calibration. Positive values indicate overconfidence, negative values indicate underconfidence."
        )
        
        # Show calibration status
        if abs(gap) < 0.05:
            st.success("✅ Model is well-calibrated and performing optimally")
        elif gap > 0:
            st.warning("⚠️ Model shows overconfidence - predicted confidence exceeds actual accuracy")
        else:
            st.warning("⚠️ Model shows underconfidence - predicted confidence is below actual accuracy")
    
    with col2:
        st.write("#### Performance by Confidence Level")
        
        # Check for valid confidence data
        valid_conf_data = filtered_df.dropna(subset=['confidence'])
        if len(valid_conf_data) < 5:
            st.warning("Insufficient valid confidence data for binning analysis.")
            return
        
        # Create confidence bins
        bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['0-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        
        valid_conf_data['confidence_bin'] = pd.cut(
            valid_conf_data['confidence'], 
            bins=bins, 
            labels=bin_labels,
            include_lowest=True
        )
        
        # Group by confidence bin
        bin_stats = valid_conf_data.groupby('confidence_bin').agg(
            sample_count=('model_correct', 'count'),
            accuracy=('model_correct', 'mean'),
            mean_confidence=('confidence', 'mean')
        ).reset_index()
        
        # Calculate calibration gap
        bin_stats['gap'] = bin_stats['mean_confidence'] - bin_stats['accuracy']
        
        # Create a table
        st.dataframe(
            bin_stats.style.format({
                'sample_count': '{:,.0f}',
                'accuracy': '{:.1%}',
                'mean_confidence': '{:.1%}',
                'gap': '{:.1%}'
            }),
            use_container_width=True
        )

def display_confidence_distribution(df):
    """Display model confidence calibration analysis"""
    st.subheader("Model Confidence Analysis")
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.info("Not enough data points for confidence distribution analysis in the selected time range.")
        return
    
    # Check if we have confidence data
    if 'confidence' not in filtered_df.columns:
        st.warning("Confidence data not available.")
        return
    
    # Remove rows with NaN confidence values
    filtered_df = filtered_df.dropna(subset=['confidence'])
    if len(filtered_df) < 5:
        st.warning("Insufficient valid confidence data for distribution analysis.")
        return
    
    # Plot confidence distribution
    fig = px.histogram(
        filtered_df,
        x='confidence',
        nbins=20,
        title='Model Confidence Distribution',
        labels={'confidence': 'Confidence Score'},
        opacity=0.7,
        color_discrete_sequence=['#4CAF50']  # Green for model performance
    )
    
    fig.update_layout(
        xaxis_title='Confidence Score',
        yaxis_title='Count',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add confidence distribution metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Confidence", 
            f"{filtered_df['confidence'].mean():.2%}"
        )
    
    with col2:
        st.metric(
            "Median Confidence", 
            f"{filtered_df['confidence'].median():.2%}"
        )
    
    with col3:
        st.metric(
            "Min Confidence", 
            f"{filtered_df['confidence'].min():.2%}"
        )
    
    with col4:
        st.metric(
            "Max Confidence", 
            f"{filtered_df['confidence'].max():.2%}"
        )
    
    # Add confidence distribution by correctness
    if 'model_correct' in filtered_df.columns:
        st.write("### Confidence Distribution by Prediction Accuracy")
        
        # Split data by correctness
        correct_df = filtered_df[filtered_df['model_correct'] == 1]
        incorrect_df = filtered_df[filtered_df['model_correct'] == 0]
        
        # Create distributions
        fig = go.Figure()
        
        # Add correct predictions
        fig.add_trace(go.Histogram(
            x=correct_df['confidence'],
            name='Correct Predictions',
            opacity=0.7,
            marker_color='#4ecdc4',
            nbinsx=20
        ))
        
        # Add incorrect predictions
        fig.add_trace(go.Histogram(
            x=incorrect_df['confidence'],
            name='Incorrect Predictions',
            opacity=0.7,
            marker_color='#ff6b6b',
            nbinsx=20
        ))
        
        # Update layout
        fig.update_layout(
            title='Confidence Distribution by Prediction Correctness',
            xaxis_title='Confidence Score',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare average confidence by correctness
        col1, col2 = st.columns(2)
        
        with col1:
            avg_correct = correct_df['confidence'].mean() if not correct_df.empty else 0
            st.metric(
                "Avg Confidence (Correct)", 
                f"{avg_correct:.2%}"
            )
        
        with col2:
            avg_incorrect = incorrect_df['confidence'].mean() if not incorrect_df.empty else 0
            st.metric(
                "Avg Confidence (Incorrect)", 
                f"{avg_incorrect:.2%}"
            )
        
        # Calculate confidence gap
        if not correct_df.empty and not incorrect_df.empty:
            gap = avg_correct - avg_incorrect
            st.metric(
                "Confidence Gap", 
                f"{gap:.2%}",
                help="Difference in average confidence between correct and incorrect predictions. Higher values indicate better confidence calibration."
            )
            
            if gap < 0.05:
                st.warning("⚠️ Small confidence gap suggests poor confidence calibration.")
            elif gap > 0.2:
                st.success("✅ Large confidence gap suggests good confidence differentiation.")

def display_calibration_error(df):
    """Display calibration error analysis by confidence level"""
    st.subheader("Calibration Error Analysis by Confidence Level")
    
    # Apply time range filter
    filtered_df = create_time_slider(df)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.info("Not enough data points for calibration error analysis in the selected time range.")
        return
    
    # Check if we have confidence and correctness data
    if 'confidence' not in filtered_df.columns or 'model_correct' not in filtered_df.columns:
        st.warning("Confidence or model correctness data not available.")
        return
    
    # Remove rows with NaN values
    filtered_df = filtered_df.dropna(subset=['confidence', 'model_correct'])
    if len(filtered_df) < 10:
        st.warning("Insufficient valid data for calibration error analysis.")
        return
    
    # Calculate expected calibration error
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(filtered_df['confidence'], bins) - 1
    
    bin_sums = np.bincount(binids, weights=filtered_df['confidence'], minlength=len(bins))
    bin_true = np.bincount(binids, weights=filtered_df['model_correct'], minlength=len(bins))
    bin_counts = np.bincount(binids, minlength=len(bins))
    
    # Calculate mean predicted probability and fraction of positives in each bin
    nonzero = bin_counts != 0
    prob_pred = bin_sums[nonzero] / bin_counts[nonzero]  # Mean predicted probability in each bin
    prob_true = bin_true[nonzero] / bin_counts[nonzero]  # Fraction of positives in each bin
    
    # Calculate ECE (weighted average of bin disparities)
    ece = np.sum(np.abs(prob_pred - prob_true) * (bin_counts[nonzero] / len(filtered_df)))
    
    # Create data for bin-wise plot
    bin_indices = np.arange(len(prob_pred))
    bin_names = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in bin_indices]
    bin_errors = np.abs(prob_pred - prob_true)
    
    # Calculate bin weights
    bin_weights = bin_counts[nonzero] / len(filtered_df)
    
    # Create dataframe for plotting
    calibration_df = pd.DataFrame({
        'bin': bin_names,
        'pred_prob': prob_pred,
        'true_prob': prob_true,
        'abs_error': bin_errors,
        'sample_weight': bin_weights,
        'sample_count': bin_counts[nonzero]
    })
    
    # Display the ECE metric
    st.metric(
        "Expected Calibration Error (ECE)", 
        f"{ece:.2%}",
        help="Weighted average of absolute differences between confidence and accuracy across bins. Lower values indicate better calibration."
    )
    
    # Add interpretation
    if ece < 0.01:
        st.success("✅ Model is exceptionally well-calibrated (< 1% error)")
    elif ece < 0.05:
        st.success("✅ Model shows good calibration with minor alignment opportunities (< 5% error)")
    elif ece < 0.10:
        st.warning("⚠️ Model requires calibration improvement (5-10% error)")
    else:
        st.error("❌ Model requires significant calibration improvement (> 10% error)")
    
    # Display calibration error by bin
    st.write("### Calibration Error by Confidence Bin")
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bar chart for calibration error by bin
    fig.add_trace(go.Bar(
        x=calibration_df['bin'],
        y=calibration_df['abs_error'],
        marker_color=np.where(calibration_df['abs_error'] > 0.1, '#ff6b6b', '#4ecdc4'),
        name='Calibration Error',
        text=[f"{err:.1%}<br>{count} samples" for err, count in zip(calibration_df['abs_error'], calibration_df['sample_count'])],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Calibration Error by Confidence Bin',
        xaxis_title='Confidence Bin',
        yaxis_title='Absolute Error (|confidence - accuracy|)',
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the table with all calibration statistics
    st.write("### Detailed Calibration Statistics")
    
    # Format the dataframe for display
    display_df = calibration_df.copy()
    display_df = display_df.rename(columns={
        'bin': 'Confidence Bin',
        'pred_prob': 'Avg Confidence',
        'true_prob': 'Accuracy',
        'abs_error': 'Abs Error',
        'sample_weight': 'Weight',
        'sample_count': 'Samples'
    })
    
    # Add contributing error column (weighted error)
    display_df['Contrib to ECE'] = display_df['Abs Error'] * display_df['Weight']
    
    # Reorder columns
    display_df = display_df[['Confidence Bin', 'Samples', 'Avg Confidence', 'Accuracy', 'Abs Error', 'Weight', 'Contrib to ECE']]
    
    # Display the table
    st.dataframe(
        display_df.style.format({
            'Samples': '{:,.0f}',
            'Avg Confidence': '{:.1%}',
            'Accuracy': '{:.1%}',
            'Abs Error': '{:.1%}',
            'Weight': '{:.1%}',
            'Contrib to ECE': '{:.2%}'
        }),
        use_container_width=True
    ) 