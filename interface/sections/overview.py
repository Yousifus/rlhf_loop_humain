"""
System Performance Dashboard

This section provides comprehensive overview of model performance,
user interaction patterns, and system health metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from interface.components.utils import (
    ANNOTATION_TARGET, 
    TARGET_ACCURACY,
    format_timestamp,
    create_time_slider
)

def show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary):
    """Display system performance overview"""
    st.header("📊 Model Performance Overview")
    
    # Key metrics in expanded cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_annotation_progress(vote_df, data_summary)
    
    with col2:
        show_accuracy_metrics(vote_df, data_summary)
    
    with col3:
        show_reflection_metrics(reflections_df, data_summary)
    
    with col4:
        show_consistency_metrics(vote_df, predictions_df, data_summary)
    
    # System health indicators
    st.subheader("System Health Status")
    
    # Create health indicators row
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        # Data freshness
        latest_annotation = vote_df['timestamp'].max() if not vote_df.empty and 'timestamp' in vote_df.columns else None
        if latest_annotation is not None:
            time_diff = (pd.Timestamp.now() - pd.to_datetime(latest_annotation)).total_seconds() / 3600
            if time_diff < 24:
                st.success("✅ Recent training data available (less than 24 hours)")
            elif time_diff < 72:
                st.warning("⚠️ Training data is moderately stale (1-3 days)")
            else:
                st.error("❌ Training data is outdated (over 3 days)")
        else:
            st.error("❌ No training data available yet")
    
    with health_col2:
        # Annotation volume
        recent_count = len(vote_df[vote_df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]) if not vote_df.empty and 'timestamp' in vote_df.columns else 0
        if recent_count >= 50:
            st.success(f"✅ Active data collection ({recent_count} entries this week)")
        elif recent_count >= 10:
            st.warning(f"⚠️ Moderate data collection activity ({recent_count} entries this week)")
        else:
            st.error(f"❌ Low data collection activity ({recent_count} entries this week)")
    
    with health_col3:
        # Error trend
        if not vote_df.empty and 'model_correct' in vote_df.columns and 'timestamp' in vote_df.columns:
            # Check if we have enough data for a trend
            if len(vote_df) >= 10:
                vote_df_sorted = vote_df.sort_values('timestamp')
                split_idx = len(vote_df_sorted) // 2
                first_half = vote_df_sorted.iloc[:split_idx]
                second_half = vote_df_sorted.iloc[split_idx:]
                
                first_accuracy = first_half['model_correct'].mean()
                second_accuracy = second_half['model_correct'].mean()
                
                accuracy_change = second_accuracy - first_accuracy
                
                if accuracy_change > 0.05:
                    st.success(f"✅ Model performance improving significantly (+{accuracy_change:.1%})")
                elif accuracy_change > 0:
                    st.info(f"ℹ️ Model performance slowly improving (+{accuracy_change:.1%})")
                elif accuracy_change > -0.05:
                    st.warning(f"⚠️ Model performance showing minor decline ({accuracy_change:.1%})")
                else:
                    st.error(f"❌ Model performance declining significantly ({accuracy_change:.1%})")
            else:
                st.info("ℹ️ More training data needed to establish performance trend")
        else:
            st.error("❌ Unable to assess model performance trend")
    
    with health_col4:
        # Reflection quality
        if not reflections_df.empty and 'quality_score' in reflections_df.columns:
            avg_quality = reflections_df['quality_score'].mean()
            if avg_quality >= 0.8:
                st.success(f"✅ High quality reflection analysis ({avg_quality:.1%})")
            elif avg_quality >= 0.5:
                st.info(f"ℹ️ Moderate quality reflection analysis ({avg_quality:.1%})")
            else:
                st.warning(f"⚠️ Low quality reflection analysis ({avg_quality:.1%})")
        else:
            st.info("ℹ️ No reflection analysis data available")
    
    # Recent activity log
    st.subheader("Recent System Activity")
    
    # Combine data for activity log
    recent_activity = []
    
    # Add annotations
    if not vote_df.empty and 'timestamp' in vote_df.columns:
        recent_votes = vote_df.sort_values('timestamp', ascending=False).head(5)
        for _, row in recent_votes.iterrows():
            recent_activity.append({
                'timestamp': pd.to_datetime(row['timestamp']),
                'action': 'Prediction',
                'details': f"{'✓ Correct prediction' if row.get('model_correct', False) else '✗ Incorrect prediction'}",
                'prompt': row.get('prompt', 'No prompt available')[:50] + '...' if len(row.get('prompt', '')) > 50 else row.get('prompt', 'No prompt available')
            })
    
    # Add reflections if available
    if not reflections_df.empty and 'timestamp' in reflections_df.columns:
        recent_reflections = reflections_df.sort_values('timestamp', ascending=False).head(3)
        for _, row in recent_reflections.iterrows():
            recent_activity.append({
                'timestamp': pd.to_datetime(row['timestamp']),
                'action': 'Reflection Analysis',
                'details': f"Quality Score: {row.get('quality_score', 0):.2f}",
                'prompt': row.get('prompt', 'No prompt available')[:50] + '...' if len(row.get('prompt', '')) > 50 else row.get('prompt', 'No prompt available')
            })
    
    # Sort by timestamp
    if recent_activity:
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        activity_df = pd.DataFrame(recent_activity)
        st.dataframe(
            activity_df[['timestamp', 'action', 'details', 'prompt']],
            use_container_width=True
        )
    else:
        st.info("No training data available yet")
    
    # Weekly velocity metrics
    st.subheader("Training Data Velocity Metrics")
    
    # Create velocity metrics
    velocity_col1, velocity_col2, velocity_col3 = st.columns(3)
    
    with velocity_col1:
        # Annotation velocity
        weekly_annotations = len(vote_df[vote_df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]) if not vote_df.empty and 'timestamp' in vote_df.columns else 0
        st.metric(
            "Weekly Training Samples",
            f"{weekly_annotations}",
            delta=f"{weekly_annotations - data_summary.get('last_week_annotations', 0)}" if 'last_week_annotations' in data_summary else None
        )
    
    with velocity_col2:
        # Error discovery velocity
        weekly_errors = len(vote_df[(vote_df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))) & (vote_df['model_correct'] == 0)]) if not vote_df.empty and 'timestamp' in vote_df.columns and 'model_correct' in vote_df.columns else 0
        st.metric(
            "Weekly Prediction Errors",
            f"{weekly_errors}",
            delta=f"{weekly_errors - data_summary.get('last_week_errors', 0)}" if 'last_week_errors' in data_summary else None
        )
    
    with velocity_col3:
        # Reflection velocity
        weekly_reflections = len(reflections_df[reflections_df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]) if not reflections_df.empty and 'timestamp' in reflections_df.columns else 0
        st.metric(
            "Weekly Reflection Analyses",
            f"{weekly_reflections}",
            delta=f"{weekly_reflections - data_summary.get('last_week_reflections', 0)}" if 'last_week_reflections' in data_summary else None
        )

def show_annotation_progress(vote_df, data_summary):
    """Show training data collection progress"""
    total_annotations = len(vote_df) if not vote_df.empty else 0
    progress_percentage = min(100, total_annotations / ANNOTATION_TARGET * 100)
    
    st.metric(
        "Training Data Progress",
        f"{total_annotations} / {ANNOTATION_TARGET}",
        f"{progress_percentage:.1f}%"
    )
    
    # Progress bar
    st.progress(progress_percentage / 100)

def show_accuracy_metrics(vote_df, data_summary):
    """Show model accuracy metrics"""
    if not vote_df.empty and 'model_correct' in vote_df.columns:
        current_accuracy = vote_df['model_correct'].mean()
        delta = current_accuracy - data_summary.get('previous_accuracy', 0) if 'previous_accuracy' in data_summary else None
        
        st.metric(
            "Model Accuracy",
            f"{current_accuracy:.1%}",
            f"{delta:.1%}" if delta is not None else None
        )
        
        # Accuracy target progress
        accuracy_percentage = min(100, current_accuracy / TARGET_ACCURACY * 100)
        st.progress(accuracy_percentage / 100)
    else:
        st.metric("Model Accuracy", "No Data Available")
        st.progress(0)

def show_reflection_metrics(reflections_df, data_summary):
    """Show reflection analysis metrics"""
    total_reflections = len(reflections_df) if not reflections_df.empty else 0
    
    # Quality score if available
    if not reflections_df.empty and 'quality_score' in reflections_df.columns:
        avg_quality = reflections_df['quality_score'].mean()
        st.metric(
            "Reflection Analyses",
            f"{total_reflections}",
            f"Avg Quality: {avg_quality:.1%}"
        )
    else:
        st.metric("Reflection Analyses", f"{total_reflections}")
    
    # Reflection ratio
    annotation_count = len(reflections_df) if not reflections_df.empty else 0
    if annotation_count > 0:
        reflection_ratio = total_reflections / annotation_count
        st.progress(min(1.0, reflection_ratio))
    else:
        st.progress(0)

def show_consistency_metrics(vote_df, predictions_df, data_summary):
    """Show model consistency and calibration metrics"""
    if not vote_df.empty and 'model_choice' in vote_df.columns and 'human_choice' in vote_df.columns:
        # Calculate agreement rate
        vote_df['agreement'] = vote_df['model_choice'] == vote_df['human_choice']
        agreement_rate = vote_df['agreement'].mean()
        
        st.metric(
            "Human-AI Agreement",
            f"{agreement_rate:.1%}"
        )
        
        # Confidence vs accuracy if available
        if not predictions_df.empty and 'confidence' in predictions_df.columns:
            avg_confidence = predictions_df['confidence'].mean()
            confidence_ratio = avg_confidence / agreement_rate if agreement_rate > 0 else 1
            
            # Progress bar showing calibration (1.0 = perfectly calibrated)
            calibration = 1 - min(0.5, abs(avg_confidence - agreement_rate)) / 0.5
            st.progress(calibration)
            
            if confidence_ratio > 1.1:
                st.caption("⚠️ Model may show overconfidence")
            elif confidence_ratio < 0.9:
                st.caption("⚠️ Model may show underconfidence")
            else:
                st.caption("✓ Model appears well-calibrated")
        else:
            st.progress(0.5)
    else:
        st.metric("Human-AI Agreement", "No Data Available")
        st.progress(0) 