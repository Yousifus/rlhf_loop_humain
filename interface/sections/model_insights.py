"""
Model Performance Insights Interface

This section provides comprehensive insights into model performance and behavior -
model learning patterns, evolution, and performance optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

def display_model_insights():
    """Display comprehensive insights into model performance and evolution"""
    st.header("🧠 Model Performance Insights")
    
    # Professional header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b9d, #c44569);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">🔬 Model Analysis Dashboard</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
            Monitor model development, learning patterns, and performance optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 Model Training",
        "🎯 Calibration",
        "📊 Performance Metrics", 
        "🔄 Drift Analysis",
        "💭 Introspection"
    ])
    
    with tab1:
        display_training_insights()
    
    with tab2:
        display_calibration_insights()
    
    with tab3:
        display_performance_metrics()
    
    with tab4:
        display_drift_analysis()
    
    with tab5:
        display_introspection_insights()

def display_training_insights():
    """Show training progress and model evolution"""
    st.subheader("🌱 Model Training Progress")
    
    # Load training log
    training_log_path = Path("models/vote_predictor_training_log.json")
    
    if training_log_path.exists():
        with open(training_log_path, 'r') as f:
            training_data = json.load(f)
        
        # Display training overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model Architecture",
                training_data.get('model_name', 'Unknown').split('/')[-1],
                help="The neural architecture used for preference learning"
            )
        
        with col2:
            st.metric(
                "Training Examples",
                training_data.get('dataset_size', 0),
                help="Number of training examples processed"
            )
        
        with col3:
            st.metric(
                "Learning Rate",
                f"{training_data.get('training_params', {}).get('learning_rate', 0):.0e}",
                help="Rate of model parameter updates during training"
            )
        
        # Training parameters
        st.subheader("🔧 Training Configuration")
        
        params = training_data.get('training_params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Parameters:**")
            st.write(f"• **Batch Size:** {params.get('batch_size', 'Unknown')}")
            st.write(f"• **Epochs:** {params.get('epochs', 'Unknown')}")
            st.write(f"• **Max Length:** {params.get('max_length', 'Unknown')} tokens")
            st.write(f"• **Weight Decay:** {params.get('weight_decay', 'Unknown')}")
        
        with col2:
            st.markdown("**Training Strategy:**")
            st.write(f"• **Validation Split:** {params.get('validation_split', 'Unknown')}")
            st.write(f"• **Save Steps:** {params.get('save_steps', 'Unknown')}")
            st.write(f"• **Early Stopping:** {params.get('early_stopping_patience', 'Unknown')} patience")
            st.write(f"• **Random Seed:** {params.get('seed', 'Unknown')}")
        
        # Training timeline
        st.subheader("📅 Training Timeline")
        timestamp = training_data.get('timestamp', '')
        if timestamp:
            training_time = datetime.fromisoformat(timestamp.replace('T', ' '))
            st.info(f"🕐 Last trained: {training_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    else:
        st.warning("No training data found. Model has not been trained yet.")
        st.info("💡 Train the model using the vote predictor training script to see learning progress here.")

def display_calibration_insights():
    """Show calibration metrics and confidence analysis"""
    st.subheader("🎯 Model Confidence Calibration")
    
    # Load calibration data
    calibration_path = Path("models/calibration_log.json")
    
    if calibration_path.exists():
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
        
        # Calibration overview
        st.markdown("### 📊 Calibration Performance")
        
        col1, col2, col3 = st.columns(3)
        
        metrics = calibration_data.get('metrics', {})
        pre_cal = metrics.get('pre_calibration', {})
        post_cal = metrics.get('post_calibration', {})
        improvement = metrics.get('improvement', {})
        
        with col1:
            st.metric(
                "Expected Calibration Error",
                f"{post_cal.get('ece', 0):.3f}",
                delta=f"{-improvement.get('ece', 0):.3f}",
                help="Measure of confidence-accuracy alignment"
            )
        
        with col2:
            st.metric(
                "Log Loss",
                f"{post_cal.get('log_loss', 0):.3f}",
                delta=f"{-improvement.get('log_loss', 0):.3f}",
                help="Prediction confidence quality metric"
            )
        
        with col3:
            st.metric(
                "Brier Score",
                f"{post_cal.get('brier_score', 0):.3f}",
                delta=f"{-improvement.get('brier_score', 0):.3f}",
                help="Overall prediction quality score"
            )
        
        # Temperature scaling info
        st.subheader("🌡️ Temperature Scaling")
        
        temp = calibration_data.get('parameters', {}).get('temperature', 1.0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Calibration Temperature",
                f"{temp:.2f}",
                help="Parameter for confidence adjustment"
            )
        
        with col2:
            method = calibration_data.get('method', 'Unknown')
            st.metric(
                "Calibration Method",
                method.replace('_', ' ').title(),
                help="Technique used for confidence calibration"
            )
        
        # Calibration history
        history = calibration_data.get('history', [])
        if history:
            st.subheader("📈 Calibration History")
            
            # Create DataFrame from history
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Plot calibration error over time
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['calibration_error'],
                mode='lines+markers',
                name='Calibration Error',
                line=dict(color='#ff6b9d', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='Calibration Improvement Over Time',
                xaxis_title='Time',
                yaxis_title='Calibration Error',
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show latest calibration stats
            latest = history[-1]
            st.markdown("### 📋 Latest Calibration Session")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Sample Count:** {latest.get('sample_count', 'Unknown')}")
                st.write(f"**Accuracy:** {latest.get('accuracy', 0):.1%}")
            
            with col2:
                st.write(f"**Confidence Before:** {latest.get('avg_confidence_before', 0):.1%}")
                st.write(f"**Confidence After:** {latest.get('avg_confidence_after', 0):.1%}")
            
            with col3:
                st.write(f"**Calibration Error:** {latest.get('calibration_error', 0):.3f}")
                st.write(f"**Notes:** {latest.get('notes', 'No notes')}")
    
    else:
        st.warning("No calibration data found. Model has not been calibrated yet.")
        st.info("💡 Run calibration to analyze model confidence accuracy.")

def display_performance_metrics():
    """Show overall performance and accuracy metrics"""
    st.subheader("📊 Model Performance Analysis")
    
    # Load introspection summary
    introspection_path = Path("models/introspection_summary.json")
    
    if introspection_path.exists():
        with open(introspection_path, 'r') as f:
            introspection_data = json.load(f)
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = introspection_data.get('accuracy', 0)
            st.metric(
                "Overall Accuracy",
                f"{accuracy:.1%}",
                help="Frequency of correct preference predictions"
            )
        
        with col2:
            total_votes = introspection_data.get('total_votes', 0)
            st.metric(
                "Total Evaluations",
                total_votes,
                help="Number of human preferences analyzed"
            )
        
        with col3:
            correct = introspection_data.get('correct_predictions', 0)
            st.metric(
                "Correct Predictions",
                correct,
                help="Number of accurate preference predictions"
            )
        
        with col4:
            high_conf_errors = introspection_data.get('error_counts', {}).get('high_confidence_error', 0)
            st.metric(
                "High Confidence Errors",
                high_conf_errors,
                help="Overconfident incorrect predictions"
            )
        
        # Error breakdown
        st.subheader("🎯 Error Analysis")
        
        error_counts = introspection_data.get('error_counts', {})
        
        if any(error_counts.values()):
            # Create pie chart of error types
            labels = []
            values = []
            colors = ['#ff6b9d', '#c44569', '#8b2635']
            
            for error_type, count in error_counts.items():
                if count > 0:
                    labels.append(error_type.replace('_', ' ').title())
                    values.append(count)
            
            if values:
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=colors[:len(labels)]
                )])
                
                fig.update_layout(
                    title="Prediction Error Categories",
                    height=400,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No categorized prediction errors detected!")
        
        # Confidence distribution
        conf_dist = introspection_data.get('confidence_distribution', {})
        if conf_dist:
            st.subheader("📈 Confidence Distribution")
            
            conf_df = pd.DataFrame([
                {'Confidence Level': k, 'Count': v} 
                for k, v in conf_dist.items()
            ])
            
            fig = px.bar(
                conf_df,
                x='Confidence Level',
                y='Count',
                title='Model Confidence Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=400, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No performance data found. Model has not been evaluated yet.")
        st.info("💡 Run introspective evaluation to analyze model performance.")

def display_drift_analysis():
    """Show drift detection and clustering analysis"""
    st.subheader("🔄 Model Drift Analysis")
    
    # Load drift analysis
    drift_path = Path("models/drift_analysis")
    
    if drift_path.exists():
        # Check for drift analysis file
        drift_file = drift_path / "drift_analysis.json"
        
        if drift_file.exists():
            with open(drift_file, 'r') as f:
                drift_data = json.load(f)
            
            st.markdown("### 📊 Drift Detection Results")
            
            # Display drift metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Drift Score",
                    f"{drift_data.get('overall_drift_score', 0):.3f}",
                    help="Measure of model behavior change over time"
                )
            
            with col2:
                st.metric(
                    "Time Windows",
                    drift_data.get('time_windows_analyzed', 0),
                    help="Number of time periods analyzed"
                )
            
            with col3:
                st.metric(
                    "Clusters Found",
                    drift_data.get('clusters_identified', 0),
                    help="Distinct patterns in model behavior"
                )
            
            # Time-based drift analysis
            time_drift = drift_data.get('time_based_analysis', {})
            if time_drift:
                st.subheader("⏰ Temporal Drift Patterns")
                
                # Create time series of drift
                if 'drift_over_time' in time_drift:
                    drift_series = time_drift['drift_over_time']
                    
                    df = pd.DataFrame([
                        {'Time': k, 'Drift': v} 
                        for k, v in drift_series.items()
                    ])
                    
                    fig = px.line(
                        df,
                        x='Time',
                        y='Drift',
                        title='Model Drift Over Time',
                        markers=True
                    )
                    
                    fig.update_layout(height=400, template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Clustering analysis
            clustering = drift_data.get('clustering_analysis', {})
            if clustering:
                st.subheader("🎯 Behavioral Patterns")
                
                st.write(f"**Algorithm Used:** {clustering.get('algorithm', 'Unknown')}")
                st.write(f"**Silhouette Score:** {clustering.get('silhouette_score', 'N/A')}")
                
                if 'cluster_characteristics' in clustering:
                    chars = clustering['cluster_characteristics']
                    
                    for cluster_id, char in chars.items():
                        with st.expander(f"Cluster {cluster_id} - {char.get('description', 'Unknown')}"):
                            st.write(f"**Size:** {char.get('size', 0)} examples")
                            st.write(f"**Accuracy:** {char.get('accuracy', 0):.1%}")
                            st.write(f"**Avg Confidence:** {char.get('avg_confidence', 0):.1%}")
        
        # Check for cluster files
        cluster_file = drift_path / "drift_clusters.jsonl"
        if cluster_file.exists():
            st.subheader("📋 Detailed Cluster Information")
            
            clusters = []
            with open(cluster_file, 'r') as f:
                for line in f:
                    clusters.append(json.loads(line))
            
            if clusters:
                cluster_df = pd.DataFrame(clusters)
                st.dataframe(cluster_df, use_container_width=True)
            else:
                st.info("No detailed cluster data available.")
    
    else:
        st.warning("No drift analysis found. Model has not been analyzed for drift yet.")
        st.info("💡 Run drift analysis to monitor model behavior changes over time.")

def display_introspection_insights():
    """Show deep introspective analysis"""
    st.subheader("💭 Model Self-Analysis")
    
    # Load meta reflection data
    reflection_path = Path("models/meta_reflection_log.jsonl")
    
    if reflection_path.exists() and reflection_path.stat().st_size > 0:
        reflections = []
        with open(reflection_path, 'r') as f:
            for line in f:
                if line.strip():
                    reflections.append(json.loads(line))
        
        if reflections:
            st.markdown(f"### 📚 {len(reflections)} Reflection Entries")
            
            # Create DataFrame for analysis
            df = pd.DataFrame(reflections)
            
            # Show reflection timeline
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Plot reflection frequency over time
                daily_reflections = df.groupby(df['timestamp'].dt.date).size()
                
                fig = px.bar(
                    x=daily_reflections.index,
                    y=daily_reflections.values,
                    title='Daily Self-Reflection Activity',
                    labels={'x': 'Date', 'y': 'Reflections'}
                )
                
                fig.update_layout(height=400, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show accuracy patterns in reflections
            if 'model_correct' in df.columns:
                accuracy_by_confidence = df.groupby('predicted_choice')['model_correct'].agg(['mean', 'count'])
                
                st.subheader("🎯 Accuracy by Prediction Type")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Choice A Predictions:**")
                    if 'A' in accuracy_by_confidence.index:
                        a_acc = accuracy_by_confidence.loc['A', 'mean']
                        a_count = accuracy_by_confidence.loc['A', 'count']
                        st.metric("Accuracy", f"{a_acc:.1%}", help=f"Based on {a_count} predictions")
                
                with col2:
                    st.write("**Choice B Predictions:**")
                    if 'B' in accuracy_by_confidence.index:
                        b_acc = accuracy_by_confidence.loc['B', 'mean']
                        b_count = accuracy_by_confidence.loc['B', 'count']
                        st.metric("Accuracy", f"{b_acc:.1%}", help=f"Based on {b_count} predictions")
            
            # Show recent reflections
            st.subheader("📖 Recent Self-Reflections")
            
            recent_reflections = sorted(reflections, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
            
            for i, reflection in enumerate(recent_reflections):
                with st.expander(f"Reflection {i+1}: {reflection.get('timestamp', 'Unknown time')}"):
                    st.write(f"**Vote ID:** {reflection.get('vote_id', 'Unknown')}")
                    st.write(f"**Model Prediction:** {reflection.get('predicted_choice', 'Unknown')}")
                    st.write(f"**Human Choice:** {reflection.get('human_choice', 'Unknown')}")
                    st.write(f"**Correct Prediction:** {'✅ Yes' if reflection.get('model_correct') else '❌ No'}")
                    
                    if 'confidence' in reflection:
                        st.write(f"**Model Confidence:** {reflection['confidence']:.1%}")
                    
                    if 'error_analysis' in reflection:
                        st.write(f"**Error Analysis:** {reflection['error_analysis']}")
        else:
            st.info("Reflection file exists but contains no valid entries.")
    
    else:
        st.warning("No introspection data found. Model has not performed self-analysis yet.")
        st.info("💡 Run introspective evaluation to see model self-analysis.")
    
    # Load meta reflection summary
    summary_path = Path("models/meta_reflection_summary.json")
    
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        st.subheader("📊 Self-Reflection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Reflections",
                summary_data.get('total_entries', 0),
                help="Number of self-analysis sessions"
            )
        
        with col2:
            st.metric(
                "Self-Assessed Accuracy",
                f"{summary_data.get('accuracy', 0):.1%}",
                help="Model's own performance assessment"
            )
        
        with col3:
            high_conf_correct = summary_data.get('mean_confidence_correct', 0)
            st.metric(
                "Confidence When Correct",
                f"{high_conf_correct:.1%}",
                help="Average confidence on correct predictions"
            )
        
        with col4:
            high_conf_wrong = summary_data.get('mean_confidence_incorrect', 0)
            st.metric(
                "Confidence When Incorrect",
                f"{high_conf_wrong:.1%}",
                help="Average confidence on incorrect predictions"
            )

def load_model_files_overview():
    """Load overview of all model files"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        return {}
    
    file_info = {}
    
    for file_path in models_dir.rglob("*"):
        if file_path.is_file():
            try:
                stat = file_path.stat()
                file_info[str(file_path.relative_to(models_dir))] = {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'type': file_path.suffix
                }
            except Exception:
                continue
    
    return file_info
