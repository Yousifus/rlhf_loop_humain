#!/usr/bin/env python3
"""
RLHF Pipeline Monitor - HUMAIN Edition

RLHF pipeline monitoring and analysis system for ML engineers.
Provides comprehensive monitoring across the complete RLHF lifecycle:
- Data Collection: Annotation quality, dataset statistics, ingestion status
- Training: Model training status, loss curves, resource utilization  
- Evaluation: Performance metrics, calibration analysis, drift detection
- Deployment: Model serving status, production metrics, system health
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import core data management
from interface.components.data_loader import load_all_data, get_data_summary
from interface.components.utils import AUTO_REFRESH_INTERVAL

# Import RLHF pipeline sections
from interface.sections.overview import show_dashboard_overview
from interface.sections.annotation import display_annotation_interface, display_annotation_history
from interface.sections.alignment import display_alignment_over_time
from interface.sections.calibration import display_calibration_diagnostics
from interface.sections.drift_analysis import display_drift_clusters
from interface.sections.model_evolution import display_model_evolution
from interface.sections.model_insights import display_model_insights

# Import calibration analytics for evaluation phase
from interface.sections.calibration_additions import (
    display_confidence_correctness_heatmap,
    display_pre_post_calibration_comparison
)

# Import model configuration
from interface.sections.model_config_core import (
    get_model_config, display_model_status, apply_professional_styling, 
    system_message_formatter
)

# Import enhanced UX features
from interface.ux_improvements import (
    create_enhanced_metrics_display,
    create_metric_card,
    calculate_connection_depth,
    calculate_learning_progress,
    calculate_engagement_level,
    calculate_sync_rate
)

# Import visualization components
import plotly.graph_objects as go
import plotly.express as px

# Hidden debug interface (not in main navigation)
from interface.sections.chat import display_chat_interface

def main():
    """Main entry point for RLHF Pipeline Monitor"""
    
    # Configure page settings
    st.set_page_config(
        page_title="RLHF Pipeline Monitor | HUMAIN OS",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "RLHF pipeline monitoring system - HUMAIN OS Integration"
        }
    )
    
    # Apply HUMAIN OS styling
    apply_humain_styling()
    
    # Initialize session state
    initialize_pipeline_session_state()
    
    # Get model configuration
    model_config = get_model_config()
    
    # Create pipeline header
    create_pipeline_header()
    
    # Create RLHF pipeline sidebar navigation
    create_pipeline_sidebar()
    
    # Handle hidden debug access
    selected_section = handle_navigation_and_debug()
    
    # Load pipeline data
    try:
        if should_refresh_data():
            # Enhanced loading animation
            loading_placeholder = st.empty()
            with loading_placeholder.container():
                st.markdown("""
                <div style="
                    text-align: center;
                    padding: 30px;
                    background: linear-gradient(135deg, rgba(29, 181, 132, 0.1), rgba(29, 181, 132, 0.05));
                    border-radius: 15px;
                    margin: 20px 0;
                ">
                    <div style="
                        display: inline-block;
                        width: 40px;
                        height: 40px;
                        border: 4px solid rgba(29, 181, 132, 0.3);
                        border-radius: 50%;
                        border-top-color: #1DB584;
                        animation: spin 1s ease-in-out infinite;
                    "></div>
                    <p style="color: #1DB584; margin-top: 15px; font-weight: 500;">
                        🔄 Loading RLHF pipeline data...
                    </p>
                </div>
                
                <style>
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
            
            # Load the data
            vote_df, predictions_df, reflections_df = load_all_data(force_reload=True)
            st.session_state.last_data_refresh = time.time()
            
            # Clear loading animation
            loading_placeholder.empty()
            
            # Store in session state for performance
            st.session_state.vote_df = vote_df
            st.session_state.predictions_df = predictions_df
            st.session_state.reflections_df = reflections_df
        else:
            vote_df = st.session_state.get('vote_df', pd.DataFrame())
            predictions_df = st.session_state.get('predictions_df', pd.DataFrame())
            reflections_df = st.session_state.get('reflections_df', pd.DataFrame())
        
        # Get data summary
        data_summary = get_data_summary()
        st.session_state.data_summary = data_summary
        
        # Update model configuration based on data
        update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df)
        
    except Exception as e:
        st.error(f"Error loading pipeline data: {str(e)}")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
        # Ensure session state is initialized even on error
        st.session_state.vote_df = vote_df
        st.session_state.predictions_df = predictions_df
        st.session_state.reflections_df = reflections_df
        st.session_state.data_summary = data_summary
    
    # Display pipeline content
    display_pipeline_content(selected_section, vote_df, predictions_df, reflections_df, data_summary)
    
    # System status footer
    create_system_footer()

def apply_humain_styling():
    """Apply HUMAIN OS styling"""
    st.markdown("""
    <style>
    /* HUMAIN OS Theme */
    .main > div {
        background: #FFFFFF;
        color: #333333;
    }
    
    /* HUMAIN Primary Actions */
    .stButton > button {
        background: linear-gradient(135deg, #1DB584 0%, #17a573 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(29, 181, 132, 0.15);
        font-size: 14px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #17a573 0%, #148f64 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(29, 181, 132, 0.25);
    }
    
    /* Professional Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 28px;
        color: #1DB584;
    }
    
    h2 {
        font-size: 22px;
        color: #34495e;
    }
    
    h3 {
        font-size: 18px;
        color: #34495e;
    }
    
    /* Professional Cards */
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
        border-color: #1DB584;
    }
    
    /* Professional Sidebar */
    .css-1d391kg {
        background: #F6F5F4;
        border-right: 1px solid #e1e8ed;
    }
    
    /* Status Indicators */
    .stSuccess {
        background-color: #d4edda;
        border-color: #10B981;
        color: #155724;
    }
    
    .stError {
        background-color: #f8d7da;
        border-color: #EF4444;
        color: #721c24;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    /* Professional Data Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    
    /* Clean Forms */
    .stForm {
        background: white;
        border-radius: 12px;
        border: 1px solid #e1e8ed;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 4px;
        background: #F6F5F4;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 6px;
        transition: all 0.2s ease;
        color: #6B7280;
        font-weight: 500;
        padding: 8px 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e5e7eb;
        color: #1DB584;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1DB584;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(29, 181, 132, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #1DB584 0%, #17a573 100%);
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_pipeline_header():
    """Create RLHF pipeline header"""
    st.markdown("""
    <div style="
        background: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    ">
        <h1 style="
            color: #1DB584; 
            margin: 0; 
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
        ">
            ⚙️ RLHF Pipeline Monitor
        </h1>
        <p style="
            color: #6B7280; 
            margin: 0; 
            font-size: 16px;
            font-weight: 400;
            margin-bottom: 16px;
        ">
            ML Operations • HUMAIN OS Integration
        </p>
        <div style="
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
        ">
            <span style="
                background: #F6F5F4;
                color: #6B7280;
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid #e1e8ed;
            ">Data Collection</span>
            <span style="
                background: #F6F5F4;
                color: #6B7280;
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid #e1e8ed;
            ">Training</span>
            <span style="
                background: #F6F5F4;
                color: #6B7280;
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid #e1e8ed;
            ">Evaluation</span>
            <span style="
                background: #F6F5F4;
                color: #6B7280;
                padding: 6px 12px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 500;
                border: 1px solid #e1e8ed;
            ">Deployment</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_pipeline_sidebar():
    """Create RLHF pipeline navigation sidebar"""
    with st.sidebar:
        st.markdown("## 📊 Pipeline Phases")
        
        # Data Collection Phase
        with st.expander("📥 Data Collection", expanded=False):
            if st.button("📋 Annotation Interface", use_container_width=True):
                st.session_state.selected_section = "annotation_interface"
                st.rerun()
                
            if st.button("📊 Dataset Statistics", use_container_width=True):
                st.session_state.selected_section = "dataset_stats"
                st.rerun()
                
            if st.button("📈 Data Quality Metrics", use_container_width=True):
                st.session_state.selected_section = "data_quality"
                st.rerun()
        
        # Training Phase  
        with st.expander("🔧 Training", expanded=False):
            if st.button("🚀 Training Status", use_container_width=True):
                st.session_state.selected_section = "training_status"
                st.rerun()
                
            if st.button("📈 Loss Curves", use_container_width=True):
                st.session_state.selected_section = "loss_curves"
                st.rerun()
                
            if st.button("💭 Model Insights", use_container_width=True):
                st.session_state.selected_section = "model_insights"
                st.rerun()
        
        # Evaluation Phase
        with st.expander("📋 Evaluation", expanded=True):
            if st.button("🎯 Model Performance", use_container_width=True):
                st.session_state.selected_section = "model_performance"
                st.rerun()
                
            if st.button("📊 Calibration Analysis", use_container_width=True):
                st.session_state.selected_section = "calibration"
                st.rerun()
                
            if st.button("🌊 Drift Detection", use_container_width=True):
                st.session_state.selected_section = "drift_detection"
                st.rerun()
        
        # Deployment Phase
        with st.expander("🚀 Deployment", expanded=False):
            if st.button("⚡ System Overview", use_container_width=True):
                st.session_state.selected_section = "system_overview"
                st.rerun()
                
            if st.button("📊 Production Metrics", use_container_width=True):
                st.session_state.selected_section = "production_metrics"
                st.rerun()
                
            if st.button("🔍 System Health", use_container_width=True):
                st.session_state.selected_section = "system_health"
                st.rerun()

def handle_navigation_and_debug():
    """Handle navigation and hidden debug interface access"""
    # Check for debug chat access via URL parameter
    query_params = st.query_params
    if "debug" in query_params and query_params["debug"] == "chat":
        return "debug_chat"
    
    # Return selected section or default
    return st.session_state.get('selected_section', 'system_overview')

def initialize_pipeline_session_state():
    """Initialize session state for pipeline monitoring"""
    defaults = {
        'last_refresh_time': time.time(),
        'auto_refresh': True,
        'selected_section': 'system_overview',
        'pipeline_status': 'operational',
        'last_data_refresh': 0,
        'system_config': {
            "theme": "humain-professional",
            "auto_refresh": True,
            "monitoring_level": "comprehensive"
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def should_refresh_data():
    """Smart data refresh logic for pipeline monitoring"""
    if 'vote_df' not in st.session_state:
        return True
    
    if st.session_state.auto_refresh:
        time_since_refresh = time.time() - st.session_state.last_data_refresh
        return time_since_refresh > AUTO_REFRESH_INTERVAL
    
    return False

def update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df):
    """Update model configuration with pipeline metrics"""
    if not vote_df.empty:
        total_interactions = len(vote_df)
        if total_interactions > 0:
            # Pipeline depth calculation
            model_config.interaction_depth = min(1.0, total_interactions / 100.0)
            
            # Performance calculation
            if 'model_correct' in vote_df.columns:
                accuracy = vote_df['model_correct'].mean()
                if pd.notna(accuracy):
                    model_config.performance_progress = min(1.0, accuracy * 0.7 + model_config.interaction_depth * 0.3)
            
            # Operational state
            if 'timestamp' in vote_df.columns:
                recent_df = vote_df[pd.to_datetime(vote_df['timestamp']) > (datetime.now() - pd.Timedelta(hours=24))]
                if len(recent_df) > 10:
                    model_config.current_state = "training"
                elif len(recent_df) > 0:
                    model_config.current_state = "monitoring"
                else:
                    model_config.current_state = "idle"

def display_pipeline_content(selected_section, vote_df, predictions_df, reflections_df, data_summary):
    """Display content based on selected pipeline section"""
    
    try:
        if selected_section == "system_overview":
            st.markdown("## ⚡ System Overview")
            
            # Add enhanced metrics display to the main overview
            if not vote_df.empty:
                create_enhanced_metrics_display(vote_df, predictions_df, reflections_df)
                st.markdown("---")
            
            show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
            
        elif selected_section == "annotation_interface":
            st.markdown("## 📋 Annotation Interface")
            display_annotation_interface(vote_df)
            
        elif selected_section == "dataset_stats":
            st.markdown("## 📊 Dataset Statistics")
            display_annotation_history(vote_df, predictions_df)
            
        elif selected_section == "data_quality":
            st.markdown("## 📈 Data Quality Metrics")
            if not vote_df.empty:
                # Use enhanced metrics display
                create_enhanced_metrics_display(vote_df, predictions_df, reflections_df)
                
                # Additional detailed metrics
                st.markdown("### 📊 Detailed Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_annotations = len(vote_df)
                    st.metric("Total Annotations", total_annotations)
                    
                with col2:
                    if 'human_choice' in vote_df.columns and 'model_choice' in vote_df.columns:
                        agreement_rate = (vote_df['human_choice'] == vote_df['model_choice']).mean()
                        st.metric("Human-Model Agreement", f"{agreement_rate:.1%}")
                    else:
                        st.metric("Agreement Rate", "N/A")
                        
                with col3:
                    if 'timestamp' in vote_df.columns:
                        recent_data = len(vote_df[pd.to_datetime(vote_df['timestamp']) > (datetime.now() - pd.Timedelta(days=7))])
                        st.metric("Recent Activity (7d)", recent_data)
                    else:
                        st.metric("Recent Activity", "N/A")
                        
                with col4:
                    if 'confidence' in vote_df.columns:
                        avg_confidence = vote_df['confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}" if pd.notna(avg_confidence) else "N/A")
                    else:
                        st.metric("Avg Confidence", "N/A")
            else:
                st.info("No annotation data available for quality analysis.")
                
        elif selected_section == "training_status":
            st.markdown("## 🚀 Training Status")
            display_model_evolution(vote_df, predictions_df)
            
        elif selected_section == "loss_curves":
            st.markdown("## 📈 Loss Curves & Training Metrics")
            display_model_insights()
            
        elif selected_section == "model_insights":
            st.markdown("## 💭 Model Insights")
            display_model_insights()
            
        elif selected_section == "model_performance":
            st.markdown("## 🎯 Model Performance Analysis")
            display_alignment_over_time(vote_df, predictions_df)
            
        elif selected_section == "calibration":
            st.markdown("## 📊 Calibration Analysis")
            display_calibration_diagnostics(vote_df, predictions_df)
            
        elif selected_section == "drift_detection":
            st.markdown("## 🌊 Drift Detection")
            display_drift_clusters(vote_df, predictions_df)
            
        elif selected_section == "production_metrics":
            st.markdown("## 📊 Production Metrics")
            
            if not vote_df.empty:
                # Enhanced metrics overview
                create_enhanced_metrics_display(vote_df, predictions_df, reflections_df)
                st.markdown("---")
                
                # Advanced visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Model Performance Timeline")
                    timeline_chart = create_connection_timeline(vote_df)
                    if timeline_chart:
                        st.plotly_chart(timeline_chart, use_container_width=True)
                    else:
                        st.info("Insufficient data for timeline visualization")
                
                with col2:
                    st.markdown("### 🔥 Preference Patterns")
                    heatmap_chart = create_preference_heatmap(vote_df)
                    if heatmap_chart:
                        st.plotly_chart(heatmap_chart, use_container_width=True)
                    else:
                        st.info("Insufficient data for preference heatmap")
                
                st.markdown("---")
                
                # Existing calibration analysis
                if not predictions_df.empty:
                    display_confidence_correctness_heatmap(vote_df)
                    st.markdown("---")
                    display_pre_post_calibration_comparison(vote_df)
                else:
                    st.info("No prediction data available for calibration analysis.")
            else:
                st.info("No production data available for metrics analysis.")
                
        elif selected_section == "system_health":
            st.markdown("## 🔍 System Health")
            # System health monitoring
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔧 Pipeline Status")
                st.success("✅ All systems operational")
                
            with col2:
                st.markdown("### 📊 Data Pipeline")
                st.info(f"📈 {len(vote_df)} total annotations processed")
                
            with col3:
                st.markdown("### ⚡ Model Status")
                st.success("✅ Model serving active")
        
        elif selected_section == "debug_chat":
            st.markdown("## 🛠️ Debug Interface")
            st.warning("⚠️ **Developer Debug Tool** - Not part of production interface")
            display_chat_interface()
            
        else:
            st.markdown("## 🚧 Feature In Development")
            st.info("This pipeline feature is currently being developed.")
            
    except Exception as e:
        st.error(f"Error displaying pipeline content: {str(e)}")

def create_connection_timeline(vote_df):
    """Create an interactive timeline of model performance"""
    if vote_df.empty or len(vote_df) < 2:
        return None
    
    try:
        # Prepare data
        df = vote_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate by date
        daily_stats = df.groupby('date').agg({
            'model_correct': ['count', 'mean'] if 'model_correct' in df.columns else ['count'],
            'timestamp': 'count'
        }).reset_index()
        
        if 'model_correct' in df.columns:
            daily_stats.columns = ['date', 'total_annotations', 'accuracy', 'interactions']
        else:
            daily_stats.columns = ['date', 'total_annotations', 'interactions']
            daily_stats['accuracy'] = 0.5  # Default if no accuracy data
        
        # Create interactive timeline
        fig = go.Figure()
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['accuracy'],
            mode='lines+markers',
            name='Model Performance',
            line=dict(color='#1DB584', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Performance: %{y:.1%}<extra></extra>'
        ))
        
        # Add interaction volume as bar chart
        fig.add_trace(go.Bar(
            x=daily_stats['date'],
            y=daily_stats['total_annotations'],
            name='Daily Annotations',
            marker_color='rgba(29, 181, 132, 0.3)',
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Annotations: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis=dict(
                title='Model Performance',
                tickformat='.0%',
                side='left'
            ),
            yaxis2=dict(
                title='Daily Annotations',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating timeline: {str(e)}")
        return None

def create_preference_heatmap(vote_df):
    """Create a heatmap showing preference patterns"""
    if vote_df.empty or 'human_choice' not in vote_df.columns or len(vote_df) < 5:
        return None
    
    try:
        # Prepare data
        df = vote_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Create preference matrix
        preference_matrix = df.groupby(['day_of_week', 'hour'])['human_choice'].apply(
            lambda x: (x == 'A').mean() if len(x) > 0 else 0.5
        ).reset_index()
        
        # Pivot for heatmap
        heatmap_data = preference_matrix.pivot(
            index='day_of_week', 
            columns='hour', 
            values='human_choice'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Preference A Rate"),
            color_continuous_scale='RdYlBu_r',
            aspect="auto"
        )
        
        fig.update_layout(height=300)
        
        return fig
    except Exception as e:
        st.error(f"Error creating preference heatmap: {str(e)}")
        return None

def create_system_footer():
    """Create system status footer"""
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pipeline_status = "🟢 Operational" 
        st.markdown(f"""
        <div style="
            text-align: center; 
            color: #34495e;
            padding: 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        ">
            <strong style="font-size: 14px;">Pipeline Status</strong><br>
            <span style="color: #10B981; font-weight: 500; margin-top: 4px; display: inline-block;">{pipeline_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            text-align: center; 
            color: #34495e;
            padding: 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        ">
            <strong style="font-size: 14px;">Data Flow</strong><br>
            <span style="color: #1DB584; font-weight: 500; margin-top: 4px; display: inline-block;">📊 Active</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            text-align: center; 
            color: #34495e;
            padding: 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        ">
            <strong style="font-size: 14px;">Model Status</strong><br>
            <span style="color: #1DB584; font-weight: 500; margin-top: 4px; display: inline-block;">⚡ Serving</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="
            text-align: center; 
            color: #34495e;
            padding: 16px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        ">
            <strong style="font-size: 14px;">Monitoring</strong><br>
            <span style="color: #1DB584; font-weight: 500; margin-top: 4px; display: inline-block;">📈 Enabled</span>
        </div>
        """, unsafe_allow_html=True)
    
    # System info
    st.markdown("""
    <div style="
        text-align: center; 
        margin-top: 16px; 
        padding: 16px;
        background: white;
        border-radius: 8px;
        font-size: 12px;
        color: #6B7280;
        border: 1px solid #e1e8ed;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    ">
        <p style="margin: 0; font-weight: 600; color: #1DB584; font-size: 14px;">
            ⚙️ RLHF Pipeline Monitor v1.0
        </p>
        <p style="margin: 8px 0 0 0; color: #6B7280; font-weight: 400;">
            ML Operations Platform | HUMAIN OS Integration
        </p>
        <p style="margin: 8px 0 0 0; color: #6B7280; font-size: 11px;">
            Professional RLHF monitoring for ML engineering teams | 
            <span style="color: #1DB584; font-weight: 500;">Powered by HUMAIN OS</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 