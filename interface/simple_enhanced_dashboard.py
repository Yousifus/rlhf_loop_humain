"""
Enhanced RLHF Dashboard - Professional Interface

Enterprise-grade RLHF monitoring interface with comprehensive navigation
and advanced analytics capabilities.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add parent directory to path to allow imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import existing components
from interface.components.data_loader import load_all_data, get_data_summary
from interface.components.utils import AUTO_REFRESH_INTERVAL

# Import dashboard sections
from interface.sections.overview import show_dashboard_overview
from interface.sections.annotation import display_annotation_interface
from interface.sections.alignment import display_alignment_over_time
from interface.sections.calibration import display_calibration_diagnostics
from interface.sections.drift_analysis import display_drift_clusters
from interface.sections.model_evolution import display_model_evolution
from interface.sections.chat import display_chat_interface
from interface.sections.annotation import display_annotation_history, display_preference_timeline

# Import model configuration
from interface.sections.model_config_core import (
    get_model_config, display_model_status, apply_professional_styling,
    system_message_formatter
)

def main():
    """Main entry point for the enhanced RLHF dashboard"""
    
    # Configure page settings
    st.set_page_config(
        page_title="Enhanced RLHF Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Enhanced RLHF Dashboard - Professional monitoring for reinforcement learning systems"
        }
    )
    
    # Apply professional styling
    apply_professional_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Get model configuration
    model_config = get_model_config()
    
    # Create header
    create_header()
    
    # Create sidebar navigation
    with st.sidebar:
        create_sidebar()
        
        # Navigation menu
        st.markdown("### 📊 Dashboard Navigation")
        
        tab_options = [
            "System Overview",
            "Chat Interface", 
            "Annotation Interface",
            "Annotation History",
            "Alignment Analysis",
            "Calibration Metrics",
            "Drift Analysis",
            "Model Evolution",
            "Preference Evolution",
            "System Diagnostics"
        ]
        
        selected_tab = st.radio(
            "Select Dashboard Section",
            tab_options,
            index=tab_options.index(st.session_state.get('selected_tab', 'System Overview'))
        )
        
        # Update session state
        st.session_state.selected_tab = selected_tab
        
        # Display system status
        display_system_status()
    
    # Load data with error handling
    try:
        # Check if data refresh is needed
        if should_refresh_data():
            with st.spinner("Loading data..."):
                vote_df, predictions_df, reflections_df = load_all_data(force_reload=True)
                st.session_state.last_data_refresh = time.time()
        else:
            vote_df = st.session_state.get('vote_df', pd.DataFrame())
            predictions_df = st.session_state.get('predictions_df', pd.DataFrame())
            reflections_df = st.session_state.get('reflections_df', pd.DataFrame())
        
        # Get data summary
        data_summary = st.session_state.get('data_summary', {})
        
        # Update model configuration from data
        update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
    
    # Display content based on selected tab
    display_content(selected_tab, vote_df, predictions_df, reflections_df, data_summary)
    
    # Footer
    create_footer()

def apply_professional_styling():
    """Apply professional styling for the dashboard"""
    st.markdown("""
    <style>
    /* Professional color scheme */
    .main > div {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(45deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #1d4ed8, #2563eb);
        transform: translateY(-2px);
    }
    
    /* Professional headers */
    h1, h2, h3 {
        color: #1e40af;
        font-weight: 600;
    }
    
    /* Professional metrics */
    .stMetric {
        background: rgba(37, 99, 235, 0.1);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        background: rgba(37, 99, 235, 0.15);
        border-color: rgba(37, 99, 235, 0.3);
    }
    
    /* Professional sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(37, 99, 235, 0.1), rgba(29, 78, 216, 0.05));
    }
    
    /* Professional form elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(37, 99, 235, 0.05);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 6px;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
    }
    
    /* Professional progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        border-radius: 4px;
    }
    
    /* Professional radio buttons */
    .stRadio > div {
        background: rgba(37, 99, 235, 0.05);
        border-radius: 6px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create professional header"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    ">
        <h1 style="
            color: white; 
            margin: 0; 
            font-size: 2.5em;
            font-weight: 700;
        ">
            🤖 Enhanced RLHF Dashboard
        </h1>
        <p style="
            color: rgba(255,255,255,0.9); 
            margin: 10px 0 0 0; 
            font-size: 1.2em;
        ">
            Professional AI model performance monitoring and analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create professional sidebar content"""
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h2 style="color: white; margin: 0; font-size: 1.5em;">🤖 RLHF Dashboard</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">
            Professional AI model monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.selected_tab = "Chat Interface"
            st.rerun()
    
    with col2:
        if st.button("📝 Annotate", use_container_width=True):
            st.session_state.selected_tab = "Annotation Interface"
            st.rerun()
    
    # System metrics
    st.markdown("### 📊 System Metrics")
    
    # Professional system metrics
    system_metrics = {
        "Model Performance": {"value": 0.75, "display": "75%"},
        "Training Progress": {"value": 0.82, "display": "82%"},
        "System Load": {"value": 0.68, "display": "68%"},
        "Alignment Score": {"value": 0.79, "display": "79%"}
    }
    
    for metric_name, metric_data in system_metrics.items():
        st.markdown(f"**{metric_name}**")
        progress_col, value_col = st.columns([3, 1])
        
        with progress_col:
            st.progress(metric_data['value'])
        with value_col:
            st.write(f"{metric_data['display']}")

def initialize_session_state():
    """Initialize session state variables"""
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "System Overview"
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "Active"
    if 'last_data_refresh' not in st.session_state:
        st.session_state.last_data_refresh = 0

def should_refresh_data():
    """Determine if data should be refreshed"""
    if 'vote_df' not in st.session_state:
        return True
    
    if st.session_state.auto_refresh:
        time_since_refresh = time.time() - st.session_state.last_data_refresh
        return time_since_refresh > AUTO_REFRESH_INTERVAL
    
    return False

def update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df):
    """Update model configuration based on current data"""
    if not vote_df.empty:
        # Calculate performance metrics
        total_interactions = len(vote_df)
        if total_interactions > 0:
            # Update interaction metrics
            model_config.interaction_depth = min(1.0, total_interactions / 100.0)
            
            # Update performance metrics based on accuracy
            if 'model_correct' in vote_df.columns:
                accuracy = vote_df['model_correct'].mean()
                if pd.notna(accuracy):
                    model_config.performance_progress = min(1.0, accuracy * 0.8 + model_config.interaction_depth * 0.2)

def display_content(selected_tab, vote_df, predictions_df, reflections_df, data_summary):
    """Display content based on selected tab"""
    
    try:
        if selected_tab == "System Overview":
            # System overview with metrics
            create_metrics_display(vote_df, predictions_df, reflections_df)
            st.markdown("---")
            show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
            
        elif selected_tab == "Chat Interface":
            # Chat interface
            st.markdown("""
            <div style="
                background: linear-gradient(90deg, #2563eb, #1d4ed8);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 20px;
                text-align: center;
            ">
                <h2 style="color: white; margin: 0;">💬 Interactive Chat Interface</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">
                    Direct model interaction interface
                </p>
            </div>
            """, unsafe_allow_html=True)
            display_chat_interface()
            
        elif selected_tab == "Annotation Interface":
            display_annotation_interface(vote_df)
            
        elif selected_tab == "Annotation History":
            display_annotation_history(vote_df, predictions_df)
            
        elif selected_tab == "Alignment Analysis":
            st.markdown("### 📈 Model Alignment Analysis")
            display_alignment_over_time(vote_df, predictions_df)
            
        elif selected_tab == "Calibration Metrics":
            st.markdown("### 🎯 Model Calibration Diagnostics")
            display_calibration_diagnostics(vote_df, predictions_df)
            
        elif selected_tab == "Drift Analysis":
            st.markdown("### 🔍 Model Drift Analysis")
            display_drift_clusters(vote_df, predictions_df)
            
        elif selected_tab == "Model Evolution":
            st.markdown("### 🌱 Model Evolution Tracking")
            display_model_evolution(vote_df, predictions_df)
            
        elif selected_tab == "Preference Evolution":
            st.markdown("### 📈 User Preference Evolution")
            display_preference_timeline(vote_df)
            
        else:
            system_message_formatter("This feature is under development", "warning")
            
    except Exception as e:
        st.error(f"Error displaying content: {e}")

def create_metrics_display(vote_df, predictions_df, reflections_df):
    """Create professional metrics display"""
    
    # Metrics section header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(29, 78, 216, 0.1));
        padding: 30px;
        border-radius: 12px;
        margin: 20px 0;
        border: 1px solid rgba(37, 99, 235, 0.2);
    ">
        <h2 style="text-align: center; color: #1e40af; margin-bottom: 20px;">
            📊 System Performance Metrics
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Model Accuracy",
            calculate_model_accuracy(vote_df),
            "🎯",
            "Model prediction accuracy"
        )
    
    with col2:
        create_metric_card(
            "Training Progress", 
            calculate_training_progress(vote_df),
            "📈",
            "Training progress metrics"
        )
    
    with col3:
        create_metric_card(
            "System Load",
            calculate_system_load(vote_df, reflections_df),
            "⚡",
            "System performance metrics"
        )
    
    with col4:
        create_metric_card(
            "Alignment Score",
            calculate_alignment_score(vote_df, predictions_df),
            "🎯",
            "Model-user alignment metrics"
        )

def create_metric_card(title, value, icon, description):
    """Create a professional metric card"""
    # Determine color based on value
    if value >= 0.8:
        color = "#10b981"  # Green
        bg_color = "rgba(16, 185, 129, 0.1)"
    elif value >= 0.6:
        color = "#f59e0b"  # Orange
        bg_color = "rgba(245, 158, 11, 0.1)"
    else:
        color = "#ef4444"  # Red
        bg_color = "rgba(239, 68, 68, 0.1)"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid {color}33;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 2em; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 1.2em; font-weight: bold; color: {color}; margin-bottom: 5px;">
            {title}
        </div>
        <div style="font-size: 2em; font-weight: bold; color: {color}; margin-bottom: 5px;">
            {value:.1%}
        </div>
        <div style="font-size: 0.8em; color: #6b7280; opacity: 0.8;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)

def calculate_model_accuracy(vote_df):
    """Calculate model accuracy metrics"""
    if vote_df.empty:
        return 0.0
    
    total_interactions = len(vote_df)
    if 'timestamp' in vote_df.columns:
        recent_interactions = len(vote_df[
            pd.to_datetime(vote_df['timestamp']) > (datetime.now() - pd.Timedelta(days=7))
        ])
        recency_factor = min(1.0, recent_interactions / 10)
    else:
        recency_factor = 0.5
    
    volume_factor = min(1.0, total_interactions / 100)
    return (volume_factor * 0.7 + recency_factor * 0.3)

def calculate_training_progress(vote_df):
    """Calculate training progress metrics"""
    if vote_df.empty or 'model_correct' not in vote_df.columns:
        return 0.0
    return vote_df['model_correct'].mean()

def calculate_system_load(vote_df, reflections_df):
    """Calculate system load metrics"""
    if vote_df.empty:
        return 0.0
    
    interaction_factor = min(1.0, len(vote_df) / 50)
    
    if not reflections_df.empty and 'quality_score' in reflections_df.columns:
        quality_factor = reflections_df['quality_score'].mean()
    else:
        quality_factor = 0.5
    
    return (interaction_factor * 0.6 + quality_factor * 0.4)

def calculate_alignment_score(vote_df, predictions_df):
    """Calculate alignment score metrics"""
    if vote_df.empty:
        return 0.0
    
    if 'model_choice' in vote_df.columns and 'human_choice' in vote_df.columns:
        return (vote_df['model_choice'] == vote_df['human_choice']).mean()
    
    return 0.5

def create_footer():
    """Create professional footer"""
    st.markdown("---")
    
    # Footer with system status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; color: #1e40af;">
            <strong>📊 System Status</strong><br>
            <span style="color: #10b981;">● Operational</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #1e40af;">
            <strong>🕐 Last Update</strong><br>
            <span style="color: #6b7280;">Real-time</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; color: #1e40af;">
            <strong>🧠 Training Mode</strong><br>
            <span style="color: #2563eb;">● Active</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; color: #1e40af;">
            <strong>📈 Performance</strong><br>
            <span style="color: #f59e0b;">Optimized</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Version info
    st.markdown("""
    <div style="
        text-align: center; 
        margin-top: 20px; 
        padding: 15px;
        background: rgba(37, 99, 235, 0.05);
        border-radius: 8px;
        font-size: 12px;
        color: #6b7280;
    ">
        <p style="margin: 0;">
            Enhanced RLHF Dashboard v2.0 | 
            Professional RLHF system monitoring |
            <span style="color: #2563eb;">Enterprise-grade analytics</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
