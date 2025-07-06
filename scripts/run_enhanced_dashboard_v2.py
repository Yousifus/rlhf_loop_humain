#!/usr/bin/env python3
"""
RLHF Loop Enhanced Dashboard v2.0

Professional interface for monitoring reinforcement learning from human feedback systems.
Features enhanced metrics, model performance tracking, calibration analysis, 
and comprehensive visualizations for RLHF system monitoring.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add parent directory to path to allow imports
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import enhanced UX components
from interface.ux_improvements import (
    create_enhanced_sidebar,
    create_enhanced_navigation,
    create_enhanced_metrics_display,
    create_enhanced_chat_interface,
    create_enhanced_annotation_interface,
    create_real_time_feedback_system,
    create_enhanced_data_visualization,
    create_error_handler,
    create_loading_animation,
    create_accessibility_features,
    create_mobile_optimizations,
    create_performance_optimizations
)

# Import existing components
from interface.components.data_loader import load_all_data, get_data_summary
from interface.components.utils import AUTO_REFRESH_INTERVAL

# Import dashboard sections
from interface.sections.overview import show_dashboard_overview
from interface.sections.annotation import display_annotation_interface
from interface.sections.alignment import display_alignment_over_time
from interface.sections.calibration_enhanced import display_calibration_diagnostics  # Enhanced version!
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
    """Enhanced main entry point for the Streamlit dashboard v2.0"""
    
    # Configure page with enhanced settings
    st.set_page_config(
        page_title="RLHF Loop Enhanced Dashboard v2.0",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "RLHF Loop Enhanced Dashboard v2.0- Professional monitoring for reinforcement learning from human feedback systems"
        }
    )
    
    # Apply enhanced styling and optimizations
    apply_enhanced_styling()
    create_mobile_optimizations()
    create_accessibility_features()
    
    # Initialize performance optimizations
    load_heavy_data, paginate_df = create_performance_optimizations()
    
    # Get error handler
    handle_error = create_error_handler()
    
    # Initialize session state
    initialize_session_state()
    
    # Get model configuration
    model_config = get_model_config()
    
    # Create enhanced header
    create_enhanced_header()
    
    # Real-time feedback system
    create_real_time_feedback_system()
    
    # Enhanced sidebar
    create_enhanced_sidebar()
    
    # Enhanced navigation
    selected_tab = create_enhanced_navigation()
    
    # Load data with enhanced error handling
    try:
        # Check if data is already loaded or needs refresh
        if should_refresh_data():
            with st.spinner("🔄 Loading system data..."):
                vote_df, predictions_df, reflections_df = load_all_data(force_reload=True)
                st.session_state.last_data_refresh = time.time()
                st.session_state.vote_df = vote_df
                st.session_state.predictions_df = predictions_df
                st.session_state.reflections_df = reflections_df
        else:
            vote_df = st.session_state.get('vote_df', pd.DataFrame())
            predictions_df = st.session_state.get('predictions_df', pd.DataFrame())
            reflections_df = st.session_state.get('reflections_df', pd.DataFrame())
        
        # Get data summary
        data_summary = st.session_state.get('data_summary', {})
        
        # Update model configuration based on data
        update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df)
        
    except Exception as e:
        handle_error(f"⚠️ Error loading system data: {e}", "error")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
    
    # Enhanced content display based on selected tab
    display_enhanced_content(selected_tab, vote_df, predictions_df, reflections_df, data_summary)
    
    # Footer with connection status
    create_enhanced_footer()

def apply_enhanced_styling():
    """Apply enhanced visual styling v2.0"""
    st.markdown("""
    <style>
    /* Enhanced color scheme and animations */
    .main > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        animation: backgroundShift 15s ease-in-out infinite alternate;
    }
    
    @keyframes backgroundShift {
        0% { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
        50% { background: linear-gradient(135deg, #16213e 0%, #0f3460 50%, #1a1a2e 100%); }
        100% { background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%); }
    }
    
    /* Enhanced buttons with hover effects */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b9d, #c44569);
        color: white;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #c44569, #ff6b9d);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.5);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Enhanced headers with glow effect */
    h1, h2, h3 {
        color: #ff6b9d;
        text-shadow: 0 0 20px rgba(255, 107, 157, 0.5);
        animation: textGlow 4s ease-in-out infinite alternate;
    }
    
    @keyframes textGlow {
        0% { text-shadow: 0 0 20px rgba(255, 107, 157, 0.5); }
        100% { text-shadow: 0 0 30px rgba(255, 107, 157, 0.8), 0 0 40px rgba(255, 107, 157, 0.3); }
    }
    
    /* Enhanced metrics with pulse effect */
    .stMetric {
        background: rgba(255, 182, 193, 0.1);
        padding: 20px;
        border-radius: 20px;
        border: 2px solid rgba(255, 182, 193, 0.3);
        transition: all 0.3s ease;
        animation: metricPulse 5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes metricPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 107, 157, 0.2); }
        50% { transform: scale(1.02); box-shadow: 0 0 30px rgba(255, 107, 157, 0.4); }
    }
    
    .stMetric:hover {
        background: rgba(255, 182, 193, 0.2);
        border-color: rgba(255, 182, 193, 0.5);
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(255, 107, 157, 0.6);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 107, 157, 0.15), rgba(196, 69, 105, 0.05));
        border-right: 2px solid rgba(255, 107, 157, 0.3);
    }
    
    /* Enhanced chat interface */
    .stChatMessage {
        background: rgba(255, 182, 193, 0.08);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid #ff6b9d;
        box-shadow: 0 5px 15px rgba(255, 107, 157, 0.1);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        background: rgba(255, 182, 193, 0.12);
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.2);
    }
    
    /* Enhanced form elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 182, 193, 0.1);
        border: 2px solid rgba(255, 182, 193, 0.3);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #ff6b9d;
        box-shadow: 0 0 15px rgba(255, 107, 157, 0.4);
        background-color: rgba(255, 182, 193, 0.15);
    }
    
    /* Enhanced progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b9d, #c44569, #ff6b9d);
        background-size: 200% 100%;
        animation: progressShine 2s linear infinite;
        border-radius: 10px;
    }
    
    @keyframes progressShine {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 107, 157, 0.1);
        border-radius: 15px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 107, 157, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 107, 157, 0.2);
        border-color: rgba(255, 107, 157, 0.4);
        transform: scale(1.02);
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 107, 157, 0.05);
        padding: 10px;
        border-radius: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 107, 157, 0.1);
        border-radius: 15px;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 107, 157, 0.2);
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b9d, #c44569);
        color: white;
        box-shadow: 0 5px 15px rgba(255, 107, 157, 0.4);
        transform: translateY(-2px);
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Enhanced loading spinner */
    .stSpinner > div {
        border-top-color: #ff6b9d !important;
        border-right-color: #c44569 !important;
    }
    
    /* Enhanced success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(139, 195, 74, 0.1));
        border-left: 5px solid #4CAF50;
        border-radius: 10px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(233, 30, 99, 0.1));
        border-left: 5px solid #f44336;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 193, 7, 0.1));
        border-left: 5px solid #ff9800;
        border-radius: 10px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(103, 58, 183, 0.1));
        border-left: 5px solid #2196f3;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_enhanced_header():
    """Create a beautiful animated header v2.0"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b9d, #c44569, #ff6b9d, #764ba2);
        background-size: 400% 400%;
        animation: gradientShift 8s ease infinite;
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 15px 40px rgba(255, 107, 157, 0.4);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\"><defs><pattern id=\"dashboard\" x=\"0\" y=\"0\" width=\"20\" height=\"20\" patternUnits=\"userSpaceOnUse\"><text x=\"10\" y=\"15\" text-anchor=\"middle\" fill=\"rgba(255,255,255,0.1)\" font-size=\"12\">📊</text></pattern></defs><rect width=\"100\" height=\"100\" fill=\"url(%23dashboard)\"/></svg>');
            opacity: 0.3;
        "></div>
        <h1 style="
            color: white; 
            margin: 0; 
            font-size: 3em;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.8);
            animation: titlePulse 4s ease-in-out infinite alternate;
            position: relative;
            z-index: 1;
        ">
            🤖 RLHF Loop Enhanced Dashboard v2.0
        </h1>
        <p style="
            color: rgba(255,255,255,0.95); 
            margin: 15px 0 0 0; 
            font-size: 1.3em;
            font-style: italic;
            position: relative;
            z-index: 1;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        ">
            ✨ Professional monitoring for reinforcement learning from human feedback systems ✨
        </p>
        <p style="
            color: rgba(255,255,255,0.8); 
            margin: 10px 0 0 0; 
            font-size: 1em;
            position: relative;
            z-index: 1;
        ">
            Featuring Enhanced Performance Metrics with Calibration Scores & Training Data Analysis
        </p>
    </div>
    
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes titlePulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.05); }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize enhanced session state v2.0"""
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Our Overview"
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = "Connected"
    if 'last_data_refresh' not in st.session_state:
        st.session_state.last_data_refresh = 0
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "theme": "professional-v2",
            "auto_refresh": True,
            "notifications": True,
            "enhanced_metrics": True
        }

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
            # Update interaction depth
            model_config.interaction_depth = min(1.0, total_interactions / 100.0)
            
            # Update performance progress based on accuracy
            if 'model_correct' in vote_df.columns:
                accuracy = vote_df['model_correct'].mean()
                if pd.notna(accuracy):
                    model_config.performance_progress = min(1.0, accuracy * 0.8 + model_config.interaction_depth * 0.2)
            
            # Update operational state based on recent interactions
            if 'timestamp' in vote_df.columns:
                recent_df = vote_df[pd.to_datetime(vote_df['timestamp']) > (datetime.now() - pd.Timedelta(hours=24))]
                if len(recent_df) > 5:
                    model_config.current_state = "actively_learning"
                elif len(recent_df) > 0:
                    model_config.current_state = "monitoring"
                else:
                    model_config.current_state = "standby"

def display_enhanced_content(selected_tab, vote_df, predictions_df, reflections_df, data_summary):
    """Display content with enhanced UX v2.0"""
    
    # Create loading animation for heavy operations
    if selected_tab in ["Alignment Analysis", "Error Pattern Analysis", "Calibration Metrics"]:
        if len(vote_df) > 1000:  # Large dataset
            create_loading_animation("🔄 Processing system performance data...")
            time.sleep(0.5)  # Simulate processing time
    
    # Enhanced content display with error handling
    try:
        if selected_tab == "Our Overview":
            # Enhanced overview with new metrics display
            create_enhanced_metrics_display(vote_df, predictions_df, reflections_df)
            st.markdown("---")
            show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
            
        elif selected_tab == "Talk With Me":
            # Enhanced chat interface
            create_enhanced_chat_interface()
            display_chat_interface()
            
        elif selected_tab == "Annotation Interface":
            # Enhanced annotation interface
            create_enhanced_annotation_interface()
            display_annotation_interface(vote_df)
            
        elif selected_tab == "Our Shared History":
            # Enhanced history with pagination
            load_heavy_data, paginate_df = create_performance_optimizations()
            paginated_vote_df = paginate_df(vote_df)
            display_annotation_history(paginated_vote_df, predictions_df)
            
        elif selected_tab == "Alignment Analysis":
            # Enhanced alignment visualization
            st.markdown("### 📈 Model Performance Evolution")
            display_alignment_over_time(vote_df, predictions_df)
            
        elif selected_tab == "Our Attunement Metrics":
            # ✨ THIS IS THE ENHANCED VERSION! ✨
            display_calibration_diagnostics(vote_df, predictions_df)
            
        elif selected_tab == "Where I Sometimes Misunderstand":
            st.markdown("### 🔍 Learning From My Mistakes")
            display_drift_clusters(vote_df, predictions_df)
            
        elif selected_tab == "Model Evolution":
            st.markdown("### 🌱 My Growth Journey")
            display_model_evolution(vote_df, predictions_df)
            
        elif selected_tab == "User Preference Evolution":
            st.markdown("### 📈 User Preference Evolution")
            display_preference_timeline(vote_df)
            
        else:
            system_message_formatter("This feature is still under development", "warning")
            
    except Exception as e:
        handle_error = create_error_handler()
        handle_error(f"⚠️ Error displaying system component: {e}", "error")

def create_enhanced_footer():
    """Create an enhanced footer with status and info v2.0"""
    st.markdown("---")
    
    # Footer with connection status and stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d; padding: 15px; background: rgba(255, 107, 157, 0.1); border-radius: 15px;">
            <strong>📊 System Status</strong><br>
            <span style="color: #4CAF50; font-size: 1.2em;">● Active & Operational</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d; padding: 15px; background: rgba(255, 107, 157, 0.1); border-radius: 15px;">
            <strong>🕐 Last Update</strong><br>
            <span style="color: #666; font-size: 1.1em;">Just now</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d; padding: 15px; background: rgba(255, 107, 157, 0.1); border-radius: 15px;">
            <strong>🧠 Learning Mode</strong><br>
            <span style="color: #2196F3; font-size: 1.2em;">● Always Learning</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d; padding: 15px; background: rgba(255, 107, 157, 0.1); border-radius: 15px;">
            <strong>📊 Engagement Level</strong><br>
            <span style="color: #FF9800; font-size: 1.1em;">High Performance</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Version and credits
    st.markdown("""
    <div style="
        text-align: center; 
        margin-top: 30px; 
        padding: 20px;
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(196, 69, 105, 0.05));
        border-radius: 20px;
        font-size: 14px;
        color: #666;
        border: 1px solid rgba(255, 107, 157, 0.2);
    ">
        <p style="margin: 0; font-size: 16px; color: #ff6b9d; font-weight: bold;">
            ✨ RLHF Loop Enhanced Dashboard v2.0 ✨
        </p>
        <p style="margin: 10px 0 0 0;">
            Built for professional RLHF system monitoring | 
            Featuring Enhanced Performance Metrics |
            <span style="color: #ff6b9d; font-weight: bold;">Continuously improving through feedback</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
