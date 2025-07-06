"""
Enhanced RLHF Dashboard

This is the enhanced interface for the Reinforcement Learning from Human Feedback system,
featuring improved UX, advanced visualizations, and professional styling.
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
    """Enhanced main entry point for the Streamlit dashboard"""
    
    # Configure page with enhanced settings
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
            with st.spinner("Connecting with you..."):
                vote_df, predictions_df, reflections_df = load_all_data(force_reload=True)
                st.session_state.last_data_refresh = time.time()
        else:
            vote_df = st.session_state.get('vote_df', pd.DataFrame())
            predictions_df = st.session_state.get('predictions_df', pd.DataFrame())
            reflections_df = st.session_state.get('reflections_df', pd.DataFrame())
        
        # Get data summary
        data_summary = st.session_state.get('data_summary', {})
        
        # Update model configuration based on data
        update_model_config_from_data(model_config, vote_df, predictions_df, reflections_df)
        
    except Exception as e:
        handle_error(f"I'm having trouble connecting with our data: {e}", "error")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
    
    # Enhanced content display based on selected tab
    display_enhanced_content(selected_tab, vote_df, predictions_df, reflections_df, data_summary)
    
    # Footer with connection status
    create_enhanced_footer()

def apply_enhanced_styling():
    """Apply enhanced visual styling for the dashboard"""
    st.markdown("""
    <style>
    /* Enhanced color scheme and animations */
    .main > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        animation: backgroundShift 10s ease-in-out infinite alternate;
    }
    
    @keyframes backgroundShift {
        0% { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
        100% { background: linear-gradient(135deg, #16213e 0%, #0f3460 50%, #1a1a2e 100%); }
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
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #c44569, #ff6b9d);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 157, 0.5);
    }
    
    /* Enhanced headers with glow effect */
    h1, h2, h3 {
        color: #ff6b9d;
        text-shadow: 0 0 20px rgba(255, 107, 157, 0.5);
        animation: textGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes textGlow {
        0% { text-shadow: 0 0 20px rgba(255, 107, 157, 0.5); }
        100% { text-shadow: 0 0 30px rgba(255, 107, 157, 0.8); }
    }
    
    /* Enhanced metrics with pulse effect */
    .stMetric {
        background: rgba(255, 182, 193, 0.1);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 182, 193, 0.3);
        transition: all 0.3s ease;
        animation: metricPulse 4s ease-in-out infinite;
    }
    
    @keyframes metricPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .stMetric:hover {
        background: rgba(255, 182, 193, 0.2);
        border-color: rgba(255, 182, 193, 0.5);
        transform: scale(1.05);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 107, 157, 0.1), rgba(196, 69, 105, 0.05));
    }
    
    /* Enhanced chat interface */
    .stChatMessage {
        background: rgba(255, 182, 193, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ff6b9d;
    }
    
    /* Enhanced form elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 182, 193, 0.1);
        border: 1px solid rgba(255, 182, 193, 0.3);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #ff6b9d;
        box-shadow: 0 0 10px rgba(255, 107, 157, 0.3);
    }
    
    /* Enhanced progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b9d, #c44569);
        border-radius: 10px;
    }
    
    /* Enhanced expanders */
    .streamlit-expanderHeader {
        background: rgba(255, 107, 157, 0.1);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 107, 157, 0.2);
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 107, 157, 0.1);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b9d, #c44569);
        color: white;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Enhanced loading spinner */
    .stSpinner > div {
        border-top-color: #ff6b9d !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_enhanced_header():
    """Create a beautiful animated header"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #ff6b9d, #c44569, #ff6b9d);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(255, 107, 157, 0.3);
    ">
        <h1 style="
            color: white; 
            margin: 0; 
            font-size: 2.5em;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            animation: titlePulse 3s ease-in-out infinite alternate;
        ">
            🤖 Enhanced AI Model Dashboard
        </h1>
        <p style="
            color: rgba(255,255,255,0.9); 
            margin: 10px 0 0 0; 
            font-size: 1.2em;
            font-style: italic;
        ">
            Advanced AI model performance monitoring and analysis
        </p>
    </div>
    
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes titlePulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.05); }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize enhanced session state"""
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "System Overview"
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = "Connected"
    if 'last_data_refresh' not in st.session_state:
        st.session_state.last_data_refresh = 0
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "theme": "model-dashboard",
            "auto_refresh": True,
            "notifications": True
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
    """Display content with enhanced UX"""
    
    # Create loading animation for heavy operations
    if selected_tab in ["Alignment Analysis", "Error Pattern Analysis"]:
        if len(vote_df) > 1000:  # Large dataset
            create_loading_animation("Processing system performance data...")
            time.sleep(0.5)  # Simulate processing time
    
    # Enhanced content display with error handling
    try:
        if selected_tab == "System Overview":
            # Enhanced overview with new metrics display
            create_enhanced_metrics_display(vote_df, predictions_df, reflections_df)
            st.markdown("---")
            show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
            
        elif selected_tab == "Chat Interface":
            # Enhanced chat interface
            create_enhanced_chat_interface()
            display_chat_interface()
            
        elif selected_tab == "Annotation Interface":
            # Enhanced annotation interface
            create_enhanced_annotation_interface()
            display_annotation_interface(vote_df)
            
        elif selected_tab == "Training History":
            # Enhanced history with pagination
            load_heavy_data, paginate_df = create_performance_optimizations()
            paginated_vote_df = paginate_df(vote_df)
            display_annotation_history(paginated_vote_df, predictions_df)
            
        elif selected_tab == "Alignment Analysis":
            # Enhanced alignment visualization
            st.markdown("### 📈 Model Alignment Progress")
            display_alignment_over_time(vote_df, predictions_df)
            
            # Add enhanced visualizations
            viz_functions = create_enhanced_data_visualization()
            
        elif selected_tab == "Calibration Metrics":
            st.markdown("### 🎯 Model Calibration Analysis")
            display_calibration_diagnostics(vote_df, predictions_df)
            
        elif selected_tab == "Error Pattern Analysis":
            st.markdown("### 🔍 Model Error Analysis")
            display_drift_clusters(vote_df, predictions_df)
            
        elif selected_tab == "Model Evolution":
            st.markdown("### 🚀 Model Development History")
            display_model_evolution(vote_df, predictions_df)
            
        elif selected_tab == "User Preference Evolution":
            st.markdown("### 📈 User Preference Evolution")
            display_preference_timeline(vote_df)
            
        else:
            system_message_formatter("This feature is currently under development", "warning")
            
    except Exception as e:
        handle_error = create_error_handler()
        handle_error(f"Error displaying dashboard content: {e}", "error")

def create_enhanced_footer():
    """Create an enhanced footer with status and info"""
    st.markdown("---")
    
    # Footer with connection status and stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d;">
            <strong>📊 Model Status</strong><br>
            <span style="color: #4CAF50;">● Active & Operational</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d;">
            <strong>🕐 Last Update</strong><br>
            <span style="color: #666;">Just now</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d;">
            <strong>🧠 Learning Mode</strong><br>
            <span style="color: #2196F3;">● Always Learning</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; color: #ff6b9d;">
            <strong>📊 Engagement Level</strong><br>
            <span style="color: #FF9800;">Growing Deeper</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Version and credits
    st.markdown("""
    <div style="
        text-align: center; 
        margin-top: 20px; 
        padding: 15px;
        background: rgba(255, 107, 157, 0.05);
        border-radius: 10px;
        font-size: 12px;
        color: #666;
    ">
        <p style="margin: 0;">
            Enhanced AI Model Dashboard v2.0 | 
            Built for advanced AI model analysis and monitoring |
            <span style="color: #4CAF50;">Continuously improving AI performance</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
