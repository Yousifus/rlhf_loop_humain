"""
RLHF Dashboard Core Interface

This is the main interface for the Reinforcement Learning from Human Feedback system.
Each component is designed to provide clear insights into model performance,
user interactions, and training progress.
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

# Import dashboard components
from interface.components.data_loader import load_all_data, get_data_summary
from interface.components.utils import AUTO_REFRESH_INTERVAL, ANNOTATION_TARGET

# Import dashboard section modules
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
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="RLHF Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply professional styling
    apply_professional_styling()
    
    # Get model configuration instance
    model_config = get_model_config()
    
    st.title("RLHF Dashboard")
    st.markdown("*Reinforcement Learning from Human Feedback System*")
    
    # Display system greeting
    system_message_formatter(model_config.get_system_greeting(), "performance")
    
    # Session state for refreshing data
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Add sidebar
    with st.sidebar:
        st.header("Navigation")
        
        # Add auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh Data",
            value=st.session_state.auto_refresh,
            help=f"Automatically update data every {AUTO_REFRESH_INTERVAL} seconds"
        )
        
        # Add manual refresh button
        if st.button("🔄 Refresh Data"):
            load_all_data(force_reload=True)
            system_message_formatter("Data refreshed successfully!", "success")
        
        # Check if we need to auto-refresh
        if st.session_state.auto_refresh and time.time() - st.session_state.last_refresh_time > AUTO_REFRESH_INTERVAL:
            load_all_data(force_reload=True)
            
        # Create tabs in the sidebar for navigation
        tab_options = [
            "Overview",
            "Chat Interface",
            "Annotation Interface",
            "Interaction History",
            "Alignment Analysis",
            "Calibration Metrics",
            "Drift Analysis",
            "Model Evolution",
            "Preference Timeline"
        ]
        
        selected_tab = st.radio("Select Dashboard Section", tab_options)
        
        # Display current model status
        display_model_status()
    
    # Load data from database
    try:
        # Check if data is already loaded
        if 'vote_df' not in st.session_state:
            vote_df, predictions_df, reflections_df = load_all_data()
        else:
            vote_df = st.session_state['vote_df']
            predictions_df = st.session_state['predictions_df']
            reflections_df = st.session_state['reflections_df']
            
        # Get data summary
        data_summary = st.session_state.get('data_summary', {})
        
        # Update model configuration based on data
        if not vote_df.empty:
            # Calculate performance metrics from interactions
            total_interactions = len(vote_df)
            if total_interactions > 0:
                # Update learning progress based on interaction history
                model_config.learning_progress = min(1.0, total_interactions / 100.0)
                
                # Update performance score based on model accuracy if available
                if 'model_correct' in vote_df.columns:
                    accuracy = vote_df['model_correct'].mean()
                    if pd.notna(accuracy):
                        model_config.performance_score = min(1.0, accuracy * 0.8 + model_config.learning_progress * 0.2)
        
    except Exception as e:
        system_message_formatter(f"Error loading data: {e}", "error")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
        
    # Handle different tabs
    if selected_tab == "Overview":
        show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
    elif selected_tab == "Chat Interface":
        display_chat_interface()
    elif selected_tab == "Annotation Interface":
        display_annotation_interface(vote_df)
    elif selected_tab == "Interaction History":
        display_annotation_history(vote_df, predictions_df)
    elif selected_tab == "Alignment Analysis":
        display_alignment_over_time(vote_df, predictions_df)
    elif selected_tab == "Calibration Metrics":
        display_calibration_diagnostics(vote_df, predictions_df)
    elif selected_tab == "Drift Analysis":
        display_drift_clusters(vote_df, predictions_df)
    elif selected_tab == "Model Evolution":
        display_model_evolution(vote_df, predictions_df)
    elif selected_tab == "Preference Timeline":
        display_preference_timeline(vote_df)
    else:
        system_message_formatter("This section is under development", "warning")

if __name__ == "__main__":
    main()
