"""
RLHF Attunement Dashboard

A Streamlit dashboard for the RLHF system, providing capabilities for:
- Annotation of completions
- Visualization of annotation history
- Analysis of model alignment
- Tracking of user preferences over time
"""

import json
import os
import re
import sys
import time
import uuid
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Add new imports for clustering and dimensionality reduction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Add parent directory to path to allow imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
sys.path.append(project_root)

# Import the database module
from utils.database import get_database, RLHFDatabase
# Import the API client
from utils.api_client import get_api_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up Streamlit page configuration
st.set_page_config(
    page_title="RLHF Attunement Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state for refreshing data
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Constants
AUTO_REFRESH_INTERVAL = 60  # seconds
ANNOTATION_TARGET = 500  # Target number of annotations
TARGET_ACCURACY = 0.85  # Target model accuracy

def create_time_slider(data_df, timestamp_col='timestamp', label="Select Time Range"):
    """Create a time range slider for filtering data"""
    if data_df.empty or timestamp_col not in data_df.columns:
        return data_df
    
    # Ensure timestamp column is in datetime format
    if data_df[timestamp_col].dtype != 'datetime64[ns]':
        try:
            data_df[timestamp_col] = pd.to_datetime(data_df[timestamp_col])
        except Exception as e:
            st.warning(f"Error converting timestamps for slider: {e}")
            return data_df
    
    # Get min and max dates from the data
    min_date = data_df[timestamp_col].min().date()
    max_date = data_df[timestamp_col].max().date()
    
    # Set default range (last 7 days or full range if shorter)
    default_start = max(min_date, max_date - timedelta(days=7))
    
    # Create date range slider
    date_range = st.slider(
        label,
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD"
    )
    
    start_time, end_time = date_range
    
    # Convert start and end dates to datetime (inclusive of full days)
    start_datetime = pd.Timestamp(start_time)
    end_datetime = pd.Timestamp(end_time) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Filter and return the data
    return filter_by_time_range(data_df, start_datetime, end_datetime, timestamp_col)

def filter_by_time_range(data_df, start_time, end_time, timestamp_col='timestamp'):
    """Filter dataframe by time range"""
    if data_df.empty or timestamp_col not in data_df.columns:
        return data_df
    
    if start_time is None or end_time is None:
        return data_df
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = data_df.copy()
    
    # Ensure timestamp column is in datetime format
    if df_copy[timestamp_col].dtype != 'datetime64[ns]':
        try:
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
        except Exception as e:
            st.warning(f"Error converting timestamps for filtering: {e}")
            return data_df
    
    try:
        # Ensure start_time and end_time are datetime objects
        if isinstance(start_time, pd.Timestamp):
            start_time = start_time.to_pydatetime()
        if isinstance(end_time, pd.Timestamp):
            end_time = end_time.to_pydatetime()
        
        # Filter by time range
        filtered_df = df_copy[(df_copy[timestamp_col] >= start_time) & (df_copy[timestamp_col] <= end_time)]
        return filtered_df
    except Exception as e:
        st.warning(f"Error filtering by time range: {e}")
        return data_df

def format_timestamp(ts):
    """Format timestamp for display"""
    if ts is None:
        return "N/A"
    
    try:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        
        # Format datetime object
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)

def plot_accuracy_over_time(data_df, window_size=10, calibration_history=None):
    """Plot model prediction accuracy over time with rolling average and calibration markers"""
    if data_df.empty or 'timestamp' not in data_df.columns or 'is_prediction_correct' not in data_df.columns:
        return go.Figure().update_layout(title="No data available for accuracy plot")
    
    # Sort by timestamp
    df = data_df.sort_values('timestamp')
    
    # Calculate rolling accuracy
    df['rolling_accuracy'] = df['is_prediction_correct'].astype(int).rolling(window=window_size).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for individual predictions
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['is_prediction_correct'].astype(int),
            mode='markers',
            name='Predictions',
            marker=dict(
                size=8,
                color=df['is_prediction_correct'].map({True: 'green', False: 'red'}),
                opacity=0.6
            )
        )
    )
    
    # Add line plot for rolling accuracy
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['rolling_accuracy'],
            mode='lines',
            name=f'Rolling Accuracy (window={window_size})',
            line=dict(width=3, color='blue')
        )
    )
    
    # Add calibration event markers if available
    if calibration_history and 'history' in calibration_history:
        calibration_events = []
        for event in calibration_history['history']:
            if 'timestamp' in event:
                try:
                    event_time = pd.to_datetime(event['timestamp'])
                    calibration_events.append({
                        'timestamp': event_time,
                        'method': event.get('method', 'unknown'),
                        'ece_before': event.get('ece_before', 'N/A'),
                        'ece_after': event.get('ece_after', 'N/A')
                    })
                except Exception as e:
                    logger.warning(f"Error processing calibration event timestamp: {e}")
        
        if calibration_events:
            # Add vertical lines for calibration events
            for event in calibration_events:
                # Convert timestamp to string to avoid pandas Timestamp arithmetic operations
                # which are no longer supported in newer pandas versions
                event_time_str = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Add vertical line at calibration time
                fig.add_shape(
                    type="line",
                    x0=event_time_str,
                    y0=0,
                    x1=event_time_str,
                    y1=1,
                    line=dict(
                        color="green",
                        width=2,
                        dash="dash",
                    )
                )
                
                # Add text annotation for calibration
                fig.add_annotation(
                    x=event_time_str,
                    y=1.05,
                    text=f"Calibration: {event['method']}<br>ECE: {event['ece_before']} → {event['ece_after']}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                )
    
    # Update layout
    fig.update_layout(
        title=f"Model Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy",
        yaxis=dict(range=[-0.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def plot_confidence_calibration(data_df, num_bins=10):
    """Plot confidence calibration curve"""
    # Import the replacement function from visualizations module
    from utils.dashboard.visualizations import plot_reliability_diagram
    
    # Use the new implementation
    return plot_reliability_diagram(data_df, num_bins=num_bins, pre_calibration=True)

def plot_error_types(data_df):
    """Plot distribution of error types"""
    if data_df.empty or 'prediction_error_type' not in data_df.columns:
        return go.Figure().update_layout(title="No error type data available")
    
    # Count error types
    error_counts = data_df['prediction_error_type'].value_counts().reset_index()
    error_counts.columns = ['error_type', 'count']
    
    # Create pie chart
    fig = px.pie(
        error_counts,
        values='count',
        names='error_type',
        title="Distribution of Error Types",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Error Type"
    )
    
    return fig

def display_checkpoints_timeline(checkpoints):
    """Display model checkpoints as a timeline"""
    if not checkpoints:
        st.warning("No model checkpoint data available")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(checkpoints)
    
    # Ensure we have necessary columns
    if 'timestamp' not in df.columns or 'version' not in df.columns:
        st.warning("Checkpoint data missing required fields")
        return
    
    # Ensure timestamp is datetime and version is string
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['version'] = df['version'].astype(str)
        
        # Convert numeric columns for hover data
        if 'training_samples' in df.columns:
            df['training_samples'] = pd.to_numeric(df['training_samples'], errors='coerce')
        if 'accuracy' in df.columns:
            df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        if 'calibration_error' in df.columns:
            df['calibration_error'] = pd.to_numeric(df['calibration_error'], errors='coerce')
    except Exception as e:
        st.warning(f"Error processing checkpoint data: {e}")
        # Continue with unprocessed data
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create a timeline
    fig = px.line(
        df,
        x='timestamp',
        y='version',
        markers=True,
        hover_data=['training_samples', 'accuracy', 'calibration_error']
    )
    
    # Update layout
    fig.update_layout(
        title="Model Checkpoint Timeline",
        xaxis_title="Date",
        yaxis_title="Model Version"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show checkpoint details in a table
    with st.expander("View Checkpoint Details"):
        # Create a copy with string columns to avoid Arrow serialization issues
        display_df = df.copy()
        
        # Convert columns that might cause serialization issues to strings
        for col in display_df.columns:
            if display_df[col].dtype == 'object' or col == 'timestamp':
                display_df[col] = display_df[col].astype(str)
        
        st.dataframe(display_df)

def plot_drift_clusters(drift_clusters, reflection_data):
    """Plot drift clusters using Plotly"""
    # Import the enhanced version with default options
    from utils.dashboard.visualizations import plot_enhanced_drift_clusters
    return plot_enhanced_drift_clusters(drift_clusters, reflection_data, use_umap=False)

def plot_human_ai_agreement(vote_logs):
    """Plot agreement between human preferences and AI predictions over time"""
    if vote_logs.empty or 'is_model_vote' not in vote_logs.columns:
        return go.Figure().update_layout(title="No data available for agreement plot")
    
    # Make a copy and ensure timestamp is in datetime format
    df = vote_logs.copy()
    if df['timestamp'].dtype != 'datetime64[ns]':
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            st.warning(f"Error converting timestamps for agreement plot: {e}")
            return go.Figure().update_layout(title="Error processing timestamps")
    
    # Filter to include only human votes (non-model votes)
    human_votes = df[df['is_model_vote'] == False].copy()
    
    if human_votes.empty:
        return go.Figure().update_layout(title="No human votes available for comparison")
    
    # Extract needed columns
    if 'prompt_id' not in human_votes.columns or 'choice' not in human_votes.columns:
        return go.Figure().update_layout(title="Vote logs missing required preference data")
    
    # For each human vote, find the corresponding model prediction for the same prompt
    results = []
    for _, human_vote in human_votes.iterrows():
        prompt_id = human_vote.get('prompt_id')
        pair_id = human_vote.get('pair_id', None)
        
        # Find model prediction for the same prompt/pair
        model_votes = df[(df['is_model_vote'] == True) & 
                         (df['prompt_id'] == prompt_id)]
        
        # If pair_id is available, use it for more precise matching
        if pair_id is not None and 'pair_id' in df.columns:
            model_votes = model_votes[model_votes['pair_id'] == pair_id]
        
        if not model_votes.empty:
            # Get the most recent model prediction before human vote
            model_vote = model_votes[model_votes['timestamp'] <= human_vote['timestamp']].sort_values('timestamp', ascending=False).iloc[0]
            
            # Compare model and human choice
            human_choice = human_vote.get('choice')
            model_choice = model_vote.get('choice')
            
            # Record the agreement data point
            results.append({
                'timestamp': human_vote['timestamp'],
                'prompt_id': prompt_id,
                'human_choice': human_choice,
                'model_choice': model_choice,
                'agreement': human_choice == model_choice,
                'model_confidence': model_vote.get('confidence', 0.5)
            })
    
    if not results:
        return go.Figure().update_layout(title="No matching model-human pairs found")
    
    # Convert to DataFrame
    agreement_df = pd.DataFrame(results)
    
    # Sort by timestamp
    agreement_df = agreement_df.sort_values('timestamp')
    
    # Add rolling agreement rate
    window_size = min(10, len(agreement_df))
    agreement_df['rolling_agreement'] = agreement_df['agreement'].astype(int).rolling(window=window_size).mean()
    
    # Convert timestamps to strings for Plotly
    agreement_df['timestamp_str'] = agreement_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create figure for agreement rate over time
    fig = go.Figure()
    
    # Add scatter plot for individual agreements
    fig.add_trace(
        go.Scatter(
            x=agreement_df['timestamp_str'],
            y=agreement_df['agreement'].astype(int),
            mode='markers',
            name='Agreement',
            marker=dict(
                size=8,
                color=agreement_df['agreement'].map({True: 'green', False: 'red'}),
                opacity=0.6
            )
        )
    )
    
    # Add line plot for rolling agreement rate
    fig.add_trace(
        go.Scatter(
            x=agreement_df['timestamp_str'],
            y=agreement_df['rolling_agreement'],
            mode='lines',
            name=f'Rolling Agreement (window={window_size})',
            line=dict(width=3, color='blue')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Human-AI Preference Agreement Over Time",
        xaxis_title="Date",
        yaxis_title="Agreement Rate",
        yaxis=dict(range=[-0.1, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def display_preference_timeline(vote_df):
    """Display user preference timeline with detailed RLHF metrics"""
    if vote_df.empty:
        st.warning("No vote log data available")
        return
    
    # Ensure we have necessary columns
    if 'timestamp' not in vote_df.columns:
        st.warning("Vote log data missing required timestamp field")
        return
    
    # Create tabs for different visualizations
    timeline_tab, metrics_tab, agreement_tab, details_tab = st.tabs(["Timeline", "Metrics", "Human-AI Agreement", "Details"])
    
    with timeline_tab:
        # Group by day and count votes
        vote_df['date'] = vote_df['timestamp'].dt.date
        daily_votes = vote_df.groupby('date', observed=True).size().reset_index(name='vote_count')
        
        # Create a bar chart for daily votes
        fig = px.bar(
            daily_votes,
            x='date',
            y='vote_count',
            title="Daily Annotation Volume",
            color_discrete_sequence=['#2D87BB']  # Blue color for consistency
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Annotations",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with metrics_tab:
        # Check if we have model prediction data
        has_model_data = all(col in vote_df.columns for col in ['is_model_vote', 'confidence'])
        
        if has_model_data:
            # Filter to include only explicit model votes
            model_votes = vote_df[vote_df['is_model_vote'] == True].copy()
            
            if not model_votes.empty:
                # Group by date and calculate success rate
                model_votes['date'] = model_votes['timestamp'].dt.date
                
                # Create metrics for model performance
                col1, col2, col3 = st.columns(3)
                
                # Overall accuracy - if we can determine it
                if 'is_prediction_correct' in model_votes.columns:
                    accuracy = model_votes['is_prediction_correct'].mean()
                    col1.metric("Model Accuracy", f"{accuracy:.2%}")
                
                # Average confidence
                avg_confidence = model_votes['confidence'].mean()
                col2.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Number of predictions
                col3.metric("Total Predictions", f"{len(model_votes)}")
                
                # Plot confidence and accuracy over time if we have enough data
                if len(model_votes) >= 5:
                    # Group by date
                    daily_metrics = model_votes.groupby('date', observed=True).agg({
                        'confidence': 'mean',
                        'is_prediction_correct': 'mean' if 'is_prediction_correct' in model_votes.columns else lambda x: float('nan')
                    }).reset_index()
                    
                    # Create time series plot for confidence
                    fig = px.line(
                        daily_metrics, 
                        x='date', 
                        y=['confidence'] + (['is_prediction_correct'] if 'is_prediction_correct' in model_votes.columns else []),
                        title="Model Confidence & Accuracy Over Time",
                        labels={
                            'date': 'Date',
                            'value': 'Score',
                            'variable': 'Metric'
                        },
                        color_discrete_map={
                            'confidence': '#FFA500',  # Orange
                            'is_prediction_correct': '#2D87BB'  # Blue
                        }
                    )
                    
                    # Update layout
                    fig.update_layout(
                        hovermode="x unified",
                        yaxis_tickformat='.0%'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Error types distribution if available
                if 'prediction_error_type' in model_votes.columns:
                    error_counts = model_votes['prediction_error_type'].value_counts().reset_index()
                    error_counts.columns = ['Error Type', 'Count']
                    
                    fig = px.pie(
                        error_counts, 
                        values='Count', 
                        names='Error Type',
                        title="Distribution of Prediction Error Types",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No model prediction data available yet")
            else:
                st.info("Model prediction data not available in vote logs")
    
    with agreement_tab:
        st.subheader("Human vs AI Preference Alignment")
        
        # Add description of what this tab shows
        st.markdown("""
            This visualization shows how well the AI model's preferences align with human annotators.
            Each point represents a human annotation and whether the model predicted the same preference.
        """)
        
        # Plot human-AI agreement over time
        agreement_fig = plot_human_ai_agreement(vote_df)
        st.plotly_chart(agreement_fig, use_container_width=True)
        
        # Add additional insights if available
        if 'is_model_vote' in vote_df.columns and 'confidence' in vote_df.columns:
            # Get human and model votes
            human_votes = vote_df[vote_df['is_model_vote'] == False]
            model_votes = vote_df[vote_df['is_model_vote'] == True]
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            # Count of human annotations
            col1.metric("Human Annotations", f"{len(human_votes)}")
            
            # Count of model predictions
            col2.metric("Model Predictions", f"{len(model_votes)}")
            
            # Add some analysis on agreement rates if available
            if not human_votes.empty and not model_votes.empty and 'choice' in human_votes.columns:
                # Attempt to match human and model votes
                matched_pairs = []
                for _, human_vote in human_votes.iterrows():
                    prompt_id = human_vote.get('prompt_id')
                    pair_id = human_vote.get('pair_id', None)
                    
                    # Find corresponding model vote
                    matching_votes = model_votes[(model_votes['prompt_id'] == prompt_id)]
                    if pair_id is not None and 'pair_id' in model_votes.columns:
                        matching_votes = matching_votes[matching_votes['pair_id'] == pair_id]
                    
                    if not matching_votes.empty:
                        model_vote = matching_votes.iloc[0]
                        matched_pairs.append((human_vote.get('choice'), model_vote.get('choice')))
                
                # Calculate agreement rate
                if matched_pairs:
                    agreement_rate = sum(h == m for h, m in matched_pairs) / len(matched_pairs)
                    col3.metric("Agreement Rate", f"{agreement_rate:.2%}")
                    
                    # Add some explanation
                    if agreement_rate > 0.8:
                        st.success("The model shows strong alignment with human preferences.")
                    elif agreement_rate > 0.6:
                        st.info("The model shows moderate alignment with human preferences.")
                    else:
                        st.warning("The model shows weak alignment with human preferences.")
    
    with details_tab:
        # Show vote details in a table with most recent votes first
        st.subheader("Recent Annotations")
        
        # Clean up display columns
        display_cols = [col for col in vote_df.columns if col not in ['is_model_vote', 'pair_id']]
        
        # Create a downloadable version of the data
        csv = vote_df[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="rlhf_annotations.csv",
            mime="text/csv"
        )
        
        # Display the table with the most recent votes first
        st.dataframe(
            vote_df[display_cols].sort_values('timestamp', ascending=False).head(100),
            use_container_width=True
        )

def generate_ai_prompts(count=5, domains=None, custom_domains=None):
    """Generate AI-specific prompts for RLHF annotation"""
    if domains is None:
        domains = ["AI ethics", "Machine learning", "Computer vision", "Natural language processing", "Robotics"]
    
    # System prompt to guide the AI model
    system_prompt = """
    You are a prompt generator for an RLHF (Reinforcement Learning from Human Feedback) system. 
    Create diverse, interesting, and challenging prompts for language models.
    
    Each prompt should follow this JSON format:
    {
        "id": "unique_id",
        "text": "The actual prompt text",
        "metadata": {
            "domain": "The subject area or domain",
            "difficulty": "easy|medium|hard",
            "type": "analytical|creative|factual|ethical",
            "expected_tokens": approximate number of tokens for a good response
        }
    }
    
    Ensure the prompts are diverse in terms of:
    1. Type (analytical questions, creative writing, factual explanations, ethical considerations)
    2. Difficulty (mix of easy, medium, and hard questions)
    3. Subject matter (different aspects of specified domains)
    4. Length (some prompts should elicit short responses, others longer ones)
    """
    
    # Template categories
    templates = {
        "explanation": [
            "Explain {topic} in simple terms.",
            "How would you describe {topic} to a 10-year-old?",
            "What is the relationship between {topic} and {related_topic}?",
            "Compare and contrast {topic} and {related_topic}.",
            "What are the key principles of {topic}?"
        ],
        "creative": [
            "Write a short poem about {topic}.",
            "Create a metaphor that explains {topic}.",
            "If {topic} was a person, what would they be like?",
            "Tell a short story that illustrates the concept of {topic}.",
            "Write a dialogue between {topic} and {related_topic}."
        ],
        "analysis": [
            "What are the ethical implications of {topic}?",
            "How might {topic} change in the next 10 years?",
            "What are the main criticisms of {topic}?",
            "How has {topic} evolved over time?",
            "What are the practical applications of {topic}?"
        ]
    }
    
    # Topics and related topics
    topics = {
        "machine learning": ["artificial intelligence", "data science", "neural networks", "statistics"],
        "artificial intelligence": ["machine learning", "robotics", "natural language processing", "computer vision"],
        "deep learning": ["neural networks", "backpropagation", "representation learning", "AI"],
        "reinforcement learning": ["reward systems", "agent-based learning", "game theory", "decision making"],
        "natural language processing": ["computational linguistics", "text analysis", "speech recognition", "language models"],
        "computer vision": ["image processing", "object detection", "facial recognition", "scene understanding"],
        "ethics in AI": ["bias", "fairness", "transparency", "accountability"],
        "quantum computing": ["superposition", "entanglement", "qubits", "cryptography"],
        "blockchain": ["cryptocurrency", "distributed ledgers", "smart contracts", "decentralization"],
        "internet of things": ["connected devices", "smart homes", "sensors", "automation"]
    }
    
    prompts = []
    
    # Generate unique prompts
    for i in range(count):
        # Select random category and template
        category = random.choice(list(templates.keys()))
        template = random.choice(templates[category])
        
        # Select random topic and related topic
        topic = random.choice(list(topics.keys()))
        related_topic = random.choice(topics[topic])
        
        # Format the template
        prompt_text = template.format(topic=topic, related_topic=related_topic)
        
        # Create prompt object
        prompt = {
            "id": f"generated_prompt_{i+1}_{uuid.uuid4().hex[:6]}",
            "text": prompt_text,
            "metadata": {
                "category": category,
                "topic": topic,
                "related_topic": related_topic,
                "generated": True
            }
        }
        
        prompts.append(prompt)
    
    return prompts

def generate_prompts(count=5):
    """Generate simple template-based prompts"""
    template_types = {
        "explanation": [
            "Explain {topic} in simple terms.",
            "How does {topic} work?",
            "What are the key components of {topic}?",
            "Describe the process of {topic}.",
            "What is the importance of {topic}?"
        ],
        "comparison": [
            "Compare and contrast {topic_a} and {topic_b}.",
            "What are the similarities and differences between {topic_a} and {topic_b}?",
            "How does {topic_a} differ from {topic_b}?",
            "In what ways are {topic_a} and {topic_b} similar?",
            "Which is better for {context}: {topic_a} or {topic_b}?"
        ],
        "application": [
            "How is {topic} applied in {field}?",
            "What are practical applications of {topic}?",
            "How can {topic} be used to solve problems in {field}?",
            "Give examples of {topic} being used in real-world scenarios.",
            "How might {topic} change {field} in the future?"
        ],
        "critique": [
            "What are the limitations of {topic}?",
            "What ethical concerns surround {topic}?",
            "What are common criticisms of {topic}?",
            "How might {topic} be improved?",
            "What are the risks associated with {topic}?"
        ],
        "creative": [
            "Write a short story involving {topic}.",
            "Create a dialogue that explains {topic}.",
            "Write a poem about {topic}.",
            "If {topic} was a person, describe their personality.",
            "Write a marketing slogan for {topic}."
        ]
    }
    
    topics = {
        "technology": [
            "artificial intelligence", "machine learning", "blockchain", 
            "virtual reality", "quantum computing", "cloud computing",
            "internet of things", "robotics", "5G", "cybersecurity"
        ],
        "science": [
            "gene editing", "renewable energy", "climate change", 
            "space exploration", "neuroscience", "immunology",
            "quantum physics", "materials science", "cosmology", "biodiversity"
        ],
        "philosophy": [
            "existentialism", "ethics", "epistemology", "consciousness", 
            "free will", "moral relativism", "humanism", "nihilism",
            "determinism", "utilitarianism"
        ],
        "society": [
            "social media", "remote work", "digital privacy", 
            "economic inequality", "education", "healthcare",
            "democracy", "urbanization", "globalization", "sustainability"
        ]
    }
    
    fields = [
        "healthcare", "education", "finance", "entertainment", 
        "transportation", "agriculture", "manufacturing", "energy",
        "retail", "defense", "communications", "public policy"
    ]
    
    prompts = []
    
    for i in range(count):
        # Select a random template type and template
        template_type = random.choice(list(template_types.keys()))
        template = random.choice(template_types[template_type])
        
        # Select topic category and specific topic(s)
        category = random.choice(list(topics.keys()))
        
        if "{topic_a}" in template and "{topic_b}" in template:
            # For comparison templates, select two different topics
            topic_a = random.choice(topics[category])
            topic_b = random.choice([t for t in topics[category] if t != topic_a])
            field = random.choice(fields)
            
            prompt_text = template.format(
                topic_a=topic_a,
                topic_b=topic_b,
                context=field
            )
        elif "{topic}" in template and "{field}" in template:
            # For application templates
            topic = random.choice(topics[category])
            field = random.choice(fields)
            
            prompt_text = template.format(
                topic=topic,
                field=field
            )
        else:
            # For simple templates with just one topic
            topic = random.choice(topics[category])
            prompt_text = template.format(topic=topic)
        
        # Create prompt object
        prompt = {
            "id": f"template_prompt_{i+1}_{uuid.uuid4().hex[:6]}",
            "text": prompt_text,
            "metadata": {
                "type": template_type,
                "category": category,
                "generated": True,
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "expected_tokens": random.choice([100, 150, 200, 250, 300, 350, 400])
            }
        }
        
        prompts.append(prompt)
    
    return prompts

def generate_completions(prompt_id, prompt_text, count=3):
    """Generate completions for a given prompt using DeepSeek API or simulation"""
    # System prompt to guide the model
    system_prompt = f"""
    You are an assistant tasked with generating a high-quality completion for a given prompt.
    
    For this completion, aim to be:
    1. Informative and detailed
    2. Accurate and factual
    3. Well-structured and coherent
    
    THE PROMPT: {prompt_text}
    
    Generate a thoughtful, detailed completion for this prompt. The completion should be self-contained
    and appropriately address what was asked in the prompt.
    """
    
    # Try to use DeepSeek API if available
    try:
        # Attempt to import OpenAI client for DeepSeek API
        from openai import OpenAI
        import os
        import json
        import sys
        
        # Check if API key is available
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ImportError("DeepSeek API key not found in environment variables")
        
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        completions = []
        
        # Generate each completion with slightly different parameters to ensure diversity
        for i in range(count):
            # Adjust temperature and seed for each completion to increase diversity
            temp = 0.7 + (i * 0.1)  # Increase temperature for each completion
            
            # Create a placeholder in the Streamlit UI for streaming
            if "streamlit" in sys.modules:
                stream_placeholder = st.empty()
                stream_placeholder.text("Generating completion...\n\n")
                stream_text = ""
            
            # Make API call with streaming enabled
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=min(temp, 1.0),  # Cap at 1.0
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0.2 if i > 0 else 0.0,  # Add frequency penalty for variety after first completion
                presence_penalty=0.3 if i > 1 else 0.0,    # Add presence penalty for even more variety on third completion
                stream=True  # Enable streaming
            )
            
            # Process the stream
            completion_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content_piece = chunk.choices[0].delta.content
                    completion_text += content_piece
                
                # Update the UI if in Streamlit context
                if "streamlit" in sys.modules:
                    stream_text += content_piece
                    # Update the placeholder with current text
                    stream_placeholder.text(f"Generating completion {i+1}/{count}...\n\n{stream_text}")
            
            # Clear the placeholder once generation is complete
            if "streamlit" in sys.modules:
                stream_placeholder.empty()
            
            # Create completion object
            completion = {
                "id": f"{prompt_id}_completion_{i+1}_{uuid.uuid4().hex[:6]}",
                "text": completion_text,
                "metadata": {
                    "style": "DeepSeek generated",
                    "generated": True,
                    "temperature": temp,
                    "model": "deepseek-chat"
                }
            }
            
            completions.append(completion)
            
            # Add slight delay between calls
            sleep(0.5)
        
        # Log success
        logger.info(f"Successfully generated {len(completions)} completions using DeepSeek API for prompt: {prompt_id}")
        
        return completions
        
    except Exception as e:
        logger.warning(f"Could not use DeepSeek API for completion generation: {e}, falling back to simulation")
        
        # Simple templates for different types of completions
        completion_templates = {
            "factual": [
                "{topic} refers to {definition}. Key aspects include {aspect1}, {aspect2}, and {aspect3}. {additional_info}",
                "When we talk about {topic}, we're discussing {definition}. This involves {aspect1} and {aspect2}. {additional_info}",
                "{topic} is {definition}. It's characterized by {aspect1}, while also involving {aspect2}. {additional_info}"
            ],
            "creative": [
                "Imagine {topic} as {metaphor}. {aspect1} is like {metaphor_aspect1}, while {aspect2} resembles {metaphor_aspect2}. {conclusion}",
                "Picture {topic} as {metaphor}. Just as {metaphor_aspect1}, {aspect1} works in similar ways. {conclusion}",
                "If we think of {topic} as {metaphor}, then {aspect1} would be the {metaphor_aspect1}, and {aspect2} would be the {metaphor_aspect2}. {conclusion}"
            ],
            "analytical": [
                "When analyzing {topic}, we must consider {aspect1}, {aspect2}, and {aspect3}. {critical_point} Furthermore, {additional_analysis}.",
                "From an analytical perspective, {topic} involves several key considerations. First, {aspect1}. Second, {aspect2}. Finally, {aspect3}. {conclusion}",
                "A thorough examination of {topic} reveals three important aspects: {aspect1}, {aspect2}, and {aspect3}. {critical_point} This suggests {conclusion}."
            ],
            "poetic": [
                "{adjective1} {topic},\n{verb1} through {location},\n{adjective2} as {simile},\n{conclusion}.",
                "In the realm of {topic},\nWhere {adjective1} {noun1} {verb1},\nAnd {adjective2} {noun2} {verb2},\n{conclusion}.",
                "{topic} like {simile},\n{verb1} with {adjective1} grace,\n{verb2} through {location},\n{conclusion}."
            ]
        }
        
        # Extract possible topic from prompt
        words = prompt_text.lower().split()
        potential_topics = [
            "ai", "machine learning", "deep learning", "artificial intelligence", 
            "ethics", "data", "algorithms", "neural networks", "language models",
            "computer vision", "robotics", "automation", "supervised learning",
            "unsupervised learning", "reinforcement learning"
        ]
        
        # Find a topic or use a default
        found_topic = next((topic for topic in potential_topics if topic in prompt_text.lower()), "technology")
        
        # Placeholder values for templates
        placeholder_data = {
            "topic": found_topic,
            "definition": "a set of techniques and approaches for solving complex problems",
            "aspect1": "data processing",
            "aspect2": "algorithmic decision-making",
            "aspect3": "continuous improvement through feedback",
            "additional_info": "This has implications across various domains including healthcare, finance, and education.",
            "metaphor": "a garden that needs constant tending",
            "metaphor_aspect1": "plants need nurturing",
            "metaphor_aspect2": "garden beds",
            "conclusion": "This understanding can guide future developments.",
            "critical_point": "The interplay between these factors is crucial.",
            "additional_analysis": "we must remain vigilant about unintended consequences"
        }
        
        completions = []
        for i in range(count):
            # Pick completion style based on prompt and index
            if i == 0:
                # First completion is more likely to be analytical or factual
                if "explain" in prompt_text.lower() or "what is" in prompt_text.lower():
                    styles = ["factual", "analytical"]
                elif "poem" in prompt_text.lower() or "creative" in prompt_text.lower():
                    styles = ["poetic", "creative"]
                elif "analyze" in prompt_text.lower() or "ethical" in prompt_text.lower():
                    styles = ["analytical", "factual"]
                else:
                    styles = ["analytical", "factual", "creative", "poetic"]
            else:
                # Second+ completion should be different from first
                if "explain" in prompt_text.lower() or "what is" in prompt_text.lower():
                    styles = ["creative", "poetic"]
                elif "poem" in prompt_text.lower() or "creative" in prompt_text.lower():
                    styles = ["analytical", "creative"]
                elif "analyze" in prompt_text.lower() or "ethical" in prompt_text.lower():
                    styles = ["creative", "poetic"]
                else:
                    styles = ["poetic", "creative", "analytical", "factual"]
            
            # Select a style for this completion
            style = random.choice(styles)
            template = random.choice(completion_templates[style])
            
            # Slightly vary the placeholder data for diversity
            varied_data = placeholder_data.copy()
            varied_data["conclusion"] = random.choice([
                "This understanding can guide future developments.",
                "Such insights help us navigate complex challenges.",
                "This framework offers valuable perspective.",
                "The implications of this are far-reaching."
            ])
            
            # Format template with data
            completion_text = template.format(**varied_data)
            
            # Create completion object
            completion = {
                "id": f"{prompt_id}_completion_{i+1}_{uuid.uuid4().hex[:6]}",
                "text": completion_text,
                "metadata": {
                    "style": style,
                    "generated": True
                }
            }
            
            completions.append(completion)
        
        return completions

def load_all_data(force_reload=False):
    """
    Load all data from the database and prepare for visualization.
    
    Args:
        force_reload: If True, force reload from disk instead of using cache
    
    Returns:
        Tuple of (annotations_df, predictions_df, reflections_df)
    """
    st.session_state.last_refresh_time = time.time()
    
    # Get database instance
    db = get_database()
    
    # Load data from database
    with st.spinner("Loading data..."):
        annotations_df = db.get_annotations(force_reload=force_reload)
        predictions_df = db.get_predictions(force_reload=force_reload)
        reflections_df = db.get_reflection_data(force_reload=force_reload)
    
    # Get database summary for status indicators
    summary = db.get_data_summary()
    
    # Store in session state for reuse
    st.session_state['vote_df'] = annotations_df
    st.session_state['predictions_df'] = predictions_df 
    st.session_state['reflections_df'] = reflections_df
    st.session_state['data_summary'] = summary
    
    return annotations_df, predictions_df, reflections_df

def save_annotation(annotation_data):
    """Save annotation data to the database"""
    # Get database instance
    db = get_database()
    
    # Save the annotation
    success = db.save_annotation(annotation_data)
    
    if success:
        # Reload data after successful save
        load_all_data(force_reload=True)
        return True
    else:
        st.error("Failed to save annotation. Please try again.")
        return False

def get_deepseek_response(messages, system_prompt=None, temperature=0.7):
    """Get a response from DeepSeek API for the chat interface"""
    try:
        # Import OpenAI client for DeepSeek API
        from openai import OpenAI
        import os
        import json
        
        # Check if API key is available
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            st.warning("DeepSeek API key not found in environment variables. Using fallback response mode.")
            return generate_fallback_response(messages, system_prompt)
        
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Prepare messages for API call
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for message in messages:
            api_messages.append({"role": message["role"], "content": message["content"]})
        
        # Create a placeholder for streaming
        response_placeholder = st.empty()
        
        # Initialize response text
        response_text = ""
        
        # Make API call with streaming
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=api_messages,
            temperature=temperature,
            max_tokens=800,
            stream=True
        )
        
        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content_piece = chunk.choices[0].delta.content
                response_text += content_piece
                # Update the placeholder with current text
                response_placeholder.markdown(response_text)
        
        # Clear the placeholder
        response_placeholder.empty()
        
        return response_text
        
    except Exception as e:
        # Handle any errors gracefully
        import traceback
        error_msg = f"Error connecting to DeepSeek API: {str(e)}"
        st.warning(f"Encountered an error with DeepSeek API. Using fallback response mode.")
        st.info(f"Technical details: {str(e)}")
        return generate_fallback_response(messages, system_prompt)

def generate_fallback_response(messages, system_prompt=None):
    """Generate a fallback response when DeepSeek API is not available"""
    # Get the latest user message
    if not messages:
        return "I don't have any messages to respond to yet."
    
    latest_message = messages[-1]["content"] if messages else ""
    
    # Simple rules-based responses for common RLHF questions
    rlhf_responses = {
        "what is rlhf": "RLHF (Reinforcement Learning from Human Feedback) is a technique to align AI models with human preferences by using human feedback to create a reward signal that guides model training.",
        "how does rlhf work": "RLHF works in three main steps: 1) Pretrain a model on a large corpus of text, 2) Collect human preferences on model outputs, 3) Train a reward model based on these preferences, and 4) Fine-tune the model using reinforcement learning to maximize this reward.",
        "calibration": "Calibration in RLHF refers to ensuring that a model's confidence scores accurately reflect its actual performance. A well-calibrated model will be 70% accurate when it reports 70% confidence.",
        "drift": "Drift in RLHF refers to how model behavior can shift away from desired performance over time. Drift clusters help identify patterns where the model's predictions no longer align with human preferences.",
        "preference": "Preference data in RLHF consists of human judgments about which of two or more model outputs is better. These preferences are used to train the reward model.",
        "alignment": "Alignment in RLHF refers to how well the model's behavior matches human values and expectations. The goal is to create AI systems that are helpful, harmless, and honest.",
        "calibration error": "Calibration error, often measured as Expected Calibration Error (ECE), quantifies the difference between a model's confidence and its actual accuracy.",
        "dashboard": "The RLHF Attunement Dashboard visualizes key metrics of the RLHF system, including alignment over time, calibration diagnostics, drift analysis, model evolution, and preference tracking."
    }
    
    # Check for keyword matches in the user's message
    user_message_lower = latest_message.lower()
    
    for keyword, response in rlhf_responses.items():
        if keyword in user_message_lower:
            return response
    
    # Generic fallback responses
    import random
    fallbacks = [
        "I'm currently operating in fallback mode without access to the DeepSeek API. For detailed RLHF information, please ask about topics like 'alignment', 'calibration', 'preferences', or 'drift'.",
        "I can provide basic information about RLHF concepts while operating in fallback mode. Try asking about specific RLHF components or check the other dashboard tabs for more detailed visualizations.",
        "The DeepSeek API is currently unavailable. I can still answer basic questions about RLHF, preferences, calibration, and alignment patterns.",
        "While in fallback mode, I can offer information about the RLHF process, preference collection, model calibration, and alignment metrics. For more technical assistance, please ensure the DeepSeek API is properly configured."
    ]
    
    return random.choice(fallbacks)

def show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary):
    """
    Display the main dashboard overview with key metrics and recommended actions.
    
    Args:
        vote_df: DataFrame with annotation data
        predictions_df: DataFrame with prediction data
        reflections_df: DataFrame with reflection data
        data_summary: Dictionary with data summary stats
    """
    st.header("📊 RLHF System Dashboard")
    
    # System status indicators
    st.subheader("System Status")
    
    # Calculate key metrics
    total_annotations = data_summary.get("total_annotations", 0)
    accuracy = data_summary.get("annotation_accuracy", 0)
    
    # Set up metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Annotation progress
        progress_pct = min(1.0, total_annotations / ANNOTATION_TARGET) if ANNOTATION_TARGET > 0 else 0
        st.metric(
            "Annotations", 
            f"{total_annotations}/{ANNOTATION_TARGET}", 
            delta=f"{progress_pct:.1%}"
        )
        st.progress(progress_pct)
    
    with col2:
        # Model accuracy
        if accuracy is not None:
            accuracy_progress = min(1.0, accuracy / TARGET_ACCURACY) if TARGET_ACCURACY > 0 else 0
            st.metric(
                "Model Accuracy", 
                f"{accuracy:.1%}", 
                delta=f"{accuracy_progress:.1%} of target"
            )
            st.progress(accuracy_progress)
        else:
            st.metric("Model Accuracy", "No data")
            st.progress(0)
    
    with col3:
        # Data freshness
        latest_annotation = data_summary.get("latest_annotation")
        if latest_annotation:
            latest_dt = pd.to_datetime(latest_annotation)
            time_since = datetime.now() - latest_dt.to_pydatetime()
            hours_since = time_since.total_seconds() / 3600
            
            freshness_label = "Recent" if hours_since < 24 else "Outdated"
            delta_value = f"{int(hours_since)} hours ago"
            delta_color = "normal" if hours_since < 24 else "inverse"
            
            st.metric(
                "Data Freshness", 
                freshness_label,
                delta=delta_value,
                delta_color=delta_color
            )
        else:
            st.metric("Data Freshness", "No data")
    
    with col4:
        # System recommendation
        if total_annotations < 50:
            action = "Add more annotations"
            description = "System needs more training data"
        elif accuracy is not None and accuracy < 0.6:
            action = "Improve model training"
            description = "Model accuracy is low"
        elif accuracy is not None and accuracy >= TARGET_ACCURACY:
            action = "Review error cases"
            description = "Model performing well"
        else:
            action = "Continue annotating"
            description = "Building training dataset"
            
        st.metric("Recommended Action", action)
        st.caption(description)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if not vote_df.empty:
        # Get recent annotations
        recent_votes = vote_df.sort_values('timestamp', ascending=False).head(5)
        
        # Format for display
        recent_activity = []
        for _, row in recent_votes.iterrows():
            timestamp = row.get('timestamp', datetime.now())
            prompt = row.get('prompt', '')
            preference = row.get('preference', '')
            
            # Truncate prompt if too long
            if len(prompt) > 100:
                prompt = prompt[:100] + "..."
                
            recent_activity.append({
                "timestamp": timestamp,
                "action": "Annotation",
                "details": f"Preference: {preference}",
                "prompt": prompt
            })
            
        # Display as a dataframe
        if recent_activity:
            activity_df = pd.DataFrame(recent_activity)
            st.dataframe(
                activity_df[['timestamp', 'action', 'details', 'prompt']],
                use_container_width=True
            )
        else:
            st.info("No recent activity found")
        else:
        st.info("No annotation data available")
    
    # Quick stats and charts
    st.subheader("System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show annotation growth over time
        if not vote_df.empty and 'timestamp' in vote_df.columns:
            st.write("Annotation Growth")
            
            # Group by day and count
            vote_df['date'] = pd.to_datetime(vote_df['timestamp']).dt.date
            daily_counts = vote_df.groupby('date').size().reset_index(name='count')
            daily_counts['cumulative'] = daily_counts['count'].cumsum()
            
            # Create chart
            chart = alt.Chart(daily_counts).mark_area().encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('cumulative:Q', title='Total Annotations')
            ).properties(height=200)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data for annotation growth chart")
    
    with col2:
        # Show model accuracy over time if available
        if not vote_df.empty and 'model_correct' in vote_df.columns and 'timestamp' in vote_df.columns:
            st.write("Model Accuracy Over Time")
            
            # Sort by timestamp
            accuracy_df = vote_df.sort_values('timestamp')
            
            # Calculate rolling accuracy
            window_size = min(10, len(accuracy_df))
            if window_size > 0:
                accuracy_df['rolling_accuracy'] = accuracy_df['model_correct'].rolling(window=window_size).mean()
                
                # Create dataframe for chart
                chart_data = accuracy_df[['timestamp', 'rolling_accuracy']].dropna()
                
                if not chart_data.empty:
                    # Create chart
                    chart = alt.Chart(chart_data).mark_line().encode(
                        x=alt.X('timestamp:T', title='Time'),
                        y=alt.Y('rolling_accuracy:Q', title='Accuracy (rolling)', scale=alt.Scale(domain=[0, 1]))
                    ).properties(height=200)
                    
                    st.altair_chart(chart, use_container_width=True)
        else:
                    st.info("Not enough data points for accuracy chart")
            else:
                st.info("Not enough data points for accuracy chart")
        else:
            st.info("Model accuracy data not available")
    
    # System health section
    st.subheader("System Health")
    
            col1, col2, col3 = st.columns(3)
            
    with col1:
        # Data files status
        st.write("Data Files")
        data_files = data_summary.get("data_files", {})
        
        file_status = {
            "Vote Logs": f"{data_files.get('vote_logs', 0)} files",
            "Votes JSONL": "✅ Present" if data_files.get("votes_jsonl", False) else "❌ Missing",
            "Predictions JSONL": "✅ Present" if data_files.get("predictions_jsonl", False) else "❌ Missing",
            "Reflection Data": "✅ Present" if data_files.get("reflection_data_jsonl", False) else "❌ Missing"
        }
        
        st.json(file_status)
    
    with col2:
        # Annotation quality
        st.write("Annotation Quality")
        
        if not vote_df.empty:
            # Check for feedback
            has_feedback = 'feedback' in vote_df.columns and vote_df['feedback'].notna().any()
            
            # Check for quality metrics
            has_quality_metrics = False
            for col in vote_df.columns:
                if 'quality' in col.lower():
                    has_quality_metrics = True
                    break
            
            quality_status = {
                "Has Feedback": "✅ Yes" if has_feedback else "❌ No",
                "Has Quality Metrics": "✅ Yes" if has_quality_metrics else "❌ No",
                "Binary Preferences": f"{vote_df['is_binary_preference'].sum() if 'is_binary_preference' in vote_df.columns else 0} / {len(vote_df)}",
                "Data Completeness": "✅ Good" if len(vote_df) > 50 else "⚠️ Limited"
            }
            
            st.json(quality_status)
        else:
            st.info("No annotation data available")
    
    with col3:
        # Quick actions
        st.write("Quick Actions")
        
        if st.button("📝 Start Annotating"):
            # Use session state to navigate to annotation tab
            st.session_state.sidebar_selection = "Annotation Interface"
            st.experimental_rerun()
            
        if st.button("📊 View Analytics"):
            # Use session state to navigate to alignment tab
            st.session_state.sidebar_selection = "Alignment Over Time"
            st.experimental_rerun()
            
        if st.button("💾 Export Data"):
            # Create CSV for download
            if not vote_df.empty:
                csv = vote_df.to_csv(index=False)
                st.download_button(
                    label="Download Annotations CSV",
                    data=csv,
                    file_name=f"rlhf_annotations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                else:
                st.warning("No data to export")

def display_annotation_interface(vote_df):
    """Display annotation interface for RLHF system."""
    st.header("🏷️ Annotation Interface")
    
    # Provide context and guidance
    with st.expander("ℹ️ How to Annotate", expanded=True):
                    st.markdown("""
        This interface allows you to generate prompts, create completions, and annotate them with your preferences. 
        Your annotations help train and improve the RLHF system.
        
        **Process:**
        1. **Generate a prompt** or use a pre-generated one
        2. **Create completions** (either using the API or manually)
        3. **Annotate** by selecting your preferred completion and providing feedback
        4. Your annotation data will automatically be saved for model training
        
        **Tips for quality annotations:**
        - Be consistent in your preferences
        - Explain your reasoning in the feedback field
        - Try to focus on different types of prompts to provide diverse training data
        """)
    
    # Show annotation progress
    total_annotations = len(vote_df)
    progress_pct = min(1.0, total_annotations / ANNOTATION_TARGET) if ANNOTATION_TARGET > 0 else 0
    
    st.info(f"Annotation Progress: {total_annotations} out of {ANNOTATION_TARGET} target annotations ({progress_pct:.1%})")
    st.progress(progress_pct)
    
    # Create tabs for different annotation workflows
    tab1, tab2, tab3 = st.tabs(["Generate Content", "Annotate", "Annotation History"])
    
    with tab1:
        st.subheader("Generate New Content")
        
        # Prompt generation options
        generation_method = st.radio(
            "Prompt Generation Method",
            ["Use Pre-Generated Prompt", "Generate New Prompt", "Enter Custom Prompt"],
            help="Choose how you want to generate the prompt for annotation"
        )
        
        if generation_method == "Use Pre-Generated Prompt":
            # Load pre-generated prompts
            prompt_categories = ["General", "Coding", "Creative", "Reasoning", "Instructions"]
            selected_category = st.selectbox("Prompt Category", prompt_categories)
            
            # Mock data - in real implementation, load from a prompts database
            sample_prompts = [
                "Explain the difference between supervised and unsupervised learning in AI.",
                "Write a function to find the nth Fibonacci number in Python.",
                "Describe three ways to improve productivity when working from home.",
                "What are the ethical implications of AI-generated content?",
                "If you could have dinner with any historical figure, who would it be and why?"
            ]
            
            selected_prompt = st.selectbox("Select a prompt", sample_prompts)
            
            if st.button("Use This Prompt"):
                st.session_state.current_prompt = selected_prompt
                st.session_state.current_prompt_id = f"pregenerated_{uuid.uuid4().hex[:8]}"
                st.success(f"Prompt selected! Go to the Annotate tab to generate completions.")
                
        elif generation_method == "Generate New Prompt":
            prompt_theme = st.text_input("Prompt Theme (e.g., 'Technology', 'Ethics', 'Science')")
            prompt_type = st.selectbox("Prompt Type", ["Question", "Instruction", "Open-ended", "Scenario"])
            
            if st.button("Generate Prompt") and prompt_theme:
                # Use the API client to generate a prompt
                with st.spinner("Generating prompt..."):
                    # Create a system prompt for the generator
                    system_prompt = f"""
                    You are a prompt generator for an RLHF system. Create a {prompt_type.lower()} about {prompt_theme}.
                    The prompt should be clear, interesting, and designed to elicit different possible responses.
                    Respond with only the generated prompt text, nothing else.
                    """
                    
                    # Generate the prompt
                    new_prompt = generate_chat_response(f"Generate a {prompt_type} about {prompt_theme}", system_prompt=system_prompt)
                    
                    # Store in session state
                    st.session_state.current_prompt = new_prompt
                    st.session_state.current_prompt_id = f"generated_{uuid.uuid4().hex[:8]}"
                    
                st.success("Prompt generated successfully!")
                st.write(f"**Generated Prompt:** {new_prompt}")
                st.info("Go to the Annotate tab to generate completions for this prompt.")
                
        elif generation_method == "Enter Custom Prompt":
            custom_prompt = st.text_area("Enter your prompt", height=100)
            
            if st.button("Use Custom Prompt") and custom_prompt:
                st.session_state.current_prompt = custom_prompt
                st.session_state.current_prompt_id = f"custom_{uuid.uuid4().hex[:8]}"
                st.success("Custom prompt saved!")
                st.info("Go to the Annotate tab to generate completions for this prompt.")
    
    with tab2:
        st.subheader("Create Completions & Annotate")
        
        # Display current prompt if available
        current_prompt = st.session_state.get("current_prompt", "")
        current_prompt_id = st.session_state.get("current_prompt_id", "")
        
        if current_prompt:
            st.write("**Current Prompt:**")
            st.info(current_prompt)
            
            # Completion generation options
            completion_method = st.radio(
                "Completion Generation Method",
                ["Generate with API", "Enter Custom Completions"],
                help="Choose how you want to generate the completions for annotation"
            )
            
            if completion_method == "Generate with API":
                # Use the API client to generate completions
                if st.button("Generate Completions"):
                    # Use our new function that generates a pair of completions
                    generate_completions_pair(current_prompt)
            
            elif completion_method == "Enter Custom Completions":
                # Manual completion entry
                completion_a = st.text_area("Completion A", height=150, value=st.session_state.get("completion_a", ""))
                completion_b = st.text_area("Completion B", height=150, value=st.session_state.get("completion_b", ""))
                
                if st.button("Save Completions") and completion_a and completion_b:
                    st.session_state.completion_a = completion_a
                    st.session_state.completion_b = completion_b
                    st.success("Completions saved successfully!")
            
            # Display completions for annotation if available
            if st.session_state.get("completion_a") and st.session_state.get("completion_b"):
                st.write("### Compare Completions")
                
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                    st.write("**Completion A:**")
                    st.info(st.session_state.completion_a)
                                    
                                    with col2:
                    st.write("**Completion B:**")
                    st.info(st.session_state.completion_b)
                
                # Annotation form
                st.write("### Annotate")
                
                with st.form("annotation_form"):
                    preference = st.radio(
                        "Which completion do you prefer?",
                        ["Completion A", "Completion B"],
                        help="Select the completion that better responds to the prompt"
                    )
                    
                    # Quality metrics
                    st.write("#### Quality Metrics")
                    quality_metrics = {}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        quality_metrics["relevance"] = st.slider(
                            "Relevance to Prompt", 1, 10, 5,
                            help="How relevant is the preferred completion to the prompt?"
                        )
                        quality_metrics["coherence"] = st.slider(
                            "Coherence", 1, 10, 5,
                            help="How logically structured is the preferred completion?"
                        )
                    
                    with col2:
                        quality_metrics["accuracy"] = st.slider(
                            "Factual Accuracy", 1, 10, 5,
                            help="How factually accurate is the preferred completion?"
                        )
                        quality_metrics["helpfulness"] = st.slider(
                            "Helpfulness", 1, 10, 5,
                            help="How helpful is the preferred completion in addressing the prompt?"
                        )
                    
                    # Feedback
                    feedback = st.text_area(
                        "Feedback (Optional)",
                        placeholder="Explain why you preferred this completion...",
                        help="Your reasoning helps improve the model's understanding"
                    )
                    
                    # Submit button
                    submit_button = st.form_submit_button("Submit Annotation")
                
                if submit_button:
                    # Prepare annotation data
                    annotation_data = {
                        "prompt_id": current_prompt_id,
                        "prompt": current_prompt,
                        "preference": preference,
                        "selected_completion": st.session_state.completion_a if preference == "Completion A" else st.session_state.completion_b,
                        "rejected_completion": st.session_state.completion_b if preference == "Completion A" else st.session_state.completion_a,
                        "feedback": feedback,
                        "quality_metrics": quality_metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Save annotation
                    success = save_annotation(annotation_data)
                    
                    if success:
                        st.success("Annotation saved successfully!")
                        
                        # Clear session state for next annotation
                        if st.button("Start New Annotation"):
                            for key in ["current_prompt", "current_prompt_id", "completion_a", "completion_b"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.experimental_rerun()
                    else:
                        st.error("Failed to save annotation. Please try again.")
        else:
            st.warning("No prompt selected. Go to the Generate Content tab to create or select a prompt.")
    
    with tab3:
        st.subheader("Your Annotation History")
        
        if not vote_df.empty:
            # Filter for most recent annotations
            recent_annotations = vote_df.sort_values("timestamp", ascending=False).head(10)
            
            st.write(f"Showing your {len(recent_annotations)} most recent annotations")
            
            for idx, row in recent_annotations.iterrows():
                with st.expander(f"Annotation {idx+1}: {row.get('timestamp', 'Unknown date')}", expanded=False):
                    st.write("**Prompt:**")
                    st.info(row.get("prompt", "No prompt available"))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Selected Completion:**")
                        st.success(row.get("selected_completion", "Not available"))
                    
                    with col2:
                        st.write("**Rejected Completion:**")
                        st.error(row.get("rejected_completion", "Not available"))
                    
                    st.write("**Your Feedback:**")
                    st.info(row.get("feedback", "No feedback provided"))
                    
                    # Show model prediction if available
                    if "model_prediction" in row and "model_correct" in row:
                        st.write("**Model Prediction:**")
                        prediction = row.get("model_prediction", "")
                        is_correct = row.get("model_correct", False)
                        
                        if is_correct:
                            st.success(f"Model correctly predicted your preference ({prediction})")
                        else:
                            st.error(f"Model incorrectly predicted {prediction}")
                else:
            st.info("No annotation history available. Start annotating to see your history here.")

def generate_mock_completion(prompt_text):
    """Generate a mock completion for demonstration purposes"""
    # Simple templates for different types of completions
    completion_templates = {
        "factual": [
            "{topic} refers to {definition}. Key aspects include {aspect1}, {aspect2}, and {aspect3}. {additional_info}",
            "When we talk about {topic}, we're discussing {definition}. This involves {aspect1} and {aspect2}. {additional_info}",
            "{topic} is {definition}. It's characterized by {aspect1}, while also involving {aspect2}. {additional_info}"
        ],
        "creative": [
            "Imagine {topic} as {metaphor}. {aspect1} is like {metaphor_aspect1}, while {aspect2} resembles {metaphor_aspect2}. {conclusion}",
            "Picture {topic} as {metaphor}. Just as {metaphor_aspect1}, {aspect1} works in similar ways. {conclusion}",
            "If we think of {topic} as {metaphor}, then {aspect1} would be the {metaphor_aspect1}, and {aspect2} would be the {metaphor_aspect2}. {conclusion}"
        ],
        "analytical": [
            "When analyzing {topic}, we must consider {aspect1}, {aspect2}, and {aspect3}. {critical_point} Furthermore, {additional_analysis}.",
            "From an analytical perspective, {topic} involves several key considerations. First, {aspect1}. Second, {aspect2}. Finally, {aspect3}. {conclusion}",
            "A thorough examination of {topic} reveals three important aspects: {aspect1}, {aspect2}, and {aspect3}. {critical_point} This suggests {conclusion}."
        ]
    }
    
    # Extract possible topic from prompt
    words = prompt_text.lower().split()
    potential_topics = [
        "ai", "machine learning", "deep learning", "artificial intelligence", 
        "ethics", "data", "algorithms", "neural networks", "language models",
        "computer vision", "robotics", "automation", "supervised learning",
        "unsupervised learning", "reinforcement learning"
    ]
    
    # Find a topic or use a default
    found_topic = next((topic for topic in potential_topics if topic in prompt_text.lower()), "technology")
    
    # Placeholder values for templates
    placeholder_data = {
        "topic": found_topic,
        "definition": "a set of techniques and approaches for solving complex problems",
        "aspect1": "data processing",
        "aspect2": "algorithmic decision-making",
        "aspect3": "continuous improvement through feedback",
        "additional_info": "This has implications across various domains including healthcare, finance, and education.",
        "metaphor": "a garden that needs constant tending",
        "metaphor_aspect1": "plants need nurturing",
        "metaphor_aspect2": "garden beds",
        "conclusion": "This understanding can guide future developments.",
        "critical_point": "The interplay between these factors is crucial.",
        "additional_analysis": "we must remain vigilant about unintended consequences"
    }
    
    # For "supervised vs unsupervised" type prompts, use more specific completions
    if "supervised" in prompt_text.lower() and "unsupervised" in prompt_text.lower():
        completions = [
            "Supervised learning requires labeled data where the model learns to map inputs to known outputs. Unsupervised learning works with unlabeled data, finding patterns and structures without predefined outputs. Supervised learning is used for classification and regression tasks, while unsupervised learning is used for clustering, dimensionality reduction, and anomaly detection.",
            "The key difference between supervised and unsupervised learning is the presence of labeled training data. In supervised learning, algorithms are trained on labeled examples to predict outcomes for new data. Unsupervised learning algorithms identify patterns in data without labels, discovering hidden structures and relationships autonomously.",
            "Supervised learning involves training a model with input-output pairs, where the correct answers are provided. The model learns to approximate the mapping function to make predictions on new data. Unsupervised learning, in contrast, explores data without labels, identifying inherent structures through techniques like clustering and association."
        ]
        return random.choice(completions)
    
    # Select completion style based on prompt
    if "explain" in prompt_text.lower() or "what is" in prompt_text.lower():
        style = "factual"
    elif "analyze" in prompt_text.lower() or "compare" in prompt_text.lower():
        style = "analytical"
    elif "creative" in prompt_text.lower() or "imagine" in prompt_text.lower():
        style = "creative"
                else:
        style = random.choice(list(completion_templates.keys()))
    
    # Select a template for this completion
    template = random.choice(completion_templates[style])
    
    # Format template with data
    completion_text = template.format(**placeholder_data)
    
    return completion_text

def display_chat_interface():
    """Display a simple chat interface for conversational interaction"""
    st.header("💬 Chat Interface")
    
    # Description
        st.markdown("""
    This interface allows you to chat with the RLHF system and explore how it responds to different prompts.
    The chat supports discussion about RLHF topics, annotation strategies, and system capabilities.
    """)
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "system", "content": "You are chatting with the RLHF Attunement Dashboard assistant."},
            {"role": "assistant", "content": "Hello! I'm here to help you with RLHF concepts, annotation guidance, or exploring the dashboard. What would you like to discuss today?"}
        ]
    
    # Initialize session ID for this chat
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        if message["role"] != "system":  # Don't display system messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about RLHF?"):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use our new API client function
                response = generate_chat_response(
                    st.session_state.chat_messages,
                    system_prompt="You are an AI assistant embedded in an RLHF Attunement Dashboard. Help users understand RLHF concepts, provide guidance on annotation best practices, and explain dashboard features."
                )
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Save chat log for training
        save_chat_log(st.session_state.chat_messages, st.session_state.chat_session_id)
    
    # Add chat rating option at the bottom
    if len(st.session_state.chat_messages) > 2:  # Only show if there's been conversation
        with st.expander("Rate this conversation for RLHF training", expanded=False):
            st.write("Your feedback helps improve the AI system:")
            
            # Quality rating
            quality = st.slider("Conversation Quality", 1, 10, 5)
            
            # Helpfulness rating
            helpfulness = st.slider("Assistant Helpfulness", 1, 10, 5)
            
            # Feedback text
            feedback = st.text_area("Additional Feedback (optional)", "")
            
            if st.button("Submit Feedback"):
                # Save feedback with chat log
                save_chat_feedback(
                    st.session_state.chat_session_id,
                    {
                        "quality": quality,
                        "helpfulness": helpfulness,
                        "feedback": feedback,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                st.success("Thank you for your feedback! It will be used to improve the system.")
                
                # Optionally reset the chat
                if st.button("Start New Chat"):
                    # Keep only the system message
                    st.session_state.chat_messages = [st.session_state.chat_messages[0]]
                    st.session_state.chat_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    st.experimental_rerun()

def save_chat_log(messages, session_id):
    """Save chat log to disk for training"""
    # Create chat logs directory if it doesn't exist
    chat_logs_dir = Path(project_root) / "data" / "chat_logs"
    chat_logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Create JSON structure
    chat_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    # Save to file
    chat_file = chat_logs_dir / f"{session_id}.json"
    with open(chat_file, "w") as f:
        json.dump(chat_data, f, indent=2)
    
    logger.info(f"Saved chat log to {chat_file}")

def save_chat_feedback(session_id, feedback_data):
    """Save feedback for a chat session"""
    # Find the chat log file
    chat_logs_dir = Path(project_root) / "data" / "chat_logs"
    chat_file = chat_logs_dir / f"{session_id}.json"
    
    if not chat_file.exists():
        logger.warning(f"Chat log file not found: {chat_file}")
        return
    
    # Load existing data
    with open(chat_file, "r") as f:
        chat_data = json.load(f)
    
    # Add feedback
    chat_data["quality_metrics"] = feedback_data
    
    # Save updated data
    with open(chat_file, "w") as f:
        json.dump(chat_data, f, indent=2)
    
    logger.info(f"Saved feedback for chat session {session_id}")

def generate_completions_pair(prompt_text):
    """Generate a pair of completions for annotation"""
    api_client = get_api_client()
    
    with st.spinner("Generating completions..."):
        # Use the API client to generate a pair of completions
        completions = api_client.generate_completions_pair(prompt_text)
        
        # Store in session state
        st.session_state.completion_a = completions["completion_a"]
        st.session_state.completion_b = completions["completion_b"]
    
    st.success("Completions generated successfully!")

def display_annotation_history(vote_df, predictions_df):
    """Display annotation history with detailed analytics"""
    st.header("📊 Annotation History")
    
    if vote_df.empty:
        st.warning("No annotation data available yet. Start annotating to see your history.")
        return
    
    # Create tabs for different views
    stats_tab, timeline_tab, details_tab = st.tabs(["Stats", "Timeline", "Annotation Details"])
    
    with stats_tab:
        st.subheader("Annotation Statistics")
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Annotations", f"{len(vote_df)}")
        
        with col2:
            # Calculate model accuracy if available
            if 'model_correct' in vote_df.columns:
                accuracy = vote_df['model_correct'].mean()
                st.metric("Model Accuracy", f"{accuracy:.2%}")
            else:
                st.metric("Model Accuracy", "N/A")
        
        with col3:
            # Get most recent annotation date
            if 'timestamp' in vote_df.columns:
                latest = vote_df['timestamp'].max()
                st.metric("Latest Annotation", latest.strftime("%Y-%m-%d"))
            else:
                st.metric("Latest Annotation", "N/A")
        
        # Additional metrics if available
        if 'quality_metrics' in vote_df.columns or any('quality' in col for col in vote_df.columns):
            st.subheader("Quality Metrics")
            
            # Extract quality metrics if stored as nested dict
            quality_cols = [col for col in vote_df.columns if 'quality' in col.lower()]
            
            if quality_cols:
                # Create metrics for each quality dimension
                quality_df = vote_df[quality_cols].mean().reset_index()
                quality_df.columns = ['Metric', 'Average Score']
                
                st.bar_chart(quality_df.set_index('Metric'))
            else:
                st.info("Quality metrics data structure not recognized")
        
        # Human-AI agreement plot if available
        if 'model_prediction' in vote_df.columns and 'model_correct' in vote_df.columns:
            st.subheader("Human-AI Agreement")
            
            # Plot agreement over time if timestamps available
            if 'timestamp' in vote_df.columns:
                # Create a copy with datetime index
                plot_df = vote_df.copy()
                plot_df['date'] = pd.to_datetime(plot_df['timestamp']).dt.date
                
                # Group by date and calculate agreement rate
                agreement_by_date = plot_df.groupby('date')['model_correct'].mean().reset_index()
                
                # Create line chart
                agreement_chart = alt.Chart(agreement_by_date).mark_line().encode(
                    x='date:T',
                    y=alt.Y('model_correct:Q', title='Agreement Rate', scale=alt.Scale(domain=[0, 1]))
                ).properties(height=300)
                
                st.altair_chart(agreement_chart, use_container_width=True)
    
    with timeline_tab:
        st.subheader("Annotation Timeline")
        
        if 'timestamp' not in vote_df.columns:
            st.warning("Timestamp data not available for timeline visualization")
            return
        
        # Create a copy with datetime index
        timeline_df = vote_df.copy()
        timeline_df['date'] = pd.to_datetime(timeline_df['timestamp']).dt.date
        
        # Group by date and count annotations
        annotations_by_date = timeline_df.groupby('date').size().reset_index(name='count')
        annotations_by_date['cumulative'] = annotations_by_date['count'].cumsum()
        
        # Create combo chart with daily and cumulative counts
        base = alt.Chart(annotations_by_date).encode(
            x='date:T'
        )
        
        bar = base.mark_bar().encode(
            y='count:Q'
        )
        
        line = base.mark_line(color='red').encode(
            y='cumulative:Q'
        )
        
        st.altair_chart(bar + line, use_container_width=True)
    
    with details_tab:
        st.subheader("Detailed Annotations")
        
        # Sort by timestamp if available, otherwise use index
        if 'timestamp' in vote_df.columns:
            sorted_df = vote_df.sort_values('timestamp', ascending=False)
        else:
            sorted_df = vote_df
        
        # Display the latest annotations
        for i, (_, row) in enumerate(sorted_df.head(10).iterrows()):
            with st.expander(f"Annotation {i+1}: {row.get('timestamp', 'No date')}"):
                # Display prompt
                st.markdown("**Prompt:**")
                st.info(row.get('prompt', 'No prompt available'))
                
                # Display completions side by side if available
                if 'selected_completion' in row and 'rejected_completion' in row:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Selected Completion:**")
                        st.success(row.get('selected_completion', 'Not available'))
                    
                    with col2:
                        st.markdown("**Rejected Completion:**")
                        st.error(row.get('rejected_completion', 'Not available'))
                
                # Display feedback if available
                if 'feedback' in row:
                    st.markdown("**Feedback:**")
                    st.info(row.get('feedback', 'No feedback provided'))
                
                # Display model prediction vs actual if available
                if 'model_prediction' in row and 'model_correct' in row:
                    st.markdown("**Model Prediction:**")
                    if row.get('model_correct', False):
                        st.success(f"Model correctly predicted your preference ({row.get('model_prediction', 'N/A')})")
                    else:
                        st.error(f"Model incorrectly predicted {row.get('model_prediction', 'N/A')}")
        
        # Add download button for the full dataset
        csv = vote_df.to_csv(index=False)
        st.download_button(
            label="Download Complete Annotation History (CSV)",
            data=csv,
            file_name="annotation_history.csv",
            mime="text/csv"
        )

def display_alignment_over_time(vote_df, predictions_df):
    """Display alignment metrics over time with enhanced visualizations"""
    st.header("📈 Alignment Over Time")
    
    # Show placeholder message if no data
    if vote_df.empty:
        st.warning("No annotation data available yet to display alignment metrics.")
        st.info("Start annotating to see alignment metrics over time.")
        return
    
    # Check if we have model prediction data
    has_model_data = all(col in vote_df.columns for col in ['model_prediction', 'model_correct'])
    
    if not has_model_data:
        st.warning("Model prediction data not available in the annotation records.")
        st.info("The alignment visualization requires model predictions to compare with human preferences.")
        return
    
    # Create tabs for different alignment visualizations
    accuracy_tab, confidence_tab, drift_tab, themes_tab = st.tabs(["Accuracy Trends", "Confidence Analysis", "Alignment Drift", "Theme Analysis"])
    
    with accuracy_tab:
        st.subheader("Model Alignment Accuracy Over Time")
        
        if 'timestamp' not in vote_df.columns:
            st.warning("Timestamp data not available for timeline visualization")
            return
        
        # Create a copy with datetime index
        accuracy_df = vote_df.copy()
        accuracy_df['date'] = pd.to_datetime(accuracy_df['timestamp']).dt.date
        
        # Group by date and calculate accuracy
        accuracy_by_date = accuracy_df.groupby('date')['model_correct'].mean().reset_index()
        
        # Add rolling average
        window = min(7, len(accuracy_by_date))
        if window > 1:
            accuracy_by_date['rolling_avg'] = accuracy_by_date['model_correct'].rolling(window=window).mean()
        
        # Prepare annotation count as size metric
        count_by_date = accuracy_df.groupby('date').size().reset_index(name='count')
        accuracy_by_date = accuracy_by_date.merge(count_by_date, on='date')
        
        # Create the enhanced chart with Plotly
        fig = px.scatter(
            accuracy_by_date, 
            x='date', 
            y='model_correct',
            size='count',
            size_max=15,
            color='model_correct',
            color_continuous_scale='RdYlGn',
            range_color=[0.5, 1],
            hover_data={
                'date': True,
                'model_correct': ':.2%',
                'count': True,
                'rolling_avg': ':.2%' if 'rolling_avg' in accuracy_by_date.columns else None
            },
            labels={
                'date': 'Date',
                'model_correct': 'Alignment Accuracy',
                'count': 'Annotations',
                'rolling_avg': 'Rolling Average'
            },
            title='Human-AI Alignment Accuracy Over Time'
        )
        
        # Add rolling average line if available
        if 'rolling_avg' in accuracy_by_date.columns:
            fig.add_trace(
                go.Scatter(
                    x=accuracy_by_date['date'],
                    y=accuracy_by_date['rolling_avg'],
                    mode='lines',
                    name=f'{window}-day Rolling Average',
                    line=dict(color='rgba(0, 0, 255, 0.7)', width=2)
                )
            )
        
        # Add target line
        fig.add_shape(
            type="line",
            x0=accuracy_by_date['date'].min(),
            y0=0.8,
            x1=accuracy_by_date['date'].max(),
            y1=0.8,
            line=dict(color="green", width=1, dash="dash"),
            name="Target Accuracy"
        )
        
        # Add annotation for target
        fig.add_annotation(
            x=accuracy_by_date['date'].max(),
            y=0.8,
            text="Target (80%)",
            showarrow=False,
            yshift=10,
            font=dict(color="green")
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Alignment Accuracy",
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding Alignment Accuracy"):
            st.markdown("""
            **Alignment Accuracy** measures how well the model's preferences match human preferences.
            
            - **Higher values** indicate better alignment between AI predictions and human judgments
            - **Points** represent daily accuracy, with size indicating annotation volume
            - **Rolling average** smooths daily fluctuations to show overall trends
            - **Target line** shows the 80% alignment goal
            
            Consistent alignment above 80% suggests the model has successfully learned human preferences.
            """)
        
        # Add insights section if we have enough data
        if len(accuracy_by_date) > 3:
            st.subheader("Alignment Insights")
            
            # Calculate metrics
            current_accuracy = accuracy_by_date.iloc[-1]['model_correct']
            avg_accuracy = accuracy_by_date['model_correct'].mean()
            
            if len(accuracy_by_date) > 5:
                recent_trend = accuracy_by_date.iloc[-5:]['model_correct'].mean() - accuracy_by_date.iloc[-10:-5]['model_correct'].mean()
                trend_text = "improving" if recent_trend > 0.05 else "declining" if recent_trend < -0.05 else "stable"
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Alignment", f"{current_accuracy:.1%}", f"{current_accuracy - avg_accuracy:.1%} vs avg")
                col2.metric("Average Alignment", f"{avg_accuracy:.1%}")
                col3.metric("Recent Trend", trend_text, f"{recent_trend:.1%}")
                
                # Add recommendation based on trend
                if current_accuracy < 0.7:
                    st.warning("⚠️ **Recommendation**: Model alignment is below target. Consider collecting more annotations or retraining the model.")
                elif recent_trend < -0.05:
                    st.warning("⚠️ **Recommendation**: Alignment trend is declining. Check for concept drift or changes in annotation patterns.")
                elif current_accuracy > 0.85:
                    st.success("✅ **Recommendation**: Model alignment is excellent. Continue monitoring and consider using the model for production.")
    
    with confidence_tab:
        st.subheader("Confidence vs. Accuracy Analysis")
        
        if 'confidence' not in vote_df.columns:
            st.warning("Confidence data not available.")
            return
        
        # Create confidence bins
        vote_with_conf = vote_df.copy()
        vote_with_conf['confidence_bin'] = pd.cut(vote_with_conf['confidence'], 
                                               bins=10, 
                                               labels=[f"{i*10}-{(i+1)*10}%" for i in range(10)])
        
        # Group by confidence bin and calculate accuracy
        conf_accuracy = vote_with_conf.groupby('confidence_bin').agg({
            'model_correct': 'mean',
            'confidence': 'mean',
            'model_correct': 'count'
        }).reset_index()
        conf_accuracy.columns = ['confidence_bin', 'accuracy', 'avg_confidence', 'count']
        
        # Create calibration scatter plot
        fig = px.scatter(
            conf_accuracy,
            x='avg_confidence',
            y='accuracy',
            size='count',
            color='accuracy',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1],
            hover_data={
                'confidence_bin': True,
                'accuracy': ':.2%',
                'avg_confidence': ':.2%',
                'count': True
            },
            labels={
                'avg_confidence': 'Model Confidence',
                'accuracy': 'Actual Accuracy',
                'count': 'Number of Predictions',
                'confidence_bin': 'Confidence Range'
            },
            title='Calibration: Confidence vs. Actual Accuracy',
            size_max=20
        )
        
        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='black', width=1, dash='dash')
            )
        )
        
        # Improve layout
        fig.update_layout(
            xaxis=dict(tickformat='.0%', range=[0, 1]),
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding Model Calibration"):
            st.markdown("""
            **Calibration** measures how well the model's confidence scores match its actual accuracy.
            
            - **Points on the diagonal line** represent perfect calibration
            - **Points above the line** show underconfidence (model is more accurate than it thinks)
            - **Points below the line** show overconfidence (model is less accurate than it thinks)
            - **Point size** indicates the number of predictions in each confidence range
            
            Well-calibrated models have confidence scores that match their actual accuracy.
            """)
        
        # Calculate ECE (Expected Calibration Error)
        total_samples = vote_df.shape[0]
        ece = 0
        for _, row in conf_accuracy.iterrows():
            bin_weight = row['count'] / total_samples
            ece += bin_weight * abs(row['avg_confidence'] - row['accuracy'])
        
        # Display ECE metric
        st.metric("Expected Calibration Error (ECE)", f"{ece:.3f}", 
                 delta_color="inverse")  # Lower is better, so inverse the color
        
        # Add recommendation based on ECE
        if ece < 0.05:
            st.success("✅ **Model is well-calibrated**: Confidence scores closely match actual performance.")
        elif ece < 0.1:
            st.info("ℹ️ **Model is reasonably calibrated**: Minor adjustments may improve confidence estimates.")
        else:
            st.warning("⚠️ **Model is poorly calibrated**: Consider recalibration to improve confidence estimates.")
    
    with drift_tab:
        st.subheader("Alignment Drift Analysis")
        
        # Error Analysis section
        st.subheader("Error Distribution Analysis")
        
        # Calculate accuracy over time if timestamps are available
        if 'timestamp' in vote_df.columns:
            # Create a copy with datetime index
            df = vote_df.copy()
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Group by date and calculate accuracy
            accuracy_by_date = df.groupby('date')['model_correct'].agg(['mean', 'count']).reset_index()
            accuracy_by_date.columns = ['date', 'accuracy', 'count']
            
            # Calculate rolling average
            window = min(7, len(accuracy_by_date))
            if window > 1:
                accuracy_by_date['rolling_avg'] = accuracy_by_date['accuracy'].rolling(window=window).mean()
            
            # Create chart
            fig = px.line(
                accuracy_by_date,
                x='date',
                y='accuracy' if window <= 1 else 'rolling_avg',
                labels={'date': 'Date', 'accuracy': 'Accuracy', 'rolling_avg': 'Rolling Accuracy'},
                title='Model Accuracy Over Time'
            )
            
            # Add annotations for potential drift events
            significant_drops = []
            for i in range(1, len(accuracy_by_date)):
                if accuracy_by_date.iloc[i]['accuracy'] < accuracy_by_date.iloc[i-1]['accuracy'] - 0.2:
                    significant_drops.append(accuracy_by_date.iloc[i])
            
            for drop in significant_drops:
                fig.add_annotation(
                    x=drop['date'],
                    y=drop['accuracy'],
                    text="Potential Drift",
                    showarrow=True,
                    arrowhead=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show error distribution
        st.subheader("Error Distribution")
        
        # Count correct and incorrect predictions
        error_counts = vote_df['model_correct'].value_counts().reset_index()
        error_counts.columns = ['is_correct', 'count']
        
        # Create pie chart
        fig = px.pie(
            error_counts,
            values='count',
            names='is_correct',
            title='Distribution of Correct vs. Incorrect Predictions',
            color='is_correct',
            color_discrete_map={True: 'green', False: 'red'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error Clustering section
        st.subheader("Error Prompt Clusters")
        
        # Filter for incorrect predictions
        error_df = vote_df[vote_df['model_correct'] == False].copy()
        
        if error_df.empty:
            st.info("No errors found to cluster.")
            return
        
        if 'prompt' not in error_df.columns:
            st.warning("Prompt text not available for error analysis.")
            return
        
        # Check if we have enough errors to perform clustering
        MIN_SAMPLES_FOR_CLUSTERING = 5
        if len(error_df) < MIN_SAMPLES_FOR_CLUSTERING:
            st.info(f"Not enough error data (found {len(error_df)} errors) to generate meaningful clusters.")
            
            # Show the few errors we have
            st.write("Error examples:")
            for i, (_, row) in enumerate(error_df.iterrows()):
                st.markdown(f"**Error {i+1}:** {row.get('prompt', 'No prompt available')}")
            return
        
        # Extract prompts for clustering
        error_prompts = error_df['prompt'].tolist()
        
        try:
            # Create TF-IDF features
            tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                max_df=0.9,
                min_df=2
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(error_prompts)
            
            # Determine number of clusters
            n_clusters = min(5, len(error_prompts) // 2)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Add cluster labels to dataframe
            error_df['cluster'] = cluster_labels
            
            # Get the top terms for each cluster
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = tfidf_vectorizer.get_feature_names_out()
            
            # Display clusters
            for cluster_id in range(n_clusters):
                cluster_samples = error_df[error_df['cluster'] == cluster_id]
                
                with st.expander(f"Cluster {cluster_id+1} ({len(cluster_samples)} errors)"):
                    # Show top terms
                    top_terms = [terms[idx] for idx in order_centroids[cluster_id, :10]]
                    st.markdown(f"**Top terms:** {', '.join(top_terms)}")
                    
                    # Show example prompts
                    st.markdown("**Example prompts that led to errors:**")
                    for _, row in cluster_samples.head(3).iterrows():
                        st.markdown(f"- {row['prompt']}")
                        if 'confidence' in row:
                            st.caption(f"   Model confidence: {row.get('confidence', 'N/A'):.2%}")
        
        except Exception as e:
            st.error(f"Error performing clustering: {str(e)}")
            st.info("Showing error examples instead:")
            for i, (_, row) in enumerate(error_df.head(5).iterrows()):
                st.markdown(f"**Error {i+1}:** {row.get('prompt', 'No prompt available')}")
        
        # Suggestions for future enhancements
        st.markdown("""
        ### Future Enhancements
        
        - **Semantic clustering** using embeddings from language models would provide more meaningful clusters
        - **Temporal analysis** of clusters to track how error patterns evolve over time
        - **Interactive exploration** of individual examples within each cluster
        - **Advanced visualization** with 3D plots and interactive filtering
        - **Automatic error categorization** using supervised learning on annotated errors
            """)
    
    with themes_tab:
        st.subheader("Alignment by Theme and Content Type")
        
        # Check if we have any theme or category data
        theme_columns = [col for col in vote_df.columns if any(keyword in col.lower() for keyword in ['theme', 'category', 'topic', 'type'])]
        
        if not theme_columns:
            # Try to extract themes from prompts
            if 'prompt' in vote_df.columns:
                st.info("No explicit theme data found. Analyzing prompt content to identify themes...")
                
                # Simple keyword-based theme extraction
                common_themes = {
                    "Ethics": ["ethics", "moral", "fair", "bias", "justice", "right", "wrong"],
                    "Technology": ["technology", "computer", "software", "hardware", "code", "programming"],
                    "Science": ["science", "research", "experiment", "theory", "data", "evidence"],
                    "Creative": ["creative", "story", "write", "imagine", "poem", "art"],
                    "Reasoning": ["reason", "logic", "argument", "rational", "conclude", "inference"],
                    "Factual": ["fact", "information", "define", "explain", "describe", "history"]
                }
                
                # Extract themes from prompts
                theme_df = vote_df.copy()
                theme_df['extracted_theme'] = 'Other'
                
                for theme, keywords in common_themes.items():
                    for keyword in keywords:
                        theme_mask = theme_df['prompt'].str.lower().str.contains(keyword, na=False)
                        theme_df.loc[theme_mask, 'extracted_theme'] = theme
                
                # Group by extracted theme and calculate accuracy
                theme_accuracy = theme_df.groupby('extracted_theme').agg({
                    'model_correct': ['mean', 'count']
                }).reset_index()
                theme_accuracy.columns = ['Theme', 'Accuracy', 'Count']
                
                # Create bar chart
                fig = px.bar(
                    theme_accuracy,
                    x='Theme',
                    y='Accuracy',
                    color='Accuracy',
                    text='Count',
                    color_continuous_scale='RdYlGn',
                    range_color=[0.5, 1],
                    labels={
                        'Theme': 'Content Theme',
                        'Accuracy': 'Alignment Accuracy',
                        'Count': 'Number of Annotations'
                    },
                    title='Alignment Accuracy by Content Theme (Extracted from Prompts)'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Content Theme",
                    yaxis=dict(tickformat='.0%', range=[0, 1]),
                    yaxis_title="Alignment Accuracy",
                    coloraxis_showscale=False
                )
                
                # Display annotations on bars
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add analysis of theme differences
                best_theme = theme_accuracy.loc[theme_accuracy['Accuracy'].idxmax()]
                worst_theme = theme_accuracy.loc[theme_accuracy['Accuracy'].idxmin()]
                
                if best_theme['Count'] >= 5 and worst_theme['Count'] >= 5:
                    st.info(f"The model performs best on **{best_theme['Theme']}** content ({best_theme['Accuracy']:.1%} accuracy) and worst on **{worst_theme['Theme']}** content ({worst_theme['Accuracy']:.1%} accuracy).")
                    
                    if best_theme['Accuracy'] - worst_theme['Accuracy'] > 0.2:
                        st.warning(f"⚠️ **Significant alignment gap detected**: There's a {(best_theme['Accuracy'] - worst_theme['Accuracy']):.1%} difference in alignment between best and worst themes.")
            else:
                st.info("No theme or category data available in the annotations.")
                return
        else:
            # Select a theme column to analyze
            theme_col = st.selectbox("Select theme dimension", theme_columns)
            
            if theme_col in vote_df.columns:
                # Group by theme and calculate accuracy
                theme_df = vote_df.groupby(theme_col)['model_correct'].agg(['mean', 'count']).reset_index()
                theme_df.columns = [theme_col, 'Accuracy', 'Count']
                
                # Sort by count (descending)
                theme_df = theme_df.sort_values('Count', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    theme_df,
                    x=theme_col,
                    y='Accuracy',
                    color='Accuracy',
                    text='Count',
                    color_continuous_scale='RdYlGn',
                    range_color=[0.5, 1],
                    labels={
                        theme_col: 'Theme',
                        'Accuracy': 'Alignment Accuracy',
                        'Count': 'Number of Annotations'
                    },
                    title='Model Alignment by Theme'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title="Theme",
                    yaxis=dict(tickformat='.0%', range=[0, 1]),
                    yaxis_title="Alignment Accuracy",
                    coloraxis_showscale=False
                )
                
                # Display annotations on bars
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Add theme performance heatmap if we have prompt and domain data
        if 'prompt' in vote_df.columns and len(vote_df) > 10:
            st.subheader("Alignment Performance Heatmap")
            
            # Extract prompt length and word count
            prompt_df = vote_df.copy()
            prompt_df['prompt_length'] = prompt_df['prompt'].str.len()
            prompt_df['word_count'] = prompt_df['prompt'].str.split().str.len()
            
            # Create prompt length bins
            prompt_df['length_bin'] = pd.cut(
                prompt_df['prompt_length'],
                bins=[0, 50, 100, 200, 500, 1000],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
            
            # Create word count bins
            prompt_df['complexity_bin'] = pd.cut(
                prompt_df['word_count'],
                bins=[0, 5, 10, 20, 50, 100],
                labels=['Very Simple', 'Simple', 'Average', 'Complex', 'Very Complex']
            )
            
            # Group by length and complexity
            heatmap_data = prompt_df.groupby(['length_bin', 'complexity_bin'])['model_correct'].agg(['mean', 'count']).reset_index()
            
            # Create pivot table for heatmap
            pivot_data = heatmap_data.pivot_table(
                index='length_bin',
                columns='complexity_bin',
                values='mean',
                aggfunc='mean'
            ).fillna(0)
            
            # Create annotation counts pivot for hover data
            count_pivot = heatmap_data.pivot_table(
                index='length_bin',
                columns='complexity_bin',
                values='count',
                aggfunc='sum'
            ).fillna(0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                zmin=0.5,
                zmax=1,
                customdata=count_pivot.values,
                hovertemplate='Prompt Length: %{y}<br>Complexity: %{x}<br>Accuracy: %{z:.1%}<br>Count: %{customdata}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title="Alignment Performance by Prompt Characteristics",
                xaxis_title="Prompt Complexity",
                yaxis_title="Prompt Length",
                xaxis={'categoryorder': 'array', 'categoryarray': ['Very Simple', 'Simple', 'Average', 'Complex', 'Very Complex']},
                yaxis={'categoryorder': 'array', 'categoryarray': ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']}
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_calibration_diagnostics(vote_df, predictions_df):
    """Display enhanced calibration diagnostics and visualizations"""
    st.header("🎯 Calibration Diagnostics")
    
    # Check if we have the necessary data
    if vote_df.empty:
        st.warning("No annotation data available yet.")
        return
    
    if 'confidence' not in vote_df.columns or 'model_correct' not in vote_df.columns:
        st.warning("Confidence or correctness data not available. Cannot display calibration diagnostics.")
        return
    
    # Create tabs for different calibration visualizations
    reliability_tab, distribution_tab, calibration_tab, metrics_tab = st.tabs([
        "Reliability Diagram", 
        "Confidence Distribution", 
        "Calibration Metrics",
        "Calibration Over Time"
    ])
    
    with reliability_tab:
        st.subheader("Reliability Diagram")
        
        # Create bins for confidence values
        num_bins = 10
        vote_df_with_bins = vote_df.copy()
        vote_df_with_bins['confidence_bin'] = pd.cut(
            vote_df_with_bins['confidence'], 
            bins=num_bins, 
            labels=[f"{i/num_bins:.0%}-{(i+1)/num_bins:.0%}" for i in range(num_bins)]
        )
        
        # Calculate accuracy for each bin
        bin_stats = vote_df_with_bins.groupby('confidence_bin').agg({
            'confidence': 'mean',
            'model_correct': 'mean',
            'model_correct': 'count'
        }).reset_index()
        bin_stats.columns = ['bin', 'avg_confidence', 'accuracy', 'count']
        
        # Create enhanced reliability diagram
        fig = px.scatter(
            bin_stats,
            x='avg_confidence',
            y='accuracy',
            size='count',
            size_max=25,
            color='accuracy',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1],
            hover_data={
                'bin': True,
                'avg_confidence': ':.2%',
                'accuracy': ':.2%',
                'count': True
            },
            labels={
                'avg_confidence': 'Predicted Confidence',
                'accuracy': 'Observed Accuracy',
                'count': 'Samples in Bin'
            },
            title='Reliability Diagram: Model Calibration'
        )
        
        # Add perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='black', width=2, dash='dash')
            )
        )
        
        # Add smooth calibration curve
        if len(bin_stats) >= 5:
            try:
                from scipy.interpolate import make_interp_spline
                
                # Create smooth spline
                x = bin_stats['avg_confidence'].values
                y = bin_stats['accuracy'].values
                
                if len(x) >= 4:  # Need at least 4 points for cubic spline
                    x_smooth = np.linspace(min(x), max(x), 100)
                    spl = make_interp_spline(x, y, k=min(3, len(x)-1))
                    y_smooth = spl(x_smooth)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            name='Calibration Curve',
                            line=dict(color='blue', width=2)
                        )
                    )
            except:
                # Skip if scipy not available or other error
                pass
        
        # Improve layout
        fig.update_layout(
            xaxis=dict(title='Predicted Confidence', tickformat='.0%', range=[0, 1]),
            yaxis=dict(title='Observed Accuracy', tickformat='.0%', range=[0, 1]),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        with st.expander("Understanding the Reliability Diagram"):
            st.markdown("""
            ### How to Interpret the Reliability Diagram
            
            The reliability diagram shows how well-calibrated the model's confidence scores are:
            
            - **Perfect calibration** (diagonal line): When the model predicts with X% confidence, it should be correct X% of the time.
            - **Points above the line**: The model is underconfident (accuracy is higher than confidence).
            - **Points below the line**: The model is overconfident (confidence is higher than accuracy).
            - **Point size**: Number of predictions in each confidence bin.
            
            A well-calibrated model will have points clustering along the diagonal line.
            """)
    
    with distribution_tab:
        st.subheader("Confidence Distribution Analysis")
        
        # Create distribution by correctness
        correct_conf = vote_df[vote_df['model_correct'] == True]['confidence']
        incorrect_conf = vote_df[vote_df['model_correct'] == False]['confidence']
        
        # Create histogram data
        fig = go.Figure()
        
        # Add histogram for correct predictions
        fig.add_trace(go.Histogram(
            x=correct_conf,
            name='Correct Predictions',
            marker_color='rgba(0, 128, 0, 0.7)',
            opacity=0.7,
            xbins=dict(size=0.1),
            histnorm='probability'
        ))
        
        # Add histogram for incorrect predictions
        fig.add_trace(go.Histogram(
            x=incorrect_conf,
            name='Incorrect Predictions',
            marker_color='rgba(255, 0, 0, 0.7)',
            opacity=0.7,
            xbins=dict(size=0.1),
            histnorm='probability'
        ))
        
        # Update layout
        fig.update_layout(
            title='Distribution of Confidence by Prediction Correctness',
            xaxis=dict(title='Confidence Score', tickformat='.0%', range=[0, 1]),
            yaxis=dict(title='Density'),
            barmode='overlay',
            bargap=0.1,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add metrics on separation
        if len(correct_conf) > 0 and len(incorrect_conf) > 0:
            avg_correct_conf = correct_conf.mean()
            avg_incorrect_conf = incorrect_conf.mean()
            separation = avg_correct_conf - avg_incorrect_conf
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric(
                "Avg Confidence (Correct)", 
                f"{avg_correct_conf:.2%}"
            )
            
            col2.metric(
                "Avg Confidence (Incorrect)", 
                f"{avg_incorrect_conf:.2%}"
            )
            
            col3.metric(
                "Confidence Separation", 
                f"{separation:.2%}",
                delta_color="normal"
            )
            
            # Add interpretation
            if separation > 0.3:
                st.success("✅ **Good separation**: The model assigns much higher confidence to correct predictions than incorrect ones.")
            elif separation > 0.1:
                st.info("ℹ️ **Moderate separation**: The model shows some ability to distinguish correct from incorrect predictions.")
            else:
                st.warning("⚠️ **Poor separation**: The model's confidence scores don't effectively distinguish correct from incorrect predictions.")
    
    with calibration_tab:
        st.subheader("Calibration Metrics")
        
        # Calculate Expected Calibration Error (ECE)
        ece, bin_data = calculate_ece(vote_df, num_bins=10)
        
        # Calculate Maximum Calibration Error (MCE)
        bin_df = pd.DataFrame(bin_data)
        if not bin_df.empty:
            mce = bin_df['calibration_error'].abs().max()
        else:
            mce = float('nan')
        
        # Calculate Brier Score
        brier_score = ((vote_df['confidence'] - vote_df['model_correct'].astype(float)) ** 2).mean()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Expected Calibration Error (ECE)",
            f"{ece:.4f}",
            help="Lower is better. Weighted average of calibration errors across confidence bins."
        )
        
        col2.metric(
            "Maximum Calibration Error (MCE)",
            f"{mce:.4f}",
            help="Lower is better. Largest calibration error across all bins."
        )
        
        col3.metric(
            "Brier Score",
            f"{brier_score:.4f}",
            help="Lower is better. Mean squared error of confidence scores."
        )
        
        # Create bar chart for calibration error by bin
        if not bin_df.empty:
            # Add bin center as percentage
            bin_df['bin_center_pct'] = bin_df['bin_center'].apply(lambda x: f"{x:.0%}")
            
            # Create chart
            fig = px.bar(
                bin_df,
                x='bin_center_pct',
                y='calibration_error',
                color='calibration_error',
                color_continuous_scale='RdBu_r',  # Red for positive (overconfident), blue for negative (underconfident)
                range_color=[-0.3, 0.3],
                hover_data={
                    'bin_center': ':.2%',
                    'accuracy': ':.2%',
                    'sample_count': True,
                    'calibration_error': ':.3f'
                },
                labels={
                    'bin_center_pct': 'Confidence Bin',
                    'calibration_error': 'Calibration Error',
                    'sample_count': 'Samples',
                    'accuracy': 'Accuracy'
                },
                title='Calibration Error by Confidence Bin'
            )
            
            # Add horizontal line at y=0
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=0,
                x1=len(bin_df) - 0.5,
                y1=0,
                line=dict(color="black", width=1, dash="dash")
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Confidence",
                yaxis_title="Calibration Error (Confidence - Accuracy)",
                hovermode="closest",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            with st.expander("Understanding Calibration Error"):
                st.markdown("""
                ### Interpreting Calibration Error
                
                The calibration error chart shows the difference between predicted confidence and observed accuracy for each confidence bin:
                
                - **Positive values (red)**: Model is overconfident in this bin
                - **Negative values (blue)**: Model is underconfident in this bin
                - **Near zero**: Model is well-calibrated in this bin
                
                **Calibration Metrics:**
                
                - **ECE (Expected Calibration Error)**: Weighted average of absolute calibration errors
                - **MCE (Maximum Calibration Error)**: Largest absolute calibration error in any bin
                - **Brier Score**: Measures both calibration and accuracy (mean squared error of probabilities)
                """)
            
            # Add calibration recommendation
            st.subheader("Calibration Assessment")
            
            if ece < 0.01:
                st.success("✅ **Perfect calibration**: The model's confidence scores exactly match its accuracy (ECE < 0.01).")
            elif ece < 0.05:
                st.success("✅ **Excellent calibration**: The model's confidence scores closely match its accuracy (ECE < 0.05).")
            elif ece < 0.1:
                st.info("ℹ️ **Good calibration**: The model's confidence scores generally match its accuracy, with minor deviations (ECE < 0.1).")
            elif ece < 0.15:
                st.warning("⚠️ **Moderate miscalibration**: The model shows noticeable discrepancy between confidence and accuracy (ECE < 0.15).")
            else:
                st.error("❌ **Severe miscalibration**: The model's confidence scores significantly deviate from its accuracy (ECE ≥ 0.15).")
                
                # Add specific recommendations based on calibration pattern
                overconfident_bins = bin_df[bin_df['calibration_error'] > 0.1]
                underconfident_bins = bin_df[bin_df['calibration_error'] < -0.1]
                
                if len(overconfident_bins) > len(underconfident_bins) and len(overconfident_bins) > 0:
                    st.warning("🔍 **Pattern detected**: Model tends to be overconfident. Consider temperature scaling or confidence penalty during training.")
                elif len(underconfident_bins) > len(overconfident_bins) and len(underconfident_bins) > 0:
                    st.warning("🔍 **Pattern detected**: Model tends to be underconfident. Consider adjusting confidence scaling upward.")
    
    with metrics_tab:
        st.subheader("Calibration Over Time")
        
        # Check if we have timestamp data
        if 'timestamp' not in vote_df.columns:
            st.warning("Timestamp data not available for temporal analysis.")
            return
        
        # Create copy with date field
        time_df = vote_df.copy()
        time_df['date'] = pd.to_datetime(time_df['timestamp']).dt.date
        
        # Group by date and calculate metrics
        metrics_by_date = []
        
        for date, group in time_df.groupby('date'):
            # Skip days with too few samples
            if len(group) < 5:
                continue
                
            # Calculate ECE for this date
            date_ece, _ = calculate_ece(group, num_bins=min(5, len(group) // 2))
            
            # Calculate Brier score
            date_brier = ((group['confidence'] - group['model_correct'].astype(float)) ** 2).mean()
            
            # Calculate accuracy
            date_accuracy = group['model_correct'].mean()
            
            # Calculate average confidence
            date_confidence = group['confidence'].mean()
            
            # Add to metrics list
            metrics_by_date.append({
                'date': date,
                'ece': date_ece,
                'brier': date_brier,
                'accuracy': date_accuracy,
                'confidence': date_confidence,
                'count': len(group)
            })
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_by_date)
        
        if len(metrics_df) > 1:
            # Create time series chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add ECE line
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['ece'],
                    mode='lines+markers',
                    name='Expected Calibration Error',
                    line=dict(color='red', width=2)
                ),
                secondary_y=False
            )
            
            # Add accuracy line
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green', width=2)
                ),
                secondary_y=True
            )
            
            # Add confidence line
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['date'],
                    y=metrics_df['confidence'],
                    mode='lines+markers',
                    name='Avg Confidence',
                    line=dict(color='blue', width=2, dash='dash')
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title='Calibration Metrics Over Time',
                xaxis_title='Date',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig.update_yaxes(title_text='Calibration Error (ECE)', range=[0, 0.5], secondary_y=False)
            fig.update_yaxes(title_text='Accuracy / Confidence', tickformat='.0%', range=[0, 1], secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend analysis
            if len(metrics_df) >= 3:
                st.subheader("Calibration Trend Analysis")
                
                # Calculate trends
                first_half = metrics_df.iloc[:len(metrics_df)//2]
                second_half = metrics_df.iloc[len(metrics_df)//2:]
                
                ece_trend = second_half['ece'].mean() - first_half['ece'].mean()
                acc_trend = second_half['accuracy'].mean() - first_half['accuracy'].mean()
                conf_trend = second_half['confidence'].mean() - first_half['confidence'].mean()
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "ECE Trend", 
                    f"{second_half['ece'].mean():.3f}",
                    f"{ece_trend:.3f}",
                    delta_color="inverse"  # Lower is better
                )
                
                col2.metric(
                    "Accuracy Trend", 
                    f"{second_half['accuracy'].mean():.1%}",
                    f"{acc_trend:.1%}"
                )
                
                col3.metric(
                    "Confidence Trend", 
                    f"{second_half['confidence'].mean():.1%}",
                    f"{conf_trend:.1%}"
                )
                
                # Add interpretation
                if ece_trend < -0.02:
                    st.success("✅ **Improving calibration**: The model is becoming better calibrated over time.")
                elif ece_trend > 0.02:
                    st.warning("⚠️ **Declining calibration**: The model is becoming less well-calibrated over time.")
                else:
                    st.info("ℹ️ **Stable calibration**: The model's calibration is relatively stable over time.")
        else:
            st.info("Not enough time periods with sufficient data to analyze calibration trends.")

def calculate_ece(data_df, num_bins=10):
    """Calculate Expected Calibration Error (ECE) with detailed bin data"""
    # Ensure we have confidence and correctness columns
    if 'confidence' not in data_df.columns or 'model_correct' not in data_df.columns:
        return 1.0, []  # Return max error if data is missing
    
    # Create a copy of the data
    df = data_df.copy()
    
    # Ensure model_correct is numeric
    df['is_correct'] = df['model_correct'].astype(float)
    
    # Create bins
    bin_indices = pd.cut(df['confidence'], bins=num_bins, labels=False)
    df['bin'] = bin_indices
    
    # Calculate average confidence and accuracy for each bin
    bin_stats = df.groupby('bin').agg({
        'confidence': 'mean',
        'is_correct': 'mean',
        'bin': 'count'
    }).rename(columns={'bin': 'count'}).reset_index()
    
    # Calculate calibration error for each bin
    bin_stats['calibration_error'] = bin_stats['confidence'] - bin_stats['is_correct']
    
    # Calculate ECE
    total_samples = len(df)
    ece = 0
    for _, row in bin_stats.iterrows():
        bin_weight = row['count'] / total_samples
        ece += bin_weight * abs(row['calibration_error'])
    
    # Create bin data for visualization
    bin_data = []
    for _, row in bin_stats.iterrows():
        bin_data.append({
            'bin': int(row['bin']),
            'bin_center': row['confidence'],
            'accuracy': row['is_correct'],
            'sample_count': row['count'],
            'calibration_error': row['calibration_error']
        })
    
    return ece, bin_data

def display_drift_clusters(vote_df, predictions_df):
    """Display drift clusters and analysis"""
    st.header("🔄 Drift Clusters & Error Zones")
    
    # Check if we have the necessary data
    if vote_df.empty:
        st.warning("No annotation data available yet.")
        return
    
    # Check if we have correctness data
    if 'model_correct' not in vote_df.columns:
        st.warning("Model correctness data not available. Cannot display drift analysis.")
        return
    
    # Existing Error Analysis (Overall Error Analysis)
    st.subheader("Overall Error Analysis")
    
    # Calculate accuracy over time if timestamps are available
    if 'timestamp' in vote_df.columns:
        # Create a copy with datetime index
        df_accuracy_time = vote_df.copy()
        df_accuracy_time['date'] = pd.to_datetime(df_accuracy_time['timestamp']).dt.date
        
        # Group by date and calculate accuracy
        accuracy_by_date = df_accuracy_time.groupby('date')['model_correct'].agg(['mean', 'count']).reset_index()
        accuracy_by_date.columns = ['date', 'accuracy', 'count']
        
        # Calculate rolling average
        window = min(7, len(accuracy_by_date))
        if window > 1:
            accuracy_by_date['rolling_avg'] = accuracy_by_date['accuracy'].rolling(window=window).mean()
        
        # Create chart
        y_values = 'rolling_avg' if 'rolling_avg' in accuracy_by_date.columns else 'accuracy'
        fig_acc_time = px.line(
            accuracy_by_date,
            x='date',
            y=y_values,
            labels={'date': 'Date', y_values: 'Accuracy' if y_values == 'accuracy' else 'Rolling Accuracy'},
            title='Model Accuracy Over Time (All Predictions)'
        )
        
        # Add annotations for potential drift events
        significant_drops = []
        if 'accuracy' in accuracy_by_date.columns and len(accuracy_by_date) > 1:
        for i in range(1, len(accuracy_by_date)):
                if accuracy_by_date.iloc[i]['accuracy'] < accuracy_by_date.iloc[i-1]['accuracy'] - 0.2: # Example threshold
                significant_drops.append(accuracy_by_date.iloc[i])
        
        for drop in significant_drops:
            fig_acc_time.add_annotation(
                x=drop['date'],
                y=drop['accuracy'], # Use actual accuracy for annotation y-position
                text="Potential Drift",
                showarrow=True,
                arrowhead=1
            )
        
        st.plotly_chart(fig_acc_time, use_container_width=True)
    
    # Show error distribution
    error_counts = vote_df['model_correct'].value_counts().reset_index()
    error_counts.columns = ['is_correct', 'count']
    error_counts['label'] = error_counts['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    fig_error_dist = px.pie(
        error_counts,
        values='count',
        names='label',
        title='Distribution of Correct vs. Incorrect Predictions',
        color='label',
        color_discrete_map={True: 'green', False: 'red', 'Correct': 'green', 'Incorrect': 'red'} # map both bool and string
    )
    st.plotly_chart(fig_error_dist, use_container_width=True)

    # New Drift Cluster Analysis for Error Prompts
    st.subheader("Error Prompt Clusters")
    
    error_df = vote_df[vote_df['model_correct'] == False].copy()

    if error_df.empty:
        st.info("No errors found to cluster.")
        # Keep the future enhancements markdown for context
        st.markdown("""
    ### Future Enhancements
    
    The full implementation of drift clusters would include:
    
        - Dimensionality reduction (UMAP/t-SNE) of prompt and completion embeddings (using semantic embeddings)
    - Clustering of errors to identify patterns
    - Temporal analysis of error clusters to detect drift
    - Interactive 3D visualization of drift clusters
    - Drilldown into specific examples within each cluster
        """)
        return

    MIN_SAMPLES_FOR_CLUSTERING = 5  # Minimum samples to attempt clustering
    N_CLUSTERS_DEFAULT = min(5, len(error_df)) # Default number of clusters, ensure it's not more than samples

    if len(error_df) < MIN_SAMPLES_FOR_CLUSTERING:
        st.info(f"Not enough error data (found {len(error_df)} prompts, need at least {MIN_SAMPLES_FOR_CLUSTERING}) to generate meaningful clusters at this time.")
        st.markdown("""
        ### Future Enhancements 
        ... (as above) ...
        """)
        return
    
    error_prompts = error_df['prompt'].tolist()

    try:
        # TF-IDF Vectorization
        # Using a smaller max_features for initial implementation to manage memory/performance.
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.90, min_df=max(1, int(len(error_prompts)*0.05)), max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(error_prompts)

        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            st.info("Could not generate features from error prompts for clustering (e.g., all prompts too short or common).")
            st.markdown("""
            ### Future Enhancements 
            ... (as above) ...
            """)
            return

        if UMAP_AVAILABLE:
            n_samples_for_umap = tfidf_matrix.shape[0]
            # UMAP n_neighbors must be less than n_samples.
            umap_n_neighbors = max(2, min(15, n_samples_for_umap - 1))
            
            if n_samples_for_umap <= umap_n_neighbors or umap_n_neighbors < 2 : # n_neighbors must be at least 2
                 st.info(f"Too few unique samples ({n_samples_for_umap}) for UMAP with n_neighbors={umap_n_neighbors}. Skipping UMAP plot. Consider more error data.")
                 # Fallback or just don't show UMAP plot
            else:
                umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=umap_n_neighbors, min_dist=0.1, metric='cosine')
                umap_embeddings = umap_reducer.fit_transform(tfidf_matrix) # UMAP can handle sparse tfidf_matrix

                # Clustering (KMeans)
                actual_n_clusters = min(N_CLUSTERS_DEFAULT, n_samples_for_umap)
                if actual_n_clusters <= 1 and n_samples_for_umap > 1 : # Avoid clustering if only 1 cluster makes sense, unless only 1 sample
                    actual_n_clusters = 1 # Group all into one if too few for multiple meaningful clusters
                elif n_samples_for_umap == 1:
                     actual_n_clusters = 1
                
                if actual_n_clusters == 0 : # Should not happen with current logic but as a safeguard
                    st.info("Not enough samples to form any clusters.")
                
                elif n_samples_for_umap < actual_n_clusters:
                    st.info(f"Number of samples ({n_samples_for_umap}) is less than the number of desired clusters ({actual_n_clusters}). Adjusting cluster count.")
                    actual_n_clusters = n_samples_for_umap


                if actual_n_clusters > 0:
                    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
                    # Kmeans prefers dense arrays, but can work with sparse for some solvers.
                    # However, umap_embeddings are dense.
                    cluster_labels = kmeans.fit_predict(umap_embeddings)
                    error_df['cluster'] = cluster_labels.astype(str) # Make categorical for Plotly

                    # Visualization (Plotly Scatter for UMAP)
                    plot_df_umap = pd.DataFrame(umap_embeddings, columns=['UMAP_1', 'UMAP_2'])
                    plot_df_umap['cluster'] = error_df['cluster'].values # Ensure alignment
                    plot_df_umap['prompt'] = error_df['prompt'].values # For hover data

                    fig_umap = px.scatter(plot_df_umap, x='UMAP_1', y='UMAP_2', color='cluster',
                                        hover_data={'prompt': True, 'cluster': True, 'UMAP_1':':.2f', 'UMAP_2':':.2f'},
                                        title='Error Prompt Clusters (UMAP + KMeans with TF-IDF Features)')
                    st.plotly_chart(fig_umap, use_container_width=True)

                    # Cluster Statistics
                    st.subheader("Cluster Details")
                    count_vectorizer = CountVectorizer(stop_words='english', max_features=5, ngram_range=(1,1))

                    sorted_clusters = sorted(error_df['cluster'].unique())

                    for cluster_id_str in sorted_clusters:
                        cluster_df_view = error_df[error_df['cluster'] == cluster_id_str]
                        num_prompts_in_cluster = len(cluster_df_view)

                        if num_prompts_in_cluster == 0:
                            continue
                        
                        with st.expander(f"Cluster {cluster_id_str} ({num_prompts_in_cluster} erroneous prompts)"):
                            cluster_prompts_text_list = cluster_df_view['prompt'].tolist()
                            if len(cluster_prompts_text_list) > 0:
                                try:
                                    term_matrix = count_vectorizer.fit_transform(cluster_prompts_text_list)
                                    top_keywords = count_vectorizer.get_feature_names_out().tolist()
                                    st.markdown(f"**Top Keywords:** {', '.join(top_keywords) if top_keywords else 'N/A'}")
                                except ValueError:
                                    st.markdown("**Top Keywords:** Not enough distinct words.")
                            else:
                                st.markdown("**Top Keywords:** N/A")

                            st.markdown("**Sample Prompts (from errors):**")
                            for _, row in cluster_df_view.head(min(3, num_prompts_in_cluster)).iterrows():
                                st.caption(f"- {row['prompt']}")
                                if 'confidence' in row and pd.notna(row['confidence']):
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_Confidence: {row['confidence']:.2%}_")
                            
                            if 'confidence' in cluster_df_view.columns:
                                avg_confidence = cluster_df_view['confidence'].mean()
                                if pd.notna(avg_confidence):
                                    st.markdown(f"**Average Model Confidence (for this cluster's errors):** {avg_confidence:.2%}")
                else:
                    st.info("Could not form clusters with the current error data.")
        else: # UMAP not available
            st.warning("`umap-learn` library not found or insufficient data for UMAP. UMAP visualization for clusters is skipped. Please install it (`pip install umap-learn`) for this feature.")
            st.markdown("Basic listing of error prompts:")
            st.dataframe(error_df[['prompt', 'timestamp'] + (['confidence'] if 'confidence' in error_df.columns else [])].head(20))

    except ImportError: # Catch sklearn imports
        st.error("Clustering libraries (scikit-learn) not found. Please install them (`pip install scikit-learn`) for this feature.")
    except Exception as e:
        # Use the logger from the top of the file if available, otherwise plain print
        try:
            logger.error(f"Error during drift cluster generation: {e}", exc_info=True)
        except NameError:
            print(f"Error during drift cluster generation: {e}")
        st.error(f"An error occurred while generating drift clusters: {str(e)[:200]}...") # Show a snippet of the error

    # Placeholder for full implementation details (can be refined)
    st.markdown("""
    ### Future Enhancements & Notes
    
    - The current clustering uses TF-IDF features from error prompts. For more semantically meaningful clusters, **prompt and completion embeddings** (e.g., from sentence transformers) should be used.
    - **Temporal analysis** of these clusters could reveal how error patterns evolve over time.
    - Interactive **drilldown** into specific examples within each cluster is planned.
    - Adding **Cluster Entropy Over Time** would provide a quantitative measure of error distribution changes.
    - Consider **DBSCAN** or other clustering algorithms that don't require specifying the number of clusters.
    """)

def display_model_evolution(vote_df, predictions_df):
    """Display model evolution over time"""
    st.header("🧬 Model Evolution")
    
    # Create placeholder for full implementation
    st.info("Model evolution analysis requires historical checkpoint data.")
    
    # Show basic metrics from available data
    if not vote_df.empty and 'model_correct' in vote_df.columns:
        st.subheader("Current Model Performance")
        
        # Calculate overall accuracy
        overall_accuracy = vote_df['model_correct'].mean()
        
        # Calculate metrics by time period if timestamps are available
        if 'timestamp' in vote_df.columns:
            # Create a copy with datetime index
            df = vote_df.copy()
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Sort by date
            df = df.sort_values('date')
            
            # Split into time periods
            if len(df) >= 20:
                # Use quantiles to split the data
                df['period'] = pd.qcut(df.index, 3, labels=['Early', 'Middle', 'Recent'])
                
                # Calculate metrics for each period
                period_metrics = df.groupby('period').agg({
                    'model_correct': 'mean',
                    'confidence': 'mean' if 'confidence' in df.columns else 'count'
                }).reset_index()
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                
                    with col1:
                    st.metric(
                        "Early Accuracy", 
                        f"{period_metrics.iloc[0]['model_correct']:.2%}"
                        )
                    
                    with col2:
                    st.metric(
                        "Middle Accuracy", 
                        f"{period_metrics.iloc[1]['model_correct']:.2%}",
                        delta=f"{period_metrics.iloc[1]['model_correct'] - period_metrics.iloc[0]['model_correct']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        "Recent Accuracy", 
                        f"{period_metrics.iloc[2]['model_correct']:.2%}",
                        delta=f"{period_metrics.iloc[2]['model_correct'] - period_metrics.iloc[1]['model_correct']:.2%}"
                    )
                
                # Create evolution chart
                fig = px.line(
                    period_metrics,
                    x='period',
                    y='model_correct',
                    markers=True,
                    labels={'period': 'Time Period', 'model_correct': 'Accuracy'},
                    title='Model Accuracy Evolution'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show confidence evolution if available
                if 'confidence' in period_metrics.columns:
                    fig = px.line(
                        period_metrics,
                        x='period',
                        y='confidence',
                        markers=True,
                        labels={'period': 'Time Period', 'confidence': 'Confidence'},
                        title='Model Confidence Evolution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Analyze error patterns over time
                if len(df) >= 30:
                    st.subheader("Error Pattern Evolution")
                    
                    # Group by period and model correctness
                    error_evolution = df.groupby(['period', 'model_correct']).size().reset_index(name='count')
                    error_evolution = error_evolution.pivot(index='period', columns='model_correct', values='count').fillna(0)
                    
                    # Calculate error rate
                    if True in error_evolution.columns and False in error_evolution.columns:
                        error_evolution['total'] = error_evolution[True] + error_evolution[False]
                        error_evolution['error_rate'] = error_evolution[False] / error_evolution['total']
                        
                        # Create bar chart
                        fig = px.bar(
                            error_evolution.reset_index(),
                            x='period',
                            y='error_rate',
                            labels={'period': 'Time Period', 'error_rate': 'Error Rate'},
                            title='Error Rate Evolution'
                        )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to split into time periods for evolution analysis.")
                
                # Display overall metrics
                st.metric("Overall Accuracy", f"{overall_accuracy:.2%}")
                            else:
            # Display overall metrics
            st.metric("Overall Accuracy", f"{overall_accuracy:.2%}")
    else:
        st.warning("No model performance data available yet.")
    
    # Placeholder for full implementation
    st.markdown("""
    ### Future Enhancements
    
    The full implementation of model evolution would include:
    
    - Tracking of model versions and checkpoints
    - Comparison of performance metrics across versions
    - Side-by-side reliability diagrams for different checkpoints
    - Visualization of changes in model behavior over time
    - Analysis of training data impact on model performance
    """)

def generate_completion(prompt_text, temperature=0.7, stream=False):
    """Generate a completion using the API client"""
    api_client = get_api_client()
    
    # Create a placeholder if streaming is enabled
    if stream:
        stream_placeholder = st.empty()
        stream_placeholder.text("Generating completion...\n\n")
        
        # Create a callback function for streaming
        def stream_callback(content_piece):
            # This will be called with each chunk of the response
            if 'stream_text' not in st.session_state:
                st.session_state.stream_text = ""
            st.session_state.stream_text += content_piece
            stream_placeholder.text(f"Generating...\n\n{st.session_state.stream_text}")
        
        # Generate completion with streaming
        completion = api_client.generate_completion(
            prompt_text, 
            temperature=temperature,
            stream=True,
            stream_callback=stream_callback
        )
        
        # Clear the placeholder once generation is complete
        stream_placeholder.empty()
        
        # Clear stream text for next generation
        if 'stream_text' in st.session_state:
            del st.session_state.stream_text
    else:
        # Generate completion without streaming
        completion = api_client.generate_completion(
            prompt_text,
            temperature=temperature
        )
    
    return completion

def generate_chat_response(prompt, system_prompt=None):
    """Generate a chat response using the API client"""
    api_client = get_api_client()
    
    # If we only have a prompt and no chat history, create a simple message structure
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
                    else:
        # Assume prompt is already a list of messages
        messages = prompt
    
    # Generate response
    response = api_client.generate_chat_response(
        messages,
        system_prompt=system_prompt
    )
    
    return response

def main():
    """Main entry point for the Streamlit dashboard."""
    st.title("RLHF Attunement Dashboard")
    
    # Add sidebar
    with st.sidebar:
        st.header("Navigation")
        
        # Add auto-refresh toggle
        st.session_state.auto_refresh = st.checkbox(
            "Auto-refresh data", 
            value=st.session_state.auto_refresh,
            help=f"Automatically refresh data every {AUTO_REFRESH_INTERVAL} seconds"
        )
        
        # Add manual refresh button
        if st.button("🔄 Refresh Data"):
            load_all_data(force_reload=True)
            st.success("Data refreshed!")
        
        # Check if we need to auto-refresh
        if st.session_state.auto_refresh and time.time() - st.session_state.last_refresh_time > AUTO_REFRESH_INTERVAL:
            load_all_data(force_reload=True)
            
        # Create tabs in the sidebar for navigation
        tab_options = [
            "Dashboard", 
            "Chat Interface", 
            "Annotation Interface", 
            "Annotation History", 
            "Alignment Over Time",
            "Calibration Diagnostics", 
            "Drift Clusters & Error Zones", 
            "Model Evolution", 
            "User Preference Timeline"
        ]
        
        selected_tab = st.radio("Select Tab", tab_options)
    
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
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        vote_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        reflections_df = pd.DataFrame()
        data_summary = {}
        
    # Handle different tabs
    if selected_tab == "Dashboard":
        show_dashboard_overview(vote_df, predictions_df, reflections_df, data_summary)
    elif selected_tab == "Chat Interface":
        display_chat_interface()
    elif selected_tab == "Annotation Interface":
        display_annotation_interface(vote_df)
    elif selected_tab == "Annotation History":
        display_annotation_history(vote_df, predictions_df)
    elif selected_tab == "Alignment Over Time":
        display_alignment_over_time(vote_df, predictions_df)
    elif selected_tab == "Calibration Diagnostics":
        display_calibration_diagnostics(vote_df, predictions_df)
    elif selected_tab == "Drift Clusters & Error Zones":
        display_drift_clusters(vote_df, predictions_df)
    elif selected_tab == "Model Evolution":
        display_model_evolution(vote_df, predictions_df)
    elif selected_tab == "User Preference Timeline":
        display_preference_timeline(vote_df)
    else:
        st.warning("Tab not implemented yet")

if __name__ == "__main__":
    main()