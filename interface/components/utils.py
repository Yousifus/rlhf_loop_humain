"""
RLHF Utility Functions

Essential utility functions for the RLHF monitoring system,
providing data processing and interface management capabilities.
"""

from datetime import datetime, timedelta
import re
import uuid
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import streamlit as st
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# System Configuration Constants
AUTO_REFRESH_INTERVAL = 60  # Data refresh interval (seconds)
ANNOTATION_TARGET = 500  # Target number of annotations for training
TARGET_ACCURACY = 0.85  # Target model accuracy threshold

def format_timestamp(ts):
    """Format timestamp for dashboard display"""
    if ts is None:
        return "N/A"
    
    try:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        
        # Format datetime object
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts)

def create_time_slider(data_df, timestamp_col='timestamp', label="Select Time Range for Analysis"):
    """Create a time range selector for filtering data"""
    if data_df.empty or timestamp_col not in data_df.columns:
        return data_df
    
    # Ensure timestamp column is in datetime format
    if data_df[timestamp_col].dtype != 'datetime64[ns]':
        try:
            data_df[timestamp_col] = pd.to_datetime(data_df[timestamp_col])
        except Exception as e:
            st.warning(f"Error processing timestamp data: {e}")
            return data_df
    
    # Get min and max dates from the data
    min_date = data_df[timestamp_col].min().date()
    max_date = data_df[timestamp_col].max().date()
    
    # Handle case where all data is from the same date
    if min_date == max_date:
        st.info(f"📅 All data is from {min_date.strftime('%Y-%m-%d')}. Showing complete dataset.")
        return data_df
    
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
    """Filter data by specified time range"""
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
            st.warning(f"Error processing timestamp data for filtering: {e}")
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
        st.warning(f"Error filtering data by time range: {e}")
        return data_df

def generate_unique_id(prefix='annotation'):
    """Generate a unique identifier for training data records"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def save_data_to_json(data, filepath):
    """Save data to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to file: {e}")
        return False

def load_data_from_json(filepath):
    """Load data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from file: {e}")
        return None 