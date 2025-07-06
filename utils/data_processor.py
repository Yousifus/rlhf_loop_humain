#!/usr/bin/env python3
"""
Data processor for RLHF training.

This module processes annotations from the dashboard into training data formats
for reward modeling and reinforcement learning.
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root
project_root = str(Path(__file__).resolve().parents[1])

# Import database module
sys.path.append(project_root)
from utils.database import get_database, RLHFDatabase

def process_annotations_for_training(output_path: Optional[str] = None) -> str:
    """
    Process annotations into training data for reward modeling.
    
    Args:
        output_path: Path to save the processed data. If None, will use default path.
        
    Returns:
        Path to the processed training data file
    """
    # Get database
    db = get_database()
    
    # Get annotations
    annotations_df = db.get_annotations(force_reload=True)
    
    if annotations_df.empty:
        logger.warning("No annotations found to process")
        return ""
    
    # Filter for binary preferences (Completion A vs Completion B)
    # Use a simpler approach that's more resilient to column differences
    binary_prefs = annotations_df
    if 'is_binary_preference' in annotations_df.columns:
        binary_prefs = annotations_df[annotations_df['is_binary_preference'] == True]
    
    # Additional filter to ensure we have both selected and rejected completions
    binary_prefs = binary_prefs[
        binary_prefs['selected_completion'].notna() & 
        binary_prefs['rejected_completion'].notna()
    ]
    
    if binary_prefs.empty:
        logger.warning("No binary preference annotations found")
        return ""
    
    # Create reward modeling format
    training_data = []
    
    for _, row in binary_prefs.iterrows():
        # Get basic data
        prompt = row.get('prompt', '')
        chosen = row.get('selected_completion', '')
        rejected = row.get('rejected_completion', '')
        
        # Convert timestamp to string if it's a pandas Timestamp
        timestamp = row.get('timestamp', '')
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        
        # Create training example
        example = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "prompt_id": row.get('prompt_id', ''),
                "annotation_id": row.get('annotation_id', ''),
                "timestamp": timestamp,
                "feedback": row.get('feedback', ''),
                "quality_metrics": row.get('quality_metrics', {})
            }
        }
        
        training_data.append(example)
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(project_root, "data", f"reward_model_training_data_{timestamp}.jsonl")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Processed {len(training_data)} annotations into training data at {output_path}")
    
    return output_path

def process_chat_data_for_training(output_path: Optional[str] = None) -> str:
    """
    Process chat interactions into training data for finetuning.
    
    Args:
        output_path: Path to save the processed data. If None, will use default path.
        
    Returns:
        Path to the processed training data file
    """
    # Define the paths
    chat_logs_dir = os.path.join(project_root, "data", "chat_logs")
    
    if not os.path.exists(chat_logs_dir):
        logger.warning(f"Chat logs directory not found: {chat_logs_dir}")
        return ""
    
    # Find all chat log files
    chat_files = list(Path(chat_logs_dir).glob("*.json"))
    
    if not chat_files:
        logger.warning("No chat log files found")
        return ""
    
    # Process chat logs
    training_data = []
    
    for chat_file in chat_files:
        try:
            with open(chat_file, 'r') as f:
                chat_data = json.load(f)
                
            # Process each chat session
            if "messages" in chat_data:
                messages = chat_data["messages"]
                
                # Only process sessions with at least one user and assistant message
                if any(m["role"] == "user" for m in messages) and any(m["role"] == "assistant" for m in messages):
                    example = {
                        "messages": messages,
                        "metadata": {
                            "session_id": chat_data.get("session_id", ""),
                            "timestamp": chat_data.get("timestamp", ""),
                            "quality_metrics": chat_data.get("quality_metrics", {})
                        }
                    }
                    training_data.append(example)
        except Exception as e:
            logger.error(f"Error processing chat file {chat_file}: {e}")
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(project_root, "data", f"chat_training_data_{timestamp}.jsonl")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Processed {len(training_data)} chat sessions into training data at {output_path}")
    
    return output_path

def visualize_training_data(training_data_path: str) -> Dict[str, Any]:
    """
    Generate statistics and visualizations for training data.
    
    Args:
        training_data_path: Path to the training data file
        
    Returns:
        Dictionary of statistics about the training data
    """
    if not os.path.exists(training_data_path):
        logger.error(f"Training data file not found: {training_data_path}")
        return {}
    
    # Load training data
    data = []
    with open(training_data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        logger.warning("No training data found")
        return {}
    
    # Calculate statistics
    stats = {
        "total_examples": len(data),
        "avg_prompt_length": sum(len(ex.get("prompt", "")) for ex in data) / max(1, len(data)),
        "avg_chosen_length": sum(len(ex.get("chosen", "")) for ex in data) / max(1, len(data)),
        "avg_rejected_length": sum(len(ex.get("rejected", "")) for ex in data) / max(1, len(data)),
        "has_quality_metrics": sum(1 for ex in data if ex.get("metadata", {}).get("quality_metrics", {})) / max(1, len(data))
    }
    
    # Add timestamp
    stats["timestamp"] = datetime.now().isoformat()
    
    # Save statistics
    stats_path = training_data_path.replace(".jsonl", "_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Generated statistics for training data at {stats_path}")
    
    return stats

if __name__ == "__main__":
    import sys
    
    # Process annotations
    training_data_path = process_annotations_for_training()
    
    if training_data_path:
        # Visualize the data
        stats = visualize_training_data(training_data_path)
        print(f"Processed {stats.get('total_examples', 0)} examples for training")
        print(f"Average prompt length: {stats.get('avg_prompt_length', 0):.1f} chars")
        print(f"Training data saved to: {training_data_path}")
    else:
        print("No annotations processed. Please create some annotations first.") 