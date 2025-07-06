#!/usr/bin/env python3
"""
Test script to generate reflection data for testing the dashboard

This script creates some test reflection data entries and writes them
to the meta_reflection_log.jsonl file in the models directory.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_test_reflection_data(count=5):
    """Generate test reflection data entries"""
    entries = []
    
    for i in range(count):
        # Generate random data
        is_correct = random.choice([True, False])
        confidence = random.uniform(0.5, 0.95)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_id": f"test_prompt_{i}",
            "prompt": f"Test prompt {i} for reflection data",
            "selected_completion_id": f"completion_a_{i}",
            "selected_completion": f"This is test completion A for prompt {i}",
            "rejected_completion_id": f"completion_b_{i}",
            "rejected_completion": f"This is test completion B for prompt {i}",
            "model_prediction": f"completion_{'a' if is_correct else 'b'}",
            "model_prediction_confidence_raw": confidence,
            "model_prediction_confidence_calibrated": confidence * 0.8,
            "is_prediction_correct": is_correct,
            "confidence_error": confidence if not is_correct else 1.0 - confidence,
            "prediction_error_type": "none" if is_correct else 
                                  "overconfidence" if confidence > 0.7 else "low_confidence",
            "drift_cluster_id": None
        }
        entries.append(entry)
    
    return entries

def save_test_reflection_data(entries, output_path=None):
    """Save test reflection data to file"""
    if output_path is None:
        output_path = os.path.join(project_root, "models", "meta_reflection_log.jsonl")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save entries
    try:
        with open(output_path, "a") as f:  # Use "a" to append to existing data instead of overwriting
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Saved {len(entries)} test reflection entries to {output_path}")
        
        # Verify the file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Reflection file size is now {file_size} bytes")
            if file_size == 0:
                logger.warning("⚠️ Warning: Reflection file exists but is empty")
        else:
            logger.error(f"❌ Error: Reflection file {output_path} doesn't exist after writing")
            
        return True
    except Exception as e:
        logger.error(f"❌ Error saving reflection data: {e}")
        return False

def main():
    """Main function"""
    logger.info("Generating test reflection data...")
    entries = generate_test_reflection_data(count=10)
    
    logger.info("Saving test reflection data...")
    success = save_test_reflection_data(entries)
    
    if success:
        logger.info("✅ Test reflection data saved successfully")
    else:
        logger.error("❌ Failed to save test reflection data")

if __name__ == "__main__":
    main() 
"""
Test script to generate reflection data for testing the dashboard

This script creates some test reflection data entries and writes them
to the meta_reflection_log.jsonl file in the models directory.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_test_reflection_data(count=5):
    """Generate test reflection data entries"""
    entries = []
    
    for i in range(count):
        # Generate random data
        is_correct = random.choice([True, False])
        confidence = random.uniform(0.5, 0.95)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_id": f"test_prompt_{i}",
            "prompt": f"Test prompt {i} for reflection data",
            "selected_completion_id": f"completion_a_{i}",
            "selected_completion": f"This is test completion A for prompt {i}",
            "rejected_completion_id": f"completion_b_{i}",
            "rejected_completion": f"This is test completion B for prompt {i}",
            "model_prediction": f"completion_{'a' if is_correct else 'b'}",
            "model_prediction_confidence_raw": confidence,
            "model_prediction_confidence_calibrated": confidence * 0.8,
            "is_prediction_correct": is_correct,
            "confidence_error": confidence if not is_correct else 1.0 - confidence,
            "prediction_error_type": "none" if is_correct else 
                                  "overconfidence" if confidence > 0.7 else "low_confidence",
            "drift_cluster_id": None
        }
        entries.append(entry)
    
    return entries

def save_test_reflection_data(entries, output_path=None):
    """Save test reflection data to file"""
    if output_path is None:
        output_path = os.path.join(project_root, "models", "meta_reflection_log.jsonl")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save entries
    try:
        with open(output_path, "a") as f:  # Use "a" to append to existing data instead of overwriting
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"Saved {len(entries)} test reflection entries to {output_path}")
        
        # Verify the file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"Reflection file size is now {file_size} bytes")
            if file_size == 0:
                logger.warning("⚠️ Warning: Reflection file exists but is empty")
        else:
            logger.error(f"❌ Error: Reflection file {output_path} doesn't exist after writing")
            
        return True
    except Exception as e:
        logger.error(f"❌ Error saving reflection data: {e}")
        return False

def main():
    """Main function"""
    logger.info("Generating test reflection data...")
    entries = generate_test_reflection_data(count=10)
    
    logger.info("Saving test reflection data...")
    success = save_test_reflection_data(entries)
    
    if success:
        logger.info("✅ Test reflection data saved successfully")
    else:
        logger.error("❌ Failed to save test reflection data")

if __name__ == "__main__":
    main() 