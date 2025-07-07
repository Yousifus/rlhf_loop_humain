#!/usr/bin/env python3
"""
Script to train a reward model from collected annotations.

This script processes annotations from the dashboard and trains a reward model
that can be used to provide rewards for reinforcement learning.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root - fix to get the actual project root, not scripts directory
project_root = str(Path(__file__).resolve().parent.parent)  # Go up one more level from scripts/
sys.path.append(project_root)

from utils.data_processor import process_annotations_for_training, visualize_training_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a reward model from RLHF annotations")
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        help="Path to the training data file. If not provided, will generate from annotations."
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=os.path.join(project_root, "models", "reward_model"),
        help="Directory to save the trained model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training iterations"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Fraction of data to use for testing"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Prepare data but don't actually train the model"
    )
    
    return parser.parse_args()

def prepare_training_features(data):
    """
    Prepare features for training from RLHF data.
    
    Args:
        data: List of training examples from RLHF annotations
        
    Returns:
        X: Feature matrix
        y: Labels (1 for chosen, 0 for rejected)
        feature_names: Names of features
    """
    logger.info("Preparing training features from RLHF data...")
    
    # Extract text data for feature engineering
    prompts = []
    chosen_completions = []
    rejected_completions = []
    
    for example in data:
        prompts.append(example.get('prompt', ''))
        chosen_completions.append(example.get('chosen', ''))
        rejected_completions.append(example.get('rejected', ''))
    
    # Create TF-IDF features for prompts and completions
    vectorizer_prompt = TfidfVectorizer(max_features=100, stop_words='english')
    vectorizer_completion = TfidfVectorizer(max_features=100, stop_words='english')
    
    # Fit vectorizers on all text
    all_prompts = prompts
    all_completions = chosen_completions + rejected_completions
    
    vectorizer_prompt.fit(all_prompts)
    vectorizer_completion.fit(all_completions)
    
    # Create training pairs (prompt + chosen vs prompt + rejected)
    X = []
    y = []
    
    for i, example in enumerate(data):
        prompt = prompts[i]
        chosen = chosen_completions[i]
        rejected = rejected_completions[i]
        
        # Features for prompt
        prompt_features = vectorizer_prompt.transform([prompt]).toarray()[0]
        
        # Features for chosen completion (positive example)
        chosen_features = vectorizer_completion.transform([chosen]).toarray()[0]
        chosen_length = len(chosen)
        chosen_example = np.concatenate([prompt_features, chosen_features, [chosen_length]])
        X.append(chosen_example)
        y.append(1)  # Chosen = positive
        
        # Features for rejected completion (negative example)
        rejected_features = vectorizer_completion.transform([rejected]).toarray()[0]
        rejected_length = len(rejected)
        rejected_example = np.concatenate([prompt_features, rejected_features, [rejected_length]])
        X.append(rejected_example)
        y.append(0)  # Rejected = negative
    
    X = np.array(X)
    y = np.array(y)
    
    feature_names = (
        [f"prompt_tfidf_{i}" for i in range(100)] +
        [f"completion_tfidf_{i}" for i in range(100)] +
        ["completion_length"]
    )
    
    logger.info(f"Created {len(X)} training examples with {X.shape[1]} features")
    
    return X, y, feature_names, vectorizer_prompt, vectorizer_completion

def train_reward_model(X, y, epochs, output_dir):
    """
    Train the actual reward model on real RLHF data.
    
    Args:
        X: Feature matrix
        y: Labels
        epochs: Number of training iterations
        output_dir: Directory to save the trained model
        
    Returns:
        model: Trained model
        metrics: Training metrics
    """
    logger.info(f"Training reward model on {len(X)} real RLHF examples...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Use simpler model for small datasets to reduce overfitting
    if len(X) < 50:  # Small dataset
        logger.info("Using simplified model for small dataset to reduce overfitting")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=0.1,  # Stronger regularization
            class_weight='balanced'
        )
    else:  # Larger dataset
        model = RandomForestClassifier(
            n_estimators=min(epochs, 50),  # Fewer trees for small data
            max_depth=3,  # Limit depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
    
    logger.info("Fitting model on training data...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])
    else:
        feature_importance = np.zeros(X.shape[1])
    
    # Calculate confidence intervals for small datasets
    confidence_info = {}
    if len(X_test) < 10:
        # For very small test sets, calculate confidence intervals
        from scipy import stats
        n_test = len(X_test)
        if n_test > 0:
            # Wilson score interval for binomial proportion
            z = 1.96  # 95% confidence
            p = test_accuracy
            lower = (p + z*z/(2*n_test) - z*np.sqrt((p*(1-p) + z*z/(4*n_test))/n_test)) / (1 + z*z/n_test)
            upper = (p + z*z/(2*n_test) + z*np.sqrt((p*(1-p) + z*z/(4*n_test))/n_test)) / (1 + z*z/n_test)
            confidence_info = {
                "test_accuracy_lower_95": max(0, lower),
                "test_accuracy_upper_95": min(1, upper),
                "small_test_set_warning": True
            }
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importance": feature_importance.tolist(),
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
        "training_completed": True,
        "timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        **confidence_info
    }
    
    # Add interpretation for small datasets
    if len(X) < 30:
        metrics["interpretation"] = {
            "dataset_size": "small",
            "reliability": "low" if test_accuracy < 0.6 else "moderate",
            "recommendation": "Collect more annotations (target: 50+) for better model reliability",
            "overfitting_risk": "high" if (train_accuracy - test_accuracy) > 0.3 else "moderate"
        }
    
    logger.info(f"Training completed! Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
    if confidence_info:
        logger.info(f"95% confidence interval for test accuracy: [{confidence_info['test_accuracy_lower_95']:.3f}, {confidence_info['test_accuracy_upper_95']:.3f}]")
    
    return model, metrics

def save_model_and_results(model, metrics, vectorizer_prompt, vectorizer_completion, output_dir):
    """Save the trained model and results for dashboard display."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, "reward_model.pkl")
    joblib.dump(model, model_path)
    
    # Save vectorizers
    joblib.dump(vectorizer_prompt, os.path.join(output_dir, "vectorizer_prompt.pkl"))
    joblib.dump(vectorizer_completion, os.path.join(output_dir, "vectorizer_completion.pkl"))
    
    # Save training history for dashboard
    training_history = {
        "model_type": "reward_model",
        "algorithm": "RandomForest",
        "train_accuracy": metrics["train_accuracy"],
        "test_accuracy": metrics["test_accuracy"],
        "train_size": metrics["train_size"],
        "test_size": metrics["test_size"],
        "feature_count": len(metrics["feature_importance"]),
        "timestamp": metrics["timestamp"],
        "status": "completed",
        "metrics": metrics
    }
    
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model config
    model_config = {
        "model_type": "reward_model",
        "algorithm": "RandomForestClassifier", 
        "model_path": model_path,
        "vectorizer_prompt_path": os.path.join(output_dir, "vectorizer_prompt.pkl"),
        "vectorizer_completion_path": os.path.join(output_dir, "vectorizer_completion.pkl"),
        "timestamp": metrics["timestamp"],
        "performance": {
            "train_accuracy": metrics["train_accuracy"],
            "test_accuracy": metrics["test_accuracy"]
        }
    }
    
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Model and results saved to {output_dir}")
    
    return model_path, history_path, config_path

def main():
    """Main entry point for reward model training."""
    # Parse command line arguments
    args = parse_args()
    
    # Process the annotations to create training data if not provided
    if args.data_path:
        data_path = args.data_path
        logger.info(f"Using provided training data: {data_path}")
    else:
        logger.info("Processing RLHF annotations to create training data...")
        data_path = process_annotations_for_training()
        if not data_path:
            logger.error("No RLHF annotations available for training. Please create some annotations first.")
            sys.exit(1)
    
    # Load training data
    logger.info(f"Loading training data from {data_path}")
    training_data = []
    with open(data_path, 'r') as f:
        for line in f:
            training_data.append(json.loads(line))
    
    if len(training_data) == 0:
        logger.error("No training data found in file.")
        sys.exit(1)
    
    # Visualize the training data
    stats = visualize_training_data(data_path)
    logger.info(f"Training data stats: {json.dumps(stats, indent=2)}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If dry run, just show data stats
    if args.dry_run:
        logger.info("Dry run requested. Data processing completed successfully.")
        logger.info(f"Ready to train on {len(training_data)} RLHF examples")
        logger.info("Remove --dry-run flag to start actual training")
        return
    
    # Prepare features
    X, y, feature_names, vectorizer_prompt, vectorizer_completion = prepare_training_features(training_data)
    
    # Train the model
    model, metrics = train_reward_model(X, y, args.epochs, args.output_dir)
    
    # Save model and results
    model_path, history_path, config_path = save_model_and_results(
        model, metrics, vectorizer_prompt, vectorizer_completion, args.output_dir
    )
    
    # Print results
    logger.info("=" * 50)
    logger.info("🎉 REWARD MODEL TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info(f"📊 Training Accuracy: {metrics['train_accuracy']:.1%}")
    logger.info(f"🎯 Test Accuracy: {metrics['test_accuracy']:.1%}")
    logger.info(f"📝 Trained on: {len(training_data)} RLHF annotations")
    logger.info(f"💾 Model saved: {model_path}")
    logger.info(f"📈 History saved: {history_path}")
    logger.info("🔄 Restart dashboard to see updated metrics!")

if __name__ == "__main__":
    main() 