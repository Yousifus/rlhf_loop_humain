#!/usr/bin/env python3
"""
RLHF Model Confidence Calibration Module

This module calibrates the confidence scores of prediction models to improve their reliability.
It analyzes prediction accuracy versus confidence to adjust confidence scores, making them
more accurately reflect the true likelihood of correct predictions.

Input: 
- Historical prediction data from models/meta_reflection_log.jsonl
- Raw model confidence scores

Output:
- Calibrated confidence scoring system
- Calibration parameters saved to models/calibration_log.json
- Reliability visualizations and calibration metrics
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "calibration.log"))
    ]
)
logger = logging.getLogger(__name__)

def load_reflection_data(reflection_path: str) -> List[Dict[str, Any]]:
    """
    Load reflection data from the meta_reflection_log.jsonl file.
    
    Args:
        reflection_path: Path to the reflection log file
        
    Returns:
        List of reflection entries
    """
    logger.info(f"Loading reflection data from {reflection_path}")
    reflections = []
    
    try:
        with open(reflection_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    reflection = json.loads(line.strip())
                    reflections.append(reflection)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON at line {line_idx}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading reflection file: {e}")
        raise
    
    logger.info(f"Loaded {len(reflections)} reflection entries")
    return reflections

def prepare_calibration_data(reflections: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract probabilities and ground truth labels from reflection data.
    
    Args:
        reflections: List of reflection entries
        
    Returns:
        Tuple of (probabilities, labels)
    """
    # Extract probabilities and binary outcomes
    probs = []
    labels = []
    
    for reflection in reflections:
        try:
            # Get the probability of the model's prediction
            if "model_probabilities" in reflection:
                model_probabilities = reflection["model_probabilities"]
                model_prediction = reflection["model_prediction"]
                model_confidence = model_probabilities[model_prediction]
                
                # Get the binary outcome (1 if correct, 0 if incorrect)
                is_correct = reflection["is_correct"]
                
                probs.append(model_confidence)
                labels.append(1 if is_correct else 0)
            elif "predicted_A_is_chosen_probability" in reflection:
                # Handle format from eval_probe.py
                pred_prob = reflection["predicted_A_is_chosen_probability"]
                pred_winner_is_a = reflection["predicted_winner_is_A"]
                is_a_chosen = reflection["is_A_chosen_by_human"]
                is_correct = (pred_winner_is_a == is_a_chosen)
                
                # Use the confidence in the prediction
                confidence = pred_prob if pred_winner_is_a else (1 - pred_prob)
                
                probs.append(confidence)
                labels.append(1 if is_correct else 0)
            else:
                logger.warning("Unrecognized reflection format. Skipping entry.")
                continue
        except (KeyError, IndexError) as e:
            logger.warning(f"Error processing reflection entry: {e}. Skipping.")
            continue
    
    if not probs:
        raise ValueError("No valid data extracted from reflection entries.")
    
    return np.array(probs), np.array(labels)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities
        labels: Ground truth labels (1 for correct, 0 for incorrect)
        n_bins: Number of bins for binning predictions
        
    Returns:
        Expected Calibration Error (lower is better)
    """
    # Handle edge case with empty or very small input
    if len(probs) < 2:
        logger.warning("Too few samples for reliable ECE calculation. Returning 0.")
        return 0.0
    
    # Adjust number of bins for small datasets
    actual_n_bins = min(n_bins, len(probs) // 2 + 1)
    if actual_n_bins < n_bins:
        logger.warning(f"Reducing bin count from {n_bins} to {actual_n_bins} due to small dataset size.")
    
    # Create bins for probabilities
    bin_indices = np.minimum(np.floor(probs * actual_n_bins).astype(int), actual_n_bins - 1)
    
    # Initialize arrays to store bin statistics
    bin_sums = np.zeros(actual_n_bins)  # Sum of probabilities in each bin
    bin_correct = np.zeros(actual_n_bins)  # Sum of correct predictions in each bin
    bin_counts = np.zeros(actual_n_bins)  # Count of samples in each bin
    
    # Accumulate bin statistics
    for i, (prob, label) in enumerate(zip(probs, labels)):
        bin_idx = bin_indices[i]
        bin_sums[bin_idx] += prob
        bin_correct[bin_idx] += label
        bin_counts[bin_idx] += 1
    
    # Calculate ECE
    ece = 0.0
    for i in range(actual_n_bins):
        if bin_counts[i] > 0:
            bin_avg_prob = bin_sums[i] / bin_counts[i]
            bin_accuracy = bin_correct[i] / bin_counts[i]
            bin_weight = bin_counts[i] / len(probs)
            ece += bin_weight * abs(bin_avg_prob - bin_accuracy)
    
    return ece

def temperature_scaling_objective(T: float, probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Objective function for temperature scaling optimization.
    
    Args:
        T: Temperature parameter
        probs: Uncalibrated probabilities
        labels: Ground truth labels
        
    Returns:
        Negative log likelihood loss
    """
    # Apply temperature scaling
    calibrated_probs = 1.0 / (1.0 + np.exp(-(np.log(probs / (1 - probs)) / T)))
    
    # Clip probabilities to avoid numerical issues
    calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
    
    # Calculate log loss
    loss = safe_log_loss(labels, calibrated_probs)
    return loss

def platt_scaling_objective(params: List[float], logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Objective function for Platt scaling optimization.
    
    Args:
        params: [a, b] parameters for Platt scaling
        logits: Uncalibrated model logits
        labels: Ground truth labels
        
    Returns:
        Negative log likelihood loss
    """
    a, b = params
    calibrated_logits = a * logits + b
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    
    # Clip probabilities to avoid numerical issues
    calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
    
    # Calculate log loss
    loss = safe_log_loss(labels, calibrated_probs)
    return loss

def safe_log_loss(y_true, y_pred):
    """
    Wrapper for log_loss that handles edge cases like single-class data.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        Log loss value, or 0 if calculation fails
    """
    try:
        # For single-class datasets, we need to provide the labels argument
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 1:
            # If all examples are correct (1) or all are wrong (0)
            only_label = unique_labels[0]
            # For a single class, log loss is -log(p) if the class is 1, or -log(1-p) if the class is 0
            if only_label == 1:
                return -np.mean(np.log(np.clip(y_pred, 1e-7, 1 - 1e-7)))
            else:
                return -np.mean(np.log(np.clip(1 - y_pred, 1e-7, 1 - 1e-7)))
        else:
            return log_loss(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Log loss calculation failed: {e}. Returning 0.")
        return 0.0

def safe_brier_score(y_true, y_pred):
    """
    Wrapper for brier_score_loss that handles edge cases.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        Brier score, or 0 if calculation fails
    """
    try:
        return brier_score_loss(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Brier score calculation failed: {e}. Returning 0.")
        # Calculate manually for small datasets
        return np.mean((np.array(y_pred) - np.array(y_true)) ** 2)

def train_temperature_scaling(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Train a temperature scaling calibration model.
    
    Args:
        probs: Uncalibrated probabilities
        labels: Ground truth labels
        
    Returns:
        Dictionary with calibration parameters and metrics
    """
    logger.info("Training temperature scaling calibration model")
    
    # Convert probabilities to range (0, 1) to avoid numerical issues
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # Calculate pre-calibration metrics
    pre_nll = safe_log_loss(labels, probs)
    pre_brier = safe_brier_score(labels, probs)
    pre_ece = expected_calibration_error(probs, labels)
    
    # Optimize for temperature parameter
    result = minimize(
        lambda T: temperature_scaling_objective(T, probs, labels),
        x0=np.array([1.0]),  # Start with T=1 (no calibration)
        method='BFGS',
        options={'disp': False}
    )
    
    # Extract optimal temperature
    T_opt = result.x[0]
    
    # Apply calibration to get calibrated probabilities
    calibrated_probs = 1.0 / (1.0 + np.exp(-(np.log(probs / (1 - probs)) / T_opt)))
    calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
    
    # Calculate post-calibration metrics
    post_nll = safe_log_loss(labels, calibrated_probs)
    post_brier = safe_brier_score(labels, calibrated_probs)
    post_ece = expected_calibration_error(calibrated_probs, labels)
    
    # Create result dictionary
    calibration_result = {
        "method": "temperature_scaling",
        "parameters": {
            "temperature": float(T_opt)
        },
        "metrics": {
            "pre_calibration": {
                "log_loss": pre_nll,
                "brier_score": pre_brier,
                "ece": pre_ece
            },
            "post_calibration": {
                "log_loss": post_nll,
                "brier_score": post_brier,
                "ece": post_ece
            },
            "improvement": {
                "log_loss": pre_nll - post_nll,
                "brier_score": pre_brier - post_brier,
                "ece": pre_ece - post_ece
            }
        }
    }
    
    logger.info(f"Temperature scaling complete. T={T_opt:.4f}")
    logger.info(f"Log loss: {pre_nll:.4f} -> {post_nll:.4f} ({pre_nll - post_nll:.4f} improvement)")
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} ({pre_ece - post_ece:.4f} improvement)")
    
    return calibration_result

def train_platt_scaling(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Train a Platt scaling calibration model.
    
    Args:
        probs: Uncalibrated probabilities
        labels: Ground truth labels
        
    Returns:
        Dictionary with calibration parameters and metrics
    """
    logger.info("Training Platt scaling calibration model")
    
    # Convert probabilities to logits
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1 - probs))
    
    # Calculate pre-calibration metrics
    pre_nll = safe_log_loss(labels, probs)
    pre_brier = safe_brier_score(labels, probs)
    pre_ece = expected_calibration_error(probs, labels)
    
    # Optimize for Platt scaling parameters
    result = minimize(
        lambda params: platt_scaling_objective(params, logits, labels),
        x0=np.array([1.0, 0.0]),  # Start with a=1, b=0 (no calibration)
        method='BFGS',
        options={'disp': False}
    )
    
    # Extract optimal parameters
    a_opt, b_opt = result.x
    
    # Apply calibration to get calibrated probabilities
    calibrated_logits = a_opt * logits + b_opt
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
    
    # Calculate post-calibration metrics
    post_nll = safe_log_loss(labels, calibrated_probs)
    post_brier = safe_brier_score(labels, calibrated_probs)
    post_ece = expected_calibration_error(calibrated_probs, labels)
    
    # Create result dictionary
    calibration_result = {
        "method": "platt_scaling",
        "parameters": {
            "a": float(a_opt),
            "b": float(b_opt)
        },
        "metrics": {
            "pre_calibration": {
                "log_loss": pre_nll,
                "brier_score": pre_brier,
                "ece": pre_ece
            },
            "post_calibration": {
                "log_loss": post_nll,
                "brier_score": post_brier,
                "ece": post_ece
            },
            "improvement": {
                "log_loss": pre_nll - post_nll,
                "brier_score": pre_brier - post_brier,
                "ece": pre_ece - post_ece
            }
        }
    }
    
    logger.info(f"Platt scaling complete. a={a_opt:.4f}, b={b_opt:.4f}")
    logger.info(f"Log loss: {pre_nll:.4f} -> {post_nll:.4f} ({pre_nll - post_nll:.4f} improvement)")
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} ({pre_ece - post_ece:.4f} improvement)")
    
    return calibration_result

def apply_calibration(raw_probs: Union[float, List[float], np.ndarray], 
                      calibration_params: Dict[str, Any], 
                      method_name: str) -> Union[float, List[float], np.ndarray]:
    """
    Apply calibration to raw probability outputs.
    
    Args:
        raw_probs: Uncalibrated probability or list of probabilities
        calibration_params: Dictionary containing calibration parameters
        method_name: Name of the calibration method to use
        
    Returns:
        Calibrated probability or list of probabilities
    """
    # Handle single probability vs. list/array
    single_input = False
    if isinstance(raw_probs, (float, int)):
        raw_probs = np.array([float(raw_probs)])
        single_input = True
    else:
        raw_probs = np.array(raw_probs)
    
    # Clip probabilities to avoid numerical issues
    raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
    
    # Apply the appropriate calibration method
    if method_name == "temperature_scaling":
        # Get temperature parameter
        T = calibration_params["parameters"]["temperature"]
        
        # Convert to logits, apply temperature, convert back to probabilities
        logits = np.log(raw_probs / (1 - raw_probs))
        calibrated_logits = logits / T
        calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    
    elif method_name == "platt_scaling":
        # Get Platt scaling parameters
        a = calibration_params["parameters"]["a"]
        b = calibration_params["parameters"]["b"]
        
        # Convert to logits, apply Platt scaling, convert back to probabilities
        logits = np.log(raw_probs / (1 - raw_probs))
        calibrated_logits = a * logits + b
        calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
    
    else:
        raise ValueError(f"Unknown calibration method: {method_name}")
    
    # Clip again to ensure valid probabilities
    calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
    
    # Return single value or array as appropriate
    if single_input:
        return float(calibrated_probs[0])
    else:
        return calibrated_probs.tolist() if isinstance(raw_probs, list) else calibrated_probs

def plot_reliability_diagram(probs: np.ndarray, 
                            labels: np.ndarray, 
                            calibrated_probs: np.ndarray, 
                            output_path: str,
                            n_bins: int = 10) -> None:
    """
    Plot reliability diagrams for uncalibrated and calibrated probabilities.
    
    Args:
        probs: Uncalibrated probabilities
        labels: Ground truth labels
        calibrated_probs: Calibrated probabilities
        output_path: Path to save the plot
        n_bins: Number of bins for binning predictions
    """
    # Check for minimum dataset size
    if len(probs) < 5:
        logger.warning("Dataset too small for meaningful reliability diagram. Skipping plot.")
        return
    
    # Adjust number of bins for small datasets
    actual_n_bins = min(n_bins, len(probs) // 2 + 1)
    if actual_n_bins < n_bins:
        logger.warning(f"Reducing bin count from {n_bins} to {actual_n_bins} for reliability diagram due to small dataset size.")
    
    plt.figure(figsize=(10, 8))
    
    # Function to bin predictions and calculate accuracies
    def get_calibration_points(probs, labels, n_bins):
        bin_size = 1.0 / n_bins
        bins = np.arange(0, 1 + bin_size, bin_size)
        binned = np.digitize(probs, bins) - 1
        bin_accs = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(len(probs)):
            bin_idx = min(binned[i], n_bins - 1)
            bin_accs[bin_idx] += labels[i]
            bin_confs[bin_idx] += probs[i]
            bin_counts[bin_idx] += 1
        
        bin_accs = np.divide(bin_accs, bin_counts, out=np.zeros_like(bin_accs), where=bin_counts > 0)
        bin_confs = np.divide(bin_confs, bin_counts, out=np.zeros_like(bin_confs), where=bin_counts > 0)
        
        return bin_accs, bin_confs, bin_counts
    
    try:
        # Get calibration points for uncalibrated probabilities
        uncal_accs, uncal_confs, uncal_counts = get_calibration_points(probs, labels, actual_n_bins)
        
        # Get calibration points for calibrated probabilities
        cal_accs, cal_confs, cal_counts = get_calibration_points(calibrated_probs, labels, actual_n_bins)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Plot reliability diagrams
        plt.plot(uncal_confs, uncal_accs, 'ro-', label='Uncalibrated')
        plt.plot(cal_confs, cal_accs, 'go-', label='Calibrated')
        
        # Add details to plot
        plt.xlabel('Confidence (predicted probability)')
        plt.ylabel('Accuracy (fraction of positive cases)')
        plt.title('Reliability Diagram')
        plt.grid(True)
        plt.legend()
        
        # Add ECE values to the plot
        uncal_ece = expected_calibration_error(probs, labels, actual_n_bins)
        cal_ece = expected_calibration_error(calibrated_probs, labels, actual_n_bins)
        plt.text(0.05, 0.95, f'Uncalibrated ECE: {uncal_ece:.4f}', transform=plt.gca().transAxes)
        plt.text(0.05, 0.90, f'Calibrated ECE: {cal_ece:.4f}', transform=plt.gca().transAxes)
        
        # Add dataset size warning if applicable
        if len(probs) < 20:
            plt.figtext(0.5, 0.01, f'Warning: Small dataset size ({len(probs)} samples)',
                       ha='center', color='red', fontsize=12)
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Reliability diagram saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating reliability diagram: {e}")
        plt.close()  # Ensure figure is closed on error

def calibrate_model(reflection_path: str, output_path: str, 
                   method: str = "temperature_scaling", 
                   test_split: float = 0.2,
                   random_state: int = 42) -> Dict[str, Any]:
    """
    Train a calibration model and evaluate its performance.
    
    Args:
        reflection_path: Path to the reflection log file
        output_path: Path to save the calibration parameters
        method: Calibration method to use
        test_split: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing calibration parameters and metrics
    """
    # Load reflection data
    reflections = load_reflection_data(reflection_path)
    
    if len(reflections) == 0:
        logger.error("No reflection data found.")
        return None
    
    # Extract human choice (for stratified split)
    y_human_choice = [r["human_choice"] for r in reflections]
    
    # Split data into train and test sets
    try:
        train_idx, test_idx = train_test_split(
            range(len(reflections)), 
            test_size=test_split, 
            random_state=random_state,
            stratify=y_human_choice
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
        # Fall back to random split if stratification fails
        train_idx, test_idx = train_test_split(
            range(len(reflections)), 
            test_size=test_split, 
            random_state=random_state
        )
    
    train_reflections = [reflections[i] for i in train_idx]
    test_reflections = [reflections[i] for i in test_idx]
    
    logger.info(f"Split data into {len(train_reflections)} training and {len(test_reflections)} testing examples")
    
    # Prepare calibration data
    train_probs, train_labels = prepare_calibration_data(train_reflections)
    test_probs, test_labels = prepare_calibration_data(test_reflections)
    
    # Train the calibration model
    if method == "temperature_scaling":
        calibration_result = train_temperature_scaling(train_probs, train_labels)
    elif method == "platt_scaling":
        calibration_result = train_platt_scaling(train_probs, train_labels)
    else:
        logger.error(f"Unknown calibration method: {method}")
        return None
    
    # Evaluate on test set
    test_calibrated_probs = apply_calibration(test_probs, calibration_result, method)
    
    test_pre_nll = safe_log_loss(test_labels, test_probs)
    test_pre_brier = safe_brier_score(test_labels, test_probs)
    test_pre_ece = expected_calibration_error(test_probs, test_labels)
    
    test_post_nll = safe_log_loss(test_labels, test_calibrated_probs)
    test_post_brier = safe_brier_score(test_labels, test_calibrated_probs)
    test_post_ece = expected_calibration_error(test_calibrated_probs, test_labels)
    
    # Add test metrics to result
    calibration_result["test_metrics"] = {
        "pre_calibration": {
            "log_loss": test_pre_nll,
            "brier_score": test_pre_brier,
            "ece": test_pre_ece
        },
        "post_calibration": {
            "log_loss": test_post_nll,
            "brier_score": test_post_brier,
            "ece": test_post_ece
        },
        "improvement": {
            "log_loss": test_pre_nll - test_post_nll,
            "brier_score": test_pre_brier - test_post_brier,
            "ece": test_pre_ece - test_post_ece
        }
    }
    
    # Add metadata to result
    calibration_result["metadata"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "reflection_path": reflection_path,
        "train_size": len(train_reflections),
        "test_size": len(test_reflections),
        "random_state": random_state,
        "is_global_calibration": True,  # For future segmented calibration
    }
    
    # Save calibration parameters
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(calibration_result, f, indent=2)
    
    logger.info(f"Calibration parameters saved to {output_path}")
    
    # Plot reliability diagram
    plot_path = os.path.join(os.path.dirname(output_path), "reliability_diagram.png")
    plot_reliability_diagram(test_probs, test_labels, test_calibrated_probs, plot_path)
    
    return calibration_result

def get_calibrated_preference_prediction(prompt: str, 
                                        completion_a: str, 
                                        completion_b: str,
                                        model_predict_fn: Callable,
                                        calibration_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get calibrated preference prediction for a pair of completions.
    This is a preview of what will be in predict.py
    
    Args:
        prompt: The prompt text
        completion_a: First completion
        completion_b: Second completion
        model_predict_fn: Function to get raw model predictions
        calibration_params: Calibration parameters
        
    Returns:
        Dictionary with prediction results
    """
    # Get raw model prediction
    raw_prediction = model_predict_fn(prompt, completion_a, completion_b)
    
    # Extract raw probabilities
    raw_probs = raw_prediction["probabilities"]
    
    # Apply calibration
    method = calibration_params["method"]
    calibrated_probs = apply_calibration(raw_probs, calibration_params, method)
    
    # Determine preferred completion
    predicted_label = np.argmax(calibrated_probs)
    confidence = calibrated_probs[predicted_label]
    
    # Create result
    result = {
        "preferred_completion": "A" if predicted_label == 0 else "B",
        "confidence": float(confidence),
        "raw_confidence": float(raw_probs[predicted_label]),
        "calibrated_probabilities": [float(p) for p in calibrated_probs],
        "raw_probabilities": [float(p) for p in raw_probs],
        "calibration_method": method
    }
    
    return result

def main():
    """Main entry point for the calibration script."""
    parser = argparse.ArgumentParser(description="Calibrate meta-evaluator model confidence")
    
    parser.add_argument("--reflection-path", type=str, 
                        default=os.path.join(project_root, "models", "meta_reflection_log.jsonl"),
                        help="Path to the reflection log file")
    
    parser.add_argument("--output-path", type=str, 
                        default=os.path.join(project_root, "models", "calibration_log.json"),
                        help="Path to save the calibration parameters")
    
    parser.add_argument("--method", type=str, 
                        choices=["temperature_scaling", "platt_scaling"],
                        default="temperature_scaling",
                        help="Calibration method to use")
    
    parser.add_argument("--test-split", type=float, 
                        default=0.2,
                        help="Fraction of data to use for testing")
    
    parser.add_argument("--random-state", type=int, 
                        default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logger.info("Starting calibration")
    calibration_result = calibrate_model(
        args.reflection_path, 
        args.output_path, 
        args.method,
        args.test_split,
        args.random_state
    )
    
    if calibration_result:
        method = calibration_result["method"]
        params_str = str(calibration_result["parameters"])
        train_ece_improvement = calibration_result["metrics"]["improvement"]["ece"]
        test_ece_improvement = calibration_result["test_metrics"]["improvement"]["ece"]
        
        print("\n=== Calibration Complete ===")
        print(f"Method: {method}")
        print(f"Parameters: {params_str}")
        print(f"Training ECE improvement: {train_ece_improvement:.4f}")
        print(f"Test ECE improvement: {test_ece_improvement:.4f}")
        print(f"Calibration log saved to: {args.output_path}")
        print(f"Reliability diagram saved to: {os.path.dirname(args.output_path)}/reliability_diagram.png")
        print("\nNext steps:")
        print("1. Implement predict.py with calibrated confidence")
        print("2. Integrate calibration into the full RLHF loop")

if __name__ == "__main__":
    main()
