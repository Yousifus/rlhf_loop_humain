#!/usr/bin/env python3
"""
Introspective Evaluation Module for Meta-Evaluator Model

This script performs reflection by comparing the model's predictions against
historical human votes, building an "error memory" to guide future improvements.

Input: 
- Trained model from models/vote_predictor_checkpoint/
- Historical votes from data/votes.jsonl

Output:
- Meta-reflection log in models/meta_reflection_log.jsonl
- Summary statistics in models/introspection_summary.json
"""

import sys
import os
import json
import argparse
import logging
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Import drift monitoring
from utils.vote_predictor.drift_monitor import run_drift_analysis, DriftAnalysisConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "eval_probe.log"))
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Tuple[Any, Any]:
    """
    Load the trained meta-evaluator model and tokenizer.
    
    Args:
        model_path: Path to the saved model directory
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set model to evaluation mode
        model.eval()
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def load_historical_votes(votes_path: str) -> List[Dict[str, Any]]:
    """
    Load historical votes from votes.jsonl.
    
    Args:
        votes_path: Path to the votes file
    
    Returns:
        List of vote entries
    """
    logger.info(f"Loading historical votes from {votes_path}")
    votes = []
    try:
        with open(votes_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    vote = json.loads(line.strip())
                    votes.append(vote)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON at line {line_idx}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading votes file: {e}")
        raise
    
    logger.info(f"Loaded {len(votes)} historical votes")
    return votes

def generate_entry_id(data: Dict[str, Any]) -> str:
    """
    Generate a deterministic ID for a reflection entry.
    
    Args:
        data: Dictionary containing prompt and completion information
        
    Returns:
        A unique ID string
    """
    # Try to use UUID4 if requested
    if os.environ.get("USE_UUID4_FOR_REFLECTION", "").lower() in ("1", "true", "yes"):
        return uuid.uuid4().hex
    
    # Otherwise use a deterministic hash
    content = (
        f"{data.get('prompt_text', '')}"
        f"{data.get('completion_A_text', '')}"
        f"{data.get('completion_B_text', '')}"
        f"{datetime.utcnow().isoformat()}"
    )
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

def calculate_token_overlap(text_a: str, text_b: str, tokenizer: Any) -> float:
    """
    Calculate Jaccard similarity between two texts based on token overlap.
    
    Args:
        text_a: First text
        text_b: Second text
        tokenizer: Tokenizer to convert text to tokens
        
    Returns:
        Jaccard similarity score (0-1)
    """
    # Tokenize texts
    tokens_a = set(tokenizer.tokenize(text_a))
    tokens_b = set(tokenizer.tokenize(text_b))
    
    # Calculate Jaccard similarity: intersection / union
    if not tokens_a and not tokens_b:
        return 1.0  # Both empty means identical
    
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    
    return len(intersection) / len(union)

def calculate_completion_features(
    completion_a: str, 
    completion_b: str, 
    tokenizer: Any
) -> Dict[str, Any]:
    """
    Calculate features comparing two completions.
    
    Args:
        completion_a: First completion text
        completion_b: Second completion text
        tokenizer: Model tokenizer
        
    Returns:
        Dictionary of completion pair features
    """
    # Tokenize completions
    tokens_a = tokenizer.tokenize(completion_a)
    tokens_b = tokenizer.tokenize(completion_b)
    
    # Calculate features
    token_overlap = calculate_token_overlap(completion_a, completion_b, tokenizer)
    length_a = len(tokens_a)
    length_b = len(tokens_b)
    length_diff = length_a - length_b
    
    return {
        "semantic_similarity_A_B": None,  # Would require embeddings model
        "token_overlap_A_B": round(token_overlap, 4),
        "length_diff_A_B": length_diff,
        "completion_A_length": length_a,
        "completion_B_length": length_b
    }

def predict_preference(
    model: Any, 
    tokenizer: Any, 
    device: str,
    prompt: str, 
    completion_a: str, 
    completion_b: str, 
    max_length: int = 512
) -> Dict[str, Any]:
    """
    Predict preference between two completions for a given prompt.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: Device to run inference on
        prompt: The input prompt
        completion_a: First completion
        completion_b: Second completion
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with prediction results (label, confidence, logits)
    """
    # Format input the same way as during training
    text_input = f"{prompt} [Option A]: {completion_a} [Option B]: {completion_b}"
    
    # Tokenize
    inputs = tokenizer(
        text_input,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Extract prediction
    prediction = int(np.argmax(probabilities))
    predicted_label_is_a = prediction == 0  # 0 means completion A is preferred
    confidence = float(probabilities[prediction])
    
    # Calculate raw confidence (0-1 scale regardless of binary choice)
    # This is how far from 0.5 the probability is, scaled to 0-1
    raw_confidence = abs(probabilities[0] - 0.5) * 2
    
    return {
        "predicted_label": prediction,
        "predicted_winner_is_A": predicted_label_is_a,
        "probabilities": probabilities.tolist(),
        "predicted_A_probability": float(probabilities[0]),
        "confidence": confidence,
        "raw_confidence": float(raw_confidence),
        "logits": logits.cpu()[0].tolist()
    }

def categorize_error(is_correct: bool, confidence: float) -> Optional[str]:
    """
    Categorize the prediction error type based on correctness and confidence.
    
    Args:
        is_correct: Whether the prediction was correct
        confidence: Model's raw confidence in the prediction
        
    Returns:
        Error type string or None if prediction was correct
    """
    if is_correct:
        return None
    
    if confidence >= 0.8:
        return "high_confidence_error"
    elif confidence >= 0.6:
        return "medium_confidence_error"
    else:
        return "low_confidence_error"

def evaluate_vote(
    model: Any, 
    tokenizer: Any,
    device: str,
    vote: Dict[str, Any],
    model_checkpoint_path: str
) -> Dict[str, Any]:
    """
    Evaluate a single vote using the current model and generate reflection.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        vote: Vote entry to evaluate
        model_checkpoint_path: Path to the model checkpoint
        
    Returns:
        Dictionary with reflection data
    """
    try:
        # Extract data from vote
        vote_id = vote.get("id") or generate_entry_id(vote)
        prompt = vote.get("prompt")
        completion_a = vote.get("completion_a")
        completion_b = vote.get("completion_b")
        human_choice = vote.get("choice")
        vote_generation_metadata = vote.get("generation_metadata", {})
        
        if not prompt or not completion_a or not completion_b or not human_choice:
            logger.warning(f"Vote {vote_id} is missing required fields. Skipping.")
            return None
        
        # Predict preference using the model
        prediction = predict_preference(
            model, tokenizer, device, prompt, completion_a, completion_b
        )
        
        # Extract prediction results
        raw_logits = prediction["logits"]
        raw_confidence = prediction["confidence"]
        predicted_winner = prediction["preferred_completion"]
        
        # Determine if prediction was correct
        human_choice_is_a = human_choice.lower() == "a"
        predicted_winner_is_a = predicted_winner.lower() == "completion_a"
        
        is_correct = (human_choice_is_a == predicted_winner_is_a)
        
        # Calculate features for the completion pair
        completion_features = calculate_completion_features(completion_a, completion_b, tokenizer)
        
        # Categorize error if prediction was wrong
        error_type = categorize_error(is_correct, raw_confidence)
        
        # Create reflection entry
        reflection = {
            "id": vote_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_checkpoint_path": model_checkpoint_path,
            
            "prompt": prompt,
            "completion_a": completion_a,
            "completion_b": completion_b,
            
            "human_choice": "completion_a" if human_choice_is_a else "completion_b",
            "model_prediction": predicted_winner,
            "model_prediction_confidence_raw": raw_confidence,
            
            "predicted_winner_is_A": predicted_winner_is_a,
            "model_prediction_confidence_raw": raw_confidence,
            
            "is_prediction_correct": is_correct,
            "prediction_error_type": error_type,
            "hardness_index": None,
            "drift_cluster_id": None,
            
            "completion_pair_features": completion_features,
            
            "original_completions_set_generation_metadata": {
                "model": vote_generation_metadata.get("model", "unknown"),
                "temperature": vote_generation_metadata.get("temperature", 0.7),
                "top_p": vote_generation_metadata.get("top_p", 0.9),
                "tokens": {
                    "prompt": vote_generation_metadata.get("prompt_tokens", 0),
                    "completion": vote_generation_metadata.get("completion_tokens", 0),
                    "total": vote_generation_metadata.get("total_tokens", 0)
                },
                "estimated_cost_usd": vote_generation_metadata.get("estimated_cost", 0.0)
            }
        }
        
        return reflection
    except Exception as e:
        logger.error(f"Error evaluating vote: {e}")
        return None

def create_confidence_histogram(reflections: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Create data for confidence histograms.
    
    Args:
        reflections: List of reflection entries
        
    Returns:
        Dictionary with confidence values for correct and incorrect predictions
    """
    correct_confidences = []
    incorrect_confidences = []
    
    for reflection in reflections:
        confidence = reflection["model_prediction_confidence_raw"]
        if reflection["is_prediction_correct"]:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    
    return {
        "correct_confidences": correct_confidences,
        "incorrect_confidences": incorrect_confidences
    }

def generate_summary_report(reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary report from the reflection entries.
    
    Args:
        reflections: List of reflection entries
        
    Returns:
        Dictionary with summary statistics
    """
    # Initialize counters
    total = len(reflections)
    correct_count = 0
    error_types = Counter()
    confidence_bins = defaultdict(int)
    confidence_bins_accuracy = defaultdict(lambda: [0, 0])  # [correct, total]
    
    # Process each reflection
    for reflection in reflections:
        # Count correct predictions
        if reflection["is_prediction_correct"]:
            correct_count += 1
        
        # Count error types
        if reflection["prediction_error_type"]:
            error_types[reflection["prediction_error_type"]] += 1
        
        # Build confidence histogram data
        confidence = reflection["model_prediction_confidence_raw"]
        bin_idx = min(int(confidence * 10), 9)  # 0.0-0.1 -> 0, 0.9-1.0 -> 9
        confidence_bins[bin_idx] += 1
        
        # Track accuracy per confidence bin
        confidence_bins_accuracy[bin_idx][1] += 1  # Increment total
        if reflection["is_prediction_correct"]:
            confidence_bins_accuracy[bin_idx][0] += 1  # Increment correct
    
    # Calculate accuracy
    accuracy = correct_count / total if total > 0 else 0
    
    # Calculate accuracy per confidence bin
    confidence_bin_accuracy = {}
    for bin_idx, counts in confidence_bins_accuracy.items():
        correct, total = counts
        bin_accuracy = correct / total if total > 0 else 0
        confidence_bin_accuracy[f"{bin_idx/10:.1f}-{(bin_idx+1)/10:.1f}"] = round(bin_accuracy, 4)
    
    # Create histogram data
    histogram_data = create_confidence_histogram(reflections)
    
    # Create summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_entries": total,
        "accuracy": round(accuracy, 4),
        "error_counts": dict(error_types),
        "high_confidence_error_count": error_types["high_confidence_error"],
        "medium_confidence_error_count": error_types["medium_confidence_error"],
        "low_confidence_error_count": error_types["low_confidence_error"],
        "confidence_distribution": dict(confidence_bins),
        "confidence_bin_accuracy": confidence_bin_accuracy,
        "mean_confidence_correct": round(np.mean(histogram_data["correct_confidences"]) if histogram_data["correct_confidences"] else 0, 4),
        "mean_confidence_incorrect": round(np.mean(histogram_data["incorrect_confidences"]) if histogram_data["incorrect_confidences"] else 0, 4)
    }
    
    return summary

def plot_confidence_histogram(reflections: List[Dict[str, Any]], output_path: str) -> None:
    """
    Create a confidence histogram plot and save it.
    
    Args:
        reflections: List of reflection entries
        output_path: Path to save the plot
    """
    histogram_data = create_confidence_histogram(reflections)
    
    plt.figure(figsize=(10, 6))
    
    # Create histograms
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    plt.hist(histogram_data["correct_confidences"], bins=bins, alpha=0.7, label="Correct Predictions")
    plt.hist(histogram_data["incorrect_confidences"], bins=bins, alpha=0.7, label="Incorrect Predictions")
    
    # Add details
    plt.xlabel("Model Confidence")
    plt.ylabel("Count")
    plt.title("Distribution of Model Confidence by Prediction Correctness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Confidence histogram saved to {output_path}")

def run_introspection(
    model_path: str, 
    votes_path: str, 
    output_path: str,
    summary_path: str,
    plot_path: str,
    run_drift_monitoring: bool = False
) -> Dict[str, Any]:
    """
    Run the introspective evaluation on all historical votes.
    
    Args:
        model_path: Path to the saved model
        votes_path: Path to the votes file
        output_path: Path to save the reflection log
        summary_path: Path to save the summary report
        plot_path: Path to save the confidence histogram
        run_drift_monitoring: Whether to run drift monitoring analysis
    
    Returns:
        Summary statistics
    """
    # Load model and votes
    model, tokenizer, device = load_model(model_path)
    votes = load_historical_votes(votes_path)
    
    # Process each vote
    reflections = []
    
    for vote_idx, vote in enumerate(votes):
        logger.info(f"Evaluating vote {vote_idx+1}/{len(votes)}")
        
        try:
            reflection = evaluate_vote(model, tokenizer, device, vote, model_path)
            if reflection:
                reflections.append(reflection)
        except Exception as e:
            logger.error(f"Error processing vote {vote_idx}: {e}")
    
    # Generate summary report
    summary = generate_summary_report(reflections)
    
    # Create confidence histogram
    plot_confidence_histogram(reflections, plot_path)
    
    # Save reflections to jsonl file
    logger.info(f"Saving {len(reflections)} reflections to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for reflection in reflections:
            json.dump(reflection, f)
            f.write('\n')
    
    # Save summary report
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved to {summary_path}")
    
    # Run drift monitoring if requested
    if run_drift_monitoring:
        logger.info("Running drift monitoring analysis")
        drift_output_dir = os.path.join(os.path.dirname(output_path), "drift_analysis")
        
        # Configure drift analysis
        drift_config = DriftAnalysisConfig(
            time_window_days=7,
            n_clusters=min(5, len(reflections) // 20) if len(reflections) > 20 else 2,
            min_examples_per_window=max(5, len(reflections) // 10)
        )
        
        # Run drift analysis
        drift_analysis = run_drift_analysis(
            reflection_path=output_path,
            output_dir=drift_output_dir,
            config=drift_config
        )
        
        # Add drift analysis summary to the summary report
        summary["drift_analysis"] = drift_analysis.get("summary", {})
        
        # Update summary report with drift information
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Drift analysis saved to {drift_output_dir}")
    
    return summary

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Evaluate vote predictor model introspectively")
    parser.add_argument("--model-path", type=str, 
                        default=os.path.join(project_root, "models", "vote_predictor_checkpoint"),
                        help="Path to the saved model")
    parser.add_argument("--votes-path", type=str, 
                        default=os.path.join(project_root, "data", "votes.jsonl"),
                        help="Path to the votes file")
    parser.add_argument("--output-path", type=str, 
                        default=os.path.join(project_root, "models", "meta_reflection_log.jsonl"),
                        help="Path to save the reflection log")
    parser.add_argument("--summary-path", type=str, 
                        default=os.path.join(project_root, "models", "meta_reflection_summary.json"),
                        help="Path to save the summary report")
    parser.add_argument("--plot-path", type=str, 
                        default=os.path.join(project_root, "models", "confidence_histogram.png"),
                        help="Path to save the confidence histogram")
    parser.add_argument("--run-drift-monitoring", action="store_true",
                        help="Run drift monitoring analysis")
    
    args = parser.parse_args()
    
    logger.info("Starting introspective evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Votes path: {args.votes_path}")
    
    summary = run_introspection(
        model_path=args.model_path, 
        votes_path=args.votes_path, 
        output_path=args.output_path,
        summary_path=args.summary_path,
        plot_path=args.plot_path,
        run_drift_monitoring=args.run_drift_monitoring
    )
    
    print("\nEvaluation Summary:")
    print(f"Total entries: {summary['total_entries']}")
    print(f"Accuracy: {summary['accuracy']:.2f}")
    print(f"High confidence errors: {summary['high_confidence_error_count']}")
    print(f"Medium confidence errors: {summary['medium_confidence_error_count']}")
    print(f"Low confidence errors: {summary['low_confidence_error_count']}")
    
    # Print drift analysis summary if available
    if "drift_analysis" in summary:
        drift_summary = summary["drift_analysis"]
        print("\nDrift Analysis Summary:")
        print(f"Time-based drift detected: {drift_summary.get('time_drift_detected', False)}")
        print(f"Number of clusters: {drift_summary.get('num_clusters', 0)}")
        print(f"Potential drift clusters: {drift_summary.get('potential_drift_clusters', [])}")
        print(f"Calibration drift detected: {drift_summary.get('calibration_drift_detected', False)}")
        print(f"Calibration trend: {drift_summary.get('calibration_trend', 'stable')}")
    
    print(f"\nFull details saved to {args.summary_path}")
    print(f"Confidence histogram saved to {args.plot_path}")

if __name__ == "__main__":
    main() 