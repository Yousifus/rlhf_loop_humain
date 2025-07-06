#!/usr/bin/env python3
"""
Meta-Evaluator Memory-Attuned Retraining Data Preparation

This module prepares training data for fine-tuning the vote predictor model based on
historical performance data recorded in the meta_reflection_log.jsonl file.

Input: meta_reflection_log.jsonl (log of model predictions and human preferences)
Output: 
  - vote_predictor_retrain_data.jsonl (weighted training examples for fine-tuning)
  - retrain_metrics.json (summary of data preparation and weighting decisions)

Key features:
1. Converts historical human feedback and model predictions into training examples
2. Weights examples based on error patterns and confidence gap
3. Emphasizes cases where the model was wrong or under-confident
4. Preserves original example metadata for traceability
"""

import os
import sys
import json
import logging
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Import needed schema classes
from utils.vote_predictor.data_prep import TokenUsage, ModelGenerationMetadata, VotePredictionExample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "retrain_data_prep.log"))
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RetrainExample(VotePredictionExample):
    """
    Extends VotePredictionExample with retraining-specific fields.
    """
    # Original fields from VotePredictionExample are inherited
    # Additional fields for retraining:
    weight: float = 1.0  # Sample weight for training (higher values emphasize this example)
    error_type: Optional[str] = None  # Type of error if model prediction was wrong
    confidence_gap: Optional[float] = None  # Gap between human and model confidence
    timestamp: Optional[str] = None  # When this example was processed
    original_prediction: Optional[Dict[str, Any]] = None  # Original model prediction details

def load_reflection_log(file_path: str) -> List[Dict[str, Any]]:
    """
    Load meta reflection data from the log file.
    
    Args:
        file_path: Path to the meta_reflection_log.jsonl file
        
    Returns:
        List of reflection entries as dictionaries
    """
    reflections = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    reflections.append(entry)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON at line {line_num}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading reflection log: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(reflections)} reflection entries from {file_path}")
    return reflections

def calculate_sample_weight(reflection: Dict[str, Any], weight_config: Dict[str, float]) -> float:
    """
    Calculate the weight for a training example based on reflection data.
    
    Args:
        reflection: Single reflection entry
        weight_config: Configuration for weight calculation
        
    Returns:
        Weight value for the training example
    """
    base_weight = weight_config.get("base_weight", 1.0)
    
    # Add weight if model prediction was wrong
    is_correct = reflection.get("is_correct", True)
    if not is_correct:
        base_weight += weight_config.get("incorrect_prediction_bonus", 2.0)
    
    # Add weight based on confidence gap (more weight when model was under-confident)
    confidence_gap = reflection.get("confidence_gap", 0.0)
    if confidence_gap > 0:  # Human was more confident than model
        confidence_factor = min(confidence_gap * weight_config.get("confidence_gap_factor", 1.0), 
                               weight_config.get("max_confidence_bonus", 2.0))
        base_weight += confidence_factor
    
    # Add weight based on error type
    error_type = reflection.get("error_type")
    if error_type:
        error_type_bonus = weight_config.get("error_type_bonuses", {}).get(error_type, 0.0)
        base_weight += error_type_bonus
    
    # Ensure weight is within valid range
    return max(min(base_weight, weight_config.get("max_weight", 5.0)), 
              weight_config.get("min_weight", 1.0))

def create_retrain_example(reflection: Dict[str, Any], weight_config: Dict[str, float]) -> List[RetrainExample]:
    """
    Create training examples from a single reflection entry.
    
    Args:
        reflection: Single reflection entry
        weight_config: Configuration for weight calculation
        
    Returns:
        List of RetrainExample objects (usually 1 or 2 examples)
    """
    examples = []
    
    # Extract basic information
    prompt = reflection.get("prompt")
    if not prompt:
        logger.warning("Reflection entry missing prompt field. Skipping.")
        return examples
    
    # Try to extract completions from the original vote metadata
    original_metadata = reflection.get("original_vote_metadata", {})
    completions = original_metadata.get("completions", [])
    
    # If completions aren't available, we can't create training examples
    if not completions:
        logger.warning("No completions found in reflection entry. Skipping.")
        return examples
    
    # Extract human choice and model prediction
    human_choice = reflection.get("human_choice")
    model_prediction = reflection.get("model_prediction")
    
    if human_choice is None:
        logger.warning("Reflection entry missing human_choice field. Skipping.")
        return examples
    
    # Extract metadata for the example
    annotation = original_metadata.get("annotation")
    gen_metadata_raw = original_metadata.get("generation_metadata")
    
    # Create metadata object if present
    metadata = None
    if gen_metadata_raw:
        try:
            tokens = None
            if 'tokens' in gen_metadata_raw:
                tokens = TokenUsage(
                    prompt=gen_metadata_raw['tokens'].get('prompt'),
                    completion=gen_metadata_raw['tokens'].get('completion'),
                    total=gen_metadata_raw['tokens'].get('total')
                )
            
            metadata = ModelGenerationMetadata(
                model=gen_metadata_raw.get('model', 'unknown'),
                temperature=gen_metadata_raw.get('temperature', 0.0),
                top_p=gen_metadata_raw.get('top_p', 1.0),
                tokens=tokens,
                estimated_cost=gen_metadata_raw.get('estimated_cost')
            )
        except Exception as e:
            logger.warning(f"Error parsing metadata: {e}")
    
    # Determine sample weight
    weight = calculate_sample_weight(reflection, weight_config)
    
    # Get data for additional retraining fields
    error_type = reflection.get("error_type")
    confidence_gap = reflection.get("confidence_gap")
    timestamp = reflection.get("timestamp")
    
    # Prepare original prediction details
    original_prediction = {
        "model_prediction": model_prediction,
        "model_confidence": reflection.get("model_confidence"),
        "model_probabilities": reflection.get("model_probabilities"),
        "model_logits": reflection.get("model_logits"),
        "is_correct": reflection.get("is_correct"),
    }
    
    # Create the example based on human preference
    if len(completions) > human_choice:
        chosen_completion = completions[human_choice]
        
        # Get indices of non-chosen completions
        non_chosen_indices = [i for i in range(len(completions)) if i != human_choice]
        
        if non_chosen_indices:
            # Randomly select a non-chosen completion
            random_non_chosen_idx = random.choice(non_chosen_indices)
            non_chosen_completion = completions[random_non_chosen_idx]
            
            # Create example 1: chosen as A, non-chosen as B, label=0
            example1 = RetrainExample(
                prompt=prompt,
                completion_a=chosen_completion,
                completion_b=non_chosen_completion,
                label=0,  # A is preferred
                confidence=reflection.get("human_confidence"),
                annotation=annotation,
                metadata=metadata,
                weight=weight,
                error_type=error_type,
                confidence_gap=confidence_gap,
                timestamp=timestamp,
                original_prediction=original_prediction
            )
            
            # Create example 2: non-chosen as A, chosen as B, label=1
            example2 = RetrainExample(
                prompt=prompt,
                completion_a=non_chosen_completion,
                completion_b=chosen_completion,
                label=1,  # B is preferred
                confidence=reflection.get("human_confidence"),
                annotation=annotation,
                metadata=metadata,
                weight=weight,
                error_type=error_type,
                confidence_gap=confidence_gap,
                timestamp=timestamp,
                original_prediction=original_prediction
            )
            
            examples.append(example1)
            examples.append(example2)
    
    return examples

def prepare_retrain_data(reflections: List[Dict[str, Any]], weight_config: Dict[str, float]) -> List[RetrainExample]:
    """
    Prepare training data from all reflection entries.
    
    Args:
        reflections: List of reflection entries
        weight_config: Configuration for weight calculation
        
    Returns:
        List of RetrainExample objects
    """
    all_examples = []
    
    for reflection in reflections:
        examples = create_retrain_example(reflection, weight_config)
        all_examples.extend(examples)
    
    return all_examples

def save_retrain_data(examples: List[RetrainExample], output_path: str) -> None:
    """
    Save retraining examples to a JSONL file.
    
    Args:
        examples: List of retraining examples
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(asdict(example)) + '\n')
        logger.info(f"Successfully saved {len(examples)} retraining examples to {output_path}")
    except Exception as e:
        logger.error(f"Error saving retraining data: {e}")
        sys.exit(1)

def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save retraining metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Successfully saved retraining metrics to {output_path}")
    except Exception as e:
        logger.error(f"Error saving retraining metrics: {e}")

def calculate_metrics(reflections: List[Dict[str, Any]], examples: List[RetrainExample]) -> Dict[str, Any]:
    """
    Calculate metrics for the retraining data preparation.
    
    Args:
        reflections: List of reflection entries
        examples: List of retraining examples
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_reflections": len(reflections),
        "total_examples": len(examples),
        "average_weight": sum(e.weight for e in examples) / len(examples) if examples else 0,
        "weight_distribution": {},
        "error_type_counts": {},
    }
    
    # Calculate weight distribution
    weight_counts = {}
    for example in examples:
        weight = round(example.weight, 2)
        weight_counts[weight] = weight_counts.get(weight, 0) + 1
    
    metrics["weight_distribution"] = {str(k): v for k, v in sorted(weight_counts.items())}
    
    # Count error types
    error_counts = {}
    for reflection in reflections:
        error_type = reflection.get("error_type")
        if error_type:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    metrics["error_type_counts"] = error_counts
    
    return metrics

def main():
    """Main entry point for the retraining data preparation script."""
    parser = argparse.ArgumentParser(description="Prepare retraining data from meta reflection log")
    parser.add_argument("--log-file", type=str, help="Path to meta_reflection_log.jsonl")
    parser.add_argument("--output-file", type=str, help="Path to save retraining data")
    parser.add_argument("--metrics-file", type=str, help="Path to save metrics")
    parser.add_argument("--base-weight", type=float, default=1.0, help="Base weight for all examples")
    parser.add_argument("--incorrect-bonus", type=float, default=2.0, help="Bonus weight for incorrect predictions")
    parser.add_argument("--confidence-factor", type=float, default=1.0, help="Multiplier for confidence gap")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Maximum weight for any example")
    
    args = parser.parse_args()
    
    # Define file paths (use command line args if provided, otherwise use defaults)
    base_dir = Path(project_root)
    log_file = args.log_file if args.log_file else base_dir / "models" / "meta_reflection_log.jsonl"
    output_file = args.output_file if args.output_file else base_dir / "data" / "vote_predictor_retrain_data.jsonl"
    metrics_file = args.metrics_file if args.metrics_file else base_dir / "models" / "retrain_metrics.json"
    
    # Create weight configuration
    weight_config = {
        "base_weight": args.base_weight,
        "incorrect_prediction_bonus": args.incorrect_bonus,
        "confidence_gap_factor": args.confidence_factor,
        "max_confidence_bonus": 2.0,
        "min_weight": 1.0,
        "max_weight": args.max_weight,
        "error_type_bonuses": {
            "low_confidence_error": 1.5,
            "high_confidence_error": 3.0,
            "calibration_error": 2.0
        }
    }
    
    # Create the output directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True)
    
    # Load reflection log
    reflections = load_reflection_log(str(log_file))
    
    if not reflections:
        logger.error("No reflection entries found.")
        sys.exit(1)
    
    # Prepare retraining data
    logger.info("Preparing retraining data")
    examples = prepare_retrain_data(reflections, weight_config)
    
    if not examples:
        logger.error("No retraining examples could be created.")
        sys.exit(1)
    
    # Calculate metrics
    logger.info("Calculating metrics")
    metrics = calculate_metrics(reflections, examples)
    metrics["weight_config"] = weight_config
    
    # Save retraining data
    logger.info(f"Saving retraining data to {output_file}")
    save_retrain_data(examples, str(output_file))
    
    # Save metrics
    logger.info(f"Saving metrics to {metrics_file}")
    save_metrics(metrics, str(metrics_file))
    
    # Print summary
    print(f"\nRetraining Data Preparation Complete:")
    print(f"- Total reflection entries processed: {len(reflections)}")
    print(f"- Total retraining examples created: {len(examples)}")
    print(f"- Average example weight: {metrics['average_weight']:.2f}")
    print(f"- Output saved to: {output_file}")
    print(f"- Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main() 