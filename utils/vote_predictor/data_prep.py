#!/usr/bin/env python3
"""
Data preparation module for the Vote Predictor model.

This script extracts pairwise training examples from raw vote logs in
data/votes.jsonl and prepares them for training the meta-evaluator model.
"""

import json
import random
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import schema classes from .cursor/rules/dataset_schema.py
# These are copied here for self-containment
@dataclass
class TokenUsage:
    """Stores token usage details for a generation event."""
    prompt: Optional[int] = None
    completion: Optional[int] = None
    total: Optional[int] = None

@dataclass
class ModelGenerationMetadata:
    """Optional metadata about the model and parameters used for generating completions."""
    model: str  # Assuming model ID should always be present if metadata is logged
    temperature: float # Assuming temperature should always be present
    top_p: float       # Assuming top_p should always be present
    tokens: Optional[TokenUsage] = None # Token usage itself might be missing from source
    estimated_cost: Optional[float] = None

@dataclass
class VotePredictionExample:
    """
    Represents a single training example for a vote prediction model.
    """
    prompt: str
    completion_a: str
    completion_b: str
    label: int  # 0 means A is preferred, 1 means B is preferred
    confidence: Optional[float] = None
    annotation: Optional[str] = None
    metadata: Optional[ModelGenerationMetadata] = None

    def to_dict(self):
        return asdict(self)

def load_votes(file_path: str) -> List[Dict[str, Any]]:
    """
    Load vote data from a JSONL file.
    
    Args:
        file_path: Path to the votes.jsonl file
        
    Returns:
        List of vote entries as dictionaries
    """
    votes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    vote = json.loads(line.strip())
                    votes.append(vote)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON at line {line_num}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading votes file: {e}")
        sys.exit(1)
    
    return votes

def extract_training_examples(votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract pairwise training examples from vote data.
    
    For each vote, create two samples:
    1. Chosen completion as A, random non-chosen as B, label=0
    2. Random non-chosen as A, chosen as B, label=1
    
    Args:
        votes: List of vote entries
        
    Returns:
        List of training examples as dictionaries
    """
    training_examples = []
    
    for vote_idx, vote in enumerate(votes):
        try:
            prompt = vote.get('prompt')
            completions = vote.get('completions', [])
            chosen_index = vote.get('chosen_index')
            confidence = vote.get('confidence')
            annotation = vote.get('annotation')
            gen_metadata = vote.get('generation_metadata')
            
            # Validate required fields
            if prompt is None or not completions or chosen_index is None:
                logger.warning(f"Vote at index {vote_idx} missing required fields. Skipping.")
                continue
                
            if chosen_index >= len(completions):
                logger.warning(f"Vote at index {vote_idx} has invalid chosen_index. Skipping.")
                continue
                
            # Create metadata object if present
            metadata = None
            if gen_metadata:
                try:
                    tokens = None
                    if 'tokens' in gen_metadata:
                        tokens = TokenUsage(
                            prompt=gen_metadata['tokens'].get('prompt'),
                            completion=gen_metadata['tokens'].get('completion'),
                            total=gen_metadata['tokens'].get('total')
                        )
                    
                    metadata = ModelGenerationMetadata(
                        model=gen_metadata.get('model', 'unknown'),
                        temperature=gen_metadata.get('temperature', 0.0),
                        top_p=gen_metadata.get('top_p', 1.0),
                        tokens=tokens,
                        estimated_cost=gen_metadata.get('estimated_cost')
                    )
                except Exception as e:
                    logger.warning(f"Error parsing metadata for vote {vote_idx}: {e}")
            
            # Get the chosen completion
            chosen_completion = completions[chosen_index]
            
            # Get indices of non-chosen completions
            non_chosen_indices = [i for i in range(len(completions)) if i != chosen_index]
            
            if not non_chosen_indices:
                logger.warning(f"Vote at index {vote_idx} has only one completion. Skipping.")
                continue
            
            # Randomly select a non-chosen completion
            random_non_chosen_idx = random.choice(non_chosen_indices)
            non_chosen_completion = completions[random_non_chosen_idx]
            
            # Create example 1: chosen as A, non-chosen as B, label=0
            example1 = VotePredictionExample(
                prompt=prompt,
                completion_a=chosen_completion,
                completion_b=non_chosen_completion,
                label=0,  # A is preferred
                confidence=confidence,
                annotation=annotation,
                metadata=metadata
            )
            
            # Create example 2: non-chosen as A, chosen as B, label=1
            example2 = VotePredictionExample(
                prompt=prompt,
                completion_a=non_chosen_completion,
                completion_b=chosen_completion,
                label=1,  # B is preferred
                confidence=confidence,
                annotation=annotation,
                metadata=metadata
            )
            
            training_examples.append(example1.to_dict())
            training_examples.append(example2.to_dict())
            
        except Exception as e:
            logger.error(f"Error processing vote at index {vote_idx}: {e}")
    
    return training_examples

def save_training_data(examples: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save training examples to a JSONL file.
    
    Args:
        examples: List of training examples
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        logger.info(f"Successfully saved {len(examples)} training examples to {output_path}")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")

def main():
    """Main entry point for the data preparation script."""
    # Define file paths
    base_dir = Path(__file__).resolve().parents[2]  # Project root directory
    votes_path = base_dir / "data" / "votes.jsonl"
    output_path = base_dir / "data" / "vote_predictor_training_data.jsonl"
    
    # Create the output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True)
    
    logger.info(f"Loading votes from {votes_path}")
    votes = load_votes(str(votes_path))
    logger.info(f"Loaded {len(votes)} votes")
    
    logger.info("Extracting training examples")
    training_examples = extract_training_examples(votes)
    logger.info(f"Extracted {len(training_examples)} training examples from {len(votes)} votes")
    
    logger.info(f"Saving training data to {output_path}")
    save_training_data(training_examples, str(output_path))
    
    # Print summary statistics
    print(f"\nData Preparation Complete:")
    print(f"- Total votes processed: {len(votes)}")
    print(f"- Total training examples extracted: {len(training_examples)}")
    print(f"- Output saved to: {output_path}")

if __name__ == "__main__":
    main() 