#!/usr/bin/env python3
"""
RLHF Vote Predictor Training Module

This module trains the neural network model to predict user preferences and choices.
The training process uses collected annotation data to learn patterns in user preferences,
enabling the model to better anticipate user selections.

Input: Training data from data/vote_predictor_training_data.jsonl
Output: Trained model saved to models/vote_predictor_checkpoint/
       Training metrics logged to models/vote_predictor_training_log.json
"""

import os
import sys
import json
import argparse
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Add the project root to the Python path so imports work correctly
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Increase the timeout for downloading models
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "vote_predictor_training.log"))
    ]
)
logger = logging.getLogger(__name__)

class VoteDataset(Dataset):
    """Dataset for vote prediction training."""
    
    def __init__(
        self, 
        examples: List[Dict[str, Any]], 
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset with vote prediction examples.
        
        Args:
            examples: List of vote prediction examples from data_prep.py
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract sample weights if available
        self.has_weights = any("weight" in example for example in examples)
        if self.has_weights:
            logger.info("Sample weights detected in training data")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Format:
            prompt: The text prompt
            completion_a: First completion option
            completion_b: Second completion option
            label: 0 if A preferred, 1 if B preferred
            weight: Optional sample weight for weighted training
        """
        example = self.examples[idx]
        
        # Prepare text input for the model by combining prompt and completions
        # Format: [CLS] prompt [SEP] completion_a [SEP] completion_b [SEP]
        text_input = (
            f"{example['prompt']} "
            f"[Option A]: {example['completion_a']} "
            f"[Option B]: {example['completion_b']}"
        )
        
        # Tokenize the input
        encoding = self.tokenizer(
            text_input,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Convert from batch format (1, seq_len) to just (seq_len)
        item = {
            key: val.squeeze(0) for key, val in encoding.items()
        }
        
        # Add the label
        item["labels"] = torch.tensor(example["label"], dtype=torch.long)
        
        # Add sample weight if available
        if "weight" in example:
            item["weight"] = torch.tensor(example["weight"], dtype=torch.float)
        
        return item

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load vote prediction examples from a JSONL file.
    
    Args:
        data_path: Path to the vote predictor training data
        
    Returns:
        List of vote prediction examples
    """
    examples = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Error parsing JSON at line {line_idx}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(examples)} examples from {data_path}")
    return examples

def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for the model.
    
    Args:
        pred: Prediction object from Trainer
        
    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions
    
    # For AUC and log loss, we need probabilities
    probs = torch.nn.functional.softmax(torch.tensor(preds), dim=1).numpy()
    pred_class = np.argmax(preds, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, pred_class)
    
    # For binary classification, we take probability of class 1
    binary_probs = probs[:, 1]
    try:
        auc = roc_auc_score(labels, binary_probs)
    except ValueError:
        # This handles the case where there's only one class in the batch
        auc = 0.0
    
    # Compute log loss
    try:
        loss = log_loss(labels, probs)
    except ValueError:
        loss = np.nan
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "log_loss": loss
    }

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(args) -> Dict[str, Any]:
    """
    Train the preference model using HuggingFace Trainer.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary of training results and metrics
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Define paths
    base_dir = Path(project_root)
    data_file = base_dir / "data" / args.data_file
    model_dir = base_dir / "models" / args.model_dir
    log_file = base_dir / "models" / "vote_predictor_training_log.json"
    
    # Create model directory if it doesn't exist
    model_dir.parent.mkdir(exist_ok=True)
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    
    # Check if we're fine-tuning from a checkpoint
    is_fine_tuning = args.checkpoint_dir and os.path.exists(args.checkpoint_dir)
    
    if is_fine_tuning:
        checkpoint_dir = Path(args.checkpoint_dir)
        logger.info(f"Fine-tuning from checkpoint: {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load training data
    logger.info(f"Loading data from {data_file}")
    examples = load_data(str(data_file))
    
    if not examples:
        logger.error("No training examples found.")
        sys.exit(1)
    
    # Check if we have sample weights
    has_weights = any("weight" in example for example in examples)
    if has_weights:
        logger.info("Sample weights detected in training data")
    
    # Create the dataset
    logger.info("Preparing dataset")
    dataset = VoteDataset(examples, tokenizer, max_length=args.max_length)
    
    # Split data into train and validation sets
    if args.validation_split > 0 and len(dataset) > 2:  # Need at least 2 examples to split
        val_size = max(1, int(len(dataset) * args.validation_split))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        logger.info(f"Split dataset: {train_size} training examples, {val_size} validation examples")
    else:
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Using all {len(dataset)} examples for training (no validation split)")
    
    # Load the model with offline mode if requested
    try:
        logger.info(f"Loading model: {args.model_name if not is_fine_tuning else checkpoint_dir}")
        
        # Define model configuration
        if is_fine_tuning:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, 
                num_labels=2
            )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=os.path.join(str(model_dir), "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,  # Only keep the 3 most recent checkpoints
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="auc" if val_dataset else None,
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none" if args.no_wandb else "wandb",
        # Handle sample weights
        include_inputs_for_metrics=True,
    )
    
    # Define callbacks
    callbacks = []
    if val_dataset and args.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        callbacks.append(early_stopping)
    
    # Create a custom loss function that supports sample weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Extract sample weights if present
            weights = None
            if "weight" in inputs:
                weights = inputs.pop("weight")
            
            # Get regular loss
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            
            # Compute standard cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="none"
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            # Apply sample weights if available
            if weights is not None:
                loss = loss * weights
            
            # Take mean or sum
            loss = loss.mean()
            
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer
    if has_weights:
        logger.info("Using weighted trainer with sample weights")
        trainer_class = WeightedTrainer
    else:
        trainer_class = Trainer
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save the trained model
    logger.info(f"Saving model to {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Calculate and save metrics
    metrics = {
        "training_time": train_result.metrics.get("train_runtime", 0),
        "training_samples": len(examples),
        "training_loss": train_result.metrics.get("train_loss", 0),
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "fine_tuned_from": str(args.checkpoint_dir) if is_fine_tuning else None,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "parameters": {
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "validation_split": args.validation_split,
            "max_length": args.max_length,
            "seed": args.seed,
        }
    }
    
    # If we have a validation set, add validation metrics
    if val_dataset:
        eval_result = trainer.evaluate()
        metrics["validation_loss"] = eval_result.get("eval_loss", 0)
        metrics["validation_accuracy"] = eval_result.get("eval_accuracy", 0)
        metrics["validation_auc"] = eval_result.get("eval_auc", 0)
    
    # Save metrics to log file
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved training metrics to {log_file}")
    except Exception as e:
        logger.error(f"Error saving training metrics: {e}")
    
    return metrics

def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train meta-evaluator model")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased", 
                        help="Model name or path for base model")
    parser.add_argument("--data-file", type=str, default="vote_predictor_training_data.jsonl",
                        help="Name of the training data file in data directory")
    parser.add_argument("--model-dir", type=str, default="vote_predictor_checkpoint",
                        help="Directory to save model in models directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing a checkpoint to fine-tune from")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--logging-steps", type=int, default=50,
                        help="Logging steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="Save steps")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args)
    
    logger.info("Training complete")

if __name__ == "__main__":
    main()
