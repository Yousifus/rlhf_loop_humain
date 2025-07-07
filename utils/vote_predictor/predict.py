#!/usr/bin/env python3
"""
RLHF Vote Predictor Inference Module

This module uses trained neural network models to predict user preferences between different
response options. The predictor analyzes input prompts and completion pairs to determine
which option a user would likely prefer.

Input: 
- Trained model from models/vote_predictor_checkpoint/
- Calibration parameters from models/calibration_log.json
- A prompt and two completion options to compare

Output:
- Prediction of which completion the user would prefer
- Confidence score for the prediction
- Detailed prediction metadata and probabilities
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from utils.vote_predictor.calibrate import apply_calibration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "prediction.log"))
    ]
)
logger = logging.getLogger(__name__)

class VotePredictor:
    """Vote prediction model with calibrated confidence."""
    
    def __init__(
        self,
        model_path: str,
        calibration_path: str,
        device: Optional[str] = None,
        checkpoint_version: Optional[str] = None
    ):
        """
        Initialize the vote predictor with a trained model and calibration parameters.
        
        Args:
            model_path: Path to the trained model directory
            calibration_path: Path to the calibration parameters file
            device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
            checkpoint_version: Optional specific checkpoint version to load (e.g., 'v2')
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Store checkpoint version if provided
        self.checkpoint_version = checkpoint_version
        
        # Load calibration parameters
        self.load_calibration(calibration_path)
        
        # Load model and tokenizer
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load the trained vote prediction model and tokenizer.
        
        Args:
            model_path: Path to the model directory
        """
        try:
            # If checkpoint version is specified, find that specific version in the model directory
            if self.checkpoint_version:
                # Look for a subdirectory with the specified version
                version_path = os.path.join(model_path, f"v{self.checkpoint_version}")
                if os.path.exists(version_path):
                    model_path = version_path
                    logger.info(f"Using checkpoint version: v{self.checkpoint_version} at {model_path}")
                else:
                    logger.warning(f"Checkpoint version v{self.checkpoint_version} not found at {model_path}, using latest")
            
            logger.info(f"Loading model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_calibration(self, calibration_path: str) -> None:
        """
        Load calibration parameters.
        
        Args:
            calibration_path: Path to the calibration parameters file
        """
        try:
            logger.info(f"Loading calibration parameters from {calibration_path}")
            
            with open(calibration_path, 'r', encoding='utf-8') as f:
                self.calibration_params = json.load(f)
            
            # Validate calibration parameters
            required_keys = ["method", "parameters"]
            for key in required_keys:
                if key not in self.calibration_params:
                    raise ValueError(f"Calibration parameters missing required key: {key}")
            
            self.calibration_method = self.calibration_params["method"]
            logger.info(f"Using calibration method: {self.calibration_method}")
        except Exception as e:
            logger.error(f"Error loading calibration parameters: {e}")
            raise
    
    def predict_raw(
        self,
        prompt: str,
        completion_a: str,
        completion_b: str
    ) -> Dict[str, Any]:
        """
        Make a raw (uncalibrated) prediction.
        
        Args:
            prompt: The prompt text
            completion_a: First completion option
            completion_b: Second completion option
        
        Returns:
            Dictionary with raw prediction results
        """
        # Prepare input text
        text_input = (
            f"{prompt} "
            f"[Option A]: {completion_a} "
            f"[Option B]: {completion_b}"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
        
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
        
        # Get prediction and confidence
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])
        
        return {
            "preferred_completion": "A" if predicted_label == 0 else "B",
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "logits": logits.tolist()
        }
    
    def predict(
        self,
        prompt: str,
        completion_a: str,
        completion_b: str
    ) -> Dict[str, Any]:
        """
        Make a calibrated prediction.
        
        Args:
            prompt: The prompt text
            completion_a: First completion option
            completion_b: Second completion option
        
        Returns:
            Dictionary with calibrated prediction results
        """
        # Get raw prediction
        raw_prediction = self.predict_raw(prompt, completion_a, completion_b)
        
        # Apply calibration
        calibrated_confidence = apply_calibration(
            raw_prediction["confidence"],
            self.calibration_params,
            self.calibration_method
        )
        
        # Update prediction with calibrated confidence
        prediction = {
            "preferred_completion": raw_prediction["preferred_completion"],
            "confidence": calibrated_confidence,
            "raw_confidence": raw_prediction["confidence"],
            "probabilities": raw_prediction["probabilities"],
            "logits": raw_prediction["logits"],
            "calibration_method": self.calibration_method
        }
        
        # Add checkpoint version if available
        if hasattr(self, 'checkpoint_version') and self.checkpoint_version:
            prediction["model_version"] = f"v{self.checkpoint_version}"
        
        return prediction
    
    def batch_predict(
        self,
        items: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Make calibrated predictions for multiple items.
        
        Args:
            items: List of dictionaries with prompt, completion_a, and completion_b keys
        
        Returns:
            List of prediction results
        """
        predictions = []
        
        for i, item in enumerate(items):
            try:
                prompt = item.get("prompt", "")
                completion_a = item.get("completion_a", "")
                completion_b = item.get("completion_b", "")
                
                # Validate inputs
                if not prompt or not completion_a or not completion_b:
                    raise ValueError("Missing required input: prompt, completion_a, or completion_b")
                
                # Make prediction
                result = self.predict(prompt, completion_a, completion_b)
                
                # Add additional metadata from the input
                if "id" in item:
                    result["id"] = item["id"]
                
                predictions.append(result)
            except Exception as e:
                logger.error(f"Error predicting item {i}: {e}")
                # Include failed item in results with error
                predictions.append({
                    "error": str(e),
                    "id": item.get("id", f"item_{i}")
                })
        
        return predictions

class MockVotePredictor:
    """Mock implementation of vote predictor for testing and development."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_version: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the mock vote predictor.
        
        Args:
            model_path: Ignored; included for API compatibility
            calibration_path: Ignored; included for API compatibility
            device: Ignored; included for API compatibility
            checkpoint_version: Optional checkpoint version for tagging predictions
            seed: Random seed for deterministic predictions
        """
        self.seed = seed
        self.checkpoint_version = checkpoint_version
        self.calibration_method = "mock_calibration"
        random.seed(seed)
        np.random.seed(seed)
        
        # Log initialization
        logger.info(f"Initialized MockVotePredictor with seed {seed}")
    
    def predict_raw(
        self,
        prompt: str,
        completion_a: str,
        completion_b: str
    ) -> Dict[str, Any]:
        """
        Make a mock raw prediction.
        
        Args:
            prompt: The prompt text
            completion_a: First completion option
            completion_b: Second completion option
        
        Returns:
            Dictionary with mock raw prediction results
        """
        # Use length as a simple heuristic for "quality"
        len_a = len(completion_a)
        len_b = len(completion_b)
        
        # Add some randomness
        rand_factor = random.uniform(-0.2, 0.2)
        
        # Compute scaled lengths
        scaled_a = len_a / max(len_a + len_b, 1) + rand_factor
        scaled_b = len_b / max(len_a + len_b, 1) - rand_factor
        
        # Normalize to [0, 1]
        total = scaled_a + scaled_b
        if total > 0:
            prob_a = scaled_a / total
            prob_b = scaled_b / total
        else:
            prob_a = 0.5
            prob_b = 0.5
        
        # Cap to avoid extreme values
        prob_a = min(max(prob_a, 0.1), 0.9)
        prob_b = min(max(prob_b, 0.1), 0.9)
        
        # Normalize again to ensure they sum to 1
        total = prob_a + prob_b
        prob_a = prob_a / total
        prob_b = prob_b / total
        
        # Create an array of probabilities
        probabilities = [prob_a, prob_b]
        
        # Get prediction and confidence
        predicted_label = 0 if prob_a > prob_b else 1
        confidence = float(probabilities[predicted_label])
        
        # Create fake logits
        logits = [float(np.log(p / (1 - p))) for p in probabilities]
        
        return {
            "preferred_completion": "A" if predicted_label == 0 else "B",
            "confidence": confidence,
            "probabilities": probabilities,
            "logits": logits
        }
    
    def predict(
        self,
        prompt: str,
        completion_a: str,
        completion_b: str
    ) -> Dict[str, Any]:
        """
        Make a mock calibrated prediction.
        
        Args:
            prompt: The prompt text
            completion_a: First completion option
            completion_b: Second completion option
        
        Returns:
            Dictionary with mock calibrated prediction results
        """
        # Get raw prediction
        raw_prediction = self.predict_raw(prompt, completion_a, completion_b)
        
        # Apply a small calibration adjustment to simulate calibration
        raw_confidence = raw_prediction["confidence"]
        calibrated_confidence = 0.5 + (raw_confidence - 0.5) * 0.8  # Reduce extremes
        
        # Update prediction with calibrated confidence
        prediction = {
            "preferred_completion": raw_prediction["preferred_completion"],
            "confidence": float(calibrated_confidence),
            "raw_confidence": float(raw_confidence),
            "probabilities": [float(p) for p in raw_prediction["probabilities"]],
            "logits": [float(l) for l in raw_prediction["logits"]],
            "calibration_method": self.calibration_method,
            "is_mock": True
        }
        
        # Add checkpoint version if available
        if hasattr(self, 'checkpoint_version') and self.checkpoint_version:
            prediction["model_version"] = f"v{self.checkpoint_version}"
        else:
            prediction["model_version"] = "mock-v1"
        
        return prediction

def load_vote_predictor(
    model_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    device: Optional[str] = None,
    use_mock: bool = False,
    checkpoint_version: Optional[str] = None
) -> Union[VotePredictor, MockVotePredictor]:
    """
    Load a vote predictor model.
    
    Args:
        model_path: Path to the model directory. If None, uses default path.
        calibration_path: Path to the calibration parameters file. If None, uses default path.
        device: Device to run the model on ('cpu', 'cuda', or None for auto-detection)
        use_mock: Whether to use a mock predictor instead of a real model
        checkpoint_version: Optional specific checkpoint version to load
    
    Returns:
        A VotePredictor or MockVotePredictor instance
    """
    # Set default paths if not provided
    if model_path is None:
        model_path = os.path.join(project_root, "models", "vote_predictor_checkpoint")
    
    if calibration_path is None:
        calibration_path = os.path.join(project_root, "models", "calibration_log.json")
    
    # Return appropriate predictor
    if use_mock:
        return MockVotePredictor(model_path, calibration_path, device, checkpoint_version)
    else:
        return VotePredictor(model_path, calibration_path, device, checkpoint_version)

def predict_single(
    prompt: str,
    completion_a: str,
    completion_b: str,
    predictor: Optional[Union[VotePredictor, MockVotePredictor]] = None,
    model_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    use_mock: bool = False,
    checkpoint_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make a single prediction using the vote predictor.
    
    Args:
        prompt: The prompt text
        completion_a: First completion option
        completion_b: Second completion option
        predictor: Existing predictor instance to use (if None, one will be loaded)
        model_path: Path to the model directory (if predictor is None)
        calibration_path: Path to the calibration parameters file (if predictor is None)
        use_mock: Whether to use a mock predictor (if predictor is None)
        checkpoint_version: Optional specific checkpoint version to load
    
    Returns:
        Prediction result
    """
    # Create a predictor if none provided
    if predictor is None:
        predictor = load_vote_predictor(
            model_path, calibration_path, None, use_mock, checkpoint_version
        )
    
    # Make prediction
    return predictor.predict(prompt, completion_a, completion_b)

def save_predictions(predictions: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save prediction results to a file.
    
    Args:
        predictions: List of prediction results
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for prediction in predictions:
                f.write(json.dumps(prediction) + '\n')
        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")

def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(description="Make vote predictions")
    parser.add_argument("--model-path", type=str, help="Path to the model directory")
    parser.add_argument("--calibration-path", type=str, help="Path to the calibration parameters file")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to run the model on")
    parser.add_argument("--use-mock", action="store_true", help="Use mock predictor instead of real model")
    parser.add_argument("--output-path", type=str, help="Path to save predictions")
    parser.add_argument("--input-file", type=str, help="Path to input file with items to predict")
    parser.add_argument("--checkpoint-version", type=str, help="Specific checkpoint version to load")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single prediction command
    single_parser = subparsers.add_parser("single", help="Make a single prediction")
    single_parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    single_parser.add_argument("--completion-a", type=str, required=True, help="First completion option")
    single_parser.add_argument("--completion-b", type=str, required=True, help="Second completion option")
    
    # Batch prediction command
    batch_parser = subparsers.add_parser("batch", help="Make predictions for multiple items")
    batch_parser.add_argument("--input-file", type=str, required=True, help="Path to input file with items to predict")
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = load_vote_predictor(
        args.model_path, args.calibration_path, args.device, args.use_mock, args.checkpoint_version
    )
    
    if args.command == "single":
        # Make single prediction
        prediction = predictor.predict(args.prompt, args.completion_a, args.completion_b)
        
        # Print result
        print(json.dumps(prediction, indent=2))
        
        # Save result if output path specified
        if args.output_path:
            save_predictions([prediction], args.output_path)
    
    elif args.command == "batch" or args.input_file:
        # Load input items
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                items = [json.loads(line) for line in f]
        except Exception as e:
            logger.error(f"Error loading input file: {e}")
            sys.exit(1)
        
        # Make predictions
        predictions = predictor.batch_predict(items)
        
        # Print summary
        print(f"Made {len(predictions)} predictions")
        print(f"Average confidence: {np.mean([p.get('confidence', 0) for p in predictions if 'error' not in p]):.4f}")
        
        # Save results if output path specified
        if args.output_path:
            save_predictions(predictions, args.output_path)
        else:
            # Print first result as sample
            print("\nSample prediction:")
            print(json.dumps(predictions[0], indent=2))
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
