#!/usr/bin/env python3
"""
RLHF Loop Interface

This module provides an interface to run the full RLHF (Reinforcement Learning from Human Feedback) loop.
It integrates all components of the system:
1. Prompt generation
2. Completion generation
3. Human feedback collection
4. Vote prediction with calibrated confidence
5. Model tuning (future integration)

The RLHF loop can be run in automated mode or with human feedback integration.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Import project modules
from utils.vote_predictor.predict import load_vote_predictor, predict_single
import interface.voting_ui as voting_ui
from utils.completions import generate_completions
from utils.vote_predictor.drift_monitor import run_drift_analysis, DriftAnalysisConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "rlhf_loop.log"))
    ]
)
logger = logging.getLogger(__name__)

class RLHFLoop:
    """RLHF Loop Manager Class."""
    
    def __init__(
        self,
        prompts_path: str,
        completions_path: str,
        votes_path: str,
        predictions_path: str,
        model_path: str,
        calibration_path: str,
        confidence_threshold: float = 0.75,
        device: Optional[str] = None,
        mock_predictor: bool = False,
        monitor_drift: bool = False
    ):
        """
        Initialize the RLHF Loop Manager.
        
        Args:
            prompts_path: Path to the prompts file
            completions_path: Path to the completions log file
            votes_path: Path to the votes file
            predictions_path: Path to save predictions
            model_path: Path to the trained model directory
            calibration_path: Path to the calibration parameters file
            confidence_threshold: Threshold for confident predictions
            device: Device to run inference on
            mock_predictor: Whether to use a mock predictor if model loading fails
            monitor_drift: Whether to monitor for model drift
        """
        self.prompts_path = prompts_path
        self.completions_path = completions_path
        self.votes_path = votes_path
        self.predictions_path = predictions_path
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.mock_predictor = mock_predictor
        self.monitor_drift = monitor_drift
        
        # Load the predictor
        try:
            if mock_predictor:
                from utils.vote_predictor.predict import MockVotePredictor
                self.predictor = MockVotePredictor(model_path, calibration_path, device)
                self.model_loaded = True
                logger.info("Using mock vote predictor")
            else:
                self.predictor = load_vote_predictor(model_path, calibration_path, device)
                self.model_loaded = True
                logger.info("Vote predictor model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load vote predictor: {e}")
            if mock_predictor:
                from utils.vote_predictor.predict import MockVotePredictor
                self.predictor = MockVotePredictor(calibration_path=calibration_path)
                self.model_loaded = True
                logger.info("Falling back to mock predictor")
            else:
                raise
        
        logger.info("RLHF Loop Manager initialized")
    
    def load_prompts(self) -> List[Dict[str, Any]]:
        """
        Load prompts from the prompts file.
        
        Returns:
            List of prompt objects
        """
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                prompts = [json.loads(line) for line in f]
            
            logger.info(f"Loaded {len(prompts)} prompts from {self.prompts_path}")
            return prompts
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise
    
    def load_existing_data(self) -> Dict[str, Any]:
        """
        Load existing data from the system.
        
        Returns:
            Dictionary with existing completions, votes, and other data
        """
        data = {
            "completions": [],
            "votes": [],
            "processed_prompt_ids": set()
        }
        
        # Load completions
        try:
            if os.path.exists(self.completions_path):
                with open(self.completions_path, 'r', encoding='utf-8') as f:
                    data["completions"] = [json.loads(line) for line in f]
                
                # Extract processed prompt IDs
                for completion in data["completions"]:
                    data["processed_prompt_ids"].add(completion["prompt_id"])
                
                logger.info(f"Loaded {len(data['completions'])} existing completions")
        except Exception as e:
            logger.warning(f"Error loading completions: {e}")
        
        # Load votes
        try:
            if os.path.exists(self.votes_path):
                with open(self.votes_path, 'r', encoding='utf-8') as f:
                    data["votes"] = [json.loads(line) for line in f]
                
                logger.info(f"Loaded {len(data['votes'])} existing votes")
        except Exception as e:
            logger.warning(f"Error loading votes: {e}")
        
        return data
    
    def run_automated_loop(self, num_prompts: int = 5, human_feedback_ratio: float = 0.2) -> None:
        """
        Run the RLHF loop in automated mode with some human feedback.
        
        Args:
            num_prompts: Number of prompts to process
            human_feedback_ratio: Ratio of prompts that should get human feedback
        """
        # Load existing data
        data = self.load_existing_data()
        
        # Load prompts
        all_prompts = self.load_prompts()
        
        # Filter unprocessed prompts
        unprocessed_prompts = [
            p for p in all_prompts 
            if p.get("id") not in data["processed_prompt_ids"]
        ]
        
        if not unprocessed_prompts:
            logger.warning("No unprocessed prompts found")
            return
        
        # Limit to requested number of prompts
        prompts_to_process = unprocessed_prompts[:num_prompts]
        
        logger.info(f"Processing {len(prompts_to_process)} prompts")
        
        # Process each prompt
        for i, prompt in enumerate(prompts_to_process, 1):
            logger.info(f"Processing prompt {i}/{len(prompts_to_process)}")
            
            # Generate completions
            completion_pairs = self._generate_completion_pairs(prompt)
            
            # Determine if this prompt should get human feedback
            should_get_human_feedback = i <= int(num_prompts * human_feedback_ratio)
            
            # Process completion pairs
            if should_get_human_feedback:
                logger.info("Getting human feedback for this prompt")
                votes = self._collect_human_feedback(prompt, completion_pairs)
            else:
                logger.info("Using model predictions for this prompt")
                votes = self._predict_votes(prompt, completion_pairs)
            
            # Log results
            logger.info(f"Processed {len(votes)} votes for prompt {prompt.get('id')}")
        
        # Run drift monitoring if enabled
        if self.monitor_drift:
            self._check_for_drift()
            
        logger.info("RLHF loop complete")
    
    def _generate_completion_pairs(self, prompt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate completion pairs for a prompt.
        
        Args:
            prompt: Prompt object
            
        Returns:
            List of completion pair objects
        """
        prompt_text = prompt.get("text", "")
        if not prompt_text and "generated_prompt" in prompt:
            prompt_text = prompt.get("generated_prompt", "")
            
        prompt_id = prompt.get("id", f"unknown-{datetime.now().isoformat()}")
        
        # Generate completions
        completions = generate_completions(prompt_text, n_completions=4)
        
        # Save completions
        with open(self.completions_path, 'a', encoding='utf-8') as f:
            for comp in completions["completions"]:  # Access the completions from result dict
                completion_obj = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "completion": comp,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(completion_obj) + '\n')
        
        # Create pairs for comparison
        pairs = []
        completions_list = completions["completions"]  # Access the completions from result dict
        for i in range(len(completions_list)):
            for j in range(i + 1, len(completions_list)):
                pair = {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "completion_a": completions_list[i],
                    "completion_b": completions_list[j],
                    "pair_id": f"{prompt_id}-{i}-{j}"
                }
                pairs.append(pair)
        
        logger.info(f"Generated {len(pairs)} completion pairs for prompt {prompt_id}")
        return pairs
    
    def _collect_human_feedback(
        self, 
        prompt: Dict[str, Any], 
        completion_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Collect human feedback on completion pairs.
        
        Args:
            prompt: Prompt object
            completion_pairs: List of completion pairs
            
        Returns:
            List of vote objects
        """
        prompt_id = prompt.get("id", "unknown")
        prompt_text = prompt.get("text", "")
        
        logger.info(f"Collecting human feedback for prompt {prompt_id}")
        print(f"\n=== Human Feedback - Prompt {prompt_id} ===")
        print(f"Prompt: {prompt_text}")
        
        votes = []
        
        # Process each completion pair
        for pair in completion_pairs:
            print("\n" + "-"*50)
            print(f"Comparing completion pair {pair['pair_id']}:")
            
            # Display completions
            print("\nCompletion A:")
            print(pair["completion_a"])
            print("\nCompletion B:")
            print(pair["completion_b"])
            
            # Get preference
            choice = ""
            while choice not in ["A", "B"]:
                choice = input("\nWhich completion do you prefer? (A/B): ").strip().upper()
            
            # Get confidence
            confidence = 0.0
            while confidence <= 0.0 or confidence > 1.0:
                try:
                    confidence = float(input("Enter your confidence (0.0-1.0): ").strip())
                except ValueError:
                    confidence = 0.0
            
            # Create vote object
            vote = {
                "prompt_id": prompt_id,
                "pair_id": pair["pair_id"],
                "prompt": prompt_text,
                "completion_a": pair["completion_a"],
                "completion_b": pair["completion_b"],
                "choice": choice,
                "confidence": confidence,
                "is_model_vote": False,
                "timestamp": datetime.now().isoformat()
            }
            
            votes.append(vote)
            
            # Save to votes file
            with open(self.votes_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(vote) + '\n')
            
            print(f"Vote recorded: Preferred {choice} with confidence {confidence:.2f}")
        
        logger.info(f"Collected {len(votes)} human votes for prompt {prompt_id}")
        return votes
    
    def _predict_votes(
        self, 
        prompt: Dict[str, Any], 
        completion_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict votes using the calibrated model.
        
        Args:
            prompt: Prompt object
            completion_pairs: List of completion pairs
            
        Returns:
            List of vote objects
        """
        prompt_id = prompt.get("id", "unknown")
        votes = []
        
        # Make predictions for each pair
        for pair in completion_pairs:
            # Get prediction using the loaded predictor (real or mock)
            prediction = self.predictor.predict(
                pair["prompt"],
                pair["completion_a"],
                pair["completion_b"]
            )
            
            # Log if using mock predictor
            if hasattr(prediction, 'get') and prediction.get('is_mock', False):
                logger.info(f"Using mock prediction: {prediction['preferred_completion']} with confidence {prediction['confidence']:.2f}")
            
            # Create vote object
            vote = {
                "prompt_id": prompt_id,
                "pair_id": pair["pair_id"],
                "prompt": pair["prompt"],
                "completion_a": pair["completion_a"],
                "completion_b": pair["completion_b"],
                "choice": "A" if prediction["preferred_completion"] == "A" else "B",
                "confidence": prediction["confidence"],
                "is_model_vote": True,
                "timestamp": datetime.now().isoformat(),
                "raw_prediction": prediction
            }
            
            # Add confidence assessment
            vote["is_confident"] = prediction["confidence"] >= self.confidence_threshold
            
            votes.append(vote)
            
            # Save to votes file
            with open(self.votes_path, 'a', encoding='utf-8') as f:
                # We'll save a simplified version to the votes file
                vote_to_save = {
                    "prompt_id": vote["prompt_id"],
                    "pair_id": vote["pair_id"],
                    "prompt": vote["prompt"],
                    "completion_a": vote["completion_a"],
                    "completion_b": vote["completion_b"],
                    "choice": vote["choice"],
                    "confidence": vote["confidence"],
                    "is_model_vote": True,
                    "timestamp": vote["timestamp"]
                }
                f.write(json.dumps(vote_to_save) + '\n')
        
        # Save full predictions
        with open(self.predictions_path, 'a', encoding='utf-8') as f:
            for vote in votes:
                f.write(json.dumps(vote) + '\n')
        
        logger.info(f"Generated {len(votes)} model votes for prompt {prompt_id}")
        return votes

    def _check_for_drift(self) -> Dict[str, Any]:
        """
        Run drift monitoring to check for model drift.
        
        Returns:
            Dictionary with drift analysis results
        """
        logger.info("Running drift monitoring analysis")
        
        # Configure paths
        reflection_path = os.path.join(project_root, "models", "meta_reflection_log.jsonl")
        drift_output_dir = os.path.join(project_root, "models", "drift_analysis")
        
        # Skip if no reflection log exists
        if not os.path.exists(reflection_path):
            logger.warning("No reflection log found for drift monitoring")
            return {}
            
        # Configure drift analysis
        drift_config = DriftAnalysisConfig(
            time_window_days=7,
            n_clusters=5,
            confidence_drift_threshold=0.1,
            alert_accuracy_change_threshold=0.1
        )
        
        # Run drift analysis
        try:
            drift_analysis = run_drift_analysis(
                reflection_path=reflection_path,
                output_dir=drift_output_dir,
                config=drift_config
            )
            
            # Log summary
            summary = drift_analysis.get("summary", {})
            logger.info("Drift Analysis Summary:")
            logger.info(f"Time-based drift detected: {summary.get('time_drift_detected', False)}")
            logger.info(f"Number of clusters: {summary.get('num_clusters', 0)}")
            logger.info(f"Potential drift clusters: {len(summary.get('potential_drift_clusters', []))}")
            logger.info(f"Calibration drift detected: {summary.get('calibration_drift_detected', False)}")
            
            return drift_analysis
        except Exception as e:
            logger.error(f"Error running drift monitoring: {e}")
            return {}

def main():
    """Main entry point for the RLHF loop interface."""
    parser = argparse.ArgumentParser(description="RLHF Loop Interface")
    
    # Paths
    parser.add_argument(
        "--prompts-path", 
        type=str, 
        default=os.path.join(project_root, "prompts", "generated_prompts.jsonl"),
        help="Path to the prompts file"
    )
    
    parser.add_argument(
        "--completions-path", 
        type=str, 
        default=os.path.join(project_root, "data", "raw_completions_log.jsonl"),
        help="Path to the completions log file"
    )
    
    parser.add_argument(
        "--votes-path", 
        type=str, 
        default=os.path.join(project_root, "data", "votes.jsonl"),
        help="Path to the votes file"
    )
    
    parser.add_argument(
        "--predictions-path", 
        type=str, 
        default=os.path.join(project_root, "data", "predictions.jsonl"),
        help="Path to save predictions"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=os.path.join(project_root, "models", "vote_predictor_checkpoint"),
        help="Path to the trained model directory"
    )
    
    parser.add_argument(
        "--calibration-path", 
        type=str, 
        default=os.path.join(project_root, "models", "calibration_log.json"),
        help="Path to the calibration parameters file"
    )
    
    # Control parameters
    parser.add_argument(
        "--num-prompts", 
        type=int, 
        default=5,
        help="Number of prompts to process"
    )
    
    parser.add_argument(
        "--human-feedback-ratio", 
        type=float, 
        default=0.2,
        help="Ratio of prompts that should get human feedback"
    )
    
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.75,
        help="Threshold for confident predictions"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--mock-predictor",
        action="store_true",
        help="Use a mock predictor if model loading fails"
    )
    
    parser.add_argument(
        "--monitor-drift",
        action="store_true",
        help="Monitor for model drift during the RLHF loop"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Configure debug output
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print(f"DEBUG: Arguments: {args}")
    
    # Create RLHF loop manager
    print("Creating RLHF loop manager...")
    rlhf_loop = RLHFLoop(
        prompts_path=args.prompts_path,
        completions_path=args.completions_path,
        votes_path=args.votes_path,
        predictions_path=args.predictions_path,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        mock_predictor=args.mock_predictor,
        monitor_drift=args.monitor_drift
    )
    
    # Run the automated loop
    print(f"Running automated loop with {args.num_prompts} prompts, " 
          f"human feedback ratio: {args.human_feedback_ratio}...")
    rlhf_loop.run_automated_loop(
        num_prompts=args.num_prompts,
        human_feedback_ratio=args.human_feedback_ratio
    )
    
    print("RLHF loop completed successfully!")

if __name__ == "__main__":
    main() 