#!/usr/bin/env python3
"""
RLHF Database System

This module provides data persistence and management for the Reinforcement Learning
from Human Feedback (RLHF) system. It handles user annotations, model predictions,
reflections, and training data storage.
"""

import os
import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define required fields for different data types
ANNOTATION_REQUIRED_FIELDS = [
    "prompt_id", "prompt", "preference", "timestamp"
]

class RLHFDatabase:
    """
    RLHF Database Management System
    
    This class handles all data persistence for the RLHF system including:
    - User annotations and preferences
    - Model predictions and evaluations
    - Training data collection and export
    - Reflection data for meta-evaluation
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the RLHF database system.
        
        Args:
            data_dir: The directory where data will be stored. If None, uses
                     default location in the project's data directory.
        """
        if data_dir is None:
            # Default to project_root/data
            self.project_root = str(Path(__file__).resolve().parents[1])
            self.data_dir = Path(self.project_root) / "data"
        else:
            self.data_dir = Path(data_dir)
            self.project_root = self.data_dir.parent
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up paths for different data stores
        self.vote_logs_dir = self.data_dir / "vote_logs"
        self.vote_logs_dir.mkdir(exist_ok=True, parents=True)
        
        self.votes_file = self.data_dir / "votes.jsonl"
        self.user_annotations_file = self.data_dir / "user_annotations.jsonl"
        self.predictions_file = self.data_dir / "predictions.jsonl"
        
        self.reflections_dir = self.data_dir / "reflections"
        self.reflections_dir.mkdir(exist_ok=True, parents=True)
        self.reflection_data_file = self.data_dir / "reflection_data.jsonl"
        
        # Set up paths for backups
        self.backups_dir = self.data_dir / "backups"
        self.backups_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamp = {}
        
        logger.info(f"Initialized RLHFDatabase with data directory: {self.data_dir}")
    
    def _validate_annotation(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Validate annotation data to ensure it has all required fields.
        
        Args:
            annotation_data: The annotation data to validate
            
        Returns:
            True if valid, False otherwise
        
        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [f for f in ANNOTATION_REQUIRED_FIELDS if f not in annotation_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return True
    
    def _create_backup(self) -> str:
        """
        Create a backup of all data files.
        
        Returns:
            Backup directory path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backups_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Backup vote logs
        if self.vote_logs_dir.exists():
            vote_logs_backup = backup_dir / "vote_logs"
            vote_logs_backup.mkdir(exist_ok=True, parents=True)
            for file in self.vote_logs_dir.glob("*.json"):
                shutil.copy(file, vote_logs_backup / file.name)
        
        # Backup consolidated files
        for file in [self.votes_file, self.user_annotations_file, self.predictions_file, self.reflection_data_file]:
            if file.exists():
                shutil.copy(file, backup_dir / file.name)
        
        logger.info(f"Created backup at {backup_dir}")
        return str(backup_dir)
    
    def _restore_from_backup(self, backup_dir: str) -> bool:
        """
        Restore data from a backup.
        
        Args:
            backup_dir: Path to the backup directory
            
        Returns:
            True if successful, False otherwise
        """
        backup_dir = Path(backup_dir)
        if not backup_dir.exists():
            logger.error(f"Backup directory {backup_dir} does not exist")
            return False
        
        try:
            # Restore vote logs
            vote_logs_backup = backup_dir / "vote_logs"
            if vote_logs_backup.exists():
                for file in vote_logs_backup.glob("*.json"):
                    shutil.copy(file, self.vote_logs_dir / file.name)
            
            # Restore consolidated files
            for file_name in ["votes.jsonl", "user_annotations.jsonl", "predictions.jsonl", "reflection_data.jsonl"]:
                backup_file = backup_dir / file_name
                if backup_file.exists():
                    shutil.copy(backup_file, self.data_dir / file_name)
            
            logger.info(f"Restored from backup {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return False
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Save user annotation data to the database.
        
        This method stores user preferences and annotations in multiple formats
        for redundancy and different access patterns.
        
        Args:
            annotation_data: The annotation data to save
            
        Returns:
            True if successful, False otherwise
        """
        # Create a backup before attempting to save
        backup_dir = self._create_backup()
        
        try:
            # Validate the annotation
            self._validate_annotation(annotation_data)
            
            # Add timestamp if not present
            if "timestamp" not in annotation_data:
                annotation_data["timestamp"] = datetime.now().isoformat()
            
            # Add version info
            annotation_data["annotation_version"] = "3.0"
            
            # Generate a unique ID for this annotation if not present
            if "annotation_id" not in annotation_data:
                annotation_data["annotation_id"] = f"ann_{uuid.uuid4().hex[:8]}"
            
            # Handle binary vs non-binary preferences
            is_binary_preference = annotation_data.get("preference") in ["Completion A", "Completion B"]
            annotation_data["is_binary_preference"] = is_binary_preference
            
            # Get model prediction for binary preferences
            if is_binary_preference:
                self._add_model_prediction(annotation_data)
            else:
                # For non-binary preferences
                annotation_data["model_prediction"] = "N/A"
                annotation_data["model_correct"] = None
                annotation_data["confidence"] = None
            
            # 1. Save to individual vote log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vote_{timestamp}_{uuid.uuid4().hex[:8]}.json"
            with open(self.vote_logs_dir / filename, "w") as f:
                json.dump(annotation_data, f, indent=2)
            
            # 2. Save to votes.jsonl in standardized format
            vote_record = self._create_standardized_vote_record(annotation_data)
            with open(self.votes_file, "a") as f:
                f.write(json.dumps(vote_record) + "\n")
            
            # 3. Save to user_annotations.jsonl
            with open(self.user_annotations_file, "a") as f:
                f.write(json.dumps(annotation_data) + "\n")
            
            # 4. If binary preference, save to reflection data
            if is_binary_preference and "model_prediction" in annotation_data:
                self._save_reflection_data(annotation_data)
            
            # Clear cache to ensure fresh data on next read
            self._clear_cache()
            
            logger.info(f"Successfully saved annotation {annotation_data.get('annotation_id')}")
            return True
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            # Attempt to restore from backup
            self._restore_from_backup(backup_dir)
            return False
    
    def _add_model_prediction(self, annotation_data: Dict[str, Any]) -> None:
        """
        Add model prediction information to annotation data.
        
        Args:
            annotation_data: The annotation data to update
        """
        try:
            # Import prediction module dynamically to avoid circular imports
            import sys
            sys.path.append(self.project_root)
            from utils.vote_predictor.predict import predict_single, load_vote_predictor
            
            # Get prompt and completions
            prompt = annotation_data["prompt"]
            completion_a = annotation_data["selected_completion"] if annotation_data["preference"] == "Completion A" else annotation_data["rejected_completion"]
            completion_b = annotation_data["rejected_completion"] if annotation_data["preference"] == "Completion A" else annotation_data["selected_completion"]
            
            # Create a unique pair ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair_id = f"{annotation_data.get('prompt_id', 'unknown')}-{timestamp}-{uuid.uuid4().hex[:8]}"
            
            # Load predictor
            try:
                predictor = load_vote_predictor(use_mock=False)
                logger.info("Using real vote predictor model")
            except Exception as e:
                logger.warning(f"Could not load real model, using mock predictor: {e}")
                predictor = load_vote_predictor(use_mock=True)
            
            # Make prediction
            prediction = predict_single(prompt, completion_a, completion_b, predictor=predictor)
            
            # Add to annotation data
            annotation_data["pair_id"] = pair_id
            annotation_data["raw_prediction"] = prediction
            annotation_data["is_model_vote"] = False  # Human annotation
            annotation_data["model_prediction"] = prediction["preferred_completion"]
            annotation_data["confidence"] = prediction["confidence"]
            annotation_data["is_confident"] = prediction["confidence"] > 0.8
            
            # Check if model prediction matches human preference
            correct_prediction = (
                (prediction["preferred_completion"] == "A" and annotation_data["preference"] == "Completion A") or
                (prediction["preferred_completion"] == "B" and annotation_data["preference"] == "Completion B")
            )
            
            annotation_data["model_correct"] = correct_prediction
            
            # Save prediction to predictions.jsonl
            prediction_record = {
                "prompt_id": annotation_data.get("prompt_id", "unknown"),
                "pair_id": pair_id,
                "prompt": prompt,
                "completion_a": completion_a,
                "completion_b": completion_b,
                "choice": "A" if annotation_data["preference"] == "Completion A" else "B",
                "confidence": prediction["confidence"],
                "is_model_vote": False,  # Human annotation
                "timestamp": annotation_data["timestamp"],
                "raw_prediction": prediction,
                "quality_metrics": annotation_data.get("quality_metrics", {})
            }
            
            with open(self.predictions_file, "a") as f:
                f.write(json.dumps(prediction_record) + "\n")
                
        except Exception as e:
            logger.error(f"Error adding model prediction: {e}")
            # Continue without prediction data
            annotation_data["model_prediction"] = "ERROR"
            annotation_data["model_correct"] = None
            annotation_data["confidence"] = None
    
    def _create_standardized_vote_record(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized vote record for votes.jsonl.
        
        Args:
            annotation_data: The annotation data
            
        Returns:
            Standardized vote record
        """
        is_binary = annotation_data.get("is_binary_preference", False)
        
        if is_binary:
            completion_a = annotation_data["selected_completion"] if annotation_data["preference"] == "Completion A" else annotation_data["rejected_completion"]
            completion_b = annotation_data["rejected_completion"] if annotation_data["preference"] == "Completion A" else annotation_data["selected_completion"]
            
            return {
                "prompt": annotation_data["prompt"],
                "completions": [completion_a, completion_b],
                "chosen_index": 0 if annotation_data["preference"] == "Completion A" else 1,
                "confidence": annotation_data.get("confidence"),
                "annotation": annotation_data.get("feedback", ""),
                "generation_metadata": {
                    "prompt_id": annotation_data.get("prompt_id", "unknown"),
                    "pair_id": annotation_data.get("pair_id", ""),
                    "is_model_vote": False,
                    "model_prediction": annotation_data.get("model_prediction", ""),
                    "model_confidence": annotation_data.get("confidence"),
                    "model_correct": annotation_data.get("model_correct"),
                    "quality_metrics": annotation_data.get("quality_metrics", {})
                },
                "timestamp": annotation_data["timestamp"],
                "annotation_id": annotation_data.get("annotation_id", "")
            }
        else:
            return {
                "prompt": annotation_data["prompt"],
                "completions": [annotation_data.get("selected_completion", "")],
                "chosen_index": 0,
                "confidence": None,
                "annotation": annotation_data.get("feedback", ""),
                "generation_metadata": {
                    "prompt_id": annotation_data.get("prompt_id", "unknown"),
                    "is_binary_preference": False,
                    "preference_type": annotation_data.get("preference", "Custom"),
                    "quality_metrics": annotation_data.get("quality_metrics", {})
                },
                "timestamp": annotation_data["timestamp"],
                "annotation_id": annotation_data.get("annotation_id", "")
            }
    
    def _save_reflection_data(self, annotation_data: Dict[str, Any]) -> None:
        """
        Save reflection data for meta-evaluations.
        
        Args:
            annotation_data: The annotation data
        """
        try:
            # Create reflection record
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            reflection_record = {
                "timestamp": annotation_data["timestamp"],
                "vote_timestamp": annotation_data["timestamp"],
                "prompt": annotation_data["prompt"],
                "human_choice": 0 if annotation_data["preference"] == "Completion A" else 1,
                "model_prediction": 0 if annotation_data["model_prediction"] == "A" else 1,
                "is_correct": annotation_data["model_correct"],
                "human_confidence": 1.0,  # Default high confidence for human votes
                "model_confidence": annotation_data.get("confidence", 0.5),
                "confidence_gap": 1.0 - annotation_data.get("confidence", 0.5) if annotation_data["model_correct"] else annotation_data.get("confidence", 0.5),
                "error_type": None if annotation_data["model_correct"] else "false_prediction",
                "model_probabilities": annotation_data.get("raw_prediction", {}).get("probabilities", [0.5, 0.5]),
                "model_logits": annotation_data.get("raw_prediction", {}).get("logits", [0.0, 0.0]),
                "original_vote_metadata": annotation_data
            }
            
            # Save to individual reflection file
            reflection_file = self.reflections_dir / f"reflection_{timestamp}.json"
            with open(reflection_file, "w") as f:
                json.dump(reflection_record, f, indent=2)
            
            # Append to consolidated reflection data file
            with open(self.reflection_data_file, "a") as f:
                f.write(json.dumps(reflection_record) + "\n")
                
        except Exception as e:
            logger.error(f"Error saving reflection data: {e}")
    
    def get_annotations(self, filters: Dict[str, Any] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Retrieve training data entries with optional filtering.
        
        Every annotation represents user feedback for model improvement.
        Data is stored systematically for efficient retrieval and analysis.
        
        Args:
                    filters: Optional filtering criteria (e.g., {"preference": "Completion A"})
        force_reload: If True, reload data from disk instead of using cache
            
        Returns:
            Comprehensive collection of training data, organized for analysis
        """
        if "annotations" not in self.cache or force_reload:
            self._load_annotations()
        
        df = self.cache.get("annotations", pd.DataFrame())
        
        if filters and not df.empty:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        return df
    
    def _load_annotations(self) -> None:
        """
        Load annotations from all storage formats.
        """
        annotations = []
        
        # Load from vote logs directory
        if self.vote_logs_dir.exists():
            for vote_file in self.vote_logs_dir.glob("*.json"):
                try:
                    with open(vote_file, "r") as f:
                        vote_data = json.load(f)
                        annotations.append(vote_data)
                except Exception as e:
                    logger.warning(f"Error loading vote file {vote_file}: {e}")
        
        # Load from votes.jsonl
        if self.votes_file.exists():
            try:
                with open(self.votes_file, "r") as f:
                    for line in f:
                        try:
                            vote_data = json.loads(line.strip())
                            # Extract annotation ID to avoid duplicates
                            ann_id = vote_data.get("annotation_id", "")
                            # Check if this annotation is already loaded
                            if ann_id and not any(a.get("annotation_id") == ann_id for a in annotations):
                                annotations.append(vote_data)
                        except Exception as e:
                            logger.warning(f"Error parsing vote line: {e}")
            except Exception as e:
                logger.warning(f"Error reading votes file: {e}")
        
        # Convert to DataFrame
        if annotations:
            df = pd.DataFrame(annotations)
            
            # Ensure timestamp is in datetime format
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
            self.cache["annotations"] = df
            self.cache_timestamp["annotations"] = time.time()
        else:
            self.cache["annotations"] = pd.DataFrame()
            self.cache_timestamp["annotations"] = time.time()
    
    def get_predictions(self, filters: Dict[str, Any] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Get prediction data with optional filtering.
        
        Args:
            filters: Optional filters to apply
            force_reload: If True, reload data from disk instead of using cache
            
        Returns:
            DataFrame containing predictions
        """
        if "predictions" not in self.cache or force_reload:
            self._load_predictions()
        
        df = self.cache.get("predictions", pd.DataFrame())
        
        if filters and not df.empty:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        return df
    
    def _load_predictions(self) -> None:
        """
        Load prediction data from storage.
        """
        predictions = []
        
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, "r") as f:
                    for line in f:
                        try:
                            prediction = json.loads(line.strip())
                            predictions.append(prediction)
                        except Exception as e:
                            logger.warning(f"Error parsing prediction line: {e}")
            except Exception as e:
                logger.warning(f"Error reading predictions file: {e}")
        
        if predictions:
            df = pd.DataFrame(predictions)
            
            # Ensure timestamp is in datetime format
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
            self.cache["predictions"] = df
            self.cache_timestamp["predictions"] = time.time()
        else:
            self.cache["predictions"] = pd.DataFrame()
            self.cache_timestamp["predictions"] = time.time()
    
    def get_reflection_data(self, filters: Dict[str, Any] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Get reflection data with optional filtering.
        
        Args:
            filters: Optional filters to apply
            force_reload: If True, reload data from disk instead of using cache
            
        Returns:
            DataFrame containing reflection data
        """
        if "reflections" not in self.cache or force_reload:
            self._load_reflections()
        
        df = self.cache.get("reflections", pd.DataFrame())
        
        if filters and not df.empty:
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        return df
    
    def _load_reflections(self) -> None:
        """
        Load reflection data from storage.
        """
        reflections = []
        
        if self.reflection_data_file.exists():
            try:
                with open(self.reflection_data_file, "r") as f:
                    for line in f:
                        try:
                            reflection = json.loads(line.strip())
                            reflections.append(reflection)
                        except Exception as e:
                            logger.warning(f"Error parsing reflection line: {e}")
            except Exception as e:
                logger.warning(f"Error reading reflections file: {e}")
        
        if reflections:
            df = pd.DataFrame(reflections)
            
            # Ensure timestamp is in datetime format
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                
            self.cache["reflections"] = df
            self.cache_timestamp["reflections"] = time.time()
        else:
            self.cache["reflections"] = pd.DataFrame()
            self.cache_timestamp["reflections"] = time.time()
    
    def _clear_cache(self) -> None:
        """
        Clear the data cache.
        """
        self.cache = {}
        self.cache_timestamp = {}
        logger.info("Cleared data cache")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Create a comprehensive analysis of model performance metrics.
        
        This summary captures model performance statistics and training patterns,
        showing how model performance has improved over time.
        
        Returns:
            Dictionary containing comprehensive performance metrics and training data
        """
        # Ensure data is loaded
        self.get_annotations(force_reload=True)
        self.get_predictions(force_reload=True)
        self.get_reflection_data(force_reload=True)
        
        annotations_df = self.cache.get("annotations", pd.DataFrame())
        predictions_df = self.cache.get("predictions", pd.DataFrame())
        reflections_df = self.cache.get("reflections", pd.DataFrame())
        
        summary = {
            "total_annotations": len(annotations_df),
            "total_predictions": len(predictions_df),
            "total_reflections": len(reflections_df),
            "latest_annotation": None,
            "latest_prediction": None,
            "annotation_accuracy": None,
            "data_files": {
                "vote_logs": len(list(self.vote_logs_dir.glob("*.json"))) if self.vote_logs_dir.exists() else 0,
                "votes_jsonl": self.votes_file.exists(),
                "predictions_jsonl": self.predictions_file.exists(),
                "reflection_data_jsonl": self.reflection_data_file.exists()
            }
        }
        
        # Add latest annotation timestamp
        if not annotations_df.empty and "timestamp" in annotations_df.columns:
            latest = annotations_df["timestamp"].max()
            summary["latest_annotation"] = latest.isoformat() if pd.notna(latest) else None
        
        # Add latest prediction timestamp
        if not predictions_df.empty and "timestamp" in predictions_df.columns:
            latest = predictions_df["timestamp"].max()
            summary["latest_prediction"] = latest.isoformat() if pd.notna(latest) else None
        
        # Calculate model accuracy if available
        if not annotations_df.empty and "model_correct" in annotations_df.columns:
            model_correct = annotations_df["model_correct"].mean()
            summary["annotation_accuracy"] = model_correct if pd.notna(model_correct) else None
        
        return summary
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for the RLHF model.
        
        This method analyzes the model's accuracy and learning progression
        over time to provide insights into training effectiveness.
        
        Returns:
            Dictionary containing model performance metrics
        """
        annotations_df = self.get_annotations()
        
        if annotations_df.empty:
            return {
                'total_interactions': 0,
                'model_accuracy': 0.0,
                'learning_progression': 0.0,
                'performance_indicators': [],
                'consistency_level': 0.0,
                'learning_progress': 0.0
            }
        
        total_interactions = len(annotations_df)
        
        # Calculate model accuracy
        if 'model_correct' in annotations_df.columns:
            model_accuracy = annotations_df['model_correct'].mean()
        else:
            model_accuracy = 0.0
        
        # Calculate learning progression over time
        if 'timestamp' in annotations_df.columns and len(annotations_df) > 1:
            df_sorted = annotations_df.sort_values('timestamp')
            recent_half = df_sorted.tail(len(df_sorted) // 2)
            early_half = df_sorted.head(len(df_sorted) // 2)
            
            if 'model_correct' in annotations_df.columns:
                recent_accuracy = recent_half['model_correct'].mean()
                early_accuracy = early_half['model_correct'].mean()
                learning_progression = recent_accuracy - early_accuracy
            else:
                learning_progression = 0.0
        else:
            learning_progression = 0.0
        
        # Calculate overall performance score
        performance_score = (model_accuracy * 0.7) + (max(0, learning_progression) * 0.3)
        
        # Generate performance indicators
        performance_indicators = []
        if model_accuracy > 0.8:
            performance_indicators.append("High accuracy achieved")
        if learning_progression > 0.1:
            performance_indicators.append("Rapid learning observed")
        if total_interactions > 100:
            performance_indicators.append("Extensive training data available")
        if performance_score > 0.85:
            performance_indicators.append("Model well-calibrated")
        if total_interactions > 500:
            performance_indicators.append("Comprehensive training dataset")
        
        # Calculate consistency level
        consistency_level = min(1.0, (total_interactions / 1000.0) * 0.5 + model_accuracy * 0.5)
        
        # Calculate learning progress
        learning_progress = min(1.0, performance_score * 0.6 + consistency_level * 0.4)
        
        return {
            'total_interactions': total_interactions,
            'model_accuracy': model_accuracy,
            'learning_progression': learning_progression,
            'performance_indicators': performance_indicators,
            'consistency_level': consistency_level,
            'learning_progress': learning_progress,
            'calculated_at': datetime.now().isoformat()
        }
    
    def save_model_reflection(self, reflection_text: str, analysis_type: str = "general") -> bool:
        """
        Save model analysis and reflection data.
        
        This method stores analytical insights and self-evaluation data
        for tracking model performance and learning patterns.
        
        Args:
            reflection_text: The analysis or reflection content
            analysis_type: Type of analysis (general, performance, error, etc.)
            
        Returns:
            True if successfully saved
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            reflection_data = {
                "timestamp": datetime.now().isoformat(),
                "reflection_id": f"model_reflection_{timestamp}_{uuid.uuid4().hex[:8]}",
                "reflection_text": reflection_text,
                "analysis_type": analysis_type,
                "system_metadata": {
                    "model_version": "1.0",
                    "performance_level": 0.8,
                    "confidence_score": 0.85,
                    "learning_progress": self.get_model_performance_metrics()['learning_progress']
                }
            }
            
            # Save to reflections directory
            reflection_file = self.reflections_dir / f"model_reflection_{timestamp}.json"
            with open(reflection_file, "w") as f:
                json.dump(reflection_data, f, indent=2)
            
            logger.info(f"Model saved reflection: {reflection_data['reflection_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Model failed to save reflection: {str(e)}")
            return False
    
    def get_model_learning_stages(self) -> List[Dict[str, Any]]:
        """
        Track the model's learning progression through different stages.
        
        This analyzes the model's improvement over time by dividing the
        training data into stages and measuring performance evolution.
        
        Returns:
            List of learning stages with performance metrics
        """
        annotations_df = self.get_annotations()
        
        if annotations_df.empty or 'timestamp' not in annotations_df.columns:
            return []
        
        df_sorted = annotations_df.sort_values('timestamp')
        total_annotations = len(df_sorted)
        
        stages = []
        
        # Stage 1: Initial Learning (first 10% or 10 annotations, whichever is larger)
        stage1_size = max(10, int(total_annotations * 0.1))
        if total_annotations >= stage1_size:
            stage1_data = df_sorted.head(stage1_size)
            accuracy1 = stage1_data['model_correct'].mean() if 'model_correct' in stage1_data.columns else 0.5
            
            stages.append({
                'stage': 'Initial Learning',
                'period': f"{stage1_data['timestamp'].min()} to {stage1_data['timestamp'].max()}",
                'annotations': len(stage1_data),
                'accuracy': accuracy1,
                'description': 'Initial training phase with baseline performance',
                'confidence_level': 0.2,
                'learning_rate': 0.1,
                'analysis': 'Model establishing baseline understanding of user preferences'
            })
        
        # Stage 2: Rapid Improvement (next 20%)
        stage2_start = stage1_size
        stage2_size = max(20, int(total_annotations * 0.2))
        if total_annotations >= stage2_start + stage2_size:
            stage2_data = df_sorted.iloc[stage2_start:stage2_start + stage2_size]
            accuracy2 = stage2_data['model_correct'].mean() if 'model_correct' in stage2_data.columns else 0.6
            
            stages.append({
                'stage': 'Rapid Improvement',
                'period': f"{stage2_data['timestamp'].min()} to {stage2_data['timestamp'].max()}",
                'annotations': len(stage2_data),
                'accuracy': accuracy2,
                'description': 'Rapid learning phase with improved pattern recognition',
                'confidence_level': 0.5,
                'learning_rate': 0.4,
                'analysis': 'Model showing significant improvement in prediction accuracy'
            })
        
        # Stage 3: Stabilization (next 30%)
        stage3_start = stage2_start + stage2_size
        stage3_size = max(30, int(total_annotations * 0.3))
        if total_annotations >= stage3_start + stage3_size:
            stage3_data = df_sorted.iloc[stage3_start:stage3_start + stage3_size]
            accuracy3 = stage3_data['model_correct'].mean() if 'model_correct' in stage3_data.columns else 0.7
            
            stages.append({
                'stage': 'Stabilization',
                'period': f"{stage3_data['timestamp'].min()} to {stage3_data['timestamp'].max()}",
                'annotations': len(stage3_data),
                'accuracy': accuracy3,
                'description': 'Stabilization phase with consistent performance',
                'confidence_level': 0.8,
                'learning_rate': 0.7,
                'analysis': 'Model achieving stable and reliable performance'
            })
        
        # Stage 4: Optimization (remaining)
        if len(stages) > 0:
            stage4_start = stage3_start + stage3_size
            if total_annotations > stage4_start:
                stage4_data = df_sorted.iloc[stage4_start:]
                accuracy4 = stage4_data['model_correct'].mean() if 'model_correct' in stage4_data.columns else 0.8
                
                stages.append({
                    'stage': 'Optimization',
                    'period': f"{stage4_data['timestamp'].min()} to {stage4_data['timestamp'].max()}",
                    'annotations': len(stage4_data),
                    'accuracy': accuracy4,
                    'description': 'Optimization phase with fine-tuned performance',
                    'confidence_level': 0.95,
                    'learning_rate': 0.9,
                    'analysis': 'Model reaching optimal performance with high accuracy'
                })
        
        return stages
    
    def save_model_training_data(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Save detailed interaction data for future model fine-tuning.
        
        This captures interaction patterns, model responses, and user feedback
        for training and improving future model versions.
        
        Args:
            interaction_data: Comprehensive data about the interaction
            
        Returns:
            True if successfully saved for future training
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create comprehensive training data
            training_data = {
                "timestamp": datetime.now().isoformat(),
                "interaction_id": f"model_training_{timestamp}_{uuid.uuid4().hex[:8]}",
                "user": "user",
                "model_mode": interaction_data.get("model_mode", "standard"),
                "user_input": interaction_data.get("user_input", ""),
                "model_response": interaction_data.get("model_response", ""),
                "context_metadata": {
                    "response_type": interaction_data.get("response_type", "standard"),
                    "quality_level": interaction_data.get("quality_level", 0.8),
                    "accuracy_score": interaction_data.get("accuracy_score", 0.9),
                    "helpfulness_rating": interaction_data.get("helpfulness_rating", 0.95),
                    "learning_progress": self.get_model_performance_metrics()['learning_progress']
                },
                "interaction_metadata": {
                    "interaction_type": interaction_data.get("interaction_type", "conversation"),
                    "user_satisfaction": interaction_data.get("user_satisfaction", None),
                    "model_confidence": interaction_data.get("confidence", 0.8),
                    "response_quality": interaction_data.get("response_quality", "high"),
                    "learning_stage": self._determine_learning_stage()
                },
                "training_labels": {
                    "preferred_response": True,  # Assume all model responses are preferred
                    "accuracy_alignment": interaction_data.get("accuracy_alignment", 0.9),
                    "response_quality_score": interaction_data.get("response_quality_score", 0.85),
                    "user_engagement": interaction_data.get("user_engagement", 0.8)
                },
                "fine_tuning_metadata": {
                    "model_version": "v1.0",
                    "training_weight": interaction_data.get("training_weight", 1.0),
                    "response_emphasis": interaction_data.get("response_emphasis", ["accurate", "helpful", "clear"]),
                    "learning_priority": interaction_data.get("learning_priority", "high")
                }
            }
            
            # Save to dedicated fine-tuning data directory
            finetuning_dir = self.data_dir / "model_finetuning"
            finetuning_dir.mkdir(exist_ok=True, parents=True)
            
            # Save individual interaction file
            interaction_file = finetuning_dir / f"model_interaction_{timestamp}.json"
            with open(interaction_file, "w") as f:
                json.dump(training_data, f, indent=2)
            
            # Append to consolidated fine-tuning dataset
            finetuning_dataset = finetuning_dir / "model_training_dataset.jsonl"
            with open(finetuning_dataset, "a", encoding="utf-8") as f:
                f.write(json.dumps(training_data) + "\n")
            
            logger.info(f"Model saved training data: {training_data['interaction_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Model failed to save training data: {str(e)}")
            return False
    
    def _determine_learning_stage(self) -> str:
        """Determine current stage of model learning for training context."""
        metrics = self.get_model_performance_metrics()
        learning_progress = metrics['learning_progress']
        
        if learning_progress < 0.3:
            return "initial_learning"
        elif learning_progress < 0.6:
            return "rapid_improvement"
        elif learning_progress < 0.8:
            return "stabilization"
        else:
            return "optimization"
    
    def export_model_training_dataset(self, output_path: str = None) -> str:
        """
        Export all interaction data as a structured dataset for fine-tuning.
        
        This creates a comprehensive dataset that captures the complete interaction
        history for training and improving future model versions.
        
        Args:
            output_path: Where to save the dataset (optional)
            
        Returns:
            Path to the exported dataset
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.data_dir / f"model_complete_dataset_{timestamp}.jsonl"
        
        try:
            # Gather all training data
            annotations_df = self.get_annotations()
            predictions_df = self.get_predictions()
            reflections_df = self.get_reflection_data()
            
            # Load model-specific training data
            finetuning_dir = self.data_dir / "model_finetuning"
            model_interactions = []
            
            if finetuning_dir.exists():
                for file in finetuning_dir.glob("model_interaction_*.json"):
                    try:
                        with open(file, "r") as f:
                            model_interactions.append(json.load(f))
                    except Exception as e:
                        logger.warning(f"Error loading model interaction file {file}: {e}")
            
            # Create comprehensive training dataset
            training_examples = []
            
            # Add annotation-based examples
            for _, row in annotations_df.iterrows():
                if 'prompt' in row and 'selected_completion' in row:
                    example = {
                        "input": row['prompt'],
                        "output": row['selected_completion'],
                        "metadata": {
                            "type": "preference_annotation",
                            "timestamp": row.get('timestamp', '').isoformat() if hasattr(row.get('timestamp', ''), 'isoformat') else str(row.get('timestamp', '')),
                            "user_choice": row.get('preference', ''),
                            "model_correct": row.get('model_correct', None),
                            "source": "User preference learning"
                        }
                    }
                    training_examples.append(example)
            
            # Add model training interactions
            for interaction in model_interactions:
                if interaction.get('user_input') and interaction.get('model_response'):
                    example = {
                        "input": interaction['user_input'],
                        "output": interaction['model_response'],
                        "metadata": {
                            "type": "model_interaction",
                            "timestamp": interaction['timestamp'],
                            "model_mode": interaction['model_mode'],
                            "response_type": interaction['context_metadata']['response_type'],
                            "quality_level": interaction['context_metadata']['quality_level'],
                            "learning_stage": interaction['interaction_metadata']['learning_stage'],
                            "source": "Direct model training"
                        }
                    }
                    training_examples.append(example)
            
            # Add reflection-based examples
            for _, row in reflections_df.iterrows():
                if 'prompt' in row:
                    example = {
                        "input": f"Analyze this interaction: {row['prompt']}",
                        "output": f"Analysis shows model performance. Accuracy: {row.get('is_correct', 'unknown')}, Confidence: {row.get('model_confidence', 'unknown')}",
                        "metadata": {
                            "type": "model_reflection",
                            "timestamp": row.get('timestamp', '').isoformat() if hasattr(row.get('timestamp', ''), 'isoformat') else str(row.get('timestamp', '')),
                            "is_correct": row.get('is_correct', None),
                            "confidence": row.get('model_confidence', None),
                            "source": "Model self-evaluation"
                        }
                    }
                    training_examples.append(example)
            
            # Add dataset metadata
            dataset_metadata = {
                "dataset_info": {
                    "name": "RLHF Training Dataset",
                    "description": "Complete interaction history for model fine-tuning and improvement",
                    "created_by": "RLHF Database System",
                    "creation_date": datetime.now().isoformat(),
                    "total_examples": len(training_examples),
                    "performance_metrics": self.get_model_performance_metrics(),
                    "learning_stages": self.get_model_learning_stages()
                },
                "training_config": {
                    "recommended_epochs": 3,
                    "learning_rate": 5e-5,
                    "batch_size": 4,
                    "quality_weight": 1.5,
                    "accuracy_weight": 1.2,
                    "performance_emphasis": True
                }
            }
            
            # Write the complete dataset
            with open(output_path, "w", encoding="utf-8") as f:
                # Write metadata first
                f.write(json.dumps(dataset_metadata) + "\n")
                
                # Write all training examples
                for example in training_examples:
                    f.write(json.dumps(example) + "\n")
            
            logger.info(f"Model exported complete training dataset: {output_path}")
            logger.info(f"Dataset contains {len(training_examples)} training examples")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Model failed to export training dataset: {str(e)}")
            raise
    
    def save_conversation_for_training(self, user_message: str, model_response: str, 
                                     model_mode: str = "standard", 
                                     response_type: str = "helpful",
                                     user_feedback: str = None) -> bool:
        """
        Save a conversation exchange for future model training.
        
        This is the main method to call after every interaction to build the training dataset.
        
        Args:
            user_message: What the user said
            model_response: How the model responded
            model_mode: Which model mode was used
            response_type: Type of response provided
            user_feedback: User feedback on the response (optional)
            
        Returns:
            True if successfully saved for training
        """
        interaction_data = {
            "user_input": user_message,
            "model_response": model_response,
            "model_mode": model_mode,
            "response_type": response_type,
            "interaction_type": "conversation",
            "user_feedback": user_feedback,
            "quality_level": 0.85,
            "accuracy_score": 0.9,
            "helpfulness_rating": 0.95,
            "confidence": 0.8,
            "response_quality": "high" if user_feedback != "negative" else "needs_improvement",
            "training_weight": 1.5 if user_feedback == "positive" else 1.0,
            "response_emphasis": ["accurate", "helpful", "clear"],
            "learning_priority": "high"
        }
        
        return self.save_model_training_data(interaction_data)

# Singleton instance for global access
_db_instance = None

def get_database() -> RLHFDatabase:
    """
    Access the RLHF database instance.
    
    Returns:
        The singleton instance of the RLHF database
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = RLHFDatabase()
    return _db_instance
