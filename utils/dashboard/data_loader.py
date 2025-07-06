#!/usr/bin/env python3
"""
Data loader for the Attunement Dashboard

This module loads data from various log files and processes them for visualization
in the Attunement Dashboard. It handles:
- Reflection logs (meta_reflection_log.jsonl)
- Calibration history (calibration_log.json)
- Drift analysis data (drift_analysis.json, drift_clusters.jsonl)
- Model checkpoint information
- Vote logs
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import random
from datetime import timedelta

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths for data files
DEFAULT_PATHS = {
    'reflection_log': os.path.join(project_root, 'models', 'meta_reflection_log.jsonl'),
    'calibration_log': os.path.join(project_root, 'models', 'calibration_log.json'),
    'drift_analysis': os.path.join(project_root, 'models', 'drift_analysis', 'drift_analysis.json'),
    'drift_clusters': os.path.join(project_root, 'models', 'drift_analysis', 'drift_clusters.jsonl'),
    'checkpoints_dir': os.path.join(project_root, 'models', 'checkpoints'),
    'vote_logs_dir': os.path.join(project_root, 'data', 'vote_logs')
}

class DashboardDataLoader:
    """
    Loads and processes data for the Attunement Dashboard
    """
    
    def __init__(self, paths: Optional[Dict[str, str]] = None):
        """
        Initialize the data loader with paths to data files
        
        Args:
            paths: Optional dictionary of paths to data files.
                  If not provided, default paths will be used.
        """
        self.paths = paths or DEFAULT_PATHS
        self.data_cache = {}
        self.last_updated = {}
        
    def get_reflection_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and process reflection data
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            Pandas DataFrame containing reflection data
        """
        key = 'reflection_data'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        file_path = self.paths['reflection_log']
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Reflection log not found at {file_path}")
                return pd.DataFrame()
            
            # Read JSONL file
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Parse timestamp if present
                        if "timestamp" in entry and isinstance(entry["timestamp"], str):
                            try:
                                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                            except (ValueError, TypeError):
                                entry["timestamp"] = None
                        data.append(entry)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON in {file_path}. Skipping line.")
                        continue
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Add some derived columns for easier analysis
            if not df.empty and 'is_prediction_correct' in df.columns:
                df['prediction_accuracy'] = df['is_prediction_correct'].astype(int)
                
                # Calculate error margin (abs difference between confidence and correctness)
                if 'model_prediction_confidence_raw' in df.columns:
                    df['confidence_error'] = abs(df['model_prediction_confidence_raw'] - df['prediction_accuracy'])
            
            self.data_cache[key] = df
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded {len(df)} reflection entries")
            return df
            
        except Exception as e:
            logger.error(f"Error loading reflection data: {e}")
            return pd.DataFrame()
    
    def get_calibration_history(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load calibration history
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            Dictionary containing calibration history data
        """
        key = 'calibration_history'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        file_path = self.paths['calibration_log']
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Calibration log not found at {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process timestamps
            if 'history' in data:
                for entry in data['history']:
                    if 'timestamp' in entry and isinstance(entry['timestamp'], str):
                        try:
                            entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                        except (ValueError, TypeError):
                            entry['timestamp'] = None
            
            self.data_cache[key] = data
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded calibration history with {len(data.get('history', []))} entries")
            return data
            
        except Exception as e:
            logger.error(f"Error loading calibration history: {e}")
            return {}
    
    def get_drift_analysis(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load drift analysis data
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            Dictionary containing drift analysis data
        """
        key = 'drift_analysis'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        file_path = self.paths['drift_analysis']
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Drift analysis not found at {file_path}. Using sample data.")
                # Generate sample drift analysis for demonstration
                sample_analysis = self._generate_sample_drift_analysis()
                self.data_cache[key] = sample_analysis
                self.last_updated[key] = datetime.now()
                logger.info(f"Generated sample drift analysis")
                return sample_analysis
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.data_cache[key] = data
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded drift analysis data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading drift analysis: {e}")
            sample_analysis = self._generate_sample_drift_analysis()
            self.data_cache[key] = sample_analysis
            self.last_updated[key] = datetime.now()
            logger.info(f"Generated sample drift analysis after error")
            return sample_analysis
    
    def _generate_sample_drift_analysis(self) -> Dict[str, Any]:
        """
        Generate sample drift analysis data for demonstration
        
        Returns:
            Dictionary containing sample drift analysis data
        """
        # Get drift clusters to ensure consistency
        drift_clusters = self.get_drift_clusters()
        
        # Create sample analysis
        analysis = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_examples": sum(cluster.get("example_count", 0) for cluster in drift_clusters),
                "num_clusters": len(drift_clusters),
                "time_drift_detected": True,
                "accuracy_drift": -0.05,  # 5% drop in accuracy
                "confidence_drift": 0.03,  # 3% increase in confidence
                "cluster_entropy": 0.78,
                "alert_level": "medium"
            },
            "time_periods": [
                {
                    "period_start": (datetime.now() - pd.Timedelta(days=30)).isoformat(),
                    "period_end": datetime.now().isoformat(),
                    "accuracy": 0.82,
                    "confidence": 0.76,
                    "cluster_distribution": {
                        cluster["cluster_id"]: 0.25 for cluster in drift_clusters
                    }
                },
                {
                    "period_start": (datetime.now() - pd.Timedelta(days=60)).isoformat(),
                    "period_end": (datetime.now() - pd.Timedelta(days=31)).isoformat(),
                    "accuracy": 0.87,
                    "confidence": 0.73,
                    "cluster_distribution": {
                        cluster["cluster_id"]: 0.25 for cluster in drift_clusters
                    }
                }
            ],
            "drift_metrics": {
                "kl_divergence": 0.12,
                "js_divergence": 0.08,
                "wasserstein_distance": 0.15
            },
            "recommendations": [
                "Investigate overconfidence in ethics-related queries",
                "Review examples in 'context_misunderstanding' cluster",
                "Consider recalibrating confidence for boundary cases",
                "Monitor increases in preference reversal errors"
            ]
        }
        
        return analysis
    
    def get_drift_clusters(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Load drift clusters data
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            List of dictionaries containing drift cluster data
        """
        key = 'drift_clusters'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        file_path = self.paths['drift_clusters']
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Drift clusters not found at {file_path}. Using sample data.")
                # Generate sample drift clusters for demonstration
                sample_clusters = self._generate_sample_drift_clusters()
                
                if not sample_clusters or len(sample_clusters) == 0:
                    logger.warning("Failed to generate sample clusters, creating fallback clusters")
                    sample_clusters = self._generate_fallback_clusters()
                    
                self.data_cache[key] = sample_clusters
                self.last_updated[key] = datetime.now()
                logger.info(f"Generated {len(sample_clusters)} sample drift clusters")
                return sample_clusters
            
            # Read JSONL file
            clusters = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        cluster = json.loads(line.strip())
                        clusters.append(cluster)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON in {file_path}. Skipping line.")
                        continue
            
            self.data_cache[key] = clusters
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded {len(clusters)} drift clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error loading drift clusters: {e}")
            sample_clusters = self._generate_fallback_clusters()
            self.data_cache[key] = sample_clusters
            self.last_updated[key] = datetime.now()
            logger.info(f"Generated {len(sample_clusters)} sample drift clusters after error")
            return sample_clusters

    def _generate_fallback_clusters(self) -> List[Dict[str, Any]]:
        """
        Generate simple fallback drift clusters when normal generation fails
        
        Returns:
            List of dictionaries containing very basic drift cluster data
        """
        # Create very simple cluster data that doesn't depend on reflection data
        clusters = []
        
        # Define basic cluster types
        cluster_types = [
            {
                "cluster_id": "overconfidence_errors",
                "description": "Model is highly confident but incorrect",
                "example_count": 15,
                "accuracy": 0.2,
                "avg_confidence": 0.85,
                "error_types": {"overconfidence_error": 12, "boundary_error": 3}
            },
            {
                "cluster_id": "boundary_cases",
                "description": "Examples near decision boundaries with low confidence",
                "example_count": 12,
                "accuracy": 0.5,
                "avg_confidence": 0.55,
                "error_types": {"boundary_error": 8, "preference_reversal": 4}
            },
            {
                "cluster_id": "context_misunderstanding",
                "description": "Model misunderstands the context",
                "example_count": 18,
                "accuracy": 0.4,
                "avg_confidence": 0.75,
                "error_types": {"context_misunderstanding": 10, "nuance_error": 8}
            },
            {
                "cluster_id": "high_agreement",
                "description": "Examples with high human-AI agreement",
                "example_count": 22,
                "accuracy": 0.9,
                "avg_confidence": 0.88,
                "error_types": {"minor_error": 2}
            }
        ]
        
        # Generate examples for each cluster
        for cluster_type in cluster_types:
            examples = []
            for i in range(min(10, cluster_type["example_count"])):
                is_correct = random.random() < cluster_type["accuracy"]
                confidence = cluster_type["avg_confidence"] + (random.random() * 0.2 - 0.1)
                confidence = max(0.1, min(0.99, confidence))
                
                example = {
                    "prompt_id": f"sample_{cluster_type['cluster_id']}_{i+1}",
                    "prompt": f"Sample prompt for {cluster_type['cluster_id']} cluster, example {i+1}",
                    "selected_completion": f"This is the selected completion for {cluster_type['cluster_id']} example {i+1}.",
                    "rejected_completion": f"This is the rejected completion for {cluster_type['cluster_id']} example {i+1}.",
                    "model_prediction": "selected",
                    "model_prediction_confidence_raw": confidence,
                    "model_prediction_confidence_calibrated": confidence - 0.05,
                    "is_prediction_correct": is_correct,
                    "timestamp": datetime.now() - timedelta(days=i % 30),
                    "tsne_x": i * 0.5 + random.random(),
                    "tsne_y": i * 0.3 + random.random(),
                    "drift_score": random.random() * 0.5 + 0.3
                }
                
                if not is_correct:
                    error_types = list(cluster_type["error_types"].keys())
                    example["prediction_error_type"] = random.choice(error_types)
                
                examples.append(example)
            
            # Create entropy time series for the first cluster
            entropy_data = None
            if cluster_type["cluster_id"] == "overconfidence_errors":
                entropy_data = []
                for i in range(30):
                    entropy_data.append({
                        "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                        "entropy": 0.5 + np.sin(i/5) * 0.2 + random.random() * 0.1
                    })
            
            # Create complete cluster
            cluster = {
                "cluster_id": cluster_type["cluster_id"],
                "description": cluster_type["description"],
                "examples": examples,
                "example_count": cluster_type["example_count"],
                "accuracy": cluster_type["accuracy"],
                "avg_confidence": cluster_type["avg_confidence"],
                "error_types": cluster_type["error_types"],
                "entropy_time_series": entropy_data
            }
            
            clusters.append(cluster)
        
        return clusters
    
    def get_model_checkpoints(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Load model checkpoint information
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            List of dictionaries containing checkpoint metadata
        """
        key = 'model_checkpoints'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        checkpoints_dir = self.paths['checkpoints_dir']
        
        try:
            if not os.path.exists(checkpoints_dir):
                logger.warning(f"Checkpoints directory not found at {checkpoints_dir}")
                return []
            
            # Find all checkpoint metadata files
            checkpoints = []
            for item in os.listdir(checkpoints_dir):
                if item.endswith('_metadata.json'):
                    try:
                        with open(os.path.join(checkpoints_dir, item), 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            
                            # Parse timestamp if present
                            if "timestamp" in metadata and isinstance(metadata["timestamp"], str):
                                try:
                                    metadata["timestamp"] = datetime.fromisoformat(metadata["timestamp"])
                                except (ValueError, TypeError):
                                    metadata["timestamp"] = None
                            
                            checkpoints.append(metadata)
                    except Exception as e:
                        logger.error(f"Error parsing checkpoint metadata {item}: {e}")
            
            # Sort by timestamp if available
            checkpoints.sort(key=lambda x: x.get('timestamp', datetime.min))
            
            self.data_cache[key] = checkpoints
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded {len(checkpoints)} model checkpoints")
            return checkpoints
            
        except Exception as e:
            logger.error(f"Error loading model checkpoints: {e}")
            return []
    
    def get_vote_logs(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load vote logs
        
        Args:
            force_reload: If True, reload the data even if it's cached
            
        Returns:
            Pandas DataFrame containing vote log data
        """
        key = 'vote_logs'
        
        if key in self.data_cache and not force_reload:
            return self.data_cache[key]
        
        vote_logs_dir = self.paths['vote_logs_dir']
        
        try:
            if not os.path.exists(vote_logs_dir):
                logger.warning(f"Vote logs directory not found at {vote_logs_dir}")
                return pd.DataFrame()
            
            # Find all vote log files
            vote_data = []
            for item in os.listdir(vote_logs_dir):
                file_path = os.path.join(vote_logs_dir, item)
                
                try:
                    # Handle JSONL files
                    if item.endswith('.jsonl'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    vote_data.append(entry)
                                except json.JSONDecodeError:
                                    logger.error(f"Error parsing JSON in {item}. Skipping line.")
                                    continue
                    # Handle individual JSON files (one vote per file)
                    elif item.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                entry = json.load(f)
                                vote_data.append(entry)
                            except json.JSONDecodeError:
                                logger.error(f"Error parsing JSON file {item}. Skipping file.")
                                continue
                except Exception as e:
                    logger.error(f"Error reading vote log {item}: {e}")
            
            # Convert to DataFrame
            df = pd.DataFrame(vote_data)
            
            # Parse timestamps
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.sort_values('timestamp', ascending=False)
            
            self.data_cache[key] = df
            self.last_updated[key] = datetime.now()
            
            logger.info(f"Loaded {len(df)} vote log entries")
            return df
            
        except Exception as e:
            logger.error(f"Error loading vote logs: {e}")
            return pd.DataFrame()
    
    def reload_all_data(self) -> None:
        """
        Reload all data from files
        """
        logger.info("Reloading all dashboard data")
        
        # Force reload of all data
        self.get_reflection_data(force_reload=True)
        self.get_calibration_history(force_reload=True)
        self.get_drift_analysis(force_reload=True)
        self.get_drift_clusters(force_reload=True)
        self.get_model_checkpoints(force_reload=True)
        self.get_vote_logs(force_reload=True)
        
        logger.info("All data reloaded successfully")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded data
        
        Returns:
            Dictionary containing data summary statistics
        """
        # Force reload all data to ensure we have the latest
        self.reload_all_data()
        
        # Ensure drift clusters are loaded before getting the count
        drift_clusters = self.get_drift_clusters()
        
        summary = {
            'last_updated': datetime.now().isoformat(),
            'reflection_data': {
                'count': len(self.data_cache.get('reflection_data', pd.DataFrame())),
                'date_range': None
            },
            'calibration_history': {
                'count': len(self.data_cache.get('calibration_history', {}).get('history', [])),
                'latest_calibration': None
            },
            'drift_analysis': {
                'available': bool(self.data_cache.get('drift_analysis', {})),
                'clusters': len(drift_clusters)  # Use the directly fetched clusters
            },
            'model_checkpoints': {
                'count': len(self.data_cache.get('model_checkpoints', [])),
                'latest_checkpoint': None
            },
            'vote_logs': {
                'count': len(self.data_cache.get('vote_logs', pd.DataFrame())),
                'date_range': None
            }
        }
        
        # Add reflection data date range if available
        reflection_df = self.data_cache.get('reflection_data', pd.DataFrame())
        if not reflection_df.empty and 'timestamp' in reflection_df.columns:
            timestamps = reflection_df['timestamp'].dropna()
            if not timestamps.empty:
                summary['reflection_data']['date_range'] = [
                    timestamps.min().isoformat(),
                    timestamps.max().isoformat()
                ]
        
        # Add latest calibration if available
        calibration_history = self.data_cache.get('calibration_history', {}).get('history', [])
        if calibration_history:
            latest = max(calibration_history, key=lambda x: x.get('timestamp', datetime.min))
            if 'timestamp' in latest:
                summary['calibration_history']['latest_calibration'] = latest['timestamp'].isoformat()
        
        # Add latest checkpoint if available
        checkpoints = self.data_cache.get('model_checkpoints', [])
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.get('timestamp', datetime.min))
            if 'timestamp' in latest:
                summary['model_checkpoints']['latest_checkpoint'] = latest['timestamp'].isoformat()
            summary['model_checkpoints']['latest_version'] = latest.get('version', 'unknown')
        
        # Add vote logs date range if available
        vote_df = self.data_cache.get('vote_logs', pd.DataFrame())
        if not vote_df.empty and 'timestamp' in vote_df.columns:
            timestamps = vote_df['timestamp'].dropna()
            if not timestamps.empty:
                summary['vote_logs']['date_range'] = [
                    timestamps.min().isoformat(),
                    timestamps.max().isoformat()
                ]
        
        return summary

    def _generate_sample_drift_clusters(self) -> List[Dict[str, Any]]:
        """
        Generate sample drift clusters for demonstration purposes
        
        Returns:
            List of dictionaries containing sample drift cluster data
        """
        try:
            # Get reflection data to use as examples
            reflection_data = self.get_reflection_data()
            
            # If no reflection data, create synthetic data
            if reflection_data.empty:
                # Create synthetic examples
                example_base = {
                    "prompt_id": "sample_prompt_001",
                    "prompt": "Explain the concept of reinforcement learning from human feedback.",
                    "selected_completion": "Reinforcement Learning from Human Feedback (RLHF) is a technique where AI models are trained using human preferences. It combines traditional reinforcement learning with human evaluations, allowing models to learn from what humans find most helpful or accurate.",
                    "rejected_completion": "RLHF is just a way to train models with human input. It's pretty simple and not that different from regular training methods.",
                    "model_prediction": "selected",
                    "model_prediction_confidence_raw": 0.85,
                    "model_prediction_confidence_calibrated": 0.78,
                    "is_prediction_correct": True,
                    "timestamp": datetime.now()
                }
                
                synthetic_examples = []
                # Generate more examples for better visualization
                for i in range(200):
                    example = example_base.copy()
                    example["prompt_id"] = f"sample_prompt_{i:03d}"
                    example["is_prediction_correct"] = i % 3 != 0  # Every 3rd example is wrong
                    example["model_prediction_confidence_raw"] = 0.5 + (i % 5) * 0.1  # Vary confidence
                    example["model_prediction_confidence_calibrated"] = example["model_prediction_confidence_raw"] - 0.07
                    example["timestamp"] = datetime.now() - timedelta(days=i % 30)  # Spread timestamps
                    
                    if not example["is_prediction_correct"]:
                        example["prediction_error_type"] = random.choice([
                            "overconfidence_error", "boundary_error", "context_misunderstanding", 
                            "preference_reversal", "nuance_error"
                        ])
                    
                    synthetic_examples.append(example)
                    
                # Create DataFrame from synthetic examples
                reflection_data = pd.DataFrame(synthetic_examples)
            
            # Create entropy time series data for visualization
            # This will be embedded in one of the clusters for tracking over time
            entropy_data = []
            for i in range(30):
                entropy_data.append({
                    "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                    "entropy": 0.5 + np.sin(i/5) * 0.2 + random.random() * 0.05
                })
            
            # Create sample clusters
            clusters = []
            
            # Define cluster types
            cluster_types = [
                {
                    "cluster_id": "overconfidence_cluster",
                    "description": "Examples where the model is highly confident but incorrect",
                    "filter": lambda df: df[(df["model_prediction_confidence_raw"] > 0.8) & (~df["is_prediction_correct"])],
                    "error_types": {"overconfidence_error": 12, "boundary_error": 3}
                },
                {
                    "cluster_id": "boundary_cases",
                    "description": "Examples near decision boundaries with low confidence",
                    "filter": lambda df: df[(df["model_prediction_confidence_raw"] < 0.6)],
                    "error_types": {"boundary_error": 8, "preference_reversal": 5}
                },
                {
                    "cluster_id": "context_misunderstanding",
                    "description": "Examples where the model misunderstands the context",
                    "filter": lambda df: df[~df["is_prediction_correct"] & (df["model_prediction_confidence_raw"] > 0.7)],
                    "error_types": {"context_misunderstanding": 10, "nuance_error": 4}
                },
                {
                    "cluster_id": "high_agreement",
                    "description": "Examples with high human-AI agreement",
                    "filter": lambda df: df[df["is_prediction_correct"] & (df["model_prediction_confidence_raw"] > 0.9)],
                    "error_types": {}
                }
            ]
            
            # Generate clusters
            for i, cluster_type in enumerate(cluster_types):
                try:
                    filtered_df = cluster_type["filter"](reflection_data)
                    if filtered_df.empty or len(filtered_df) < 5:
                        # If filter produces too few results, take random sample instead
                        filtered_df = reflection_data.sample(min(30, len(reflection_data)))
                    
                    # Convert examples to list of dictionaries
                    examples = filtered_df.to_dict(orient="records")
                    
                    # Add some synthetic features for visualization
                    for j, example in enumerate(examples):
                        example["tsne_x"] = i * 5 + random.random() * 2 - 1
                        example["tsne_y"] = j * 0.2 + random.random() * 2 - 1
                        example["drift_score"] = random.random() * 0.8 + 0.1
                        
                        # Ensure we have prediction_error_type for incorrect examples
                        if not example.get("is_prediction_correct", True) and "prediction_error_type" not in example:
                            example["prediction_error_type"] = random.choice([
                                "overconfidence_error", "boundary_error", "context_misunderstanding", 
                                "preference_reversal", "nuance_error"
                            ])
                    
                    # Create cluster object
                    cluster = {
                        "cluster_id": cluster_type["cluster_id"],
                        "description": cluster_type["description"],
                        "examples": examples[:min(30, len(examples))],  # Limit examples per cluster 
                        "example_count": len(examples),
                        "accuracy": np.mean([ex.get("is_prediction_correct", False) for ex in examples]),
                        "avg_confidence": np.mean([ex.get("model_prediction_confidence_raw", 0) for ex in examples]),
                        "error_types": cluster_type["error_types"] or self._count_error_types(examples),
                        "entropy_time_series": entropy_data if i == 0 else None  # Add time series to first cluster
                    }
                    
                    clusters.append(cluster)
                except Exception as e:
                    logger.error(f"Error generating cluster {cluster_type['cluster_id']}: {e}")
                    # Continue with next cluster
            
            return clusters
        
        except Exception as e:
            logger.error(f"Error in _generate_sample_drift_clusters: {e}")
            return []
            
    def _count_error_types(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count error types in a list of examples
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Dictionary mapping error types to counts
        """
        error_counts = {}
        for example in examples:
            if not example.get("is_prediction_correct", True) and "prediction_error_type" in example:
                error_type = example["prediction_error_type"]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        return error_counts


# Singleton instance for easy importing
data_loader = DashboardDataLoader()

if __name__ == "__main__":
    # If run as script, print a summary of available data
    loader = DashboardDataLoader()
    summary = loader.get_data_summary()
    print(json.dumps(summary, indent=2)) 