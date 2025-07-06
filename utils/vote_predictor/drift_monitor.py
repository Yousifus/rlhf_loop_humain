#!/usr/bin/env python3
"""
Drift Monitoring Module

This module implements drift detection and monitoring for the vote predictor model in the RLHF system.
It analyzes historical prediction data to detect shifts in model performance or underlying data patterns.

Key features:
1. Time-based drift analysis: Detects changes in model accuracy over time
2. Semantic clustering: Identifies clusters of similar examples where the model performs differently
3. Confidence calibration drift: Monitors changes in the model's calibration quality
4. Feature drift detection: Tracks changes in the distribution of input features

Input: meta_reflection_log.jsonl (log of model predictions and human preferences)
Output: 
  - drift_analysis.json (summary of drift detection results)
  - drift_clusters.jsonl (examples grouped by identified drift patterns)
  - drift_visualizations/ (optional directory with drift visualization files)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "drift_monitor.log"))
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DriftAnalysisConfig:
    """Configuration parameters for drift analysis."""
    time_window_days: int = 7  # Size of time window for temporal analysis
    min_examples_per_window: int = 5  # Minimum examples required for valid window (reduced from 20)
    min_samples_required: int = 10  # Minimum samples required for full drift analysis
    n_clusters: int = 5  # Number of clusters for KMeans
    dbscan_eps: float = 0.5  # DBSCAN epsilon parameter
    dbscan_min_samples: int = 5  # DBSCAN min_samples parameter
    confidence_drift_threshold: float = 0.1  # Threshold for confidence calibration drift
    embedding_dimensions: int = 32  # Dimensions for TSNE visualization
    random_seed: int = 42  # Random seed for reproducibility
    alert_accuracy_change_threshold: float = 0.1  # Alert threshold for accuracy change

@dataclass
class DriftCluster:
    """Represents a cluster of examples with similar drift characteristics."""
    cluster_id: str
    description: str
    examples: List[Dict[str, Any]]
    centroid: Optional[List[float]] = None
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    feature_distribution: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "cluster_id": self.cluster_id,
            "description": self.description,
            "centroid": self.centroid,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "error_types": self.error_types,
            "feature_distribution": self.feature_distribution,
            "example_count": len(self.examples)
        }

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
                    # Parse timestamp if present
                    if "timestamp" in entry and isinstance(entry["timestamp"], str):
                        try:
                            entry["timestamp_parsed"] = datetime.fromisoformat(entry["timestamp"])
                        except (ValueError, TypeError):
                            entry["timestamp_parsed"] = None
                    reflections.append(entry)
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON at line {line_num}. Skipping.")
    except Exception as e:
        logger.error(f"Error loading reflection log: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(reflections)} reflection entries from {file_path}")
    return reflections

def extract_features_for_clustering(reflections: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    """
    Extract numerical features from reflections for clustering.
    
    Args:
        reflections: List of reflection entries
        
    Returns:
        Tuple of (feature_matrix, example_indices) where feature_matrix is a numpy array of 
        extracted features and example_indices maps back to the original reflections
    """
    features = []
    valid_indices = []
    
    for i, reflection in enumerate(reflections):
        try:
            # Extract features from completion_pair_features
            pair_features = reflection.get("completion_pair_features", {})
            
            feature_vector = [
                # Prediction-related features
                reflection.get("model_prediction_confidence_raw", 0.5),
                1 if reflection.get("is_prediction_correct", False) else 0,
                
                # Completion comparison features
                pair_features.get("length_difference", 0),
                pair_features.get("token_overlap", 0),
                pair_features.get("sentiment_difference", 0),
                pair_features.get("complexity_difference", 0)
            ]
            
            # Convert error type to numeric value
            error_type = reflection.get("prediction_error_type")
            if error_type == "high_confidence_error":
                feature_vector.append(3)
            elif error_type == "medium_confidence_error":
                feature_vector.append(2)
            elif error_type == "low_confidence_error":
                feature_vector.append(1)
            else:
                feature_vector.append(0)
                
            features.append(feature_vector)
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Error extracting features from reflection {i}: {e}")
    
    # Convert to numpy array
    if not features:
        return np.array([]), []
    
    feature_matrix = np.array(features)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    return normalized_features, valid_indices

def perform_time_based_analysis(
    reflections: List[Dict[str, Any]], 
    config: DriftAnalysisConfig
) -> Dict[str, Any]:
    """
    Perform time-based drift analysis by dividing data into time windows.
    
    Args:
        reflections: List of reflection entries
        config: Configuration parameters
        
    Returns:
        Dictionary with time-based drift analysis results
    """
    # Filter reflections with valid timestamps
    timestamped_reflections = [
        r for r in reflections 
        if "timestamp_parsed" in r and r["timestamp_parsed"] is not None
    ]
    
    if not timestamped_reflections:
        logger.warning("No reflections with valid timestamps found")
        return {"time_windows": [], "drift_detected": False}
    
    # Sort by timestamp
    timestamped_reflections.sort(key=lambda r: r["timestamp_parsed"])
    
    # Determine time range
    start_time = timestamped_reflections[0]["timestamp_parsed"]
    end_time = timestamped_reflections[-1]["timestamp_parsed"]
    total_days = (end_time - start_time).days + 1
    
    # Create time windows
    windows = []
    window_size = timedelta(days=config.time_window_days)
    
    current_start = start_time
    while current_start <= end_time:
        current_end = current_start + window_size
        
        # Get reflections in this window
        window_reflections = [
            r for r in timestamped_reflections
            if current_start <= r["timestamp_parsed"] < current_end
        ]
        
        # Only process windows with enough examples
        if len(window_reflections) >= config.min_examples_per_window:
            # Calculate metrics for this window
            correct_count = sum(1 for r in window_reflections if r.get("is_prediction_correct", False))
            accuracy = correct_count / len(window_reflections)
            
            # Calculate average confidence
            confidences = [r.get("model_prediction_confidence_raw", 0.5) for r in window_reflections]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Count error types
            error_types = Counter([r.get("prediction_error_type", "none") for r in window_reflections])
            
            windows.append({
                "start_time": current_start.isoformat(),
                "end_time": current_end.isoformat(),
                "example_count": len(window_reflections),
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "error_types": dict(error_types)
            })
        
        current_start = current_end
    
    # Detect significant changes between windows
    detected_drifts = []
    for i in range(1, len(windows)):
        prev_window = windows[i-1]
        curr_window = windows[i]
        
        # Check for accuracy drift
        accuracy_change = abs(curr_window["accuracy"] - prev_window["accuracy"])
        if accuracy_change >= config.alert_accuracy_change_threshold:
            detected_drifts.append({
                "type": "accuracy_drift",
                "window_indices": [i-1, i],
                "change_magnitude": accuracy_change,
                "previous": prev_window["accuracy"],
                "current": curr_window["accuracy"],
                "description": f"Accuracy changed by {accuracy_change:.2f} between windows"
            })
    
    return {
        "time_windows": windows,
        "drift_detected": len(detected_drifts) > 0,
        "detected_drifts": detected_drifts
    }

def perform_clustering_analysis(
    reflections: List[Dict[str, Any]], 
    config: DriftAnalysisConfig
) -> Dict[str, Any]:
    """
    Perform clustering analysis to identify groups of similar examples.
    
    Args:
        reflections: List of reflection entries
        config: Configuration parameters
        
    Returns:
        Dictionary with clustering analysis results and DriftCluster objects
    """
    # Extract features for clustering
    features, valid_indices = extract_features_for_clustering(reflections)
    
    if len(features) == 0:
        logger.warning("No valid features extracted for clustering")
        return {
            "clusters": [],
            "algorithm": "none",
            "quality_scores": {},
            "drift_clusters": []
        }
    
    # Check if we have enough data for clustering
    if len(features) < 3:
        logger.warning(f"Not enough samples ({len(features)}) for clustering analysis (minimum 3 required)")
        
        # For extremely small datasets, create a single cluster with all examples
        if len(features) > 0:
            cluster_description = "All examples (insufficient data for clustering)"
            
            # Calculate basic metrics
            all_correct = sum(1 for r in reflections if r.get("is_prediction_correct", False))
            accuracy = all_correct / len(reflections) if reflections else 0
            
            # Calculate average confidence
            confidences = [r.get("model_prediction_confidence_raw", 0.5) for r in reflections]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Count error types
            error_types = Counter([r.get("prediction_error_type", "none") for r in reflections])
            
            cluster = {
                "cluster_id": "cluster_all",
                "size": len(reflections),
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "error_types": dict(error_types),
                "description": cluster_description
            }
            
            drift_cluster = DriftCluster(
                cluster_id="cluster_all",
                description=cluster_description,
                examples=reflections,
                accuracy=accuracy,
                avg_confidence=avg_confidence,
                error_types=dict(error_types)
            )
            
            return {
                "clusters": [cluster],
                "algorithm": "single_cluster",
                "quality_scores": {},
                "drift_clusters": [drift_cluster],
                "limited_data": True
            }
        
        return {
            "clusters": [],
            "algorithm": "none",
            "quality_scores": {},
            "drift_clusters": [],
            "limited_data": True
        }
    
    # Adjust number of clusters for small datasets
    n_clusters = min(config.n_clusters, len(features) - 1)
    
    # Try different clustering approaches and pick the best one
    clustering_results = {}
    
    # KMeans clustering
    try:
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=config.random_seed
        )
        kmeans_labels = kmeans.fit_predict(features)
        
        # Evaluate clustering quality
        if len(np.unique(kmeans_labels)) > 1:
            silhouette = silhouette_score(features, kmeans_labels)
            calinski = calinski_harabasz_score(features, kmeans_labels)
            
            clustering_results["kmeans"] = {
                "labels": kmeans_labels,
                "centroids": kmeans.cluster_centers_,
                "quality": {
                    "silhouette": silhouette,
                    "calinski_harabasz": calinski
                }
            }
    except Exception as e:
        logger.warning(f"Error during KMeans clustering: {e}")
    
    # DBSCAN clustering - only try if we have enough samples
    if len(features) >= config.dbscan_min_samples:
        try:
            # Adjust DBSCAN parameters for small datasets
            min_samples = min(config.dbscan_min_samples, len(features) // 3)
            
            dbscan = DBSCAN(eps=config.dbscan_eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(features)
            
            # Evaluate clustering quality if we have more than one cluster
            unique_labels = np.unique(dbscan_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:  # No noise points
                silhouette = silhouette_score(features, dbscan_labels)
                
                clustering_results["dbscan"] = {
                    "labels": dbscan_labels,
                    "quality": {
                        "silhouette": silhouette,
                        "num_clusters": len(unique_labels)
                    }
                }
        except Exception as e:
            logger.warning(f"Error during DBSCAN clustering: {e}")
    
    # Select best clustering approach based on silhouette score
    selected_algorithm = None
    selected_labels = None
    best_quality = -1
    
    for algo, result in clustering_results.items():
        quality = result["quality"].get("silhouette", -1)
        if quality > best_quality:
            best_quality = quality
            selected_algorithm = algo
            selected_labels = result["labels"]
    
    # Fallback to simpler clustering for small datasets if needed
    if selected_algorithm is None and len(features) < 10:
        logger.warning("Using simple clustering for small dataset")
        # For very small datasets, just use binary clustering based on correctness
        is_correct = np.array([reflections[valid_indices[i]].get("is_prediction_correct", False) 
                              for i in range(len(valid_indices))])
        selected_labels = is_correct.astype(int)
        selected_algorithm = "binary"
    elif selected_algorithm is None:
        logger.warning("No suitable clustering algorithm found")
        return {
            "clusters": [],
            "algorithm": "none",
            "quality_scores": {},
            "drift_clusters": []
        }
    
    # Create cluster summary
    cluster_summaries = []
    drift_clusters = []
    
    # Group reflections by cluster
    clusters = defaultdict(list)
    for idx, cluster_id in enumerate(selected_labels):
        reflection_idx = valid_indices[idx]
        clusters[int(cluster_id)].append(reflections[reflection_idx])
    
    # Create summary for each cluster
    for cluster_id, cluster_reflections in clusters.items():
        # Calculate metrics
        correct_count = sum(1 for r in cluster_reflections if r.get("is_prediction_correct", False))
        accuracy = correct_count / len(cluster_reflections) if cluster_reflections else 0
        
        # Calculate average confidence
        confidences = [r.get("model_prediction_confidence_raw", 0.5) for r in cluster_reflections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count error types
        error_types = Counter([r.get("prediction_error_type", "none") for r in cluster_reflections])
        
        # Generate cluster description
        if selected_algorithm == "binary":
            if cluster_id == 1:
                description = f"Cluster {cluster_id}: Correctly predicted examples. "
            else:
                description = f"Cluster {cluster_id}: Incorrectly predicted examples. "
        else:
            description = f"Cluster {cluster_id}: "
            if accuracy < 0.5:
                description += "Low accuracy cluster. "
            elif accuracy > 0.8:
                description += "High accuracy cluster. "
                
            if avg_confidence < 0.6 and accuracy > 0.7:
                description += "Model is under-confident. "
            elif avg_confidence > 0.8 and accuracy < 0.7:
                description += "Model is over-confident. "
        
        # Get most common error type
        most_common_error = error_types.most_common(1)
        if most_common_error and most_common_error[0][0] != "none":
            description += f"Common error: {most_common_error[0][0]}. "
        
        # Get centroid if available
        centroid = None
        if selected_algorithm == "kmeans" and "centroids" in clustering_results["kmeans"]:
            centroid = clustering_results["kmeans"]["centroids"][cluster_id].tolist()
        
        # Create cluster summary
        summary = {
            "cluster_id": f"cluster_{cluster_id}",
            "size": len(cluster_reflections),
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "error_types": dict(error_types),
            "description": description
        }
        
        cluster_summaries.append(summary)
        
        # Create drift cluster object
        drift_cluster = DriftCluster(
            cluster_id=f"cluster_{cluster_id}",
            description=description,
            examples=cluster_reflections,
            centroid=centroid,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            error_types=dict(error_types)
        )
        
        drift_clusters.append(drift_cluster)
    
    # Identify potential drift clusters (those with notably different accuracy)
    global_accuracy = sum(1 for r in reflections if r.get("is_prediction_correct", False)) / len(reflections)
    
    potential_drift_clusters = []
    for summary in cluster_summaries:
        # Check if cluster accuracy differs significantly from global accuracy
        if abs(summary["accuracy"] - global_accuracy) >= config.alert_accuracy_change_threshold:
            potential_drift_clusters.append(summary["cluster_id"])
    
    return {
        "clusters": cluster_summaries,
        "algorithm": selected_algorithm,
        "quality_scores": clustering_results.get(selected_algorithm, {}).get("quality", {}),
        "drift_clusters": drift_clusters,
        "potential_drift_clusters": potential_drift_clusters
    }

def analyze_confidence_calibration_drift(reflections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze drift in confidence calibration over time.
    
    Args:
        reflections: List of reflection entries
        
    Returns:
        Dictionary with confidence calibration drift analysis
    """
    # Filter reflections with valid timestamps
    timestamped_reflections = [
        r for r in reflections 
        if "timestamp_parsed" in r and r["timestamp_parsed"] is not None
    ]
    
    if not timestamped_reflections:
        logger.warning("No reflections with valid timestamps found")
        return {"calibration_drift_detected": False}
        
    # Check if we have enough data for quartile analysis
    if len(timestamped_reflections) < 4:
        logger.warning(f"Not enough data for confidence calibration analysis: {len(timestamped_reflections)} samples (minimum 4 required)")
        return {
            "calibration_drift_detected": False,
            "limited_data": True,
            "sample_count": len(timestamped_reflections)
        }
    
    # Sort by timestamp
    timestamped_reflections.sort(key=lambda r: r["timestamp_parsed"])
    
    # Divide into quartiles by time or fewer bins for small datasets
    n = len(timestamped_reflections)
    
    # Use 2 bins instead of 4 for smaller datasets
    if n < 10:
        num_bins = 2
        logger.warning(f"Using {num_bins} time bins instead of quartiles due to small dataset size ({n} samples)")
    else:
        num_bins = 4  # default to quartiles
        
    bin_size = max(n // num_bins, 1)
    
    time_bins = [
        timestamped_reflections[i:i+bin_size] 
        for i in range(0, n, bin_size)
    ]
    
    # Calculate expected vs actual confidence for each time bin
    bin_calibration = []
    
    for i, bin_data in enumerate(time_bins):
        if not bin_data:
            continue
            
        # Calculate average confidence
        avg_confidence = sum(r.get("model_prediction_confidence_raw", 0.5) for r in bin_data) / len(bin_data)
        
        # Calculate actual accuracy
        accuracy = sum(1 for r in bin_data if r.get("is_prediction_correct", False)) / len(bin_data)
        
        # Calculate calibration error (how far is confidence from actual accuracy)
        calibration_error = abs(avg_confidence - accuracy)
        
        bin_calibration.append({
            "bin_index": i,
            "sample_count": len(bin_data),
            "avg_confidence": avg_confidence,
            "accuracy": accuracy,
            "calibration_error": calibration_error,
            "is_calibrated": calibration_error < 0.1,  # threshold for "well calibrated"
            "start_time": bin_data[0]["timestamp_parsed"].isoformat(),
            "end_time": bin_data[-1]["timestamp_parsed"].isoformat()
        })
    
    # Check for significant drift in calibration quality
    calibration_drift_detected = False
    calibration_trend = "stable"
    
    if len(bin_calibration) >= 2:
        # Compare first and last time bins
        first_bin = bin_calibration[0]
        last_bin = bin_calibration[-1]
        
        calibration_change = last_bin["calibration_error"] - first_bin["calibration_error"]
        
        if abs(calibration_change) > 0.1:
            calibration_drift_detected = True
            if calibration_change > 0:
                calibration_trend = "worsening"
            else:
                calibration_trend = "improving"
    
    return {
        "quartile_calibration": bin_calibration,
        "calibration_drift_detected": calibration_drift_detected,
        "calibration_trend": calibration_trend
    }

def generate_visualizations(
    reflections: List[Dict[str, Any]], 
    clustering_result: Dict[str, Any],
    output_dir: str,
    small_dataset: bool = False
) -> Dict[str, str]:
    """
    Generate visualizations for drift analysis.
    
    Args:
        reflections: List of reflection entries
        clustering_result: Result of clustering analysis
        output_dir: Directory to save visualizations
        small_dataset: Boolean indicating if the dataset is small
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    visualization_paths = {}
    
    # Extract features for visualization
    features, valid_indices = extract_features_for_clustering(reflections)
    
    if len(features) == 0:
        logger.warning("No valid features extracted for visualization")
        return visualization_paths
    
    # Get cluster labels if available
    labels = None
    if "algorithm" in clustering_result and clustering_result["algorithm"] != "none":
        for cluster in clustering_result.get("clusters", []):
            cluster_id = cluster["cluster_id"]
            cluster_reflections = [r for r in reflections if r.get("drift_cluster_id") == cluster_id]
            cluster_indices = [i for i, r in enumerate(reflections) if r.get("drift_cluster_id") == cluster_id]
    
    # Create TSNE embedding for visualization if we have enough data points
    if len(features) >= 3:  # TSNE needs at least 3 samples
        try:
            # For small datasets, use simpler parameters
            if small_dataset:
                tsne = TSNE(
                    n_components=2, 
                    random_state=42,
                    perplexity=min(len(features) - 1, 5),  # Adjust perplexity for small datasets
                    n_iter=500,  # Reduce iterations for small datasets
                    learning_rate='auto'
                )
            else:
                tsne = TSNE(n_components=2, random_state=42)
                
            embeddings = tsne.fit_transform(features)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            
            # Color points by prediction correctness
            is_correct = [reflections[valid_indices[i]].get("is_prediction_correct", False) for i in range(len(valid_indices))]
            
            correct_points = embeddings[is_correct]
            incorrect_points = embeddings[~np.array(is_correct)]
            
            plt.scatter(correct_points[:, 0], correct_points[:, 1], c='green', alpha=0.6, label='Correct Predictions')
            plt.scatter(incorrect_points[:, 0], incorrect_points[:, 1], c='red', alpha=0.6, label='Incorrect Predictions')
            
            title = 'TSNE Visualization of Model Predictions'
            if small_dataset:
                title += ' (Limited Data)'
            plt.title(title)
            plt.xlabel('TSNE Dimension 1')
            plt.ylabel('TSNE Dimension 2')
            plt.legend()
            
            # Save plot
            tsne_path = os.path.join(output_dir, 'tsne_visualization.png')
            plt.savefig(tsne_path)
            plt.close()
            
            visualization_paths['tsne'] = tsne_path
        except Exception as e:
            logger.warning(f"Error generating TSNE visualization: {e}")
    else:
        logger.warning(f"Not enough samples ({len(features)}) for TSNE visualization (minimum 3 required)")
    
    # Create accuracy over time plot
    try:
        # Filter reflections with valid timestamps
        timestamped_reflections = [
            r for r in reflections 
            if "timestamp_parsed" in r and r["timestamp_parsed"] is not None
        ]
        
        if timestamped_reflections:
            # Sort by timestamp
            timestamped_reflections.sort(key=lambda r: r["timestamp_parsed"])
            
            # Create rolling window accuracy
            if small_dataset:
                # For very small datasets, just plot the raw data
                window_size = 1
            else:
                window_size = min(20, max(3, len(timestamped_reflections) // 5))
                
            if len(timestamped_reflections) >= window_size:
                timestamps = [r["timestamp_parsed"] for r in timestamped_reflections]
                is_correct = [r.get("is_prediction_correct", False) for r in timestamped_reflections]
                
                rolling_accuracy = []
                for i in range(len(is_correct) - window_size + 1):
                    window_accuracy = sum(is_correct[i:i+window_size]) / window_size
                    rolling_accuracy.append(window_accuracy)
                
                plt.figure(figsize=(12, 6))
                
                if window_size == 1:
                    # For window_size=1, we're plotting raw values, so use step plot
                    plt.step(timestamps, is_correct, marker='o', where='mid', label='Correct (1) / Incorrect (0)')
                    plt.title('Model Predictions Over Time (Raw Data)')
                    plt.yticks([0, 1], ['Incorrect', 'Correct'])
                else:
                    plt.plot(timestamps[window_size-1:], rolling_accuracy, marker='o', linestyle='-')
                    plt.title(f'Model Accuracy Over Time (Rolling Window: {window_size} examples)')
                
                plt.xlabel('Time')
                plt.ylabel('Accuracy')
                plt.ylim(-0.1, 1.1)  # Give a bit of space for the step plot
                plt.grid(True, alpha=0.3)
                
                # Add best fit line only if we have enough data
                if len(timestamps) >= 4 and not small_dataset:
                    try:
                        import numpy as np
                        from sklearn.linear_model import LinearRegression
                        
                        X = np.array([(t - timestamps[0]).total_seconds() for t in timestamps[window_size-1:]]).reshape(-1, 1)
                        y = np.array(rolling_accuracy)
                        
                        model = LinearRegression().fit(X, y)
                        y_pred = model.predict(X)
                        
                        plt.plot(timestamps[window_size-1:], y_pred, 'r--', label=f'Trend (slope: {model.coef_[0]:.6f})')
                        plt.legend()
                    except Exception as e:
                        logger.warning(f"Error generating trend line: {e}")
                
                # Save plot
                accuracy_path = os.path.join(output_dir, 'accuracy_over_time.png')
                plt.savefig(accuracy_path)
                plt.close()
                
                visualization_paths['accuracy_over_time'] = accuracy_path
            else:
                logger.warning(f"Not enough timestamped reflections ({len(timestamped_reflections)}) for window size {window_size}")
    except Exception as e:
        logger.warning(f"Error generating accuracy over time visualization: {e}")
    
    return visualization_paths

def save_drift_clusters(drift_clusters: List[DriftCluster], output_path: str) -> None:
    """
    Save drift clusters to a file.
    
    Args:
        drift_clusters: List of DriftCluster objects
        output_path: Path to save the drift clusters
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format and save
        with open(output_path, 'w', encoding='utf-8') as f:
            for cluster in drift_clusters:
                cluster_dict = cluster.to_dict()
                json.dump(cluster_dict, f)
                f.write('\n')
        
        logger.info(f"Saved {len(drift_clusters)} drift clusters to {output_path}")
    except Exception as e:
        logger.error(f"Error saving drift clusters: {e}")

def update_reflections_with_drift_info(
    reflections: List[Dict[str, Any]], 
    clustering_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Update reflection entries with drift cluster information.
    
    Args:
        reflections: List of reflection entries
        clustering_result: Result of clustering analysis
        
    Returns:
        Updated list of reflection entries
    """
    # Create mapping from example to cluster
    example_to_cluster = {}
    
    for cluster in clustering_result.get("drift_clusters", []):
        cluster_id = cluster.cluster_id
        for example in cluster.examples:
            example_id = example.get("id") or example.get("entry_id")
            if example_id:
                example_to_cluster[example_id] = cluster_id
    
    # Update reflections
    updated_reflections = []
    for reflection in reflections:
        example_id = reflection.get("id") or reflection.get("entry_id")
        if example_id and example_id in example_to_cluster:
            reflection["drift_cluster_id"] = example_to_cluster[example_id]
        updated_reflections.append(reflection)
    
    return updated_reflections

def run_drift_analysis(
    reflection_path: str,
    output_dir: str,
    config: Optional[DriftAnalysisConfig] = None
) -> Dict[str, Any]:
    """
    Run full drift analysis on reflection data.
    
    Args:
        reflection_path: Path to meta_reflection_log.jsonl file
        output_dir: Directory to save output files
        config: Configuration parameters (or None to use defaults)
        
    Returns:
        Dictionary with drift analysis results
    """
    # Use default config if not provided
    if config is None:
        config = DriftAnalysisConfig()
    
    # Load reflection data
    reflections = load_reflection_log(reflection_path)
    
    if not reflections:
        logger.error("No reflection data loaded")
        return {"error": "No reflection data loaded"}
    
    # Check if we have sufficient data for meaningful analysis
    small_dataset = len(reflections) < config.min_samples_required
    if small_dataset:
        logger.warning(f"Small dataset detected: {len(reflections)} samples, which is below the recommended minimum of {config.min_samples_required} samples")
        logger.warning("Limited drift analysis will be performed, some features will be disabled")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform time-based analysis if sufficient data
    logger.info("Performing time-based drift analysis")
    if small_dataset:
        time_analysis = {"time_windows": [], "drift_detected": False, "limited_data": True}
    else:
        time_analysis = perform_time_based_analysis(reflections, config)
    
    # Perform clustering analysis with adjusted parameters for small datasets
    logger.info("Performing clustering analysis")
    if small_dataset:
        # Adjust cluster parameters for small datasets
        small_config = DriftAnalysisConfig(**asdict(config))
        small_config.n_clusters = min(3, len(reflections) // 2)  # Reduce clusters for small datasets
        small_config.dbscan_min_samples = min(3, len(reflections) // 3)  # Adjust DBSCAN params
        clustering_result = perform_clustering_analysis(reflections, small_config)
    else:
        clustering_result = perform_clustering_analysis(reflections, config)
    
    # Analyze confidence calibration drift
    logger.info("Analyzing confidence calibration drift")
    if small_dataset and len(reflections) < 4:  # Too small for quartile analysis
        calibration_analysis = {"calibration_drift_detected": False, "limited_data": True}
    else:
        calibration_analysis = analyze_confidence_calibration_drift(reflections)
    
    # Generate visualizations with adjusted parameters for small datasets
    logger.info("Generating visualizations")
    visualization_dir = os.path.join(output_dir, "visualizations")
    visualization_paths = generate_visualizations(reflections, clustering_result, visualization_dir, small_dataset=small_dataset)
    
    # Update reflections with drift information
    logger.info("Updating reflections with drift cluster information")
    updated_reflections = update_reflections_with_drift_info(reflections, clustering_result)
    
    # Save drift clusters
    drift_clusters_path = os.path.join(output_dir, "drift_clusters.jsonl")
    save_drift_clusters(clustering_result.get("drift_clusters", []), drift_clusters_path)
    
    # Combine all results
    drift_analysis = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": asdict(config),
        "summary": {
            "total_examples": len(reflections),
            "small_dataset": small_dataset,
            "time_drift_detected": time_analysis.get("drift_detected", False),
            "num_clusters": len(clustering_result.get("clusters", [])),
            "potential_drift_clusters": clustering_result.get("potential_drift_clusters", []),
            "calibration_drift_detected": calibration_analysis.get("calibration_drift_detected", False),
            "calibration_trend": calibration_analysis.get("calibration_trend", "stable")
        },
        "time_analysis": time_analysis,
        "clustering_analysis": {
            "algorithm": clustering_result.get("algorithm", "none"),
            "quality_scores": clustering_result.get("quality_scores", {}),
            "clusters": clustering_result.get("clusters", [])
        },
        "calibration_analysis": calibration_analysis,
        "visualization_paths": visualization_paths
    }
    
    # Save full analysis results
    analysis_path = os.path.join(output_dir, "drift_analysis.json")
    try:
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(drift_analysis, f, indent=2)
        logger.info(f"Drift analysis saved to {analysis_path}")
    except Exception as e:
        logger.error(f"Error saving drift analysis: {e}")
    
    # Save updated reflections if clusters were identified
    if clustering_result.get("drift_clusters"):
        updated_reflections_path = os.path.join(output_dir, "reflections_with_drift.jsonl")
        try:
            with open(updated_reflections_path, 'w', encoding='utf-8') as f:
                for reflection in updated_reflections:
                    json.dump(reflection, f)
                    f.write('\n')
            logger.info(f"Updated reflections saved to {updated_reflections_path}")
        except Exception as e:
            logger.error(f"Error saving updated reflections: {e}")
    
    return drift_analysis

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Drift monitoring for RLHF vote predictor")
    parser.add_argument("--reflection-path", type=str, default=os.path.join(project_root, "models", "meta_reflection_log.jsonl"),
                        help="Path to the meta_reflection_log.jsonl file")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "models", "drift_analysis"),
                        help="Directory to save drift analysis results")
    parser.add_argument("--time-window-days", type=int, default=7,
                        help="Size of time window in days for temporal analysis")
    parser.add_argument("--min-examples-per-window", type=int, default=5,
                        help="Minimum examples required for valid window")
    parser.add_argument("--min-samples-required", type=int, default=10,
                        help="Minimum samples required for full drift analysis")
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="Number of clusters for KMeans")
    parser.add_argument("--confidence-drift-threshold", type=float, default=0.1,
                        help="Threshold for confidence calibration drift")
    parser.add_argument("--accuracy-change-threshold", type=float, default=0.1,
                        help="Alert threshold for accuracy change")
    
    args = parser.parse_args()
    
    # Create config from args
    config = DriftAnalysisConfig(
        time_window_days=args.time_window_days,
        min_examples_per_window=args.min_examples_per_window,
        min_samples_required=args.min_samples_required,
        n_clusters=args.n_clusters,
        confidence_drift_threshold=args.confidence_drift_threshold,
        alert_accuracy_change_threshold=args.accuracy_change_threshold
    )
    
    # Run drift analysis
    drift_analysis = run_drift_analysis(
        reflection_path=args.reflection_path,
        output_dir=args.output_dir,
        config=config
    )
    
    # Output summary
    summary = drift_analysis.get("summary", {})
    print("\nDrift Analysis Summary:")
    print(f"Total examples analyzed: {summary.get('total_examples', 0)}")
    if summary.get('small_dataset', False):
        print("WARNING: Limited analysis performed due to small dataset size")
    print(f"Time-based drift detected: {summary.get('time_drift_detected', False)}")
    print(f"Number of clusters: {summary.get('num_clusters', 0)}")
    print(f"Potential drift clusters: {summary.get('potential_drift_clusters', [])}")
    print(f"Calibration drift detected: {summary.get('calibration_drift_detected', False)}")
    print(f"Calibration trend: {summary.get('calibration_trend', 'stable')}")
    print(f"\nAnalysis saved to {args.output_dir}")

if __name__ == "__main__":
    main() 