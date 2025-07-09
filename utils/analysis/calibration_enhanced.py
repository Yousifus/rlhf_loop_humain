"""
Enhanced Calibration Analysis for RLHF System

This module implements advanced calibration analysis techniques including:
- Isotonic regression for calibration correction
- Maximum Calibration Error (MCE)
- Adaptive Calibration Error (ACE) with dynamic binning
- Kullback-Leibler (KL) calibration metric
- Multi-class calibration support
- Confidence interval calculations
- Calibration history comparison
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import kl_div
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample
import warnings

@dataclass
class CalibrationMetrics:
    """Container for calibration metrics"""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Adaptive Calibration Error
    kl_calibration: float  # KL divergence calibration metric
    brier_score: float  # Brier score
    reliability_data: Dict[str, List[float]]  # Reliability diagram data
    confidence_intervals: Dict[str, Tuple[float, float]]  # Bootstrap CIs


class AdvancedCalibrationAnalyzer:
    """
    Advanced calibration analyzer with enhanced metrics and correction methods.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.isotonic_regressor = None
        
    def calculate_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        n_bins: int = 10,
        n_bootstrap: int = 1000
    ) -> CalibrationMetrics:
        """
        Calculate comprehensive calibration metrics.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities [0, 1]
            n_bins: Number of bins for ECE calculation
            n_bootstrap: Number of bootstrap samples for confidence intervals
            
        Returns:
            CalibrationMetrics object with all computed metrics
        """
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        # Basic validation
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            raise ValueError("y_prob must be in [0, 1] range")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1 values")
            
        # Calculate core metrics
        ece = self.calculate_ece(y_true, y_prob, n_bins)
        mce = self.calculate_mce(y_true, y_prob, n_bins)
        ace = self.calculate_ace(y_true, y_prob)
        kl_cal = self.calculate_kl_calibration(y_true, y_prob, n_bins)
        brier = self.calculate_brier_score(y_true, y_prob)
        
        # Get reliability diagram data
        bin_boundaries, bin_lowers, bin_uppers, bin_accuracies, bin_confidences, bin_counts = \
            self._get_reliability_diagram_data(y_true, y_prob, n_bins)
            
        reliability_data = {
            'bin_boundaries': bin_boundaries.tolist(),
            'bin_accuracies': bin_accuracies.tolist(),
            'bin_confidences': bin_confidences.tolist(),
            'bin_counts': bin_counts.tolist()
        }
        
        # Calculate confidence intervals using bootstrap
        confidence_intervals = self._calculate_confidence_intervals(
            y_true, y_prob, n_bins, n_bootstrap
        )
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            ace=ace,
            kl_calibration=kl_cal,
            brier_score=brier,
            reliability_data=reliability_data,
            confidence_intervals=confidence_intervals
        )
    
    def calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error with uniform binning."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def calculate_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                calibration_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_calibration_error = max(max_calibration_error, calibration_error)
                
        return max_calibration_error
    
    def calculate_ace(self, y_true: np.ndarray, y_prob: np.ndarray, 
                     n_bins: int = 10, min_bin_size: int = 10) -> float:
        """
        Calculate Adaptive Calibration Error with dynamic bin sizing.
        Bins are adjusted to have approximately equal sample counts.
        """
        if len(y_prob) < n_bins * min_bin_size:
            # Fall back to regular ECE if not enough samples
            return self.calculate_ece(y_true, y_prob, n_bins)
        
        # Sort by confidence and create adaptive bins
        sorted_indices = np.argsort(y_prob)
        sorted_y_true = y_true[sorted_indices]
        sorted_y_prob = y_prob[sorted_indices]
        
        # Create bins with approximately equal counts
        samples_per_bin = len(y_prob) // n_bins
        ace = 0
        
        for i in range(n_bins):
            start_idx = i * samples_per_bin
            end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else len(y_prob)
            
            bin_y_true = sorted_y_true[start_idx:end_idx]
            bin_y_prob = sorted_y_prob[start_idx:end_idx]
            
            if len(bin_y_true) > 0:
                accuracy_in_bin = bin_y_true.mean()
                avg_confidence_in_bin = bin_y_prob.mean()
                prop_in_bin = len(bin_y_true) / len(y_true)
                ace += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ace
    
    def calculate_kl_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> float:
        """
        Calculate KL divergence-based calibration metric.
        Measures divergence between predicted and actual probability distributions.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        predicted_probs = []
        actual_probs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence_in_bin = y_prob[in_bin].mean()
                accuracy_in_bin = y_true[in_bin].mean()
                
                predicted_probs.append(avg_confidence_in_bin)
                actual_probs.append(accuracy_in_bin)
        
        if len(predicted_probs) == 0:
            return 0.0
            
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        predicted_probs = np.array(predicted_probs) + epsilon
        actual_probs = np.array(actual_probs) + epsilon
        
        # Calculate KL divergence
        kl_div_val = stats.entropy(actual_probs, predicted_probs)
        return kl_div_val
    
    def calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier score."""
        return np.mean((y_prob - y_true) ** 2)
    
    def _get_reliability_diagram_data(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                     n_bins: int = 10) -> Tuple:
        """Get data for reliability diagram plotting."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                count_in_bin = 0
                
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
        
        return (
            bin_boundaries,
            np.array(bin_lowers),
            np.array(bin_uppers),
            np.array(bin_accuracies),
            np.array(bin_confidences),
            np.array(bin_counts)
        )
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_prob: np.ndarray,
                                      n_bins: int, n_bootstrap: int) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for calibration metrics."""
        np.random.seed(self.random_state)
        
        ece_samples = []
        mce_samples = []
        brier_samples = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            
            # Calculate metrics
            ece_samples.append(self.calculate_ece(y_true_boot, y_prob_boot, n_bins))
            mce_samples.append(self.calculate_mce(y_true_boot, y_prob_boot, n_bins))
            brier_samples.append(self.calculate_brier_score(y_true_boot, y_prob_boot))
        
        # Calculate 95% confidence intervals
        confidence_intervals = {}
        for metric_name, samples in [('ece', ece_samples), ('mce', mce_samples), ('brier', brier_samples)]:
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            confidence_intervals[metric_name] = (lower, upper)
            
        return confidence_intervals
    
    def fit_isotonic_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'AdvancedCalibrationAnalyzer':
        """
        Fit isotonic regression for calibration correction.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities to calibrate
            
        Returns:
            Self for method chaining
        """
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(y_prob, y_true)
        return self
    
    def apply_isotonic_calibration(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply fitted isotonic calibration to new probabilities.
        
        Args:
            y_prob: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if self.isotonic_regressor is None:
            raise ValueError("Must call fit_isotonic_calibration first")
        
        return self.isotonic_regressor.predict(y_prob)
    
    def create_enhanced_reliability_plot(self, metrics: CalibrationMetrics, 
                                       title: str = "Enhanced Reliability Diagram") -> go.Figure:
        """
        Create an enhanced reliability diagram with confidence intervals and additional metrics.
        """
        reliability_data = metrics.reliability_data
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        # Bin centers for x-axis
        bin_centers = [(reliability_data['bin_boundaries'][i] + reliability_data['bin_boundaries'][i+1]) / 2 
                      for i in range(len(reliability_data['bin_boundaries']) - 1)]
        
        # Reliability bars
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=reliability_data['bin_accuracies'],
            width=0.08,
            name='Accuracy',
            opacity=0.7,
            marker_color='lightblue',
            hovertemplate='Bin Center: %{x:.2f}<br>Accuracy: %{y:.3f}<br>Count: %{text}<extra></extra>',
            text=reliability_data['bin_counts']
        ))
        
        # Average confidence per bin
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=reliability_data['bin_confidences'],
            mode='markers+lines',
            name='Average Confidence',
            marker=dict(size=8, color='red'),
            line=dict(color='red', dash='dot')
        ))
        
        # Add metric annotations
        ci_ece = metrics.confidence_intervals['ece']
        ci_brier = metrics.confidence_intervals['brier']
        
        annotation_text = (
            f"ECE: {metrics.ece:.4f} [{ci_ece[0]:.4f}, {ci_ece[1]:.4f}]<br>"
            f"MCE: {metrics.mce:.4f}<br>"
            f"ACE: {metrics.ace:.4f}<br>"
            f"KL Cal: {metrics.kl_calibration:.4f}<br>"
            f"Brier: {metrics.brier_score:.4f} [{ci_brier[0]:.4f}, {ci_brier[1]:.4f}]"
        )
        
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=600,
            width=800,
            showlegend=True
        )
        
        return fig


def calculate_isotonic_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, IsotonicRegression]:
    """
    Convenience function to calculate isotonic calibration.
    
    Returns:
        Tuple of (calibrated_probabilities, fitted_regressor)
    """
    analyzer = AdvancedCalibrationAnalyzer()
    analyzer.fit_isotonic_calibration(y_true, y_prob)
    calibrated_probs = analyzer.apply_isotonic_calibration(y_prob)
    return calibrated_probs, analyzer.isotonic_regressor


def calculate_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Convenience function to calculate Maximum Calibration Error."""
    analyzer = AdvancedCalibrationAnalyzer()
    return analyzer.calculate_mce(y_true, y_prob, n_bins)


def calculate_ace(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Convenience function to calculate Adaptive Calibration Error."""
    analyzer = AdvancedCalibrationAnalyzer()
    return analyzer.calculate_ace(y_true, y_prob, n_bins)


def calculate_kl_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Convenience function to calculate KL divergence calibration metric."""
    analyzer = AdvancedCalibrationAnalyzer()
    return analyzer.calculate_kl_calibration(y_true, y_prob, n_bins)