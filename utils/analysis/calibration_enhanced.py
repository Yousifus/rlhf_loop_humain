"""
Enhanced Calibration Analysis for RLHF Loop
============================================

Comprehensive calibration analysis tools including:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)  
- Adaptive Calibration Error (ACE)
- Kullback-Leibler calibration metric
- Brier Score and Log Loss
- Reliability diagrams
- Confidence distribution analysis
- Temperature scaling calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class CalibrationMetrics(NamedTuple):
    """Container for all calibration metrics"""
    ece: float
    mce: float
    ace: float
    kl_calibration: float
    brier_score: float
    log_loss: float
    reliability_data: Dict
    confidence_intervals: Dict
    overall_metrics: Dict
    bin_stats: List[Dict]
    confidence_distribution: Dict

class AdvancedCalibrationAnalyzer:
    """
    Advanced calibration analysis with comprehensive metrics
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                             n_bins: int = 10, n_bootstrap: int = 100) -> CalibrationMetrics:
        """
        Calculate comprehensive calibration metrics
        
        Args:
            y_true: True binary labels (0/1)
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            n_bootstrap: Number of bootstrap samples for confidence intervals
        
        Returns:
            CalibrationMetrics object with all metrics
        """
        if len(y_true) == 0 or len(y_prob) == 0:
            return self._empty_metrics()
            
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        
        # Remove invalid values
        valid_mask = ~(np.isnan(y_prob) | np.isnan(y_true))
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]
        
        if len(y_true) == 0:
            return self._empty_metrics()
        
        # Ensure probabilities are in [0, 1]
        y_prob = np.clip(y_prob, 0.0, 1.0)
        
        # Calculate basic metrics
        ece = self._calculate_ece(y_true, y_prob, n_bins)
        mce = self._calculate_mce(y_true, y_prob, n_bins)
        ace = self._calculate_ace(y_true, y_prob)
        kl_calibration = self._calculate_kl_calibration(y_true, y_prob, n_bins)
        brier_score = self._calculate_brier_score(y_true, y_prob)
        log_loss = self._calculate_log_loss(y_true, y_prob)
        
        # Calculate reliability data for plots
        reliability_data = self._calculate_reliability_data(y_true, y_prob, n_bins)
        
        # Calculate bin statistics
        bin_stats = self._calculate_bin_stats(y_true, y_prob, n_bins)
        
        # Calculate confidence distributions
        confidence_distribution = self._calculate_confidence_distribution(y_true, y_prob)
        
        # Calculate confidence intervals via bootstrap
        confidence_intervals = self._calculate_confidence_intervals(
            y_true, y_prob, n_bootstrap
        )
        
        # Overall metrics summary
        overall_metrics = {
            'ece': ece,
            'mce': mce,
            'ace': ace,
            'avg_confidence': np.mean(y_prob),
            'accuracy': np.mean(y_true),
            'confidence_gap': np.mean(y_prob) - np.mean(y_true),
            'brier_score': brier_score,
            'log_loss': log_loss,
            'kl_calibration': kl_calibration
        }
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            ace=ace,
            kl_calibration=kl_calibration,
            brier_score=brier_score,
            log_loss=log_loss,
            reliability_data=reliability_data,
            confidence_intervals=confidence_intervals,
            overall_metrics=overall_metrics,
            bin_stats=bin_stats,
            confidence_distribution=confidence_distribution
        )
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(y_prob)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_ace(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Adaptive Calibration Error with optimal binning"""
        if len(y_prob) < 10:
            return self._calculate_ece(y_true, y_prob, 5)  # Fewer bins for small datasets
        
        # Use quantile-based binning for ACE
        quantiles = np.linspace(0, 1, 11)  # 10 bins
        bin_boundaries = np.quantile(y_prob, quantiles)
        
        ace = 0
        total_samples = len(y_prob)
        
        for i in range(len(bin_boundaries) - 1):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == len(bin_boundaries) - 2:  # Last bin includes upper boundary
                in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ace += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ace
    
    def _calculate_kl_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
        """Calculate KL-divergence based calibration metric"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        observed_freq = []
        expected_freq = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                observed_freq.append(accuracy_in_bin)
                expected_freq.append(avg_confidence_in_bin)
        
        if len(observed_freq) == 0:
            return 0.0
        
        observed_freq = np.array(observed_freq)
        expected_freq = np.array(expected_freq)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        observed_freq = np.clip(observed_freq, epsilon, 1 - epsilon)
        expected_freq = np.clip(expected_freq, epsilon, 1 - epsilon)
        
        # Calculate symmetric KL divergence
        kl1 = np.sum(observed_freq * np.log(observed_freq / expected_freq))
        kl2 = np.sum(expected_freq * np.log(expected_freq / observed_freq))
        
        return (kl1 + kl2) / 2
    
    def _calculate_brier_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Brier Score"""
        return np.mean((y_prob - y_true) ** 2)
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Log Loss"""
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped))
    
    def _calculate_reliability_data(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> Dict:
        """Calculate reliability diagram data"""
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
        
        return {
            'bin_boundaries': bin_boundaries,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    
    def _calculate_bin_stats(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> List[Dict]:
        """Calculate detailed bin statistics"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_stats = []
        total_samples = len(y_prob)
        
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            count_in_bin = in_bin.sum()
            
            if count_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                abs_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                weight = count_in_bin / total_samples
                contrib_to_ece = abs_error * weight
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
                abs_error = 0
                weight = 0
                contrib_to_ece = 0
            
            bin_stats.append({
                'bin': f'[{bin_lower:.1f}, {bin_upper:.1f}]',
                'bin_range': [bin_lower, bin_upper],
                'samples': count_in_bin,
                'avg_confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'abs_error': abs_error,
                'weight': weight,
                'contrib_to_ece': contrib_to_ece
            })
        
        return bin_stats
    
    def _calculate_confidence_distribution(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate confidence distributions for correct vs incorrect predictions"""
        correct_mask = y_true == 1
        incorrect_mask = y_true == 0
        
        # Create confidence histograms
        bins = np.linspace(0, 1, 21)  # 20 bins for distribution
        
        correct_conf = y_prob[correct_mask] if correct_mask.sum() > 0 else np.array([])
        incorrect_conf = y_prob[incorrect_mask] if incorrect_mask.sum() > 0 else np.array([])
        
        correct_hist, _ = np.histogram(correct_conf, bins=bins)
        incorrect_hist, _ = np.histogram(incorrect_conf, bins=bins)
        
        # Convert to format expected by frontend
        correct_dist = [
            {'confidence': (bins[i] + bins[i+1]) / 2, 'count': int(correct_hist[i])}
            for i in range(len(correct_hist))
        ]
        
        incorrect_dist = [
            {'confidence': (bins[i] + bins[i+1]) / 2, 'count': int(incorrect_hist[i])}
            for i in range(len(incorrect_hist))
        ]
        
        return {
            'correct': correct_dist,
            'incorrect': incorrect_dist
        }
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                      n_bootstrap: int) -> Dict:
        """Calculate confidence intervals via bootstrap"""
        if len(y_true) < 10:
            return {'ece': [0, 0], 'brier_score': [0, 0]}
        
        bootstrap_eces = []
        bootstrap_briers = []
        
        for _ in range(min(n_bootstrap, 50)):  # Limit bootstrap for performance
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            boot_y_true = y_true[indices]
            boot_y_prob = y_prob[indices]
            
            boot_ece = self._calculate_ece(boot_y_true, boot_y_prob, 10)
            boot_brier = self._calculate_brier_score(boot_y_true, boot_y_prob)
            
            bootstrap_eces.append(boot_ece)
            bootstrap_briers.append(boot_brier)
        
        ece_ci = [np.percentile(bootstrap_eces, 2.5), np.percentile(bootstrap_eces, 97.5)]
        brier_ci = [np.percentile(bootstrap_briers, 2.5), np.percentile(bootstrap_briers, 97.5)]
        
        return {
            'ece': ece_ci,
            'brier_score': brier_ci
        }
    
    def _empty_metrics(self) -> CalibrationMetrics:
        """Return empty metrics when no data is available"""
        return CalibrationMetrics(
            ece=0.0,
            mce=0.0,
            ace=0.0,
            kl_calibration=0.0,
            brier_score=0.0,
            log_loss=0.0,
            reliability_data={'bin_boundaries': [], 'bin_accuracies': [], 'bin_confidences': [], 'bin_counts': []},
            confidence_intervals={'ece': [0, 0], 'brier_score': [0, 0]},
            overall_metrics={
                'ece': 0.0, 'mce': 0.0, 'ace': 0.0, 'avg_confidence': 0.0,
                'accuracy': 0.0, 'confidence_gap': 0.0, 'brier_score': 0.0,
                'log_loss': 0.0, 'kl_calibration': 0.0
            },
            bin_stats=[],
            confidence_distribution={'correct': [], 'incorrect': []}
        )

    def calculate_temperature_scaling(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Calculate temperature scaling calibration
        """
        if len(y_true) < 10:
            return {
                'pre_calibration': {'ece': 0.0, 'log_loss': 0.0, 'brier_score': 0.0},
                'post_calibration': {'ece': 0.0, 'log_loss': 0.0, 'brier_score': 0.0},
                'temperature': 1.0,
                'improvement': {'ece': 0.0, 'log_loss': 0.0, 'brier_score': 0.0}
            }
        
        # Pre-calibration metrics
        pre_ece = self._calculate_ece(y_true, y_prob, 10)
        pre_log_loss = self._calculate_log_loss(y_true, y_prob)
        pre_brier = self._calculate_brier_score(y_true, y_prob)
        
        # Find optimal temperature
        def neg_log_likelihood(temp):
            scaled_probs = self._apply_temperature(y_prob, temp)
            return self._calculate_log_loss(y_true, scaled_probs)
        
        # Optimize temperature
        result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method='bounded')
        optimal_temp = result.x
        
        # Post-calibration metrics
        post_probs = self._apply_temperature(y_prob, optimal_temp)
        post_ece = self._calculate_ece(y_true, post_probs, 10)
        post_log_loss = self._calculate_log_loss(y_true, post_probs)
        post_brier = self._calculate_brier_score(y_true, post_probs)
        
        return {
            'pre_calibration': {
                'ece': pre_ece,
                'log_loss': pre_log_loss,
                'brier_score': pre_brier
            },
            'post_calibration': {
                'ece': post_ece,
                'log_loss': post_log_loss,
                'brier_score': post_brier
            },
            'temperature': optimal_temp,
            'improvement': {
                'ece': pre_ece - post_ece,
                'log_loss': pre_log_loss - post_log_loss,
                'brier_score': pre_brier - post_brier
            }
        }
    
    def _apply_temperature(self, y_prob: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities"""
        # Convert to logits, scale, then back to probabilities
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        scaled_logits = logits / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs 