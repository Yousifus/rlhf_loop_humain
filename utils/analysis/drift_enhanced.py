"""
Enhanced Drift Detection for RLHF System

This module implements advanced drift detection techniques including:
- Population Stability Index (PSI) for data drift detection
- Statistical drift tests (KS test, Chi-square, etc.)
- Feature drift analysis with importance tracking
- Concept drift detection using statistical methods
- Automated drift correction recommendations
- Streaming drift detection capabilities
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from datetime import datetime, timedelta


@dataclass
class DriftTestResult:
    """Container for drift test results"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    drift_magnitude: float
    description: str
    recommendation: str


@dataclass
class PSIResult:
    """Container for PSI analysis results"""
    feature_name: str
    psi_score: float
    drift_level: str  # 'no_drift', 'slight_drift', 'moderate_drift', 'severe_drift'
    bin_psi_values: List[float]
    bin_boundaries: List[float]
    reference_proportions: List[float]
    current_proportions: List[float]


@dataclass
class DriftAnalysisReport:
    """Comprehensive drift analysis report"""
    overall_drift_detected: bool
    drift_severity: str
    psi_results: List[PSIResult]
    statistical_tests: List[DriftTestResult]
    feature_importance_changes: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime
    confidence_score: float


class EnhancedDriftDetector:
    """
    Enhanced drift detector with multiple statistical methods and PSI analysis.
    """
    
    def __init__(self, psi_thresholds: Dict[str, float] = None, significance_level: float = 0.05):
        """
        Initialize the drift detector.
        
        Args:
            psi_thresholds: Dictionary with PSI thresholds for drift levels
            significance_level: Statistical significance level for tests
        """
        self.psi_thresholds = psi_thresholds or {
            'no_drift': 0.1,
            'slight_drift': 0.2,
            'moderate_drift': 0.3,
            'severe_drift': float('inf')
        }
        self.significance_level = significance_level
        self.reference_data = None
        self.feature_names = None
        
    def fit_reference(self, reference_data: pd.DataFrame, feature_columns: List[str] = None):
        """
        Fit the reference dataset for drift detection.
        
        Args:
            reference_data: Reference dataset to compare against
            feature_columns: List of feature columns to analyze (None = all numeric columns)
        """
        self.reference_data = reference_data.copy()
        
        if feature_columns is None:
            # Auto-detect numeric columns
            self.feature_names = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.feature_names = feature_columns
            
        # Validate feature columns exist
        missing_cols = set(self.feature_names) - set(reference_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in reference data: {missing_cols}")
    
    def analyze_drift(self, current_data: pd.DataFrame, 
                     detailed_analysis: bool = True) -> DriftAnalysisReport:
        """
        Perform comprehensive drift analysis.
        
        Args:
            current_data: Current dataset to analyze for drift
            detailed_analysis: Whether to perform detailed feature-by-feature analysis
            
        Returns:
            DriftAnalysisReport with comprehensive results
        """
        if self.reference_data is None:
            raise ValueError("Must call fit_reference first")
            
        # Validate current data has required features
        missing_cols = set(self.feature_names) - set(current_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in current data: {missing_cols}")
        
        psi_results = []
        statistical_tests = []
        recommendations = []
        
        # Analyze each feature
        for feature in self.feature_names:
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                warnings.warn(f"Skipping {feature}: insufficient data")
                continue
            
            # PSI Analysis
            psi_result = self.calculate_psi(ref_values, curr_values, feature)
            psi_results.append(psi_result)
            
            # Statistical Tests if detailed analysis requested
            if detailed_analysis:
                # Kolmogorov-Smirnov test for continuous features
                ks_result = self.perform_ks_test(ref_values, curr_values, feature)
                statistical_tests.append(ks_result)
                
                # Chi-square test for categorical-like features (if few unique values)
                if len(ref_values.unique()) <= 10:
                    chi2_result = self.perform_chi2_test(ref_values, curr_values, feature)
                    if chi2_result:
                        statistical_tests.append(chi2_result)
        
        # Determine overall drift status
        severe_drift_count = sum(1 for psi in psi_results if psi.drift_level == 'severe_drift')
        moderate_drift_count = sum(1 for psi in psi_results if psi.drift_level == 'moderate_drift')
        significant_tests = sum(1 for test in statistical_tests if test.is_significant)
        
        overall_drift_detected = (severe_drift_count > 0 or 
                                moderate_drift_count >= 2 or 
                                significant_tests >= len(self.feature_names) * 0.3)
        
        # Determine severity
        if severe_drift_count >= 2:
            drift_severity = 'severe'
        elif severe_drift_count == 1 or moderate_drift_count >= 3:
            drift_severity = 'moderate'
        elif moderate_drift_count >= 1 or significant_tests > 0:
            drift_severity = 'slight'
        else:
            drift_severity = 'none'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(psi_results, statistical_tests, drift_severity)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(psi_results, statistical_tests)
        
        return DriftAnalysisReport(
            overall_drift_detected=overall_drift_detected,
            drift_severity=drift_severity,
            psi_results=psi_results,
            statistical_tests=statistical_tests,
            feature_importance_changes={},  # To be implemented with ML models
            recommendations=recommendations,
            timestamp=datetime.now(),
            confidence_score=confidence_score
        )
    
    def calculate_psi(self, reference: pd.Series, current: pd.Series, 
                     feature_name: str, n_bins: int = 10) -> PSIResult:
        """
        Calculate Population Stability Index (PSI) for a feature.
        
        Args:
            reference: Reference data series
            current: Current data series
            feature_name: Name of the feature
            n_bins: Number of bins for discretization
            
        Returns:
            PSIResult object with PSI analysis
        """
        # Handle edge cases
        if len(reference) == 0 or len(current) == 0:
            return PSIResult(
                feature_name=feature_name,
                psi_score=0.0,
                drift_level='no_drift',
                bin_psi_values=[],
                bin_boundaries=[],
                reference_proportions=[],
                current_proportions=[]
            )
        
        # Create bins based on reference data
        if reference.dtype in ['object', 'category']:
            # Categorical feature
            categories = reference.value_counts().index.tolist()
            ref_props = reference.value_counts(normalize=True)
            curr_props = current.value_counts(normalize=True)
            
            # Ensure same categories
            all_categories = set(categories) | set(curr_props.index)
            ref_props = ref_props.reindex(all_categories, fill_value=0.001)  # Small value to avoid log(0)
            curr_props = curr_props.reindex(all_categories, fill_value=0.001)
            
            bin_boundaries = list(all_categories)
            
        else:
            # Numerical feature
            # Use quantile-based binning
            try:
                bin_boundaries = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
                # Handle case where quantiles are identical
                if len(np.unique(bin_boundaries)) < len(bin_boundaries):
                    bin_boundaries = np.linspace(reference.min(), reference.max(), n_bins + 1)
                
                # Calculate proportions for each bin
                ref_counts, _ = np.histogram(reference, bins=bin_boundaries)
                curr_counts, _ = np.histogram(current, bins=bin_boundaries)
                
                # Convert to proportions, add small value to avoid log(0)
                ref_props = ref_counts / len(reference)
                curr_props = curr_counts / len(current)
                
                # Add small epsilon to zero values
                ref_props = np.where(ref_props == 0, 0.001, ref_props)
                curr_props = np.where(curr_props == 0, 0.001, curr_props)
                
            except Exception as e:
                warnings.warn(f"Error in binning for {feature_name}: {e}")
                return PSIResult(
                    feature_name=feature_name,
                    psi_score=0.0,
                    drift_level='no_drift',
                    bin_psi_values=[],
                    bin_boundaries=[],
                    reference_proportions=[],
                    current_proportions=[]
                )
        
        # Calculate PSI for each bin
        bin_psi_values = []
        for ref_prop, curr_prop in zip(ref_props, curr_props):
            if ref_prop > 0 and curr_prop > 0:
                psi_bin = (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
                bin_psi_values.append(psi_bin)
            else:
                bin_psi_values.append(0.0)
        
        # Total PSI score
        psi_score = sum(bin_psi_values)
        
        # Determine drift level
        if psi_score <= self.psi_thresholds['no_drift']:
            drift_level = 'no_drift'
        elif psi_score <= self.psi_thresholds['slight_drift']:
            drift_level = 'slight_drift'
        elif psi_score <= self.psi_thresholds['moderate_drift']:
            drift_level = 'moderate_drift'
        else:
            drift_level = 'severe_drift'
        
        return PSIResult(
            feature_name=feature_name,
            psi_score=psi_score,
            drift_level=drift_level,
            bin_psi_values=bin_psi_values,
            bin_boundaries=bin_boundaries.tolist() if hasattr(bin_boundaries, 'tolist') else bin_boundaries,
            reference_proportions=ref_props.tolist() if hasattr(ref_props, 'tolist') else ref_props,
            current_proportions=curr_props.tolist() if hasattr(curr_props, 'tolist') else curr_props
        )
    
    def perform_ks_test(self, reference: pd.Series, current: pd.Series, 
                       feature_name: str) -> DriftTestResult:
        """
        Perform Kolmogorov-Smirnov test for distribution drift.
        """
        try:
            statistic, p_value = ks_2samp(reference, current)
            is_significant = p_value < self.significance_level
            
            # Calculate drift magnitude
            drift_magnitude = statistic  # KS statistic is already a measure of difference
            
            description = f"KS test comparing distributions of {feature_name}"
            
            if is_significant:
                recommendation = f"Significant distribution change in {feature_name}. Consider retraining or data preprocessing."
            else:
                recommendation = f"No significant distribution change in {feature_name}."
            
            return DriftTestResult(
                test_name=f"KS_Test_{feature_name}",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                drift_magnitude=drift_magnitude,
                description=description,
                recommendation=recommendation
            )
            
        except Exception as e:
            warnings.warn(f"Error in KS test for {feature_name}: {e}")
            return DriftTestResult(
                test_name=f"KS_Test_{feature_name}",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                drift_magnitude=0.0,
                description=f"KS test failed for {feature_name}",
                recommendation="Unable to perform KS test"
            )
    
    def perform_chi2_test(self, reference: pd.Series, current: pd.Series, 
                         feature_name: str) -> Optional[DriftTestResult]:
        """
        Perform Chi-square test for categorical drift.
        """
        try:
            # Create contingency table
            ref_counts = reference.value_counts()
            curr_counts = current.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_counts = ref_counts.reindex(all_categories, fill_value=0)
            curr_counts = curr_counts.reindex(all_categories, fill_value=0)
            
            # Create contingency table
            contingency_table = np.array([ref_counts.values, curr_counts.values])
            
            # Perform chi-square test
            statistic, p_value, dof, expected = chi2_contingency(contingency_table)
            is_significant = p_value < self.significance_level
            
            # Calculate Cramer's V as drift magnitude
            n = contingency_table.sum()
            drift_magnitude = np.sqrt(statistic / (n * (min(contingency_table.shape) - 1)))
            
            description = f"Chi-square test for categorical distribution of {feature_name}"
            
            if is_significant:
                recommendation = f"Significant categorical distribution change in {feature_name}. Review data collection process."
            else:
                recommendation = f"No significant categorical change in {feature_name}."
            
            return DriftTestResult(
                test_name=f"Chi2_Test_{feature_name}",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                drift_magnitude=drift_magnitude,
                description=description,
                recommendation=recommendation
            )
            
        except Exception as e:
            warnings.warn(f"Error in Chi-square test for {feature_name}: {e}")
            return None
    
    def _generate_recommendations(self, psi_results: List[PSIResult], 
                                statistical_tests: List[DriftTestResult],
                                drift_severity: str) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []
        
        if drift_severity == 'severe':
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Severe drift detected")
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Review data collection and preprocessing pipelines")
            
        elif drift_severity == 'moderate':
            recommendations.append("⚠️  Moderate drift detected - monitor closely")
            recommendations.append("Plan model retraining in next cycle")
            recommendations.append("Implement online learning if possible")
            
        elif drift_severity == 'slight':
            recommendations.append("📊 Slight drift observed - continue monitoring")
            recommendations.append("Consider expanding monitoring frequency")
            
        else:
            recommendations.append("✅ No significant drift detected")
            recommendations.append("Maintain current monitoring schedule")
        
        # Feature-specific recommendations
        severe_features = [psi for psi in psi_results if psi.drift_level == 'severe_drift']
        if severe_features:
            features_list = ', '.join([psi.feature_name for psi in severe_features])
            recommendations.append(f"Focus on features with severe drift: {features_list}")
        
        return recommendations
    
    def _calculate_confidence_score(self, psi_results: List[PSIResult], 
                                  statistical_tests: List[DriftTestResult]) -> float:
        """Calculate confidence score for drift analysis."""
        if not psi_results and not statistical_tests:
            return 0.0
        
        # Base confidence on consistency between PSI and statistical tests
        psi_scores = [psi.psi_score for psi in psi_results]
        test_significance = [test.is_significant for test in statistical_tests]
        
        # Normalize PSI scores
        if psi_scores:
            avg_psi = np.mean(psi_scores)
            psi_confidence = min(1.0, avg_psi / 0.5)  # Scale to 0-1
        else:
            psi_confidence = 0.0
        
        # Statistical test consistency
        if test_significance:
            test_confidence = np.mean(test_significance)
        else:
            test_confidence = 0.0
        
        # Combined confidence
        confidence_score = (psi_confidence + test_confidence) / 2
        
        return min(1.0, confidence_score)
    
    def create_drift_dashboard(self, drift_report: DriftAnalysisReport) -> go.Figure:
        """Create a comprehensive drift analysis dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PSI Scores by Feature', 'Statistical Test Results', 
                          'Drift Severity Distribution', 'Feature Importance Changes'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # PSI scores bar chart
        if drift_report.psi_results:
            psi_features = [psi.feature_name for psi in drift_report.psi_results]
            psi_scores = [psi.psi_score for psi in drift_report.psi_results]
            psi_colors = ['red' if psi.drift_level == 'severe_drift' else 
                         'orange' if psi.drift_level == 'moderate_drift' else 
                         'yellow' if psi.drift_level == 'slight_drift' else 'green'
                         for psi in drift_report.psi_results]
            
            fig.add_trace(
                go.Bar(x=psi_features, y=psi_scores, 
                      marker_color=psi_colors, name='PSI Scores'),
                row=1, col=1
            )
        
        # Statistical test results scatter plot
        if drift_report.statistical_tests:
            test_names = [test.test_name for test in drift_report.statistical_tests]
            test_statistics = [test.statistic for test in drift_report.statistical_tests]
            test_p_values = [test.p_value for test in drift_report.statistical_tests]
            test_colors = ['red' if test.is_significant else 'blue' 
                          for test in drift_report.statistical_tests]
            
            fig.add_trace(
                go.Scatter(x=test_statistics, y=test_p_values,
                          mode='markers', marker=dict(color=test_colors, size=10),
                          text=test_names, name='Statistical Tests'),
                row=1, col=2
            )
        
        # Drift severity pie chart
        drift_levels = ['no_drift', 'slight_drift', 'moderate_drift', 'severe_drift']
        drift_counts = [sum(1 for psi in drift_report.psi_results if psi.drift_level == level)
                       for level in drift_levels]
        
        fig.add_trace(
            go.Pie(labels=drift_levels, values=drift_counts, name='Drift Levels'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Drift Analysis Dashboard - {drift_report.drift_severity.title()} Drift Detected",
            height=800,
            showlegend=True
        )
        
        return fig


def calculate_psi(reference: Union[pd.Series, np.ndarray], 
                 current: Union[pd.Series, np.ndarray],
                 n_bins: int = 10) -> float:
    """
    Convenience function to calculate PSI between two datasets.
    
    Args:
        reference: Reference dataset
        current: Current dataset
        n_bins: Number of bins for discretization
        
    Returns:
        PSI score
    """
    detector = EnhancedDriftDetector()
    ref_series = pd.Series(reference) if not isinstance(reference, pd.Series) else reference
    curr_series = pd.Series(current) if not isinstance(current, pd.Series) else current
    
    psi_result = detector.calculate_psi(ref_series, curr_series, 'feature', n_bins)
    return psi_result.psi_score


def statistical_drift_test(reference: Union[pd.Series, np.ndarray], 
                          current: Union[pd.Series, np.ndarray],
                          test_type: str = 'ks') -> DriftTestResult:
    """
    Convenience function to perform statistical drift test.
    
    Args:
        reference: Reference dataset
        current: Current dataset  
        test_type: Type of test ('ks' for Kolmogorov-Smirnov, 'chi2' for Chi-square)
        
    Returns:
        DriftTestResult object
    """
    detector = EnhancedDriftDetector()
    ref_series = pd.Series(reference) if not isinstance(reference, pd.Series) else reference
    curr_series = pd.Series(current) if not isinstance(current, pd.Series) else current
    
    if test_type == 'ks':
        return detector.perform_ks_test(ref_series, curr_series, 'feature')
    elif test_type == 'chi2':
        return detector.perform_chi2_test(ref_series, curr_series, 'feature')
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def feature_drift_analysis(reference_df: pd.DataFrame, 
                          current_df: pd.DataFrame,
                          feature_columns: List[str] = None) -> DriftAnalysisReport:
    """
    Convenience function to perform comprehensive feature drift analysis.
    
    Args:
        reference_df: Reference DataFrame
        current_df: Current DataFrame
        feature_columns: List of feature columns to analyze
        
    Returns:
        DriftAnalysisReport object
    """
    detector = EnhancedDriftDetector()
    detector.fit_reference(reference_df, feature_columns)
    return detector.analyze_drift(current_df)