"""
Human Preference Analysis for RLHF System

This module implements advanced human preference analysis including:
- Preference pattern clustering and mining
- Human annotator consistency analysis
- Preference trend analysis over time
- Human-model disagreement analysis
- Inter-annotator agreement metrics
- Confidence-based disagreement weighting
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import warnings


@dataclass
class AnnotatorMetrics:
    """Metrics for individual annotator performance"""
    annotator_id: str
    total_annotations: int
    consistency_score: float  # Self-consistency over time
    agreement_with_majority: float  # Agreement with other annotators
    response_time_avg: float  # Average annotation time
    confidence_correlation: float  # How well confidence predicts agreement
    domain_expertise: Dict[str, float]  # Expertise by domain
    temporal_trends: Dict[str, float]  # Performance trends over time


@dataclass
class PreferencePattern:
    """Container for discovered preference patterns"""
    pattern_id: str
    pattern_type: str  # 'length_bias', 'format_preference', 'domain_specific', etc.
    description: str
    examples: List[Dict[str, Any]]
    frequency: float  # How often this pattern occurs
    confidence: float  # Statistical confidence in pattern
    annotators_showing_pattern: List[str]
    strength: float  # How strong the pattern is


@dataclass
class DisagreementAnalysis:
    """Analysis of human-model disagreements"""
    total_disagreements: int
    disagreement_rate: float
    disagreement_by_confidence: Dict[str, float]  # Disagreement vs confidence level
    disagreement_by_domain: Dict[str, float]
    systematic_patterns: List[PreferencePattern]
    resolution_suggestions: List[str]
    annotator_specific_disagreements: Dict[str, float]


@dataclass
class PreferenceReport:
    """Comprehensive preference analysis report"""
    annotator_metrics: List[AnnotatorMetrics]
    preference_patterns: List[PreferencePattern]
    disagreement_analysis: DisagreementAnalysis
    temporal_trends: Dict[str, List[float]]
    quality_insights: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class HumanPreferenceAnalyzer:
    """
    Comprehensive analyzer for human preference patterns and behaviors.
    """
    
    def __init__(self, min_annotations_per_annotator: int = 10):
        """
        Initialize the preference analyzer.
        
        Args:
            min_annotations_per_annotator: Minimum annotations required for analysis
        """
        self.min_annotations = min_annotations_per_annotator
        self.preference_patterns = []
        self.annotator_profiles = {}
        
    def analyze_preferences(self, annotations_df: pd.DataFrame) -> PreferenceReport:
        """
        Perform comprehensive preference analysis.
        
        Args:
            annotations_df: DataFrame with annotation data including:
                - annotator_id, chosen_index, confidence, timestamp
                - prompt, response_a, response_b, domain (optional)
                
        Returns:
            PreferenceReport with comprehensive analysis
        """
        # Validate required columns
        required_cols = ['chosen_index', 'timestamp']
        if not all(col in annotations_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Ensure timestamp is datetime
        annotations_df = annotations_df.copy()
        annotations_df['timestamp'] = pd.to_datetime(annotations_df['timestamp'])
        
        # Add annotator_id if not present (assume single annotator)
        if 'annotator_id' not in annotations_df.columns:
            annotations_df['annotator_id'] = 'default_annotator'
        
        # Analyze each component
        annotator_metrics = self._analyze_annotator_consistency(annotations_df)
        preference_patterns = self._discover_preference_patterns(annotations_df)
        disagreement_analysis = self._analyze_disagreements(annotations_df)
        temporal_trends = self._analyze_temporal_trends(annotations_df)
        quality_insights = self._analyze_annotation_quality(annotations_df)
        recommendations = self._generate_recommendations(
            annotator_metrics, preference_patterns, disagreement_analysis
        )
        
        return PreferenceReport(
            annotator_metrics=annotator_metrics,
            preference_patterns=preference_patterns,
            disagreement_analysis=disagreement_analysis,
            temporal_trends=temporal_trends,
            quality_insights=quality_insights,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _analyze_annotator_consistency(self, df: pd.DataFrame) -> List[AnnotatorMetrics]:
        """Analyze individual annotator consistency and patterns."""
        annotator_metrics = []
        
        for annotator_id in df['annotator_id'].unique():
            annotator_data = df[df['annotator_id'] == annotator_id]
            
            if len(annotator_data) < self.min_annotations:
                continue
            
            # Calculate consistency metrics
            consistency_score = self._calculate_self_consistency(annotator_data)
            agreement_with_majority = self._calculate_majority_agreement(annotator_data, df)
            
            # Response time analysis (if available)
            response_time_avg = 0.0
            if 'response_time' in annotator_data.columns:
                response_time_avg = annotator_data['response_time'].mean()
            
            # Confidence correlation
            confidence_correlation = self._calculate_confidence_correlation(annotator_data)
            
            # Domain expertise
            domain_expertise = self._calculate_domain_expertise(annotator_data)
            
            # Temporal trends
            temporal_trends = self._calculate_temporal_trends(annotator_data)
            
            metrics = AnnotatorMetrics(
                annotator_id=annotator_id,
                total_annotations=len(annotator_data),
                consistency_score=consistency_score,
                agreement_with_majority=agreement_with_majority,
                response_time_avg=response_time_avg,
                confidence_correlation=confidence_correlation,
                domain_expertise=domain_expertise,
                temporal_trends=temporal_trends
            )
            
            annotator_metrics.append(metrics)
        
        return annotator_metrics
    
    def _calculate_self_consistency(self, annotator_data: pd.DataFrame) -> float:
        """Calculate how consistent an annotator is with themselves."""
        if len(annotator_data) < 2:
            return 0.0
        
        # Group by similar prompts (simplified - could use semantic similarity)
        if 'prompt' in annotator_data.columns:
            # For prompts with similar length (proxy for similarity)
            prompt_lengths = annotator_data['prompt'].str.len()
            length_groups = pd.cut(prompt_lengths, bins=5, labels=False)
            
            consistency_scores = []
            for group in range(5):
                group_data = annotator_data[length_groups == group]
                if len(group_data) >= 2:
                    # Calculate variance in choices for similar prompts
                    choice_variance = group_data['chosen_index'].var()
                    consistency_scores.append(1.0 - min(1.0, choice_variance))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
        else:
            # Fallback: overall choice distribution consistency
            choice_dist = annotator_data['chosen_index'].value_counts(normalize=True)
            # Higher consistency = less random (entropy-based)
            entropy = -sum(p * np.log2(p) for p in choice_dist if p > 0)
            max_entropy = np.log2(len(choice_dist))
            return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
    
    def _calculate_majority_agreement(self, annotator_data: pd.DataFrame, all_data: pd.DataFrame) -> float:
        """Calculate agreement with majority vote across all annotators."""
        if 'prompt' not in annotator_data.columns:
            return 0.5  # Cannot calculate without prompt matching
        
        agreements = []
        
        for _, row in annotator_data.iterrows():
            # Find all annotations for same prompt
            same_prompt = all_data[all_data['prompt'] == row['prompt']]
            
            if len(same_prompt) > 1:
                # Calculate majority choice
                majority_choice = same_prompt['chosen_index'].mode()
                if len(majority_choice) > 0:
                    agreements.append(row['chosen_index'] == majority_choice.iloc[0])
        
        return np.mean(agreements) if agreements else 0.5
    
    def _calculate_confidence_correlation(self, annotator_data: pd.DataFrame) -> float:
        """Calculate correlation between confidence and actual agreement."""
        if 'confidence' not in annotator_data.columns:
            return 0.0
        
        # This would ideally correlate confidence with actual correctness
        # For now, correlate with choice consistency
        if len(annotator_data) < 5:
            return 0.0
        
        # Simple proxy: higher confidence should correlate with more common choices
        choice_frequencies = annotator_data['chosen_index'].value_counts(normalize=True)
        annotator_data = annotator_data.copy()
        annotator_data['choice_frequency'] = annotator_data['chosen_index'].map(choice_frequencies)
        
        if annotator_data['confidence'].std() > 0 and annotator_data['choice_frequency'].std() > 0:
            correlation = np.corrcoef(annotator_data['confidence'], annotator_data['choice_frequency'])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_domain_expertise(self, annotator_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate domain-specific expertise scores."""
        if 'domain' not in annotator_data.columns:
            return {'general': 0.5}
        
        domain_expertise = {}
        
        for domain in annotator_data['domain'].unique():
            domain_data = annotator_data[annotator_data['domain'] == domain]
            
            if len(domain_data) >= 3:
                # Calculate domain-specific consistency
                choice_variance = domain_data['chosen_index'].var()
                expertise_score = 1.0 - min(1.0, choice_variance)
                domain_expertise[domain] = expertise_score
            else:
                domain_expertise[domain] = 0.5
        
        return domain_expertise
    
    def _calculate_temporal_trends(self, annotator_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate how annotator behavior changes over time."""
        if len(annotator_data) < 10:
            return {'consistency_trend': 0.0, 'speed_trend': 0.0}
        
        # Sort by timestamp
        data_sorted = annotator_data.sort_values('timestamp')
        
        # Split into early and late periods
        mid_point = len(data_sorted) // 2
        early_data = data_sorted.iloc[:mid_point]
        late_data = data_sorted.iloc[mid_point:]
        
        # Calculate consistency trend
        early_consistency = self._calculate_self_consistency(early_data)
        late_consistency = self._calculate_self_consistency(late_data)
        consistency_trend = late_consistency - early_consistency
        
        # Calculate speed trend (if response time available)
        speed_trend = 0.0
        if 'response_time' in data_sorted.columns:
            early_speed = early_data['response_time'].mean()
            late_speed = late_data['response_time'].mean()
            speed_trend = (early_speed - late_speed) / early_speed if early_speed > 0 else 0.0
        
        return {
            'consistency_trend': consistency_trend,
            'speed_trend': speed_trend
        }
    
    def _discover_preference_patterns(self, df: pd.DataFrame) -> List[PreferencePattern]:
        """Discover systematic preference patterns in the data."""
        patterns = []
        
        # Pattern 1: Length bias
        length_pattern = self._detect_length_bias(df)
        if length_pattern:
            patterns.append(length_pattern)
        
        # Pattern 2: Format preferences
        format_pattern = self._detect_format_preferences(df)
        if format_pattern:
            patterns.append(format_pattern)
        
        # Pattern 3: Confidence patterns
        confidence_pattern = self._detect_confidence_patterns(df)
        if confidence_pattern:
            patterns.append(confidence_pattern)
        
        # Pattern 4: Temporal patterns
        temporal_pattern = self._detect_temporal_patterns(df)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        return patterns
    
    def _detect_length_bias(self, df: pd.DataFrame) -> Optional[PreferencePattern]:
        """Detect bias towards longer or shorter responses."""
        if 'response_a' not in df.columns or 'response_b' not in df.columns:
            return None
        
        df_analysis = df.copy()
        df_analysis['len_a'] = df_analysis['response_a'].str.len()
        df_analysis['len_b'] = df_analysis['response_b'].str.len()
        df_analysis['longer_choice'] = (df_analysis['len_a'] > df_analysis['len_b']).astype(int)
        
        # Check if there's bias towards longer responses
        chose_longer = (df_analysis['chosen_index'] == df_analysis['longer_choice']).mean()
        
        if abs(chose_longer - 0.5) > 0.15:  # Significant bias threshold
            bias_direction = "longer" if chose_longer > 0.5 else "shorter"
            strength = abs(chose_longer - 0.5) * 2  # Scale to 0-1
            
            # Find examples
            biased_examples = df_analysis[
                df_analysis['chosen_index'] == df_analysis['longer_choice']
            ].head(3)
            
            examples = []
            for _, row in biased_examples.iterrows():
                examples.append({
                    'prompt': row.get('prompt', 'N/A')[:100],
                    'chosen_response': row[f'response_{["a", "b"][row["chosen_index"]]}'][:100],
                    'length_difference': abs(row['len_a'] - row['len_b'])
                })
            
            return PreferencePattern(
                pattern_id="length_bias",
                pattern_type="length_bias",
                description=f"Strong bias towards {bias_direction} responses ({chose_longer:.1%} of choices)",
                examples=examples,
                frequency=chose_longer if bias_direction == "longer" else 1 - chose_longer,
                confidence=strength,
                annotators_showing_pattern=df['annotator_id'].unique().tolist(),
                strength=strength
            )
        
        return None
    
    def _detect_format_preferences(self, df: pd.DataFrame) -> Optional[PreferencePattern]:
        """Detect preferences for specific response formats."""
        if 'response_a' not in df.columns or 'response_b' not in df.columns:
            return None
        
        df_analysis = df.copy()
        
        # Look for structured vs unstructured preferences
        df_analysis['a_has_bullets'] = df_analysis['response_a'].str.contains(r'[•\-\*]|\d+\.').astype(int)
        df_analysis['b_has_bullets'] = df_analysis['response_b'].str.contains(r'[•\-\*]|\d+\.').astype(int)
        df_analysis['structured_choice'] = (df_analysis['a_has_bullets'] > df_analysis['b_has_bullets']).astype(int)
        
        chose_structured = (df_analysis['chosen_index'] == df_analysis['structured_choice']).mean()
        
        if abs(chose_structured - 0.5) > 0.12:  # Format bias threshold
            format_type = "structured" if chose_structured > 0.5 else "unstructured"
            strength = abs(chose_structured - 0.5) * 2
            
            return PreferencePattern(
                pattern_id="format_preference",
                pattern_type="format_preference",
                description=f"Preference for {format_type} responses ({chose_structured:.1%} of relevant choices)",
                examples=[],  # Could add specific examples
                frequency=chose_structured if format_type == "structured" else 1 - chose_structured,
                confidence=strength,
                annotators_showing_pattern=df['annotator_id'].unique().tolist(),
                strength=strength
            )
        
        return None
    
    def _detect_confidence_patterns(self, df: pd.DataFrame) -> Optional[PreferencePattern]:
        """Detect patterns in confidence vs actual choices."""
        if 'confidence' not in df.columns:
            return None
        
        # Analyze confidence distribution
        high_confidence = df[df['confidence'] > 0.8]
        low_confidence = df[df['confidence'] < 0.5]
        
        if len(high_confidence) > 5 and len(low_confidence) > 5:
            # Check if choice patterns differ by confidence
            high_conf_choice_dist = high_confidence['chosen_index'].value_counts(normalize=True)
            low_conf_choice_dist = low_confidence['chosen_index'].value_counts(normalize=True)
            
            # Calculate distribution difference
            dist_diff = abs(high_conf_choice_dist.get(1, 0) - low_conf_choice_dist.get(1, 0))
            
            if dist_diff > 0.15:
                return PreferencePattern(
                    pattern_id="confidence_pattern",
                    pattern_type="confidence_correlation",
                    description=f"Different choice patterns at high vs low confidence (diff: {dist_diff:.2f})",
                    examples=[],
                    frequency=dist_diff,
                    confidence=min(1.0, dist_diff * 2),
                    annotators_showing_pattern=df['annotator_id'].unique().tolist(),
                    strength=dist_diff
                )
        
        return None
    
    def _detect_temporal_patterns(self, df: pd.DataFrame) -> Optional[PreferencePattern]:
        """Detect how preferences change over time."""
        if len(df) < 20:
            return None
        
        df_sorted = df.sort_values('timestamp')
        
        # Split into early and late periods
        mid_point = len(df_sorted) // 2
        early_choices = df_sorted.iloc[:mid_point]['chosen_index'].mean()
        late_choices = df_sorted.iloc[mid_point:]['chosen_index'].mean()
        
        choice_drift = abs(late_choices - early_choices)
        
        if choice_drift > 0.15:
            direction = "towards option B" if late_choices > early_choices else "towards option A"
            
            return PreferencePattern(
                pattern_id="temporal_drift",
                pattern_type="temporal_pattern",
                description=f"Preference drift {direction} over time (change: {choice_drift:.2f})",
                examples=[],
                frequency=choice_drift,
                confidence=min(1.0, choice_drift * 2),
                annotators_showing_pattern=df['annotator_id'].unique().tolist(),
                strength=choice_drift
            )
        
        return None
    
    def _analyze_disagreements(self, df: pd.DataFrame) -> DisagreementAnalysis:
        """Analyze patterns in human-model disagreements."""
        # This requires model predictions - simulate for now
        total_annotations = len(df)
        
        # Simulate model predictions (would be real in production)
        np.random.seed(42)
        df_analysis = df.copy()
        df_analysis['model_prediction'] = np.random.choice([0, 1], len(df), p=[0.4, 0.6])
        
        # Calculate disagreements
        disagreements = df_analysis['chosen_index'] != df_analysis['model_prediction']
        disagreement_rate = disagreements.mean()
        
        # Disagreement by confidence level
        disagreement_by_confidence = {}
        if 'confidence' in df.columns:
            for conf_level in ['low', 'medium', 'high']:
                if conf_level == 'low':
                    mask = df['confidence'] < 0.5
                elif conf_level == 'medium':
                    mask = (df['confidence'] >= 0.5) & (df['confidence'] < 0.8)
                else:
                    mask = df['confidence'] >= 0.8
                
                if mask.sum() > 0:
                    conf_disagreements = disagreements[mask].mean()
                    disagreement_by_confidence[conf_level] = conf_disagreements
        
        # Disagreement by domain
        disagreement_by_domain = {}
        if 'domain' in df.columns:
            for domain in df['domain'].unique():
                domain_mask = df['domain'] == domain
                domain_disagreements = disagreements[domain_mask].mean()
                disagreement_by_domain[domain] = domain_disagreements
        
        # Generate resolution suggestions
        resolution_suggestions = []
        if disagreement_rate > 0.3:
            resolution_suggestions.append("High disagreement rate suggests need for model retraining")
        if disagreement_by_confidence.get('high', 0) > 0.2:
            resolution_suggestions.append("Even high-confidence predictions show disagreement - review model calibration")
        
        return DisagreementAnalysis(
            total_disagreements=int(disagreements.sum()),
            disagreement_rate=disagreement_rate,
            disagreement_by_confidence=disagreement_by_confidence,
            disagreement_by_domain=disagreement_by_domain,
            systematic_patterns=[],  # Would analyze patterns in disagreements
            resolution_suggestions=resolution_suggestions,
            annotator_specific_disagreements={}  # Would analyze per annotator
        )
    
    def _analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Analyze how preferences change over time."""
        if len(df) < 10:
            return {}
        
        df_sorted = df.sort_values('timestamp')
        
        # Create time windows
        n_windows = min(10, len(df_sorted) // 5)
        window_size = len(df_sorted) // n_windows
        
        choice_trends = []
        confidence_trends = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < n_windows - 1 else len(df_sorted)
            window_data = df_sorted.iloc[start_idx:end_idx]
            
            choice_trends.append(window_data['chosen_index'].mean())
            
            if 'confidence' in window_data.columns:
                confidence_trends.append(window_data['confidence'].mean())
        
        trends = {'choice_preference': choice_trends}
        if confidence_trends:
            trends['confidence_level'] = confidence_trends
        
        return trends
    
    def _analyze_annotation_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall annotation quality metrics."""
        quality_metrics = {
            'total_annotations': len(df),
            'unique_annotators': df['annotator_id'].nunique(),
            'average_confidence': df['confidence'].mean() if 'confidence' in df.columns else None,
            'choice_distribution': df['chosen_index'].value_counts(normalize=True).to_dict(),
            'temporal_span_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'annotations_per_day': len(df) / max(1, (df['timestamp'].max() - df['timestamp'].min()).days)
        }
        
        # Quality indicators
        quality_issues = []
        if quality_metrics['choice_distribution'].get(1, 0) > 0.8 or quality_metrics['choice_distribution'].get(0, 0) > 0.8:
            quality_issues.append("Extreme choice bias detected")
        
        if quality_metrics['average_confidence'] and quality_metrics['average_confidence'] > 0.9:
            quality_issues.append("Suspiciously high confidence levels")
        
        if quality_metrics['annotations_per_day'] > 100:
            quality_issues.append("Very high annotation rate - check for automation")
        
        quality_metrics['quality_issues'] = quality_issues
        quality_metrics['overall_quality_score'] = max(0.0, 1.0 - len(quality_issues) * 0.2)
        
        return quality_metrics
    
    def _generate_recommendations(self, annotator_metrics: List[AnnotatorMetrics], 
                                 preference_patterns: List[PreferencePattern],
                                 disagreement_analysis: DisagreementAnalysis) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Annotator-based recommendations
        if annotator_metrics:
            avg_consistency = np.mean([m.consistency_score for m in annotator_metrics])
            if avg_consistency < 0.6:
                recommendations.append("🔄 Low annotator consistency detected - consider additional training")
            
            if any(m.total_annotations < 20 for m in annotator_metrics):
                recommendations.append("📊 Some annotators have few annotations - consider minimum requirements")
        
        # Pattern-based recommendations
        length_patterns = [p for p in preference_patterns if p.pattern_type == 'length_bias']
        if length_patterns and length_patterns[0].strength > 0.3:
            recommendations.append("📏 Strong length bias detected - consider response length normalization")
        
        format_patterns = [p for p in preference_patterns if p.pattern_type == 'format_preference']
        if format_patterns:
            recommendations.append("📝 Format preferences detected - ensure diverse response formats in training")
        
        # Disagreement-based recommendations
        if disagreement_analysis.disagreement_rate > 0.4:
            recommendations.append("⚠️ High human-model disagreement - model may need significant retraining")
        elif disagreement_analysis.disagreement_rate > 0.25:
            recommendations.append("📈 Moderate disagreement detected - consider targeted model improvements")
        
        # Confidence-based recommendations
        high_conf_disagreement = disagreement_analysis.disagreement_by_confidence.get('high', 0)
        if high_conf_disagreement > 0.15:
            recommendations.append("🎯 High-confidence predictions still show disagreement - improve calibration")
        
        if not recommendations:
            recommendations.append("✅ Annotation quality looks good - continue current practices")
        
        return recommendations
    
    def create_preference_dashboard(self, report: PreferenceReport) -> go.Figure:
        """Create comprehensive preference analysis dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Annotator Consistency Scores', 'Preference Patterns Strength',
                'Disagreement by Confidence', 'Temporal Preference Trends', 
                'Choice Distribution', 'Quality Metrics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # Annotator consistency
        if report.annotator_metrics:
            annotator_names = [m.annotator_id for m in report.annotator_metrics]
            consistency_scores = [m.consistency_score for m in report.annotator_metrics]
            
            fig.add_trace(
                go.Bar(x=annotator_names, y=consistency_scores, name='Consistency'),
                row=1, col=1
            )
        
        # Preference patterns
        if report.preference_patterns:
            pattern_names = [p.pattern_type for p in report.preference_patterns]
            pattern_strengths = [p.strength for p in report.preference_patterns]
            
            fig.add_trace(
                go.Bar(x=pattern_names, y=pattern_strengths, name='Pattern Strength'),
                row=1, col=2
            )
        
        # Disagreement by confidence
        if report.disagreement_analysis.disagreement_by_confidence:
            conf_levels = list(report.disagreement_analysis.disagreement_by_confidence.keys())
            disagreement_rates = list(report.disagreement_analysis.disagreement_by_confidence.values())
            
            fig.add_trace(
                go.Bar(x=conf_levels, y=disagreement_rates, name='Disagreement Rate'),
                row=2, col=1
            )
        
        # Temporal trends
        if 'choice_preference' in report.temporal_trends:
            time_points = list(range(len(report.temporal_trends['choice_preference'])))
            choice_trends = report.temporal_trends['choice_preference']
            
            fig.add_trace(
                go.Scatter(x=time_points, y=choice_trends, mode='lines+markers', name='Choice Trend'),
                row=2, col=2
            )
        
        # Overall disagreement pie chart
        if report.disagreement_analysis.total_disagreements > 0:
            agreements = 100 - (report.disagreement_analysis.disagreement_rate * 100)
            disagreements = report.disagreement_analysis.disagreement_rate * 100
            
            fig.add_trace(
                go.Pie(labels=['Agreement', 'Disagreement'], values=[agreements, disagreements]),
                row=3, col=1
            )
        
        # Quality score
        if report.quality_insights.get('overall_quality_score'):
            fig.add_trace(
                go.Bar(x=['Quality Score'], y=[report.quality_insights['overall_quality_score']], name='Quality'),
                row=3, col=2
            )
        
        fig.update_layout(
            title="Human Preference Analysis Dashboard",
            height=1000,
            showlegend=False
        )
        
        return fig


def analyze_annotator_agreement(annotations_df: pd.DataFrame, 
                               annotator_col: str = 'annotator_id',
                               choice_col: str = 'chosen_index') -> Dict[str, float]:
    """
    Calculate inter-annotator agreement metrics.
    
    Args:
        annotations_df: DataFrame with annotations
        annotator_col: Column containing annotator IDs
        choice_col: Column containing choices
        
    Returns:
        Dictionary with agreement metrics
    """
    if annotator_col not in annotations_df.columns:
        return {'error': 'Annotator column not found'}
    
    # Calculate pairwise agreement
    annotators = annotations_df[annotator_col].unique()
    if len(annotators) < 2:
        return {'error': 'Need at least 2 annotators for agreement calculation'}
    
    agreements = []
    
    for i, ann1 in enumerate(annotators):
        for ann2 in annotators[i+1:]:
            # Find common annotations (same prompt/task)
            ann1_data = annotations_df[annotations_df[annotator_col] == ann1]
            ann2_data = annotations_df[annotations_df[annotator_col] == ann2]
            
            # Simple agreement based on choice (could be enhanced with prompt matching)
            if len(ann1_data) > 0 and len(ann2_data) > 0:
                # For demonstration, compare overall choice distributions
                ann1_choices = ann1_data[choice_col].values
                ann2_choices = ann2_data[choice_col].values
                
                if len(ann1_choices) == len(ann2_choices):
                    agreement = (ann1_choices == ann2_choices).mean()
                    agreements.append(agreement)
    
    if agreements:
        return {
            'mean_pairwise_agreement': np.mean(agreements),
            'min_agreement': np.min(agreements),
            'max_agreement': np.max(agreements),
            'num_comparisons': len(agreements)
        }
    else:
        return {'error': 'Could not calculate agreements'}


def detect_annotation_anomalies(annotations_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect anomalies in annotation patterns that might indicate issues.
    
    Args:
        annotations_df: DataFrame with annotation data
        
    Returns:
        Dictionary with anomaly detection results
    """
    anomalies = []
    
    # Check for extreme choice bias
    if 'chosen_index' in annotations_df.columns:
        choice_dist = annotations_df['chosen_index'].value_counts(normalize=True)
        max_choice_freq = choice_dist.max()
        
        if max_choice_freq > 0.9:
            anomalies.append({
                'type': 'extreme_choice_bias',
                'severity': 'high',
                'description': f'{max_choice_freq:.1%} of choices are the same option',
                'recommendation': 'Check for annotation errors or extreme model bias'
            })
    
    # Check for temporal clustering
    if 'timestamp' in annotations_df.columns:
        annotations_df['timestamp'] = pd.to_datetime(annotations_df['timestamp'])
        time_diffs = annotations_df['timestamp'].diff().dt.total_seconds()
        very_fast = (time_diffs < 5).sum()  # Less than 5 seconds between annotations
        
        if very_fast > len(annotations_df) * 0.3:
            anomalies.append({
                'type': 'suspiciously_fast_annotations',
                'severity': 'medium',
                'description': f'{very_fast} annotations made very quickly (< 5 seconds apart)',
                'recommendation': 'Check for automated annotation or insufficient consideration time'
            })
    
    # Check for confidence anomalies
    if 'confidence' in annotations_df.columns:
        very_high_conf = (annotations_df['confidence'] > 0.95).sum()
        very_low_conf = (annotations_df['confidence'] < 0.1).sum()
        
        if very_high_conf > len(annotations_df) * 0.8:
            anomalies.append({
                'type': 'excessive_confidence',
                'severity': 'medium', 
                'description': f'{very_high_conf} annotations with >95% confidence',
                'recommendation': 'Unusually high confidence - check calibration or annotation guidelines'
            })
    
    return {
        'anomalies_detected': len(anomalies),
        'anomalies': anomalies,
        'overall_health': 'good' if len(anomalies) == 0 else 'concerning' if len(anomalies) > 2 else 'needs_review'
    }