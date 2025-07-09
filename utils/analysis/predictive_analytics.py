"""
Predictive Analytics for RLHF System

This module implements predictive analytics capabilities including:
- Performance forecasting and trend prediction
- Training time estimation and early stopping prediction
- Data requirement prediction for target performance
- Annotation effort optimization and active learning
- Model capacity and scaling predictions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import warnings


@dataclass
class PerformanceForecast:
    """Container for performance prediction results"""
    metric_name: str  # 'accuracy', 'calibration_error', etc.
    current_value: float
    predicted_values: List[float]  # Future values
    prediction_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]  # CI for each prediction
    trend_direction: str  # 'improving', 'declining', 'stable'
    model_confidence: float  # Confidence in the prediction model
    factors_influencing: List[str]  # Key factors affecting the metric


@dataclass
class TrainingOptimization:
    """Training optimization recommendations"""
    estimated_training_time: float  # Hours to reach target
    data_requirements: Dict[str, int]  # Data needed for different targets
    early_stopping_recommendations: Dict[str, Any]
    learning_rate_suggestions: List[float]
    batch_size_optimization: Dict[str, Any]
    resource_allocation: Dict[str, float]


@dataclass
class ActiveLearningRecommendations:
    """Active learning and annotation optimization"""
    priority_prompts: List[Dict[str, Any]]  # High-value prompts to annotate
    annotation_effort_reduction: float  # Potential effort savings
    uncertainty_hotspots: List[Dict[str, Any]]  # Areas needing more data
    annotator_workload_balance: Dict[str, float]
    quality_vs_quantity_tradeoffs: Dict[str, Any]


@dataclass
class PredictiveReport:
    """Comprehensive predictive analytics report"""
    performance_forecasts: List[PerformanceForecast]
    training_optimization: TrainingOptimization
    active_learning: ActiveLearningRecommendations
    scaling_predictions: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class PerformancePredictor:
    """
    Advanced performance prediction and forecasting system.
    """
    
    def __init__(self, forecast_horizon_days: int = 30):
        """
        Initialize the performance predictor.
        
        Args:
            forecast_horizon_days: How far into the future to predict
        """
        self.forecast_horizon = forecast_horizon_days
        self.models = {}
        self.feature_importance = {}
        
    def create_predictive_report(self, 
                               historical_data: pd.DataFrame,
                               current_metrics: Dict[str, float] = None) -> PredictiveReport:
        """
        Create comprehensive predictive analytics report.
        
        Args:
            historical_data: Historical performance data with timestamps
            current_metrics: Current performance metrics
            
        Returns:
            PredictiveReport with forecasts and recommendations
        """
        # Validate data
        if len(historical_data) < 10:
            raise ValueError("Need at least 10 historical data points for predictions")
        
        # Ensure datetime index
        if 'timestamp' in historical_data.columns:
            historical_data = historical_data.set_index('timestamp')
        historical_data.index = pd.to_datetime(historical_data.index)
        
        # Generate forecasts for key metrics
        performance_forecasts = self._forecast_performance_metrics(historical_data)
        
        # Training optimization analysis
        training_optimization = self._analyze_training_optimization(historical_data)
        
        # Active learning recommendations
        active_learning = self._generate_active_learning_recommendations(historical_data)
        
        # Scaling predictions
        scaling_predictions = self._predict_scaling_requirements(historical_data)
        
        # Risk assessment
        risk_assessment = self._assess_prediction_risks(performance_forecasts, historical_data)
        
        # Generate recommendations
        recommendations = self._generate_predictive_recommendations(
            performance_forecasts, training_optimization, risk_assessment
        )
        
        return PredictiveReport(
            performance_forecasts=performance_forecasts,
            training_optimization=training_optimization,
            active_learning=active_learning,
            scaling_predictions=scaling_predictions,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
    
    def _forecast_performance_metrics(self, data: pd.DataFrame) -> List[PerformanceForecast]:
        """Forecast key performance metrics into the future."""
        forecasts = []
        
        # Define metrics to forecast
        metrics_to_forecast = []
        if 'accuracy' in data.columns:
            metrics_to_forecast.append('accuracy')
        if 'calibration_error' in data.columns:
            metrics_to_forecast.append('calibration_error')
        if 'confidence' in data.columns:
            metrics_to_forecast.append('confidence')
        
        for metric in metrics_to_forecast:
            if metric in data.columns and data[metric].notna().sum() >= 5:
                forecast = self._forecast_single_metric(data, metric)
                if forecast:
                    forecasts.append(forecast)
        
        return forecasts
    
    def _forecast_single_metric(self, data: pd.DataFrame, metric: str) -> Optional[PerformanceForecast]:
        """Forecast a single performance metric."""
        try:
            # Prepare time series data
            metric_data = data[metric].dropna()
            if len(metric_data) < 5:
                return None
            
            # Create time features
            time_index = np.arange(len(metric_data))
            X = time_index.reshape(-1, 1)
            y = metric_data.values
            
            # Try different models and select best
            models = {
                'linear': LinearRegression(),
                'polynomial': self._create_polynomial_model(degree=2),
                'ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_score = -np.inf
            
            for model_name, model in models.items():
                try:
                    if model_name == 'polynomial':
                        # Handle polynomial features separately
                        poly_features = PolynomialFeatures(degree=2)
                        X_poly = poly_features.fit_transform(X)
                        model.fit(X_poly, y)
                        score = r2_score(y, model.predict(X_poly))
                    else:
                        model.fit(X, y)
                        score = r2_score(y, model.predict(X))
                    
                    if score > best_score:
                        best_score = score
                        best_model = (model, model_name)
                except Exception:
                    continue
            
            if best_model is None:
                return None
            
            model, model_name = best_model
            
            # Generate future predictions
            future_steps = min(30, len(metric_data) // 2)  # Reasonable forecast horizon
            future_time = np.arange(len(metric_data), len(metric_data) + future_steps).reshape(-1, 1)
            
            if model_name == 'polynomial':
                poly_features = PolynomialFeatures(degree=2)
                poly_features.fit(X)  # Fit on original X
                future_X = poly_features.transform(future_time)
                predictions = model.predict(future_X)
            else:
                predictions = model.predict(future_time)
            
            # Generate prediction timestamps
            last_timestamp = data.index[-1]
            time_delta = (data.index[-1] - data.index[-2]) if len(data) > 1 else timedelta(days=1)
            prediction_timestamps = [
                last_timestamp + (i + 1) * time_delta for i in range(future_steps)
            ]
            
            # Calculate confidence intervals (simplified)
            residuals = y - model.predict(X if model_name != 'polynomial' else poly_features.transform(X))
            residual_std = np.std(residuals)
            confidence_intervals = [
                (pred - 1.96 * residual_std, pred + 1.96 * residual_std)
                for pred in predictions
            ]
            
            # Determine trend
            trend_direction = 'stable'
            if len(predictions) > 1:
                slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                if slope > 0.01:
                    trend_direction = 'improving' if metric != 'calibration_error' else 'declining'
                elif slope < -0.01:
                    trend_direction = 'declining' if metric != 'calibration_error' else 'improving'
            
            # Factors influencing (simplified analysis)
            factors_influencing = ['temporal_trend']
            if len(data.columns) > 1:
                factors_influencing.extend(['data_volume', 'training_iterations'])
            
            return PerformanceForecast(
                metric_name=metric,
                current_value=float(y[-1]),
                predicted_values=predictions.tolist(),
                prediction_timestamps=prediction_timestamps,
                confidence_intervals=confidence_intervals,
                trend_direction=trend_direction,
                model_confidence=max(0.0, min(1.0, best_score)),
                factors_influencing=factors_influencing
            )
            
        except Exception as e:
            warnings.warn(f"Error forecasting {metric}: {e}")
            return None
    
    def _create_polynomial_model(self, degree: int = 2):
        """Create a polynomial regression model."""
        return LinearRegression()  # Will be used with PolynomialFeatures
    
    def _analyze_training_optimization(self, data: pd.DataFrame) -> TrainingOptimization:
        """Analyze training process and provide optimization recommendations."""
        
        # Estimate training time to reach targets
        estimated_time = self._estimate_training_time(data)
        
        # Data requirements analysis
        data_requirements = self._analyze_data_requirements(data)
        
        # Early stopping recommendations
        early_stopping = self._recommend_early_stopping(data)
        
        # Learning rate suggestions
        learning_rates = self._suggest_learning_rates(data)
        
        # Batch size optimization
        batch_optimization = self._optimize_batch_size(data)
        
        # Resource allocation recommendations
        resource_allocation = self._recommend_resource_allocation(data)
        
        return TrainingOptimization(
            estimated_training_time=estimated_time,
            data_requirements=data_requirements,
            early_stopping_recommendations=early_stopping,
            learning_rate_suggestions=learning_rates,
            batch_size_optimization=batch_optimization,
            resource_allocation=resource_allocation
        )
    
    def _estimate_training_time(self, data: pd.DataFrame) -> float:
        """Estimate time to reach performance targets."""
        if 'accuracy' not in data.columns or len(data) < 5:
            return 24.0  # Default estimate in hours
        
        # Analyze improvement rate
        accuracy_values = data['accuracy'].dropna()
        if len(accuracy_values) < 3:
            return 24.0
        
        # Calculate improvement rate per time period
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 3600  # Hours
        accuracy_diffs = accuracy_values.diff()
        
        # Filter out invalid data
        valid_mask = (time_diffs > 0) & (accuracy_diffs.notna())
        if valid_mask.sum() < 2:
            return 24.0
        
        improvement_rate = accuracy_diffs[valid_mask].mean() / time_diffs[valid_mask].mean()
        
        # Estimate time to reach 90% accuracy (or 10% improvement from current)
        current_accuracy = accuracy_values.iloc[-1]
        target_accuracy = min(0.9, current_accuracy + 0.1)
        
        if improvement_rate > 0:
            estimated_hours = (target_accuracy - current_accuracy) / improvement_rate
            return max(1.0, min(168.0, estimated_hours))  # Between 1 hour and 1 week
        
        return 24.0
    
    def _analyze_data_requirements(self, data: pd.DataFrame) -> Dict[str, int]:
        """Analyze data requirements for different performance targets."""
        current_samples = len(data)
        
        # Estimate samples needed for different accuracy targets
        # This is a simplified model - in practice would be more sophisticated
        base_samples = max(100, current_samples)
        
        requirements = {
            'maintain_current': current_samples,
            'improve_5_percent': int(base_samples * 1.5),
            'improve_10_percent': int(base_samples * 2.0),
            'reach_90_percent_accuracy': int(base_samples * 3.0),
            'production_ready': int(base_samples * 5.0)
        }
        
        return requirements
    
    def _recommend_early_stopping(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Recommend early stopping strategies."""
        if 'accuracy' not in data.columns:
            return {
                'enable_early_stopping': False,
                'patience': 10,
                'min_delta': 0.01,
                'monitor_metric': 'accuracy'
            }
        
        # Analyze convergence patterns
        accuracy_values = data['accuracy'].dropna()
        if len(accuracy_values) < 10:
            return {
                'enable_early_stopping': True,
                'patience': 5,
                'min_delta': 0.01,
                'monitor_metric': 'accuracy'
            }
        
        # Calculate recent improvement
        recent_window = min(10, len(accuracy_values) // 3)
        recent_improvement = accuracy_values.iloc[-recent_window:].std()
        
        # Recommend based on convergence
        if recent_improvement < 0.01:
            patience = 3  # Stop quickly if converged
            min_delta = 0.005
        else:
            patience = 8  # Allow more training if still improving
            min_delta = 0.01
        
        return {
            'enable_early_stopping': True,
            'patience': patience,
            'min_delta': min_delta,
            'monitor_metric': 'accuracy',
            'estimated_convergence': recent_improvement < 0.01
        }
    
    def _suggest_learning_rates(self, data: pd.DataFrame) -> List[float]:
        """Suggest optimal learning rates based on training dynamics."""
        # Analyze training stability and suggest learning rates
        base_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        
        if 'accuracy' in data.columns and len(data) > 5:
            accuracy_values = data['accuracy'].dropna()
            
            # Check training stability
            accuracy_variance = accuracy_values.var()
            
            if accuracy_variance > 0.1:
                # High variance suggests learning rate too high
                suggested_rates = [1e-5, 2e-5, 5e-5]
            elif accuracy_variance < 0.01:
                # Low variance might mean learning rate too low
                suggested_rates = [5e-4, 1e-3, 2e-3]
            else:
                # Seems reasonable
                suggested_rates = [5e-5, 1e-4, 5e-4]
        else:
            suggested_rates = [5e-5, 1e-4, 5e-4]  # Conservative defaults
        
        return suggested_rates
    
    def _optimize_batch_size(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Recommend optimal batch sizes."""
        dataset_size = len(data)
        
        # Simple heuristics for batch size optimization
        if dataset_size < 100:
            optimal_batch = min(16, dataset_size // 4)
        elif dataset_size < 1000:
            optimal_batch = 32
        elif dataset_size < 10000:
            optimal_batch = 64
        else:
            optimal_batch = 128
        
        return {
            'recommended_batch_size': optimal_batch,
            'alternative_sizes': [optimal_batch // 2, optimal_batch, optimal_batch * 2],
            'memory_considerations': 'Adjust based on GPU memory availability',
            'training_speed_tradeoff': 'Larger batches = faster training but more memory'
        }
    
    def _recommend_resource_allocation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Recommend compute resource allocation."""
        dataset_size = len(data)
        
        # Estimate resource needs (simplified)
        if dataset_size < 1000:
            gpu_hours = 2.0
            cpu_cores = 4
            memory_gb = 8
        elif dataset_size < 10000:
            gpu_hours = 8.0
            cpu_cores = 8
            memory_gb = 16
        else:
            gpu_hours = 24.0
            cpu_cores = 16
            memory_gb = 32
        
        return {
            'estimated_gpu_hours': gpu_hours,
            'recommended_cpu_cores': cpu_cores,
            'memory_gb_required': memory_gb,
            'parallel_training_benefit': dataset_size > 5000
        }
    
    def _generate_active_learning_recommendations(self, data: pd.DataFrame) -> ActiveLearningRecommendations:
        """Generate active learning and annotation optimization recommendations."""
        
        # Priority prompts (simulate uncertainty-based selection)
        priority_prompts = self._identify_priority_prompts(data)
        
        # Estimate annotation effort reduction
        effort_reduction = self._estimate_effort_reduction(data)
        
        # Identify uncertainty hotspots
        uncertainty_hotspots = self._identify_uncertainty_hotspots(data)
        
        # Workload balancing
        workload_balance = self._optimize_annotator_workload(data)
        
        # Quality vs quantity tradeoffs
        quality_tradeoffs = self._analyze_quality_quantity_tradeoffs(data)
        
        return ActiveLearningRecommendations(
            priority_prompts=priority_prompts,
            annotation_effort_reduction=effort_reduction,
            uncertainty_hotspots=uncertainty_hotspots,
            annotator_workload_balance=workload_balance,
            quality_vs_quantity_tradeoffs=quality_tradeoffs
        )
    
    def _identify_priority_prompts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify high-priority prompts for annotation."""
        priorities = []
        
        # Simulate uncertainty-based selection
        if 'confidence' in data.columns:
            # Find low-confidence examples
            low_confidence = data[data['confidence'] < 0.6]
            for idx, row in low_confidence.head(5).iterrows():
                priorities.append({
                    'prompt_id': f"prompt_{idx}",
                    'confidence': row['confidence'],
                    'priority_score': 1.0 - row['confidence'],
                    'reason': 'Low model confidence',
                    'estimated_value': 'High learning potential'
                })
        
        # Add diversity-based selections
        priorities.append({
            'prompt_id': 'diverse_prompt_1',
            'priority_score': 0.8,
            'reason': 'Underrepresented domain',
            'estimated_value': 'Domain coverage improvement'
        })
        
        return priorities
    
    def _estimate_effort_reduction(self, data: pd.DataFrame) -> float:
        """Estimate potential annotation effort reduction through active learning."""
        # Simple heuristic: active learning can reduce effort by 20-40%
        if len(data) > 100:
            return 0.3  # 30% reduction for larger datasets
        else:
            return 0.2  # 20% reduction for smaller datasets
    
    def _identify_uncertainty_hotspots(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify areas with high uncertainty needing more data."""
        hotspots = []
        
        if 'domain' in data.columns and 'confidence' in data.columns:
            # Group by domain and find low-confidence areas
            domain_stats = data.groupby('domain')['confidence'].agg(['mean', 'count']).reset_index()
            
            for _, row in domain_stats.iterrows():
                if row['mean'] < 0.7 and row['count'] < 20:
                    hotspots.append({
                        'area': row['domain'],
                        'avg_confidence': row['mean'],
                        'sample_count': row['count'],
                        'recommendation': 'Needs more training data'
                    })
        
        return hotspots
    
    def _optimize_annotator_workload(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize workload distribution among annotators."""
        if 'annotator_id' not in data.columns:
            return {'single_annotator': 1.0}
        
        # Analyze current workload distribution
        workload = data['annotator_id'].value_counts(normalize=True)
        
        # Recommend more balanced distribution
        ideal_balance = 1.0 / len(workload)
        adjustments = {}
        
        for annotator, current_load in workload.items():
            adjustment = ideal_balance / current_load
            adjustments[annotator] = min(2.0, max(0.5, adjustment))  # Cap adjustments
        
        return adjustments
    
    def _analyze_quality_quantity_tradeoffs(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze quality vs quantity tradeoffs in annotation."""
        return {
            'current_quality_score': 0.8,  # Would calculate from actual data
            'annotation_speed_target': '10 annotations/hour',
            'quality_threshold': 0.85,
            'recommendations': [
                'Focus on quality over quantity for complex prompts',
                'Use batch annotation for similar prompts',
                'Implement quality checks every 50 annotations'
            ]
        }
    
    def _predict_scaling_requirements(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict scaling requirements for larger deployments."""
        current_size = len(data)
        
        scaling_predictions = {
            '10x_scale': {
                'data_size': current_size * 10,
                'training_time_multiplier': 3.2,  # Sub-linear scaling
                'memory_requirements': '2x current',
                'annotation_effort': current_size * 8  # Some efficiency gains
            },
            '100x_scale': {
                'data_size': current_size * 100,
                'training_time_multiplier': 10.0,
                'memory_requirements': '5x current',
                'annotation_effort': current_size * 60  # Greater efficiency gains
            },
            'production_scale': {
                'estimated_users': 10000,
                'daily_annotations_needed': 500,
                'infrastructure_requirements': 'Distributed training recommended',
                'quality_assurance': 'Automated QA systems required'
            }
        }
        
        return scaling_predictions
    
    def _assess_prediction_risks(self, forecasts: List[PerformanceForecast], 
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Assess risks associated with predictions."""
        risks = []
        
        # Check prediction confidence
        low_confidence_forecasts = [f for f in forecasts if f.model_confidence < 0.6]
        if low_confidence_forecasts:
            risks.append({
                'type': 'low_prediction_confidence',
                'severity': 'medium',
                'description': f'{len(low_confidence_forecasts)} forecasts have low confidence',
                'mitigation': 'Collect more historical data for better predictions'
            })
        
        # Check for declining trends
        declining_forecasts = [f for f in forecasts if f.trend_direction == 'declining']
        if declining_forecasts:
            risks.append({
                'type': 'performance_decline',
                'severity': 'high',
                'description': f'{len(declining_forecasts)} metrics showing decline',
                'mitigation': 'Investigate root causes and adjust training'
            })
        
        # Data sufficiency check
        if len(data) < 50:
            risks.append({
                'type': 'insufficient_data',
                'severity': 'high',
                'description': 'Limited historical data may affect prediction accuracy',
                'mitigation': 'Collect more data before making major decisions'
            })
        
        return {
            'risks_identified': len(risks),
            'risks': risks,
            'overall_risk_level': 'high' if any(r['severity'] == 'high' for r in risks) else 'medium' if risks else 'low'
        }
    
    def _generate_predictive_recommendations(self, forecasts: List[PerformanceForecast],
                                           training_opt: TrainingOptimization,
                                           risks: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        # Performance-based recommendations
        improving_metrics = [f for f in forecasts if f.trend_direction == 'improving']
        declining_metrics = [f for f in forecasts if f.trend_direction == 'declining']
        
        if improving_metrics:
            recommendations.append(f"✅ {len(improving_metrics)} metrics showing improvement - maintain current approach")
        
        if declining_metrics:
            recommendations.append(f"⚠️ {len(declining_metrics)} metrics declining - investigate and adjust training")
        
        # Training optimization recommendations
        if training_opt.estimated_training_time > 48:
            recommendations.append("⏱️ Training time estimate >48h - consider optimizing hyperparameters")
        
        if training_opt.early_stopping_recommendations['enable_early_stopping']:
            patience = training_opt.early_stopping_recommendations['patience']
            recommendations.append(f"🛑 Enable early stopping with patience={patience} to avoid overfitting")
        
        # Risk-based recommendations
        if risks['overall_risk_level'] == 'high':
            recommendations.append("🚨 High prediction risk detected - validate with additional data")
        
        # Data requirements
        current_data = training_opt.data_requirements['maintain_current']
        production_data = training_opt.data_requirements['production_ready']
        if production_data > current_data * 3:
            recommendations.append(f"📊 Need {production_data} samples for production (currently {current_data})")
        
        if not recommendations:
            recommendations.append("📈 Predictions look stable - continue monitoring trends")
        
        return recommendations
    
    def create_prediction_dashboard(self, report: PredictiveReport) -> go.Figure:
        """Create comprehensive predictive analytics dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Performance Forecasts', 'Training Time Estimates',
                'Data Requirements', 'Resource Allocation',
                'Risk Assessment', 'Confidence Scores'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ]
        )
        
        # Performance forecasts
        for forecast in report.performance_forecasts:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(forecast.predicted_values))),
                    y=forecast.predicted_values,
                    mode='lines+markers',
                    name=f'{forecast.metric_name} forecast'
                ),
                row=1, col=1
            )
        
        # Training time estimates
        training_times = [report.training_optimization.estimated_training_time]
        fig.add_trace(
            go.Bar(x=['Estimated Training Time'], y=training_times, name='Hours'),
            row=1, col=2
        )
        
        # Data requirements
        data_req = report.training_optimization.data_requirements
        req_names = list(data_req.keys())
        req_values = list(data_req.values())
        
        fig.add_trace(
            go.Bar(x=req_names, y=req_values, name='Samples Needed'),
            row=2, col=1
        )
        
        # Resource allocation
        resources = report.training_optimization.resource_allocation
        resource_names = list(resources.keys())
        resource_values = list(resources.values())
        
        fig.add_trace(
            go.Bar(x=resource_names, y=resource_values, name='Resource Units'),
            row=2, col=2
        )
        
        # Risk assessment pie chart
        if report.risk_assessment['risks']:
            risk_severities = [risk['severity'] for risk in report.risk_assessment['risks']]
            severity_counts = {sev: risk_severities.count(sev) for sev in set(risk_severities)}
            
            fig.add_trace(
                go.Pie(labels=list(severity_counts.keys()), values=list(severity_counts.values())),
                row=3, col=1
            )
        
        # Model confidence scores
        if report.performance_forecasts:
            metric_names = [f.metric_name for f in report.performance_forecasts]
            confidence_scores = [f.model_confidence for f in report.performance_forecasts]
            
            fig.add_trace(
                go.Bar(x=metric_names, y=confidence_scores, name='Confidence'),
                row=3, col=2
            )
        
        fig.update_layout(
            title="Predictive Analytics Dashboard",
            height=1200,
            showlegend=False
        )
        
        return fig


def predict_annotation_value(prompt_features: Dict[str, Any], 
                           historical_data: pd.DataFrame) -> Dict[str, float]:
    """
    Predict the value of annotating a specific prompt.
    
    Args:
        prompt_features: Features of the prompt to evaluate
        historical_data: Historical annotation data
        
    Returns:
        Dictionary with value prediction metrics
    """
    # Simplified value prediction
    base_value = 0.5
    
    # Adjust based on confidence if available
    if 'confidence' in prompt_features:
        # Lower confidence = higher annotation value
        confidence_value = 1.0 - prompt_features['confidence']
        base_value += confidence_value * 0.3
    
    # Adjust based on domain representation
    if 'domain' in prompt_features and 'domain' in historical_data.columns:
        domain = prompt_features['domain']
        domain_count = (historical_data['domain'] == domain).sum()
        total_count = len(historical_data)
        
        # Underrepresented domains have higher value
        if domain_count < total_count * 0.1:  # Less than 10% representation
            base_value += 0.2
    
    # Normalize to 0-1 range
    annotation_value = min(1.0, max(0.0, base_value))
    
    return {
        'annotation_value': annotation_value,
        'expected_improvement': annotation_value * 0.05,  # Expected accuracy improvement
        'priority_score': annotation_value,
        'recommendation': 'high_priority' if annotation_value > 0.7 else 'medium_priority' if annotation_value > 0.4 else 'low_priority'
    }


def estimate_model_capacity(current_performance: Dict[str, float],
                           data_size: int,
                           target_performance: Dict[str, float]) -> Dict[str, Any]:
    """
    Estimate if current model has capacity to reach target performance.
    
    Args:
        current_performance: Current performance metrics
        data_size: Current training data size
        target_performance: Target performance metrics
        
    Returns:
        Capacity estimation with recommendations
    """
    capacity_analysis = {}
    
    # Analyze each metric
    for metric, current_value in current_performance.items():
        if metric in target_performance:
            target_value = target_performance[metric]
            improvement_needed = abs(target_value - current_value)
            
            # Simple heuristic for capacity estimation
            if improvement_needed < 0.05:
                capacity = 'sufficient'
                data_multiplier = 1.2
            elif improvement_needed < 0.1:
                capacity = 'marginal'
                data_multiplier = 2.0
            else:
                capacity = 'insufficient'
                data_multiplier = 5.0
            
            capacity_analysis[metric] = {
                'current': current_value,
                'target': target_value,
                'gap': improvement_needed,
                'capacity_assessment': capacity,
                'data_size_multiplier': data_multiplier,
                'estimated_samples_needed': int(data_size * data_multiplier)
            }
    
    # Overall assessment
    insufficient_count = sum(1 for analysis in capacity_analysis.values() 
                           if analysis['capacity_assessment'] == 'insufficient')
    
    if insufficient_count > len(capacity_analysis) / 2:
        overall_capacity = 'insufficient'
        recommendation = 'Consider model architecture changes or significant data increase'
    elif insufficient_count > 0:
        overall_capacity = 'mixed'
        recommendation = 'Focus on underperforming metrics with targeted data collection'
    else:
        overall_capacity = 'sufficient'
        recommendation = 'Current model capacity appears adequate for targets'
    
    return {
        'overall_capacity': overall_capacity,
        'metric_analysis': capacity_analysis,
        'recommendation': recommendation,
        'max_data_multiplier': max([analysis['data_size_multiplier'] for analysis in capacity_analysis.values()]) if capacity_analysis else 1.0
    }