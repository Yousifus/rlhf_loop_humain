"""
Real-time Monitoring for RLHF System

This module implements real-time monitoring capabilities including:
- Real-time calibration drift alerts
- Performance trend monitoring
- Automated alert management
- Performance prediction and forecasting
- Streaming analysis capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import queue
from collections import deque
import warnings


@dataclass
class Alert:
    """Container for monitoring alerts"""
    alert_id: str
    alert_type: str  # 'calibration_drift', 'performance_drop', 'data_quality', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    is_resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Container for real-time performance metrics"""
    timestamp: datetime
    accuracy: float
    calibration_error: float
    confidence_avg: float
    prediction_count: int
    drift_score: float = 0.0
    quality_score: float = 1.0


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    calibration_drift_threshold: float = 0.05
    performance_drop_threshold: float = 0.1
    alert_cooldown_minutes: int = 30
    metrics_window_size: int = 100
    prediction_horizon_hours: int = 24
    enable_auto_alerts: bool = True
    alert_callbacks: List[Callable] = field(default_factory=list)


class RealTimeMonitor:
    """
    Real-time monitor for RLHF system performance and calibration.
    """
    
    def __init__(self, config: MonitoringConfig = None):
        """
        Initialize the real-time monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.metrics_buffer = deque(maxlen=self.config.metrics_window_size)
        self.alert_manager = AlertManager(self.config)
        self.predictor = PerformancePredictor()
        
        # Threading components
        self.monitoring_thread = None
        self.is_monitoring = False
        self.data_queue = queue.Queue()
        
        # Baseline metrics for comparison
        self.baseline_metrics = None
        self.last_alert_times = {}
        
    def start_monitoring(self):
        """Start real-time monitoring in a separate thread."""
        if self.is_monitoring:
            warnings.warn("Monitoring is already running")
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
    def add_observation(self, y_true: int, y_prob: float, features: Dict[str, Any] = None):
        """
        Add a new observation for real-time monitoring.
        
        Args:
            y_true: True label (0 or 1)
            y_prob: Predicted probability
            features: Additional features for drift detection
        """
        observation = {
            'timestamp': datetime.now(),
            'y_true': y_true,
            'y_prob': y_prob,
            'features': features or {}
        }
        
        if self.is_monitoring:
            self.data_queue.put(observation)
        else:
            # Process immediately if not in monitoring mode
            self._process_observation(observation)
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        batch_size = 10
        batch_timeout = 5.0
        
        while self.is_monitoring:
            observations = []
            start_time = time.time()
            
            # Collect observations in batches
            while (len(observations) < batch_size and 
                   time.time() - start_time < batch_timeout and 
                   self.is_monitoring):
                try:
                    obs = self.data_queue.get(timeout=1.0)
                    observations.append(obs)
                except queue.Empty:
                    continue
            
            # Process batch
            if observations:
                self._process_batch(observations)
                
    def _process_observation(self, observation: Dict[str, Any]):
        """Process a single observation."""
        self._process_batch([observation])
        
    def _process_batch(self, observations: List[Dict[str, Any]]):
        """Process a batch of observations."""
        if not observations:
            return
            
        # Calculate metrics for this batch
        y_true_batch = [obs['y_true'] for obs in observations]
        y_prob_batch = [obs['y_prob'] for obs in observations]
        timestamp = observations[-1]['timestamp']  # Use latest timestamp
        
        # Calculate current metrics
        accuracy = np.mean([yt == (yp > 0.5) for yt, yp in zip(y_true_batch, y_prob_batch)])
        calibration_error = np.mean([abs(yp - yt) for yt, yp in zip(y_true_batch, y_prob_batch)])
        confidence_avg = np.mean(y_prob_batch)
        
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            accuracy=accuracy,
            calibration_error=calibration_error,
            confidence_avg=confidence_avg,
            prediction_count=len(observations)
        )
        
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Set baseline if this is the first batch
        if self.baseline_metrics is None and len(self.metrics_buffer) >= 10:
            self._set_baseline()
            
        # Check for alerts
        if self.baseline_metrics and self.config.enable_auto_alerts:
            self._check_alerts(metrics)
    
    def _set_baseline(self):
        """Set baseline metrics from initial observations."""
        if len(self.metrics_buffer) < 10:
            return
            
        recent_metrics = list(self.metrics_buffer)[-10:]
        
        self.baseline_metrics = {
            'accuracy': np.mean([m.accuracy for m in recent_metrics]),
            'calibration_error': np.mean([m.calibration_error for m in recent_metrics]),
            'confidence_avg': np.mean([m.confidence_avg for m in recent_metrics])
        }
        
    def _check_alerts(self, current_metrics: PerformanceMetrics):
        """Check for alert conditions."""
        now = datetime.now()
        
        # Calibration drift alert
        cal_drift = abs(current_metrics.calibration_error - self.baseline_metrics['calibration_error'])
        if cal_drift > self.config.calibration_drift_threshold:
            if self._should_alert('calibration_drift', now):
                alert = Alert(
                    alert_id=f"cal_drift_{int(now.timestamp())}",
                    alert_type='calibration_drift',
                    severity='high' if cal_drift > self.config.calibration_drift_threshold * 2 else 'medium',
                    message=f"Calibration drift detected: {cal_drift:.4f}",
                    details={'drift_magnitude': cal_drift, 'threshold': self.config.calibration_drift_threshold},
                    timestamp=now
                )
                self.alert_manager.add_alert(alert)
                self.last_alert_times['calibration_drift'] = now
        
        # Performance drop alert
        acc_drop = self.baseline_metrics['accuracy'] - current_metrics.accuracy
        if acc_drop > self.config.performance_drop_threshold:
            if self._should_alert('performance_drop', now):
                alert = Alert(
                    alert_id=f"perf_drop_{int(now.timestamp())}",
                    alert_type='performance_drop',
                    severity='critical' if acc_drop > self.config.performance_drop_threshold * 2 else 'high',
                    message=f"Performance drop detected: {acc_drop:.4f}",
                    details={'accuracy_drop': acc_drop, 'threshold': self.config.performance_drop_threshold},
                    timestamp=now
                )
                self.alert_manager.add_alert(alert)
                self.last_alert_times['performance_drop'] = now
    
    def _should_alert(self, alert_type: str, now: datetime) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if alert_type not in self.last_alert_times:
            return True
            
        time_since_last = now - self.last_alert_times[alert_type]
        return time_since_last.total_seconds() > self.config.alert_cooldown_minutes * 60
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.metrics_buffer:
            return {'status': 'no_data'}
            
        latest_metrics = self.metrics_buffer[-1]
        recent_metrics = list(self.metrics_buffer)[-10:] if len(self.metrics_buffer) >= 10 else list(self.metrics_buffer)
        
        # Calculate trends
        accuracy_trend = self._calculate_trend([m.accuracy for m in recent_metrics])
        calibration_trend = self._calculate_trend([m.calibration_error for m in recent_metrics])
        
        return {
            'status': 'monitoring',
            'latest_metrics': {
                'accuracy': latest_metrics.accuracy,
                'calibration_error': latest_metrics.calibration_error,
                'confidence_avg': latest_metrics.confidence_avg,
                'timestamp': latest_metrics.timestamp.isoformat()
            },
            'trends': {
                'accuracy': accuracy_trend,
                'calibration': calibration_trend
            },
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'baseline_set': self.baseline_metrics is not None,
            'buffer_size': len(self.metrics_buffer)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return 'stable'
            
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get performance predictions."""
        if len(self.metrics_buffer) < 20:
            return {'status': 'insufficient_data'}
            
        return self.predictor.predict_performance(list(self.metrics_buffer))


class AlertManager:
    """
    Manager for handling and routing alerts.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts = []
        self.alert_callbacks = config.alert_callbacks
        
    def add_alert(self, alert: Alert):
        """Add a new alert."""
        self.alerts.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.is_resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.is_resolved = True
                alert.resolution_timestamp = datetime.now()
                break
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        active_alerts = self.get_active_alerts()
        
        severity_counts = {}
        type_counts = {}
        
        for alert in active_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            'total_active': len(active_alerts),
            'total_all_time': len(self.alerts),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'latest_alert': active_alerts[-1].timestamp.isoformat() if active_alerts else None
        }


class PerformancePredictor:
    """
    Predictor for future performance trends.
    """
    
    def predict_performance(self, metrics_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        Predict future performance based on historical metrics.
        
        Args:
            metrics_history: List of historical performance metrics
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if len(metrics_history) < 20:
            return {'status': 'insufficient_data'}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        accuracies = [m.accuracy for m in metrics_history]
        calibration_errors = [m.calibration_error for m in metrics_history]
        
        # Simple trend-based prediction (can be enhanced with more sophisticated models)
        predictions = {}
        
        # Predict accuracy trend
        acc_trend = self._predict_trend(accuracies)
        predictions['accuracy'] = {
            'trend': acc_trend['trend'],
            'predicted_change': acc_trend['predicted_change'],
            'confidence': acc_trend['confidence']
        }
        
        # Predict calibration trend
        cal_trend = self._predict_trend(calibration_errors)
        predictions['calibration_error'] = {
            'trend': cal_trend['trend'],
            'predicted_change': cal_trend['predicted_change'],
            'confidence': cal_trend['confidence']
        }
        
        # Overall health prediction
        health_score = self._calculate_health_score(metrics_history[-10:])
        predictions['health_score'] = health_score
        
        # Risk assessment
        risk_level = self._assess_risk(predictions)
        predictions['risk_assessment'] = risk_level
        
        return {
            'status': 'predictions_available',
            'predictions': predictions,
            'prediction_timestamp': datetime.now().isoformat(),
            'based_on_samples': len(metrics_history)
        }
    
    def _predict_trend(self, values: List[float]) -> Dict[str, Any]:
        """Predict trend for a single metric."""
        if len(values) < 5:
            return {'trend': 'unknown', 'predicted_change': 0.0, 'confidence': 0.0}
        
        # Use simple linear regression for trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Predict next value
        next_x = len(values)
        predicted_value = slope * next_x + intercept
        current_value = values[-1]
        predicted_change = predicted_value - current_value
        
        # Calculate confidence based on R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'improving' if predicted_change > 0 else 'declining'
        else:
            trend = 'declining' if predicted_change < 0 else 'improving'
        
        return {
            'trend': trend,
            'predicted_change': predicted_change,
            'confidence': confidence
        }
    
    def _calculate_health_score(self, recent_metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall health score from recent metrics."""
        if not recent_metrics:
            return 0.0
        
        # Weight different aspects of health
        accuracy_score = np.mean([m.accuracy for m in recent_metrics])
        
        # Lower calibration error is better, so invert it
        cal_errors = [m.calibration_error for m in recent_metrics]
        calibration_score = 1.0 - min(1.0, np.mean(cal_errors))
        
        # Confidence should be reasonable (not too high, not too low)
        confidences = [m.confidence_avg for m in recent_metrics]
        confidence_score = 1.0 - abs(np.mean(confidences) - 0.7)  # Optimal around 0.7
        
        # Weighted health score
        health_score = (
            0.5 * accuracy_score +
            0.3 * calibration_score +
            0.2 * confidence_score
        )
        
        return max(0.0, min(1.0, health_score))
    
    def _assess_risk(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level based on predictions."""
        risks = []
        
        # Check accuracy trend
        acc_pred = predictions.get('accuracy', {})
        if acc_pred.get('trend') == 'declining' and acc_pred.get('confidence', 0) > 0.7:
            risks.append({
                'type': 'performance_decline',
                'severity': 'high' if acc_pred.get('predicted_change', 0) < -0.1 else 'medium',
                'description': 'Model accuracy is predicted to decline'
            })
        
        # Check calibration trend
        cal_pred = predictions.get('calibration_error', {})
        if cal_pred.get('trend') == 'declining' and cal_pred.get('confidence', 0) > 0.7:  # Declining means getting worse (higher error)
            risks.append({
                'type': 'calibration_degradation',
                'severity': 'medium',
                'description': 'Model calibration is predicted to worsen'
            })
        
        # Overall risk level
        if any(risk['severity'] == 'high' for risk in risks):
            overall_risk = 'high'
        elif any(risk['severity'] == 'medium' for risk in risks):
            overall_risk = 'medium'
        elif risks:
            overall_risk = 'low'
        else:
            overall_risk = 'minimal'
        
        return {
            'overall_risk': overall_risk,
            'specific_risks': risks,
            'risk_count': len(risks)
        }