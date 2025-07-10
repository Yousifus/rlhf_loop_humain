"""
Performance & Scalability Optimizer for RLHF System

This module implements comprehensive performance optimization including:
- Large dataset analysis optimization and memory management
- Parallel processing for analysis tasks
- Incremental analysis updates and delta processing
- Analysis result memoization and caching strategies
- Real-time analytics with streaming capabilities
- Performance monitoring and bottleneck detection
"""

import numpy as np
import pandas as pd
import asyncio
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import time
import psutil
import multiprocessing as mp
from collections import deque
import warnings
import weakref
import gc


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    cache_hit_rate: float
    throughput_items_per_sec: float
    bottlenecks: List[str]
    optimization_suggestions: List[str]


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization"""
    enable_parallel_processing: bool = True
    max_workers: int = None
    chunk_size: int = 10000
    enable_memoization: bool = True
    cache_size_mb: int = 512
    enable_incremental_updates: bool = True
    memory_threshold_mb: int = 1024
    enable_streaming: bool = True
    batch_size: int = 1000


@dataclass
class StreamingAnalysisConfig:
    """Configuration for streaming analysis"""
    window_size: int = 1000
    slide_interval: int = 100
    aggregation_functions: List[str] = field(default_factory=lambda: ['mean', 'std', 'count'])
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class PerformanceOptimizer:
    """
    Advanced performance optimization system for RLHF analysis.
    
    Provides intelligent scaling, caching, parallel processing, and 
    real-time analytics capabilities for large-scale data analysis.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Set max workers based on system capabilities
        if self.config.max_workers is None:
            self.config.max_workers = min(8, mp.cpu_count())
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        # Memoization cache
        self.memo_cache = {}
        self.cache_access_times = {}
        
        # Incremental processing state
        self.incremental_state = {}
        self.data_fingerprints = {}
        
        # Streaming analysis components
        self.streaming_windows = {}
        self.stream_processors = {}
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
    def optimize_dataframe_analysis(self, 
                                  df: pd.DataFrame, 
                                  analysis_func: Callable,
                                  **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """
        Optimize DataFrame analysis with intelligent chunking and parallel processing.
        
        Args:
            df: DataFrame to analyze
            analysis_func: Analysis function to apply
            **kwargs: Additional arguments for analysis function
            
        Returns:
            Tuple of (analysis_result, performance_metrics)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Determine optimal processing strategy
        if len(df) <= self.config.chunk_size or not self.config.enable_parallel_processing:
            # Single-threaded processing for small datasets
            result = analysis_func(df, **kwargs)
            bottlenecks = []
        else:
            # Parallel processing for large datasets
            result, bottlenecks = self._parallel_dataframe_analysis(df, analysis_func, **kwargs)
        
        # Calculate performance metrics
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage()
        memory_usage = end_memory - start_memory
        cpu_utilization = psutil.cpu_percent(interval=0.1)
        
        # Calculate cache hit rate
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        # Calculate throughput
        throughput = len(df) / execution_time if execution_time > 0 else 0
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            len(df), execution_time, memory_usage, cpu_utilization
        )
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_utilization=cpu_utilization,
            cache_hit_rate=cache_hit_rate,
            throughput_items_per_sec=throughput,
            bottlenecks=bottlenecks,
            optimization_suggestions=optimization_suggestions
        )
        
        # Store performance history
        self.performance_history.append(metrics)
        
        return result, metrics
    
    def _parallel_dataframe_analysis(self, 
                                   df: pd.DataFrame, 
                                   analysis_func: Callable,
                                   **kwargs) -> Tuple[Any, List[str]]:
        """Execute DataFrame analysis in parallel chunks."""
        # Split DataFrame into chunks
        chunks = self._create_optimal_chunks(df)
        bottlenecks = []
        
        # Track chunk processing times
        chunk_times = []
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, analysis_func, **kwargs): i
                for i, chunk in enumerate(chunks)
            }
            
            chunk_results = {}
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_result, chunk_time = future.result()
                    chunk_results[chunk_idx] = chunk_result
                    chunk_times.append(chunk_time)
                except Exception as e:
                    print(f"Chunk {chunk_idx} failed: {e}")
                    bottlenecks.append(f"chunk_{chunk_idx}_error")
        
        # Analyze chunk performance
        if chunk_times:
            avg_time = np.mean(chunk_times)
            std_time = np.std(chunk_times)
            if std_time > avg_time * 0.5:  # High variance indicates load imbalance
                bottlenecks.append("load_imbalance")
        
        # Combine results
        if hasattr(analysis_func, 'combine_results'):
            # Custom combine function
            result = analysis_func.combine_results([chunk_results[i] for i in sorted(chunk_results.keys())])
        else:
            # Default combination for common result types
            result = self._combine_chunk_results(list(chunk_results.values()))
        
        return result, bottlenecks
    
    def _create_optimal_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Create optimal chunks based on data characteristics and system resources."""
        total_rows = len(df)
        
        # Estimate memory per row
        memory_per_row = df.memory_usage(deep=True).sum() / total_rows
        
        # Calculate optimal chunk size based on available memory
        available_memory = psutil.virtual_memory().available * 0.8  # Use 80% of available memory
        max_chunk_memory = available_memory / self.config.max_workers
        optimal_chunk_size = int(max_chunk_memory / memory_per_row)
        
        # Respect configured limits
        chunk_size = min(optimal_chunk_size, self.config.chunk_size, total_rows)
        chunk_size = max(chunk_size, 100)  # Minimum chunk size
        
        # Create chunks
        chunks = []
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunks.append(df.iloc[start:end].copy())
        
        return chunks
    
    def _process_chunk(self, 
                      chunk: pd.DataFrame, 
                      analysis_func: Callable,
                      **kwargs) -> Tuple[Any, float]:
        """Process a single chunk and return result with timing."""
        start_time = time.time()
        
        try:
            result = analysis_func(chunk, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            raise Exception(f"Chunk processing failed after {execution_time:.2f}s: {e}")
    
    def _combine_chunk_results(self, results: List[Any]) -> Any:
        """Combine results from parallel chunks."""
        if not results:
            return None
        
        # Handle different result types
        first_result = results[0]
        
        if isinstance(first_result, pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(first_result, dict):
            # Combine dictionaries by merging values
            combined = {}
            for result in results:
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = []
                    combined[key].append(value)
            
            # Average numeric values
            for key, values in combined.items():
                if all(isinstance(v, (int, float)) for v in values):
                    combined[key] = np.mean(values)
                elif all(isinstance(v, list) for v in values):
                    combined[key] = [item for sublist in values for item in sublist]
            
            return combined
        elif isinstance(first_result, (int, float)):
            # Average numeric results
            return np.mean(results)
        elif isinstance(first_result, list):
            # Concatenate lists
            combined = []
            for result in results:
                combined.extend(result)
            return combined
        else:
            # Return first result for unsupported types
            return first_result
    
    def memoize_analysis(self, 
                        cache_key: str = None, 
                        ttl_hours: int = 24):
        """
        Decorator for memoizing expensive analysis functions.
        
        Args:
            cache_key: Custom cache key (auto-generated if None)
            ttl_hours: Time to live for cache entries
            
        Returns:
            Decorated function with memoization
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_memoization:
                    return func(*args, **kwargs)
                
                # Generate cache key
                if cache_key is None:
                    key = self._generate_cache_key(func.__name__, args, kwargs)
                else:
                    key = cache_key
                
                # Check cache
                cached_result = self._get_from_cache(key, ttl_hours)
                if cached_result is not None:
                    self.cache_stats["hits"] += 1
                    return cached_result
                
                # Execute function
                self.cache_stats["misses"] += 1
                result = func(*args, **kwargs)
                
                # Store in cache
                self._store_in_cache(key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def incremental_analysis(self, 
                           data: pd.DataFrame,
                           analysis_func: Callable,
                           data_key: str,
                           **kwargs) -> Any:
        """
        Perform incremental analysis only on changed data.
        
        Args:
            data: Current dataset
            analysis_func: Analysis function to apply
            data_key: Unique identifier for this dataset
            **kwargs: Additional arguments for analysis function
            
        Returns:
            Analysis result (full or incremental)
        """
        if not self.config.enable_incremental_updates:
            return analysis_func(data, **kwargs)
        
        # Calculate data fingerprint
        current_fingerprint = self._calculate_data_fingerprint(data)
        
        # Check if data has changed
        if data_key in self.data_fingerprints:
            previous_fingerprint = self.data_fingerprints[data_key]
            
            if current_fingerprint == previous_fingerprint:
                # Data unchanged, return cached result
                if data_key in self.incremental_state:
                    return self.incremental_state[data_key]["result"]
        
        # Identify changes
        if data_key in self.incremental_state:
            previous_data = self.incremental_state[data_key]["data"]
            new_data, updated_data = self._identify_data_changes(previous_data, data)
            
            if hasattr(analysis_func, 'incremental_update'):
                # Function supports incremental updates
                previous_result = self.incremental_state[data_key]["result"]
                result = analysis_func.incremental_update(
                    previous_result, new_data, updated_data, **kwargs
                )
            else:
                # Full recomputation
                result = analysis_func(data, **kwargs)
        else:
            # First time analysis
            result = analysis_func(data, **kwargs)
        
        # Update state
        self.incremental_state[data_key] = {
            "data": data.copy(),
            "result": result,
            "timestamp": datetime.now()
        }
        self.data_fingerprints[data_key] = current_fingerprint
        
        return result
    
    def create_streaming_analyzer(self, 
                                config: StreamingAnalysisConfig) -> 'StreamingAnalyzer':
        """
        Create a streaming analysis processor for real-time data.
        
        Args:
            config: Streaming analysis configuration
            
        Returns:
            StreamingAnalyzer instance
        """
        analyzer = StreamingAnalyzer(config, self)
        return analyzer
    
    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning caches and running garbage collection."""
        # Clean expired cache entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, access_time in self.cache_access_times.items():
            if current_time - access_time > timedelta(hours=24):
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.memo_cache:
                del self.memo_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
        
        self.cache_stats["evictions"] += len(expired_keys)
        
        # Clean incremental state for old data
        old_keys = []
        for key, state in self.incremental_state.items():
            if current_time - state["timestamp"] > timedelta(days=7):
                old_keys.append(key)
        
        for key in old_keys:
            del self.incremental_state[key]
            if key in self.data_fingerprints:
                del self.data_fingerprints[key]
        
        # Force garbage collection
        gc.collect()
        
        print(f"🧹 Memory optimization: removed {len(expired_keys)} cache entries, {len(old_keys)} old states")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        recent_metrics = list(self.performance_history)[-100:]  # Last 100 executions
        
        avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cpu_utilization = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_items_per_sec for m in recent_metrics])
        
        # Cache statistics
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        # Common bottlenecks
        all_bottlenecks = [b for m in recent_metrics for b in m.bottlenecks]
        bottleneck_frequency = {}
        for bottleneck in all_bottlenecks:
            bottleneck_frequency[bottleneck] = bottleneck_frequency.get(bottleneck, 0) + 1
        
        # System resources
        system_info = {
            "cpu_count": mp.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "memory_percent_used": psutil.virtual_memory().percent
        }
        
        return {
            "performance_summary": {
                "avg_execution_time_sec": avg_execution_time,
                "avg_memory_usage_mb": avg_memory_usage,
                "avg_cpu_utilization_percent": avg_cpu_utilization,
                "avg_throughput_items_per_sec": avg_throughput
            },
            "cache_statistics": {
                "hit_rate": cache_hit_rate,
                "total_hits": self.cache_stats["hits"],
                "total_misses": self.cache_stats["misses"],
                "total_evictions": self.cache_stats["evictions"]
            },
            "bottlenecks": bottleneck_frequency,
            "system_resources": system_info,
            "optimization_config": {
                "parallel_processing": self.config.enable_parallel_processing,
                "max_workers": self.config.max_workers,
                "chunk_size": self.config.chunk_size,
                "cache_enabled": self.config.enable_memoization,
                "incremental_updates": self.config.enable_incremental_updates
            }
        }
    
    # Helper methods
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        import hashlib
        
        # Create string representation of arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        
        # Hash for shorter key
        key_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
        
        return f"{func_name}_{key_hash}"
    
    def _get_from_cache(self, key: str, ttl_hours: int) -> Any:
        """Retrieve value from cache if not expired."""
        if key not in self.memo_cache:
            return None
        
        # Check TTL
        if key in self.cache_access_times:
            age = datetime.now() - self.cache_access_times[key]
            if age > timedelta(hours=ttl_hours):
                del self.memo_cache[key]
                del self.cache_access_times[key]
                return None
        
        # Update access time
        self.cache_access_times[key] = datetime.now()
        return self.memo_cache[key]
    
    def _store_in_cache(self, key: str, value: Any):
        """Store value in cache."""
        self.memo_cache[key] = value
        self.cache_access_times[key] = datetime.now()
        
        # Manage cache size
        if len(self.memo_cache) > 1000:  # Simple size limit
            oldest_key = min(self.cache_access_times.keys(), 
                           key=lambda k: self.cache_access_times[k])
            del self.memo_cache[oldest_key]
            del self.cache_access_times[oldest_key]
            self.cache_stats["evictions"] += 1
    
    def _calculate_data_fingerprint(self, data: pd.DataFrame) -> str:
        """Calculate fingerprint for data change detection."""
        import hashlib
        
        # Use shape, column names, and sample of data for fingerprint
        fingerprint_data = [
            str(data.shape),
            str(list(data.columns)),
            str(data.dtypes.to_dict()),
            str(data.iloc[::max(1, len(data)//100)].sum().sum())  # Sample checksum
        ]
        
        fingerprint_str = '|'.join(fingerprint_data)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _identify_data_changes(self, 
                             previous_data: pd.DataFrame, 
                             current_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Identify new and updated data."""
        # Simplified change detection - would be more sophisticated in production
        if len(current_data) > len(previous_data):
            new_data = current_data.iloc[len(previous_data):]
            updated_data = pd.DataFrame()  # Simplified - no update detection
        else:
            new_data = pd.DataFrame()
            updated_data = pd.DataFrame()
        
        return new_data, updated_data
    
    def _generate_optimization_suggestions(self, 
                                         data_size: int,
                                         execution_time: float,
                                         memory_usage: float,
                                         cpu_utilization: float) -> List[str]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        if execution_time > 10 and data_size > 50000:
            suggestions.append("Consider enabling parallel processing for large datasets")
        
        if memory_usage > self.config.memory_threshold_mb:
            suggestions.append("High memory usage detected - consider chunking or streaming")
        
        if cpu_utilization < 50 and self.config.enable_parallel_processing:
            suggestions.append("Low CPU utilization - consider increasing worker count")
        
        if cpu_utilization > 90:
            suggestions.append("High CPU utilization - consider reducing parallel workers")
        
        cache_hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) if (self.cache_stats["hits"] + self.cache_stats["misses"]) > 0 else 0
        if cache_hit_rate < 0.3:
            suggestions.append("Low cache hit rate - consider enabling memoization")
        
        return suggestions


class StreamingAnalyzer:
    """
    Real-time streaming analysis processor.
    
    Processes continuous data streams with sliding windows and real-time aggregations.
    """
    
    def __init__(self, config: StreamingAnalysisConfig, optimizer: PerformanceOptimizer):
        """
        Initialize streaming analyzer.
        
        Args:
            config: Streaming configuration
            optimizer: Parent performance optimizer
        """
        self.config = config
        self.optimizer = optimizer
        self.window_data = deque(maxlen=config.window_size)
        self.processed_count = 0
        self.last_alert_time = {}
        self.streaming_results = deque(maxlen=1000)
        
    def add_data_point(self, data_point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a new data point and return analysis results if window is ready.
        
        Args:
            data_point: New data point to add
            
        Returns:
            Analysis results if window processing is triggered
        """
        self.window_data.append(data_point)
        self.processed_count += 1
        
        # Check if we should process the window
        if self.processed_count % self.config.slide_interval == 0:
            return self._process_window()
        
        return None
    
    def _process_window(self) -> Dict[str, Any]:
        """Process current window and return aggregated results."""
        if not self.window_data:
            return {}
        
        # Convert window to DataFrame for analysis
        df = pd.DataFrame(list(self.window_data))
        
        # Calculate aggregations
        results = {
            "timestamp": datetime.now().isoformat(),
            "window_size": len(self.window_data),
            "aggregations": {}
        }
        
        for column in df.select_dtypes(include=[np.number]).columns:
            column_aggs = {}
            for func_name in self.config.aggregation_functions:
                if hasattr(df[column], func_name):
                    value = getattr(df[column], func_name)()
                    column_aggs[func_name] = float(value) if not pd.isna(value) else None
            
            results["aggregations"][column] = column_aggs
        
        # Check for alerts
        alerts = self._check_alerts(results["aggregations"])
        if alerts:
            results["alerts"] = alerts
        
        # Store results
        self.streaming_results.append(results)
        
        return results
    
    def _check_alerts(self, aggregations: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Check for alert conditions in current aggregations."""
        alerts = []
        current_time = datetime.now()
        
        for metric, thresholds in self.config.alert_thresholds.items():
            if metric in aggregations:
                value = aggregations[metric].get('mean')
                if value is not None:
                    # Check threshold
                    threshold = thresholds.get('threshold')
                    condition = thresholds.get('condition', 'greater')  # greater, less, equal
                    
                    triggered = False
                    if condition == 'greater' and value > threshold:
                        triggered = True
                    elif condition == 'less' and value < threshold:
                        triggered = True
                    elif condition == 'equal' and abs(value - threshold) < 0.001:
                        triggered = True
                    
                    if triggered:
                        # Check cooldown period
                        cooldown = thresholds.get('cooldown_minutes', 5)
                        last_alert = self.last_alert_time.get(metric)
                        
                        if (last_alert is None or 
                            current_time - last_alert > timedelta(minutes=cooldown)):
                            
                            alerts.append({
                                "metric": metric,
                                "value": value,
                                "threshold": threshold,
                                "condition": condition,
                                "severity": thresholds.get('severity', 'warning'),
                                "timestamp": current_time.isoformat()
                            })
                            
                            self.last_alert_time[metric] = current_time
        
        return alerts
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent streaming analysis results."""
        return list(self.streaming_results)[-count:]


def create_demo_performance_optimization() -> PerformanceOptimizer:
    """
    Create demonstration of performance optimization capabilities.
    
    Returns:
        Configured performance optimizer with demo analysis
    """
    # Initialize optimizer
    config = OptimizationConfig(
        enable_parallel_processing=True,
        max_workers=4,
        chunk_size=5000,
        enable_memoization=True,
        enable_incremental_updates=True
    )
    
    optimizer = PerformanceOptimizer(config)
    
    # Create demo datasets of different sizes
    small_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'value': np.random.normal(100, 15, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    large_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50000, freq='min'),
        'value': np.random.normal(100, 15, 50000),
        'category': np.random.choice(['A', 'B', 'C'], 50000),
        'score': np.random.uniform(0, 1, 50000)
    })
    
    # Demo analysis function
    def demo_analysis(df):
        """Demo analysis function for performance testing."""
        result = {
            'mean_value': df['value'].mean(),
            'std_value': df['value'].std(),
            'category_counts': df['category'].value_counts().to_dict(),
            'row_count': len(df)
        }
        
        if 'score' in df.columns:
            result['mean_score'] = df['score'].mean()
            result['score_correlation'] = df['value'].corr(df['score'])
        
        return result
    
    # Test regular analysis
    print("🔄 Testing regular analysis...")
    result1, metrics1 = optimizer.optimize_dataframe_analysis(small_data, demo_analysis)
    print(f"✅ Small dataset: {metrics1.execution_time:.3f}s, {metrics1.memory_usage_mb:.1f}MB")
    
    result2, metrics2 = optimizer.optimize_dataframe_analysis(large_data, demo_analysis)
    print(f"✅ Large dataset: {metrics2.execution_time:.3f}s, {metrics2.memory_usage_mb:.1f}MB")
    
    # Test memoization
    print("\n🧠 Testing memoization...")
    
    @optimizer.memoize_analysis(ttl_hours=1)
    def memoized_analysis(df):
        time.sleep(0.1)  # Simulate expensive computation
        return demo_analysis(df)
    
    start_time = time.time()
    result3 = memoized_analysis(small_data)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result4 = memoized_analysis(small_data)  # Should be cached
    second_call_time = time.time() - start_time
    
    print(f"✅ First call: {first_call_time:.3f}s")
    print(f"✅ Cached call: {second_call_time:.3f}s (speedup: {first_call_time/second_call_time:.1f}x)")
    
    # Test incremental analysis
    print("\n📈 Testing incremental analysis...")
    
    def incremental_demo_analysis(df):
        return {
            'total_rows': len(df),
            'mean_value': df['value'].mean(),
            'latest_timestamp': df['timestamp'].max()
        }
    
    result5 = optimizer.incremental_analysis(small_data, incremental_demo_analysis, "demo_dataset")
    
    # Add more data
    additional_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-02-15', periods=500, freq='H'),
        'value': np.random.normal(110, 20, 500),
        'category': np.random.choice(['A', 'B', 'C'], 500)
    })
    
    extended_data = pd.concat([small_data, additional_data], ignore_index=True)
    result6 = optimizer.incremental_analysis(extended_data, incremental_demo_analysis, "demo_dataset")
    
    print(f"✅ Original data: {result5['total_rows']} rows")
    print(f"✅ Extended data: {result6['total_rows']} rows")
    
    # Test streaming analysis
    print("\n🌊 Testing streaming analysis...")
    
    streaming_config = StreamingAnalysisConfig(
        window_size=100,
        slide_interval=10,
        aggregation_functions=['mean', 'std', 'min', 'max'],
        alert_thresholds={
            'value': {'threshold': 120, 'condition': 'greater', 'severity': 'warning'}
        }
    )
    
    stream_analyzer = optimizer.create_streaming_analyzer(streaming_config)
    
    # Simulate streaming data
    streaming_results = []
    for i in range(150):
        data_point = {
            'timestamp': datetime.now(),
            'value': np.random.normal(100 + i * 0.2, 15),  # Gradually increasing trend
            'category': np.random.choice(['A', 'B', 'C'])
        }
        
        result = stream_analyzer.add_data_point(data_point)
        if result:
            streaming_results.append(result)
    
    print(f"✅ Processed 150 streaming points, generated {len(streaming_results)} window results")
    
    # Show alerts
    alerts = []
    for result in streaming_results:
        if 'alerts' in result:
            alerts.extend(result['alerts'])
    
    print(f"✅ Generated {len(alerts)} alerts")
    
    # Generate performance report
    print("\n📊 Performance Report:")
    report = optimizer.get_performance_report()
    
    print(f"Average execution time: {report['performance_summary']['avg_execution_time_sec']:.3f}s")
    print(f"Cache hit rate: {report['cache_statistics']['hit_rate']:.1%}")
    print(f"System CPU count: {report['system_resources']['cpu_count']}")
    print(f"Available memory: {report['system_resources']['memory_available_gb']:.1f}GB")
    
    if report['bottlenecks']:
        print(f"Common bottlenecks: {list(report['bottlenecks'].keys())}")
    
    # Memory optimization
    print("\n🧹 Running memory optimization...")
    optimizer.optimize_memory_usage()
    
    return optimizer


if __name__ == "__main__":
    print("⚡ Performance & Scalability Optimizer - Demo Mode")
    print("=" * 55)
    
    # Run demo
    optimizer = create_demo_performance_optimization()
    
    print(f"\n🎯 Performance Optimization Demo Complete!")
    print(f"Phase 3 Module 4: ✅ READY FOR INTEGRATION") 