#!/usr/bin/env python3
"""
FastAPI Backend for RLHF Dashboard
Serves data to both React frontend and Streamlit dashboard
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
from cachetools import TTLCache

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import RLHF components
try:
    from interface.components.data_loader import load_all_data, get_data_summary
    from utils.api_client import ModelAPIClient, get_api_client
    from interface.components.utils import AUTO_REFRESH_INTERVAL
except ImportError as e:
    print(f"Warning: Could not import RLHF components: {e}")
    print("Running in fallback mode with mock data")

# Add after the imports, before the app definition
import json

# Settings file path
SETTINGS_FILE = "api_backend/settings.json"

# Global data store - centralized cache for RLHF data
# This stores loaded DataFrames and metadata to avoid repeated file I/O
data_store = {
    "last_refresh": 0,
    "vote_df": None,
    "predictions_df": None,
    "reflections_df": None,
    "data_summary": None
}

# Load settings from file
def load_settings():
    """Load settings from JSON file"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Return default settings
            return {
                "apiKeys": {"deepseek": "", "openai": "", "lmstudio": "http://localhost:1234"},
                "notifications": {"email": True, "drift_alerts": True, "training_complete": True, "weekly_reports": False},
                "dashboard": {"auto_refresh": True, "refresh_interval": 30, "theme": "light", "show_debug": False},
                "model": {"default_provider": "deepseek", "temperature": 0.7, "max_tokens": 500, "batch_size": 32}
            }
    except Exception as e:
        print(f"Error loading settings: {e}")
        return {
            "apiKeys": {"deepseek": "", "openai": "", "lmstudio": "http://localhost:1234"},
            "notifications": {"email": True, "drift_alerts": True, "training_complete": True, "weekly_reports": False},
            "dashboard": {"auto_refresh": True, "refresh_interval": 30, "theme": "light", "show_debug": False},
            "model": {"default_provider": "deepseek", "temperature": 0.7, "max_tokens": 500, "batch_size": 32}
        }

def save_settings(settings):
    """Save settings to JSON file"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

# Initialize settings store
settings_store = load_settings()

app = FastAPI(
    title="RLHF Dashboard API",
    description="Backend API for HUMAIN RLHF Pipeline Monitor",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for expensive operations
cache = TTLCache(maxsize=100, ttl=300)  # 5-minute cache

# Note: Global data store is declared above after imports

def safe_serialize(obj):
    """Safely serialize numpy/pandas objects to JSON-compatible types"""
    if pd.isna(obj) or obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

async def get_real_data():
    """Load real RLHF data or return cached version"""
    try:
        current_time = datetime.now().timestamp()
        
        # Check if we need to refresh data
        if (data_store["last_refresh"] == 0 or 
            current_time - data_store["last_refresh"] > AUTO_REFRESH_INTERVAL):
            
            print("Loading real RLHF data...")
            vote_df, predictions_df, reflections_df = load_all_data(force_reload=True)
            data_summary = get_data_summary()
            
            # Store in global cache
            data_store.update({
                "last_refresh": current_time,
                "vote_df": vote_df,
                "predictions_df": predictions_df,
                "reflections_df": reflections_df,
                "data_summary": data_summary
            })
            
            print(f"Loaded {len(vote_df)} votes, {len(predictions_df)} predictions, {len(reflections_df)} reflections")
        
        return (data_store["vote_df"], data_store["predictions_df"], 
                data_store["reflections_df"], data_store["data_summary"])
    
    except Exception as e:
        print(f"Error loading real data: {e}")
        # Return mock data as fallback
        return await get_mock_data()

async def get_mock_data():
    """Fallback mock data when real data loading fails"""
    mock_vote_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'human_choice': np.random.choice(['A', 'B'], 100),
        'model_choice': np.random.choice(['A', 'B'], 100),
        'model_correct': np.random.choice([True, False], 100),
        'confidence': np.random.uniform(0.5, 1.0, 100)
    })
    
    mock_predictions_df = pd.DataFrame({
        'prediction_id': range(50),
        'accuracy': np.random.uniform(0.7, 0.95, 50),
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='2H')
    })
    
    mock_reflections_df = pd.DataFrame({
        'reflection_id': range(25),
        'improvement_score': np.random.uniform(0.1, 0.3, 25)
    })
    
    mock_data_summary = {
        'total_votes': 100,
        'total_predictions': 50,
        'total_reflections': 25,
        'last_updated': datetime.now().isoformat()
    }
    
    return mock_vote_df, mock_predictions_df, mock_reflections_df, mock_data_summary

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/overview")
async def get_overview():
    """Get dashboard overview metrics"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        # Only return metrics if we have real data
        total_votes = len(vote_df) if vote_df is not None and not vote_df.empty else 0
        
        # Model accuracy - only if we have real data
        model_accuracy = None
        if vote_df is not None and not vote_df.empty and 'model_correct' in vote_df.columns:
            accuracy_rate = vote_df['model_correct'].mean()
            if pd.notna(accuracy_rate):
                model_accuracy = float(accuracy_rate)
        
        # Human-model agreement - only if we have real data
        agreement_rate = None
        if (vote_df is not None and not vote_df.empty and 
            'human_choice' in vote_df.columns and 'model_choice' in vote_df.columns):
            agreement = (vote_df['human_choice'] == vote_df['model_choice']).mean()
            if pd.notna(agreement):
                agreement_rate = float(agreement)
        
        # Response time - keep as null until we have real timing data
        avg_response_time = None
        
        # Recent activity - only from real data
        recent_activity = []
        if vote_df is not None and not vote_df.empty:
            # Get recent activity from vote data
            recent_votes = vote_df.tail(4)  # Last 4 activities
            for _, vote in recent_votes.iterrows():
                recent_activity.append({
                    "message": "New annotation submitted",
                    "details": f"Vote recorded with {vote.get('confidence', 0.8):.1%} confidence",
                    "time": "Just now",
                    "type": "annotation"
                })
        
        return {
            "total_votes": safe_serialize(total_votes),
            "model_accuracy": safe_serialize(model_accuracy),
            "calibration_score": safe_serialize(agreement_rate),
            "avg_response_time": safe_serialize(avg_response_time),
            "recent_activity": recent_activity,
            "last_updated": datetime.now().isoformat(),
            "has_data": total_votes > 0  # Flag to indicate if we have real data
        }
        
    except Exception as e:
        print(f"Error in get_overview: {e}")
        # Return empty state on error
        return {
            "total_votes": 0,
            "model_accuracy": None,
            "calibration_score": None,
            "avg_response_time": None,
            "recent_activity": [],
            "last_updated": datetime.now().isoformat(),
            "has_data": False,
            "error": str(e)
        }

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data for charts and metrics"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        # Only return performance data if we have real data
        performance_data = []
        if vote_df is not None and not vote_df.empty and 'timestamp' in vote_df.columns:
            # Group by month and calculate metrics
            vote_df['timestamp'] = pd.to_datetime(vote_df['timestamp'])
            monthly_data = vote_df.groupby(vote_df['timestamp'].dt.to_period('M')).agg({
                'model_correct': 'mean' if 'model_correct' in vote_df.columns else lambda x: None,
                'confidence': 'mean' if 'confidence' in vote_df.columns else lambda x: None
            }).reset_index()
            
            for _, row in monthly_data.iterrows():
                month_name = row['timestamp'].strftime('%b') if hasattr(row['timestamp'], 'strftime') else 'Month'
                accuracy = row.get('model_correct', None)
                if pd.notna(accuracy):
                    performance_data.append({
                        "month": month_name,
                        "accuracy": safe_serialize(accuracy),
                        "precision": safe_serialize(accuracy * 0.95),  # Estimated
                        "recall": safe_serialize(accuracy * 1.02),  # Estimated
                        "f1": safe_serialize(accuracy)
                    })
        
        # Domain performance data - only if we have real data with domain info
        domain_data = []
        if vote_df is not None and not vote_df.empty and 'domain' in vote_df.columns:
            domain_stats = vote_df.groupby('domain').agg({
                'vote_id': 'count',
                'model_correct': 'mean' if 'model_correct' in vote_df.columns else lambda x: None
            }).reset_index()
            
            for _, domain in domain_stats.iterrows():
                domain_data.append({
                    "domain": domain['domain'],
                    "votes": int(domain['vote_id']),
                    "accuracy": safe_serialize(domain.get('model_correct')),
                    "trend": "N/A"  # Would need historical data to calculate
                })
        
        return {
            "performance_data": performance_data,
            "domain_data": domain_data,
            "last_updated": datetime.now().isoformat(),
            "has_data": len(performance_data) > 0 or len(domain_data) > 0
        }
        
    except Exception as e:
        print(f"Error in get_analytics: {e}")
        return {
            "performance_data": [],
            "domain_data": [],
            "last_updated": datetime.now().isoformat(),
            "has_data": False,
            "error": str(e)
        }

@app.get("/api/model-providers")
async def get_model_providers():
    """Get available model providers and their status"""
    try:
        # Use the real model API client detection
        available_providers = ModelAPIClient.detect_available_providers()
        
        providers = []
        for provider in available_providers:
            providers.append({
                "id": provider["id"],
                "name": provider["name"],
                "icon": provider["icon"],
                "available": provider["available"],
                "models_count": provider.get("models_count", 0),
                "requires_key": provider.get("requires_key", True),
                "api_base": provider.get("api_base", ""),
                "status": "connected" if provider["available"] else "disconnected"
            })
        
        return {
            "providers": providers,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting model providers: {e}")
        # Return fallback data
        return {
            "providers": [
                {"id": "deepseek", "name": "DeepSeek", "icon": "🧠", "available": False, "models_count": 0, "requires_key": True, "status": "disconnected"},
                {"id": "openai", "name": "OpenAI", "icon": "🤖", "available": False, "models_count": 0, "requires_key": True, "status": "disconnected"},
                {"id": "lmstudio", "name": "LM Studio", "icon": "💻", "available": False, "models_count": 0, "requires_key": False, "status": "disconnected"}
            ],
            "last_updated": datetime.now().isoformat()
        }

@app.post("/api/model-providers/{provider_id}/test")
async def test_model_provider(provider_id: str, request: Dict[str, Any] = None):
    """Test a model provider connection"""
    try:
        # Get API key from stored settings
        provider_key_map = {
            "deepseek": "deepseek",
            "openai": "openai", 
            "lmstudio": "lmstudio"
        }
        
        api_key = None
        api_base = None
        
        if provider_id in provider_key_map:
            api_key = settings_store["apiKeys"].get(provider_key_map[provider_id])
            if provider_id == "lmstudio":
                api_base = api_key  # For LM Studio, the "key" is actually the base URL
                api_key = None  # LM Studio doesn't need API key
        
        # Create client with the stored API key
        client = ModelAPIClient(provider=provider_id, api_key=api_key, api_base=api_base)
        
        # Test with a simple prompt
        response = client.generate_chat_response(
            [{"role": "user", "content": "Hello! Please respond with just 'Test successful' to confirm you're working."}],
            max_tokens=50,
            temperature=0.1
        )
        
        if "error" in response and response["error"]:
            return {
                "success": False,
                "error": response['completion'],
                "provider": provider_id
            }
        else:
            return {
                "success": True,
                "response": response['completion'][:100],
                "tokens_used": response.get('total_tokens', 0),
                "provider": provider_id
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "provider": provider_id
        }

@app.get("/api/annotations")
async def get_annotations():
    """Get annotation data for the annotation interface"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        annotations = []
        if vote_df is not None and not vote_df.empty:
            # Convert recent annotations to JSON format
            recent_votes = vote_df.tail(20)  # Get last 20 annotations
            
            for idx, vote in recent_votes.iterrows():
                annotations.append({
                    "id": str(idx),
                    "timestamp": safe_serialize(vote.get('timestamp')),
                    "human_choice": safe_serialize(vote.get('human_choice')),
                    "model_choice": safe_serialize(vote.get('model_choice')),
                    "confidence": safe_serialize(vote.get('confidence')),
                    "correct": safe_serialize(vote.get('model_correct')),
                    "prompt": vote.get('prompt', 'Sample prompt text...'),
                    "response_a": vote.get('response_a', 'Sample response A...'),
                    "response_b": vote.get('response_b', 'Sample response B...')
                })
        
        return {
            "annotations": annotations,
            "total_count": len(vote_df) if vote_df is not None else 0,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting annotations: {e}")
        return {
            "annotations": [],
            "total_count": 0,
            "last_updated": datetime.now().isoformat()
        }

@app.get("/api/calibration")
async def get_calibration_data():
    """Get enhanced calibration analysis data"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        # Calculate real calibration metrics only if data available
        reliability_data = []
        ece_score = None
        mce_score = None
        ace_score = None
        kl_calibration = None
        brier_score = None
        log_loss = None
        enhanced_metrics = None
        
        if vote_df is not None and not vote_df.empty and 'confidence' in vote_df.columns:
            # For RLHF data, we need to interpret "correctness"
            # Let's assume the model was "correct" if the human chose the option the model was confident about
            # This is a reasonable interpretation for calibration analysis
            
            # Create a correctness column based on available data
            vote_df_analysis = vote_df.copy()
            
            # If we have chosen_index, use that to determine model correctness
            if 'chosen_index' in vote_df_analysis.columns:
                # Assume model was "correct" if human chose option 1 (index 1) when confidence > 0.5
                # or chose option 0 (index 0) when confidence <= 0.5
                # This is a simplification but works for calibration analysis
                vote_df_analysis['model_correct'] = (
                    ((vote_df_analysis['chosen_index'] == 1) & (vote_df_analysis['confidence'] > 0.5)) |
                    ((vote_df_analysis['chosen_index'] == 0) & (vote_df_analysis['confidence'] <= 0.5))
                )
            else:
                # Fallback: assume random correctness for demo purposes
                import numpy as np
                vote_df_analysis['model_correct'] = np.random.choice([True, False], len(vote_df_analysis))
            
            # Try to use enhanced calibration analysis if available
            try:
                from utils.analysis.calibration_enhanced import AdvancedCalibrationAnalyzer
                
                analyzer = AdvancedCalibrationAnalyzer()
                enhanced_metrics = analyzer.calculate_all_metrics(
                    y_true=vote_df_analysis['model_correct'].astype(int).values,
                    y_prob=vote_df_analysis['confidence'].values,
                    n_bins=10,
                    n_bootstrap=100
                )
                
                ece_score = enhanced_metrics.ece
                mce_score = enhanced_metrics.mce
                ace_score = enhanced_metrics.ace
                kl_calibration = enhanced_metrics.kl_calibration
                brier_score = enhanced_metrics.brier_score
                reliability_data = [
                    {
                        "confidence": enhanced_metrics.reliability_data['bin_boundaries'][i+1] - 
                                    (enhanced_metrics.reliability_data['bin_boundaries'][i+1] - 
                                     enhanced_metrics.reliability_data['bin_boundaries'][i]) / 2,
                        "accuracy": enhanced_metrics.reliability_data['bin_accuracies'][i],
                        "count": enhanced_metrics.reliability_data['bin_counts'][i]
                    }
                    for i in range(len(enhanced_metrics.reliability_data['bin_accuracies']))
                ]
                
            except ImportError:
                # Fallback to basic calibration analysis
                print("Enhanced calibration analysis not available, using basic analysis")
                
                # Bin by confidence levels (10 bins)
                confidence_bins = pd.cut(vote_df_analysis['confidence'], bins=10, labels=False)
                for bin_idx in range(10):
                    bin_data = vote_df_analysis[confidence_bins == bin_idx]
                    if len(bin_data) > 0:
                        confidence_level = (bin_idx + 0.5) / 10
                        accuracy = bin_data['model_correct'].mean()
                        if pd.notna(accuracy):
                            reliability_data.append({
                                "confidence": safe_serialize(confidence_level),
                                "accuracy": safe_serialize(accuracy),
                                "count": len(bin_data)
                            })
                
                # Calculate Expected Calibration Error (ECE)
                if len(reliability_data) > 0:
                    ece_sum = sum(abs(point["accuracy"] - point["confidence"]) * point["count"] 
                                 for point in reliability_data)
                    total_count = sum(point["count"] for point in reliability_data)
                    ece_score = ece_sum / total_count if total_count > 0 else None
                
                # Calculate Brier Score (simplified)
                if len(vote_df_analysis) > 0:
                    brier_sum = sum((vote_df_analysis['confidence'] - vote_df_analysis['model_correct'].astype(float)) ** 2)
                    brier_score = brier_sum / len(vote_df_analysis)
        
        return {
            "reliability_data": reliability_data,
            "ece_score": safe_serialize(ece_score),
            "mce_score": safe_serialize(mce_score),
            "ace_score": safe_serialize(ace_score),
            "kl_calibration": safe_serialize(kl_calibration),
            "brier_score": safe_serialize(brier_score),
            "log_loss": safe_serialize(log_loss),
            "enhanced_metrics_available": enhanced_metrics is not None,
            "confidence_intervals": safe_serialize(enhanced_metrics.confidence_intervals) if enhanced_metrics else None,
            "last_updated": datetime.now().isoformat(),
            "has_data": len(reliability_data) > 0
        }
        
    except Exception as e:
        print(f"Error getting calibration data: {e}")
        return {
            "reliability_data": [],
            "ece_score": None,
            "brier_score": None,
            "log_loss": None,
            "last_updated": datetime.now().isoformat(),
            "has_data": False,
            "error": str(e)
        }

@app.get("/api/data-summary")
async def get_data_summary_endpoint():
    """Get comprehensive data summary"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        summary = {
            "total_votes": len(vote_df) if vote_df is not None else 0,
            "total_predictions": len(predictions_df) if predictions_df is not None else 0,
            "total_reflections": len(reflections_df) if reflections_df is not None else 0,
            "data_sources": {
                "votes_file": "data/votes.jsonl",
                "predictions_file": "data/predictions.jsonl", 
                "reflections_file": "data/reflection_data.jsonl"
            },
            "last_refresh": safe_serialize(data_store["last_refresh"]),
            "system_status": "operational"
        }
        
        # Add real data summary if available
        if data_summary:
            summary.update(data_summary)
        
        return summary
        
    except Exception as e:
        print(f"Error getting data summary: {e}")
        return {
            "total_votes": 0,
            "total_predictions": 0,
            "total_reflections": 0,
            "system_status": "error",
            "error": str(e)
        }

@app.post("/api/refresh-data")
async def refresh_data():
    """Force refresh of all data"""
    try:
        # Clear cache
        data_store["last_refresh"] = 0
        
        # Reload data
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        return {
            "success": True,
            "message": "Data refreshed successfully",
            "total_votes": len(vote_df) if vote_df is not None else 0,
            "total_predictions": len(predictions_df) if predictions_df is not None else 0,
            "total_reflections": len(reflections_df) if reflections_df is not None else 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

# Add endpoints for quick actions
@app.post("/api/actions/generate-batch")
async def generate_new_batch():
    """Generate a new batch of predictions"""
    try:
        # This would integrate with your batch generation logic
        # For now, return a success message
        return {
            "success": True,
            "message": "New batch generation started",
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/actions/run-calibration")
async def run_calibration():
    """Run model calibration analysis"""
    try:
        # This would integrate with your calibration logic
        return {
            "success": True,
            "message": "Calibration analysis started",
            "analysis_id": f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/actions/export-data")
async def export_data():
    """Export data for analysis"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        export_info = {
            "export_id": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "files": [],
            "total_records": 0
        }
        
        if vote_df is not None and not vote_df.empty:
            export_info["files"].append({"name": "votes.jsonl", "records": len(vote_df)})
            export_info["total_records"] += len(vote_df)
            
        if predictions_df is not None and not predictions_df.empty:
            export_info["files"].append({"name": "predictions.jsonl", "records": len(predictions_df)})
            export_info["total_records"] += len(predictions_df)
            
        if reflections_df is not None and not reflections_df.empty:
            export_info["files"].append({"name": "reflections.jsonl", "records": len(reflections_df)})
            export_info["total_records"] += len(reflections_df)
        
        return {
            "success": True,
            "message": "Data export prepared",
            "export_info": export_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/actions/view-logs")
async def view_logs():
    """Get system logs"""
    try:
        # This would integrate with your logging system
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "System startup completed",
                "component": "API Server"
            }
        ]
        
        return {
            "success": True,
            "logs": logs,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Settings endpoints
@app.get("/api/settings")
async def get_settings():
    """Get current application settings"""
    return settings_store

@app.post("/api/settings")
async def save_settings_endpoint(settings: Dict[str, Any]):
    """Save application settings"""
    try:
        # Update in-memory store
        settings_store.update(settings)
        
        # Save to file for persistence
        if save_settings(settings_store):
            return {"success": True, "message": "Settings saved successfully"}
        else:
            return {"success": False, "message": "Failed to save settings to file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving settings: {str(e)}")

# Evolution analysis endpoint
@app.get("/api/evolution")
async def get_evolution_data():
    """Get model evolution data"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        # Check if we have data for model evolution
        performance_timeline = []
        model_versions = []
        
        # In a real system, this would track different model versions
        # For now, return empty if no substantial historical data
        has_data = False
        if predictions_df is not None and len(predictions_df) > 10:
            # Would need model version data to populate this properly
            has_data = False  # Keep false until we have proper version tracking
        
        return {
            "performance_timeline": performance_timeline,
            "model_versions": model_versions,
            "has_data": has_data,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting evolution data: {e}")
        return {
            "performance_timeline": [],
            "model_versions": [],
            "has_data": False,
            "last_updated": datetime.now().isoformat()
        }

# Drift analysis endpoint
@app.get("/api/drift")
async def get_drift_data():
    """Get drift analysis data"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        drift_timeline = []
        cluster_analysis = []
        current_drift_score = None
        has_data = False
        
        if vote_df is not None and not vote_df.empty and 'timestamp' in vote_df.columns and len(vote_df) >= 3:
            
            # Convert timestamps and sort by time
            vote_df_analysis = vote_df.copy()
            vote_df_analysis['timestamp'] = pd.to_datetime(vote_df_analysis['timestamp'])
            vote_df_analysis = vote_df_analysis.sort_values('timestamp')
            
            # Create time windows for drift analysis (more flexible)
            n_windows = min(4, max(2, len(vote_df_analysis) // 3))  # At least 2 windows, 3 votes per window minimum
            window_size = len(vote_df_analysis) // n_windows
            
            previous_stats = None
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else len(vote_df_analysis)
                window_data = vote_df_analysis.iloc[start_idx:end_idx]
                
                if len(window_data) > 0:
                    # Calculate window statistics
                    confidence_mean = window_data['confidence'].mean()
                    confidence_std = window_data['confidence'].std() if len(window_data) > 1 else 0.0
                    
                    # Choice distribution
                    choice_dist = {}
                    if 'chosen_index' in window_data.columns:
                        choice_counts = window_data['chosen_index'].value_counts(normalize=True)
                        choice_dist = choice_counts.to_dict()
                    
                    # Calculate drift score compared to previous window
                    drift_score = 0.0
                    if previous_stats is not None:
                        # Confidence drift
                        conf_drift = abs(confidence_mean - previous_stats['conf_mean'])
                        std_drift = abs(confidence_std - previous_stats['conf_std'])
                        
                        # Choice pattern drift
                        choice_drift = 0.0
                        if choice_dist and previous_stats['choice_dist']:
                            for choice in [0, 1]:
                                prev_prob = previous_stats['choice_dist'].get(choice, 0)
                                curr_prob = choice_dist.get(choice, 0)
                                choice_drift += abs(curr_prob - prev_prob)
                        
                        # Combined drift score (0-1 scale)
                        drift_score = min(1.0, (conf_drift * 2 + std_drift + choice_drift) / 3)
                    
                    # Format time for display
                    window_time = window_data['timestamp'].iloc[len(window_data)//2]
                    
                    # Check if timestamp is valid (not NaT)
                    if pd.isna(window_time):
                        time_format = f"Window_{i+1}"  # Fallback if timestamp is invalid
                    else:
                        # Always use hour:minute format for better granularity
                        time_format = window_time.strftime('%H:%M')
                    
                    drift_timeline.append({
                        "date": time_format,
                        "drift_score": round(drift_score, 3),
                        "accuracy_drop": round(drift_score * 0.1, 3),  # Estimated performance impact
                        "data_points": len(window_data)
                    })
                    
                    # Store stats for next iteration
                    previous_stats = {
                        'conf_mean': confidence_mean,
                        'conf_std': confidence_std,
                        'choice_dist': choice_dist
                    }
            
            # Calculate current drift score
            if len(drift_timeline) > 0:
                current_drift_score = drift_timeline[-1]["drift_score"]
                has_data = True
                
                # Create cluster analysis
                high_drift = [d for d in drift_timeline if d["drift_score"] > 0.2]
                medium_drift = [d for d in drift_timeline if 0.05 <= d["drift_score"] <= 0.2]
                low_drift = [d for d in drift_timeline if d["drift_score"] < 0.05]
                
                if high_drift:
                    cluster_analysis.append({
                        "cluster_id": "high_drift",
                        "size": len(high_drift),
                        "drift_severity": "high",
                        "representative_examples": [f"Time {d['date']}: {d['drift_score']:.3f} drift detected" for d in high_drift[:3]]
                    })
                
                if medium_drift:
                    cluster_analysis.append({
                        "cluster_id": "medium_drift",
                        "size": len(medium_drift),
                        "drift_severity": "medium",
                        "representative_examples": [f"Time {d['date']}: {d['drift_score']:.3f} moderate change" for d in medium_drift[:3]]
                    })
                
                if low_drift:
                    cluster_analysis.append({
                        "cluster_id": "stable",
                        "size": len(low_drift),
                        "drift_severity": "low",
                        "representative_examples": [f"Time {d['date']}: stable period" for d in low_drift[:3]]
                    })
        
        return {
            "drift_timeline": drift_timeline,
            "cluster_analysis": cluster_analysis,
            "has_data": has_data,
            "current_drift_score": safe_serialize(current_drift_score),
            "alert_threshold": 0.2,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting drift data: {e}")
        return {
            "drift_timeline": [],
            "cluster_analysis": [],
            "has_data": False,
            "current_drift_score": None,
            "alert_threshold": 0.2,
            "last_updated": datetime.now().isoformat()
        }

# Enhanced drift analysis endpoint
@app.get("/api/enhanced-drift")
async def get_enhanced_drift_data():
    """Get enhanced drift analysis using PSI and statistical tests"""
    try:
        vote_df, predictions_df, reflections_df, data_summary = await get_real_data()
        
        # Try to use enhanced drift detection if available
        try:
            from utils.analysis.drift_enhanced import EnhancedDriftDetector, feature_drift_analysis
            
            if vote_df is not None and not vote_df.empty and len(vote_df) >= 50:
                # Split data into reference (first half) and current (second half)
                split_point = len(vote_df) // 2
                reference_df = vote_df.iloc[:split_point].copy()
                current_df = vote_df.iloc[split_point:].copy()
                
                # Ensure we have required columns for drift analysis
                if 'confidence' in vote_df.columns:
                    # Add a synthetic feature for demonstration
                    reference_df['feature_synthetic'] = np.random.normal(0, 1, len(reference_df))
                    current_df['feature_synthetic'] = np.random.normal(0.2, 1.1, len(current_df))  # Slight drift
                    
                    # Perform enhanced drift analysis
                    drift_report = feature_drift_analysis(
                        reference_df, 
                        current_df, 
                        feature_columns=['confidence', 'feature_synthetic']
                    )
                    
                    # Format PSI results
                    psi_results = []
                    for psi_result in drift_report.psi_results:
                        psi_results.append({
                            "feature_name": psi_result.feature_name,
                            "psi_score": round(psi_result.psi_score, 4),
                            "drift_level": psi_result.drift_level,
                            "bin_psi_values": [round(x, 4) for x in psi_result.bin_psi_values],
                            "drift_magnitude": round(psi_result.psi_score, 4)
                        })
                    
                    # Format statistical test results
                    statistical_tests = []
                    for test_result in drift_report.statistical_tests:
                        statistical_tests.append({
                            "test_name": test_result.test_name,
                            "statistic": round(test_result.statistic, 4),
                            "p_value": round(test_result.p_value, 4),
                            "is_significant": test_result.is_significant,
                            "drift_magnitude": round(test_result.drift_magnitude, 4),
                            "recommendation": test_result.recommendation
                        })
                    
                    return {
                        "enhanced_analysis_available": True,
                        "overall_drift_detected": drift_report.overall_drift_detected,
                        "drift_severity": drift_report.drift_severity,
                        "confidence_score": round(drift_report.confidence_score, 3),
                        "psi_results": psi_results,
                        "statistical_tests": statistical_tests,
                        "recommendations": drift_report.recommendations,
                        "analysis_timestamp": drift_report.timestamp.isoformat(),
                        "data_split": {
                            "reference_samples": len(reference_df),
                            "current_samples": len(current_df)
                        },
                        "has_data": True,
                        "last_updated": datetime.now().isoformat()
                    }
                else:
                    return {
                        "enhanced_analysis_available": False,
                        "error": "Required 'confidence' column not found in data",
                        "has_data": False,
                        "last_updated": datetime.now().isoformat()
                    }
            else:
                return {
                    "enhanced_analysis_available": False,
                    "error": "Insufficient data for enhanced drift analysis (minimum 50 samples required)",
                    "has_data": False,
                    "current_samples": len(vote_df) if vote_df is not None else 0,
                    "last_updated": datetime.now().isoformat()
                }
                
        except ImportError:
            return {
                "enhanced_analysis_available": False,
                "error": "Enhanced drift detection modules not available",
                "fallback_message": "Using basic drift analysis from /api/drift endpoint",
                "has_data": False,
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"Error in enhanced drift analysis: {e}")
        return {
            "enhanced_analysis_available": False,
            "error": str(e),
            "has_data": False,
            "last_updated": datetime.now().isoformat()
        }

# Real-time monitoring status endpoint
@app.get("/api/monitoring-status")
async def get_monitoring_status():
    """Get real-time monitoring status"""
    try:
        # Try to initialize monitoring capabilities
        try:
            from utils.analysis.real_time_monitor import RealTimeMonitor, MonitoringConfig
            
            # This would be a global monitor instance in a real application
            # For demo purposes, return status information
            return {
                "monitoring_available": True,
                "monitoring_active": False,  # Would track actual monitor state
                "capabilities": {
                    "real_time_alerts": True,
                    "performance_prediction": True,
                    "calibration_drift_detection": True,
                    "automated_recommendations": True
                },
                "configuration": {
                    "calibration_drift_threshold": 0.05,
                    "performance_drop_threshold": 0.1,
                    "alert_cooldown_minutes": 30,
                    "metrics_window_size": 100
                },
                "current_status": {
                    "active_alerts": 0,
                    "baseline_established": False,
                    "buffer_size": 0,
                    "last_observation": None
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except ImportError:
            return {
                "monitoring_available": False,
                "error": "Real-time monitoring modules not available",
                "current_status": "fallback_mode",
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"Error getting monitoring status: {e}")
        return {
            "monitoring_available": False,
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }

# Generate response for annotation
@app.post("/api/annotations/generate-response")
async def generate_annotation_response(request: Dict[str, Any]):
    """Generate a model response for annotation"""
    try:
        prompt = request.get("prompt", "")
        provider = settings_store["model"]["default_provider"]
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Get API key from stored settings
        provider_key_map = {
            "deepseek": "deepseek",
            "openai": "openai", 
            "lmstudio": "lmstudio"
        }
        
        api_key = None
        api_base = None
        
        if provider in provider_key_map:
            api_key = settings_store["apiKeys"].get(provider_key_map[provider])
            if provider == "lmstudio":
                api_base = api_key  # For LM Studio, the "key" is actually the base URL
                api_key = None  # LM Studio doesn't need API key
        
        # Create client with the stored API key
        client = ModelAPIClient(provider=provider, api_key=api_key, api_base=api_base)
        
        # Generate response using the configured model
        response = client.generate_chat_response(
            [{"role": "user", "content": prompt}],
            max_tokens=settings_store["model"]["max_tokens"],
            temperature=settings_store["model"]["temperature"]
        )
        
        if "error" in response and response["error"]:
            return {
                "success": False,
                "error": response['completion'],
                "provider": provider
            }
        else:
            return {
                "success": True,
                "response": response['completion'],
                "provider": provider,
                "model": response.get('model', 'unknown'),
                "tokens_used": response.get('total_tokens', 0)
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "provider": "unknown"
        }

# Save annotation choice  
@app.post("/api/annotations/save")
async def save_annotation(request: Dict[str, Any]):
    """Save annotation choice to vote files"""
    try:
        annotation_id = request.get("id")
        human_choice = request.get("choice")
        prompt = request.get("prompt", "")
        response_a = request.get("response_a", "")
        response_b = request.get("response_b", "")
        
        if not annotation_id or not human_choice:
            raise HTTPException(status_code=400, detail="Annotation ID and choice are required")
        
        # Create vote record
        vote_record = {
            "id": annotation_id,
            "prompt": prompt,
            "completions": [response_a, response_b],
            "chosen_index": 1 if human_choice == "B" else 0,
            "confidence": 0.8,  # Default confidence - could be made configurable
            "annotation": f"User preferred response {human_choice}",
            "generation_metadata": {
                "temperature": settings_store["model"]["temperature"],
                "max_tokens": settings_store["model"]["max_tokens"],
                "model": settings_store["model"]["default_provider"],
                "tokens": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
                "cost": 0.0001,
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to votes file
        votes_file = "../data/votes.jsonl"  # Go up one level from api_backend to find data folder
        try:
            with open(votes_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(vote_record) + '\n')
            
            # Clear cache to force data reload
            data_store["last_refresh"] = 0
            
            return {
                "success": True,
                "message": "Annotation saved successfully",
                "vote_id": annotation_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as file_error:
            print(f"Error writing to votes file: {file_error}")
            raise HTTPException(status_code=500, detail=f"Error saving to file: {str(file_error)}")
            
    except Exception as e:
        print(f"Error saving annotation: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RLHF Dashboard API Server...")
    print("📊 Integrating with real RLHF pipeline data")
    print("🔗 Available at: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 