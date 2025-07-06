"""
RLHF Model Configuration Core

This module contains the core configuration and state management for the RLHF system.
It handles model parameters, performance tracking, and interface customization
for the reinforcement learning from human feedback system.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import uuid

class ModelConfig:
    """
    Core configuration and state management for the RLHF model.
    
    This class manages model parameters, performance metrics, and learning state
    to provide consistent configuration across the RLHF system.
    """
    
    def __init__(self):
        self.model_name = "RLHF Model"
        self.version = "1.0"
        self.performance_metrics = {
            "accuracy": 0.0,
            "consistency": 0.0,
            "responsiveness": 0.0,
            "helpfulness": 0.0,
            "reliability": 0.0,
            "learning_efficiency": 0.0,
            "user_satisfaction": 0.0
        }
        self.training_modes = [
            "standard", "reinforcement_learning", "preference_learning",
            "feedback_integration", "performance_optimization", "adaptive_learning"
        ]
        self.current_mode = "standard"
        self.learning_progress = 0.0
        self.performance_score = 0.0
        
    def get_system_greeting(self) -> str:
        """Generate a professional system greeting."""
        greetings = [
            "Welcome to the RLHF Training Interface",
            "Ready to continue model training and evaluation",
            "RLHF System initialized - ready for interaction",
            "Model training interface loaded and ready",
            "Welcome back to the preference learning system"
        ]
        
        import random
        return random.choice(greetings)
    
    def get_training_context(self, interaction_type: str = "general") -> Dict[str, Any]:
        """Get current training context for interactions."""
        return {
            "training_mode": self.current_mode,
            "accuracy_level": self.performance_metrics["accuracy"],
            "learning_progress": self.learning_progress,
            "performance_score": self.performance_score,
            "model_version": self.version,
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_user_feedback(self, feedback: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user feedback and adjust model configuration accordingly."""
        feedback_lower = feedback.lower()
        
        # Analyze feedback sentiment
        positive_indicators = ["good", "correct", "accurate", "helpful", "excellent", "clear"]
        negative_indicators = ["wrong", "incorrect", "unclear", "unhelpful", "confusing", "poor"]
        
        is_positive = any(indicator in feedback_lower for indicator in positive_indicators)
        is_negative = any(indicator in feedback_lower for indicator in negative_indicators)
        
        response_data = {
            "feedback_received": feedback,
            "sentiment": "positive" if is_positive else "negative" if is_negative else "neutral",
            "config_adjustment": {},
            "system_response": "",
            "learning_note": ""
        }
        
        if is_positive:
            # Increase performance metrics
            self.learning_progress = min(1.0, self.learning_progress + 0.02)
            self.performance_score = min(1.0, self.performance_score + 0.01)
            
            response_data["system_response"] = "Positive feedback received - model performance improved"
            response_data["learning_note"] = "This response pattern was successful - reinforcing approach"
            response_data["config_adjustment"] = {
                "learning_progress": self.learning_progress,
                "performance_score": self.performance_score,
                "reinforcement": True
            }
            
        elif is_negative:
            # Adjust for improvement
            self.performance_metrics["accuracy"] = max(0.0, self.performance_metrics["accuracy"] - 0.01)
            
            response_data["system_response"] = "Negative feedback received - adjusting model parameters"
            response_data["learning_note"] = "This response needs improvement - updating training focus"
            response_data["config_adjustment"] = {
                "accuracy_adjustment": True,
                "improved_accuracy_target": self.performance_metrics["accuracy"] + 0.05,
                "training_focus": "enhanced"
            }
        
        return response_data
    
    def apply_response_enhancement(self, base_response: str, context: Dict[str, Any] = None) -> str:
        """Enhance response quality based on model configuration."""
        
        # Add professional enhancement patterns
        clarity_phrases = [
            "Based on the analysis",
            "According to the data",
            "The model indicates",
            "Evidence suggests",
            "Analysis shows"
        ]
        
        # Add reliability indicators
        confidence_expressions = [
            "with high confidence",
            "based on training data",
            "according to learned patterns",
            "following established guidelines",
            "using validated approaches"
        ]
        
        # Transform the response for better clarity
        enhanced_response = base_response
        
        # Add professional context if appropriate
        if context and context.get("add_professional_context", True):
            import random
            if self.performance_metrics["accuracy"] > 0.7:
                clarity_prefix = random.choice(clarity_phrases)
                enhanced_response = f"{clarity_prefix}, {enhanced_response.lower()}"
        
        return enhanced_response
    
    def get_learning_priorities(self) -> List[str]:
        """Get current learning priorities based on model state."""
        priorities = []
        
        if self.learning_progress < 0.7:
            priorities.append("Improve user preference understanding")
        
        if self.performance_score < 0.8:
            priorities.append("Enhance response quality and accuracy")
        
        if self.performance_metrics["accuracy"] < 0.9:
            priorities.append("Strengthen prediction accuracy")
        
        priorities.extend([
            "Optimize response relevance",
            "Improve consistency across interactions",
            "Enhance user satisfaction metrics",
            "Strengthen feedback integration",
            "Develop better calibration"
        ])
        
        return priorities[:5]  # Return top 5 priorities
    
    def save_model_state(self, database_instance) -> bool:
        """Save current model state to database for persistence."""
        try:
            model_data = {
                "timestamp": datetime.now().isoformat(),
                "model_id": f"model_state_{uuid.uuid4().hex[:8]}",
                "performance_metrics": self.performance_metrics,
                "training_mode": self.current_mode,
                "learning_progress": self.learning_progress,
                "performance_score": self.performance_score,
                "learning_priorities": self.get_learning_priorities(),
                "model_version": self.version
            }
            
            return database_instance.save_model_training_data({
                "interaction_type": "model_state_save",
                "model_mode": "standard",
                "response_type": "system",
                "user_input": "System: Saving model state",
                "model_response": f"Model state saved with performance score {self.performance_score:.2f}",
                "quality_level": self.learning_progress,
                "accuracy_score": self.performance_metrics["accuracy"],
                "helpfulness_rating": self.performance_metrics["helpfulness"],
                "confidence": 0.9,
                "response_quality": "system",
                "training_weight": 0.5,
                "response_emphasis": ["accurate", "helpful", "clear"],
                "learning_priority": "medium",
                "model_state_data": model_data
            })
            
        except Exception as e:
            print(f"Error saving model state: {e}")
            return False
    
    def load_model_state(self, database_instance) -> bool:
        """Load model state from database if available."""
        try:
            # This would load the most recent model state
            # For now, we'll use default values
            return True
        except Exception as e:
            print(f"Error loading model state: {e}")
            return False

# Global model configuration instance
_model_config = None

def get_model_config() -> ModelConfig:
    """Get the global model configuration instance."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config

def display_model_status():
    """Display current model status in Streamlit interface."""
    config = get_model_config()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance")
    
    # Performance metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Learning Progress", f"{config.learning_progress:.1%}")
        st.metric("Accuracy", f"{config.performance_metrics['accuracy']:.1%}")
    with col2:
        st.metric("Performance Score", f"{config.performance_score:.1%}")
        st.metric("Reliability", f"{config.performance_metrics['reliability']:.1%}")
    
    # Training mode
    st.sidebar.write(f"**Training Mode:** {config.current_mode.replace('_', ' ').title()}")
    
    # Learning priorities
    with st.sidebar.expander("Current Learning Focus"):
        priorities = config.get_learning_priorities()
        for i, priority in enumerate(priorities, 1):
            st.write(f"{i}. {priority}")

def apply_professional_styling():
    """Apply professional styling to the Streamlit interface."""
    st.markdown("""
    <style>
    .main > div {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(173, 216, 230, 0.1);
        border: 1px solid rgba(173, 216, 230, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4A90E2, #357ABD);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #357ABD, #4A90E2);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
    }
    
    h1, h2, h3 {
        color: #4A90E2;
        text-shadow: 0 0 8px rgba(74, 144, 226, 0.2);
    }
    
    .stMetric {
        background: rgba(173, 216, 230, 0.1);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(173, 216, 230, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def system_message_formatter(message: str, message_type: str = "info") -> None:
    """Format messages in a professional system style."""
    
    if message_type == "success":
        st.success(f"✅ {message}")
    elif message_type == "error":
        st.error(f"❌ {message}")
    elif message_type == "warning":
        st.warning(f"⚠️ {message}")
    elif message_type == "performance":
        st.markdown(f"""
        <div style="
            background: linear-gradient(45deg, rgba(74, 144, 226, 0.1), rgba(53, 122, 189, 0.1));
            border-left: 4px solid #4A90E2;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <p style="color: #4A90E2; font-weight: 500; margin: 0;">
                📊 {message}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ {message}")
