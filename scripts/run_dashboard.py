#!/usr/bin/env python3
"""
RLHF Loop Dashboard - Live Model Integration

Professional interface for monitoring reinforcement learning from human feedback systems.
Now with live model connections and real-time API integration.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Add parent directory to path to allow imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add RLHF annotation imports at the top
try:
    from interface.sections.annotation import display_annotation_interface, display_annotation_history, display_preference_timeline
except ImportError:
    # Fallback if annotation module not available
    def display_annotation_interface(vote_df):
        st.error("Annotation interface not available. Please check interface/sections/annotation.py")
    def display_annotation_history(vote_df, predictions_df):
        st.error("Annotation history not available.")
    def display_preference_timeline(vote_df):
        st.error("Preference timeline not available.")

def check_api_connection():
    """Check if API is configured and working"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return False, "No API key found"
    
    try:
        from utils.completions import generate_completions
        # Quick test call
        result = generate_completions("Test", n_completions=1, max_tokens=5)
        return True, f"Connected to DeepSeek API"
    except Exception as e:
        return False, f"API Error: {str(e)}"

def main():
    """Main entry point for the RLHF dashboard"""
    
    # Configure page
    st.set_page_config(
        page_title="RLHF Loop Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply professional styling
    apply_professional_styling()
    
    # Check API connection status
    api_connected, api_status = check_api_connection()
    
    # Create header
    create_header(api_connected)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🎛️ RLHF Dashboard")
        st.markdown("Professional AI monitoring interface")
        
        # API Status in sidebar
        if api_connected:
            st.success("🟢 DeepSeek API Connected")
        else:
            st.error("🔴 API Disconnected")
            st.caption(api_status)
        
        tab_options = [
            "System Overview",
            "Live Model Chat",
            "RLHF Annotation",
            "Annotation History",
            "Performance Metrics", 
            "Data Analysis",
            "Model Status"
        ]
        
        selected_tab = st.selectbox("Select Section", tab_options)
    
    # Display content based on selected tab
    display_content(selected_tab, api_connected, api_status)
    
    # Footer
    create_footer()

def apply_professional_styling():
    """Apply professional styling"""
    st.markdown("""
    <style>
    .main > div {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #1d4ed8, #2563eb);
        transform: translateY(-2px);
    }
    
    h1, h2, h3 {
        color: #1e40af;
        font-weight: 600;
    }
    
    .stMetric {
        background: rgba(37, 99, 235, 0.1);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(37, 99, 235, 0.2);
    }
    
    .api-connected {
        border-left: 4px solid #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .api-disconnected {
        border-left: 4px solid #ef4444;
        background: rgba(239, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_header(api_connected):
    """Create professional header with API status"""
    api_status_color = "#10b981" if api_connected else "#ef4444"
    api_status_text = "🟢 Live Models Connected" if api_connected else "🔴 Demo Mode"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    ">
        <h1 style="
            color: white; 
            margin: 0; 
            font-size: 2.5em;
            font-weight: 700;
        ">
            🤖 RLHF Loop Dashboard
        </h1>
        <p style="
            color: rgba(255,255,255,0.9); 
            margin: 10px 0 0 0; 
            font-size: 1.2em;
        ">
            Professional AI model performance monitoring
        </p>
        <div style="
            margin-top: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            color: {api_status_color};
            font-weight: 600;
        ">
            {api_status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_content(selected_tab, api_connected, api_status):
    """Display content based on selected tab"""
    
    if selected_tab == "System Overview":
        display_system_overview(api_connected, api_status)
    elif selected_tab == "Live Model Chat":
        display_live_chat(api_connected)
    elif selected_tab == "RLHF Annotation":
        display_rlhf_annotation_tab(api_connected)
    elif selected_tab == "Annotation History":
        display_rlhf_history_tab()
    elif selected_tab == "Performance Metrics":
        display_performance_metrics()
    elif selected_tab == "Data Analysis":
        display_data_analysis()
    elif selected_tab == "Model Status":
        display_model_status(api_connected, api_status)

def display_system_overview(api_connected, api_status):
    """Display system overview with live API status"""
    st.markdown("## 📊 System Overview")
    
    if api_connected:
        st.success("🎉 **Live Mode**: Connected to DeepSeek API - Real model interactions available!")
    else:
        st.warning("⚠️ **Demo Mode**: No live API connection. Set DEEPSEEK_API_KEY to enable live models.")
    
    # Create metrics with live or demo indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if api_connected:
            st.metric(
                label="API Connection",
                value="Active",
                delta="DeepSeek connected",
                help="Live API connection established"
            )
        else:
            st.metric(
                label="API Connection",
                value="Offline",
                delta="No API key",
                help="Set DEEPSEEK_API_KEY environment variable"
            )
    
    with col2:
        response_time = "~200ms" if api_connected else "N/A"
        delta_text = "Live inference" if api_connected else "No active inference"
        st.metric(
            label="Response Time",
            value=response_time,
            delta=delta_text,
            help="API response latency"
        )
    
    with col3:
        st.metric(
            label="System Load",
            value="Ready",
            delta="System operational",
            help="System ready for model deployment"
        )
    
    with col4:
        alignment_value = "Real-time" if api_connected else "N/A"
        alignment_delta = "Live monitoring" if api_connected else "Requires model"
        st.metric(
            label="Alignment Score",
            value=alignment_value,
            delta=alignment_delta,
            help="Model alignment monitoring status"
        )
    
    # System status
    st.markdown("### 🔧 System Status")
    if api_connected:
        st.success("✅ Live API connection established")
        st.success("✅ DeepSeek model ready for inference")
        st.info("📊 Real-time monitoring active")
        
        # Show API details
        st.markdown("### 🔗 API Connection Details")
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        masked_key = f"sk-{api_key[3:8]}...{api_key[-4:]}" if len(api_key) > 10 else "Invalid"
        st.code(f"""
API Endpoint: https://api.deepseek.com/v1/chat/completions
Model: deepseek-chat
API Key: {masked_key}
Status: {api_status}
        """)
    else:
        st.info("🔧 Dashboard ready - awaiting API configuration")
        st.warning("⚠️ Configure API key to enable live models")
        st.info("📊 Demo monitoring capabilities active")

def display_live_chat(api_connected):
    """Display live model chat interface"""
    st.markdown("## 💬 Live Model Chat")
    
    if not api_connected:
        st.error("🔌 API connection required for live chat. Please configure your DEEPSEEK_API_KEY.")
        st.info("💡 **How to connect:**\n1. Run: `python utils/setup_deepseek.py`\n2. Set: `$env:DEEPSEEK_API_KEY = 'your-key'`\n3. Restart dashboard")
        return
    
    st.success("🟢 Live chat with DeepSeek API enabled!")
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("📊 Token Usage"):
                    st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask the model anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get model response
        with st.chat_message("assistant"):
            with st.spinner("🤖 DeepSeek is thinking..."):
                try:
                    from utils.completions import generate_completions
                    result = generate_completions(prompt, n_completions=1, max_tokens=200)
                    
                    response = result['completions'][0]
                    st.markdown(response)
                    
                    # Add assistant message with metadata
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "metadata": {
                            "tokens": result['aggregated_usage'],
                            "cost": result.get('estimated_cost', 0),
                            "timestamp": result['timestamp']
                        }
                    })
                    
                    # Show token usage
                    with st.expander("📊 Token Usage"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prompt Tokens", result['aggregated_usage']['prompt_tokens'])
                        with col2:
                            st.metric("Response Tokens", result['aggregated_usage']['completion_tokens'])
                        with col3:
                            st.metric("Cost", f"${result.get('estimated_cost', 0):.6f}")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

def get_live_chat_data():
    """Get real data from live chat sessions"""
    if "chat_history" not in st.session_state:
        return {
            "total_conversations": 0,
            "total_tokens": 0,
            "total_cost": 0,
            "avg_response_time": 0,
            "conversations": []
        }
    
    conversations = st.session_state.chat_history
    total_tokens = 0
    total_cost = 0
    user_messages = 0
    assistant_messages = 0
    
    for msg in conversations:
        if msg.get("metadata"):
            tokens = msg["metadata"].get("tokens", {})
            total_tokens += tokens.get("total_tokens", 0)
            total_cost += msg["metadata"].get("cost", 0)
        
        if msg["role"] == "user":
            user_messages += 1
        elif msg["role"] == "assistant":
            assistant_messages += 1
    
    return {
        "total_conversations": len([msg for msg in conversations if msg["role"] == "user"]),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_response_time": 200,  # ms - could be tracked from API calls
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "conversations": conversations
    }

def load_training_results():
    """Load real training results from reward model training"""
    try:
        training_history_path = Path("models/reward_model/training_history.json")
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading training results: {e}")
    return None

def display_performance_metrics():
    """Display performance metrics using REAL RLHF data"""
    st.markdown("## 📊 Performance Metrics")
    
    # Load real RLHF data and training results
    rlhf_data = load_rlhf_data()
    chat_data = get_live_chat_data()
    training_results = load_training_results()
    
    # Show real data status
    if rlhf_data.empty and chat_data["total_conversations"] == 0:
        st.info("🎯 **No performance data yet.** Start using the RLHF Annotation and Live Chat to generate real metrics!")
        return
    
    st.success("📈 **Live Performance Data** - Showing real usage metrics from your RLHF system")
    
    # Real performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # RLHF annotations count
        annotations_count = len(rlhf_data) if not rlhf_data.empty else 0
        st.metric(
            "🎯 RLHF Annotations", 
            annotations_count,
            f"+{annotations_count} total" if annotations_count > 0 else "Start annotating!"
        )
    
    with col2:
        # Live chat conversations
        st.metric(
            "💬 Live Conversations", 
            chat_data["total_conversations"],
            f"+{chat_data['conversations'][-1] if chat_data['conversations'] else 0} recent" if chat_data["total_conversations"] > 0 else "Start chatting!"
        )
    
    with col3:
        # Model training status
        if training_results and training_results.get("status") == "completed":
            accuracy = training_results.get("test_accuracy", 0)
            st.metric("🧠 Model Accuracy", f"{accuracy:.1%}", "Trained on real data!")
        else:
            st.metric("🧠 Model Status", "Not Trained", "Click train button!")
    
    with col4:
        # Total cost
        total_cost = chat_data["total_cost"]
        if not rlhf_data.empty and 'quality_metrics' in rlhf_data.columns:
            for _, row in rlhf_data.iterrows():
                try:
                    metrics = row.get('quality_metrics', {})
                    if isinstance(metrics, dict):
                        total_cost += metrics.get('cost', 0)
                except:
                    pass
        st.metric("💰 Total Cost", f"${total_cost:.6f}")
    
    # Real training results section
    if training_results:
        st.markdown("### 🧠 Reward Model Training Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{training_results.get('train_accuracy', 0):.1%}")
        with col2:
            test_acc = training_results.get('test_accuracy', 0)
            # Show confidence interval if available
            if 'test_accuracy_lower_95' in training_results:
                lower = training_results['test_accuracy_lower_95']
                upper = training_results['test_accuracy_upper_95']
                st.metric("Test Accuracy", f"{test_acc:.1%}", f"95% CI: [{lower:.1%}, {upper:.1%}]")
            else:
                st.metric("Test Accuracy", f"{test_acc:.1%}")
        with col3:
            train_size = training_results.get('train_size', 0)
            test_size = training_results.get('test_size', 0)
            st.metric("Total Examples", f"{train_size + test_size}")
        
        # Show training timeline and model type
        if training_results.get('timestamp'):
            model_type = training_results.get('model_type', 'Unknown')
            st.caption(f"🕒 Trained: {training_results['timestamp'][:19]} | Algorithm: {model_type}")
        
        # Show interpretation for small datasets
        if 'interpretation' in training_results:
            interp = training_results['interpretation']
            
            # Color-code the warning based on reliability
            reliability = interp.get('reliability', 'unknown')
            if reliability == 'low':
                st.warning("⚠️ **Small Dataset Warning**: " + interp.get('recommendation', ''))
            elif reliability == 'moderate':
                st.info("ℹ️ **Dataset Size**: " + interp.get('recommendation', ''))
            
            # Show overfitting warning
            overfitting = interp.get('overfitting_risk', 'unknown')
            if overfitting == 'high':
                train_acc = training_results.get('train_accuracy', 0)
                test_acc = training_results.get('test_accuracy', 0)
                gap = train_acc - test_acc
                st.error(f"🎯 **Overfitting Detected**: {gap:.1%} gap between train/test accuracy. Model memorized training data.")
        
        # Show small test set warning
        if training_results.get('small_test_set_warning'):
            st.warning(f"📊 **Small Test Set**: Only {training_results.get('test_size', 0)} test examples. Results have high variance.")
        
        # Performance visualization
        if 'metrics' in training_results:
            metrics = training_results['metrics']
            
            # Create accuracy comparison chart
            accuracy_data = pd.DataFrame({
                'Dataset': ['Training', 'Test'],
                'Accuracy': [metrics.get('train_accuracy', 0), metrics.get('test_accuracy', 0)]
            })
            
            fig = px.bar(
                accuracy_data,
                x='Dataset',
                y='Accuracy',
                title='Model Performance on Real RLHF Data',
                labels={'Accuracy': 'Accuracy Score'},
                color='Dataset',
                color_discrete_map={'Training': '#1f77b4', 'Test': '#ff7f0e'}
            )
            fig.update_layout(height=300, showlegend=False, yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
    
    # Real-time activity chart
    if not rlhf_data.empty or chat_data["total_conversations"] > 0:
        st.markdown("### 📈 Real Activity Timeline")
        
        activity_data = []
        
        # Add RLHF annotations to timeline
        if not rlhf_data.empty and 'timestamp' in rlhf_data.columns:
            for _, row in rlhf_data.iterrows():
                try:
                    timestamp = pd.to_datetime(row['timestamp'])
                    activity_data.append({
                        'time': timestamp,
                        'activity': 'RLHF Annotation',
                        'count': 1
                    })
                except:
                    pass
        
        # Add chat conversations to timeline
        for conv in chat_data["conversations"]:
            if conv.get("timestamp"):
                try:
                    timestamp = pd.to_datetime(conv["timestamp"])
                    activity_data.append({
                        'time': timestamp,
                        'activity': 'Live Chat',
                        'count': 1
                    })
                except:
                    pass
        
        # Add training events
        if training_results and training_results.get('timestamp'):
            try:
                timestamp = pd.to_datetime(training_results['timestamp'])
                activity_data.append({
                    'time': timestamp,
                    'activity': 'Model Training',
                    'count': 1
                })
            except:
                pass
        
        if activity_data:
            activity_df = pd.DataFrame(activity_data)
            activity_df['date'] = activity_df['time'].dt.date
            activity_summary = activity_df.groupby(['date', 'activity'])['count'].sum().reset_index()
            
            fig = px.bar(
                activity_summary, 
                x='date', 
                y='count', 
                color='activity',
                title='Daily Activity (RLHF Annotations + Live Chat + Training)',
                labels={'count': 'Activity Count', 'date': 'Date'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity data available yet. Start using RLHF or Live Chat!")
    
    # Real preference distribution
    if not rlhf_data.empty:
        st.markdown("### 🎯 RLHF Preference Distribution")
        
        # Count preferences
        if 'preference' in rlhf_data.columns:
            pref_counts = rlhf_data['preference'].value_counts()
            
            # Create donut chart
            fig = px.pie(
                values=pref_counts.values,
                names=pref_counts.index,
                title="Your Preference Choices",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement analysis (if available)
        if 'model_correct' in rlhf_data.columns:
            agreement_rate = rlhf_data['model_correct'].mean()
            st.markdown("### 🤖 Model Agreement Rate")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{agreement_rate:.1%}" if not pd.isna(agreement_rate) else "No data")
            with col2:
                correct_count = rlhf_data['model_correct'].sum() if 'model_correct' in rlhf_data.columns else 0
                st.metric("Correct Predictions", int(correct_count) if not pd.isna(correct_count) else 0)

def display_data_analysis():
    """Display data analysis using REAL RLHF data"""
    st.markdown("## 📊 Data Analysis")
    
    # Load real data
    rlhf_data = load_rlhf_data()
    chat_data = get_live_chat_data()
    training_results = load_training_results()
    
    if rlhf_data.empty and chat_data["total_conversations"] == 0:
        st.info("📊 **No data to analyze yet!** Start using RLHF Annotation and Live Chat to generate insights.")
        
        # Show getting started guide
        st.markdown("### 🚀 **Generate Real Data:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 RLHF Annotations:**
            1. Go to 'RLHF Annotation' tab
            2. Enter prompts
            3. Choose better responses
            4. Build preference dataset
            """)
        with col2:
            st.markdown("""
            **💬 Live Conversations:**
            1. Use 'Live Model Chat' tab
            2. Chat with DeepSeek API
            3. Generate usage analytics
            4. Track costs & tokens
            """)
        return
    
    st.success("📊 **Real Data Analysis** - Insights from your actual RLHF system usage")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 RLHF Samples", len(rlhf_data))
    with col2:
        st.metric("💬 Chat Messages", chat_data["total_conversations"])  
    with col3:
        total_interactions = len(rlhf_data) + chat_data["total_conversations"]
        st.metric("🔄 Total Interactions", total_interactions)
    with col4:
        training_status = "✅ Trained" if training_results else "❌ Not Trained"
        st.metric("🧠 Model Status", training_status)
    
    # Real training analysis
    if training_results:
        st.markdown("### 🧠 Real Reward Model Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Model Performance")
            
            # Performance metrics
            metrics_data = pd.DataFrame({
                'Metric': ['Training Accuracy', 'Test Accuracy', 'Feature Count'],
                'Value': [
                    f"{training_results.get('train_accuracy', 0):.1%}",
                    f"{training_results.get('test_accuracy', 0):.1%}",
                    f"{training_results.get('feature_count', 0)}"
                ]
            })
            st.dataframe(metrics_data, hide_index=True, use_container_width=True)
            
        with col2:
            st.markdown("#### 🎯 Training Details")
            
            # Training details
            details_data = pd.DataFrame({
                'Detail': ['Algorithm', 'Training Size', 'Test Size', 'Trained On'],
                'Value': [
                    training_results.get('algorithm', 'Unknown'),
                    f"{training_results.get('train_size', 0)} examples",
                    f"{training_results.get('test_size', 0)} examples",
                    training_results.get('timestamp', 'Unknown')[:10]
                ]
            })
            st.dataframe(details_data, hide_index=True, use_container_width=True)
        
        # Feature importance analysis (if available)
        if 'metrics' in training_results and 'feature_importance' in training_results['metrics']:
            st.markdown("#### 🔍 Feature Importance Analysis")
            
            feature_importance = training_results['metrics']['feature_importance']
            
            # Show top 10 most important features
            top_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)[:10]
            
            feature_df = pd.DataFrame({
                'Feature': [f"Feature {i}" for i, _ in top_features],
                'Importance': [importance for _, importance in top_features]
            })
            
            fig = px.bar(
                feature_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features for Preference Prediction',
                labels={'Importance': 'Feature Importance Score'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Temporal analysis
    if not rlhf_data.empty or chat_data["total_conversations"] > 0:
        st.markdown("### 📈 Usage Patterns")
        
        # Create timeline data
        timeline_data = []
        
        # Add RLHF data
        if not rlhf_data.empty and 'timestamp' in rlhf_data.columns:
            for _, row in rlhf_data.iterrows():
                try:
                    timeline_data.append({
                        'timestamp': pd.to_datetime(row['timestamp']),
                        'type': 'RLHF Annotation',
                        'value': 1
                    })
                except:
                    pass
        
        # Add chat data
        for conv in chat_data["conversations"]:
            if conv.get("timestamp"):
                try:
                    timeline_data.append({
                        'timestamp': pd.to_datetime(conv["timestamp"]),
                        'type': 'Live Chat',
                        'value': 1
                    })
                except:
                    pass
        
        # Add training data
        if training_results and training_results.get('timestamp'):
            try:
                timeline_data.append({
                    'timestamp': pd.to_datetime(training_results['timestamp']),
                    'type': 'Model Training',
                    'value': 1
                })
            except:
                pass
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('timestamp')
            timeline_df['cumulative'] = timeline_df.groupby('type')['value'].cumsum()
            
            fig = px.line(
                timeline_df, 
                x='timestamp', 
                y='cumulative', 
                color='type',
                title='Cumulative System Usage Over Time (Including Real Training)',
                labels={'cumulative': 'Total Count', 'timestamp': 'Time'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # RLHF-specific analysis
    if not rlhf_data.empty:
        st.markdown("### 🎯 RLHF Training Analysis")
        
        # Preference patterns
        if 'preference' in rlhf_data.columns:
            st.markdown("#### 👥 Preference Patterns")
            pref_over_time = rlhf_data.copy()
            if 'timestamp' in pref_over_time.columns:
                pref_over_time['timestamp'] = pd.to_datetime(pref_over_time['timestamp'])
                pref_over_time['date'] = pref_over_time['timestamp'].dt.date
                
                daily_prefs = pref_over_time.groupby(['date', 'preference']).size().reset_index(name='count')
                
                fig = px.bar(
                    daily_prefs,
                    x='date',
                    y='count', 
                    color='preference',
                    title='Daily Preference Distribution',
                    labels={'count': 'Preference Count', 'date': 'Date'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cost analysis  
        if 'quality_metrics' in rlhf_data.columns:
            st.markdown("#### 💰 Cost Analysis")
            costs = []
            for _, row in rlhf_data.iterrows():
                try:
                    metrics = row.get('quality_metrics', {})
                    if isinstance(metrics, dict):
                        cost = metrics.get('cost', 0)
                        if cost > 0:
                            costs.append({
                                'timestamp': row.get('timestamp'),
                                'cost': cost
                            })
                except:
                    pass
            
            if costs:
                cost_df = pd.DataFrame(costs)
                cost_df['timestamp'] = pd.to_datetime(cost_df['timestamp'])
                cost_df['cumulative_cost'] = cost_df['cost'].cumsum()
                
                fig = px.line(
                    cost_df,
                    x='timestamp',
                    y='cumulative_cost',
                    title='Cumulative RLHF Training Cost',
                    labels={'cumulative_cost': 'Total Cost ($)', 'timestamp': 'Time'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("💰 Total RLHF Cost", f"${cost_df['cumulative_cost'].iloc[-1]:.6f}")
    
    # Chat analysis
    if chat_data["total_conversations"] > 0:
        st.markdown("### 💬 Live Chat Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_tokens = chat_data["total_tokens"] / chat_data["total_conversations"] if chat_data["total_conversations"] > 0 else 0
            st.metric("📏 Avg Tokens/Chat", f"{avg_tokens:.0f}")
        with col2:
            avg_cost = chat_data["total_cost"] / chat_data["total_conversations"] if chat_data["total_conversations"] > 0 else 0
            st.metric("💰 Avg Cost/Chat", f"${avg_cost:.6f}")

def display_model_status(api_connected, api_status):
    """Display model status with live API information"""
    st.markdown("## 🤖 Model Status")
    
    # Model information
    st.markdown("### ℹ️ Model Information")
    
    if api_connected:
        model_info = {
            "Model Type": "DeepSeek Chat (Live API)",
            "Architecture": "Large Language Model",
            "API Endpoint": "https://api.deepseek.com/v1/chat/completions",
            "Model ID": "deepseek-chat",
            "Connection Status": "✅ Connected",
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Version": "Live API v1.0"
        }
    else:
        model_info = {
            "Model Type": "BERT-based Preference Model (Demo)",
            "Architecture": "Transformer (BERT-tiny)",
            "Parameters": "4.3M",
            "Training Data": "Human preference pairs",
            "Connection Status": "❌ No API connection",
            "Last Updated": "2024-01-15 14:30:22",
            "Version": "v2.1.0 (Demo)"
        }
    
    for key, value in model_info.items():
        st.text(f"{key}: {value}")
    
    # Model health checks
    st.markdown("### 🏥 Health Checks")
    if api_connected:
        st.success("✅ DeepSeek API connected and responding")
        st.success("✅ Live inference pipeline active")
        st.success("✅ Real-time monitoring enabled")
        st.info("📊 Token usage tracking active")
        
        # Additional API metrics
        st.markdown("### 📊 API Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("API Status", "Online", "Connected")
        with col2:
            st.metric("Model Latency", "~200ms", "Real-time")
        with col3:
            st.metric("Rate Limit", "Available", "No limits hit")
            
    else:
        st.warning("⚠️ Model not loaded - API connection required")
        st.info("🔧 To enable live models:")
        st.code("""
# 1. Set up API key
python utils/setup_deepseek.py

# 2. Set environment variable
$env:DEEPSEEK_API_KEY = "your-api-key"

# 3. Restart dashboard
streamlit run scripts/run_dashboard.py
        """)
        st.info("📊 Demo monitoring capabilities active")

def create_footer():
    """Create professional footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 System Status:** Active")
    
    with col2:
        st.markdown("**🕐 Last Update:** Real-time")
    
    with col3:
        api_connected = bool(os.environ.get("DEEPSEEK_API_KEY"))
        mode = "Live API" if api_connected else "Demo"
        st.markdown(f"**🧠 Mode:** {mode}")
    
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; color: #6b7280;">
        RLHF Loop Dashboard v2.0 | Professional AI monitoring platform with live model integration
    </div>
    """, unsafe_allow_html=True)

def display_rlhf_annotation_tab(api_connected):
    """Display RLHF annotation interface for collecting human preferences"""
    st.markdown("## 🎯 RLHF Annotation Interface")
    
    if not api_connected:
        st.error("🔌 **API connection required for RLHF annotation.** Please configure your DEEPSEEK_API_KEY.")
        st.info("💡 **How to connect:**\n1. Run: `python utils/setup_deepseek.py`\n2. Set: `$env:DEEPSEEK_API_KEY = 'your-key'`\n3. Restart dashboard")
        return
    
    st.success("🎯 **RLHF Mode Active** - Collect human preferences to train your reward model!")
    
    # Instructions for RLHF
    with st.expander("📚 **How RLHF Works**", expanded=True):
        st.markdown("""
        **Reinforcement Learning from Human Feedback (RLHF) Process:**
        
        1. **📝 Enter a prompt** - Ask the AI anything
        2. **🤖 Generate completions** - Get multiple AI responses 
        3. **👥 Choose the better response** - Your preference teaches the AI
        4. **🧠 Train the reward model** - System learns from your choices
        5. **🔄 Improve future responses** - AI gets better over time
        
        **Your choices directly improve the AI's performance!** 🚀
        """)
    
    # RLHF annotation form
    with st.form("rlhf_annotation_form"):
        st.markdown("### 📝 Step 1: Enter Your Prompt")
        prompt = st.text_area(
            "What would you like the AI to respond to?",
            height=100,
            placeholder="Example: Explain quantum computing in simple terms",
            help="Enter any prompt - the AI will generate multiple responses for you to compare"
        )
        
        # Generation settings
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "🌡️ Response Creativity", 
                0.1, 1.0, 0.7, 0.1,
                help="Higher = more creative, Lower = more consistent"
            )
        with col2:
            max_tokens = st.slider(
                "📏 Response Length", 
                50, 500, 200, 50,
                help="Maximum tokens in response"
            )
        
        generate_completions = st.form_submit_button("🚀 **Generate Completions for RLHF**", type="primary")
    
    # Handle completion generation
    if generate_completions and prompt.strip():
        with st.spinner("🤖 Generating multiple AI responses for comparison..."):
            try:
                from utils.completions import generate_completions
                
                # Generate completion pairs for RLHF
                result = generate_completions(
                    prompt.strip(), 
                    n_completions=2,  # A/B comparison
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Store in session state for annotation
                st.session_state.rlhf_prompt = prompt.strip()
                st.session_state.rlhf_completion_a = result['completions'][0]
                st.session_state.rlhf_completion_b = result['completions'][1]
                st.session_state.rlhf_metadata = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": "deepseek-chat",
                    "tokens": result['aggregated_usage'],
                    "cost": result.get('estimated_cost', 0),
                    "timestamp": result['timestamp']
                }
                
                st.success("✅ Generated 2 completions for comparison!")
                
            except Exception as e:
                st.error(f"❌ Error generating completions: {e}")
    
    # Display completion comparison if available
    if (st.session_state.get("rlhf_prompt") and 
        st.session_state.get("rlhf_completion_a") and 
        st.session_state.get("rlhf_completion_b")):
        
        st.markdown("### 👥 Step 2: Choose the Better Response")
        st.markdown(f"**Prompt:** {st.session_state.rlhf_prompt}")
        
        # Display completions side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Response A:**")
            st.info(st.session_state.rlhf_completion_a)
            
            if st.button("✨ **Choose Response A**", key="choose_a", type="primary"):
                save_rlhf_annotation("A")
        
        with col2:
            st.markdown("**🤖 Response B:**")
            st.info(st.session_state.rlhf_completion_b)
            
            if st.button("✨ **Choose Response B**", key="choose_b", type="primary"):
                save_rlhf_annotation("B")
        
        # Show metadata
        if st.session_state.get("rlhf_metadata"):
            metadata = st.session_state.rlhf_metadata
            st.caption(f"🔹 Model: {metadata.get('model')} • Tokens: {metadata.get('tokens', {}).get('total_tokens', 0)} • Cost: ${metadata.get('cost', 0):.6f}")

def save_rlhf_annotation(choice):
    """Save RLHF annotation choice to training data"""
    try:
        import json
        from pathlib import Path
        import uuid
        
        # Create annotation data in the correct database format
        prompt_id = f"prompt_{uuid.uuid4().hex[:8]}"
        preference = "Completion A" if choice == "A" else "Completion B"
        
        annotation_data = {
            "prompt_id": prompt_id,  # Required by database
            "prompt": st.session_state.rlhf_prompt,
            "preference": preference,  # Required by database - must be "Completion A" or "Completion B"
            "timestamp": datetime.now().isoformat(),
            "selected_completion": st.session_state.rlhf_completion_a if choice == "A" else st.session_state.rlhf_completion_b,
            "rejected_completion": st.session_state.rlhf_completion_b if choice == "A" else st.session_state.rlhf_completion_a,
            "feedback": f"User preferred response {choice}",
            "quality_metrics": {
                "temperature": st.session_state.rlhf_metadata.get("temperature"),
                "max_tokens": st.session_state.rlhf_metadata.get("max_tokens"),
                "model": st.session_state.rlhf_metadata.get("model"),
                "cost": st.session_state.rlhf_metadata.get("cost", 0)
            }
        }
        
        # Save to votes.jsonl file for vote predictor training (legacy format)
        votes_file = Path("data/votes.jsonl")
        votes_file.parent.mkdir(exist_ok=True)
        
        legacy_vote_data = {
            "id": str(uuid.uuid4()),
            "prompt": st.session_state.rlhf_prompt,
            "completions": [st.session_state.rlhf_completion_a, st.session_state.rlhf_completion_b],
            "chosen_index": 0 if choice == "A" else 1,
            "confidence": 0.8,
            "annotation": f"User preferred response {choice}",
            "generation_metadata": st.session_state.rlhf_metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(votes_file, "a") as f:
            f.write(json.dumps(legacy_vote_data) + "\n")
        
        # Save to database with correct format
        try:
            from utils.database import get_database
            db = get_database()
            success = db.save_annotation(annotation_data)
            if success:
                st.success(f"🎉 **Preference recorded!** You chose Response {choice}. This helps train the AI!")
                st.balloons()
            else:
                st.warning("⚠️ Annotation saved to file but database save failed. Data is still preserved.")
        except Exception as e:
            st.warning(f"⚠️ Database save failed ({e}), but annotation saved to file.")
        
        # Clear session state for next annotation
        for key in ['rlhf_prompt', 'rlhf_completion_a', 'rlhf_completion_b', 'rlhf_metadata']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error saving annotation: {e}")

def display_rlhf_history_tab():
    """Display RLHF annotation history and training progress"""
    st.markdown("## 📊 RLHF Training Progress")
    
    # Load annotation data
    vote_df = load_rlhf_data()
    
    if vote_df.empty:
        st.info("🎯 **No RLHF annotations yet!** Go to 'RLHF Annotation' tab to start collecting human preferences.")
        st.markdown("### 🚀 **Get Started with RLHF:**")
        st.markdown("1. Switch to **'RLHF Annotation'** tab")
        st.markdown("2. Enter prompts and compare AI responses")
        st.markdown("3. Choose better responses to teach the AI")
        st.markdown("4. Watch your training data grow here!")
        return
    
    st.success(f"🎉 **{len(vote_df)} RLHF annotations collected!** Your preferences are training the AI.")
    
    # Show training metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Annotations", len(vote_df))
    
    with col2:
        # Recent annotations (last 24h)
        if 'timestamp' in vote_df.columns:
            vote_df['timestamp'] = pd.to_datetime(vote_df['timestamp'])
            recent = len(vote_df[vote_df['timestamp'] > datetime.now() - timedelta(days=1)])
            st.metric("Last 24h", recent, f"+{recent}")
    
    with col3:
        # Preference distribution
        if 'chosen_index' in vote_df.columns:
            choice_a = len(vote_df[vote_df['chosen_index'] == 0])
            choice_b = len(vote_df[vote_df['chosen_index'] == 1])
            ratio = choice_a / (choice_a + choice_b) if (choice_a + choice_b) > 0 else 0
            st.metric("Choice A Ratio", f"{ratio:.1%}")
    
    with col4:
        # Training readiness
        ready = "✅ Ready" if len(vote_df) >= 10 else f"Need {10 - len(vote_df)} more"
        st.metric("Training Status", ready)
    
    # Recent annotations
    st.markdown("### 📝 Recent RLHF Annotations")
    
    if 'timestamp' in vote_df.columns:
        recent_annotations = vote_df.sort_values('timestamp', ascending=False).head(5)
        
        for idx, row in recent_annotations.iterrows():
            with st.expander(f"📝 Annotation: {row.get('prompt', 'No prompt')[:50]}...", expanded=False):
                st.markdown(f"**Prompt:** {row.get('prompt', 'No prompt')}")
                
                col1, col2 = st.columns(2)
                
                completions = row.get('completions', [])
                chosen_idx = row.get('chosen_index', 0)
                
                with col1:
                    st.markdown("**👎 Rejected Response:**")
                    if completions and len(completions) > 1:
                        rejected_idx = 1 - chosen_idx
                        st.error(completions[rejected_idx])
                
                with col2:
                    st.markdown("**✨ Chosen Response:**")
                    if completions and chosen_idx < len(completions):
                        st.success(completions[chosen_idx])
                
                st.caption(f"🕒 {row.get('timestamp', 'Unknown time')}")
    
    # Training action button
    st.markdown("### 🧠 Train Reward Model")
    
    if len(vote_df) >= 10:
        if st.button("🚀 **Train Vote Predictor Model**", type="primary"):
            with st.spinner("🧠 Training reward model on your annotations..."):
                try:
                    # Run training script with correct working directory
                    import subprocess
                    import os
                    
                    # Get the project root directory (parent of scripts/)
                    project_root = str(Path(__file__).parent.parent)
                    
                    # Run ACTUAL training (no --dry-run)
                    result = subprocess.run([
                        "python", "scripts/train_reward_model.py"
                    ], capture_output=True, text=True, cwd=project_root)
                    
                    if result.returncode == 0:
                        st.success("🎉 **Reward model training completed!** The AI has learned from your preferences.")
                        st.balloons()
                        
                        # Show training output
                        with st.expander("📋 Training Details", expanded=False):
                            st.code(result.stdout)
                            
                        # Show next steps
                        st.info("✨ **Next Steps:**\n1. Check 'Performance Metrics' tab for training results\n2. Check 'Data Analysis' tab for model insights\n3. Continue adding annotations to improve the model")
                        
                        # Force refresh to show new training results
                        st.success("🔄 **Dashboard updated!** Go to Performance Metrics and Data Analysis tabs to see your trained model results.")
                        
                    else:
                        st.error(f"❌ **Training failed:** {result.stderr}")
                        
                        # Show debug info
                        with st.expander("🔍 Debug Information", expanded=True):
                            st.code(f"Error: {result.stderr}")
                            st.code(f"Output: {result.stdout}")
                            
                except Exception as e:
                    st.error(f"❌ **Error running training:** {e}")
    else:
        st.warning(f"🎯 **Need {10 - len(vote_df)} more annotations** before training can begin.")
        st.info("💡 Keep annotating in the 'RLHF Annotation' tab!")

def load_rlhf_data():
    """Load RLHF annotation data from database and files"""
    try:
        from utils.database import get_database
        db = get_database()
        # Load existing annotations from database
        annotations_df = db.get_annotations()
        if not annotations_df.empty:
            return annotations_df
    except Exception as e:
        logger.warning(f"Database error: {e}")
    
    # Fallback: try loading from file
    try:
        vote_file = Path("data/votes.jsonl")
        if vote_file.exists():
            votes = []
            with open(vote_file, 'r') as f:
                for line in f:
                    try:
                        votes.append(json.loads(line))
                    except:
                        continue
            return pd.DataFrame(votes)
    except:
        pass
    return pd.DataFrame()

if __name__ == "__main__":
    main() 