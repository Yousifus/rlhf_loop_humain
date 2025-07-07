"""
RLHF UX Enhancement Suite

Professional UX improvements for the RLHF monitoring system,
making the interface more intuitive, accessible, and functional.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import uuid
from typing import Dict, List, Any, Optional

# Enhanced UX Components

def create_enhanced_sidebar():
    """Create a more intuitive and beautiful sidebar experience"""
    with st.sidebar:
        # Beautiful header with status
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff6b9d, #c44569);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
        ">
                    <h2 style="color: white; margin: 0;">🤖 RLHF Dashboard</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">
            Professional monitoring system
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick actions section
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat Now", use_container_width=True):
                st.session_state.selected_tab = "Talk With Me"
                st.rerun()
        
        with col2:
            if st.button("🔗 Connect", use_container_width=True):
                st.session_state.selected_tab = "Annotation Interface"
                st.rerun()
        
        # Connection status with real-time updates
        st.markdown("### 📊 Connection Status")
        
        # Get connection metrics
        connection_metrics = get_connection_metrics()
        
        # Display metrics with progress bars
        for metric_name, metric_data in connection_metrics.items():
            st.markdown(f"**{metric_name}**")
            progress_col, value_col = st.columns([3, 1])
            
            with progress_col:
                st.progress(metric_data['value'])
            with value_col:
                st.write(f"{metric_data['display']}")
        
        # Recent activity feed
        st.markdown("### 📝 Recent Activity")
        display_activity_feed()

def create_enhanced_navigation():
    """Create a more intuitive navigation system"""
    # Create tabs with better organization and icons
    tab_config = {
        "🏠 Overview": {
            "key": "Our Overview",
            "description": "View model performance metrics",
            "icon": "🏠"
        },
        "💬 Chat": {
            "key": "Talk With Me", 
            "description": "Direct conversation with me",
            "icon": "💬"
        },
        "🔗 Connect": {
            "key": "Training Interface",
            "description": "Provide model training feedback",
            "icon": "🔗"
        },
        "📚 History": {
            "key": "Training History",
            "description": "View training data history",
            "icon": "📚"
        },
        "📈 Analytics": {
            "key": "Model Alignment Analytics",
            "description": "View model training progression",
            "icon": "📈"
        },
        "🔧 Advanced": {
            "key": "advanced_menu",
            "description": "Advanced features and settings",
            "icon": "🔧"
        }
    }
    
    # Create navigation with better visual hierarchy
    st.markdown("### 🧭 Dashboard Navigation")
    
    # Primary navigation
    primary_tabs = ["🏠 Overview", "💬 Chat", "🔗 Connect", "📚 History"]
    selected_primary = st.radio(
        "Main Areas",
        primary_tabs,
        format_func=lambda x: f"{tab_config[x]['icon']} {x.split(' ', 1)[1]}",
        help="Choose dashboard section to view"
    )
    
    # Secondary navigation for advanced features
    with st.expander("🔧 Advanced Features"):
        advanced_tabs = [
            "Model Alignment Analytics",
            "Performance Metrics", 
            "Error Analysis",
            "Model Evolution",
            "Training Progress Analysis"
        ]
        
        selected_advanced = st.selectbox(
            "Advanced Analytics",
            ["None"] + advanced_tabs,
            help="Deep dive into model performance analytics"
        )
        
        if selected_advanced != "None":
            st.session_state.selected_tab = selected_advanced
        else:
            st.session_state.selected_tab = tab_config[selected_primary]["key"]
    
    return st.session_state.get("selected_tab", "Our Overview")

def create_enhanced_metrics_display(vote_df, predictions_df, reflections_df):
    """Create more engaging and informative metrics display"""
    
    # Hero metrics section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(196, 69, 105, 0.1));
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 107, 157, 0.2);
    ">
        <h2 style="text-align: center; color: #1DB584; margin-bottom: 20px; font-weight: 600;">
            📊 Model Performance Metrics
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 4 main metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Model Performance",
            calculate_connection_depth(vote_df),
            "🔗",
            "Overall model performance level"
        )
    
    with col2:
        create_metric_card(
            "Learning Progress", 
            calculate_learning_progress(vote_df),
            "🧠",
            "User preference learning progress"
        )
    
    with col3:
        create_metric_card(
            "Interaction Quality",
            calculate_engagement_level(vote_df, reflections_df),
            "💬",
            "User-model interaction effectiveness"
        )
    
    with col4:
        create_metric_card(
            "Alignment Rate",
            calculate_sync_rate(vote_df, predictions_df),
            "🎯",
            "Model-user preference alignment"
        )

def create_metric_card(title, value, icon, description):
    """Create a beautiful metric card"""
    # Determine color based on value
    if value >= 0.8:
        color = "#4CAF50"  # Green
        bg_color = "rgba(76, 175, 80, 0.1)"
    elif value >= 0.6:
        color = "#FF9800"  # Orange
        bg_color = "rgba(255, 152, 0, 0.1)"
    else:
        color = "#F44336"  # Red
        bg_color = "rgba(244, 67, 54, 0.1)"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid {color}33;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 2em; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 1.2em; font-weight: bold; color: {color}; margin-bottom: 5px;">
            {title}
        </div>
        <div style="font-size: 2em; font-weight: bold; color: {color}; margin-bottom: 5px;">
            {value:.1%}
        </div>
        <div style="font-size: 0.8em; color: #666; opacity: 0.8;">
            {description}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_chat_interface():
    """Create a more engaging chat interface"""
    
    # Chat header with HUMAIN styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1DB584 0%, #17a573 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(29, 181, 132, 0.1);
    ">
        <h2 style="color: white; margin: 0; font-weight: 600;">💬 Interactive Chat Interface</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-weight: 400;">
            Direct user-model interaction interface
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced chat settings in an elegant sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Conversation Settings")
        
        # Personality mode with descriptions
        personality_options = {
                    "analytical": "🔍 Analytical & Thorough",
        "conversational": "💬 Natural & Conversational", 
        "precise": "🎯 Precise & Concise"
        }
        
        selected_personality = st.selectbox(
            "My Personality Mode",
            list(personality_options.keys()),
            format_func=lambda x: personality_options[x],
            help="Choose how I express myself to you"
        )
        
        # Advanced settings in expander
        with st.expander("🔧 Advanced Settings"):
            temperature = st.slider(
                "Response Creativity",
                0.0, 1.0, 0.7, 0.1,
                help="Higher = more creative and unpredictable"
            )
            
            max_tokens = st.slider(
                "Response Length",
                100, 2000, 500, 100,
                help="How detailed my responses will be"
            )
            
            # Save settings
            if st.button("💾 Save My Preferences"):
                save_chat_preferences(selected_personality, temperature, max_tokens)
                st.success("User preferences have been saved successfully!")

def create_enhanced_annotation_interface():
    """Create a more intuitive annotation interface"""
    
    # Progress indicator
    progress_data = get_annotation_progress()
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(196, 69, 105, 0.1));
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    ">
        <h3 style="color: #1DB584; margin-bottom: 15px; font-weight: 600;">📈 Model Performance Progress</h3>
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Training Sessions: {progress_data['current']}</span>
                <span>Target: {progress_data['target']}</span>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 5px; height: 10px;">
                <div style="
                    background: linear-gradient(135deg, #1DB584 0%, #17a573 100%);
                    height: 100%;
                    border-radius: 5px;
                    width: {progress_data['percentage']:.1f}%;
                "></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced prompt input with suggestions
    st.markdown("### 💭 Provide Training Input")
    
    # Prompt suggestions
    with st.expander("💡 Need inspiration? Try these prompts"):
        suggestion_categories = {
            "Personal": [
                "Tell me about your dreams and aspirations",
                "What makes you feel most alive?",
                "Describe your perfect day"
            ],
            "Creative": [
                "Write a short story about...",
                "Imagine a world where...",
                "Create a poem about..."
            ],
            "Analytical": [
                "Explain the concept of...",
                "Compare and contrast...",
                "Analyze the implications of..."
            ]
        }
        
        for category, prompts in suggestion_categories.items():
            st.markdown(f"**{category}:**")
            for prompt in prompts:
                if st.button(f"💫 {prompt}", key=f"suggestion_{prompt}"):
                    st.session_state.suggested_prompt = prompt

def create_real_time_feedback_system():
    """Create immediate feedback and response system"""
    
    # Real-time connection status
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = "Connected"
    
    # Status indicator
    status_colors = {
        "Connected": "#4CAF50",
        "Thinking": "#FF9800", 
        "Learning": "#2196F3",
        "Error": "#F44336"
    }
    
    status_color = status_colors.get(st.session_state.connection_status, "#666")
    
    st.markdown(f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        background: {status_color};
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    ">
        ● {st.session_state.connection_status}
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_data_visualization():
    """Create more engaging and informative visualizations"""
    
    def create_connection_timeline(vote_df):
        """Create an interactive timeline of model training"""
        if vote_df.empty:
            return None
        
        # Prepare data
        df = vote_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate by date
        daily_stats = df.groupby('date').agg({
            'model_correct': ['count', 'mean'],
            'timestamp': 'count'
        }).reset_index()
        
        daily_stats.columns = ['date', 'total_annotations', 'accuracy', 'interactions']
        
        # Create interactive timeline
        fig = go.Figure()
        
        # Add accuracy line
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['accuracy'],
            mode='lines+markers',
            name='Understanding Level',
            line=dict(color='#1DB584', width=3),
            marker=dict(size=8),
            hovertemplate='<b>%{x}</b><br>Understanding: %{y:.1%}<extra></extra>'
        ))
        
        # Add interaction volume as bar chart
        fig.add_trace(go.Bar(
            x=daily_stats['date'],
            y=daily_stats['total_annotations'],
            name='Daily Interactions',
            marker_color='rgba(29, 181, 132, 0.3)',
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Interactions: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title='Model Performance Journey Over Time',
            xaxis_title='Date',
            yaxis=dict(
                title='Understanding Level',
                tickformat='.0%',
                side='left'
            ),
            yaxis2=dict(
                title='Daily Interactions',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_preference_heatmap(vote_df):
        """Create a heatmap showing preference patterns"""
        if vote_df.empty or 'human_choice' not in vote_df.columns:
            return None
        
        # Prepare data
        df = vote_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        # Create preference matrix
        preference_matrix = df.groupby(['day_of_week', 'hour'])['human_choice'].apply(
            lambda x: (x == 'A').mean()
        ).reset_index()
        
        # Pivot for heatmap
        heatmap_data = preference_matrix.pivot(
            index='day_of_week', 
            columns='hour', 
            values='human_choice'
        )
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            title='When You Prefer Different Response Styles',
            labels=dict(x="Hour of Day", y="Day of Week", color="Preference A Rate"),
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig

# Helper functions for metrics calculation

def calculate_connection_depth(vote_df):
    """Calculate model-user interaction quality metrics"""
    if vote_df.empty:
        return 0.0
    
    # Base on number of interactions and recency
    total_interactions = len(vote_df)
    if 'timestamp' in vote_df.columns:
        recent_interactions = len(vote_df[
            pd.to_datetime(vote_df['timestamp']) > (datetime.now() - timedelta(days=7))
        ])
        recency_factor = min(1.0, recent_interactions / 10)
    else:
        recency_factor = 0.5
    
    volume_factor = min(1.0, total_interactions / 100)
    
    return (volume_factor * 0.7 + recency_factor * 0.3)

def calculate_learning_progress(vote_df):
    """Calculate model learning progress from training data"""
    if vote_df.empty or 'model_correct' not in vote_df.columns:
        return 0.0
    
    return vote_df['model_correct'].mean()

def calculate_engagement_level(vote_df, reflections_df):
    """Calculate user-model interaction quality level"""
    if vote_df.empty:
        return 0.0
    
    # Base on interaction frequency and reflection quality
    interaction_factor = min(1.0, len(vote_df) / 50)
    
    if not reflections_df.empty and 'quality_score' in reflections_df.columns:
        reflection_factor = reflections_df['quality_score'].mean()
    else:
        reflection_factor = 0.5
    
    return (interaction_factor * 0.6 + reflection_factor * 0.4)

def calculate_sync_rate(vote_df, predictions_df):
    """Calculate model-user preference alignment rate"""
    if vote_df.empty:
        return 0.0
    
    if 'model_choice' in vote_df.columns and 'human_choice' in vote_df.columns:
        return (vote_df['model_choice'] == vote_df['human_choice']).mean()
    
    return 0.5

def get_connection_metrics():
    """Get current connection metrics for sidebar"""
    # This would normally pull from session state or database
    return {
        "Model Performance": {"value": 0.75, "display": "75%"},
        "Learning Progress": {"value": 0.82, "display": "82%"},
        "Interaction Quality": {"value": 0.68, "display": "68%"},
        "Alignment Rate": {"value": 0.79, "display": "79%"}
    }

def display_activity_feed():
    """Display recent activity in sidebar"""
    # Mock recent activities
    activities = [
        {"time": "2 min ago", "action": "💬 Chat message", "status": "✓"},
        {"time": "15 min ago", "action": "📊 Preference learned", "status": "✓"},
        {"time": "1 hour ago", "action": "🧠 Model updated", "status": "✓"},
        {"time": "3 hours ago", "action": "📊 Analysis completed", "status": "✓"}
    ]
    
    for activity in activities:
        st.markdown(f"""
        <div style="
            background: rgba(29, 181, 132, 0.1);
            padding: 8px;
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: 12px;
        ">
            <div style="display: flex; justify-content: space-between;">
                <span>{activity['action']}</span>
                <span style="color: #4CAF50;">{activity['status']}</span>
            </div>
            <div style="color: #666; font-size: 10px;">{activity['time']}</div>
        </div>
        """, unsafe_allow_html=True)

def get_annotation_progress():
    """Get annotation progress data"""
    # This would normally come from database
    return {
        "current": 127,
        "target": 500,
        "percentage": 25.4
    }

def save_chat_preferences(personality, temperature, max_tokens):
    """Save user's chat preferences"""
    preferences = {
        "personality": personality,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to session state
    st.session_state.chat_preferences = preferences
    
    # Could also save to database here
    return True

# Enhanced error handling and user feedback

def create_error_handler():
    """Create beautiful error handling"""
    
    def handle_error(error_message, error_type="error"):
        """Display errors in system style"""
        
        if error_type == "error":
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.05));
                border-left: 4px solid #F44336;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <p style="color: #F44336; font-weight: bold; margin: 0;">
                    ⚠️ System connection interrupted
                </p>
                <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">
                    {error_message}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        elif error_type == "warning":
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(255, 152, 0, 0.05));
                border-left: 4px solid #FF9800;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            ">
                <p style="color: #FF9800; font-weight: bold; margin: 0;">
                    ⚠️ System notification required
                </p>
                <p style="color: #666; margin: 5px 0 0 0; font-size: 14px;">
                    {error_message}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    return handle_error

# Loading states and progress indicators

def create_loading_animation(message="I'm thinking about you..."):
    """Create beautiful loading animations"""
    
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1), rgba(196, 69, 105, 0.1));
        border-radius: 15px;
        margin: 20px 0;
    ">
        <div style="
            display: inline-block;
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 107, 157, 0.3);
            border-radius: 50%;
            border-top-color: #1DB584;
            animation: spin 1s ease-in-out infinite;
        "></div>
        <p style="color: #1DB584; margin-top: 15px; font-style: italic;">
            {message}
        </p>
    </div>
    
    <style>
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Accessibility improvements

def create_accessibility_features():
    """Add accessibility features"""
    
    # Keyboard shortcuts
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to submit forms
        if (e.ctrlKey && e.key === 'Enter') {
            const submitButtons = document.querySelectorAll('button[kind="primary"]');
            if (submitButtons.length > 0) {
                submitButtons[0].click();
            }
        }
        
        // Escape to close modals/expanders
        if (e.key === 'Escape') {
            const expandedElements = document.querySelectorAll('[data-testid="stExpander"][aria-expanded="true"]');
            expandedElements.forEach(el => el.click());
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # High contrast mode toggle
    if st.sidebar.checkbox("🔆 High Contrast Mode"):
        st.markdown("""
        <style>
        .stApp {
            filter: contrast(1.5) brightness(1.2);
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Font size adjustment
    font_size = st.sidebar.slider("📝 Text Size", 12, 20, 14)
    st.markdown(f"""
    <style>
    .stApp {{
        font-size: {font_size}px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Mobile responsiveness

def create_mobile_optimizations():
    """Optimize for mobile devices"""
    
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .stColumns > div {
            min-width: 100% !important;
            margin-bottom: 1rem;
        }
        
        .stSidebar {
            width: 100% !important;
        }
        
        .stButton > button {
            width: 100% !important;
            margin-bottom: 0.5rem;
        }
        
        .metric-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Performance optimizations

def create_performance_optimizations():
    """Add performance optimizations"""
    
    # Lazy loading for heavy components
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_heavy_visualization_data():
        """Load and cache heavy visualization data"""
        # This would load expensive computations
        return {"status": "loaded"}
    
    # Pagination for large datasets
    def paginate_dataframe(df, page_size=50):
        """Paginate large dataframes"""
        if df.empty:
            return df
        
        total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
        
        if total_pages > 1:
            page = st.selectbox(
                "Page",
                range(1, total_pages + 1),
                format_func=lambda x: f"Page {x} of {total_pages}"
            )
            
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            return df.iloc[start_idx:end_idx]
        
        return df
    
    return load_heavy_visualization_data, paginate_dataframe
