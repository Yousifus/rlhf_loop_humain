"""
RLHF Annotation Interface

Interface for collecting user preferences and annotations on model outputs.
Supports comparison-based evaluation and preference data collection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import uuid
import random
import time
import logging

from interface.components.utils import (
    create_time_slider,
    filter_by_time_range,
    format_timestamp,
    generate_unique_id
)
from interface.components.data_loader import save_annotation

# Import additional annotation visualizations
from interface.sections.annotation_additions import (
    display_annotation_wordcloud,
    display_theme_based_agreement
)

# Configure logging
logger = logging.getLogger(__name__)

def display_annotation_interface(vote_df):
    """Display the model training interface for user feedback"""
    st.header("🔧 Model Training Interface")
    
    # Provide context and guidance for users
    with st.expander("ℹ️ How to Use This Interface", expanded=True):
        st.markdown("""
        This is the model training interface where you can provide feedback and shape the AI's behavior:
        
        1. Enter your prompt or question in the text area
        2. Generate responses using the 'Generate Responses' button
        3. Select the response that better meets your requirements
        4. Submit your choice to improve model performance
        
        Each selection provides valuable feedback for model training and improvement.
        """)
    
    # Model settings in sidebar with system configuration
    with st.sidebar:
        st.subheader("⚙️ Model Configuration")
        model_mode = st.selectbox(
            "Response Style",
            options=["analytical", "conversational", "precise"],
            index=0,
            help="Choose the AI's response personality and style"
        )
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Response randomness: 0 = consistent, 1 = creative"
        )
        max_tokens = st.slider(
            "Response Length",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum response length in tokens"
        )
        
        # API key input
        api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            help="Optional API key for model access"
        )
        
        # Apply settings button
        if st.button("Update Settings"):
            st.session_state.annotation_settings = {
                "model_mode": model_mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": api_key if api_key else None
            }
            st.success("Settings have been updated successfully!")
    
    # Create feedback collection form
    with st.form("connection_form"):
        # Prompt input
        prompt = st.text_area(
            "Enter your prompt:",
            height=100,
            placeholder="Enter your prompt or question here..."
        )
        
        # Generate completions button
        generate_submitted = st.form_submit_button("Generate Responses")
        
        # Store completions in session state if newly generated
        if generate_submitted and prompt.strip():
            with st.spinner("Generating completions..."):
                try:
                    # Get API client with current settings
                    from utils.api_client import get_api_client, ModelAPIClient
                    
                    # Get settings
                    settings = st.session_state.get("annotation_settings", {})
                    api_key = settings.get("api_key")
                    model_mode = settings.get("model_mode", "analytical")
                    temperature = settings.get("temperature", 0.7)
                    max_tokens = settings.get("max_tokens", 300)
                    
                    # Initialize API client with correct DeepSeek model
                    api_client = ModelAPIClient(api_key=api_key, model="deepseek-chat")
                    
                    # Set the response mode/personality
                    api_client.set_model_mode(model_mode)
                    
                    # Generate completions
                    completions = api_client.generate_comparison(
                        prompt=prompt,
                        num_completions=2,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    # Store in session state
                    st.session_state.prompt = prompt
                    st.session_state.prompt_id = f"prompt_{int(time.time())}"
                    st.session_state.completion_a = completions[0]["completion"]
                    st.session_state.completion_b = completions[1]["completion"]
                    st.session_state.completions_metadata = {
                        "model": "deepseek-chat",
                        "model_mode": model_mode,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    st.error(f"Error generating completions: {str(e)}")
                    logger.error(f"Completion generation error: {str(e)}")
    
            # Show completions if available
    if st.session_state.get("prompt") and st.session_state.get("completion_a") and st.session_state.get("completion_b"):
        st.write("### Select Preferred Response")
        
        # Show model info
        if st.session_state.get("completions_metadata"):
            metadata = st.session_state.get("completions_metadata")
            model_info = f"Model: {metadata.get('model', 'deepseek-chat')}"
            if 'model_mode' in metadata:
                model_info += f" ({metadata.get('model_mode')} mode)"
            model_info += f" | Temperature: {metadata.get('temperature', 0.7)}"
            st.info(model_info)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Response A:**")
            st.info(st.session_state.completion_a)
            
            select_a = st.button("Select Response A")
        
        with col2:
            st.write("**Response B:**")
            st.info(st.session_state.completion_b)
            
            select_b = st.button("Select Response B")
        
        # Handle selection
        if select_a or select_b:
            selected_completion = st.session_state.completion_a if select_a else st.session_state.completion_b
            unselected_completion = st.session_state.completion_b if select_a else st.session_state.completion_a
            
            # Create annotation data
            annotation_data = {
                "id": generate_unique_id("vote"),
                "prompt_id": st.session_state.get("prompt_id", f"prompt_{int(time.time())}"),
                "prompt": st.session_state.prompt,
                "completion_a": st.session_state.completion_a,
                "completion_b": st.session_state.completion_b,
                "selected_completion": selected_completion,
                "rejected_completion": unselected_completion,
                "timestamp": datetime.now().isoformat(),
                "model_choice": "A" if select_a else "B",
                "human_choice": "A" if select_a else "B",
                "preference": "Completion A" if select_a else "Completion B",
                "model_correct": True,  # For demonstration, assuming model is always correct
                "metadata": st.session_state.get("completions_metadata", {})
            }
            
            # Save annotation feedback using centralized database
            if save_annotation(annotation_data):
                # Also save to database for comprehensive tracking
                try:
                    from utils.database import get_database
                    db = get_database()
                    
                    # Save as model training data
                    training_success = db.save_model_training_data({
                        "user_input": annotation_data["prompt"],
                        "model_response": annotation_data["selected_completion"],
                        "training_mode": "annotation",
                        "emotional_state": "learning",
                        "interaction_type": "annotation",
                        "user_feedback": "positive",  # They selected this completion
                        "confidence": 0.85,
                        "response_quality": "preferred",
                        "training_weight": 1.5,  # Higher weight for explicit preferences
                        "personality_emphasis": ["responsive", "adaptive", "learning"],
                        "learning_priority": "high"
                    })
                    
                    # Save reflection about the annotation
                    reflection_text = f"User chose completion {annotation_data['human_choice']} for prompt: {annotation_data['prompt'][:100]}... This improves model alignment."
                    db.save_model_reflection(reflection_text, analysis_type="learning")
                    
                except Exception as e:
                    logger.error(f"Error saving to database: {str(e)}")
                
                st.success("Feedback recorded successfully. Your input helps improve model performance.")
                
                # Clear form for next annotation
                st.session_state.prompt = ""
                st.session_state.completion_a = ""
                st.session_state.completion_b = ""
                st.session_state.prompt_id = ""
                st.session_state.completions_metadata = {}
                
                # Force rerun to update UI
                st.rerun()
    
            # Show training statistics
    if not vote_df.empty:
        st.subheader("System Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Annotations", len(vote_df))
        
        with col2:
            # Calculate model agreement rate if available
            if 'model_correct' in vote_df.columns:
                agreement_rate = vote_df['model_correct'].mean() * 100
                st.metric("Model Accuracy", f"{agreement_rate:.1f}%")
        
        with col3:
            # Calculate recent annotation count
            if 'timestamp' in vote_df.columns:
                df = vote_df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent_count = len(df[df['timestamp'] > (datetime.now() - timedelta(days=7))])
                st.metric("Recent Activity (7d)", recent_count)

def display_annotation_history(vote_df, predictions_df):
    """Display training data history"""
    st.header("📊 Training Data History")
    
    if vote_df.empty:
        st.warning("No annotation data available yet.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Show recent annotations
    st.subheader("Recent Annotations")
    
    # Filter for most recent annotations
    recent_annotations = filtered_df.sort_values("timestamp", ascending=False).head(10)
    
    if not recent_annotations.empty:
        for idx, row in recent_annotations.iterrows():
            with st.expander(f"Annotation #{idx+1}: {row.get('timestamp', 'Unknown date')}", expanded=False):
                st.write("**Prompt:**")
                st.info(row.get("prompt", "No prompt available"))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Selected Completion:**")
                    st.success(row.get("selected_completion", "Not available"))
                
                with col2:
                    st.write("**Unselected Completion:**")
                    st.error(row.get("rejected_completion", "Not available"))
                
                # Show model correctness if available
                if 'model_correct' in row:
                    st.write("**Model Correctness:**")
                    if row['model_correct']:
                        st.success("✓ Model prediction matched human preference")
                    else:
                        st.error("✗ Model prediction did not match human preference")
    else:
        st.info("No annotations found in the selected time range.")
    
    # Annotation activity calendar
    if 'timestamp' in filtered_df.columns:
        st.subheader("Annotation Activity")
        
        # Convert timestamp to datetime
        filtered_df['date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
        
        # Count annotations per day
        date_counts = filtered_df.groupby('date').size().reset_index(name='count')
        
        # Create calendar heatmap
        # We'll use a workaround with a bar chart since Streamlit doesn't natively support calendar heatmaps
        fig = px.bar(
            date_counts,
            x='date',
            y='count',
            title='Daily Annotation Count',
            labels={'date': 'Date', 'count': 'Annotations'},
            color='count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Annotations',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download Annotation Data (CSV)",
            csv,
            file_name=f"rlhf_annotations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data to export")

def display_preference_timeline(vote_df):
    """Display user preference evolution over time"""
    st.header("📈 User Preference Evolution")
    
    if vote_df.empty:
        st.warning("No annotation data available yet.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Check if we have the necessary data
    if 'human_choice' not in filtered_df.columns or 'timestamp' not in filtered_df.columns:
        st.warning("Human choice or timestamp data not available.")
        return
    
    # Convert timestamp to datetime
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    
    # Group by date and calculate preference patterns
    filtered_df['date'] = filtered_df['timestamp'].dt.date
    
    # Show preference trends over time
    st.subheader("Preference Patterns Over Time")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Preference Distribution",
        "Model Agreement",
        "Annotation Content Wordcloud",
        "Theme-Based Agreement Shifts"
    ])
    
    with tab1:
        # Calculate preference distribution over time
        if 'human_choice' in filtered_df.columns:
            # Group by date and count preferences
            pref_counts = filtered_df.groupby(['date', 'human_choice']).size().reset_index(name='count')
            
            # Create a pivot table for easier plotting
            pivot_df = pref_counts.pivot(index='date', columns='human_choice', values='count').fillna(0)
            
            # Reset index to make date a column again
            pivot_df = pivot_df.reset_index()
            
            # Make sure we have 'A' and 'B' columns
            if 'A' not in pivot_df.columns:
                pivot_df['A'] = 0
            if 'B' not in pivot_df.columns:
                pivot_df['B'] = 0
            
            # Create stacked area chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=pivot_df['date'],
                y=pivot_df['A'],
                name='Preference A',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(26, 118, 255, 0.5)'
            ))
            
            fig.add_trace(go.Scatter(
                x=pivot_df['date'],
                y=pivot_df['B'],
                name='Preference B',
                mode='lines',
                stackgroup='one',
                fillcolor='rgba(255, 107, 107, 0.5)'
            ))
            
            fig.update_layout(
                title='User Preference Distribution Over Time',
                xaxis_title='Date',
                yaxis_title='Count',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate preference ratio
            preference_ratio = filtered_df['human_choice'].value_counts(normalize=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create donut chart for overall preference distribution
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=preference_ratio.index,
                    values=preference_ratio.values,
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=['rgba(26, 118, 255, 0.8)', 'rgba(255, 107, 107, 0.8)'])
                ))
                
                fig.update_layout(
                    title='Overall Preference Distribution',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Calculate preference stability
                filtered_df['prev_choice'] = filtered_df.sort_values('timestamp')['human_choice'].shift(1)
                filtered_df['choice_changed'] = filtered_df['human_choice'] != filtered_df['prev_choice']
                
                # Calculate change rate
                change_rate = filtered_df.dropna(subset=['prev_choice'])['choice_changed'].mean()
                stability = 1 - change_rate
                
                # Create gauge chart for preference stability
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=stability * 100,
                    title={'text': "Preference Stability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Calculate model agreement over time
        if 'model_correct' in filtered_df.columns:
            # Group by date and calculate agreement rate
            agreement_df = filtered_df.groupby('date')['model_correct'].agg(['mean', 'count']).reset_index()
            agreement_df = agreement_df.rename(columns={'mean': 'agreement_rate'})
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=agreement_df['date'],
                y=agreement_df['agreement_rate'],
                mode='lines+markers',
                name='Agreement Rate',
                line=dict(color='rgba(26, 118, 255, 0.8)', width=3),
                marker=dict(size=8)
            ))
            
            # Add reference line for 50% (random chance)
            fig.add_shape(
                type="line",
                x0=agreement_df['date'].min(),
                y0=0.5,
                x1=agreement_df['date'].max(),
                y1=0.5,
                line=dict(color="red", dash="dash")
            )
            
            # Add annotation for random chance
            fig.add_annotation(
                x=agreement_df['date'].max(),
                y=0.5,
                text="Random Chance",
                showarrow=False,
                yshift=10,
                font=dict(color="red")
            )
            
            fig.update_layout(
                title='Model-Human Agreement Over Time',
                xaxis_title='Date',
                yaxis_title='Agreement Rate',
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add metrics for agreement
            col1, col2, col3 = st.columns(3)
            
            with col1:
                overall_agreement = filtered_df['model_correct'].mean()
                st.metric(
                    "Overall Agreement", 
                    f"{overall_agreement:.1%}"
                )
            
            with col2:
                recent_agreement = filtered_df.sort_values('timestamp').tail(30)['model_correct'].mean()
                st.metric(
                    "Recent Agreement", 
                    f"{recent_agreement:.1%}", 
                    delta=f"{recent_agreement - overall_agreement:.1%}"
                )
            
            with col3:
                confidence_col = 'confidence' if 'confidence' in filtered_df.columns else None
                if confidence_col:
                    avg_confidence = filtered_df[confidence_col].mean()
                    calibration_gap = avg_confidence - overall_agreement
                    
                    st.metric(
                        "Calibration Gap", 
                        f"{calibration_gap:.1%}", 
                        help="Difference between average confidence and actual agreement rate"
                    )
                else:
                    st.metric(
                        "Annotations", 
                        f"{len(filtered_df)}"
                    )

    with tab3:
        display_annotation_wordcloud(vote_df)
    
    with tab4:
        display_theme_based_agreement(vote_df)

def generate_mock_completion(prompt_text, completion_id="a"):
    """Generate a mock completion for demonstration"""
    # Define some completion templates for demonstration
    completion_templates = {
        "factual": [
            "Here's a straightforward explanation: {prompt_content}...",
            "Let me explain this clearly: {prompt_content}...",
            "The answer is quite simple: {prompt_content}..."
        ],
        "analytical": [
            "Analyzing this question: {prompt_content}...",
            "There are several aspects to consider: {prompt_content}...",
            "Looking at this from multiple perspectives: {prompt_content}..."
        ],
        "creative": [
            "Thinking creatively about this: {prompt_content}...",
            "Here's an imaginative take: {prompt_content}...",
            "Let's explore this creatively: {prompt_content}..."
        ]
    }
    
    # Select completion style based on prompt
    if "explain" in prompt_text.lower() or "what is" in prompt_text.lower():
        style = "factual"
    elif "analyze" in prompt_text.lower() or "compare" in prompt_text.lower():
        style = "analytical"
    elif "creative" in prompt_text.lower() or "imagine" in prompt_text.lower():
        style = "creative"
    else:
        style = random.choice(list(completion_templates.keys()))
    
    # Get random template from selected style
    template = random.choice(completion_templates[style])
    
    # Generate completion
    completion = template.format(prompt_content=prompt_text[:50])
    
    # Add some variation based on completion_id
    if completion_id == "a":
        completion += "\n\nThis is particularly important because it demonstrates key principles in this domain."
    else:
        completion += "\n\nWe can see how this applies to several real-world scenarios and examples."
    
    return completion
