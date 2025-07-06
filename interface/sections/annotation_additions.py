"""
RLHF Annotation Analytics

This module provides advanced analytics for annotation data:
1. Text Analysis and Word Pattern Recognition
2. Topic-Based Performance Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter

from interface.components.utils import create_time_slider, filter_by_time_range

# Try to import wordcloud, but provide fallback if not available
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

def display_annotation_wordcloud(vote_df):
    """Display wordcloud visualization of annotation language patterns"""
    st.subheader("Word Pattern Analysis")
    
    # Check if we have the necessary data
    if vote_df.empty:
        st.warning("Insufficient training data for analysis.")
        return
    
    # Check if we have prompt and completion data
    if 'prompt' not in vote_df.columns or 'selected_completion' not in vote_df.columns:
        st.warning("Cannot find prompt and completion data for analysis.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Check if we have enough data
    if len(filtered_df) < 5:
        st.info("Insufficient data in this time period for meaningful word pattern analysis.")
        return
    
    # Create tabs for different content types
    tab1, tab2, tab3 = st.tabs(["User Prompts", "Selected Completions", "Combined Analysis"])
    
    # Check if wordcloud is available
    if not WORDCLOUD_AVAILABLE:
        for tab in [tab1, tab2, tab3]:
            with tab:
                st.warning("WordCloud package not available. Please install it with: pip install wordcloud matplotlib")
        return
    
    # Function to generate and display wordcloud
    def generate_wordcloud(text_data, title):
        # Combine all text
        all_text = " ".join(text_data)
        
        # Clean text (remove URLs, special chars, etc.)
        all_text = re.sub(r'http\S+', '', all_text)
        all_text = re.sub(r'[^\w\s]', '', all_text)
        all_text = re.sub(r'\s+', ' ', all_text).strip().lower()
        
        # Remove common stopwords
        stopwords = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'it', 'for', 'with', 'as', 'be', 'this', 'on', 'an', 'by', 'not', 'are', 'or', 'at', 'from', 'was', 'were', 'would', 'will', 'could', 'should', 'can', 'has', 'have', 'had', 'been', 'may', 'might', 'must', 'shall'])
        word_tokens = all_text.split()
        filtered_text = ' '.join([word for word in word_tokens if word.lower() not in stopwords])
        
        # Generate wordcloud
        if filtered_text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=100,
                contour_width=1,
                contour_color='steelblue'
            ).generate(filtered_text)
            
            # Display the wordcloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title)
            st.pyplot(fig)
            
            # Display top words
            word_counts = Counter(filtered_text.split())
            top_words = word_counts.most_common(10)
            
            st.write("### Most Frequent Terms")
            
            # Create bar chart of top words
            top_words_df = pd.DataFrame(top_words, columns=['word', 'count'])
            
            fig = px.bar(
                top_words_df,
                x='count',
                y='word',
                orientation='h',
                title='Term Frequency Analysis',
                labels={'count': 'Frequency', 'word': 'Word'},
                color='count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient text data in this time period for meaningful analysis.")
    
    # Generate wordclouds for each tab
    with tab1:
        prompts = filtered_df['prompt'].fillna("").tolist()
        generate_wordcloud(prompts, "User Prompt Analysis")
    
    with tab2:
        completions = filtered_df['selected_completion'].fillna("").tolist()
        generate_wordcloud(completions, "Selected Completion Analysis")
    
    with tab3:
        combined = filtered_df['prompt'].fillna("").tolist() + filtered_df['selected_completion'].fillna("").tolist()
        generate_wordcloud(combined, "Combined Text Analysis")

def display_theme_based_agreement(vote_df):
    """Display model performance across different topic categories"""
    st.subheader("Performance Analysis by Topic Category")
    
    # Check if we have the necessary data
    if vote_df.empty:
        st.warning("Insufficient training data for analysis.")
        return
    
    # Check if we have model correctness and prompt data
    if 'model_correct' not in vote_df.columns or 'prompt' not in vote_df.columns or 'timestamp' not in vote_df.columns:
        st.warning("Cannot find required data for topic-based analysis.")
        return
    
    # Apply time range filter
    filtered_df = create_time_slider(vote_df)
    
    # Check if we have enough data
    if len(filtered_df) < 10:
        st.info("Insufficient data in this time period for meaningful topic analysis.")
        return
    
    # Convert timestamp to datetime if needed
    if filtered_df['timestamp'].dtype != 'datetime64[ns]':
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    
    # Define themes and their keywords
    themes = {
        "Technical": ["code", "programming", "software", "algorithm", "function", "data", "system", "technical", "technology", "computer", "api", "database"],
        "Creative": ["story", "creative", "write", "design", "art", "music", "imagine", "novel", "poem", "fiction", "creative", "artistic"],
        "Philosophical": ["meaning", "consciousness", "existence", "philosophy", "mind", "reality", "truth", "knowledge", "wisdom", "being", "perception", "thought"],
        "Emotional": ["feel", "emotion", "sentiment", "mood", "affect", "response", "reaction", "expression", "attitude", "perspective", "viewpoint", "opinion"],
        "Analytical": ["analysis", "analyze", "compare", "contrast", "evaluate", "examine", "assess", "interpret", "explain", "describe", "research"]
    }
    
    # Function to assign theme to a prompt
    def assign_theme(prompt):
        prompt_lower = prompt.lower()
        theme_scores = {}
        
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            theme_scores[theme] = score
        
        # Get theme with highest score
        max_score = max(theme_scores.values())
        if max_score > 0:
            # Get all themes with the max score
            max_themes = [theme for theme, score in theme_scores.items() if score == max_score]
            return max_themes[0]  # Return the first theme if there are ties
        else:
            return "Other"
    
    # Add theme column to dataframe
    filtered_df['theme'] = filtered_df['prompt'].fillna("").apply(assign_theme)
    
    # Group data by time period and theme
    filtered_df['month'] = filtered_df['timestamp'].dt.to_period('M')
    
    # Calculate agreement rate by theme and time period
    theme_agreement = filtered_df.groupby(['month', 'theme'])['model_correct'].agg(['mean', 'count']).reset_index()
    theme_agreement = theme_agreement.rename(columns={'mean': 'agreement_rate'})
    
    # Convert month period to string for plotting
    theme_agreement['month_str'] = theme_agreement['month'].astype(str)
    
    # Create visualization
    st.write("### Model Performance by Topic Over Time")
    
    # Create line chart
    fig = px.line(
        theme_agreement,
        x='month_str',
        y='agreement_rate',
        color='theme',
        markers=True,
        title='Model Performance by Topic Category Over Time',
        labels={
            'month_str': 'Time Period',
            'agreement_rate': 'Accuracy Rate',
            'theme': 'Topic Category'
        },
        hover_data=['count']
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time Period',
        yaxis_title='Model Accuracy',
        yaxis=dict(tickformat='.0%'),
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Create theme distribution visualization
    st.write("### Topic Distribution")
    
    # Count themes
    theme_counts = filtered_df['theme'].value_counts().reset_index()
    theme_counts.columns = ['theme', 'count']
    
    # Create pie chart
    fig = px.pie(
        theme_counts,
        values='count',
        names='theme',
        title='Distribution of Topics in Training Data',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create theme performance comparison
    st.write("### Model Performance by Topic Category")
    
    # Calculate overall agreement rate by theme
    theme_overall = filtered_df.groupby('theme')['model_correct'].agg(['mean', 'count']).reset_index()
    theme_overall = theme_overall.rename(columns={'mean': 'agreement_rate'})
    theme_overall = theme_overall.sort_values('agreement_rate', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        theme_overall,
        x='theme',
        y='agreement_rate',
        color='theme',
        title='Model Accuracy by Topic Category',
        labels={
            'theme': 'Topic Category',
            'agreement_rate': 'Accuracy Rate'
        },
        text_auto='.0%',
        hover_data=['count']
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Topic Category',
        yaxis_title='Model Accuracy',
        yaxis=dict(tickformat='.0%'),
        height=400,
        showlegend=False
    )
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    st.write("### Performance Analysis Summary")
    
    # Find themes with highest and lowest agreement
    best_theme = theme_overall.iloc[0]['theme']
    worst_theme = theme_overall.iloc[-1]['theme']
    
    st.write(f"""
    This analysis reveals topic categories where model performance varies:
    
    - **Highest Performance**: The model performs best on **{best_theme}** topics ({theme_overall.iloc[0]['agreement_rate']:.1%} accuracy)
    - **Improvement Needed**: The model needs improvement on **{worst_theme}** topics ({theme_overall.iloc[-1]['agreement_rate']:.1%} accuracy)
    
    **Key Insights:**
    
    1. **Topic-Specific Calibration**: Different topic categories show varying levels of alignment with user preferences
    2. **Training Data Distribution**: Performance may correlate with the quantity and quality of training data per topic
    3. **Model Optimization**: Topic-specific fine-tuning could improve performance on underperforming categories
    
    **Recommendations:**
    
    - Collect additional training data for underperforming topic categories
    - Implement topic-aware model calibration techniques
    - Monitor performance drift across different topic categories over time
    """)

def display_annotation_content_analysis(vote_df):
    st.subheader("Annotation Content Analysis")
    
    # Get all prompts and completions
    prompt_text = ' '.join(vote_df['prompt'].dropna().astype(str))
    completion_text = ' '.join(vote_df['selected_completion'].dropna().astype(str) + ' ' + 
                              vote_df['rejected_completion'].dropna().astype(str))
    
    combined_text = prompt_text + ' ' + completion_text
    
    # Check if we have text to analyze
    if not combined_text.strip():
        st.warning("No text data available for analysis.")
        return
    
    # Create word frequency visualization
    words = combined_text.lower().split()
    word_freq = pd.Series(words).value_counts().head(20)
    
    if not word_freq.empty:
        # Create bar chart
        fig = px.bar(
            x=word_freq.index, 
            y=word_freq.values,
            title='Most Common Words in Annotations',
            labels={'x': 'Word', 'y': 'Word Frequency'},
            color=word_freq.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title='Word',
            yaxis_title='Frequency',
            height=400,
            xaxis={'categoryorder': 'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)