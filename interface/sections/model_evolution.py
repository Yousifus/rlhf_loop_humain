"""
This section analyzes how the model has evolved and improved over time
to meet user requirements more effectively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from interface.components.data_loader import get_model_checkpoints
from interface.components.utils import create_time_slider, filter_by_time_range, format_timestamp

def display_model_evolution(vote_df, predictions_df):
    """Display model development and improvement history"""
    st.header("🚀 Model Evolution History")
    
    # Get model checkpoints data
    checkpoints = get_model_checkpoints()
    
    if not checkpoints:
        st.info("No model checkpoint data available yet.")
        
        # If no checkpoint data, show basic performance stats
        if not vote_df.empty and 'model_correct' in vote_df.columns:
            st.subheader("Current Model Performance")
            
            # Calculate overall accuracy
            current_accuracy = vote_df['model_correct'].mean()
            
            st.metric("Model Accuracy", f"{current_accuracy:.2%}")
            
            # Show placeholder for future development
            st.markdown("""
            #### Model Development Roadmap
            This section will track model versions and improvements across different iterations.
            
            Features planned for development:
            - Checkpoint-to-checkpoint performance comparison
            - Training progress visualization
            - Model architecture evolution tracking
            - Performance optimization history
            """)
            
        return
        
    # Convert checkpoint data to DataFrame
    checkpoint_df = pd.DataFrame(checkpoints)
    
    # Convert timestamp strings to datetime
    checkpoint_df['timestamp'] = pd.to_datetime(checkpoint_df['timestamp'])
    
    # Sort by timestamp
    checkpoint_df = checkpoint_df.sort_values('timestamp')
    
    # Display checkpoint overview
    st.subheader("Model Development Timeline")
    
    # Create visualization showing model evolution
    fig = go.Figure()
    
    # Add accuracy trace
    fig.add_trace(go.Scatter(
        x=checkpoint_df['timestamp'],
        y=checkpoint_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#4ecdc4', width=3),
        hovertemplate='%{y:.2%}<br>%{text}',
        text=checkpoint_df['notes']
    ))
    
    # Add calibration error trace
    fig.add_trace(go.Scatter(
        x=checkpoint_df['timestamp'],
        y=checkpoint_df['calibration_error'],
        mode='lines+markers',
        name='Calibration Error',
        line=dict(color='#ff6b6b', width=3),
        yaxis='y2',
        hovertemplate='%{y:.2%}<br>%{text}',
        text=checkpoint_df['notes']
    ))
    
    # Add confidence trend
    fig.add_trace(go.Scatter(
        x=checkpoint_df['timestamp'],
        y=checkpoint_df['confidence_avg'],
        mode='lines+markers',
        name='Avg Confidence',
        line=dict(color='#f9c74f', width=2, dash='dash'),
        hovertemplate='%{y:.2%}<br>%{text}',
        text=checkpoint_df['notes']
    ))
    
    # Add training sample size as bubble size
    fig.add_trace(go.Scatter(
        x=checkpoint_df['timestamp'],
        y=[0.5] * len(checkpoint_df),  # Fixed position
        mode='markers',
        marker=dict(
            size=checkpoint_df['training_samples'] / 50,  # Scale for visualization
            sizemode='area',
            sizeref=max(checkpoint_df['training_samples']) / (50**2),
            sizemin=5,
            color='rgba(100, 150, 240, 0.5)'
        ),
        name='Training Data Size',
        yaxis='y3',
        hovertemplate='%{text} samples',
        text=checkpoint_df['training_samples'].astype(str),
        showlegend=False
    ))
    
    # Add version annotations
    for i, row in checkpoint_df.iterrows():
        fig.add_annotation(
            x=row['timestamp'],
            y=row['accuracy'],
            text=row['version'],
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-30
        )
    
    # Update layout
    fig.update_layout(
        title='Model Performance Evolution',
        xaxis=dict(title='Development Timeline'),
        yaxis=dict(
            title='Accuracy',
            range=[0, 1],
            tickformat='.0%',
            side='left'
        ),
        yaxis2=dict(
            title='Calibration Error',
            range=[0, 0.5],
            tickformat='.0%',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        yaxis3=dict(
            range=[0.4, 0.6],  # Fixed range for bubble positioning
            showticklabels=False,
            showgrid=False,
            overlaying='y'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=600,
        hovermode='x unified'
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display checkpoint details
    st.subheader("Model Version History")
    
    # Create tabs for different checkpoints
    checkpoint_tabs = st.tabs([ckpt['version'] for ckpt in checkpoints])
    
    for i, tab in enumerate(checkpoint_tabs):
        with tab:
            ckpt = checkpoints[i]
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Accuracy", f"{ckpt['accuracy']:.2%}")
                st.write(f"**Architecture:** {ckpt['model_architecture']}")
            
            with col2:
                st.metric("Calibration Error", f"{ckpt['calibration_error']:.2%}")
                st.write(f"**Learning Rate:** {ckpt['learning_rate']}")
            
            with col3:
                st.metric("Average Confidence", f"{ckpt['confidence_avg']:.2%}")
                st.write(f"**Training Samples:** {ckpt['training_samples']}")
            
            # Show notes
            st.subheader("Version Changes")
            st.info(ckpt['notes'])
            
            # Add comparison with previous version if not the first
            if i > 0:
                prev_ckpt = checkpoints[i-1]
                
                st.subheader("Performance Improvements")
                
                # Calculate changes
                acc_change = ckpt['accuracy'] - prev_ckpt['accuracy']
                cal_change = ckpt['calibration_error'] - prev_ckpt['calibration_error']
                conf_change = ckpt['confidence_avg'] - prev_ckpt['confidence_avg']
                sample_change = ckpt['training_samples'] - prev_ckpt['training_samples']
                
                # Display metrics with changes
                delta_col1, delta_col2, delta_col3, delta_col4 = st.columns(4)
                
                with delta_col1:
                    st.metric(
                        "Accuracy Change",
                        f"{acc_change:.2%}", 
                        help="Change in model accuracy from previous version"
                    )
                
                with delta_col2:
                    st.metric(
                        "Calibration Change",
                        f"{cal_change:.2%}",
                        delta=f"{-cal_change:.2%}",  # Negative is better for calibration error
                        delta_color="inverse",
                        help="Change in calibration error from previous version"
                    )
                
                with delta_col3:
                    st.metric(
                        "Confidence Change",
                        f"{conf_change:.2%}", 
                        help="Change in average confidence from previous version"
                    )
                
                with delta_col4:
                    st.metric(
                        "Additional Training Data",
                        f"{sample_change:,}", 
                        help="Additional training samples from previous version"
                    )
    
    # Show performance projection
    st.subheader("Performance Projection")
    
    # Create simple projection chart
    if len(checkpoint_df) >= 2:
        # Create future dates for projection
        last_date = checkpoint_df['timestamp'].max()
        
        # Get average time between checkpoints
        if len(checkpoint_df) > 1:
            date_diffs = checkpoint_df['timestamp'].diff().dropna()
            avg_diff = date_diffs.mean()
        else:
            avg_diff = pd.Timedelta(days=30)  # Default 30 days
        
        # Create projected dates (3 future checkpoints)
        future_dates = [last_date + avg_diff * (i + 1) for i in range(3)]
        
        # Simple linear regression for projection
        x = np.array(range(len(checkpoint_df))).reshape(-1, 1)
        y = checkpoint_df['accuracy'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        # Predict future values
        future_x = np.array(range(len(checkpoint_df), len(checkpoint_df) + 3)).reshape(-1, 1)
        future_y = model.predict(future_x)
        
        # Cap predictions at 1.0
        future_y = np.minimum(future_y, 1.0)
        
        # Create projection dataframe
        projection_df = pd.DataFrame({
            'timestamp': future_dates,
            'accuracy': future_y,
            'type': 'Projected'
        })
        
        # Add historical data
        historical_df = pd.DataFrame({
            'timestamp': checkpoint_df['timestamp'],
            'accuracy': checkpoint_df['accuracy'],
            'type': 'Historical'
        })
        
        # Combine dataframes
        combined_df = pd.concat([historical_df, projection_df])
        
        # Create projection chart
        proj_fig = px.line(
            combined_df,
            x='timestamp',
            y='accuracy',
            color='type',
            title='Model Performance Projection',
            color_discrete_map={'Historical': '#4ecdc4', 'Projected': '#aaaaaa'},
            line_dash='type',
            line_dash_map={'Historical': 'solid', 'Projected': 'dash'}
        )
        
        # Instead of add_vline, add a vertical line by creating a separate scatter trace
        # Note the last historical point and the first projected point
        boundary_time = combined_df[combined_df['type'] == 'Projected']['timestamp'].min()
        
        # Add a vertical line as a separate trace
        line_x = [boundary_time, boundary_time]
        line_y = [0, 1]  # From bottom to top of the chart
        
        proj_fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='none',
            name='Projection Start'
        ))
        
        # Add text annotation
        proj_fig.add_annotation(
            x=boundary_time,
            y=0.1,
            text="Future Projection",
            showarrow=False,
            textangle=-90,
            xanchor='left',
            font=dict(color='gray')
        )
        
        # Update layout
        proj_fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Model Accuracy',
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        
        # Display chart
        st.plotly_chart(proj_fig, use_container_width=True)
        
        # Show when we might reach target performance
        last_projected = future_y[-1]
        target_performance = 0.95  # 95% accuracy target
        
        if last_projected >= target_performance:
            st.success(f"Model is projected to reach {target_performance:.0%} accuracy within the next {len(future_dates)} versions!")
        else:
            # Estimate how many more checkpoints needed
            if model.coef_[0] > 0:  # Check if trend is positive
                checkpoints_needed = int(np.ceil((target_performance - last_projected) / model.coef_[0]))
                # Use pandas method to add timedeltas correctly
                est_date = future_dates[-1] + pd.Timedelta(days=avg_diff.days * checkpoints_needed)
                st.info(f"At current improvement rate, model may reach {target_performance:.0%} accuracy around {est_date.strftime('%B %Y')} (after approximately {checkpoints_needed} more versions).")
            else:
                st.warning("Current trend suggests model performance optimization may need adjustment to reach target accuracy.")
    else:
        st.info("At least 2 model versions required for performance projection analysis.")