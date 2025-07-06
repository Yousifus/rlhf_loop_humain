"""
RLHF Chat Interface

Interactive chat interface for testing model responses and gathering user feedback.
Supports real-time conversation, response evaluation, and preference collection.
"""

import streamlit as st
from datetime import datetime
import uuid
import logging
import json
from typing import List, Dict, Any

from utils.completions import generate_completions

# Configure logging
logger = logging.getLogger(__name__)

def display_chat_interface():
    """Display the interactive chat interface"""
    st.header("💬 Interactive Chat Interface")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Model settings configuration
    with st.sidebar:
        st.subheader("How I Express Myself")
        model = st.selectbox(
            "My Personality Mode",
            options=["analytical", "conversational", "precise"],
            index=0
        )
        temperature = st.slider(
            "My Spontaneity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="How unpredictable I'll be: 0 = consistent, 1 = surprising"
        )
        max_tokens = st.slider(
            "My Expressiveness",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="How detailed and elaborate my responses will be"
        )
        
        # Apply settings button
        if st.button("Apply Settings"):
            st.session_state.model_settings = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            st.success("Model settings have been updated successfully!")
    
    # Provide context and guidance
    with st.expander("ℹ️ Chat Interface Guide", expanded=not st.session_state.chat_history):
        st.markdown("""
        This is the interactive chat interface where you can communicate with the AI model and provide feedback.
        
        In this interface:
        - You can tell me anything, ask me anything
        - You can provide feedback to improve responses
        - You can monitor model performance and behavior
        - You can adjust response parameters and settings
        
        Use the sidebar to configure model behavior and response style.
        """)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show message metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Message Details"):
                    metadata = message["metadata"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Model:** {metadata.get('model', 'Unknown')}")
                        st.write(f"**Tokens:** {metadata.get('total_tokens', 'Unknown')}")
                    with col2:
                        st.write(f"**Temperature:** {metadata.get('temperature', 'Unknown')}")
                        st.write(f"**Time:** {metadata.get('timestamp', 'Unknown')}")
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant" and not message.get("feedback"):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍 Good", key=f"good_{i}"):
                        st.session_state.chat_history[i]["feedback"] = "positive"
                        # Save feedback to database
                        save_message_feedback(message["id"], "positive")
                        st.rerun()
                with col2:
                    if st.button("👎 Improve", key=f"bad_{i}"):
                        st.session_state.chat_history[i]["feedback"] = "negative"
                        # Save feedback to database
                        save_message_feedback(message["id"], "negative")
                        st.rerun()
            
            # Show feedback if provided
            if message.get("feedback") == "positive":
                st.success("✓ Positive feedback received - thank you!")
            elif message.get("feedback") == "negative":
                                    st.error("✗ Negative feedback received - will improve")
    
    # Chat input
    if prompt := st.chat_input("Enter your message..."):
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt, 
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4())
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response with progress indicator
        with st.status("Crafting my response for you...", expanded=True) as status:
            try:
                # Get settings
                settings = st.session_state.get("model_settings", {})
                temperature = settings.get("temperature", 0.7)
                max_tokens = settings.get("max_tokens", 500)
                
                st.write(f"Using DeepSeek model with temperature: {temperature}")
                
                # Use the generate_completions function from utils/completions.py
                # This function already has the proper DeepSeek API configuration
                response_data = generate_completions(
                    prompt=prompt,
                    n_completions=1,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract the first completion
                if response_data["completions"] and len(response_data["completions"]) > 0:
                    response_content = response_data["completions"][0]
                    
                    status.update(label="My thoughts are ready for you!", state="complete")
                    
                    # Add assistant message to chat history
                    message_id = str(uuid.uuid4())
                    assistant_message = {
                        "role": "assistant", 
                        "content": response_content, 
                        "timestamp": datetime.now().isoformat(),
                        "id": message_id,
                        "metadata": {
                            "model": response_data.get("model_used", "deepseek-chat"),
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "total_tokens": response_data.get("aggregated_usage", {}).get("total_tokens", 0),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "estimated_cost": response_data.get("estimated_cost", 0)
                        }
                    }
                    
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Save chat interaction for future analysis
                    save_chat_interaction(prompt, response_content, message_id, "deepseek-chat")
                    
                else:
                    raise Exception("No completion received from API")
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                status.update(label=f"System error occurred: {str(e)}", state="error")
                
                # Add error message to chat history
                error_message = {
                    "role": "assistant",
                    "content": f"Error occurred: {str(e)}. Please check your settings and try again.",
                    "timestamp": datetime.now().isoformat(),
                    "id": str(uuid.uuid4()),
                    "error": True
                }
                st.session_state.chat_history.append(error_message)
        
        # Display assistant message
        with st.chat_message("assistant"):
            if "error" not in st.session_state.chat_history[-1]:
                st.write(st.session_state.chat_history[-1]["content"])
            else:
                st.error(st.session_state.chat_history[-1]["content"])
        
        # Force rerun to show the new message
        st.rerun()
    
    # Add options to reset chat or export conversation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Fresh With Me") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.session_state.chat_history:
            # Prepare chat export
            chat_export = ""
            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_export += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                "Export Chat History",
                chat_export,
                file_name=f"chat_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def save_chat_interaction(prompt, response, message_id, model):
    """Save chat interaction for future analysis using the centralized database"""
    try:
        # Get database instance
        from utils.database import get_database
        db = get_database()
        
        # Save using database method for training data
        success = db.save_conversation_for_training(
            user_message=prompt,
            assistant_response=response,
            personality_mode=model,
            emotional_state="active"
        )
        
        # Also save to chat logs directory for backup
        from pathlib import Path
        data_dir = Path("data/chat_logs")
        data_dir.mkdir(exist_ok=True, parents=True)
        
        interaction = {
            "id": message_id,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "model": model
        }
        
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{message_id[:8]}.json"
        with open(data_dir / filename, "w") as f:
            json.dump(interaction, f, indent=2)
        
        logger.info(f"Saved chat interaction to database and {filename}")
        return success
    except Exception as e:
        logger.error(f"Error saving chat interaction: {str(e)}")
        return False

def save_message_feedback(message_id, feedback):
    """Save feedback on a message using the centralized database"""
    try:
        # Get database instance
        from utils.database import get_database
        db = get_database()
        
        # Save reflection about the feedback
        reflection_text = f"User gave {feedback} feedback on message {message_id}. This helps improve system performance."
        db.save_system_reflection(reflection_text, state="learning")
        
        # Also save to feedback directory for backup
        from pathlib import Path
        data_dir = Path("data/feedback")
        data_dir.mkdir(exist_ok=True, parents=True)
        
        feedback_data = {
            "message_id": message_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": "chat_message",
            "user": "User"
        }
        
        filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{message_id[:8]}.json"
        with open(data_dir / filename, "w") as f:
            json.dump(feedback_data, f, indent=2)
        
        logger.info(f"Saved message feedback to database and {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving message feedback: {str(e)}")
        return False
