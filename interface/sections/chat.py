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

from utils.api_client import get_api_client

# Configure logging
logger = logging.getLogger(__name__)

def display_chat_interface():
    """Display the interactive chat interface"""
    st.header("💬 Interactive Chat Interface")
    
    # Get selected provider from session state
    selected_provider = st.session_state.get('selected_provider', 'deepseek')
    
    # Get API client for selected provider
    try:
        api_client = get_api_client(selected_provider)
        provider_info = api_client.get_provider_info()
    except Exception as e:
        st.error(f"Error accessing {selected_provider} provider: {str(e)}")
        return
    
    # Show current provider status
    if provider_info["available"]:
        st.success(f"✅ Connected to {provider_info['icon']} {provider_info['name']} (Model: {provider_info['model']})")
    else:
        st.error(f"❌ {provider_info['name']} is not available. Please check configuration in sidebar.")
        return
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Model settings configuration
    with st.sidebar:
        st.subheader("🎯 Chat Settings")
        
        # Model mode selection
        mode_info = api_client.get_mode_info()
        current_mode = mode_info["current"]
        available_modes = mode_info["available_modes"]
        
        selected_mode = st.selectbox(
            "Response Style",
            options=available_modes,
            index=available_modes.index(current_mode),
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Choose how the model should respond"
        )
        
        # Update mode if changed
        if selected_mode != current_mode:
            api_client.set_model_mode(selected_mode)
        
        temperature = st.slider(
            "Creativity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0 = Consistent and predictable, 1 = Creative and varied"
        )
        max_tokens = st.slider(
            "Response Length",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum length of responses"
        )
        
        # Apply settings button
        if st.button("Apply Settings", use_container_width=True):
            st.session_state.chat_settings = {
                "mode": selected_mode,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            st.success("Chat settings updated!")
    
    # Provide context and guidance
    with st.expander("ℹ️ Chat Interface Guide", expanded=not st.session_state.chat_history):
        st.markdown(f"""
        **Currently using:** {provider_info['icon']} {provider_info['name']} with model `{provider_info['model']}`
        
        **In this interface you can:**
        - Have natural conversations with the AI model
        - Test different response styles and creativity levels
        - Provide feedback to improve model performance
        - Export chat history for analysis
        
        **Tips:**
        - Use the sidebar to adjust response style and creativity
        - Switch providers in the main sidebar if needed
        - Give feedback with 👍/👎 buttons to help improve responses
        """)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show message metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Response Details"):
                    metadata = message["metadata"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Provider:** {metadata.get('provider', 'Unknown')}")
                        st.write(f"**Model:** {metadata.get('model', 'Unknown')}")
                        st.write(f"**Tokens:** {metadata.get('total_tokens', 'Unknown')}")
                    with col2:
                        st.write(f"**Mode:** {metadata.get('mode', 'Unknown')}")
                        st.write(f"**Temperature:** {metadata.get('temperature', 'Unknown')}")
                        st.write(f"**Time:** {metadata.get('timestamp', 'Unknown')}")
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant" and not message.get("feedback"):
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍 Good", key=f"good_{i}"):
                        st.session_state.chat_history[i]["feedback"] = "positive"
                        save_message_feedback(message["id"], "positive")
                        st.rerun()
                with col2:
                    if st.button("👎 Improve", key=f"bad_{i}"):
                        st.session_state.chat_history[i]["feedback"] = "negative"
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
        with st.status("Generating response...", expanded=True) as status:
            try:
                # Get chat settings
                settings = st.session_state.get("chat_settings", {})
                temperature = settings.get("temperature", 0.7)
                max_tokens = settings.get("max_tokens", 300)
                mode = settings.get("mode", "analytical")
                
                st.write(f"Using {provider_info['name']} with {mode} mode")
                
                # Prepare conversation history for context
                conversation_messages = []
                for msg in st.session_state.chat_history[-10:]:  # Last 10 messages for context
                    if msg["role"] in ["user", "assistant"]:
                        conversation_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Generate response using the selected provider
                response_data = api_client.generate_chat_response(
                    messages=conversation_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if "error" in response_data and response_data["error"]:
                    raise Exception(response_data["completion"])
                
                response_content = response_data["completion"]
                status.update(label="Response generated successfully!", state="complete")
                
                # Add assistant message to chat history
                message_id = str(uuid.uuid4())
                assistant_message = {
                    "role": "assistant", 
                    "content": response_content, 
                    "timestamp": datetime.now().isoformat(),
                    "id": message_id,
                    "metadata": {
                        "provider": response_data.get("provider", selected_provider),
                        "model": response_data.get("model", "unknown"),
                        "mode": mode,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "total_tokens": response_data.get("total_tokens", 0),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                
                st.session_state.chat_history.append(assistant_message)
                
                # Save chat interaction for analysis
                save_chat_interaction(
                    prompt, 
                    response_content, 
                    message_id, 
                    f"{selected_provider}:{response_data.get('model', 'unknown')}"
                )
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                status.update(label=f"Error: {str(e)}", state="error")
                
                # Add error message to chat history
                error_message = {
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
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
        if st.button("🔄 Clear Chat", use_container_width=True) and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.session_state.chat_history:
            # Prepare chat export
            chat_export = f"Chat Conversation - {provider_info['name']}\n"
            chat_export += f"Model: {provider_info['model']}\n"
            chat_export += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            chat_export += "=" * 50 + "\n\n"
            
            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_export += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                "📥 Export Chat",
                chat_export,
                file_name=f"chat_{selected_provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
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
