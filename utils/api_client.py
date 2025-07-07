#!/usr/bin/env python3
"""
RLHF Model API Client

This module provides the API interface for connecting with language models
in the RLHF system. It handles different response modes, communication patterns,
and ensures consistent model behavior across interactions.
"""

import os
import json
import logging
import random
import time
import uuid
from typing import List, Dict, Any, Optional, Union
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelAPIClient:
    """RLHF Model API Client - Interface for language model interactions"""
    
    # Model personality modes and their characteristics
    MODEL_MODES = {
        "analytical": {
            "system_prompt": "You are an AI assistant that provides thorough, analytical responses. Focus on breaking down complex topics, providing detailed explanations, and offering structured insights. Maintain a professional, informative tone.",
            "temperature": 0.7,
            "description": "Analytical and thorough responses"
        },
        "conversational": {
            "system_prompt": "You are an AI assistant that provides natural, conversational responses. Be engaging, friendly, and approachable while maintaining accuracy and helpfulness. Use clear, accessible language.",
            "temperature": 0.8,
            "description": "Natural and conversational responses"
        },
        "precise": {
            "system_prompt": "You are an AI assistant that provides precise, concise responses. Focus on accuracy, clarity, and directness. Avoid unnecessary elaboration while ensuring all important information is covered.",
            "temperature": 0.6,
            "description": "Precise and concise responses"
        }
    }
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """
        Initialize the model API client.
        
        Args:
            api_key: API key for authentication (reads from DEEPSEEK_API_KEY env var if None)
            api_base: API base URL (uses DeepSeek by default)
            model: Model identifier for API calls
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            logger.warning("API key required for model communication. Set DEEPSEEK_API_KEY environment variable.")
            
        self.api_base = api_base or "https://api.deepseek.com/v1"
        self.model = model or "deepseek-chat"
        self.request_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Track current model mode
        self.current_mode = "analytical"
        
        logger.info(f"Model API client initialized with model: {self.model}")
    
    def set_model_mode(self, mode: str) -> bool:
        """
        Set the model's response mode.
        
        Args:
            mode: The response mode to use ('analytical', 'conversational', 'precise')
            
        Returns:
            True if mode was set successfully
        """
        if mode in self.MODEL_MODES:
            self.current_mode = mode
            logger.info(f"Model mode set to: {mode}")
            return True
        else:
            logger.warning(f"Unknown model mode: {mode}")
            return False
    
    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about the current model mode."""
        return {
            "current": self.current_mode,
            "description": self.MODEL_MODES[self.current_mode]["description"],
            "available_modes": list(self.MODEL_MODES.keys())
        }
    
    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Prepare messages with appropriate system context.
        
        Args:
            messages: Original message list
            
        Returns:
            Messages with system prompt prepended if needed
        """
        # Get current mode configuration
        mode_config = self.MODEL_MODES[self.current_mode]
        
        # Check if system message already exists
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if not has_system:
            # Prepend system message
            prepared_messages = [
                {
                    "role": "system",
                    "content": mode_config["system_prompt"]
                }
            ] + messages
        else:
            # Use existing messages as-is
            prepared_messages = messages
            
        return prepared_messages
    
    def generate_completion(self, prompt: str, max_tokens: int = 500, 
                           temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a single completion from the API.
        
        Args:
            prompt: The prompt to generate a completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0-1)
            
        Returns:
            Response from the API including the completion text
        """
        if not self.api_key:
            raise ValueError("API key is required for generating completions")
        
        endpoint = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.request_headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract completion text
            completion = result["choices"][0]["message"]["content"].strip()
            
            # Create structured response
            structured_response = {
                "completion": completion,
                "model": result["model"],
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
                "finish_reason": result["choices"][0]["finish_reason"],
                "raw_response": result
            }
            
            return structured_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def generate_chat_response(self, messages: List[Dict[str, str]], 
                              max_tokens: int = 500, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a chat response from the model.
        
        Args:
            messages: Conversation history with "role" and "content" keys
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            Model response with metadata
        """
        if not self.api_key:
            raise ValueError("API key required for chat response generation")
        
        endpoint = f"{self.api_base}/chat/completions"
        
        # Prepare messages with system context
        prepared_messages = self._prepare_messages(messages)
        
        # Use mode-specific temperature if not overridden
        mode_config = self.MODEL_MODES[self.current_mode]
        if temperature == 0.7:  # Default value, use mode setting
            temperature = mode_config["temperature"]
        
        payload = {
            "model": self.model,
            "messages": prepared_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.request_headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract completion text
            completion = result["choices"][0]["message"]["content"].strip()
            
            # Create structured response
            structured_response = {
                "completion": completion,
                "role": "assistant",
                "model": result["model"],
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
                "finish_reason": result["choices"][0]["finish_reason"],
                "raw_response": result
            }
            
            return structured_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "completion": f"Error: {str(e)}",
                "role": "assistant",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_comparison(self, prompt: str, num_completions: int = 2, 
                           max_tokens: int = 500, temperature: float = 0.7) -> List[Dict[str, Any]]:
        """
        Generate multiple completions for comparison and preference ranking.
        
        Args:
            prompt: The prompt to generate completions for
            num_completions: Number of different completions to generate
            max_tokens: Maximum tokens per completion
            temperature: Temperature for sampling (0-1)
            
        Returns:
            List of completions with metadata for comparison
        """
        completions = []
        
        # Define distinct approaches for diversity
        approaches = [
            {
                "temperature": 0.5,
                "system_addition": " Provide a structured, step-by-step analysis.",
                "top_p": 0.9
            },
            {
                "temperature": 0.9,
                "system_addition": " Use creative examples and analogies to explain concepts.",
                "top_p": 0.95
            },
            {
                "temperature": 0.7,
                "system_addition": " Focus on practical applications and real-world implications.",
                "top_p": 0.92
            }
        ]
        
        for i in range(num_completions):
            try:
                # Use different approaches for each completion
                approach = approaches[i % len(approaches)]
                
                # Prepare messages with varied system context
                messages = [{"role": "user", "content": prompt}]
                mode_config = self.MODEL_MODES[self.current_mode]
                
                # Modify system prompt for diversity
                varied_system_prompt = mode_config["system_prompt"] + approach["system_addition"]
                
                prepared_messages = [
                    {"role": "system", "content": varied_system_prompt}
                ] + messages
                
                if not self.api_key:
                    raise ValueError("API key required for completion generation")
                
                endpoint = f"{self.api_base}/chat/completions"
                
                # Use the approach-specific parameters for diversity
                payload = {
                    "model": self.model,
                    "messages": prepared_messages,
                    "max_tokens": max_tokens,
                    "temperature": approach["temperature"],
                    "top_p": approach["top_p"],
                    "frequency_penalty": 0.1 if i % 2 == 1 else 0.0,  # Add variety
                    "presence_penalty": 0.1 if i % 2 == 0 else 0.0,   # Add variety
                }
                
                response = requests.post(
                    endpoint,
                    headers=self.request_headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract completion text
                completion_text = result["choices"][0]["message"]["content"].strip()
                
                # Create structured response
                structured_response = {
                    "completion": completion_text,
                    "model": result["model"],
                    "timestamp": datetime.now().isoformat(),
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "total_tokens": result["usage"]["total_tokens"],
                    "finish_reason": result["choices"][0]["finish_reason"],
                    "model_mode": self.current_mode,
                    "temperature_used": approach["temperature"],
                    "top_p_used": approach["top_p"],
                    "approach": approach["system_addition"].strip(),
                    "completion_index": i,
                    "raw_response": result
                }
                
                completions.append(structured_response)
                
                # Add a small delay to avoid rate limiting
                if i < num_completions - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error generating completion {i+1}: {str(e)}")
                approach = approaches[i % len(approaches)]  # Get the approach that was being used
                completions.append({
                    "completion": f"Error generating completion: {str(e)}",
                    "error": True,
                    "timestamp": datetime.now().isoformat(),
                    "model_mode": self.current_mode,
                    "approach": approach["system_addition"].strip(),
                    "completion_index": i
                })
        
        return completions

# Create a singleton instance
_api_client = None

def get_api_client() -> ModelAPIClient:
    """Get or create the API client singleton"""
    global _api_client
    if _api_client is None:
        _api_client = ModelAPIClient()
    return _api_client
