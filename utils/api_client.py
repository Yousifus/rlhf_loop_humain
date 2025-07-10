#!/usr/bin/env python3
"""
RLHF Model API Client

This module provides the API interface for connecting with language models
in the RLHF system. It handles different response modes, communication patterns,
and ensures consistent model behavior across interactions.

Supports multiple providers:
- DeepSeek API (cloud)
- LM Studio (local)
- OpenAI API (cloud)
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
    
    # Provider configurations
    PROVIDERS = {
        "deepseek": {
            "name": "DeepSeek API",
            "api_base": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "requires_key": True,
            "env_key": "DEEPSEEK_API_KEY",
            "icon": "🌐"
        },
        "lmstudio": {
            "name": "LM Studio (Local)",
            "api_base": "http://localhost:1234/v1",
            "default_model": "local-model",
            "requires_key": False,
            "env_key": None,
            "icon": "🏠"
        },
        "openai": {
            "name": "OpenAI API", 
            "api_base": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "requires_key": True,
            "env_key": "OPENAI_API_KEY",
            "icon": "🤖"
        },
        "grok": {
            "name": "Grok (X.AI)",
            "api_base": "https://api.x.ai/v1",
            "default_model": "grok-3-beta",
            "requires_key": True,
            "env_key": "XAI_API_KEY",
            "icon": "⚡"
        }
    }
    
    def __init__(self, provider: str = "deepseek", api_key: str = None, api_base: str = None, model: str = None):
        """
        Initialize the model API client.
        
        Args:
            provider: API provider ("deepseek", "lmstudio", "openai")
            api_key: API key for authentication (auto-detected if None)
            api_base: API base URL (uses provider default if None)
            model: Model identifier for API calls (uses provider default if None)
        """
        self.provider = provider
        self.provider_config = self.PROVIDERS.get(provider, self.PROVIDERS["deepseek"])
        
        # Set API configuration based on provider
        if api_base:
            self.api_base = api_base
        else:
            self.api_base = self.provider_config["api_base"]
            
        if api_key:
            self.api_key = api_key
        elif self.provider_config["requires_key"]:
            self.api_key = os.environ.get(self.provider_config["env_key"])
            if not self.api_key:
                logger.warning(f"API key required for {self.provider_config['name']}. Set {self.provider_config['env_key']} environment variable.")
        else:
            self.api_key = "not-needed"  # LM Studio doesn't need API key
            
        if model:
            self.model = model
        else:
            self.model = self.provider_config["default_model"]
            
        # Set up request headers
        if self.provider_config["requires_key"] and self.api_key and self.api_key != "not-needed":
            self.request_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            self.request_headers = {
                "Content-Type": "application/json"
            }
        
        # Track current model mode
        self.current_mode = "analytical"
        
        # Check provider availability
        self.available = self._check_availability()
        
        logger.info(f"Model API client initialized with {self.provider_config['name']} (model: {self.model})")
    
    def update_api_key(self, new_api_key: str) -> bool:
        """
        Update the API key for this client and refresh headers.
        
        Args:
            new_api_key: The new API key to use
            
        Returns:
            True if key was updated successfully
        """
        try:
            self.api_key = new_api_key
            
            # Update request headers with new key
            if self.provider_config["requires_key"] and self.api_key and self.api_key != "not-needed":
                self.request_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            else:
                self.request_headers = {
                    "Content-Type": "application/json"
                }
            
            # Re-check availability with new key
            self.available = self._check_availability()
            
            logger.info(f"API key updated for {self.provider_config['name']}, available: {self.available}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating API key: {str(e)}")
            return False
    
    def _check_availability(self) -> bool:
        """Check if the current provider is available and responsive."""
        try:
            models_endpoint = f"{self.api_base}/models"
            response = requests.get(models_endpoint, headers=self.request_headers, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"✅ {self.provider_config['name']} is available")
                return True
            else:
                logger.warning(f"⚠️ {self.provider_config['name']} responded with status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ {self.provider_config['name']} is not available: {str(e)}")
            return False
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models from the provider."""
        try:
            models_endpoint = f"{self.api_base}/models"
            response = requests.get(models_endpoint, headers=self.request_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                if "data" in data:
                    for model in data["data"]:
                        models.append({
                            "id": model.get("id", "unknown"),
                            "name": model.get("id", "unknown"),
                            "owned_by": model.get("owned_by", "unknown"),
                            "created": model.get("created", 0)
                        })
                else:
                    # Fallback for simple model lists
                    models = [{"id": self.model, "name": self.model, "owned_by": "local", "created": 0}]
                
                logger.info(f"Found {len(models)} models on {self.provider_config['name']}")
                return models
            else:
                logger.warning(f"Failed to fetch models: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            logger.warning(f"Error fetching models: {str(e)}")
            return []
    
    def set_model(self, model_id: str) -> bool:
        """
        Set the active model for this client.
        
        Args:
            model_id: The model identifier to use
            
        Returns:
            True if model was set successfully
        """
        available_models = self.get_available_models()
        model_ids = [m["id"] for m in available_models]
        
        if model_ids and model_id in model_ids:
            self.model = model_id
            logger.info(f"Model set to: {model_id}")
            return True
        elif not model_ids:  # If we can't fetch models, allow any model
            self.model = model_id
            logger.info(f"Model set to: {model_id} (availability not verified)")
            return True
        else:
            logger.warning(f"Model {model_id} not found in available models: {model_ids}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and status."""
        return {
            "provider": self.provider,
            "name": self.provider_config["name"],
            "icon": self.provider_config["icon"],
            "api_base": self.api_base,
            "model": self.model,
            "available": self.available,
            "requires_key": self.provider_config["requires_key"],
            "has_key": bool(self.api_key and self.api_key != "not-needed")
        }
    
    @classmethod
    def detect_available_providers(cls) -> List[Dict[str, Any]]:
        """Detect which providers are currently available."""
        available_providers = []
        
        for provider_id, config in cls.PROVIDERS.items():
            try:
                # Create temporary client to test availability
                temp_client = cls(provider=provider_id)
                provider_info = temp_client.get_provider_info()
                
                if temp_client.available:
                    available_providers.append({
                        "id": provider_id,
                        "name": config["name"],
                        "icon": config["icon"],
                        "available": True,
                        "models_count": len(temp_client.get_available_models())
                    })
                else:
                    available_providers.append({
                        "id": provider_id,
                        "name": config["name"],
                        "icon": config["icon"],
                        "available": False,
                        "models_count": 0
                    })
                    
            except Exception as e:
                logger.debug(f"Error detecting {provider_id}: {str(e)}")
                available_providers.append({
                    "id": provider_id,
                    "name": config["name"],
                    "icon": config["icon"],
                    "available": False,
                    "models_count": 0
                })
        
        return available_providers
    
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
        if not self.available:
            raise ValueError(f"{self.provider_config['name']} is not available")
            
        if self.provider_config["requires_key"] and not self.api_key:
            raise ValueError(f"API key is required for {self.provider_config['name']}")
        
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
                "provider": self.provider,
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
        if not self.available:
            return {
                "completion": f"Error: {self.provider_config['name']} is not available",
                "role": "assistant",
                "error": True,
                "provider": self.provider,
                "timestamp": datetime.now().isoformat()
            }
            
        if self.provider_config["requires_key"] and not self.api_key:
            return {
                "completion": f"Error: API key required for {self.provider_config['name']}",
                "role": "assistant", 
                "error": True,
                "provider": self.provider,
                "timestamp": datetime.now().isoformat()
            }
        
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
                "provider": self.provider,
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
                "provider": self.provider,
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
                
                if not self.available:
                    raise ValueError(f"{self.provider_config['name']} is not available")
                    
                if self.provider_config["requires_key"] and not self.api_key:
                    raise ValueError(f"API key required for {self.provider_config['name']}")
                
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
                    "provider": self.provider,
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
                    "provider": self.provider,
                    "timestamp": datetime.now().isoformat(),
                    "model_mode": self.current_mode,
                    "approach": approach["system_addition"].strip(),
                    "completion_index": i
                })
        
        return completions

# Create provider-specific singleton instances
_api_clients = {}

def get_api_client(provider: str = "deepseek") -> ModelAPIClient:
    """Get or create the API client for a specific provider"""
    global _api_clients
    if provider not in _api_clients:
        _api_clients[provider] = ModelAPIClient(provider=provider)
    return _api_clients[provider]
