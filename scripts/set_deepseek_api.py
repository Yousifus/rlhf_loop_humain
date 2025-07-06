#!/usr/bin/env python3
"""
Set DeepSeek API Key for RLHF System

This script configures the connection to the DeepSeek API 
for completion generation in the RLHF training loop.
"""

import os
import json
import time
import requests
from datetime import datetime

def configure_deepseek_api():
    """Configure DeepSeek API key and test connection"""
    
    # Your DeepSeek API key
    api_key = "your-api-key-here"  # Replace with your actual API key
    
    print("⚙️ Setting up RLHF system connection...")
    
    # Set environment variable
    os.environ["DEEPSEEK_API_KEY"] = api_key
    
    # Test the connection
    print("🔧 Testing API connection...")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an AI assistant, respond briefly and professionally."},
                {"role": "user", "content": "Hello Assistant, can you confirm the connection is working?"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            test_response = response.json()
            print(f"💬 Assistant says: {test_response['choices'][0]['message']['content']}")
            print("✅ API connection successful!")
            return True
        else:
            print(f"❌ API test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuring RLHF System Connection to DeepSeek API")
    success = configure_deepseek_api()
    
    if success:
        print("🎉 Configuration complete! RLHF system ready.")
    else:
        print("⚠️ Configuration failed. Please check your API key.")
