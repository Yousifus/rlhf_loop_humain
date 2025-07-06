import os
import time
import json
import requests
from typing import List, Optional, Dict
from datetime import datetime

# Read API key from environment variable for security
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_ID = "deepseek-chat"

COMPLETION_LOG_PATH = "data/raw_completions_log.jsonl"

# Optional: define pricing per 1M tokens
RATE_TABLE = {
    "deepseek-chat": {
        "input": 0.27 / 1_000_000,
        "output": 1.10 / 1_000_000
    }
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost for API usage"""
    input_rate = RATE_TABLE.get(model, {}).get("input", 0)
    output_rate = RATE_TABLE.get(model, {}).get("output", 0)
    return round((prompt_tokens * input_rate + completion_tokens * output_rate), 6)

def generate_completions(
    prompt: str,
    n_completions: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    calculate_cost: bool = True,
    fallback_on_error: bool = True,
    max_retries: int = 2
) -> Dict:
    """
    Generate multiple completions for a given prompt using the DeepSeek API.
    
    Args:
        prompt: The input prompt for completion generation
        n_completions: Number of completions to generate
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens per completion
        calculate_cost: Whether to calculate estimated API cost
        fallback_on_error: Whether to use fallback on API errors
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing completions and metadata
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable must be set")
    
    completions = []
    metadata_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    fallback_triggered = False

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "X-Model-Version": "1.0"
    }

    for i in range(n_completions):
        body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": "You are an AI assistant that provides helpful, accurate, and professional responses."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        attempt = 0
        success = False
        while attempt <= max_retries:
            try:
                response = requests.post(DEEPSEEK_API_URL, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]["message"]["content"]
                usage = payload.get("usage", {})

                completions.append(choice)
                metadata_list.append({
                    "model": MODEL_ID,
                    "temperature": temperature,
                    "top_p": top_p,
                    "created_unix": payload.get("created"),
                    "finish_reason": payload["choices"][0].get("finish_reason", "stop"),
                    "usage": usage
                })

                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                success = True
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"[!] Completion {i+1} failed after retries: {e}")
                    if fallback_on_error:
                        completions.append(f"[Fallback completion {i+1}]")
                        metadata_list.append({
                            "model": MODEL_ID,
                            "temperature": temperature,
                            "top_p": top_p,
                            "error_message": str(e),
                            "fallback": True
                        })
                        fallback_triggered = True
                    break
                time.sleep(0.5)

    result = {
        "prompt": prompt,
        "completions": completions,
        "completion_details": metadata_list,
        "aggregated_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        },
        "model_used": MODEL_ID,
        "timestamp": datetime.utcnow().isoformat(),
        "fallback_triggered": fallback_triggered
    }

    if calculate_cost:
        result["estimated_cost"] = estimate_cost(
            MODEL_ID, total_prompt_tokens, total_completion_tokens
        )

    # Log the completion results
    os.makedirs(os.path.dirname(COMPLETION_LOG_PATH), exist_ok=True)
    with open(COMPLETION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result

if __name__ == "__main__":
    print("--- RLHF Loop Completion Generator ---")
    prompt = "Explain the concept of reinforcement learning from human feedback."
    output = generate_completions(prompt, temperature=0.7)
    print(json.dumps(output, indent=2))
