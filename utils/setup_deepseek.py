#!/usr/bin/env python3
"""
Setup script for DeepSeek API integration.

This script helps users set up their DeepSeek API key as an environment variable.
"""

import os
import sys
import argparse
from pathlib import Path

def setup_api_key(api_key=None, force=False):
    """Set up the DeepSeek API key.
    
    Args:
        api_key (str, optional): The API key to use. If not provided, will prompt the user.
        force (bool, optional): Whether to override existing key if present.
        
    Returns:
        bool: True if setup was successful, False otherwise.
    """
    # Determine the appropriate location for saving the key
    home_dir = Path.home()
    env_file = home_dir / ".deepseek_env"
    
    # Check if key already exists and we don't want to force override
    existing_key = os.environ.get("DEEPSEEK_API_KEY")
    if existing_key and not force:
        print(f"\nDeepSeek API key already exists in the environment.")
        override = input("Would you like to replace it? (y/n): ").strip().lower()
        if override != 'y':
            print("Keeping existing API key.")
            return True
    
    # Check if env file exists and we don't want to force override
    if env_file.exists() and not force:
        print(f"\nDeepSeek environment file already exists at {env_file}")
        override = input("Would you like to replace it? (y/n): ").strip().lower()
        if override != 'y':
            print("Keeping existing environment file.")
            return True
    
    # If API key is not provided, prompt for it
    attempts = 0
    max_attempts = 3
    
    while not api_key and attempts < max_attempts:
        attempts += 1
        print("\n=== DeepSeek API Setup ===")
        print("You need a DeepSeek API key to use the API integration features.")
        print("You can get one by signing up at https://platform.deepseek.com\n")
        
        api_key = input("Enter your DeepSeek API key: ").strip()
        
        if not api_key:
            if attempts < max_attempts:
                print(f"API key cannot be empty. Please try again ({attempts}/{max_attempts}).")
            else:
                print(f"Maximum attempts reached. Setup failed.")
                return False
    
    # Save the API key to a file that can be sourced
    try:
        with open(env_file, "w") as f:
            f.write(f"# DeepSeek API key environment variable\n")
            f.write(f"DEEPSEEK_API_KEY={api_key}\n")
        
        env_file.chmod(0o600)  # Set permissions to be readable only by the user
        
        print(f"\nAPI key saved to {env_file}")
        
        # Set the key in current environment
        os.environ["DEEPSEEK_API_KEY"] = api_key
        print("API key set in current environment")
        
        # Print instructions for activating the key (masked for security)
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else "***"
        
        if sys.platform.startswith("win"):
            print("\nTo activate the API key in your current PowerShell session, run:")
            print(f"$env:DEEPSEEK_API_KEY = \"<your_api_key_here>\"")
            print(f"# Your key: {masked_key}")
            print("\nTo make it permanent, add to your PowerShell profile:")
            print(f"[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', '<your_api_key_here>', 'User')")
        else:
            print("\nTo activate the API key in your current shell session, run:")
            print(f"export DEEPSEEK_API_KEY=\"<your_api_key_here>\"")
            print(f"# Your key: {masked_key}")
            print("\nTo make it permanent, add to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
            print(f"export DEEPSEEK_API_KEY=\"<your_api_key_here>\"")
            
        return True
    
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Set up DeepSeek API key")
    parser.add_argument("--key", help="DeepSeek API key (if not provided, will prompt for it)")
    parser.add_argument("--force", action="store_true", help="Force override existing key")
    args = parser.parse_args()
    
    result = setup_api_key(args.key, args.force)
    
    if result:
        print("\nSetup completed successfully!")
    else:
        print("\nSetup failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 