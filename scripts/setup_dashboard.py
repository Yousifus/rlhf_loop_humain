#!/usr/bin/env python3
"""
Setup script for the RLHF Attunement Dashboard.

This script installs all the required packages for the dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Install all required packages for the dashboard"""
    print("Setting up RLHF Attunement Dashboard...")
    
    # Install packages from requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install optional packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn"])
        print("Successfully installed UMAP (for advanced dimensionality reduction)")
    except Exception as e:
        print(f"Warning: Could not install UMAP. Dashboard will fall back to PCA: {e}")
    
    # Create necessary directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\nSetup completed successfully!")
    print("\nTo run the dashboard:")
    print("  streamlit run scripts/run_dashboard.py")

if __name__ == "__main__":
    main() 