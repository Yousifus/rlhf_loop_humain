#!/usr/bin/env python3
"""
Script to launch the RLHF Attunement Dashboard with DeepSeek API integration.

This script sets the necessary environment variables for DeepSeek API access
and launches the dashboard.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DeepSeek API credentials
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-api-key-here") 
MODEL_ID = "deepseek-chat"

def main():
    """Run the Streamlit dashboard with DeepSeek API integration."""
    # Set environment variables
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["MODEL_ID"] = MODEL_ID
    
    logger.info(f"Set DEEPSEEK_API_KEY and MODEL_ID environment variables")
    
    # Get the dashboard script path
    script_dir = Path(__file__).resolve().parent
    dashboard_path = script_dir / "run_dashboard.py"
    
    if not dashboard_path.exists():
        logger.error(f"Dashboard script not found at {dashboard_path}")
        sys.exit(1)
    
    # Run the dashboard script as a subprocess
    logger.info(f"Starting RLHF Attunement Dashboard with DeepSeek API integration...")
    
    # Set up environment for the subprocess
    env = os.environ.copy()
    
    try:
        # Run the dashboard script
        subprocess.run([sys.executable, str(dashboard_path)], env=env)
    except KeyboardInterrupt:
        logger.info("\nDashboard stopped.")
    except Exception as e:
        logger.error(f"Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 