#!/usr/bin/env python3
"""
RLHF Loop Dashboard Launcher

This launches the main dashboard for monitoring and analyzing 
the reinforcement learning from human feedback system.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_structure():
    """Ensure basic data structure exists"""
    data_dir = Path("data")
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Create required subdirectories
    subdirs = ["completions", "prompts"]
    for subdir in subdirs:
        (data_dir / subdir).mkdir(exist_ok=True)
    
    # Create empty data files if they don't exist
    data_files = [
        "predictions.jsonl",
        "votes.jsonl"
    ]
    
    for file_name in data_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            file_path.touch()
            logger.info(f"Created empty data file: {file_path}")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        # Change to project directory
        project_root = Path(__file__).resolve().parent
        os.chdir(project_root)
        
        logger.info("Starting RLHF Loop Dashboard...")
        logger.info("Dashboard will be available at: http://localhost:8501")
        
        # Launch streamlit
        dashboard_file = "interface/attunement_dashboard.py"
        if not Path(dashboard_file).exists():
            # Fallback to basic dashboard
            dashboard_file = "interface/rlhf_loop.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        logger.info(f"Launching: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch dashboard: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    logger.info("🚀 RLHF Loop Dashboard Launcher")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed")
        sys.exit(1)
    
    logger.info("✅ Dependencies check passed")
    
    # Check data structure
    check_data_structure()
    logger.info("✅ Data structure verified")
    
    # Launch dashboard
    logger.info("🎯 Launching RLHF dashboard...")
    success = launch_dashboard()
    
    if success:
        logger.info("✅ Dashboard session completed")
    else:
        logger.error("❌ Dashboard failed to launch")
        sys.exit(1)

if __name__ == "__main__":
    main()
