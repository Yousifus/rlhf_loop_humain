#!/usr/bin/env python3
"""
Enhanced RLHF Dashboard Launcher

Professional launcher for the enhanced RLHF monitoring interface with 
improved UX and advanced analytics capabilities.
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

def ensure_directories_exist():
    """Ensure all required data directories exist."""
    # Get project root
    project_root = Path(__file__).resolve().parent
    
    # Define required directories
    data_dir = project_root / "data"
    vote_logs_dir = data_dir / "vote_logs"
    reflections_dir = data_dir / "reflections"
    backups_dir = data_dir / "backups"
    chat_logs_dir = data_dir / "chat_logs"
    feedback_dir = data_dir / "feedback"
    
    # Create directories if they don't exist
    for directory in [data_dir, vote_logs_dir, reflections_dir, backups_dir, chat_logs_dir, feedback_dir]:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Ensured directory exists: {directory}")
    
    return project_root

def initialize_database():
    """Initialize the database for the RLHF system."""
    # Add project root to path to allow imports
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Import database module
    try:
        from utils.database import get_database
        
        # Initialize database
        db = get_database()
        
        # Get data summary to verify database is working
        summary = db.get_data_summary()
        
        logger.info(f"Database initialized successfully. Found {summary.get('total_annotations', 0)} annotations.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'altair',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run the Enhanced Streamlit dashboard."""
    print("🌟 Initializing Enhanced RLHF Dashboard...")
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Initialize database
    success = initialize_database()
    if not success:
        logger.warning("⚠️ Database initialization failed. Some features may not work correctly.")
    
    # Get the enhanced dashboard script path
    script_dir = Path(__file__).resolve().parent
    dashboard_path = script_dir / "interface" / "simple_enhanced_dashboard.py"
    
    if not dashboard_path.exists():
        logger.error(f"❌ Enhanced dashboard script not found at {dashboard_path}")
        logger.info("💡 Falling back to standard dashboard...")
        dashboard_path = script_dir / "interface" / "dashboard_core.py"
        
        if not dashboard_path.exists():
            logger.error(f"❌ No dashboard script found!")
            sys.exit(1)
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(script_dir)
    
    # Disable telemetry to avoid CORS errors
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Check if API key is provided
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    model_id = os.environ.get("MODEL_ID", "deepseek-chat")
    
    if api_key:
        logger.info(f"✅ Using DeepSeek API with model: {model_id}")
    else:
        logger.warning("⚠️ No DeepSeek API key found. API features will use fallback mode.")
    
    # Print professional startup message
    print("\n" + "="*60)
    print("🤖 RLHF LOOP ENHANCED DASHBOARD")
    print("="*60)
    print("🌟 Starting professional RLHF monitoring system...")
    print("📊 Advanced analytics and performance tracking")
    print("🎨 Beautiful, intuitive, and highly functional")
    print("="*60 + "\n")
    
    # Create a new environment dictionary for the subprocess with telemetry disabled
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Enhanced Streamlit configuration
    cmd = [
        "streamlit", "run", 
        str(dashboard_path),
        "--server.port", "8504",  # Different port for enhanced version
        "--server.address", "0.0.0.0",
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#ff6b9d",
        "--theme.backgroundColor", "#0E1117",
        "--theme.secondaryBackgroundColor", "#262730",
        "--theme.textColor", "#FAFAFA",
        "--server.maxUploadSize", "200",
        "--server.maxMessageSize", "200"
    ]
    
    logger.info(f"🚀 Launching Enhanced RLHF Dashboard on http://localhost:8504")
    logger.info("💡 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard connection closed. Session ended.")
        logger.info("Dashboard stopped by user.")
    except Exception as e:
        logger.error(f"❌ Error launching enhanced dashboard: {e}")
        print(f"\n⚠️ Error occurred: {e}")
        print("\n💡 To run manually, use:")
        print(f"  streamlit run {dashboard_path} --server.port 8504 --browser.gatherUsageStats=false")
        sys.exit(1)

if __name__ == "__main__":
    main()
