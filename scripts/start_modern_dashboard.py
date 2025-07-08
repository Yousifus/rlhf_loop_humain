#!/usr/bin/env python3
"""
🚀 RLHF Modern Dashboard Launcher

Starts both the FastAPI backend and provides instructions for React frontend.
Gives users choice between modern React UI and classic Streamlit interface.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
from threading import Thread
import platform

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🚀 RLHF MODERN DASHBOARD LAUNCHER")
    print("="*70)
    print("High-performance React frontend + FastAPI backend")
    print("Built for speed, designed for professionals")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies using pip list instead of import
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "list"
        ], capture_output=True, text=True, check=True)
        
        pip_output = result.stdout.lower()
        
        # Check for required packages
        required_packages = ["fastapi", "uvicorn", "pandas"]
        missing_packages = []
        
        for package in required_packages:
            if package not in pip_output:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
            print("💡 Install with: pip install -r api_backend/requirements.txt")
            return False
        else:
            print("✅ FastAPI backend dependencies: OK")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  Could not check Python dependencies: {e}")
        print("🔄 Proceeding anyway - will try to start backend...")
    
    # Check Node.js (optional for backend-only mode)
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            node_version = result.stdout.strip()
            print(f"✅ Node.js: {node_version}")
        else:
            print("⚠️  Node.js not found - React frontend won't be available")
            print("💡 Install Node.js 18+ from: https://nodejs.org/")
    except FileNotFoundError:
        print("⚠️  Node.js not found - React frontend won't be available")
        print("💡 Install Node.js 18+ from: https://nodejs.org/")
    
    return True  # Continue even if some dependencies are missing

def start_fastapi_backend():
    """Start FastAPI backend server"""
    print("🔧 Starting FastAPI backend...")
    
    try:
        # Get paths
        project_root = str(Path(__file__).resolve().parents[1])
        api_backend_path = Path(project_root) / "api_backend"
        
        # Set up environment
        env = os.environ.copy()
        
        # Add project root to Python path (Windows-friendly)
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{project_root};{current_pythonpath}"
        else:
            env['PYTHONPATH'] = project_root
        
        print(f"📁 Project root: {project_root}")
        print(f"📁 API backend: {api_backend_path}")
        
        # Test uvicorn first
        test_result = subprocess.run([
            sys.executable, "-m", "uvicorn", "--help"
        ], capture_output=True, text=True)
        
        if test_result.returncode != 0:
            print("❌ uvicorn not available, trying direct python execution...")
            # Fallback: try running the script directly
            subprocess.Popen([
                sys.executable, str(api_backend_path / "main.py")
            ], cwd=api_backend_path, env=env)
        else:
            # Normal uvicorn execution
            subprocess.Popen([
                sys.executable, "-m", "uvicorn", "main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=api_backend_path, env=env)
        
        print("✅ FastAPI backend starting on http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Failed to start FastAPI backend: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return False

def check_react_setup():
    """Check if React frontend is set up"""
    web_modern_path = Path(__file__).resolve().parents[1] / "web_modern"
    package_json = web_modern_path / "package.json"
    node_modules = web_modern_path / "node_modules"
    
    if not package_json.exists():
        return False, "package.json not found"
    
    if not node_modules.exists():
        return False, "dependencies not installed"
    
    return True, "ready"

def print_instructions():
    """Print usage instructions"""
    print("\n" + "="*70)
    print("📱 FRONTEND OPTIONS")
    print("="*70)
    
    # Check React setup
    react_ready, react_status = check_react_setup()
    
    if react_ready:
        print("🚀 REACT DASHBOARD (Recommended - Fast & Modern)")
        print("   • Navigate to: cd web_modern")
        print("   • Start with: npm run dev")
        print("   • Access at: http://localhost:3000")
        print("   • Features: Real-time updates, mobile-responsive, 10x faster")
    else:
        print("🚀 REACT DASHBOARD (Setup Required)")
        print("   • Navigate to: cd web_modern")
        print("   • Install deps: npm install")
        print("   • Start with: npm run dev")
        print("   • Access at: http://localhost:3000")
    
    print("\n🛠️  STREAMLIT DASHBOARD (Classic - Stable)")
    print("   • Start with: python scripts/run_dashboard.py")
    print("   • Access at: http://localhost:8501")
    print("   • Features: Full-featured, familiar interface")
    
    print("\n📊 API BACKEND")
    print("   • Running at: http://localhost:8000")
    print("   • API docs: http://localhost:8000/api/docs")
    print("   • Health check: http://localhost:8000/api/health")
    
    print("\n" + "="*70)
    print("💡 QUICK START TIPS")
    print("="*70)
    print("• For best performance: Use React dashboard")
    print("• For admin tasks: Use Streamlit dashboard")
    print("• Both share the same data via FastAPI backend")
    print("• Hot reload enabled for development")
    print("="*70 + "\n")

def open_browser_tabs():
    """Open browser tabs for key endpoints"""
    time.sleep(3)  # Wait for servers to start
    
    try:
        # Open API docs
        webbrowser.open("http://localhost:8000/api/docs")
        print("🌐 Opened API documentation in browser")
        
        # Check if React is running, if so open it
        try:
            import requests
            response = requests.get("http://localhost:3000", timeout=2)
            if response.status_code == 200:
                webbrowser.open("http://localhost:3000")
                print("🌐 Opened React dashboard in browser")
        except:
            pass  # React not running yet
            
    except Exception as e:
        print(f"Could not open browser: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check dependencies (but continue even if some are missing)
    check_dependencies()
    
    print("\n🚀 Starting services...\n")
    
    # Start FastAPI backend
    if not start_fastapi_backend():
        print("\n❌ Failed to start backend services.\n")
        return False
    
    # Wait a moment for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(2)
    
    # Open browser tabs in background
    browser_thread = Thread(target=open_browser_tabs)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Print instructions
    print_instructions()
    
    # Keep the script running
    try:
        print("🎯 Backend running! Choose your frontend and start coding!")
        print("📝 Press Ctrl+C to stop the backend server\n")
        
        # Keep alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down RLHF Dashboard...")
        print("Thanks for using the modern dashboard!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 