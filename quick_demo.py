#!/usr/bin/env python3
"""
🎮 Quick Demo Launcher for RLHF Dashboard

This script automatically enables demo mode and launches the dashboard
for instant showcase of all features. Perfect for:
• Portfolio demonstrations
• Feature exploration  
• Understanding RLHF concepts

Usage: python quick_demo.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 RLHF Dashboard - Quick Demo Launcher")
    print("=" * 50)
    
    # Check current demo status first
    print("🔍 Checking current demo mode status...")
    try:
        result = subprocess.run([
            "python", "scripts/demo_mode.py", "status"
        ], check=True, capture_output=True, text=True)
        
        # Check if already in demo mode
        if "🎮 Current Mode: DEMO" in result.stdout:
            print("✅ Demo mode already enabled!")
        else:
            print("🎯 Enabling demo mode with rich sample data...")
            # Enable demo mode
            subprocess.run([
                "python", "scripts/demo_mode.py", "enable"
            ], check=True, capture_output=True, text=True)
            print("✅ Demo mode enabled!")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error with demo mode: {e}")
        print("🔄 Continuing anyway - dashboard may work with existing data...")
    except FileNotFoundError:
        print("❌ Python not found in PATH")
        print("🔄 Continuing anyway - dashboard may work with existing data...")
    
    print("\n🚀 Launching RLHF Pipeline Monitor...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("📈 Features: 30 diverse prompts with realistic data patterns")
    print("🎯 No API keys required - everything works instantly!")
    print("\n💡 Press Ctrl+C to stop the dashboard when done")
    print("-" * 50)
    
    # Launch dashboard
    try:
        # Use streamlit directly from current environment instead of sys.executable
        subprocess.run([
            "streamlit", "run", 
            "scripts/run_dashboard.py",
            "--server.runOnSave=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n🎉 Demo session completed!")
        print("💾 Demo data preserved for future use")
        print("🔄 To switch back to real data: python scripts/demo_mode.py disable")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("💡 Try running directly: streamlit run scripts/run_dashboard.py")
        return False
    except FileNotFoundError:
        print("\n❌ Streamlit not found in current environment")
        print("💡 Make sure you're in the virtual environment with: pip install streamlit")
        print("💡 Or run directly: streamlit run scripts/run_dashboard.py")
        return False
    
    return True

if __name__ == "__main__":
    main() 