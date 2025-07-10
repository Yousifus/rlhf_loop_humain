#!/usr/bin/env python3
"""
Simple FastAPI Backend Starter - Backup launcher
Use this if the main launcher has issues
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)

try:
    print("🚀 Starting RLHF Dashboard API (Simple Mode)...")
    print(f"📁 Project root: {project_root}")
    
    # Import and run the app directly
    from main import app
    import uvicorn
    
    print("✅ Modules imported successfully!")
    print("🌐 Starting server on http://localhost:8000")
    print("📊 API docs will be at: http://localhost:8000/api/docs")
    print("🔄 Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled reload to fix startup issue
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you've installed dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    print("🔍 Check that port 8000 is not already in use")
    sys.exit(1) 