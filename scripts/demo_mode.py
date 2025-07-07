#!/usr/bin/env python3
"""
Demo Mode Controller for RLHF Dashboard

This script allows you to easily switch between demo data and real data
to showcase dashboard capabilities or work with actual project data.

The demo mode now includes RICH, comprehensive data:
• 450+ diverse prompts spanning 6 months of usage
• Realistic model evolution (58% → 87% accuracy)
• 6 distinct content domains with authentic patterns
• Comprehensive calibration and confidence data
• Rich metadata and reflection analysis

Perfect for showcases immediate feature exploration!

Usage:
    python scripts/demo_mode.py enable    # Switch to rich demo data
    python scripts/demo_mode.py disable   # Switch back to real data  
    python scripts/demo_mode.py status    # Check current mode
    python scripts/demo_mode.py refresh   # Regenerate demo data with latest features
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse
import json
from datetime import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# File mappings for demo mode
FILE_MAPPINGS = {
    "votes.jsonl": "demo_votes.jsonl",
    "predictions.jsonl": "demo_predictions.jsonl", 
    "reflection_data.jsonl": "demo_reflection_data.jsonl",
}

# Additional demo files that don't replace real files
ADDITIONAL_DEMO_FILES = [
    "models/demo_calibration_log.json",
    "models/checkpoints/demo_checkpoint_v1_metadata.json",
    "models/checkpoints/demo_checkpoint_v2_metadata.json", 
    "models/checkpoints/demo_checkpoint_v3_metadata.json",
    "models/checkpoints/demo_checkpoint_v4_metadata.json",
]

BACKUP_SUFFIX = ".real_backup"

def create_status_file(is_demo_mode: bool):
    """Create a status file to track current mode"""
    status_file = DATA_DIR / ".demo_mode_status"
    status = {
        "demo_mode": is_demo_mode,
        "timestamp": datetime.now().isoformat(),
        "note": "Rich demo mode - 450+ prompts with realistic evolution patterns" if is_demo_mode else "Real data mode",
        "data_stats": get_demo_data_stats() if is_demo_mode else None
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

def get_demo_data_stats() -> dict:
    """Get statistics about demo data"""
    stats = {
        "total_prompts": 0,
        "date_range": None,
        "categories": {},
        "model_evolution": None
    }
    
    try:
        demo_votes_path = DATA_DIR / "demo_votes.jsonl"
        if demo_votes_path.exists():
            with open(demo_votes_path, 'r') as f:
                votes = [json.loads(line) for line in f]
                
            stats["total_prompts"] = len(votes)
            
            if votes:
                timestamps = [v["timestamp"] for v in votes]
                stats["date_range"] = [min(timestamps)[:10], max(timestamps)[:10]]
                
                # Category breakdown
                for vote in votes:
                    cat = vote.get("generation_metadata", {}).get("category", "unknown")
                    stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
                    
                # Model evolution (compare first vs last 50 entries)
                if len(votes) >= 100:
                    early_accuracy = sum(1 for v in votes[:50] if v.get("chosen_index") == 1) / 50
                    late_accuracy = sum(1 for v in votes[-50:] if v.get("chosen_index") == 1) / 50
                    stats["model_evolution"] = {
                        "early_accuracy": round(early_accuracy, 2),
                        "late_accuracy": round(late_accuracy, 2),
                        "improvement": round(late_accuracy - early_accuracy, 2)
                    }
    except Exception as e:
        print(f"   ⚠ Error calculating demo stats: {e}")
        
    return stats

def get_current_mode() -> bool:
    """Check if currently in demo mode"""
    status_file = DATA_DIR / ".demo_mode_status"
    if not status_file.exists():
        return False
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
            return status.get("demo_mode", False)
    except:
        return False

def backup_real_files():
    """Backup real data files before switching to demo mode"""
    print("📦 Backing up real data files...")
    
    for real_file, demo_file in FILE_MAPPINGS.items():
        real_path = DATA_DIR / real_file
        backup_path = DATA_DIR / (real_file + BACKUP_SUFFIX)
        
        if real_path.exists():
            shutil.copy2(real_path, backup_path)
            print(f"   ✓ Backed up {real_file}")
        else:
            print(f"   ⚠ {real_file} not found (will be created from demo)")

def restore_real_files():
    """Restore real data files from backup"""
    print("🔄 Restoring real data files...")
    
    for real_file, demo_file in FILE_MAPPINGS.items():
        real_path = DATA_DIR / real_file
        backup_path = DATA_DIR / (real_file + BACKUP_SUFFIX)
        
        if backup_path.exists():
            shutil.copy2(backup_path, real_path)
            print(f"   ✓ Restored {real_file}")
        else:
            print(f"   ⚠ No backup found for {real_file}")

def generate_rich_demo_data():
    """Generate comprehensive demo data using the rich generator"""
    print("🎨 Generating rich demo data...")
    
    try:
        # Run the rich demo data generator
        result = subprocess.run([
            sys.executable, 
            str(PROJECT_ROOT / "scripts" / "generate_rich_demo_data.py")
        ], capture_output=True, text=True, check=True)
        
        print("   ✅ Rich demo data generated successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error generating demo data: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("   ❌ Rich demo data generator not found!")
        return False

def enable_demo_mode():
    """Switch to rich demo data"""
    if get_current_mode():
        print("✨ Already in demo mode!")
        show_demo_stats()
        return True
    
    print("🚀 Enabling Rich Demo Mode...")
    print("   📊 This will showcase 450+ prompts with realistic RLHF patterns")
    
    # Backup real files
    backup_real_files()
    
    # Generate fresh demo data if it doesn't exist or is outdated
    demo_votes_path = DATA_DIR / "demo_votes.jsonl"
    needs_generation = True
    
    if demo_votes_path.exists():
        # Check if demo data is recent and comprehensive
        try:
            with open(demo_votes_path, 'r') as f:
                line_count = sum(1 for _ in f)
            if line_count >= 400:  # Rich demo should have 400+ entries
                needs_generation = False
                print("   📁 Using existing rich demo data")
        except:
            pass
    
    if needs_generation:
        print("   🎨 Generating fresh comprehensive demo data...")
        if not generate_rich_demo_data():
            print("   ❌ Failed to generate demo data!")
            return False
    
    # Copy demo files to main locations
    print("\n📋 Installing demo data files...")
    for real_file, demo_file in FILE_MAPPINGS.items():
        real_path = DATA_DIR / real_file
        demo_path = DATA_DIR / demo_file
        
        if demo_path.exists():
            shutil.copy2(demo_path, real_path)
            print(f"   ✓ Installed {demo_file} as {real_file}")
        else:
            print(f"   ❌ Demo file {demo_file} not found!")
            return False
    
    # Update status
    create_status_file(True)
    
    print("\n🎉 Rich Demo Mode Enabled!")
    show_demo_stats()
    print("\n   🚀 Run the dashboard to explore all visualizations!")
    
    return True

def disable_demo_mode():
    """Switch back to real data"""
    if not get_current_mode():
        print("📊 Already using real data!")
        return True
    
    print("🔄 Disabling Demo Mode...")
    print("   Switching back to your real project data")
    
    # Restore real files
    restore_real_files()
    
    # Clean up backup files
    print("\n🧹 Cleaning up backup files...")
    for real_file in FILE_MAPPINGS.keys():
        backup_path = DATA_DIR / (real_file + BACKUP_SUFFIX)
        if backup_path.exists():
            backup_path.unlink()
            print(f"   ✓ Removed backup {real_file + BACKUP_SUFFIX}")
    
    # Update status
    create_status_file(False)
    
    print("\n✅ Demo Mode Disabled!")
    print("   📊 Dashboard now shows your real project data")
    print("   💎 All demo data preserved in demo_*.jsonl files")
    
    return True

def refresh_demo_data():
    """Regenerate demo data with latest features"""
    print("🔄 Refreshing Demo Data...")
    print("   Regenerating with latest features and patterns")
    
    was_in_demo_mode = get_current_mode()
    
    # Generate fresh demo data
    if not generate_rich_demo_data():
        print("   ❌ Failed to refresh demo data!")
        return False
    
    # If we were in demo mode, reinstall the refreshed data
    if was_in_demo_mode:
        print("\n📋 Reinstalling refreshed demo data...")
        for real_file, demo_file in FILE_MAPPINGS.items():
            real_path = DATA_DIR / real_file
            demo_path = DATA_DIR / demo_file
            
            if demo_path.exists():
                shutil.copy2(demo_path, real_path)
                print(f"   ✓ Updated {real_file}")
        
        # Update status
        create_status_file(True)
    
    print("\n✅ Demo Data Refreshed!")
    if was_in_demo_mode:
        show_demo_stats()
    print("   🎨 Latest features and improvements included")
    
    return True

def show_demo_stats():
    """Show rich statistics about demo data"""
    stats = get_demo_data_stats()
    
    if stats["total_prompts"] > 0:
        print(f"\n📊 Rich Demo Dataset Statistics:")
        print(f"   📝 Total prompts: {stats['total_prompts']}")
        
        if stats["date_range"]:
            print(f"   📅 Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
            
        if stats["categories"]:
            print(f"   📚 Content categories:")
            for cat, count in stats["categories"].items():
                pct = count / stats["total_prompts"] * 100
                print(f"     • {cat}: {count} prompts ({pct:.1f}%)")
                
        if stats["model_evolution"]:
            evo = stats["model_evolution"]
            print(f"   📈 Model evolution: {evo['early_accuracy']:.1%} → {evo['late_accuracy']:.1%} accuracy (+{evo['improvement']:.1%})")

def show_status():
    """Show current demo mode status"""
    try:
        is_demo = get_current_mode()
        
        print("📊 RLHF Dashboard Data Mode Status")
        print("=" * 50)
        
        if is_demo:
            print("🎮 Current Mode: RICH DEMO DATA")
            print("📋 Data Source: Comprehensive demonstration dataset")
            show_demo_stats()
            print("\n💡 Perfect for:")
            print("   • Portfolio showcases and interviews")
            print("   • Exploring all dashboard features instantly")
            print("   • Understanding RLHF patterns and evolution")
            print("   • Demonstrating model improvement over time")
            print("\n🔄 To switch to real data: python scripts/demo_mode.py disable")
        else:
            print("🔬 Current Mode: REAL DATA")
            print("📋 Data Source: Your actual project data")
            print("📊 Features: Live training data and annotations")
            print("\n💡 Perfect for:")
            print("   • Active RLHF training and development")
            print("   • Real model development and monitoring")
            print("   • Production system monitoring")
            print("\n🎮 To switch to rich demo: python scripts/demo_mode.py enable")
        
        # Show file status
        print(f"\n📁 Data Files (in {DATA_DIR}):")
        for real_file, demo_file in FILE_MAPPINGS.items():
            try:
                real_path = DATA_DIR / real_file
                demo_path = DATA_DIR / demo_file
                backup_path = DATA_DIR / (real_file + BACKUP_SUFFIX)
                
                status = "✓" if real_path.exists() else "❌"
                size = f"({real_path.stat().st_size // 1024}KB)" if real_path.exists() else ""
                
                print(f"   {status} {real_file} {size}")
                
                if is_demo and demo_path.exists():
                    demo_size = f"({demo_path.stat().st_size // 1024}KB)"
                    print(f"       └── Source: {demo_file} {demo_size}")
                elif backup_path.exists():
                    print(f"       └── Backup available: {real_file + BACKUP_SUFFIX}")
            except Exception as e:
                print(f"   ❌ Error checking {real_file}: {e}")
                
    except Exception as e:
        print(f"❌ Error showing status: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Control rich demo mode for RLHF Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo_mode.py enable     # Switch to rich demo data (450+ prompts)
  python scripts/demo_mode.py disable    # Switch to real data  
  python scripts/demo_mode.py status     # Check current mode
  python scripts/demo_mode.py refresh    # Regenerate demo data with latest features

Rich Demo Mode provides comprehensive data perfect for:
• Portfolio showcases and interviews
• Understanding all dashboard features instantly
• Exploring realistic RLHF patterns and model evolution
• Demonstrating 6 months of system usage in seconds

The rich demo includes:
• 450+ diverse prompts across 6 content domains
• Realistic model evolution (58% → 87% accuracy)
• Comprehensive calibration and confidence data
• Rich metadata and reflection analysis
• Authentic usage patterns and temporal trends
        """
    )
    
    parser.add_argument(
        "action",
        choices=["enable", "disable", "status", "refresh"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    if args.action == "enable":
        success = enable_demo_mode()
        sys.exit(0 if success else 1)
    elif args.action == "disable":
        success = disable_demo_mode()
        sys.exit(0 if success else 1)
    elif args.action == "refresh":
        success = refresh_demo_data()
        sys.exit(0 if success else 1)
    elif args.action == "status":
        success = show_status()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 