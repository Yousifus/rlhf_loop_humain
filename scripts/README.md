# Scripts & Automation

This directory contains all automation scripts, runners, and deployment utilities for the RLHF Loop system.

## 📁 Directory Contents

### Dashboard Runners
- **`run_dashboard.py`** - Main dashboard launcher
- **`run_enhanced_dashboard.py`** - Enhanced dashboard with advanced features
- **`run_enhanced_dashboard_v2.py`** - Latest dashboard version with full feature set
- **`setup_dashboard.py`** - Dashboard setup and dependency installation

### API & Integration Scripts
- **`run_with_deepseek.py`** - DeepSeek AI integration runner
- **`set_deepseek_api.py`** - DeepSeek API configuration utility
- **`run_with_api.ps1`** - Generic API runner (PowerShell)
- **`run_with_deepseek.ps1`** - DeepSeek integration (PowerShell)
- **`set_deepseek_api.ps1`** - API key configuration (PowerShell)
- **`set_deepseek_key.ps1`** - API key setup utility

### Training & Processing
- **`train_reward_model.py`** - Reward model training pipeline
- **`run_batch_processor.ps1`** - Batch processing automation
- **`run_rlhf_pipeline.ps1`** - Full RLHF training pipeline

### Development Utilities
- **`start_dashboard.ps1`** - Quick dashboard startup
- **`temp_profile.ps1`** - Temporary PowerShell profile for development

## 🚀 Quick Start

### Launch Dashboard
```bash
# Python Dashboard
python scripts/run_enhanced_dashboard_v2.py

# Or using PowerShell
scripts/start_dashboard.ps1
```

### Run Training Pipeline
```bash
# Full RLHF pipeline
scripts/run_rlhf_pipeline.ps1

# Reward model training only
python scripts/train_reward_model.py
```

### API Configuration
```bash
# Configure DeepSeek API
python scripts/set_deepseek_api.py
# Or via PowerShell
scripts/set_deepseek_key.ps1
```

## 🔧 Requirements

- **Python 3.8+** - For Python scripts
- **PowerShell 5.1+** - For PowerShell automation
- **API Keys** - DeepSeek, OpenAI, or LM Studio as configured

## 📊 Integration

All scripts are designed to work seamlessly with:
- Main RLHF system (`../interface/`)
- Model training (`../models/`)
- Data processing (`../data/`)
- Web interfaces (`../web/`) 