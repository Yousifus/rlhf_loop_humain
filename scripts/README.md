# Scripts & Automation

This directory contains essential automation scripts for the RLHF Loop system - streamlined for professional deployment and easy first-time setup.

## 📁 Essential Scripts

### 🎛️ **Dashboard & Interface**
- **`run_dashboard.py`** - Main RLHF monitoring dashboard
  ```bash
  streamlit run scripts/run_dashboard.py
  ```

### 🚀 **Core Pipeline**
- **`run_rlhf_pipeline.ps1`** - Complete RLHF training pipeline
  ```powershell
  scripts/run_rlhf_pipeline.ps1
  ```

### 🧠 **Model Training**
- **`train_reward_model.py`** - Reward model training from annotations
  ```bash
  python scripts/train_reward_model.py
  ```

### ⚙️ **Setup & Processing**
- **`setup_dashboard.py`** - Install dependencies and configure environment
  ```bash
  python scripts/setup_dashboard.py
  ```
- **`run_batch_processor.ps1`** - Batch completion generation
  ```powershell
  scripts/run_batch_processor.ps1 -InputFile data/prompts.json
  ```

## 🚀 Quick Start Guide

### 1. **First-Time Setup**
```bash
# Install dependencies
python scripts/setup_dashboard.py

# Verify installation
python -c "import streamlit, pandas, plotly; print('Dependencies installed successfully')"
```

### 2. **Launch Dashboard**
```bash
# Start the RLHF monitoring interface
streamlit run scripts/run_dashboard.py
```
*Dashboard will be available at: http://localhost:8501*

### 3. **Run Full Pipeline**
```powershell
# Execute complete RLHF training workflow
scripts/run_rlhf_pipeline.ps1
```

### 4. **Train Reward Model**
```bash
# Train preference model from collected annotations
python scripts/train_reward_model.py --epochs 3
```

## 🔧 System Requirements

- **Python 3.8+** - Core runtime
- **PowerShell 5.1+** - For automation scripts
- **8GB RAM** - Recommended for model training
- **API Access** - DeepSeek, OpenAI, or LM Studio (optional)

## 📊 Integration

All scripts integrate seamlessly with:
- **Data Processing** (`../data/`) - Training datasets and results
- **Model Storage** (`../models/`) - Trained models and checkpoints  
- **Interface Components** (`../interface/`) - Dashboard and UI elements
- **Utilities** (`../utils/`) - Core processing functions

## 🛡️ Security Notes

- **No hardcoded API keys** - All credentials via environment variables
- **Secure configuration** - API keys stored safely outside repository
- **Clean architecture** - No temporary or development files included

## 📈 Professional Features

- **Real-time monitoring** - Live performance metrics
- **Calibrated predictions** - Confidence score validation
- **Drift detection** - Performance degradation alerts
- **Batch processing** - Efficient completion generation
- **Model versioning** - Training history and checkpoints

---

*This streamlined script collection provides everything needed for professional RLHF system deployment and monitoring.* 