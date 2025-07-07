# RLHF Pipeline Monitor

The RLHF Pipeline Monitor provides a comprehensive visualization and monitoring system for the RLHF training lifecycle. It helps track model performance, pipeline health, and system metrics across 4 distinct operational phases.

## Structure

The dashboard follows a clean architecture organized around the RLHF pipeline lifecycle:

### Core Entry Point

- `scripts/run_dashboard.py` - Main RLHF Pipeline Monitor entry point with HUMAIN OS styling

### Reusable Components

- `components/` - Shared components and utilities
  - `data_loader.py` - Data loading and preprocessing
  - `utils.py` - Helper functions and constants
  - `visualization.py` - Reusable visualization components

### Pipeline Phases

The dashboard is organized into 4 RLHF pipeline phases:

#### **📊 Phase 1: Data Collection**
- `sections/overview.py` - Data ingestion pipeline monitoring
- `sections/annotation.py` - Annotation quality control and management
- `sections/model_insights.py` - Dataset analytics and distribution analysis

#### **🚀 Phase 2: Training** 
- `sections/model_evolution.py` - Training status and progress monitoring
- `sections/calibration.py` - Loss curve analysis and metrics tracking
- `sections/drift_analysis.py` - Resource utilization monitoring

#### **🎯 Phase 3: Evaluation**
- `sections/model_config_core.py` - Performance metrics and validation
- `sections/calibration.py` - Calibration analysis and confidence validation
- `sections/drift_analysis.py` - Drift detection and statistical monitoring

#### **🌐 Phase 4: Deployment**
- `sections/alignment.py` - Serving status and deployment health
- `sections/model_insights.py` - Production metrics and inference monitoring
- `sections/overview.py` - System health and infrastructure monitoring

### Supporting Components

- `sections/chat.py` - Hidden debug chat interface (accessible via ?debug=chat)
- `ux_improvements.py` - HUMAIN OS styling and design system

## Running the Dashboard

To run the RLHF Pipeline Monitor:

```bash
python scripts/run_dashboard.py
```

Access the dashboard at:
- **Main Interface:** http://localhost:8501
- **Debug Mode:** http://localhost:8501?debug=chat

## Key Features

### 🏭 **Pipeline Monitoring**

1. **📊 Data Collection Phase**
   - Real-time data ingestion monitoring
   - Annotation quality control with inter-annotator agreement
   - Dataset analytics and distribution analysis
   - Data pipeline health and validation metrics

2. **🚀 Training Phase**
   - Live training status and progress tracking
   - Multi-metric loss curve analysis and monitoring
   - Resource utilization (GPU/CPU/Memory) monitoring
   - Training efficiency and performance optimization

3. **🎯 Evaluation Phase**
   - Model performance metrics and accuracy tracking
   - Calibration analysis and confidence validation
   - Statistical drift detection and change monitoring
   - Evaluation pipeline health and validation

4. **🌐 Deployment Phase**
   - Production serving status monitoring
   - Live inference performance and metrics
   - System health and infrastructure monitoring
   - Production deployment validation and alerts

### 🎨 **HUMAIN OS Design System**

- **Professional Styling** - HUMAIN teal (`#1DB584`) color scheme
- **Clean Interface** - Professional white backgrounds with subtle shadows
- **Responsive Layout** - Mobile-friendly, accessible design
- **Performance Optimized** - Fast loading and smooth interactions
- **Accessibility** - WCAG 2.1 AA compliant interface elements

### 🔧 **Technical Features**

- **API Integration** - Built-in DeepSeek/OpenAI API key management
- **Real-time Updates** - Live data refresh and monitoring
- **Interactive Visualizations** - Dynamic charts with drill-down capabilities
- **Error Handling** - Comprehensive validation and recovery
- **Session Management** - Efficient state management and caching

## Requirements

See `requirements.txt` for necessary dependencies:

- **streamlit** - Web application framework
- **plotly** - Interactive visualizations
- **pandas** - Data processing
- **numpy** - Numerical computations
- **transformers** - AI model integration
- **scikit-learn** - ML utilities

## Architecture

The RLHF Pipeline Monitor implements a clean separation of concerns:

```
scripts/run_dashboard.py           # Main entry point
├── 4-Phase Pipeline Structure     # Organized workflow
├── HUMAIN OS Styling             # Professional design system
├── API Key Management            # Secure credential handling
├── Real-time Monitoring          # Live data updates
└── Debug Chat Interface          # Hidden development tool
```

## Configuration

The dashboard supports:
- **Environment Variables** - API key configuration
- **Session Storage** - Temporary API key storage
- **Debug Mode** - Enhanced development features
- **Custom Styling** - HUMAIN OS theme 