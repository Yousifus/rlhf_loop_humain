# RLHF Attunement Dashboard

The RLHF Attunement Dashboard provides a comprehensive visualization and monitoring system for the RLHF training loop. It helps track model alignment, performance metrics, and error patterns over time.

## Structure

The dashboard follows a modular architecture with the following components:

### Core Modules

- `dashboard_core.py` - Main entry point and navigation controller
- `components/` - Reusable components and utilities
  - `data_loader.py` - Data loading and preprocessing
  - `utils.py` - Helper functions and constants
  - `visualization.py` - Reusable visualization components

### Visualization Sections

- `sections/overview.py` - Dashboard overview and summary
- `sections/annotation.py` - Annotation interface and history
- `sections/alignment.py` - Alignment over time visualizations
- `sections/calibration.py` - Calibration diagnostics
- `sections/drift_analysis.py` - Drift clusters and error zones
- `sections/model_evolution.py` - Model evolution tracking
- `sections/chat.py` - Interactive chat interface

## Running the Dashboard

To run the dashboard:

```bash
streamlit run interface/dashboard_core.py
```

## Key Features

1. **Alignment Over Time**
   - Accuracy trends over time
   - Confidence analysis
   - Agreement metrics
   - Error distribution

2. **Calibration Diagnostics**
   - Calibration curves
   - Confidence histograms
   - Reliability diagrams
   - Calibration metrics

3. **Drift Clusters & Error Zones**
   - Error case clustering (TF-IDF and semantic)
   - Temporal drift analysis
   - Semantic change detection
   - Cluster analysis

4. **Model Evolution**
   - Checkpoint comparison
   - Performance trajectory
   - Training data impact
   - Version details

5. **User Preference Timeline**
   - Historical annotations
   - User feedback patterns
   - Preference trends

## Requirements

See `requirements.txt` for necessary dependencies. 