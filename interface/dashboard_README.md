# RLHF Pipeline Monitor

The RLHF Pipeline Monitor provides comprehensive monitoring and analysis across the complete RLHF lifecycle, from data collection through deployment.

## Features

The dashboard consists of five main tabs:

1. **Alignment Over Time**: Track model prediction accuracy and confidence over time
2. **Calibration Diagnostics**: Analyze model calibration quality and history
3. **Drift Clusters & Error Zones**: Identify and analyze clusters of prediction errors
4. **Model Evolution**: Compare model checkpoints and track improvements
5. **User Preference Timeline**: Visualize user feedback patterns

## Getting Started

### Prerequisites

Ensure you have all required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Dashboard

To launch the dashboard, run:

```bash
cd /path/to/project
streamlit run scripts/run_dashboard.py
```

The dashboard will be available at http://localhost:8501 by default.

## Data Sources

The dashboard uses the following data sources:

- `models/meta_reflection_log.jsonl`: Model predictions and reflections
- `models/calibration_log.json`: Calibration history and parameters
- `models/drift_analysis/`: Drift detection results
- `models/checkpoints/`: Model checkpoint metadata
- `data/vote_logs/`: User preference vote logs

## Tips for Use

- Use the time slider to focus on specific time periods
- Adjust the bin count in calibration diagnostics to see different granularity
- Examine drift clusters to identify systematic error patterns
- Compare checkpoint versions to track model improvements
- View recent completion pairs to understand user preferences

## Troubleshooting

If you encounter issues:

1. Ensure all data files exist in their expected locations
2. Check that the files are properly formatted (valid JSON/JSONL)
3. Verify that the dashboard has permission to read the files
4. Check the log output for specific error messages

### Common Issues

#### CORS Errors in Browser Console

If you see CORS-related errors in the browser console related to "webhooks.fivetran.com", these are due to Streamlit's telemetry services and don't impact dashboard functionality. 

To completely disable Streamlit telemetry and resolve these errors, you can use one of these methods:

1. Use the main dashboard script which already disables telemetry:
   ```bash
   streamlit run scripts/run_dashboard.py
   ```
2. Set environment variables before running Streamlit:
   ```bash
   set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   streamlit run scripts/run_dashboard.py
   ```
3. Create a `.streamlit/config.toml` file in your project root with:
   ```toml
   [browser]
   gatherUsageStats = false
   ```
4. Use command line flags:
   ```bash
   streamlit run scripts/run_dashboard.py --browser.gatherUsageStats=false
   ```

If you still see the errors, try clearing your browser cache or restarting your browser.

#### Streamlit Version Compatibility

The dashboard is designed to work with Streamlit 1.20.0 or later. If you encounter API-related errors, make sure your Streamlit version is up to date:

```bash
pip install --upgrade streamlit
``` 