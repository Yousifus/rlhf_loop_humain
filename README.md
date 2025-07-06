# RLHF Loop System

A comprehensive Reinforcement Learning from Human Feedback (RLHF) system with predictive modeling and calibrated confidence scores.

## Project Overview

This project implements a full RLHF loop system with the following components:

1. **Prompt Generation** - Creates prompts for language model responses
2. **Completion Generation** - Generates multiple completions for each prompt
3. **Human Feedback Collection** - CLI interface for collecting human preferences between completions
4. **Vote Prediction** - Transformer-based model to predict human preferences with calibrated confidence
5. **RLHF Integration** - System to tie all components together into a feedback loop
6. **Drift Monitoring** - Detects and analyzes model preference drift over time
7. **Batch Processing** - Tools for processing multiple prompts in parallel

## Architecture

![RLHF Loop Architecture](docs/architecture.png)

### Data Flow

```
prompts/generator.py → prompts/generated_prompts.jsonl
utils/completions.py → data/raw_completions_log.jsonl
interface/voting_ui.py → data/votes.jsonl
utils/vote_predictor/data_prep.py → data/vote_predictor_training_data.jsonl
utils/vote_predictor/train.py → models/vote_predictor_checkpoint/
interface/eval_probe.py → models/meta_reflection_log.jsonl
utils/vote_predictor/calibrate.py → models/calibration_log.json
utils/vote_predictor/predict.py → data/predictions.jsonl
utils/vote_predictor/retrain_data_prep.py → data/vote_predictor_retrain_data.jsonl
utils/vote_predictor/drift_monitor.py → models/drift_analysis/
interface/rlhf_loop.py → Integrated loop control
```

## Project Components

### 1. Vote Predictor

The core of this system is a binary preference model that predicts human choices between two completions:

- **Data Preparation** (`utils/vote_predictor/data_prep.py`) - Transforms raw votes into training data
- **Model Training** (`utils/vote_predictor/train.py`) - Trains a transformer model on preference data
- **Model Calibration** (`utils/vote_predictor/calibrate.py`) - Applies temperature/Platt scaling to align confidence with accuracy
- **Prediction API** (`utils/vote_predictor/predict.py`) - Makes calibrated predictions on new completion pairs
- **Retraining Data Preparation** (`utils/vote_predictor/retrain_data_prep.py`) - Prepares training data for fine-tuning based on error patterns
- **Drift Monitoring** (`utils/vote_predictor/drift_monitor.py`) - Detects and analyzes model preference drift over time

### 2. Interface Components

- **Voting UI** (`interface/voting_ui.py`) - CLI for collecting human preferences
- **Evaluation Probe** (`interface/eval_probe.py`) - Introspection tool to assess model vs. human alignment
- **RLHF Loop** (`interface/rlhf_loop.py`) - Main control interface for running the full RLHF cycle
- **Drift Analysis Runner** (`interface/run_drift_analysis.py`) - Runs drift monitoring analysis with visualizations

### 3. Utility Modules

- **Completions Generator** (`utils/completions.py`) - Generates model completions for prompts

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers (HuggingFace)
- NumPy, Matplotlib, SciPy
- scikit-learn

### Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the RLHF Loop

To run the full RLHF loop with default settings:

```bash
python interface/rlhf_loop.py
```

With custom parameters and drift monitoring:

```bash
python interface/rlhf_loop.py --num-prompts 10 --human-feedback-ratio 0.3 --confidence-threshold 0.8 --monitor-drift
```

### Running the Attunement Dashboard

To run the Streamlit-based Attunement Dashboard:

```bash
python run_dashboard.py
```

To run with DeepSeek AI integration (for domain generation and completions):

```powershell
.\run_with_deepseek.ps1
```

### Using Batch Processing

The batch processor allows you to generate completions for multiple prompts in parallel:

1. **Create a JSON file with prompts**:
   Create a file with prompts following the format in `prompts/sample_batch.json`

2. **Run the batch processor**:
   ```powershell
   .\run_batch_processor.ps1 -InputFile prompts/sample_batch.json
   ```

3. **Customizing batch processing**:
   ```powershell
   .\run_batch_processor.ps1 -InputFile prompts/sample_batch.json -MaxWorkers 4 -Temperature 0.8 -MaxTokens 300
   ```

Batch processing results are saved as both JSON and CSV in the output directory for easy analysis.

### Training the Vote Predictor

1. **Prepare data**:
   ```bash
   python utils/vote_predictor/data_prep.py
   ```

2. **Train model**:
   ```bash
   python utils/vote_predictor/train.py
   ```

3. **Calibrate confidence**:
   ```bash
   python utils/vote_predictor/calibrate.py
   ```

4. **Run predictions**:
   ```bash
   python utils/vote_predictor/predict.py --interactive
   ```

5. **Prepare retraining data** (after collecting feedback):
   ```bash
   python utils/vote_predictor/retrain_data_prep.py
   ```

6. **Fine-tune model**:
   ```bash
   python utils/vote_predictor/train.py --retrain --checkpoint models/vote_predictor_checkpoint
   ```

## Evaluation and Monitoring

### Model Evaluation

The vote predictor model performance can be evaluated using:

```bash
python interface/eval_probe.py
```

This will compare model predictions with human judgments and generate a reliability diagram for confidence calibration.

### Drift Monitoring

Monitor model drift over time to detect changes in model performance or shifts in data patterns:

```bash
python interface/run_drift_analysis.py --generate-report --visualization-mode detailed
```

Key drift monitoring features:
- Time-based drift analysis
- Semantic clustering of examples with similar characteristics
- Confidence calibration drift detection
- Visualizations and HTML reporting

To enable continuous drift monitoring in the RLHF loop:

```bash
python interface/rlhf_loop.py --monitor-drift
```

## Project Structure

```
rlhf_loop/
├── data/
│   ├── votes.jsonl
│   ├── raw_completions_log.jsonl
│   ├── vote_predictor_training_data.jsonl
│   ├── vote_predictor_retrain_data.jsonl
│   ├── predictions.jsonl
│   └── batch_results/
├── docs/
│   └── architecture.png
├── interface/
│   ├── eval_probe.py
│   ├── rlhf_loop.py
│   ├── voting_ui.py
│   ├── run_drift_analysis.py
│   └── attunement_dashboard.py
├── models/
│   ├── vote_predictor_checkpoint/
│   ├── meta_reflection_log.jsonl
│   ├── calibration_log.json
│   └── drift_analysis/
├── prompts/
│   ├── generator.py
│   ├── generated_prompts.jsonl
│   └── sample_batch.json
├── utils/
│   ├── completions.py
│   ├── batch_processor.py
│   ├── setup_deepseek.py
│   └── vote_predictor/
│       ├── data_prep.py
│       ├── train.py
│       ├── calibrate.py
│       ├── predict.py
│       ├── retrain_data_prep.py
│       └── drift_monitor.py
├── run_dashboard.py
├── run_with_deepseek.ps1
├── run_batch_processor.ps1
└── README.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# RLHF Attunement Dashboard

We've implemented a new modular dashboard architecture for better visualization and monitoring of the RLHF training loop. The dashboard provides comprehensive visualizations for:

- **Alignment Over Time**: Track accuracy trends, confidence analysis, agreement metrics, and error distribution.
- **Calibration Diagnostics**: Analyze model calibration with calibration curves, confidence histograms, and reliability diagrams.
- **Drift Clusters & Error Zones**: Discover error patterns through clustering (TF-IDF and semantic), temporal drift analysis, and semantic change detection.
- **Model Evolution**: Compare checkpoints, visualize performance trajectory, and analyze training data impact.
- **User Preference Timeline**: Review historical annotations and preference trends.

### Dashboard Structure

The dashboard follows a modular architecture:

- `interface/dashboard_core.py` - Main entry point
- `interface/components/` - Reusable components 
- `interface/sections/` - Visualization sections

### Running the Dashboard

```bash
# Install required packages
python setup_dashboard.py

# Run the dashboard
streamlit run interface/dashboard_core.py
```

See `interface/README.md` for more details on the dashboard.
