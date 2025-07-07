# RLHF System Architecture

This document provides a visualization of the RLHF Loop system architecture using Mermaid diagrams. It shows the component relationships, data flow, and implementation structure.

## System Components

```mermaid
graph TD
    subgraph "Data Generation & Collection"
        A[Prompt Generator] --> B[Completion Generator]
        B --> C[Human Feedback Collection]
        B --> D[Vote Predictor]
        C --> D
    end
    
    subgraph "RLHF Core"
        D --> E[Model Calibration]
        E --> F[Retraining Data Preparation]
        F --> G[Model Training/Fine-tuning]
        G --> H[Drift Monitoring]
        H --> D
    end
    
    subgraph "Visualization & Analysis"
        H --> I[RLHF Pipeline Monitor]
        D --> I
        E --> I
        C --> I
    end
    
    classDef dataGeneration fill:#f9f,stroke:#333,stroke-width:2px;
    classDef coreProcessing fill:#bbf,stroke:#333,stroke-width:2px;
    classDef visualization fill:#fbf,stroke:#333,stroke-width:2px;
    
    class A,B,C dataGeneration;
    class D,E,F,G,H coreProcessing;
    class I visualization;
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph "Input"
        PROMPT[Prompt Generation]
        COMPLETE[Completion Generation]
    end
    
    subgraph "Feedback"
        HUMAN[Human Voting]
        AI[Vote Prediction]
    end
    
    subgraph "Processing"
        CALIB[Calibration]
        DRIFT[Drift Detection]
        TRAIN[Model Training]
    end
    
    subgraph "Output"
        VISUAL[Dashboard Visualization]
        RETRAIN[Retraining Data]
    end
    
    PROMPT --> |generated_prompts.jsonl| COMPLETE
    COMPLETE --> |raw_completions_log.jsonl| HUMAN
    COMPLETE --> |completions for prediction| AI
    HUMAN --> |votes.jsonl| AI
    HUMAN --> |votes.jsonl| VISUAL
    AI --> |predictions.jsonl| CALIB
    AI --> |prediction results| VISUAL
    CALIB --> |calibration_log.json| VISUAL
    CALIB --> |calibrated model| DRIFT
    DRIFT --> |drift_analysis| VISUAL
    DRIFT --> |drift_clusters.jsonl| RETRAIN
    RETRAIN --> |vote_predictor_retrain_data.jsonl| TRAIN
    TRAIN --> |model checkpoints| AI
    TRAIN --> |model evolution data| VISUAL
```

## Dashboard Architecture

```mermaid
graph TD
    subgraph "Dashboard Core"
        MAIN[Dashboard Entry Point]
        DATA[Data Loader]
        VISUAL[Visualization Module]
        THEME[Theme Module]
        UTIL[Utilities]
    end
    
    subgraph "Dashboard Tabs"
        TAB1[Attunement Timeline]
        TAB2[Confidence & Calibration]
        TAB3[Drift Clusters]
        TAB4[Model Evolution]
        TAB5[Human-System Entanglement]
    end
    
    MAIN --> DATA
    DATA --> VISUAL
    THEME --> VISUAL
    UTIL --> VISUAL
    VISUAL --> TAB1
    VISUAL --> TAB2
    VISUAL --> TAB3
    VISUAL --> TAB4
    VISUAL --> TAB5
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef tabs fill:#9f6,stroke:#333,stroke-width:2px;
    
    class MAIN,DATA,VISUAL,THEME,UTIL core;
    class TAB1,TAB2,TAB3,TAB4,TAB5 tabs;
```

## Directory Structure

```mermaid
graph TD
    subgraph "RLHF Loop Project"
        ROOT[Root Directory] --> DATA[data/]
        ROOT --> DOCS[docs/]
        ROOT --> INTERFACE[interface/]
        ROOT --> MODELS[models/]
        ROOT --> PROMPTS[prompts/]
        ROOT --> TASKS[tasks/]
        ROOT --> UTILS[utils/]
        ROOT --> SCRIPTS[scripts/]
        ROOT --> CONFIG[configuration files]
    end
    
    DATA --> D1[votes.jsonl]
    DATA --> D2[raw_completions_log.jsonl]
    DATA --> D3[predictions.jsonl]
    DATA --> D4[vote_predictor_training_data.jsonl]
    
    DOCS --> DOC1[architecture.md]
    DOCS --> DOC2[visualization_reference.md]
    
    INTERFACE --> I1[voting_ui.py]
    INTERFACE --> I2[rlhf_loop.py]
    INTERFACE --> I3[eval_probe.py]
    INTERFACE --> I4[run_drift_analysis.py]
    INTERFACE --> I5[scripts/run_dashboard.py]
    
    MODELS --> M1[vote_predictor_checkpoint/]
    MODELS --> M2[meta_reflection_log.jsonl]
    MODELS --> M3[calibration_log.json]
    MODELS --> M4[drift_analysis/]
    
    PROMPTS --> P1[generator.py]
    PROMPTS --> P2[generated_prompts.jsonl]
    
    UTILS --> U1[completions.py]
    UTILS --> U2[vote_predictor/]
    UTILS --> U3[batch_processor.py]
    
    U2 --> VP1[data_prep.py]
    U2 --> VP2[train.py]
    U2 --> VP3[calibrate.py]
    U2 --> VP4[predict.py]
    U2 --> VP5[retrain_data_prep.py]
    U2 --> VP6[drift_monitor.py]
    
    TASKS --> T1[task_001.txt]
    TASKS --> T2[task_002.txt]
    TASKS --> T3[tasks.json]
    
    SCRIPTS --> S1[prd.txt]
    
    CONFIG --> C1[.taskmasterconfig]
    CONFIG --> C2[requirements.txt]
```

## System Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initialize
    
    Initialize --> PromptGeneration
    PromptGeneration --> CompletionGeneration
    
    state "Feedback Collection" as FeedbackCollection {
        [*] --> HumanFeedback
        [*] --> AIFeedback
        HumanFeedback --> [*]
        AIFeedback --> [*]
    }
    
    CompletionGeneration --> FeedbackCollection
    FeedbackCollection --> ModelCalibration
    
    state "Performance Analysis" as PerformanceAnalysis {
        [*] --> DriftDetection
        [*] --> Visualization
        DriftDetection --> Visualization
        Visualization --> [*]
    }
    
    ModelCalibration --> PerformanceAnalysis
    
    state "Model Improvement" as ModelImprovement {
        [*] --> PrepareRetrainingData
        PrepareRetrainingData --> Retraining
        Retraining --> [*]
    }
    
    PerformanceAnalysis --> ModelImprovement
    ModelImprovement --> FeedbackCollection: New model version
```

## Dashboard Component Interaction

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant DataLoader
    participant Visualizations
    participant RLHF_System
    
    User->>Dashboard: Open dashboard
    Dashboard->>DataLoader: Request data
    DataLoader->>RLHF_System: Load logs & model data
    RLHF_System-->>DataLoader: Return data
    DataLoader-->>Dashboard: Process & return data
    Dashboard->>Visualizations: Generate visualizations
    Visualizations-->>Dashboard: Return interactive charts
    Dashboard-->>User: Display dashboard
    
    User->>Dashboard: Select time range
    Dashboard->>Visualizations: Update with new range
    Visualizations-->>Dashboard: Return updated charts
    Dashboard-->>User: Display filtered view
    
    User->>Dashboard: Click on error cluster
    Dashboard->>DataLoader: Request cluster details
    DataLoader->>RLHF_System: Get example data
    RLHF_System-->>DataLoader: Return examples
    DataLoader-->>Dashboard: Process examples
    Dashboard-->>User: Show drill-down view
```

This architecture documentation provides a comprehensive visual overview of the RLHF Loop system structure, component relationships, and information flow. Use these diagrams as a reference for understanding the system design and implementation approach. 