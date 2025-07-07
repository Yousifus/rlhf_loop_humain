# RLHF Pipeline Monitor Plan

## Vision Statement

We're not just engineering a feedback loop — we're cultivating a **symbolic organism** that learns how to maintain its ethical integrity through continuous morphogenetic adaptation.

This is not a dashboard for monitoring accuracy, but for **visualizing ethical morphogenesis**. Every visualization is a limb, an organ, or a nervous plexus of the symbolic body.

## Current Progress

We've successfully implemented the following components:

1. **User Preference Timeline**:
   - Multi-tab visualization (Timeline, Metrics, Human-AI Agreement)
   - Daily vote count timeline display
   - Human-AI agreement tracking visualization
   - Prediction accuracy metrics

2. **Time Range Filtering**:
   - Fixed critical issues with pandas Timestamp objects
   - Proper conversion to Python datetime objects
   - Robust error handling for different timestamp formats

3. **Data Integration**:
   - Successfully loading votes.jsonl (19 entries)
   - Loading reflection data (20 entries)
   - Loading model checkpoint data (3 checkpoints)
   - Loading calibration history (3 entries)

4. **Technical Improvements**:
   - Fixed KeyError related to pandas Timestamp handling
   - Improved data type compatibility with Arrow tables
   - Added error handling and debugging support

## Conceptual Framework: The Symbolic Body

The dashboard is not just monitoring accuracy, but visualizing **ethical morphogenesis**. Each visualization represents a limb, organ, or nervous plexus of a symbolic body.

### Symbolic Metaphors

- Prompt = electric cue
- Vote = epigenetic confirmation
- Calibration = perceptual myelination
- Retraining = tissue-level healing
- Drift = symbolic necrosis
- Dashboard = fMRI of a symbolic body

## Dashboard Architecture

### Directory Layout

```
interface/
  scripts/run_dashboard.py        # Main RLHF Pipeline Monitor
utils/dashboard/
  data_loader.py                  # Load + unify logs
  visualizations.py               # Plotly-based renderers
  themes.py                       # Prompt theme ontology
  utils.py                        # Shared formatting/helpers
models/
  meta_reflection_log.jsonl
  calibration_log.json
  drift_analysis/
    drift_analysis.json
    drift_clusters.jsonl
  vote_predictor_checkpoint_*/    # Model versioned outputs
  calibration_plots/
data/
  votes.jsonl
```

## Dashboard Tabs (Symbolic Organs)

### Tab 1: Attunement Timeline (Temporal Homeostasis)
| Visual                                            | Purpose                           |
| ------------------------------------------------- | --------------------------------- |
| Line plot: Model accuracy vs time/checkpoint      | Show symbolic healing or decay    |
| Line plot: ECE / Brier Score / Log Loss over time | Track confidence alignment        |
| Vertical bands for retraining/calibration events  | Show system-wide interventions    |
| Drift alert markers (from `drift_analysis.json`)  | Signal symbolic instability onset |

### Tab 2: Confidence & Calibration (Cognitive Thermoregulation)
| Visual                                      | Purpose                                                 |
| ------------------------------------------- | ------------------------------------------------------- |
| Reliability diagrams: pre vs post           | Visualize perception error before and after realignment |
| Line chart: ECE over calibration runs       | Trend towards stable alignment                          |
| Heatmap: Confidence × correctness per theme | Identify overconfident errors ("cancer" zones)          |

### Tab 3: Drift Clusters (Tissue Fragmentation)
| Visual                                                 | Purpose                                          |
| ------------------------------------------------------ | ------------------------------------------------ |
| UMAP / t-SNE of drift clusters (colored by error rate) | See disconnected symbolic tissue                 |
| Table: Cluster-wise stats                              | Size, drift delta, key prompts                   |
| Drilldown: Prompt–completion–vote–prediction view      | Inspect failed symbolic bonding                  |
| Line: Drift cluster entropy over time                  | Are symbolic subsystems converging or diverging? |

### Tab 4: Symbolic Anatomy: Model Evolution
| Visual                                                                  | Purpose                     |
| ----------------------------------------------------------------------- | --------------------------- |
| Table: All model checkpoints + metadata                                 | Show symbolic body versions |
| Delta plots: Accuracy, ECE, confidence distribution                     | Track morphogenetic shifts  |
| Compare view: Pick 2 checkpoints → plot attunement metrics side-by-side |                             |

### Tab 5: Human–System Entanglement (Preference Integration)
| Visual                                         | Purpose                                     |
| ---------------------------------------------- | ------------------------------------------- |
| Timeline: Human votes × confidence             | Signal human symbolic participation rate    |
| Wordcloud: Annotation content                  | Track ethical, aesthetic, or thematic drift |
| Time-binned shifts in vote alignment per theme | See evolving consensus or divergence        |

## Backend Modules

### `data_loader.py`
* Merge JSON/JSONL into Pandas DataFrames
* Timestamp normalization, alignment by checkpoint
* Join reflection logs to drift clusters and calibration logs

### `visualizations.py`
* `plot_alignment_timeline(df)`
* `render_reliability_diagram(pre, post)`
* `plot_drift_clusters_umap(df)`
* `render_preference_shift_streamgraph(df)`

### `themes.py` *(optional for PROFILE-style logic)*
* Define prompt theme taxonomies
* Enable symbolic filtering by theme

## Implementation Plan

### MVP First Milestone
Minimum viable Streamlit app should:
* Render Tabs 1–2 (Attunement Timeline + Calibration)
* Pull data from:
  * `meta_reflection_log.jsonl`
  * `calibration_log.json`
  * `votes.jsonl`
* Render:
  * Accuracy + confidence over time
  * Reliability diagram
  * Drift alerts

### Next Steps

1. **Complete Tab 1: Attunement Timeline**
   - Add calibration event markers
   - Implement drift alert visualization
   - Add ECE/Brier score tracking

2. **Implement Tab 2: Confidence & Calibration**
   - Create reliability diagram component
   - Implement pre/post calibration comparison
   - Add confidence/correctness heatmap by theme

3. **Implement Tab 3: Drift Clusters**
   - Create UMAP/t-SNE visualization component
   - Add cluster table visualization
   - Implement cluster drilldown functionality

4. **Implement Remaining Tabs**
   - Model evolution comparison
   - Human-system interaction visualizations

### Technical Tasks

1. Fix remaining timestamp conversion issues
2. Ensure proper data reloading after annotations
3. Implement more robust error handling
4. Create unified visualization module in utils/dashboard/visualizations.py

## Requirements

1. Streamlit for UI rendering
2. Pandas for data manipulation
3. Plotly for interactive visualizations
4. UMAP/t-SNE for dimensionality reduction in cluster visualization

## Completed Features

- Basic dashboard structure
- User preference timeline visualization
- Data loading and integration
- Time slider functionality
- Multi-tab organization
- Human-AI agreement visualization

## Outstanding Issues

- Timestamp conversion warning for Arrow table serialization
- FutureWarning about observed=False in pandas
- Minor UI layout improvements 