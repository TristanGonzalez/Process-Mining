# Decision Mining
A Python-based process mining tool that discovers decision points in business processes and trains machine learning models to explain decision-making behavior from event logs.
## Overview
This project performs **decision mining** on event logs in XES format. It:
1. **Discovers a process model** (Petri net) from event logs using various algorithms (Alpha, Heuristic, or Inductive Miner)
2. **Identifies decision points** where the process branches into multiple paths
3. **Trains decision tree classifiers** for each decision point to explain which features influence the routing decisions
4. **Generates interpretable rules and visualizations** for analysis

## Project Structure

```
Process mining/
├── python/
│   ├── main.py                    # Main pipeline orchestrator
│   ├── data.py                    # XES log loading and DataFrame conversion
│   ├── petri.py                   # Process discovery and decision point detection
│   ├── decision_mining_ml.py      # ML-based decision mining
│   └── requirements.txt           # Python dependencies
├── data/
│   ├── xes/                       # Place your .xes event logs here
│   ├── decision_output/           # Generated CSV files with decision features
│   ├── petri_output/              # Generated Petri net visualizations
│   └── pkl/                       # Cached parsed logs (auto-generated)
├── models/                        # Decision tree models and results (auto-generated)
│   └── <LogName>/
│       ├── dt_default_*.png       # Default tree visualizations
│       ├── dt_pruned_*.png        # Pruned tree visualizations
│       ├── dt_default_*.txt       # Default tree rules (text)
│       ├── dt_pruned_*.txt        # Pruned tree rules (text)
│       └── eval_*.txt             # Evaluation metrics
└── config.yaml                    # Configuration file
```

## Installation
### 1. Install Requirements
Using the project's configured package manager (`condavenv`):

```
pip install -r python/requirements.txt
```

The main dependencies are:
-  `pm4py` - Process mining library
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib` - Visualization
- `pyyaml` - Configuration parsing
- `tqdm` - Parallel processing and progress bars `joblib`

## Data Setup
### 1. Place Your XES Event Logs
Copy your event log files into the directory: `.xes``data/xes/`

```
data/xes/
├── running-example.xes
├── Road_Traffic_Fine_Management_Process.xes
└── BPI Challenge 2017.xes
```

The project includes support for these sample datasets (see for sources). `data/description.txt`
## Configuration
### Configuring `config.yaml`
The file controls all aspects of the analysis: `config.yaml`
#### 1. **Log Path** (Required)

Specify which event log to process: yaml
``` yaml
log_path: "data/xes/BPI Challenge 2017.xes"
```



Uncomment one of the provided paths or add your own.
2. Process Discovery Method
Choose the algorithm for discovering the process model: yaml

``` yaml
process_model_type: "heuristic"  # Options: "alpha", "heuristic", "inductive"
```


alpha: Alpha Miner (works best on noise-free logs)
heuristic: Heuristics Miner (robust to noise, most commonly used)
inductive: Inductive Miner (guarantees sound models)
3. Heuristic Parameters (if using heuristic) yaml

``` yaml
heuristic:
  dependency_threshold: 0.5   # Min dependency frequency (0-1)
  and_threshold: 0.4          # Threshold for parallel patterns
  loop_two_threshold: 0.3     # Threshold for detecting loops
```

Lower thresholds = more lenient (include weaker dependencies)
Higher thresholds = stricter (only strong dependencies)

4. Inductive Parameters (if using inductive)

``` yaml
inductive:
  noise_threshold: 0.2          # Fraction of behavior to filter as noise (0-1)
  disable_fallthroughs: false   # Whether to disable fallthrough behavior
```

5. Decision Tree Parameters
Control the machine learning models:

``` yaml
decision_tree:
  # Default tree parameters
  criterion: "gini"              # Split criterion: "gini" or "entropy"
  max_depth: 10                  # Maximum tree depth
  min_samples_split: 2           # Min samples required to split
  min_samples_leaf: 1            # Min samples required at leaf
  max_features: "sqrt"           # Features to consider: "sqrt", "log2", or int
  random_state: 42               # Random seed for reproducibility
  class_weight: "balanced"       # Handle imbalanced classes
  ccp_alpha: 0.0                 # Complexity pruning parameter

  # Aggressive pruning parameters (for simpler, more readable trees)
  prune_depth: 3                 # Max depth for pruned tree
  prune_min_samples_leaf: 10     # Min samples per leaf for pruned tree
```

Two trees are trained per decision point:
Default tree: Uses the full parameter set for maximum accuracy
Pruned tree: Uses prune_depth and prune_min_samples_leaf for interpretability

## Running the Pipeline
Execute the Main Script from the project root directory:

``` bash
python python/main.py
```

Or from the python/ directory:

``` bash
cd python
python main.py
```
