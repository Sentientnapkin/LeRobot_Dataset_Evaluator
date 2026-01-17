<p align="center">
  <h1 align="center">LeRobot Dataset Evaluator</h1>
</p>

<p align="center">
  <a href="https://github.com/Sentientnapkin/LeRobot_Dataset_Evaluator/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/built%20with-Streamlit-ff4b4b.svg"></a>
  <a href="https://huggingface.co/docs/hub/index"><img alt="HuggingFace" src="https://img.shields.io/badge/datasets-HuggingFace-yellow.svg"></a>
</p>

<p align="center">
  An interactive dashboard for evaluating LeRobot datasets before training ACT models.
  <br>
  Save time and compute by identifying data quality issues early.
</p>

---

## Overview

Training robotics models can take hours or even days. This tool helps you evaluate your dataset quality **before** training, allowing you to identify data collection issues, understand spatial coverage, analyze motion quality, and get actionable recommendations for improvement.

## Features

| Category | Metrics |
|----------|---------|
| **Spatial Coverage** | Workspace coverage, starting position variance, approach angle diversity, position distribution |
| **Motion Quality** | Jerk metrics, smoothness analysis (SPARC), acceleration profiles, movement efficiency |
| **Data Consistency** | Action-observation alignment, temporal consistency, anomaly detection, missing data analysis |
| **Success Indicators** | Episode length distributions, trajectory similarity, task completion patterns |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Sentientnapkin/LeRobot_Dataset_Evaluator.git
cd LeRobot_Dataset_Evaluator

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Supported Robots

| Robot | DOF | Description |
|-------|-----|-------------|
| `ur5` | 6 | Universal Robots UR5 collaborative arm |
| `panda` | 7 | Franka Emika Panda collaborative arm |
| `so-arm100` | 6 | SO-ARM100 6-DOF robotic arm |
| `generic_6dof` | 6 | Generic 6-DOF robot with standard limits |

Each robot configuration includes accurate Denavit-Hartenberg (DH) parameters for proper forward kinematics computation.

## Configuration

Edit `config.yaml` to customize settings:

```yaml
dataset:
  hf_account: "your-account"    # HuggingFace account name
  cache_dir: "./cache"          # Local cache directory
  streaming: true               # Use streaming mode for large datasets

robot:
  type: "generic_6dof"          # Robot type (ur5, panda, so-arm100, generic_6dof)
  dof: 6                        # Degrees of freedom

metrics:
  spatial_coverage:
    voxel_size: 0.05            # Workspace discretization (meters)
    min_coverage_threshold: 0.4 # Minimum acceptable coverage

  motion_quality:
    jerk_threshold_multiplier: 2.0  # Flag trajectories > 2 sigma
    smoothness_method: "sparc"      # Smoothness metric method

quality_scoring:
  weights:
    motion_quality: 0.40        # Highest weight (most critical for ACT)
    spatial_coverage: 0.30
    data_consistency: 0.20
    success_indicators: 0.10
```

## Understanding the Metrics

### Motion Quality (Critical for ACT)

Research shows that jerk and smoothness are the strongest predictors of ACT training success. Poor motion quality leads to training instability, poor generalization, and suboptimal learned behaviors.

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Jerk** | Rate of change of acceleration | Lower = smoother, more human-like |
| **SPARC** | Spectral Arc Length smoothness | Lower = smoother trajectories |
| **Movement Units** | Purposeful movement segments | Higher = more skilled teleoperation |
| **Hesitations** | Velocity reversals per second | Lower = more confident control |

### Spatial Coverage

Diverse training data improves generalization. The tool now uses **reachable workspace coverage** which compares visited voxels against the robot's actual reachable space (not just the bounding box).

| Coverage Level | Threshold | Interpretation |
|----------------|-----------|----------------|
| Excellent | >= 50% | Comprehensive workspace exploration |
| Good | >= 30% | Solid coverage for most tasks |
| Moderate | >= 15% | Focused on specific regions |
| Limited | < 15% | Consider expanding exploration |

### Data Consistency

Ensures training reliability through temporal alignment checks, frame rate consistency, and anomaly detection.

## Project Structure

```
LeRobot_Dataset_Evaluator/
├── app.py                      # Main Streamlit application
├── config.yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md
│
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── loader.py
│   │   └── preprocessor.py
│   │
│   ├── metrics/                # Metric computation modules
│   │   ├── spatial_coverage.py
│   │   ├── execution_quality.py
│   │   └── data_consistency.py
│   │
│   ├── visualizations/         # Plotly visualization components
│   │   ├── spatial_viz.py
│   │   ├── robot_arm_3d.py
│   │   └── execution_quality_viz.py
│   │
│   ├── ui/                     # UI components and styling
│   │   ├── styles.py
│   │   └── components.py
│   │
│   └── utils/                  # Utility functions
│       ├── config.py
│       ├── robot_models.py
│       └── robot_kinematics.py
│
└── tests/                      # Unit and integration tests
```

## Contributing

Contributions are welcome. The codebase is organized for extensibility:

1. **Adding New Metrics**: Create a module in `src/metrics/`
2. **Custom Visualizations**: Add functions to `src/visualizations/`
3. **Robot Support**: Add DH parameters to `src/utils/robot_kinematics.py` and configuration to `src/utils/robot_models.py`

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{lerobot_dataset_evaluator,
  author = {SentientNapkin},
  title = {LeRobot Dataset Evaluator},
  year = {2024},
  url = {https://github.com/Sentientnapkin/LeRobot_Dataset_Evaluator}
}
```

## Acknowledgments

- Built on [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace
- Powered by [Streamlit](https://streamlit.io/) for the interactive dashboard
- Visualizations with [Plotly](https://plotly.com/)
- Data processing with [Polars](https://www.pola.rs/)

---

<p align="center">
  <sub>Built for robotics researchers</sub>
</p>
