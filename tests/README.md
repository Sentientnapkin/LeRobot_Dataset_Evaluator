# Tests Directory

This directory contains all tests for the LeRobot Dataset Evaluator.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_motion_quality.py        # Motion quality metrics
│   ├── test_smoothness_debug.py      # Smoothness metric debugging
│   └── test_joint_space_fix.py       # Joint space detection
│
├── integration/             # Integration tests for full workflows
│   ├── test_basic.py                 # Basic integration test
│   ├── test_full_pipeline.py         # Full data pipeline
│   ├── test_streamlit_compatibility.py  # Streamlit app compatibility
│   └── test_comprehensive.py         # Comprehensive dataset comparison
│
├── data/                    # Data loading and API tests
│   ├── test_lerobot_api.py           # LeRobot API structure
│   ├── test_episode_structure.py     # Episode data structure
│   ├── test_fixed_loader.py          # Fixed data loader
│   └── test_user_dataset.py          # User dataset analysis
│
└── test_spatial_coverage_and_data_consistency.py  # Legacy test
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Data tests only
pytest tests/data/
```

### Run Specific Test File
```bash
pytest tests/unit/test_motion_quality.py
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Descriptions

### Unit Tests

**test_motion_quality.py**
- Tests individual motion quality metrics
- Validates jerk, smoothness, acceleration calculations
- Uses synthetic test data

**test_smoothness_debug.py**
- Debugs SPARC smoothness metric
- Tests with known smooth/jerky trajectories
- Validates normalization

**test_joint_space_fix.py**
- Tests joint space vs Cartesian space detection
- Validates workspace coverage in joint space
- Tests adaptive binning

### Integration Tests

**test_basic.py**
- Basic integration test without pytest
- Tests data loading → preprocessing → metrics
- Validates end-to-end pipeline

**test_full_pipeline.py**
- Complete pipeline test with real dataset
- Tests loader → preprocessor → all metrics
- Validates metric outputs

**test_streamlit_compatibility.py**
- Tests Streamlit app without running Streamlit
- Validates all imports and workflows
- Simulates dashboard user flow

**test_comprehensive.py**
- Compares multiple datasets
- Tests user dataset vs reference datasets
- Generates comparison reports

### Data Tests

**test_lerobot_api.py**
- Explores actual LeRobot API structure
- Validates available attributes
- Documents API differences

**test_episode_structure.py**
- Tests episode filtering and indexing
- Validates HuggingFace dataset structure
- Tests episode_index column usage

**test_fixed_loader.py**
- Tests fixed data loader implementation
- Validates num_episodes property
- Tests get_episode_data method

**test_user_dataset.py**
- Analyzes user's specific dataset
- Tests with SentientNapkin/stack-center-ACT-dual-cam
- Debugs data quality issues

## Test Data

Tests use a combination of:
- **Synthetic data**: Generated test trajectories with known properties
- **Real datasets**: Public LeRobot datasets (lerobot/pusht, berkeley_autolab_ur5)
- **User datasets**: SentientNapkin/stack-center-ACT-dual-cam

## Adding New Tests

### Unit Test Template
```python
"""Test for [component name]."""

import numpy as np
import pytest
from src.metrics.your_metric import YourMetricAnalyzer

def test_metric_basic():
    """Test basic metric calculation."""
    analyzer = YourMetricAnalyzer()

    # Create test data
    data = np.array([...])

    # Run metric
    result = analyzer.compute_metric(data)

    # Assert expected behavior
    assert result['score'] > 0
    assert result['score'] <= 1.0

def test_metric_edge_cases():
    """Test edge cases."""
    analyzer = YourMetricAnalyzer()

    # Test with empty data
    with pytest.raises(ValueError):
        analyzer.compute_metric(np.array([]))
```

### Integration Test Template
```python
"""Integration test for [workflow name]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DatasetLoader
from src.metrics.your_metric import YourMetricAnalyzer

def test_workflow():
    """Test complete workflow."""
    # Load data
    loader = DatasetLoader("test/dataset")
    dataset = loader.load_dataset()

    # Run analysis
    analyzer = YourMetricAnalyzer()
    results = analyzer.analyze(dataset)

    # Validate results
    assert 'overall_score' in results
    assert results['overall_score'] > 0
```

## Continuous Integration

When CI is set up, tests will run automatically on:
- Every push to main branch
- Every pull request
- Nightly builds

## Test Coverage Goals

- **Unit tests**: > 80% coverage
- **Integration tests**: All major workflows
- **Data tests**: All LeRobot API interactions

## Troubleshooting

### Import Errors
If you get import errors, make sure to:
1. Install all dependencies: `pip install -r requirements.txt`
2. Run from project root: `cd /path/to/LeRobot_Dataset_Evaluator`
3. Use proper Python path: `PYTHONPATH=. pytest tests/`

### Dataset Download Issues
Some tests require downloading datasets from HuggingFace:
- Ensure internet connection
- Authenticate if needed: `huggingface-cli login`
- Tests cache downloads in `./cache/`

### Slow Tests
Integration tests with real datasets can be slow:
- Use `pytest -x` to stop on first failure
- Use `pytest -k test_name` to run specific tests
- Use `--maxfail=1` to stop after one failure

## Contributing

When adding new features:
1. Write unit tests first (TDD)
2. Add integration tests for workflows
3. Update this README
4. Ensure all tests pass before committing

---

**Last Updated**: 2026-01-15
**Test Framework**: pytest 8.0+
**Coverage**: pytest-cov
