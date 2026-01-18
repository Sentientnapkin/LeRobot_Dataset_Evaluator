"""Shared pytest fixtures for LeRobot Dataset Evaluator tests."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================

@pytest.fixture
def sample_states():
    """Generate sample state data (100 frames, 6 DOF)."""
    np.random.seed(42)
    return np.random.randn(100, 6)


@pytest.fixture
def sample_actions():
    """Generate sample action data (100 frames, 6 DOF)."""
    np.random.seed(42)
    return np.random.randn(100, 6)


@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps (100 frames at 10 Hz)."""
    return np.linspace(0, 10, 100)


@pytest.fixture
def sample_episode_data(sample_states, sample_actions, sample_timestamps):
    """Create a complete episode data dictionary."""
    return {
        'episode_index': 0,
        'states': sample_states,
        'actions': sample_actions,
        'timestamps': sample_timestamps,
        'length': 100,
    }


@pytest.fixture
def sample_episodes_list(sample_episode_data):
    """Create a list of 5 sample episodes."""
    episodes = []
    for i in range(5):
        ep = sample_episode_data.copy()
        ep['episode_index'] = i
        # Add slight variation to each episode
        ep['states'] = ep['states'] + np.random.randn(*ep['states'].shape) * 0.1
        ep['actions'] = ep['actions'] + np.random.randn(*ep['actions'].shape) * 0.1
        episodes.append(ep)
    return episodes


# ============================================================================
# Synthetic Trajectory Fixtures
# ============================================================================

@pytest.fixture
def smooth_trajectory():
    """Create a smooth sinusoidal trajectory for testing."""
    n_frames = 100
    n_dof = 6
    t = np.linspace(0, 10, n_frames)

    positions = np.zeros((n_frames, n_dof))
    for i in range(n_dof):
        freq = 0.5 + 0.2 * i
        positions[:, i] = np.sin(2 * np.pi * freq * t / 10)

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t,
    }


@pytest.fixture
def jerky_trajectory():
    """Create a jerky trajectory with sudden changes for testing."""
    n_frames = 100
    n_dof = 6
    t = np.linspace(0, 10, n_frames)

    np.random.seed(123)
    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        base_motion = np.sin(2 * np.pi * 0.5 * t / 10)
        jumps = np.zeros(n_frames)
        jump_indices = np.random.choice(n_frames, size=10, replace=False)
        jumps[jump_indices] = np.random.randn(10) * 0.5
        positions[:, i] = base_motion + jumps

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t,
    }


@pytest.fixture
def noisy_trajectory():
    """Create a noisy trajectory for testing filtering/smoothing."""
    n_frames = 100
    n_dof = 6
    t = np.linspace(0, 10, n_frames)

    np.random.seed(456)
    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        freq = 0.5 + 0.1 * i
        positions[:, i] = (np.sin(2 * np.pi * freq * t / 10) +
                          np.random.randn(n_frames) * 0.1)

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t,
    }


# ============================================================================
# Feature Fixtures (for metrics tests)
# ============================================================================

@pytest.fixture
def sample_features(smooth_trajectory):
    """Create preprocessed features from a smooth trajectory."""
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    return preprocessor.extract_features(smooth_trajectory)


@pytest.fixture
def multiple_episode_features(sample_episodes_list):
    """Create preprocessed features for multiple episodes."""
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    features_list = []
    for episode in sample_episodes_list:
        features = preprocessor.extract_features(episode)
        features_list.append(features)
    return features_list


# ============================================================================
# Robot Configuration Fixtures
# ============================================================================

@pytest.fixture
def generic_robot_config():
    """Get generic 6-DOF robot configuration."""
    from src.utils.robot_models import get_robot_config
    return get_robot_config('generic_6dof')


# ============================================================================
# Analyzer Fixtures
# ============================================================================

@pytest.fixture
def motion_analyzer():
    """Create a MotionQualityAnalyzer instance."""
    from src.metrics.motion_quality import MotionQualityAnalyzer
    return MotionQualityAnalyzer()


@pytest.fixture
def spatial_analyzer(generic_robot_config):
    """Create a SpatialCoverageAnalyzer instance."""
    from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
    return SpatialCoverageAnalyzer(
        robot_config=generic_robot_config,
        voxel_size=0.05
    )


@pytest.fixture
def consistency_analyzer(generic_robot_config):
    """Create a DataConsistencyAnalyzer instance."""
    from src.metrics.data_consistency import DataConsistencyAnalyzer
    return DataConsistencyAnalyzer(robot_config=generic_robot_config)


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    from src.data.preprocessor import DataPreprocessor
    return DataPreprocessor()


# ============================================================================
# Utility Functions (available in tests via conftest)
# ============================================================================

def create_synthetic_trajectory(n_frames=100, n_dof=6, noise_level=0.01):
    """Create synthetic smooth trajectory for testing.

    Args:
        n_frames: Number of frames
        n_dof: Degrees of freedom
        noise_level: Amount of noise to add

    Returns:
        Dictionary with positions and timestamps
    """
    t = np.linspace(0, 10, n_frames)
    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        freq = 0.5 + 0.2 * i
        positions[:, i] = np.sin(2 * np.pi * freq * t / 10) + noise_level * np.random.randn(n_frames)

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t
    }


def create_test_episodes(n_episodes=10, n_frames=50):
    """Create synthetic test episodes.

    Args:
        n_episodes: Number of episodes to create
        n_frames: Number of frames per episode

    Returns:
        List of episode dictionaries
    """
    episodes = []

    for i in range(n_episodes):
        t = np.linspace(0, 5, n_frames)
        positions = np.zeros((n_frames, 6))

        for j in range(6):
            freq = 0.5 + 0.1 * i
            positions[:, j] = np.sin(2 * np.pi * freq * t / 5) * (0.5 + 0.1 * i)

        positions += np.random.randn(*positions.shape) * 0.01

        episodes.append({
            'states': positions,
            'actions': positions.copy(),
            'timestamps': t
        })

    return episodes
