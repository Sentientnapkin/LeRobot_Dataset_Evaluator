"""Tests for motion quality metrics."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.motion_quality import MotionQualityAnalyzer
from src.metrics.utils import compute_statistics, detect_outliers_zscore
from src.data.preprocessor import DataPreprocessor


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

    # Create smooth sinusoidal trajectory (minimum jerk-like)
    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        # Sinusoidal motion with different frequencies per joint
        freq = 0.5 + 0.2 * i
        positions[:, i] = np.sin(2 * np.pi * freq * t / 10) + noise_level * np.random.randn(n_frames)

    return {
        'states': positions,
        'actions': positions.copy(),  # For simplicity
        'timestamps': t
    }


def create_jerky_trajectory(n_frames=100, n_dof=6):
    """Create jerky trajectory with sudden changes.

    Args:
        n_frames: Number of frames
        n_dof: Degrees of freedom

    Returns:
        Dictionary with positions and timestamps
    """
    t = np.linspace(0, 10, n_frames)

    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        # Create trajectory with sudden jumps
        base_motion = np.sin(2 * np.pi * 0.5 * t / 10)

        # Add random jumps
        jumps = np.zeros(n_frames)
        jump_indices = np.random.choice(n_frames, size=10, replace=False)
        jumps[jump_indices] = np.random.randn(10) * 0.5

        positions[:, i] = base_motion + jumps

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t
    }


class TestDataPreprocessor:
    """Test data preprocessing functionality."""

    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.robot_config is None

    def test_compute_velocities(self):
        """Test velocity computation."""
        preprocessor = DataPreprocessor()

        # Create simple trajectory
        trajectory = create_synthetic_trajectory(n_frames=50)
        positions = trajectory['states']
        timestamps = trajectory['timestamps']

        velocities = preprocessor.compute_velocities(positions, timestamps)

        assert velocities.shape == positions.shape
        assert not np.any(np.isnan(velocities))
        assert not np.any(np.isinf(velocities))

    def test_compute_accelerations(self):
        """Test acceleration computation."""
        preprocessor = DataPreprocessor()

        trajectory = create_synthetic_trajectory(n_frames=50)
        positions = trajectory['states']
        timestamps = trajectory['timestamps']

        velocities = preprocessor.compute_velocities(positions, timestamps)
        accelerations = preprocessor.compute_accelerations(velocities, timestamps)

        assert accelerations.shape == positions.shape
        assert not np.any(np.isnan(accelerations))

    def test_compute_jerk(self):
        """Test jerk computation."""
        preprocessor = DataPreprocessor()

        trajectory = create_synthetic_trajectory(n_frames=50)
        positions = trajectory['states']
        timestamps = trajectory['timestamps']

        velocities = preprocessor.compute_velocities(positions, timestamps)
        accelerations = preprocessor.compute_accelerations(velocities, timestamps)
        jerk = preprocessor.compute_jerk(accelerations, timestamps)

        assert jerk.shape == positions.shape
        assert not np.any(np.isnan(jerk))

    def test_extract_features(self):
        """Test feature extraction."""
        preprocessor = DataPreprocessor()

        episode = create_synthetic_trajectory(n_frames=50)
        features = preprocessor.extract_features(episode)

        assert 'positions' in features
        assert 'velocities' in features
        assert 'accelerations' in features
        assert 'jerk' in features
        assert 'timestamps' in features

        # All should have same shape
        assert features['velocities'].shape == features['positions'].shape
        assert features['accelerations'].shape == features['positions'].shape
        assert features['jerk'].shape == features['positions'].shape


class TestMotionQualityMetrics:
    """Test motion quality metric computation."""

    def test_analyzer_init(self):
        """Test analyzer initialization."""
        analyzer = MotionQualityAnalyzer()
        assert analyzer.jerk_threshold_multiplier == 2.0
        assert analyzer.smoothness_method == "sparc"

    def test_jerk_metrics_smooth_trajectory(self):
        """Test jerk metrics on smooth trajectory."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        # Create smooth trajectory
        episode = create_synthetic_trajectory(n_frames=100, noise_level=0.001)
        features = preprocessor.extract_features(episode, smooth_sigma=1.0)

        result = analyzer.analyze_episode(features)

        # Smooth trajectory should have high jerk score
        assert result['individual_scores']['jerk'] > 0.5
        assert result['overall_score'] > 0.5

        # Should have low outlier count
        assert result['jerk']['outlier_percentage'] < 20

    def test_jerk_metrics_jerky_trajectory(self):
        """Test jerk metrics on jerky trajectory."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        # Create jerky trajectory
        episode = create_jerky_trajectory(n_frames=100)
        features = preprocessor.extract_features(episode, smooth_sigma=1.0)

        result = analyzer.analyze_episode(features)

        # Jerky trajectory should have lower jerk score
        # (though smoothing helps)
        assert 'jerk' in result['individual_scores']
        assert result['jerk']['outlier_percentage'] >= 0

    def test_smoothness_metrics(self):
        """Test smoothness metric computation."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        episode = create_synthetic_trajectory(n_frames=100)
        features = preprocessor.extract_features(episode)

        smoothness = analyzer.compute_smoothness_metrics(
            features['positions'],
            features['velocities'],
            features['jerk'],
            features['timestamps']
        )

        assert 'sparc' in smoothness
        assert 'ldlj' in smoothness
        assert 'score' in smoothness

        # Score should be between 0 and 1
        assert 0 <= smoothness['score'] <= 1

    def test_acceleration_metrics(self):
        """Test acceleration metrics."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        episode = create_synthetic_trajectory(n_frames=100)
        features = preprocessor.extract_features(episode)

        acc_metrics = analyzer.compute_acceleration_metrics(
            features['accelerations'],
            features['timestamps']
        )

        assert 'rms_acceleration' in acc_metrics
        assert 'discontinuity_count' in acc_metrics
        assert 'score' in acc_metrics

        assert 0 <= acc_metrics['score'] <= 1

    def test_velocity_metrics(self):
        """Test velocity metrics."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        episode = create_synthetic_trajectory(n_frames=100)
        features = preprocessor.extract_features(episode)

        vel_metrics = analyzer.compute_velocity_metrics(
            features['velocities'],
            features['positions'],
            features['timestamps']
        )

        assert 'rms_velocity' in vel_metrics
        assert 'peak_velocity' in vel_metrics
        assert 'score' in vel_metrics

        assert vel_metrics['peak_velocity'] > 0
        assert 0 <= vel_metrics['score'] <= 1

    def test_multiple_episodes(self):
        """Test analyzing multiple episodes."""
        analyzer = MotionQualityAnalyzer()
        preprocessor = DataPreprocessor()

        # Create multiple episodes
        episodes_features = []
        for i in range(5):
            episode = create_synthetic_trajectory(n_frames=50, noise_level=0.01 * (i + 1))
            features = preprocessor.extract_features(episode)
            episodes_features.append(features)

        results = analyzer.analyze_multiple_episodes(episodes_features, show_progress=False)

        assert 'overall_score' in results
        assert 'n_episodes' in results
        assert results['n_episodes'] == 5

        assert 'mean' in results['overall_score']
        assert 'std' in results['overall_score']

        # Mean score should be reasonable
        assert 0 <= results['overall_score']['mean'] <= 1


class TestUtilities:
    """Test utility functions."""

    def test_compute_statistics(self):
        """Test statistics computation."""
        data = np.random.randn(100)
        stats = compute_statistics(data)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

        assert stats['min'] < stats['mean'] < stats['max']

    def test_detect_outliers_zscore(self):
        """Test z-score outlier detection."""
        # Create data with outliers
        data = np.concatenate([
            np.random.randn(90),  # Normal data
            np.random.randn(10) * 10  # Outliers
        ])

        outliers, z_scores = detect_outliers_zscore(data, threshold=3.0)

        assert len(outliers) == len(data)
        assert len(z_scores) == len(data)
        assert np.sum(outliers) > 0  # Should detect some outliers


class TestVisualizationsMotionQuality:
    """Test motion quality visualization generation."""

    @pytest.mark.unit
    def test_visualizations_generation(self):
        """Test that visualizations can be generated without error."""
        analyzer = MotionQualityAnalyzer(
            jerk_threshold_multiplier=2.0,
            smoothness_method="sparc"
        )
        preprocessor = DataPreprocessor()

        # Create synthetic dataset
        episodes_features = []
        for i in range(5):
            episode = create_synthetic_trajectory(n_frames=50, noise_level=0.01 * (i + 1))
            features = preprocessor.extract_features(episode, smooth_sigma=1.0)
            episodes_features.append(features)

        results = analyzer.analyze_multiple_episodes(episodes_features, show_progress=False)

        from src.visualizations.motion_viz import (
            plot_overall_scores_distribution,
            plot_metric_comparison_bars,
            plot_jerk_outliers,
            plot_smoothness_comparison
        )

        # Just verify plots can be created without error
        fig1 = plot_overall_scores_distribution(results['overall_score']['scores'])
        assert fig1 is not None

        fig2 = plot_metric_comparison_bars(results)
        assert fig2 is not None

        fig3 = plot_jerk_outliers(results['episode_results'])
        assert fig3 is not None

        fig4 = plot_smoothness_comparison(results['episode_results'])
        assert fig4 is not None
