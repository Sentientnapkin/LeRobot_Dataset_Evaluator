"""Tests for data consistency metrics."""

import pytest
import numpy as np

from src.metrics.data_consistency import DataConsistencyAnalyzer
from src.utils.robot_models import get_robot_config


class TestDataConsistencyAnalyzer:
    """Test data consistency metric computation."""

    @pytest.mark.unit
    def test_analyzer_initialization(self, generic_robot_config):
        """Test analyzer initialization."""
        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        assert analyzer.robot_config is not None

    @pytest.mark.unit
    def test_analyze_data_consistency(self, generic_robot_config):
        """Test data consistency analysis with synthetic data."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        # Verify results structure
        assert 'overall_score' in results
        assert 'individual_scores' in results
        assert 'anomalies' in results

        # Verify score bounds
        assert 0 <= results['overall_score'] <= 1

    @pytest.mark.unit
    def test_individual_scores_structure(self, generic_robot_config):
        """Test that all expected individual scores are present."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        individual_scores = results['individual_scores']

        # Check expected score keys
        expected_keys = ['alignment', 'temporal', 'missing_data', 'anomalies', 'joint_limits']
        for key in expected_keys:
            assert key in individual_scores, f"Missing key: {key}"
            assert 0 <= individual_scores[key] <= 1, f"Score {key} out of bounds"

    @pytest.mark.unit
    def test_anomaly_detection(self, generic_robot_config):
        """Test anomaly detection functionality."""
        # Create episodes with anomalies
        episodes = self._create_episodes_with_anomalies(n_episodes=10, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        anomalies = results['anomalies']
        assert 'n_anomalies' in anomalies

    @pytest.mark.unit
    @pytest.mark.parametrize("fps", [10.0, 30.0, 50.0])
    def test_different_fps_values(self, generic_robot_config, fps):
        """Test analysis with different expected FPS values."""
        episodes = self._create_test_episodes(n_episodes=5, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=fps)

        assert 'overall_score' in results
        assert 0 <= results['overall_score'] <= 1

    @pytest.mark.unit
    def test_clean_data_high_score(self, generic_robot_config):
        """Test that clean, consistent data receives high scores."""
        # Create very clean, consistent data
        episodes = self._create_clean_episodes(n_episodes=10, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        # Clean data should have high scores
        assert results['overall_score'] > 0.5

    @pytest.mark.unit
    def test_missing_data_detection(self, generic_robot_config):
        """Test missing data detection."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = DataConsistencyAnalyzer(robot_config=generic_robot_config)
        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        missing_data = results.get('missing_data', {})
        # Should have completeness metric
        assert 'completeness_percentage' in missing_data or results['individual_scores']['missing_data'] is not None

    def _create_test_episodes(self, n_episodes=10, n_frames=50):
        """Create synthetic test episodes."""
        np.random.seed(42)
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

    def _create_clean_episodes(self, n_episodes=10, n_frames=50):
        """Create very clean, consistent episodes."""
        episodes = []

        for i in range(n_episodes):
            t = np.linspace(0, 5, n_frames)
            positions = np.zeros((n_frames, 6))

            for j in range(6):
                # Very smooth sinusoidal motion
                freq = 0.5
                positions[:, j] = np.sin(2 * np.pi * freq * t / 5) * 0.5

            episodes.append({
                'states': positions,
                'actions': positions.copy(),
                'timestamps': t
            })

        return episodes

    def _create_episodes_with_anomalies(self, n_episodes=10, n_frames=50):
        """Create episodes with deliberate anomalies."""
        np.random.seed(123)
        episodes = []

        for i in range(n_episodes):
            t = np.linspace(0, 5, n_frames)
            positions = np.zeros((n_frames, 6))

            for j in range(6):
                freq = 0.5 + 0.1 * i
                positions[:, j] = np.sin(2 * np.pi * freq * t / 5) * (0.5 + 0.1 * i)

            # Add anomalies (sudden jumps)
            if i % 3 == 0:
                anomaly_idx = n_frames // 2
                positions[anomaly_idx:anomaly_idx + 3] += np.random.randn(3, 6) * 5

            episodes.append({
                'states': positions,
                'actions': positions.copy(),
                'timestamps': t
            })

        return episodes
