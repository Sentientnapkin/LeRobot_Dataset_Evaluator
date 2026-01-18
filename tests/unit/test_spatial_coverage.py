"""Tests for spatial coverage metrics."""

import pytest
import numpy as np

from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.utils.robot_models import get_robot_config


class TestSpatialCoverageAnalyzer:
    """Test spatial coverage metric computation."""

    @pytest.mark.unit
    def test_analyzer_initialization(self, generic_robot_config):
        """Test analyzer initialization."""
        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )
        assert analyzer.voxel_size == 0.05
        assert analyzer.robot_config is not None

    @pytest.mark.unit
    def test_analyze_spatial_coverage(self, generic_robot_config):
        """Test spatial coverage analysis with synthetic data."""
        # Create test episodes
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes)

        # Verify results structure
        assert 'overall_score' in results
        assert 'workspace_coverage' in results
        assert 'individual_scores' in results

        # Verify score bounds
        assert 0 <= results['overall_score'] <= 1

    @pytest.mark.unit
    def test_workspace_coverage_metrics(self, generic_robot_config):
        """Test workspace coverage computation."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes)
        coverage = results['workspace_coverage']

        # Verify coverage metrics
        assert 'coverage_percentage' in coverage
        assert 'voxel_grid' in coverage
        assert 'voxel_edges' in coverage
        assert coverage['coverage_percentage'] >= 0

    @pytest.mark.unit
    def test_individual_scores_structure(self, generic_robot_config):
        """Test that all expected individual scores are present."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)

        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes)
        individual_scores = results['individual_scores']

        # Check expected score keys
        assert 'starting_variance' in individual_scores
        assert 'position_distribution' in individual_scores
        assert 'angle_diversity' in individual_scores

        # All scores should be in valid range
        for key, score in individual_scores.items():
            assert 0 <= score <= 1, f"Score {key} out of bounds: {score}"

    @pytest.mark.unit
    @pytest.mark.parametrize("voxel_size", [0.01, 0.05, 0.1, 0.2])
    def test_different_voxel_sizes(self, generic_robot_config, voxel_size):
        """Test analysis with different voxel sizes."""
        episodes = self._create_test_episodes(n_episodes=5, n_frames=30)

        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=voxel_size
        )

        results = analyzer.analyze_spatial_coverage(episodes)

        assert 'overall_score' in results
        assert 0 <= results['overall_score'] <= 1

    @pytest.mark.unit
    def test_empty_episodes_handling(self, generic_robot_config):
        """Test handling of empty episode list."""
        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )

        # Empty list should either return empty results or raise an error gracefully
        try:
            results = analyzer.analyze_spatial_coverage([])
            # If it doesn't raise, verify it returns sensible defaults
            assert 'overall_score' in results
        except (ValueError, IndexError):
            # Acceptable to raise an error for empty input
            pass

    @pytest.mark.unit
    def test_single_episode(self, generic_robot_config):
        """Test analysis with a single episode."""
        episodes = self._create_test_episodes(n_episodes=1, n_frames=50)

        analyzer = SpatialCoverageAnalyzer(
            robot_config=generic_robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes)

        assert 'overall_score' in results
        assert 0 <= results['overall_score'] <= 1

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
