"""Test the full pipeline: loader -> preprocessor -> metrics.

This integration test requires network access to download datasets from HuggingFace.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Integration tests for the full analysis pipeline."""

    @pytest.fixture
    def loader(self):
        """Create a dataset loader for testing."""
        from src.data.loader import DatasetLoader
        return DatasetLoader(
            dataset_id="lerobot/pusht",
            cache_dir="./cache",
            streaming=False
        )

    def test_dataset_loading(self, loader):
        """Test that dataset can be loaded."""
        loader.load_dataset()
        assert loader.num_episodes > 0
        assert loader.num_samples > 0

    def test_episode_data_loading(self, loader):
        """Test loading episode data."""
        loader.load_dataset()
        episodes = loader.get_all_episodes_data(
            max_episodes=3,
            include_images=False,
            show_progress=False
        )

        assert len(episodes) == 3
        for ep in episodes:
            assert 'states' in ep
            assert 'episode_index' in ep
            assert ep['states'] is not None

    def test_feature_extraction(self, loader):
        """Test feature extraction from episodes."""
        from src.data.preprocessor import DataPreprocessor

        loader.load_dataset()
        episodes = loader.get_all_episodes_data(
            max_episodes=3,
            include_images=False,
            show_progress=False
        )

        preprocessor = DataPreprocessor()
        episodes_features = []

        for ep in episodes:
            features = preprocessor.extract_features(ep)
            episodes_features.append(features)

            # Verify feature structure
            assert 'positions' in features
            assert 'velocities' in features
            assert 'accelerations' in features
            assert 'jerk' in features

        assert len(episodes_features) == 3

    def test_motion_quality_analysis(self, loader):
        """Test motion quality metric computation."""
        from src.data.preprocessor import DataPreprocessor
        from src.metrics.motion_quality import MotionQualityAnalyzer

        loader.load_dataset()
        episodes = loader.get_all_episodes_data(
            max_episodes=3,
            include_images=False,
            show_progress=False
        )

        preprocessor = DataPreprocessor()
        episodes_features = [preprocessor.extract_features(ep) for ep in episodes]

        analyzer = MotionQualityAnalyzer()
        results = analyzer.analyze_multiple_episodes(
            episodes_features,
            show_progress=False
        )

        assert 'overall_score' in results
        assert 'n_episodes' in results
        assert results['n_episodes'] == 3
        assert 0 <= results['overall_score']['mean'] <= 1

    def test_spatial_coverage_analysis(self, loader):
        """Test spatial coverage analysis."""
        from src.data.preprocessor import DataPreprocessor
        from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
        from src.utils.robot_models import get_robot_config

        loader.load_dataset()
        episodes = loader.get_all_episodes_data(
            max_episodes=3,
            include_images=False,
            show_progress=False
        )

        preprocessor = DataPreprocessor()
        episodes_features = [preprocessor.extract_features(ep) for ep in episodes]

        robot_config = get_robot_config('generic_6dof')
        analyzer = SpatialCoverageAnalyzer(
            robot_config=robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes_features)

        assert 'overall_score' in results
        # Score might be a dict or float depending on implementation
        if isinstance(results['overall_score'], dict):
            assert 0 <= results['overall_score']['mean'] <= 1
        else:
            assert 0 <= results['overall_score'] <= 1


@pytest.mark.integration
class TestSyntheticPipeline:
    """Integration tests using synthetic data (no network required)."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline with synthetic data."""
        from src.data.preprocessor import DataPreprocessor
        from src.metrics.motion_quality import MotionQualityAnalyzer
        from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
        from src.metrics.data_consistency import DataConsistencyAnalyzer
        from src.utils.robot_models import get_robot_config

        # Create synthetic data
        episodes = self._create_synthetic_episodes(n_episodes=10, n_frames=50)

        # Extract features
        preprocessor = DataPreprocessor()
        episodes_features = [preprocessor.extract_features(ep) for ep in episodes]

        # Run motion quality analysis
        motion_analyzer = MotionQualityAnalyzer()
        motion_results = motion_analyzer.analyze_multiple_episodes(
            episodes_features,
            show_progress=False
        )

        assert 'overall_score' in motion_results
        assert 0 <= motion_results['overall_score']['mean'] <= 1

        # Run spatial coverage analysis
        robot_config = get_robot_config('generic_6dof')
        spatial_analyzer = SpatialCoverageAnalyzer(
            robot_config=robot_config,
            voxel_size=0.05
        )
        spatial_results = spatial_analyzer.analyze_spatial_coverage(episodes_features)

        assert 'overall_score' in spatial_results

        # Run data consistency analysis
        consistency_analyzer = DataConsistencyAnalyzer(robot_config=robot_config)
        consistency_results = consistency_analyzer.analyze_data_consistency(
            episodes,
            expected_fps=10.0
        )

        assert 'overall_score' in consistency_results

    def test_config_loading(self):
        """Test configuration management."""
        from src.utils.config import get_config

        config = get_config()

        assert config.hf_account is not None
        assert config.robot_type is not None
        assert config.robot_dof is not None

    def test_robot_models_loading(self):
        """Test robot model configuration loading."""
        from src.utils.robot_models import get_robot_config

        robot_config = get_robot_config('generic_6dof')

        assert robot_config.name is not None
        assert robot_config.dof == 6
        assert robot_config.workspace_bounds is not None

    def test_preprocessor_numerical_stability(self):
        """Test that preprocessing produces numerically stable results."""
        from src.data.preprocessor import DataPreprocessor

        episodes = self._create_synthetic_episodes(n_episodes=5, n_frames=50)
        preprocessor = DataPreprocessor()

        for ep in episodes:
            features = preprocessor.extract_features(ep, smooth_sigma=1.0)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(features['velocities']))
            assert not np.any(np.isnan(features['accelerations']))
            assert not np.any(np.isnan(features['jerk']))
            assert not np.any(np.isinf(features['velocities']))
            assert not np.any(np.isinf(features['accelerations']))
            assert not np.any(np.isinf(features['jerk']))

    def test_utility_functions(self):
        """Test metric utility functions."""
        from src.metrics.utils import (
            normalize_metric,
            detect_outliers_zscore,
            compute_path_efficiency
        )

        # Test normalization
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_metric(values, lower_is_better=True)
        assert np.all((normalized >= 0) & (normalized <= 1))

        # Test outlier detection
        np.random.seed(42)
        data = np.concatenate([np.random.randn(90), np.random.randn(10) * 10])
        outliers, z_scores = detect_outliers_zscore(data, threshold=3.0)
        assert np.sum(outliers) > 0

        # Test path efficiency
        trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        efficiency = compute_path_efficiency(trajectory)
        assert 0 <= efficiency <= 1

    def _create_synthetic_episodes(self, n_episodes=10, n_frames=50):
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
