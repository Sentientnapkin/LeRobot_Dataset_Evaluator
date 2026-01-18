"""Tests for success indicators analysis."""

import pytest
import numpy as np

from src.metrics.success_indicators import SuccessIndicatorAnalyzer, EpisodeSuccessInfo


class TestSuccessIndicatorAnalyzer:
    """Test success indicator computation."""

    @pytest.mark.unit
    def test_analyzer_initialization(self):
        """Test analyzer initialization with default thresholds."""
        analyzer = SuccessIndicatorAnalyzer()

        assert analyzer.success_threshold == 0.6
        assert analyzer.excellent_threshold == 0.75
        assert analyzer.marginal_threshold == 0.45

    @pytest.mark.unit
    def test_analyzer_custom_thresholds(self):
        """Test analyzer initialization with custom thresholds."""
        analyzer = SuccessIndicatorAnalyzer(
            success_threshold=0.7,
            excellent_threshold=0.85,
            marginal_threshold=0.5
        )

        assert analyzer.success_threshold == 0.7
        assert analyzer.excellent_threshold == 0.85
        assert analyzer.marginal_threshold == 0.5

    @pytest.mark.unit
    def test_analyze_success_indicators(self):
        """Test basic success indicator analysis."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)
        motion_results = self._create_mock_motion_results(n_episodes=10)

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        # Verify results structure
        assert 'n_episodes' in results
        assert results['n_episodes'] == 10
        assert 'length_stats' in results
        assert 'quality_stats' in results
        assert 'episode_info' in results
        assert 'success_rates' in results

    @pytest.mark.unit
    def test_length_statistics(self):
        """Test episode length statistics computation."""
        episodes = self._create_variable_length_episodes()

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(episodes)

        length_stats = results['length_stats']

        assert 'mean' in length_stats
        assert 'std' in length_stats
        assert 'min' in length_stats
        assert 'max' in length_stats
        assert 'median' in length_stats

        # Basic sanity checks
        assert length_stats['min'] <= length_stats['mean'] <= length_stats['max']
        assert length_stats['min'] <= length_stats['median'] <= length_stats['max']

    @pytest.mark.unit
    def test_quality_statistics(self):
        """Test quality score statistics computation."""
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)
        motion_results = self._create_mock_motion_results(n_episodes=10)

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        quality_stats = results['quality_stats']

        assert 'mean' in quality_stats
        assert 'std' in quality_stats
        assert 'min' in quality_stats
        assert 'max' in quality_stats
        assert 'scores' in quality_stats

        # Scores should be in valid range
        assert 0 <= quality_stats['min'] <= 1
        assert 0 <= quality_stats['max'] <= 1

    @pytest.mark.unit
    @pytest.mark.parametrize("classification,min_score,max_score", [
        ("excellent", 0.75, 1.0),
        ("good", 0.6, 0.75),
        ("marginal", 0.45, 0.6),
        ("poor", 0.0, 0.45),
    ])
    def test_episode_classification(self, classification, min_score, max_score):
        """Test episode classification by quality score."""
        analyzer = SuccessIndicatorAnalyzer()

        # Create episode with score in the middle of the range
        score = (min_score + max_score) / 2
        if classification == "excellent":
            score = 0.85  # Ensure it's in the excellent range
        elif classification == "good":
            score = 0.65
        elif classification == "marginal":
            score = 0.5
        else:
            score = 0.3

        episodes = [{'states': np.random.randn(50, 6)}]
        motion_results = {
            'episode_results': [{'overall_score': score}]
        }

        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        episode_info = results['episode_info'][0]
        assert episode_info.classification == classification

    @pytest.mark.unit
    def test_success_rates_calculation(self):
        """Test success rate calculations."""
        # Create episodes with mixed quality
        episodes = self._create_test_episodes(n_episodes=10, n_frames=50)
        motion_results = self._create_mock_motion_results(n_episodes=10)

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        success_rates = results['success_rates']

        assert 'overall' in success_rates
        assert 'overall_percentage' in success_rates
        assert 'n_successful' in success_rates
        assert 'n_total' in success_rates
        assert 'counts' in success_rates

        # Rates should be valid
        assert 0 <= success_rates['overall'] <= 1
        assert 0 <= success_rates['overall_percentage'] <= 100
        assert success_rates['n_total'] == 10

    @pytest.mark.unit
    def test_best_worst_episodes(self):
        """Test identification of best and worst episodes."""
        episodes = self._create_test_episodes(n_episodes=20, n_frames=50)
        motion_results = self._create_mock_motion_results(n_episodes=20)

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        best = results['best_episodes']
        worst = results['worst_episodes']

        # Should have 5 each by default
        assert len(best) <= 5
        assert len(worst) <= 5

        # Best should have higher scores than worst
        if best and worst:
            assert best[0]['quality_score'] >= worst[-1]['quality_score']

    @pytest.mark.unit
    def test_length_quality_correlation(self):
        """Test length-quality correlation analysis."""
        episodes = self._create_variable_length_episodes()
        motion_results = self._create_mock_motion_results(n_episodes=len(episodes))

        analyzer = SuccessIndicatorAnalyzer()
        results = analyzer.analyze_success_indicators(
            episodes,
            motion_quality_results=motion_results
        )

        correlation = results['length_quality_correlation']

        assert 'coefficient' in correlation
        assert 'strength' in correlation
        assert 'direction' in correlation
        assert 'interpretation' in correlation

        # Coefficient should be in valid range
        assert -1 <= correlation['coefficient'] <= 1
        assert correlation['strength'] in ['weak', 'moderate', 'strong']
        assert correlation['direction'] in ['positive', 'negative']

    @pytest.mark.unit
    def test_thresholds_in_results(self):
        """Test that thresholds are included in results."""
        episodes = self._create_test_episodes(n_episodes=5, n_frames=50)

        analyzer = SuccessIndicatorAnalyzer(
            success_threshold=0.65,
            excellent_threshold=0.8,
            marginal_threshold=0.4
        )
        results = analyzer.analyze_success_indicators(episodes)

        thresholds = results['thresholds']

        assert thresholds['success'] == 0.65
        assert thresholds['excellent'] == 0.8
        assert thresholds['marginal'] == 0.4

    def _create_test_episodes(self, n_episodes=10, n_frames=50):
        """Create synthetic test episodes."""
        np.random.seed(42)
        episodes = []

        for i in range(n_episodes):
            t = np.linspace(0, 5, n_frames)
            positions = np.random.randn(n_frames, 6)

            episodes.append({
                'states': positions,
                'actions': positions.copy(),
                'timestamps': t
            })

        return episodes

    def _create_variable_length_episodes(self, n_episodes=10):
        """Create episodes with variable lengths."""
        np.random.seed(42)
        episodes = []
        lengths = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

        for i, length in enumerate(lengths[:n_episodes]):
            t = np.linspace(0, 5, length)
            positions = np.random.randn(length, 6)

            episodes.append({
                'states': positions,
                'actions': positions.copy(),
                'timestamps': t
            })

        return episodes

    def _create_mock_motion_results(self, n_episodes=10):
        """Create mock motion quality results."""
        np.random.seed(42)
        scores = np.random.uniform(0.3, 0.9, n_episodes)

        return {
            'episode_results': [
                {'overall_score': float(score)}
                for score in scores
            ]
        }
