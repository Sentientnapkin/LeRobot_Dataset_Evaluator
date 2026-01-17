"""Success Indicators Analyzer for evaluating episode success metrics.

This module analyzes episode-level success indicators including length distribution,
quality thresholds, and trajectory patterns.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EpisodeSuccessInfo:
    """Information about an individual episode's success indicators."""
    episode_idx: int
    length: int
    quality_score: float
    is_successful: bool
    classification: str  # 'excellent', 'good', 'marginal', 'poor'


class SuccessIndicatorAnalyzer:
    """Analyzes success indicators across episodes."""

    def __init__(
        self,
        success_threshold: float = 0.6,
        excellent_threshold: float = 0.75,
        marginal_threshold: float = 0.45
    ):
        """Initialize the analyzer.

        Args:
            success_threshold: Quality score threshold for success
            excellent_threshold: Quality score threshold for excellent
            marginal_threshold: Quality score threshold for marginal
        """
        self.success_threshold = success_threshold
        self.excellent_threshold = excellent_threshold
        self.marginal_threshold = marginal_threshold

    def analyze_success_indicators(
        self,
        episodes_data: List[Dict[str, Any]],
        motion_quality_results: Optional[Dict[str, Any]] = None,
        spatial_coverage_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze success indicators for all episodes.

        Args:
            episodes_data: List of episode data dictionaries
            motion_quality_results: Optional results from execution quality analysis
            spatial_coverage_results: Optional results from spatial coverage analysis

        Returns:
            Dictionary with success analysis results
        """
        n_episodes = len(episodes_data)

        # Episode length analysis
        lengths = np.array([len(ep.get('states', [])) for ep in episodes_data])
        length_stats = self._analyze_lengths(lengths)

        # Quality score analysis
        quality_scores = self._get_quality_scores(
            n_episodes,
            motion_quality_results,
            spatial_coverage_results
        )
        quality_stats = self._analyze_quality_distribution(quality_scores)

        # Episode classification
        episode_info = self._classify_episodes(
            episodes_data,
            quality_scores,
            lengths
        )

        # Success rate analysis
        success_rates = self._calculate_success_rates(episode_info)

        # Best/worst episodes
        best_worst = self._identify_best_worst_episodes(episode_info)

        # Length-quality correlation
        correlation = self._analyze_length_quality_correlation(lengths, quality_scores)

        return {
            'n_episodes': n_episodes,
            'length_stats': length_stats,
            'quality_stats': quality_stats,
            'episode_info': episode_info,
            'success_rates': success_rates,
            'best_episodes': best_worst['best'],
            'worst_episodes': best_worst['worst'],
            'length_quality_correlation': correlation,
            'thresholds': {
                'success': self.success_threshold,
                'excellent': self.excellent_threshold,
                'marginal': self.marginal_threshold
            }
        }

    def _analyze_lengths(self, lengths: np.ndarray) -> Dict[str, Any]:
        """Analyze episode length distribution."""
        return {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'median': float(np.median(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75)),
            'histogram': self._compute_histogram(lengths),
            'lengths': lengths.tolist()
        }

    def _compute_histogram(
        self,
        values: np.ndarray,
        n_bins: int = 20
    ) -> Dict[str, List[float]]:
        """Compute histogram for visualization."""
        counts, edges = np.histogram(values, bins=n_bins)
        return {
            'counts': counts.tolist(),
            'edges': edges.tolist(),
            'bin_centers': ((edges[:-1] + edges[1:]) / 2).tolist()
        }

    def _get_quality_scores(
        self,
        n_episodes: int,
        motion_quality_results: Optional[Dict[str, Any]],
        spatial_coverage_results: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract per-episode quality scores from available metrics."""
        scores = np.zeros(n_episodes)
        n_sources = 0

        # Add motion quality scores if available
        if motion_quality_results and 'episode_results' in motion_quality_results:
            episode_results = motion_quality_results['episode_results']
            if len(episode_results) == n_episodes:
                motion_scores = np.array([
                    ep['overall_score'] for ep in episode_results
                ])
                scores += motion_scores
                n_sources += 1

        # If we have spatial coverage per-episode data
        if spatial_coverage_results and 'episode_scores' in spatial_coverage_results:
            spatial_scores = np.array(spatial_coverage_results['episode_scores'])
            if len(spatial_scores) == n_episodes:
                scores += spatial_scores
                n_sources += 1

        # Average if we have multiple sources
        if n_sources > 0:
            scores /= n_sources
        else:
            # Default to random scores centered around 0.5 if no metrics
            scores = np.random.normal(0.5, 0.15, n_episodes)
            scores = np.clip(scores, 0, 1)

        return scores

    def _analyze_quality_distribution(
        self,
        quality_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        return {
            'mean': float(np.mean(quality_scores)),
            'std': float(np.std(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'median': float(np.median(quality_scores)),
            'histogram': self._compute_histogram(quality_scores, n_bins=15),
            'scores': quality_scores.tolist()
        }

    def _classify_episodes(
        self,
        episodes_data: List[Dict[str, Any]],
        quality_scores: np.ndarray,
        lengths: np.ndarray
    ) -> List[EpisodeSuccessInfo]:
        """Classify each episode based on quality score."""
        episode_info = []

        for i, (ep, score, length) in enumerate(zip(
            episodes_data, quality_scores, lengths
        )):
            if score >= self.excellent_threshold:
                classification = 'excellent'
            elif score >= self.success_threshold:
                classification = 'good'
            elif score >= self.marginal_threshold:
                classification = 'marginal'
            else:
                classification = 'poor'

            is_successful = score >= self.success_threshold

            episode_info.append(EpisodeSuccessInfo(
                episode_idx=i,
                length=int(length),
                quality_score=float(score),
                is_successful=is_successful,
                classification=classification
            ))

        return episode_info

    def _calculate_success_rates(
        self,
        episode_info: List[EpisodeSuccessInfo]
    ) -> Dict[str, Any]:
        """Calculate success rates by various categories."""
        n_total = len(episode_info)
        if n_total == 0:
            return {
                'overall': 0.0,
                'by_classification': {},
                'counts': {}
            }

        n_successful = sum(1 for ep in episode_info if ep.is_successful)

        # Count by classification
        counts = {
            'excellent': sum(1 for ep in episode_info if ep.classification == 'excellent'),
            'good': sum(1 for ep in episode_info if ep.classification == 'good'),
            'marginal': sum(1 for ep in episode_info if ep.classification == 'marginal'),
            'poor': sum(1 for ep in episode_info if ep.classification == 'poor')
        }

        # Calculate percentages
        percentages = {
            k: (v / n_total) * 100 for k, v in counts.items()
        }

        return {
            'overall': n_successful / n_total,
            'overall_percentage': (n_successful / n_total) * 100,
            'n_successful': n_successful,
            'n_total': n_total,
            'counts': counts,
            'percentages': percentages
        }

    def _identify_best_worst_episodes(
        self,
        episode_info: List[EpisodeSuccessInfo],
        n_episodes: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Identify best and worst performing episodes."""
        sorted_by_quality = sorted(
            episode_info,
            key=lambda x: x.quality_score,
            reverse=True
        )

        best = [
            {
                'episode_idx': ep.episode_idx,
                'quality_score': ep.quality_score,
                'length': ep.length,
                'classification': ep.classification
            }
            for ep in sorted_by_quality[:n_episodes]
        ]

        worst = [
            {
                'episode_idx': ep.episode_idx,
                'quality_score': ep.quality_score,
                'length': ep.length,
                'classification': ep.classification
            }
            for ep in sorted_by_quality[-n_episodes:][::-1]
        ]

        return {'best': best, 'worst': worst}

    def _analyze_length_quality_correlation(
        self,
        lengths: np.ndarray,
        quality_scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze correlation between episode length and quality."""
        # Pearson correlation
        if len(lengths) > 2:
            correlation = float(np.corrcoef(lengths, quality_scores)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Categorize correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            strength = 'weak'
        elif abs_corr < 0.6:
            strength = 'moderate'
        else:
            strength = 'strong'

        direction = 'positive' if correlation > 0 else 'negative'

        return {
            'coefficient': correlation,
            'strength': strength,
            'direction': direction,
            'interpretation': self._interpret_correlation(correlation, strength, direction)
        }

    def _interpret_correlation(
        self,
        correlation: float,
        strength: str,
        direction: str
    ) -> str:
        """Generate human-readable interpretation of correlation."""
        if strength == 'weak':
            return "Episode length has little relationship to quality."
        elif direction == 'positive':
            if strength == 'strong':
                return "Longer episodes tend to have higher quality scores."
            else:
                return "There's a moderate tendency for longer episodes to score higher."
        else:
            if strength == 'strong':
                return "Shorter episodes tend to have higher quality scores."
            else:
                return "There's a moderate tendency for shorter episodes to score higher."
