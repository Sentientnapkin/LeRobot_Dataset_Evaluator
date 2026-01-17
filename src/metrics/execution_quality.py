# src/metrics/execution_quality.py
"""
Execution quality metrics for teleoperated robotics data.

These metrics are designed to discriminate between good and poor teleoperation
quality, normalized for human operator performance (not ideal trajectories).

Based on research from:
- Flash & Hogan (1985): Movement units
- Rohrer et al. (2002): Hesitations and corrections
- Meyer et al. (1988): Submovement analysis
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .utils import compute_statistics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ExecutionQualityAnalyzer:
    """Analyzer for teleoperation execution quality metrics."""

    def __init__(
        self,
        velocity_threshold_factor: float = 0.05,
        acceleration_peak_prominence: float = 0.1
    ):
        """Initialize execution quality analyzer.

        Args:
            velocity_threshold_factor: Factor of max velocity for hesitation detection
            acceleration_peak_prominence: Minimum prominence for acceleration peaks
        """
        self.velocity_threshold_factor = velocity_threshold_factor
        self.acceleration_peak_prominence = acceleration_peak_prominence

    def compute_number_of_movement_units(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Number of Movement Units (NMU).

        A movement unit is a segment with a single acceleration-deceleration cycle.
        Skilled operators use fewer, larger movement units.

        Args:
            velocities: Velocity array, shape (n_frames, n_dof)
            timestamps: Timestamp array

        Returns:
            Dictionary containing NMU metrics
        """
        # Compute velocity magnitude
        vel_magnitude = np.linalg.norm(velocities, axis=1)

        # Smooth velocity to reduce noise
        vel_smooth = gaussian_filter1d(vel_magnitude, sigma=2.0)

        # Find velocity peaks (local maxima)
        # Each peak represents a movement unit
        peaks, properties = signal.find_peaks(
            vel_smooth,
            prominence=np.max(vel_smooth) * 0.05,  # 5% of max velocity
            distance=5  # Minimum 5 frames between peaks
        )

        n_movement_units = len(peaks)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0

        # Normalize by duration (units per second)
        nmu_per_second = n_movement_units / duration if duration > 0 else 0

        # Scoring: fewer units = better (more purposeful movements)
        # Typical ranges:
        # - Excellent: 0.5-1.5 units/sec
        # - Good: 1.5-3.0 units/sec
        # - Acceptable: 3.0-5.0 units/sec
        # - Poor: > 5.0 units/sec
        if nmu_per_second <= 1.5:
            score = 1.0
        elif nmu_per_second <= 3.0:
            score = 1.0 - (nmu_per_second - 1.5) / 1.5 * 0.3  # 0.7-1.0
        elif nmu_per_second <= 5.0:
            score = 0.7 - (nmu_per_second - 3.0) / 2.0 * 0.4  # 0.3-0.7
        else:
            score = max(0.0, 0.3 - (nmu_per_second - 5.0) / 5.0 * 0.3)  # 0.0-0.3

        return {
            'n_movement_units': n_movement_units,
            'nmu_per_second': nmu_per_second,
            'duration': duration,
            'peak_velocities': vel_smooth[peaks],
            'peak_times': timestamps[peaks],
            'score': score
        }

    def compute_hesitation_rate(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute hesitation rate (velocity reversals per second).

        Hesitations indicate uncertainty, poor control mapping, or task difficulty.
        Measured as number of times velocity direction reverses.

        Args:
            velocities: Velocity array, shape (n_frames, n_dof)
            timestamps: Timestamp array

        Returns:
            Dictionary containing hesitation metrics
        """
        # Compute velocity magnitude
        vel_magnitude = np.linalg.norm(velocities, axis=1)

        # Smooth to reduce noise-induced reversals
        vel_smooth = gaussian_filter1d(vel_magnitude, sigma=1.5)

        # Define movement threshold (5% of max velocity)
        vel_threshold = np.max(vel_smooth) * self.velocity_threshold_factor

        # Find periods of movement (above threshold)
        moving = vel_smooth > vel_threshold

        # Detect velocity reversals (direction changes during movement)
        vel_diff = np.diff(vel_smooth)
        direction_changes = np.diff(np.sign(vel_diff))

        # Count reversals only during movement periods
        # Adjust moving mask length to match direction_changes
        moving_trimmed = moving[:len(direction_changes)]
        reversals = np.sum(np.abs(direction_changes[moving_trimmed]) > 1)

        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        hesitation_rate = reversals / duration if duration > 0 else 0

        # Scoring: fewer hesitations = better
        # Typical ranges:
        # - Excellent: 0-0.5 reversals/sec
        # - Good: 0.5-1.5 reversals/sec
        # - Acceptable: 1.5-3.0 reversals/sec
        # - Poor: > 3.0 reversals/sec
        if hesitation_rate <= 0.5:
            score = 1.0
        elif hesitation_rate <= 1.5:
            score = 1.0 - (hesitation_rate - 0.5) / 1.0 * 0.3  # 0.7-1.0
        elif hesitation_rate <= 3.0:
            score = 0.7 - (hesitation_rate - 1.5) / 1.5 * 0.4  # 0.3-0.7
        else:
            score = max(0.0, 0.3 - (hesitation_rate - 3.0) / 3.0 * 0.3)  # 0.0-0.3

        return {
            'n_hesitations': int(reversals),
            'hesitation_rate': hesitation_rate,
            'duration': duration,
            'velocity_threshold': vel_threshold,
            'score': score
        }

    def compute_submovement_index(
        self,
        accelerations: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute submovement index (number of acceleration peaks).

        Smooth movements have fewer acceleration peaks. More peaks indicate
        corrective submovements and less skilled execution.

        Args:
            accelerations: Acceleration array, shape (n_frames, n_dof)
            timestamps: Timestamp array

        Returns:
            Dictionary containing submovement metrics
        """
        # Compute acceleration magnitude
        acc_magnitude = np.linalg.norm(accelerations, axis=1)

        # Smooth to reduce noise
        acc_smooth = gaussian_filter1d(acc_magnitude, sigma=2.0)

        # Find acceleration peaks (both positive and negative)
        # Positive peaks (accelerations)
        peaks_pos, _ = signal.find_peaks(
            acc_smooth,
            prominence=np.std(acc_smooth) * self.acceleration_peak_prominence
        )

        # Negative peaks (decelerations) - invert signal
        peaks_neg, _ = signal.find_peaks(
            -acc_smooth,
            prominence=np.std(acc_smooth) * self.acceleration_peak_prominence
        )

        n_submovements = len(peaks_pos) + len(peaks_neg)
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0

        # Normalize by duration
        submovement_index = n_submovements / duration if duration > 0 else 0

        # Scoring: fewer submovements = better
        # Typical ranges:
        # - Excellent: 1-3 submovements/sec
        # - Good: 3-5 submovements/sec
        # - Acceptable: 5-8 submovements/sec
        # - Poor: > 8 submovements/sec
        if submovement_index <= 3.0:
            score = 1.0
        elif submovement_index <= 5.0:
            score = 1.0 - (submovement_index - 3.0) / 2.0 * 0.3  # 0.7-1.0
        elif submovement_index <= 8.0:
            score = 0.7 - (submovement_index - 5.0) / 3.0 * 0.4  # 0.3-0.7
        else:
            score = max(0.0, 0.3 - (submovement_index - 8.0) / 4.0 * 0.3)  # 0.0-0.3

        return {
            'n_submovements': n_submovements,
            'n_accelerations': len(peaks_pos),
            'n_decelerations': len(peaks_neg),
            'submovement_index': submovement_index,
            'duration': duration,
            'score': score
        }

    def compute_normalized_jerk_percentile(
        self,
        jerk: np.ndarray,
        timestamps: np.ndarray,
        positions: np.ndarray,
        reference_distribution: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute normalized jerk with percentile-based scoring.

        Instead of absolute thresholds, compare to reference distribution
        of teleoperated data.

        Args:
            jerk: Jerk array, shape (n_frames, n_dof)
            timestamps: Timestamp array
            positions: Position array (for path length)
            reference_distribution: Optional reference jerk values for percentile

        Returns:
            Dictionary containing jerk metrics
        """
        # Compute jerk magnitude
        jerk_magnitude = np.linalg.norm(jerk, axis=1)

        # Compute dimensionless jerk
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        if duration > 0 and path_length > 0:
            # Integrate squared jerk
            dt = np.diff(timestamps)
            dt = np.where(dt == 0, 1e-6, dt)
            jerk_squared = jerk_magnitude[:-1] ** 2
            jerk_cost = np.sum(jerk_squared * dt)

            # Dimensionless jerk
            dimensionless_jerk = jerk_cost * (duration ** 3) / (path_length ** 2)
        else:
            dimensionless_jerk = 0.0
            jerk_cost = 0.0

        # Percentile-based scoring
        if reference_distribution is not None and len(reference_distribution) > 0:
            # Score based on percentile rank (lower jerk = higher percentile = better)
            percentile = (np.sum(reference_distribution > dimensionless_jerk) /
                         len(reference_distribution) * 100)
            score = percentile / 100.0
        else:
            # Fallback: use empirical thresholds for teleoperated data
            # Based on observed ranges in real datasets
            if dimensionless_jerk <= 1e4:
                score = 1.0
            elif dimensionless_jerk <= 1e5:
                score = 1.0 - (np.log10(dimensionless_jerk) - 4) / 1 * 0.3  # 0.7-1.0
            elif dimensionless_jerk <= 1e6:
                score = 0.7 - (np.log10(dimensionless_jerk) - 5) / 1 * 0.4  # 0.3-0.7
            else:
                score = max(0.0, 0.3 - (np.log10(dimensionless_jerk) - 6) / 1 * 0.3)

        stats = compute_statistics(jerk_magnitude)

        # Detect outliers in jerk magnitude
        outliers = self._detect_outliers(jerk_magnitude, threshold_multiplier=2.0)

        return {
            'jerk_magnitude_mean': stats['mean'],
            'jerk_magnitude_max': stats['max'],
            'dimensionless_jerk': dimensionless_jerk,
            'jerk_cost': jerk_cost,
            'statistics': stats,
            'outliers': outliers,  # Add outlier information
            'score': score
        }

    def _detect_outliers(
        self,
        values: np.ndarray,
        threshold_multiplier: float = 2.0
    ) -> Dict[str, Any]:
        """Detect outliers using mean ± threshold * std.

        Args:
            values: Array of values to check for outliers
            threshold_multiplier: Number of standard deviations for threshold

        Returns:
            Dictionary with outlier statistics:
                - outlier_percentage: Percentage of values that are outliers
                - outlier_indices: Array indices of outliers
                - threshold: The outlier threshold value
                - n_outliers: Count of outliers
        """
        if len(values) == 0:
            return {
                'outlier_percentage': 0.0,
                'outlier_indices': np.array([]),
                'threshold': 0.0,
                'n_outliers': 0
            }

        # Calculate mean and std
        mean = np.mean(values)
        std = np.std(values)

        # Define outlier threshold
        threshold = mean + (threshold_multiplier * std)

        # Find outliers (values exceeding threshold)
        outlier_mask = values > threshold
        outlier_indices = np.where(outlier_mask)[0]
        n_outliers = len(outlier_indices)
        outlier_percentage = (n_outliers / len(values)) * 100.0

        return {
            'outlier_percentage': float(outlier_percentage),
            'outlier_indices': outlier_indices.tolist(),
            'threshold': float(threshold),
            'n_outliers': int(n_outliers)
        }

    def analyze_episode(
        self,
        episode_features: Dict[str, np.ndarray],
        reference_jerk_dist: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze execution quality for a single episode.

        Args:
            episode_features: Dictionary with kinematic features
            reference_jerk_dist: Optional reference jerk distribution

        Returns:
            Dictionary containing all execution quality metrics
        """
        velocities = episode_features['velocities']
        accelerations = episode_features['accelerations']
        jerk = episode_features['jerk']
        timestamps = episode_features['timestamps']
        positions = episode_features['positions']

        # Compute all metrics
        nmu_results = self.compute_number_of_movement_units(velocities, timestamps)
        hesitation_results = self.compute_hesitation_rate(velocities, timestamps)
        submovement_results = self.compute_submovement_index(accelerations, timestamps)
        jerk_results = self.compute_normalized_jerk_percentile(
            jerk, timestamps, positions, reference_jerk_dist
        )

        # Individual scores
        individual_scores = {
            'movement_units': nmu_results['score'],
            'hesitations': hesitation_results['score'],
            'submovements': submovement_results['score'],
            'jerk': jerk_results['score']
        }

        # Weighted overall score
        weights = {
            'movement_units': 0.30,
            'hesitations': 0.25,
            'submovements': 0.25,
            'jerk': 0.20
        }

        overall_score = sum(
            individual_scores[k] * weights[k]
            for k in individual_scores.keys()
        )

        return {
            'movement_units': nmu_results,
            'hesitations': hesitation_results,
            'submovements': submovement_results,
            'jerk': jerk_results,
            'individual_scores': individual_scores,
            'overall_score': overall_score,
            'weights': weights
        }

    def analyze_multiple_episodes(
        self,
        episodes_features: List[Dict[str, np.ndarray]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Analyze execution quality for multiple episodes.

        Args:
            episodes_features: List of episode feature dictionaries
            show_progress: Whether to show progress

        Returns:
            Dictionary containing aggregated results
        """
        from tqdm import tqdm

        # First pass: collect jerk distribution for percentile-based scoring
        all_jerk_values = []
        for features in episodes_features:
            jerk = features['jerk']
            timestamps = features['timestamps']
            positions = features['positions']

            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1.0
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

            if duration > 0 and path_length > 0:
                dt = np.diff(timestamps)
                dt = np.where(dt == 0, 1e-6, dt)
                jerk_squared = jerk_magnitude[:-1] ** 2
                jerk_cost = np.sum(jerk_squared * dt)
                dimensionless_jerk = jerk_cost * (duration ** 3) / (path_length ** 2)
                all_jerk_values.append(dimensionless_jerk)

        reference_jerk_dist = np.array(all_jerk_values) if all_jerk_values else None

        # Second pass: analyze each episode with reference distribution
        episode_results = []

        iterator = episodes_features
        if show_progress:
            iterator = tqdm(iterator, desc="Analyzing execution quality")

        for features in iterator:
            try:
                result = self.analyze_episode(features, reference_jerk_dist)
                episode_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze episode: {e}")
                continue

        if not episode_results:
            logger.error("No episodes successfully analyzed")
            return {}

        # Aggregate results
        aggregated = self._aggregate_episode_results(episode_results)

        return aggregated

    def _aggregate_episode_results(
        self,
        episode_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple episodes."""
        # Extract overall scores
        overall_scores = [r['overall_score'] for r in episode_results]

        # Extract individual metric scores
        nmu_scores = [r['individual_scores']['movement_units'] for r in episode_results]
        hesitation_scores = [r['individual_scores']['hesitations'] for r in episode_results]
        submovement_scores = [r['individual_scores']['submovements'] for r in episode_results]
        jerk_scores = [r['individual_scores']['jerk'] for r in episode_results]

        # Compute statistics with scores arrays included
        def compute_stats_with_scores(scores_list):
            """Compute statistics and include scores array."""
            stats = compute_statistics(np.array(scores_list))
            stats['scores'] = scores_list  # Add scores array for visualizations
            return stats

        # Compute statistics
        aggregated = {
            'overall_score': {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': np.min(overall_scores),
                'max': np.max(overall_scores),
                'scores': overall_scores
            },
            'movement_units_scores': compute_stats_with_scores(nmu_scores),
            'hesitation_scores': compute_stats_with_scores(hesitation_scores),
            'submovement_scores': compute_stats_with_scores(submovement_scores),
            'jerk_scores': compute_stats_with_scores(jerk_scores),
            'n_episodes': len(episode_results),
            'episode_results': episode_results
        }

        return aggregated


def analyze_execution_quality(
    episodes_features: List[Dict[str, np.ndarray]],
    velocity_threshold_factor: float = 0.05,
    acceleration_peak_prominence: float = 0.1
) -> Dict[str, Any]:
    """Convenience function to analyze execution quality.

    Args:
        episodes_features: List of episode feature dictionaries
        velocity_threshold_factor: Factor for hesitation detection
        acceleration_peak_prominence: Prominence for submovement detection

    Returns:
        Aggregated execution quality analysis
    """
    analyzer = ExecutionQualityAnalyzer(
        velocity_threshold_factor=velocity_threshold_factor,
        acceleration_peak_prominence=acceleration_peak_prominence
    )

    return analyzer.analyze_multiple_episodes(episodes_features)
