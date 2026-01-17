"""Motion quality metrics"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import signal, fft

from .utils import (
    compute_statistics,
    create_metric_summary,
    detect_outliers_zscore,
    compute_rms,
    compute_path_efficiency
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MotionQualityAnalyzer:
    """Analyzer for motion quality metrics."""

    def __init__(
        self,
        jerk_threshold_multiplier: float = 2.0,
        smoothness_method: str = "sparc"
    ):
        """Initialize motion quality analyzer.

        Args:
            jerk_threshold_multiplier: Multiplier for jerk threshold (σ)
            smoothness_method: Smoothness metric ('sparc' or 'ldlj')
        """
        self.jerk_threshold_multiplier = jerk_threshold_multiplier
        self.smoothness_method = smoothness_method

    def compute_jerk_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        jerk: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive jerk-based metrics.

        Args:
            positions: Position array, shape (n_frames, n_dof)
            velocities: Velocity array
            accelerations: Acceleration array
            jerk: Jerk array
            timestamps: Timestamp array

        Returns:
            Dictionary containing jerk metrics
        """
        # Compute jerk magnitude (Euclidean norm across DOFs)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)

        # Statistics
        stats = compute_statistics(jerk_magnitude)

        # Detect outliers (trajectories with excessive jerk)
        outliers, z_scores = detect_outliers_zscore(
            jerk_magnitude,
            threshold=self.jerk_threshold_multiplier
        )

        # Jerk cost (integral of squared jerk)
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        jerk_squared = jerk_magnitude[:-1] ** 2
        jerk_cost = np.sum(jerk_squared * dt)

        # Normalized jerk (dimensionless)
        duration = timestamps[-1] - timestamps[0]
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

        if duration > 0 and path_length > 0:
            normalized_jerk = jerk_cost * (duration ** 3) / (path_length ** 2)
        else:
            normalized_jerk = 0.0

        # Quality score (0-1, higher is better)
        # Lower jerk = higher score
        mean_jerk = stats['mean']
        if mean_jerk > 0:
            # Inverse sigmoid scaling
            jerk_score = 1.0 / (1.0 + mean_jerk / 100.0)
        else:
            jerk_score = 1.0

        result = {
            'jerk_magnitude': jerk_magnitude,
            'jerk_cost': jerk_cost,
            'normalized_jerk': normalized_jerk,
            'statistics': stats,
            'outlier_count': int(np.sum(outliers)),
            'outlier_percentage': float(np.mean(outliers) * 100),
            'outlier_indices': np.where(outliers)[0],
            'z_scores': z_scores,
            'score': jerk_score
        }

        return result

    def compute_sparc_smoothness(
        self,
        velocity: np.ndarray,
        timestamps: np.ndarray,
        padlevel: int = 4
    ) -> float:
        """Compute SPARC (Spectral Arc Length) smoothness metric.

        Lower values indicate smoother movements.

        Args:
            velocity: Velocity magnitude array
            timestamps: Timestamp array
            padlevel: Zero-padding level for FFT

        Returns:
            SPARC value (lower is smoother)
        """
        # Compute velocity magnitude if multi-dimensional
        if velocity.ndim > 1:
            velocity_mag = np.linalg.norm(velocity, axis=1)
        else:
            velocity_mag = velocity

        # Compute FFT
        nfft = int(2 ** np.ceil(np.log2(len(velocity_mag)) + padlevel))
        velocity_fft = fft.fft(velocity_mag, n=nfft)
        velocity_spectrum = np.abs(velocity_fft[:nfft//2])

        # Frequency vector
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            dt = 1.0 / 30.0  # Default to 30 Hz

        freq = fft.fftfreq(nfft, d=dt)[:nfft//2]

        # Normalize spectrum
        velocity_spectrum_normalized = velocity_spectrum / np.sum(velocity_spectrum)

        # Compute arc length
        # Arc length in frequency-amplitude space
        df = freq[1] - freq[0]
        arc_length = 0.0

        for i in range(1, len(freq)):
            dx = df
            dy = velocity_spectrum_normalized[i] - velocity_spectrum_normalized[i-1]
            arc_length += np.sqrt(dx**2 + dy**2)

        # Normalize by direct distance
        direct_distance = np.sqrt(
            (freq[-1] - freq[0])**2 +
            (velocity_spectrum_normalized[-1] - velocity_spectrum_normalized[0])**2
        )

        if direct_distance > 0:
            sparc = -arc_length / direct_distance
        else:
            sparc = 0.0

        return sparc

    def compute_ldlj_smoothness(
        self,
        jerk: np.ndarray,
        timestamps: np.ndarray,
        positions: np.ndarray
    ) -> float:
        """Compute Log Dimensionless Jerk (LDLJ) smoothness metric.

        Lower (more negative) values indicate smoother movements.

        Args:
            jerk: Jerk array
            timestamps: Timestamp array
            positions: Position array

        Returns:
            LDLJ value (more negative is smoother)
        """
        # Compute movement duration
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return 0.0

        # Compute path length
        path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        if path_length <= 0:
            return 0.0

        # Compute jerk magnitude
        jerk_magnitude = np.linalg.norm(jerk, axis=1)

        # Integrate squared jerk
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        jerk_squared = jerk_magnitude[:-1] ** 2
        jerk_integral = np.sum(jerk_squared * dt)

        # Dimensionless jerk
        dimensionless_jerk = (jerk_integral * (duration ** 5)) / (path_length ** 2)

        # Log dimensionless jerk
        if dimensionless_jerk > 0:
            ldlj = -np.log(dimensionless_jerk)
        else:
            ldlj = 0.0

        return ldlj

    def compute_smoothness_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        jerk: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute smoothness metrics.

        Args:
            positions: Position array
            velocities: Velocity array
            jerk: Jerk array
            timestamps: Timestamp array

        Returns:
            Dictionary containing smoothness metrics
        """
        results = {}

        # SPARC smoothness
        try:
            sparc = self.compute_sparc_smoothness(velocities, timestamps)
            results['sparc'] = sparc
        except Exception as e:
            logger.warning(f"Failed to compute SPARC: {e}")
            results['sparc'] = None

        # LDLJ smoothness
        try:
            ldlj = self.compute_ldlj_smoothness(jerk, timestamps, positions)
            results['ldlj'] = ldlj
        except Exception as e:
            logger.warning(f"Failed to compute LDLJ: {e}")
            results['ldlj'] = None

        # Overall smoothness score (0-1, higher is better)
        # Based on primary method
        if self.smoothness_method == "sparc" and results['sparc'] is not None:
            # SPARC ranges from about -1.0 (jerky) to -3.0+ (smooth)
            # More negative = smoother movement
            # Normalize to 0-1 (higher is better)
            sparc_val = results['sparc']
            # Map -1.0 -> 0.0 and -3.0 -> 1.0
            smoothness_score = np.clip((-sparc_val - 1.0) / 2.0, 0, 1)
        elif self.smoothness_method == "ldlj" and results['ldlj'] is not None:
            # LDLJ typically ranges from -10 to 10 (more positive is better)
            ldlj_val = results['ldlj']
            smoothness_score = np.clip((ldlj_val + 10) / 20, 0, 1)
        else:
            smoothness_score = 0.5

        results['score'] = smoothness_score

        return results

    def compute_acceleration_metrics(
        self,
        accelerations: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute acceleration-based metrics.

        Args:
            accelerations: Acceleration array, shape (n_frames, n_dof)
            timestamps: Timestamp array

        Returns:
            Dictionary containing acceleration metrics
        """
        # Acceleration magnitude
        acc_magnitude = np.linalg.norm(accelerations, axis=1)

        # Statistics
        stats = compute_statistics(acc_magnitude)

        # RMS acceleration
        rms_acc = compute_rms(acc_magnitude)

        # Detect abrupt accelerations (discontinuities)
        acc_diff = np.diff(acc_magnitude)
        discontinuity_threshold = 3.0 * np.std(acc_diff)
        discontinuities = np.abs(acc_diff) > discontinuity_threshold
        n_discontinuities = np.sum(discontinuities)

        # Quality score (lower acceleration variance is better)
        if stats['std'] > 0:
            acc_score = 1.0 / (1.0 + stats['std'] / 10.0)
        else:
            acc_score = 1.0

        result = {
            'acceleration_magnitude': acc_magnitude,
            'rms_acceleration': rms_acc,
            'statistics': stats,
            'discontinuity_count': int(n_discontinuities),
            'discontinuity_percentage': float(np.mean(discontinuities) * 100),
            'score': acc_score
        }

        return result

    def compute_velocity_metrics(
        self,
        velocities: np.ndarray,
        positions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Compute velocity-based metrics.

        Args:
            velocities: Velocity array, shape (n_frames, n_dof)
            positions: Position array (for workspace analysis)
            timestamps: Timestamp array

        Returns:
            Dictionary containing velocity metrics
        """
        # Velocity magnitude
        vel_magnitude = np.linalg.norm(velocities, axis=1)

        # Statistics
        stats = compute_statistics(vel_magnitude)

        # RMS velocity
        rms_vel = compute_rms(vel_magnitude)

        # Peak velocity
        peak_velocity = np.max(vel_magnitude)

        # Velocity profile (time to peak)
        peak_idx = np.argmax(vel_magnitude)
        time_to_peak = timestamps[peak_idx] - timestamps[0]
        total_duration = timestamps[-1] - timestamps[0]

        if total_duration > 0:
            normalized_time_to_peak = time_to_peak / total_duration
        else:
            normalized_time_to_peak = 0.5

        # Quality score (based on velocity consistency)
        if stats['mean'] > 0:
            cv = stats['std'] / stats['mean']  # Coefficient of variation
            vel_score = 1.0 / (1.0 + cv)
        else:
            vel_score = 0.5

        result = {
            'velocity_magnitude': vel_magnitude,
            'rms_velocity': rms_vel,
            'peak_velocity': peak_velocity,
            'time_to_peak': time_to_peak,
            'normalized_time_to_peak': normalized_time_to_peak,
            'statistics': stats,
            'score': vel_score
        }

        return result

    def compute_movement_efficiency(
        self,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Compute movement efficiency metrics.

        Args:
            positions: Position array, shape (n_frames, n_dof)

        Returns:
            Dictionary containing efficiency metrics
        """
        # Path efficiency
        efficiency = compute_path_efficiency(positions)

        # Quality score (efficiency itself is 0-1)
        efficiency_score = efficiency

        result = {
            'path_efficiency': efficiency,
            'score': efficiency_score
        }

        return result

    def analyze_episode(
        self,
        episode_features: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze motion quality for a single episode.

        Args:
            episode_features: Dictionary with 'positions', 'velocities',
                            'accelerations', 'jerk', 'timestamps'

        Returns:
            Dictionary containing all motion quality metrics
        """
        positions = episode_features['positions']
        velocities = episode_features['velocities']
        accelerations = episode_features['accelerations']
        jerk = episode_features['jerk']
        timestamps = episode_features['timestamps']

        # Compute all metrics
        jerk_metrics = self.compute_jerk_metrics(
            positions, velocities, accelerations, jerk, timestamps
        )

        smoothness_metrics = self.compute_smoothness_metrics(
            positions, velocities, jerk, timestamps
        )

        acceleration_metrics = self.compute_acceleration_metrics(
            accelerations, timestamps
        )

        velocity_metrics = self.compute_velocity_metrics(
            velocities, positions, timestamps
        )

        efficiency_metrics = self.compute_movement_efficiency(positions)

        # Aggregate scores
        individual_scores = {
            'jerk': jerk_metrics['score'],
            'smoothness': smoothness_metrics['score'],
            'acceleration': acceleration_metrics['score'],
            'velocity': velocity_metrics['score'],
            'efficiency': efficiency_metrics['score']
        }

        # Weighted overall score (jerk and smoothness are most important)
        weights = {
            'jerk': 0.35,
            'smoothness': 0.35,
            'acceleration': 0.15,
            'velocity': 0.10,
            'efficiency': 0.05
        }

        overall_score = sum(
            individual_scores[k] * weights[k]
            for k in individual_scores.keys()
        )

        return {
            'jerk': jerk_metrics,
            'smoothness': smoothness_metrics,
            'acceleration': acceleration_metrics,
            'velocity': velocity_metrics,
            'efficiency': efficiency_metrics,
            'individual_scores': individual_scores,
            'overall_score': overall_score,
            'weights': weights
        }

    def analyze_multiple_episodes(
        self,
        episodes_features: List[Dict[str, np.ndarray]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Analyze motion quality for multiple episodes.

        Args:
            episodes_features: List of episode feature dictionaries
            show_progress: Whether to show progress

        Returns:
            Dictionary containing aggregated results
        """
        from tqdm import tqdm

        episode_results = []

        iterator = episodes_features
        if show_progress:
            iterator = tqdm(iterator, desc="Analyzing motion quality")

        for features in iterator:
            try:
                result = self.analyze_episode(features)
                episode_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze episode: {e}")
                continue

        if not episode_results:
            logger.error("No episodes successfully analyzed")
            return {}

        # Aggregate results across episodes
        aggregated = self._aggregate_episode_results(episode_results)

        return aggregated

    def _aggregate_episode_results(
        self,
        episode_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple episodes.

        Args:
            episode_results: List of episode analysis results

        Returns:
            Aggregated results dictionary
        """
        # Extract overall scores
        overall_scores = [r['overall_score'] for r in episode_results]

        # Extract individual metric scores
        jerk_scores = [r['individual_scores']['jerk'] for r in episode_results]
        smoothness_scores = [r['individual_scores']['smoothness'] for r in episode_results]
        acceleration_scores = [r['individual_scores']['acceleration'] for r in episode_results]
        velocity_scores = [r['individual_scores']['velocity'] for r in episode_results]
        efficiency_scores = [r['individual_scores']['efficiency'] for r in episode_results]

        # Compute statistics
        aggregated = {
            'overall_score': {
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'min': np.min(overall_scores),
                'max': np.max(overall_scores),
                'scores': overall_scores
            },
            'jerk_scores': compute_statistics(np.array(jerk_scores)),
            'smoothness_scores': compute_statistics(np.array(smoothness_scores)),
            'acceleration_scores': compute_statistics(np.array(acceleration_scores)),
            'velocity_scores': compute_statistics(np.array(velocity_scores)),
            'efficiency_scores': compute_statistics(np.array(efficiency_scores)),
            'n_episodes': len(episode_results),
            'episode_results': episode_results
        }

        return aggregated


def analyze_motion_quality(
    episodes_features: List[Dict[str, np.ndarray]],
    jerk_threshold_multiplier: float = 2.0,
    smoothness_method: str = "sparc"
) -> Dict[str, Any]:
    """Convenience function to analyze motion quality.

    Args:
        episodes_features: List of episode feature dictionaries
        jerk_threshold_multiplier: Jerk outlier threshold
        smoothness_method: Smoothness metric to use

    Returns:
        Aggregated motion quality analysis
    """
    analyzer = MotionQualityAnalyzer(
        jerk_threshold_multiplier=jerk_threshold_multiplier,
        smoothness_method=smoothness_method
    )

    return analyzer.analyze_multiple_episodes(episodes_features)
