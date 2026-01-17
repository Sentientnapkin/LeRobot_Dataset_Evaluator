"""Data consistency and quality metrics."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .utils import compute_statistics
from ..utils.logging_config import get_logger
from ..utils.robot_models import RobotConfig, is_within_joint_limits

logger = get_logger(__name__)


class DataConsistencyAnalyzer:
    """Analyzer for data consistency and quality."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        anomaly_method: str = 'isolation_forest',
        contamination: float = 0.1
    ):
        """Initialize data consistency analyzer.

        Args:
            robot_config: Robot configuration for validation
            anomaly_method: Anomaly detection method ('isolation_forest' or 'lof')
            contamination: Expected proportion of outliers
        """
        self.robot_config = robot_config
        self.anomaly_method = anomaly_method
        self.contamination = contamination

    def check_action_observation_alignment(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Check temporal alignment between actions and observations.

        Args:
            observations: Observation array
            actions: Action array
            timestamps: Timestamp array

        Returns:
            Dictionary containing alignment metrics
        """
        issues = []
        alignment_score = 1.0

        # Check dimension match
        if len(observations) != len(actions):
            issues.append(f"Dimension mismatch: {len(observations)} obs vs {len(actions)} actions")
            alignment_score *= 0.5

        if len(observations) != len(timestamps):
            issues.append(f"Timestamp mismatch: {len(observations)} obs vs {len(timestamps)} timestamps")
            alignment_score *= 0.5

        # Check timestamp monotonicity
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            negative_diffs = np.sum(time_diffs < 0)

            if negative_diffs > 0:
                issues.append(f"Non-monotonic timestamps: {negative_diffs} violations")
                alignment_score *= 0.7

            # Check for zero time differences (duplicates)
            zero_diffs = np.sum(time_diffs == 0)
            if zero_diffs > 0:
                issues.append(f"Duplicate timestamps: {zero_diffs} instances")
                alignment_score *= 0.9

        # Check for large time gaps
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            mean_dt = np.mean(time_diffs)
            std_dt = np.std(time_diffs)

            large_gaps = np.sum(time_diffs > mean_dt + 3 * std_dt)
            if large_gaps > 0:
                issues.append(f"Large time gaps: {large_gaps} instances")
                alignment_score *= 0.95

        return {
            'aligned': len(issues) == 0,
            'issues': issues,
            'n_issues': len(issues),
            'score': alignment_score
        }

    def check_temporal_consistency(
        self,
        timestamps: np.ndarray,
        expected_fps: float = 30.0
    ) -> Dict[str, Any]:
        """Check frame rate and temporal consistency.

        Args:
            timestamps: Timestamp array
            expected_fps: Expected frames per second

        Returns:
            Dictionary containing temporal consistency metrics
        """
        if len(timestamps) < 2:
            return {
                'mean_fps': 0.0,
                'std_fps': 0.0,
                'dropout_percentage': 0.0,
                'jitter': 0.0,
                'score': 0.0
            }

        # Compute frame rate
        time_diffs = np.diff(timestamps)
        instantaneous_fps = 1.0 / (time_diffs + 1e-10)

        mean_fps = np.mean(instantaneous_fps)
        std_fps = np.std(instantaneous_fps)

        # Compute jitter (variation in frame timing)
        expected_dt = 1.0 / expected_fps
        jitter = np.std(time_diffs) / expected_dt if expected_dt > 0 else 0.0

        # Detect frame dropouts (time diffs > 2x expected)
        dropout_threshold = 2.0 * expected_dt
        dropouts = np.sum(time_diffs > dropout_threshold)
        dropout_percentage = (dropouts / len(time_diffs)) * 100

        # Consistency score (lower jitter and dropouts = better)
        jitter_score = max(0, 1.0 - jitter / 0.5)  # Jitter < 0.5 is good
        dropout_score = max(0, 1.0 - dropout_percentage / 10.0)  # < 10% dropouts is acceptable

        consistency_score = 0.6 * jitter_score + 0.4 * dropout_score

        return {
            'mean_fps': mean_fps,
            'std_fps': std_fps,
            'expected_fps': expected_fps,
            'dropout_count': int(dropouts),
            'dropout_percentage': dropout_percentage,
            'jitter': jitter,
            'score': consistency_score
        }

    def detect_anomalies(
        self,
        trajectories: np.ndarray,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect anomalous trajectories using ML methods.

        Args:
            trajectories: Trajectory data, shape (n_samples, n_features)
            method: Detection method (overrides instance setting)

        Returns:
            Dictionary containing anomaly detection results
        """
        method = method or self.anomaly_method

        if len(trajectories) < 10:
            logger.warning("Too few samples for anomaly detection")
            return {
                'anomaly_indices': np.array([]),
                'anomaly_scores': np.array([]),
                'n_anomalies': 0,
                'anomaly_percentage': 0.0,
                'score': 1.0
            }

        try:
            if method == 'isolation_forest':
                detector = IsolationForest(
                    contamination=self.contamination,
                    random_state=42
                )
            elif method == 'lof':
                detector = LocalOutlierFactor(
                    contamination=self.contamination,
                    novelty=False
                )
            else:
                raise ValueError(f"Unknown anomaly detection method: {method}")

            # Fit and predict
            if method == 'isolation_forest':
                predictions = detector.fit_predict(trajectories)
                anomaly_scores = -detector.score_samples(trajectories)  # Negative for consistency
            else:  # LOF
                predictions = detector.fit_predict(trajectories)
                anomaly_scores = -detector.negative_outlier_factor_

            # Anomalies are labeled as -1
            anomaly_mask = predictions == -1
            anomaly_indices = np.where(anomaly_mask)[0]

            n_anomalies = np.sum(anomaly_mask)
            anomaly_percentage = (n_anomalies / len(trajectories)) * 100

            # Quality score (fewer anomalies = better)
            quality_score = max(0, 1.0 - anomaly_percentage / 20.0)  # < 20% is acceptable

            return {
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': anomaly_scores,
                'anomaly_mask': anomaly_mask,
                'n_anomalies': int(n_anomalies),
                'anomaly_percentage': anomaly_percentage,
                'method': method,
                'score': quality_score
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                'anomaly_indices': np.array([]),
                'anomaly_scores': np.array([]),
                'n_anomalies': 0,
                'anomaly_percentage': 0.0,
                'score': 0.5,
                'error': str(e)
            }

    def check_missing_data(
        self,
        episodes_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for missing or invalid data.

        Args:
            episodes_data: List of episode dictionaries

        Returns:
            Dictionary containing missing data analysis
        """
        total_frames = 0
        nan_count = 0
        inf_count = 0
        incomplete_episodes = 0

        for i, episode in enumerate(episodes_data):
            states = episode.get('states')
            actions = episode.get('actions')
            timestamps = episode.get('timestamps')

            # Check for None
            if states is None or actions is None or timestamps is None:
                incomplete_episodes += 1
                continue

            # Count frames
            total_frames += len(states)

            # Check for NaN
            nan_count += np.sum(np.isnan(states))
            nan_count += np.sum(np.isnan(actions))
            nan_count += np.sum(np.isnan(timestamps))

            # Check for Inf
            inf_count += np.sum(np.isinf(states))
            inf_count += np.sum(np.isinf(actions))
            inf_count += np.sum(np.isinf(timestamps))

        # Completeness metrics
        complete_episodes = len(episodes_data) - incomplete_episodes
        completeness_percentage = (complete_episodes / len(episodes_data)) * 100 if episodes_data else 0

        nan_percentage = (nan_count / (total_frames * 1.0 + 1e-10)) * 100 if total_frames > 0 else 0
        inf_percentage = (inf_count / (total_frames * 1.0 + 1e-10)) * 100 if total_frames > 0 else 0

        # Quality score
        completeness_score = completeness_percentage / 100.0
        validity_score = 1.0 - (nan_percentage + inf_percentage) / 200.0  # Penalize NaN/Inf
        quality_score = 0.7 * completeness_score + 0.3 * max(0, validity_score)

        return {
            'complete_episodes': complete_episodes,
            'incomplete_episodes': incomplete_episodes,
            'completeness_percentage': completeness_percentage,
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'nan_percentage': nan_percentage,
            'inf_percentage': inf_percentage,
            'total_frames': total_frames,
            'score': quality_score
        }

    def check_joint_limits(
        self,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check if joint positions are within valid limits.

        Args:
            positions: Joint positions, shape (n_samples, n_dof)

        Returns:
            Dictionary containing joint limit violations
        """
        if self.robot_config is None:
            return {
                'violations': 0,
                'violation_percentage': 0.0,
                'score': 1.0,
                'warning': 'No robot config provided'
            }

        # Check limits
        within_limits = is_within_joint_limits(positions, self.robot_config)

        n_violations = np.sum(~within_limits)
        violation_percentage = (n_violations / len(positions)) * 100

        # Quality score
        quality_score = max(0, 1.0 - violation_percentage / 10.0)  # < 10% violations acceptable

        return {
            'violations': int(n_violations),
            'violation_percentage': violation_percentage,
            'violation_indices': np.where(~within_limits)[0],
            'score': quality_score
        }

    def analyze_data_consistency(
        self,
        episodes_data: List[Dict[str, Any]],
        expected_fps: float = 30.0
    ) -> Dict[str, Any]:
        """Analyze data consistency for multiple episodes.

        Args:
            episodes_data: List of episode dictionaries
            expected_fps: Expected frame rate

        Returns:
            Dictionary containing data consistency analysis
        """
        # Collect all data
        all_observations = []
        all_actions = []
        all_timestamps = []

        alignment_scores = []
        temporal_scores = []

        logger.info("Checking alignment and temporal consistency...")

        for episode in episodes_data:
            states = episode.get('states')
            actions = episode.get('actions')
            timestamps = episode.get('timestamps')

            if states is None or actions is None or timestamps is None:
                continue

            all_observations.append(states)
            all_actions.append(actions)
            all_timestamps.append(timestamps)

            # Check alignment for this episode
            alignment = self.check_action_observation_alignment(
                states, actions, timestamps
            )
            alignment_scores.append(alignment['score'])

            # Check temporal consistency
            temporal = self.check_temporal_consistency(timestamps, expected_fps)
            temporal_scores.append(temporal['score'])

        # Aggregate alignment and temporal results
        mean_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0
        mean_temporal_score = np.mean(temporal_scores) if temporal_scores else 0.0

        # Check missing data
        logger.info("Checking for missing data...")
        missing_data = self.check_missing_data(episodes_data)

        # Anomaly detection on flattened trajectories
        logger.info("Detecting anomalies...")
        if all_observations:
            # Flatten all observations
            flat_observations = np.vstack(all_observations)

            # Sample for anomaly detection if too large
            if len(flat_observations) > 10000:
                sample_indices = np.random.choice(len(flat_observations), 10000, replace=False)
                flat_observations = flat_observations[sample_indices]

            anomaly_results = self.detect_anomalies(flat_observations)
        else:
            anomaly_results = {
                'n_anomalies': 0,
                'anomaly_percentage': 0.0,
                'score': 1.0
            }

        # Check joint limits
        if all_observations and self.robot_config:
            logger.info("Checking joint limits...")
            flat_observations = np.vstack(all_observations)
            joint_limit_results = self.check_joint_limits(flat_observations)
        else:
            joint_limit_results = {
                'violations': 0,
                'violation_percentage': 0.0,
                'score': 1.0
            }

        # Overall consistency score (weighted average)
        weights = {
            'alignment': 0.25,
            'temporal': 0.25,
            'missing_data': 0.20,
            'anomalies': 0.20,
            'joint_limits': 0.10
        }

        overall_score = (
            weights['alignment'] * mean_alignment_score +
            weights['temporal'] * mean_temporal_score +
            weights['missing_data'] * missing_data['score'] +
            weights['anomalies'] * anomaly_results['score'] +
            weights['joint_limits'] * joint_limit_results['score']
        )

        return {
            'alignment': {
                'mean_score': mean_alignment_score,
                'episodes_analyzed': len(alignment_scores)
            },
            'temporal_consistency': {
                'mean_score': mean_temporal_score,
                'episodes_analyzed': len(temporal_scores)
            },
            'missing_data': missing_data,
            'anomalies': anomaly_results,
            'joint_limits': joint_limit_results,
            'overall_score': overall_score,
            'individual_scores': {
                'alignment': mean_alignment_score,
                'temporal': mean_temporal_score,
                'missing_data': missing_data['score'],
                'anomalies': anomaly_results['score'],
                'joint_limits': joint_limit_results['score']
            },
            'weights': weights,
            'n_episodes': len(episodes_data)
        }


def analyze_data_consistency(
    episodes_data: List[Dict[str, Any]],
    robot_config: Optional[RobotConfig] = None,
    expected_fps: float = 30.0
) -> Dict[str, Any]:
    """Convenience function to analyze data consistency.

    Args:
        episodes_data: List of episode dictionaries
        robot_config: Optional robot configuration
        expected_fps: Expected frame rate

    Returns:
        Data consistency analysis results
    """
    analyzer = DataConsistencyAnalyzer(robot_config=robot_config)

    return analyzer.analyze_data_consistency(episodes_data, expected_fps)
