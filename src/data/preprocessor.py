"""Data preprocessing and feature extraction."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d

from ..utils.logging_config import get_logger
from ..utils.robot_models import RobotConfig, is_within_joint_limits

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocessor for robot trajectory data."""

    def __init__(self, robot_config: Optional[RobotConfig] = None):
        """Initialize preprocessor.

        Args:
            robot_config: Robot configuration for validation
        """
        self.robot_config = robot_config

    def validate_episode_data(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and report issues in episode data.

        Args:
            episode_data: Episode data dictionary

        Returns:
            Dictionary containing validation results
        """
        issues = []

        # Check for required fields
        required_fields = ['states', 'actions', 'timestamps']
        for field in required_fields:
            if field not in episode_data or episode_data[field] is None:
                issues.append(f"Missing required field: {field}")

        if issues:
            return {'valid': False, 'issues': issues}

        states = episode_data['states']
        actions = episode_data['actions']
        timestamps = episode_data['timestamps']

        # Check dimensions match
        if len(states) != len(actions):
            issues.append(f"State-action dimension mismatch: {len(states)} vs {len(actions)}")

        if len(states) != len(timestamps):
            issues.append(f"State-timestamp dimension mismatch: {len(states)} vs {len(timestamps)}")

        # Check for NaN or Inf values
        if np.any(~np.isfinite(states)):
            issues.append("States contain NaN or Inf values")

        if np.any(~np.isfinite(actions)):
            issues.append("Actions contain NaN or Inf values")

        if np.any(~np.isfinite(timestamps)):
            issues.append("Timestamps contain NaN or Inf values")

        # Check timestamp monotonicity
        if not np.all(np.diff(timestamps) >= 0):
            issues.append("Timestamps are not monotonically increasing")

        # Check joint limits (if robot config available)
        if self.robot_config is not None and states.shape[1] == self.robot_config.dof:
            within_limits = is_within_joint_limits(states, self.robot_config)
            if not np.all(within_limits):
                n_violations = np.sum(~within_limits)
                issues.append(f"{n_violations} states violate joint limits")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_frames': len(states)
        }

    def compute_velocities(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        smooth_sigma: float = 0.0
    ) -> np.ndarray:
        """Compute velocities from positions using numerical differentiation.

        Args:
            positions: Position array, shape (n_frames, n_dof)
            timestamps: Timestamp array, shape (n_frames,)
            smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)

        Returns:
            Velocity array, shape (n_frames, n_dof)
        """
        # Compute time deltas
        dt = np.diff(timestamps)

        # Handle edge case where dt is zero
        dt = np.where(dt == 0, 1e-6, dt)

        # Compute position differences
        dpos = np.diff(positions, axis=0)

        # Compute velocities
        velocities = dpos / dt[:, np.newaxis]

        # Pad to match original length (repeat first velocity)
        velocities = np.vstack([velocities[0:1], velocities])

        # Optional smoothing
        if smooth_sigma > 0:
            for i in range(velocities.shape[1]):
                velocities[:, i] = gaussian_filter1d(velocities[:, i], sigma=smooth_sigma)

        return velocities

    def compute_accelerations(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray,
        smooth_sigma: float = 0.0
    ) -> np.ndarray:
        """Compute accelerations from velocities.

        Args:
            velocities: Velocity array, shape (n_frames, n_dof)
            timestamps: Timestamp array, shape (n_frames,)
            smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)

        Returns:
            Acceleration array, shape (n_frames, n_dof)
        """
        # Compute time deltas
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)

        # Compute velocity differences
        dvel = np.diff(velocities, axis=0)

        # Compute accelerations
        accelerations = dvel / dt[:, np.newaxis]

        # Pad to match original length
        accelerations = np.vstack([accelerations[0:1], accelerations])

        # Optional smoothing
        if smooth_sigma > 0:
            for i in range(accelerations.shape[1]):
                accelerations[:, i] = gaussian_filter1d(accelerations[:, i], sigma=smooth_sigma)

        return accelerations

    def compute_jerk(
        self,
        accelerations: np.ndarray,
        timestamps: np.ndarray,
        smooth_sigma: float = 0.0
    ) -> np.ndarray:
        """Compute jerk (derivative of acceleration).

        Args:
            accelerations: Acceleration array, shape (n_frames, n_dof)
            timestamps: Timestamp array, shape (n_frames,)
            smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)

        Returns:
            Jerk array, shape (n_frames, n_dof)
        """
        # Compute time deltas
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)

        # Compute acceleration differences
        dacc = np.diff(accelerations, axis=0)

        # Compute jerk
        jerk = dacc / dt[:, np.newaxis]

        # Pad to match original length
        jerk = np.vstack([jerk[0:1], jerk])

        # Optional smoothing
        if smooth_sigma > 0:
            for i in range(jerk.shape[1]):
                jerk[:, i] = gaussian_filter1d(jerk[:, i], sigma=smooth_sigma)

        return jerk

    def extract_features(
        self,
        episode_data: Dict[str, Any],
        smooth_sigma: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Extract kinematic features from episode data.

        Args:
            episode_data: Episode data dictionary
            smooth_sigma: Smoothing parameter for derivatives

        Returns:
            Dictionary containing extracted features
        """
        states = episode_data['states']
        timestamps = episode_data['timestamps']

        # Compute derivatives
        velocities = self.compute_velocities(states, timestamps, smooth_sigma)
        accelerations = self.compute_accelerations(velocities, timestamps, smooth_sigma)
        jerk = self.compute_jerk(accelerations, timestamps, smooth_sigma)

        features = {
            'states': states,  # Keep for compatibility with spatial coverage
            'positions': states,  # Alias for motion quality analysis
            'velocities': velocities,
            'accelerations': accelerations,
            'jerk': jerk,
            'timestamps': timestamps,
            'actions': episode_data['actions']
        }

        return features

    def detect_outliers(
        self,
        data: np.ndarray,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> np.ndarray:
        """Detect outlier frames in data.

        Args:
            data: Data array, shape (n_frames, n_features)
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Boolean array indicating outliers
        """
        if method == 'iqr':
            # Interquartile range method
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = np.any((data < lower_bound) | (data > upper_bound), axis=1)

        elif method == 'zscore':
            # Z-score method
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            z_scores = np.abs((data - mean) / (std + 1e-8))
            outliers = np.any(z_scores > threshold, axis=1)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers

    def handle_missing_data(
        self,
        data: np.ndarray,
        method: str = 'interpolate'
    ) -> Tuple[np.ndarray, int]:
        """Handle missing or invalid data points.

        Args:
            data: Data array
            method: Handling method ('interpolate', 'forward_fill', or 'remove')

        Returns:
            Tuple of (cleaned data, number of missing points)
        """
        # Find missing/invalid data
        missing_mask = ~np.isfinite(data)
        n_missing = np.sum(missing_mask)

        if n_missing == 0:
            return data, 0

        if method == 'interpolate':
            # Linear interpolation
            for col in range(data.shape[1]):
                col_data = data[:, col]
                valid_mask = np.isfinite(col_data)

                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(~valid_mask)[0]

                    if len(valid_indices) > 1:
                        data[invalid_indices, col] = np.interp(
                            invalid_indices,
                            valid_indices,
                            col_data[valid_indices]
                        )

        elif method == 'forward_fill':
            # Forward fill
            for col in range(data.shape[1]):
                col_data = data[:, col]
                valid_mask = np.isfinite(col_data)

                if np.any(valid_mask):
                    last_valid = col_data[valid_mask][0]
                    for i in range(len(col_data)):
                        if not np.isfinite(col_data[i]):
                            col_data[i] = last_valid
                        else:
                            last_valid = col_data[i]

        elif method == 'remove':
            # Remove rows with missing data
            valid_mask = np.all(np.isfinite(data), axis=1)
            data = data[valid_mask]

        else:
            raise ValueError(f"Unknown missing data handling method: {method}")

        return data, n_missing

    def preprocess_episodes(
        self,
        episodes: List[Dict[str, Any]],
        smooth_sigma: float = 1.0,
        validate: bool = True,
        extract_features: bool = True
    ) -> List[Dict[str, Any]]:
        """Preprocess multiple episodes.

        Args:
            episodes: List of episode data dictionaries
            smooth_sigma: Smoothing parameter
            validate: Whether to validate data
            extract_features: Whether to extract kinematic features

        Returns:
            List of preprocessed episode data
        """
        preprocessed = []

        logger.info(f"Preprocessing {len(episodes)} episodes...")

        for i, episode in enumerate(episodes):
            try:
                # Validate if requested
                if validate:
                    validation = self.validate_episode_data(episode)
                    if not validation['valid']:
                        logger.warning(f"Episode {i} validation failed: {validation['issues']}")
                        episode['validation'] = validation

                # Extract features if requested
                if extract_features:
                    features = self.extract_features(episode, smooth_sigma)
                    episode['features'] = features

                preprocessed.append(episode)

            except Exception as e:
                logger.error(f"Failed to preprocess episode {i}: {e}")
                continue

        logger.info(f"Successfully preprocessed {len(preprocessed)} episodes")

        return preprocessed


def preprocess_episode_batch(
    episodes: List[Dict[str, Any]],
    robot_config: Optional[RobotConfig] = None,
    smooth_sigma: float = 1.0
) -> List[Dict[str, Any]]:
    """Convenience function to preprocess a batch of episodes.

    Args:
        episodes: List of episode data
        robot_config: Robot configuration for validation
        smooth_sigma: Smoothing parameter

    Returns:
        Preprocessed episodes
    """
    preprocessor = DataPreprocessor(robot_config)
    return preprocessor.preprocess_episodes(
        episodes,
        smooth_sigma=smooth_sigma,
        validate=True,
        extract_features=True
    )
