"""Shared utilities for metric computation."""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import signal


def normalize_metric(
    values: np.ndarray,
    lower_is_better: bool = True,
    clip: bool = True
) -> np.ndarray:
    """Normalize metric values to [0, 1] range.

    Args:
        values: Metric values
        lower_is_better: If True, lower values get higher scores
        clip: Whether to clip to [0, 1]

    Returns:
        Normalized scores in [0, 1]
    """
    if len(values) == 0:
        return np.array([])

    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val < 1e-10:
        return np.ones_like(values) * 0.5

    normalized = (values - min_val) / (max_val - min_val)

    if lower_is_better:
        normalized = 1.0 - normalized

    if clip:
        normalized = np.clip(normalized, 0, 1)

    return normalized


def compute_percentiles(
    data: np.ndarray,
    percentiles: list = [25, 50, 75]
) -> Dict[str, float]:
    """Compute percentile statistics.

    Args:
        data: Data array
        percentiles: List of percentiles to compute

    Returns:
        Dictionary with percentile values
    """
    stats = {}

    for p in percentiles:
        stats[f'p{p}'] = np.percentile(data, p)

    return stats


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for a data array.

    Args:
        data: Data array

    Returns:
        Dictionary containing statistics
    """
    if len(data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'p25': np.nan,
            'p75': np.nan,
            'range': np.nan
        }

    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'p25': np.percentile(data, 25),
        'p75': np.percentile(data, 75),
        'range': np.ptp(data)
    }

    return stats


def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect outliers using z-score method.

    Args:
        data: Data array
        threshold: Z-score threshold

    Returns:
        Tuple of (outlier_mask, z_scores)
    """
    mean = np.mean(data)
    std = np.std(data)

    if std < 1e-10:
        return np.zeros(len(data), dtype=bool), np.zeros(len(data))

    z_scores = np.abs((data - mean) / std)
    outliers = z_scores > threshold

    return outliers, z_scores


def detect_outliers_iqr(
    data: np.ndarray,
    threshold: float = 1.5
) -> np.ndarray:
    """Detect outliers using IQR method.

    Args:
        data: Data array
        threshold: IQR multiplier threshold

    Returns:
        Boolean array indicating outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers


def smooth_signal(
    signal_data: np.ndarray,
    method: str = 'gaussian',
    window_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Smooth signal using various methods.

    Args:
        signal_data: 1D signal data
        method: Smoothing method ('gaussian', 'savgol', 'moving_average')
        window_size: Window size for smoothing
        sigma: Sigma for Gaussian smoothing

    Returns:
        Smoothed signal
    """
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(signal_data, sigma=sigma)

    elif method == 'savgol':
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        return signal.savgol_filter(signal_data, window_size, polyorder=3)

    elif method == 'moving_average':
        return np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def compute_signal_energy(signal_data: np.ndarray) -> float:
    """Compute signal energy (sum of squares).

    Args:
        signal_data: Signal data

    Returns:
        Signal energy
    """
    return np.sum(signal_data ** 2)


def compute_rms(signal_data: np.ndarray) -> float:
    """Compute root mean square.

    Args:
        signal_data: Signal data

    Returns:
        RMS value
    """
    return np.sqrt(np.mean(signal_data ** 2))


def aggregate_metric_results(
    metric_values: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Aggregate multiple metric results into overall score.

    Args:
        metric_values: Dictionary of metric name -> score (0-1)
        weights: Optional weights for each metric

    Returns:
        Dictionary with aggregated results
    """
    if weights is None:
        weights = {k: 1.0 for k in metric_values.keys()}

    # Normalize weights
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}

    # Compute weighted average
    weighted_sum = 0.0
    for metric_name, score in metric_values.items():
        weight = normalized_weights.get(metric_name, 0.0)
        weighted_sum += score * weight

    return {
        'overall_score': weighted_sum,
        'individual_scores': metric_values,
        'weights': normalized_weights
    }


def create_metric_summary(
    metric_name: str,
    raw_values: np.ndarray,
    score: float,
    threshold: Optional[float] = None,
    passed: Optional[bool] = None,
    description: str = ""
) -> Dict[str, Any]:
    """Create standardized metric summary.

    Args:
        metric_name: Name of the metric
        raw_values: Raw metric values
        score: Normalized score (0-1)
        threshold: Optional threshold for pass/fail
        passed: Whether metric passes quality check
        description: Description of the metric

    Returns:
        Standardized metric summary dictionary
    """
    statistics = compute_statistics(raw_values)

    summary = {
        'name': metric_name,
        'score': float(score),
        'raw_values': raw_values,
        'statistics': statistics,
        'description': description
    }

    if threshold is not None:
        summary['threshold'] = threshold

    if passed is not None:
        summary['passed'] = passed

    return summary


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """Compute total path length of a trajectory.

    Args:
        trajectory: Trajectory array, shape (n_frames, n_dims)

    Returns:
        Total path length
    """
    if len(trajectory) < 2:
        return 0.0

    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(distances)

    return total_length


def compute_straight_line_distance(
    start: np.ndarray,
    end: np.ndarray
) -> float:
    """Compute straight-line distance between two points.

    Args:
        start: Start position
        end: End position

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(end - start)


def compute_path_efficiency(trajectory: np.ndarray) -> float:
    """Compute path efficiency (straight line / actual path).

    Args:
        trajectory: Trajectory array, shape (n_frames, n_dims)

    Returns:
        Efficiency ratio (0-1, higher is better)
    """
    if len(trajectory) < 2:
        return 1.0

    actual_length = compute_trajectory_length(trajectory)
    straight_length = compute_straight_line_distance(trajectory[0], trajectory[-1])

    if actual_length < 1e-10:
        return 1.0

    efficiency = straight_length / actual_length

    return min(efficiency, 1.0)
