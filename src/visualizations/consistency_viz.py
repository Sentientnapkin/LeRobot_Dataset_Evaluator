"""Data consistency visualizations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional


def plot_temporal_consistency(
    episodes_data: List[Dict[str, Any]],
    title: str = "Frame Rate Consistency Across Episodes"
) -> go.Figure:
    """Plot frame rate consistency over episodes.

    Args:
        episodes_data: List of episode dictionaries with timestamps
        title: Plot title

    Returns:
        Plotly figure
    """
    episode_indices = []
    mean_fps_list = []
    std_fps_list = []

    for i, episode in enumerate(episodes_data):
        timestamps = episode.get('timestamps')
        if timestamps is None or len(timestamps) < 2:
            continue

        time_diffs = np.diff(timestamps)
        fps = 1.0 / (time_diffs + 1e-10)

        episode_indices.append(i + 1)
        mean_fps_list.append(np.mean(fps))
        std_fps_list.append(np.std(fps))

    fig = go.Figure()

    # Mean FPS with error bars
    fig.add_trace(go.Scatter(
        x=episode_indices,
        y=mean_fps_list,
        error_y=dict(type='data', array=std_fps_list),
        mode='lines+markers',
        name='Mean FPS',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))

    # Expected FPS line (30 Hz)
    if episode_indices:
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Expected (30 Hz)"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Frame Rate (Hz)",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_missing_data_heatmap(
    episodes_data: List[Dict[str, Any]],
    title: str = "Missing Data Analysis"
) -> go.Figure:
    """Plot heatmap showing missing data across episodes and features.

    Args:
        episodes_data: List of episode dictionaries
        title: Plot title

    Returns:
        Plotly figure
    """
    # Check for missing data
    episode_indices = []
    has_states = []
    has_actions = []
    has_timestamps = []
    nan_in_states = []
    nan_in_actions = []

    for i, episode in enumerate(episodes_data):
        episode_indices.append(i + 1)

        states = episode.get('states')
        actions = episode.get('actions')
        timestamps = episode.get('timestamps')

        has_states.append(1 if states is not None else 0)
        has_actions.append(1 if actions is not None else 0)
        has_timestamps.append(1 if timestamps is not None else 0)

        # Check for NaN within data
        if states is not None:
            nan_in_states.append(1 if np.any(np.isnan(states)) else 0)
        else:
            nan_in_states.append(0)

        if actions is not None:
            nan_in_actions.append(1 if np.any(np.isnan(actions)) else 0)
        else:
            nan_in_actions.append(0)

    # Create heatmap data
    heatmap_data = np.array([
        has_states,
        has_actions,
        has_timestamps,
        nan_in_states,
        nan_in_actions
    ])

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=episode_indices,
        y=['States Present', 'Actions Present', 'Timestamps Present',
           'NaN in States', 'NaN in Actions'],
        colorscale=[[0, 'red'], [1, 'green']],
        showscale=False,
        hovertemplate='Episode: %{x}<br>%{y}: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Data Field",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_anomaly_distribution(
    anomaly_scores: np.ndarray,
    anomaly_mask: np.ndarray,
    title: str = "Anomaly Score Distribution"
) -> go.Figure:
    """Plot distribution of anomaly scores.

    Args:
        anomaly_scores: Anomaly scores for all samples
        anomaly_mask: Boolean mask indicating anomalies
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Histogram of all scores
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        nbinsx=50,
        name='All Samples',
        marker_color='#1f77b4',
        opacity=0.7
    ))

    # Histogram of anomalies
    if np.any(anomaly_mask):
        anomaly_score_subset = anomaly_scores[anomaly_mask]
        fig.add_trace(go.Histogram(
            x=anomaly_score_subset,
            nbinsx=50,
            name='Anomalies',
            marker_color='red',
            opacity=0.7
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        template="plotly_dark",
        barmode='overlay',
        height=400
    )

    return fig


def plot_consistency_summary(
    consistency_results: Dict[str, Any],
    title: str = "Data Consistency Summary"
) -> go.Figure:
    """Create bar chart summarizing consistency metrics.

    Args:
        consistency_results: Data consistency analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    # Extract scores
    scores = consistency_results['individual_scores']

    metric_names = [
        'Action-Obs<br>Alignment',
        'Temporal<br>Consistency',
        'Missing<br>Data',
        'Anomaly<br>Detection',
        'Joint<br>Limits'
    ]

    metric_values = [
        scores['alignment'],
        scores['temporal'],
        scores['missing_data'],
        scores['anomalies'],
        scores['joint_limits']
    ]

    # Color code by score
    colors = []
    for v in metric_values:
        if v >= 0.8:
            colors.append('#2ca02c')  # Green
        elif v >= 0.6:
            colors.append('#ff7f0e')  # Orange
        else:
            colors.append('#d62728')  # Red

    fig = go.Figure(data=[go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in metric_values],
        textposition='outside'
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1.1],
        template="plotly_dark",
        height=400
    )

    return fig


def plot_alignment_issues(
    episodes_data: List[Dict[str, Any]],
    title: str = "Action-Observation Alignment Issues"
) -> go.Figure:
    """Plot alignment issues across episodes.

    Args:
        episodes_data: List of episode dictionaries
        title: Plot title

    Returns:
        Plotly figure
    """
    from src.metrics.data_consistency import DataConsistencyAnalyzer

    analyzer = DataConsistencyAnalyzer()

    episode_indices = []
    alignment_scores = []
    issue_counts = []

    for i, episode in enumerate(episodes_data):
        states = episode.get('states')
        actions = episode.get('actions')
        timestamps = episode.get('timestamps')

        if states is None or actions is None or timestamps is None:
            continue

        alignment = analyzer.check_action_observation_alignment(
            states, actions, timestamps
        )

        episode_indices.append(i + 1)
        alignment_scores.append(alignment['score'])
        issue_counts.append(alignment['n_issues'])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Alignment scores
    fig.add_trace(
        go.Scatter(
            x=episode_indices,
            y=alignment_scores,
            mode='lines+markers',
            name='Alignment Score',
            line=dict(color='#1f77b4', width=2)
        ),
        secondary_y=False
    )

    # Issue counts
    fig.add_trace(
        go.Bar(
            x=episode_indices,
            y=issue_counts,
            name='Issue Count',
            marker_color='red',
            opacity=0.6
        ),
        secondary_y=True
    )

    fig.update_xaxes(title_text="Episode")
    fig.update_yaxes(title_text="Alignment Score", secondary_y=False)
    fig.update_yaxes(title_text="Number of Issues", secondary_y=True)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400
    )

    return fig


def plot_joint_limit_violations(
    positions: np.ndarray,
    robot_config,
    title: str = "Joint Limit Violations"
) -> go.Figure:
    """Plot joint limit violations over time.

    Args:
        positions: Joint positions
        robot_config: Robot configuration
        title: Plot title

    Returns:
        Plotly figure
    """
    from src.utils.robot_models import is_within_joint_limits

    # Check limits
    within_limits = is_within_joint_limits(positions, robot_config)

    # Count violations per DOF
    violations_per_dof = []
    joint_names = [f"Joint {i+1}" for i in range(positions.shape[1])]

    for i in range(positions.shape[1]):
        lower = robot_config.joint_limits_lower[i]
        upper = robot_config.joint_limits_upper[i]

        below_lower = np.sum(positions[:, i] < lower)
        above_upper = np.sum(positions[:, i] > upper)

        violations_per_dof.append(below_lower + above_upper)

    fig = go.Figure(data=[go.Bar(
        x=joint_names,
        y=violations_per_dof,
        marker_color='red',
        text=violations_per_dof,
        textposition='outside'
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Joint",
        yaxis_title="Number of Violations",
        template="plotly_dark",
        height=400
    )

    return fig


def create_consistency_dashboard(
    consistency_results: Dict[str, Any],
    episodes_data: List[Dict[str, Any]]
) -> Dict[str, go.Figure]:
    """Create comprehensive data consistency dashboard.

    Args:
        consistency_results: Data consistency analysis results
        episodes_data: List of episode dictionaries

    Returns:
        Dictionary of figure names to Plotly figures
    """
    figures = {}

    # 1. Consistency summary
    figures['consistency_summary'] = plot_consistency_summary(
        consistency_results,
        title="Data Consistency Metrics Summary"
    )

    # 2. Temporal consistency
    figures['temporal_consistency'] = plot_temporal_consistency(
        episodes_data,
        title="Frame Rate Consistency Across Episodes"
    )

    # 3. Missing data heatmap
    figures['missing_data'] = plot_missing_data_heatmap(
        episodes_data,
        title="Missing Data Heatmap"
    )

    # 4. Anomaly distribution
    if 'anomalies' in consistency_results and 'anomaly_scores' in consistency_results['anomalies']:
        anomalies = consistency_results['anomalies']
        if len(anomalies['anomaly_scores']) > 0:
            figures['anomaly_distribution'] = plot_anomaly_distribution(
                anomalies['anomaly_scores'],
                anomalies.get('anomaly_mask', np.array([])),
                title=f"Anomaly Detection ({anomalies['n_anomalies']} found)"
            )

    # 5. Alignment issues
    figures['alignment_issues'] = plot_alignment_issues(
        episodes_data,
        title="Action-Observation Alignment by Episode"
    )

    return figures
