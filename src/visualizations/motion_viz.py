"""Motion quality visualizations."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from ..utils.config import get_config


def plot_jerk_timeseries(
    jerk_magnitude: np.ndarray,
    timestamps: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Jerk Over Time"
) -> go.Figure:
    """Plot jerk magnitude time series.

    Args:
        jerk_magnitude: Jerk magnitude array
        timestamps: Timestamp array
        threshold: Optional threshold line
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Main jerk plot
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=jerk_magnitude,
        mode='lines',
        name='Jerk Magnitude',
        line=dict(color='#1f77b4', width=2)
    ))

    # Threshold line
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold:.2f}",
            annotation_position="right"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Jerk Magnitude",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_jerk_distribution(
    jerk_magnitude: np.ndarray,
    threshold_multiplier: float = 2.0,
    title: str = "Jerk Distribution"
) -> go.Figure:
    """Plot jerk distribution histogram with threshold markers.

    Args:
        jerk_magnitude: Jerk magnitude array
        threshold_multiplier: Multiplier for threshold
        title: Plot title

    Returns:
        Plotly figure
    """
    mean_jerk = np.mean(jerk_magnitude)
    std_jerk = np.std(jerk_magnitude)
    threshold = mean_jerk + threshold_multiplier * std_jerk

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=jerk_magnitude,
        nbinsx=50,
        name='Jerk Distribution',
        marker_color='#1f77b4',
        opacity=0.7
    ))

    # Mean line
    fig.add_vline(
        x=mean_jerk,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: {mean_jerk:.2f}",
        annotation_position="top"
    )

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold_multiplier}σ): {threshold:.2f}",
        annotation_position="top"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Jerk Magnitude",
        yaxis_title="Count",
        template="plotly_dark",
        showlegend=True,
        height=400
    )

    return fig


def plot_acceleration_heatmap(
    accelerations: np.ndarray,
    timestamps: np.ndarray,
    joint_names: Optional[List[str]] = None,
    title: str = "Acceleration Heatmap"
) -> go.Figure:
    """Plot acceleration heatmap over time.

    Args:
        accelerations: Acceleration array, shape (n_frames, n_dof)
        timestamps: Timestamp array
        joint_names: Optional joint names
        title: Plot title

    Returns:
        Plotly figure
    """
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(accelerations.shape[1])]

    fig = go.Figure(data=go.Heatmap(
        z=accelerations.T,
        x=timestamps,
        y=joint_names,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Acceleration")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Joint",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_velocity_profile(
    velocities: np.ndarray,
    timestamps: np.ndarray,
    joint_names: Optional[List[str]] = None,
    title: str = "Velocity Profiles"
) -> go.Figure:
    """Plot velocity profiles for all joints.

    Args:
        velocities: Velocity array, shape (n_frames, n_dof)
        timestamps: Timestamp array
        joint_names: Optional joint names
        title: Plot title

    Returns:
        Plotly figure
    """
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(velocities.shape[1])]

    fig = go.Figure()

    # Plot each joint
    for i, joint_name in enumerate(joint_names):
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=velocities[:, i],
            mode='lines',
            name=joint_name,
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Velocity",
        template="plotly_dark",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_smoothness_comparison(
    episode_results: List[Dict[str, Any]],
    title: str = "Smoothness Scores Across Episodes"
) -> go.Figure:
    """Plot smoothness scores across episodes.

    Args:
        episode_results: List of episode analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    episodes = list(range(1, len(episode_results) + 1))
    smoothness_scores = [r['individual_scores']['smoothness'] for r in episode_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=episodes,
        y=smoothness_scores,
        mode='lines+markers',
        name='Smoothness Score',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=8)
    ))

    # Mean line
    mean_smoothness = np.mean(smoothness_scores)
    fig.add_hline(
        y=mean_smoothness,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean: {mean_smoothness:.3f}"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Smoothness Score (0-1)",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_motion_quality_radar(
    individual_scores: Dict[str, float],
    title: str = "Motion Quality Radar Chart"
) -> go.Figure:
    """Create radar chart for motion quality metrics.

    Args:
        individual_scores: Dictionary of metric scores
        title: Plot title

    Returns:
        Plotly figure
    """
    categories = list(individual_scores.keys())
    values = list(individual_scores.values())

    # Close the radar chart
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Scores',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        template="plotly_dark",
        height=500
    )

    return fig


def plot_overall_scores_distribution(
    overall_scores: List[float],
    title: str = "Overall Motion Quality Scores"
) -> go.Figure:
    """Plot distribution of overall motion quality scores.

    Args:
        overall_scores: List of overall scores
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=overall_scores,
        nbinsx=30,
        name='Score Distribution',
        marker_color='#2ca02c',
        opacity=0.7
    ))

    # Statistics
    mean_score = np.mean(overall_scores)
    median_score = np.median(overall_scores)

    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {mean_score:.3f}"
    )

    fig.add_vline(
        x=median_score,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Median: {median_score:.3f}"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Overall Score (0-1)",
        yaxis_title="Count",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_metric_comparison_bars(
    aggregated_results: Dict[str, Any],
    title: str = "Average Metric Scores"
) -> go.Figure:
    """Plot bar chart comparing average metric scores.

    Args:
        aggregated_results: Aggregated analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    metrics = ['jerk_scores', 'smoothness_scores', 'acceleration_scores',
               'velocity_scores', 'efficiency_scores']
    metric_names = ['Jerk', 'Smoothness', 'Acceleration', 'Velocity', 'Efficiency']

    mean_scores = [aggregated_results[m]['mean'] for m in metrics]
    std_scores = [aggregated_results[m]['std'] for m in metrics]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metric_names,
        y=mean_scores,
        error_y=dict(type='data', array=std_scores),
        marker_color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd'],
        text=[f"{s:.3f}" for s in mean_scores],
        textposition='outside'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Average Score (0-1)",
        yaxis_range=[0, 1.1],
        template="plotly_dark",
        height=400
    )

    return fig


def plot_jerk_outliers(
    episode_results: List[Dict[str, Any]],
    title: str = "Episodes with Jerk Outliers"
) -> go.Figure:
    """Plot episodes with jerk outliers.

    Args:
        episode_results: List of episode analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    episodes = list(range(1, len(episode_results) + 1))
    outlier_percentages = [r['jerk']['outlier_percentage'] for r in episode_results]

    fig = go.Figure()

    # Bar chart
    colors = ['red' if p > 10 else 'orange' if p > 5 else 'green'
              for p in outlier_percentages]

    fig.add_trace(go.Bar(
        x=episodes,
        y=outlier_percentages,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in outlier_percentages],
        textposition='outside'
    ))

    # Threshold lines
    fig.add_hline(y=5, line_dash="dash", line_color="orange",
                  annotation_text="Warning (5%)")
    fig.add_hline(y=10, line_dash="dash", line_color="red",
                  annotation_text="Critical (10%)")

    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Outlier Percentage (%)",
        template="plotly_dark",
        height=400
    )

    return fig


def plot_efficiency_vs_smoothness(
    episode_results: List[Dict[str, Any]],
    title: str = "Path Efficiency vs Smoothness"
) -> go.Figure:
    """Scatter plot of efficiency vs smoothness.

    Args:
        episode_results: List of episode analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    efficiency_scores = [r['individual_scores']['efficiency'] for r in episode_results]
    smoothness_scores = [r['individual_scores']['smoothness'] for r in episode_results]
    overall_scores = [r['overall_score'] for r in episode_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=efficiency_scores,
        y=smoothness_scores,
        mode='markers',
        marker=dict(
            size=10,
            color=overall_scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Overall<br>Score")
        ),
        text=[f"Episode {i+1}" for i in range(len(episode_results))],
        hovertemplate='<b>%{text}</b><br>' +
                      'Efficiency: %{x:.3f}<br>' +
                      'Smoothness: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Path Efficiency (0-1)",
        yaxis_title="Smoothness Score (0-1)",
        template="plotly_dark",
        height=500
    )

    return fig


def create_motion_quality_dashboard(
    aggregated_results: Dict[str, Any],
    sample_episode_idx: int = 0
) -> Dict[str, go.Figure]:
    """Create comprehensive motion quality dashboard.

    Args:
        aggregated_results: Aggregated analysis results
        sample_episode_idx: Index of sample episode to visualize

    Returns:
        Dictionary of figure names to Plotly figures
    """
    episode_results = aggregated_results['episode_results']

    if not episode_results:
        return {}

    # Sample episode for detailed plots
    sample_episode = episode_results[sample_episode_idx]

    figures = {}

    # 1. Overall scores distribution
    overall_scores = aggregated_results['overall_score']['scores']
    figures['overall_distribution'] = plot_overall_scores_distribution(
        overall_scores,
        title="Distribution of Overall Motion Quality Scores"
    )

    # 2. Metric comparison bars
    figures['metric_comparison'] = plot_metric_comparison_bars(
        aggregated_results,
        title="Average Scores Across All Metrics"
    )

    # 3. Radar chart for sample episode
    figures['radar_chart'] = plot_motion_quality_radar(
        sample_episode['individual_scores'],
        title=f"Motion Quality Profile (Episode {sample_episode_idx + 1})"
    )

    # 4. Jerk time series for sample episode
    if 'jerk_magnitude' in sample_episode['jerk']:
        jerk_data = sample_episode['jerk']['jerk_magnitude']
        # Need to get timestamps from somewhere - would need to pass them in
        # For now, create dummy timestamps
        timestamps = np.linspace(0, len(jerk_data)/30, len(jerk_data))
        figures['jerk_timeseries'] = plot_jerk_timeseries(
            jerk_data,
            timestamps,
            title=f"Jerk Over Time (Episode {sample_episode_idx + 1})"
        )

    # 5. Jerk outliers across episodes
    figures['jerk_outliers'] = plot_jerk_outliers(
        episode_results,
        title="Jerk Outliers Across Episodes"
    )

    # 6. Smoothness comparison
    figures['smoothness_comparison'] = plot_smoothness_comparison(
        episode_results,
        title="Smoothness Scores Across Episodes"
    )

    # 7. Efficiency vs Smoothness scatter
    figures['efficiency_vs_smoothness'] = plot_efficiency_vs_smoothness(
        episode_results,
        title="Path Efficiency vs Smoothness"
    )

    return figures
