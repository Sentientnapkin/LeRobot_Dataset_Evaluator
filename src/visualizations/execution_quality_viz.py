"""Visualization functions for execution quality metrics.

This module provides visualization functions specifically designed for teleoperation
quality assessment using the ExecutionQualityAnalyzer metrics.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any


# Color scheme for quality levels
COLOR_EXCELLENT = '#00C851'  # Green
COLOR_GOOD = '#7CB342'       # Light green
COLOR_ACCEPTABLE = '#FFB300' # Yellow/Orange
COLOR_POOR = '#FF4444'       # Red
COLOR_NEUTRAL = '#90A4AE'    # Grey


def get_quality_color(score: float) -> str:
    """Get color based on quality score.

    Args:
        score: Quality score (0-1)

    Returns:
        Hex color string
    """
    if score > 0.75:
        return COLOR_EXCELLENT
    elif score > 0.65:
        return COLOR_GOOD
    elif score > 0.45:
        return COLOR_ACCEPTABLE
    else:
        return COLOR_POOR


def plot_quality_gauge(overall_score: float, title: str = "Overall Execution Quality") -> go.Figure:
    """Create a gauge chart showing overall quality score.

    Args:
        overall_score: Overall execution quality score (0-1)
        title: Chart title

    Returns:
        Plotly figure
    """
    # Determine quality level
    if overall_score > 0.65:
        quality_text = "EXCELLENT"
        quality_color = COLOR_EXCELLENT
    elif overall_score > 0.45:
        quality_text = "ACCEPTABLE"
        quality_color = COLOR_ACCEPTABLE
    else:
        quality_text = "NEEDS IMPROVEMENT"
        quality_color = COLOR_POOR

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score,
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 0.65, 'increasing': {'color': COLOR_EXCELLENT}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': quality_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.45], 'color': '#FFCDD2'},      # Light red
                {'range': [0.45, 0.65], 'color': '#FFF9C4'},   # Light yellow
                {'range': [0.65, 1.0], 'color': '#C8E6C9'}     # Light green
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.45
            }
        },
        number={'font': {'size': 50}},
    ))

    # Add quality label
    fig.add_annotation(
        text=quality_text,
        xref="paper", yref="paper",
        x=0.5, y=0.2,
        showarrow=False,
        font=dict(size=18, color=quality_color, family="Arial Black")
    )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def plot_metric_comparison_bars(results: Dict[str, Any], title: str = "Metric Comparison") -> go.Figure:
    """Create bar chart comparing the 4 execution quality metrics.

    Args:
        results: Results from ExecutionQualityAnalyzer.analyze_multiple_episodes()
        title: Chart title

    Returns:
        Plotly figure
    """
    # Extract metric names and scores
    metrics = ['movement_units_scores', 'hesitation_scores', 'submovement_scores', 'jerk_scores']
    metric_names = ['Movement Units', 'Hesitations', 'Submovements', 'Jerk']

    mean_scores = [results[m]['mean'] for m in metrics]
    colors = [get_quality_color(score) for score in mean_scores]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metric_names,
        y=mean_scores,
        marker=dict(color=colors, line=dict(color='rgb(8,48,107)', width=1.5)),
        text=[f"{score:.3f}" for score in mean_scores],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
    ))

    # Add threshold lines
    fig.add_hline(y=0.65, line_dash="dash", line_color="green",
                  annotation_text="Good threshold (0.65)")
    fig.add_hline(y=0.45, line_dash="dash", line_color="orange",
                  annotation_text="Acceptable threshold (0.45)")

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        hovermode='closest',
        showlegend=False,
        height=400
    )

    return fig


def plot_quality_trends(episode_results: List[Dict[str, Any]], title: str = "Quality Trends Across Episodes") -> go.Figure:
    """Create line chart showing overall quality score across episodes.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    episodes = list(range(len(episode_results)))
    scores = [ep['overall_score'] for ep in episode_results]

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Color code points by quality
    colors = [get_quality_color(score) for score in scores]

    fig = go.Figure()

    # Add scatter plot with colored points
    fig.add_trace(go.Scatter(
        x=episodes,
        y=scores,
        mode='markers+lines',
        marker=dict(size=8, color=colors, line=dict(width=1, color='DarkSlateGrey')),
        line=dict(color='lightgray', width=1),
        name='Episode Score',
        hovertemplate='Episode %{x}<br>Score: %{y:.3f}<extra></extra>'
    ))

    # Add mean line
    fig.add_hline(y=mean_score, line_dash="solid", line_color="blue", line_width=2,
                  annotation_text=f"Mean: {mean_score:.3f}")

    # Add ±1σ bands
    fig.add_hrect(y0=mean_score - std_score, y1=mean_score + std_score,
                  fillcolor="lightblue", opacity=0.2, line_width=0,
                  annotation_text="±1σ", annotation_position="right")

    # Add threshold lines
    fig.add_hline(y=0.65, line_dash="dash", line_color="green",
                  annotation_text="Good", annotation_position="left")
    fig.add_hline(y=0.45, line_dash="dash", line_color="orange",
                  annotation_text="Acceptable", annotation_position="left")

    fig.update_layout(
        title=title,
        xaxis_title="Episode Number",
        yaxis_title="Overall Quality Score",
        yaxis=dict(range=[0, 1.05]),
        hovermode='closest',
        height=400
    )

    return fig


def plot_episode_heatmap(episode_results: List[Dict[str, Any]], title: str = "Episode Quality Heatmap") -> go.Figure:
    """Create grid heatmap of all episodes colored by overall score.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    n_episodes = len(episode_results)

    # Determine grid size (roughly square)
    n_cols = int(np.ceil(np.sqrt(n_episodes)))
    n_rows = int(np.ceil(n_episodes / n_cols))

    # Create grid matrix
    grid = np.full((n_rows, n_cols), np.nan)
    hover_text = [['' for _ in range(n_cols)] for _ in range(n_rows)]

    for i, ep in enumerate(episode_results):
        row = i // n_cols
        col = i % n_cols
        grid[row, col] = ep['overall_score']

        # Create hover text with all scores
        hover_text[row][col] = (
            f"Episode {i}<br>"
            f"Overall: {ep['overall_score']:.3f}<br>"
            f"Movement Units: {ep['individual_scores']['movement_units']:.3f}<br>"
            f"Hesitations: {ep['individual_scores']['hesitations']:.3f}<br>"
            f"Submovements: {ep['individual_scores']['submovements']:.3f}<br>"
            f"Jerk: {ep['individual_scores']['jerk']:.3f}"
        )

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0.0, COLOR_POOR],
            [0.45, COLOR_ACCEPTABLE],
            [0.65, COLOR_GOOD],
            [1.0, COLOR_EXCELLENT]
        ],
        colorbar=dict(title="Score", tickvals=[0, 0.45, 0.65, 1.0]),
        zmid=0.55
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=400
    )

    return fig


def plot_metric_distributions(results: Dict[str, Any], title: str = "Metric Score Distributions") -> go.Figure:
    """Create 2x2 subplot with histograms for each metric.

    Args:
        results: Results from ExecutionQualityAnalyzer.analyze_multiple_episodes()
        title: Chart title

    Returns:
        Plotly figure
    """
    metrics = ['movement_units_scores', 'hesitation_scores', 'submovement_scores', 'jerk_scores']
    metric_names = ['Movement Units', 'Hesitations', 'Submovements', 'Jerk']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_names,
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (metric, name, pos) in zip(metrics, metric_names, positions):
        scores = results[metric]['scores']
        mean = results[metric]['mean']
        median = results[metric]['median']

        row, col = pos

        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=20,
                marker=dict(color=get_quality_color(mean), opacity=0.7),
                name=name,
                showlegend=False,
                hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )

        # Add mean line
        fig.add_vline(x=mean, line_dash="solid", line_color="red", line_width=2,
                     row=row, col=col)

        # Add median line
        fig.add_vline(x=median, line_dash="dash", line_color="blue", line_width=1.5,
                     row=row, col=col)

        # Update axes
        fig.update_xaxes(title_text="Score", range=[0, 1], row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )

    return fig


def plot_movement_units_analysis(episode_results: List[Dict[str, Any]], title: str = "Movement Units Analysis") -> go.Figure:
    """Create scatter plot of movement units per second vs overall score.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    nmu_per_sec = [ep['movement_units']['nmu_per_second'] for ep in episode_results]
    overall_scores = [ep['overall_score'] for ep in episode_results]
    colors = [get_quality_color(score) for score in overall_scores]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=nmu_per_sec,
        y=overall_scores,
        mode='markers',
        marker=dict(size=10, color=colors, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
        text=[f"Episode {i}" for i in range(len(episode_results))],
        hovertemplate='%{text}<br>NMU/sec: %{x:.2f}<br>Overall Score: %{y:.3f}<extra></extra>'
    ))

    # Add trendline
    if len(nmu_per_sec) > 1:
        z = np.polyfit(nmu_per_sec, overall_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(nmu_per_sec), max(nmu_per_sec), 100)
        y_trend = p(x_trend)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Trend',
            showlegend=True,
            hovertemplate='Trend line<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Movement Units per Second",
        yaxis_title="Overall Quality Score",
        yaxis=dict(range=[0, 1.05]),
        hovermode='closest',
        height=400
    )

    return fig


def plot_hesitation_analysis(episode_results: List[Dict[str, Any]], title: str = "Hesitation Rate Analysis") -> go.Figure:
    """Create bar chart of hesitation rate per episode.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    episodes = list(range(len(episode_results)))
    hesitation_rates = [ep['hesitations']['hesitation_rate'] for ep in episode_results]
    hesitation_scores = [ep['individual_scores']['hesitations'] for ep in episode_results]
    colors = [get_quality_color(score) for score in hesitation_scores]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=episodes,
        y=hesitation_rates,
        marker=dict(color=colors, line=dict(color='rgb(8,48,107)', width=1)),
        text=[f"{rate:.2f}" for rate in hesitation_rates],
        textposition='outside',
        textangle=0,
        hovertemplate='Episode %{x}<br>Hesitations/sec: %{y:.2f}<br>Score: %{text}<extra></extra>',
        customdata=hesitation_scores
    ))

    # Add acceptable threshold line (around 2-3 hesitations/sec is reasonable)
    mean_rate = np.mean(hesitation_rates)
    fig.add_hline(y=mean_rate, line_dash="dash", line_color="blue",
                  annotation_text=f"Mean: {mean_rate:.2f}/sec")

    fig.update_layout(
        title=title,
        xaxis_title="Episode Number",
        yaxis_title="Hesitation Rate (reversals/second)",
        hovermode='closest',
        showlegend=False,
        height=400
    )

    return fig


def plot_submovement_patterns(episode_results: List[Dict[str, Any]], title: str = "Submovement Patterns") -> go.Figure:
    """Create scatter plot of submovement index vs overall score.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    submovement_indices = [ep['submovements']['submovement_index'] for ep in episode_results]
    overall_scores = [ep['overall_score'] for ep in episode_results]
    colors = [get_quality_color(score) for score in overall_scores]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=submovement_indices,
        y=overall_scores,
        mode='markers',
        marker=dict(size=10, color=colors, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
        text=[f"Episode {i}" for i in range(len(episode_results))],
        hovertemplate='%{text}<br>Submovement Index: %{x:.2f}<br>Overall Score: %{y:.3f}<extra></extra>'
    ))

    # Add percentile markers
    p50 = np.percentile(submovement_indices, 50)
    p75 = np.percentile(submovement_indices, 75)
    p95 = np.percentile(submovement_indices, 95)

    fig.add_vline(x=p50, line_dash="dot", line_color="gray",
                  annotation_text="p50", annotation_position="top")
    fig.add_vline(x=p75, line_dash="dot", line_color="gray",
                  annotation_text="p75", annotation_position="top")
    fig.add_vline(x=p95, line_dash="dot", line_color="gray",
                  annotation_text="p95", annotation_position="top")

    fig.update_layout(
        title=title,
        xaxis_title="Submovement Index (peaks/second)",
        yaxis_title="Overall Quality Score",
        yaxis=dict(range=[0, 1.05]),
        hovermode='closest',
        height=400
    )

    return fig


def plot_jerk_distribution_with_outliers(episode_results: List[Dict[str, Any]], title: str = "Jerk Score Distribution") -> go.Figure:
    """Create histogram of jerk scores with outlier highlighting.

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    jerk_scores = [ep['individual_scores']['jerk'] for ep in episode_results]

    # Get outlier information from first episode that has it
    has_outlier_data = any('outliers' in ep['jerk'] for ep in episode_results)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=jerk_scores,
        nbinsx=25,
        marker=dict(color=COLOR_GOOD, opacity=0.7),
        name='Jerk Scores',
        hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))

    # Add percentile lines
    p50 = np.percentile(jerk_scores, 50)
    p75 = np.percentile(jerk_scores, 75)
    p95 = np.percentile(jerk_scores, 95)

    fig.add_vline(x=p50, line_dash="solid", line_color="blue", line_width=2,
                  annotation_text=f"Median: {p50:.3f}")
    fig.add_vline(x=p75, line_dash="dash", line_color="orange", line_width=1.5,
                  annotation_text=f"p75: {p75:.3f}")
    fig.add_vline(x=p95, line_dash="dot", line_color="red", line_width=1.5,
                  annotation_text=f"p95: {p95:.3f}")

    # Add outlier threshold if available
    if has_outlier_data:
        # Calculate mean outlier percentage
        outlier_percentages = [ep['jerk']['outliers']['outlier_percentage']
                              for ep in episode_results if 'outliers' in ep['jerk']]
        mean_outlier_pct = np.mean(outlier_percentages)

        fig.add_annotation(
            text=f"Avg Outliers: {mean_outlier_pct:.1f}%",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=12, color="red"),
            bgcolor="white",
            bordercolor="red",
            borderwidth=1
        )

    fig.update_layout(
        title=title,
        xaxis_title="Jerk Score",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )

    return fig


def plot_control_confidence_composite(episode_results: List[Dict[str, Any]], title: str = "Control Confidence Score") -> go.Figure:
    """Create combined metric showing control confidence (hesitation + submovement).

    Args:
        episode_results: List of episode results from analyzer
        title: Chart title

    Returns:
        Plotly figure
    """
    episodes = list(range(len(episode_results)))

    # Calculate composite control confidence score
    confidence_scores = [
        (ep['individual_scores']['hesitations'] + ep['individual_scores']['submovements']) / 2
        for ep in episode_results
    ]

    colors = [get_quality_color(score) for score in confidence_scores]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=episodes,
        y=confidence_scores,
        marker=dict(color=colors, line=dict(color='rgb(8,48,107)', width=1)),
        text=[f"{score:.3f}" for score in confidence_scores],
        textposition='outside',
        hovertemplate='Episode %{x}<br>Control Confidence: %{y:.3f}<extra></extra>'
    ))

    # Add threshold zones
    fig.add_hrect(y0=0.65, y1=1.0, fillcolor="green", opacity=0.1, line_width=0,
                  annotation_text="Good", annotation_position="right")
    fig.add_hrect(y0=0.45, y1=0.65, fillcolor="yellow", opacity=0.1, line_width=0,
                  annotation_text="Acceptable", annotation_position="right")
    fig.add_hrect(y0=0.0, y1=0.45, fillcolor="red", opacity=0.1, line_width=0,
                  annotation_text="Poor", annotation_position="right")

    # Add mean line
    mean_confidence = np.mean(confidence_scores)
    fig.add_hline(y=mean_confidence, line_dash="dash", line_color="blue", line_width=2,
                  annotation_text=f"Mean: {mean_confidence:.3f}")

    fig.update_layout(
        title=title,
        xaxis_title="Episode Number",
        yaxis_title="Control Confidence Score",
        yaxis=dict(range=[0, 1.1]),
        showlegend=False,
        height=400
    )

    return fig


# Keep the old plot_overall_scores_distribution for backward compatibility
def plot_overall_scores_distribution(overall_scores: List[float], title: str = "Overall Score Distribution") -> go.Figure:
    """Create histogram of overall quality scores.

    Args:
        overall_scores: List of overall scores
        title: Chart title

    Returns:
        Plotly figure
    """
    mean_score = np.mean(overall_scores)
    median_score = np.median(overall_scores)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=overall_scores,
        nbinsx=20,
        marker=dict(color=get_quality_color(mean_score), opacity=0.7),
        hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))

    fig.add_vline(x=mean_score, line_dash="solid", line_color="red", line_width=2,
                  annotation_text=f"Mean: {mean_score:.3f}")
    fig.add_vline(x=median_score, line_dash="dash", line_color="blue", line_width=1.5,
                  annotation_text=f"Median: {median_score:.3f}")

    # Add threshold lines
    fig.add_vline(x=0.65, line_dash="dot", line_color="green",
                  annotation_text="Good")
    fig.add_vline(x=0.45, line_dash="dot", line_color="orange",
                  annotation_text="Acceptable")

    fig.update_layout(
        title=title,
        xaxis_title="Overall Score",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )

    return fig
