"""Success Indicators Visualizations.

This module provides visualization functions for success indicator analysis.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np


# Color scheme
COLORS = {
    'excellent': '#059669',
    'good': '#0891B2',
    'marginal': '#D97706',
    'poor': '#DC2626',
    'primary': '#2563EB',
    'secondary': '#64748B',
    'background': '#1E293B',
    'grid': '#334155',
    'text': '#F8FAFC'
}


def plot_episode_length_histogram(results: Dict[str, Any]) -> go.Figure:
    """Create episode length distribution histogram.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    length_stats = results['length_stats']
    lengths = np.array(length_stats['lengths'])

    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=lengths,
        nbinsx=25,
        marker_color=COLORS['primary'],
        opacity=0.8,
        name='Episodes'
    ))

    # Add mean line
    mean_val = length_stats['mean']
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLORS['excellent'],
        annotation_text=f"Mean: {mean_val:.0f}",
        annotation_position="top"
    )

    # Add median line
    median_val = length_stats['median']
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=COLORS['marginal'],
        annotation_text=f"Median: {median_val:.0f}",
        annotation_position="bottom"
    )

    fig.update_layout(
        title="Episode Length Distribution",
        xaxis_title="Episode Length (frames)",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        showlegend=False,
        height=400
    )

    return fig


def plot_success_rate_pie(results: Dict[str, Any]) -> go.Figure:
    """Create success rate pie chart by classification.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    success_rates = results['success_rates']
    counts = success_rates['counts']

    labels = ['Excellent', 'Good', 'Marginal', 'Poor']
    values = [
        counts['excellent'],
        counts['good'],
        counts['marginal'],
        counts['poor']
    ]
    colors = [
        COLORS['excellent'],
        COLORS['good'],
        COLORS['marginal'],
        COLORS['poor']
    ]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='percent+label',
        textfont_size=12,
        pull=[0.05, 0, 0, 0.05]
    )])

    fig.update_layout(
        title="Episode Quality Distribution",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400,
        annotations=[dict(
            text=f"{success_rates['overall_percentage']:.0f}%<br>Success",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )]
    )

    return fig


def plot_quality_score_distribution(results: Dict[str, Any]) -> go.Figure:
    """Create quality score distribution with threshold lines.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    quality_stats = results['quality_stats']
    scores = np.array(quality_stats['scores'])
    thresholds = results['thresholds']

    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color=COLORS['primary'],
        opacity=0.7,
        name='Quality Scores'
    ))

    # Add threshold lines
    fig.add_vline(
        x=thresholds['excellent'],
        line_dash="solid",
        line_color=COLORS['excellent'],
        annotation_text="Excellent",
        annotation_position="top right"
    )
    fig.add_vline(
        x=thresholds['success'],
        line_dash="solid",
        line_color=COLORS['good'],
        annotation_text="Success",
        annotation_position="top right"
    )
    fig.add_vline(
        x=thresholds['marginal'],
        line_dash="solid",
        line_color=COLORS['marginal'],
        annotation_text="Marginal",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Quality Score Distribution with Thresholds",
        xaxis_title="Quality Score",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        showlegend=False,
        height=400,
        xaxis=dict(range=[0, 1])
    )

    return fig


def plot_quality_scatter(results: Dict[str, Any]) -> go.Figure:
    """Create scatter plot of episode quality vs length.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    episode_info = results['episode_info']

    episodes = [ep.episode_idx for ep in episode_info]
    lengths = [ep.length for ep in episode_info]
    scores = [ep.quality_score for ep in episode_info]
    classifications = [ep.classification for ep in episode_info]

    color_map = {
        'excellent': COLORS['excellent'],
        'good': COLORS['good'],
        'marginal': COLORS['marginal'],
        'poor': COLORS['poor']
    }
    colors = [color_map[c] for c in classifications]

    fig = go.Figure()

    for classification in ['excellent', 'good', 'marginal', 'poor']:
        mask = [c == classification for c in classifications]
        fig.add_trace(go.Scatter(
            x=[l for l, m in zip(lengths, mask) if m],
            y=[s for s, m in zip(scores, mask) if m],
            mode='markers',
            name=classification.capitalize(),
            marker=dict(
                color=color_map[classification],
                size=8,
                opacity=0.7
            ),
            text=[f"Episode {e}" for e, m in zip(episodes, mask) if m],
            hovertemplate="<b>%{text}</b><br>Length: %{x}<br>Quality: %{y:.3f}<extra></extra>"
        ))

    # Add correlation line
    correlation = results['length_quality_correlation']
    if abs(correlation['coefficient']) > 0.1:
        z = np.polyfit(lengths, scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lengths), max(lengths), 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name=f"Trend (r={correlation['coefficient']:.2f})",
            line=dict(color=COLORS['secondary'], dash='dash', width=2)
        ))

    fig.update_layout(
        title="Episode Length vs Quality Score",
        xaxis_title="Episode Length (frames)",
        yaxis_title="Quality Score",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_best_worst_comparison(results: Dict[str, Any]) -> go.Figure:
    """Create comparison chart of best vs worst episodes.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    best = results['best_episodes']
    worst = results['worst_episodes']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Best Episodes", "Worst Episodes"),
        horizontal_spacing=0.15
    )

    # Best episodes
    fig.add_trace(go.Bar(
        x=[f"Ep {ep['episode_idx']}" for ep in best],
        y=[ep['quality_score'] for ep in best],
        marker_color=COLORS['excellent'],
        text=[f"{ep['quality_score']:.3f}" for ep in best],
        textposition='outside',
        name='Best',
        showlegend=False
    ), row=1, col=1)

    # Worst episodes
    fig.add_trace(go.Bar(
        x=[f"Ep {ep['episode_idx']}" for ep in worst],
        y=[ep['quality_score'] for ep in worst],
        marker_color=COLORS['poor'],
        text=[f"{ep['quality_score']:.3f}" for ep in worst],
        textposition='outside',
        name='Worst',
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title="Best vs Worst Performing Episodes",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400,
        yaxis=dict(title="Quality Score", range=[0, 1]),
        yaxis2=dict(title="Quality Score", range=[0, 1])
    )

    return fig


def plot_classification_bar(results: Dict[str, Any]) -> go.Figure:
    """Create stacked bar chart of episode classifications.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    success_rates = results['success_rates']
    counts = success_rates['counts']
    percentages = success_rates['percentages']

    categories = ['Excellent', 'Good', 'Marginal', 'Poor']
    values = [
        counts['excellent'],
        counts['good'],
        counts['marginal'],
        counts['poor']
    ]
    colors = [
        COLORS['excellent'],
        COLORS['good'],
        COLORS['marginal'],
        COLORS['poor']
    ]

    fig = go.Figure()

    for i, (cat, val, col) in enumerate(zip(categories, values, colors)):
        pct = percentages[cat.lower()]
        fig.add_trace(go.Bar(
            x=[cat],
            y=[val],
            name=cat,
            marker_color=col,
            text=[f"{val} ({pct:.1f}%)"],
            textposition='outside'
        ))

    fig.update_layout(
        title="Episodes by Quality Classification",
        xaxis_title="Classification",
        yaxis_title="Number of Episodes",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=400,
        showlegend=False,
        bargap=0.3
    )

    return fig


def plot_length_quality_correlation(results: Dict[str, Any]) -> go.Figure:
    """Create visualization of length-quality correlation.

    Args:
        results: Success indicator analysis results

    Returns:
        Plotly figure
    """
    episode_info = results['episode_info']
    correlation = results['length_quality_correlation']

    lengths = np.array([ep.length for ep in episode_info])
    scores = np.array([ep.quality_score for ep in episode_info])

    fig = go.Figure()

    # Add 2D histogram for density
    fig.add_trace(go.Histogram2dContour(
        x=lengths,
        y=scores,
        colorscale='Blues',
        showscale=False,
        contours=dict(showlines=False),
        ncontours=10,
        name='Density'
    ))

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=lengths,
        y=scores,
        mode='markers',
        marker=dict(
            color=scores,
            colorscale='RdYlGn',
            size=6,
            opacity=0.6,
            showscale=True,
            colorbar=dict(title="Quality")
        ),
        name='Episodes'
    ))

    # Add trend line
    z = np.polyfit(lengths, scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(lengths), max(lengths), 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name=f"Trend (r={correlation['coefficient']:.3f})",
        line=dict(color='white', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f"Length-Quality Correlation: {correlation['strength'].capitalize()} {correlation['direction'].capitalize()} (r={correlation['coefficient']:.3f})",
        xaxis_title="Episode Length (frames)",
        yaxis_title="Quality Score",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=450
    )

    return fig


def create_success_dashboard(results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """Create all success indicator visualizations.

    Args:
        results: Success indicator analysis results

    Returns:
        Dictionary of figure names to Plotly figures
    """
    return {
        'length_histogram': plot_episode_length_histogram(results),
        'success_rate_pie': plot_success_rate_pie(results),
        'quality_distribution': plot_quality_score_distribution(results),
        'quality_scatter': plot_quality_scatter(results),
        'best_worst': plot_best_worst_comparison(results),
        'classification_bar': plot_classification_bar(results),
        'correlation': plot_length_quality_correlation(results)
    }
