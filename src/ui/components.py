"""Reusable UI components for LeRobot Dataset Evaluator.

This module provides Streamlit-compatible HTML components that follow
the design system defined in styles.py.
"""

import streamlit as st
from typing import Optional, Literal
from .styles import COLORS


QualityLevel = Literal['excellent', 'good', 'moderate', 'limited', 'poor']
AlertType = Literal['success', 'warning', 'error', 'info']


def render_status_badge(
    label: str,
    quality: QualityLevel
) -> None:
    """Render a status badge with quality indicator.

    Args:
        label: Text to display in the badge
        quality: Quality level for styling
    """
    html = f"""
    <div class="status-badge status-{quality}">
        <span class="status-dot"></span>
        {label.upper()}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    description: Optional[str] = None,
    delta: Optional[str] = None,
    delta_type: Literal['positive', 'negative', 'neutral'] = 'neutral',
    small: bool = False
) -> None:
    """Render a metric card component.

    Args:
        label: Metric label/title
        value: Metric value to display
        description: Optional description text
        delta: Optional change indicator
        delta_type: Type of delta for styling
        small: Use smaller text for value
    """
    size_class = 'small' if small else ''
    delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>' if delta else ''
    desc_html = f'<div class="metric-description">{description}</div>' if description else ''

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {size_class}">{value}</div>
        {delta_html}
        {desc_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(
    metrics: list,
    columns: int = 4
) -> None:
    """Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: label, value, description (optional)
        columns: Number of columns
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                description=metric.get('description'),
                delta=metric.get('delta'),
                delta_type=metric.get('delta_type', 'neutral'),
                small=metric.get('small', False)
            )


def render_section_header(
    title: str,
    subtitle: Optional[str] = None
) -> None:
    """Render a section header.

    Args:
        title: Section title
        subtitle: Optional subtitle
    """
    subtitle_html = f'<div style="color: {COLORS["text_secondary"]}; font-size: 0.875rem; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''
    html = f"""
    <div class="section-header">
        {title}
        {subtitle_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: Optional[str] = None
) -> None:
    """Render a page header.

    Args:
        title: Page title
        subtitle: Optional subtitle
    """
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)


def render_alert(
    message: str,
    alert_type: AlertType = 'info',
    title: Optional[str] = None
) -> None:
    """Render an alert component.

    Args:
        message: Alert message
        alert_type: Type of alert for styling
        title: Optional title
    """
    title_html = f'<div class="alert-title">{title}</div>' if title else ''
    html = f"""
    <div class="alert alert-{alert_type}">
        {title_html}
        <div class="alert-message">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_quality_indicator(
    score: float,
    label: str,
    description: str,
    thresholds: Optional[dict] = None
) -> None:
    """Render a quality indicator with score and description.

    Args:
        score: Numeric score (0-1)
        label: Quality label (e.g., "Excellent")
        description: Description of what this score means
        thresholds: Optional dict with 'excellent', 'good', 'moderate' thresholds
    """
    if thresholds is None:
        thresholds = {'excellent': 0.65, 'good': 0.45, 'moderate': 0.25}

    if score >= thresholds['excellent']:
        quality_class = 'excellent'
    elif score >= thresholds['good']:
        quality_class = 'good'
    elif score >= thresholds['moderate']:
        quality_class = 'moderate'
    else:
        quality_class = 'poor'

    html = f"""
    <div class="quality-indicator">
        <div class="quality-score {quality_class}">{score:.2f}</div>
        <div class="quality-details">
            <div class="quality-label">{label}</div>
            <div class="quality-description">{description}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_quality_level(score: float, thresholds: Optional[dict] = None) -> QualityLevel:
    """Get quality level from score.

    Args:
        score: Numeric score (0-1)
        thresholds: Optional custom thresholds

    Returns:
        Quality level string
    """
    if thresholds is None:
        thresholds = {'excellent': 0.65, 'good': 0.45, 'moderate': 0.25}

    if score >= thresholds['excellent']:
        return 'excellent'
    elif score >= thresholds['good']:
        return 'good'
    elif score >= thresholds['moderate']:
        return 'moderate'
    else:
        return 'poor'


def render_coverage_metrics(
    reachable_coverage: Optional[float],
    bounding_box_coverage: float,
    coverage_quality: str
) -> None:
    """Render coverage metrics with both reachable and bounding box coverage.

    Args:
        reachable_coverage: Reachable workspace coverage (0-1), or None if unavailable
        bounding_box_coverage: Bounding box coverage (0-1)
        coverage_quality: Quality label (Excellent, Good, Moderate, Limited)
    """
    col1, col2 = st.columns(2)

    with col1:
        if reachable_coverage is not None:
            quality_lower = coverage_quality.lower()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Reachable Coverage</div>
                <div class="metric-value">{reachable_coverage * 100:.1f}%</div>
                <div class="status-badge status-{quality_lower}" style="margin-top: 0.5rem;">
                    <span class="status-dot"></span>
                    {coverage_quality.upper()}
                </div>
                <div class="metric-description">
                    Percentage of robot's actual reachable workspace visited
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Bounding Box Coverage</div>
                <div class="metric-value">{bounding_box_coverage * 100:.1f}%</div>
                <div class="metric-description">
                    Conservative estimate using full workspace bounds
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Bounding Box Coverage</div>
            <div class="metric-value small">{bounding_box_coverage * 100:.1f}%</div>
            <div class="metric-description">
                Conservative estimate using rectangular workspace bounds
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_recommendation(
    priority: str,
    metric: str,
    score: float,
    issue: str,
    advice: str,
    technique: str
) -> None:
    """Render a recommendation card.

    Args:
        priority: HIGH, MEDIUM, or LOW
        metric: Metric name
        score: Current score
        issue: Description of the issue
        advice: What to do
        technique: How to do it
    """
    priority_colors = {
        'HIGH': COLORS['error'],
        'MEDIUM': COLORS['warning'],
        'LOW': COLORS['success']
    }
    color = priority_colors.get(priority, COLORS['text_muted'])

    with st.expander(f"**{metric}** - Score: {score:.3f} ({priority} Priority)"):
        st.markdown(f"""
        <div style="border-left: 3px solid {color}; padding-left: 1rem;">
            <p><strong>Issue:</strong> {issue}</p>
            <p><strong>What to do:</strong> {advice}</p>
            <p><strong>Technique:</strong> {technique}</p>
        </div>
        """, unsafe_allow_html=True)
