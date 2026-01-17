"""UI module for LeRobot Dataset Evaluator.

This module provides centralized styling and reusable UI components
for the Streamlit application.
"""

from .styles import (
    COLORS,
    get_base_css,
    get_navigation_css,
    get_metric_card_css,
    get_status_badge_css,
    get_full_css,
)

from .components import (
    render_status_badge,
    render_metric_card,
    render_metric_row,
    render_section_header,
    render_page_header,
    render_alert,
    render_quality_indicator,
)

__all__ = [
    # Styles
    'COLORS',
    'get_base_css',
    'get_navigation_css',
    'get_metric_card_css',
    'get_status_badge_css',
    'get_full_css',
    # Components
    'render_status_badge',
    'render_metric_card',
    'render_metric_row',
    'render_section_header',
    'render_page_header',
    'render_alert',
    'render_quality_indicator',
]
