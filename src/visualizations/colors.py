"""Shared color schemes for visualizations.

This module provides consistent color constants used across all visualization modules.
"""

# Primary color scheme for classification/quality levels
QUALITY_COLORS = {
    'excellent': '#059669',  # Green
    'good': '#0891B2',       # Cyan
    'marginal': '#D97706',   # Amber
    'poor': '#DC2626',       # Red
}

# UI and theme colors
UI_COLORS = {
    'primary': '#2563EB',    # Blue
    'secondary': '#64748B',  # Slate
    'background': '#1E293B', # Dark slate
    'grid': '#334155',       # Slate gray
    'text': '#F8FAFC',       # Near white
}

# Trajectory visualization colors
TRAJECTORY_COLORS = {
    'trajectory': '#2563EB', # Blue
    'current': '#DC2626',    # Red
    'start': '#059669',      # Green
    'end': '#D97706',        # Amber
}

# Combined color scheme for success_viz.py compatibility
COLORS = {
    **QUALITY_COLORS,
    **UI_COLORS,
}

# Combined color scheme for trajectory_playback.py compatibility
TRAJECTORY_PLAYBACK_COLORS = {
    **TRAJECTORY_COLORS,
    'background': UI_COLORS['background'],
    'grid': UI_COLORS['grid'],
    'text': UI_COLORS['text'],
    'secondary': UI_COLORS['secondary'],
}

# Colorscale for multi-dimensional data
DIMENSION_COLORS = [
    '#2563EB',  # Blue
    '#059669',  # Green
    '#D97706',  # Amber
    '#DC2626',  # Red
    '#8B5CF6',  # Purple
    '#EC4899',  # Pink
]


def get_quality_color(classification: str) -> str:
    """Get color for a quality classification.

    Args:
        classification: One of 'excellent', 'good', 'marginal', 'poor'

    Returns:
        Hex color string
    """
    return QUALITY_COLORS.get(classification.lower(), UI_COLORS['secondary'])


def get_dimension_color(index: int) -> str:
    """Get color for a dimension index (cycles through available colors).

    Args:
        index: Dimension index

    Returns:
        Hex color string
    """
    return DIMENSION_COLORS[index % len(DIMENSION_COLORS)]
