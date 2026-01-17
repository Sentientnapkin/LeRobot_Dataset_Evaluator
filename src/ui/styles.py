"""Centralized CSS styles and design tokens for LeRobot Dataset Evaluator.

This module defines the visual design system including colors, typography,
and CSS styles for all UI components.
"""

from typing import Dict

# Design Tokens - Color Palette
COLORS: Dict[str, str] = {
    # Primary colors
    'primary': '#2563EB',
    'primary_hover': '#1D4ED8',
    'primary_light': '#3B82F6',

    # Semantic colors
    'success': '#059669',
    'success_light': '#10B981',
    'success_bg': '#064E3B',
    'warning': '#D97706',
    'warning_light': '#F59E0B',
    'warning_bg': '#78350F',
    'error': '#DC2626',
    'error_light': '#EF4444',
    'error_bg': '#7F1D1D',
    'info': '#0891B2',
    'info_light': '#06B6D4',
    'info_bg': '#164E63',

    # Background colors
    'bg_primary': '#0F172A',
    'bg_secondary': '#1E293B',
    'bg_tertiary': '#334155',
    'bg_card': '#1E293B',
    'bg_hover': '#334155',

    # Text colors
    'text_primary': '#F8FAFC',
    'text_secondary': '#94A3B8',
    'text_muted': '#64748B',

    # Border colors
    'border': '#334155',
    'border_light': '#475569',
}


def get_base_css() -> str:
    """Get base CSS styles for the application."""
    return f"""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Root variables */
        :root {{
            --color-primary: {COLORS['primary']};
            --color-success: {COLORS['success']};
            --color-warning: {COLORS['warning']};
            --color-error: {COLORS['error']};
            --color-bg-primary: {COLORS['bg_primary']};
            --color-bg-secondary: {COLORS['bg_secondary']};
            --color-text-primary: {COLORS['text_primary']};
            --color-text-secondary: {COLORS['text_secondary']};
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Base typography */
        .main .block-container {{
            padding-top: 1rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        /* Headers */
        .main-header {{
            font-size: 2.25rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}

        .sub-header {{
            font-size: 1.125rem;
            color: {COLORS['text_secondary']};
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }}

        .section-header {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {COLORS['border']};
        }}

        /* Code and monospace */
        code, .mono {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }}
    </style>
    """


def get_navigation_css() -> str:
    """Get CSS styles for the navigation bar."""
    return f"""
    <style>
        /* Navigation bar container */
        .nav-bar-container {{
            background: linear-gradient(180deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
            padding: 0.75rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            border-bottom: 1px solid {COLORS['border']};
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        .nav-buttons-row {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .nav-button-wrapper {{
            flex: 1;
            min-width: 0;
        }}

        .nav-button-wrapper button {{
            width: 100%;
            height: 48px;
            font-size: 13px;
            font-weight: 500;
            border-radius: 6px;
            border: 1px solid {COLORS['border']};
            background: rgba(255,255,255,0.05);
            color: {COLORS['text_secondary']};
            transition: all 0.2s ease;
            padding: 0.5rem 0.75rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .nav-button-wrapper button:hover {{
            background: rgba(255,255,255,0.1);
            border-color: {COLORS['primary']};
            color: {COLORS['text_primary']};
            transform: translateY(-1px);
        }}

        .nav-button-wrapper button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_hover']} 100%);
            border-color: {COLORS['primary']};
            color: {COLORS['text_primary']};
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            font-weight: 600;
        }}

        .nav-button-wrapper button[kind="primary"]:hover {{
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
            transform: translateY(-2px);
        }}
    </style>
    """


def get_metric_card_css() -> str:
    """Get CSS styles for metric cards."""
    return f"""
    <style>
        .metric-card {{
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 1.25rem;
            margin: 0.5rem 0;
            transition: all 0.2s ease;
        }}

        .metric-card:hover {{
            border-color: {COLORS['border_light']};
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        .metric-label {{
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }}

        .metric-value.small {{
            font-size: 1.5rem;
        }}

        .metric-delta {{
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}

        .metric-delta.positive {{
            color: {COLORS['success']};
        }}

        .metric-delta.negative {{
            color: {COLORS['error']};
        }}

        .metric-delta.neutral {{
            color: {COLORS['text_muted']};
        }}

        .metric-description {{
            font-size: 0.75rem;
            color: {COLORS['text_muted']};
            margin-top: 0.5rem;
        }}
    </style>
    """


def get_status_badge_css() -> str:
    """Get CSS styles for status badges."""
    return f"""
    <style>
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.375rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        .status-excellent {{
            background: {COLORS['success_bg']};
            color: {COLORS['success_light']};
        }}

        .status-excellent .status-dot {{
            background: {COLORS['success']};
            box-shadow: 0 0 8px {COLORS['success']};
        }}

        .status-good {{
            background: {COLORS['info_bg']};
            color: {COLORS['info_light']};
        }}

        .status-good .status-dot {{
            background: {COLORS['info']};
            box-shadow: 0 0 8px {COLORS['info']};
        }}

        .status-moderate {{
            background: {COLORS['warning_bg']};
            color: {COLORS['warning_light']};
        }}

        .status-moderate .status-dot {{
            background: {COLORS['warning']};
            box-shadow: 0 0 8px {COLORS['warning']};
        }}

        .status-limited, .status-poor {{
            background: {COLORS['error_bg']};
            color: {COLORS['error_light']};
        }}

        .status-limited .status-dot, .status-poor .status-dot {{
            background: {COLORS['error']};
            box-shadow: 0 0 8px {COLORS['error']};
        }}
    </style>
    """


def get_alert_css() -> str:
    """Get CSS styles for alert components."""
    return f"""
    <style>
        .alert {{
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }}

        .alert-success {{
            background: {COLORS['success_bg']};
            border-color: {COLORS['success']};
            color: {COLORS['success_light']};
        }}

        .alert-warning {{
            background: {COLORS['warning_bg']};
            border-color: {COLORS['warning']};
            color: {COLORS['warning_light']};
        }}

        .alert-error {{
            background: {COLORS['error_bg']};
            border-color: {COLORS['error']};
            color: {COLORS['error_light']};
        }}

        .alert-info {{
            background: {COLORS['info_bg']};
            border-color: {COLORS['info']};
            color: {COLORS['info_light']};
        }}

        .alert-title {{
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}

        .alert-message {{
            font-size: 0.875rem;
            opacity: 0.9;
        }}
    </style>
    """


def get_quality_indicator_css() -> str:
    """Get CSS styles for quality indicators."""
    return f"""
    <style>
        .quality-indicator {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.5rem;
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            margin: 1rem 0;
        }}

        .quality-score {{
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .quality-score.excellent {{
            color: {COLORS['success']};
        }}

        .quality-score.good {{
            color: {COLORS['info']};
        }}

        .quality-score.moderate {{
            color: {COLORS['warning']};
        }}

        .quality-score.poor {{
            color: {COLORS['error']};
        }}

        .quality-details {{
            flex: 1;
        }}

        .quality-label {{
            font-size: 1rem;
            font-weight: 600;
            color: {COLORS['text_primary']};
            margin-bottom: 0.25rem;
        }}

        .quality-description {{
            font-size: 0.875rem;
            color: {COLORS['text_secondary']};
        }}
    </style>
    """


def get_table_css() -> str:
    """Get CSS styles for tables."""
    return f"""
    <style>
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}

        .data-table th {{
            background: {COLORS['bg_tertiary']};
            color: {COLORS['text_secondary']};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid {COLORS['border']};
        }}

        .data-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid {COLORS['border']};
            color: {COLORS['text_primary']};
        }}

        .data-table tr:hover td {{
            background: {COLORS['bg_hover']};
        }}

        .data-table .numeric {{
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
        }}
    </style>
    """


def get_full_css() -> str:
    """Get all CSS styles combined."""
    return (
        get_base_css() +
        get_navigation_css() +
        get_metric_card_css() +
        get_status_badge_css() +
        get_alert_css() +
        get_quality_indicator_css() +
        get_table_css()
    )
