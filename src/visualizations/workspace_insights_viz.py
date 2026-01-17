"""Visualizations for workspace coverage insights and analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional


def plot_exploration_density_heatmap(
    voxel_grid: np.ndarray,
    voxel_edges: tuple,
    title: str = "Exploration Density Heatmap"
) -> go.Figure:
    """Create a 3D heatmap showing exploration density.
    
    Args:
        voxel_grid: 3D voxel grid with visit counts
        voxel_edges: Tuple of bin edges
        title: Plot title
        
    Returns:
        Plotly figure
    """
    if len(voxel_edges) != 3:
        # Not 3D, return empty
        fig = go.Figure()
        fig.add_annotation(text="3D visualization only", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x_edges, y_edges, z_edges = voxel_edges
    
    # Find occupied voxels
    occupied_indices = np.where(voxel_grid > 0)
    
    if len(occupied_indices[0]) == 0:
        fig = go.Figure()
        fig.update_layout(title="No data to display", template="plotly_dark", height=600)
        return fig
    
    # Get voxel centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    
    # Extract positions and visit counts
    x_pos = x_centers[occupied_indices[0]]
    y_pos = y_centers[occupied_indices[1]]
    z_pos = z_centers[occupied_indices[2]]
    visit_counts = voxel_grid[occupied_indices]
    
    # Normalize counts for color scale
    max_count = np.max(visit_counts)
    normalized_counts = visit_counts / (max_count + 1e-10)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        mode='markers',
        marker=dict(
            size=8,
            color=visit_counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Visit<br>Count", len=0.7),
            opacity=0.8,
            cmin=0,
            cmax=max_count
        ),
        text=[f"Visits: {int(c)}" for c in visit_counts],
        hovertemplate='<b>Position</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      'Z: %{z:.3f}<br>' +
                      'Visits: %{text}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        template="plotly_dark",
        height=600
    )
    
    return fig


def plot_coverage_insights_dashboard(
    coverage_results: Dict[str, Any],
    title: str = "Workspace Coverage Insights"
) -> go.Figure:
    """Create a comprehensive dashboard showing workspace coverage insights.
    
    Args:
        coverage_results: Coverage results dictionary with exploration_metrics
        title: Plot title
        
    Returns:
        Plotly figure with subplots
    """
    exploration = coverage_results.get('exploration_metrics', {})
    
    # Create subplot with metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Coverage Summary",
            "Exploration Statistics",
            "Spatial Distribution",
            "Quality Breakdown"
        ),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Coverage gauge
    coverage_pct = coverage_results.get('coverage_percentage', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=coverage_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Coverage %"},
        delta={'reference': 20},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 10], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [10, 30], 'color': "rgba(251, 191, 36, 0.2)"},
                {'range': [30, 100], 'color': "rgba(34, 197, 94, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ), row=1, col=1)
    
    # 2. Exploration statistics
    stats_names = ["Mean Visits", "Median Visits", "Max Visits", "Hotspots"]
    stats_values = [
        exploration.get('mean_visits_per_voxel', 0),
        exploration.get('median_visits_per_voxel', 0),
        exploration.get('max_visits_per_voxel', 0),
        exploration.get('n_hotspots', 0)
    ]
    
    fig.add_trace(go.Bar(
        x=stats_names,
        y=stats_values,
        marker_color='#3b82f6',
        text=[f"{v:.1f}" if v < 100 else f"{int(v)}" for v in stats_values],
        textposition='outside'
    ), row=1, col=2)
    
    # 3. Spatial distribution (center of mass)
    com = exploration.get('center_of_mass', (0, 0, 0))
    fig.add_trace(go.Scatter(
        x=['X', 'Y', 'Z'],
        y=[com[0], com[1], com[2]],
        mode='markers+lines',
        marker=dict(size=15, color='#10b981'),
        line=dict(color='#10b981', width=2),
        name="Center of Mass"
    ), row=2, col=1)
    
    # 4. Quality breakdown
    score = coverage_results.get('score', 0)
    uniformity = exploration.get('exploration_uniformity', 0)
    spread = exploration.get('spread_score', 0)
    
    quality_names = ["Overall", "Uniformity", "Spread"]
    quality_values = [score, uniformity, spread]
    
    colors = ['#1f77b4' if v > 0.5 else '#ef4444' for v in quality_values]
    
    fig.add_trace(go.Bar(
        x=quality_names,
        y=quality_values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in quality_values],
        textposition='outside'
    ), row=2, col=2)
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=700,
        showlegend=False
    )
    
    return fig
