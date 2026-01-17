"""Spatial coverage visualizations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

from ..utils.config import get_config


def plot_workspace_coverage_3d(
    voxel_grid: np.ndarray,
    voxel_edges: tuple,
    title: str = "Workspace Coverage Heatmap",
    all_positions: Optional[np.ndarray] = None,
    data_space: str = 'cartesian'
) -> go.Figure:
    """Plot 3D workspace coverage heatmap.

    Args:
        voxel_grid: 3D voxel occupancy grid (or higher dimensional for joint space)
        voxel_edges: Tuple of bin edges (x_edges, y_edges, z_edges) or more for joint space
        title: Plot title
        all_positions: Original positions array (needed for PCA projection of joint space)
        data_space: 'cartesian' or 'joint'

    Returns:
        Plotly figure
    """
    # Handle joint space (more than 3 dimensions) by using PCA projection
    if len(voxel_edges) > 3 or data_space == 'joint':
        if all_positions is None:
            # Fallback if positions not provided
            fig = go.Figure()
            fig.add_annotation(
                text="Joint space data detected. Original positions needed for 3D visualization.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=title + " (Joint Space)",
                template="plotly_dark",
                height=600
            )
            return fig
        
        # Use PCA to project joint space to 3D
        from sklearn.decomposition import PCA
        
        # Sample positions if too many (for performance)
        n_samples = len(all_positions)
        max_samples = 10000
        if n_samples > max_samples:
            indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
            positions_sample = all_positions[indices]
        else:
            positions_sample = all_positions
        
        # Apply PCA to project to 3D
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(positions_sample)
        explained_variance = pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance) * 100
        
        # Analyze which joints contribute most to each principal component
        # PCA components show how each original dimension (joint) contributes
        n_joints = all_positions.shape[1]
        pca_components = pca.components_  # Shape: (3, n_joints)
        
        def get_top_joints(pc_idx, top_k=2):
            """Get the top contributing joints for a principal component."""
            component = pca_components[pc_idx]
            abs_component = np.abs(component)
            top_indices = np.argsort(abs_component)[-top_k:][::-1]
            top_contributions = component[top_indices]
            return top_indices, top_contributions
        
        # Get top contributing joints for each PC
        pc1_joints, pc1_contribs = get_top_joints(0)
        pc2_joints, pc2_contribs = get_top_joints(1)
        pc3_joints, pc3_contribs = get_top_joints(2)
        
        def format_joint_label(joint_indices, contributions):
            """Format joint label showing which joints contribute most."""
            labels = []
            for idx, contrib in zip(joint_indices, contributions):
                sign = "+" if contrib > 0 else "-"
                labels.append(f"J{idx+1}{sign}")
            return ", ".join(labels)
        
        pc1_label = format_joint_label(pc1_joints, pc1_contribs)
        pc2_label = format_joint_label(pc2_joints, pc2_contribs)
        pc3_label = format_joint_label(pc3_joints, pc3_contribs)
        
        # Create 3D voxel grid in PCA space
        # Use fixed number of bins for consistent visualization
        # Adaptive binning based on data spread
        n_bins = 25  # Target number of bins per dimension
        
        # Calculate ranges
        x_min, x_max = positions_3d[:, 0].min(), positions_3d[:, 0].max()
        y_min, y_max = positions_3d[:, 1].min(), positions_3d[:, 1].max()
        z_min, z_max = positions_3d[:, 2].min(), positions_3d[:, 2].max()
        
        # Use fixed binning (simpler and more consistent)
        x_bins = n_bins
        y_bins = n_bins
        z_bins = n_bins
        
        # Create histogram edges
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(y_min, y_max, y_bins + 1)
        z_edges = np.linspace(z_min, z_max, z_bins + 1)
        
        # Create 3D histogram
        hist_3d, _ = np.histogramdd(
            positions_3d,
            bins=[x_bins, y_bins, z_bins],
            range=[(x_edges[0], x_edges[-1]), (y_edges[0], y_edges[-1]), (z_edges[0], z_edges[-1])]
        )
        
        # Find occupied voxels
        occupied_indices = np.where(hist_3d > 0)
        
        if len(occupied_indices[0]) == 0:
            fig = go.Figure()
            fig.update_layout(title="No data to display", template="plotly_dark", height=600)
            return fig
        
        # Get voxel centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        
        # Extract occupied voxel positions and counts
        x_pos = x_centers[occupied_indices[0]]
        y_pos = y_centers[occupied_indices[1]]
        z_pos = z_centers[occupied_indices[2]]
        counts = hist_3d[occupied_indices]
        
        # Normalize counts for color
        normalized_counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-10)
        
        # Create visualization with enhanced PCA axis labels
        fig = go.Figure(data=[go.Scatter3d(
            x=x_pos,
            y=y_pos,
            z=z_pos,
            mode='markers',
            marker=dict(
                size=5,
                color=normalized_counts,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Visit<br>Count"),
                opacity=0.8
            ),
            text=[f"Visits: {int(c)}" for c in counts],
            hovertemplate='<b>Joint Space Coverage (PCA Projection)</b><br>' +
                          f'<b>PC1</b> ({pc1_label}): %{{x:.3f}}<br>' +
                          f'<b>PC2</b> ({pc2_label}): %{{y:.3f}}<br>' +
                          f'<b>PC3</b> ({pc3_label}): %{{z:.3f}}<br>' +
                          '%{text}<br>' +
                          '<extra></extra>'
        )])
        
        # Create informative axis titles with joint contributions
        # Format: "PC1 (XX% var) | Top: J1+, J2-"
        pc1_title = f"PC1 ({explained_variance[0]*100:.1f}% var)<br>Top joints: {pc1_label}"
        pc2_title = f"PC2 ({explained_variance[1]*100:.1f}% var)<br>Top joints: {pc2_label}"
        pc3_title = f"PC3 ({explained_variance[2]*100:.1f}% var)<br>Top joints: {pc3_label}"
        
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub style='font-size:0.75em;color:#aaa'>PCA Projection: {n_joints}D joint space → 3D ({total_variance:.1f}% variance explained)</sub>",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title=pc1_title,
                yaxis_title=pc2_title,
                zaxis_title=pc3_title,
                aspectmode='data',
                xaxis=dict(
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    tickfont=dict(size=10)
                ),
                zaxis=dict(
                    tickfont=dict(size=10)
                )
            ),
            template="plotly_dark",
            height=650,
            annotations=[
                dict(
                    text=f"<b>PCA Explained:</b><br>" +
                         f"• Each axis combines multiple joint movements<br>" +
                         f"• PC1 = {explained_variance[0]*100:.1f}% variance (most important)<br>" +
                         f"• PC2 = {explained_variance[1]*100:.1f}% variance<br>" +
                         f"• PC3 = {explained_variance[2]*100:.1f}% variance<br>" +
                         f"• Labels show top contributing joints (J1+ = joint 1 positive, J2- = joint 2 negative)<br>" +
                         f"• Color = visit frequency",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(0,0,0,0.75)",
                    bordercolor="rgba(255,255,255,0.4)",
                    borderwidth=1,
                    borderpad=8,
                    font=dict(size=9.5, color="white"),
                    showarrow=False,
                    align="left"
                )
            ]
        )
        
        return fig
    
    # Cartesian space (3D)
    x_edges, y_edges, z_edges = voxel_edges

    # Find occupied voxels
    occupied_indices = np.where(voxel_grid > 0)

    if len(occupied_indices[0]) == 0:
        # Empty plot if no data
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    # Get voxel centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Extract occupied voxel positions and counts
    x_pos = x_centers[occupied_indices[0]]
    y_pos = y_centers[occupied_indices[1]]
    z_pos = z_centers[occupied_indices[2]]
    counts = voxel_grid[occupied_indices]

    # Normalize counts for color
    normalized_counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-10)

    fig = go.Figure(data=[go.Scatter3d(
        x=x_pos,
        y=y_pos,
        z=z_pos,
        mode='markers',
        marker=dict(
            size=5,
            color=normalized_counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Visit<br>Count"),
            opacity=0.8
        ),
        text=[f"Visits: {int(c)}" for c in counts],
        hovertemplate='<b>Position</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      'Z: %{z:.3f}<br>' +
                      '%{text}<br>' +
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


def plot_starting_position_variance(
    starting_positions: np.ndarray,
    joint_names: Optional[List[str]] = None,
    title: str = "Starting Position Variance"
) -> go.Figure:
    """Plot variance in starting positions across DOFs.

    Args:
        starting_positions: Starting positions, shape (n_episodes, n_dof)
        joint_names: Optional joint names
        title: Plot title

    Returns:
        Plotly figure
    """
    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(starting_positions.shape[1])]

    fig = go.Figure()

    # Box plot for each joint
    for i, joint_name in enumerate(joint_names):
        fig.add_trace(go.Box(
            y=starting_positions[:, i],
            name=joint_name,
            marker_color='#1f77b4',
            boxmean='sd'  # Show mean and std dev
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Joint",
        yaxis_title="Position (rad or m)",
        template="plotly_dark",
        showlegend=False,
        height=400
    )

    return fig


def plot_starting_position_distribution(
    starting_positions: np.ndarray,
    title: str = "Starting Position Distribution (PCA)"
) -> go.Figure:
    """Plot 2D projection of starting positions using PCA.

    Args:
        starting_positions: Starting positions, shape (n_episodes, n_dof)
        title: Plot title

    Returns:
        Plotly figure
    """
    from sklearn.decomposition import PCA

    if len(starting_positions) < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough data for PCA")
        return fig

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    positions_2d = pca.fit_transform(starting_positions)

    explained_variance = pca.explained_variance_ratio_

    fig = go.Figure(data=[go.Scatter(
        x=positions_2d[:, 0],
        y=positions_2d[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color='#1f77b4',
            opacity=0.7
        ),
        text=[f"Episode {i+1}" for i in range(len(positions_2d))],
        hovertemplate='<b>%{text}</b><br>' +
                      'PC1: %{x:.3f}<br>' +
                      'PC2: %{y:.3f}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({explained_variance[0]*100:.1f}% variance)",
        yaxis_title=f"PC2 ({explained_variance[1]*100:.1f}% variance)",
        template="plotly_dark",
        height=500
    )

    return fig


def plot_position_heatmap_2d(
    positions: np.ndarray,
    title: str = "Position Density Heatmap (Top View)"
) -> go.Figure:
    """Plot 2D heatmap of position density.

    Args:
        positions: Positions array, shape (n_samples, n_dims)
        title: Plot title

    Returns:
        Plotly figure
    """
    # Use first 2 dimensions (typically x, y)
    if positions.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough dimensions for 2D heatmap")
        return fig

    x = positions[:, 0]
    y = positions[:, 1]

    # Create 2D histogram
    fig = go.Figure(data=[go.Histogram2d(
        x=x,
        y=y,
        colorscale='Viridis',
        nbinsx=50,
        nbinsy=50,
        colorbar=dict(title="Count")
    )])

    fig.update_layout(
        title=title,
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        template="plotly_dark",
        height=500
    )

    return fig


def plot_coverage_summary(
    spatial_results: Dict[str, Any],
    title: str = "Spatial Coverage Summary"
) -> go.Figure:
    """Create bar chart summarizing spatial coverage metrics.

    Args:
        spatial_results: Spatial coverage analysis results
        title: Plot title

    Returns:
        Plotly figure
    """
    # Extract scores
    scores = spatial_results['individual_scores']

    metric_names = [
        'Workspace<br>Coverage',
        'Starting<br>Variance',
        'Position<br>Distribution',
        'Angle<br>Diversity'
    ]

    metric_values = [
        scores['workspace_coverage'],
        scores['starting_variance'],
        scores['position_distribution'],
        scores['angle_diversity']
    ]

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

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


def plot_workspace_bounds(
    workspace_bounds: Dict[str, tuple],
    actual_positions: np.ndarray,
    title: str = "Workspace Utilization"
) -> go.Figure:
    """Plot workspace bounds vs actual positions.

    Args:
        workspace_bounds: Dictionary with x, y, z bounds
        actual_positions: Actual positions visited
        title: Plot title

    Returns:
        Plotly figure
    """
    # Extract bounds
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    z_min, z_max = workspace_bounds['z']

    # Create workspace boundary box (wireframe)
    # Define the 12 edges of the box
    edges_x = [
        [x_min, x_max], [x_min, x_max], [x_min, x_max], [x_min, x_max],  # Bottom and top
        [x_min, x_min], [x_max, x_max], [x_min, x_min], [x_max, x_max]   # Vertical
    ]
    edges_y = [
        [y_min, y_min], [y_max, y_max], [y_min, y_min], [y_max, y_max],  # Bottom and top
        [y_min, y_max], [y_min, y_max], [y_min, y_max], [y_min, y_max]   # Vertical
    ]
    edges_z = [
        [z_min, z_min], [z_min, z_min], [z_max, z_max], [z_max, z_max],  # Bottom and top
        [z_min, z_min], [z_min, z_min], [z_max, z_max], [z_max, z_max]   # Vertical
    ]

    fig = go.Figure()

    # Add workspace boundary
    for ex, ey, ez in zip(edges_x, edges_y, edges_z):
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add actual positions (sample if too many)
    if len(actual_positions) > 5000:
        sample_indices = np.random.choice(len(actual_positions), 5000, replace=False)
        sample_positions = actual_positions[sample_indices]
    else:
        sample_positions = actual_positions

    fig.add_trace(go.Scatter3d(
        x=sample_positions[:, 0],
        y=sample_positions[:, 1],
        z=sample_positions[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='#1f77b4',
            opacity=0.5
        ),
        name='Visited Positions',
        hoverinfo='skip'
    ))

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


def create_spatial_coverage_dashboard(
    spatial_results: Dict[str, Any],
    all_positions: np.ndarray,
    starting_positions: np.ndarray
) -> Dict[str, go.Figure]:
    """Create comprehensive spatial coverage dashboard.

    Args:
        spatial_results: Spatial coverage analysis results
        all_positions: All positions from episodes
        starting_positions: Starting positions from episodes

    Returns:
        Dictionary of figure names to Plotly figures
    """
    figures = {}

    # 1. Workspace coverage 3D
    coverage = spatial_results['workspace_coverage']
    
    # Check if data is in Cartesian space (3D) or joint space (higher dimensional)
    data_space = coverage.get('data_space', 'cartesian')
    
    # Create 3D visualization (works for both Cartesian and joint space via PCA)
    if 'voxel_grid' in coverage and 'voxel_edges' in coverage:
        figures['workspace_3d'] = plot_workspace_coverage_3d(
            coverage['voxel_grid'],
            coverage['voxel_edges'],
            title=f"Workspace Coverage: {coverage['coverage_percentage']:.1f}%",
            all_positions=all_positions,
            data_space=data_space
        )
    else:
        # Fallback if voxel data is not available
        fig = go.Figure()
        fig.add_annotation(
            text="3D visualization not available for this data type.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Workspace Coverage", template="plotly_dark", height=600)
        figures['workspace_3d'] = fig

    # 2. Starting position variance
    figures['starting_variance'] = plot_starting_position_variance(
        starting_positions,
        title="Starting Position Variance Across Episodes"
    )

    # 3. Starting position PCA
    figures['starting_pca'] = plot_starting_position_distribution(
        starting_positions,
        title="Starting Position Diversity (PCA Projection)"
    )

    # 4. Position heatmap 2D
    figures['position_heatmap'] = plot_position_heatmap_2d(
        all_positions,
        title="Position Density Heatmap (XY Plane)"
    )

    # 5. Coverage summary
    figures['coverage_summary'] = plot_coverage_summary(
        spatial_results,
        title="Spatial Coverage Metrics Summary"
    )

    # 6. Workspace bounds
    if 'workspace_bounds' in coverage:
        figures['workspace_bounds'] = plot_workspace_bounds(
            coverage['workspace_bounds'],
            all_positions,
            title="Workspace Utilization vs Limits"
        )

    return figures
