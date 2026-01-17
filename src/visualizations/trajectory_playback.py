"""Trajectory Playback Visualizations.

This module provides visualization functions for trajectory playback and animation.
"""

from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# Color scheme
COLORS = {
    'trajectory': '#2563EB',
    'current': '#DC2626',
    'start': '#059669',
    'end': '#D97706',
    'background': '#1E293B',
    'grid': '#334155',
    'text': '#F8FAFC',
    'secondary': '#64748B'
}


def create_3d_trajectory_plot(
    states: np.ndarray,
    current_frame: int = 0,
    title: str = "3D Trajectory"
) -> go.Figure:
    """Create 3D trajectory plot with current position highlight.

    Args:
        states: Array of shape (n_frames, n_dims) with state data
        current_frame: Current frame index to highlight
        title: Plot title

    Returns:
        Plotly figure
    """
    n_frames = len(states)
    n_dims = states.shape[1] if len(states.shape) > 1 else 1

    # Use first 3 dimensions for 3D plot
    if n_dims >= 3:
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
    else:
        x = states[:, 0] if n_dims >= 1 else np.arange(n_frames)
        y = states[:, 1] if n_dims >= 2 else np.zeros(n_frames)
        z = np.zeros(n_frames)

    fig = go.Figure()

    # Full trajectory
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        name='Trajectory',
        line=dict(color=COLORS['trajectory'], width=3),
        opacity=0.7
    ))

    # Trajectory up to current frame
    if current_frame > 0:
        fig.add_trace(go.Scatter3d(
            x=x[:current_frame+1],
            y=y[:current_frame+1],
            z=z[:current_frame+1],
            mode='lines',
            name='Path Traveled',
            line=dict(color=COLORS['current'], width=5),
            opacity=1.0
        ))

    # Start position
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode='markers',
        name='Start',
        marker=dict(
            color=COLORS['start'],
            size=12,
            symbol='circle'
        )
    ))

    # End position
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode='markers',
        name='End',
        marker=dict(
            color=COLORS['end'],
            size=12,
            symbol='diamond'
        )
    ))

    # Current position
    current_frame = min(current_frame, n_frames - 1)
    fig.add_trace(go.Scatter3d(
        x=[x[current_frame]],
        y=[y[current_frame]],
        z=[z[current_frame]],
        mode='markers',
        name='Current',
        marker=dict(
            color=COLORS['current'],
            size=15,
            symbol='cross'
        )
    ))

    fig.update_layout(
        title=f"{title} (Frame {current_frame + 1}/{n_frames})",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor=COLORS['background'],
            xaxis=dict(gridcolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid']),
            zaxis=dict(gridcolor=COLORS['grid'])
        ),
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_state_timeline(
    states: np.ndarray,
    current_frame: int = 0,
    labels: Optional[List[str]] = None,
    title: str = "State Timeline"
) -> go.Figure:
    """Create timeline chart showing state evolution.

    Args:
        states: Array of shape (n_frames, n_dims)
        current_frame: Current frame index to highlight
        labels: Optional labels for each dimension
        title: Plot title

    Returns:
        Plotly figure
    """
    n_frames = len(states)
    n_dims = states.shape[1] if len(states.shape) > 1 else 1
    frames = np.arange(n_frames)

    if labels is None:
        labels = [f"Dim {i+1}" for i in range(n_dims)]

    # Limit to first 6 dimensions for readability
    n_dims_show = min(n_dims, 6)

    fig = make_subplots(
        rows=n_dims_show, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=labels[:n_dims_show]
    )

    colors = ['#2563EB', '#059669', '#D97706', '#DC2626', '#8B5CF6', '#EC4899']

    for i in range(n_dims_show):
        dim_data = states[:, i] if n_dims > 1 else states

        # Full trace
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=dim_data,
                mode='lines',
                name=labels[i],
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ),
            row=i+1, col=1
        )

        # Current position marker
        fig.add_trace(
            go.Scatter(
                x=[current_frame],
                y=[dim_data[min(current_frame, n_frames-1)]],
                mode='markers',
                marker=dict(color=COLORS['current'], size=10),
                showlegend=False
            ),
            row=i+1, col=1
        )

    # Add current frame line
    fig.add_vline(
        x=current_frame,
        line_dash="dash",
        line_color=COLORS['current'],
        line_width=2
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=100 + n_dims_show * 100,
        showlegend=False
    )

    fig.update_xaxes(title_text="Frame", row=n_dims_show, col=1)

    return fig


def create_action_timeline(
    actions: np.ndarray,
    current_frame: int = 0,
    labels: Optional[List[str]] = None,
    title: str = "Action Timeline"
) -> go.Figure:
    """Create timeline chart showing action evolution.

    Args:
        actions: Array of shape (n_frames, n_dims)
        current_frame: Current frame index to highlight
        labels: Optional labels for each dimension
        title: Plot title

    Returns:
        Plotly figure
    """
    return create_state_timeline(
        actions,
        current_frame=current_frame,
        labels=labels,
        title=title
    )


def create_animated_trajectory(
    states: np.ndarray,
    fps: int = 10,
    title: str = "Animated Trajectory"
) -> go.Figure:
    """Create animated 3D trajectory plot.

    Args:
        states: Array of shape (n_frames, n_dims)
        fps: Frames per second for animation
        title: Plot title

    Returns:
        Plotly figure with animation frames
    """
    n_frames = len(states)
    n_dims = states.shape[1] if len(states.shape) > 1 else 1

    # Use first 3 dimensions
    if n_dims >= 3:
        x = states[:, 0]
        y = states[:, 1]
        z = states[:, 2]
    else:
        x = states[:, 0] if n_dims >= 1 else np.arange(n_frames)
        y = states[:, 1] if n_dims >= 2 else np.zeros(n_frames)
        z = np.zeros(n_frames)

    # Create figure with initial data
    fig = go.Figure(
        data=[
            # Full trajectory (faded)
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name='Full Path',
                line=dict(color=COLORS['trajectory'], width=2),
                opacity=0.3
            ),
            # Animated path
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='lines',
                name='Current Path',
                line=dict(color=COLORS['current'], width=4)
            ),
            # Current position marker
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                name='Current Position',
                marker=dict(color=COLORS['current'], size=12)
            ),
            # Start marker
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                name='Start',
                marker=dict(color=COLORS['start'], size=10)
            )
        ]
    )

    # Create frames for animation (subsample for performance)
    step = max(1, n_frames // 100)  # Max 100 frames
    frame_indices = list(range(0, n_frames, step))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    frames = []
    for i in frame_indices:
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=x, y=y, z=z),  # Full trajectory stays same
                go.Scatter3d(x=x[:i+1], y=y[:i+1], z=z[:i+1]),  # Path up to current
                go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]]),  # Current position
                go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]])  # Start stays same
            ],
            name=str(i)
        ))

    fig.frames = frames

    # Add animation controls
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            bgcolor=COLORS['background'],
            aspectmode='data'
        ),
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=600,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1000//fps, redraw=True),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True)
                            )
                        ],
                        label=str(i)
                    )
                    for i in frame_indices
                ],
                x=0.1,
                len=0.8,
                y=-0.05,
                currentvalue=dict(
                    prefix="Frame: ",
                    visible=True,
                    xanchor="center"
                ),
                transition=dict(duration=0)
            )
        ]
    )

    return fig


def create_velocity_profile(
    states: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    current_frame: int = 0,
    title: str = "Velocity Profile"
) -> go.Figure:
    """Create velocity profile visualization.

    Args:
        states: Array of shape (n_frames, n_dims)
        timestamps: Optional array of timestamps
        current_frame: Current frame index
        title: Plot title

    Returns:
        Plotly figure
    """
    n_frames = len(states)

    if timestamps is None:
        timestamps = np.arange(n_frames) / 30.0  # Assume 30 fps

    # Calculate velocities
    dt = np.diff(timestamps)
    dt = np.where(dt > 0, dt, 1e-6)  # Avoid division by zero

    if len(states.shape) > 1:
        # Calculate magnitude of velocity
        dpos = np.diff(states, axis=0)
        velocity = np.sqrt(np.sum(dpos**2, axis=1)) / dt
    else:
        velocity = np.abs(np.diff(states)) / dt

    frames = np.arange(len(velocity))

    fig = go.Figure()

    # Velocity profile
    fig.add_trace(go.Scatter(
        x=frames,
        y=velocity,
        mode='lines',
        name='Velocity',
        line=dict(color=COLORS['trajectory'], width=2),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.2)'
    ))

    # Current frame marker
    if current_frame > 0 and current_frame < len(velocity):
        fig.add_trace(go.Scatter(
            x=[current_frame],
            y=[velocity[current_frame-1]],
            mode='markers',
            name='Current',
            marker=dict(color=COLORS['current'], size=12)
        ))

    # Current frame line
    fig.add_vline(
        x=current_frame,
        line_dash="dash",
        line_color=COLORS['current'],
        line_width=2
    )

    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title="Velocity (units/s)",
        template="plotly_dark",
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=300,
        showlegend=False
    )

    return fig


def create_trajectory_dashboard(
    episode_data: Dict[str, Any],
    current_frame: int = 0
) -> Dict[str, go.Figure]:
    """Create all trajectory playback visualizations for an episode.

    Args:
        episode_data: Episode data dictionary with 'states', 'actions', etc.
        current_frame: Current frame index

    Returns:
        Dictionary of figure names to Plotly figures
    """
    states = np.array(episode_data.get('states', []))
    actions = np.array(episode_data.get('actions', []))
    timestamps = episode_data.get('timestamps')

    if timestamps is not None:
        timestamps = np.array(timestamps)

    figures = {}

    if len(states) > 0:
        figures['trajectory_3d'] = create_3d_trajectory_plot(
            states, current_frame,
            title="3D Trajectory"
        )

        figures['state_timeline'] = create_state_timeline(
            states, current_frame,
            title="State Evolution"
        )

        figures['velocity_profile'] = create_velocity_profile(
            states, timestamps, current_frame,
            title="Velocity Profile"
        )

        figures['animated_trajectory'] = create_animated_trajectory(
            states,
            title="Animated Trajectory Playback"
        )

    if len(actions) > 0:
        figures['action_timeline'] = create_action_timeline(
            actions, current_frame,
            title="Action Evolution"
        )

    return figures
