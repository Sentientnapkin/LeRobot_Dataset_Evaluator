"""Interactive 3D robot arm visualization with joint angle controls."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple, List
from ..utils.robot_models import RobotConfig
from ..utils.robot_kinematics import get_dh_params, get_joint_positions


def create_interactive_robot_arm_figure(
    robot_config: Optional[RobotConfig] = None,
    joint_angles: Optional[np.ndarray] = None
) -> go.Figure:
    """Create an interactive 3D robot arm figure that can be updated with joint angles.

    Uses accurate DH-based forward kinematics for supported robots.

    Args:
        robot_config: Robot configuration
        joint_angles: Joint angles array

    Returns:
        Plotly figure ready for interactive updates
    """
    if robot_config is None:
        n_dof = 6
        robot_type = 'generic_6dof'
        if joint_angles is None:
            joint_angles = np.array([0.0, np.pi/4, -np.pi/4, 0.0, np.pi/4, 0.0])
    else:
        n_dof = robot_config.dof
        robot_type = robot_config.robot_type_key if robot_config.robot_type_key else 'generic_6dof'
        if joint_angles is None:
            joint_angles = (robot_config.joint_limits_lower + robot_config.joint_limits_upper) / 2

    # Get DH parameters for this robot type
    dh_params = get_dh_params(robot_type)
    n_joints = min(len(joint_angles), len(dh_params))

    # Ensure we have the right number of angles
    if len(joint_angles) < n_joints:
        joint_angles = np.pad(joint_angles, (0, n_joints - len(joint_angles)), 'constant')
    joint_angles = joint_angles[:n_joints]

    # Compute joint positions using DH-based forward kinematics
    positions = get_joint_positions(joint_angles, dh_params)

    fig = go.Figure()

    # Draw links with thicker, more visible lines
    link_colors = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4"]
    link_names = ["Base Link", "Shoulder Link", "Elbow Link", "Wrist 1", "Wrist 2", "Wrist 3", "End Effector Link"]

    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]

        # Draw link as a thick line
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+markers',
            line=dict(color=link_colors[i % len(link_colors)], width=12),
            marker=dict(size=8, color=link_colors[i % len(link_colors)]),
            name=link_names[i] if i < len(link_names) else f"Link {i+1}",
            showlegend=False,
            hovertemplate=f'<b>{link_names[i] if i < len(link_names) else f"Link {i+1}"}</b><br>' +
                         f'Start: ({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f})<br>' +
                         f'End: ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f})<br>' +
                         '<extra></extra>'
        ))

    # Draw joints with labels
    joint_names = [f"J{i+1}" for i in range(n_joints)]
    joint_descriptions = [
        "Base Rotation",
        "Shoulder Pitch",
        "Elbow Pitch",
        "Wrist Roll",
        "Wrist Pitch",
        "Wrist Yaw",
        "Flange"
    ]

    for i, pos in enumerate(positions[1:n_joints+1]):
        if i < len(joint_names):
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode='markers+text',
                marker=dict(
                    size=18,
                    color="#EF4444",
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=[joint_names[i]],
                textposition="top center",
                textfont=dict(size=16, color="white", family="Arial Black"),
                name=joint_names[i],
                showlegend=False,
                hovertemplate=f'<b>{joint_names[i]}: {joint_descriptions[i] if i < len(joint_descriptions) else ""}</b><br>' +
                             f'Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})<br>' +
                             f'Angle: {joint_angles[i]:.3f} rad ({np.degrees(joint_angles[i]):.1f} deg)<br>' +
                             '<extra></extra>'
            ))

    # Base platform
    base_size = 0.15
    base_points = np.array([
        [-base_size, -base_size, 0],
        [base_size, -base_size, 0],
        [base_size, base_size, 0],
        [-base_size, base_size, 0],
        [-base_size, -base_size, 0]
    ])

    fig.add_trace(go.Scatter3d(
        x=base_points[:, 0],
        y=base_points[:, 1],
        z=base_points[:, 2],
        mode='lines',
        line=dict(color='#6B7280', width=4),
        name="Base",
        showlegend=False,
        hoverinfo='skip'
    ))

    # Base center marker
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=20, color="#374151", symbol='diamond'),
        name="Base",
        showlegend=False,
        hovertemplate='<b>Base</b><br>Position: (0, 0, 0)<extra></extra>'
    ))

    # End effector marker
    if len(positions) > 1:
        ee_pos = positions[-1]
        fig.add_trace(go.Scatter3d(
            x=[ee_pos[0]],
            y=[ee_pos[1]],
            z=[ee_pos[2]],
            mode='markers',
            marker=dict(size=15, color="#10B981", symbol='diamond'),
            name="End Effector",
            showlegend=False,
            hovertemplate=f'<b>End Effector</b><br>' +
                         f'Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})<br>' +
                         '<extra></extra>'
        ))

    # Calculate workspace bounds for better view
    all_positions = positions
    if len(all_positions) > 0:
        x_range = [all_positions[:, 0].min() - 0.2, all_positions[:, 0].max() + 0.2]
        y_range = [all_positions[:, 1].min() - 0.2, all_positions[:, 1].max() + 0.2]
        z_range = [min(0, all_positions[:, 2].min() - 0.1), all_positions[:, 2].max() + 0.2]
    else:
        x_range = [-0.5, 0.5]
        y_range = [-0.5, 0.5]
        z_range = [0, 0.8]

    robot_name = robot_config.name if robot_config else "Generic 6-DOF Robot"
    fig.update_layout(
        title=dict(
            text=f"<b>{robot_name} - Interactive Joint Control</b><br>" +
                 "<span style='color:#94A3B8;font-size:0.8em'>Adjust sliders to move joints</span>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title="X (m)", range=x_range, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Y (m)", range=y_range, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Z (m)", range=z_range, backgroundcolor="rgba(0,0,0,0)"),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5),
                center=dict(x=0, y=0, z=0.3),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=False
    )

    return fig
