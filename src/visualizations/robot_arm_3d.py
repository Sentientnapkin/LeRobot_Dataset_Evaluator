"""3D robot arm visualization with accurate DH-based kinematics."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple, List
from ..utils.robot_models import RobotConfig
from ..utils.robot_kinematics import (
    get_dh_params,
    get_joint_positions,
    get_link_lengths,
    forward_kinematics,
    DHParameters
)


def forward_kinematics_dh(
    joint_angles: np.ndarray,
    robot_type: str = 'generic_6dof'
) -> np.ndarray:
    """Compute forward kinematics using DH parameters.

    Args:
        joint_angles: Array of joint angles (radians)
        robot_type: Robot type for DH parameter lookup

    Returns:
        Array of joint positions, shape (n_joints + 1, 3)
    """
    dh_params = get_dh_params(robot_type)
    return get_joint_positions(joint_angles, dh_params)


def forward_kinematics_6dof(
    joint_angles: np.ndarray,
    link_lengths: Optional[List[float]] = None,
    robot_type: str = 'generic_6dof'
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute forward kinematics for a 6-DOF robot arm using DH parameters.

    This function is maintained for backward compatibility but now uses
    accurate DH-based kinematics internally.

    Args:
        joint_angles: Array of 6 joint angles (radians)
        link_lengths: Deprecated - ignored, DH params used instead
        robot_type: Robot type for DH parameter lookup

    Returns:
        List of (position, direction) tuples for each joint
    """
    dh_params = get_dh_params(robot_type)
    n_joints = min(len(joint_angles), len(dh_params))

    # Get joint positions using DH kinematics
    positions = get_joint_positions(joint_angles[:n_joints], dh_params)

    # Compute directions from position differences
    results = []
    for i in range(len(positions)):
        pos = positions[i]
        if i < len(positions) - 1:
            direction = positions[i + 1] - positions[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])
        else:
            # End effector - use last direction
            if i > 0:
                direction = positions[i] - positions[i - 1]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                else:
                    direction = np.array([0, 0, 1])
            else:
                direction = np.array([0, 0, 1])

        results.append((pos, direction))

    return results


def create_robot_arm_3d(
    robot_config: Optional[RobotConfig] = None,
    joint_angles: Optional[np.ndarray] = None,
    show_labels: bool = True
) -> go.Figure:
    """Create a 3D visualization of a robot arm with joint labels.

    Uses accurate DH-based forward kinematics for supported robots.

    Args:
        robot_config: Robot configuration
        joint_angles: Optional joint angles (defaults to neutral pose)
        show_labels: Whether to show joint labels

    Returns:
        Plotly figure
    """
    if robot_config is None:
        n_dof = 6
        robot_type = 'generic_6dof'
        joint_angles = np.array([0.0, np.pi/4, -np.pi/4, 0.0, np.pi/4, 0.0])
    else:
        n_dof = robot_config.dof
        robot_type = robot_config.robot_type_key if robot_config.robot_type_key else 'generic_6dof'
        if joint_angles is None:
            joint_angles = (robot_config.joint_limits_lower + robot_config.joint_limits_upper) / 2

    # Get DH parameters and compute FK
    dh_params = get_dh_params(robot_type)
    n_joints = min(len(joint_angles), len(dh_params))
    positions = get_joint_positions(joint_angles[:n_joints], dh_params)

    fig = go.Figure()

    # Draw links (lines connecting joints)
    link_colors = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4"]
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]

        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines+markers',
            line=dict(color=link_colors[i % len(link_colors)], width=8),
            marker=dict(size=6, color=link_colors[i % len(link_colors)]),
            name=f"Link {i+1}",
            showlegend=False
        ))

    # Draw joints (spheres)
    joint_names = [f"J{i+1}" for i in range(n_dof)]
    for i, (pos, name) in enumerate(zip(positions[1:n_dof+1], joint_names)):
        fig.add_trace(go.Scatter3d(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=12, color="#EF4444", symbol='circle'),
            text=[name] if show_labels else None,
            textposition="top center",
            textfont=dict(size=14, color="white"),
            name=name,
            showlegend=False,
            hovertemplate=f'<b>{name}</b><br>' +
                         f'Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})<br>' +
                         f'Angle: {joint_angles[i]:.3f} rad ({np.degrees(joint_angles[i]):.1f} deg)<br>' +
                         '<extra></extra>'
        ))

    # Base marker
    fig.add_trace(go.Scatter3d(
        x=[positions[0][0]],
        y=[positions[0][1]],
        z=[positions[0][2]],
        mode='markers',
        marker=dict(size=15, color="#374151", symbol='diamond'),
        name="Base",
        showlegend=False,
        hovertemplate='<b>Base</b><br>Position: (0, 0, 0)<extra></extra>'
    ))

    # End effector marker
    ee_pos = positions[-1]
    fig.add_trace(go.Scatter3d(
        x=[ee_pos[0]],
        y=[ee_pos[1]],
        z=[ee_pos[2]],
        mode='markers',
        marker=dict(size=10, color="#10B981", symbol='diamond'),
        name="End Effector",
        showlegend=False,
        hovertemplate=f'<b>End Effector</b><br>' +
                     f'Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})<br>' +
                     '<extra></extra>'
    ))

    # Set layout
    robot_name = robot_config.name if robot_config else "Generic 6-DOF Robot"
    fig.update_layout(
        title=dict(
            text=f"<b>{robot_name} - Joint Visualization</b>",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0.3)
            )
        ),
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig
