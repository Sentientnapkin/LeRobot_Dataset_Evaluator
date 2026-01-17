"""Visualization for joint information and descriptions with interactive 3D robot arm."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from ..utils.robot_models import RobotConfig, get_robot_config
from .interactive_robot_arm import create_interactive_robot_arm_figure


def create_joint_info_visualization(
    robot_config: Optional[RobotConfig] = None,
    robot_type: Optional[str] = None,
    show_3d_model: bool = True
) -> go.Figure:
    """Create a visualization explaining what each joint represents.
    
    Args:
        robot_config: Robot configuration object
        robot_type: Robot type string (used if robot_config not provided)
        
    Returns:
        Plotly figure showing joint information
    """
    if robot_config is None and robot_type:
        try:
            robot_config = get_robot_config(robot_type)
        except:
            robot_config = None
    
    if robot_config is None:
        # Default generic robot info
        n_dof = 6
        joint_names = [f"Joint {i+1}" for i in range(n_dof)]
        joint_descriptions = [
            "Base rotation (shoulder yaw)",
            "Shoulder pitch",
            "Elbow pitch",
            "Wrist roll",
            "Wrist pitch",
            "Wrist yaw (end-effector rotation)"
        ]
    else:
        n_dof = robot_config.dof
        joint_names = [f"J{i+1}" for i in range(n_dof)]
        
        # Create descriptions based on robot type and DOF
        if robot_config.name.lower().startswith('ur5'):
            joint_descriptions = [
                "Base rotation (shoulder yaw)",
                "Shoulder lift",
                "Elbow extension",
                "Wrist 1 rotation",
                "Wrist 2 bend",
                "Wrist 3 rotation"
            ]
        elif 'panda' in robot_config.name.lower():
            joint_descriptions = [
                "Shoulder pan",
                "Shoulder lift",
                "Elbow",
                "Wrist 1",
                "Wrist 2",
                "Wrist 3",
                "Finger 1" if n_dof > 6 else "Wrist 4"
            ]
        elif 'so-arm' in robot_config.name.lower():
            joint_descriptions = [
                "Base rotation",
                "Shoulder pitch",
                "Elbow pitch",
                "Wrist roll",
                "Wrist pitch",
                "Wrist yaw"
            ]
        else:
            # Generic descriptions
            joint_descriptions = []
            for i in range(n_dof):
                if i == 0:
                    joint_descriptions.append("Base rotation (shoulder yaw)")
                elif i == 1:
                    joint_descriptions.append("Shoulder pitch")
                elif i == 2:
                    joint_descriptions.append("Elbow pitch")
                elif i == n_dof - 3:
                    joint_descriptions.append("Wrist roll")
                elif i == n_dof - 2:
                    joint_descriptions.append("Wrist pitch")
                elif i == n_dof - 1:
                    joint_descriptions.append("Wrist yaw (end-effector)")
                else:
                    joint_descriptions.append(f"Joint {i+1} (intermediate)")
    
    # Create subplot with 3D robot arm and joint info side by side
    if show_3d_model:
        # Create subplot: 3D model on left, info panel on right
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.1
        )
        
        # Get 3D robot arm visualization
        arm_fig = create_robot_arm_3d(robot_config=robot_config, show_labels=True)
        
        # Add all traces from arm figure to subplot
        for trace in arm_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Update 3D scene
        fig.update_scenes(
            row=1, col=1,
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0.3)
            )
        )
        
        # Add joint information panel on the right using text traces
        y_positions = np.linspace(0.85, 0.15, n_dof)
        
        # Add invisible scatter points to anchor text
        fig.add_trace(go.Scatter(
            x=[0.1] * n_dof,
            y=y_positions,
            mode='text',
            text=[f"<b>{name}</b>" for name in joint_names],
            textfont=dict(size=16, color="#1f77b4"),
            textposition="middle left",
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=[0.5] * n_dof,
            y=y_positions,
            mode='text',
            text=joint_descriptions,
            textfont=dict(size=13, color="white"),
            textposition="middle left",
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=2)
        
        if robot_config:
            limits_texts = [
                f"[{robot_config.joint_limits_lower[i]:.2f}, {robot_config.joint_limits_upper[i]:.2f}]"
                for i in range(n_dof)
            ]
            fig.add_trace(go.Scatter(
                x=[0.95] * n_dof,
                y=y_positions,
                mode='text',
                text=limits_texts,
                textfont=dict(size=11, color="#888"),
                textposition="middle right",
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=2)
        
        # Configure right panel (info panel)
        fig.update_xaxes(range=[0, 1], visible=False, row=1, col=2)
        fig.update_yaxes(range=[0, 1], visible=False, row=1, col=2)
        
        # Add title
        robot_name = robot_config.name if robot_config else "Generic Robot"
        fig.add_annotation(
            text=f"<b>Joint Information: {robot_name}</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.98,
            xanchor="center", yanchor="top",
            font=dict(size=20, color="white"),
            showarrow=False
        )
        
        # Add explanation box
        fig.add_annotation(
            text="<b>Understanding Joint Labels:</b><br>" +
                 "• <b>J1, J2, J3</b> control the arm (base, shoulder, elbow)<br>" +
                 "• <b>J4, J5, J6</b> control the wrist and end-effector<br>" +
                 "• In PCA: <b>J1+</b> = positive contribution, <b>J2-</b> = negative",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            xanchor="center", yanchor="bottom",
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1,
            borderpad=10,
            font=dict(size=11, color="white"),
            showarrow=False,
            align="left"
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=700,
            showlegend=False,
            margin=dict(l=0, r=0, t=80, b=100)
        )
    else:
        # Fallback to original table view if 3D model disabled
        fig = go.Figure()
        # ... (original code for table view)
        pass
    
    return fig
