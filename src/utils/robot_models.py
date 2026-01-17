"""Robot model configurations and specifications."""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .robot_kinematics import RobotKinematics


@dataclass
class RobotConfig:
    """Configuration for a specific robot model."""

    name: str
    dof: int
    joint_limits_lower: np.ndarray  # radians or meters
    joint_limits_upper: np.ndarray
    workspace_bounds: Dict[str, tuple]  # {'x': (min, max), 'y': (min, max), 'z': (min, max)}
    max_velocity: np.ndarray  # rad/s or m/s per joint
    max_acceleration: np.ndarray  # rad/s^2 or m/s^2 per joint
    description: str = ""
    robot_type_key: str = ""  # Key for looking up kinematics

    def get_kinematics(self) -> 'RobotKinematics':
        """Get the kinematics instance for this robot.

        Returns:
            RobotKinematics instance with DH parameters
        """
        from .robot_kinematics import RobotKinematics

        key = self.robot_type_key if self.robot_type_key else 'generic_6dof'
        return RobotKinematics(
            robot_type=key,
            joint_limits=(self.joint_limits_lower, self.joint_limits_upper)
        )


# Robot configurations database
ROBOT_CONFIGS = {
    'ur5': RobotConfig(
        name='UR5',
        dof=6,
        joint_limits_lower=np.array([-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
        joint_limits_upper=np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
        workspace_bounds={
            'x': (-0.85, 0.85),
            'y': (-0.85, 0.85),
            'z': (0.0, 1.3)
        },
        max_velocity=np.array([np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
        max_acceleration=np.array([10.0, 10.0, 10.0, 15.0, 15.0, 15.0]),
        description="Universal Robots UR5 collaborative robot arm",
        robot_type_key='ur5'
    ),
    'panda': RobotConfig(
        name='Franka Panda',
        dof=7,  # 7-DOF but can be configured for 6-DOF
        joint_limits_lower=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        joint_limits_upper=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
        workspace_bounds={
            'x': (-0.855, 0.855),
            'y': (-0.855, 0.855),
            'z': (0.0, 1.19)
        },
        max_velocity=np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]),
        max_acceleration=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
        description="Franka Emika Panda collaborative robot arm",
        robot_type_key='panda'
    ),
    'so-arm100': RobotConfig(
        name='SO-ARM100',
        dof=6,
        joint_limits_lower=np.array([-3.05, -1.57, -1.57, -3.05, -1.57, -3.05]),
        joint_limits_upper=np.array([3.05, 1.57, 3.05, 3.05, 1.57, 3.05]),
        workspace_bounds={
            'x': (-0.6, 0.6),
            'y': (-0.6, 0.6),
            'z': (0.0, 0.8)
        },
        max_velocity=np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
        max_acceleration=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        description="SO-ARM100 6-DOF robotic arm",
        robot_type_key='so-arm100'
    ),
    'generic_6dof': RobotConfig(
        name='Generic 6-DOF Arm',
        dof=6,
        joint_limits_lower=np.array([-np.pi] * 6),
        joint_limits_upper=np.array([np.pi] * 6),
        workspace_bounds={
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.5)
        },
        max_velocity=np.array([2.0] * 6),
        max_acceleration=np.array([10.0] * 6),
        description="Generic 6-DOF robot arm with standard limits",
        robot_type_key='generic_6dof'
    )
}


def get_robot_config(robot_type: str) -> RobotConfig:
    """Get robot configuration by type.

    Args:
        robot_type: Robot type identifier (e.g., 'ur5', 'panda', 'so-arm100', 'generic_6dof')

    Returns:
        RobotConfig object

    Raises:
        ValueError: If robot type is not found
    """
    robot_type_lower = robot_type.lower()

    if robot_type_lower not in ROBOT_CONFIGS:
        available = ', '.join(ROBOT_CONFIGS.keys())
        raise ValueError(
            f"Unknown robot type: {robot_type}. Available types: {available}"
        )

    return ROBOT_CONFIGS[robot_type_lower]


def is_within_joint_limits(
    joint_positions: np.ndarray,
    robot_config: RobotConfig,
    tolerance: float = 0.01
) -> np.ndarray:
    """Check if joint positions are within robot limits.

    Args:
        joint_positions: Array of joint positions, shape (n_samples, n_joints)
        robot_config: Robot configuration
        tolerance: Tolerance in radians/meters

    Returns:
        Boolean array indicating which samples are within limits
    """
    if joint_positions.ndim == 1:
        joint_positions = joint_positions.reshape(1, -1)

    lower_check = joint_positions >= (robot_config.joint_limits_lower - tolerance)
    upper_check = joint_positions <= (robot_config.joint_limits_upper + tolerance)

    return np.all(lower_check & upper_check, axis=1)


def is_within_workspace(
    positions: np.ndarray,
    robot_config: RobotConfig,
    tolerance: float = 0.01
) -> np.ndarray:
    """Check if Cartesian positions are within robot workspace.

    Args:
        positions: Array of [x, y, z] positions, shape (n_samples, 3)
        robot_config: Robot configuration
        tolerance: Tolerance in meters

    Returns:
        Boolean array indicating which samples are within workspace
    """
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)

    bounds = robot_config.workspace_bounds
    x_check = (positions[:, 0] >= bounds['x'][0] - tolerance) & \
              (positions[:, 0] <= bounds['x'][1] + tolerance)
    y_check = (positions[:, 1] >= bounds['y'][0] - tolerance) & \
              (positions[:, 1] <= bounds['y'][1] + tolerance)
    z_check = (positions[:, 2] >= bounds['z'][0] - tolerance) & \
              (positions[:, 2] <= bounds['z'][1] + tolerance)

    return x_check & y_check & z_check


def get_workspace_volume(robot_config: RobotConfig) -> float:
    """Calculate approximate workspace volume in cubic meters.

    Args:
        robot_config: Robot configuration

    Returns:
        Workspace volume (m^3)
    """
    bounds = robot_config.workspace_bounds
    x_range = bounds['x'][1] - bounds['x'][0]
    y_range = bounds['y'][1] - bounds['y'][0]
    z_range = bounds['z'][1] - bounds['z'][0]

    return x_range * y_range * z_range


def normalize_joint_positions(
    joint_positions: np.ndarray,
    robot_config: RobotConfig
) -> np.ndarray:
    """Normalize joint positions to [-1, 1] range based on limits.

    Args:
        joint_positions: Array of joint positions
        robot_config: Robot configuration

    Returns:
        Normalized positions in [-1, 1]
    """
    joint_range = robot_config.joint_limits_upper - robot_config.joint_limits_lower
    joint_center = (robot_config.joint_limits_upper + robot_config.joint_limits_lower) / 2

    normalized = (joint_positions - joint_center) / (joint_range / 2)

    return normalized


def denormalize_joint_positions(
    normalized_positions: np.ndarray,
    robot_config: RobotConfig
) -> np.ndarray:
    """Denormalize joint positions from [-1, 1] range to actual values.

    Args:
        normalized_positions: Normalized positions in [-1, 1]
        robot_config: Robot configuration

    Returns:
        Actual joint positions
    """
    joint_range = robot_config.joint_limits_upper - robot_config.joint_limits_lower
    joint_center = (robot_config.joint_limits_upper + robot_config.joint_limits_lower) / 2

    actual = normalized_positions * (joint_range / 2) + joint_center

    return actual


def print_robot_info(robot_type: str):
    """Print detailed information about a robot configuration.

    Args:
        robot_type: Robot type identifier
    """
    config = get_robot_config(robot_type)

    print(f"\n{'='*60}")
    print(f"Robot: {config.name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")
    print(f"Degrees of Freedom: {config.dof}")
    print(f"\nJoint Limits (radians/meters):")
    for i in range(config.dof):
        print(f"  Joint {i+1}: [{config.joint_limits_lower[i]:.3f}, "
              f"{config.joint_limits_upper[i]:.3f}]")

    print(f"\nWorkspace Bounds (meters):")
    for axis, bounds in config.workspace_bounds.items():
        print(f"  {axis.upper()}: [{bounds[0]:.3f}, {bounds[1]:.3f}]")

    print(f"\nWorkspace Volume: {get_workspace_volume(config):.3f} m³")

    print(f"\nMax Velocity (rad/s or m/s):")
    print(f"  {config.max_velocity}")

    print(f"\nMax Acceleration (rad/s² or m/s²):")
    print(f"  {config.max_acceleration}")
    print(f"{'='*60}\n")
