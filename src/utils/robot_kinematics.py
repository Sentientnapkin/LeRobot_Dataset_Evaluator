"""Robot kinematics using Denavit-Hartenberg parameters.

This module provides accurate forward kinematics computation for supported robots
using standard DH parameters. Each robot has its own set of parameters that
accurately represent its physical configuration.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DHParameters:
    """Denavit-Hartenberg parameters for a single joint.

    Standard DH convention:
    - d: offset along previous z to common normal
    - a: length of common normal (joint offset)
    - alpha: angle about common normal from old z to new z
    - theta_offset: fixed offset added to joint angle (for joints not at 0 at home)
    """
    d: float      # Link offset (m)
    a: float      # Link length (m)
    alpha: float  # Link twist (rad)
    theta_offset: float = 0.0  # Joint angle offset (rad)


# DH Parameters for supported robots
# Reference: Robot manufacturers' documentation and URDF files

UR5_DH_PARAMS = [
    # Joint 1: Base rotation
    DHParameters(d=0.089159, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 2: Shoulder
    DHParameters(d=0.0, a=-0.42500, alpha=0.0, theta_offset=0.0),
    # Joint 3: Elbow
    DHParameters(d=0.0, a=-0.39225, alpha=0.0, theta_offset=0.0),
    # Joint 4: Wrist 1
    DHParameters(d=0.10915, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 5: Wrist 2
    DHParameters(d=0.09465, a=0.0, alpha=-np.pi/2, theta_offset=0.0),
    # Joint 6: Wrist 3
    DHParameters(d=0.0823, a=0.0, alpha=0.0, theta_offset=0.0),
]

PANDA_DH_PARAMS = [
    # Joint 1
    DHParameters(d=0.333, a=0.0, alpha=0.0, theta_offset=0.0),
    # Joint 2
    DHParameters(d=0.0, a=0.0, alpha=-np.pi/2, theta_offset=0.0),
    # Joint 3
    DHParameters(d=0.316, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 4
    DHParameters(d=0.0, a=0.0825, alpha=np.pi/2, theta_offset=0.0),
    # Joint 5
    DHParameters(d=0.384, a=-0.0825, alpha=-np.pi/2, theta_offset=0.0),
    # Joint 6
    DHParameters(d=0.0, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 7
    DHParameters(d=0.107, a=0.088, alpha=np.pi/2, theta_offset=0.0),
]

# SO-ARM100: Scaled parameters based on ~0.4m reach
SO_ARM100_DH_PARAMS = [
    # Joint 1: Base rotation
    DHParameters(d=0.05, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 2: Shoulder
    DHParameters(d=0.0, a=-0.12, alpha=0.0, theta_offset=0.0),
    # Joint 3: Elbow
    DHParameters(d=0.0, a=-0.10, alpha=0.0, theta_offset=0.0),
    # Joint 4: Wrist 1
    DHParameters(d=0.04, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    # Joint 5: Wrist 2
    DHParameters(d=0.04, a=0.0, alpha=-np.pi/2, theta_offset=0.0),
    # Joint 6: Wrist 3
    DHParameters(d=0.03, a=0.0, alpha=0.0, theta_offset=0.0),
]

# Generic 6-DOF arm with reasonable default parameters
GENERIC_6DOF_DH_PARAMS = [
    DHParameters(d=0.10, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    DHParameters(d=0.0, a=-0.25, alpha=0.0, theta_offset=0.0),
    DHParameters(d=0.0, a=-0.20, alpha=0.0, theta_offset=0.0),
    DHParameters(d=0.08, a=0.0, alpha=np.pi/2, theta_offset=0.0),
    DHParameters(d=0.08, a=0.0, alpha=-np.pi/2, theta_offset=0.0),
    DHParameters(d=0.05, a=0.0, alpha=0.0, theta_offset=0.0),
]

# Registry of DH parameters by robot type
DH_PARAMS_REGISTRY: Dict[str, List[DHParameters]] = {
    'ur5': UR5_DH_PARAMS,
    'panda': PANDA_DH_PARAMS,
    'so-arm100': SO_ARM100_DH_PARAMS,
    'generic_6dof': GENERIC_6DOF_DH_PARAMS,
}


def dh_transform_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Compute the homogeneous transformation matrix for a single DH joint.

    Uses standard DH convention:
    T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

    Args:
        theta: Joint angle (rad)
        d: Link offset (m)
        a: Link length (m)
        alpha: Link twist (rad)

    Returns:
        4x4 homogeneous transformation matrix
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1]
    ])


def forward_kinematics(
    joint_angles: np.ndarray,
    dh_params: List[DHParameters],
    return_all_frames: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute forward kinematics using DH parameters.

    Args:
        joint_angles: Array of joint angles (rad), shape (n_joints,)
        dh_params: List of DH parameters for each joint
        return_all_frames: If True, return transformation matrices for all frames

    Returns:
        Tuple of:
        - End-effector position (x, y, z)
        - List of 4x4 transformation matrices for each joint frame (if return_all_frames)
    """
    n_joints = min(len(joint_angles), len(dh_params))

    # Initialize with identity (base frame)
    T_base_to_current = np.eye(4)
    frames = [T_base_to_current.copy()]

    for i in range(n_joints):
        params = dh_params[i]
        theta = joint_angles[i] + params.theta_offset

        T_joint = dh_transform_matrix(theta, params.d, params.a, params.alpha)
        T_base_to_current = T_base_to_current @ T_joint

        if return_all_frames:
            frames.append(T_base_to_current.copy())

    # Extract end-effector position
    ee_position = T_base_to_current[:3, 3]

    return ee_position, frames


def get_joint_positions(
    joint_angles: np.ndarray,
    dh_params: List[DHParameters]
) -> np.ndarray:
    """Get 3D positions of all joints for visualization.

    Args:
        joint_angles: Array of joint angles (rad)
        dh_params: List of DH parameters

    Returns:
        Array of joint positions, shape (n_joints + 1, 3)
        First row is base (0, 0, 0), last row is end-effector
    """
    _, frames = forward_kinematics(joint_angles, dh_params, return_all_frames=True)

    positions = np.array([frame[:3, 3] for frame in frames])
    return positions


def get_joint_axes(
    joint_angles: np.ndarray,
    dh_params: List[DHParameters]
) -> np.ndarray:
    """Get rotation axes (z-axis direction) for each joint.

    Args:
        joint_angles: Array of joint angles (rad)
        dh_params: List of DH parameters

    Returns:
        Array of z-axis directions, shape (n_joints, 3)
    """
    _, frames = forward_kinematics(joint_angles, dh_params, return_all_frames=True)

    # Z-axis is the third column of rotation matrix
    axes = np.array([frame[:3, 2] for frame in frames[:-1]])  # Exclude end-effector
    return axes


def get_dh_params(robot_type: str) -> List[DHParameters]:
    """Get DH parameters for a robot type.

    Args:
        robot_type: Robot type identifier (e.g., 'ur5', 'panda', 'so-arm100')

    Returns:
        List of DHParameters for the robot

    Raises:
        ValueError: If robot type is not supported
    """
    robot_type_lower = robot_type.lower()

    if robot_type_lower not in DH_PARAMS_REGISTRY:
        available = ', '.join(DH_PARAMS_REGISTRY.keys())
        raise ValueError(
            f"Unknown robot type: {robot_type}. Available types: {available}"
        )

    return DH_PARAMS_REGISTRY[robot_type_lower]


def compute_workspace_samples(
    dh_params: List[DHParameters],
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    n_samples: int = 10000,
    seed: Optional[int] = None
) -> np.ndarray:
    """Sample the reachable workspace using Monte Carlo sampling.

    Args:
        dh_params: DH parameters for the robot
        joint_limits_lower: Lower joint limits (rad)
        joint_limits_upper: Upper joint limits (rad)
        n_samples: Number of random configurations to sample
        seed: Random seed for reproducibility

    Returns:
        Array of end-effector positions, shape (n_samples, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    n_joints = len(dh_params)

    # Generate random joint configurations within limits
    joint_configs = np.random.uniform(
        joint_limits_lower[:n_joints],
        joint_limits_upper[:n_joints],
        size=(n_samples, n_joints)
    )

    # Compute end-effector positions for all configurations
    positions = np.zeros((n_samples, 3))

    for i, joints in enumerate(joint_configs):
        ee_pos, _ = forward_kinematics(joints, dh_params, return_all_frames=False)
        positions[i] = ee_pos

    return positions


def estimate_arm_reach(dh_params: List[DHParameters]) -> float:
    """Estimate the maximum reach of a robot arm.

    This computes an approximate maximum reach by summing link lengths.

    Args:
        dh_params: DH parameters for the robot

    Returns:
        Estimated maximum reach in meters
    """
    # Sum of absolute values of a and d parameters gives approximate reach
    reach = sum(abs(p.a) + abs(p.d) for p in dh_params)
    return reach


def get_link_lengths(dh_params: List[DHParameters]) -> List[float]:
    """Get visual link lengths for 3D rendering.

    Returns the Euclidean distance that each link spans.

    Args:
        dh_params: DH parameters for the robot

    Returns:
        List of link lengths (m)
    """
    lengths = []
    for p in dh_params:
        # Link length is approximately sqrt(a^2 + d^2)
        length = np.sqrt(p.a**2 + p.d**2)
        # Ensure minimum visible length for visualization
        length = max(length, 0.02)
        lengths.append(length)
    return lengths


class RobotKinematics:
    """Class for robot kinematics computations.

    Provides a convenient interface for forward kinematics and workspace analysis.
    """

    def __init__(self, robot_type: str, joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Initialize robot kinematics.

        Args:
            robot_type: Robot type identifier
            joint_limits: Optional tuple of (lower, upper) joint limits
        """
        self.robot_type = robot_type
        self.dh_params = get_dh_params(robot_type)
        self.n_joints = len(self.dh_params)

        if joint_limits is not None:
            self.joint_limits_lower = joint_limits[0]
            self.joint_limits_upper = joint_limits[1]
        else:
            # Default limits (will be overridden if RobotConfig is used)
            self.joint_limits_lower = np.array([-np.pi] * self.n_joints)
            self.joint_limits_upper = np.array([np.pi] * self.n_joints)

        self._workspace_samples = None
        self._workspace_voxels = None

    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute forward kinematics.

        Args:
            joint_angles: Joint angles (rad)

        Returns:
            Tuple of (end-effector position, list of frame transforms)
        """
        return forward_kinematics(joint_angles, self.dh_params)

    def get_joint_positions(self, joint_angles: np.ndarray) -> np.ndarray:
        """Get 3D positions of all joints.

        Args:
            joint_angles: Joint angles (rad)

        Returns:
            Array of positions, shape (n_joints + 1, 3)
        """
        return get_joint_positions(joint_angles, self.dh_params)

    def get_max_reach(self) -> float:
        """Get estimated maximum reach of the robot."""
        return estimate_arm_reach(self.dh_params)

    def sample_workspace(self, n_samples: int = 10000, seed: Optional[int] = 42) -> np.ndarray:
        """Sample the reachable workspace.

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            Array of end-effector positions
        """
        if self._workspace_samples is None or len(self._workspace_samples) != n_samples:
            self._workspace_samples = compute_workspace_samples(
                self.dh_params,
                self.joint_limits_lower,
                self.joint_limits_upper,
                n_samples,
                seed
            )
        return self._workspace_samples

    def compute_reachable_voxels(
        self,
        voxel_size: float = 0.05,
        n_samples: int = 10000
    ) -> Tuple[np.ndarray, Dict]:
        """Compute voxel grid of reachable workspace.

        Args:
            voxel_size: Size of each voxel (m)
            n_samples: Number of workspace samples

        Returns:
            Tuple of:
            - Boolean voxel grid (True = reachable)
            - Dict with grid bounds and metadata
        """
        # Sample workspace
        positions = self.sample_workspace(n_samples)

        # Compute bounds with padding
        mins = positions.min(axis=0) - voxel_size
        maxs = positions.max(axis=0) + voxel_size

        # Create voxel grid
        x_bins = int((maxs[0] - mins[0]) / voxel_size) + 1
        y_bins = int((maxs[1] - mins[1]) / voxel_size) + 1
        z_bins = int((maxs[2] - mins[2]) / voxel_size) + 1

        # Count voxels visited by workspace samples
        voxel_grid, edges = np.histogramdd(
            positions,
            bins=[x_bins, y_bins, z_bins],
            range=[(mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])]
        )

        # Convert to boolean (reachable = visited at least once)
        reachable_grid = voxel_grid > 0

        metadata = {
            'bounds': {
                'x': (mins[0], maxs[0]),
                'y': (mins[1], maxs[1]),
                'z': (mins[2], maxs[2])
            },
            'edges': edges,
            'voxel_size': voxel_size,
            'n_reachable_voxels': int(np.sum(reachable_grid)),
            'n_total_voxels': x_bins * y_bins * z_bins
        }

        self._workspace_voxels = (reachable_grid, metadata)
        return reachable_grid, metadata
