"""Test joint space detection fix."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.utils.robot_models import get_robot_config

print("Testing joint space detection fix...\n")

# Create sample joint space data (like user's dataset)
joint_angles = np.array([
    [-41.0, -99.0, -80.0, 10.0, 20.0, 30.0],
    [35.0, 61.0, 99.0, -10.0, -20.0, -30.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [20.0, -50.0, 40.0, 5.0, 10.0, 15.0],
    [-20.0, 50.0, -40.0, -5.0, -10.0, -15.0],
])

print(f"Sample data shape: {joint_angles.shape}")
print(f"Data range: [{np.min(joint_angles):.1f}, {np.max(joint_angles):.1f}]")

# Test detection
robot_config = get_robot_config('generic_6dof')
analyzer = SpatialCoverageAnalyzer(
    robot_config=robot_config,
    voxel_size=0.05
)

# Test _detect_data_space
data_space = analyzer._detect_data_space(joint_angles)
print(f"\nDetected data space: {data_space}")

# Test compute_workspace_coverage
result = analyzer.compute_workspace_coverage(joint_angles)

print(f"\nCoverage result:")
print(f"  - Data space: {result.get('data_space', 'N/A')}")
print(f"  - Coverage %: {result['coverage_percentage']:.1f}%")
print(f"  - Voxels visited: {result['n_voxels_visited']}")
print(f"  - Total voxels: {result['n_voxels_total']}")
print(f"  - Score: {result['score']:.3f}")

if 'joint_ranges' in result:
    print(f"  - Joint ranges:")
    for i, (jmin, jmax) in enumerate(result['joint_ranges']):
        print(f"      DOF {i}: [{jmin:.1f}, {jmax:.1f}]")

print("\n✅ Joint space detection test complete!")
