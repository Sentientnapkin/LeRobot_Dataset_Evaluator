"""Test with user's actual dataset to debug issues."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.metrics.motion_quality import MotionQualityAnalyzer
from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.utils.robot_models import get_robot_config

print("Testing with SentientNapkin/stack-center-ACT-dual-cam...\n")

# 1. Load dataset
print("1. Loading dataset...")
loader = DatasetLoader(
    dataset_id="SentientNapkin/stack-center-ACT-dual-cam",
    cache_dir="./cache",
    streaming=False
)

try:
    loader.load_dataset()
    print(f"   ✓ Loaded: {loader.num_episodes} episodes, {loader.num_samples} samples")
except Exception as e:
    print(f"   ✗ Failed to load: {e}")
    sys.exit(1)

# 2. Load sample episodes to inspect data
print("\n2. Loading sample episodes to inspect data...")
n_sample_episodes = min(5, loader.num_episodes)
episodes = loader.get_all_episodes_data(
    max_episodes=n_sample_episodes,
    include_images=False,
    show_progress=False
)

print(f"   ✓ Loaded {len(episodes)} episodes")
for i, ep in enumerate(episodes):
    print(f"     Episode {i}:")
    print(f"       - States shape: {ep['states'].shape}")
    print(f"       - States range: [{np.min(ep['states']):.3f}, {np.max(ep['states']):.3f}]")
    print(f"       - Actions shape: {ep['actions'].shape}")
    print(f"       - Actions range: [{np.min(ep['actions']):.3f}, {np.max(ep['actions']):.3f}]")
    print(f"       - Length: {ep['length']} frames")

# 3. Extract features
print("\n3. Extracting features...")
preprocessor = DataPreprocessor()
episodes_features = []
for ep in episodes:
    features = preprocessor.extract_features(ep)
    episodes_features.append(features)

# Print feature statistics
first_features = episodes_features[0]
print(f"   ✓ Extracted features:")
print(f"     - Positions shape: {first_features['positions'].shape}")
print(f"     - Positions range: [{np.min(first_features['positions']):.3f}, {np.max(first_features['positions']):.3f}]")
print(f"     - Velocities range: [{np.min(first_features['velocities']):.3f}, {np.max(first_features['velocities']):.3f}]")
print(f"     - Accelerations range: [{np.min(first_features['accelerations']):.3f}, {np.max(first_features['accelerations']):.3f}]")
print(f"     - Jerk range: [{np.min(first_features['jerk']):.3f}, {np.max(first_features['jerk']):.3f}]")

# 4. Test motion quality
print("\n4. Computing motion quality metrics...")
motion_analyzer = MotionQualityAnalyzer()
motion_results = motion_analyzer.analyze_multiple_episodes(
    episodes_features,
    show_progress=False
)

print(f"   Motion Quality Results:")
print(f"     - Overall score: {motion_results['overall_score']['mean']:.3f}")
print(f"     - Jerk score: {motion_results['jerk_scores']['mean']:.3f}")
print(f"     - Smoothness score: {motion_results['smoothness_scores']['mean']:.3f}")
print(f"     - Acceleration score: {motion_results['acceleration_scores']['mean']:.3f}")
print(f"     - Velocity score: {motion_results['velocity_scores']['mean']:.3f}")
print(f"     - Efficiency score: {motion_results['efficiency_scores']['mean']:.3f}")

# Check individual episode results
print(f"\n   Individual episode smoothness metrics:")
for i, ep_result in enumerate(motion_results['episode_results'][:3]):
    print(f"     Episode {i}:")
    print(f"       - SPARC: {ep_result['smoothness'].get('sparc', 'N/A')}")
    print(f"       - LDLJ: {ep_result['smoothness'].get('ldlj', 'N/A')}")
    print(f"       - Smoothness score: {ep_result['smoothness']['score']:.3f}")

# 5. Test spatial coverage
print("\n5. Computing spatial coverage metrics...")
robot_config = get_robot_config('generic_6dof')
print(f"   Using robot config:")
print(f"     - Workspace X: {robot_config.workspace_bounds['x']}")
print(f"     - Workspace Y: {robot_config.workspace_bounds['y']}")
print(f"     - Workspace Z: {robot_config.workspace_bounds['z']}")

spatial_analyzer = SpatialCoverageAnalyzer(
    robot_config=robot_config,
    voxel_size=0.05
)

spatial_results = spatial_analyzer.analyze_spatial_coverage(
    episodes_features
)

print(f"\n   Spatial Coverage Results:")
print(f"     - Overall score: {spatial_results['overall_score']:.3f}")
print(f"     - Workspace coverage %: {spatial_results['workspace_coverage'].get('coverage_percentage', 0):.1f}%")
print(f"     - Voxels visited: {spatial_results['workspace_coverage'].get('n_voxels_visited', 0)}")
print(f"     - Total voxels: {spatial_results['workspace_coverage'].get('n_voxels_total', 0)}")
print(f"     - Starting variance score: {spatial_results['starting_variance'].get('score', 0):.3f}")

# Check position ranges vs workspace bounds
all_positions = np.vstack([f['positions'] for f in episodes_features])
print(f"\n   Position ranges (first 3 DOFs):")
for i in range(min(3, all_positions.shape[1])):
    print(f"     DOF {i}: [{np.min(all_positions[:, i]):.3f}, {np.max(all_positions[:, i]):.3f}]")

print(f"\n   Position statistics:")
print(f"     - Mean: {np.mean(all_positions, axis=0)[:3]}")
print(f"     - Std: {np.std(all_positions, axis=0)[:3]}")
