"""Comprehensive test with user's dataset and reference datasets."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.metrics.motion_quality import MotionQualityAnalyzer
from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.utils.robot_models import get_robot_config

def analyze_dataset(dataset_id, robot_type, max_episodes=50):
    """Analyze a dataset and print results."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset_id}")
    print(f"Robot: {robot_type}")
    print(f"Max episodes: {max_episodes}")
    print('='*80)

    try:
        # Load dataset
        print("\n1. Loading dataset...")
        loader = DatasetLoader(
            dataset_id=dataset_id,
            cache_dir="./cache",
            streaming=False
        )
        loader.load_dataset()
        print(f"   ✓ Total episodes: {loader.num_episodes}")

        # Load episodes
        n_episodes = min(max_episodes, loader.num_episodes)
        print(f"\n2. Loading {n_episodes} episodes...")
        episodes = loader.get_all_episodes_data(
            max_episodes=n_episodes,
            include_images=False,
            show_progress=True
        )
        print(f"   ✓ Loaded {len(episodes)} episodes")

        # Extract features
        print(f"\n3. Extracting features...")
        robot_config = get_robot_config(robot_type)
        preprocessor = DataPreprocessor(robot_config)
        episodes_features = []
        for ep in episodes:
            features = preprocessor.extract_features(ep)
            episodes_features.append(features)
        print(f"   ✓ Features extracted")

        # Data statistics
        all_positions = np.vstack([f['positions'] for f in episodes_features])
        print(f"\n4. Data statistics:")
        print(f"   - Position range: [{np.min(all_positions):.1f}, {np.max(all_positions):.1f}]")
        print(f"   - Position mean: {np.mean(all_positions):.2f}")
        print(f"   - Position std: {np.std(all_positions):.2f}")

        # Motion quality
        print(f"\n5. Motion Quality Analysis...")
        motion_analyzer = MotionQualityAnalyzer()
        motion_results = motion_analyzer.analyze_multiple_episodes(
            episodes_features,
            show_progress=False
        )
        print(f"   Results:")
        print(f"     - Overall score: {motion_results['overall_score']['mean']:.3f}")
        print(f"     - Jerk score: {motion_results['jerk_scores']['mean']:.3f}")
        print(f"     - Smoothness score: {motion_results['smoothness_scores']['mean']:.3f}")
        print(f"     - Acceleration score: {motion_results['acceleration_scores']['mean']:.3f}")

        # Spatial coverage
        print(f"\n6. Spatial Coverage Analysis...")
        spatial_analyzer = SpatialCoverageAnalyzer(
            robot_config=robot_config,
            voxel_size=0.05
        )
        spatial_results = spatial_analyzer.analyze_spatial_coverage(
            episodes_features
        )
        print(f"   Results:")
        print(f"     - Overall score: {spatial_results['overall_score']:.3f}")
        print(f"     - Data space: {spatial_results.get('data_space', 'N/A')}")
        print(f"     - Coverage: {spatial_results['workspace_coverage']['coverage_percentage']:.2f}%")
        print(f"     - Voxels: {spatial_results['workspace_coverage']['n_voxels_visited']:,} / {spatial_results['workspace_coverage']['n_voxels_total']:,}")
        print(f"     - Starting variance score: {spatial_results['starting_variance']['score']:.3f}")

        return {
            'motion_quality': motion_results['overall_score']['mean'],
            'spatial_coverage': spatial_results['overall_score'],
            'smoothness': motion_results['smoothness_scores']['mean'],
            'coverage_pct': spatial_results['workspace_coverage']['coverage_percentage']
        }

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test datasets
print("COMPREHENSIVE DATASET ANALYSIS")
print("="*80)

results = {}

# User's dataset (more episodes)
results['user'] = analyze_dataset(
    "SentientNapkin/stack-center-ACT-dual-cam",
    "generic_6dof",
    max_episodes=100
)

# Reference dataset 1: sherryxychen (SO-101)
results['sherry'] = analyze_dataset(
    "sherryxychen/2025-09-01_pick-and-place-block",
    "so-arm100",
    max_episodes=50
)

# Reference dataset 2: berkeley autolab (UR5)
results['berkeley'] = analyze_dataset(
    "lerobot/berkeley_autolab_ur5",
    "ur5",
    max_episodes=50
)

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY COMPARISON")
print('='*80)
print(f"{'Dataset':<40} {'Motion':<10} {'Smooth':<10} {'Spatial':<10} {'Cov %':<10}")
print('-'*80)

for name, result in results.items():
    if result:
        print(f"{name:<40} {result['motion_quality']:<10.3f} {result['smoothness']:<10.3f} {result['spatial_coverage']:<10.3f} {result['coverage_pct']:<10.2f}")

print('='*80)
