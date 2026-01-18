"""Test Spatial Coverage and Data Consistency."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.metrics.data_consistency import DataConsistencyAnalyzer
from src.utils.robot_models import get_robot_config


def create_test_episodes(n_episodes=10, n_frames=50):
    """Create synthetic test episodes."""
    episodes = []

    for i in range(n_episodes):
        t = np.linspace(0, 5, n_frames)
        positions = np.zeros((n_frames, 6))

        # Create varying trajectories
        for j in range(6):
            freq = 0.5 + 0.1 * i  # Vary frequency across episodes
            positions[:, j] = np.sin(2 * np.pi * freq * t / 5) * (0.5 + 0.1 * i)

        # Add small noise
        positions += np.random.randn(*positions.shape) * 0.01

        episodes.append({
            'states': positions,
            'actions': positions.copy(),
            'timestamps': t
        })

    return episodes


def main():
    """Run Phase 3 tests."""
    print("\n" + "="*70)
    print("  Phase 3: Spatial Coverage & Data Consistency Tests")
    print("="*70 + "\n")

    # Create test data
    print("Creating test episodes...")
    episodes = create_test_episodes(n_episodes=10, n_frames=50)
    print(f"✓ Created {len(episodes)} episodes with {len(episodes[0]['states'])} frames each\n")

    # Test 1: Spatial Coverage
    print("Test 1: Spatial Coverage Analysis")
    print("-" * 70)
    try:
        robot_config = get_robot_config('generic_6dof')

        analyzer = SpatialCoverageAnalyzer(
            robot_config=robot_config,
            voxel_size=0.05
        )

        results = analyzer.analyze_spatial_coverage(episodes)

        print("✓ Spatial coverage analysis complete")
        print(f"  - Overall score: {results['overall_score']:.3f}")
        print(f"  - Workspace coverage: {results['workspace_coverage']['coverage_percentage']:.2f}%")
        print(f"  - Starting variance score: {results['individual_scores']['starting_variance']:.3f}")
        print(f"  - Position distribution score: {results['individual_scores']['position_distribution']:.3f}")
        print(f"  - Angle diversity score: {results['individual_scores']['angle_diversity']:.3f}")

    except Exception as e:
        print(f"✗ Spatial coverage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 2: Data Consistency
    print("Test 2: Data Consistency Analysis")
    print("-" * 70)
    try:
        analyzer = DataConsistencyAnalyzer(
            robot_config=robot_config
        )

        results = analyzer.analyze_data_consistency(episodes, expected_fps=10.0)

        print("✓ Data consistency analysis complete")
        print(f"  - Overall score: {results['overall_score']:.3f}")
        print(f"  - Alignment score: {results['individual_scores']['alignment']:.3f}")
        print(f"  - Temporal consistency score: {results['individual_scores']['temporal']:.3f}")
        print(f"  - Missing data score: {results['individual_scores']['missing_data']:.3f}")
        print(f"  - Anomaly score: {results['individual_scores']['anomalies']:.3f}")
        print(f"  - Joint limits score: {results['individual_scores']['joint_limits']:.3f}")
        print(f"  - Anomalies detected: {results['anomalies']['n_anomalies']}")

    except Exception as e:
        print(f"✗ Data consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 3: Spatial Visualizations
    print("Test 3: Spatial Visualizations")
    print("-" * 70)
    try:
        from src.visualizations.spatial_viz import (
            plot_workspace_coverage_3d,
            plot_starting_position_variance,
            plot_coverage_summary
        )

        # Use spatial results from Test 1
        robot_config = get_robot_config('generic_6dof')
        spatial_analyzer = SpatialCoverageAnalyzer(robot_config=robot_config, voxel_size=0.05)
        spatial_results = spatial_analyzer.analyze_spatial_coverage(episodes)

        # Test workspace coverage 3D
        coverage = spatial_results['workspace_coverage']
        fig = plot_workspace_coverage_3d(
            coverage['voxel_grid'],
            coverage['voxel_edges']
        )
        print("✓ Workspace coverage 3D plot")

        # Test starting position variance
        starting_positions = np.array([ep['states'][0] for ep in episodes])
        fig = plot_starting_position_variance(starting_positions)
        print("✓ Starting position variance plot")

        # Test coverage summary
        fig = plot_coverage_summary(spatial_results)
        print("✓ Coverage summary plot")

    except Exception as e:
        print(f"✗ Spatial visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 4: Consistency Visualizations
    print("Test 4: Consistency Visualizations")
    print("-" * 70)
    try:
        from src.visualizations.consistency_viz import (
            plot_temporal_consistency,
            plot_missing_data_heatmap,
            plot_consistency_summary
        )

        # Test temporal consistency
        fig = plot_temporal_consistency(episodes)
        print("✓ Temporal consistency plot")

        # Test missing data heatmap
        fig = plot_missing_data_heatmap(episodes)
        print("✓ Missing data heatmap")

        # Test consistency summary
        robot_config = get_robot_config('generic_6dof')
        consistency_analyzer = DataConsistencyAnalyzer(robot_config=robot_config)
        consistency_results = consistency_analyzer.analyze_data_consistency(episodes)
        fig = plot_consistency_summary(consistency_results)
        print("✓ Consistency summary plot")

    except Exception as e:
        print(f"✗ Consistency visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # All tests passed
    print("="*70)
    print("  ALL PHASE 3 TESTS PASSED ✓")
    print("="*70)
    print()
    print("Summary:")
    print("  ✓ Spatial coverage metrics working")
    print("  ✓ Data consistency checks functional")
    print("  ✓ Spatial visualizations generated")
    print("  ✓ Consistency visualizations generated")
    print()
    print("Phase 3 implementation is complete and ready! 🎉")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Phase 3 test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
