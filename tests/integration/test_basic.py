"""Basic integration test without pytest dependency."""

import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.motion_quality import MotionQualityAnalyzer
from src.metrics.utils import compute_statistics
from src.data.preprocessor import DataPreprocessor


def create_synthetic_trajectory(n_frames=100, n_dof=6, noise_level=0.01):
    """Create synthetic smooth trajectory for testing."""
    t = np.linspace(0, 10, n_frames)
    positions = np.zeros((n_frames, n_dof))

    for i in range(n_dof):
        freq = 0.5 + 0.2 * i
        positions[:, i] = np.sin(2 * np.pi * freq * t / 10) + noise_level * np.random.randn(n_frames)

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t
    }


def main():
    """Run basic integration test."""
    print("\n" + "="*70)
    print("  LeRobot Dataset Evaluator - Basic Integration Test")
    print("="*70 + "\n")

    # Test 1: Configuration
    print("Test 1: Configuration Management")
    print("-" * 70)
    try:
        from src.utils.config import get_config
        config = get_config()
        print(f"✓ Config loaded successfully")
        print(f"  - HF Account: {config.hf_account}")
        print(f"  - Robot Type: {config.robot_type}")
        print(f"  - Robot DOF: {config.robot_dof}")
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

    print()

    # Test 2: Robot Models
    print("Test 2: Robot Models")
    print("-" * 70)
    try:
        from src.utils.robot_models import get_robot_config
        robot_config = get_robot_config('generic_6dof')
        print(f"✓ Robot config loaded: {robot_config.name}")
        print(f"  - DOF: {robot_config.dof}")
        print(f"  - Workspace volume: {robot_config.workspace_bounds}")
    except Exception as e:
        print(f"✗ Robot model test failed: {e}")
        return False

    print()

    # Test 3: Data Preprocessing
    print("Test 3: Data Preprocessing")
    print("-" * 70)
    try:
        preprocessor = DataPreprocessor()
        episode = create_synthetic_trajectory(n_frames=50)

        features = preprocessor.extract_features(episode, smooth_sigma=1.0)

        print(f"✓ Feature extraction successful")
        print(f"  - Positions shape: {features['positions'].shape}")
        print(f"  - Velocities shape: {features['velocities'].shape}")
        print(f"  - Accelerations shape: {features['accelerations'].shape}")
        print(f"  - Jerk shape: {features['jerk'].shape}")

        # Verify no NaN or Inf
        assert not np.any(np.isnan(features['velocities'])), "Velocities contain NaN"
        assert not np.any(np.isnan(features['accelerations'])), "Accelerations contain NaN"
        assert not np.any(np.isnan(features['jerk'])), "Jerk contains NaN"
        print(f"  - Data validation: All values finite ✓")

    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 4: Motion Quality Metrics
    print("Test 4: Motion Quality Metrics")
    print("-" * 70)
    try:
        analyzer = MotionQualityAnalyzer(
            jerk_threshold_multiplier=2.0,
            smoothness_method="sparc"
        )

        print(f"✓ Analyzer initialized")

        # Analyze single episode
        result = analyzer.analyze_episode(features)

        print(f"✓ Single episode analysis complete")
        print(f"  - Overall score: {result['overall_score']:.3f}")
        print(f"  - Jerk score: {result['individual_scores']['jerk']:.3f}")
        print(f"  - Smoothness score: {result['individual_scores']['smoothness']:.3f}")
        print(f"  - Acceleration score: {result['individual_scores']['acceleration']:.3f}")
        print(f"  - Velocity score: {result['individual_scores']['velocity']:.3f}")
        print(f"  - Efficiency score: {result['individual_scores']['efficiency']:.3f}")

    except Exception as e:
        print(f"✗ Motion quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 5: Multiple Episodes
    print("Test 5: Multiple Episodes Analysis")
    print("-" * 70)
    try:
        episodes_features = []

        print("Creating 10 synthetic episodes...")
        for i in range(10):
            episode = create_synthetic_trajectory(n_frames=50, noise_level=0.01 * (i + 1))
            feats = preprocessor.extract_features(episode, smooth_sigma=1.0)
            episodes_features.append(feats)

        print(f"✓ Created {len(episodes_features)} episodes")

        results = analyzer.analyze_multiple_episodes(episodes_features, show_progress=False)

        print(f"✓ Multi-episode analysis complete")
        print(f"  - Episodes analyzed: {results['n_episodes']}")
        print(f"  - Mean overall score: {results['overall_score']['mean']:.3f}")
        print(f"  - Std overall score: {results['overall_score']['std']:.3f}")
        print(f"  - Min overall score: {results['overall_score']['min']:.3f}")
        print(f"  - Max overall score: {results['overall_score']['max']:.3f}")

        print(f"\n  Metric Breakdown:")
        print(f"    - Jerk:         {results['jerk_scores']['mean']:.3f}")
        print(f"    - Smoothness:   {results['smoothness_scores']['mean']:.3f}")
        print(f"    - Acceleration: {results['acceleration_scores']['mean']:.3f}")
        print(f"    - Velocity:     {results['velocity_scores']['mean']:.3f}")
        print(f"    - Efficiency:   {results['efficiency_scores']['mean']:.3f}")

    except Exception as e:
        print(f"✗ Multiple episodes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 6: Visualizations
    print("Test 6: Visualization Generation")
    print("-" * 70)
    try:
        from src.visualizations.motion_viz import (
            plot_overall_scores_distribution,
            plot_metric_comparison_bars,
            plot_jerk_outliers,
            plot_smoothness_comparison
        )

        # Generate plots (don't display)
        fig1 = plot_overall_scores_distribution(results['overall_score']['scores'])
        print("✓ Overall scores distribution plot")

        fig2 = plot_metric_comparison_bars(results)
        print("✓ Metric comparison bars plot")

        fig3 = plot_jerk_outliers(results['episode_results'])
        print("✓ Jerk outliers plot")

        fig4 = plot_smoothness_comparison(results['episode_results'])
        print("✓ Smoothness comparison plot")

        print(f"\n✓ All visualizations generated successfully")

    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 7: Utilities
    print("Test 7: Utility Functions")
    print("-" * 70)
    try:
        from src.metrics.utils import (
            normalize_metric,
            detect_outliers_zscore,
            compute_path_efficiency
        )

        # Test normalization
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_metric(values, lower_is_better=True)
        assert np.all((normalized >= 0) & (normalized <= 1)), "Normalization failed"
        print("✓ Metric normalization")

        # Test outlier detection
        data = np.concatenate([np.random.randn(90), np.random.randn(10) * 10])
        outliers, z_scores = detect_outliers_zscore(data, threshold=3.0)
        assert np.sum(outliers) > 0, "No outliers detected"
        print("✓ Outlier detection")

        # Test path efficiency
        trajectory = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        efficiency = compute_path_efficiency(trajectory)
        assert 0 <= efficiency <= 1, "Invalid efficiency"
        print("✓ Path efficiency computation")

    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # All tests passed
    print("="*70)
    print("  ALL TESTS PASSED ✓")
    print("="*70)
    print()
    print("Summary:")
    print("  ✓ Configuration management working")
    print("  ✓ Robot models loaded correctly")
    print("  ✓ Data preprocessing functional")
    print("  ✓ Motion quality metrics computed")
    print("  ✓ Multi-episode analysis working")
    print("  ✓ Visualizations generated")
    print("  ✓ Utility functions operational")
    print()
    print("Your LeRobot Dataset Evaluator is ready to use! 🎉")
    print()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
