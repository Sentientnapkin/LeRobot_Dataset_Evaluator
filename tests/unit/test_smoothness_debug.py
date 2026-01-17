"""Debug smoothness metric computation."""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.motion_quality import MotionQualityAnalyzer
from src.data.preprocessor import DataPreprocessor


def create_test_trajectory():
    """Create simple test trajectory."""
    t = np.linspace(0, 10, 100)
    positions = np.zeros((100, 6))

    for i in range(6):
        positions[:, i] = np.sin(2 * np.pi * 0.5 * t / 10)

    return {
        'states': positions,
        'actions': positions.copy(),
        'timestamps': t
    }


def main():
    print("\n" + "="*70)
    print("  Debugging Smoothness Metric")
    print("="*70 + "\n")

    preprocessor = DataPreprocessor()
    analyzer = MotionQualityAnalyzer(smoothness_method="sparc")

    episode = create_test_trajectory()
    features = preprocessor.extract_features(episode, smooth_sigma=1.0)

    print("Feature shapes:")
    print(f"  Positions: {features['positions'].shape}")
    print(f"  Velocities: {features['velocities'].shape}")
    print(f"  Jerk: {features['jerk'].shape}")
    print()

    # Test SPARC computation
    print("Testing SPARC computation...")
    try:
        sparc_value = analyzer.compute_sparc_smoothness(
            features['velocities'],
            features['timestamps']
        )
        print(f"  SPARC value: {sparc_value}")
        print(f"  SPARC type: {type(sparc_value)}")
    except Exception as e:
        print(f"  Error computing SPARC: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test LDLJ computation
    print("Testing LDLJ computation...")
    try:
        ldlj_value = analyzer.compute_ldlj_smoothness(
            features['jerk'],
            features['timestamps'],
            features['positions']
        )
        print(f"  LDLJ value: {ldlj_value}")
        print(f"  LDLJ type: {type(ldlj_value)}")
    except Exception as e:
        print(f"  Error computing LDLJ: {e}")
        import traceback
        traceback.print_exc()

    print()

    # Test full smoothness metrics
    print("Testing full smoothness metrics...")
    try:
        smoothness_result = analyzer.compute_smoothness_metrics(
            features['positions'],
            features['velocities'],
            features['jerk'],
            features['timestamps']
        )
        print(f"  Result: {smoothness_result}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
