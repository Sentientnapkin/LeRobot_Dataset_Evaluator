"""Test Streamlit app compatibility without actually running Streamlit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing Streamlit app compatibility...\n")

# Test all imports
print("1. Testing imports...")
try:
    from src.data.loader import DatasetLoader
    from src.data.preprocessor import DataPreprocessor
    from src.metrics.motion_quality import MotionQualityAnalyzer
    from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
    from src.metrics.data_consistency import DataConsistencyAnalyzer
    from src.visualizations.motion_viz import (
        plot_jerk_timeseries,
        plot_jerk_distribution,
        plot_overall_scores_distribution
    )
    from src.visualizations.spatial_viz import (
        plot_workspace_coverage_3d,
        plot_starting_position_variance
    )
    from src.visualizations.consistency_viz import (
        plot_temporal_consistency,
        plot_missing_data_heatmap
    )
    from src.utils.config import get_config
    from src.utils.robot_models import get_robot_config
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test configuration loading
print("\n2. Testing configuration...")
try:
    config = get_config()
    print(f"   ✓ Config loaded: {config.hf_account}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    sys.exit(1)

# Test robot config
print("\n3. Testing robot config...")
try:
    robot_config = get_robot_config('generic_6dof')
    print(f"   ✓ Robot config loaded: {robot_config.name}")
except Exception as e:
    print(f"   ✗ Robot config failed: {e}")
    sys.exit(1)

# Simulate Streamlit workflow (without actual Streamlit)
print("\n4. Simulating Streamlit workflow...")
try:
    # Initialize loader
    loader = DatasetLoader(
        dataset_id="lerobot/pusht",
        cache_dir="./cache",
        streaming=False
    )

    # Load dataset
    loader.load_dataset()

    # Get number of episodes (this was causing the original error)
    num_episodes = loader.num_episodes
    print(f"   ✓ Number of episodes: {num_episodes}")

    # Load a few episodes
    max_episodes = min(3, num_episodes)
    episodes = loader.get_all_episodes_data(
        max_episodes=max_episodes,
        include_images=False,
        show_progress=False
    )
    print(f"   ✓ Loaded {len(episodes)} episodes")

    # Extract features
    preprocessor = DataPreprocessor()
    episodes_features = []
    for ep in episodes:
        features = preprocessor.extract_features(ep)
        episodes_features.append(features)
    print(f"   ✓ Extracted features")

    # Compute motion quality
    motion_analyzer = MotionQualityAnalyzer()
    motion_results = motion_analyzer.analyze_multiple_episodes(
        episodes_features,
        show_progress=False
    )
    print(f"   ✓ Motion quality: {motion_results['overall_score']['mean']:.3f}")

    # Compute spatial coverage
    spatial_analyzer = SpatialCoverageAnalyzer(
        robot_config=robot_config,
        voxel_size=0.05
    )
    spatial_results = spatial_analyzer.analyze_spatial_coverage(
        episodes_features
    )
    print(f"   ✓ Spatial coverage: {spatial_results['overall_score']:.3f}")

    # Compute data consistency
    consistency_analyzer = DataConsistencyAnalyzer(
        robot_config=robot_config
    )
    consistency_results = consistency_analyzer.analyze_data_consistency(
        episodes_features
    )
    print(f"   ✓ Data consistency: {consistency_results['overall_score']:.3f}")

    print("\n✅ Streamlit compatibility test passed!")
    print("\nThe Streamlit app should now work correctly.")
    print("To test it, run: streamlit run app.py")

except Exception as e:
    print(f"   ✗ Workflow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
