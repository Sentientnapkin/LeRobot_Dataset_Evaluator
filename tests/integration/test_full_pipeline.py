"""Test the full pipeline: loader -> preprocessor -> metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.metrics.motion_quality import MotionQualityAnalyzer
from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
from src.utils.robot_models import get_robot_config

print("Testing full pipeline...\n")

# 1. Load dataset
print("1. Loading dataset...")
loader = DatasetLoader(
    dataset_id="lerobot/pusht",
    cache_dir="./cache",
    streaming=False
)
loader.load_dataset()
print(f"   ✓ Loaded: {loader.num_episodes} episodes")

# 2. Load episodes
print("\n2. Loading episode data...")
episodes = loader.get_all_episodes_data(
    max_episodes=5,
    include_images=False,
    show_progress=False
)
print(f"   ✓ Loaded {len(episodes)} episodes")

# 3. Extract features
print("\n3. Extracting features...")
preprocessor = DataPreprocessor()
episodes_features = []
for ep in episodes:
    features = preprocessor.extract_features(ep)
    episodes_features.append(features)
print(f"   ✓ Extracted features for {len(episodes_features)} episodes")

# 4. Compute motion quality metrics
print("\n4. Computing motion quality metrics...")
motion_analyzer = MotionQualityAnalyzer()
motion_results = motion_analyzer.analyze_multiple_episodes(
    episodes_features,
    show_progress=False
)
print(f"   ✓ Motion quality score: {motion_results['overall_score']['mean']:.3f}")

# 5. Compute spatial coverage metrics
print("\n5. Computing spatial coverage metrics...")
robot_config = get_robot_config('generic_6dof')
spatial_analyzer = SpatialCoverageAnalyzer(
    robot_config=robot_config,
    voxel_size=0.05
)
spatial_results = spatial_analyzer.analyze_spatial_coverage(
    episodes_features
)
print(f"   ✓ Spatial coverage computed")
print(f"     Keys: {list(spatial_results.keys())}")

print("\n✅ Full pipeline test passed!")
print(f"\nSummary:")
print(f"  - Episodes analyzed: {len(episodes)}")
print(f"  - Motion quality: {motion_results['overall_score']['mean']:.3f}")
if 'overall_score' in spatial_results:
    if isinstance(spatial_results['overall_score'], dict):
        print(f"  - Spatial coverage: {spatial_results['overall_score']['mean']:.3f}")
    else:
        print(f"  - Spatial coverage: {spatial_results['overall_score']:.3f}")
