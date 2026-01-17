"""Test new execution quality metrics on real datasets."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.metrics.execution_quality import ExecutionQualityAnalyzer
from src.metrics.motion_quality import MotionQualityAnalyzer  # Old metrics for comparison
from src.utils.robot_models import get_robot_config


def analyze_dataset_with_new_metrics(dataset_id, robot_type, max_episodes=50):
    """Analyze a dataset with new execution quality metrics."""
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

        # Analyze with OLD metrics (for comparison)
        print(f"\n4. Computing OLD motion quality metrics...")
        old_analyzer = MotionQualityAnalyzer()
        old_results = old_analyzer.analyze_multiple_episodes(
            episodes_features,
            show_progress=False
        )

        # Analyze with NEW metrics
        print(f"\n5. Computing NEW execution quality metrics...")
        new_analyzer = ExecutionQualityAnalyzer()
        new_results = new_analyzer.analyze_multiple_episodes(
            episodes_features,
            show_progress=False
        )

        # Print comparison
        print(f"\n{'='*80}")
        print(f"RESULTS COMPARISON")
        print('='*80)

        print(f"\n📊 OLD METRICS (SPARC-based smoothness):")
        print(f"   Overall Score: {old_results['overall_score']['mean']:.3f} ± {old_results['overall_score']['std']:.3f}")
        print(f"   - Jerk: {old_results['jerk_scores']['mean']:.3f}")
        print(f"   - Smoothness (SPARC): {old_results['smoothness_scores']['mean']:.3f}")
        print(f"   - Acceleration: {old_results['acceleration_scores']['mean']:.3f}")
        print(f"   - Velocity: {old_results['velocity_scores']['mean']:.3f}")
        print(f"   - Efficiency: {old_results['efficiency_scores']['mean']:.3f}")

        print(f"\n✨ NEW METRICS (Execution quality):")
        print(f"   Overall Score: {new_results['overall_score']['mean']:.3f} ± {new_results['overall_score']['std']:.3f}")
        print(f"   - Movement Units: {new_results['movement_units_scores']['mean']:.3f}")
        print(f"   - Hesitations: {new_results['hesitation_scores']['mean']:.3f}")
        print(f"   - Submovements: {new_results['submovement_scores']['mean']:.3f}")
        print(f"   - Jerk (recalibrated): {new_results['jerk_scores']['mean']:.3f}")

        # Interpret new score
        new_score = new_results['overall_score']['mean']
        if new_score > 0.65:
            quality = "🟢 GOOD"
        elif new_score > 0.45:
            quality = "🟡 ACCEPTABLE"
        else:
            quality = "🔴 POOR"

        print(f"\n   Quality Assessment: {quality}")

        # Show example episode details
        print(f"\n📋 Example Episode Details (Episode 0):")
        example = new_results['episode_results'][0]
        print(f"   Movement Units: {example['movement_units']['n_movement_units']} units "
              f"({example['movement_units']['nmu_per_second']:.2f}/sec)")
        print(f"   Hesitations: {example['hesitations']['n_hesitations']} reversals "
              f"({example['hesitations']['hesitation_rate']:.2f}/sec)")
        print(f"   Submovements: {example['submovements']['n_submovements']} peaks "
              f"({example['submovements']['submovement_index']:.2f}/sec)")

        return {
            'dataset_id': dataset_id,
            'old_score': old_results['overall_score']['mean'],
            'new_score': new_results['overall_score']['mean'],
            'movement_units': new_results['movement_units_scores']['mean'],
            'hesitations': new_results['hesitation_scores']['mean'],
            'submovements': new_results['submovement_scores']['mean'],
            'jerk': new_results['jerk_scores']['mean'],
            'quality': quality
        }

    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test datasets
print("="*80)
print("TESTING NEW EXECUTION QUALITY METRICS ON REAL DATASETS")
print("="*80)

results = {}

# 1. Your dataset
print("\n\n🎯 YOUR DATASET")
results['user'] = analyze_dataset_with_new_metrics(
    "SentientNapkin/stack-center-ACT-dual-cam",
    "generic_6dof",
    max_episodes=100
)

# 2. Reference: berkeley autolab
print("\n\n📚 REFERENCE DATASET #1: Berkeley Autolab UR5")
results['berkeley'] = analyze_dataset_with_new_metrics(
    "lerobot/berkeley_autolab_ur5",
    "ur5",
    max_episodes=50
)

# 3. Reference: sherryxychen
print("\n\n📚 REFERENCE DATASET #2: Sherry's Pick-and-Place")
results['sherry'] = analyze_dataset_with_new_metrics(
    "sherryxychen/2025-09-01_pick-and-place-block",
    "so-arm100",
    max_episodes=50
)

# Final comparison
print("\n\n" + "="*80)
print("FINAL COMPARISON - OLD vs NEW METRICS")
print("="*80)

print(f"\n{'Dataset':<45} {'Old Score':<12} {'New Score':<12} {'Quality':<15}")
print("-"*80)

for name, result in results.items():
    if result:
        dataset_name = result['dataset_id'].split('/')[-1][:40]
        print(f"{dataset_name:<45} {result['old_score']:<12.3f} {result['new_score']:<12.3f} {result['quality']:<15}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

if all(r for r in results.values()):
    # Calculate improvements
    user_improvement = results['user']['new_score'] - results['user']['old_score']
    berkeley_improvement = results['berkeley']['new_score'] - results['berkeley']['old_score']
    sherry_improvement = results['sherry']['new_score'] - results['sherry']['old_score']

    print(f"\n✅ Score Improvements (New - Old):")
    print(f"   Your dataset:     +{user_improvement:.3f} ({user_improvement/results['user']['old_score']*100:.0f}% increase)")
    print(f"   Berkeley UR5:     +{berkeley_improvement:.3f} ({berkeley_improvement/results['berkeley']['old_score']*100:.0f}% increase)")
    print(f"   Sherry's dataset: +{sherry_improvement:.3f} ({sherry_improvement/results['sherry']['old_score']*100:.0f}% increase)")

    # Check discrimination
    new_scores = [results['user']['new_score'], results['berkeley']['new_score'], results['sherry']['new_score']]
    score_range = max(new_scores) - min(new_scores)
    old_scores = [results['user']['old_score'], results['berkeley']['old_score'], results['sherry']['old_score']]
    old_range = max(old_scores) - min(old_scores)

    print(f"\n📊 Discrimination:")
    print(f"   Old metrics range: {old_range:.3f} (narrow, not discriminative)")
    print(f"   New metrics range: {score_range:.3f} (wider, more discriminative)")

    # Detailed breakdown for your dataset
    print(f"\n🎯 YOUR DATASET DETAILED BREAKDOWN:")
    print(f"   Movement Units Score: {results['user']['movement_units']:.3f}")
    print(f"   Hesitation Score:     {results['user']['hesitations']:.3f}")
    print(f"   Submovement Score:    {results['user']['submovements']:.3f}")
    print(f"   Jerk Score:           {results['user']['jerk']:.3f}")
    print(f"   → Overall:            {results['user']['new_score']:.3f} {results['user']['quality']}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR YOUR DATASET:")
    if results['user']['hesitations'] < 0.5:
        print(f"   ⚠️  Hesitation score is low ({results['user']['hesitations']:.3f})")
        print(f"       → Work on reducing velocity reversals and corrections")
    if results['user']['movement_units'] < 0.5:
        print(f"   ⚠️  Movement units score is low ({results['user']['movement_units']:.3f})")
        print(f"       → Try making fewer, larger purposeful movements")
    if results['user']['submovements'] < 0.5:
        print(f"   ⚠️  Submovement score is low ({results['user']['submovements']:.3f})")
        print(f"       → Reduce corrective submovements by improving control confidence")

    if results['user']['new_score'] > 0.65:
        print(f"   ✅ Your dataset quality is GOOD for teleoperation!")
        print(f"   ✅ Ready for ACT training")

print("\n" + "="*80)
print("✅ NEW METRICS TEST COMPLETE!")
print("="*80)
