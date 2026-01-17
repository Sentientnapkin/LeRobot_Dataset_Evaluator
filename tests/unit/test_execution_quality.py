"""Test execution quality metrics."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metrics.execution_quality import ExecutionQualityAnalyzer


def generate_smooth_trajectory(n_frames=100, n_dof=6):
    """Generate a smooth, skilled trajectory."""
    t = np.linspace(0, 5, n_frames)

    # Smooth sinusoidal motion (skilled operator)
    positions = np.zeros((n_frames, n_dof))
    for i in range(n_dof):
        positions[:, i] = np.sin(2 * np.pi * 0.5 * t + i * 0.1) * 50

    # Compute derivatives
    dt = t[1] - t[0]
    velocities = np.gradient(positions, dt, axis=0)
    accelerations = np.gradient(velocities, dt, axis=0)
    jerk = np.gradient(accelerations, dt, axis=0)

    return {
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'jerk': jerk,
        'timestamps': t
    }


def generate_jerky_trajectory(n_frames=100, n_dof=6):
    """Generate a jerky, unskilled trajectory with hesitations."""
    t = np.linspace(0, 5, n_frames)

    # Jerky motion with random hesitations (unskilled operator)
    positions = np.zeros((n_frames, n_dof))
    for i in range(n_dof):
        # Base motion with high frequency noise
        positions[:, i] = np.sin(2 * np.pi * 0.5 * t + i * 0.1) * 50
        # Add submovements (corrective actions)
        positions[:, i] += np.sin(2 * np.pi * 3.0 * t) * 10
        # Add random hesitations
        positions[:, i] += np.random.randn(n_frames) * 5

    # Compute derivatives
    dt = t[1] - t[0]
    velocities = np.gradient(positions, dt, axis=0)
    accelerations = np.gradient(velocities, dt, axis=0)
    jerk = np.gradient(accelerations, dt, axis=0)

    return {
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'jerk': jerk,
        'timestamps': t
    }


def test_movement_units_discrimination():
    """Test that NMU discriminates between smooth and jerky trajectories."""
    analyzer = ExecutionQualityAnalyzer()

    smooth_traj = generate_smooth_trajectory()
    jerky_traj = generate_jerky_trajectory()

    smooth_nmu = analyzer.compute_number_of_movement_units(
        smooth_traj['velocities'],
        smooth_traj['timestamps']
    )

    jerky_nmu = analyzer.compute_number_of_movement_units(
        jerky_traj['velocities'],
        jerky_traj['timestamps']
    )

    print(f"\nSmooth trajectory NMU: {smooth_nmu['nmu_per_second']:.2f} units/sec, score: {smooth_nmu['score']:.3f}")
    print(f"Jerky trajectory NMU: {jerky_nmu['nmu_per_second']:.2f} units/sec, score: {jerky_nmu['score']:.3f}")

    # Smooth should have fewer movement units
    assert smooth_nmu['nmu_per_second'] < jerky_nmu['nmu_per_second']

    # Smooth should have higher score
    assert smooth_nmu['score'] > jerky_nmu['score']

    # Scores should be in valid range
    assert 0 <= smooth_nmu['score'] <= 1.0
    assert 0 <= jerky_nmu['score'] <= 1.0


def test_hesitation_discrimination():
    """Test that hesitation rate discriminates between smooth and jerky."""
    analyzer = ExecutionQualityAnalyzer()

    smooth_traj = generate_smooth_trajectory()
    jerky_traj = generate_jerky_trajectory()

    smooth_hes = analyzer.compute_hesitation_rate(
        smooth_traj['velocities'],
        smooth_traj['timestamps']
    )

    jerky_hes = analyzer.compute_hesitation_rate(
        jerky_traj['velocities'],
        jerky_traj['timestamps']
    )

    print(f"\nSmooth trajectory hesitations: {smooth_hes['hesitation_rate']:.2f} /sec, score: {smooth_hes['score']:.3f}")
    print(f"Jerky trajectory hesitations: {jerky_hes['hesitation_rate']:.2f} /sec, score: {jerky_hes['score']:.3f}")

    # Jerky should have more hesitations
    assert jerky_hes['hesitation_rate'] > smooth_hes['hesitation_rate']

    # Smooth should have higher score
    assert smooth_hes['score'] > jerky_hes['score']

    # Scores should be in valid range
    assert 0 <= smooth_hes['score'] <= 1.0
    assert 0 <= jerky_hes['score'] <= 1.0


def test_submovement_discrimination():
    """Test that submovement index discriminates between smooth and jerky."""
    analyzer = ExecutionQualityAnalyzer()

    smooth_traj = generate_smooth_trajectory()
    jerky_traj = generate_jerky_trajectory()

    smooth_sub = analyzer.compute_submovement_index(
        smooth_traj['accelerations'],
        smooth_traj['timestamps']
    )

    jerky_sub = analyzer.compute_submovement_index(
        jerky_traj['accelerations'],
        jerky_traj['timestamps']
    )

    print(f"\nSmooth trajectory submovements: {smooth_sub['submovement_index']:.2f} /sec, score: {smooth_sub['score']:.3f}")
    print(f"Jerky trajectory submovements: {jerky_sub['submovement_index']:.2f} /sec, score: {jerky_sub['score']:.3f}")

    # Jerky should have more submovements
    assert jerky_sub['submovement_index'] > smooth_sub['submovement_index']

    # Smooth should have higher score
    assert smooth_sub['score'] > jerky_sub['score']

    # Scores should be in valid range
    assert 0 <= smooth_sub['score'] <= 1.0
    assert 0 <= jerky_sub['score'] <= 1.0


def test_overall_execution_quality():
    """Test overall execution quality scoring."""
    analyzer = ExecutionQualityAnalyzer()

    smooth_traj = generate_smooth_trajectory()
    jerky_traj = generate_jerky_trajectory()

    smooth_result = analyzer.analyze_episode(smooth_traj)
    jerky_result = analyzer.analyze_episode(jerky_traj)

    print(f"\n=== SMOOTH TRAJECTORY ===")
    print(f"Overall score: {smooth_result['overall_score']:.3f}")
    print(f"Individual scores:")
    for metric, score in smooth_result['individual_scores'].items():
        print(f"  {metric}: {score:.3f}")

    print(f"\n=== JERKY TRAJECTORY ===")
    print(f"Overall score: {jerky_result['overall_score']:.3f}")
    print(f"Individual scores:")
    for metric, score in jerky_result['individual_scores'].items():
        print(f"  {metric}: {score:.3f}")

    # Smooth should have significantly higher overall score
    assert smooth_result['overall_score'] > jerky_result['overall_score']

    # Difference should be meaningful (at least 0.2)
    score_difference = smooth_result['overall_score'] - jerky_result['overall_score']
    assert score_difference > 0.2, f"Score difference too small: {score_difference:.3f}"

    # Scores should be in valid range
    assert 0 <= smooth_result['overall_score'] <= 1.0
    assert 0 <= jerky_result['overall_score'] <= 1.0

    print(f"\n✅ Score difference: {score_difference:.3f} (meaningful discrimination!)")


def test_multiple_episodes():
    """Test analysis of multiple episodes."""
    analyzer = ExecutionQualityAnalyzer()

    # Generate 5 smooth and 5 jerky trajectories
    smooth_episodes = [generate_smooth_trajectory() for _ in range(5)]
    jerky_episodes = [generate_jerky_trajectory() for _ in range(5)]

    smooth_results = analyzer.analyze_multiple_episodes(smooth_episodes, show_progress=False)
    jerky_results = analyzer.analyze_multiple_episodes(jerky_episodes, show_progress=False)

    print(f"\n=== SMOOTH DATASET (5 episodes) ===")
    print(f"Overall score: {smooth_results['overall_score']['mean']:.3f} ± {smooth_results['overall_score']['std']:.3f}")

    print(f"\n=== JERKY DATASET (5 episodes) ===")
    print(f"Overall score: {jerky_results['overall_score']['mean']:.3f} ± {jerky_results['overall_score']['std']:.3f}")

    # Smooth dataset should have higher average score
    assert smooth_results['overall_score']['mean'] > jerky_results['overall_score']['mean']

    # Difference should be meaningful (at least 0.1)
    score_diff = smooth_results['overall_score']['mean'] - jerky_results['overall_score']['mean']
    assert score_diff > 0.1

    print(f"\n✅ Dataset score difference: {score_diff:.3f} (datasets are discriminable!)")


if __name__ == "__main__":
    print("Testing new execution quality metrics...\n")
    print("="*80)

    test_movement_units_discrimination()
    test_hesitation_discrimination()
    test_submovement_discrimination()
    test_overall_execution_quality()
    test_multiple_episodes()

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("\nNew metrics successfully discriminate between good and poor execution!")
