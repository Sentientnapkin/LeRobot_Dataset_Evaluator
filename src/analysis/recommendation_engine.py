"""Recommendation Engine for generating actionable dataset improvement suggestions.

This module analyzes computed metrics and generates prioritized recommendations
to improve dataset quality for ACT model training.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Recommendation priority levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Recommendation:
    """A single recommendation with priority and details."""
    priority: Priority
    category: str
    metric: str
    score: float
    issue: str
    advice: str
    technique: str


class RecommendationEngine:
    """Generates actionable recommendations based on computed metrics."""

    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommendations: List[Recommendation] = []

    def generate_execution_quality_recommendations(
        self,
        motion_quality_results: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations from execution quality metrics.

        Args:
            motion_quality_results: Results from ExecutionQualityAnalyzer

        Returns:
            List of recommendations
        """
        recommendations = []

        if not motion_quality_results:
            return recommendations

        # Movement Units
        movement_units_mean = motion_quality_results['movement_units_scores']['mean']
        if movement_units_mean < 0.5:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Execution Quality",
                metric="Movement Units",
                score=movement_units_mean,
                issue="Too many small adjustments instead of purposeful movements",
                advice="Make fewer, larger purposeful movements instead of many small adjustments",
                technique="Plan your motion path before executing. Think in terms of complete trajectories rather than incremental corrections."
            ))
        elif movement_units_mean < 0.7:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Execution Quality",
                metric="Movement Units",
                score=movement_units_mean,
                issue="Some room for improvement in movement purposefulness",
                advice="Continue working on more deliberate, planned movements",
                technique="Before each movement, visualize the complete path. Reduce the number of corrective adjustments."
            ))

        # Hesitations
        hesitation_mean = motion_quality_results['hesitation_scores']['mean']
        if hesitation_mean < 0.5:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Execution Quality",
                metric="Hesitations",
                score=hesitation_mean,
                issue="Excessive velocity reversals indicating uncertainty",
                advice="Work on reducing velocity reversals and corrections during movements",
                technique="Build confidence in your control. Practice consistent velocity profiles. Avoid back-and-forth corrections."
            ))
        elif hesitation_mean < 0.7:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Execution Quality",
                metric="Hesitations",
                score=hesitation_mean,
                issue="Some hesitation in control execution",
                advice="Continue improving control confidence",
                technique="Focus on smooth acceleration and deceleration. Minimize velocity reversals."
            ))

        # Submovements
        submovement_mean = motion_quality_results['submovement_scores']['mean']
        if submovement_mean < 0.5:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Execution Quality",
                metric="Submovements",
                score=submovement_mean,
                issue="High number of corrective submovements",
                advice="Reduce corrective submovements by improving control confidence",
                technique="Practice moving in single, smooth strokes rather than multiple corrective phases."
            ))
        elif submovement_mean < 0.7:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Execution Quality",
                metric="Submovements",
                score=submovement_mean,
                issue="Some corrective movements present",
                advice="Work on smoother single-phase movements",
                technique="Reduce the number of acceleration/deceleration phases within each movement."
            ))

        # Jerk
        jerk_mean = motion_quality_results['jerk_scores']['mean']
        if jerk_mean < 0.3:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Execution Quality",
                metric="Jerk",
                score=jerk_mean,
                issue="Excessive jerkiness in movements",
                advice="Focus on smoother, more continuous motions",
                technique="Avoid abrupt changes in acceleration. Practice fluid, continuous movements."
            ))
        elif jerk_mean < 0.5:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Execution Quality",
                metric="Jerk",
                score=jerk_mean,
                issue="Movement smoothness could be improved",
                advice="Work on reducing jerkiness",
                technique="Smooth out acceleration changes. Avoid sudden starts and stops."
            ))

        return recommendations

    def generate_spatial_coverage_recommendations(
        self,
        spatial_coverage_results: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations from spatial coverage metrics.

        Args:
            spatial_coverage_results: Results from SpatialCoverageAnalyzer

        Returns:
            List of recommendations
        """
        recommendations = []

        if not spatial_coverage_results:
            return recommendations

        # Coverage Quality
        coverage = spatial_coverage_results.get('workspace_coverage', {})
        coverage_quality = coverage.get('coverage_quality', '').lower()

        if coverage_quality == 'limited':
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Spatial Coverage",
                metric="Coverage Quality",
                score=spatial_coverage_results.get('overall_score', 0),
                issue="Limited workspace exploration",
                advice="Expand data collection to cover more of the robot's reachable workspace",
                technique="Deliberately collect data from under-represented regions. Vary starting positions and target locations."
            ))
        elif coverage_quality == 'moderate':
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Spatial Coverage",
                metric="Coverage Quality",
                score=spatial_coverage_results.get('overall_score', 0),
                issue="Focused on specific regions only",
                advice="Consider expanding to more diverse workspace regions",
                technique="Identify unexplored areas using the 3D coverage visualization and collect additional data there."
            ))

        # Starting Variance
        individual_scores = spatial_coverage_results.get('individual_scores', {})
        starting_variance = individual_scores.get('starting_variance', 1.0)
        if starting_variance < 0.4:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Spatial Coverage",
                metric="Starting Variance",
                score=starting_variance,
                issue="Insufficient diversity in starting positions",
                advice="Vary robot starting positions more during data collection",
                technique="Use different initial arm configurations for each episode. Avoid always starting from the same pose."
            ))

        # Angle Diversity
        angle_diversity = individual_scores.get('angle_diversity', 1.0)
        if angle_diversity < 0.4:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Spatial Coverage",
                metric="Angle Diversity",
                score=angle_diversity,
                issue="Limited approach angles",
                advice="Approach targets from different angles",
                technique="Deliberately vary the direction from which you approach objects. Include left, right, front, and angled approaches."
            ))

        return recommendations

    def generate_data_consistency_recommendations(
        self,
        consistency_results: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate recommendations from data consistency metrics.

        Args:
            consistency_results: Results from DataConsistencyAnalyzer

        Returns:
            List of recommendations
        """
        recommendations = []

        if not consistency_results:
            return recommendations

        # Overall Score
        overall_score = consistency_results.get('overall_score', 1.0)
        if overall_score < 0.6:
            recommendations.append(Recommendation(
                priority=Priority.HIGH,
                category="Data Consistency",
                metric="Overall Consistency",
                score=overall_score,
                issue="Significant data quality issues detected",
                advice="Review and address data collection pipeline issues",
                technique="Check sensor synchronization, data recording frequency, and hardware reliability."
            ))

        # Anomalies
        anomalies = consistency_results.get('anomalies', {})
        anomaly_pct = anomalies.get('anomaly_percentage', 0)
        if anomaly_pct > 5:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Data Consistency",
                metric="Anomaly Rate",
                score=1 - (anomaly_pct / 100),
                issue=f"Detected {anomaly_pct:.1f}% anomalous data points",
                advice="Investigate and potentially remove anomalous episodes",
                technique="Review episodes flagged as anomalous. Check for sensor glitches, communication drops, or recording errors."
            ))

        # Completeness
        missing_data = consistency_results.get('missing_data', {})
        completeness = missing_data.get('completeness_percentage', 100)
        if completeness < 95:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Data Consistency",
                metric="Completeness",
                score=completeness / 100,
                issue=f"Data is only {completeness:.1f}% complete",
                advice="Ensure all data fields are properly recorded",
                technique="Check data pipeline for dropped frames or missing values. Verify all sensors are recording consistently."
            ))

        # Temporal consistency
        individual_scores = consistency_results.get('individual_scores', {})
        temporal_score = individual_scores.get('temporal', 1.0)
        if temporal_score < 0.7:
            recommendations.append(Recommendation(
                priority=Priority.MEDIUM,
                category="Data Consistency",
                metric="Temporal Consistency",
                score=temporal_score,
                issue="Irregular timing in data recording",
                advice="Improve data recording timing consistency",
                technique="Ensure consistent frame rate during recording. Check for system load issues that might cause timing jitter."
            ))

        return recommendations

    def prioritize_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Sort recommendations by priority and score.

        Args:
            recommendations: List of recommendations to sort

        Returns:
            Sorted list of recommendations
        """
        priority_order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        return sorted(
            recommendations,
            key=lambda x: (priority_order[x.priority], x.score)
        )

    def identify_problematic_episodes(
        self,
        motion_quality_results: Optional[Dict[str, Any]] = None,
        spatial_coverage_results: Optional[Dict[str, Any]] = None,
        consistency_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[int]]:
        """Identify episodes that appear in multiple worst lists.

        Args:
            motion_quality_results: Results from ExecutionQualityAnalyzer
            spatial_coverage_results: Results from SpatialCoverageAnalyzer
            consistency_results: Results from DataConsistencyAnalyzer

        Returns:
            Dict with episode indices organized by issue count
        """
        episode_issues: Dict[int, List[str]] = {}

        # Collect worst episodes from execution quality
        if motion_quality_results and 'episode_results' in motion_quality_results:
            episode_results = motion_quality_results['episode_results']
            scores = [(i, ep['overall_score']) for i, ep in enumerate(episode_results)]
            scores.sort(key=lambda x: x[1])

            # Bottom 10% are problematic
            n_worst = max(1, len(scores) // 10)
            for i, score in scores[:n_worst]:
                if i not in episode_issues:
                    episode_issues[i] = []
                episode_issues[i].append("Low execution quality")

        # Collect problematic episodes from consistency
        if consistency_results and 'anomalies' in consistency_results:
            anomaly_episodes = consistency_results['anomalies'].get('anomaly_episodes', [])
            for ep_idx in anomaly_episodes:
                if ep_idx not in episode_issues:
                    episode_issues[ep_idx] = []
                episode_issues[ep_idx].append("Contains anomalies")

        # Organize by number of issues
        result = {
            'multiple_issues': [],
            'single_issue': [],
            'issue_details': episode_issues
        }

        for ep_idx, issues in episode_issues.items():
            if len(issues) > 1:
                result['multiple_issues'].append(ep_idx)
            else:
                result['single_issue'].append(ep_idx)

        result['multiple_issues'].sort()
        result['single_issue'].sort()

        return result

    def calculate_overall_health_score(
        self,
        motion_quality_results: Optional[Dict[str, Any]] = None,
        spatial_coverage_results: Optional[Dict[str, Any]] = None,
        consistency_results: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate overall dataset health score.

        Args:
            motion_quality_results: Results from ExecutionQualityAnalyzer
            spatial_coverage_results: Results from SpatialCoverageAnalyzer
            consistency_results: Results from DataConsistencyAnalyzer

        Returns:
            Overall health score between 0 and 1
        """
        scores = []
        weights = []

        if motion_quality_results:
            scores.append(motion_quality_results['overall_score']['mean'])
            weights.append(0.4)

        if spatial_coverage_results:
            scores.append(spatial_coverage_results['overall_score'])
            weights.append(0.3)

        if consistency_results:
            scores.append(consistency_results['overall_score'])
            weights.append(0.3)

        if not scores:
            return 0.0

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        return sum(s * w for s, w in zip(scores, weights))

    def generate_comprehensive_report(
        self,
        motion_quality_results: Optional[Dict[str, Any]] = None,
        spatial_coverage_results: Optional[Dict[str, Any]] = None,
        consistency_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive recommendation report.

        Args:
            motion_quality_results: Results from ExecutionQualityAnalyzer
            spatial_coverage_results: Results from SpatialCoverageAnalyzer
            consistency_results: Results from DataConsistencyAnalyzer

        Returns:
            Comprehensive report with all recommendations and analysis
        """
        all_recommendations = []

        # Collect recommendations from all sources
        if motion_quality_results:
            all_recommendations.extend(
                self.generate_execution_quality_recommendations(motion_quality_results)
            )

        if spatial_coverage_results:
            all_recommendations.extend(
                self.generate_spatial_coverage_recommendations(spatial_coverage_results)
            )

        if consistency_results:
            all_recommendations.extend(
                self.generate_data_consistency_recommendations(consistency_results)
            )

        # Prioritize
        prioritized = self.prioritize_recommendations(all_recommendations)

        # Count by priority
        priority_counts = {
            Priority.HIGH: len([r for r in prioritized if r.priority == Priority.HIGH]),
            Priority.MEDIUM: len([r for r in prioritized if r.priority == Priority.MEDIUM]),
            Priority.LOW: len([r for r in prioritized if r.priority == Priority.LOW])
        }

        # Group by category
        by_category = {}
        for rec in prioritized:
            if rec.category not in by_category:
                by_category[rec.category] = []
            by_category[rec.category].append(rec)

        # Identify problematic episodes
        problematic_episodes = self.identify_problematic_episodes(
            motion_quality_results,
            spatial_coverage_results,
            consistency_results
        )

        # Calculate overall health
        health_score = self.calculate_overall_health_score(
            motion_quality_results,
            spatial_coverage_results,
            consistency_results
        )

        return {
            'health_score': health_score,
            'recommendations': prioritized,
            'priority_counts': priority_counts,
            'by_category': by_category,
            'problematic_episodes': problematic_episodes,
            'total_recommendations': len(prioritized)
        }
