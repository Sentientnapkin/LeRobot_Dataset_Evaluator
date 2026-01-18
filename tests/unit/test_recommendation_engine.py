"""Tests for recommendation engine."""

import pytest
import numpy as np

from src.analysis.recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    Priority
)


class TestRecommendationEngine:
    """Test recommendation engine functionality."""

    @pytest.mark.unit
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = RecommendationEngine()
        assert engine.recommendations == []

    @pytest.mark.unit
    def test_execution_quality_recommendations_low_scores(self):
        """Test recommendations for low execution quality scores."""
        engine = RecommendationEngine()

        motion_results = {
            'movement_units_scores': {'mean': 0.3},
            'hesitation_scores': {'mean': 0.3},
            'submovement_scores': {'mean': 0.3},
            'jerk_scores': {'mean': 0.2}
        }

        recommendations = engine.generate_execution_quality_recommendations(motion_results)

        # Should generate high priority recommendations for all metrics
        assert len(recommendations) >= 3
        high_priority = [r for r in recommendations if r.priority == Priority.HIGH]
        assert len(high_priority) >= 3

    @pytest.mark.unit
    def test_execution_quality_recommendations_medium_scores(self):
        """Test recommendations for medium execution quality scores."""
        engine = RecommendationEngine()

        motion_results = {
            'movement_units_scores': {'mean': 0.55},
            'hesitation_scores': {'mean': 0.55},
            'submovement_scores': {'mean': 0.55},
            'jerk_scores': {'mean': 0.4}
        }

        recommendations = engine.generate_execution_quality_recommendations(motion_results)

        # Should generate medium priority recommendations
        medium_priority = [r for r in recommendations if r.priority == Priority.MEDIUM]
        assert len(medium_priority) >= 2

    @pytest.mark.unit
    def test_execution_quality_recommendations_high_scores(self):
        """Test that high scores don't generate recommendations."""
        engine = RecommendationEngine()

        motion_results = {
            'movement_units_scores': {'mean': 0.9},
            'hesitation_scores': {'mean': 0.9},
            'submovement_scores': {'mean': 0.9},
            'jerk_scores': {'mean': 0.8}
        }

        recommendations = engine.generate_execution_quality_recommendations(motion_results)

        # High scores should not generate recommendations
        assert len(recommendations) == 0

    @pytest.mark.unit
    def test_spatial_coverage_recommendations_limited(self):
        """Test recommendations for limited spatial coverage."""
        engine = RecommendationEngine()

        spatial_results = {
            'overall_score': 0.3,
            'workspace_coverage': {
                'coverage_quality': 'limited'
            },
            'individual_scores': {
                'starting_variance': 0.3,
                'angle_diversity': 0.3
            }
        }

        recommendations = engine.generate_spatial_coverage_recommendations(spatial_results)

        # Should generate high priority for limited coverage
        high_priority = [r for r in recommendations if r.priority == Priority.HIGH]
        assert len(high_priority) >= 1

    @pytest.mark.unit
    def test_spatial_coverage_recommendations_moderate(self):
        """Test recommendations for moderate spatial coverage."""
        engine = RecommendationEngine()

        spatial_results = {
            'overall_score': 0.5,
            'workspace_coverage': {
                'coverage_quality': 'moderate'
            },
            'individual_scores': {
                'starting_variance': 0.5,
                'angle_diversity': 0.5
            }
        }

        recommendations = engine.generate_spatial_coverage_recommendations(spatial_results)

        # Should generate some recommendations
        assert len(recommendations) >= 1

    @pytest.mark.unit
    def test_data_consistency_recommendations_low_score(self):
        """Test recommendations for low data consistency."""
        engine = RecommendationEngine()

        consistency_results = {
            'overall_score': 0.4,
            'anomalies': {
                'anomaly_percentage': 10
            },
            'missing_data': {
                'completeness_percentage': 85
            },
            'individual_scores': {
                'temporal': 0.5
            }
        }

        recommendations = engine.generate_data_consistency_recommendations(consistency_results)

        # Should generate high priority recommendation
        high_priority = [r for r in recommendations if r.priority == Priority.HIGH]
        assert len(high_priority) >= 1

    @pytest.mark.unit
    def test_prioritize_recommendations(self):
        """Test recommendation prioritization."""
        engine = RecommendationEngine()

        recommendations = [
            Recommendation(Priority.MEDIUM, "Cat1", "Metric1", 0.5, "Issue1", "Advice1", "Tech1"),
            Recommendation(Priority.HIGH, "Cat2", "Metric2", 0.3, "Issue2", "Advice2", "Tech2"),
            Recommendation(Priority.LOW, "Cat3", "Metric3", 0.7, "Issue3", "Advice3", "Tech3"),
            Recommendation(Priority.HIGH, "Cat4", "Metric4", 0.2, "Issue4", "Advice4", "Tech4"),
        ]

        prioritized = engine.prioritize_recommendations(recommendations)

        # HIGH priority should come first
        assert prioritized[0].priority == Priority.HIGH
        assert prioritized[1].priority == Priority.HIGH
        # Within same priority, lower score should come first
        assert prioritized[0].score <= prioritized[1].score

    @pytest.mark.unit
    def test_identify_problematic_episodes(self):
        """Test identification of problematic episodes."""
        engine = RecommendationEngine()

        motion_results = {
            'episode_results': [
                {'overall_score': 0.8},
                {'overall_score': 0.2},  # Bad
                {'overall_score': 0.7},
                {'overall_score': 0.1},  # Bad
                {'overall_score': 0.9},
            ]
        }

        consistency_results = {
            'anomalies': {
                'anomaly_episodes': [1, 3]  # Episodes 1 and 3 have anomalies
            }
        }

        problematic = engine.identify_problematic_episodes(
            motion_quality_results=motion_results,
            consistency_results=consistency_results
        )

        # Episodes 1 and 3 should be identified
        assert 'multiple_issues' in problematic
        assert 'single_issue' in problematic
        assert 'issue_details' in problematic

    @pytest.mark.unit
    def test_calculate_overall_health_score(self):
        """Test overall health score calculation."""
        engine = RecommendationEngine()

        motion_results = {'overall_score': {'mean': 0.8}}
        spatial_results = {'overall_score': 0.7}
        consistency_results = {'overall_score': 0.9}

        health_score = engine.calculate_overall_health_score(
            motion_quality_results=motion_results,
            spatial_coverage_results=spatial_results,
            consistency_results=consistency_results
        )

        # Should be weighted average
        assert 0 <= health_score <= 1
        # With all good scores, should be reasonably high
        assert health_score > 0.7

    @pytest.mark.unit
    def test_calculate_health_score_partial_data(self):
        """Test health score with partial data."""
        engine = RecommendationEngine()

        # Only motion quality results
        motion_results = {'overall_score': {'mean': 0.6}}

        health_score = engine.calculate_overall_health_score(
            motion_quality_results=motion_results
        )

        # Should still calculate something
        assert 0 <= health_score <= 1

    @pytest.mark.unit
    def test_calculate_health_score_no_data(self):
        """Test health score with no data."""
        engine = RecommendationEngine()

        health_score = engine.calculate_overall_health_score()

        # Should return 0 for no data
        assert health_score == 0.0

    @pytest.mark.unit
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        engine = RecommendationEngine()

        motion_results = {
            'overall_score': {'mean': 0.5},
            'movement_units_scores': {'mean': 0.4},
            'hesitation_scores': {'mean': 0.4},
            'submovement_scores': {'mean': 0.4},
            'jerk_scores': {'mean': 0.3},
            'episode_results': [
                {'overall_score': 0.5} for _ in range(10)
            ]
        }

        spatial_results = {
            'overall_score': 0.5,
            'workspace_coverage': {'coverage_quality': 'moderate'},
            'individual_scores': {
                'starting_variance': 0.4,
                'angle_diversity': 0.4
            }
        }

        consistency_results = {
            'overall_score': 0.6,
            'anomalies': {
                'anomaly_percentage': 3,
                'anomaly_episodes': []
            },
            'missing_data': {'completeness_percentage': 97},
            'individual_scores': {'temporal': 0.7}
        }

        report = engine.generate_comprehensive_report(
            motion_quality_results=motion_results,
            spatial_coverage_results=spatial_results,
            consistency_results=consistency_results
        )

        # Verify report structure
        assert 'health_score' in report
        assert 'recommendations' in report
        assert 'priority_counts' in report
        assert 'by_category' in report
        assert 'problematic_episodes' in report
        assert 'total_recommendations' in report

        # Health score should be valid
        assert 0 <= report['health_score'] <= 1

    @pytest.mark.unit
    def test_recommendation_structure(self):
        """Test that recommendations have correct structure."""
        rec = Recommendation(
            priority=Priority.HIGH,
            category="Test Category",
            metric="Test Metric",
            score=0.5,
            issue="Test issue",
            advice="Test advice",
            technique="Test technique"
        )

        assert rec.priority == Priority.HIGH
        assert rec.category == "Test Category"
        assert rec.metric == "Test Metric"
        assert rec.score == 0.5
        assert rec.issue == "Test issue"
        assert rec.advice == "Test advice"
        assert rec.technique == "Test technique"

    @pytest.mark.unit
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        engine = RecommendationEngine()

        # Empty results should return empty list
        rec1 = engine.generate_execution_quality_recommendations({})
        rec2 = engine.generate_spatial_coverage_recommendations({})
        rec3 = engine.generate_data_consistency_recommendations({})

        assert rec1 == []
        assert rec2 == []
        assert rec3 == []

    @pytest.mark.unit
    def test_none_results_handling(self):
        """Test handling of None results."""
        engine = RecommendationEngine()

        # None results should return empty list
        rec1 = engine.generate_execution_quality_recommendations(None)
        rec2 = engine.generate_spatial_coverage_recommendations(None)
        rec3 = engine.generate_data_consistency_recommendations(None)

        assert rec1 == []
        assert rec2 == []
        assert rec3 == []
