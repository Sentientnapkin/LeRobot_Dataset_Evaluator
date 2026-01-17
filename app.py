"""LeRobot Dataset Evaluator - Main Streamlit Application."""

import streamlit as st
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import get_config
from src.utils.logging_config import setup_logging, get_logger
from src.ui import get_full_css, render_page_header, render_section_header, render_alert

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="LeRobot Dataset Evaluator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply centralized CSS
st.markdown(get_full_css(), unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False

    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None

    if 'metadata' not in st.session_state:
        st.session_state.metadata = None

    if 'loader' not in st.session_state:
        st.session_state.loader = None

    if 'episodes_data' not in st.session_state:
        st.session_state.episodes_data = None

    if 'episodes_features' not in st.session_state:
        st.session_state.episodes_features = None

    if 'motion_quality_results' not in st.session_state:
        st.session_state.motion_quality_results = None

    if 'spatial_coverage_results' not in st.session_state:
        st.session_state.spatial_coverage_results = None

    if 'consistency_results' not in st.session_state:
        st.session_state.consistency_results = None

    if 'success_indicator_results' not in st.session_state:
        st.session_state.success_indicator_results = None

    if 'metrics_computed' not in st.session_state:
        st.session_state.metrics_computed = False

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Overview"


def render_sidebar():
    """Render sidebar with dataset selection and configuration."""
    st.sidebar.title("Configuration")

    # Load config
    config = get_config()

    # Dataset selection
    st.sidebar.header("Dataset Selection")

    # HuggingFace account
    hf_account = st.sidebar.text_input(
        "HuggingFace Account",
        value=config.hf_account,
        help="Your HuggingFace account name"
    )

    # Dataset name
    dataset_name = st.sidebar.text_input(
        "Dataset Name",
        value="",
        help="Name of the dataset on HuggingFace Hub"
    )

    # Full dataset ID
    if dataset_name:
        dataset_id = f"{hf_account}/{dataset_name}"
        st.sidebar.info(f"Full ID: `{dataset_id}`")
    else:
        dataset_id = None

    # Robot configuration
    st.sidebar.header("Robot Configuration")

    robot_type = st.sidebar.selectbox(
        "Robot Type",
        options=['generic_6dof', 'ur5', 'panda', 'so-arm100'],
        index=0,
        help="Select your robot type for accurate metrics"
    )

    # Analysis options
    st.sidebar.header("Analysis Options")

    max_episodes = st.sidebar.number_input(
        "Max Episodes to Analyze",
        min_value=1,
        max_value=10000,
        value=100,
        help="Number of episodes to analyze (lower = faster)"
    )

    sampling_strategy = st.sidebar.selectbox(
        "Sampling Strategy",
        options=['auto', 'all', 'random', 'sequential'],
        index=0,
        help="""How to select episodes for analysis:
        - Auto (Recommended): All episodes if <500, random sample if >500
        - All: Analyze all episodes (most accurate, slower for large datasets)
        - Random: Random sample of max episodes (unbiased)
        - Sequential: Episodes 0 to max-1 (may have temporal bias)"""
    )

    use_streaming = st.sidebar.checkbox(
        "Use Streaming Mode",
        value=True,
        help="Stream data without full download (recommended for large datasets)"
    )

    # Load button
    load_button = st.sidebar.button(
        "Load Dataset",
        type="primary",
        use_container_width=True,
        disabled=(dataset_id is None)
    )

    return {
        'dataset_id': dataset_id,
        'robot_type': robot_type,
        'max_episodes': max_episodes,
        'sampling_strategy': sampling_strategy,
        'use_streaming': use_streaming,
        'load_button': load_button
    }


def render_home_page():
    """Render home page with modern app-like design."""
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            LeRobot Dataset Evaluator
        </h1>
        <p style="font-size: 1.25rem; color: #94A3B8; max-width: 600px; margin: 0 auto;">
            Analyze your robotics datasets before training. Catch issues early, save compute time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # If dataset is loaded, show stats prominently
    if st.session_state.dataset_loaded and st.session_state.metadata:
        metadata = st.session_state.metadata

        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E3A5F 0%, #1E293B 100%); border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid #334155;">
            <h3 style="color: #F8FAFC; margin: 0 0 1rem 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">Currently Loaded</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Dataset", st.session_state.dataset_id.split('/')[-1] if st.session_state.dataset_id else "N/A")
        with col2:
            st.metric("Episodes", metadata.get('total_episodes', 'N/A'))
        with col3:
            st.metric("Frames", f"{metadata.get('total_frames', 0):,}")
        with col4:
            st.metric("FPS", metadata.get('fps', 'N/A'))
        with col5:
            avg_length = metadata.get('total_frames', 0) / max(metadata.get('total_episodes', 1), 1)
            st.metric("Avg Length", f"{avg_length:.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Loader Section (inline, not sidebar)
    if not st.session_state.dataset_loaded:
        st.markdown("""
        <div style="background: #1E293B; border-radius: 12px; padding: 2rem; margin-bottom: 2rem; border: 1px solid #334155;">
            <h3 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">Load a Dataset</h3>
            <p style="color: #94A3B8; margin: 0 0 1.5rem 0;">Enter your HuggingFace dataset details to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                hf_account = st.text_input(
                    "HuggingFace Account",
                    value=get_config().hf_account,
                    placeholder="your-username"
                )
            with subcol2:
                dataset_name = st.text_input(
                    "Dataset Name",
                    value="",
                    placeholder="dataset-name"
                )

            if hf_account and dataset_name:
                st.caption(f"Full ID: `{hf_account}/{dataset_name}`")

        with col2:
            robot_type = st.selectbox(
                "Robot Type",
                options=['generic_6dof', 'ur5', 'panda', 'so-arm100'],
                index=0
            )

        # Advanced options in expander
        with st.expander("Advanced Options"):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            with adv_col1:
                max_episodes = st.number_input(
                    "Max Episodes",
                    min_value=1,
                    max_value=10000,
                    value=100
                )
            with adv_col2:
                sampling_strategy = st.selectbox(
                    "Sampling",
                    options=['auto', 'all', 'random', 'sequential'],
                    index=0
                )
            with adv_col3:
                use_streaming = st.checkbox("Streaming Mode", value=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Load button
        dataset_id = f"{hf_account}/{dataset_name}" if hf_account and dataset_name else None
        if st.button("Load Dataset", type="primary", use_container_width=True, disabled=(dataset_id is None)):
            if dataset_id:
                load_dataset(dataset_id, max_episodes, sampling_strategy, use_streaming)
                st.rerun()

    # Feature Cards
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background: #1E293B; border-radius: 8px; padding: 1.5rem; height: 100%; border: 1px solid #334155;">
            <div style="color: #3B82F6; font-size: 1.5rem; margin-bottom: 0.75rem;">01</div>
            <h4 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">Execution Quality</h4>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Teleoperation-calibrated metrics for movement smoothness, hesitation, and control quality.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #1E293B; border-radius: 8px; padding: 1.5rem; height: 100%; border: 1px solid #334155;">
            <div style="color: #059669; font-size: 1.5rem; margin-bottom: 0.75rem;">02</div>
            <h4 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">Spatial Coverage</h4>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                3D workspace analysis, position diversity, and exploration uniformity metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: #1E293B; border-radius: 8px; padding: 1.5rem; height: 100%; border: 1px solid #334155;">
            <div style="color: #D97706; font-size: 1.5rem; margin-bottom: 0.75rem;">03</div>
            <h4 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">Data Consistency</h4>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Anomaly detection, temporal alignment, and data completeness checks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: #1E293B; border-radius: 8px; padding: 1.5rem; height: 100%; border: 1px solid #334155;">
            <div style="color: #DC2626; font-size: 1.5rem; margin-bottom: 0.75rem;">04</div>
            <h4 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">Recommendations</h4>
            <p style="color: #94A3B8; font-size: 0.875rem; margin: 0;">
                Actionable insights prioritized by impact to improve your dataset quality.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 2rem 0;">'
        'Built by Sebastian Vargas'
        '</div>',
        unsafe_allow_html=True
    )


def load_dataset(dataset_id: str, max_episodes: int, sampling_strategy: str, use_streaming: bool):
    """Load dataset and update session state."""
    try:
        from src.data.loader import DatasetLoader

        with st.spinner(f"Loading dataset {dataset_id}..."):
            # Create loader
            loader = DatasetLoader(
                dataset_id=dataset_id,
                streaming=use_streaming
            )

            # Load metadata
            metadata = loader.load_metadata()

            # Load dataset
            loader.load_dataset(max_episodes=max_episodes)

            # Update session state
            st.session_state.dataset_loaded = True
            st.session_state.dataset_id = dataset_id
            st.session_state.metadata = metadata
            st.session_state.loader = loader
            st.session_state.max_episodes_choice = max_episodes
            st.session_state.sampling_strategy = sampling_strategy

        # Now immediately load and preprocess episodes
        load_and_preprocess_episodes()

        # Display smart message based on sampling strategy
        if st.session_state.data_preprocessing_complete:
            n_episodes = len(st.session_state.episodes_data)
            total_eps = metadata['total_episodes']

            if sampling_strategy == 'auto':
                if total_eps < 500:
                    msg = f"Dataset loaded and preprocessed. Analyzed all {n_episodes} episodes (auto mode)."
                else:
                    msg = f"Dataset loaded and preprocessed. Analyzed {n_episodes} of {total_eps} episodes (auto mode)."
            elif sampling_strategy == 'all':
                msg = f"Dataset loaded and preprocessed. Analyzed all {n_episodes} episodes."
            elif sampling_strategy == 'random':
                msg = f"Dataset loaded and preprocessed. Analyzed {n_episodes} randomly sampled episodes."
            else:
                msg = f"Dataset loaded and preprocessed. Analyzed episodes 0-{n_episodes-1}."

            st.success(msg)
            logger.info(f"Loaded and preprocessed dataset: {dataset_id} ({n_episodes} episodes)")
        else:
            st.warning("Dataset loaded but preprocessing failed. Check error messages above.")
            logger.warning(f"Dataset loaded but preprocessing failed: {dataset_id}")

    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")
        logger.error(f"Dataset loading error: {e}", exc_info=True)


def load_and_preprocess_episodes():
    """Load and preprocess episodes immediately after dataset load."""
    try:
        from src.data.loader import DatasetLoader
        from src.data.preprocessor import DataPreprocessor
        from src.utils.robot_models import get_robot_config
        from src.utils.config import get_config

        loader = st.session_state.loader
        config = get_config()

        # Get robot config
        robot_config = get_robot_config(config.robot_type)

        # Load episodes using user's choices from sidebar
        max_episodes_choice = st.session_state.get('max_episodes_choice', 100)
        sampling_strategy = st.session_state.get('sampling_strategy', 'auto')

        with st.spinner("Preprocessing episodes..."):
            # Load episodes
            episodes = loader.get_all_episodes_data(
                max_episodes=max_episodes_choice,
                include_images=False,
                show_progress=True,
                sampling_strategy=sampling_strategy
            )

            # Preprocess episodes to extract features
            preprocessor = DataPreprocessor(robot_config)
            episodes_features = []

            for episode in episodes:
                features = preprocessor.extract_features(episode, smooth_sigma=1.0)
                episodes_features.append(features)

            # Store in session state for all pages to access
            st.session_state.episodes_features = episodes_features
            st.session_state.episodes_data = episodes
            st.session_state.data_preprocessing_complete = True

            logger.info(f"Preprocessed {len(episodes)} episodes successfully")

    except Exception as e:
        st.error(f"Failed to preprocess episodes: {str(e)}")
        logger.error(f"Episode preprocessing error: {e}", exc_info=True)
        st.session_state.data_preprocessing_complete = False


def render_motion_quality_page():
    """Render execution quality analysis page."""
    render_page_header(
        "Execution Quality Analysis",
        "Teleoperation quality metrics calibrated for real-world robotics data"
    )

    st.markdown("---")

    # Check if dataset is loaded and preprocessed
    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    # Compute execution quality metrics if not already done
    if st.session_state.motion_quality_results is None:
        with st.spinner("Computing execution quality metrics..."):
            try:
                from src.metrics.execution_quality import ExecutionQualityAnalyzer

                analyzer = ExecutionQualityAnalyzer()

                results = analyzer.analyze_multiple_episodes(
                    st.session_state.episodes_features,
                    show_progress=False
                )

                st.session_state.motion_quality_results = results

            except Exception as e:
                st.error(f"Failed to compute metrics: {str(e)}")
                logger.error(f"Metric computation error: {e}", exc_info=True)
                return

    # Display results
    results = st.session_state.motion_quality_results

    if not results:
        st.error("No results available.")
        return

    # Section 1: Quality Score Hero
    render_section_header("Overall Execution Quality")

    col1, col2 = st.columns([1, 2])

    overall_mean = results['overall_score']['mean']

    with col1:
        # Large score display with gauge
        try:
            from src.visualizations.execution_quality_viz import plot_quality_gauge
            fig_gauge = plot_quality_gauge(overall_mean)
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.metric("Overall Quality Score", f"{overall_mean:.3f}")
            logger.error(f"Gauge visualization error: {e}", exc_info=True)

    with col2:
        # Quality status and interpretation
        if overall_mean > 0.65:
            render_alert(
                title="EXCELLENT - Ready for ACT training",
                message="""Your dataset shows high-quality teleoperation with smooth, purposeful movements,
                confident control with minimal hesitations, low corrective submovements, and appropriate
                jerk levels for the task. This dataset is well-suited for training an ACT model.""",
                alert_type="success"
            )
        elif overall_mean > 0.45:
            render_alert(
                title="ACCEPTABLE - Can proceed with training",
                message="""Your dataset has acceptable quality and can be used for ACT training,
                though there may be room for improvement. Some metrics may benefit from refinement.
                Consider reviewing recommendations below. Training should still produce reasonable results.""",
                alert_type="info"
            )
        else:
            render_alert(
                title="NEEDS IMPROVEMENT - Review recommendations",
                message="""Your dataset quality needs attention before optimal ACT training.
                Multiple metrics show room for improvement. Review detailed recommendations below.
                Consider re-collecting some episodes with improved technique.""",
                alert_type="warning"
            )

        # Show min/max as smaller metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Min Score", f"{results['overall_score']['min']:.3f}")
        with col_b:
            st.metric("Max Score", f"{results['overall_score']['max']:.3f}")

    st.markdown("---")

    # Metric breakdown
    render_section_header("Metric Breakdown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        movement_units_mean = results['movement_units_scores']['mean']
        st.metric("Movement Units", f"{movement_units_mean:.3f}",
                 help="Measures purposeful movement segments (higher = more skilled)")

    with col2:
        hesitation_mean = results['hesitation_scores']['mean']
        st.metric("Hesitations", f"{hesitation_mean:.3f}",
                 help="Velocity reversals per second (higher = more confident)")

    with col3:
        submovement_mean = results['submovement_scores']['mean']
        st.metric("Submovements", f"{submovement_mean:.3f}",
                 help="Acceleration smoothness (higher = fewer corrections)")

    with col4:
        jerk_mean = results['jerk_scores']['mean']
        st.metric("Jerk (Recalibrated)", f"{jerk_mean:.3f}",
                 help="Motion smoothness calibrated for teleoperation")

    st.markdown("---")

    # Section 3: Visualization Tabs
    render_section_header("Detailed Analysis")

    try:
        from src.visualizations.execution_quality_viz import (
            plot_quality_trends,
            plot_episode_heatmap,
            plot_metric_comparison_bars,
            plot_metric_distributions,
            plot_movement_units_analysis,
            plot_hesitation_analysis,
            plot_submovement_patterns,
            plot_jerk_distribution_with_outliers
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Metric Details",
            "Movement Quality",
            "Control Quality",
            "Jerk and Smoothness"
        ])

        with tab1:
            st.markdown("#### Quality Trends Across Episodes")
            fig1 = plot_quality_trends(results['episode_results'])
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### Episode Quality Heatmap")
            fig2 = plot_episode_heatmap(results['episode_results'])
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("#### Metric Comparison")
            fig1 = plot_metric_comparison_bars(results)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### Metric Distributions")
            fig2 = plot_metric_distributions(results)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.markdown("#### Movement Units Analysis")
            st.caption("Shows correlation between purposeful movement segments and overall quality")
            fig1 = plot_movement_units_analysis(results['episode_results'])
            st.plotly_chart(fig1, use_container_width=True)

        with tab4:
            st.markdown("#### Hesitation Analysis")
            st.caption("Lower hesitation rates indicate more confident control")
            fig1 = plot_hesitation_analysis(results['episode_results'])
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("#### Submovement Patterns")
            st.caption("Lower submovement index indicates smoother, more direct control")
            fig2 = plot_submovement_patterns(results['episode_results'])
            st.plotly_chart(fig2, use_container_width=True)

        with tab5:
            st.markdown("#### Jerk Distribution with Outlier Detection")
            st.caption("Shows jerk score distribution and identifies episodes with excessive jerkiness")
            fig1 = plot_jerk_distribution_with_outliers(results['episode_results'])
            st.plotly_chart(fig1, use_container_width=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logger.error(f"Visualization error: {e}", exc_info=True)

    st.markdown("---")

    # Section 4: Enhanced Recommendations
    render_section_header("Actionable Recommendations")

    # Build prioritized recommendations based on scores
    recommendations = []

    if movement_units_mean < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'metric': 'Movement Units',
            'score': movement_units_mean,
            'issue': 'Too many small adjustments instead of purposeful movements',
            'advice': 'Make fewer, larger purposeful movements instead of many small adjustments',
            'technique': 'Plan your motion path before executing. Think in terms of complete trajectories rather than incremental corrections.'
        })
    elif movement_units_mean < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'metric': 'Movement Units',
            'score': movement_units_mean,
            'issue': 'Some room for improvement in movement purposefulness',
            'advice': 'Continue working on more deliberate, planned movements',
            'technique': 'Before each movement, visualize the complete path. Reduce the number of corrective adjustments.'
        })

    if hesitation_mean < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'metric': 'Hesitations',
            'score': hesitation_mean,
            'issue': 'Excessive velocity reversals indicating uncertainty',
            'advice': 'Work on reducing velocity reversals and corrections during movements',
            'technique': 'Build confidence in your control. Practice consistent velocity profiles. Avoid back-and-forth corrections.'
        })
    elif hesitation_mean < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'metric': 'Hesitations',
            'score': hesitation_mean,
            'issue': 'Some hesitation in control execution',
            'advice': 'Continue improving control confidence',
            'technique': 'Focus on smooth acceleration and deceleration. Minimize velocity reversals.'
        })

    if submovement_mean < 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'metric': 'Submovements',
            'score': submovement_mean,
            'issue': 'High number of corrective submovements',
            'advice': 'Reduce corrective submovements by improving control confidence',
            'technique': 'Practice moving in single, smooth strokes rather than multiple corrective phases.'
        })
    elif submovement_mean < 0.7:
        recommendations.append({
            'priority': 'MEDIUM',
            'metric': 'Submovements',
            'score': submovement_mean,
            'issue': 'Some corrective movements present',
            'advice': 'Work on smoother single-phase movements',
            'technique': 'Reduce the number of acceleration/deceleration phases within each movement.'
        })

    if jerk_mean < 0.3:
        recommendations.append({
            'priority': 'HIGH',
            'metric': 'Jerk',
            'score': jerk_mean,
            'issue': 'Excessive jerkiness in movements',
            'advice': 'Focus on smoother, more continuous motions',
            'technique': 'Avoid abrupt changes in acceleration. Practice fluid, continuous movements.'
        })
    elif jerk_mean < 0.5:
        recommendations.append({
            'priority': 'MEDIUM',
            'metric': 'Jerk',
            'score': jerk_mean,
            'issue': 'Movement smoothness could be improved',
            'advice': 'Work on reducing jerkiness',
            'technique': 'Smooth out acceleration changes. Avoid sudden starts and stops.'
        })

    # Display recommendations
    if recommendations:
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations_sorted = sorted(
            recommendations,
            key=lambda x: (priority_order[x['priority']], x['score'])
        )

        for rec in recommendations_sorted:
            priority_colors = {'HIGH': '#DC2626', 'MEDIUM': '#D97706', 'LOW': '#059669'}
            color = priority_colors.get(rec['priority'], '#64748B')

            with st.expander(f"**{rec['metric']}** - Score: {rec['score']:.3f} ({rec['priority']} Priority)"):
                st.markdown(f"""
                <div style="border-left: 3px solid {color}; padding-left: 1rem;">
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>What to do:</strong> {rec['advice']}</p>
                    <p><strong>Technique:</strong> {rec['technique']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        render_alert(
            title="Excellent work!",
            message="All metrics are in good range. Your teleoperation technique shows strong quality across all dimensions. This dataset is well-suited for ACT training.",
            alert_type="success"
        )

    # Section 5: Episode Browser
    st.markdown("---")
    render_section_header("Episode Browser")

    with st.expander("View All Episodes with Sorting", expanded=False):
        import pandas as pd

        episode_data = []
        episode_results = results['episode_results']

        for i, ep in enumerate(episode_results):
            episode_data.append({
                'Episode': i,
                'Overall': ep['overall_score'],
                'Movement Units': ep['individual_scores']['movement_units'],
                'Hesitations': ep['individual_scores']['hesitations'],
                'Submovements': ep['individual_scores']['submovements'],
                'Jerk': ep['individual_scores']['jerk']
            })

        df = pd.DataFrame(episode_data)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=['Episode', 'Overall', 'Movement Units', 'Hesitations', 'Submovements', 'Jerk'],
                index=1
            )
        with col2:
            ascending = st.checkbox("Ascending", value=False)
        with col3:
            show_count = st.number_input("Show episodes", min_value=5, max_value=len(df), value=min(20, len(df)), step=5)

        df_sorted = df.sort_values(by=sort_by, ascending=ascending).head(show_count)

        def color_score(val):
            if isinstance(val, (int, float)) and val != int(val):
                if val > 0.65:
                    color = '#064E3B'
                elif val > 0.45:
                    color = '#78350F'
                else:
                    color = '#7F1D1D'
                return f'background-color: {color}'
            return ''

        styled_df = df_sorted.style.map(
            color_score,
            subset=['Overall', 'Movement Units', 'Hesitations', 'Submovements', 'Jerk']
        ).format({
            'Overall': '{:.3f}',
            'Movement Units': '{:.3f}',
            'Hesitations': '{:.3f}',
            'Submovements': '{:.3f}',
            'Jerk': '{:.3f}'
        })

        st.dataframe(styled_df, use_container_width=True, height=400)

        st.markdown("#### Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Episodes", len(df))
        with col2:
            excellent = len(df[df['Overall'] > 0.65])
            st.metric("Excellent (>0.65)", excellent, f"{excellent/len(df)*100:.1f}%")
        with col3:
            acceptable = len(df[(df['Overall'] > 0.45) & (df['Overall'] <= 0.65)])
            st.metric("Acceptable (0.45-0.65)", acceptable, f"{acceptable/len(df)*100:.1f}%")
        with col4:
            poor = len(df[df['Overall'] <= 0.45])
            st.metric("Needs Improvement (<0.45)", poor, f"{poor/len(df)*100:.1f}%")
        with col5:
            st.metric("Best Episode", int(df.loc[df['Overall'].idxmax(), 'Episode']), f"Score: {df['Overall'].max():.3f}")


def render_spatial_coverage_page():
    """Render spatial coverage analysis page."""
    render_page_header(
        "Spatial Coverage Analysis",
        "Workspace coverage and position diversity"
    )

    st.markdown("---")

    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    # Compute spatial coverage if not already done
    if 'spatial_coverage_results' not in st.session_state or st.session_state.spatial_coverage_results is None:
        with st.spinner("Computing spatial coverage metrics..."):
            try:
                from src.metrics.spatial_coverage import SpatialCoverageAnalyzer
                from src.utils.robot_models import get_robot_config
                from src.utils.config import get_config

                config = get_config()
                robot_config = get_robot_config(config.robot_type)

                analyzer = SpatialCoverageAnalyzer(
                    robot_config=robot_config,
                    voxel_size=config.get('metrics.spatial_coverage.voxel_size', 0.05)
                )

                results = analyzer.analyze_spatial_coverage(st.session_state.episodes_data)
                st.session_state.spatial_coverage_results = results

            except Exception as e:
                st.error(f"Failed to compute spatial coverage: {str(e)}")
                logger.error(f"Spatial coverage error: {e}", exc_info=True)
                return

    results = st.session_state.spatial_coverage_results

    # Display overall score with enhanced insights
    render_section_header("Overall Spatial Coverage Score")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Score", f"{results['overall_score']:.3f}")

    with col2:
        st.metric("Episodes Analyzed", results['n_episodes'])

    with col3:
        st.metric("Total Samples", results['n_total_samples'])

    with col4:
        coverage = results['workspace_coverage']
        # Check for reachable coverage first
        if coverage.get('reachable_coverage') is not None:
            st.metric("Reachable Coverage", f"{coverage['reachable_coverage_percentage']:.1f}%",
                     help="Percentage of robot's actual reachable workspace visited")
        else:
            st.metric("Workspace Coverage", f"{coverage['coverage_percentage']:.1f}%")

    st.markdown("---")

    # Enhanced metric breakdown with insights
    render_section_header("Coverage Analysis and Insights")

    # Show coverage quality badge if available
    if 'coverage_quality' in coverage:
        quality = coverage['coverage_quality']
        quality_lower = quality.lower()
        quality_descriptions = {
            'excellent': 'Comprehensive exploration of the reachable workspace',
            'good': 'Solid coverage for most tasks',
            'moderate': 'Focused on specific regions',
            'limited': 'Consider expanding exploration'
        }
        st.markdown(f"""
        <div class="status-badge status-{quality_lower}" style="margin-bottom: 1rem;">
            <span class="status-dot"></span>
            {quality.upper()} COVERAGE
        </div>
        <p style="color: #94A3B8; font-size: 0.875rem;">{quality_descriptions.get(quality_lower, '')}</p>
        """, unsafe_allow_html=True)

    # Show exploration metrics if available
    if 'exploration_metrics' in coverage:
        exp_metrics = coverage['exploration_metrics']

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Exploration Uniformity", f"{exp_metrics.get('exploration_uniformity', 0):.3f}",
                     help="How evenly distributed the exploration is (1.0 = perfectly uniform)")

        with col2:
            st.metric("Hotspots", exp_metrics.get('n_hotspots', 0),
                     help="Number of highly-explored regions")

        with col3:
            st.metric("Avg Visits/Voxel", f"{exp_metrics.get('mean_visits_per_voxel', 0):.1f}",
                     help="Average number of visits per explored voxel")

        with col4:
            variance_score = results['individual_scores']['starting_variance']
            st.metric("Starting Variance", f"{variance_score:.3f}",
                     help="Diversity in starting positions")

        with col5:
            angle_score = results['individual_scores']['angle_diversity']
            st.metric("Angle Diversity", f"{angle_score:.3f}",
                     help="Diversity in approach angles")

        if 'center_of_mass' in exp_metrics:
            com = exp_metrics['center_of_mass']
            st.info(f"Exploration Center of Mass: ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}) m")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            coverage_pct = results['workspace_coverage']['coverage_percentage']
            st.metric("Workspace Coverage", f"{coverage_pct:.1f}%")

        with col2:
            variance_score = results['individual_scores']['starting_variance']
            st.metric("Starting Variance", f"{variance_score:.3f}")

        with col3:
            dist_score = results['individual_scores']['position_distribution']
            st.metric("Position Distribution", f"{dist_score:.3f}")

        with col4:
            angle_score = results['individual_scores']['angle_diversity']
            st.metric("Angle Diversity", f"{angle_score:.3f}")

    st.markdown("---")

    # Visualizations
    render_section_header("Visualizations")

    try:
        from src.visualizations.spatial_viz import create_spatial_coverage_dashboard

        all_positions = np.vstack([ep['states'] for ep in st.session_state.episodes_data])
        starting_positions = np.array([ep['states'][0] for ep in st.session_state.episodes_data])

        figures = create_spatial_coverage_dashboard(
            results,
            all_positions,
            starting_positions
        )

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "3D Coverage",
            "Exploration Insights",
            "Starting Positions",
            "Position Density",
            "Summary",
            "Joint Information"
        ])

        with tab1:
            if 'workspace_3d' in figures:
                st.plotly_chart(figures['workspace_3d'], use_container_width=True)

        with tab2:
            try:
                from src.visualizations.workspace_insights_viz import (
                    plot_exploration_density_heatmap,
                    plot_coverage_insights_dashboard
                )

                coverage = results['workspace_coverage']
                if 'voxel_grid' in coverage and 'voxel_edges' in coverage:
                    st.markdown("#### Exploration Density Heatmap")
                    st.caption("Shows which regions were visited most frequently (hotter colors = more visits)")
                    density_fig = plot_exploration_density_heatmap(
                        coverage['voxel_grid'],
                        coverage['voxel_edges'],
                        title="Exploration Density - Where Did the Robot Spend Most Time?"
                    )
                    st.plotly_chart(density_fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Coverage Insights Dashboard")
                    insights_fig = plot_coverage_insights_dashboard(
                        coverage,
                        title="Workspace Coverage Insights"
                    )
                    st.plotly_chart(insights_fig, use_container_width=True)
                else:
                    st.info("Exploration insights available for 3D workspace data only.")
            except Exception as e:
                st.error(f"Could not load exploration insights: {str(e)}")
                logger.error(f"Exploration insights error: {e}", exc_info=True)

        with tab3:
            if 'starting_variance' in figures:
                st.plotly_chart(figures['starting_variance'], use_container_width=True)
            if 'starting_pca' in figures:
                st.plotly_chart(figures['starting_pca'], use_container_width=True)

        with tab4:
            if 'position_heatmap' in figures:
                st.plotly_chart(figures['position_heatmap'], use_container_width=True)

        with tab5:
            if 'coverage_summary' in figures:
                st.plotly_chart(figures['coverage_summary'], use_container_width=True)

        with tab6:
            try:
                from src.visualizations.interactive_robot_arm import create_interactive_robot_arm_figure
                from src.utils.config import get_config
                from src.utils.robot_models import get_robot_config

                config = get_config()
                robot_config = get_robot_config(config.robot_type)

                st.markdown("### Interactive Robot Arm - Move Joints to See How They Work")
                st.markdown("Use the sliders below to adjust each joint angle and see how the robot arm moves in real-time.")

                if 'interactive_joint_angles' not in st.session_state:
                    if robot_config:
                        st.session_state.interactive_joint_angles = (
                            robot_config.joint_limits_lower + robot_config.joint_limits_upper
                        ) / 2
                    else:
                        st.session_state.interactive_joint_angles = np.array([0.0, np.pi/4, -np.pi/4, 0.0, np.pi/4, 0.0])

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("#### Joint Controls")
                    joint_names = ["J1: Base Rotation", "J2: Shoulder Pitch", "J3: Elbow Pitch",
                                  "J4: Wrist Roll", "J5: Wrist Pitch", "J6: Wrist Yaw"]

                    updated_angles = []
                    for i, name in enumerate(joint_names[:robot_config.dof]):
                        if robot_config:
                            min_val = float(robot_config.joint_limits_lower[i])
                            max_val = float(robot_config.joint_limits_upper[i])
                        else:
                            min_val = -np.pi
                            max_val = np.pi

                        angle = st.slider(
                            name,
                            min_value=min_val,
                            max_value=max_val,
                            value=float(st.session_state.interactive_joint_angles[i]),
                            step=0.05,
                            key=f"joint_slider_{i}",
                            help=f"Adjust {name} to see how it affects the robot arm position"
                        )
                        updated_angles.append(angle)

                    while len(updated_angles) < 6:
                        updated_angles.append(0.0)

                    st.session_state.interactive_joint_angles = np.array(updated_angles)

                    if st.button("Reset to Neutral Pose", use_container_width=True):
                        if robot_config:
                            st.session_state.interactive_joint_angles = (
                                robot_config.joint_limits_lower + robot_config.joint_limits_upper
                            ) / 2
                        else:
                            st.session_state.interactive_joint_angles = np.array([0.0, np.pi/4, -np.pi/4, 0.0, np.pi/4, 0.0])
                        st.rerun()

                    st.markdown("---")
                    st.markdown("#### Joint Information")
                    joint_descriptions = [
                        ("J1", "Base rotation (shoulder yaw) - Rotates entire arm"),
                        ("J2", "Shoulder pitch - Lifts/lowers upper arm"),
                        ("J3", "Elbow pitch - Bends elbow joint"),
                        ("J4", "Wrist roll - Rotates wrist around arm axis"),
                        ("J5", "Wrist pitch - Bends wrist up/down"),
                        ("J6", "Wrist yaw - Rotates end-effector")
                    ]

                    for i, (j_name, desc) in enumerate(joint_descriptions[:robot_config.dof]):
                        with st.expander(f"{j_name}: {desc.split(' - ')[0]}"):
                            st.markdown(f"**Description:** {desc}")
                            if robot_config:
                                st.markdown(f"**Limits:** [{robot_config.joint_limits_lower[i]:.2f}, {robot_config.joint_limits_upper[i]:.2f}] rad")

                with col2:
                    arm_fig = create_interactive_robot_arm_figure(
                        robot_config=robot_config,
                        joint_angles=st.session_state.interactive_joint_angles
                    )
                    st.plotly_chart(arm_fig, use_container_width=True, key="interactive_arm")

            except Exception as e:
                st.error(f"Could not load interactive joint visualization: {str(e)}")
                logger.error(f"Interactive joint visualization error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logger.error(f"Spatial visualization error: {e}", exc_info=True)

    st.markdown("---")

    # Insights
    render_section_header("Key Insights")

    overall_score = results['overall_score']
    coverage_pct = results['workspace_coverage']['coverage_percentage']

    if overall_score > 0.7 and coverage_pct > 50:
        render_alert(
            title="Excellent spatial coverage!",
            message="Your data explores a diverse range of positions.",
            alert_type="success"
        )
    elif overall_score > 0.5:
        render_alert(
            title="Good spatial coverage with room for improvement",
            message="Consider expanding workspace exploration.",
            alert_type="info"
        )
    else:
        render_alert(
            title="Limited spatial coverage",
            message="Data collection may be concentrated in specific regions.",
            alert_type="warning"
        )


def render_data_consistency_page():
    """Render data consistency analysis page."""
    render_page_header(
        "Data Consistency Analysis",
        "Data quality and integrity checks"
    )

    st.markdown("---")

    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    # Compute consistency if not already done
    if 'consistency_results' not in st.session_state or st.session_state.consistency_results is None:
        with st.spinner("Analyzing data consistency..."):
            try:
                from src.metrics.data_consistency import DataConsistencyAnalyzer
                from src.utils.robot_models import get_robot_config
                from src.utils.config import get_config

                config = get_config()
                robot_config = get_robot_config(config.robot_type)

                analyzer = DataConsistencyAnalyzer(robot_config=robot_config)

                results = analyzer.analyze_data_consistency(
                    st.session_state.episodes_data,
                    expected_fps=30.0
                )

                st.session_state.consistency_results = results

            except Exception as e:
                st.error(f"Failed to analyze consistency: {str(e)}")
                logger.error(f"Consistency analysis error: {e}", exc_info=True)
                return

    results = st.session_state.consistency_results

    # Display overall score
    render_section_header("Overall Data Consistency Score")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Score", f"{results['overall_score']:.3f}")

    with col2:
        st.metric("Episodes Analyzed", results['n_episodes'])

    with col3:
        anomalies = results['anomalies']['n_anomalies']
        st.metric("Anomalies Detected", anomalies)

    st.markdown("---")

    # Metric breakdown
    render_section_header("Consistency Checks")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        align_score = results['individual_scores']['alignment']
        st.metric("Alignment", f"{align_score:.3f}")

    with col2:
        temporal_score = results['individual_scores']['temporal']
        st.metric("Temporal", f"{temporal_score:.3f}")

    with col3:
        missing_score = results['individual_scores']['missing_data']
        st.metric("Completeness", f"{missing_score:.3f}")

    with col4:
        anomaly_score = results['individual_scores']['anomalies']
        st.metric("Anomaly Score", f"{anomaly_score:.3f}")

    with col5:
        joint_score = results['individual_scores']['joint_limits']
        st.metric("Joint Limits", f"{joint_score:.3f}")

    st.markdown("---")

    # Visualizations
    render_section_header("Visualizations")

    try:
        from src.visualizations.consistency_viz import create_consistency_dashboard

        figures = create_consistency_dashboard(
            results,
            st.session_state.episodes_data
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Summary",
            "Temporal",
            "Missing Data",
            "Alignment"
        ])

        with tab1:
            if 'consistency_summary' in figures:
                st.plotly_chart(figures['consistency_summary'], use_container_width=True)

        with tab2:
            if 'temporal_consistency' in figures:
                st.plotly_chart(figures['temporal_consistency'], use_container_width=True)

        with tab3:
            if 'missing_data' in figures:
                st.plotly_chart(figures['missing_data'], use_container_width=True)

        with tab4:
            if 'alignment_issues' in figures:
                st.plotly_chart(figures['alignment_issues'], use_container_width=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logger.error(f"Consistency visualization error: {e}", exc_info=True)

    st.markdown("---")

    # Insights
    render_section_header("Key Insights")

    overall_score = results['overall_score']
    missing_pct = results['missing_data']['completeness_percentage']
    anomaly_pct = results['anomalies']['anomaly_percentage']

    if overall_score > 0.8 and missing_pct > 95:
        render_alert(
            title="Excellent data quality!",
            message="Data is consistent, complete, and well-aligned.",
            alert_type="success"
        )
    elif overall_score > 0.6:
        render_alert(
            title="Good data quality",
            message=f"Found {anomaly_pct:.1f}% anomalies and {100-missing_pct:.1f}% missing data.",
            alert_type="info"
        )
    else:
        render_alert(
            title="Data quality issues detected",
            message="Review anomalies and missing data carefully.",
            alert_type="warning"
        )


def render_video_playback_page():
    """Render video playback page with trajectory visualization."""
    render_page_header(
        "Trajectory Playback",
        "Visualize and explore episode trajectories"
    )

    st.markdown("---")

    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    episodes_data = st.session_state.episodes_data
    n_episodes = len(episodes_data)

    # Episode selector
    render_section_header("Episode Selection")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_episode = st.selectbox(
            "Select Episode",
            options=list(range(n_episodes)),
            format_func=lambda x: f"Episode {x} ({len(episodes_data[x].get('states', []))} frames)",
            key="video_episode_selector"
        )

    with col2:
        episode_data = episodes_data[selected_episode]
        n_frames = len(episode_data.get('states', []))
        st.metric("Total Frames", n_frames)

    # Frame slider
    if n_frames > 0:
        current_frame = st.slider(
            "Frame",
            min_value=0,
            max_value=n_frames - 1,
            value=0,
            key="video_frame_slider"
        )
    else:
        current_frame = 0

    st.markdown("---")

    # Trajectory visualization
    try:
        from src.visualizations.trajectory_playback import create_trajectory_dashboard

        figures = create_trajectory_dashboard(episode_data, current_frame)

        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "3D Trajectory",
            "State Timeline",
            "Action Timeline",
            "Animated Playback",
            "Camera Playback"
        ])

        with tab1:
            render_section_header("3D Trajectory Visualization")

            if 'trajectory_3d' in figures:
                st.plotly_chart(figures['trajectory_3d'], use_container_width=True)

                # Position info
                states = np.array(episode_data.get('states', []))
                if len(states) > 0:
                    current_state = states[min(current_frame, len(states)-1)]
                    n_dims = len(current_state) if hasattr(current_state, '__len__') else 1

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("X Position", f"{current_state[0]:.4f}" if n_dims >= 1 else "N/A")
                    with col2:
                        st.metric("Y Position", f"{current_state[1]:.4f}" if n_dims >= 2 else "N/A")
                    with col3:
                        st.metric("Z Position", f"{current_state[2]:.4f}" if n_dims >= 3 else "N/A")

            if 'velocity_profile' in figures:
                st.markdown("#### Velocity Profile")
                st.plotly_chart(figures['velocity_profile'], use_container_width=True)

        with tab2:
            render_section_header("State Evolution Over Time")

            if 'state_timeline' in figures:
                st.plotly_chart(figures['state_timeline'], use_container_width=True)

            # State statistics
            states = np.array(episode_data.get('states', []))
            if len(states) > 0:
                st.markdown("#### State Statistics")
                import pandas as pd

                n_dims = states.shape[1] if len(states.shape) > 1 else 1
                stats_data = []
                for i in range(min(n_dims, 6)):
                    dim_data = states[:, i] if n_dims > 1 else states
                    stats_data.append({
                        'Dimension': f"Dim {i+1}",
                        'Mean': f"{np.mean(dim_data):.4f}",
                        'Std': f"{np.std(dim_data):.4f}",
                        'Min': f"{np.min(dim_data):.4f}",
                        'Max': f"{np.max(dim_data):.4f}"
                    })

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with tab3:
            render_section_header("Action Evolution Over Time")

            if 'action_timeline' in figures:
                st.plotly_chart(figures['action_timeline'], use_container_width=True)

            # Action statistics
            actions = np.array(episode_data.get('actions', []))
            if len(actions) > 0:
                st.markdown("#### Action Statistics")
                import pandas as pd

                n_dims = actions.shape[1] if len(actions.shape) > 1 else 1
                stats_data = []
                for i in range(min(n_dims, 6)):
                    dim_data = actions[:, i] if n_dims > 1 else actions
                    stats_data.append({
                        'Dimension': f"Action {i+1}",
                        'Mean': f"{np.mean(dim_data):.4f}",
                        'Std': f"{np.std(dim_data):.4f}",
                        'Min': f"{np.min(dim_data):.4f}",
                        'Max': f"{np.max(dim_data):.4f}"
                    })

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("No action data available for this episode.")

        with tab4:
            render_section_header("Animated Trajectory Playback")

            if 'animated_trajectory' in figures:
                st.markdown("""
                **Controls:**
                - Click **Play** to start the animation
                - Click **Pause** to stop
                - Use the slider below the plot to jump to specific frames
                """)
                st.plotly_chart(figures['animated_trajectory'], use_container_width=True)
            else:
                st.info("Animation not available for this episode.")

        with tab5:
            render_section_header("Camera Playback")

            # Get loader and check for cameras
            loader = st.session_state.get('loader')
            if loader is None:
                st.warning("Dataset loader not available.")
            elif not hasattr(loader, 'get_camera_names'):
                st.warning("Please restart the Streamlit app to enable camera playback (new features require app restart).")
            else:
                try:
                    # Cache camera names in session state (only detect once)
                    if 'available_cameras' not in st.session_state:
                        with st.spinner("Detecting cameras..."):
                            st.session_state.available_cameras = loader.get_camera_names()

                    camera_names = st.session_state.available_cameras

                    if not camera_names:
                        st.info("No camera data found in this dataset.")
                    else:
                        st.success(f"Found {len(camera_names)} camera(s): {', '.join(camera_names)}")

                        # Get the global frame indices for this episode
                        episode_idx = episodes_data[selected_episode]['episode_index']
                        frame_indices = loader.get_episode_frame_indices(episode_idx)

                        if frame_indices and current_frame < len(frame_indices):
                            global_frame_idx = frame_indices[current_frame]

                            # Load only the current frame's images (on-demand)
                            frame_images = loader.get_frame_images(global_frame_idx, camera_names)

                            if frame_images:
                                # Display cameras in columns
                                n_cameras = len(frame_images)
                                if n_cameras == 1:
                                    cols = [st.container()]
                                elif n_cameras == 2:
                                    cols = st.columns(2)
                                else:
                                    cols = st.columns(min(n_cameras, 3))

                                for i, (cam_name, img) in enumerate(frame_images.items()):
                                    col_idx = i % len(cols)
                                    with cols[col_idx]:
                                        st.markdown(f"**{cam_name}**")
                                        st.image(
                                            img,
                                            caption=f"Frame {current_frame + 1}/{n_frames}",
                                            use_container_width=True
                                        )

                                # Frame info
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Current Frame", current_frame + 1)
                                with col2:
                                    st.metric("Total Frames", n_frames)
                                with col3:
                                    if n_frames > 0:
                                        progress = (current_frame + 1) / n_frames * 100
                                        st.metric("Progress", f"{progress:.1f}%")

                                st.caption("Use the frame slider above to scrub through the video.")
                            else:
                                st.warning("Could not load images for this frame.")
                        else:
                            st.warning("Frame indices not available for this episode.")

                except Exception as e:
                    error_msg = str(e)
                    # Check for FFmpeg/torchcodec error
                    if 'torchcodec' in error_msg.lower() or 'ffmpeg' in error_msg.lower() or 'libtorchcodec' in error_msg.lower():
                        st.error("**FFmpeg 7 Required for Video Playback**")
                        st.markdown("""
                        Camera playback requires FFmpeg (version 4-7) to decode video data from LeRobot datasets.

                        **Note:** FFmpeg 8 is NOT supported. You need version 7 or earlier.

                        **To install FFmpeg 7 on macOS:**
                        ```bash
                        brew install ffmpeg@7
                        brew link ffmpeg@7 --force
                        ```

                        **On Ubuntu/Debian:**
                        ```bash
                        sudo apt-get install ffmpeg
                        ```

                        After installing, restart the Streamlit app.
                        """)
                    else:
                        st.error(f"Error loading camera data: {error_msg}")
                    logger.error(f"Camera playback error: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logger.error(f"Trajectory visualization error: {e}", exc_info=True)

    st.markdown("---")

    # Episode info
    render_section_header("Episode Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Episode Index", selected_episode)

    with col2:
        st.metric("Total Frames", n_frames)

    with col3:
        # Calculate duration if timestamps available
        timestamps = episode_data.get('timestamps')
        if timestamps is not None and len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            st.metric("Duration", f"{duration:.2f}s")
        else:
            st.metric("Duration", f"{n_frames / 30:.2f}s (est.)")

    with col4:
        states = episode_data.get('states', [])
        n_dims = states[0].shape[0] if len(states) > 0 and hasattr(states[0], 'shape') else len(states[0]) if len(states) > 0 else 0
        st.metric("State Dimensions", n_dims)


def render_success_indicators_page():
    """Render success indicators analysis page."""
    render_page_header(
        "Success Indicators",
        "Episode-level success analysis and quality distribution"
    )

    st.markdown("---")

    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    # Compute success indicators if not already done
    if st.session_state.success_indicator_results is None:
        with st.spinner("Analyzing success indicators..."):
            try:
                from src.metrics.success_indicators import SuccessIndicatorAnalyzer

                analyzer = SuccessIndicatorAnalyzer()

                # Get motion quality results if available for quality scores
                motion_results = st.session_state.get('motion_quality_results')
                spatial_results = st.session_state.get('spatial_coverage_results')

                results = analyzer.analyze_success_indicators(
                    st.session_state.episodes_data,
                    motion_quality_results=motion_results,
                    spatial_coverage_results=spatial_results
                )

                st.session_state.success_indicator_results = results

            except Exception as e:
                st.error(f"Failed to analyze success indicators: {str(e)}")
                logger.error(f"Success indicator error: {e}", exc_info=True)
                return

    results = st.session_state.success_indicator_results

    # Section 1: Overview metrics
    render_section_header("Success Overview")

    col1, col2, col3, col4 = st.columns(4)

    success_rates = results['success_rates']

    with col1:
        success_pct = success_rates['overall_percentage']
        color = "#059669" if success_pct >= 70 else "#D97706" if success_pct >= 50 else "#DC2626"
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: #1E293B; border-radius: 8px;">
            <div style="font-size: 2rem; font-weight: 700; color: {color};">
                {success_pct:.1f}%
            </div>
            <div style="color: #94A3B8; font-size: 0.875rem;">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Total Episodes", results['n_episodes'])

    with col3:
        st.metric("Successful Episodes", success_rates['n_successful'])

    with col4:
        length_stats = results['length_stats']
        st.metric("Avg Length", f"{length_stats['mean']:.0f} frames")

    st.markdown("---")

    # Section 2: Quality Classification
    render_section_header("Quality Classification")

    col1, col2, col3, col4 = st.columns(4)
    counts = success_rates['counts']
    percentages = success_rates['percentages']

    with col1:
        st.markdown(f"""
        <div style="background: rgba(5, 150, 105, 0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #059669;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #059669;">{counts['excellent']}</div>
            <div style="color: #94A3B8;">Excellent ({percentages['excellent']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: rgba(8, 145, 178, 0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #0891B2;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #0891B2;">{counts['good']}</div>
            <div style="color: #94A3B8;">Good ({percentages['good']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: rgba(217, 119, 6, 0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #D97706;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #D97706;">{counts['marginal']}</div>
            <div style="color: #94A3B8;">Marginal ({percentages['marginal']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: rgba(220, 38, 38, 0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #DC2626;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #DC2626;">{counts['poor']}</div>
            <div style="color: #94A3B8;">Poor ({percentages['poor']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 3: Visualizations
    render_section_header("Visualizations")

    try:
        from src.visualizations.success_viz import create_success_dashboard

        figures = create_success_dashboard(results)

        tab1, tab2, tab3, tab4 = st.tabs([
            "Distribution",
            "Quality Analysis",
            "Best/Worst Episodes",
            "Correlations"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figures['length_histogram'], use_container_width=True)
            with col2:
                st.plotly_chart(figures['success_rate_pie'], use_container_width=True)

            st.plotly_chart(figures['classification_bar'], use_container_width=True)

        with tab2:
            st.plotly_chart(figures['quality_distribution'], use_container_width=True)
            st.plotly_chart(figures['quality_scatter'], use_container_width=True)

        with tab3:
            st.plotly_chart(figures['best_worst'], use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Best Performing Episodes")
                import pandas as pd
                best_df = pd.DataFrame(results['best_episodes'])
                best_df.columns = ['Episode', 'Quality', 'Length', 'Classification']
                st.dataframe(best_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### Worst Performing Episodes")
                worst_df = pd.DataFrame(results['worst_episodes'])
                worst_df.columns = ['Episode', 'Quality', 'Length', 'Classification']
                st.dataframe(worst_df, use_container_width=True, hide_index=True)

        with tab4:
            st.plotly_chart(figures['correlation'], use_container_width=True)

            correlation = results['length_quality_correlation']
            st.info(f"**Interpretation:** {correlation['interpretation']}")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        logger.error(f"Success visualization error: {e}", exc_info=True)

    st.markdown("---")

    # Section 4: Length Statistics
    render_section_header("Episode Length Statistics")

    length_stats = results['length_stats']

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean", f"{length_stats['mean']:.1f}")
    with col2:
        st.metric("Median", f"{length_stats['median']:.1f}")
    with col3:
        st.metric("Std Dev", f"{length_stats['std']:.1f}")
    with col4:
        st.metric("Min", length_stats['min'])
    with col5:
        st.metric("Max", length_stats['max'])

    st.caption(f"25th percentile: {length_stats['q25']:.0f} | 75th percentile: {length_stats['q75']:.0f}")

    # Insights
    st.markdown("---")
    render_section_header("Key Insights")

    success_pct = success_rates['overall_percentage']
    if success_pct >= 70:
        render_alert(
            title="Strong success rate!",
            message=f"{success_pct:.1f}% of episodes meet quality thresholds. Dataset is well-suited for training.",
            alert_type="success"
        )
    elif success_pct >= 50:
        render_alert(
            title="Moderate success rate",
            message=f"{success_pct:.1f}% of episodes meet quality thresholds. Consider reviewing lower-quality episodes.",
            alert_type="info"
        )
    else:
        render_alert(
            title="Low success rate",
            message=f"Only {success_pct:.1f}% of episodes meet quality thresholds. Review recommendations for improvement.",
            alert_type="warning"
        )


def render_recommendations_page():
    """Render recommendations page with prioritized suggestions."""
    render_page_header(
        "Recommendations",
        "Actionable suggestions to improve your dataset quality"
    )

    st.markdown("---")

    if not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first from the Overview page.")
        return

    if not st.session_state.get('data_preprocessing_complete', False):
        st.error("Episodes not preprocessed. Please reload the dataset from the Overview page.")
        return

    # Generate recommendations
    try:
        from src.analysis.recommendation_engine import RecommendationEngine, Priority

        engine = RecommendationEngine()

        # Get all available results
        motion_results = st.session_state.get('motion_quality_results')
        spatial_results = st.session_state.get('spatial_coverage_results')
        consistency_results = st.session_state.get('consistency_results')

        # Check if any metrics have been computed
        if not any([motion_results, spatial_results, consistency_results]):
            render_alert(
                title="No metrics computed yet",
                message="Please visit the Execution, Coverage, or Consistency pages first to compute metrics before viewing recommendations.",
                alert_type="info"
            )
            return

        # Generate comprehensive report
        report = engine.generate_comprehensive_report(
            motion_quality_results=motion_results,
            spatial_coverage_results=spatial_results,
            consistency_results=consistency_results
        )

        # Section 1: Overall Health Score
        render_section_header("Dataset Health Score")

        col1, col2 = st.columns([1, 2])

        with col1:
            health_score = report['health_score']

            # Color based on score
            if health_score >= 0.7:
                color = "#059669"
                status = "Healthy"
            elif health_score >= 0.5:
                color = "#D97706"
                status = "Moderate"
            else:
                color = "#DC2626"
                status = "Needs Attention"

            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #1E293B; border-radius: 12px; border: 1px solid #334155;">
                <div style="font-size: 3.5rem; font-weight: 700; color: {color}; font-family: 'JetBrains Mono', monospace;">
                    {health_score:.0%}
                </div>
                <div style="font-size: 1.25rem; color: {color}; font-weight: 600; margin-top: 0.5rem;">
                    {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Summary cards
            priority_counts = report['priority_counts']

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(
                    "High Priority",
                    priority_counts[Priority.HIGH],
                    help="Critical issues that should be addressed"
                )
            with col_b:
                st.metric(
                    "Medium Priority",
                    priority_counts[Priority.MEDIUM],
                    help="Important issues for optimization"
                )
            with col_c:
                st.metric(
                    "Low Priority",
                    priority_counts[Priority.LOW],
                    help="Minor improvements"
                )

            # Status message
            if priority_counts[Priority.HIGH] == 0 and priority_counts[Priority.MEDIUM] == 0:
                render_alert(
                    title="Excellent dataset quality!",
                    message="No significant issues detected. Your dataset is ready for ACT training.",
                    alert_type="success"
                )
            elif priority_counts[Priority.HIGH] > 0:
                render_alert(
                    title=f"{priority_counts[Priority.HIGH]} high priority issue(s) detected",
                    message="Address these issues to significantly improve training outcomes.",
                    alert_type="warning"
                )
            else:
                render_alert(
                    title="Good quality with room for improvement",
                    message="Consider addressing medium priority items for optimal results.",
                    alert_type="info"
                )

        st.markdown("---")

        # Section 2: Priority Recommendations
        if report['total_recommendations'] > 0:
            render_section_header("Priority Recommendations")

            for rec in report['recommendations']:
                priority_colors = {
                    Priority.HIGH: '#DC2626',
                    Priority.MEDIUM: '#D97706',
                    Priority.LOW: '#059669'
                }
                color = priority_colors.get(rec.priority, '#64748B')

                with st.expander(
                    f"**[{rec.priority.value}]** {rec.metric} - Score: {rec.score:.3f}",
                    expanded=(rec.priority == Priority.HIGH)
                ):
                    st.markdown(f"""
                    <div style="border-left: 3px solid {color}; padding-left: 1rem; margin: 0.5rem 0;">
                        <p><strong>Category:</strong> {rec.category}</p>
                        <p><strong>Issue:</strong> {rec.issue}</p>
                        <p><strong>What to do:</strong> {rec.advice}</p>
                        <p><strong>Technique:</strong> {rec.technique}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

        # Section 3: Category Breakdown
        render_section_header("Recommendations by Category")

        tab_names = list(report['by_category'].keys())
        if tab_names:
            tabs = st.tabs(tab_names)

            for tab, category in zip(tabs, tab_names):
                with tab:
                    category_recs = report['by_category'][category]
                    if category_recs:
                        for rec in category_recs:
                            priority_colors = {
                                Priority.HIGH: '#DC2626',
                                Priority.MEDIUM: '#D97706',
                                Priority.LOW: '#059669'
                            }
                            color = priority_colors.get(rec.priority, '#64748B')

                            st.markdown(f"""
                            <div style="background: #1E293B; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid {color};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 600; color: #F8FAFC;">{rec.metric}</span>
                                    <span style="color: {color}; font-weight: 500;">{rec.priority.value}</span>
                                </div>
                                <p style="color: #94A3B8; margin: 0.5rem 0; font-size: 0.9rem;">{rec.issue}</p>
                                <p style="color: #F8FAFC; margin: 0.25rem 0; font-size: 0.875rem;"><strong>Advice:</strong> {rec.advice}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No recommendations for this category.")
        else:
            st.success("No recommendations needed - all metrics are in good range!")

        st.markdown("---")

        # Section 4: Problematic Episodes
        render_section_header("Problematic Episodes")

        problematic = report['problematic_episodes']

        if problematic['multiple_issues'] or problematic['single_issue']:
            import pandas as pd

            # Episodes with multiple issues
            if problematic['multiple_issues']:
                st.markdown("#### Episodes with Multiple Issues")
                st.caption("These episodes appear in multiple worst-performing lists and should be reviewed first.")

                multi_data = []
                for ep_idx in problematic['multiple_issues']:
                    issues = problematic['issue_details'].get(ep_idx, [])
                    multi_data.append({
                        'Episode': ep_idx,
                        'Issues': ', '.join(issues),
                        'Count': len(issues)
                    })

                df_multi = pd.DataFrame(multi_data)
                st.dataframe(df_multi, use_container_width=True, hide_index=True)

            # Episodes with single issues
            if problematic['single_issue']:
                with st.expander(f"Episodes with Single Issue ({len(problematic['single_issue'])} episodes)"):
                    single_data = []
                    for ep_idx in problematic['single_issue'][:20]:  # Limit to 20
                        issues = problematic['issue_details'].get(ep_idx, [])
                        single_data.append({
                            'Episode': ep_idx,
                            'Issue': issues[0] if issues else 'Unknown'
                        })

                    df_single = pd.DataFrame(single_data)
                    st.dataframe(df_single, use_container_width=True, hide_index=True)

                    if len(problematic['single_issue']) > 20:
                        st.caption(f"Showing 20 of {len(problematic['single_issue'])} episodes")
        else:
            st.success("No particularly problematic episodes identified.")

    except Exception as e:
        st.error(f"Failed to generate recommendations: {str(e)}")
        logger.error(f"Recommendations error: {e}", exc_info=True)


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Navigation - Top panel for better UX
    if st.session_state.dataset_loaded:
        pages = [
            "Overview",
            "Execution",
            "Coverage",
            "Consistency",
            "Success",
            "Video",
            "Recommendations"
        ]

        # Create modern top navigation bar
        current_idx = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0

        # Navigation bar container
        st.markdown('<div class="nav-bar-container">', unsafe_allow_html=True)
        st.markdown('<div class="nav-buttons-row">', unsafe_allow_html=True)

        cols = st.columns(len(pages))
        selected_page = st.session_state.current_page

        for i, (col, page) in enumerate(zip(cols, pages)):
            with col:
                st.markdown('<div class="nav-button-wrapper">', unsafe_allow_html=True)
                button_type = "primary" if i == current_idx else "secondary"
                if st.button(page, key=f"nav_{i}", use_container_width=True, type=button_type):
                    st.session_state.current_page = page
                    selected_page = page
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Route to appropriate page
        if selected_page == "Overview":
            render_home_page()
        elif selected_page == "Execution":
            render_motion_quality_page()
        elif selected_page == "Coverage":
            render_spatial_coverage_page()
        elif selected_page == "Consistency":
            render_data_consistency_page()
        elif selected_page == "Success":
            render_success_indicators_page()
        elif selected_page == "Video":
            render_video_playback_page()
        elif selected_page == "Recommendations":
            render_recommendations_page()
    else:
        # Show home page if no dataset loaded
        render_home_page()


if __name__ == "__main__":
    main()
