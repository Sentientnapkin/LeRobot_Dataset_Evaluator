"""Spatial coverage and diversity metrics."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import entropy

from .utils import compute_statistics, create_metric_summary
from ..utils.logging_config import get_logger
from ..utils.robot_models import RobotConfig, is_within_workspace

logger = get_logger(__name__)


# Coverage quality interpretation thresholds
COVERAGE_QUALITY_THRESHOLDS = {
    'excellent': 0.50,  # >= 50% reachable coverage
    'good': 0.30,       # >= 30% reachable coverage
    'moderate': 0.15,   # >= 15% reachable coverage
    # < 15% = limited
}


def get_coverage_quality(reachable_coverage: float) -> str:
    """Get quality interpretation for reachable coverage percentage.

    Args:
        reachable_coverage: Coverage as a fraction (0-1)

    Returns:
        Quality string: 'Excellent', 'Good', 'Moderate', or 'Limited'
    """
    if reachable_coverage >= COVERAGE_QUALITY_THRESHOLDS['excellent']:
        return 'Excellent'
    elif reachable_coverage >= COVERAGE_QUALITY_THRESHOLDS['good']:
        return 'Good'
    elif reachable_coverage >= COVERAGE_QUALITY_THRESHOLDS['moderate']:
        return 'Moderate'
    else:
        return 'Limited'


class SpatialCoverageAnalyzer:
    """Analyzer for spatial coverage and diversity metrics."""

    def __init__(
        self,
        robot_config: Optional[RobotConfig] = None,
        voxel_size: float = 0.05,
        min_coverage_threshold: float = 0.4,
        workspace_samples: int = 10000
    ):
        """Initialize spatial coverage analyzer.

        Args:
            robot_config: Robot configuration for workspace bounds
            voxel_size: Size of voxel grid cells (meters)
            min_coverage_threshold: Minimum acceptable coverage (0-1)
            workspace_samples: Number of samples for Monte Carlo workspace estimation
        """
        self.robot_config = robot_config
        self.voxel_size = voxel_size
        self.min_coverage_threshold = min_coverage_threshold
        self.workspace_samples = workspace_samples
        self._reachable_workspace_cache = None

    def _compute_reachable_workspace(self) -> Tuple[np.ndarray, Dict]:
        """Compute the reachable workspace using Monte Carlo sampling.

        Uses forward kinematics to sample end-effector positions for random
        joint configurations within joint limits.

        Returns:
            Tuple of:
            - Boolean voxel grid indicating reachable voxels
            - Metadata dict with bounds and statistics
        """
        if self._reachable_workspace_cache is not None:
            return self._reachable_workspace_cache

        if self.robot_config is None:
            # No robot config - cannot compute reachable workspace
            return None, {}

        try:
            # Get kinematics instance from robot config
            kinematics = self.robot_config.get_kinematics()

            # Compute reachable voxels using FK-based sampling
            reachable_grid, metadata = kinematics.compute_reachable_voxels(
                voxel_size=self.voxel_size,
                n_samples=self.workspace_samples
            )

            self._reachable_workspace_cache = (reachable_grid, metadata)
            return reachable_grid, metadata

        except Exception as e:
            logger.warning(f"Could not compute reachable workspace: {e}")
            return None, {}

    def compute_reachable_coverage(
        self,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Compute coverage as percentage of reachable workspace visited.

        This is the improved metric that compares visited voxels against
        the robot's actual reachable workspace (not the bounding box).

        Args:
            positions: End-effector positions, shape (n_samples, 3)

        Returns:
            Dictionary with reachable coverage metrics
        """
        # Get reachable workspace
        reachable_grid, workspace_meta = self._compute_reachable_workspace()

        if reachable_grid is None:
            # Fallback to bounding box coverage
            logger.info("Reachable workspace not available, using bounding box")
            return {
                'reachable_coverage': None,
                'reachable_coverage_available': False
            }

        # Extract bounds from metadata
        bounds = workspace_meta['bounds']
        edges = workspace_meta['edges']

        # Create histogram of visited positions using same grid
        visited_grid, _ = np.histogramdd(
            positions[:, :3],
            bins=[len(e) - 1 for e in edges],
            range=[bounds['x'], bounds['y'], bounds['z']]
        )

        # Count voxels visited that are also reachable
        visited_reachable = (visited_grid > 0) & reachable_grid
        n_visited_reachable = int(np.sum(visited_reachable))
        n_reachable_total = workspace_meta['n_reachable_voxels']

        # Compute reachable coverage percentage
        if n_reachable_total > 0:
            reachable_coverage = n_visited_reachable / n_reachable_total
        else:
            reachable_coverage = 0.0

        # Get quality interpretation
        coverage_quality = get_coverage_quality(reachable_coverage)

        return {
            'reachable_coverage': reachable_coverage,
            'reachable_coverage_percentage': reachable_coverage * 100,
            'n_visited_reachable_voxels': n_visited_reachable,
            'n_reachable_voxels': n_reachable_total,
            'coverage_quality': coverage_quality,
            'reachable_coverage_available': True,
            'workspace_bounds': bounds
        }

    def _detect_data_space(self, positions: np.ndarray) -> str:
        """Detect if positions are in joint space or Cartesian space.

        Args:
            positions: Position array, shape (n_samples, n_dof)

        Returns:
            'joint' or 'cartesian'
        """
        # Check if values are outside typical Cartesian workspace (-5 to 5 meters)
        pos_min = np.min(positions)
        pos_max = np.max(positions)

        # If values are outside typical Cartesian range, assume joint space
        if pos_min < -10 or pos_max > 10:
            return 'joint'

        # If values are in typical Cartesian range
        return 'cartesian'

    def _compute_joint_space_coverage(
        self,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Compute coverage in joint space using joint angle ranges.

        Args:
            positions: Joint angles, shape (n_samples, n_dof)

        Returns:
            Dictionary containing coverage metrics
        """
        n_dof = positions.shape[1]

        # Always use data range with padding (joint limits from config may not match actual data units)
        joint_ranges = []
        for i in range(n_dof):
            data_min = np.min(positions[:, i])
            data_max = np.max(positions[:, i])
            padding = (data_max - data_min) * 0.1
            joint_ranges.append((data_min - padding, data_max + padding))

        # Create multi-dimensional histogram
        # Adaptive binning based on data density
        n_samples = positions.shape[0]
        bins_per_dof = []

        # Scale bins based on number of samples to avoid sparse histograms
        # Target: total_bins^n_dof should be ~10-100 times number of samples
        target_bins_per_dof = max(5, min(15, int(np.power(n_samples * 50, 1.0 / n_dof))))

        for joint_min, joint_max in joint_ranges:
            bins_per_dof.append(target_bins_per_dof)

        # Create histogram
        hist, edges = np.histogramdd(
            positions,
            bins=bins_per_dof,
            range=joint_ranges
        )

        # Count occupied voxels
        n_voxels_visited = np.sum(hist > 0)
        n_voxels_total = np.prod(bins_per_dof)

        # Coverage percentage
        coverage_percentage = (n_voxels_visited / n_voxels_total) * 100

        # Additional insightful metrics for joint space
        visit_counts = hist.flatten()
        non_zero_counts = visit_counts[visit_counts > 0]
        
        if len(non_zero_counts) > 0:
            mean_visits = np.mean(non_zero_counts)
            median_visits = np.median(non_zero_counts)
            max_visits = np.max(non_zero_counts)
            exploration_std = np.std(non_zero_counts)
            exploration_uniformity = 1.0 / (1.0 + exploration_std / (mean_visits + 1e-10))
            
            hotspot_threshold = np.percentile(non_zero_counts, 75)
            n_hotspots = np.sum(hist >= hotspot_threshold)
        else:
            mean_visits = 0
            median_visits = 0
            max_visits = 0
            exploration_uniformity = 0.0
            n_hotspots = 0

        # Quality score (0-1, higher is better)
        # For teleoperation data, use logarithmic scaling to be more forgiving
        # Teleoperation naturally focuses on task-relevant regions
        if coverage_percentage > 0:
            max_reasonable_coverage = 50.0  # 50% is excellent for teleoperation
            log_coverage = np.log1p(coverage_percentage)
            log_max = np.log1p(max_reasonable_coverage)
            coverage_component = min(log_coverage / log_max, 1.0)
            
            # Combined score: coverage (60%), uniformity (40%)
            coverage_score = 0.6 * coverage_component + 0.4 * exploration_uniformity
        else:
            coverage_score = 0.0

        return {
            'coverage_percentage': coverage_percentage,
            'n_voxels_visited': int(n_voxels_visited),
            'n_voxels_total': int(n_voxels_total),
            'voxel_grid': hist,
            'voxel_edges': edges,
            'joint_ranges': joint_ranges,
            'data_space': 'joint',
            'n_dof': n_dof,
            'score': coverage_score,
            # New insightful metrics
            'exploration_metrics': {
                'mean_visits_per_voxel': float(mean_visits),
                'median_visits_per_voxel': float(median_visits),
                'max_visits_per_voxel': int(max_visits),
                'n_hotspots': int(n_hotspots),
                'exploration_uniformity': float(exploration_uniformity)
            }
        }

    def compute_workspace_coverage(
        self,
        positions: np.ndarray,
        workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Compute workspace coverage using 3D voxel grid.

        Automatically detects if data is in joint space or Cartesian space.

        Args:
            positions: Cartesian positions or joint angles, shape (n_samples, 3) or (n_samples, n_dof)
            workspace_bounds: Optional workspace bounds, defaults to robot config

        Returns:
            Dictionary containing coverage metrics
        """
        # Detect data space
        data_space = self._detect_data_space(positions)

        if data_space == 'joint':
            # For joint space, compute coverage over joint angle ranges
            logger.info("Detected joint space data, computing joint space coverage")
            return self._compute_joint_space_coverage(positions)

        # Cartesian space coverage
        logger.info("Detected Cartesian space data, computing workspace coverage")

        # Use robot config bounds if available
        if workspace_bounds is None and self.robot_config is not None:
            workspace_bounds = self.robot_config.workspace_bounds

        if workspace_bounds is None:
            # Default workspace bounds if none provided
            workspace_bounds = {
                'x': (-1.0, 1.0),
                'y': (-1.0, 1.0),
                'z': (0.0, 1.5)
            }

        # Extract x, y, z positions (first 3 dimensions if more available)
        if positions.shape[1] >= 3:
            xyz_positions = positions[:, :3]
        else:
            logger.warning("Positions have less than 3 dimensions, cannot compute workspace coverage")
            return {
                'coverage_percentage': 0.0,
                'n_voxels_visited': 0,
                'n_voxels_total': 0,
                'score': 0.0,
                'data_space': 'cartesian'
            }

        # Create voxel grid
        x_range = workspace_bounds['x']
        y_range = workspace_bounds['y']
        z_range = workspace_bounds['z']

        x_bins = int((x_range[1] - x_range[0]) / self.voxel_size)
        y_bins = int((y_range[1] - y_range[0]) / self.voxel_size)
        z_bins = int((z_range[1] - z_range[0]) / self.voxel_size)

        # Create 3D histogram
        voxel_grid, edges = np.histogramdd(
            xyz_positions,
            bins=[x_bins, y_bins, z_bins],
            range=[x_range, y_range, z_range]
        )

        # Count occupied voxels
        n_voxels_visited = np.sum(voxel_grid > 0)
        n_voxels_total = x_bins * y_bins * z_bins

        # Coverage percentage
        coverage_percentage = (n_voxels_visited / n_voxels_total) * 100

        # Additional insightful metrics for teleoperation data
        
        # 1. Exploration density analysis
        # Identify well-explored vs under-explored regions
        visit_counts = voxel_grid.flatten()
        non_zero_counts = visit_counts[visit_counts > 0]
        
        if len(non_zero_counts) > 0:
            # Density metrics
            mean_visits = np.mean(non_zero_counts)
            median_visits = np.median(non_zero_counts)
            max_visits = np.max(non_zero_counts)
            
            # Identify hotspots (highly visited regions) and coldspots (rarely visited)
            hotspot_threshold = np.percentile(non_zero_counts, 75)  # Top 25%
            coldspot_threshold = np.percentile(non_zero_counts, 25)  # Bottom 25%
            
            n_hotspots = np.sum(voxel_grid >= hotspot_threshold)
            n_coldspots = np.sum((voxel_grid > 0) & (voxel_grid <= coldspot_threshold))
            
            # Exploration uniformity (lower std = more uniform)
            exploration_std = np.std(non_zero_counts)
            exploration_uniformity = 1.0 / (1.0 + exploration_std / (mean_visits + 1e-10))
        else:
            mean_visits = 0
            median_visits = 0
            max_visits = 0
            n_hotspots = 0
            n_coldspots = 0
            exploration_uniformity = 0.0
        
        # 2. Spatial distribution analysis
        # Check if data is concentrated in specific regions
        # Compute center of mass of visited regions
        visited_indices = np.where(voxel_grid > 0)
        if len(visited_indices[0]) > 0:
            # Get voxel centers
            x_centers = (edges[0][:-1] + edges[0][1:]) / 2
            y_centers = (edges[1][:-1] + edges[1][1:]) / 2
            z_centers = (edges[2][:-1] + edges[2][1:]) / 2
            
            # Weighted center of mass (weighted by visit count)
            weights = voxel_grid[visited_indices]
            com_x = np.average(x_centers[visited_indices[0]], weights=weights)
            com_y = np.average(y_centers[visited_indices[1]], weights=weights)
            com_z = np.average(z_centers[visited_indices[2]], weights=weights)
            
            # Spread metric (how spread out the exploration is)
            # Distance from center of mass
            distances_from_com = []
            for i, j, k in zip(visited_indices[0], visited_indices[1], visited_indices[2]):
                vox_x = x_centers[i]
                vox_y = y_centers[j]
                vox_z = z_centers[k]
                dist = np.sqrt((vox_x - com_x)**2 + (vox_y - com_y)**2 + (vox_z - com_z)**2)
                distances_from_com.append(dist * weights[len(distances_from_com)])
            
            mean_spread = np.mean(distances_from_com) if distances_from_com else 0
            workspace_diagonal = np.sqrt(
                (x_range[1] - x_range[0])**2 + 
                (y_range[1] - y_range[0])**2 + 
                (z_range[1] - z_range[0])**2
            )
            spread_score = min(mean_spread / (workspace_diagonal * 0.3), 1.0)  # Good if spread > 30% of diagonal
        else:
            com_x = com_y = com_z = 0
            mean_spread = 0
            spread_score = 0.0
        
        # 3. Quality score (improved for teleoperation)
        # Combines coverage, uniformity, and spread
        if coverage_percentage > 0:
            max_reasonable_coverage = 50.0  # 50% is excellent for teleoperation
            log_coverage = np.log1p(coverage_percentage)
            log_max = np.log1p(max_reasonable_coverage)
            coverage_component = min(log_coverage / log_max, 1.0)
            
            # Combined score: coverage (40%), uniformity (30%), spread (30%)
            coverage_score = (
                0.4 * coverage_component +
                0.3 * exploration_uniformity +
                0.3 * spread_score
            )
        else:
            coverage_score = 0.0

        # Compute reachable workspace coverage (improved metric)
        reachable_results = self.compute_reachable_coverage(xyz_positions)

        # Build result dictionary
        result = {
            # Bounding box coverage (conservative, original metric)
            'coverage_percentage': coverage_percentage,
            'bounding_box_coverage': coverage_percentage / 100.0,  # As fraction
            'n_voxels_visited': int(n_voxels_visited),
            'n_voxels_total': int(n_voxels_total),
            'voxel_grid': voxel_grid,
            'voxel_edges': edges,
            'workspace_bounds': workspace_bounds,
            'data_space': 'cartesian',
            'score': coverage_score,
            # Exploration metrics
            'exploration_metrics': {
                'mean_visits_per_voxel': float(mean_visits),
                'median_visits_per_voxel': float(median_visits),
                'max_visits_per_voxel': int(max_visits),
                'n_hotspots': int(n_hotspots),
                'n_coldspots': int(n_coldspots),
                'exploration_uniformity': float(exploration_uniformity),
                'center_of_mass': (float(com_x), float(com_y), float(com_z)),
                'mean_spread': float(mean_spread),
                'spread_score': float(spread_score)
            }
        }

        # Add reachable coverage metrics (primary metric)
        if reachable_results.get('reachable_coverage_available', False):
            result['reachable_coverage'] = reachable_results['reachable_coverage']
            result['reachable_coverage_percentage'] = reachable_results['reachable_coverage_percentage']
            result['coverage_quality'] = reachable_results['coverage_quality']
            result['n_visited_reachable_voxels'] = reachable_results['n_visited_reachable_voxels']
            result['n_reachable_voxels'] = reachable_results['n_reachable_voxels']

            # Update score to use reachable coverage as primary
            # Reachable coverage is more accurate, so weight it higher
            reachable_score = min(reachable_results['reachable_coverage'] / 0.5, 1.0)
            result['score'] = 0.6 * reachable_score + 0.2 * exploration_uniformity + 0.2 * spread_score
        else:
            result['reachable_coverage'] = None
            result['coverage_quality'] = get_coverage_quality(coverage_percentage / 100.0)

        return result

    def compute_starting_position_variance(
        self,
        starting_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Compute variance in starting positions.

        Args:
            starting_positions: Starting positions, shape (n_episodes, n_dof)

        Returns:
            Dictionary containing variance metrics
        """
        if len(starting_positions) < 2:
            return {
                'variance_per_dof': np.zeros(starting_positions.shape[1]),
                'total_variance': 0.0,
                'covariance_matrix': np.zeros((starting_positions.shape[1], starting_positions.shape[1])),
                'score': 0.0
            }

        # Compute variance per DOF
        variance_per_dof = np.var(starting_positions, axis=0)

        # Total variance (sum across DOFs)
        total_variance = np.sum(variance_per_dof)

        # Covariance matrix
        covariance_matrix = np.cov(starting_positions.T)

        # Compute principal components (for diversity measure)
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        eigenvalues = np.real(eigenvalues)  # Take real part
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

        # Diversity score based on variance
        # Higher variance = higher diversity = better
        # Normalize by expected variance (heuristic: 0.5 rad^2 is good)
        expected_variance = 0.5
        diversity_score = min(total_variance / expected_variance, 1.0)

        return {
            'variance_per_dof': variance_per_dof,
            'total_variance': total_variance,
            'mean_variance': np.mean(variance_per_dof),
            'covariance_matrix': covariance_matrix,
            'eigenvalues': eigenvalues,
            'score': diversity_score
        }

    def compute_approach_angle_diversity(
        self,
        trajectories: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Compute diversity in approach angles from trajectories.

        Args:
            trajectories: List of trajectory arrays, each shape (n_frames, n_dims)
                         Approach angle is computed from velocity vectors near the end

        Returns:
            Dictionary containing angle diversity metrics
        """
        if len(trajectories) < 2:
            return {
                'angular_entropy': 0.0,
                'angular_variance': 0.0,
                'score': 0.0
            }

        # Compute approach angles from trajectories
        # Use last 20% of trajectory to compute approach direction
        approach_vectors = []
        
        for traj in trajectories:
            if len(traj) < 5:
                continue
            
            # Get last portion of trajectory (last 20% or at least 5 frames)
            n_frames = len(traj)
            start_idx = max(0, n_frames - max(5, int(n_frames * 0.2)))
            approach_segment = traj[start_idx:]
            
            if len(approach_segment) < 2:
                continue
            
            # Compute average velocity vector in approach segment
            # This represents the approach direction
            if approach_segment.shape[1] >= 3:
                # Use 3D positions
                velocities = np.diff(approach_segment, axis=0)
                if len(velocities) > 0:
                    avg_velocity = np.mean(velocities, axis=0)
                    # Normalize to get direction
                    norm = np.linalg.norm(avg_velocity)
                    if norm > 1e-6:
                        approach_vectors.append(avg_velocity / norm)
            else:
                # For lower dimensions, use as is
                velocities = np.diff(approach_segment, axis=0)
                if len(velocities) > 0:
                    avg_velocity = np.mean(velocities, axis=0)
                    norm = np.linalg.norm(avg_velocity)
                    if norm > 1e-6:
                        approach_vectors.append(avg_velocity / norm)
        
        if len(approach_vectors) < 2:
            return {
                'angular_entropy': 0.0,
                'angular_variance': 0.0,
                'score': 0.0
            }
        
        approach_vectors = np.array(approach_vectors)
        
        # If 3D vectors, compute angles
        if approach_vectors.shape[1] == 3:
            # Compute angles from vectors
            # Azimuth (horizontal angle)
            azimuth = np.arctan2(approach_vectors[:, 1], approach_vectors[:, 0])

            # Elevation (vertical angle)
            r_xy = np.sqrt(approach_vectors[:, 0]**2 + approach_vectors[:, 1]**2)
            elevation = np.arctan2(approach_vectors[:, 2], r_xy)

            angles = np.column_stack([azimuth, elevation])
        else:
            angles = approach_vectors

        # Compute angular variance
        angular_variance = np.var(angles, axis=0)
        mean_angular_variance = np.mean(angular_variance)

        # Compute angular entropy (discretize into bins)
        n_bins = 20
        angle_bins = []

        for i in range(angles.shape[1]):
            hist, _ = np.histogram(angles[:, i], bins=n_bins)
            angle_bins.append(hist)

        # Compute entropy for each angle dimension
        entropies = []
        for hist in angle_bins:
            # Normalize to probability distribution
            prob = hist / (np.sum(hist) + 1e-10)
            ent = entropy(prob + 1e-10)  # Add small value to avoid log(0)
            entropies.append(ent)

        angular_entropy = np.mean(entropies)

        # Diversity score based on entropy
        # Max entropy for 20 bins is log(20) ≈ 3.0
        max_entropy = np.log(n_bins)
        diversity_score = min(angular_entropy / max_entropy, 1.0)

        return {
            'angular_entropy': angular_entropy,
            'angular_variance': angular_variance,
            'mean_angular_variance': mean_angular_variance,
            'score': diversity_score
        }

    def compute_position_distribution(
        self,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Compute spatial distribution metrics using simple distance-based clustering.

        Args:
            positions: Position array, shape (n_samples, n_dims)

        Returns:
            Dictionary containing distribution metrics
        """
        if len(positions) < 10:
            return {
                'spatial_entropy': 0.0,
                'cluster_count': 0,
                'score': 0.0
            }

        # Extract 3D positions if available
        pos_3d = positions[:, :3] if positions.shape[1] >= 3 else positions

        # Compute pairwise distances using numpy (avoids scipy.pdist multiprocessing issues)
        # For large datasets, sample to avoid O(n^2) memory issues
        n_samples = len(pos_3d)
        max_samples_for_dist = 1000  # Limit samples for distance computation
        
        if n_samples > max_samples_for_dist:
            # Sample positions for distance computation
            indices = np.linspace(0, n_samples - 1, max_samples_for_dist, dtype=int)
            pos_sample = pos_3d[indices]
        else:
            pos_sample = pos_3d
        
        # Compute pairwise distances using vectorized numpy (avoids scipy.pdist multiprocessing)
        # Use broadcasting: (n, 1, d) - (1, n, d) = (n, n, d) -> (n, n) distances
        n = len(pos_sample)
        if n > 1:
            # Compute all pairwise distances at once using broadcasting
            diff = pos_sample[:, np.newaxis, :] - pos_sample[np.newaxis, :, :]
            all_distances = np.linalg.norm(diff, axis=2)
            
            # Extract upper triangle (avoid duplicates and self-distances)
            triu_indices = np.triu_indices(n, k=1)
            distances = all_distances[triu_indices]
        else:
            distances = np.array([0.0])
        
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Simple greedy clustering using distance threshold
        # This avoids scipy.cluster.hierarchy which can cause multiprocessing issues in Streamlit
        def simple_distance_clustering(positions, threshold):
            """Simple greedy clustering based on distance threshold."""
            n = len(positions)
            if n == 0:
                return []
            if n == 1:
                return [0]
            
            clusters = np.full(n, -1, dtype=int)
            cluster_id = 0
            
            for i in range(n):
                if clusters[i] == -1:
                    # Start new cluster
                    clusters[i] = cluster_id
                    
                    # Find all points within threshold
                    for j in range(i + 1, n):
                        if clusters[j] == -1:
                            dist = np.linalg.norm(positions[i] - positions[j])
                            if dist <= threshold:
                                clusters[j] = cluster_id
                    
                    cluster_id += 1
            
            return clusters

        if len(positions) > 1:
            # Use mean distance as threshold for clustering
            clusters = simple_distance_clustering(pos_3d, mean_distance)
            n_clusters = len(np.unique(clusters))
        else:
            n_clusters = 1

        # Spatial entropy (based on position variance)
        position_variance = np.var(positions, axis=0)
        spatial_entropy = np.sum(np.log(position_variance + 1e-10))

        # Distribution score (more clusters = better distribution)
        # Normalize by expected number of clusters (heuristic: 5-10)
        expected_clusters = 7
        distribution_score = min(n_clusters / expected_clusters, 1.0)

        return {
            'spatial_entropy': spatial_entropy,
            'mean_pairwise_distance': mean_distance,
            'std_pairwise_distance': std_distance,
            'cluster_count': n_clusters,
            'score': distribution_score
        }

    def analyze_spatial_coverage(
        self,
        episodes_data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Analyze spatial coverage for multiple episodes.

        Args:
            episodes_data: List of episode dictionaries with 'states' key

        Returns:
            Dictionary containing spatial coverage analysis
        """
        # Collect all positions
        all_positions = []
        starting_positions = []

        for episode in episodes_data:
            states = episode['states']
            all_positions.append(states)
            starting_positions.append(states[0])

        all_positions = np.vstack(all_positions)
        starting_positions = np.array(starting_positions)

        # Workspace coverage
        logger.info("Computing workspace coverage...")
        coverage_results = self.compute_workspace_coverage(all_positions)

        # Starting position variance
        logger.info("Computing starting position variance...")
        variance_results = self.compute_starting_position_variance(starting_positions)

        # Position distribution
        logger.info("Computing position distribution...")
        distribution_results = self.compute_position_distribution(all_positions)

        # Approach angle diversity (compute from trajectories)
        logger.info("Computing approach angle diversity from trajectories...")
        trajectories = [ep['states'] for ep in episodes_data]
        angle_results = self.compute_approach_angle_diversity(trajectories)

        # Overall spatial coverage score (weighted average)
        weights = {
            'workspace_coverage': 0.40,
            'starting_variance': 0.30,
            'position_distribution': 0.20,
            'angle_diversity': 0.10
        }

        overall_score = (
            weights['workspace_coverage'] * coverage_results['score'] +
            weights['starting_variance'] * variance_results['score'] +
            weights['position_distribution'] * distribution_results['score'] +
            weights['angle_diversity'] * angle_results['score']
        )

        return {
            'workspace_coverage': coverage_results,
            'starting_variance': variance_results,
            'position_distribution': distribution_results,
            'angle_diversity': angle_results,
            'overall_score': overall_score,
            'individual_scores': {
                'workspace_coverage': coverage_results['score'],
                'starting_variance': variance_results['score'],
                'position_distribution': distribution_results['score'],
                'angle_diversity': angle_results['score']
            },
            'weights': weights,
            'n_episodes': len(episodes_data),
            'n_total_samples': len(all_positions)
        }


def analyze_spatial_coverage(
    episodes_data: List[Dict[str, np.ndarray]],
    robot_config: Optional[RobotConfig] = None,
    voxel_size: float = 0.05
) -> Dict[str, Any]:
    """Convenience function to analyze spatial coverage.

    Args:
        episodes_data: List of episode dictionaries
        robot_config: Optional robot configuration
        voxel_size: Voxel size for workspace grid

    Returns:
        Spatial coverage analysis results
    """
    analyzer = SpatialCoverageAnalyzer(
        robot_config=robot_config,
        voxel_size=voxel_size
    )

    return analyzer.analyze_spatial_coverage(episodes_data)
