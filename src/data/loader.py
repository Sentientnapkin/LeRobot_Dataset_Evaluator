from __future__ import annotations

"""LeRobot dataset loader with streaming support."""

import os
import functools
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm.auto import tqdm

from streamlit.delta_generator import DeltaGenerator
import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

from ..utils.logging_config import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class DatasetLoader:
    """Loader for LeRobot datasets from HuggingFace Hub."""

    def __init__(
        self,
        dataset_id: str,
        cache_dir: Optional[str] = None,
        streaming: bool = True
    ):
        """Initialize dataset loader.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., 'SentientNapkin/dataset_name')
            cache_dir: Directory for caching downloaded data
            streaming: Whether to use streaming mode (avoids full download)
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot package not found. Please install: pip install lerobot>=0.4.0"
            )

        self.dataset_id = dataset_id
        self.streaming = streaming

        # Get config
        config = get_config()
        self.cache_dir = cache_dir or str(config.cache_dir)

        # Dataset objects (lazy loaded)
        self._dataset: Optional[LeRobotDataset] = None
        self._metadata: Optional[LeRobotDatasetMetadata] = None

        logger.info(f"Initialized loader for dataset: {dataset_id}")

    def _get_episode_frame_indices(self, episode_index: int) -> np.ndarray:
        """Get frame indices for a specific episode (vectorized).

        Args:
            episode_index: Episode index

        Returns:
            Array of frame indices belonging to the episode
        """
        if self._dataset is None:
            self.load_dataset()

        episode_indices = np.array(self._dataset.hf_dataset['episode_index'])
        return np.where(episode_indices == episode_index)[0]

    def load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata without downloading full dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        logger.info(f"Loading metadata for {self.dataset_id}...")

        try:
            self._metadata = LeRobotDatasetMetadata(self.dataset_id)

            metadata = {
                'dataset_id': self.dataset_id,
                'total_episodes': self._metadata.total_episodes,
                'total_frames': self._metadata.total_frames,
                'fps': self._metadata.fps,
                'robot_type': getattr(self._metadata, 'robot_type', 'unknown'),
                'features': self._metadata.features if hasattr(self._metadata, 'features') else {},
                'camera_keys': self._metadata.camera_keys if hasattr(self._metadata, 'camera_keys') else [],
            }

            logger.info(f"Loaded metadata: {metadata['total_episodes']} episodes, "
                       f"{metadata['total_frames']} frames")

            return metadata

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def load_dataset(self, max_episodes: Optional[int] = None) -> LeRobotDataset:
        """Load LeRobot dataset.

        Args:
            max_episodes: Maximum number of episodes to load (None = all)

        Returns:
            LeRobotDataset object
        """
        if self._dataset is not None:
            return self._dataset

        logger.info(f"Loading dataset {self.dataset_id}...")

        try:
            # Load dataset with optional streaming
            if self.streaming:
                logger.info("Using streaming mode...")

            if max_episodes is not None:
                self._dataset = LeRobotDataset(
                    self.dataset_id,
                    root=self.cache_dir,
                    episodes=list(range(max_episodes)),
                    video_backend="pyav"
                )
            else:
                self._dataset = LeRobotDataset(
                    self.dataset_id,
                    root=self.cache_dir,
                    video_backend="pyav"
                )

            logger.info(f"Dataset loaded successfully: {len(self._dataset)} samples")

            return self._dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            index: Sample index

        Returns:
            Dictionary containing sample data
        """
        if self._dataset is None:
            self.load_dataset()

        return self._dataset[index]

    def get_episode_data(
        self,
        episode_index: int,
        include_images: bool = False,
        show_progress: bool = False
    ) -> Dict[str, Any]:
        """Get all data for a specific episode.

        Args:
            episode_index: Episode index
            include_images: Whether to include image observations
            show_progress: Whether to show progress bar

        Returns:
            Dictionary containing episode data
        """
        if self._dataset is None:
            self.load_dataset()

        # Validate episode index
        if episode_index >= self._dataset.num_episodes:
            raise ValueError(f"Episode index {episode_index} out of range (max: {self._dataset.num_episodes - 1})")

        # Filter HuggingFace dataset by episode_index (vectorized)
        hf_dataset = self._dataset.hf_dataset
        episode_frames = self._get_episode_frame_indices(episode_index).tolist()

        if not episode_frames:
            raise ValueError(f"No frames found for episode {episode_index}")

        start_idx = episode_frames[0]
        end_idx = episode_frames[-1] + 1

        # Extract episode data
        states = []
        actions = []
        timestamps = []
        images = {} if include_images else None

        logger.info(f"Loading episode {episode_index} "
                   f"({len(episode_frames)} frames)...")

        # Extract data from HuggingFace dataset (no video loading)
        frame_iterator = episode_frames
        if show_progress:
            frame_iterator = tqdm(
                episode_frames,
                desc=f"Loading episode {episode_index}",
                unit="frame"
            )

        for frame_idx in frame_iterator:
            row = hf_dataset[frame_idx]

            # Extract state
            if 'observation.state' in row:
                state_tensor = row['observation.state']
                states.append(state_tensor.numpy() if hasattr(state_tensor, 'numpy') else state_tensor)

            # Extract action
            if 'action' in row:
                action_tensor = row['action']
                actions.append(action_tensor.numpy() if hasattr(action_tensor, 'numpy') else action_tensor)

            # Extract timestamp
            if 'timestamp' in row:
                timestamps.append(row['timestamp'])

            # Note: Skipping images for now to avoid video loading issues
            # Images can be loaded separately if needed

        episode_data = {
            'episode_index': episode_index,
            'states': np.array(states) if states else None,
            'actions': np.array(actions) if actions else None,
            'timestamps': np.array(timestamps) if timestamps else None,
            'start_frame': start_idx,
            'end_frame': end_idx,
            'length': len(episode_frames),
        }

        if include_images and images:
            episode_data['images'] = {
                cam: np.array(imgs) for cam, imgs in images.items()
            }

        return episode_data

    def get_all_episodes_data(
        self,
        max_episodes: Optional[int] = None,
        include_images: bool = False,
        show_progress: bool = True,
        sampling_strategy: str = 'auto',
        progress_bar: DeltaGenerator = None,
    ) -> List[Dict[str, Any]]:
        """Get data for all episodes.

        Args:
            max_episodes: Maximum number of episodes to load
            include_images: Whether to include image observations
            show_progress: Whether to show progress bar
            sampling_strategy: How to select episodes ('auto', 'all', 'random', 'sequential')
                - 'auto': All if <500 episodes, random sample otherwise
                - 'all': Load all episodes (ignores max_episodes)
                - 'random': Random sample of max_episodes
                - 'sequential': Sequential from 0 to max_episodes-1

        Returns:
            List of episode data dictionaries
        """
        if self._dataset is None:
            self.load_dataset()

        total_episodes = self._dataset.num_episodes

        # Determine sampling strategy
        if sampling_strategy == 'auto':
            if total_episodes < 500:
                # Small dataset: analyze all episodes
                n_episodes = total_episodes
                episode_indices = list(range(total_episodes))
                logger.info(f"Auto mode: Loading all {n_episodes} episodes (dataset < 500)")
            else:
                # Large dataset: random sample
                n_episodes = min(max_episodes or 100, total_episodes)
                episode_indices = np.random.choice(total_episodes, n_episodes, replace=False).tolist()
                logger.info(f"Auto mode: Random sampling {n_episodes} of {total_episodes} episodes")
        elif sampling_strategy == 'all':
            # Load all episodes regardless of max_episodes
            n_episodes = total_episodes
            episode_indices = list(range(total_episodes))
            logger.info(f"Loading all {n_episodes} episodes")
        elif sampling_strategy == 'random':
            # Random sample up to max_episodes
            n_episodes = min(max_episodes or 100, total_episodes)
            episode_indices = np.random.choice(total_episodes, n_episodes, replace=False).tolist()
            logger.info(f"Random sampling {n_episodes} of {total_episodes} episodes")
        else:  # 'sequential'
            # Sequential from 0 to max_episodes-1
            n_episodes = min(max_episodes or total_episodes, total_episodes)
            episode_indices = list(range(n_episodes))
            logger.info(f"Sequential: Loading episodes 0-{n_episodes-1}")

        episodes = []
        iterator = episode_indices

        if show_progress:
            iterator = tqdm(iterator, desc="Loading episodes")

        for episode_idx in iterator:
            try:
                episode_data = self.get_episode_data(
                    episode_idx,
                    include_images=include_images
                )
                episodes.append(episode_data)
                progress_bar.progress(episode_idx/ (len(iterator) * 2), "Preprocessing episodes...")
            except Exception as e:
                logger.warning(f"Failed to load episode {episode_idx}: {e}")
                continue

        logger.info(f"Successfully loaded {len(episodes)} episodes")

        return episodes

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Compute basic statistics about the dataset.

        Returns:
            Dictionary containing dataset statistics
        """
        metadata = self.load_metadata()

        # Load a sample to get dimensionality info
        sample = self.get_sample(0)

        state_dim = sample['observation.state'].shape[0] if 'observation.state' in sample else None
        action_dim = sample['action'].shape[0] if 'action' in sample else None

        # Get camera information
        camera_keys = [k.replace('observation.images.', '')
                      for k in sample.keys()
                      if k.startswith('observation.images.')]

        stats = {
            'total_episodes': metadata['total_episodes'],
            'total_frames': metadata['total_frames'],
            'fps': metadata['fps'],
            'state_dimension': state_dim,
            'action_dimension': action_dim,
            'cameras': camera_keys,
            'avg_episode_length': metadata['total_frames'] / metadata['total_episodes'],
        }

        return stats

    @functools.lru_cache(maxsize=1)
    def get_camera_names(self) -> Tuple[str, ...]:
        """Get list of available camera names in the dataset.

        Returns:
            Tuple of camera name strings (cached)
        """
        if self._dataset is None:
            self.load_dataset()

        sample = self.get_sample(0)
        camera_keys = tuple(k.replace('observation.images.', '')
                           for k in sample.keys()
                           if k.startswith('observation.images.'))

        return camera_keys

    def get_frame_images(
        self,
        frame_index: int,
        camera_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Get images for a specific frame.

        Args:
            frame_index: Global frame index in dataset
            camera_names: List of camera names to load (None = all cameras)

        Returns:
            Dictionary mapping camera names to image arrays
        """
        if self._dataset is None:
            self.load_dataset()

        # Use LeRobotDataset indexing to get decoded video frames
        sample = self._dataset[frame_index]

        if camera_names is None:
            camera_names = self.get_camera_names()

        images = {}
        for cam in camera_names:
            key = f'observation.images.{cam}'
            if key in sample:
                img_data = sample[key]
                # Handle different image formats (typically torch tensors from video decoding)
                if hasattr(img_data, 'numpy'):
                    img_array = img_data.numpy()
                elif isinstance(img_data, np.ndarray):
                    img_array = img_data
                else:
                    # PIL Image or other format
                    img_array = np.array(img_data)

                # Ensure proper shape (H, W, C) for display
                if len(img_array.shape) == 3:
                    # Check if channels first (C, H, W) and convert to (H, W, C)
                    if img_array.shape[0] in [1, 3, 4] and img_array.shape[0] < img_array.shape[1]:
                        img_array = np.transpose(img_array, (1, 2, 0))

                images[cam] = img_array

        return images

    def get_episode_images(
        self,
        episode_index: int,
        camera_names: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
        show_progress: bool = False
    ) -> Dict[str, List[np.ndarray]]:
        """Get all images for an episode.

        Args:
            episode_index: Episode index
            camera_names: List of camera names to load (None = all cameras)
            max_frames: Maximum frames to load (None = all, useful for large episodes)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping camera names to lists of image arrays
        """
        if self._dataset is None:
            self.load_dataset()

        # Get frame indices for this episode (vectorized)
        episode_frames = self._get_episode_frame_indices(episode_index).tolist()

        if not episode_frames:
            raise ValueError(f"No frames found for episode {episode_index}")

        # Subsample if requested
        if max_frames is not None and len(episode_frames) > max_frames:
            step = len(episode_frames) // max_frames
            episode_frames = episode_frames[::step][:max_frames]

        if camera_names is None:
            camera_names = self.get_camera_names()

        # Initialize storage
        images = {cam: [] for cam in camera_names}

        iterator = episode_frames
        if show_progress:
            iterator = tqdm(iterator, desc=f"Loading images for episode {episode_index}")

        for frame_idx in iterator:
            frame_images = self.get_frame_images(frame_idx, camera_names)
            for cam, img in frame_images.items():
                images[cam].append(img)

        return images

    def get_episode_frame_indices(self, episode_index: int) -> List[int]:
        """Get the global frame indices for a specific episode.

        Args:
            episode_index: Episode index

        Returns:
            List of global frame indices
        """
        return self._get_episode_frame_indices(episode_index).tolist()

    def get_video_paths(self, episode_index: int) -> Dict[str, Path]:
        """Get paths to video files for a specific episode.

        Args:
            episode_index: Episode index

        Returns:
            Dictionary mapping camera names to video file paths
        """
        if self._dataset is None:
            self.load_dataset()

        video_paths = {}
        camera_names = self.get_camera_names()

        # LeRobot stores videos in: {root}/{repo_id}/videos/{camera_name}/episode_{index:06d}.mp4
        dataset_path = Path(self.cache_dir) / self.dataset_id

        for cam in camera_names:
            video_file = dataset_path / "videos" / cam / f"episode_{episode_index:06d}.mp4"
            if video_file.exists():
                video_paths[cam] = video_file
            else:
                # Try alternative naming conventions
                alt_file = dataset_path / "videos" / f"observation.images.{cam}" / f"episode_{episode_index:06d}.mp4"
                if alt_file.exists():
                    video_paths[cam] = alt_file

        return video_paths

    @property
    def num_episodes(self) -> int:
        """Get number of episodes in dataset."""
        if self._dataset is None:
            metadata = self.load_metadata()
            return metadata['total_episodes']

        return self._dataset.num_episodes

    @property
    def num_samples(self) -> int:
        """Get total number of samples in dataset."""
        if self._dataset is None:
            metadata = self.load_metadata()
            return metadata['total_frames']

        return len(self._dataset)


def load_dataset_from_hub(
    dataset_id: str,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
    max_episodes: Optional[int] = None
) -> Tuple[DatasetLoader, Dict[str, Any]]:
    """Convenience function to load a dataset and its metadata.

    Args:
        dataset_id: HuggingFace dataset ID
        cache_dir: Cache directory
        streaming: Use streaming mode
        max_episodes: Maximum episodes to load

    Returns:
        Tuple of (DatasetLoader, metadata dict)
    """
    loader = DatasetLoader(dataset_id, cache_dir=cache_dir, streaming=streaming)
    metadata = loader.load_metadata()
    loader.load_dataset(max_episodes=max_episodes)

    return loader, metadata
