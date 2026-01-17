"""Test to understand episode structure without loading videos."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    print("✓ Using lerobot.datasets path")
except ImportError:
    print("✗ LeRobot not installed")
    sys.exit(1)

# Load dataset
print("\nLoading lerobot/pusht...")
dataset = LeRobotDataset("lerobot/pusht")

print(f"✓ Dataset loaded")
print(f"  Total frames: {len(dataset)}")
print(f"  Total episodes: {dataset.num_episodes}")

# Check hf_dataset structure
print(f"\n  HF Dataset info:")
print(f"    - Type: {type(dataset.hf_dataset)}")
print(f"    - Columns: {dataset.hf_dataset.column_names}")
print(f"    - Length: {len(dataset.hf_dataset)}")

# Check if there's episode information in hf_dataset
print(f"\n  First row sample (no images):")
first_row = dataset.hf_dataset[0]
for key, value in first_row.items():
    if 'image' not in key.lower() and 'video' not in key.lower():
        print(f"    - {key}: {value}")

# Check if there's an episode_index column
if 'episode_index' in dataset.hf_dataset.column_names:
    print(f"\n  Episode indices found!")
    # Get unique episode indices
    import numpy as np
    episode_indices = dataset.hf_dataset['episode_index']
    unique_episodes = np.unique(episode_indices)
    print(f"    - Unique episodes: {len(unique_episodes)}")
    print(f"    - Episode range: {min(episode_indices)} to {max(episode_indices)}")

    # Check how to filter by episode
    print(f"\n  Filtering episode 0:")
    episode_0_mask = [idx == 0 for idx in episode_indices]
    episode_0_length = sum(episode_0_mask)
    print(f"    - Episode 0 has {episode_0_length} frames")
