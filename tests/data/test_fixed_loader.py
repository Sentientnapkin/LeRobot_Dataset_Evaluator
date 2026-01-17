"""Test the fixed data loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DatasetLoader
import numpy as np

print("Testing fixed DatasetLoader...\n")

# Initialize loader
print("1. Initializing loader...")
loader = DatasetLoader(
    dataset_id="lerobot/pusht",
    cache_dir="./cache",
    streaming=False
)
print("   ✓ Loader initialized")

# Load dataset
print("\n2. Loading dataset...")
dataset = loader.load_dataset()
print(f"   ✓ Dataset loaded: {len(dataset)} frames")

# Test num_episodes property
print("\n3. Testing num_episodes property...")
num_episodes = loader.num_episodes
print(f"   ✓ Number of episodes: {num_episodes}")

# Test get_episode_data
print("\n4. Testing get_episode_data...")
episode_0 = loader.get_episode_data(episode_index=0, include_images=False)
print(f"   ✓ Episode 0 loaded:")
print(f"     - States shape: {episode_0['states'].shape}")
print(f"     - Actions shape: {episode_0['actions'].shape}")
print(f"     - Timestamps shape: {episode_0['timestamps'].shape}")
print(f"     - Length: {episode_0['length']} frames")

# Test get_all_episodes_data (just 3 episodes)
print("\n5. Testing get_all_episodes_data (3 episodes)...")
episodes = loader.get_all_episodes_data(
    max_episodes=3,
    include_images=False,
    show_progress=False
)
print(f"   ✓ Loaded {len(episodes)} episodes")
for i, ep in enumerate(episodes):
    print(f"     - Episode {i}: {ep['length']} frames")

print("\n✅ All tests passed!")
