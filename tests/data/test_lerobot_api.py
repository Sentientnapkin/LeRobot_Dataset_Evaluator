"""Test to understand the actual LeRobot API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    print("✓ Using lerobot.common.datasets path")
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print("✓ Using lerobot.datasets path")
    except ImportError:
        print("✗ LeRobot not installed")
        sys.exit(1)

# Try loading a public dataset
print("\nTesting with lerobot/pusht...")

try:
    dataset = LeRobotDataset("lerobot/pusht")
    print(f"✓ Dataset loaded")
    print(f"  Type: {type(dataset)}")
    print(f"  Length: {len(dataset)}")

    # Check available attributes
    print(f"\n  Available attributes:")
    for attr in dir(dataset):
        if not attr.startswith('_'):
            print(f"    - {attr}")

    # Check specific attributes we need
    print(f"\n  Key properties:")
    if hasattr(dataset, 'num_episodes'):
        print(f"    - num_episodes: {dataset.num_episodes}")
    if hasattr(dataset, 'episode_data_index'):
        print(f"    - episode_data_index: {type(dataset.episode_data_index)}")
    if hasattr(dataset, 'episodes'):
        print(f"    - episodes: {type(dataset.episodes)}")
    if hasattr(dataset, 'hf_dataset'):
        print(f"    - hf_dataset: {type(dataset.hf_dataset)}")

    # Try to get a sample
    print(f"\n  Sample data:")
    sample = dataset[0]
    print(f"    - Type: {type(sample)}")
    print(f"    - Keys: {list(sample.keys())}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
