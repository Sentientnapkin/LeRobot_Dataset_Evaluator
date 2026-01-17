"""Configuration management for LeRobot Dataset Evaluator."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager for loading and accessing settings."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, looks in project root.
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., 'dataset.hf_account')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> config.get('dataset.hf_account')
            'SentientNapkin'
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration section."""
        return self._config.get('dataset', {})

    def get_robot_config(self) -> Dict[str, Any]:
        """Get robot configuration section."""
        return self._config.get('robot', {})

    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics configuration section."""
        return self._config.get('metrics', {})

    def get_quality_scoring_config(self) -> Dict[str, Any]:
        """Get quality scoring configuration section."""
        return self._config.get('quality_scoring', {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self._config.get('visualization', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration section."""
        return self._config.get('performance', {})

    def update(self, key_path: str, value: Any):
        """Update configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file.

        Args:
            path: Path to save config. If None, saves to original path.
        """
        save_path = Path(path) if path else self.config_path

        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path, creating if it doesn't exist."""
        cache_dir = Path(self.get('dataset.cache_dir', './cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def hf_account(self) -> str:
        """Get HuggingFace account name."""
        return self.get('dataset.hf_account', '')

    @property
    def robot_dof(self) -> int:
        """Get robot degrees of freedom."""
        return self.get('robot.dof', 6)

    @property
    def robot_type(self) -> str:
        """Get robot type."""
        return self.get('robot.type', 'generic_6dof')


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to config file. Only used on first call.

    Returns:
        Global Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


def reset_config():
    """Reset global configuration instance. Useful for testing."""
    global _config_instance
    _config_instance = None
