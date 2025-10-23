"""
Feature Flag System

Enables/disables features without code changes.
Useful for experimentation and gradual rollouts.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from enum import Enum


class FeatureState(Enum):
    """Feature state."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"


class FeatureFlags:
    """Manages feature flags."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/features.json")
        self.flags: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load feature flags from config file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.flags = json.load(f)
        else:
            # Default flags
            self.flags = {
                'face_recognition': {
                    'state': 'enabled',
                    'priority': 100,
                    'config': {'tolerance': 0.6, 'model': 'hog'}
                },
                'emotion_detection': {
                    'state': 'enabled',
                    'priority': 90,
                    'config': {'confidence_threshold': 0.5}
                },
                'ocr_dialogue': {
                    'state': 'enabled',
                    'priority': 80,
                    'config': {'languages': 'eng+kor'}
                },
                'visual_analysis': {
                    'state': 'enabled',
                    'priority': 70,
                    'config': {}
                },
                'scene_classification': {
                    'state': 'enabled',
                    'priority': 60,
                    'config': {}
                },
                'action_detection': {
                    'state': 'experimental',
                    'priority': 50,
                    'config': {}
                },
                'parallel_processing': {
                    'state': 'enabled',
                    'config': {'max_workers': 4}
                },
                'caching': {
                    'state': 'enabled',
                    'config': {'cache_size': 1000}
                }
            }
            self._save()

    def _save(self) -> None:
        """Save feature flags to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.flags, f, indent=2)

    def is_enabled(self, feature: str) -> bool:
        """Check if feature is enabled."""
        if feature not in self.flags:
            return False

        state = self.flags[feature].get('state', 'disabled')
        return state in ['enabled', 'experimental']

    def is_experimental(self, feature: str) -> bool:
        """Check if feature is experimental."""
        if feature not in self.flags:
            return False

        return self.flags[feature].get('state') == 'experimental'

    def get_config(self, feature: str) -> Dict[str, Any]:
        """Get configuration for a feature."""
        if feature not in self.flags:
            return {}

        return self.flags[feature].get('config', {})

    def enable(self, feature: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Enable a feature."""
        if feature not in self.flags:
            self.flags[feature] = {}

        self.flags[feature]['state'] = 'enabled'
        if config:
            self.flags[feature]['config'] = config
        self._save()

    def disable(self, feature: str) -> None:
        """Disable a feature."""
        if feature in self.flags:
            self.flags[feature]['state'] = 'disabled'
            self._save()

    def set_experimental(self, feature: str) -> None:
        """Mark feature as experimental."""
        if feature in self.flags:
            self.flags[feature]['state'] = 'experimental'
            self._save()

    def list_features(self) -> Dict[str, str]:
        """List all features and their states."""
        return {
            feature: data.get('state', 'unknown')
            for feature, data in self.flags.items()
        }


# Global instance
_feature_flags = None


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags
