"""
Analyzer Plugin System

Enables easy addition of new ML analyzers without modifying core code.
Each analyzer is a plugin that implements the AnalyzerPlugin interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np

from .models import AnalysisResult, Character, Emotion, DialogueEntry, Action, SceneContext, VisualProperties


class AnalyzerPlugin(ABC):
    """Base class for all analyzer plugins."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 100)

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """Python package dependencies."""
        return []

    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed."""
        import importlib
        for dep in self.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                return False
        return True

    @abstractmethod
    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        """
        Analyze a panel image.

        Args:
            image: Panel image as numpy array (BGR format)
            panel_path: Path to panel file

        Returns:
            AnalysisResult with detected information
        """
        pass

    def can_run(self) -> bool:
        """Check if plugin can run (enabled + dependencies met)."""
        return self.enabled and self.check_dependencies()


class CharacterAnalyzerPlugin(AnalyzerPlugin):
    """Base class for character detection plugins."""

    @abstractmethod
    def detect_characters(self, image: np.ndarray) -> List[Character]:
        """Detect characters in image."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        characters = self.detect_characters(image)
        return AnalysisResult(characters=characters)


class EmotionAnalyzerPlugin(AnalyzerPlugin):
    """Base class for emotion detection plugins."""

    @abstractmethod
    def detect_emotions(self, image: np.ndarray, characters: List[Character]) -> List[Emotion]:
        """Detect emotions for characters."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        # Note: Emotion detection usually needs characters first
        # This is handled by the pipeline orchestrator
        return AnalysisResult()


class DialogueAnalyzerPlugin(AnalyzerPlugin):
    """Base class for dialogue/text detection plugins."""

    @abstractmethod
    def detect_dialogue(self, image: np.ndarray) -> List[DialogueEntry]:
        """Detect text/dialogue in image."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        dialogue = self.detect_dialogue(image)
        return AnalysisResult(dialogue=dialogue)


class ActionAnalyzerPlugin(AnalyzerPlugin):
    """Base class for action detection plugins."""

    @abstractmethod
    def detect_actions(self, image: np.ndarray, characters: List[Character]) -> List[Action]:
        """Detect actions in panel."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        return AnalysisResult()


class SceneAnalyzerPlugin(AnalyzerPlugin):
    """Base class for scene classification plugins."""

    @abstractmethod
    def analyze_scene(self, image: np.ndarray) -> SceneContext:
        """Analyze scene context."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        scene = self.analyze_scene(image)
        return AnalysisResult(scene=scene)


class VisualAnalyzerPlugin(AnalyzerPlugin):
    """Base class for visual analysis plugins."""

    @abstractmethod
    def analyze_visuals(self, image: np.ndarray) -> VisualProperties:
        """Analyze visual properties."""
        pass

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        visual = self.analyze_visuals(image)
        return AnalysisResult(visual=visual)


class PluginRegistry:
    """Registry for managing analyzer plugins."""

    def __init__(self):
        self._plugins: Dict[str, AnalyzerPlugin] = {}

    def register(self, plugin: AnalyzerPlugin) -> None:
        """Register a plugin."""
        if not plugin.can_run():
            print(f"Warning: Plugin '{plugin.name}' cannot run (disabled or missing dependencies)")
            return

        self._plugins[plugin.name] = plugin
        print(f"Registered plugin: {plugin.name} v{plugin.version}")

    def get(self, name: str) -> Optional[AnalyzerPlugin]:
        """Get plugin by name."""
        return self._plugins.get(name)

    def get_all(self) -> List[AnalyzerPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def get_by_type(self, plugin_type: type) -> List[AnalyzerPlugin]:
        """Get all plugins of a specific type."""
        return [p for p in self._plugins.values() if isinstance(p, plugin_type)]

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())


# Global registry instance
_registry = PluginRegistry()


def register_plugin(plugin: AnalyzerPlugin) -> None:
    """Register a plugin with the global registry."""
    _registry.register(plugin)


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _registry
