"""
Core System Components

The core module provides the fundamental building blocks for the panel database system.
"""

from .models import (
    Panel, Episode, Database,
    Character, Emotion, DialogueEntry, Action, SceneContext, VisualProperties,
    Tag, TagCategory, TagSource,
    BoundingBox, AnalysisResult
)
from .analyzer_plugin import (
    AnalyzerPlugin,
    CharacterAnalyzerPlugin, EmotionAnalyzerPlugin, DialogueAnalyzerPlugin,
    ActionAnalyzerPlugin, SceneAnalyzerPlugin, VisualAnalyzerPlugin,
    PluginRegistry, register_plugin, get_registry
)
from .pipeline import Pipeline, PipelineBuilder, PipelineContext
from .feature_flags import FeatureFlags, get_feature_flags
from .logger import setup_logging, get_logger, LoggerMixin

__all__ = [
    # Models
    'Panel', 'Episode', 'Database',
    'Character', 'Emotion', 'DialogueEntry', 'Action', 'SceneContext', 'VisualProperties',
    'Tag', 'TagCategory', 'TagSource',
    'BoundingBox', 'AnalysisResult',

    # Plugins
    'AnalyzerPlugin',
    'CharacterAnalyzerPlugin', 'EmotionAnalyzerPlugin', 'DialogueAnalyzerPlugin',
    'ActionAnalyzerPlugin', 'SceneAnalyzerPlugin', 'VisualAnalyzerPlugin',
    'PluginRegistry', 'register_plugin', 'get_registry',

    # Pipeline
    'Pipeline', 'PipelineBuilder', 'PipelineContext',

    # Feature Flags
    'FeatureFlags', 'get_feature_flags',

    # Logging
    'setup_logging', 'get_logger', 'LoggerMixin',
]
