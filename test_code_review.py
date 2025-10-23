"""
Comprehensive Code Review Test Suite

Tests all modules for common issues.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all modules can be imported."""
    print("\n🔍 Testing Imports...")

    issues = []

    try:
        import core.models
        print("  ✅ core.models")
    except Exception as e:
        issues.append(f"core.models: {e}")
        print(f"  ❌ core.models: {e}")

    try:
        import core.analyzer_plugin
        print("  ✅ core.analyzer_plugin")
    except Exception as e:
        issues.append(f"core.analyzer_plugin: {e}")
        print(f"  ❌ core.analyzer_plugin: {e}")

    try:
        import core.pipeline
        print("  ✅ core.pipeline")
    except Exception as e:
        issues.append(f"core.pipeline: {e}")
        print(f"  ❌ core.pipeline: {e}")

    try:
        import core.feature_flags
        print("  ✅ core.feature_flags")
    except Exception as e:
        issues.append(f"core.feature_flags: {e}")
        print(f"  ❌ core.feature_flags: {e}")

    try:
        import analyzers
        print("  ✅ analyzers")
    except Exception as e:
        issues.append(f"analyzers: {e}")
        print(f"  ❌ analyzers: {e}")

    return issues


def test_model_creation():
    """Test creating model instances."""
    print("\n🔍 Testing Model Creation...")

    issues = []

    try:
        from core.models import (
            Panel, Character, Emotion, DialogueEntry, Action,
            SceneContext, VisualProperties, Tag, TagCategory, TagSource,
            BoundingBox, AnalysisResult
        )

        # Test BoundingBox
        bbox = BoundingBox(x=10, y=20, width=100, height=200)
        assert bbox.area == 20000
        print("  ✅ BoundingBox")

        # Test Character
        char = Character(name="Test", confidence=0.9, bbox=bbox)
        char_dict = char.to_dict()
        assert char_dict['name'] == "Test"
        print("  ✅ Character")

        # Test Tag with enums
        tag = Tag(
            category=TagCategory.CHARACTER,
            value="test",
            confidence=0.9,
            source=TagSource.AI
        )
        tag_dict = tag.to_dict()
        assert tag_dict['category'] == 'character'
        print("  ✅ Tag")

        # Test AnalysisResult
        analysis = AnalysisResult(
            characters=[char],
            overall_confidence=0.8
        )
        analysis_dict = analysis.to_dict()
        assert len(analysis_dict['characters']) == 1
        print("  ✅ AnalysisResult")

    except Exception as e:
        issues.append(f"Model creation: {e}")
        print(f"  ❌ Model creation: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_pipeline_creation():
    """Test creating pipeline."""
    print("\n🔍 Testing Pipeline Creation...")

    issues = []

    try:
        from core.pipeline import PipelineBuilder
        from core.analyzer_plugin import AnalyzerPlugin
        from core.models import AnalysisResult
        import numpy as np

        # Create mock analyzer
        class MockAnalyzer(AnalyzerPlugin):
            @property
            def name(self) -> str:
                return "mock"

            @property
            def version(self) -> str:
                return "1.0.0"

            def analyze(self, image, panel_path) -> AnalysisResult:
                return AnalysisResult()

        # Build pipeline
        pipeline = (PipelineBuilder()
            .configure(max_workers=1)
            .build()
        )

        print("  ✅ Pipeline creation")

    except Exception as e:
        issues.append(f"Pipeline creation: {e}")
        print(f"  ❌ Pipeline creation: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_feature_flags():
    """Test feature flag system."""
    print("\n🔍 Testing Feature Flags...")

    issues = []

    try:
        from core.feature_flags import FeatureFlags

        flags = FeatureFlags()

        # Test enable/disable
        flags.enable('test_feature', {'param': 'value'})
        assert flags.is_enabled('test_feature')

        flags.disable('test_feature')
        assert not flags.is_enabled('test_feature')

        print("  ✅ Feature flags")

    except Exception as e:
        issues.append(f"Feature flags: {e}")
        print(f"  ❌ Feature flags: {e}")
        import traceback
        traceback.print_exc()

    return issues


def test_tag_enum_consistency():
    """Test that Tag creation uses enums consistently."""
    print("\n🔍 Testing Tag Enum Consistency...")

    issues = []

    try:
        from core.models import Tag, TagCategory, TagSource

        # Test with string (should work but is not type-safe)
        tag1 = Tag(
            category="character",  # String instead of enum
            value="test",
            confidence=0.9,
            source=TagSource.AI
        )

        # This works in Python but is not ideal
        if not isinstance(tag1.category, TagCategory):
            issues.append("Tag accepts strings for category (not type-safe)")
            print("  ⚠️  Tag accepts strings for category (not type-safe)")

        # Test with enum (correct way)
        tag2 = Tag(
            category=TagCategory.CHARACTER,
            value="test",
            confidence=0.9,
            source=TagSource.AI
        )

        if isinstance(tag2.category, TagCategory):
            print("  ✅ Tag works with enums")

    except Exception as e:
        issues.append(f"Tag enum: {e}")
        print(f"  ❌ Tag enum: {e}")

    return issues


if __name__ == "__main__":
    print("=" * 60)
    print("  Code Review Test Suite")
    print("=" * 60)

    all_issues = []

    all_issues.extend(test_imports())
    all_issues.extend(test_model_creation())
    all_issues.extend(test_pipeline_creation())
    all_issues.extend(test_feature_flags())
    all_issues.extend(test_tag_enum_consistency())

    print("\n" + "=" * 60)
    if all_issues:
        print(f"❌ Found {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✅ All tests passed!")
    print("=" * 60)
