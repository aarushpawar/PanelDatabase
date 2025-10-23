#!/usr/bin/env python
"""
Validate ML tagging results.

Analyzes the panel database to check:
- Tagging coverage (% of panels tagged)
- Character distribution
- Confidence score distribution
- Data quality metrics

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --database path/to/panels_database.json
"""

import sys
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core.paths import get_path_manager


def load_database(db_path: Path) -> Dict[str, Any]:
    """Load panel database from JSON file."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with open(db_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_database(database: Dict[str, Any]) -> None:
    """
    Validate and analyze the panel database.

    Args:
        database: Loaded database dictionary
    """
    panels = database.get('panels', [])
    total_panels = len(panels)

    if total_panels == 0:
        print("\n❌ ERROR: No panels found in database!")
        return

    print("\n" + "=" * 60)
    print("  Panel Database Validation Report")
    print("=" * 60)

    # Basic statistics
    print("\n📊 Basic Statistics:")
    print(f"    Total panels: {total_panels}")
    print(f"    Total episodes: {database.get('total_episodes', 'N/A')}")

    if database.get('failed_episodes'):
        failed = database['failed_episodes']
        print(f"    Failed episodes: {len(failed)} ({', '.join(failed)})")

    # Check tagging coverage
    print("\n🏷️  Tagging Coverage:")

    tagged_panels = [p for p in panels if p.get('tags', {}).get('characters')]
    tagging_rate = (len(tagged_panels) / total_panels) * 100

    print(f"    Panels with characters: {len(tagged_panels)} ({tagging_rate:.1f}%)")

    panels_with_emotions = [
        p for p in panels
        if p.get('tags', {}).get('emotions')
    ]
    emotion_rate = (len(panels_with_emotions) / total_panels) * 100
    print(f"    Panels with emotions: {len(panels_with_emotions)} ({emotion_rate:.1f}%)")

    # Character distribution
    print("\n👥 Character Distribution:")

    all_characters = []
    for panel in tagged_panels:
        chars = panel.get('tags', {}).get('characters', [])
        all_characters.extend([c['name'] for c in chars])

    if all_characters:
        char_counts = Counter(all_characters)
        print(f"    Unique characters: {len(char_counts)}")
        print("\n    Top 10 characters:")
        for char, count in char_counts.most_common(10):
            percentage = (count / len(tagged_panels)) * 100
            print(f"        {char}: {count} panels ({percentage:.1f}%)")
    else:
        print("    No characters detected")

    # Confidence distribution
    print("\n📈 Confidence Scores:")

    confidences = [
        p.get('tags', {}).get('overall_confidence', 0)
        for p in panels
        if p.get('tags')
    ]

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        print(f"    Average: {avg_conf:.3f}")
        print(f"    Min: {min_conf:.3f}")
        print(f"    Max: {max_conf:.3f}")

        # Distribution buckets
        high_conf = [c for c in confidences if c >= 0.8]
        medium_conf = [c for c in confidences if 0.5 <= c < 0.8]
        low_conf = [c for c in confidences if c < 0.5]

        print(f"\n    Distribution:")
        print(f"        High (≥0.8): {len(high_conf)} ({len(high_conf)/len(confidences)*100:.1f}%)")
        print(f"        Medium (0.5-0.8): {len(medium_conf)} ({len(medium_conf)/len(confidences)*100:.1f}%)")
        print(f"        Low (<0.5): {len(low_conf)} ({len(low_conf)/len(confidences)*100:.1f}%)")
    else:
        print("    No confidence scores available")

    # Emotion distribution
    print("\n😊 Emotion Distribution:")

    all_emotions = []
    for panel in panels_with_emotions:
        emotions = panel.get('tags', {}).get('emotions', {})
        for char, emotion_data in emotions.items():
            all_emotions.append(emotion_data.get('emotion', 'unknown'))

    if all_emotions:
        emotion_counts = Counter(all_emotions)
        print(f"    Total emotion detections: {len(all_emotions)}")
        print("\n    Top emotions:")
        for emotion, count in emotion_counts.most_common(5):
            percentage = (count / len(all_emotions)) * 100
            print(f"        {emotion}: {count} ({percentage:.1f}%)")
    else:
        print("    No emotions detected")

    # Panel dimensions
    print("\n📏 Panel Dimensions:")

    heights = [p.get('dimensions', {}).get('height', 0) for p in panels]
    widths = [p.get('dimensions', {}).get('width', 0) for p in panels]

    if heights and widths:
        avg_height = sum(heights) / len(heights)
        avg_width = sum(widths) / len(widths)

        print(f"    Average size: {avg_width:.0f} x {avg_height:.0f}px")
        print(f"    Height range: {min(heights)} - {max(heights)}px")
        print(f"    Width range: {min(widths)} - {max(widths)}px")

    # Quality issues
    print("\n⚠️  Quality Checks:")

    # Find low confidence panels
    low_conf_panels = [
        p for p in panels
        if p.get('tags', {}).get('overall_confidence', 1.0) < 0.5
    ]

    if low_conf_panels:
        print(f"    Low confidence panels: {len(low_conf_panels)}")
        print(f"        (Review these for accuracy)")

        # Sample a few
        sample_size = min(5, len(low_conf_panels))
        print(f"\n    Sample low confidence panels:")
        for panel in low_conf_panels[:sample_size]:
            filename = panel.get('filename', 'unknown')
            conf = panel.get('tags', {}).get('overall_confidence', 0)
            print(f"        {filename} (confidence: {conf:.3f})")

    # Find panels with no characters
    no_char_panels = [
        p for p in panels
        if not p.get('tags', {}).get('characters')
    ]

    if no_char_panels:
        percentage = (len(no_char_panels) / total_panels) * 100
        print(f"\n    Panels with no characters: {len(no_char_panels)} ({percentage:.1f}%)")

    # Detect potential issues
    issues = []

    if tagging_rate < 50:
        issues.append("Low tagging rate - check character database")

    if avg_conf < 0.5 if confidences else False:
        issues.append("Low average confidence - review detection settings")

    if len(low_conf_panels) > total_panels * 0.3:
        issues.append("Many low confidence panels - consider reprocessing")

    if issues:
        print(f"\n🔍 Potential Issues:")
        for issue in issues:
            print(f"    • {issue}")

    # Overall assessment
    print("\n" + "=" * 60)
    print("  Overall Assessment")
    print("=" * 60)

    if tagging_rate >= 70 and (avg_conf >= 0.7 if confidences else True):
        print("\n✅ GOOD - Database quality looks good!")
    elif tagging_rate >= 50 and (avg_conf >= 0.5 if confidences else True):
        print("\n⚠️  MODERATE - Some improvements possible")
    else:
        print("\n❌ NEEDS WORK - Significant quality issues detected")

    print("\n💡 Recommendations:")

    if tagging_rate < 70:
        print("    1. Build character database with more face samples")
        print("       Run: python scripts/build_character_database.py")

    if confidences and avg_conf < 0.7:
        print("    2. Adjust detection confidence thresholds in config")

    if len(no_char_panels) > total_panels * 0.5:
        print("    3. Check if character images are clear and frontal")

    print("\n📁 Database location: " + str(get_path_manager().get('frontend.data') / 'panels_database.json'))
    print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate panel database')
    parser.add_argument(
        '--database',
        type=Path,
        help='Path to panels_database.json (default: frontend/data/panels_database.json)'
    )

    args = parser.parse_args()

    # Determine database path
    if args.database:
        db_path = args.database
    else:
        paths = get_path_manager()
        db_path = paths.get('frontend.data') / 'panels_database.json'

    # Load and validate
    try:
        database = load_database(db_path)
        validate_database(database)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nRun the pipeline first:")
        print("    python scraper/pipeline.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
