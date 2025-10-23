#!/usr/bin/env python
"""
Comprehensive Panel Tagging System

This script processes all panels with advanced ML tagging including:
- Characters (face recognition)
- Dialogue (OCR)
- Emotions (facial analysis)
- Actions (scene understanding)
- Context (scene classification)
- Visual analysis (colors, mood)

Output: Enhanced panels_database.json with comprehensive tags
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.advanced_panel_analyzer import AdvancedPanelAnalyzer
from scraper.core.paths import get_path_manager
from scraper.core.logger import setup_logging, get_logger

setup_logging(Path("logs/comprehensive_tagging.log"))
logger = get_logger(__name__)


def load_existing_database():
    """Load existing panel database."""
    paths = get_path_manager()
    db_path = paths.get('frontend.data') / 'panels_database.json'

    if db_path.exists():
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Create empty database structure
        return {
            'total_panels': 0,
            'total_episodes': 0,
            'panels': []
        }


def tag_all_panels(skip_existing=False):
    """
    Tag all panels with comprehensive ML analysis.

    Args:
        skip_existing: If True, skip panels that already have AI tags
    """
    print("\n" + "=" * 60)
    print("  Comprehensive Panel Tagging System")
    print("=" * 60)
    print()

    # Initialize analyzer
    print("🤖 Initializing ML analyzer...")
    analyzer = AdvancedPanelAnalyzer()

    # Load database
    print("📊 Loading panel database...")
    database = load_existing_database()
    panels = database.get('panels', [])

    if not panels:
        print("❌ No panels found in database!")
        print("   Run the extraction pipeline first.")
        return

    print(f"✅ Found {len(panels)} panels")
    print()

    # Tag each panel
    tagged_count = 0
    skipped_count = 0
    failed_count = 0

    for panel in tqdm(panels, desc="Tagging panels"):
        # Check if already tagged
        if skip_existing and panel.get('ai_tags'):
            skipped_count += 1
            continue

        # Get panel path
        panel_path = Path(panel.get('path', ''))

        # Try frontend images path if original path doesn't exist
        if not panel_path.exists():
            filename = panel.get('filename', '')
            episode = panel.get('episode', '')
            alt_path = Path(f"/home/user/PanelDatabase/frontend/images/s2/{episode}/{filename}")

            if alt_path.exists():
                panel_path = alt_path
            else:
                logger.warning(f"Panel file not found: {panel_path}")
                failed_count += 1
                continue

        try:
            # Analyze panel
            analysis = analyzer.analyze_panel(panel_path)

            # Store AI tags separately from user tags
            panel['ai_tags'] = {
                'characters': analysis['characters'],
                'emotions': analysis['emotions'],
                'dialogue': analysis['dialogue'],
                'actions': analysis['actions'],
                'scene': analysis['scene'],
                'visual': analysis['visual'],
                'tags': analysis['tags'],
                'confidence': analysis['overall_confidence'],
                'version': '2.0',  # Tag version for tracking
                'method': 'ml_comprehensive'
            }

            # Keep user tags separate (if they exist)
            if 'user_tags' not in panel:
                panel['user_tags'] = {
                    'characters': [],
                    'emotions': [],
                    'custom_tags': [],
                    'notes': ''
                }

            tagged_count += 1

        except Exception as e:
            logger.error(f"Failed to tag {panel_path}: {e}")
            failed_count += 1

    # Update database
    database['total_panels'] = len(panels)
    database['tagged_panels'] = tagged_count
    database['tagging_version'] = '2.0'
    database['tagging_method'] = 'ml_comprehensive'

    # Save updated database
    paths = get_path_manager()
    db_path = paths.get('frontend.data') / 'panels_database.json'
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("✅ Tagging Complete!")
    print("=" * 60)
    print(f"✅ Tagged: {tagged_count} panels")
    print(f"⏭️  Skipped: {skipped_count} panels")
    print(f"❌ Failed: {failed_count} panels")
    print()
    print(f"💾 Database saved to: {db_path}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tag all panels with comprehensive ML analysis')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip panels that already have AI tags'
    )

    args = parser.parse_args()

    tag_all_panels(skip_existing=args.skip_existing)
