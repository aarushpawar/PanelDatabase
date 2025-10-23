#!/usr/bin/env python
"""
Panel Database CLI

Unified command-line interface for all operations.
Makes it easy to run different processing pipelines.
"""

import sys
import click
from pathlib import Path
from typing import Optional

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import PipelineBuilder
from core.feature_flags import get_feature_flags
from core.models import Database, Episode, Panel
from core.data_loader import load_existing_database, save_to_frontend_format
from analyzers import (
    FaceRecognitionAnalyzer,
    DeepFaceEmotionAnalyzer,
    TesseractOCRAnalyzer,
    ColorAnalyzer,
    BasicSceneAnalyzer
)


@click.group()
@click.version_option(version='2.0.0')
def cli():
    """Panel Database CLI - Process and analyze webtoon panels."""
    pass


@cli.command()
@click.option('--episode', '-e', help='Process specific episode (e.g., ep001)')
@click.option('--parallel/--serial', default=True, help='Process panels in parallel')
@click.option('--workers', '-w', default=4, help='Number of parallel workers')
@click.option('--output', '-o', default='frontend/data/panels_database.json', help='Output database path')
def process(episode: Optional[str], parallel: bool, workers: int, output: str):
    """Process panels through ML analysis pipeline."""
    click.echo("🤖 Panel Processing Pipeline")
    click.echo("=" * 60)

    # Get feature flags
    flags = get_feature_flags()

    # Build pipeline
    click.echo("\n📦 Building pipeline...")
    builder = PipelineBuilder().configure(
        max_workers=workers if parallel else 1,
        continue_on_error=True
    )

    # Add analyzers based on feature flags
    if flags.is_enabled('face_recognition'):
        click.echo("  ✅ Character detection (face_recognition)")
        builder.add_character_detection(
            FaceRecognitionAnalyzer(flags.get_config('face_recognition'))
        )

    if flags.is_enabled('emotion_detection'):
        click.echo("  ✅ Emotion detection (deepface)")
        builder.add_emotion_detection(
            DeepFaceEmotionAnalyzer(flags.get_config('emotion_detection'))
        )

    if flags.is_enabled('ocr_dialogue'):
        click.echo("  ✅ Dialogue extraction (tesseract)")
        builder.add_dialogue_detection(
            TesseractOCRAnalyzer(flags.get_config('ocr_dialogue'))
        )

    if flags.is_enabled('visual_analysis'):
        click.echo("  ✅ Visual analysis (color)")
        builder.add_visual_analysis(
            ColorAnalyzer(flags.get_config('visual_analysis'))
        )

    if flags.is_enabled('scene_classification'):
        click.echo("  ✅ Scene classification (basic)")
        builder.add_scene_analysis(
            BasicSceneAnalyzer(flags.get_config('scene_classification'))
        )

    pipeline = builder.build()

    # Load database
    click.echo(f"\n📊 Loading database from {output}...")
    try:
        db = load_existing_database(output)
        total_panels = sum(len(ep.panels) for ep in db.episodes)
        click.echo(f"  Found {len(db.episodes)} episodes, {total_panels} panels")
    except Exception as e:
        click.echo(f"  Failed to load database: {e}")
        click.echo("  Creating new database")
        db = Database()

    # Filter episodes if specified
    if episode:
        episodes_to_process = [ep for ep in db.episodes if ep.id == episode]
        if not episodes_to_process:
            click.echo(f"❌ Episode {episode} not found")
            return
    else:
        episodes_to_process = db.episodes

    # Process panels
    click.echo(f"\n🚀 Processing {len(episodes_to_process)} episodes...")
    mode = "parallel" if parallel else "serial"
    click.echo(f"  Mode: {mode} ({workers} workers)" if parallel else f"  Mode: {mode}")

    with click.progressbar(episodes_to_process, label='Processing') as bar:
        for ep in bar:
            processed_panels = pipeline.process_panels(ep.panels, parallel=parallel)
            ep.panels = processed_panels

    # Save database
    click.echo(f"\n💾 Saving database to {output}...")
    save_to_frontend_format(db, output)
    click.echo("✅ Complete!")


@cli.command()
@click.argument('feature')
@click.option('--state', type=click.Choice(['enabled', 'disabled', 'experimental']), help='Set feature state')
def feature(feature: str, state: Optional[str]):
    """Manage feature flags."""
    flags = get_feature_flags()

    if state:
        # Update state
        if state == 'enabled':
            flags.enable(feature)
        elif state == 'disabled':
            flags.disable(feature)
        elif state == 'experimental':
            flags.set_experimental(feature)

        click.echo(f"✅ Feature '{feature}' set to {state}")
    else:
        # Show current state
        current_state = flags.flags.get(feature, {}).get('state', 'unknown')
        click.echo(f"Feature '{feature}': {current_state}")


@cli.command()
def features():
    """List all features and their states."""
    flags = get_feature_flags()

    click.echo("\n🎛️  Feature Flags")
    click.echo("=" * 60)

    for feature, state in flags.list_features().items():
        icon = "✅" if state == "enabled" else "🧪" if state == "experimental" else "❌"
        click.echo(f"  {icon} {feature}: {state}")


@cli.command()
@click.option('--episodes', '-e', default=100, help='Number of episodes')
@click.option('--panels-per-episode', '-p', default=50, help='Average panels per episode')
def validate(episodes: int, panels_per_episode: int):
    """Validate database integrity and quality."""
    click.echo("\n🔍 Database Validation")
    click.echo("=" * 60)

    # Load database
    db_path = "frontend/data/panels_database.json"
    try:
        db = load_existing_database(db_path)
    except Exception as e:
        click.echo(f"❌ Failed to load database: {e}")
        return

    # Statistics
    total_panels = sum(len(ep.panels) for ep in db.episodes)
    tagged_panels = sum(
        1 for ep in db.episodes for p in ep.panels
        if p.ai_analysis is not None
    )

    click.echo(f"\n📊 Statistics:")
    click.echo(f"  Episodes: {len(db.episodes)}")
    click.echo(f"  Total panels: {total_panels}")
    click.echo(f"  Tagged panels: {tagged_panels} ({tagged_panels/total_panels*100:.1f}%)")

    # Quality checks
    click.echo(f"\n✅ Quality Checks:")

    if len(db.episodes) < episodes:
        click.echo(f"  ⚠️  Low episode count ({len(db.episodes)} < {episodes})")
    else:
        click.echo(f"  ✅ Episode count OK")

    avg_panels = total_panels / len(db.episodes) if db.episodes else 0
    if avg_panels < panels_per_episode * 0.5:
        click.echo(f"  ⚠️  Low average panels per episode ({avg_panels:.1f})")
    else:
        click.echo(f"  ✅ Panels per episode OK")

    if tagged_panels / total_panels < 0.8:
        click.echo(f"  ⚠️  Low tagging coverage")
    else:
        click.echo(f"  ✅ Tagging coverage OK")


@cli.command()
@click.argument('character_name')
@click.argument('images_dir', type=click.Path(exists=True))
def add_character(character_name: str, images_dir: str):
    """Add character to face recognition database."""
    click.echo(f"\n👤 Adding character: {character_name}")
    click.echo(f"  Images directory: {images_dir}")

    # This would build face encodings from the images
    # Implementation similar to build_character_database.py
    click.echo("  [Not yet implemented - use build_character_database.py]")


@cli.command()
def info():
    """Show system information."""
    import sys
    click.echo("\n📦 System Information")
    click.echo("=" * 60)
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  Database version: 2.0")

    # Check dependencies
    click.echo(f"\n📚 Dependencies:")
    deps = [
        ('face_recognition', 'Face recognition'),
        ('deepface', 'Emotion detection'),
        ('pytesseract', 'OCR'),
        ('sklearn', 'Visual analysis'),
        ('cv2', 'Image processing'),
    ]

    for module, description in deps:
        try:
            __import__(module)
            click.echo(f"  ✅ {description} ({module})")
        except ImportError:
            click.echo(f"  ❌ {description} ({module}) - NOT INSTALLED")


if __name__ == '__main__':
    cli()
