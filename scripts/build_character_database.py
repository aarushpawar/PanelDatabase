#!/usr/bin/env python
"""
Build character face encoding database.

This script collects sample face images for each character
and builds a face recognition database for fast matching.

Directory structure expected:
    data/character_images/
        CharacterName1/
            sample_001.jpg
            sample_002.jpg
            ...
        CharacterName2/
            sample_001.jpg
            ...

Usage:
    python scripts/build_character_database.py

Requirements:
    - face_recognition library
    - Sample character face images in data/character_images/
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core.ml_tagger import CharacterDatabase
from scraper.core.paths import get_path_manager
from scraper.core.logger import setup_logging, get_logger

# Setup logging
setup_logging(Path("logs/build_character_db.log"))
logger = get_logger(__name__)


def build_database():
    """Build character database from sample images."""
    try:
        import face_recognition
    except ImportError:
        logger.error(
            "face_recognition not installed. "
            "Install with: pip install face-recognition"
        )
        print("\n❌ ERROR: face_recognition library not installed")
        print("Install it with: pip install face-recognition")
        return

    paths = get_path_manager()

    # Character images directory
    char_dir = paths.get('data.character_images', create=True)

    if not char_dir.exists():
        logger.error(f"Character images directory not found: {char_dir}")
        print(f"\n❌ ERROR: Character images directory not found:")
        print(f"    {char_dir}")
        print("\nPlease create this directory and add character face samples:")
        print(f"    {char_dir}/CharacterName/sample_001.jpg")
        print(f"    {char_dir}/CharacterName/sample_002.jpg")
        print("    ...")
        return

    # Output database
    db_path = paths.get('metadata.character_embeddings', create=True)
    db = CharacterDatabase(db_path)

    # Find all character subdirectories
    char_folders = [d for d in char_dir.iterdir() if d.is_dir()]

    if not char_folders:
        logger.error(f"No character folders found in {char_dir}")
        print(f"\n❌ ERROR: No character folders found in:")
        print(f"    {char_dir}")
        print("\nPlease create subdirectories for each character with face samples.")
        return

    print(f"\n🔍 Building character database from {len(char_folders)} characters...")
    print(f"    Source: {char_dir}")
    print(f"    Output: {db_path}")
    print()

    total_encodings = 0
    failed_images = 0

    for char_folder in char_folders:
        char_name = char_folder.name

        # Find all images for this character
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(char_folder.glob(ext)))

        if not images:
            print(f"⚠️  {char_name}: No images found, skipping")
            logger.warning(f"No images found for {char_name}")
            continue

        print(f"📸 {char_name}: Processing {len(images)} samples...")

        encodings_added = 0

        for img_path in images:
            try:
                # Load image
                img = face_recognition.load_image_file(str(img_path))

                # Get face encodings
                encodings = face_recognition.face_encodings(img)

                if encodings:
                    # Use first detected face
                    db.add_character(char_name, encodings[0])
                    encodings_added += 1
                    total_encodings += 1
                else:
                    print(f"   ⚠️  No face found in {img_path.name}")
                    logger.warning(f"No face found in {img_path}")
                    failed_images += 1

            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
                logger.error(f"Error processing {img_path}: {e}")
                failed_images += 1

        if encodings_added > 0:
            print(f"   ✅ Added {encodings_added} face encodings")
        else:
            print(f"   ❌ No valid face encodings added")

    # Save database
    if total_encodings > 0:
        db.save()

        print()
        print("=" * 60)
        print("✅ Character database built successfully!")
        print("=" * 60)
        print(f"📊 Statistics:")
        print(f"    Total characters: {db.get_character_count()}")
        print(f"    Total encodings: {total_encodings}")
        print(f"    Failed images: {failed_images}")
        print()
        print(f"📁 Database saved to: {db_path}")
        print()
        print("Character breakdown:")
        for name, encodings in db.characters.items():
            print(f"    {name}: {len(encodings)} encodings")
        print()
        print("✨ You can now use the MLTagger to detect characters in panels!")

        logger.info(
            f"Database built: {db.get_character_count()} characters, "
            f"{total_encodings} encodings"
        )
    else:
        print()
        print("=" * 60)
        print("❌ No character encodings were created")
        print("=" * 60)
        print()
        print("Possible issues:")
        print("  1. No face detected in the images")
        print("  2. Images are corrupted or invalid")
        print("  3. Images don't contain clear, frontal faces")
        print()
        print("Tips for better results:")
        print("  - Use clear, well-lit images")
        print("  - Ensure faces are clearly visible and frontal")
        print("  - Use multiple samples per character (10-50 recommended)")
        print("  - Crop images to focus on the face")

        logger.error("No encodings created")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Character Database Builder")
    print("=" * 60)
    build_database()
