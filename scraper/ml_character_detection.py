"""
ML-Based Character Detection
Uses CLIP (OpenAI) for zero-shot character detection in panels
Downloads character images from wiki and matches them to panels
"""

import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# Check if transformers and torch are available
try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    print("CLIP not available. Install with: pip install transformers torch torchvision")
    HAS_CLIP = False

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
WIKI_DATA = SCRIPT_DIR / "data" / "wiki_categories.json"
CHARACTER_IMAGES_DIR = SCRIPT_DIR / "data" / "character_images"
PANELS_DIR = SCRIPT_DIR / "data" / "panels_original"
DATABASE_FILE = SCRIPT_DIR / "frontend" / "data" / "panels_database.json"

# Create directories
CHARACTER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Main characters to focus on (most important for detection)
PRIORITY_CHARACTERS = [
    "Min Sayeon", "Sayeon",
    "Cha Jaehee", "Jaehee",
    "Ryujin", "Jungwoo Ryujin",
    "Samin", "Lee Samin",
    "Dahee", "Yoo Dahee",
    "Iseul", "Lee Iseul"
]

def download_character_image(character_url, character_name):
    """Download character profile image from wiki"""
    try:
        response = requests.get(character_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find profile image (usually in infobox)
        infobox = soup.select_one('.portable-infobox')
        if infobox:
            img = infobox.select_one('img')
            if img and img.get('src'):
                img_url = img['src']

                # Download image
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()

                # Save image
                safe_name = character_name.replace(' ', '_').replace('/', '_')
                img_path = CHARACTER_IMAGES_DIR / f"{safe_name}.jpg"

                with open(img_path, 'wb') as f:
                    f.write(img_response.content)

                return img_path

        return None

    except Exception as e:
        print(f"Error downloading {character_name}: {e}")
        return None

def download_all_character_images():
    """Download character images from wiki"""
    print("Downloading Character Images from Wiki")
    print("=" * 70)

    with open(WIKI_DATA, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    characters = wiki['categories']['characters']
    print(f"Found {len(characters)} characters")

    downloaded = []
    for char in tqdm(characters, desc="Downloading"):
        if any(priority in char['name'] for priority in PRIORITY_CHARACTERS):
            img_path = download_character_image(char['url'], char['name'])
            if img_path:
                downloaded.append({
                    'name': char['name'],
                    'image_path': str(img_path)
                })

    print(f"\\nDownloaded {len(downloaded)} character images")
    return downloaded

def detect_characters_with_clip(panel_path, character_images, model, processor, threshold=0.25):
    """
    Use CLIP to detect which characters appear in a panel
    Returns list of detected character names
    """
    try:
        # Load panel image
        panel_img = Image.open(panel_path).convert('RGB')

        detected_chars = []

        for char_data in character_images:
            char_name = char_data['name']
            char_img_path = char_data['image_path']

            # Load character reference image
            char_img = Image.open(char_img_path).convert('RGB')

            # Prepare inputs
            inputs = processor(
                text=[f"a photo of {char_name}", "a different person", "no person"],
                images=[panel_img],
                return_tensors="pt",
                padding=True
            )

            # Get similarity scores
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)

            # Check if character is detected (first option has high probability)
            if probs[0][0] > threshold:
                detected_chars.append(char_name)

        return detected_chars

    except Exception as e:
        print(f"Error detecting in {panel_path}: {e}")
        return []

def process_all_panels(character_images):
    """Process all panels with CLIP character detection"""
    if not HAS_CLIP:
        print("ERROR: CLIP not available. Install with:")
        print("  pip install transformers torch torchvision")
        return

    print("\\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load database
    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        database = json.load(f)

    print(f"Processing {len(database['panels'])} panels...")

    # Process each panel
    tagged_count = 0
    for panel in tqdm(database['panels'], desc="Detecting characters"):
        # Build path to panel image (flat structure, no seasons)
        episode = panel['episode']
        panel_id = panel['filename'].replace('.jpg', '')

        panel_path = PANELS_DIR / f"ep{episode:03d}" / f"{panel_id}.jpg"

        if not panel_path.exists():
            continue

        # Detect characters
        detected = detect_characters_with_clip(
            panel_path,
            character_images,
            model,
            processor
        )

        # Update panel data
        if detected:
            panel['manual']['characters'] = detected
            panel['metadata']['tagged'] = True
            panel['metadata']['auto_tagged_ml'] = True
            tagged_count += 1

    print(f"\\nTagged {tagged_count} panels with ML detection")

    # Save updated database
    print("Saving database...")
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    # Update statistics
    stats_file = DATABASE_FILE.parent / "statistics.json"
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    stats['tagged_panels'] = tagged_count
    stats['ml_tagged'] = True

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Done! {tagged_count} panels now have ML-detected characters")

def main():
    print("ML-Based Character Detection System")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Download character images from wiki")
    print("  2. Use CLIP ML model to detect characters in each panel")
    print("  3. Update database with detected characters")
    print()
    print("Note: This requires ~1GB download for CLIP model on first run")
    print("      Processing 5000 panels will take 30-60 minutes")
    print()

    if not HAS_CLIP:
        print("ERROR: Required libraries not installed")
        print("Install with: pip install transformers torch torchvision Pillow")
        return

    # Step 1: Download character images
    character_images = download_all_character_images()

    if not character_images:
        print("ERROR: No character images downloaded")
        return

    # Step 2: Process panels with ML detection
    process_all_panels(character_images)

if __name__ == "__main__":
    main()
