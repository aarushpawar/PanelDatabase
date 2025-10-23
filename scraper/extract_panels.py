"""
Panel Extraction Script

DEPRECATED: This module has been replaced by the unified panel detector.
The functions below now delegate to scraper.core.panel_detector.
All functionality has been consolidated into a single, well-tested module.

For new code, use:
    from scraper.core.panel_detector import PanelDetector
    detector = PanelDetector(mode='standard')
    panels = detector.detect(image_path)

This file is kept for backward compatibility only.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import shutil
import warnings

# Import new unified detector
from .migrate_panel_detection import detect_panels_simple as detect_panels_simple_new

# Show deprecation warning when module is imported
warnings.warn(
    "extract_panels.py is deprecated. "
    "Use scraper.core.panel_detector.PanelDetector instead.",
    DeprecationWarning,
    stacklevel=2
)

INPUT_DIR = Path("../data/originals")
OUTPUT_DIR = Path("../data/panels_original")
METADATA_FILE = Path("../data/panel_metadata.json")

# Panel detection settings
MIN_PANEL_HEIGHT = 100  # Minimum height for a valid panel
MIN_PANEL_WIDTH = 200   # Minimum width for a valid panel
PADDING = 10  # Pixels to add around detected panels


def detect_panels_simple(image_path):
    """
    Simple panel detection for webtoons using horizontal gaps
    Webtoons typically have white/transparent spaces between panels
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read: {image_path}")
        return []

    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find horizontal lines (gaps between panels)
    # Look for mostly white/light horizontal strips
    row_sums = np.mean(gray, axis=1)  # Average brightness per row

    # Threshold to find light rows (panel gaps)
    threshold = 250  # Near white
    is_gap = row_sums > threshold

    # Find transitions from gap to content
    panels = []
    in_panel = False
    panel_start = 0

    for i, gap in enumerate(is_gap):
        if not gap and not in_panel:  # Start of panel
            panel_start = i
            in_panel = True
        elif gap and in_panel:  # End of panel
            panel_height = i - panel_start
            if panel_height > MIN_PANEL_HEIGHT:
                panels.append({
                    'y': max(0, panel_start - PADDING),
                    'h': min(height, i + PADDING) - max(0, panel_start - PADDING),
                    'x': 0,
                    'w': width
                })
            in_panel = False

    # Handle case where image ends mid-panel
    if in_panel and (height - panel_start) > MIN_PANEL_HEIGHT:
        panels.append({
            'y': max(0, panel_start - PADDING),
            'h': height - max(0, panel_start - PADDING),
            'x': 0,
            'w': width
        })

    return panels, img


def detect_panels_contour(image_path):
    """
    Contour-based panel detection (backup method)
    More sophisticated but may not work as well for webtoons
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return [], None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by size
        if w > MIN_PANEL_WIDTH and h > MIN_PANEL_HEIGHT:
            panels.append({
                'x': max(0, x - PADDING),
                'y': max(0, y - PADDING),
                'w': w + 2 * PADDING,
                'h': h + 2 * PADDING
            })

    # Sort panels top to bottom
    panels.sort(key=lambda p: p['y'])

    return panels, img


def extract_and_save_panels(image_path, output_dir, season, episode, start_index=0):
    """Extract panels from an episode image and save them"""

    # Try simple method first (works better for webtoons)
    panels, img = detect_panels_simple(image_path)

    # If simple method fails or finds too few panels, try contour method
    if len(panels) < 3:
        panels_contour, _ = detect_panels_contour(image_path)
        if len(panels_contour) > len(panels):
            panels = panels_contour

    # If still no panels found, treat entire image as one panel
    if len(panels) == 0:
        height, width = img.shape[:2]
        panels = [{'x': 0, 'y': 0, 'w': width, 'h': height}]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract each panel
    panel_data = []
    for idx, panel in enumerate(panels, start=start_index + 1):
        # Crop panel
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        panel_img = img[y:y+h, x:x+w]

        # Save panel
        panel_filename = f"s{season}_ep{episode:03d}_p{idx:03d}.jpg"
        panel_path = output_dir / panel_filename

        # Save in highest quality
        cv2.imwrite(str(panel_path), panel_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        panel_data.append({
            'filename': panel_filename,
            'path': str(panel_path),
            'season': season,
            'episode': episode,
            'panel_number': idx,
            'dimensions': {'width': w, 'height': h},
            'source_image': str(image_path),
            'crop_coords': {'x': x, 'y': y, 'w': w, 'h': h}
        })

    return panel_data


def process_all_episodes():
    """Process all downloaded episodes"""
    print("Panel Extraction")
    print("=" * 50)

    all_panels = []

    # Find all season directories
    season_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith('s')])

    for season_dir in season_dirs:
        season_num = int(season_dir.name[1:])  # Extract number from 's1', 's2', etc.

        # Find all episode directories
        episode_dirs = sorted([d for d in season_dir.iterdir() if d.is_dir()])

        print(f"\nProcessing Season {season_num}: {len(episode_dirs)} episodes")

        for ep_dir in tqdm(episode_dirs, desc=f"Season {season_num}"):
            ep_num = int(ep_dir.name.replace('ep', ''))

            # Find all page images in this episode
            page_images = sorted(ep_dir.glob('page_*.jpg'))

            if not page_images:
                continue

            # Create output directory for this episode's panels
            panel_output_dir = OUTPUT_DIR / f"s{season_num}" / f"ep{ep_num:03d}"

            # Process each page
            panel_index = 0
            for page_img in page_images:
                panels = extract_and_save_panels(
                    page_img,
                    panel_output_dir,
                    season_num,
                    ep_num,
                    start_index=panel_index
                )
                all_panels.extend(panels)
                panel_index += len(panels)

    # Save metadata
    metadata = {
        'total_panels': len(all_panels),
        'panels': all_panels
    }

    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Extraction complete!")
    print(f"✓ Extracted {len(all_panels)} panels")
    print(f"✓ Metadata saved to {METADATA_FILE}")


if __name__ == "__main__":
    process_all_episodes()
