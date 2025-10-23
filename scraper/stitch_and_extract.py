"""
Stitch and Extract Panels
STEP 1: Stitches all images from each episode into one long vertical image
STEP 2: Detects panel boundaries in the stitched image
STEP 3: Extracts individual panels

This is necessary because webtoon panels often span multiple downloaded images
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Import new unified panel detector
from .migrate_panel_detection import detect_panel_gaps as detect_panel_gaps_new

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Use absolute paths relative to script location
SCRIPT_DIR = Path(__file__).parent.parent
INPUT_DIR = SCRIPT_DIR / "data" / "originals"
STITCHED_DIR = SCRIPT_DIR / "data" / "stitched"
OUTPUT_DIR = SCRIPT_DIR / "data" / "panels_original"
METADATA_FILE = SCRIPT_DIR / "data" / "panel_metadata.json"

# Panel detection settings
MIN_PANEL_HEIGHT = 200  # Minimum height for a valid panel
MIN_GAP_HEIGHT = 10     # Minimum white gap to split panels
WHITE_THRESHOLD = 245   # Brightness threshold for "white" gaps
PADDING = 5             # Pixels to add around panels

# Create output directories
STITCHED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stitch_episode_images(episode_dir):
    """
    Stitch all page images from an episode into one long vertical image
    Returns the stitched image
    """
    # Get all page images sorted
    page_images = sorted(episode_dir.glob('page_*.jpg'), key=lambda x: int(x.stem.split('_')[1]))

    if not page_images:
        print(f"  [WARNING] No images found in {episode_dir}", flush=True)
        return None

    print(f"  Stitching {len(page_images)} images...", flush=True)

    # Load all images
    images = []
    total_height = 0
    max_width = 0

    for img_path in page_images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARNING] Failed to read {img_path}", flush=True)
            continue

        images.append(img)
        total_height += img.shape[0]
        max_width = max(max_width, img.shape[1])

    if not images:
        return None

    # Create blank canvas
    stitched = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    stitched.fill(255)  # White background

    # Paste images vertically
    current_y = 0
    for img in images:
        h, w = img.shape[:2]
        # Center image horizontally if needed
        x_offset = (max_width - w) // 2
        stitched[current_y:current_y+h, x_offset:x_offset+w] = img
        current_y += h

    print(f"  Stitched image size: {max_width} x {total_height} px", flush=True)

    return stitched


def detect_panel_gaps(stitched_image):
    """
    Detect horizontal white gaps between panels
    Returns list of (start_y, end_y) tuples for each panel

    NOTE: This function now uses the new unified panel detector.
    The old implementation is preserved below for reference but is no longer used.
    """
    # Use new unified panel detector
    panels = detect_panel_gaps_new(
        stitched_image,
        min_panel_height=MIN_PANEL_HEIGHT,
        min_gap_height=MIN_GAP_HEIGHT,
        white_threshold=WHITE_THRESHOLD
    )

    print(f"  Detected {len(panels)} panels", flush=True)
    return panels


# DEPRECATED: Old implementation preserved for reference
# This code is no longer used but kept for comparison/rollback if needed
def _detect_panel_gaps_old(stitched_image):
    """
    DEPRECATED: Old panel detection implementation.
    Use detect_panel_gaps() instead, which wraps the new unified detector.
    """
    height, width = stitched_image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)

    # Calculate average brightness for each row
    row_brightness = np.mean(gray, axis=1)

    # Find white rows (gaps between panels)
    white_rows = row_brightness > WHITE_THRESHOLD

    # Find continuous stretches of white rows
    gaps = []
    in_gap = False
    gap_start = 0

    for y in range(height):
        if white_rows[y] and not in_gap:
            gap_start = y
            in_gap = True
        elif not white_rows[y] and in_gap:
            gap_length = y - gap_start
            if gap_length >= MIN_GAP_HEIGHT:
                gaps.append((gap_start, y))
            in_gap = False

    # Convert gaps to panel boundaries
    panels = []
    current_y = 0

    for gap_start, gap_end in gaps:
        if gap_start - current_y > MIN_PANEL_HEIGHT:
            panels.append((current_y, gap_start))
        current_y = gap_end

    # Add final panel if there's content after last gap
    if height - current_y > MIN_PANEL_HEIGHT:
        panels.append((current_y, height))

    # If no gaps found, treat entire image as panels (split every ~2000px)
    if not panels:
        print("  [INFO] No clear gaps found, using fallback splitting", flush=True)
        chunk_size = 2000
        for y in range(0, height, chunk_size):
            end_y = min(y + chunk_size, height)
            if end_y - y > MIN_PANEL_HEIGHT:
                panels.append((y, end_y))

    return panels


def extract_panels(stitched_image, panel_boundaries, output_dir, season, episode):
    """
    Extract individual panels from stitched image
    """
    height, width = stitched_image.shape[:2]
    panel_data = []

    for idx, (start_y, end_y) in enumerate(panel_boundaries, 1):
        # Add padding
        y1 = max(0, start_y - PADDING)
        y2 = min(height, end_y + PADDING)

        # Extract panel
        panel_img = stitched_image[y1:y2, :]

        # Save panel
        panel_filename = f"s{season}_ep{episode:03d}_p{idx:03d}.jpg"
        panel_path = output_dir / panel_filename

        # Save in highest quality
        cv2.imwrite(str(panel_path), panel_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        panel_h, panel_w = panel_img.shape[:2]

        panel_data.append({
            'filename': panel_filename,
            'path': str(panel_path),
            'season': season,
            'episode': episode,
            'panel_number': idx,
            'dimensions': {'width': panel_w, 'height': panel_h},
            'position_in_stitched': {'y_start': y1, 'y_end': y2}
        })

    return panel_data


def process_all_episodes():
    """Process all downloaded episodes"""
    print("=" * 60, flush=True)
    print("Stitch and Extract Panels", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    all_panels = []

    # Find all season directories
    season_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith('s')])

    for season_dir in season_dirs:
        season_num = int(season_dir.name[1:])

        # Find all episode directories
        episode_dirs = sorted([d for d in season_dir.iterdir() if d.is_dir()])

        print(f"Season {season_num}: {len(episode_dirs)} episodes", flush=True)
        print(flush=True)

        for ep_dir in episode_dirs:
            ep_num = int(ep_dir.name.replace('ep', ''))

            print(f"[S{season_num} E{ep_num}] Processing...", flush=True)

            # STEP 1: Stitch images
            stitched_image = stitch_episode_images(ep_dir)

            if stitched_image is None:
                print(f"  [ERROR] Failed to stitch episode", flush=True)
                print(flush=True)
                continue

            # Save stitched image for reference
            stitched_path = STITCHED_DIR / f"s{season_num}" / f"ep{ep_num:03d}"
            stitched_path.mkdir(parents=True, exist_ok=True)
            stitched_file = stitched_path / "stitched.jpg"
            cv2.imwrite(str(stitched_file), stitched_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  Saved stitched: {stitched_file}", flush=True)

            # STEP 2: Detect panels
            panel_boundaries = detect_panel_gaps(stitched_image)

            # STEP 3: Extract panels
            panel_output_dir = OUTPUT_DIR / f"s{season_num}" / f"ep{ep_num:03d}"
            panel_output_dir.mkdir(parents=True, exist_ok=True)

            panels = extract_panels(stitched_image, panel_boundaries, panel_output_dir, season_num, ep_num)

            all_panels.extend(panels)

            print(f"  Extracted {len(panels)} panels", flush=True)
            print(flush=True)

    # Save metadata
    metadata = {
        'total_panels': len(all_panels),
        'panels': all_panels
    }

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 60, flush=True)
    print("[OK] Panel extraction complete!", flush=True)
    print(f"[OK] Total panels extracted: {len(all_panels)}", flush=True)
    print(f"[OK] Metadata saved to: {METADATA_FILE}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    process_all_episodes()
