"""
Smart Panel Detection - IMPROVED VERSION

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
import sys
import warnings

# Import new unified detector
from .migrate_panel_detection import detect_panels_with_content_awareness

# Show deprecation warning when module is imported
warnings.warn(
    "smart_panel_detection.py is deprecated. "
    "Use scraper.core.panel_detector.PanelDetector instead.",
    DeprecationWarning,
    stacklevel=2
)

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
INPUT_DIR = SCRIPT_DIR / "data" / "originals"
STITCHED_DIR = SCRIPT_DIR / "data" / "stitched"
OUTPUT_DIR = SCRIPT_DIR / "data" / "panels_original"
METADATA_FILE = SCRIPT_DIR / "data" / "panel_metadata.json"

# Detection settings - MUCH more conservative
MIN_PANEL_HEIGHT = 200  # Minimum panel height (increased)
MIN_WHITE_GAP = 50      # Minimum white gap size to consider splitting
WHITE_THRESHOLD = 245   # Brightness threshold for "white" rows
CONTENT_BUFFER = 100    # Large buffer around any detected content
MAX_PANEL_HEIGHT = 3000 # Maximum panel height before forcing a split

# Create output directories
STITCHED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def stitch_episode_images(episode_dir):
    """Stitch all page images from an episode"""
    page_images = sorted(episode_dir.glob('page_*.jpg'),
                        key=lambda x: int(x.stem.split('_')[1]))

    if not page_images:
        return None

    images = []
    total_height = 0
    max_width = 0

    for img_path in page_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        images.append(img)
        total_height += img.shape[0]
        max_width = max(max_width, img.shape[1])

    if not images:
        return None

    # Create stitched image
    stitched = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    stitched.fill(255)

    current_y = 0
    for img in images:
        h, w = img.shape[:2]
        x_offset = (max_width - w) // 2
        stitched[current_y:current_y+h, x_offset:x_offset+w] = img
        current_y += h

    return stitched


def detect_content_regions(image):
    """
    Detect ALL content in the image using edge detection.
    Returns array where True = has content, False = empty/white
    """
    height = image.shape[0]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Method 1: Brightness check (white = empty)
    brightness = np.mean(gray, axis=1)  # Average brightness per row
    is_white = brightness > WHITE_THRESHOLD

    # Method 2: Edge detection (catches all drawn lines/content)
    edges = cv2.Canny(gray, 30, 100)  # Lower thresholds to catch more edges
    edge_density = np.sum(edges, axis=1)  # Count edge pixels per row
    has_edges = edge_density > 50  # If more than 50 edge pixels, consider it content

    # Combine: A row is "empty" only if it's white AND has no edges
    has_content = ~is_white | has_edges

    return has_content


def find_safe_split_points(image, has_content):
    """
    Find safe places to split the image.
    Only splits in the middle of large white gaps with no content.
    """
    height = image.shape[0]

    # Find white gaps (consecutive rows with no content)
    gaps = []
    in_gap = False
    gap_start = 0

    for y in range(height):
        if not has_content[y]:  # No content = potential gap
            if not in_gap:
                gap_start = y
                in_gap = True
        else:  # Has content
            if in_gap:
                gap_end = y
                gap_size = gap_end - gap_start

                # Only consider gaps that are large enough
                if gap_size >= MIN_WHITE_GAP:
                    gap_middle = (gap_start + gap_end) // 2
                    gaps.append({
                        'start': gap_start,
                        'end': gap_end,
                        'middle': gap_middle,
                        'size': gap_size
                    })

                in_gap = False

    # Handle final gap if image ends with white space
    if in_gap:
        gap_end = height
        gap_size = gap_end - gap_start
        if gap_size >= MIN_WHITE_GAP:
            gap_middle = (gap_start + gap_end) // 2
            gaps.append({
                'start': gap_start,
                'end': gap_end,
                'middle': gap_middle,
                'size': gap_size
            })

    return gaps


def create_panels_from_gaps(image, gaps, has_content):
    """
    Create panel boundaries from detected gaps.
    Adds large buffer zones around content to avoid cutting through anything.
    """
    height = image.shape[0]
    panels = []
    current_y = 0

    for gap in gaps:
        # Check if this gap is far enough from the last split
        potential_panel_height = gap['start'] - current_y

        if potential_panel_height < MIN_PANEL_HEIGHT:
            # Panel would be too small, skip this gap
            continue

        # Extra safety: Check buffer zone around split point
        split_y = gap['middle']
        buffer_start = max(0, split_y - CONTENT_BUFFER)
        buffer_end = min(height, split_y + CONTENT_BUFFER)

        # Check if buffer zone is truly content-free
        buffer_has_content = np.any(has_content[buffer_start:buffer_end])

        if buffer_has_content:
            # Buffer zone has content, not safe to split here
            continue

        # Safe to split! Create panel
        panels.append((current_y, gap['start']))
        current_y = gap['end']

    # Add final panel if there's enough content left
    if height - current_y >= MIN_PANEL_HEIGHT:
        panels.append((current_y, height))
    elif panels:
        # Extend last panel to end of image
        panels[-1] = (panels[-1][0], height)
    else:
        # No good splits found, treat whole episode as one panel
        panels.append((0, height))

    return panels


def detect_panels_with_content_awareness(stitched_image):
    """
    Main panel detection algorithm using content-aware splitting.
    """
    height = stitched_image.shape[0]

    print(f"  Analyzing content ({height}px tall)...")

    # Step 1: Detect where content exists
    has_content = detect_content_regions(stitched_image)
    content_rows = np.sum(has_content)
    empty_rows = height - content_rows

    print(f"  Content rows: {content_rows}, Empty rows: {empty_rows}")

    # Step 2: Find white gaps
    gaps = find_safe_split_points(stitched_image, has_content)
    print(f"  Found {len(gaps)} potential white gaps")

    # Step 3: Create panels from safe gaps
    panels = create_panels_from_gaps(stitched_image, gaps, has_content)

    # Step 4: Handle very long panels (force split if necessary)
    final_panels = []
    for start_y, end_y in panels:
        panel_height = end_y - start_y

        # If panel is too tall, we need to force a split
        if panel_height > MAX_PANEL_HEIGHT:
            # Find the best place to split within this long panel
            chunk_start = start_y
            while chunk_start < end_y:
                chunk_end = min(chunk_start + MAX_PANEL_HEIGHT, end_y)

                # Try to find a white gap near the target split point
                search_start = max(chunk_start, chunk_end - 200)
                search_end = min(end_y, chunk_end + 200)

                # Find least-content row in search range
                search_region = has_content[search_start:search_end]
                if len(search_region) > 0:
                    # Find row with least content
                    gray = cv2.cvtColor(stitched_image[search_start:search_end], cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray, axis=1)
                    best_split_idx = np.argmax(brightness)  # Brightest = least content
                    best_split = search_start + best_split_idx
                else:
                    best_split = chunk_end

                final_panels.append((chunk_start, best_split))
                chunk_start = best_split
        else:
            final_panels.append((start_y, end_y))

    print(f"  Created {len(final_panels)} panels")

    # Validate panels
    for i, (start, end) in enumerate(final_panels):
        height = end - start
        print(f"    Panel {i+1}: {height}px tall")

    return final_panels


def extract_panels(stitched_image, panel_boundaries, output_dir, episode):
    """Extract individual panels"""
    height, width = stitched_image.shape[:2]
    panel_data = []

    for idx, (start_y, end_y) in enumerate(panel_boundaries, 1):
        # Extract panel
        y1 = max(0, start_y)
        y2 = min(height, end_y)

        panel_img = stitched_image[y1:y2, :]

        # Save panel
        panel_filename = f"ep{episode:03d}_p{idx:03d}.jpg"
        panel_path = output_dir / panel_filename

        cv2.imwrite(str(panel_path), panel_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        panel_h, panel_w = panel_img.shape[:2]

        panel_data.append({
            'filename': panel_filename,
            'path': str(panel_path),
            'episode': episode,
            'panel_number': idx,
            'dimensions': {'width': panel_w, 'height': panel_h},
            'position_in_stitched': {'y_start': y1, 'y_end': y2}
        })

    return panel_data


def process_all_episodes():
    """Process all episodes with smart panel detection"""
    print("=" * 70)
    print("Smart Panel Detection - Content-Aware Edge Detection")
    print("=" * 70)
    print()

    all_panels = []

    # Find all episode directories (flat structure, no seasons)
    episode_dirs = sorted([d for d in INPUT_DIR.iterdir()
                          if d.is_dir() and d.name.startswith('ep')])

    print(f"Found {len(episode_dirs)} episodes")
    print()

    for ep_dir in tqdm(episode_dirs, desc="Processing"):
        ep_num = int(ep_dir.name.replace('ep', ''))

        # Stitch images
        stitched_image = stitch_episode_images(ep_dir)
        if stitched_image is None:
            continue

        # Save stitched image
        stitched_path = STITCHED_DIR / f"ep{ep_num:03d}"
        stitched_path.mkdir(parents=True, exist_ok=True)
        stitched_file = stitched_path / "stitched.jpg"
        cv2.imwrite(str(stitched_file), stitched_image,
                   [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Detect panels intelligently
        panel_boundaries = detect_panels_with_content_awareness(stitched_image)

        # Extract panels
        panel_output_dir = OUTPUT_DIR / f"ep{ep_num:03d}"
        panel_output_dir.mkdir(parents=True, exist_ok=True)

        panels = extract_panels(stitched_image, panel_boundaries,
                              panel_output_dir, ep_num)

        all_panels.extend(panels)

    # Save metadata
    metadata = {
        'total_panels': len(all_panels),
        'panels': all_panels,
        'detection_method': 'content_aware_edge_detection'
    }

    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"Extracted {len(all_panels)} panels with content-aware detection")
    print(f"Metadata: {METADATA_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    process_all_episodes()
