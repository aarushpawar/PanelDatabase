"""
Migration wrapper for gradual transition to new panel detector.

Provides backward-compatible functions that wrap the new PanelDetector
so existing scripts continue to work without changes.

This module allows existing code to use the new unified panel detection
system without requiring immediate refactoring of all scripts.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import tempfile

from .core.panel_detector import PanelDetector, DetectionMode


def detect_panels_simple(image_path: Path) -> Tuple[List[Dict], np.ndarray]:
    """
    Backward-compatible wrapper for extract_panels.py

    Replaces the original simple brightness-based detection with
    the new unified detector.

    Args:
        image_path: Path to the stitched episode image

    Returns:
        (panels, image) tuple matching old format:
        - panels: List of dicts with keys 'x', 'y', 'w', 'h'
        - image: Loaded image as numpy array

    Example:
        >>> panels, img = detect_panels_simple(Path('ep001.jpg'))
        >>> print(f"Found {len(panels)} panels")
    """
    detector = PanelDetector(mode=DetectionMode.STANDARD)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Detect panels
    panel_bounds = detector.detect(image_path, apply_overlap=False)

    # Convert to old format
    panels = [pb.to_dict() for pb in panel_bounds]

    return panels, image


def detect_panels_with_content_awareness(
    stitched_image: np.ndarray,
    image_path: Path = None
) -> List[Tuple[int, int]]:
    """
    Backward-compatible wrapper for smart_panel_detection.py

    Replaces the original content-aware detection with the new
    unified detector.

    Args:
        stitched_image: Stitched episode image as numpy array
        image_path: Optional path if image is already saved

    Returns:
        List of (start_y, end_y) tuples matching old format

    Example:
        >>> img = cv2.imread('ep001.jpg')
        >>> panels = detect_panels_with_content_awareness(img)
        >>> for start, end in panels:
        ...     panel = img[start:end, :]
    """
    # Save image temporarily if not provided
    if image_path is None:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, stitched_image)
            tmp_path = Path(tmp.name)
    else:
        tmp_path = image_path

    try:
        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panel_bounds = detector.detect(tmp_path, apply_overlap=True)

        # Convert to old format (just y coordinates)
        result = [(pb.y_start, pb.y_end) for pb in panel_bounds]
        return result
    finally:
        # Clean up temp file if created
        if image_path is None and tmp_path.exists():
            tmp_path.unlink()


def detect_panel_gaps(
    stitched_image: np.ndarray,
    min_panel_height: int = 200,
    min_gap_height: int = 10,
    white_threshold: int = 245
) -> List[Tuple[int, int]]:
    """
    Backward-compatible wrapper for stitch_and_extract.py

    Replaces the original gap detection with the new unified detector.

    Args:
        stitched_image: Stitched episode image as numpy array
        min_panel_height: Minimum panel height (passed to detector)
        min_gap_height: Minimum gap height (passed to detector)
        white_threshold: Brightness threshold for white detection

    Returns:
        List of (start_y, end_y) tuples for each panel

    Example:
        >>> img = cv2.imread('ep001.jpg')
        >>> panels = detect_panel_gaps(img, min_panel_height=200)
        >>> for start, end in panels:
        ...     panel = img[start:end, :]
    """
    # Save image temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, stitched_image)
        tmp_path = Path(tmp.name)

    try:
        # Create detector with custom config
        config_override = {
            'detection.min_panel_height': min_panel_height,
            'detection.min_gap_height': min_gap_height,
            'detection.white_threshold': white_threshold
        }

        detector = PanelDetector(
            mode=DetectionMode.STANDARD,
            config_override=config_override
        )

        panel_bounds = detector.detect(tmp_path, apply_overlap=True)

        # Convert to old format
        result = [(pb.y_start, pb.y_end) for pb in panel_bounds]
        return result
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def detect_panels_contour(image_path: Path) -> Tuple[List[Dict], np.ndarray]:
    """
    Backward-compatible wrapper for contour-based detection.

    The new detector doesn't use contours, but we provide this function
    for compatibility. It uses the standard detection method.

    Args:
        image_path: Path to the stitched episode image

    Returns:
        (panels, image) tuple matching old format

    Example:
        >>> panels, img = detect_panels_contour(Path('ep001.jpg'))
    """
    # Just use standard detection
    return detect_panels_simple(image_path)
