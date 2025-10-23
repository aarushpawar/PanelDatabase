"""
Pytest configuration and shared fixtures.

Provides common test utilities and fixtures for the panel database tests.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def test_data_dir():
    """Directory for test data."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_panel_dir(test_data_dir):
    """Directory with sample panel images."""
    return test_data_dir / "sample_panels"


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test outputs.
    Automatically cleaned up after test.
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_with_gaps():
    """
    Create a synthetic test image with clear panel gaps.

    Creates an 800x4000px image with:
    - 3 panels (dark content areas)
    - 2 white gaps between them

    Returns:
        numpy array (H, W, 3) of the test image
    """
    # Create 800x4000 image
    img = np.ones((4000, 800, 3), dtype=np.uint8) * 255  # White background

    # Panel 1: rows 0-1000 (dark)
    img[0:1000, :, :] = 50

    # Gap 1: rows 1000-1200 (white) - already white

    # Panel 2: rows 1200-2400 (dark)
    img[1200:2400, :, :] = 50

    # Gap 2: rows 2400-2600 (white) - already white

    # Panel 3: rows 2600-4000 (dark)
    img[2600:4000, :, :] = 50

    return img


@pytest.fixture
def sample_image_no_gaps():
    """
    Create an image with no clear gaps (should trigger fallback).

    Returns:
        numpy array (H, W, 3) of uniform gray image
    """
    img = np.ones((2000, 800, 3), dtype=np.uint8) * 100  # Gray
    return img


@pytest.fixture
def sample_image_with_edges():
    """
    Create an image with edge features for testing edge detection.

    Returns:
        numpy array with drawn lines and shapes
    """
    img = np.ones((3000, 800, 3), dtype=np.uint8) * 255  # White background

    # Panel 1: Draw some lines/rectangles
    cv2.rectangle(img, (100, 100), (700, 900), (0, 0, 0), 2)
    cv2.line(img, (100, 500), (700, 500), (0, 0, 0), 3)

    # Gap: rows 1000-1300 (white) - already white

    # Panel 2: More shapes
    cv2.circle(img, (400, 1800), 150, (0, 0, 0), 3)
    cv2.rectangle(img, (200, 1400), (600, 2200), (0, 0, 0), 2)

    # Gap: rows 2300-2500 (white) - already white

    # Panel 3: Text-like features
    cv2.putText(img, "Test Content", (200, 2700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img


@pytest.fixture
def save_test_image(temp_dir):
    """
    Helper fixture to save test images to temp directory.

    Returns:
        Function that takes an image and filename, returns Path to saved file
    """
    def _save(image: np.ndarray, filename: str) -> Path:
        """Save image to temp directory and return path."""
        filepath = temp_dir / filename
        cv2.imwrite(str(filepath), image)
        return filepath

    return _save


@pytest.fixture
def mock_episode_structure(temp_dir):
    """
    Create a mock episode directory structure for integration tests.

    Creates:
        temp_dir/
            ep001/
                page_0.jpg
                page_1.jpg
                page_2.jpg

    Returns:
        Path to temp directory
    """
    ep_dir = temp_dir / "ep001"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 simple test pages
    for i in range(3):
        # Create a simple gradient image
        img = np.ones((1200, 800, 3), dtype=np.uint8) * (50 + i * 50)

        # Add some content
        cv2.rectangle(img, (100, 100), (700, 1100), (0, 0, 0), 2)
        cv2.putText(
            img,
            f"Page {i}",
            (300, 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3
        )

        page_path = ep_dir / f"page_{i}.jpg"
        cv2.imwrite(str(page_path), img)

    return temp_dir
