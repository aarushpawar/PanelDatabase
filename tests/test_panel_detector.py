"""
Unit tests for unified panel detector.

Tests the core functionality of the PanelDetector class including:
- Detection modes (strict/standard/aggressive)
- Gap finding
- Edge detection
- Overlap application
- Confidence scoring
- Fallback behavior
- Error handling
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from scraper.core.panel_detector import (
    PanelDetector,
    DetectionMode,
    PanelBounds,
    Gap
)


class TestPanelBounds:
    """Test the PanelBounds dataclass."""

    def test_panel_bounds_creation(self):
        """Test creating a PanelBounds object."""
        panel = PanelBounds(
            y_start=100,
            y_end=500,
            x_start=0,
            x_end=800,
            confidence=0.85,
            gap_size=200
        )

        assert panel.y_start == 100
        assert panel.y_end == 500
        assert panel.confidence == 0.85

    def test_panel_bounds_properties(self):
        """Test computed properties."""
        panel = PanelBounds(
            y_start=100,
            y_end=500,
            x_start=0,
            x_end=800
        )

        assert panel.height == 400
        assert panel.width == 800

    def test_panel_bounds_to_dict(self):
        """Test converting PanelBounds to dictionary."""
        panel = PanelBounds(
            y_start=100,
            y_end=500,
            x_start=0,
            x_end=800,
            confidence=0.85,
            gap_size=200
        )

        d = panel.to_dict()

        assert d['x'] == 0
        assert d['y'] == 100
        assert d['w'] == 800
        assert d['h'] == 400
        assert d['confidence'] == 0.85
        assert d['gap_size'] == 200


class TestGap:
    """Test the Gap dataclass."""

    def test_gap_creation(self):
        """Test creating a Gap object."""
        gap = Gap(start=100, end=200, size=100, middle=150)

        assert gap.start == 100
        assert gap.end == 200
        assert gap.size == 100
        assert gap.middle == 150

    def test_gap_is_valid(self):
        """Test gap validation."""
        valid_gap = Gap(start=100, end=200, size=100, middle=150)
        assert valid_gap.is_valid

        invalid_gap = Gap(start=200, end=100, size=-100, middle=150)
        assert not invalid_gap.is_valid


class TestPanelDetector:
    """Test the main PanelDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = PanelDetector(mode=DetectionMode.STANDARD)

        assert detector.mode == DetectionMode.STANDARD
        assert detector.min_panel_height > 0
        assert detector.white_threshold > 0

    def test_detector_with_config_override(self):
        """Test detector can override config values."""
        detector = PanelDetector(
            mode=DetectionMode.STANDARD,
            config_override={'detection.min_gap_height': 100}
        )

        assert detector.min_gap_height == 100

    def test_detect_panels_with_clear_gaps(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test detection with clear white gaps."""
        # Save test image
        img_path = save_test_image(sample_image_with_gaps, "test_gaps.jpg")

        # Detect panels
        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path, apply_overlap=False)

        # Should detect 3 panels
        assert len(panels) == 3

        # Check first panel (approximately)
        assert panels[0].y_start == 0
        assert 900 < panels[0].y_end < 1100

        # Check second panel
        assert 1100 < panels[1].y_start < 1300
        assert 2300 < panels[1].y_end < 2500

        # Check third panel
        assert 2500 < panels[2].y_start < 2700
        assert panels[2].y_end == 4000

    def test_detect_panels_with_overlap(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test that overlap is applied correctly."""
        img_path = save_test_image(sample_image_with_gaps, "test_overlap.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels_no_overlap = detector.detect(img_path, apply_overlap=False)
        panels_with_overlap = detector.detect(img_path, apply_overlap=True)

        # With overlap, panels should extend into each other
        assert panels_with_overlap[0].y_end > panels_no_overlap[0].y_end
        assert panels_with_overlap[1].y_start < panels_no_overlap[1].y_start
        assert panels_with_overlap[1].y_end > panels_no_overlap[1].y_end
        assert panels_with_overlap[2].y_start < panels_no_overlap[2].y_start

    def test_fallback_mode(self, sample_image_no_gaps, save_test_image):
        """Test fallback chunking when no gaps detected."""
        img_path = save_test_image(sample_image_no_gaps, "test_no_gaps.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path)

        # Should use fallback chunking (2000px image = 1 chunk)
        assert len(panels) >= 1

        # Fallback panels should have lower confidence
        assert all(p.confidence < 0.7 for p in panels)

    def test_confidence_scoring(self, sample_image_with_gaps, save_test_image):
        """Test confidence scores are assigned correctly."""
        img_path = save_test_image(sample_image_with_gaps, "test_confidence.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path)

        # All panels should have confidence between 0 and 1
        for panel in panels:
            assert 0.0 <= panel.confidence <= 1.0

        # Large gaps should have high confidence
        for panel in panels:
            if panel.gap_size > 200:
                assert panel.confidence > 0.7

    def test_strict_mode(self, sample_image_with_gaps, save_test_image):
        """Test strict mode only splits on large gaps."""
        img_path = save_test_image(sample_image_with_gaps, "test_strict.jpg")

        # Standard mode
        standard_detector = PanelDetector(mode=DetectionMode.STANDARD)
        standard_panels = standard_detector.detect(img_path)

        # Strict mode (requires larger gaps)
        strict_detector = PanelDetector(mode=DetectionMode.STRICT)
        strict_panels = strict_detector.detect(img_path)

        # Strict mode should find same or fewer panels
        assert len(strict_panels) <= len(standard_panels)

    def test_aggressive_mode(self, sample_image_with_gaps, save_test_image):
        """Test aggressive mode splits on smaller gaps."""
        img_path = save_test_image(sample_image_with_gaps, "test_aggressive.jpg")

        # Standard mode
        standard_detector = PanelDetector(mode=DetectionMode.STANDARD)
        standard_panels = standard_detector.detect(img_path)

        # Aggressive mode (detects smaller gaps)
        aggressive_detector = PanelDetector(mode=DetectionMode.AGGRESSIVE)
        aggressive_panels = aggressive_detector.detect(img_path)

        # Aggressive mode should find same or more panels
        assert len(aggressive_panels) >= len(standard_panels)

    def test_invalid_image_path(self):
        """Test error handling for nonexistent image."""
        detector = PanelDetector()

        with pytest.raises(FileNotFoundError):
            detector.detect(Path("nonexistent.jpg"))

    def test_edge_detection_validation(
        self,
        sample_image_with_edges,
        save_test_image
    ):
        """Test that edge detection validates gaps properly."""
        img_path = save_test_image(sample_image_with_edges, "test_edges.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path)

        # Should detect multiple panels
        assert len(panels) > 0

        # All panels should be valid
        for panel in panels:
            assert panel.height >= detector.min_panel_height

    def test_panel_dimensions_within_image(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test that all panels are within image bounds."""
        img_path = save_test_image(sample_image_with_gaps, "test_bounds.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path, apply_overlap=True)

        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]

        # Check all panels are within bounds
        for panel in panels:
            assert panel.y_start >= 0
            assert panel.y_end <= height
            assert panel.x_start >= 0
            assert panel.x_end <= width

    def test_no_overlapping_panels(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test panels without overlap don't overlap each other."""
        img_path = save_test_image(sample_image_with_gaps, "test_no_overlap.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path, apply_overlap=False)

        # Check no overlaps (end of panel i <= start of panel i+1)
        for i in range(len(panels) - 1):
            assert panels[i].y_end <= panels[i + 1].y_start

    def test_panels_cover_full_image(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test that panels cover the full image (no gaps)."""
        img_path = save_test_image(sample_image_with_gaps, "test_coverage.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path, apply_overlap=False)

        # Load image to get height
        img = cv2.imread(str(img_path))
        height = img.shape[0]

        # First panel should start at 0
        assert panels[0].y_start == 0

        # Last panel should end at image height
        assert panels[-1].y_end == height

    def test_long_panel_splitting(self, save_test_image):
        """Test that very long panels are split."""
        # Create an extra long panel (6000px)
        long_img = np.ones((6000, 800, 3), dtype=np.uint8) * 100
        img_path = save_test_image(long_img, "test_long.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path)

        # Should be split into multiple panels
        assert len(panels) > 1

        # No panel should exceed max_panel_height
        for panel in panels:
            assert panel.height <= detector.max_panel_height

    def test_minimum_panel_height_respected(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test that panels below minimum height are discarded."""
        img_path = save_test_image(sample_image_with_gaps, "test_min_height.jpg")

        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path)

        # All panels should meet minimum height
        for panel in panels:
            assert panel.height >= detector.min_panel_height


class TestMigrationWrappers:
    """Test the migration wrapper functions for backward compatibility."""

    def test_detect_panels_simple_wrapper(
        self,
        sample_image_with_gaps,
        save_test_image
    ):
        """Test the simple detection wrapper."""
        from scraper.migrate_panel_detection import detect_panels_simple

        img_path = save_test_image(sample_image_with_gaps, "test_wrapper.jpg")

        panels, image = detect_panels_simple(img_path)

        # Should return list of dicts and image array
        assert isinstance(panels, list)
        assert isinstance(image, np.ndarray)
        assert len(panels) > 0

        # Check dict format
        for panel in panels:
            assert 'x' in panel
            assert 'y' in panel
            assert 'w' in panel
            assert 'h' in panel

    def test_detect_panels_with_content_awareness_wrapper(
        self,
        sample_image_with_gaps
    ):
        """Test the content-aware detection wrapper."""
        from scraper.migrate_panel_detection import (
            detect_panels_with_content_awareness
        )

        panels = detect_panels_with_content_awareness(sample_image_with_gaps)

        # Should return list of (y_start, y_end) tuples
        assert isinstance(panels, list)
        assert len(panels) > 0

        for start, end in panels:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert end > start

    def test_detect_panel_gaps_wrapper(self, sample_image_with_gaps):
        """Test the gap detection wrapper."""
        from scraper.migrate_panel_detection import detect_panel_gaps

        panels = detect_panel_gaps(sample_image_with_gaps)

        # Should return list of (y_start, y_end) tuples
        assert isinstance(panels, list)
        assert len(panels) > 0

        for start, end in panels:
            assert end > start


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_detection_workflow(
        self,
        sample_image_with_gaps,
        save_test_image,
        temp_dir
    ):
        """Test complete detection and extraction workflow."""
        # Save test image
        img_path = save_test_image(sample_image_with_gaps, "full_test.jpg")

        # Detect panels
        detector = PanelDetector(mode=DetectionMode.STANDARD)
        panels = detector.detect(img_path, apply_overlap=True)

        # Load image
        img = cv2.imread(str(img_path))

        # Extract each panel
        for i, panel in enumerate(panels):
            # Extract panel region
            panel_img = img[panel.y_start:panel.y_end, panel.x_start:panel.x_end]

            # Save panel
            panel_path = temp_dir / f"panel_{i:03d}.jpg"
            cv2.imwrite(str(panel_path), panel_img)

            # Verify panel was saved and has correct dimensions
            assert panel_path.exists()

            saved_img = cv2.imread(str(panel_path))
            assert saved_img.shape[0] == panel.height
            assert saved_img.shape[1] == panel.width
