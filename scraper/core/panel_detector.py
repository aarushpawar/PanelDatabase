"""
Unified Panel Detection Module.

Consolidates all panel detection logic into a single, well-tested implementation.
Replaces the duplicated code in extract_panels.py, smart_panel_detection.py,
and stitch_and_extract.py.

Features:
- Multiple detection strategies (strict/standard/aggressive)
- Content-aware splitting with edge validation
- Adaptive overlap to prevent content loss
- Configurable via YAML
- Comprehensive logging and error handling
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .config import get_config
from .logger import get_logger, LoggerMixin


logger = get_logger(__name__)


class DetectionMode(Enum):
    """Panel detection modes with different aggressiveness levels."""
    STRICT = "strict"           # Only split on large gaps (scene transitions)
    STANDARD = "standard"       # Balanced approach (default)
    AGGRESSIVE = "aggressive"   # Split on any detectable gap
    FALLBACK = "fallback"       # Chunking when no gaps detected


@dataclass
class PanelBounds:
    """
    Represents the boundaries of a detected panel.

    Attributes:
        y_start: Starting Y coordinate (top of panel)
        y_end: Ending Y coordinate (bottom of panel)
        x_start: Starting X coordinate (left edge, usually 0)
        x_end: Ending X coordinate (right edge, usually image width)
        confidence: Detection confidence score (0.0-1.0)
        gap_size: Size of gap above this panel (in pixels)
    """
    y_start: int
    y_end: int
    x_start: int = 0
    x_end: int = 0
    confidence: float = 1.0
    gap_size: int = 0

    @property
    def height(self) -> int:
        """Panel height in pixels."""
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        """Panel width in pixels."""
        return self.x_end - self.x_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'x': self.x_start,
            'y': self.y_start,
            'w': self.width,
            'h': self.height,
            'confidence': self.confidence,
            'gap_size': self.gap_size
        }


@dataclass
class Gap:
    """Represents a white gap between panels."""
    start: int        # Y coordinate of gap start
    end: int          # Y coordinate of gap end
    size: int         # Gap height in pixels
    middle: int       # Middle point of gap

    @property
    def is_valid(self) -> bool:
        """Check if gap is valid (not negative)."""
        return self.size > 0 and self.end > self.start


class PanelDetector(LoggerMixin):
    """
    Unified panel detector with multiple strategies.

    Uses brightness-based gap detection with edge validation to find
    panel boundaries in stitched webtoon images.

    Example:
        >>> detector = PanelDetector(mode='standard')
        >>> panels = detector.detect(image_path)
        >>> print(f"Found {len(panels)} panels")
    """

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.STANDARD,
        config_override: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize panel detector.

        Args:
            mode: Detection mode (strict/standard/aggressive/fallback)
            config_override: Optional dict to override config values
        """
        self.mode = mode

        # Load configuration
        self.config = get_config('panel_detection')

        # Apply overrides if provided
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)

        # Load mode-specific parameters
        self._load_parameters()

        self.logger.info(
            f"Initialized PanelDetector (mode={mode.value}, "
            f"min_gap={self.min_gap_height}px, threshold={self.white_threshold})"
        )

    def _load_parameters(self) -> None:
        """Load detection parameters from configuration."""
        # Get mode-specific settings
        mode_config = self.config.get(f'modes.{self.mode.value}', {})

        # Core parameters (with mode overrides)
        self.min_panel_height = mode_config.get(
            'min_panel_height',
            self.config.get('detection.min_panel_height', 200)
        )
        self.min_gap_height = mode_config.get(
            'min_gap_height',
            self.config.get('detection.min_gap_height', 50)
        )
        self.white_threshold = mode_config.get(
            'white_threshold',
            self.config.get('detection.white_threshold', 245)
        )
        self.padding = self.config.get('detection.padding', 5)
        self.max_panel_height = self.config.get('detection.max_panel_height', 3000)

        # Overlap settings
        overlap_config = self.config.get_section('detection').get('overlap', {})
        self.overlap_enabled = overlap_config.get('enabled', True)
        self.overlap_margin_top = overlap_config.get('margin_top', 50)
        self.overlap_margin_bottom = overlap_config.get('margin_bottom', 50)
        self.overlap_adaptive = overlap_config.get('adaptive', True)

        # Edge detection settings
        edge_config = self.config.get_section('detection').get('edge_detection', {})
        self.edge_detection_enabled = edge_config.get('enabled', True)
        self.canny_low = edge_config.get('canny_low', 30)
        self.canny_high = edge_config.get('canny_high', 100)
        self.min_edge_density = edge_config.get('min_edge_density', 50)
        self.content_buffer = edge_config.get('content_buffer', 100)

        # Fallback settings
        fallback_config = self.config.get_section('fallback')
        self.fallback_enabled = fallback_config.get('enabled', True)
        self.fallback_chunk_size = fallback_config.get('chunk_size', 2000)
        self.fallback_search_range = fallback_config.get('search_range', 400)

    def detect(
        self,
        image_path: Path,
        apply_overlap: Optional[bool] = None
    ) -> List[PanelBounds]:
        """
        Detect panels in a stitched webtoon image.

        Args:
            image_path: Path to the stitched episode image
            apply_overlap: Override overlap setting (None = use config)

        Returns:
            List of PanelBounds objects representing detected panels

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If image cannot be loaded or is invalid
        """
        # Load image
        image = self._load_image(image_path)
        height, width = image.shape[:2]

        self.logger.info(
            f"Detecting panels in {image_path.name} ({width}x{height}px)"
        )

        # Step 1: Detect content regions
        has_content = self._detect_content_regions(image)

        # Step 2: Find white gaps
        gaps = self._find_white_gaps(has_content, height)

        self.logger.debug(f"Found {len(gaps)} potential gaps")

        # Step 3: Validate gaps with edge detection
        if self.edge_detection_enabled and gaps:
            gaps = self._validate_gaps_with_edges(image, gaps, has_content)
            self.logger.debug(f"After validation: {len(gaps)} valid gaps")

        # Step 4: Create panel boundaries
        panels = self._create_panels_from_gaps(gaps, height, width)

        # Step 5: Handle very long panels (force split if needed)
        panels = self._split_long_panels(panels, image, has_content)

        # Step 6: Apply overlap if enabled
        use_overlap = apply_overlap if apply_overlap is not None else self.overlap_enabled
        if use_overlap:
            panels = self._apply_overlap(panels, height)

        # Step 7: Validate results
        panels = self._validate_panels(panels, height)

        self.logger.info(f"Detected {len(panels)} panels")

        return panels

    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from file with error handling.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W, 3)

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image cannot be decoded
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        if img.size == 0:
            raise ValueError(f"Image is empty: {image_path}")

        return img

    def _detect_content_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect which rows contain content vs are empty/white.

        Uses both brightness analysis and edge detection for robustness.

        Args:
            image: Input image (H, W, 3)

        Returns:
            Boolean array of length H where True = has content, False = empty
        """
        height = image.shape[0]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Method 1: Brightness check (white = empty)
        brightness = np.mean(gray, axis=1)  # Average brightness per row
        is_white = brightness > self.white_threshold

        # Method 2: Edge detection (catches all drawn lines/content)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        edge_density = np.sum(edges, axis=1)  # Count edge pixels per row
        has_edges = edge_density > self.min_edge_density

        # Combine: A row is "empty" only if it's white AND has no edges
        has_content = ~is_white | has_edges

        return has_content

    def _find_white_gaps(self, has_content: np.ndarray, height: int) -> List[Gap]:
        """
        Find continuous stretches of white/empty rows.

        Args:
            has_content: Boolean array indicating content per row
            height: Image height

        Returns:
            List of Gap objects
        """
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
                    if gap_size >= self.min_gap_height:
                        gaps.append(Gap(
                            start=gap_start,
                            end=gap_end,
                            size=gap_size,
                            middle=(gap_start + gap_end) // 2
                        ))

                    in_gap = False

        # Handle final gap if image ends with white space
        if in_gap:
            gap_end = height
            gap_size = gap_end - gap_start
            if gap_size >= self.min_gap_height:
                gaps.append(Gap(
                    start=gap_start,
                    end=gap_end,
                    size=gap_size,
                    middle=(gap_start + gap_end) // 2
                ))

        return gaps

    def _validate_gaps_with_edges(
        self,
        image: np.ndarray,
        gaps: List[Gap],
        has_content: np.ndarray
    ) -> List[Gap]:
        """
        Validate gaps using edge detection to ensure they're truly empty.

        Args:
            image: Input image
            gaps: List of candidate gaps
            has_content: Content detection array

        Returns:
            List of validated gaps
        """
        height = image.shape[0]
        valid_gaps = []

        for gap in gaps:
            # Check buffer zone around split point
            split_y = gap.middle
            buffer_start = max(0, split_y - self.content_buffer)
            buffer_end = min(height, split_y + self.content_buffer)

            # Check if buffer zone is truly content-free
            buffer_has_content = np.any(has_content[buffer_start:buffer_end])

            if not buffer_has_content:
                valid_gaps.append(gap)
            else:
                self.logger.debug(
                    f"Rejecting gap at y={gap.middle} (content in buffer zone)"
                )

        return valid_gaps

    def _create_panels_from_gaps(
        self,
        gaps: List[Gap],
        height: int,
        width: int
    ) -> List[PanelBounds]:
        """
        Create panel boundaries from detected gaps.

        Args:
            gaps: List of validated gaps
            height: Image height
            width: Image width

        Returns:
            List of PanelBounds
        """
        panels = []
        current_y = 0

        for gap in gaps:
            # Check if panel would be large enough
            panel_height = gap.start - current_y

            if panel_height < self.min_panel_height:
                # Panel too small, skip this gap
                self.logger.debug(
                    f"Skipping small panel (height={panel_height}px at y={current_y})"
                )
                continue

            # Create panel
            panels.append(PanelBounds(
                y_start=current_y,
                y_end=gap.start,
                x_start=0,
                x_end=width,
                confidence=self._calculate_confidence(gap, panel_height),
                gap_size=gap.size
            ))

            current_y = gap.end

        # Add final panel
        if height - current_y >= self.min_panel_height:
            panels.append(PanelBounds(
                y_start=current_y,
                y_end=height,
                x_start=0,
                x_end=width,
                confidence=0.8,  # Lower confidence for last panel
                gap_size=0
            ))
        elif panels:
            # Extend last panel to end of image
            panels[-1].y_end = height
        else:
            # No valid gaps found, treat entire image as one panel (low confidence)
            if self.fallback_enabled:
                self.logger.warning(
                    f"No valid gaps found, using fallback strategy"
                )
                panels = self._fallback_chunking(height, width)
            else:
                # Single panel with low confidence
                panels.append(PanelBounds(
                    y_start=0,
                    y_end=height,
                    x_start=0,
                    x_end=width,
                    confidence=0.3,
                    gap_size=0
                ))

        return panels

    def _calculate_confidence(self, gap: Gap, panel_height: int) -> float:
        """
        Calculate confidence score for a panel split.

        Larger gaps and reasonable panel sizes = higher confidence.

        Args:
            gap: The gap used for splitting
            panel_height: Height of the resulting panel

        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from gap size
        if gap.size > 500:
            confidence = 0.95  # Large gap = very confident
        elif gap.size > 200:
            confidence = 0.85  # Standard gap
        elif gap.size > 100:
            confidence = 0.75  # Moderate gap
        else:
            confidence = 0.60  # Small but valid gap

        # Reduce confidence if panel is too small or too large
        if panel_height < 300:
            confidence *= 0.9
        elif panel_height > 2500:
            confidence *= 0.85

        return min(confidence, 1.0)

    def _split_long_panels(
        self,
        panels: List[PanelBounds],
        image: np.ndarray,
        has_content: np.ndarray
    ) -> List[PanelBounds]:
        """
        Force split very long panels that exceed max height.

        Args:
            panels: List of detected panels
            image: Input image
            has_content: Content detection array

        Returns:
            List with long panels split
        """
        final_panels = []

        for panel in panels:
            if panel.height <= self.max_panel_height:
                final_panels.append(panel)
                continue

            # Panel is too long, need to force split
            self.logger.warning(
                f"Forcing split of long panel (height={panel.height}px)"
            )

            chunk_start = panel.y_start
            while chunk_start < panel.y_end:
                chunk_end = min(chunk_start + self.max_panel_height, panel.y_end)

                # Try to find best split point near target
                best_split = self._find_best_split_point(
                    chunk_start,
                    chunk_end,
                    panel.y_end,
                    image,
                    has_content
                )

                final_panels.append(PanelBounds(
                    y_start=chunk_start,
                    y_end=best_split,
                    x_start=panel.x_start,
                    x_end=panel.x_end,
                    confidence=0.5,  # Lower confidence for forced splits
                    gap_size=0
                ))

                chunk_start = best_split

        return final_panels

    def _find_best_split_point(
        self,
        target_start: int,
        target_end: int,
        max_end: int,
        image: np.ndarray,
        has_content: np.ndarray
    ) -> int:
        """
        Find the best point to split within a range.

        Looks for the brightest/emptiest row near the target split.

        Args:
            target_start: Start of search range
            target_end: End of search range
            max_end: Maximum allowed end (panel boundary)
            image: Input image
            has_content: Content detection array

        Returns:
            Y coordinate of best split point
        """
        search_start = max(target_start, target_end - self.fallback_search_range)
        search_end = min(max_end, target_end + self.fallback_search_range)

        if search_start >= search_end:
            return target_end

        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(image[search_start:search_end], cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray, axis=1)

        # Find brightest (least content) row
        best_idx = np.argmax(brightness)
        best_split = search_start + best_idx

        return best_split

    def _fallback_chunking(self, height: int, width: int) -> List[PanelBounds]:
        """
        Fallback strategy when no gaps detected: split into chunks.

        Args:
            height: Image height
            width: Image width

        Returns:
            List of PanelBounds from chunking
        """
        panels = []
        current_y = 0

        while current_y < height:
            chunk_end = min(current_y + self.fallback_chunk_size, height)

            if chunk_end - current_y >= self.min_panel_height:
                panels.append(PanelBounds(
                    y_start=current_y,
                    y_end=chunk_end,
                    x_start=0,
                    x_end=width,
                    confidence=0.4,  # Low confidence for fallback
                    gap_size=0
                ))

            current_y = chunk_end

        return panels

    def _apply_overlap(
        self,
        panels: List[PanelBounds],
        height: int
    ) -> List[PanelBounds]:
        """
        Apply overlap margins to panels to prevent content loss.

        Args:
            panels: List of panels
            height: Image height

        Returns:
            List of panels with overlap applied
        """
        if len(panels) <= 1:
            return panels

        for i, panel in enumerate(panels):
            # Determine overlap size (adaptive based on confidence)
            if self.overlap_adaptive:
                if panel.confidence > 0.85:
                    overlap_top = self.overlap_margin_top // 2  # Less overlap for high confidence
                    overlap_bottom = self.overlap_margin_bottom // 2
                elif panel.confidence < 0.6:
                    overlap_top = self.overlap_margin_top * 2  # More overlap for low confidence
                    overlap_bottom = self.overlap_margin_bottom * 2
                else:
                    overlap_top = self.overlap_margin_top
                    overlap_bottom = self.overlap_margin_bottom
            else:
                overlap_top = self.overlap_margin_top
                overlap_bottom = self.overlap_margin_bottom

            # Apply top overlap (extend upward into previous panel)
            if i > 0:
                panel.y_start = max(0, panel.y_start - overlap_top)

            # Apply bottom overlap (extend downward into next panel)
            if i < len(panels) - 1:
                panel.y_end = min(height, panel.y_end + overlap_bottom)

        return panels

    def _validate_panels(
        self,
        panels: List[PanelBounds],
        height: int
    ) -> List[PanelBounds]:
        """
        Validate and clean up detected panels.

        Args:
            panels: List of panels to validate
            height: Image height

        Returns:
            Validated list of panels
        """
        valid_panels = []

        for i, panel in enumerate(panels):
            # Ensure bounds are within image
            panel.y_start = max(0, panel.y_start)
            panel.y_end = min(height, panel.y_end)

            # Ensure panel is not empty or invalid
            if panel.height < self.min_panel_height:
                self.logger.warning(
                    f"Discarding invalid panel {i+1} (height={panel.height}px)"
                )
                continue

            valid_panels.append(panel)

        # Check for quality issues
        validation_config = self.config.get_section('validation')
        if validation_config.get('enabled', True):
            min_panels = validation_config.get('min_panels', 1)
            max_panels = validation_config.get('max_panels', 200)

            if len(valid_panels) < min_panels:
                self.logger.warning(
                    f"Too few panels detected: {len(valid_panels)} < {min_panels}"
                )

            if len(valid_panels) > max_panels:
                self.logger.warning(
                    f"Unusually many panels detected: {len(valid_panels)} > {max_panels}"
                )

        return valid_panels
