"""
Visual Analyzer Plugin

Analyzes visual properties like colors, brightness, contrast.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from core.models import VisualProperties
from core.analyzer_plugin import VisualAnalyzerPlugin
from core.logger import get_logger

logger = get_logger(__name__)


class ColorAnalyzer(VisualAnalyzerPlugin):
    """Analyzes color and visual properties."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.num_colors = self.config.get('num_colors', 5)
        self.sample_rate = self.config.get('sample_rate', 10)

    @property
    def name(self) -> str:
        return "color_visual_analyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ['sklearn']

    def analyze_visuals(self, image: np.ndarray) -> VisualProperties:
        """Analyze visual properties."""
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image)

        # Calculate brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray) / 255.0)
        contrast = float(np.std(gray) / 128.0)

        # Determine color palette
        palette = self._determine_palette(dominant_colors)

        return VisualProperties(
            dominant_colors=dominant_colors,
            brightness=brightness,
            contrast=contrast,
            color_palette=palette
        )

    def _extract_dominant_colors(self, image: np.ndarray) -> List[List[int]]:
        """Extract dominant colors using K-means."""
        if not self.check_dependencies():
            return []

        from sklearn.cluster import KMeans

        # Reshape and sample
        pixels = image.reshape(-1, 3)
        pixels = pixels[::self.sample_rate]

        # K-means clustering
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Convert to list of RGB values
        colors = [list(map(int, color)) for color in kmeans.cluster_centers_]
        return colors

    def _determine_palette(self, colors: List[List[int]]) -> str:
        """Determine overall color palette type."""
        if not colors:
            return "unknown"

        avg_color = np.mean(colors, axis=0)
        r, g, b = avg_color

        if r > g + 20 and r > b + 20:
            return "warm"
        elif b > r + 20 and b > g + 20:
            return "cool"
        elif abs(r - g) < 10 and abs(g - b) < 10:
            return "monochrome"
        else:
            return "balanced"
