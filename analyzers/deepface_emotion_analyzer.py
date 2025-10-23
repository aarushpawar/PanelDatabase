"""
DeepFace Emotion Analyzer Plugin

Uses DeepFace for emotion detection.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from core.models import Character, Emotion
from core.analyzer_plugin import EmotionAnalyzerPlugin
from core.logger import get_logger

logger = get_logger(__name__)


class DeepFaceEmotionAnalyzer(EmotionAnalyzerPlugin):
    """Emotion detection using DeepFace."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.backend = self.config.get('backend', 'opencv')

    @property
    def name(self) -> str:
        return "deepface_emotion_detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ['deepface']

    def detect_emotions(self, image: np.ndarray, characters: List[Character]) -> List[Emotion]:
        """Detect emotions for each character's face."""
        if not self.check_dependencies():
            logger.warning("DeepFace not available, skipping emotion detection")
            return []

        from deepface import DeepFace

        emotions = []

        for char in characters:
            if not char.bbox:
                continue

            # Extract face region
            x, y, w, h = char.bbox.x, char.bbox.y, char.bbox.width, char.bbox.height
            face_img = image[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            try:
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend=self.backend
                )

                if result:
                    emotion_scores = result[0]['emotion']
                    dominant = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant] / 100

                    if confidence >= self.confidence_threshold:
                        emotions.append(Emotion(
                            character=char.name,
                            emotion=dominant.lower(),
                            confidence=confidence,
                            intensity=confidence,
                            distribution=emotion_scores
                        ))

            except Exception as e:
                logger.warning(f"Emotion detection failed for {char.name}: {e}")

        return emotions
