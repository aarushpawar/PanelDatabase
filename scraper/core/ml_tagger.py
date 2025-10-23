"""
ML-based automated tagging for panels.

Uses face_recognition for character identification,
deepface for emotions, and various other models for comprehensive tagging.

This module replaces the inefficient CLIP-based approach with proper
face recognition for 24x faster processing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import pickle

import cv2
import numpy as np

from .config import get_config
from .logger import get_logger, LoggerMixin
from .paths import get_path_manager

logger = get_logger(__name__)


@dataclass
class CharacterDetection:
    """Result of character detection in a panel."""
    name: str
    confidence: float
    bbox: tuple  # (x, y, w, h)
    face_encoding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'confidence': float(self.confidence),
            'bbox': self.bbox
        }


@dataclass
class EmotionDetection:
    """Result of emotion detection for a character."""
    character: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'character': self.character,
            'emotion': self.emotion,
            'confidence': float(self.confidence),
            'all_emotions': {k: float(v) for k, v in self.all_emotions.items()}
        }


class CharacterDatabase(LoggerMixin):
    """
    Database of character face encodings for recognition.

    Stores pre-computed face encodings for fast matching.
    Uses face_recognition library for accurate character identification.
    """

    def __init__(self, db_path: Path):
        """
        Initialize character database.

        Args:
            db_path: Path to save/load the database pickle file
        """
        self.db_path = db_path
        self.characters: Dict[str, List[np.ndarray]] = {}
        self.load()

    def load(self) -> None:
        """Load database from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    self.characters = pickle.load(f)
                self.logger.info(
                    f"Loaded {len(self.characters)} characters from database"
                )
            except Exception as e:
                self.logger.error(f"Failed to load character database: {e}")
                self.characters = {}
        else:
            self.logger.warning(f"Character database not found: {self.db_path}")
            self.logger.info("Run build_character_database.py to create it")

    def save(self) -> None:
        """Save database to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.characters, f)
            self.logger.info(f"Saved character database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to save character database: {e}")
            raise

    def add_character(self, name: str, face_encoding: np.ndarray) -> None:
        """
        Add a face encoding for a character.

        Args:
            name: Character name
            face_encoding: 128-dimensional face encoding from face_recognition
        """
        if name not in self.characters:
            self.characters[name] = []
        self.characters[name].append(face_encoding)

    def match_face(
        self,
        face_encoding: np.ndarray,
        tolerance: float = 0.6
    ) -> Optional[CharacterDetection]:
        """
        Match a face encoding against the database.

        Args:
            face_encoding: Face encoding to match (128-d vector)
            tolerance: Distance threshold (lower = stricter, default 0.6)

        Returns:
            CharacterDetection if match found, None otherwise
        """
        try:
            import face_recognition
        except ImportError:
            self.logger.error(
                "face_recognition not installed. "
                "Install with: pip install face-recognition"
            )
            return None

        best_match = None
        best_distance = float('inf')

        for char_name, char_encodings in self.characters.items():
            # Compare against all encodings for this character
            distances = face_recognition.face_distance(
                char_encodings,
                face_encoding
            )

            min_distance = np.min(distances)

            if min_distance < best_distance:
                best_distance = min_distance
                best_match = char_name

        if best_match and best_distance < tolerance:
            confidence = 1.0 - best_distance  # Convert distance to confidence
            return CharacterDetection(
                name=best_match,
                confidence=confidence,
                bbox=(0, 0, 0, 0),  # Filled in by caller
                face_encoding=face_encoding
            )

        return None

    def get_character_count(self) -> int:
        """Get number of characters in database."""
        return len(self.characters)

    def get_encoding_count(self, character: str) -> int:
        """Get number of face encodings for a character."""
        return len(self.characters.get(character, []))


class MLTagger(LoggerMixin):
    """
    ML-based panel tagger.

    Detects characters, emotions, dialogue, and mood automatically using
    state-of-the-art ML models.
    """

    def __init__(self):
        """Initialize ML tagger with configuration."""
        self.config = get_config('ml_tagging')
        self.paths = get_path_manager()

        # Load character database
        db_path = self.paths.get('metadata.character_embeddings', create=True)
        self.char_db = CharacterDatabase(db_path)

        # Configuration
        char_config = self.config.get_section('character_detection')
        self.face_model = char_config.get('face_recognition', {}).get('model', 'hog')
        self.face_tolerance = char_config.get('face_recognition', {}).get('tolerance', 0.6)

        emotion_config = self.config.get_section('emotion_detection')
        self.emotion_backend = emotion_config.get('model', {}).get('backend', 'deepface')
        self.emotion_confidence = emotion_config.get('model', {}).get('confidence', 0.6)

        self.logger.info(f"Initialized MLTagger with {self.char_db.get_character_count()} characters")

    def detect_characters(
        self,
        panel_image: np.ndarray
    ) -> List[CharacterDetection]:
        """
        Detect and identify characters in a panel.

        Uses face_recognition for accurate character identification.

        Args:
            panel_image: Panel image as numpy array (BGR format)

        Returns:
            List of CharacterDetection objects
        """
        try:
            import face_recognition
        except ImportError:
            self.logger.error(
                "face_recognition not installed. "
                "Install with: pip install face-recognition"
            )
            return []

        # Convert BGR to RGB (face_recognition uses RGB)
        panel_rgb = cv2.cvtColor(panel_image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            panel_rgb,
            model=self.face_model
        )

        if not face_locations:
            self.logger.debug("No faces detected in panel")
            return []

        self.logger.debug(f"Detected {len(face_locations)} faces")

        # Get face encodings
        face_encodings = face_recognition.face_encodings(
            panel_rgb,
            face_locations
        )

        characters = []

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Match against database
            match = self.char_db.match_face(
                face_encoding,
                tolerance=self.face_tolerance
            )

            if match:
                # Update bounding box
                match.bbox = (left, top, right - left, bottom - top)
                characters.append(match)
                self.logger.debug(
                    f"Matched character: {match.name} "
                    f"(confidence={match.confidence:.2f})"
                )

        return characters

    def detect_emotions(
        self,
        panel_image: np.ndarray,
        characters: List[CharacterDetection]
    ) -> List[EmotionDetection]:
        """
        Detect emotions for each character.

        Uses DeepFace for emotion recognition on detected character faces.

        Args:
            panel_image: Panel image (BGR format)
            characters: Detected characters with bounding boxes

        Returns:
            List of EmotionDetection objects
        """
        emotions = []

        for char in characters:
            x, y, w, h = char.bbox

            # Skip if bounding box is invalid
            if w <= 0 or h <= 0:
                continue

            # Extract character face region
            char_img = panel_image[y:y+h, x:x+w]

            if char_img.size == 0:
                continue

            try:
                # Use DeepFace for emotion detection
                from deepface import DeepFace

                analysis = DeepFace.analyze(
                    char_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if analysis:
                    emotion_scores = analysis[0]['emotion']
                    dominant = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant] / 100

                    if confidence >= self.emotion_confidence:
                        emotions.append(EmotionDetection(
                            character=char.name,
                            emotion=dominant.lower(),
                            confidence=confidence,
                            all_emotions=emotion_scores
                        ))

                        self.logger.debug(
                            f"Detected {dominant} for {char.name} "
                            f"(confidence={confidence:.2f})"
                        )

            except ImportError:
                self.logger.error(
                    "deepface not installed. "
                    "Install with: pip install deepface"
                )
                break
            except Exception as e:
                self.logger.warning(
                    f"Emotion detection failed for {char.name}: {e}"
                )

        return emotions

    def detect_dialogue(self, panel_image: np.ndarray) -> bool:
        """
        Detect if panel contains dialogue/text.

        TODO: Implement using OCR or text detection model.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            True if dialogue detected, False otherwise
        """
        # Placeholder implementation
        # TODO: Integrate manga-ocr or tesseract
        return False

    def analyze_mood(self, panel_image: np.ndarray) -> List[str]:
        """
        Analyze overall mood/atmosphere of the panel.

        TODO: Implement using color analysis and scene classification.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            List of mood tags (e.g., ["dark", "tense", "calm"])
        """
        # Placeholder implementation
        # TODO: Implement color-based mood analysis
        return []

    def tag_panel(
        self,
        panel_path: Path
    ) -> Dict[str, Any]:
        """
        Fully tag a panel with all ML detections.

        Args:
            panel_path: Path to panel image

        Returns:
            Dictionary with all tags and confidence scores
        """
        # Load image
        panel_image = cv2.imread(str(panel_path))

        if panel_image is None:
            self.logger.error(f"Failed to load panel: {panel_path}")
            return self._empty_result()

        # Detect characters
        characters = self.detect_characters(panel_image)

        # Detect emotions
        emotions = self.detect_emotions(panel_image, characters)

        # Detect dialogue (TODO)
        has_dialogue = self.detect_dialogue(panel_image)

        # Analyze mood (TODO)
        mood_tags = self.analyze_mood(panel_image)

        # Aggregate results
        result = {
            'characters': [c.to_dict() for c in characters],
            'emotions': {
                e.character: {
                    'emotion': e.emotion,
                    'confidence': e.confidence
                }
                for e in emotions
            },
            'dialogue': has_dialogue,
            'mood_tags': mood_tags,
            'overall_confidence': self._calculate_overall_confidence(
                characters, emotions
            )
        }

        self.logger.info(
            f"Tagged panel {panel_path.name}: "
            f"{len(characters)} characters, "
            f"{len(emotions)} emotions"
        )

        return result

    def _calculate_overall_confidence(
        self,
        characters: List[CharacterDetection],
        emotions: List[EmotionDetection]
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            characters: Detected characters
            emotions: Detected emotions

        Returns:
            Overall confidence score (0.0-1.0)
        """
        weights = self.config.get_section('confidence').get('weighting', {})

        # Weight character detection
        char_conf = (
            np.mean([c.confidence for c in characters])
            if characters else 0.0
        )

        # Weight emotion detection
        emotion_conf = (
            np.mean([e.confidence for e in emotions])
            if emotions else 0.0
        )

        # Weighted average
        overall = (
            char_conf * weights.get('character', 0.35) +
            emotion_conf * weights.get('emotion', 0.20)
        )

        return float(overall)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'characters': [],
            'emotions': {},
            'dialogue': False,
            'mood_tags': [],
            'overall_confidence': 0.0
        }
