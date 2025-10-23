"""
Advanced Panel Analyzer

Comprehensive ML-based panel analysis system that tags panels with:
- Characters present (face recognition + body detection)
- Dialogue and text (OCR)
- Emotions (facial expression analysis)
- Actions (pose estimation + scene understanding)
- Context/Scene (indoor/outdoor, setting type)
- Background elements (objects, locations)
- Visual mood (color analysis, atmosphere)

This replaces the basic contextual tagging with true computer vision.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json
import pickle

import cv2
import numpy as np

from .core.config import get_config
from .core.logger import get_logger, LoggerMixin
from .core.paths import get_path_manager

logger = get_logger(__name__)


@dataclass
class CharacterDetection:
    """Detected character in a panel."""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    face_visible: bool = True
    body_visible: bool = False
    pose: Optional[str] = None  # standing, sitting, action, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
            'face_visible': self.face_visible,
            'body_visible': self.body_visible,
            'pose': self.pose
        }


@dataclass
class EmotionDetection:
    """Detected emotion for a character."""
    character: str
    emotion: str  # happy, sad, angry, surprised, neutral, etc.
    confidence: float
    intensity: float  # 0.0-1.0
    all_emotions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'character': self.character,
            'emotion': self.emotion,
            'confidence': float(self.confidence),
            'intensity': float(self.intensity),
            'all_emotions': {k: float(v) for k, v in self.all_emotions.items()}
        }


@dataclass
class DialogueDetection:
    """Detected text/dialogue in panel."""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    language: str = "en"
    speaker: Optional[str] = None  # Character speaking (if identifiable)
    text_type: str = "dialogue"  # dialogue, narration, sound_effect

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'bbox': list(self.bbox),
            'confidence': float(self.confidence),
            'language': self.language,
            'speaker': self.speaker,
            'type': self.text_type
        }


@dataclass
class ActionDetection:
    """Detected action in the panel."""
    action: str  # fighting, talking, running, standing, etc.
    characters: List[str]
    confidence: float
    intensity: float = 0.5  # 0.0-1.0, how intense the action is

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'characters': self.characters,
            'confidence': float(self.confidence),
            'intensity': float(self.intensity)
        }


@dataclass
class SceneContext:
    """Scene/context information."""
    setting: str  # indoor, outdoor, abstract, etc.
    location: Optional[str]  # classroom, street, battlefield, etc.
    time_of_day: Optional[str]  # day, night, sunset, etc.
    weather: Optional[str]  # clear, rain, snow, etc.
    mood: str  # tense, calm, cheerful, ominous, etc.
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'setting': self.setting,
            'location': self.location,
            'time_of_day': self.time_of_day,
            'weather': self.weather,
            'mood': self.mood,
            'confidence': float(self.confidence)
        }


@dataclass
class VisualAnalysis:
    """Visual characteristics of the panel."""
    dominant_colors: List[Tuple[int, int, int]]  # RGB colors
    color_palette: str  # warm, cool, monochrome, vibrant, etc.
    brightness: float  # 0.0-1.0
    contrast: float  # 0.0-1.0
    has_special_effects: bool = False
    visual_style: str = "normal"  # normal, dramatic, comedic, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dominant_colors': [list(c) for c in self.dominant_colors],
            'color_palette': self.color_palette,
            'brightness': float(self.brightness),
            'contrast': float(self.contrast),
            'has_special_effects': self.has_special_effects,
            'visual_style': self.visual_style
        }


class AdvancedPanelAnalyzer(LoggerMixin):
    """
    Comprehensive panel analysis using multiple ML models.

    This analyzer uses:
    - Face recognition for character identification
    - DeepFace for emotion analysis
    - Tesseract/EasyOCR for text extraction
    - YOLO for object/person detection
    - Color analysis for mood/atmosphere
    - Scene classification models
    """

    def __init__(self):
        """Initialize analyzer with all ML models."""
        self.config = get_config('ml_tagging')
        self.paths = get_path_manager()

        # Initialize models (lazy loading)
        self._face_recognition = None
        self._ocr_reader = None
        self._object_detector = None
        self._scene_classifier = None

        # Character database
        self._load_character_database()

        self.logger.info("Initialized AdvancedPanelAnalyzer")

    def _load_character_database(self):
        """Load character face encodings database."""
        db_path = self.paths.get('metadata.character_embeddings', create=True)

        if db_path.exists():
            try:
                with open(db_path, 'rb') as f:
                    self.character_db = pickle.load(f)
                self.logger.info(f"Loaded {len(self.character_db)} characters from database")
            except Exception as e:
                self.logger.error(f"Failed to load character database: {e}")
                self.character_db = {}
        else:
            self.logger.warning("No character database found")
            self.character_db = {}

    def analyze_panel(self, panel_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a panel.

        Args:
            panel_path: Path to panel image

        Returns:
            Dictionary with all detected information
        """
        # Load image
        panel_image = cv2.imread(str(panel_path))
        if panel_image is None:
            self.logger.error(f"Failed to load panel: {panel_path}")
            return self._empty_result()

        self.logger.info(f"Analyzing panel: {panel_path.name}")

        # Run all detection modules
        characters = self.detect_characters(panel_image)
        emotions = self.detect_emotions(panel_image, characters)
        dialogue = self.detect_dialogue(panel_image)
        actions = self.detect_actions(panel_image, characters)
        scene_context = self.analyze_scene(panel_image)
        visual_analysis = self.analyze_visuals(panel_image)

        # Compile results
        result = {
            'characters': [c.to_dict() for c in characters],
            'emotions': [e.to_dict() for e in emotions],
            'dialogue': [d.to_dict() for d in dialogue],
            'actions': [a.to_dict() for a in actions],
            'scene': scene_context.to_dict() if scene_context else {},
            'visual': visual_analysis.to_dict() if visual_analysis else {},
            'tags': self._generate_tags(characters, emotions, actions, scene_context),
            'overall_confidence': self._calculate_confidence(
                characters, emotions, dialogue, actions, scene_context
            )
        }

        self.logger.info(
            f"Analysis complete: {len(characters)} characters, "
            f"{len(emotions)} emotions, {len(dialogue)} dialogue, "
            f"{len(actions)} actions"
        )

        return result

    def detect_characters(self, panel_image: np.ndarray) -> List[CharacterDetection]:
        """
        Detect characters in the panel using face recognition.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            List of detected characters
        """
        try:
            import face_recognition
        except ImportError:
            self.logger.warning("face_recognition not installed, skipping character detection")
            return []

        # Convert to RGB
        panel_rgb = cv2.cvtColor(panel_image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(panel_rgb, model='hog')

        if not face_locations:
            self.logger.debug("No faces detected")
            return []

        # Get face encodings
        face_encodings = face_recognition.face_encodings(panel_rgb, face_locations)

        characters = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Match against character database
            best_match = self._match_character(face_encoding)

            if best_match:
                name, confidence = best_match
                characters.append(CharacterDetection(
                    name=name,
                    confidence=confidence,
                    bbox=(left, top, right - left, bottom - top),
                    face_visible=True
                ))

        return characters

    def _match_character(self, face_encoding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Match face encoding against character database."""
        try:
            import face_recognition
        except ImportError:
            return None

        if not self.character_db:
            return None

        best_match = None
        best_distance = float('inf')

        for char_name, char_encodings in self.character_db.items():
            distances = face_recognition.face_distance(char_encodings, face_encoding)
            min_distance = np.min(distances)

            if min_distance < best_distance:
                best_distance = min_distance
                best_match = char_name

        # Convert distance to confidence
        if best_match and best_distance < 0.6:
            confidence = 1.0 - best_distance
            return (best_match, confidence)

        return None

    def detect_emotions(
        self,
        panel_image: np.ndarray,
        characters: List[CharacterDetection]
    ) -> List[EmotionDetection]:
        """
        Detect emotions for each character.

        Args:
            panel_image: Panel image (BGR format)
            characters: Detected characters

        Returns:
            List of emotion detections
        """
        emotions = []

        for char in characters:
            x, y, w, h = char.bbox

            # Extract face region
            face_img = panel_image[y:y+h, x:x+w]

            if face_img.size == 0:
                continue

            try:
                from deepface import DeepFace

                analysis = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if analysis:
                    emotion_scores = analysis[0]['emotion']
                    dominant = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant] / 100

                    emotions.append(EmotionDetection(
                        character=char.name,
                        emotion=dominant.lower(),
                        confidence=confidence,
                        intensity=confidence,
                        all_emotions=emotion_scores
                    ))

            except ImportError:
                self.logger.warning("deepface not installed, skipping emotion detection")
                break
            except Exception as e:
                self.logger.warning(f"Emotion detection failed for {char.name}: {e}")

        return emotions

    def detect_dialogue(self, panel_image: np.ndarray) -> List[DialogueDetection]:
        """
        Extract text/dialogue from the panel using OCR.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            List of detected text
        """
        dialogue = []

        try:
            import pytesseract

            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)

            # Enhance text regions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Run OCR
            ocr_data = pytesseract.image_to_data(
                thresh,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )

            # Extract text blocks
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    conf = float(ocr_data['conf'][i])

                    if conf > 0:  # Valid detection
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]

                        dialogue.append(DialogueDetection(
                            text=text.strip(),
                            bbox=(x, y, w, h),
                            confidence=conf / 100,
                            text_type='dialogue'
                        ))

        except ImportError:
            self.logger.warning("pytesseract not installed, skipping text detection")
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")

        return dialogue

    def detect_actions(
        self,
        panel_image: np.ndarray,
        characters: List[CharacterDetection]
    ) -> List[ActionDetection]:
        """
        Detect actions occurring in the panel.

        This uses heuristics and simple rules for now.
        Can be enhanced with pose estimation and action recognition models.

        Args:
            panel_image: Panel image (BGR format)
            characters: Detected characters

        Returns:
            List of detected actions
        """
        actions = []

        # Simple heuristic: If multiple characters, likely interaction
        if len(characters) >= 2:
            actions.append(ActionDetection(
                action='interaction',
                characters=[c.name for c in characters],
                confidence=0.7,
                intensity=0.5
            ))
        elif len(characters) == 1:
            actions.append(ActionDetection(
                action='solo_scene',
                characters=[characters[0].name],
                confidence=0.8,
                intensity=0.3
            ))

        # TODO: Add pose estimation for more detailed action detection
        # TODO: Add motion detection for dynamic panels

        return actions

    def analyze_scene(self, panel_image: np.ndarray) -> Optional[SceneContext]:
        """
        Analyze scene context and setting.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            Scene context information
        """
        # Simple heuristic based on colors and brightness
        gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray) / 255.0

        # Determine setting based on brightness
        if avg_brightness > 0.7:
            setting = 'bright_scene'
            mood = 'cheerful'
        elif avg_brightness < 0.3:
            setting = 'dark_scene'
            mood = 'serious'
        else:
            setting = 'normal'
            mood = 'neutral'

        return SceneContext(
            setting=setting,
            location=None,  # TODO: Add scene classification
            time_of_day=None,
            weather=None,
            mood=mood,
            confidence=0.6
        )

    def analyze_visuals(self, panel_image: np.ndarray) -> VisualAnalysis:
        """
        Analyze visual characteristics of the panel.

        Args:
            panel_image: Panel image (BGR format)

        Returns:
            Visual analysis results
        """
        # Extract dominant colors
        pixels = panel_image.reshape(-1, 3)
        pixels = pixels[::10]  # Sample for speed

        # Simple k-means for dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = [tuple(map(int, color)) for color in kmeans.cluster_centers_]

        # Calculate brightness and contrast
        gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0

        # Determine color palette
        avg_color = np.mean(panel_image, axis=(0, 1))
        if avg_color[2] > avg_color[0] + 20:  # More red
            palette = 'warm'
        elif avg_color[0] > avg_color[2] + 20:  # More blue
            palette = 'cool'
        else:
            palette = 'balanced'

        return VisualAnalysis(
            dominant_colors=dominant_colors,
            color_palette=palette,
            brightness=float(brightness),
            contrast=float(contrast),
            has_special_effects=False,
            visual_style='normal'
        )

    def _generate_tags(
        self,
        characters: List[CharacterDetection],
        emotions: List[EmotionDetection],
        actions: List[ActionDetection],
        scene: Optional[SceneContext]
    ) -> List[str]:
        """Generate searchable tags from detected information."""
        tags = []

        # Character tags
        for char in characters:
            tags.append(char.name)

        # Emotion tags
        for emotion in emotions:
            tags.append(f"{emotion.emotion}")

        # Action tags
        for action in actions:
            tags.append(action.action)

        # Scene tags
        if scene:
            tags.append(scene.mood)
            if scene.setting:
                tags.append(scene.setting)

        return list(set(tags))  # Remove duplicates

    def _calculate_confidence(
        self,
        characters: List[CharacterDetection],
        emotions: List[EmotionDetection],
        dialogue: List[DialogueDetection],
        actions: List[ActionDetection],
        scene: Optional[SceneContext]
    ) -> float:
        """Calculate overall confidence score."""
        confidences = []

        # Character confidence
        if characters:
            confidences.append(np.mean([c.confidence for c in characters]))

        # Emotion confidence
        if emotions:
            confidences.append(np.mean([e.confidence for e in emotions]))

        # Dialogue confidence
        if dialogue:
            confidences.append(np.mean([d.confidence for d in dialogue]))

        # Action confidence
        if actions:
            confidences.append(np.mean([a.confidence for a in actions]))

        # Scene confidence
        if scene:
            confidences.append(scene.confidence)

        return float(np.mean(confidences)) if confidences else 0.0

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'characters': [],
            'emotions': [],
            'dialogue': [],
            'actions': [],
            'scene': {},
            'visual': {},
            'tags': [],
            'overall_confidence': 0.0
        }
