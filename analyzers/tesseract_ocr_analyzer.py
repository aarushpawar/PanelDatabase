"""
Tesseract OCR Analyzer Plugin

Extracts text/dialogue using Tesseract OCR.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import cv2

from core.models import DialogueEntry, BoundingBox
from core.analyzer_plugin import DialogueAnalyzerPlugin
from core.logger import get_logger

logger = get_logger(__name__)


class TesseractOCRAnalyzer(DialogueAnalyzerPlugin):
    """Text extraction using Tesseract OCR."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.confidence_threshold = self.config.get('confidence_threshold', 60)
        self.languages = self.config.get('languages', 'eng+kor')
        self.psm = self.config.get('psm', 6)

    @property
    def name(self) -> str:
        return "tesseract_ocr_detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def dependencies(self) -> List[str]:
        return ['pytesseract']

    def detect_dialogue(self, image: np.ndarray) -> List[DialogueEntry]:
        """Extract text using OCR."""
        if not self.check_dependencies():
            logger.warning("Tesseract not available, skipping OCR")
            return []

        import pytesseract

        # Preprocess for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        dialogue = []

        try:
            # Run OCR
            data = pytesseract.image_to_data(
                thresh,
                lang=self.languages,
                config=f'--psm {self.psm}',
                output_type=pytesseract.Output.DICT
            )

            # Extract text blocks
            for i, text in enumerate(data['text']):
                if not text.strip():
                    continue

                conf = float(data['conf'][i])
                if conf < self.confidence_threshold:
                    continue

                dialogue.append(DialogueEntry(
                    text=text.strip(),
                    confidence=conf / 100,
                    bbox=BoundingBox(
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    )
                ))

        except Exception as e:
            logger.warning(f"OCR failed: {e}")

        return dialogue
