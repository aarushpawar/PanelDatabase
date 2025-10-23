"""
Complete ML-based tagging pipeline.

Processes all panels with unified detection and ML tagging.
Integrates stitching, panel detection, and automated character/emotion tagging.
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import cv2
import numpy as np
from tqdm import tqdm

from .core.panel_detector import PanelDetector, DetectionMode
from .core.ml_tagger import MLTagger
from .core.paths import get_path_manager
from .core.logger import setup_logging, get_logger, LoggerMixin


logger = get_logger(__name__)


class TaggingPipeline(LoggerMixin):
    """
    End-to-end pipeline for panel detection and ML tagging.

    Processes episodes from original images through to tagged panel database.
    """

    def __init__(self, mode: str = 'standard'):
        """
        Initialize tagging pipeline.

        Args:
            mode: Detection mode (strict/standard/aggressive)
        """
        self.paths = get_path_manager()

        # Initialize detector
        try:
            self.detector = PanelDetector(mode=DetectionMode[mode.upper()])
        except KeyError:
            self.logger.warning(
                f"Invalid mode '{mode}', using STANDARD"
            )
            self.detector = PanelDetector(mode=DetectionMode.STANDARD)

        # Initialize ML tagger
        self.tagger = MLTagger()

        # Setup logging
        log_file = self.paths.get('logs.ml_tagging', create=True)
        setup_logging(log_file, level='INFO')

        self.logger.info(f"Initialized TaggingPipeline (mode={mode})")

    def stitch_episode(self, episode_dir: Path) -> np.ndarray:
        """
        Stitch all page images from an episode into one long vertical image.

        Args:
            episode_dir: Directory containing page_*.jpg files

        Returns:
            Stitched image as numpy array

        Raises:
            ValueError: If no valid images found
        """
        # Get all page images sorted
        page_images = sorted(
            episode_dir.glob('page_*.jpg'),
            key=lambda x: int(x.stem.split('_')[1])
        )

        if not page_images:
            raise ValueError(f"No page images found in {episode_dir}")

        self.logger.info(f"Stitching {len(page_images)} pages from {episode_dir.name}")

        # Load all images
        images = []
        total_height = 0
        max_width = 0

        for img_path in page_images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Failed to read {img_path}")
                continue

            images.append(img)
            total_height += img.shape[0]
            max_width = max(max_width, img.shape[1])

        if not images:
            raise ValueError(f"No valid images loaded from {episode_dir}")

        # Create blank canvas
        stitched = np.zeros((total_height, max_width, 3), dtype=np.uint8)
        stitched.fill(255)  # White background

        # Paste images vertically
        current_y = 0
        for img in images:
            h, w = img.shape[:2]
            # Center image horizontally if needed
            x_offset = (max_width - w) // 2
            stitched[current_y:current_y+h, x_offset:x_offset+w] = img
            current_y += h

        self.logger.info(f"Stitched image size: {max_width}x{total_height}px")

        return stitched

    def extract_panels(
        self,
        stitched_image: np.ndarray,
        stitched_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract panels from stitched image.

        Args:
            stitched_image: Stitched episode image
            stitched_path: Path to save stitched image

        Returns:
            List of panel boundary information
        """
        # Save stitched image
        stitched_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(stitched_path),
            stitched_image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        # Detect panels
        panel_bounds = self.detector.detect(stitched_path, apply_overlap=True)

        self.logger.info(f"Detected {len(panel_bounds)} panels")

        # Convert to dict format with image data
        panels = []
        for i, pb in enumerate(panel_bounds):
            panels.append({
                'index': i + 1,
                'bounds': pb.to_dict(),
                'image': stitched_image[pb.y_start:pb.y_end, pb.x_start:pb.x_end]
            })

        return panels

    def save_panels(
        self,
        panels: List[Dict[str, Any]],
        output_dir: Path,
        episode_name: str
    ) -> List[Path]:
        """
        Save extracted panels to disk.

        Args:
            panels: List of panels with image data
            output_dir: Directory to save panels
            episode_name: Episode identifier (e.g., "ep001")

        Returns:
            List of saved panel paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        panel_paths = []

        for panel in panels:
            idx = panel['index']
            img = panel['image']

            # Generate filename
            filename = f"{episode_name}_p{idx:03d}.jpg"
            panel_path = output_dir / filename

            # Save with high quality
            cv2.imwrite(
                str(panel_path),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

            panel_paths.append(panel_path)

        self.logger.info(f"Saved {len(panel_paths)} panels to {output_dir}")

        return panel_paths

    def process_episode(
        self,
        episode_dir: Path,
        skip_tagging: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process one episode: stitch, detect, extract, and tag.

        Args:
            episode_dir: Directory with page_*.jpg files
            skip_tagging: If True, skip ML tagging (faster)

        Returns:
            List of panel data with tags
        """
        episode_name = episode_dir.name

        self.logger.info(f"Processing episode: {episode_name}")

        # 1. Stitch images
        stitched = self.stitch_episode(episode_dir)

        # 2. Detect and extract panels
        stitched_dir = self.paths.get('data.stitched', create=True)
        stitched_path = stitched_dir / episode_name / "stitched.jpg"

        panels = self.extract_panels(stitched, stitched_path)

        # 3. Save panels
        panels_dir = self.paths.get('data.panels_original', create=True)
        panel_output_dir = panels_dir / episode_name

        panel_paths = self.save_panels(panels, panel_output_dir, episode_name)

        # 4. Tag each panel with ML (if enabled)
        panel_data = []

        if skip_tagging:
            # Just save panel metadata without ML tags
            for path, panel in zip(panel_paths, panels):
                panel_data.append({
                    'path': str(path),
                    'filename': path.name,
                    'episode': episode_name,
                    'panel_number': panel['index'],
                    'dimensions': {
                        'width': panel['bounds']['w'],
                        'height': panel['bounds']['h']
                    },
                    'bounds': panel['bounds'],
                    'tags': {}
                })
        else:
            # Full ML tagging
            for path, panel in tqdm(
                zip(panel_paths, panels),
                desc=f"Tagging {episode_name}",
                leave=False
            ):
                tags = self.tagger.tag_panel(path)

                panel_data.append({
                    'path': str(path),
                    'filename': path.name,
                    'episode': episode_name,
                    'panel_number': panel['index'],
                    'dimensions': {
                        'width': panel['bounds']['w'],
                        'height': panel['bounds']['h']
                    },
                    'bounds': panel['bounds'],
                    'tags': tags
                })

        self.logger.info(
            f"Completed {episode_name}: "
            f"{len(panel_data)} panels processed"
        )

        return panel_data

    def process_all(self, skip_tagging: bool = False) -> Dict[str, Any]:
        """
        Process all episodes.

        Args:
            skip_tagging: If True, skip ML tagging (faster)

        Returns:
            Complete database dictionary
        """
        originals_dir = self.paths.get('data.originals')

        # Find all episode directories
        episodes = sorted([
            d for d in originals_dir.iterdir()
            if d.is_dir() and d.name.startswith('ep')
        ])

        self.logger.info(f"Processing {len(episodes)} episodes")

        all_data = []
        failed_episodes = []

        for ep_dir in tqdm(episodes, desc="Episodes"):
            try:
                ep_data = self.process_episode(ep_dir, skip_tagging=skip_tagging)
                all_data.extend(ep_data)
            except Exception as e:
                self.logger.error(f"Failed to process {ep_dir.name}: {e}", exc_info=True)
                failed_episodes.append(ep_dir.name)

        # Build database
        database = {
            'total_panels': len(all_data),
            'total_episodes': len(episodes),
            'failed_episodes': failed_episodes,
            'panels': all_data
        }

        # Save database
        self._save_database(database)

        self.logger.info(
            f"Complete! Processed {len(all_data)} panels from {len(episodes)} episodes"
        )

        if failed_episodes:
            self.logger.warning(
                f"Failed episodes ({len(failed_episodes)}): "
                f"{', '.join(failed_episodes)}"
            )

        return database

    def _save_database(self, database: Dict[str, Any]) -> None:
        """
        Save panel database to JSON file.

        Args:
            database: Complete database dictionary
        """
        db_path = self.paths.get('frontend.data') / 'panels_database.json'
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Database saved to {db_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run panel detection and tagging pipeline')
    parser.add_argument(
        '--mode',
        choices=['strict', 'standard', 'aggressive'],
        default='standard',
        help='Panel detection mode (default: standard)'
    )
    parser.add_argument(
        '--skip-tagging',
        action='store_true',
        help='Skip ML tagging (faster, only extract panels)'
    )
    parser.add_argument(
        '--episode',
        type=str,
        help='Process specific episode only (e.g., ep001)'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Panel Detection & Tagging Pipeline")
    print("=" * 60)
    print()

    pipeline = TaggingPipeline(mode=args.mode)

    if args.episode:
        # Process single episode
        paths = get_path_manager()
        ep_dir = paths.get('data.originals') / args.episode

        if not ep_dir.exists():
            print(f"❌ Episode not found: {ep_dir}")
        else:
            print(f"Processing episode: {args.episode}")
            panel_data = pipeline.process_episode(ep_dir, skip_tagging=args.skip_tagging)
            print(f"✅ Processed {len(panel_data)} panels")
    else:
        # Process all episodes
        database = pipeline.process_all(skip_tagging=args.skip_tagging)
        print()
        print("=" * 60)
        print("✅ Pipeline complete!")
        print("=" * 60)
        print(f"Total panels: {database['total_panels']}")
        print(f"Total episodes: {database['total_episodes']}")

        if database['failed_episodes']:
            print(f"Failed episodes: {len(database['failed_episodes'])}")
