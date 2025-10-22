"""
Complete Rebuild Pipeline
Properly rebuilds everything from scratch with ML
"""

import subprocess
import sys
from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).parent

def check_requirements():
    """Check if all required libraries are installed"""
    print("Checking requirements...")
    print("=" * 70)

    required = {
        'opencv-python': 'cv2',
        'pytesseract': 'pytesseract',
        'transformers': 'transformers',
        'torch': 'torch',
        'Pillow': 'PIL'
    }

    missing = []
    for package, import_name in required.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print()
        print("Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Note: pytesseract also requires Tesseract-OCR:")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

    print()
    print("✓ All requirements met!")
    return True


def clean_old_data():
    """Delete old panel data to start fresh"""
    print()
    print("=" * 70)
    print("Cleaning old panel data...")
    print("=" * 70)

    dirs_to_clean = [
        SCRIPT_DIR / "data" / "panels_original",
        SCRIPT_DIR / "data" / "panels_web",
        SCRIPT_DIR / "frontend" / "images" / "s2"
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"Deleting {dir_path.relative_to(SCRIPT_DIR)}...")
            shutil.rmtree(dir_path)

    print("✓ Old data cleaned")


def run_smart_panel_detection():
    """Step 1: Extract panels with smart detection"""
    print()
    print("=" * 70)
    print("STEP 1: Smart Panel Detection")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, "scraper/smart_panel_detection.py"],
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        print("ERROR: Panel detection failed")
        return False

    print("✓ Panel detection complete")
    return True


def run_image_optimization():
    """Step 2: Optimize images for web"""
    print()
    print("=" * 70)
    print("STEP 2: Image Optimization")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, "scraper/optimize_images.py"],
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        print("ERROR: Image optimization failed")
        return False

    # Copy to frontend
    print("Copying images to frontend...")
    src = SCRIPT_DIR / "data" / "panels_web"
    dst = SCRIPT_DIR / "frontend" / "images"
    dst.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src / "s2", dst / "s2", dirs_exist_ok=True)

    print("✓ Image optimization complete")
    return True


def run_ml_character_detection():
    """Step 3: ML-based character detection"""
    print()
    print("=" * 70)
    print("STEP 3: ML Character Detection")
    print("=" * 70)
    print()
    print("This will take 30-60 minutes to process all panels...")
    print()

    result = subprocess.run(
        [sys.executable, "scraper/ml_character_detection.py"],
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        print("ERROR: ML character detection failed")
        return False

    print("✓ ML character detection complete")
    return True


def build_database():
    """Step 4: Build final database"""
    print()
    print("=" * 70)
    print("STEP 4: Build Database")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, "scraper/build_database.py"],
        cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        print("ERROR: Database build failed")
        return False

    print("✓ Database build complete")
    return True


def main():
    print("=" * 70)
    print("COMPLETE REBUILD PIPELINE")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Delete old panel data")
    print("  2. Re-extract panels with smart detection (avoiding text/faces)")
    print("  3. Optimize images for web")
    print("  4. Run ML character detection (per-panel, accurate)")
    print("  5. Build final database")
    print()
    print("Total time: ~2-3 hours")
    print()

    # Check requirements
    if not check_requirements():
        print()
        print("Please install missing packages first, then run this script again.")
        return

    print()
    input("Press Enter to start the complete rebuild, or Ctrl+C to cancel...")

    # Execute pipeline
    steps = [
        ("Cleaning", clean_old_data),
        ("Panel Detection", run_smart_panel_detection),
        ("Image Optimization", run_image_optimization),
        ("ML Detection", run_ml_character_detection),
        ("Database Build", build_database)
    ]

    for step_name, step_func in steps:
        if not step_func():
            print()
            print(f"FAILED at step: {step_name}")
            return

    # Success
    print()
    print("=" * 70)
    print("✓ COMPLETE REBUILD SUCCESSFUL!")
    print("=" * 70)
    print()
    print("Your database is now ready with:")
    print("  - Smart panel detection (no cut-off heads/text)")
    print("  - ML character detection (per-panel accuracy)")
    print("  - Optimized web images")
    print("  - Complete searchable database")
    print()
    print("Start the frontend:")
    print("  cd frontend")
    print("  python server.py")
    print()
    print("Then open: http://localhost:8000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Cancelled by user")
