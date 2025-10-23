# Quick Start Guide

Get up and running in 5 minutes.

---

## Installation

```bash
# Core dependencies
pip install opencv-python numpy click

# ML dependencies (optional)
pip install face-recognition deepface pytesseract scikit-learn

# System dependencies for OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-kor

# macOS:
brew install tesseract tesseract-lang
```

---

## Basic Usage

### 1. Check System

```bash
./cli.py info
```

Shows installed dependencies and system status.

### 2. Process Panels

```bash
# Process all panels
./cli.py process

# Process specific episode
./cli.py process --episode ep001

# Use more workers
./cli.py process --workers 8
```

### 3. Manage Features

```bash
# List features
./cli.py features

# Disable emotion detection
./cli.py feature emotion_detection --state disabled

# Enable experimental feature
./cli.py feature action_detection --state experimental
```

### 4. Validate Database

```bash
./cli.py validate
```

---

## Adding a New Analyzer

**1. Create the analyzer:**

```python
# analyzers/my_analyzer.py
from core.analyzer_plugin import AnalyzerPlugin
from core.models import AnalysisResult

class MyAnalyzer(AnalyzerPlugin):
    @property
    def name(self) -> str:
        return "my_analyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def analyze(self, image, panel_path) -> AnalysisResult:
        # Your logic here
        return AnalysisResult(...)
```

**2. Register it:**

```python
# analyzers/__init__.py
from .my_analyzer import MyAnalyzer
__all__ = [..., 'MyAnalyzer']
```

**3. Add to pipeline:**

```python
# cli.py (in process command)
if flags.is_enabled('my_feature'):
    builder.add_custom_analyzer('my_feature', MyAnalyzer())
```

**4. Enable and run:**

```bash
./cli.py feature my_feature --state enabled
./cli.py process
```

---

## Configuration

Edit `config/features.json`:

```json
{
  "my_feature": {
    "state": "enabled",
    "priority": 85,
    "config": {
      "threshold": 0.7,
      "model": "best"
    }
  }
}
```

Your analyzer automatically gets this config:

```python
def __init__(self, config):
    super().__init__(config)
    self.threshold = self.config.get('threshold', 0.5)
```

---

## Project Structure

```
PanelDatabase/
├── core/                    # Core system
│   ├── models.py           # Data models
│   ├── analyzer_plugin.py  # Plugin system
│   ├── pipeline.py         # Orchestration
│   └── feature_flags.py    # Feature flags
│
├── analyzers/              # Analyzer implementations
│   ├── face_recognition_analyzer.py
│   ├── deepface_emotion_analyzer.py
│   └── ...
│
├── cli.py                  # Command-line interface
├── config/                 # Configuration files
└── data/                   # Panel data
```

---

## Common Tasks

### Test a Single Panel

```python
from core.pipeline import PipelineBuilder
from analyzers import FaceRecognitionAnalyzer

pipeline = (PipelineBuilder()
    .add_character_detection(FaceRecognitionAnalyzer())
    .build()
)

panel = Panel(id='test', path='test.jpg', ...)
result = pipeline.process_panel(panel)
print(result.ai_analysis.characters)
```

### Build Character Database

```python
import pickle
import face_recognition

encodings = {}
for char_name in ['Sayeon', 'Jaehee']:
    img = face_recognition.load_image_file(f'{char_name}.jpg')
    encodings[char_name] = face_recognition.face_encodings(img)

with open('data/character_encodings.pkl', 'wb') as f:
    pickle.dump(encodings, f)
```

### Debug Pipeline

```python
from core import setup_logging
setup_logging(level='DEBUG')

# Now all analyzers log debug info
pipeline.process_panels(panels)
```

---

## Troubleshooting

### "Module not found"

```bash
pip install <module-name>
# or
./cli.py info  # Shows missing dependencies
```

### "Character database not found"

Create `data/character_encodings.pkl` with face encodings (see above).

### "OCR not working"

Install Tesseract:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor
```

### Pipeline fails

Enable debugging:
```python
PipelineBuilder().configure(continue_on_error=True)
```

---

## Examples

See `ARCHITECTURE.md` for detailed examples of:
- Adding new analyzers
- Extending the data model
- Creating custom pipeline stages
- A/B testing features

---

**Ready to build? Start with `./cli.py info` and go from there!**
