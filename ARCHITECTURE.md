## Scalable Architecture for High Feature Velocity

Built for rapid feature development without technical debt.

---

## Design Philosophy

This architecture prioritizes **feature velocity** - how quickly you can add new capabilities without breaking existing code. Key principles:

1. **Plugin-based**: Add new ML models without touching core code
2. **Configuration-driven**: Change behavior via config files, not code edits
3. **Feature flags**: Enable/disable features instantly
4. **Clean separation**: Domain models, analysis logic, and storage are independent
5. **Pipeline orchestration**: Manage complex workflows declaratively

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│  (cli.py - Unified interface for all operations)           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                   Core System                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Models     │  │   Pipeline   │  │Feature Flags │     │
│  │ (Domain)     │  │(Orchestrator)│  │  (Config)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  Plugin System                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │        AnalyzerPlugin (Base Interface)           │      │
│  └──────────────────┬───────────────────────────────┘      │
│                     │                                       │
│  ┌──────────────────┼───────────────────────────────┐      │
│  │                  │                               │      │
│  ▼                  ▼                               ▼      │
│  Character      Emotion       Dialogue    ...more plugins  │
│  Analyzer       Analyzer       Analyzer                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Domain Models (`core/models.py`)

**Pure data structures** with no business logic. Easy to serialize, validate, and extend.

```python
@dataclass
class Panel:
    id: str
    episode: str
    ai_analysis: Optional[AnalysisResult]
    user_tags: List[Tag]

@dataclass
class Character:
    name: str
    confidence: float
    bbox: Optional[BoundingBox]
```

**Why this matters:**
- ✅ Add new fields without breaking existing code
- ✅ Easy to serialize to JSON/database
- ✅ Type-safe with dataclasses
- ✅ Clear separation between data and behavior

### 2. Plugin System (`core/analyzer_plugin.py`)

**Add new ML models** by implementing a simple interface:

```python
class YourNewAnalyzer(AnalyzerPlugin):
    @property
    def name(self) -> str:
        return "your_analyzer"

    def analyze(self, image: np.ndarray, panel_path: Path) -> AnalysisResult:
        # Your ML logic here
        return AnalysisResult(...)
```

**Benefits:**
- ✅ No core code changes needed
- ✅ Automatic dependency checking
- ✅ Enable/disable via feature flags
- ✅ Easy to A/B test different models

**Example**: Adding a new pose estimation analyzer takes <100 lines:

```python
# analyzers/pose_analyzer.py
class MediaPipePoseAnalyzer(ActionAnalyzerPlugin):
    def detect_actions(self, image, characters):
        # Use MediaPipe to detect poses
        poses = self.mediapipe.process(image)
        return [Action(...) for pose in poses]

# That's it! Register and use:
register_plugin(MediaPipePoseAnalyzer())
```

### 3. Pipeline Orchestration (`core/pipeline.py`)

**Declare workflows** instead of coding them:

```python
pipeline = (PipelineBuilder()
    .add_character_detection(FaceRecognitionAnalyzer())
    .add_emotion_detection(DeepFaceEmotionAnalyzer())  # Depends on characters
    .add_dialogue_detection(TesseractOCRAnalyzer())
    .add_visual_analysis(ColorAnalyzer())
    .configure(max_workers=4, continue_on_error=True)
    .build()
)

# Process panels
panels = pipeline.process_panels(panels_list, parallel=True)
```

**Features:**
- ✅ Automatic dependency resolution (emotions need characters first)
- ✅ Parallel execution where possible
- ✅ Error recovery (continue on failure)
- ✅ Timeout handling
- ✅ Progress tracking

### 4. Feature Flags (`core/feature_flags.py`)

**Control features at runtime** without deployments:

```python
flags = get_feature_flags()

if flags.is_enabled('emotion_detection'):
    pipeline.add_emotion_detection(analyzer)

# Or via CLI:
# ./cli.py feature emotion_detection --state disabled
```

**Use cases:**
- ✅ Experiment with new models
- ✅ Gradual rollouts
- ✅ A/B testing
- ✅ Quick bug mitigation (disable broken feature)

---

## Adding New Features

### Example: Add Object Detection

**1. Create Plugin** (5 minutes)

```python
# analyzers/yolo_object_detector.py
from core.analyzer_plugin import AnalyzerPlugin
from core.models import AnalysisResult

class YOLOObjectDetector(AnalyzerPlugin):
    @property
    def name(self) -> str:
        return "yolo_object_detector"

    @property
    def dependencies(self) -> List[str]:
        return ['ultralytics']

    def analyze(self, image, panel_path) -> AnalysisResult:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(image)

        # Convert to our format
        objects = [...]
        return AnalysisResult(metadata={'objects': objects})
```

**2. Register Plugin** (1 minute)

```python
# analyzers/__init__.py
from .yolo_object_detector import YOLOObjectDetector

__all__ = [..., 'YOLOObjectDetector']
```

**3. Add to Pipeline** (2 minutes)

```python
# cli.py - add to process command
if flags.is_enabled('object_detection'):
    builder.add_custom_analyzer(
        'object_detection',
        YOLOObjectDetector(flags.get_config('object_detection'))
    )
```

**4. Enable Feature** (30 seconds)

```bash
./cli.py feature object_detection --state enabled
```

**Total time: ~10 minutes** to add a complete new ML capability!

---

## Extending the System

### Add New Tag Category

```python
# core/models.py
class TagCategory(Enum):
    # Existing...
    OBJECT = "object"  # Just add new value!

# Automatically works everywhere
tag = Tag(category=TagCategory.OBJECT, value="sword", ...)
```

### Add New Pipeline Stage

```python
class CustomStage(PipelineStage):
    def execute(self, context):
        # Access previous results
        characters = context.intermediate_results['character_detection']

        # Do custom processing
        result = your_custom_logic(characters)

        # Store for next stage
        context.intermediate_results['custom_stage'] = result
        return StageResult(...)

pipeline.add_stage('custom', CustomStage(), dependencies=['character_detection'])
```

### Add New CLI Command

```python
# cli.py
@cli.command()
@click.option('--param', help='Description')
def your_command(param):
    """Your command description."""
    # Implementation
    click.echo("Running your command...")
```

---

## Configuration System

### Feature Flags (`config/features.json`)

```json
{
  "face_recognition": {
    "state": "enabled",
    "priority": 100,
    "config": {
      "tolerance": 0.6,
      "model": "hog"
    }
  },
  "emotion_detection": {
    "state": "experimental",
    "config": {
      "confidence_threshold": 0.7
    }
  }
}
```

### Analyzer Configuration

Each analyzer gets its own config from feature flags:

```python
class YourAnalyzer(AnalyzerPlugin):
    def __init__(self, config):
        super().__init__(config)
        self.threshold = self.config.get('threshold', 0.5)
        self.model_name = self.config.get('model', 'default')
```

Change behavior without code:

```bash
./cli.py feature your_analyzer --state enabled
# Edit config/features.json to adjust parameters
./cli.py process  # Uses new config!
```

---

## Testing Strategy

### Unit Tests for Plugins

```python
def test_character_analyzer():
    analyzer = FaceRecognitionAnalyzer({'tolerance': 0.6})
    image = load_test_image()

    result = analyzer.analyze(image, test_path)

    assert len(result.characters) > 0
    assert result.characters[0].confidence > 0.6
```

### Integration Tests for Pipeline

```python
def test_full_pipeline():
    pipeline = (PipelineBuilder()
        .add_character_detection(MockCharacterAnalyzer())
        .add_emotion_detection(MockEmotionAnalyzer())
        .build()
    )

    panels = [create_test_panel()]
    results = pipeline.process_panels(panels)

    assert results[0].ai_analysis.characters
    assert results[0].ai_analysis.emotions
```

### Mocking for Fast Tests

```python
class MockAnalyzer(AnalyzerPlugin):
    def analyze(self, image, path):
        # Return fake data instantly
        return AnalysisResult(characters=[Character(...)])
```

---

## Performance Optimization

### Parallel Processing

```python
# Process panels in parallel
pipeline.process_panels(panels, parallel=True)

# Control workers
PipelineBuilder().configure(max_workers=8)
```

### Lazy Loading

Plugins check dependencies only when needed:

```python
def analyze(self, image, path):
    if not hasattr(self, '_model'):
        self._model = load_heavy_model()  # Load once
    return self._model.process(image)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_character_encoding(character_name):
    # Cached per character
    return load_encoding(character_name)
```

---

## Error Handling

### Graceful Degradation

```python
# If one analyzer fails, others continue
PipelineBuilder().configure(continue_on_error=True)
```

### Dependency Checking

```python
class YourAnalyzer(AnalyzerPlugin):
    @property
    def dependencies(self) -> List[str]:
        return ['some_library']

    def check_dependencies(self) -> bool:
        # Automatically checked before running
        return super().check_dependencies()
```

### Logging

```python
from core import get_logger

logger = get_logger(__name__)
logger.info("Processing panel...")
logger.warning("Low confidence: %.2f", conf)
logger.error("Failed to detect", exc_info=True)
```

---

## CLI Usage

```bash
# Show system info and dependencies
./cli.py info

# List all features
./cli.py features

# Process all panels
./cli.py process

# Process specific episode
./cli.py process --episode ep001

# Process in serial (debugging)
./cli.py process --serial

# Adjust parallelism
./cli.py process --workers 8

# Enable/disable features
./cli.py feature emotion_detection --state disabled
./cli.py feature new_feature --state experimental

# Validate database
./cli.py validate
```

---

## Migration Path

### From Old System

Old monolithic code:
```python
def process_panel(panel):
    # 500 lines of mixed logic
    characters = detect_chars(...)
    emotions = detect_emotions(...)
    dialogue = extract_text(...)
    # ... all in one function
```

New plugin-based:
```python
# Each concern is a separate plugin
pipeline = (PipelineBuilder()
    .add_character_detection(CharAnalyzer())
    .add_emotion_detection(EmotionAnalyzer())
    .add_dialogue_detection(DialogueAnalyzer())
    .build()
)

panels = pipeline.process_panels(panels)
```

### Compatibility

Old scripts can still work via adapters:

```python
# old_script.py
from core.pipeline import Pipeline
from legacy_adapter import adapt_old_function

pipeline = Pipeline()
pipeline.add_stage('legacy', adapt_old_function(old_detect_panels))
```

---

## Benefits Summary

### For Developers

- **Add features in minutes**, not days
- **No merge conflicts** - plugins are independent
- **Easy testing** - mock individual components
- **Clear code organization** - know where to look
- **Type safety** - catch errors at development time

### For System

- **Maintainable** - each plugin is <200 lines
- **Extensible** - add new capabilities without changing core
- **Debuggable** - clear separation of concerns
- **Performant** - parallel execution, lazy loading
- **Resilient** - errors in one plugin don't break others

### For Users

- **Faster iterations** - ship features quickly
- **Better quality** - easier to test and validate
- **More features** - low cost to add new capabilities
- **Reliability** - isolated failures, graceful degradation

---

## Next Steps

1. **Add your first plugin**
   - Pick an ML model you want to integrate
   - Create analyzer in `analyzers/`
   - Register and test

2. **Configure features**
   - Edit `config/features.json`
   - Enable/disable analyzers
   - Adjust parameters

3. **Process panels**
   - Run `./cli.py process`
   - Check results in database

4. **Iterate**
   - Add more analyzers
   - Improve existing ones
   - A/B test different configurations

---

**This architecture scales with your ambitions, not against them.**
