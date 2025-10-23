# Comprehensive System Overhaul Plan

**Goal**: Transform the panel database into a sophisticated, AI-powered tagging system with user interaction and active learning.

---

## Current State Analysis

### ❌ Problems Identified
1. **Panel Splitting**: Episodes 24, 76, 98 have only 2-3 panels (should be 50+)
2. **No Real Tagging**: Database shows 0% tagged despite documentation claiming 68.6%
3. **Wrong Metadata**: Panel dimensions in metadata don't match actual files
4. **Incomplete Pipeline**: Only ep000 fully processed, rest partially done

### ✅ What's Working
- Panel extraction works correctly for ep000 (47 panels properly split)
- Images are web-optimized and accessible
- Basic frontend exists
- Configuration system in place

---

## Phase 1: Fix Panel Splitting 🔧

### Problem
Current brightness-based detection fails on:
- Dense layouts with no white gaps
- Continuous art panels
- Gray dividing lines (threshold too strict)

### Solution: Multi-Strategy Detection
```python
Strategy 1: Brightness Analysis (current - works for 80% of panels)
Strategy 2: Edge Density Detection (find content vs empty regions)
Strategy 3: Semantic Segmentation (ML-based boundary detection)
Strategy 4: Adaptive Thresholding (adjust per-image instead of fixed 245)
Strategy 5: Manual Split Points (for problematic episodes)
```

### Implementation
- Create `SmartPanelSplitter` class
- Try strategies in order until successful
- Validate splits don't cut through content
- Re-process episodes 24, 76, 98, and others with <10 panels

---

## Phase 2: Comprehensive ML Tagging 🤖

### Tag Categories

#### 1. **Characters** (Face Recognition + Body Detection)
```json
{
  "characters": [
    {
      "name": "Sayeon",
      "confidence": 0.92,
      "bbox": [100, 50, 200, 400],
      "face_visible": true,
      "body_visible": true,
      "pose": "standing"
    }
  ]
}
```

**Models**: face_recognition, YOLOv8 person detection

#### 2. **Dialogue/Text** (OCR)
```json
{
  "dialogue": [
    {
      "text": "There is no balance. Only suffering.",
      "bbox": [50, 20, 300, 80],
      "confidence": 0.95,
      "speaker": "Sayeon",
      "type": "dialogue"
    }
  ]
}
```

**Models**: Tesseract OCR, EasyOCR (Korean/English)

#### 3. **Emotions** (Facial Expression Analysis)
```json
{
  "emotions": [
    {
      "character": "Sayeon",
      "emotion": "determined",
      "confidence": 0.88,
      "intensity": 0.7,
      "all_emotions": {
        "happy": 0.05,
        "sad": 0.10,
        "angry": 0.20,
        "determined": 0.88
      }
    }
  ]
}
```

**Models**: DeepFace, FER (Facial Expression Recognition)

#### 4. **Actions** (Pose Estimation + Scene Understanding)
```json
{
  "actions": [
    {
      "action": "fighting",
      "characters": ["Sayeon", "Instructor Gyeong"],
      "confidence": 0.85,
      "intensity": 0.9
    }
  ]
}
```

**Models**: MediaPipe Pose, Action Recognition CNN

#### 5. **Scene/Context** (Scene Classification)
```json
{
  "scene": {
    "setting": "indoor",
    "location": "training_room",
    "time_of_day": "day",
    "mood": "intense",
    "confidence": 0.80
  }
}
```

**Models**: ResNet scene classifier, Custom webtoon classifier

#### 6. **Visual Analysis** (Color/Mood)
```json
{
  "visual": {
    "dominant_colors": [[255, 100, 100], [50, 50, 50]],
    "color_palette": "warm",
    "brightness": 0.6,
    "contrast": 0.8,
    "has_special_effects": false,
    "visual_style": "dramatic"
  }
}
```

**Methods**: K-means clustering, histogram analysis

### Complete Tag Structure
```json
{
  "ai_tags": {
    // All ML-generated tags
    "characters": [...],
    "emotions": [...],
    "dialogue": [...],
    "actions": [...],
    "scene": {...},
    "visual": {...},
    "tags": ["Sayeon", "fighting", "intense", "training"],
    "confidence": 0.87,
    "version": "2.0",
    "method": "ml_comprehensive"
  },
  "user_tags": {
    // User-added tags (separate!)
    "characters": ["Sayeon", "Jaehee"],
    "emotions": ["determined"],
    "custom_tags": ["important_plot_point", "character_development"],
    "notes": "This is when Sayeon first shows her true power",
    "corrections": {
      // User corrections to AI tags
      "characters": {"remove": ["Unknown_1"], "add": ["Ryujin"]}
    }
  },
  "merged_tags": {
    // Final tags combining AI + user (for search)
    "characters": ["Sayeon", "Jaehee", "Ryujin"],
    "all_tags": ["Sayeon", "fighting", "intense", "important_plot_point"]
  }
}
```

---

## Phase 3: UI Overhaul 🎨

### New Features

#### 1. **Tag Differentiation**
```html
<div class="tags">
  <!-- AI tags with robot icon -->
  <span class="tag ai-tag">🤖 Sayeon (92%)</span>
  <span class="tag ai-tag">🤖 fighting (85%)</span>

  <!-- User tags with user icon -->
  <span class="tag user-tag">👤 character_development</span>
  <span class="tag user-tag">👤 important_scene</span>

  <!-- Corrected tags with checkmark -->
  <span class="tag corrected-tag">✓ Ryujin</span>
</div>
```

#### 2. **Advanced Tagging Interface**
- Side-by-side AI suggestions vs user tags
- Click AI tag to accept/reject
- Add custom tags with autocomplete
- Batch tag similar panels
- Tag confidence visualization

#### 3. **Image Caching**
```javascript
// Service Worker for aggressive caching
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('panel-images-v1').then((cache) => {
      // Pre-cache first 50 panels
      return cache.addAll(panel_urls);
    })
  );
});

// IndexedDB for metadata caching
const db = await openDB('panel-db', 1, {
  upgrade(db) {
    db.createObjectStore('panels', { keyPath: 'id' });
    db.createObjectStore('tags', { keyPath: 'panel_id' });
  }
});
```

#### 4. **Lazy Loading**
```javascript
// Intersection Observer for infinite scroll
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      loadPanel(entry.target);
    }
  });
});
```

#### 5. **Real-time Search**
- Search by character name
- Search by emotion
- Search by dialogue text
- Search by scene type
- Combine filters (character + emotion + action)

---

## Phase 4: Active Learning System 🧠

### Concept
User corrections improve AI predictions over time.

### Implementation

#### 1. **Feedback Collection**
```json
{
  "panel_id": "s2_ep001_p015",
  "feedback": {
    "timestamp": "2025-10-23T10:30:00Z",
    "user_id": "admin",
    "corrections": {
      "characters": {
        "ai_predicted": ["Unknown_1"],
        "user_corrected": ["Ryujin"],
        "confidence_was": 0.65
      },
      "emotions": {
        "ai_predicted": "neutral",
        "user_corrected": "determined",
        "confidence_was": 0.50
      }
    }
  }
}
```

#### 2. **Learning Loop**
```python
class ActiveLearner:
    def collect_feedback(self, panel_id, corrections):
        # Store user corrections
        self.feedback_db.append({
            'panel_id': panel_id,
            'corrections': corrections,
            'timestamp': datetime.now()
        })

    def retrain_models(self):
        # Every 100 corrections, fine-tune models
        if len(self.feedback_db) >= 100:
            # Extract training data from corrections
            training_data = self.prepare_training_data()

            # Fine-tune character recognition
            self.character_model.fit(training_data)

            # Fine-tune emotion detection
            self.emotion_model.fit(training_data)

            # Update confidence thresholds
            self.adjust_thresholds()

    def suggest_high_impact_tags(self):
        # Find panels where user tagging would help most
        # (low confidence, high importance, inconsistent predictions)
        candidates = []
        for panel in self.database:
            if panel['ai_tags']['confidence'] < 0.6:
                candidates.append(panel)

        return sorted(candidates, key=lambda p: p['importance'])
```

#### 3. **Smart Retagging**
```python
def retag_similar_panels(corrected_panel):
    """When user corrects a panel, find similar panels to retag."""

    # Find visually similar panels
    similar_panels = find_similar_panels(
        corrected_panel,
        threshold=0.85,
        max_results=50
    )

    # Apply learned corrections
    for panel in similar_panels:
        # If AI made same mistake, apply correction
        if panel['ai_tags']['characters'] == corrected_panel['ai_tags']['characters']:
            panel['ai_tags']['characters'] = corrected_panel['user_tags']['characters']
            panel['ai_tags']['confidence'] += 0.1  # Increase confidence
            panel['ai_tags']['learned_from'] = corrected_panel['id']
```

#### 4. **Confidence Boosting**
```python
def update_confidence_scores():
    """Update AI confidence based on user agreement."""

    for panel in database:
        ai_tags = panel['ai_tags']
        user_tags = panel['user_tags']

        # If user agrees with AI tags
        if ai_tags['characters'] == user_tags['characters']:
            ai_tags['confidence'] = min(1.0, ai_tags['confidence'] + 0.1)

        # If user corrects AI tags
        elif user_tags['corrections']:
            ai_tags['confidence'] = max(0.0, ai_tags['confidence'] - 0.2)
```

---

## Implementation Timeline

### Week 1: Panel Splitting Fix
- [ ] Create SmartPanelSplitter with multiple strategies
- [ ] Re-process problematic episodes (24, 76, 98, etc.)
- [ ] Validate all splits visually
- [ ] Update metadata to match actual files

### Week 2: Basic ML Tagging
- [ ] Set up face_recognition pipeline
- [ ] Set up Tesseract OCR for dialogue
- [ ] Set up DeepFace for emotions
- [ ] Process first 100 panels as test

### Week 3: Advanced Tagging
- [ ] Add action detection
- [ ] Add scene classification
- [ ] Add visual analysis
- [ ] Process all panels

### Week 4: UI Overhaul
- [ ] Build new tagging interface
- [ ] Implement AI/user tag distinction
- [ ] Add service worker caching
- [ ] Add lazy loading
- [ ] Improve search functionality

### Week 5: Active Learning
- [ ] Build feedback collection system
- [ ] Implement correction propagation
- [ ] Create retraining pipeline
- [ ] Add suggestion system

---

## Success Metrics

### Panel Splitting
- ✅ All episodes have 30-100 panels (not 2-3)
- ✅ No panels >3000px height
- ✅ Visual inspection confirms no cut-off content

### Tagging Quality
- ✅ 90%+ panels tagged with characters
- ✅ 80%+ panels tagged with emotions
- ✅ 70%+ panels tagged with dialogue
- ✅ 60%+ panels tagged with actions
- ✅ Overall confidence >0.7

### UI Performance
- ✅ Initial load <2 seconds
- ✅ Panel images cached (instant re-load)
- ✅ Smooth infinite scroll
- ✅ Real-time search <100ms

### Active Learning
- ✅ User corrections applied to similar panels
- ✅ AI confidence increases with user agreement
- ✅ Retraining improves accuracy by 10%+

---

## Next Steps

1. **Test one episode end-to-end**
   - Process ep000 with new system
   - Verify panel splitting
   - Check tag quality
   - Review UI improvements

2. **Scale to all episodes**
   - Process remaining 99 episodes
   - Monitor and fix issues
   - Build character database

3. **Launch active learning**
   - Tag 100 panels manually
   - Train initial models
   - Deploy feedback loop

4. **Continuous improvement**
   - Monitor user corrections
   - Retrain models monthly
   - Add new tag categories as needed
