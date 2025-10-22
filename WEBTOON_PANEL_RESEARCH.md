# Webtoon/Manhwa Panel Detection Research

## Research Date
October 22, 2025

## Overview
This document contains research findings about webtoon/manhwa panel structure and characteristics to inform the design of the panel detection system for the Hand Jumper database project.

---

## 1. Webtoon Format Fundamentals

### What Makes Webtoons Different from Traditional Comics?

**Vertical Scroll Format:**
- Webtoons use an "infinite canvas" format with panels stacked vertically in one long continuous strip
- Designed specifically for mobile devices and vertical scrolling
- No pagination - reading continues uninterrupted from top to bottom
- Reading direction is strictly "up to down" (unlike traditional manga's Z-pattern)

**Mobile-First Design:**
- Optimized for smartphone screens
- Typically 800px wide (some creators work at 1600px @ 600dpi then downscale)
- Can extend indefinitely in height
- Standard canvas recommendation: ≤800px wide × 1280px long per screen

### Panel Arrangement

**Vertical Stacking:**
- Panels are primarily arranged vertically (one after another)
- 1-3 panels per "screen" is recommended
- Horizontal arrangements limited to max 2 panels side-by-side
- Vulnerable to horizontal movement/action (format limitation)

---

## 2. Panel Spacing & White Space

### Official WEBTOON Guidelines

**Standard Spacing:**
- **Minimum panel-to-panel spacing:** 200px
- **Scene/location transitions:** 600-1000px of vertical white space
- **Top margin:** 300px minimum (header blocks first section)

### Spacing Purposes

**Pacing Control:**
- Closer panels = faster scrolling = quicker pace
- Larger gaps = slower pacing = emphasis/pause
- White space indicates passage of time

**Readability:**
- Prevents clutter and confusion
- Gives readers visual "rest points"
- Emphasizes key moments or actions

**Scene Transitions:**
- Large white spaces (600-1000px) signal major scene changes
- Helps readers mentally prepare for context shifts

### Practical Spacing Recommendations

One common grid system approach:
- Base unit: 400px × 400px squares
- Transition space: 4 sections = ~1600px
- Panel width: 400-600px recommended for comfortable reading

---

## 3. Technical Detection Approaches

### Classical Computer Vision Methods

**Edge Detection (Most Common):**
1. Canny edge detection to find panel borders
2. Edge thickening through dilation to make borders more robust
3. Connected component labeling to find panel blocks
4. Clustering algorithms to merge overlapping bounding boxes

**Recursive Binary Splitting:**
1. Connected component labeling identifies panel blocks
2. Recursive partitioning splits blocks into sub-blocks
3. Optimal splitting lines determined adaptively at each level
4. Works well for digitally-created content with pixel-perfect delineations

**Brightness-Based Detection:**
- Calculate average brightness per horizontal row
- Identify continuous stretches of "white" rows (threshold ~245-250)
- White gaps of sufficient size indicate panel boundaries
- Simple but effective for clean digital webtoons

### Deep Learning Methods

**Object Detection:**
- WEBTOON uses proprietary detection technology
- Detects panels, speech bubbles, and dialogue areas
- Implementations often use Faster R-CNN
- Requires training data (datasets like Webtoon-Manhwa Panels Object Detection exist)

**Key Consideration:**
- Classical methods work well for digitally-created formats
- Deep learning may be overkill for pixel-perfect panel delineations
- Clever heuristics can achieve good results with less complexity

---

## 4. Hand Jumper Specific Analysis

### Observed Panel Characteristics (from metadata)

**Dimensions:**
- Standard width: 800px (consistent across all panels)
- Heights vary dramatically: 446px to 2101px+ observed
- Average panel height: ~900px (estimated)

**Actual Gaps Measured (Episode 0 sample):**
- Panel 1→2 gap: 193px
- Panel 2→3 gap: 177px
- Panel 3→4 gap: 312px (larger - possible emphasis/transition)
- Panel 4→5 gap: 157px
- Panel 5→6 gap: 250px

**Observations:**
- Gaps typically range 150-350px
- Aligns with WEBTOON's 200px minimum guideline
- Larger gaps (300+px) likely indicate scene transitions or dramatic moments
- Spacing is not uniform (intentional pacing variation)

---

## 5. Detection Requirements for Our System

### Critical Features

**Must Handle:**
1. Variable panel heights (200px to 3000px+)
2. Inconsistent gap sizes (150-1000px)
3. Full-width panels (800px standard)
4. White space as primary separator
5. Potential for no clear gaps (fallback needed)

**Must Avoid:**
1. Cutting through panel content
2. Mistaking in-panel white space for gaps
3. Missing small but valid gaps
4. Creating panels that are too large (>3000px)

### Recommended Detection Strategy

**Multi-Stage Approach:**

1. **Primary: White Gap Detection**
   - Scan for horizontal rows with high brightness (>245)
   - Identify continuous white stretches
   - Only split on gaps ≥ MIN_GAP_SIZE (recommend 50-100px)
   - Respect minimum panel height (200px)

2. **Secondary: Edge Detection Validation**
   - Use Canny edge detection to find content boundaries
   - Ensure split points have no detected edges nearby
   - Add buffer zones (50-100px) around potential splits
   - Validate that "white gaps" are truly content-free

3. **Tertiary: Content Analysis**
   - Calculate edge density per row
   - Row is "content" if edges detected OR brightness < threshold
   - Only split where both methods agree it's safe

4. **Fallback: Smart Chunking**
   - If no valid gaps found and panel > MAX_HEIGHT (3000px)
   - Find "least bad" split point within search range
   - Choose brightest/emptiest row available
   - Prefer approximate WEBTOON spacing (600-800px chunks)

### Quality Validation

**Post-Detection Checks:**
- Verify no overlapping panels
- Check panel size distribution (flag anomalies)
- Validate total height matches source image
- Generate confidence scores based on gap clarity

---

## 6. Implementation Recommendations

### Configuration Parameters

```python
# Panel size constraints
MIN_PANEL_HEIGHT = 200      # Minimum valid panel
MAX_PANEL_HEIGHT = 3000     # Force split if exceeded
STANDARD_WIDTH = 800        # Expected panel width

# Gap detection
MIN_GAP_SIZE = 50           # Minimum gap to consider splitting
WHITE_THRESHOLD = 245       # Brightness for "white" rows
SCENE_TRANSITION_GAP = 600  # Large gap = scene change

# Safety buffers
CONTENT_BUFFER = 50         # Pixels around split to verify empty
EDGE_BUFFER = 100           # Buffer zone for edge detection
PADDING = 5                 # Extra pixels to include in panels

# Edge detection
CANNY_LOW = 30              # Lower threshold for edges
CANNY_HIGH = 100            # Upper threshold for edges
MIN_EDGE_DENSITY = 50       # Edge pixels per row = content
```

### Detection Strategies (Priority Order)

1. **Strict Mode** - Only split on clear, large white gaps (600+px)
   - Best for: Episodes with obvious scene transitions
   - Risk: May miss subtle panel breaks

2. **Standard Mode** - Split on validated gaps (200+px)
   - Best for: Most episodes
   - Uses edge detection validation
   - Default recommended approach

3. **Aggressive Mode** - Split on any detectable gap (50+px)
   - Best for: Episodes with minimal spacing
   - Higher risk of false splits

4. **Fallback Mode** - Chunking with smart heuristics
   - Best for: Episodes with no clear gaps
   - Last resort only

### Testing Requirements

**Unit Tests:**
- Test each detection method independently
- Verify gap detection accuracy
- Validate edge detection sensitivity
- Test buffer zone calculations

**Integration Tests:**
- Process known-good episodes
- Compare against ground truth
- Measure precision/recall
- Benchmark performance

**Edge Cases:**
- All-black or all-white panels
- Very tall panels (>2000px)
- No gaps (continuous art)
- Minimal gaps (<100px)
- Nested content (panels within panels)

---

## 7. References

### Key Sources
1. WEBTOON Official Guidelines - Panel spacing recommendations
2. Art Rocket / CLIP STUDIO - Webtoon creation tutorials
3. Academic Papers - Manga/comic panel segmentation research
4. S-Morishita Studio - Webtoon formatting guides

### Related Projects
- ComicPanelSegmentation (GitHub - reidenong)
- WORD-pytorch (Webtoon Object Recognition)
- Roboflow Webtoon-Manhwa Panels Dataset

---

## 8. Conclusion

**Key Takeaways:**

1. **White space is intentional** - Gaps serve narrative and pacing purposes
2. **Variation is expected** - Panel heights and gaps vary by design
3. **Conservative approach preferred** - Better to keep panels together than split incorrectly
4. **Multi-method validation essential** - No single detection method is perfect
5. **Context matters** - Large gaps signal transitions; small gaps maintain flow

**Recommended Approach:**
- Start with brightness-based white gap detection (simple, effective)
- Validate with edge detection (ensure gaps are truly empty)
- Use adaptive thresholds based on gap size distribution
- Implement robust fallback for edge cases
- Generate quality metrics and confidence scores
- Allow manual review/correction of low-confidence splits

**Success Metrics:**
- 95%+ panels correctly identified
- Zero splits through panel content
- Preserves narrative pacing (respects spacing intent)
- Handles 100+ episode corpus reliably

---

## 9. Panel Overlap Strategy (NEW REQUIREMENT)

### Problem Statement
If panel detection splits are slightly inaccurate, content near the boundary could be cut off or split between panels. This makes searching incomplete, as a character or object might be partially in one panel and partially in another.

### Solution: Overlapping Panel Regions

**Concept:**
Instead of panels ending exactly where the next begins, create intentional overlap zones where both adjacent panels include the same content. This ensures:
1. No content is lost due to imprecise splits
2. Search completeness - if something appears near a boundary, it's fully captured in at least one panel
3. Redundancy provides safety margin for detection errors

### Implementation Strategy

**Overlap Parameters:**
```python
OVERLAP_MARGIN_TOP = 50      # Pixels to extend upward into previous panel
OVERLAP_MARGIN_BOTTOM = 50   # Pixels to extend downward into next panel
```

**Example:**
```
Original split points:
Panel 1: y=0 to y=1000
Panel 2: y=1000 to y=2000
Panel 3: y=2000 to y=3000

With overlap:
Panel 1: y=0 to y=1050       (+50px overlap with Panel 2)
Panel 2: y=950 to y=2050     (+50px from Panel 1, +50px into Panel 3)
Panel 3: y=1950 to y=3000    (+50px from Panel 2)
```

### Visual Representation

```
[---- Panel 1 Content ----]
                      [Overlap Zone]
                      [---- Panel 2 Content ----]
                                          [Overlap Zone]
                                          [---- Panel 3 Content ----]
```

### Considerations

**Storage Impact:**
- Minimal increase in file size (~5-10% per panel)
- Overlap zones typically white space anyway
- Benefit of search completeness outweighs storage cost

**Deduplication:**
- Search results may return adjacent panels
- Frontend can implement duplicate detection
- Show context: "This panel also appears in adjacent panel X"

**Optimal Overlap Size:**
- 50-100px recommended (6-12% of minimum panel height)
- Large enough to capture typical objects/characters
- Small enough to not significantly increase file size
- Adjustable based on detection confidence

**Adaptive Overlap:**
```python
if gap_size < 100:
    # Small gap = less confident split
    overlap = 100  # Use larger overlap for safety
elif gap_size > 500:
    # Large gap = very confident split
    overlap = 25   # Minimal overlap needed
else:
    # Standard gap
    overlap = 50   # Default overlap
```

### Benefits

1. **Content Completeness**: Guarantees full capture of all visual elements
2. **Search Reliability**: Objects near boundaries will be fully searchable
3. **Error Tolerance**: Allows for imperfect detection without data loss
4. **Quality Assurance**: Overlap zones can be validated for consistency

---

## 10. Enhanced Emotion Tagging System (NEW REQUIREMENT)

### Current State
The tagging system has an `emotions` field, but it's underpopulated and lacks a comprehensive taxonomy.

### Proposed Emotion Taxonomy

**Primary Emotions (Basic 6):**
- Happy
- Sad
- Angry
- Fearful
- Disgusted
- Surprised

**Extended Emotions (Facial Expressions):**
- Shocked
- Confused
- Worried
- Anxious
- Embarrassed
- Shy
- Blushing
- Crying
- Laughing
- Smiling
- Smirking
- Frowning
- Scowling
- Glaring
- Panicked
- Terrified
- Calm
- Serene
- Determined
- Focused
- Confident
- Proud
- Smug
- Annoyed
- Frustrated
- Tired
- Exhausted
- Excited
- Enthusiastic
- Hopeful
- Hopeless
- Depressed
- Lonely
- Content
- Relieved
- Guilty
- Remorseful
- Jealous
- Envious
- Suspicious
- Distrustful
- Loving
- Affectionate
- Compassionate
- Sympathetic

**Mood/Atmosphere Tags:**
- Tense
- Peaceful
- Chaotic
- Ominous
- Suspenseful
- Dramatic
- Romantic
- Comedic
- Melancholic
- Nostalgic
- Mysterious
- Intense
- Light-hearted
- Dark
- Bittersweet

**Character State Tags:**
- Injured
- Unconscious
- Awakening
- Transforming
- Powered-up
- Weakened
- Sleeping
- Dreaming

**Interaction Emotions:**
- Arguing
- Fighting
- Reconciling
- Comforting
- Threatening
- Protecting
- Bonding
- Confronting

### Implementation Recommendations

**1. Hierarchical Organization:**
```json
{
  "emotion_categories": {
    "basic": ["Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"],
    "expressions": ["Shocked", "Confused", "Worried", ...],
    "moods": ["Tense", "Peaceful", "Chaotic", ...],
    "states": ["Injured", "Unconscious", ...],
    "interactions": ["Arguing", "Fighting", "Reconciling", ...]
  }
}
```

**2. Multi-Select Support:**
- Panels can have multiple emotions (e.g., "Happy + Crying" = tears of joy)
- Characters in same panel can have different emotions
- Support per-character emotion tagging

**3. Enhanced UI for Tagging:**

**Quick Select Buttons (Most Common):**
```
[Happy] [Sad] [Angry] [Shocked] [Determined] [Worried]
```

**Categorized Dropdowns:**
```
Primary Emotions: [Dropdown]
Facial Expressions: [Dropdown with autocomplete]
Mood/Atmosphere: [Dropdown with autocomplete]
```

**Character-Specific Emotions:**
```
Sayeon: [Happy] [Determined] + Add emotion
Jaehee: [Worried] [Protective] + Add emotion
```

**4. Autocomplete Enhancement:**
Update `tagging.js` to include emotion autocomplete:
```javascript
populateDatalist('emotionList', EMOTION_TAXONOMY);
```

**5. Auto-Tagging Support:**

**Expression Detection (ML):**
- Use pre-trained facial expression models
- Detect faces in panels
- Classify expressions automatically
- Generate confidence scores

**Scene Detection:**
- Analyze color palette for mood (dark = ominous, bright = cheerful)
- Detect action lines for "intense" or "chaotic"
- Text analysis of dialogue for emotional context

**6. Search & Filter Integration:**

**Emotion-Based Search:**
- "Show all panels where Sayeon is angry"
- "Find determined facial expressions"
- "Show tense/suspenseful scenes"

**Mood-Based Browsing:**
- Filter by atmosphere (romantic, action, comedic)
- Create mood boards/collections
- Track emotional arcs across episodes

### Data Structure Updates

**Panel Object with Enhanced Emotions:**
```json
{
  "manual": {
    "emotions": {
      "overall": ["Tense", "Dramatic"],
      "characters": {
        "Sayeon": ["Determined", "Angry"],
        "Jaehee": ["Worried", "Protective"]
      },
      "atmosphere": "Ominous"
    }
  },
  "automated": {
    "detected_emotions": {
      "faces": [
        {
          "character": "Unknown",
          "emotion": "Shocked",
          "confidence": 0.87,
          "bbox": [x, y, w, h]
        }
      ],
      "scene_mood": {
        "mood": "Dark",
        "confidence": 0.72
      }
    }
  }
}
```

### Testing & Validation

**Inter-Rater Reliability:**
- Multiple taggers tag same panels
- Measure agreement on emotion labels
- Establish tagging guidelines

**ML Model Validation:**
- Compare automated tags with manual tags
- Calculate precision/recall per emotion
- Fine-tune thresholds

**Search Effectiveness:**
- Test emotion-based searches
- Verify results match expectations
- User feedback on relevance

### Benefits

1. **Enhanced Searchability**: Find specific emotional moments
2. **Story Analysis**: Track character emotional arcs
3. **Scene Discovery**: Find specific moods/atmospheres
4. **Fan Engagement**: Create emotion-based collections
5. **Content Recommendations**: "More scenes like this"

### Migration Plan

1. **Update database schema** to support enhanced emotion structure
2. **Update tagging UI** with new emotion taxonomy
3. **Add emotion autocomplete** with categorization
4. **Implement ML emotion detection** (optional, future enhancement)
5. **Backfill existing panels** with basic emotion tags
6. **Add emotion filters** to search interface
