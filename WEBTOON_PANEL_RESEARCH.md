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
