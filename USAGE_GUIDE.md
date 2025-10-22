# Usage Guide

Complete guide to using the Hand Jumper Panel Database.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running the Scraping Pipeline](#running-the-scraping-pipeline)
3. [Tagging Panels](#tagging-panels)
4. [Using the Search Interface](#using-the-search-interface)
5. [Browsing Panels](#browsing-panels)
6. [Tips and Best Practices](#tips-and-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Time Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python --version  # Should be 3.8 or higher
   pip list          # Check all packages installed
   ```

3. **Understanding the Project Structure**
   - `scraper/` - Python scripts for data collection
   - `data/` - Generated data (ignored by git)
   - `frontend/` - The website you'll deploy

---

## Running the Scraping Pipeline

### Step 1: Download Episodes

```bash
cd scraper
python scrape_webtoon.py
```

**What it does:**
- Fetches all Hand Jumper episodes from webtoons.com
- Saves original high-quality images to `data/originals/`
- Creates metadata in `data/episode_metadata.json`

**Options you can modify in the script:**
- `CONCURRENT_DOWNLOADS` - Number of simultaneous downloads (default: 3)
- `DELAY_BETWEEN_REQUESTS` - Delay in seconds (default: 1.0)
- `MAX_RETRIES` - Retry attempts for failed downloads (default: 3)

**Expected time:** 30-60 minutes depending on internet speed

**Output:**
```
data/originals/
├── s1/
│   ├── ep001/
│   │   ├── page_001.jpg
│   │   ├── page_002.jpg
│   │   └── ...
│   └── ...
└── s2/
    └── ...
```

### Step 2: Extract Individual Panels

```bash
python extract_panels.py
```

**What it does:**
- Uses OpenCV to detect panel boundaries
- Splits each episode page into individual panels
- Saves to `data/panels_original/` (highest quality)
- Creates `data/panel_metadata.json`

**How it works:**
- Detects horizontal white spaces between panels
- Falls back to contour detection if needed
- Treats entire image as one panel if detection fails

**Expected time:** 15-30 minutes

**Output:**
```
data/panels_original/
├── s1/
│   └── ep001/
│       ├── s1_ep001_p001.jpg
│       ├── s1_ep001_p002.jpg
│       └── ...
```

### Step 3: Optimize Images for Web

```bash
python optimize_images.py
```

**What it does:**
- Creates web-optimized versions (~100-200KB each)
- Resizes to max 1200x2000 pixels
- Compresses using progressive JPEG
- Keeps originals untouched

**Settings:**
- `MAX_WIDTH` - 1200px
- `MAX_HEIGHT` - 2000px
- `JPEG_QUALITY` - 85 (adjusts down if needed)
- `TARGET_SIZE_KB` - 200KB

**Expected time:** 10-20 minutes

**Output:**
```
data/panels_web/           # These go to frontend
└── s1/
    └── ep001/
        ├── s1_ep001_p001.jpg  (~150KB)
        ├── s1_ep001_p002.jpg
        └── ...
```

**Metadata saved to:** `data/optimization_metadata.json`

### Step 4: Scrape Wiki Data

```bash
python scrape_wiki.py
```

**What it does:**
- Scrapes Hand Jumper wiki for:
  - Character names
  - Location names
  - Organizations
  - Items
  - Character details and descriptions
- Saves to `data/wiki_categories.json`

**Expected time:** 5-10 minutes

**Output structure:**
```json
{
  "categories": {
    "characters": [
      {
        "name": "Sayeon Lee",
        "url": "...",
        "details": { "description": "..." }
      }
    ],
    "locations": [...],
    "organizations": [...],
    "items": [...]
  },
  "tag_lists": {
    "character_names": ["Sayeon Lee", "Ryujin Kang", ...],
    "location_names": [...],
    ...
  }
}
```

### Step 5: Build the Database

```bash
python build_database.py
```

**What it does:**
- Combines all metadata into final JSON database
- Creates season-specific databases for performance
- Integrates manual tags from `data/manual_tags/`
- Outputs to `frontend/data/`

**Output:**
```
frontend/data/
├── panels_database.json      # Complete database
├── panels_s1.json            # Season 1 only
├── panels_s2.json            # Season 2 only
├── wiki_data.json            # Wiki reference data
└── statistics.json           # Database stats
```

### Step 6: Copy Images to Frontend

```bash
# Windows
xcopy /E /I ..\data\panels_web ..\frontend\images

# Linux/Mac
cp -r ../data/panels_web/* ../frontend/images/
```

**Or create a symbolic link (faster, but won't work for deployment):**
```bash
# Linux/Mac
ln -s ../data/panels_web ../frontend/images

# Windows (requires admin)
mklink /D ..\frontend\images ..\data\panels_web
```

---

## Tagging Panels

### Opening the Tagging Interface

1. Navigate to `frontend/`
2. Open `tagging.html` in a browser
3. OR use a local server:
   ```bash
   cd frontend
   python -m http.server 8000
   # Visit http://localhost:8000/tagging.html
   ```

### Interface Overview

**Left Sidebar:**
- Thumbnail list of all panels
- Filter by season/episode
- Show untagged only

**Center:**
- Large panel preview
- Navigation buttons (previous/next)
- Panel ID and dimensions

**Right Sidebar:**
- Tagging form with categories:
  - Characters (autocomplete from wiki)
  - Locations (autocomplete from wiki)
  - Organizations (autocomplete)
  - Items (autocomplete)
  - General Tags (quick suggestions)
  - Emotions (quick buttons)
  - Checkboxes (dialogue, action)
  - Notes (free text)

### Tagging Workflow

1. **Select a panel** from the left sidebar
2. **Add character tags:**
   - Type character name (autocomplete helps)
   - Press Enter or click "Add"
   - Remove with × button

3. **Add other metadata** similarly

4. **Use quick tag buttons** for common tags:
   - dialogue, action, essence-use, flashback, dramatic

5. **Add emotional context** with emotion buttons

6. **Check boxes** if applicable:
   - ☑ Has Dialogue
   - ☑ Action Scene

7. **Add notes** for plot significance or context

8. **Save:**
   - Click "Save Tags" (Ctrl+S)
   - OR "Save & Next" to move to next panel

### Keyboard Shortcuts

- `←` / `→` - Navigate between panels
- `Ctrl+S` - Save current panel
- `Enter` (in tag inputs) - Add tag

### Tag Export

- Tags are automatically exported as JSON files
- Downloads to your browser's download folder
- Filename: `s1_ep001_p001_tags.json`

**To integrate tags:**
1. Move JSON files to `data/manual_tags/`
2. Re-run `build_database.py`

### Tagging Tips

**For Characters:**
- Include all visible characters
- Be consistent with names (use wiki names)
- Tag background characters if relevant

**For Locations:**
- Use general location names (e.g., "Aberrant Corps HQ")
- Don't over-specify (avoid "Hallway near Room 3B")

**For Tags:**
- Use existing tags when possible
- Create new tags sparingly
- Common useful tags:
  - dialogue, action, essence-use
  - flashback, memory, dream
  - dramatic, comedic, emotional
  - first-appearance, key-moment

**For Notes:**
- Plot significance: "First time Sayeon uses her gift"
- Context: "Continuation of conversation from previous episode"
- Connections: "References events from S1 E5"

---

## Using the Search Interface

### Basic Search

1. Open `frontend/index.html`
2. Type in the search box:
   - Character names: "Sayeon Lee"
   - Locations: "Aberrant Corps"
   - Tags: "essence-use"
   - Keywords: "gift"
3. Click "Search" or press Enter

### Fuzzy Search

The search uses Fuse.js for fuzzy matching:
- Typos are tolerated: "Saeyon" finds "Sayeon"
- Partial matches work: "Ryu" finds "Ryujin Kang"
- Multiple words: "Sayeon action" finds panels with both

### Advanced Filters

Click "Advanced Filters" to show filter panel:

**Season Filter:**
- Select one or multiple seasons
- Use Ctrl+Click to select multiple

**Episode Range:**
- Set minimum and/or maximum episode numbers
- Leave blank for no limit

**Characters:**
- Type and press Enter to add
- Filter shows only panels with ALL selected characters

**Locations:**
- Same as characters
- Shows panels at ALL selected locations

**Quick Tags:**
- Click tag buttons to toggle
- Active tags are highlighted green

**Sort Options:**
- Relevance (default) - by search score
- Episode (Oldest First) - chronological
- Episode (Newest First) - reverse chronological
- Panel Number - by panel number

### Reading Results

**Panel Cards show:**
- Panel image (lazy loaded)
- Panel ID (e.g., "s1_ep001_p001")
- Season, Episode, Panel number
- Top 3 character tags
- Top 2 location tags

**Click a card** to open detail modal:
- Full-size panel image
- All metadata
- Character, location, and general tags
- Notes if available
- Navigation to prev/next panel
- "View Original Quality" button

### Pagination

- First 50 results load automatically
- Click "Load More" to load next 50
- Continues until all results shown

---

## Browsing Panels

### By Episode

1. Open `frontend/browse.html`
2. Select "By Episode" tab
3. Choose a season from dropdown
4. Click an episode number
5. View all panels from that episode

**Use case:** Read through an episode panel-by-panel

### By Character

1. Click "By Character" tab
2. Browse list of all tagged characters
3. Click a character name
4. See all panels featuring that character

**Use case:** Character study, tracking character development

### By Location

1. Click "By Location" tab
2. Browse list of all tagged locations
3. Click a location
4. See all panels at that location

**Use case:** Analyzing setting, world-building

---

## Tips and Best Practices

### Data Management

**Keep originals separate:**
- `data/originals/` - Never modify these
- `data/panels_original/` - Highest quality panels
- `data/panels_web/` - For website only

**Backup important files:**
- `data/manual_tags/` - Your tagging work!
- `data/episode_metadata.json`
- `data/panel_metadata.json`

### Performance Optimization

**Large databases:**
- Split JSON by season (already done)
- Compress images aggressively
- Use lazy loading (already implemented)

**GitHub Pages limits:**
- Max file size: 100MB
- Max repo size: 1GB (soft), 5GB (hard)
- Use Git LFS if needed

### Tagging Strategy

**Start with important panels:**
- Key character moments
- Plot-critical scenes
- Dramatic moments

**Batch tag by category:**
- Tag all panels from one episode
- Tag all panels with one character
- Makes workflow more efficient

**Use consistent terminology:**
- Stick to wiki names
- Maintain a tag list
- Review existing tags before creating new ones

### Deployment

**Before deploying:**
- Test locally first
- Check all images load
- Test search functionality
- Verify all pages work

**When updating:**
- Re-run only changed scripts
- Rebuild database
- Copy new images
- Test before pushing

---

## Troubleshooting

### Common Issues

**Scraper fails to download:**
- Check internet connection
- Verify webtoons.com is accessible
- Reduce `CONCURRENT_DOWNLOADS`
- Increase `DELAY_BETWEEN_REQUESTS`

**Panel extraction poor quality:**
- Webtoons have irregular layouts
- Manual verification needed
- Adjust `MIN_PANEL_HEIGHT` and `MIN_PANEL_WIDTH` in script

**Images too large:**
- Lower `JPEG_QUALITY` in optimize_images.py
- Reduce `MAX_WIDTH` and `MAX_HEIGHT`
- Check `TARGET_SIZE_KB`

**Search not working:**
- Check browser console for errors
- Verify Fuse.js CDN is loading
- Check JSON files are in `frontend/data/`

**Tags not saving:**
- Check browser's download folder
- Ensure LocalStorage is enabled
- Move JSON files to `data/manual_tags/`

**GitHub Pages images not loading:**
- Verify image paths are relative
- Check files are committed to git
- Look for 404 errors in browser console
- May need Git LFS for large files

### Getting Help

**Check console:**
- Browser console (F12) for frontend errors
- Python console for scraper errors

**Verify files:**
- Check JSON files are valid
- Verify image files exist
- Confirm directory structure

**Re-run pipeline:**
- Sometimes easiest to start over
- Keep backups of manual tags!

---

## Advanced Usage

### Customizing the Schema

Edit `build_database.py` to add custom fields:

```python
panel_entry = {
    # ... existing fields ...
    "custom": {
        "myCustomField": "value"
    }
}
```

### Adding Custom Search Fields

Edit `search.js` Fuse configuration:

```javascript
const fuseOptions = {
    keys: [
        // ... existing keys ...
        { name: 'custom.myCustomField', weight: 1.0 }
    ]
}
```

### Styling

- Edit `frontend/css/styles.css` for main theme
- Edit `frontend/css/tagging.css` for tagging interface
- CSS variables in `:root` for easy color changes

### Batch Operations

**Tag all panels in an episode:**
```python
import json
from pathlib import Path

episode = "s1_ep001"
tags = {"characters": ["Sayeon Lee"], "tags": ["dialogue"]}

for panel_file in Path(f"data/manual_tags/").glob(f"{episode}_*.json"):
    with open(panel_file, 'r+') as f:
        data = json.load(f)
        data['manual'].update(tags)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
```

---

Made with ❤️ for Hand Jumper fans
