# Hand Jumper Panel Database

A searchable, browsable database of all panels from the **Hand Jumper** webtoon by SLEEPACROSS. Features comprehensive tagging by character, location, and lore details, designed for easy deployment as a static website.

![License](https://img.shields.io/badge/license-Educational%20Use-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![JavaScript](https://img.shields.io/badge/javascript-vanilla-yellow)

## ✨ Features

- 🔍 **Powerful Search**: Fuzzy search with Fuse.js across characters, locations, tags, and keywords
- 🏷️ **Advanced Filtering**: Filter by season, episode range, characters, locations, and custom tags
- 🖼️ **Dual Image Storage**: Maintains both archival-quality originals (highest quality) and web-optimized versions
- 📱 **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- ⚡ **Performance Optimized**: Lazy loading, pagination, and compressed assets for fast loading
- 🎨 **Manual Tagging Interface**: Comprehensive UI for adding detailed metadata to panels
- 📊 **Browse by Category**: Browse panels by episode, character, or location
- 🚀 **Static Deployment**: No backend required—deploys directly to GitHub Pages
- 🎯 **Hybrid Tagging**: Automated basic tags (season, episode) + manual detailed tags (emotions, plot points)

## 📁 Project Structure

```
webtoon-database/
├── scraper/                    # Python processing scripts
│   ├── scrape_webtoon.py      # Downloads episodes from webtoons.com
│   ├── extract_panels.py      # Extracts individual panels using OpenCV
│   ├── optimize_images.py     # Creates web-optimized versions
│   ├── scrape_wiki.py         # Scrapes wiki for character/lore data
│   └── build_database.py      # Builds final JSON database
│
├── data/                       # Generated data (not in git)
│   ├── originals/             # High-quality original episode images
│   ├── panels_original/       # Extracted panels (highest quality)
│   ├── panels_web/            # Web-optimized panels (~100-200KB each)
│   ├── panel_metadata.json    # Panel extraction metadata
│   ├── wiki_categories.json   # Scraped wiki data
│   └── manual_tags/           # Manual tag JSON files
│
├── frontend/                   # Static website (deploy this)
│   ├── index.html             # Search interface
│   ├── browse.html            # Browse by episode/character/location
│   ├── tagging.html           # Manual tagging interface
│   ├── about.html             # Project information
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript (search, browse, tagging)
│   ├── data/                  # JSON database files
│   │   ├── panels_database.json
│   │   ├── panels_s1.json
│   │   ├── panels_s2.json
│   │   └── wiki_data.json
│   └── images/                # Web-optimized panel images
│       ├── s1/ep001/
│       └── s2/ep001/
│
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
└── DEPLOYMENT.md             # Detailed deployment guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- A modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/hand-jumper-database.git
   cd hand-jumper-database
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the scraping pipeline** (in order)
   ```bash
   cd scraper

   # Step 1: Download episodes (may take a while)
   python scrape_webtoon.py

   # Step 2: Extract individual panels
   python extract_panels.py

   # Step 3: Create web-optimized versions
   python optimize_images.py

   # Step 4: Scrape wiki data for autocomplete
   python scrape_wiki.py

   # Step 5: Build the final database
   python build_database.py
   ```

4. **Copy web-optimized images to frontend**
   ```bash
   # Windows
   xcopy /E /I ..\data\panels_web ..\frontend\images

   # Linux/Mac
   cp -r ../data/panels_web/* ../frontend/images/
   ```

5. **Open the site locally**
   - Open `frontend/index.html` in your browser
   - Or use a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     # Visit http://localhost:8000
     ```

## 🏷️ Tagging Panels

1. Open `frontend/tagging.html` in your browser
2. Select a panel from the left sidebar
3. Add tags:
   - **Characters**: Who appears in the panel
   - **Locations**: Where the scene takes place
   - **Tags**: General tags (dialogue, action, essence-use, etc.)
   - **Emotions**: Emotional tone of the panel
   - **Notes**: Additional context
4. Click "Save & Next" to tag multiple panels quickly
5. Tagged data is saved as JSON files in `data/manual_tags/`
6. Run `build_database.py` again to incorporate new tags

## 🌐 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick deploy to GitHub Pages:**

1. Push your repository to GitHub
2. Enable GitHub Pages in repository settings
3. Deploy the `frontend/` directory
4. Your site will be live at `https://YOUR-USERNAME.github.io/REPO-NAME/`

**Note**: Due to GitHub's file size limits, you may need to use Git LFS for image files. See DEPLOYMENT.md for details.

## 🛠️ Technology Stack

### Backend (Python)
- **httpx** + **asyncio**: Async HTTP requests for fast scraping
- **BeautifulSoup4**: HTML parsing
- **OpenCV**: Panel detection and extraction
- **Pillow**: Image optimization and processing
- **tqdm**: Progress bars

### Frontend
- **HTML/CSS/JavaScript**: Vanilla, no frameworks
- **Fuse.js**: Fuzzy search library
- **Intersection Observer API**: Lazy loading for images
- **LocalStorage**: Temporary tag storage

### Data Storage
- **JSON**: Lightweight, static-friendly database format
- **File-based images**: Organized by season/episode

## 📊 Database Schema

Each panel in the database contains:

```json
{
  "id": "s1_ep001_p001",
  "season": 1,
  "episode": 1,
  "panelNumber": 1,
  "imagePath": "images/s1/ep001/s1_ep001_p001.jpg",
  "originalPath": "data/panels_original/s1/ep001/s1_ep001_p001.jpg",
  "dimensions": { "width": 800, "height": 1200 },
  "automated": {
    "detected": [],
    "confidence": {}
  },
  "manual": {
    "characters": ["Sayeon Lee", "Ryujin Kang"],
    "locations": ["Aberrant Corps HQ"],
    "organizations": ["Aberrant Corps"],
    "items": ["Essence Cuff"],
    "tags": ["dialogue", "essence-use"],
    "emotions": ["determined"],
    "plotPoints": [],
    "dialogue": true,
    "action": false,
    "notes": "First appearance of Sayeon's gift"
  },
  "metadata": {
    "tagged": true,
    "verified": false,
    "lastModified": "2025-01-15T10:30:00Z"
  }
}
```

## 🤝 Contributing

This is a personal portfolio project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## ⚖️ Legal Notice

**Hand Jumper** is created by **SLEEPACROSS** and published by **LINE Webtoon**. All rights to the original work belong to the creator and publisher.

This database is an **unofficial fan project** created for:
- Educational purposes
- Portfolio demonstration
- Archival and reference
- Non-commercial use only

This project is protected under fair use for commentary and analysis. If you are the copyright holder and have concerns, please contact me and I will address them promptly.

## 📚 Resources

- [Read Hand Jumper on Webtoon](https://www.webtoons.com/en/action/hand-jumper/list?title_no=2702)
- [Hand Jumper Wiki](https://hand-jumper.fandom.com/)
- [Deployment Guide](DEPLOYMENT.md)

## 📝 License

This project is for educational use only. The codebase itself is MIT licensed, but all Hand Jumper content belongs to SLEEPACROSS and LINE Webtoon.

---

Made with ❤️ by a Hand Jumper fan
