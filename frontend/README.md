# Hand Jumper Panel Database - Frontend

A searchable web interface for browsing and searching Hand Jumper webtoon panels.

## Features

- **Full-text search** - Search panels by characters, locations, tags, and keywords
- **Advanced filters** - Filter by season, episode range, characters, locations, and tags
- **Browse mode** - Browse panels by episode
- **Tagging interface** - Add manual tags to panels
- **High-quality images** - View original quality panels
- **Responsive design** - Works on desktop and mobile

## Quick Start

### Method 1: Using the Python server (Recommended)

```bash
# From the frontend directory
python server.py
```

Then open your browser to: **http://localhost:8000**

### Method 2: Using Python's built-in server

```bash
# From the frontend directory
python -m http.server 8000
```

Then open your browser to: **http://localhost:8000/index.html**

### Method 3: Using any other HTTP server

You can use any HTTP server (Node.js, nginx, Apache, etc.) to serve the frontend directory.

## Usage

### Search Page (index.html)
- Enter search terms in the search bar
- Use advanced filters for more specific searches
- Click on any panel to view details
- Navigate between panels using arrow keys or navigation buttons

### Browse Page (browse.html)
- Browse panels organized by episode
- Filter by season and episode
- View episode statistics

### Tagging Page (tagging.html)
- Add manual tags to panels
- Tag characters, locations, organizations, items, and more
- Tags are saved to the database for future searches

## Database Files

The frontend loads data from:
- `data/panels_database.json` - Complete panel database (4,968 panels)
- `data/panels_s2.json` - Season 2 specific data
- `data/wiki_data.json` - Character and location metadata
- `data/statistics.json` - Database statistics

## Image Files

Web-optimized panel images are located in:
- `images/s2/ep*/` - Season 2 episodes
- Each image is optimized to ~200KB for fast loading

## Project Structure

```
frontend/
├── index.html          # Search page
├── browse.html         # Browse by episode
├── tagging.html        # Manual tagging interface
├── about.html          # About page
├── server.py           # Simple Python HTTP server
├── css/
│   ├── styles.css      # Main styles
│   └── tagging.css     # Tagging interface styles
├── js/
│   ├── search.js       # Search functionality
│   ├── browse.js       # Browse page logic
│   └── tagging.js      # Tagging interface
├── data/               # JSON databases
└── images/             # Web-optimized panel images
```

## Technologies Used

- **Fuse.js** - Fuzzy search
- **Vanilla JavaScript** - No frameworks, fast and lightweight
- **Intersection Observer API** - Lazy loading images
- **CSS Grid** - Responsive layout

## Notes

- This is a fan-made project for educational and archival purposes
- Hand Jumper © SLEEPACROSS. All rights reserved.
- Original panel images are available in higher quality via the "View Original Quality" button

## Troubleshooting

**Images not loading?**
- Make sure you're using an HTTP server (not opening index.html directly)
- Check that the images/ directory contains the panel images

**Search not working?**
- Check browser console for errors
- Make sure data files are present in the data/ directory

**Slow performance?**
- The initial load processes 4,968 panels, which may take a moment
- Images are lazy-loaded as you scroll to improve performance
