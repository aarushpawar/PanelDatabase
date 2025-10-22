# Deployment Guide

## GitHub Pages Deployment

This project is designed to be deployed as a static site on GitHub Pages.

### Prerequisites

1. A GitHub account
2. Git installed on your local machine
3. Python 3.8+ for running scraper scripts

### Step 1: Prepare Your Data

Before deploying, you need to scrape and process the webtoon panels:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the scraping pipeline
cd scraper
python scrape_webtoon.py      # Download episodes
python extract_panels.py       # Extract individual panels
python optimize_images.py      # Create web-optimized versions
python scrape_wiki.py          # Get character/lore data
python build_database.py       # Build final JSON database
```

### Step 2: Organize Files for Deployment

The `frontend/` directory contains everything needed for deployment:

```
frontend/
├── index.html              # Main search page
├── browse.html             # Browse by episode/character
├── tagging.html            # Tagging interface
├── about.html              # About page
├── css/
│   ├── styles.css
│   └── tagging.css
├── js/
│   ├── search.js
│   ├── browse.js
│   └── tagging.js
├── data/
│   ├── panels_database.json
│   ├── panels_s1.json
│   ├── panels_s2.json
│   ├── wiki_data.json
│   └── statistics.json
└── images/                  # Web-optimized panel images
    ├── s1/
    │   ├── ep001/
    │   │   ├── s1_ep001_p001.jpg
    │   │   └── ...
    │   └── ...
    └── s2/
        └── ...
```

### Step 3: Copy Web-Optimized Images

Copy the optimized images to the frontend directory:

```bash
# Windows
xcopy /E /I ..\data\panels_web frontend\images

# Linux/Mac
cp -r ../data/panels_web/* frontend/images/
```

### Step 4: Initialize Git Repository

```bash
# Navigate to your project root
cd webtoon-database

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Hand Jumper Panel Database"
```

### Step 5: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `hand-jumper-database`
3. **Don't** initialize with README (you already have files)
4. Copy the repository URL

### Step 6: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR-USERNAME/hand-jumper-database.git

# Push to main branch
git branch -M main
git push -u origin main
```

### Step 7: Deploy to GitHub Pages

#### Option A: Deploy from root or /docs folder

1. Go to your repository settings
2. Scroll to "GitHub Pages" section
3. Under "Source", select the branch (main) and folder (/root or /docs)
4. Click "Save"

Since your frontend is in `frontend/`, you should:
- Either move frontend files to root, OR
- Move frontend files to a `/docs` folder, OR
- Use the GitHub Actions method below

#### Option B: Deploy using GitHub Actions (Recommended)

This allows you to keep `frontend/` as a subfolder:

1. Create `.github/workflows/deploy.yml` in your repository root:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./frontend
```

2. Commit and push:

```bash
git add .github/workflows/deploy.yml
git commit -m "Add GitHub Actions deployment"
git push
```

3. Go to repository Settings → Pages
4. Set source to "gh-pages" branch
5. Your site will be live at: `https://YOUR-USERNAME.github.io/hand-jumper-database/`

### Step 8: Important Considerations

#### File Size Limits

GitHub has file size limits:
- Individual files: 100MB max
- Repository size: 1GB recommended, 5GB max
- **Use Git LFS for large files if needed**

If your images are too large:

1. Install Git LFS:
   ```bash
   git lfs install
   ```

2. Track large image files:
   ```bash
   git lfs track "frontend/images/**/*.jpg"
   git lfs track "frontend/images/**/*.png"
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   ```

3. Push with LFS:
   ```bash
   git push origin main
   ```

#### Optimize Your Deployment

To keep the repository size manageable:

1. **Only include web-optimized images** in `frontend/images/`
2. **Don't commit** original high-quality images to Git
3. **Compress JSON files** if they're very large:
   ```python
   import json
   # When saving JSON, use separators with no spaces
   json.dump(data, f, separators=(',', ':'))
   ```

4. **Split large JSON files** by season (already done in build_database.py)

### Step 9: Test Your Deployment

1. Wait a few minutes for GitHub Pages to build
2. Visit your site at `https://YOUR-USERNAME.github.io/REPO-NAME/`
3. Test:
   - Search functionality
   - Filtering
   - Image loading
   - Modal dialogs
   - Browse features

### Troubleshooting

#### Images not loading

- Check that image paths in JSON are relative: `images/s1/ep001/panel.jpg`
- Verify images are committed to git
- Check browser console for 404 errors

#### Search not working

- Ensure Fuse.js CDN is accessible
- Check browser console for JavaScript errors
- Verify JSON files are loading correctly

#### Page not updating

- Clear GitHub Pages cache (Settings → Pages → "Clear cache")
- Force refresh in browser (Ctrl+Shift+R / Cmd+Shift+R)
- Check GitHub Actions status if using workflow

### Alternative: Deploy to Netlify

If GitHub Pages has issues:

1. Go to [Netlify](https://netlify.com)
2. Connect your GitHub repository
3. Set build directory to `frontend`
4. Deploy

### Updating Your Site

When you add new panels or update tags:

```bash
# Re-run pipeline
python scraper/build_database.py

# Copy new images if needed
cp -r data/panels_web/* frontend/images/

# Commit and push
git add .
git commit -m "Update panel database"
git push
```

GitHub Pages will automatically redeploy!

### Custom Domain (Optional)

1. Buy a domain (e.g., handjumper-database.com)
2. In repository, add a `CNAME` file to `frontend/` with your domain
3. Configure DNS records with your domain provider
4. Enable HTTPS in GitHub Pages settings

## Support

For issues or questions, refer to:
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Git LFS Documentation](https://git-lfs.github.com/)
