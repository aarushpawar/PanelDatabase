# Deploying to GitHub Pages

This frontend is **100% static** and ready for GitHub Pages deployment.

## What Makes It Static?

- ✅ Pure HTML/CSS/JavaScript (no server-side code)
- ✅ All logic runs in the browser
- ✅ Data loaded from JSON files
- ✅ External dependencies loaded from CDN (Fuse.js)
- ✅ No backend, no API calls, no database

## Deployment Steps

### Option 1: Deploy frontend directory directly

```bash
# 1. Create a new GitHub repository
# 2. Navigate to the frontend directory
cd "E:\code\webtoon database\frontend"

# 3. Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: Hand Jumper Panel Database"

# 4. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/hand-jumper-database.git
git branch -M main
git push -u origin main

# 5. Enable GitHub Pages
# Go to: Settings → Pages → Source: main branch → root directory → Save
```

### Option 2: Deploy as subdirectory

```bash
# If you want to deploy the whole project:
cd "E:\code\webtoon database"
git init
git add .
git commit -m "Initial commit"
git push -u origin main

# Then in GitHub Pages settings:
# Source: main branch → /frontend directory
```

## After Deployment

Your site will be available at:
```
https://YOUR_USERNAME.github.io/hand-jumper-database/
```

Or with a custom domain:
```
https://handjumper.yourdomain.com
```

## Important Notes

### File Size Considerations

⚠️ **Warning**: Your `images/` folder contains **4,959 images** (~200KB each).

**Total size**: ~1 GB of images

GitHub has some limits:
- Repository size: Recommended < 1 GB
- Individual file size: < 100 MB
- Push size: < 2 GB

**Recommendation**: Consider these options:

1. **Use GitHub Large File Storage (LFS)**
   ```bash
   git lfs install
   git lfs track "frontend/images/**/*.jpg"
   git add .gitattributes
   ```

2. **Host images externally**
   - Use Cloudinary, Imgur, or AWS S3
   - Update image paths in `panels_database.json`

3. **Split into multiple repos**
   - One repo for code/data
   - Separate repos for image batches

### .gitignore Setup

Create `frontend/.gitignore`:

```gitignore
# Server script (not needed for GitHub Pages)
server.py

# Local testing
*.log
.DS_Store
Thumbs.db

# Optional: exclude images if hosting externally
# images/**/*.jpg
```

## Local Testing

Since GitHub Pages uses HTTPS and serves files directly, you can test locally:

### Method 1: Python (what server.py does)
```bash
cd frontend
python -m http.server 8000
# Open http://localhost:8000
```

### Method 2: Node.js
```bash
npx http-server frontend -p 8000
```

### Method 3: VS Code
Use the "Live Server" extension

## Optimizations for GitHub Pages

### 1. Add a 404 page

Create `frontend/404.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/index.html">
    <title>Redirecting...</title>
</head>
<body>
    <p>Redirecting to homepage...</p>
</body>
</html>
```

### 2. Add robots.txt

Create `frontend/robots.txt`:
```
User-agent: *
Allow: /

Sitemap: https://YOUR_USERNAME.github.io/hand-jumper-database/sitemap.xml
```

### 3. Enable caching

GitHub Pages automatically handles caching for static files.

### 4. Compress images further (optional)

If 1 GB is too large:
```bash
# Reduce quality further
cd data
python -c "from PIL import Image; from pathlib import Path; [Image.open(p).save(p, quality=70, optimize=True) for p in Path('panels_web').rglob('*.jpg')]"
```

## Custom Domain Setup

1. Buy a domain (e.g., handjumperpanels.com)
2. Add CNAME record: `YOUR_USERNAME.github.io`
3. In GitHub repo: Settings → Pages → Custom domain → Save
4. Wait for DNS propagation (can take 24-48 hours)

## Troubleshooting

### Images not loading on GitHub Pages

Check that paths are correct:
- ✅ `images/s2/ep000/s2_ep000_p001.jpg`
- ❌ `/images/s2/ep000/s2_ep000_p001.jpg` (no leading slash)
- ❌ `../images/` (no relative paths)

### 404 errors

Make sure:
- Repository is public
- GitHub Pages is enabled
- Branch is correct (main or gh-pages)
- Path is correct (root or /frontend)

### CORS errors

GitHub Pages handles CORS correctly. If you see CORS errors:
- Make sure you're accessing via the GitHub Pages URL
- Not mixing HTTP and HTTPS

## Performance Tips

1. **Enable lazy loading** (already implemented)
2. **Use CDN for libraries** (already using Fuse.js from CDN)
3. **Minify CSS/JS** (optional, but recommended for production)
4. **Add service worker** (optional, for offline support)

## Security Considerations

Since this is a static site with no user data:
- ✅ No SQL injection risks
- ✅ No XSS from user input (no forms)
- ✅ No authentication needed
- ✅ All content is public anyway

Just make sure:
- Don't commit any API keys or secrets
- All external links use HTTPS
- External CDN links are from trusted sources

## Monitoring

### GitHub Pages includes:

- Basic traffic analytics
- Custom domain support
- Automatic HTTPS
- CDN distribution

### External monitoring (optional):

- Google Analytics
- Cloudflare (if using custom domain)
- UptimeRobot for monitoring

## Update Process

To update the site after changes:

```bash
cd frontend

# Make your changes, then:
git add .
git commit -m "Update: description of changes"
git push

# GitHub Pages will automatically rebuild (takes 1-2 minutes)
```

## Alternative Hosting Options

If GitHub Pages doesn't work for you:

1. **Netlify** - Easier deployment, better for large sites
2. **Vercel** - Fast builds, good analytics
3. **Cloudflare Pages** - Fast CDN, unlimited bandwidth
4. **AWS S3 + CloudFront** - More control, costs money

All of these support static sites and have similar deployment processes.

## Example Repository Structure

```
hand-jumper-database/
├── .github/
│   └── workflows/
│       └── deploy.yml        # Optional: CI/CD
├── .gitignore
├── .gitattributes           # For Git LFS
├── README.md                # Project documentation
├── about.html
├── browse.html
├── index.html
├── tagging.html
├── css/
│   ├── styles.css
│   └── tagging.css
├── js/
│   ├── search.js
│   ├── browse.js
│   └── tagging.js
├── data/
│   ├── panels_database.json
│   ├── panels_s2.json
│   ├── wiki_data.json
│   └── statistics.json
└── images/
    └── s2/
        ├── ep000/
        ├── ep001/
        └── ...
```

## Questions?

- **Q**: Can I use this with a private repo?
  **A**: Yes, but GitHub Pages requires a paid plan for private repos

- **Q**: What if my images are too large?
  **A**: Use Git LFS or external image hosting (Cloudinary, ImgBB, etc.)

- **Q**: Can I add a backend later?
  **A**: For GitHub Pages, no. But you can:
    - Use Netlify Functions (serverless)
    - Add a separate API on Heroku/Railway
    - Use Firebase for backend features

- **Q**: How do I update the database?
  **A**: Run the scraper scripts locally, then push the updated JSON files

## Ready to Deploy!

Your frontend is **100% static and ready** for GitHub Pages. Just push it and you're live!

```bash
# Quick deploy command:
cd "E:\code\webtoon database\frontend"
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_REPO_URL
git push -u origin main
# Then enable GitHub Pages in repo settings!
```

That's it! 🎉
