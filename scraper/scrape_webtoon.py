"""
Hand Jumper Webtoon Scraper
Downloads all episodes from webtoons.com in highest quality
Includes rate limiting to avoid overwhelming the server
"""

import httpx
import asyncio
from pathlib import Path
import json
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# Configuration
TITLE_NO = "2702"  # Hand Jumper title number
BASE_URL = "https://www.webtoons.com/en/thriller/hand-jumper"  # FIX: Changed from /action/ to /thriller/
EPISODE_LIST_URL = f"{BASE_URL}/list?title_no={TITLE_NO}"

# Use absolute paths relative to script location
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "originals"
METADATA_FILE = SCRIPT_DIR / "data" / "episode_metadata.json"

# Rate limiting settings
CONCURRENT_DOWNLOADS = 3  # Number of simultaneous downloads
DELAY_BETWEEN_REQUESTS = 1.0  # Seconds between requests
MAX_RETRIES = 3

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def get_episode_list(client):
    """Fetch the list of all available episodes"""
    print("Fetching episode list...")
    episodes = []
    page = 1

    while True:
        url = f"{EPISODE_LIST_URL}&page={page}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find episode links
            episode_links = soup.select('#_listUl li a')

            if not episode_links:
                break

            for link in episode_links:
                episode_url = link.get('href')
                if episode_url and '/episode/' in episode_url:
                    # Extract episode number from URL or title
                    title = link.select_one('.subj span').text.strip() if link.select_one('.subj span') else ""
                    episodes.append({
                        'url': f"https://www.webtoons.com{episode_url}" if episode_url.startswith('/') else episode_url,
                        'title': title
                    })

            page += 1
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    print(f"Found {len(episodes)} episodes")
    return episodes


async def download_episode_images(client, episode, semaphore):
    """Download all images from a single episode"""
    async with semaphore:
        try:
            # Get episode page
            response = await client.get(episode['url'])
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract episode number from title or URL
            episode_title = soup.select_one('.subj_episode').text.strip() if soup.select_one('.subj_episode') else episode['title']

            # Parse season and episode number
            season, ep_num = parse_episode_title(episode_title)

            # Find the viewer container with images
            viewer = soup.select_one('#_imageList')
            if not viewer:
                print(f"No viewer found for: {episode_title}")
                return None

            image_urls = []
            for img in viewer.select('img'):
                img_url = img.get('data-url') or img.get('src')
                if img_url:
                    image_urls.append(img_url)

            if not image_urls:
                print(f"No images found for: {episode_title}")
                return None

            # Create episode directory
            ep_dir = OUTPUT_DIR / f"s{season}" / f"ep{ep_num:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            # Download each image
            downloaded_files = []
            for idx, img_url in enumerate(image_urls, 1):
                img_path = ep_dir / f"page_{idx:03d}.jpg"

                if img_path.exists():
                    downloaded_files.append(str(img_path))
                    continue

                # Download with retries
                for attempt in range(MAX_RETRIES):
                    try:
                        headers = {
                            'Referer': 'https://www.webtoons.com/',
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                        img_response = await client.get(img_url, headers=headers, timeout=30.0)
                        img_response.raise_for_status()

                        # Save image
                        img_path.write_bytes(img_response.content)
                        downloaded_files.append(str(img_path))
                        break

                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"Failed to download {img_url}: {e}")
                        else:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff

                await asyncio.sleep(0.5)  # Small delay between images

            return {
                'season': season,
                'episode': ep_num,
                'title': episode_title,
                'url': episode['url'],
                'image_count': len(downloaded_files),
                'directory': str(ep_dir),
                'files': downloaded_files
            }

        except Exception as e:
            print(f"Error downloading episode {episode.get('title', 'unknown')}: {e}")
            return None


def parse_episode_title(title):
    """Extract season and episode number from title like '(S2) Ep. 99 - Title'"""
    season = 1
    episode = 0

    # Look for season marker
    if '(S2)' in title or 'S2' in title:
        season = 2
    elif '(S1)' in title or 'Season 1' in title:
        season = 1

    # Look for episode number
    import re
    ep_match = re.search(r'Ep\.?\s*(\d+)', title, re.IGNORECASE)
    if ep_match:
        episode = int(ep_match.group(1))
    else:
        # Try to extract just a number
        num_match = re.search(r'(\d+)', title)
        if num_match:
            episode = int(num_match.group(1))

    return season, episode


async def main():
    """Main scraping function"""
    print("Hand Jumper Webtoon Scraper")
    print("=" * 50)

    # Create async HTTP client
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Get all episodes
        episodes = await get_episode_list(client)

        if not episodes:
            print("No episodes found!")
            return

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

        # Download all episodes
        print(f"\nDownloading {len(episodes)} episodes...")
        tasks = []
        for episode in episodes:
            task = download_episode_images(client, episode, semaphore)
            tasks.append(task)

        # Run with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"):
            result = await coro
            if result:
                results.append(result)

        # Save metadata
        metadata = {
            'total_episodes': len(results),
            'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'episodes': results
        }

        METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Download complete!")
        print(f"✓ Downloaded {len(results)} episodes")
        print(f"✓ Metadata saved to {METADATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
