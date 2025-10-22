"""
Hand Jumper Wiki Scraper
Extracts character names, locations, organizations, and lore from the wiki
to use as tags for panel categorization

Uses the MediaWiki API to properly handle pagination and get ALL category members
"""

import httpx
import asyncio
from bs4 import BeautifulSoup
from pathlib import Path
import json
from tqdm import tqdm

WIKI_BASE = "https://hand-jumper.fandom.com"
API_URL = f"{WIKI_BASE}/api.php"

# Use absolute paths relative to script location
SCRIPT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = SCRIPT_DIR / "data"
CATEGORIES_FILE = OUTPUT_DIR / "wiki_categories.json"
MANUAL_CHARACTER_LIST = OUTPUT_DIR / "manual_character_list.json"

# Categories to scrape
CATEGORIES = {
    'characters': 'Characters',
    'locations': 'Locations',
    'organizations': 'Organizations',
    'aberrants': 'Aberrants',
    'items': 'Items'
}

# Known main characters to search for (in case they're not in categories)
KNOWN_CHARACTERS = [
    "Sayeon Lee", "Ryujin Kang", "Min Woo", "Iseul Kim", "Juni Chang",
    "Jaeil Park", "Samin Lee", "Sara Lee", "Lilith", "Augur",
    "Jungwoo", "Geum", "Sungwoo Han", "Yesol Na", "Instructor Han",
    "Instructor Kim", "Taeho Kim", "Dahee", "Yujin", "Haeun",
    "Gayoung", "Cell 4", "Seo", "Jiwoo", "Minnie", "Gyeoul"
]


async def get_category_members_api(client, category_name, depth=0):
    """
    Get ALL members of a wiki category using the MediaWiki API
    Handles pagination automatically with cmcontinue
    Recursively fetches subcategories as well
    """
    members = []
    subcategories = []

    # Get both pages and subcategories
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category_name}',
        'cmlimit': '500',  # Max allowed by API
        'cmtype': 'page|subcat',  # Get both pages and subcategories
        'format': 'json'
    }

    continue_token = None
    page_num = 1

    while True:
        if continue_token:
            params['cmcontinue'] = continue_token

        try:
            response = await client.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract members from response
            category_members = data.get('query', {}).get('categorymembers', [])

            for member in category_members:
                title = member.get('title', '')
                page_id = member.get('pageid', 0)
                ns = member.get('ns', 0)  # Namespace: 14 = Category

                if ns == 14:  # This is a subcategory
                    subcategories.append(title.replace('Category:', ''))
                else:  # Regular page
                    members.append({
                        'name': title,
                        'page_id': page_id,
                        'url': f"{WIKI_BASE}/wiki/{title.replace(' ', '_')}"
                    })

            indent = "  " * (depth + 2)
            print(f"{indent}Page {page_num}: Found {len(category_members)} items (pages: {len(members)}, subcats: {len(subcategories)})", flush=True)

            # Check for continuation
            if 'continue' in data and 'cmcontinue' in data['continue']:
                continue_token = data['continue']['cmcontinue']
                page_num += 1
                await asyncio.sleep(0.3)  # Rate limiting
            else:
                break

        except Exception as e:
            indent = "  " * (depth + 2)
            print(f"{indent}[ERROR] API request failed: {e}", flush=True)
            break

    # Recursively get members from subcategories (limit depth to avoid infinite loops)
    if subcategories and depth < 2:
        indent = "  " * (depth + 1)
        print(f"{indent}[INFO] Found {len(subcategories)} subcategories, fetching recursively...", flush=True)
        for subcat in subcategories:
            print(f"{indent}  Subcategory: {subcat}", flush=True)
            subcat_members = await get_category_members_api(client, subcat, depth + 1)
            # Merge, avoiding duplicates
            for member in subcat_members:
                if not any(m['name'] == member['name'] for m in members):
                    members.append(member)
            await asyncio.sleep(0.3)

    return members


async def search_for_page(client, page_title):
    """
    Search for a specific page by title using MediaWiki API
    Returns page info if found, None otherwise
    """
    params = {
        'action': 'query',
        'titles': page_title,
        'format': 'json'
    }

    try:
        response = await client.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        for page_id, page_info in pages.items():
            if page_id != '-1':  # Page exists
                return {
                    'name': page_info.get('title', page_title),
                    'page_id': int(page_id),
                    'url': f"{WIKI_BASE}/wiki/{page_info.get('title', page_title).replace(' ', '_')}"
                }

    except Exception as e:
        pass

    return None


async def fetch_page(client, url):
    """Fetch a single wiki page"""
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"    [ERROR] Fetching {url}: {e}", flush=True)
        return None


async def get_character_details(client, character_url):
    """Get detailed information about a character"""
    html = await fetch_page(client, character_url)
    if not html:
        return {}

    soup = BeautifulSoup(html, 'html.parser')

    details = {}

    # Try to extract from infobox
    infobox = soup.select_one('.portable-infobox')
    if infobox:
        # Extract key-value pairs
        for row in infobox.select('.pi-item'):
            label = row.select_one('.pi-data-label')
            value = row.select_one('.pi-data-value')

            if label and value:
                key = label.get_text().strip().lower()
                val = value.get_text().strip()
                details[key] = val

    # Get character description
    content = soup.select_one('.mw-parser-output')
    if content:
        # Get first paragraph as description
        paragraphs = content.find_all('p', recursive=False)
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 50:  # Skip very short paragraphs
                details['description'] = text
                break

    return details


async def scrape_all_categories():
    """Scrape all category pages from the wiki using MediaWiki API"""
    print("Hand Jumper Wiki Scraper (API-based)", flush=True)
    print("=" * 60, flush=True)
    print("Using MediaWiki API to get complete category listings", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        all_data = {}

        for category_key, category_name in CATEGORIES.items():
            print(f"[{category_key.upper()}] Fetching from Category:{category_name}...", flush=True)

            members = await get_category_members_api(client, category_name)

            if not members:
                print(f"  [WARNING] No members found for {category_key}", flush=True)
                all_data[category_key] = []
            else:
                print(f"  [OK] Found {len(members)} {category_key} from category", flush=True)

            # For characters, also search for known characters that might not be in the category
            if category_key == 'characters':
                print(f"  [INFO] Searching for known characters not in category...", flush=True)
                existing_names = {m['name'] for m in members}

                for char_name in KNOWN_CHARACTERS:
                    if char_name not in existing_names:
                        char_info = await search_for_page(client, char_name)
                        if char_info:
                            members.append(char_info)
                            print(f"    [+] Found: {char_name}", flush=True)
                        await asyncio.sleep(0.2)

                print(f"  [OK] Total characters after search: {len(members)}", flush=True)

                # Get details for all characters
                print(f"  [INFO] Fetching details for {len(members)} characters...", flush=True)
                detailed_members = []
                for idx, member in enumerate(members, 1):
                    details = await get_character_details(client, member['url'])
                    detailed_members.append({
                        **member,
                        'details': details
                    })
                    if idx % 10 == 0:
                        print(f"    Processed {idx}/{len(members)} characters...", flush=True)
                    await asyncio.sleep(0.3)  # Rate limiting
                all_data[category_key] = detailed_members
                print(f"  [OK] Character details complete", flush=True)
            else:
                all_data[category_key] = members if members else []

            print(flush=True)

        # Create simplified tag lists
        tag_lists = {
            'character_names': sorted([c['name'] for c in all_data.get('characters', [])]),
            'location_names': sorted([l['name'] for l in all_data.get('locations', [])]),
            'organization_names': sorted([o['name'] for o in all_data.get('organizations', [])]),
            'aberrant_names': sorted([a['name'] for a in all_data.get('aberrants', [])]),
            'item_names': sorted([i['name'] for i in all_data.get('items', [])])
        }

        # Save full data
        output = {
            'categories': all_data,
            'tag_lists': tag_lists,
            'total_entries': sum(len(v) for v in all_data.values()),
            'scrape_date': asyncio.get_event_loop().time()
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print("=" * 60, flush=True)
        print("[OK] Wiki scraping complete!", flush=True)
        print(f"[OK] Total entries: {output['total_entries']}", flush=True)
        for cat, items in all_data.items():
            print(f"  - {cat.capitalize()}: {len(items)}", flush=True)
        print(f"[OK] Data saved to {CATEGORIES_FILE}", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(scrape_all_categories())
