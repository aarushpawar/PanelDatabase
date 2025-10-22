"""
Automated Panel Tagging
Uses episode metadata and wiki data to automatically tag panels
"""

import json
from pathlib import Path
import re
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
EPISODE_METADATA = SCRIPT_DIR / "data" / "episode_metadata.json"
WIKI_DATA = SCRIPT_DIR / "data" / "wiki_categories.json"
PANEL_METADATA = SCRIPT_DIR / "data" / "panel_metadata.json"
DATABASE_FILE = SCRIPT_DIR / "frontend" / "data" / "panels_database.json"

# Character name variations and common mentions
CHARACTER_KEYWORDS = {
    "Sayeon": ["Sayeon", "Sa-yeon", "Min Sayeon"],
    "Jaehee": ["Jaehee", "Jae-hee", "Cha Jaehee"],
    "Ryujin": ["Ryujin", "Ryu-jin", "Jungwoo Ryujin"],
    "Samin": ["Samin", "Sa-min", "Lee Samin"],
    "Dahee": ["Dahee", "Da-hee", "Yoo Dahee"],
    "Iseul": ["Iseul", "Lee Iseul"],
    "Juni": ["Juni", "Seo Juni"],
    "Minnie": ["Minnie", "Park Minnie"],
    "Gyeong": ["Gyeong", "Instructor Gyeong"],
}

# Episode-based tagging rules (based on typical story arcs)
EPISODE_TAGS = {
    # Early episodes - Introduction
    range(0, 10): {
        "tags": ["introduction", "world-building"],
        "likely_characters": ["Sayeon", "Jaehee"]
    },
    # Training arc
    range(10, 30): {
        "tags": ["training", "academy"],
        "locations": ["Aberrant Corps Academy"],
        "likely_characters": ["Sayeon", "Jaehee", "Instructor Gyeong"]
    },
    # Mission arcs
    range(30, 60): {
        "tags": ["mission", "action"],
        "likely_characters": ["Sayeon", "Jaehee", "Ryujin", "Samin"]
    },
}

def load_data():
    """Load all necessary data files"""
    with open(EPISODE_METADATA, 'r', encoding='utf-8') as f:
        episodes = json.load(f)

    with open(WIKI_DATA, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
        database = json.load(f)

    return episodes, wiki, database

def extract_characters_from_title(title):
    """Extract character names from episode titles"""
    found_characters = []

    for char_name, variations in CHARACTER_KEYWORDS.items():
        for variation in variations:
            if variation.lower() in title.lower():
                if char_name not in found_characters:
                    found_characters.append(char_name)

    return found_characters

def get_episode_context(episode_num):
    """Get contextual tags for an episode based on its number"""
    tags = []
    locations = []
    likely_chars = []

    for ep_range, context in EPISODE_TAGS.items():
        if episode_num in ep_range:
            tags.extend(context.get("tags", []))
            locations.extend(context.get("locations", []))
            likely_chars.extend(context.get("likely_characters", []))

    return {
        "tags": tags,
        "locations": locations,
        "characters": likely_chars
    }

def auto_tag_panels(database, episodes):
    """Automatically tag panels based on available data"""

    print("Auto-Tagging Panels")
    print("=" * 70)
    print()

    # Create episode lookup
    episode_lookup = {}
    for ep_data in episodes.get("episodes", []):
        key = (ep_data["season"], ep_data["episode"])
        episode_lookup[key] = ep_data

    tagged_count = 0

    for panel in database["panels"]:
        season = panel["season"]
        episode = panel["episode"]

        # Get episode data
        ep_key = (season, episode)
        ep_data = episode_lookup.get(ep_key, {})
        ep_title = ep_data.get("title", "")

        # Extract characters from episode title
        characters = extract_characters_from_title(ep_title)

        # Get contextual tags
        context = get_episode_context(episode)

        # Combine with context
        if context["characters"]:
            characters.extend([c for c in context["characters"] if c not in characters])

        # Update panel
        if characters or context["tags"] or context["locations"]:
            panel["manual"]["characters"] = characters[:3]  # Max 3 from auto-tagging
            panel["manual"]["tags"] = context["tags"]
            panel["manual"]["locations"] = context["locations"]
            panel["metadata"]["tagged"] = True
            tagged_count += 1

    print(f"Tagged {tagged_count} / {len(database['panels'])} panels")
    print()

    return database, tagged_count

def generate_panel_statistics(database):
    """Generate statistics about tagged panels"""
    stats = defaultdict(int)

    all_characters = set()
    all_locations = set()
    all_tags = set()

    for panel in database["panels"]:
        if panel["metadata"]["tagged"]:
            stats["tagged"] += 1

            all_characters.update(panel["manual"]["characters"])
            all_locations.update(panel["manual"]["locations"])
            all_tags.update(panel["manual"]["tags"])

    stats["total"] = len(database["panels"])
    stats["unique_characters"] = len(all_characters)
    stats["unique_locations"] = len(all_locations)
    stats["unique_tags"] = len(all_tags)

    return stats, {
        "characters": sorted(all_characters),
        "locations": sorted(all_locations),
        "tags": sorted(all_tags)
    }

def save_database(database):
    """Save updated database"""
    # Update statistics
    tagged_count = sum(1 for p in database["panels"] if p["metadata"]["tagged"])

    # Update main database file
    with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    # Update statistics file
    stats_file = DATABASE_FILE.parent / "statistics.json"
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    stats["tagged_panels"] = tagged_count

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Update season-specific databases
    panels_by_season = defaultdict(list)
    for panel in database["panels"]:
        panels_by_season[panel["season"]].append(panel)

    for season, panels in panels_by_season.items():
        season_file = DATABASE_FILE.parent / f"panels_s{season}.json"
        season_db = {
            "version": database["version"],
            "season": season,
            "panels": panels
        }
        with open(season_file, 'w', encoding='utf-8') as f:
            json.dump(season_db, f, indent=2, ensure_ascii=False)

def main():
    print("Automated Panel Tagging")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    episodes, wiki, database = load_data()
    print(f"Loaded {len(database['panels'])} panels")
    print(f"Loaded {len(episodes.get('episodes', []))} episode metadata")
    print()

    # Auto-tag panels
    database, tagged_count = auto_tag_panels(database, episodes)

    # Generate statistics
    stats, unique_items = generate_panel_statistics(database)

    # Save
    print("Saving updated database...")
    save_database(database)

    # Print summary
    print()
    print("=" * 70)
    print("✓ Auto-Tagging Complete!")
    print("=" * 70)
    print(f"Total panels: {stats['total']}")
    print(f"Tagged panels: {stats['tagged']} ({stats['tagged']/stats['total']*100:.1f}%)")
    print(f"Unique characters: {stats['unique_characters']}")
    print(f"Unique locations: {stats['unique_locations']}")
    print(f"Unique tags: {stats['unique_tags']}")
    print()

    if unique_items["characters"]:
        print("Characters found:", ", ".join(unique_items["characters"][:10]))
    if unique_items["tags"]:
        print("Tags applied:", ", ".join(unique_items["tags"]))
    print()

    print("⚠️  Note: Auto-tagging provides basic tags based on episode context.")
    print("   For more accurate tagging, use the manual tagging interface:")
    print("   http://localhost:8000/tagging.html")
    print("=" * 70)

if __name__ == "__main__":
    main()
