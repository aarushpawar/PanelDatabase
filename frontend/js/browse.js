// Browse page functionality

let panels = [];
let wikiData = null;
let groupedData = {
    bySeason: {},
    byCharacter: {},
    byLocation: {}
};

async function init() {
    await loadData();
    groupPanels();
    setupTabs();
    loadEpisodesBrowse();
}

async function loadData() {
    try {
        const panelsResponse = await fetch('data/panels_database.json');
        const panelsData = await panelsResponse.json();
        panels = panelsData.panels || [];

        try {
            const wikiResponse = await fetch('data/wiki_data.json');
            wikiData = await wikiResponse.json();
        } catch (error) {
            console.warn("Wiki data not available");
        }

        console.log(`Loaded ${panels.length} panels`);
    } catch (error) {
        console.error("Error loading data:", error);
    }
}

function groupPanels() {
    // Group by season and episode
    panels.forEach(panel => {
        const key = `s${panel.season}`;
        if (!groupedData.bySeason[key]) {
            groupedData.bySeason[key] = {};
        }
        const epKey = `ep${panel.episode}`;
        if (!groupedData.bySeason[key][epKey]) {
            groupedData.bySeason[key][epKey] = [];
        }
        groupedData.bySeason[key][epKey].push(panel);

        // Group by character
        panel.manual.characters.forEach(char => {
            if (!groupedData.byCharacter[char]) {
                groupedData.byCharacter[char] = [];
            }
            groupedData.byCharacter[char].push(panel);
        });

        // Group by location
        panel.manual.locations.forEach(loc => {
            if (!groupedData.byLocation[loc]) {
                groupedData.byLocation[loc] = [];
            }
            groupedData.byLocation[loc].push(panel);
        });
    });
}

function setupTabs() {
    const tabs = document.querySelectorAll('.browse-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));

            // Add active to clicked tab
            tab.classList.add('active');
            const tabName = tab.dataset.tab;
            document.getElementById(tabName + 'Tab').classList.add('active');

            // Load appropriate content
            if (tabName === 'episodes') {
                loadEpisodesBrowse();
            } else if (tabName === 'characters') {
                loadCharactersBrowse();
            } else if (tabName === 'locations') {
                loadLocationsBrowse();
            }
        });
    });
}

function loadEpisodesBrowse() {
    const seasonSelect = document.getElementById('seasonSelect');
    const episodeList = document.getElementById('episodeList');

    // Populate season select
    const seasons = Object.keys(groupedData.bySeason).sort();
    seasonSelect.innerHTML = '';
    seasons.forEach(season => {
        const option = document.createElement('option');
        option.value = season;
        option.textContent = `Season ${season.substring(1)}`;
        seasonSelect.appendChild(option);
    });

    // Load first season
    if (seasons.length > 0) {
        seasonSelect.value = seasons[0];
        loadEpisodesForSeason(seasons[0]);
    }

    seasonSelect.addEventListener('change', (e) => {
        loadEpisodesForSeason(e.target.value);
    });
}

function loadEpisodesForSeason(seasonKey) {
    const episodeList = document.getElementById('episodeList');
    const episodes = groupedData.bySeason[seasonKey];

    episodeList.innerHTML = '';

    Object.keys(episodes).sort((a, b) => {
        const numA = parseInt(a.substring(2));
        const numB = parseInt(b.substring(2));
        return numA - numB;
    }).forEach(epKey => {
        const panels = episodes[epKey];
        const epNum = parseInt(epKey.substring(2));

        const card = document.createElement('div');
        card.className = 'episode-card';
        card.innerHTML = `
            <div class="episode-number">${epNum}</div>
            <div class="episode-info">${panels.length} panels</div>
        `;

        card.addEventListener('click', () => {
            showEpisodePanels(seasonKey, epKey, panels);
        });

        episodeList.appendChild(card);
    });
}

function showEpisodePanels(seasonKey, epKey, panels) {
    const resultsTitle = document.getElementById('resultsTitle');
    const browseResults = document.getElementById('browseResults');

    const seasonNum = seasonKey.substring(1);
    const epNum = epKey.substring(2);

    resultsTitle.textContent = `Season ${seasonNum}, Episode ${epNum}`;

    browseResults.innerHTML = '';

    panels.forEach((panel, index) => {
        const card = createPanelCard(panel);
        browseResults.appendChild(card);
    });

    // Scroll to results
    resultsTitle.scrollIntoView({ behavior: 'smooth' });
}

function loadCharactersBrowse() {
    const characterGrid = document.getElementById('characterGrid');
    characterGrid.innerHTML = '';

    const characters = Object.keys(groupedData.byCharacter).sort();

    characters.forEach(character => {
        const panels = groupedData.byCharacter[character];

        const card = document.createElement('div');
        card.className = 'browse-item';
        card.innerHTML = `
            <h3>${character}</h3>
            <p>${panels.length} panel${panels.length !== 1 ? 's' : ''}</p>
        `;

        card.addEventListener('click', () => {
            showCharacterPanels(character, panels);
        });

        characterGrid.appendChild(card);
    });

    if (characters.length === 0) {
        characterGrid.innerHTML = '<p style="color: var(--text-secondary);">No characters tagged yet. Start tagging panels!</p>';
    }
}

function showCharacterPanels(character, panels) {
    const resultsTitle = document.getElementById('resultsTitle');
    const browseResults = document.getElementById('browseResults');

    resultsTitle.textContent = `Panels featuring ${character}`;

    browseResults.innerHTML = '';

    panels.forEach(panel => {
        const card = createPanelCard(panel);
        browseResults.appendChild(card);
    });

    resultsTitle.scrollIntoView({ behavior: 'smooth' });
}

function loadLocationsBrowse() {
    const locationGrid = document.getElementById('locationGrid');
    locationGrid.innerHTML = '';

    const locations = Object.keys(groupedData.byLocation).sort();

    locations.forEach(location => {
        const panels = groupedData.byLocation[location];

        const card = document.createElement('div');
        card.className = 'browse-item';
        card.innerHTML = `
            <h3>${location}</h3>
            <p>${panels.length} panel${panels.length !== 1 ? 's' : ''}</p>
        `;

        card.addEventListener('click', () => {
            showLocationPanels(location, panels);
        });

        locationGrid.appendChild(card);
    });

    if (locations.length === 0) {
        locationGrid.innerHTML = '<p style="color: var(--text-secondary);">No locations tagged yet. Start tagging panels!</p>';
    }
}

function showLocationPanels(location, panels) {
    const resultsTitle = document.getElementById('resultsTitle');
    const browseResults = document.getElementById('browseResults');

    resultsTitle.textContent = `Panels at ${location}`;

    browseResults.innerHTML = '';

    panels.forEach(panel => {
        const card = createPanelCard(panel);
        browseResults.appendChild(card);
    });

    resultsTitle.scrollIntoView({ behavior: 'smooth' });
}

function createPanelCard(panel) {
    const card = document.createElement('div');
    card.className = 'panel-card';

    const img = document.createElement('img');
    img.className = 'panel-card-image';
    img.src = panel.imagePath;
    img.alt = panel.id;

    const content = document.createElement('div');
    content.className = 'panel-card-content';

    const title = document.createElement('div');
    title.className = 'panel-card-title';
    title.textContent = panel.id;

    const meta = document.createElement('div');
    meta.className = 'panel-card-meta';
    meta.textContent = `S${panel.season} E${panel.episode} • Panel ${panel.panelNumber}`;

    const tags = document.createElement('div');
    tags.className = 'panel-card-tags';

    // Add character tags
    panel.manual.characters.slice(0, 3).forEach(char => {
        const tag = document.createElement('span');
        tag.className = 'panel-tag character';
        tag.textContent = char;
        tags.appendChild(tag);
    });

    content.appendChild(title);
    content.appendChild(meta);
    content.appendChild(tags);

    card.appendChild(img);
    card.appendChild(content);

    return card;
}

window.addEventListener('DOMContentLoaded', init);
