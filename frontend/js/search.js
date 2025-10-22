// Hand Jumper Panel Database - Search Interface

let panels = [];
let filteredPanels = [];
let displayedPanels = [];
let wikiData = null;
let fuse = null;
let currentFilters = {
    seasons: [],
    episodeMin: null,
    episodeMax: null,
    characters: [],
    locations: [],
    tags: [],
    sort: 'relevance'
};

const PANELS_PER_PAGE = 50;
let currentPage = 0;
let currentModalIndex = -1;

// Initialize application
async function init() {
    console.log("Initializing Hand Jumper Panel Database...");

    await loadData();
    setupEventListeners();
    setupIntersectionObserver();

    // Perform initial search
    performSearch();
}

// Load panel database and wiki data
async function loadData() {
    try {
        // Load panels
        const panelsResponse = await fetch('data/panels_database.json');
        const panelsData = await panelsResponse.json();
        panels = panelsData.panels || [];

        // Initialize Fuse.js for fuzzy search
        const fuseOptions = {
            keys: [
                { name: 'id', weight: 0.1 },
                { name: 'manual.characters', weight: 2.0 },
                { name: 'manual.locations', weight: 1.5 },
                { name: 'manual.organizations', weight: 1.2 },
                { name: 'manual.tags', weight: 1.0 },
                { name: 'manual.items', weight: 0.8 },
                { name: 'manual.notes', weight: 0.5 }
            ],
            threshold: 0.4,
            includeScore: true,
            useExtendedSearch: true
        };

        fuse = new Fuse(panels, fuseOptions);

        console.log(`Loaded ${panels.length} panels`);

        // Load wiki data
        try {
            const wikiResponse = await fetch('data/wiki_data.json');
            wikiData = await wikiResponse.json();

            if (wikiData && wikiData.tag_lists) {
                populateFilters();
            }
        } catch (error) {
            console.warn("Wiki data not available:", error);
        }

        // Populate season filter
        const seasons = [...new Set(panels.map(p => p.season))].sort();
        const seasonFilter = document.getElementById('filterSeason');
        seasons.forEach(season => {
            const option = document.createElement('option');
            option.value = season;
            option.textContent = `Season ${season}`;
            seasonFilter.appendChild(option);
        });

    } catch (error) {
        console.error("Error loading data:", error);
        showError("Failed to load panel database");
    }
}

function populateFilters() {
    if (!wikiData || !wikiData.tag_lists) return;

    populateDatalist('characterList', wikiData.tag_lists.character_names);
    populateDatalist('locationList', wikiData.tag_lists.location_names);
}

function populateDatalist(listId, items) {
    const datalist = document.getElementById(listId);
    if (!datalist || !items) return;

    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        datalist.appendChild(option);
    });
}

// Setup event listeners
function setupEventListeners() {
    // Search
    document.getElementById('searchInput').addEventListener('keyup', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    document.getElementById('searchButton').addEventListener('click', performSearch);
    document.getElementById('clearSearch').addEventListener('click', clearSearch);

    // Toggle filters
    document.getElementById('toggleFilters').addEventListener('click', toggleFilters);

    // Apply/Reset filters
    document.getElementById('applyFilters').addEventListener('click', applyFilters);
    document.getElementById('resetFilters').addEventListener('click', resetFilters);

    // Tag filters
    document.querySelectorAll('.tag-filter').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.target.classList.toggle('active');
        });
    });

    // Character/Location filter inputs
    document.getElementById('filterCharacters').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            addFilterTag('characters', e.target.value.trim());
            e.target.value = '';
        }
    });

    document.getElementById('filterLocations').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
            addFilterTag('locations', e.target.value.trim());
            e.target.value = '';
        }
    });

    // Load more
    document.getElementById('loadMore').addEventListener('click', loadMorePanels);

    // Modal
    document.querySelector('.modal-close').addEventListener('click', closeModal);
    document.querySelector('.modal-overlay').addEventListener('click', closeModal);
    document.getElementById('modalPrev').addEventListener('click', () => navigateModal(-1));
    document.getElementById('modalNext').addEventListener('click', () => navigateModal(1));
    document.getElementById('viewOriginal').addEventListener('click', viewOriginalQuality);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();

        const modal = document.getElementById('panelModal');
        if (!modal.classList.contains('hidden')) {
            if (e.key === 'ArrowLeft') navigateModal(-1);
            if (e.key === 'ArrowRight') navigateModal(1);
        }
    });
}

// Search and Filter Functions
function performSearch() {
    const query = document.getElementById('searchInput').value.trim();

    showLoading(true);

    setTimeout(() => {
        if (query === '') {
            // No search query - show all panels
            filteredPanels = [...panels];
        } else {
            // Perform fuzzy search
            const results = fuse.search(query);
            filteredPanels = results.map(result => result.item);
        }

        // Apply filters
        filteredPanels = applyActiveFilters(filteredPanels);

        // Sort results
        sortResults();

        // Reset pagination
        currentPage = 0;
        displayedPanels = [];

        // Render results
        renderResults();

        showLoading(false);
    }, 100);
}

function applyActiveFilters(panelsToFilter) {
    let filtered = [...panelsToFilter];

    // Season filter
    if (currentFilters.seasons.length > 0) {
        filtered = filtered.filter(p => currentFilters.seasons.includes(p.season));
    }

    // Episode range
    if (currentFilters.episodeMin !== null) {
        filtered = filtered.filter(p => p.episode >= currentFilters.episodeMin);
    }
    if (currentFilters.episodeMax !== null) {
        filtered = filtered.filter(p => p.episode <= currentFilters.episodeMax);
    }

    // Characters
    if (currentFilters.characters.length > 0) {
        filtered = filtered.filter(p =>
            currentFilters.characters.some(char =>
                p.manual.characters.includes(char)
            )
        );
    }

    // Locations
    if (currentFilters.locations.length > 0) {
        filtered = filtered.filter(p =>
            currentFilters.locations.some(loc =>
                p.manual.locations.includes(loc)
            )
        );
    }

    // Tags
    if (currentFilters.tags.length > 0) {
        filtered = filtered.filter(p =>
            currentFilters.tags.some(tag =>
                p.manual.tags.includes(tag)
            )
        );
    }

    return filtered;
}

function sortResults() {
    const sortBy = document.getElementById('sortBy').value;

    switch (sortBy) {
        case 'episode-asc':
            filteredPanels.sort((a, b) => {
                if (a.season !== b.season) return a.season - b.season;
                if (a.episode !== b.episode) return a.episode - b.episode;
                return a.panelNumber - b.panelNumber;
            });
            break;
        case 'episode-desc':
            filteredPanels.sort((a, b) => {
                if (a.season !== b.season) return b.season - a.season;
                if (a.episode !== b.episode) return b.episode - a.episode;
                return b.panelNumber - a.panelNumber;
            });
            break;
        case 'panel-asc':
            filteredPanels.sort((a, b) => a.panelNumber - b.panelNumber);
            break;
        // 'relevance' - keep fuse.js order
        default:
            break;
    }
}

function applyFilters() {
    // Get season selections
    const seasonSelect = document.getElementById('filterSeason');
    currentFilters.seasons = Array.from(seasonSelect.selectedOptions).map(opt => parseInt(opt.value));

    // Get episode range
    const episodeMin = document.getElementById('filterEpisodeMin').value;
    const episodeMax = document.getElementById('filterEpisodeMax').value;
    currentFilters.episodeMin = episodeMin ? parseInt(episodeMin) : null;
    currentFilters.episodeMax = episodeMax ? parseInt(episodeMax) : null;

    // Get active tag filters
    currentFilters.tags = Array.from(document.querySelectorAll('.tag-filter.active'))
        .map(btn => btn.dataset.tag);

    performSearch();
}

function resetFilters() {
    // Clear all filter inputs
    document.getElementById('filterSeason').selectedIndex = -1;
    document.getElementById('filterEpisodeMin').value = '';
    document.getElementById('filterEpisodeMax').value = '';

    // Clear tag filters
    document.querySelectorAll('.tag-filter.active').forEach(btn => {
        btn.classList.remove('active');
    });

    // Clear selected characters/locations
    currentFilters.characters = [];
    currentFilters.locations = [];
    document.getElementById('selectedCharacters').innerHTML = '';
    document.getElementById('selectedLocations').innerHTML = '';

    // Reset sort
    document.getElementById('sortBy').value = 'relevance';

    // Reset filters object
    currentFilters = {
        seasons: [],
        episodeMin: null,
        episodeMax: null,
        characters: [],
        locations: [],
        tags: [],
        sort: 'relevance'
    };

    performSearch();
}

function addFilterTag(type, value) {
    if (!currentFilters[type].includes(value)) {
        currentFilters[type].push(value);

        const container = document.getElementById(type === 'characters' ? 'selectedCharacters' : 'selectedLocations');
        const tag = document.createElement('div');
        tag.className = 'filter-tag';
        tag.innerHTML = `
            <span>${value}</span>
            <span class="remove" onclick="removeFilterTag('${type}', '${value}')">&times;</span>
        `;
        container.appendChild(tag);
    }
}

function removeFilterTag(type, value) {
    const index = currentFilters[type].indexOf(value);
    if (index > -1) {
        currentFilters[type].splice(index, 1);
    }

    const container = document.getElementById(type === 'characters' ? 'selectedCharacters' : 'selectedLocations');
    container.innerHTML = '';
    currentFilters[type].forEach(item => {
        const tag = document.createElement('div');
        tag.className = 'filter-tag';
        tag.innerHTML = `
            <span>${item}</span>
            <span class="remove" onclick="removeFilterTag('${type}', '${item}')">&times;</span>
        `;
        container.appendChild(tag);
    });
}

function clearSearch() {
    document.getElementById('searchInput').value = '';
    performSearch();
}

function toggleFilters() {
    const filters = document.getElementById('advancedFilters');
    const button = document.getElementById('toggleFilters');

    filters.classList.toggle('hidden');
    button.classList.toggle('active');
}

// Rendering Functions
function renderResults() {
    const grid = document.getElementById('resultsGrid');
    const noResults = document.getElementById('noResults');
    const loadMore = document.getElementById('loadMore');

    // Clear grid
    grid.innerHTML = '';

    // Update stats
    document.getElementById('resultsCount').textContent =
        `${filteredPanels.length} panel${filteredPanels.length !== 1 ? 's' : ''} found`;

    if (filteredPanels.length === 0) {
        noResults.classList.remove('hidden');
        loadMore.classList.add('hidden');
        return;
    }

    noResults.classList.add('hidden');

    // Show first page
    loadMorePanels();
}

function loadMorePanels() {
    const grid = document.getElementById('resultsGrid');
    const loadMore = document.getElementById('loadMore');

    const start = currentPage * PANELS_PER_PAGE;
    const end = Math.min(start + PANELS_PER_PAGE, filteredPanels.length);

    for (let i = start; i < end; i++) {
        const panel = filteredPanels[i];
        displayedPanels.push(panel);

        const card = createPanelCard(panel, i);
        grid.appendChild(card);
    }

    currentPage++;

    // Show/hide load more button
    if (end < filteredPanels.length) {
        loadMore.classList.remove('hidden');
    } else {
        loadMore.classList.add('hidden');
    }
}

function createPanelCard(panel, index) {
    const card = document.createElement('div');
    card.className = 'panel-card';
    card.dataset.index = index;

    const img = document.createElement('img');
    img.className = 'panel-card-image lazy';
    img.dataset.src = panel.imagePath;
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

    // Add location tags
    panel.manual.locations.slice(0, 2).forEach(loc => {
        const tag = document.createElement('span');
        tag.className = 'panel-tag location';
        tag.textContent = loc;
        tags.appendChild(tag);
    });

    content.appendChild(title);
    content.appendChild(meta);
    content.appendChild(tags);

    card.appendChild(img);
    card.appendChild(content);

    card.addEventListener('click', () => {
        // Navigate to episode viewer and highlight this panel
        const currentUrl = window.location.href;
        window.location.href = `episode.html?season=${panel.season}&episode=${panel.episode}&panel=${panel.id}&from=${encodeURIComponent(currentUrl)}`;
    });

    return card;
}

// Lazy Loading with Intersection Observer
let imageObserver;

function setupIntersectionObserver() {
    const options = {
        root: null,
        rootMargin: '50px',
        threshold: 0.01
    };

    imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                observer.unobserve(img);
            }
        });
    }, options);
}

// Observe new images
function observeImages() {
    const lazyImages = document.querySelectorAll('img.lazy');
    lazyImages.forEach(img => imageObserver.observe(img));
}

// Modal Functions
function openModal(index) {
    const panel = filteredPanels[index];
    currentModalIndex = index;

    document.getElementById('modalImage').src = panel.imagePath;
    document.getElementById('modalTitle').textContent = panel.id;
    document.getElementById('modalSeason').textContent = panel.season;
    document.getElementById('modalEpisode').textContent = panel.episode;
    document.getElementById('modalPanel').textContent = panel.panelNumber;

    // Populate tags
    populateModalTags('modalCharacters', 'Characters', panel.manual.characters, 'character');
    populateModalTags('modalLocations', 'Locations', panel.manual.locations, 'location');
    populateModalTags('modalTags', 'Tags', panel.manual.tags);

    // Notes
    const notesEl = document.getElementById('modalNotes');
    if (panel.manual.notes) {
        notesEl.textContent = panel.manual.notes;
        notesEl.style.display = 'block';
    } else {
        notesEl.style.display = 'none';
    }

    // Update navigation buttons
    document.getElementById('modalPrev').disabled = index === 0;
    document.getElementById('modalNext').disabled = index === filteredPanels.length - 1;

    document.getElementById('panelModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('panelModal').classList.add('hidden');
    document.body.style.overflow = '';
}

function navigateModal(direction) {
    const newIndex = currentModalIndex + direction;
    if (newIndex >= 0 && newIndex < filteredPanels.length) {
        openModal(newIndex);
    }
}

function populateModalTags(containerId, title, items, cssClass = '') {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (items && items.length > 0) {
        const heading = document.createElement('h4');
        heading.textContent = title;
        container.appendChild(heading);

        const tagsDiv = document.createElement('div');
        tagsDiv.className = 'panel-card-tags';

        items.forEach(item => {
            const tag = document.createElement('span');
            tag.className = 'panel-tag' + (cssClass ? ' ' + cssClass : '');
            tag.textContent = item;
            tagsDiv.appendChild(tag);
        });

        container.appendChild(tagsDiv);
    }
}

function viewOriginalQuality() {
    const panel = filteredPanels[currentModalIndex];
    if (panel.originalPath) {
        window.open(panel.originalPath, '_blank');
    }
}

// Utility Functions
function showLoading(show) {
    const indicator = document.getElementById('loadingIndicator');
    if (show) {
        indicator.classList.remove('hidden');
    } else {
        indicator.classList.add('hidden');
    }
}

function showError(message) {
    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--error-color);">
            <h2>Error</h2>
            <p>${message}</p>
        </div>
    `;
}

// Watch for new images after render
const originalLoadMore = loadMorePanels;
loadMorePanels = function() {
    originalLoadMore();
    setTimeout(observeImages, 100);
};

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);
