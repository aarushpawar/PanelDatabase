// Tagging Interface JavaScript

let panels = [];
let currentPanelIndex = 0;
let currentTags = {
    characters: [],
    locations: [],
    organizations: [],
    items: [],
    tags: [],
    emotions: [],
    dialogue: false,
    action: false,
    notes: ""
};
let wikiData = null;

// Initialize the application
async function init() {
    console.log("Initializing tagging interface...");

    // Load panel database
    await loadPanels();

    // Load wiki data for autocomplete
    await loadWikiData();

    // Setup event listeners
    setupEventListeners();

    // Load first panel
    if (panels.length > 0) {
        loadPanel(0);
    }

    updateStats();
}

async function loadPanels() {
    try {
        const response = await fetch('data/panels_database.json');
        const data = await response.json();
        panels = data.panels || [];

        // Populate season filter
        const seasons = [...new Set(panels.map(p => p.season))].sort();
        const seasonFilter = document.getElementById('seasonFilter');
        seasons.forEach(season => {
            const option = document.createElement('option');
            option.value = season;
            option.textContent = `Season ${season}`;
            seasonFilter.appendChild(option);
        });

        renderThumbnails();
        console.log(`Loaded ${panels.length} panels`);
    } catch (error) {
        console.error("Error loading panels:", error);
        showStatus("Error loading panel database!", "error");
    }
}

async function loadWikiData() {
    try {
        const response = await fetch('data/wiki_data.json');
        wikiData = await response.json();

        // Populate autocomplete lists
        if (wikiData && wikiData.tag_lists) {
            populateDatalist('characterList', wikiData.tag_lists.character_names);
            populateDatalist('locationList', wikiData.tag_lists.location_names);
            populateDatalist('organizationList', wikiData.tag_lists.organization_names);
            populateDatalist('itemList', wikiData.tag_lists.item_names);
        }

        console.log("Loaded wiki data");
    } catch (error) {
        console.warn("Wiki data not available:", error);
    }
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

function setupEventListeners() {
    // Navigation buttons
    document.getElementById('prevPanel').addEventListener('click', () => navigatePanel(-1));
    document.getElementById('nextPanel').addEventListener('click', () => navigatePanel(1));

    // Filters
    document.getElementById('seasonFilter').addEventListener('change', filterPanels);
    document.getElementById('episodeFilter').addEventListener('change', filterPanels);
    document.getElementById('showUntaggedOnly').addEventListener('change', filterPanels);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'ArrowLeft') navigatePanel(-1);
        if (e.key === 'ArrowRight') navigatePanel(1);
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveTags();
        }
    });

    // Checkboxes
    document.getElementById('hasDialogue').addEventListener('change', (e) => {
        currentTags.dialogue = e.target.checked;
    });

    document.getElementById('isAction').addEventListener('change', (e) => {
        currentTags.action = e.target.checked;
    });

    // Notes
    document.getElementById('notesInput').addEventListener('input', (e) => {
        currentTags.notes = e.target.value;
    });
}

function renderThumbnails() {
    const container = document.getElementById('panelThumbnails');
    container.innerHTML = '';

    const filteredPanels = getFilteredPanels();

    filteredPanels.forEach((panel, index) => {
        const div = document.createElement('div');
        div.className = 'thumbnail-item';
        if (panel.metadata.tagged) div.classList.add('tagged');
        if (index === currentPanelIndex) div.classList.add('active');

        div.innerHTML = `
            <img src="${panel.imagePath}" alt="${panel.id}" loading="lazy">
            <div class="thumbnail-label">${panel.id}</div>
        `;

        div.addEventListener('click', () => loadPanel(panels.indexOf(panel)));

        container.appendChild(div);
    });
}

function getFilteredPanels() {
    let filtered = [...panels];

    const seasonFilter = document.getElementById('seasonFilter').value;
    const episodeFilter = document.getElementById('episodeFilter').value;
    const showUntaggedOnly = document.getElementById('showUntaggedOnly').checked;

    if (seasonFilter) {
        filtered = filtered.filter(p => p.season === parseInt(seasonFilter));
    }

    if (episodeFilter) {
        filtered = filtered.filter(p => p.episode === parseInt(episodeFilter));
    }

    if (showUntaggedOnly) {
        filtered = filtered.filter(p => !p.metadata.tagged);
    }

    return filtered;
}

function filterPanels() {
    // Update episode filter based on season
    const seasonFilter = document.getElementById('seasonFilter').value;
    const episodeFilter = document.getElementById('episodeFilter');

    if (seasonFilter) {
        const episodes = [...new Set(
            panels
                .filter(p => p.season === parseInt(seasonFilter))
                .map(p => p.episode)
        )].sort((a, b) => a - b);

        episodeFilter.innerHTML = '<option value="">All Episodes</option>';
        episodes.forEach(ep => {
            const option = document.createElement('option');
            option.value = ep;
            option.textContent = `Episode ${ep}`;
            episodeFilter.appendChild(option);
        });
        episodeFilter.disabled = false;
    } else {
        episodeFilter.innerHTML = '<option value="">All Episodes</option>';
        episodeFilter.disabled = true;
    }

    renderThumbnails();
}

function loadPanel(index) {
    if (index < 0 || index >= panels.length) return;

    currentPanelIndex = index;
    const panel = panels[index];

    // Load image
    document.getElementById('currentPanel').src = panel.imagePath;
    document.getElementById('panelId').textContent = panel.id;
    document.getElementById('panelDimensions').textContent =
        `${panel.dimensions.width} × ${panel.dimensions.height}px`;

    // Load tags
    currentTags = {
        characters: [...(panel.manual.characters || [])],
        locations: [...(panel.manual.locations || [])],
        organizations: [...(panel.manual.organizations || [])],
        items: [...(panel.manual.items || [])],
        tags: [...(panel.manual.tags || [])],
        emotions: [...(panel.manual.emotions || [])],
        dialogue: panel.manual.dialogue || false,
        action: panel.manual.action || false,
        notes: panel.manual.notes || ""
    };

    updateFormFromTags();
    renderThumbnails();
    updateNavigationButtons();
}

function updateFormFromTags() {
    // Clear and update tag displays
    renderTags('characters', 'characterTags');
    renderTags('locations', 'locationTags');
    renderTags('organizations', 'organizationTags');
    renderTags('items', 'itemTags');
    renderTags('tags', 'generalTags');
    renderTags('emotions', 'emotionTags');

    // Update checkboxes
    document.getElementById('hasDialogue').checked = currentTags.dialogue;
    document.getElementById('isAction').checked = currentTags.action;

    // Update notes
    document.getElementById('notesInput').value = currentTags.notes;
}

function renderTags(category, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    currentTags[category].forEach((tag, index) => {
        const chip = document.createElement('div');
        chip.className = 'tag-chip';
        chip.innerHTML = `
            <span>${tag}</span>
            <span class="remove" onclick="removeTag('${category}', ${index})">×</span>
        `;
        container.appendChild(chip);
    });
}

function addTag(category, inputId) {
    const input = document.getElementById(inputId);
    const value = input.value.trim();

    if (value && !currentTags[category].includes(value)) {
        currentTags[category].push(value);
        renderTags(category, category + 'Tags');
        input.value = '';
    }
}

function addPredefinedTag(category, value) {
    if (!currentTags[category].includes(value)) {
        currentTags[category].push(value);
        renderTags(category, category + 'Tags');
    }
}

function removeTag(category, index) {
    currentTags[category].splice(index, 1);
    renderTags(category, category + 'Tags');
}

function navigatePanel(direction) {
    const newIndex = currentPanelIndex + direction;
    if (newIndex >= 0 && newIndex < panels.length) {
        loadPanel(newIndex);
    }
}

function updateNavigationButtons() {
    document.getElementById('prevPanel').disabled = currentPanelIndex === 0;
    document.getElementById('nextPanel').disabled = currentPanelIndex === panels.length - 1;
}

function saveTags() {
    const panel = panels[currentPanelIndex];

    // Update panel with current tags
    panel.manual = {
        characters: currentTags.characters,
        locations: currentTags.locations,
        organizations: currentTags.organizations,
        items: currentTags.items,
        tags: currentTags.tags,
        emotions: currentTags.emotions,
        dialogue: currentTags.dialogue,
        action: currentTags.action,
        notes: currentTags.notes
    };

    panel.metadata.tagged = true;
    panel.metadata.lastModified = new Date().toISOString();

    // Save to local storage (temporary solution)
    // In production, this would send to a backend
    saveToLocalStorage();

    // Also export as JSON file
    exportPanelTags(panel);

    showStatus("Tags saved successfully!", "success");
    updateStats();
    renderThumbnails();
}

function saveToLocalStorage() {
    try {
        localStorage.setItem('handjumper_panels', JSON.stringify(panels));
    } catch (error) {
        console.warn("Could not save to localStorage:", error);
    }
}

function exportPanelTags(panel) {
    // Create JSON file for manual tags
    const tagData = {
        id: panel.id,
        season: panel.season,
        episode: panel.episode,
        panelNumber: panel.panelNumber,
        manual: panel.manual,
        lastModified: panel.metadata.lastModified
    };

    // Create download link
    const dataStr = JSON.stringify(tagData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `${panel.id}_tags.json`;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function clearForm() {
    currentTags = {
        characters: [],
        locations: [],
        organizations: [],
        items: [],
        tags: [],
        emotions: [],
        dialogue: false,
        action: false,
        notes: ""
    };
    updateFormFromTags();
}

function saveAndNext() {
    saveTags();
    setTimeout(() => navigatePanel(1), 500);
}

function showStatus(message, type) {
    const statusEl = document.getElementById('saveStatus');
    statusEl.textContent = message;
    statusEl.className = `save-status ${type}`;

    setTimeout(() => {
        statusEl.style.display = 'none';
    }, 3000);
}

function updateStats() {
    const total = panels.length;
    const tagged = panels.filter(p => p.metadata.tagged).length;
    const remaining = total - tagged;

    document.getElementById('totalPanels').textContent = total;
    document.getElementById('taggedPanels').textContent = tagged;
    document.getElementById('remainingPanels').textContent = remaining;
}

// Load from localStorage if available
window.addEventListener('load', () => {
    try {
        const saved = localStorage.getItem('handjumper_panels');
        if (saved) {
            const savedPanels = JSON.parse(saved);
            console.log("Loaded saved progress from localStorage");
            // Could merge with loaded data here
        }
    } catch (error) {
        console.warn("Could not load from localStorage:", error);
    }

    init();
});
