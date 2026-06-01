// State Variables
let curriculumData = null;
let activePhaseId = 1;
let activeDayNumber = 1;
let filterStatus = 'all'; // 'all', 'todo', 'completed'
let searchQuery = '';

// LocalStorage Progress Template
let userProgress = {
    completedDays: [], // Array of completed day numbers
    dayNotes: {},      // Day number -> notes string
    dayTasksChecked: {}, // Day number -> Array of booleans representing checked subtasks
    streakCount: 0,
    lastCompletedDate: null
};

// SVG circular gauge variables
const CIRCLE_RADIUS = 50;
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * CIRCLE_RADIUS;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initProgressRing();
    loadProgress();
    fetchCurriculum();
    setupEventListeners();
});

// Initialize SVG Progress Ring Circle
function initProgressRing() {
    const circle = document.getElementById('global-gauge');
    if (circle) {
        circle.style.strokeDasharray = `${CIRCLE_CIRCUMFERENCE} ${CIRCLE_CIRCUMFERENCE}`;
        circle.style.strokeDashoffset = CIRCLE_CIRCUMFERENCE;
    }
}

// Fetch Curriculum data from curriculum.json
async function fetchCurriculum() {
    try {
        const response = await fetch('curriculum.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        curriculumData = await response.json();
        
        renderPhaseNavigation();
        renderActivePhaseBanner();
        renderDaysList();
        
        // Select first day by default
        selectDay(activeDayNumber);
        updateProgressUI();
    } catch (e) {
        console.error('Could not load curriculum data: ', e);
        document.getElementById('days-list-container').innerHTML = `
            <div style="padding:20px; text-align:center; color: var(--color-red);">
                <i class="fa-solid fa-triangle-exclamation" style="font-size: 30px; margin-bottom:10px;"></i>
                <p>Failed to load curriculum.json. Please make sure the JSON file exists and is formatted correctly.</p>
            </div>
        `;
    }
}

// Load Progress from LocalStorage
function loadProgress() {
    const saved = localStorage.getItem('ds_roadmap_100_days_progress');
    if (saved) {
        try {
            userProgress = JSON.parse(saved);
            // Ensure fields exist
            if (!userProgress.completedDays) userProgress.completedDays = [];
            if (!userProgress.dayNotes) userProgress.dayNotes = {};
            if (!userProgress.dayTasksChecked) userProgress.dayTasksChecked = {};
            if (userProgress.streakCount === undefined) userProgress.streakCount = 0;
            
            // Check streak expiration
            validateStreak();
        } catch (e) {
            console.error('Error parsing local storage progress. Resetting...', e);
            saveProgress();
        }
    } else {
        saveProgress();
    }
}

// Save Progress to LocalStorage
function saveProgress() {
    localStorage.setItem('ds_roadmap_100_days_progress', JSON.stringify(userProgress));
}

// Validate if streak is broken (more than 24 hours of inactivity)
function validateStreak() {
    if (!userProgress.lastCompletedDate) {
        userProgress.streakCount = 0;
        return;
    }
    
    const today = getLocalDateString();
    const lastDate = userProgress.lastCompletedDate;
    
    if (today === lastDate) {
        return; // Streak is valid (completed today)
    }
    
    // Check if yesterday was last completed date
    const yesterday = getLocalDateString(new Date(Date.now() - 86400000));
    if (lastDate !== yesterday) {
        // More than 1 day missed, reset streak
        userProgress.streakCount = 0;
        saveProgress();
    }
}

// Get Local Date string in YYYY-MM-DD format
function getLocalDateString(dateObj = new Date()) {
    const offset = dateObj.getTimezoneOffset();
    const adjustedDate = new Date(dateObj.getTime() - (offset * 60 * 1000));
    return adjustedDate.toISOString().split('T')[0];
}

// Setup Event Listeners
function setupEventListeners() {
    // Search Bar Input
    const searchInput = document.getElementById('search-input');
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value.toLowerCase();
        renderDaysList();
    });

    // Reset Button
    const resetBtn = document.getElementById('reset-progress-btn');
    resetBtn.addEventListener('click', () => {
        if (confirm('Are you absolutely sure you want to reset all progress, notes, and streak count? This cannot be undone.')) {
            userProgress = {
                completedDays: [],
                dayNotes: {},
                dayTasksChecked: {},
                streakCount: 0,
                lastCompletedDate: null
            };
            saveProgress();
            
            // Re-render
            updateProgressUI();
            renderPhaseNavigation();
            renderActivePhaseBanner();
            renderDaysList();
            if (activeDayNumber) selectDay(activeDayNumber);
            alert('All progress has been reset successfully.');
        }
    });

    // Filter pills
    const pills = document.querySelectorAll('.filter-pill');
    pills.forEach(pill => {
        pill.addEventListener('click', (e) => {
            pills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            filterStatus = pill.getAttribute('data-filter');
            renderDaysList();
        });
    });

    // Complete Toggle Button in Detail Pane
    const completeToggle = document.getElementById('detail-complete-toggle');
    completeToggle.addEventListener('click', () => {
        toggleDayCompleted(activeDayNumber);
    });

    // Tab switching
    const tabs = document.querySelectorAll('.detail-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const targetTab = tab.getAttribute('data-tab');
            if (targetTab === 'content') {
                document.getElementById('tab-content').classList.remove('hidden');
                document.getElementById('tab-notes').classList.add('hidden');
            } else {
                document.getElementById('tab-content').classList.add('hidden');
                document.getElementById('tab-notes').classList.remove('hidden');
            }
        });
    });

    // Notes textarea changes
    const textarea = document.getElementById('notes-textarea');
    textarea.addEventListener('input', (e) => {
        saveNotes(activeDayNumber, e.target.value);
    });
}

// Render Phase navigation sidebar list
function renderPhaseNavigation() {
    const list = document.getElementById('phase-list');
    if (!list || !curriculumData) return;
    
    list.innerHTML = '';
    
    curriculumData.phases.forEach(phase => {
        // Calculate completed days in this phase
        const phaseDays = phase.days.map(d => d.day);
        const completedInPhase = phaseDays.filter(d => userProgress.completedDays.includes(d)).length;
        const totalInPhase = phaseDays.length;
        const isPhaseComplete = completedInPhase === totalInPhase;
        
        const btn = document.createElement('button');
        btn.className = `phase-item-btn ${phase.id === activePhaseId ? 'active' : ''}`;
        btn.onclick = () => selectPhase(phase.id);
        
        btn.innerHTML = `
            <div class="phase-btn-content">
                <span class="phase-btn-name">${phase.name}</span>
                <span class="phase-btn-days">${completedInPhase}/${totalInPhase} Days</span>
            </div>
            <span class="phase-btn-badge ${isPhaseComplete ? 'completed' : ''}">
                ${isPhaseComplete ? '<i class="fa-solid fa-check"></i>' : Math.round((completedInPhase / totalInPhase) * 100) + '%'}
            </span>
        `;
        
        list.appendChild(btn);
    });
}

// Select a specific phase
function selectPhase(phaseId) {
    activePhaseId = phaseId;
    renderPhaseNavigation();
    renderActivePhaseBanner();
    renderDaysList();
    
    // Find first day of the new phase and select it
    const phase = curriculumData.phases.find(p => p.id === phaseId);
    if (phase && phase.days.length > 0) {
        selectDay(phase.days[0].day);
    }
}

// Render the active phase banner details
function renderActivePhaseBanner() {
    const bannerTitle = document.getElementById('banner-phase-name');
    const fill = document.getElementById('banner-progress-fill');
    const text = document.getElementById('banner-progress-text');
    
    if (!curriculumData) return;
    
    const phase = curriculumData.phases.find(p => p.id === activePhaseId);
    if (!phase) return;
    
    bannerTitle.textContent = phase.name;
    
    const phaseDays = phase.days.map(d => d.day);
    const completed = phaseDays.filter(d => userProgress.completedDays.includes(d)).length;
    const total = phaseDays.length;
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    fill.style.width = `${percent}%`;
    text.textContent = `${percent}%`;
}

// Render Days List for the active phase (Left Panel)
function renderDaysList() {
    const container = document.getElementById('days-list-container');
    if (!container || !curriculumData) return;
    
    container.innerHTML = '';
    
    const phase = curriculumData.phases.find(p => p.id === activePhaseId);
    if (!phase) return;
    
    // Filter days based on query and status filter
    const filteredDays = phase.days.filter(day => {
        // Filter by search query
        const matchesSearch = day.title.toLowerCase().includes(searchQuery) || 
                              day.description.toLowerCase().includes(searchQuery) ||
                              day.exercise.toLowerCase().includes(searchQuery);
                              
        if (!matchesSearch) return false;
        
        // Filter by completion status
        const isCompleted = userProgress.completedDays.includes(day.day);
        if (filterStatus === 'todo') return !isCompleted;
        if (filterStatus === 'completed') return isCompleted;
        
        return true;
    });
    
    if (filteredDays.length === 0) {
        container.innerHTML = `
            <div style="text-align:center; padding:30px; color: var(--text-muted);">
                <i class="fa-solid fa-magnifying-glass" style="font-size:24px; margin-bottom:10px;"></i>
                <p>No days found matching current filters.</p>
            </div>
        `;
        return;
    }
    
    filteredDays.forEach(day => {
        const card = document.createElement('div');
        const isCompleted = userProgress.completedDays.includes(day.day);
        
        card.className = `day-card ${isCompleted ? 'completed' : ''} ${day.day === activeDayNumber ? 'selected' : ''}`;
        
        // Card content
        card.innerHTML = `
            <div class="day-card-left">
                <div class="card-checkbox" onclick="event.stopPropagation(); toggleDayCompleted(${day.day})">
                    <i class="fa-solid fa-check"></i>
                </div>
                <div class="card-info" onclick="selectDay(${day.day})">
                    <span class="card-day-num">Day ${day.day}</span>
                    <span class="card-title" title="${day.title}">${day.title}</span>
                </div>
            </div>
            <i class="fa-solid fa-chevron-right day-card-arrow" onclick="selectDay(${day.day})"></i>
        `;
        
        container.appendChild(card);
    });
}

// Select a specific day to show in details panel (Right Panel)
function selectDay(dayNumber) {
    if (!curriculumData) return;
    
    activeDayNumber = dayNumber;
    
    // Find the day details in curriculum JSON
    let selectedDayData = null;
    let selectedPhaseId = 1;
    
    for (const phase of curriculumData.phases) {
        const found = phase.days.find(d => d.day === dayNumber);
        if (found) {
            selectedDayData = found;
            selectedPhaseId = phase.id;
            break;
        }
    }
    
    if (!selectedDayData) return;
    
    // Update active phase if it differs
    if (selectedPhaseId !== activePhaseId) {
        activePhaseId = selectedPhaseId;
        renderPhaseNavigation();
        renderActivePhaseBanner();
    }
    
    // Update selected class in left list
    const cards = document.querySelectorAll('.day-card');
    cards.forEach(card => {
        card.classList.remove('selected');
    });
    // Add selected class dynamically (re-rendering list does this too, but this is snappy)
    renderDaysList();
    
    // Unhide the details pane
    document.getElementById('detail-empty-state').classList.add('hidden');
    const detailsContainer = document.getElementById('day-details-container');
    detailsContainer.classList.remove('hidden');
    
    // Populate details data
    document.getElementById('detail-day-number').textContent = `DAY ${selectedDayData.day}`;
    document.getElementById('detail-day-title').textContent = selectedDayData.title;
    document.getElementById('detail-day-description').textContent = selectedDayData.description;
    document.getElementById('detail-day-exercise').textContent = selectedDayData.exercise;
    
    // Update Day Checklist subtasks
    renderSubtasks(selectedDayData);
    
    // Update Resources grid
    renderResources(selectedDayData);
    
    // Load existing notes
    const textarea = document.getElementById('notes-textarea');
    textarea.value = userProgress.dayNotes[dayNumber] || '';
    updateNotesPreview(textarea.value);
    
    // Update Complete toggle button UI
    updateDayCompleteToggleUI(dayNumber);
}

// Render daily subtasks list
function renderSubtasks(dayData) {
    const list = document.getElementById('detail-tasks-list');
    list.innerHTML = '';
    
    const dayNum = dayData.day;
    const taskStates = userProgress.dayTasksChecked[dayNum] || new Array(dayData.tasks.length).fill(false);
    
    dayData.tasks.forEach((task, index) => {
        const isChecked = taskStates[index] || false;
        
        const item = document.createElement('li');
        item.className = `task-item ${isChecked ? 'checked' : ''}`;
        
        item.innerHTML = `
            <input type="checkbox" id="subtask-${index}" ${isChecked ? 'checked' : ''}>
            <span>${task}</span>
        `;
        
        // Handle clicking anywhere on the item row
        item.onclick = (e) => {
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (e.target !== checkbox) {
                checkbox.checked = !checkbox.checked;
            }
            toggleSubtaskState(dayNum, index, checkbox.checked, dayData.tasks.length);
            
            if (checkbox.checked) {
                item.classList.add('checked');
            } else {
                item.classList.remove('checked');
            }
        };
        
        list.appendChild(item);
    });
    
    updateSubtaskProgressBar(dayNum, dayData.tasks.length);
}

// Toggle subtask checked state
function toggleSubtaskState(dayNum, taskIndex, isChecked, totalTasks) {
    if (!userProgress.dayTasksChecked[dayNum]) {
        userProgress.dayTasksChecked[dayNum] = new Array(totalTasks).fill(false);
    }
    
    userProgress.dayTasksChecked[dayNum][taskIndex] = isChecked;
    saveProgress();
    
    updateSubtaskProgressBar(dayNum, totalTasks);
    
    // Auto-complete the day if all subtasks are checked! (Nice UX)
    const allChecked = userProgress.dayTasksChecked[dayNum].every(t => t === true);
    const dayCompleted = userProgress.completedDays.includes(dayNum);
    
    if (allChecked && !dayCompleted) {
        toggleDayCompleted(dayNum, true); // Mark complete
    } else if (!allChecked && dayCompleted) {
        toggleDayCompleted(dayNum, false); // Mark incomplete
    }
}

// Update subtask progress bar fill
function updateSubtaskProgressBar(dayNum, totalTasks) {
    const fill = document.getElementById('detail-task-progress-fill');
    if (!fill) return;
    
    const taskStates = userProgress.dayTasksChecked[dayNum] || [];
    const checkedCount = taskStates.filter(t => t === true).length;
    
    const percent = totalTasks > 0 ? Math.round((checkedCount / totalTasks) * 100) : 0;
    fill.style.width = `${percent}%`;
}

// Render resources list
function renderResources(dayData) {
    const grid = document.getElementById('detail-resources-grid');
    grid.innerHTML = '';
    
    dayData.resources.forEach(res => {
        const card = document.createElement('a');
        card.className = 'resource-card';
        card.href = res.url;
        card.target = '_blank';
        
        card.innerHTML = `
            <div class="resource-info">
                <i class="fa-solid fa-circle-play resource-icon"></i>
                <span class="resource-name">${res.name}</span>
            </div>
            <i class="fa-solid fa-arrow-up-right-from-square resource-arrow"></i>
        `;
        
        grid.appendChild(card);
    });
}

// Update the "Mark Complete" toggle button styling and text
function updateDayCompleteToggleUI(dayNumber) {
    const btn = document.getElementById('detail-complete-toggle');
    const isCompleted = userProgress.completedDays.includes(dayNumber);
    
    if (isCompleted) {
        btn.classList.add('is-completed');
        btn.innerHTML = `<i class="fa-solid fa-circle-check"></i> Completed`;
    } else {
        btn.classList.remove('is-completed');
        btn.innerHTML = `<i class="fa-regular fa-circle-check"></i> Mark Complete`;
    }
}

// Toggle overall completion state of a specific day
function toggleDayCompleted(dayNumber, forceState = null) {
    const index = userProgress.completedDays.indexOf(dayNumber);
    const currentlyCompleted = index !== -1;
    
    let targetState = forceState;
    if (targetState === null) {
        targetState = !currentlyCompleted;
    }
    
    if (targetState === currentlyCompleted) return; // No change
    
    if (targetState) {
        // Mark complete
        userProgress.completedDays.push(dayNumber);
        
        // Handle streak incrementing
        handleStreakIncrement();
        
        // Also autocheck all subtasks if manually marking day completed
        autoCheckAllSubtasks(dayNumber, true);
    } else {
        // Mark incomplete
        if (index !== -1) {
            userProgress.completedDays.splice(index, 1);
        }
        
        // Auto-uncheck all subtasks
        autoCheckAllSubtasks(dayNumber, false);
    }
    
    saveProgress();
    
    // Refresh UIs
    updateDayCompleteToggleUI(dayNumber);
    updateProgressUI();
    renderPhaseNavigation();
    renderActivePhaseBanner();
    renderDaysList();
}

// Helper to auto-check or uncheck all subtasks when day status changes
function autoCheckAllSubtasks(dayNumber, checkAll) {
    let dayData = null;
    for (const phase of curriculumData.phases) {
        const found = phase.days.find(d => d.day === dayNumber);
        if (found) { dayData = found; break; }
    }
    
    if (dayData) {
        const total = dayData.tasks.length;
        userProgress.dayTasksChecked[dayNumber] = new Array(total).fill(checkAll);
        
        // Update checkbox list if active
        if (activeDayNumber === dayNumber) {
            renderSubtasks(dayData);
        }
    }
}

// Handle streak tracking calculations
function handleStreakIncrement() {
    const todayStr = getLocalDateString();
    const lastDate = userProgress.lastCompletedDate;
    
    if (!lastDate) {
        // First day completed
        userProgress.streakCount = 1;
    } else if (lastDate === todayStr) {
        // Completed another day today, streak doesn't change
    } else {
        // Check if last completion was yesterday
        const yesterdayStr = getLocalDateString(new Date(Date.now() - 86400000));
        if (lastDate === yesterdayStr) {
            userProgress.streakCount += 1;
        } else {
            // Missed day(s), reset streak to 1
            userProgress.streakCount = 1;
        }
    }
    
    userProgress.lastCompletedDate = todayStr;
}

// Save study notes to local storage
function saveNotes(dayNumber, noteText) {
    userProgress.dayNotes[dayNumber] = noteText;
    saveProgress();
    updateNotesPreview(noteText);
}

// Render markdown notes preview (Basic renderer)
function updateNotesPreview(markdownText) {
    const preview = document.getElementById('notes-preview');
    if (!preview) return;
    
    if (!markdownText.trim()) {
        preview.innerHTML = `<p class="preview-placeholder">Notes preview will render here...</p>`;
        return;
    }
    
    // Parse simple Markdown
    let html = markdownText
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        // Headings (e.g. ### Title)
        .replace(/^### (.*$)/gim, '<h5 style="font-size:14px; font-weight:700; margin:12px 0 6px 0; color: #FFF;">$1</h5>')
        .replace(/^## (.*$)/gim, '<h4 style="font-size:16px; font-weight:700; margin:14px 0 8px 0; color: #FFF;">$1</h4>')
        .replace(/^# (.*$)/gim, '<h3 style="font-size:18px; font-weight:800; margin:16px 0 10px 0; color: #FFF;">$1</h3>')
        // Bold (**text**)
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        // Italics (*text*)
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        // Code block (inline `code`)
        .replace(/`(.*?)`/gim, '<code style="font-family:monospace; background:rgba(255,255,255,0.08); padding:2px 4px; border-radius:4px; color:#FFF;">$1</code>')
        // Lists (- item)
        .replace(/^\s*-\s*(.*$)/gim, '<li style="margin-left:16px; font-size:13px; color:var(--text-secondary);">$1</li>')
        // Linebreaks
        .replace(/\n/g, '<br>');
        
    preview.innerHTML = html;
}

// Update overall dashboard progress bar and gauges
function updateProgressUI() {
    const completedCount = userProgress.completedDays.length;
    const totalCount = 100;
    const percentage = Math.round((completedCount / totalCount) * 100);
    
    // Set numeric labels
    document.getElementById('progress-percent').textContent = `${percentage}%`;
    document.getElementById('completed-days-count').textContent = completedCount;
    document.getElementById('streak-count').textContent = `${userProgress.streakCount} Day${userProgress.streakCount === 1 ? '' : 's'}`;
    
    // Set Circular SVG gauge stroke offset
    const circle = document.getElementById('global-gauge');
    if (circle) {
        const offset = CIRCLE_CIRCUMFERENCE - (percentage / 100) * CIRCLE_CIRCUMFERENCE;
        circle.style.strokeDashoffset = offset;
    }
}
