const POLL_MS = 600;

// Local parking state — source of truth for UI (not from API)
const parkingState = [false, false, false, false];

// ---- Clock ----
function tickClock() {
    const now    = new Date();
    const h      = String(now.getHours()).padStart(2, '0');
    const m      = String(now.getMinutes()).padStart(2, '0');
    const s      = String(now.getSeconds()).padStart(2, '0');
    const days   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const el_t   = document.getElementById('clock');
    const el_d   = document.getElementById('date');
    if (el_t) el_t.textContent = `${h}:${m}:${s}`;
    if (el_d) el_d.textContent = `${days[now.getDay()]} ${now.getDate()} ${months[now.getMonth()]} ${now.getFullYear()}`;
}
setInterval(tickClock, 1000);
tickClock();

// ---- Slot toggle (click) ----
async function toggleSlot(i) {
    parkingState[i] = !parkingState[i];
    renderSlots();
    try {
        await fetch(`/api/parking/${i}/${parkingState[i] ? 1 : 0}`);
    } catch (e) {
        console.warn('Slot toggle failed:', e);
    }
}

function renderSlots() {
    let vacant = 0, occ = 0;
    for (let i = 0; i < 4; i++) {
        const card  = document.getElementById(`slot-${i}`);
        const state = document.getElementById(`slot-state-${i}`);
        if (!card) continue;
        const occupied = parkingState[i];
        card.classList.toggle('occupied', occupied);
        if (state) state.textContent = occupied ? 'OCCUPIED' : 'VACANT';
        occupied ? occ++ : vacant++;
    }
    const cv = document.getElementById('count-vacant');
    const co = document.getElementById('count-occ');
    if (cv) cv.textContent = vacant;
    if (co) co.textContent = occ;
}

// ---- Status Poll (only cam/sub online + gate + last_seen) ----
async function pollStatus() {
    try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        console.warn('Poll failed:', e);
    }
}

function updateDashboard(data) {
    setOnline('dot-cam', 'val-cam', data.cam_online);
    setOnline('dot-sub', 'val-sub', data.sub_online);

    const gateState = typeof data.servo === 'string'
        ? data.servo
        : (Array.isArray(data.servo) ? data.servo[0] : 'Closed');
    setGate('gate-main', gateState);

    const ls = document.getElementById('last-seen');
    if (ls) ls.textContent = data.last_seen || '—';
}

function setOnline(dotId, valId, online) {
    const dot = document.getElementById(dotId);
    const val = document.getElementById(valId);
    if (dot) {
        dot.classList.toggle('online',  !!online);
        dot.classList.toggle('offline', !online);
    }
    if (val) {
        val.textContent = online ? 'ONLINE' : 'OFFLINE';
        val.classList.toggle('online', !!online);
    }
}

function setGate(id, state) {
    const el = document.getElementById(id);
    if (!el) return;
    const isOpen = state === 'Open';
    el.textContent = isOpen ? 'OPEN' : 'CLOSED';
    el.classList.toggle('open', isOpen);
}

// ---- Manual Gate Trigger ----
async function triggerGate(type) {
    try {
        await fetch(`/api/gate/${type}`);
    } catch (e) {
        console.warn('Gate trigger failed:', e);
    }
}

// Init
renderSlots();
setInterval(pollStatus, POLL_MS);
pollStatus();