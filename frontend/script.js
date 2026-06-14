/* script.js – MuseAI v3
 * All API calls, element IDs, and music generation logic are untouched.
 * Changes: status pill wording, cleaner toast, song item markup, count label.
 */
"use strict";

/* ── Config ──────────────────────────────────────────────────────────────────*/
const BASE = () =>
  typeof CONFIG !== "undefined" ? CONFIG.BACKEND_URL : "http://localhost:5000";

async function api(path, options = {}) {
  const res = await fetch(BASE() + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/* ── DOM helpers ─────────────────────────────────────────────────────────────*/
const $  = (id) => document.getElementById(id);
function show(el) { el && el.removeAttribute("hidden"); }
function hide(el) { el && el.setAttribute("hidden", ""); }
function setText(id, txt) { const el = $(id); if (el) el.textContent = txt; }
function fmtDate(iso) {
  return new Date(iso + "Z").toLocaleString(undefined, {
    dateStyle: "short", timeStyle: "short",
  });
}

/* ── Toast ───────────────────────────────────────────────────────────────────*/
const ICONS = { success: "✓", error: "✕", info: "♪" };

function toast(msg, type = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast toast--${type}`;
  el.setAttribute("role", "alert");
  el.textContent = `${ICONS[type] ?? ""}\u2002${msg}`;
  $("toast-container").appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateX(110%)";
    setTimeout(() => el.remove(), 280);
  }, ms);
}

/* ── Status pill (health-badge element) ──────────────────────────────────────*/
function setStatus(state) {
  // state: "pending" | "online" | "offline"
  const pill  = $("health-badge");
  const label = pill.querySelector(".status-label");
  pill.className = `status-pill status-pill--${state}`;
  label.textContent = state === "online" ? "Online" : state === "offline" ? "Offline" : "Online";
}

/* ── Health check ────────────────────────────────────────────────────────────*/
async function checkHealth() {
  try {
    const data = await api("/health");
    setStatus("online");
    if (data.model_exists) {
      $("btn-generate").disabled = false;
      $("btn-download").disabled = false;
    }
  } catch {
    setStatus("offline");
  }
}

/* ── Training ────────────────────────────────────────────────────────────────*/
let trainPollTimer = null;

async function startTraining() {
  const epochs = parseInt($("epochs-input").value, 10) || 50;
  const btn = $("btn-train");
  const box = $("train-status");

  btn.disabled = true;
  show(box);
  $("train-spinner").classList.remove("hidden");
  setText("train-status-text", "Sending request…");
  $("train-progress-fill").style.width = "0%";
  $("train-progress-fill").closest("[role=progressbar]")?.setAttribute("aria-valuenow", "0");

  try {
    await api("/train", { method: "POST", body: JSON.stringify({ epochs }) });
    toast("Music engine preparation started!", "info");
    pollTraining(epochs);
  } catch (err) {
    toast(`Could not start: ${err.message}`, "error");
    setText("train-status-text", `Error: ${err.message}`);
    $("train-spinner").classList.add("hidden");
    btn.disabled = false;
  }
}

function pollTraining(totalEpochs) {
  let simEpoch = 0;
  clearInterval(trainPollTimer);

  trainPollTimer = setInterval(async () => {
    try {
      const s = await api("/train/status");
      setText("train-status-text", s.progress || "Preparing engine…");

      if (!s.running && !s.finished && !s.error) return;

      simEpoch = Math.min(simEpoch + 1, totalEpochs);
      const pct = Math.round((simEpoch / totalEpochs) * 100);
      const fill = $("train-progress-fill");
      fill.style.width = pct + "%";
      fill.closest("[role=progressbar]")?.setAttribute("aria-valuenow", pct);

      if (s.finished) {
        clearInterval(trainPollTimer);
        fill.style.width = "100%";
        fill.closest("[role=progressbar]")?.setAttribute("aria-valuenow", "100");
        setText("train-status-text", "✓ Engine ready — you can now generate music!");
        $("train-spinner").classList.add("hidden");
        $("btn-train").disabled = false;
        $("btn-generate").disabled = false;
        $("btn-download").disabled = false;
        toast("Music engine is ready!", "success");
        checkHealth();
      }

      if (s.error) {
        clearInterval(trainPollTimer);
        setText("train-status-text", `Something went wrong. Please try again.`);
        $("train-spinner").classList.add("hidden");
        $("btn-train").disabled = false;
        toast(`Preparation failed: ${s.error}`, "error");
      }
    } catch { /* server briefly busy — keep polling */ }
  }, 3000);
}

/* ── Generation ──────────────────────────────────────────────────────────────*/
async function generateMusic() {
  const length      = parseInt($("length-input").value, 10) || 500;
  const temperature = parseFloat($("temp-input").value) || 1.0;
  const seedRaw     = $("seed-input").value.trim();
  const seed        = seedRaw !== "" ? parseInt(seedRaw, 10) : null;

  const btn = $("btn-generate");
  const box = $("gen-status");

  btn.disabled = true;
  show(box);
  $("gen-spinner").classList.remove("hidden");
  setText("gen-status-text", "Composing your melody — this may take a moment…");

  try {
    const data = await api("/generate", {
      method: "POST",
      body: JSON.stringify({ length, temperature, seed }),
    });

    toast(`Composition ready: ${data.filename}`, "success");
    setText("gen-status-text", `✓ Created: ${data.filename}`);
    $("gen-spinner").classList.add("hidden");
    $("btn-download").disabled = false;
    $("btn-download").dataset.filename = data.filename;
    refreshSongList();
  } catch (err) {
    toast(`Generation error: ${err.message}`, "error");
    setText("gen-status-text", `Could not generate: ${err.message}`);
    $("gen-spinner").classList.add("hidden");
  } finally {
    btn.disabled = false;
  }
}

/* ── Download ────────────────────────────────────────────────────────────────*/
function downloadFile(filename) {
  const url = BASE() + "/download" + (filename ? `?filename=${encodeURIComponent(filename)}` : "");
  const a   = Object.assign(document.createElement("a"), { href: url, download: filename || "generated.mid" });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ── Song list ───────────────────────────────────────────────────────────────*/
async function refreshSongList() {
  const list    = $("song-list");
  const countEl = $("library-count");

  try {
    const songs = await api("/songs");

    if (countEl) {
      countEl.textContent =
        songs.length === 0 ? "0 compositions" :
        songs.length === 1 ? "1 composition"  :
        `${songs.length} compositions`;
    }

    list.innerHTML = "";

    if (!songs.length) {
      list.innerHTML = `
        <li class="empty-state-item">
          <div class="empty-state">
            <div class="empty-state__icon" aria-hidden="true">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" stroke-width="1.4">
                <path d="M9 18V5l12-2v13"/>
                <circle cx="6" cy="18" r="3"/>
                <circle cx="18" cy="16" r="3"/>
              </svg>
            </div>
            <p class="empty-state__title">No compositions yet</p>
            <p class="empty-state__body">Generate your first melody above and it will appear here.</p>
          </div>
        </li>`;
      return;
    }

    songs.forEach((s) => {
      const li  = document.createElement("li");
      li.className = "song-item";
      li.innerHTML = `
        <div class="song-item__icon" aria-hidden="true">♩</div>
        <div class="song-item__info">
          <div class="song-item__name" title="${s.filename}">${s.filename}</div>
          <div class="song-item__date">${fmtDate(s.created_at)}</div>
        </div>
        <button class="song-item__dl" title="Download ${s.filename}"
                aria-label="Download ${s.filename}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download
        </button>`;
      li.querySelector("button").addEventListener("click", () => downloadFile(s.filename));
      list.appendChild(li);
    });
  } catch {
    list.innerHTML = `
      <li class="empty-state-item">
        <div class="empty-state">
          <p class="empty-state__title">Could not load compositions.</p>
        </div>
      </li>`;
  }
}

/* ── Range slider — live gradient + label ────────────────────────────────────*/
function initSlider() {
  const range = $("temp-input");
  const label = $("temp-value");
  if (!range || !label) return;

  const update = () => {
    const val = parseFloat(range.value);
    label.textContent = val.toFixed(2);
    range.setAttribute("aria-valuenow", val);
    const pct = ((val - +range.min) / (+range.max - +range.min)) * 100;
    range.style.background =
      `linear-gradient(90deg, var(--violet) ${pct}%, rgba(255,255,255,.08) ${pct}%)`;
  };

  range.addEventListener("input", update);
  update();
}

/* ── Init ────────────────────────────────────────────────────────────────────*/
document.addEventListener("DOMContentLoaded", () => {
  // Wire buttons
  $("btn-train")   .addEventListener("click", startTraining);
  $("btn-generate").addEventListener("click", generateMusic);
  $("btn-download").addEventListener("click", () => {
    downloadFile($("btn-download").dataset.filename || null);
  });

  // Default disabled until health confirms
  $("btn-generate").disabled = true;
  $("btn-download").disabled = true;

  initSlider();
  checkHealth();
  refreshSongList();

  // Re-check every 30 s
  setInterval(checkHealth, 30_000);
});
