/* ================================================================
   RestoreAI — Frontend Application Logic
   ================================================================ */
const API = "http://127.0.0.1:6060/api";

// ── DOM refs ──────────────────────────────────────────────────────
const dropzone          = document.getElementById("dropzone");
const fileInput         = document.getElementById("file-input");
const inputPreviewWrap  = document.getElementById("input-preview-wrap");
const inputPreview      = document.getElementById("input-preview");
const clearBtn          = document.getElementById("clear-btn");
const runBtn            = document.getElementById("run-btn");
const runLabel          = document.getElementById("run-label");
const modeSelect        = document.getElementById("mode-select");
const degradeToggle     = document.getElementById("degrade-toggle");
const intensityRow      = document.getElementById("intensity-row");
const intensitySlider   = document.getElementById("intensity-slider");
const intensityVal      = document.getElementById("intensity-val");
const sharpenSlider     = document.getElementById("sharpen-slider");
const sharpenVal        = document.getElementById("sharpen-val");
const upscaleToggle     = document.getElementById("upscale-toggle");
const checkpointSelect  = document.getElementById("checkpoint-select");
const loadBtn           = document.getElementById("load-btn");

const statusDot         = document.getElementById("status-dot");
const statusText        = document.getElementById("status-text");
const idleState         = document.getElementById("idle-state");
const processingState   = document.getElementById("processing-state");
const resultsGrid       = document.getElementById("results-grid");
const sliderSection     = document.getElementById("slider-section");
const infoStrip         = document.getElementById("info-strip");
const downloadRow       = document.getElementById("download-row");
const toast             = document.getElementById("toast");
const toastMsg          = document.getElementById("toast-msg");
const toastIcon         = document.getElementById("toast-icon");

const imgDegraded       = document.getElementById("img-degraded");
const imgRestored       = document.getElementById("img-restored");
const imgColorized      = document.getElementById("img-colorized");
const compareBefore     = document.getElementById("compare-before");
const compareAfter      = document.getElementById("compare-after");
const compareDivider    = document.getElementById("compare-divider");

const infoCheckpoint    = document.getElementById("info-checkpoint");
const infoEpoch         = document.getElementById("info-epoch");
const infoDevice        = document.getElementById("info-device");

let toastTimer = null;
let currentFile = null;
let lastResults = null;  // store last API results for compare tab switching

// ── TOAST ─────────────────────────────────────────────────────────
function showToast(msg, type = "ok") {
  clearTimeout(toastTimer);
  toastMsg.textContent = msg;
  toastIcon.textContent = type === "error" ? "✕" : "✓";
  toast.className = "toast show" + (type === "error" ? " toast-error" : "");
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

// ── API HELPERS ───────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  try {
    const res = await fetch(API + path, opts);
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
    return json;
  } catch (e) {
    console.error("API error:", e);
    return null;
  }
}

// ── STATUS POLL ───────────────────────────────────────────────────
async function pollStatus() {
  const data = await apiFetch("/status");
  if (!data) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API Offline";
    return;
  }
  statusDot.className = "status-dot online";
  statusText.textContent = data.model_ready
    ? `Epoch ${data.epoch} · ${data.device.toUpperCase()}`
    : "No model loaded";

  // Populate checkpoint dropdown
  const current = checkpointSelect.value;
  checkpointSelect.innerHTML = "";
  (data.available || []).forEach(f => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = f.replace("checkpoint_epoch_", "Epoch ").replace(".pth", "");
    if (f === (data.loaded || current)) opt.selected = true;
    checkpointSelect.appendChild(opt);
  });
  if (!checkpointSelect.value && data.available.length)
    checkpointSelect.value = data.available[data.available.length - 1];

  // Info strip
  if (data.loaded) {
    infoCheckpoint.textContent = data.loaded;
    infoEpoch.textContent      = `${data.epoch} / 150`;
    infoDevice.textContent     = data.device.toUpperCase();
    infoStrip.style.display    = "flex";
  }
}
pollStatus();
setInterval(pollStatus, 10000);  // Refresh every 10s

// ── LOAD CHECKPOINT ───────────────────────────────────────────────
loadBtn.addEventListener("click", async () => {
  const filename = checkpointSelect.value;
  if (!filename) return showToast("Select a checkpoint first", "error");

  loadBtn.textContent = "Loading...";
  loadBtn.disabled = true;
  const data = await apiFetch("/load_checkpoint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  loadBtn.textContent = "Load";
  loadBtn.disabled = false;

  if (!data) return showToast("Failed to load checkpoint", "error");
  showToast(`Loaded Epoch ${data.epoch} successfully!`);
  pollStatus();
});

// ── FILE UPLOAD ───────────────────────────────────────────────────
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showToast("Please upload a valid image file", "error");
    return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  inputPreview.src = url;
  dropzone.style.display = "none";
  inputPreviewWrap.style.display = "block";
  runBtn.disabled = false;

  // Reset results
  idleState.style.display        = "flex";
  processingState.style.display  = "none";
  resultsGrid.style.display      = "none";
  sliderSection.style.display    = "none";
  lastResults = null;
}

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", e => { if (e.key === "Enter") fileInput.click(); });
fileInput.addEventListener("change", e => e.target.files[0] && setFile(e.target.files[0]));

dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  setFile(e.dataTransfer.files[0]);
});

clearBtn.addEventListener("click", () => {
  currentFile = null;
  fileInput.value = "";
  inputPreviewWrap.style.display = "none";
  dropzone.style.display         = "flex";
  runBtn.disabled                = true;
  idleState.style.display        = "flex";
  processingState.style.display  = "none";
  resultsGrid.style.display      = "none";
  sliderSection.style.display    = "none";
  lastResults = null;
});

// ── CONTROLS ─────────────────────────────────────────────────────
degradeToggle.addEventListener("change", () => {
  intensityRow.style.display = degradeToggle.checked ? "flex" : "none";
});
intensitySlider.addEventListener("input", () => {
  intensityVal.textContent = intensitySlider.value;
});
sharpenSlider.addEventListener("input", () => {
  sharpenVal.textContent = (sharpenSlider.value / 10).toFixed(1);
});

// ── STAGE ANIMATION ───────────────────────────────────────────────
function setStage(stage) {
  ["degrade", "restore", "color"].forEach(s => {
    const el = document.getElementById(`stage-${s}`);
    el.classList.remove("active", "done");
  });
  const stages = ["degrade", "restore", "color"];
  const idx = stages.indexOf(stage);
  stages.forEach((s, i) => {
    const el = document.getElementById(`stage-${s}`);
    if (i < idx) el.classList.add("done");
    else if (i === idx) el.classList.add("active");
  });
}

// ── RUN PIPELINE ──────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  // Show processing state
  idleState.style.display       = "none";
  resultsGrid.style.display     = "none";
  sliderSection.style.display   = "none";
  processingState.style.display = "flex";
  runBtn.disabled               = true;
  runLabel.textContent          = "Processing...";

  const stageDelay = ms => new Promise(r => setTimeout(r, ms));

  setStage("degrade");
  await stageDelay(500);
  setStage("restore");

  const formData = new FormData();
  formData.append("image", currentFile);
  formData.append("add_degradation", degradeToggle.checked);
  formData.append("scratch_intensity", intensitySlider.value);
  formData.append("sharpen_amount", (sharpenSlider.value / 10).toFixed(1));
  formData.append("upscale", upscaleToggle.checked);
  formData.append("mode", modeSelect.value);

  const data = await fetch(`${API}/process`, { method: "POST", body: formData })
    .then(r => r.ok ? r.json() : null)
    .catch(() => null);

  setStage("color");
  await stageDelay(400);

  runBtn.disabled      = false;
  runLabel.textContent = "Run AI Pipeline";
  processingState.style.display = "none";

  if (!data) {
    idleState.style.display = "flex";
    return showToast("Pipeline failed. Is the API running?", "error");
  }

  lastResults = data;
  renderResults(data);
  showToast("Pipeline complete!");

  // Update info strip epoch
  if (data.epoch) {
    infoEpoch.textContent = `${data.epoch} / 150`;
  }
});

// ── RENDER RESULTS ────────────────────────────────────────────────
function renderResults(data) {
  const b64 = (str) => `data:image/png;base64,${str}`;

  imgDegraded.src  = b64(data.degraded);
  imgRestored.src  = b64(data.restored);
  imgColorized.src = b64(data.colorized);

  resultsGrid.style.display   = "grid";
  sliderSection.style.display = "flex";

  // Default: full comparison (degraded vs colorized)
  compareBefore.src = b64(data.degraded);
  compareAfter.src  = b64(data.colorized);
  compareAfter.style.clipPath = "inset(0 50% 0 0)";
  compareDivider.style.left   = "50%";

  // Animate cards
  document.querySelectorAll(".result-card").forEach((c, i) => {
    c.style.animationDelay = `${i * 0.1}s`;
  });
}

// ── BEFORE/AFTER SLIDER ───────────────────────────────────────────
let isDragging = false;

function updateSlider(x) {
  const rect = document.getElementById("compare-wrap").getBoundingClientRect();
  let pct = Math.min(Math.max((x - rect.left) / rect.width, 0.02), 0.98);
  const rightClip = `${((1 - pct) * 100).toFixed(2)}%`;
  compareAfter.style.clipPath  = `inset(0 ${rightClip} 0 0)`;
  compareDivider.style.left    = `${(pct * 100).toFixed(2)}%`;
}

const compareWrap = document.getElementById("compare-wrap");
compareWrap.addEventListener("mousedown",  e => { isDragging = true; updateSlider(e.clientX); });
compareWrap.addEventListener("touchstart", e => { isDragging = true; updateSlider(e.touches[0].clientX); }, { passive: true });
window.addEventListener("mousemove",  e => { if (isDragging) updateSlider(e.clientX); });
window.addEventListener("touchmove",  e => { if (isDragging) updateSlider(e.touches[0].clientX); }, { passive: true });
window.addEventListener("mouseup",  () => isDragging = false);
window.addEventListener("touchend", () => isDragging = false);

// ── COMPARE TABS ─────────────────────────────────────────────────
document.querySelectorAll(".stab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".stab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    if (!lastResults) return;
    const b64 = str => `data:image/png;base64,${str}`;
    if (tab.dataset.pair === "degraded-colorized") {
      compareBefore.src = b64(lastResults.degraded);
      compareAfter.src  = b64(lastResults.colorized);
    } else {
      compareBefore.src = b64(lastResults.degraded);
      compareAfter.src  = b64(lastResults.restored);
    }
    // Reset slider to middle
    compareAfter.style.clipPath = "inset(0 50% 0 0)";
    compareDivider.style.left   = "50%";
  });
});

// ── DOWNLOAD ─────────────────────────────────────────────────────
function downloadB64(b64Data, filename) {
  const link   = document.createElement("a");
  link.href    = `data:image/png;base64,${b64Data}`;
  link.download = filename;
  link.click();
}

document.getElementById("dl-degraded").addEventListener("click", () => {
  if (lastResults) downloadB64(lastResults.degraded, "degraded_input.png");
});
document.getElementById("dl-restored").addEventListener("click", () => {
  if (lastResults) downloadB64(lastResults.restored, "restored.png");
});
document.getElementById("dl-colorized").addEventListener("click", () => {
  if (lastResults) downloadB64(lastResults.colorized, "colorized.png");
});
