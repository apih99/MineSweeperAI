// UI: builds the board once, then only touches cells whose appearance changed.
import { Board, mulberry32, PRESETS } from "./game.js";
import { Agent, MODELS, modelKeyFor } from "./agent.js";

const ORT_VERSION = "1.23.2";
const DELAYS = [900, 450, 220, 100, 35, 0]; // ms per move for speed 0..5

const $ = (id) => document.getElementById(id);
const els = {
  board: $("board"), wrap: $("board-wrap"), status: $("status"), minesLeft: $("mines-left"),
  legend: $("legend"), run: $("btn-run"), step: $("btn-step"), hint: $("btn-hint"), newGame: $("btn-new"),
  speed: $("speed"), autonext: $("autonext"), preset: $("preset"), seed: $("seed"), heat: $("heat"),
  modelInfo: $("model-info"), tally: $("tally"), tallyPlay: $("tally-play"), ctlWatch: $("controls-watch"), ctlPlay: $("controls-play"),
  tabs: [...document.querySelectorAll("[data-mode]")],
};

const state = {
  mode: "watch", preset: "beginner", board: null, seed: 0, gid: 0,
  flags: null, cells: [], cache: [], focus: 0,
  agent: null, modelKey: "", running: false, looping: false, busy: false,
  q: null, qKey: "", mark: -1, markKind: "", moves: 0,
  tally: { watch: { games: 0, wins: 0 }, play: { games: 0, wins: 0 } },
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const delay = () => DELAYS[+els.speed.value];
const setStatus = (text, cls = "") => { els.status.textContent = text; els.status.className = "status " + cls; };

// ---------- colour: viridis for the heatmap (same colormap as the README GIFs) ----------
const VIRIDIS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"].map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
function viridis(t) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  const c = VIRIDIS[i].map((v, k) => Math.round(v + (VIRIDIS[i + 1][k] - v) * f));
  return `rgb(${c[0]} ${c[1]} ${c[2]})`;
}

// ---------- grid ----------
function buildGrid() {
  const { height: h, width: w } = state.board;
  els.board.textContent = "";
  els.board.style.setProperty("--cols", w);
  state.cells = [];
  state.cache = new Array(h * w).fill("");
  const frag = document.createDocumentFragment();
  for (let i = 0; i < h * w; i++) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "cell";
    b.dataset.i = i;
    b.tabIndex = i === 0 ? 0 : -1;
    state.cells.push(b);
    frag.appendChild(b);
  }
  els.board.appendChild(frag);
  state.focus = 0;
  layoutBoard();
}

function layoutBoard() {
  if (!state.board) return;
  const w = state.board.width;
  const avail = els.wrap.clientWidth - 8;
  const max = w <= 5 ? 56 : 44;
  // Tighter gaps on narrow screens so 16x16 fits a phone without inner scrolling.
  const gap = avail < 420 && w > 9 ? 2 : 3;
  const cell = Math.max(14, Math.min(max, Math.floor((avail - gap * (w - 1)) / w)));
  els.board.style.setProperty("--cell", cell + "px");
  els.board.style.setProperty("--gap", gap + "px");
}

function describe(i) {
  const b = state.board, w = b.width, r = (i / w) | 0, c = i % w;
  let s;
  if (b.revealed[i]) s = b.mines[i] ? "mine" : b.counts[i] === 0 ? "empty" : `${b.counts[i]} adjacent`;
  else s = state.flags[i] ? "flagged" : "hidden";
  return `Row ${r + 1}, column ${c + 1}, ${s}`;
}

function renderCell(i) {
  const b = state.board, el = state.cells[i];
  const open = !!b.revealed[i] && !b.mines[i];
  const showMine = b.mines[i] && (b.revealed[i] || b.hitMine);
  const flagged = !!state.flags[i];
  let cls = "cell";
  let text = "";
  let n = "";
  let bg = "";
  if (open) {
    cls += " open";
    if (b.counts[i] > 0) { text = String(b.counts[i]); n = String(b.counts[i]); }
  } else if (showMine && !flagged) {
    cls += " mine";
    if (i === b.exploded) cls += " boom";
  } else {
    if (flagged) cls += " flag";
    if (b.hitMine && flagged && !b.mines[i]) cls += " wrongflag";
    if (state.q && els.heat.checked && !b.done && !flagged && Number.isFinite(state.q[i])) bg = heatColor(i);
  }
  if (state.mode === "watch" || b.done || open) cls += " locked";
  if (state.mark === i) cls += state.markKind === "hint" ? " hint" : " pick";
  const key = cls + "|" + text + "|" + bg + "|" + (open || showMine || flagged ? describe(i) : "");
  if (state.cache[i] === key) return;
  state.cache[i] = key;
  el.className = cls;
  el.textContent = text;
  if (n) el.dataset.n = n; else delete el.dataset.n;
  el.style.backgroundColor = bg;
  el.setAttribute("aria-label", describe(i));
}

let heatRange = [0, 1];
function heatColor(i) {
  const [lo, hi] = heatRange;
  return viridis(hi > lo ? (state.q[i] - lo) / (hi - lo) : 0.5);
}

function renderAll() {
  const b = state.board;
  if (state.q && els.heat.checked) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < state.q.length; i++) {
      const v = state.q[i];
      if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    heatRange = [lo, hi];
  }
  for (let i = 0; i < b.height * b.width; i++) renderCell(i);
  const flags = state.flags.reduce((a, v) => a + v, 0);
  const left = b.won ? 0 : b.nMines - flags;
  els.minesLeft.textContent = `Mines ${left}`;
  els.legend.hidden = !els.heat.checked;
  updateButtons();
}

function updateButtons() {
  const ready = !!state.agent;
  const over = state.board.done;
  els.run.textContent = state.running ? "Pause" : "Play";
  els.run.disabled = !ready;
  els.step.disabled = !ready || state.busy;
  els.hint.disabled = !ready || over || state.busy;
  els.seed.value = state.seed;
}

function updateTally() {
  const fmt = (t) => t.games === 0 ? "No games yet."
    : `${t.games} ${t.games === 1 ? "game" : "games"}, ${t.wins} won (${((100 * t.wins) / t.games).toFixed(1)}%)`;
  els.tally.textContent = "This session: " + fmt(state.tally.watch);
  els.tallyPlay.textContent = "This session: " + fmt(state.tally.play);
}

// ---------- game flow ----------
function newGame(seed) {
  state.gid++;
  state.seed = seed === undefined ? Math.floor(Math.random() * 2 ** 32) : seed >>> 0;
  const p = PRESETS[state.preset];
  const sameShape = state.board && state.board.height === p.height && state.board.width === p.width;
  state.board = new Board(p.height, p.width, p.mines, mulberry32(state.seed));
  state.flags = new Uint8Array(p.height * p.width);
  state.q = null; state.qKey = ""; state.mark = -1; state.moves = 0; state.busy = false;
  if (!sameShape || state.cells.length !== p.height * p.width) buildGrid();
  else state.cache.fill("");
  const loading = useModel();
  if (!loading) setStatus(state.mode === "watch" ? (state.running ? "New game." : "Press Play to watch the agent.") : "Your move. Your first click is always safe.");
  renderAll();
  refreshHeat();
}

async function getQ() {
  const b = state.board, key = `${state.gid}:${b.nRevealed}`;
  if (state.qKey === key && state.q) return { q: state.q, action: state.action };
  const gid = state.gid;
  const r = await state.agent.evaluate(b);
  if (gid !== state.gid) return null;
  state.q = r.q; state.action = r.action; state.qKey = key;
  return r;
}

async function refreshHeat() {
  if (!els.heat.checked || !state.agent || state.board.done) {
    if (state.q && (!els.heat.checked || state.board.done)) { state.q = null; state.qKey = ""; renderAll(); }
    return;
  }
  const r = await getQ();
  if (r) renderAll();
}

function finishIfDone() {
  const b = state.board;
  if (!b.done) return false;
  const t = state.tally[state.mode];
  t.games++;
  if (b.won) t.wins++;
  state.q = null; state.qKey = ""; state.mark = -1;
  if (b.won) {
    for (let i = 0; i < b.mines.length; i++) if (b.mines[i]) state.flags[i] = 1;
  }
  const who = state.mode === "watch" ? "The agent" : "You";
  setStatus(b.won ? `${who} cleared the board.` : `${who} hit a mine.`, b.won ? "win" : "lose");
  updateTally();
  return true;
}

function reveal(i) {
  const b = state.board;
  if (b.done || b.revealed[i]) return;
  state.mark = -1;
  state.flags[i] = 0;
  b.reveal((i / b.width) | 0, i % b.width);
  state.moves++;
  state.q = null; state.qKey = "";
  finishIfDone();
  renderAll();
  refreshHeat();
}

// ---------- watch mode ----------
async function agentMove() {
  if (state.busy || !state.agent) return;
  const b = state.board;
  if (b.done) return;
  const gid = state.gid;
  state.busy = true; updateButtons();
  try {
    const r = await getQ();
    if (!r || gid !== state.gid) return;
    state.mark = r.action; state.markKind = "pick";
    renderAll();
    const d = delay();
    if (d > 0) await sleep(d * 0.5);
    if (gid !== state.gid) return;
    const row = (r.action / b.width) | 0, col = r.action % b.width;
    state.busy = false;
    reveal(r.action);
    if (!state.board.done) setStatus(`Move ${state.moves}: opened row ${row + 1}, column ${col + 1}.`);
  } finally {
    state.busy = false; updateButtons();
  }
}

async function runLoop() {
  if (state.looping) return;
  state.looping = true;
  try {
    while (state.running && state.mode === "watch" && state.agent) {
      if (state.board.done) {
        if (!els.autonext.checked) { state.running = false; break; }
        await sleep(Math.max(700, delay() * 3));
        if (state.running && state.board.done) newGame();
        continue;
      }
      await agentMove();
      await sleep(delay() * 0.5);
    }
  } finally {
    state.looping = false; updateButtons();
  }
}

function toggleRun() {
  state.running = !state.running;
  updateButtons();
  if (state.running) runLoop();
}

async function stepOnce() {
  state.running = false;
  if (state.board.done) { newGame(); return; }
  await agentMove();
}

// ---------- play mode ----------
function toggleFlag(i) {
  const b = state.board;
  if (state.mode !== "play" || b.done || b.revealed[i]) return;
  state.flags[i] ^= 1;
  renderAll();
}

async function hint() {
  if (state.busy || !state.agent || state.board.done) return;
  state.busy = true; updateButtons();
  const gid = state.gid;
  try {
    const r = await getQ();
    if (!r || gid !== state.gid) return;
    state.mark = r.action; state.markKind = "hint";
    const w = state.board.width;
    setStatus(`Hint: the agent would open row ${((r.action / w) | 0) + 1}, column ${(r.action % w) + 1}. It is a best guess, not a guarantee.`);
    renderAll();
  } finally {
    state.busy = false; updateButtons();
  }
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  state.running = false;
  els.tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.mode === mode)));
  els.ctlWatch.hidden = mode !== "watch";
  els.ctlPlay.hidden = mode !== "play";
  newGame();
}

// ---------- input ----------
let longPressTimer = 0, longPressed = false, lastPointer = "mouse", downXY = null;
const cellIndex = (e) => { const t = e.target.closest(".cell"); return t ? +t.dataset.i : -1; };

els.board.addEventListener("pointerdown", (e) => {
  lastPointer = e.pointerType;
  const i = cellIndex(e);
  if (i < 0 || e.pointerType === "mouse") return;
  longPressed = false;
  downXY = [e.clientX, e.clientY];
  clearTimeout(longPressTimer);
  longPressTimer = setTimeout(() => {
    longPressed = true;
    toggleFlag(i);
    if (navigator.vibrate) { try { navigator.vibrate(15); } catch { /* ignore */ } }
  }, 420);
});
const cancelLongPress = () => clearTimeout(longPressTimer);
els.board.addEventListener("pointerup", cancelLongPress);
els.board.addEventListener("pointercancel", cancelLongPress);
els.board.addEventListener("pointermove", (e) => {
  if (downXY && Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]) > 10) cancelLongPress();
});
els.board.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const i = cellIndex(e);
  if (i >= 0 && lastPointer === "mouse") toggleFlag(i);
});
els.board.addEventListener("click", (e) => {
  const i = cellIndex(e);
  if (i < 0) return;
  if (longPressed) { longPressed = false; return; }
  if (state.mode !== "play" || state.board.done || state.flags[i]) return;
  reveal(i);
});
els.board.addEventListener("keydown", (e) => {
  const i = cellIndex(e);
  if (i < 0) return;
  const { width: w, height: h } = state.board;
  let r = (i / w) | 0, c = i % w;
  if (e.key === "ArrowUp") r--; else if (e.key === "ArrowDown") r++;
  else if (e.key === "ArrowLeft") c--; else if (e.key === "ArrowRight") c++;
  else if (e.key === "f" || e.key === "F") { toggleFlag(i); return; }
  else return;
  e.preventDefault();
  r = Math.max(0, Math.min(h - 1, r)); c = Math.max(0, Math.min(w - 1, c));
  focusCell(r * w + c);
});
els.board.addEventListener("focusin", (e) => { const i = cellIndex(e); if (i >= 0) setTabStop(i); });
function setTabStop(i) {
  if (state.cells[state.focus]) state.cells[state.focus].tabIndex = -1;
  state.focus = i; state.cells[i].tabIndex = 0;
}
function focusCell(i) { setTabStop(i); state.cells[i].focus(); }

els.tabs.forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
els.run.addEventListener("click", toggleRun);
els.step.addEventListener("click", stepOnce);
els.hint.addEventListener("click", hint);
els.newGame.addEventListener("click", () => newGame());
els.preset.addEventListener("change", () => { state.preset = els.preset.value; newGame(); });
els.seed.addEventListener("change", () => {
  const v = Number(els.seed.value);
  if (Number.isFinite(v) && v >= 0) newGame(Math.floor(v)); else els.seed.value = state.seed;
});
els.heat.addEventListener("change", () => {
  if (els.heat.checked) { renderAll(); refreshHeat(); } else { state.q = null; state.qKey = ""; renderAll(); }
});
new ResizeObserver(layoutBoard).observe(els.wrap);

// ---------- models: loaded lazily, one session per model, cached ----------
const loadedAgents = {};   // key -> Agent
const loadPromises = {};   // key -> Promise<Agent> (dropped on failure so a later switch retries)

function loadAgent(key) {
  if (loadPromises[key]) return loadPromises[key];
  const ort = window.ort;
  if (!ort) return Promise.reject(new Error("onnxruntime-web could not be downloaded from the CDN (check your connection or ad blocker)."));
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on static hosting, so no worker threads
  const p = Agent.load(ort, MODELS[key].url).then((a) => { loadedAgents[key] = a; return a; });
  loadPromises[key] = p;
  p.catch(() => { delete loadPromises[key]; });
  return p;
}

/** Point state.agent at the model for the current board, loading it if needed. Returns true while loading. */
function useModel() {
  const key = modelKeyFor(state.board.height, state.board.width);
  state.modelKey = key;
  if (loadedAgents[key]) {
    state.agent = loadedAgents[key];
    els.modelInfo.textContent = MODELS[key].label;
    return false;
  }
  state.agent = null;
  els.modelInfo.textContent = "";
  setStatus(`Loading the ${key === "large" ? "16x16 " : ""}model…`);
  loadAgent(key).then((agent) => {
    if (state.modelKey !== key) return; // the user switched board size while this was loading
    state.agent = agent;
    els.modelInfo.textContent = MODELS[key].label;
    setStatus(state.mode === "watch" ? "Model ready. Press Play to watch the agent." : "Model ready. Your move.");
    updateButtons();
    refreshHeat();
    if (state.mode === "watch" && !state.looping && (state.running || (!autoStarted && !matchMedia("(prefers-reduced-motion: reduce)").matches))) {
      state.running = true;
      runLoop();
    }
    autoStarted = true;
  }).catch((err) => {
    console.error(err);
    if (state.modelKey !== key) return;
    setStatus(`The AI model could not be loaded: ${err.message || err}. You can still play without hints.`, "error");
    updateButtons();
  });
  return true;
}

// ---------- start ----------
let autoStarted = false;
newGame();
updateTally();
