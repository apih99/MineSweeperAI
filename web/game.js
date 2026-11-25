// Pure Minesweeper logic: a port of src/minesweeper_rl/env/core.py (no DOM, no dependencies).
// Same rules, same observation encoding: 10 channels, channel 0 = hidden, channel 1+n = revealed number n.

/** Seedable PRNG (mulberry32). Returns a function giving floats in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const PRESETS = {
  tiny: { label: "Tiny", height: 5, width: 5, mines: 3 },
  beginner: { label: "Beginner", height: 9, width: 9, mines: 10 },
  intermediate: { label: "Intermediate", height: 16, width: 16, mines: 40 },
};

export class GameOverError extends Error {}

export class Board {
  constructor(height, width, nMines, rng) {
    if (height < 1 || width < 1) throw new RangeError("height and width must be >= 1");
    if (!(nMines >= 1 && nMines <= height * width - 1)) {
      throw new RangeError(`nMines must be in [1, ${height * width - 1}], got ${nMines}`);
    }
    this.height = height;
    this.width = width;
    this.nMines = nMines;
    this.rng = rng;
    const n = height * width;
    this.mines = new Uint8Array(n);
    this.revealed = new Uint8Array(n);
    this.counts = new Int8Array(n);
    this.minesPlaced = false;
    this.hitMine = false;
    this.exploded = -1; // flat index of the mine that was hit
    this.nRevealed = 0;
  }

  get nSafeTotal() { return this.height * this.width - this.nMines; }
  get nSafeRemaining() {
    return this.nSafeTotal - (this.nRevealed - (this.hitMine ? 1 : 0));
  }
  get won() { return !this.hitMine && this.minesPlaced && this.nSafeRemaining === 0; }
  get done() { return this.hitMine || this.won; }

  /** Place mines from a 0/1 array (test hook; also used internally). */
  placeMines(mask) {
    const { height: h, width: w } = this;
    if (mask.length !== h * w) throw new RangeError("mask size mismatch");
    let total = 0;
    for (let i = 0; i < mask.length; i++) total += mask[i] ? 1 : 0;
    if (total !== this.nMines) throw new RangeError("mask must contain exactly nMines mines");
    this.mines = Uint8Array.from(mask, (v) => (v ? 1 : 0));
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        let k = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < h && nc >= 0 && nc < w) k += this.mines[nr * w + nc];
          }
        }
        this.counts[r * w + c] = k;
      }
    }
    this.minesPlaced = true;
  }

  _placeRandom(safeIdx) {
    const n = this.height * this.width;
    const cand = [];
    for (let i = 0; i < n; i++) if (i !== safeIdx) cand.push(i);
    const mask = new Uint8Array(n);
    // partial Fisher-Yates: choose nMines of the candidates without replacement
    for (let i = 0; i < this.nMines; i++) {
      const j = i + Math.floor(this.rng() * (cand.length - i));
      [cand[i], cand[j]] = [cand[j], cand[i]];
      mask[cand[i]] = 1;
    }
    this.placeMines(mask);
  }

  reveal(row, col) {
    if (this.done) throw new GameOverError("game is over");
    const { height: h, width: w } = this;
    if (!(row >= 0 && row < h && col >= 0 && col < w)) throw new RangeError(`cell (${row}, ${col}) out of bounds`);
    const idx = row * w + col;
    if (this.revealed[idx]) return { newlyRevealed: 0, hitMine: false, won: false, firstClick: false, valid: false };

    const first = !this.minesPlaced;
    if (first) this._placeRandom(idx);

    if (this.mines[idx]) {
      this.revealed[idx] = 1;
      this.nRevealed += 1;
      this.hitMine = true;
      this.exploded = idx;
      return { newlyRevealed: 1, hitMine: true, won: false, firstClick: first, valid: true };
    }

    let newly = 1;
    this.revealed[idx] = 1;
    const stack = [idx];
    while (stack.length) {
      const cur = stack.pop();
      if (this.counts[cur] !== 0) continue;
      const r = (cur / w) | 0, c = cur % w;
      for (let nr = Math.max(0, r - 1); nr < Math.min(h, r + 2); nr++) {
        for (let nc = Math.max(0, c - 1); nc < Math.min(w, c + 2); nc++) {
          const ni = nr * w + nc;
          if (!this.revealed[ni]) {
            this.revealed[ni] = 1; // neighbours of a zero cell are never mines
            newly += 1;
            stack.push(ni);
          }
        }
      }
    }
    this.nRevealed += newly;
    return { newlyRevealed: newly, hitMine: false, won: this.won, firstClick: first, valid: true };
  }

  /** Per-cell channel index: 0 = hidden, 1 + count when revealed (the exploded mine uses its count). */
  observationCodes() {
    const n = this.height * this.width;
    const codes = new Uint8Array(n);
    for (let i = 0; i < n; i++) codes[i] = this.revealed[i] ? 1 + this.counts[i] : 0;
    return codes;
  }

  /** Flat float32 one-hot of shape (10, H, W), identical to Board.observation() in Python. */
  observation() {
    const n = this.height * this.width;
    const obs = new Float32Array(10 * n);
    const codes = this.observationCodes();
    for (let i = 0; i < n; i++) obs[codes[i] * n + i] = 1;
    return obs;
  }

  /** 1 for cells that may still be revealed (hidden), 0 otherwise. */
  actionMask() {
    const n = this.height * this.width;
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) m[i] = this.revealed[i] ? 0 : 1;
    return m;
  }
}
