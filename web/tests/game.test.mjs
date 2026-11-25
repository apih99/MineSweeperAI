import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Board, mulberry32, GameOverError, PRESETS } from "../game.js";
import { maskQ, argmax } from "../agent.js";

const fixtures = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url))).boards;

test("fixtures: JS matches Python observations, reveal results and masks", () => {
  assert.ok(fixtures.length >= 5);
  for (const fx of fixtures) {
    const b = new Board(fx.height, fx.width, fx.mines.length, mulberry32(1));
    const mask = new Uint8Array(fx.height * fx.width);
    for (const [r, c] of fx.mines) mask[r * fx.width + c] = 1;
    b.placeMines(mask);
    assert.deepEqual(Array.from(b.counts), fx.counts, fx.name);
    for (const s of fx.steps) {
      const res = b.reveal(...s.click);
      assert.equal(res.newlyRevealed, s.newly_revealed, fx.name);
      assert.equal(res.hitMine, s.hit_mine, fx.name);
      assert.equal(res.won, s.won, fx.name);
      assert.equal(res.valid, s.valid, fx.name);
      assert.equal(b.done, s.done, fx.name);
      assert.deepEqual(Array.from(b.observationCodes()), s.obs, fx.name);
      assert.deepEqual(Array.from(b.actionMask()), s.mask, fx.name);
      // full one-hot tensor agrees with the codes and is a valid one-hot
      const n = fx.height * fx.width, obs = b.observation();
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let ch = 0; ch < 10; ch++) sum += obs[ch * n + i];
        assert.equal(sum, 1);
        assert.equal(obs[s.obs[i] * n + i], 1);
      }
    }
  }
  assert.ok(fixtures.some((f) => f.steps.some((s) => s.hit_mine)));
  assert.ok(fixtures.some((f) => f.steps.some((s) => s.won)));
});

test("first click is always safe, mine count is exact (many seeds, all presets)", () => {
  for (const p of Object.values(PRESETS)) {
    for (let seed = 0; seed < 300; seed++) {
      const b = new Board(p.height, p.width, p.mines, mulberry32(seed));
      const idx = seed % (p.height * p.width);
      const res = b.reveal(Math.floor(idx / p.width), idx % p.width);
      assert.equal(res.firstClick, true);
      assert.equal(res.hitMine, false);
      assert.equal(b.mines[idx], 0);
      assert.equal(b.mines.reduce((a, v) => a + v, 0), p.mines);
    }
  }
});

test("same seed gives the same board; different seeds differ", () => {
  const mk = (s) => { const b = new Board(9, 9, 10, mulberry32(s)); b.reveal(4, 4); return Array.from(b.mines).join(""); };
  assert.equal(mk(7), mk(7));
  assert.notEqual(mk(7), mk(8));
});

test("flood fill: single mine in a corner reveals everything else, iteratively on a big board", () => {
  const b = new Board(60, 60, 1, mulberry32(0));
  const mask = new Uint8Array(3600); mask[0] = 1;
  b.placeMines(mask);
  const res = b.reveal(59, 59);
  assert.equal(res.newlyRevealed, 3599);
  assert.equal(res.won, true);
  assert.equal(b.done, true);
  assert.throws(() => b.reveal(0, 1), GameOverError);
});

test("revealing a revealed cell is invalid and a no-op; out of bounds throws", () => {
  const b = new Board(5, 5, 3, mulberry32(3));
  b.reveal(2, 2);
  const before = b.nRevealed;
  const res = b.reveal(2, 2);
  assert.equal(res.valid, false);
  assert.equal(b.nRevealed, before);
  assert.throws(() => b.reveal(5, 0), RangeError);
});

test("random playouts always terminate with consistent win/loss", () => {
  for (let seed = 0; seed < 200; seed++) {
    const b = new Board(9, 9, 10, mulberry32(seed));
    const r = mulberry32(seed + 1000);
    while (!b.done) {
      const m = b.actionMask();
      const cells = []; m.forEach((v, i) => v && cells.push(i));
      const a = cells[Math.floor(r() * cells.length)];
      b.reveal(Math.floor(a / 9), a % 9);
    }
    assert.notEqual(b.won, b.hitMine);
    if (b.won) assert.equal(b.nSafeRemaining, 0);
  }
});

test("masked argmax never returns a revealed cell", () => {
  const rng = mulberry32(42);
  for (let g = 0; g < 100; g++) {
    const b = new Board(9, 9, 10, mulberry32(g));
    while (!b.done) {
      const q = Float32Array.from({ length: 81 }, () => rng() * 10 - 5);
      // make revealed cells look attractive so that masking must matter
      for (let i = 0; i < 81; i++) if (b.revealed[i]) q[i] = 100;
      const a = argmax(maskQ(q, b.actionMask()));
      assert.ok(a >= 0 && !b.revealed[a]);
      b.reveal(Math.floor(a / 9), a % 9);
    }
  }
  assert.equal(argmax(maskQ(Float32Array.of(1, 2), Uint8Array.of(0, 0))), -1);
});
