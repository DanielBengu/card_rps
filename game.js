"use strict";

/* ============================================================================
   1. MODEL
   ============================================================================ */

const ROCK = 0, PAPER = 1, SCISSORS = 2;
const TYPE_NAME  = ["Rock", "Paper", "Scissors"];
const TYPE_CLASS = ["rock", "paper", "scissors"];

const GLYPH = [
  // rock — a blob
  '<svg viewBox="0 0 100 100"><path d="M22 60c-5-16 6-33 26-37 20-4 34 7 36 24 2 17-11 30-30 30-16 0-28-5-32-17z"/><path d="M36 44c4-4 10-5 15-2M52 66c6 1 12-1 15-6"/></svg>',
  // paper — sheet with a folded corner
  '<svg viewBox="0 0 100 100"><path d="M28 18h34l14 15v49H28z"/><path d="M62 18v15h14"/></svg>',
  // scissors — crossed blades with finger rings
  '<svg viewBox="0 0 100 100"><path d="M30 18l38 54M70 18L32 72"/><circle class="ring" cx="28" cy="80" r="8"/><circle class="ring" cx="72" cy="80" r="8"/></svg>'
];

/** a beats b?  paper>rock, scissors>paper, rock>scissors */
function typeBeats(a, b) { return (a - b + 3) % 3 === 1; }

/** +1 player wins the round, -1 opponent wins, 0 tie */
function clash(c1, c2) {
  if (c1.t !== c2.t) return typeBeats(c1.t, c2.t) ? 1 : -1;
  return Math.sign(c1.p - c2.p);
}

function clashText(c1, c2) {
  if (c1.t !== c2.t) {
    const [w, l] = clash(c1, c2) > 0 ? [c1, c2] : [c2, c1];
    const verb = w.t === PAPER ? "covers" : w.t === ROCK ? "crushes" : "cut";
    return `${TYPE_NAME[w.t]} ${verb} ${TYPE_NAME[l.t]}`;
  }
  if (c1.p === c2.p) return `Two ${TYPE_NAME[c1.t]} ${c1.p}s — dead tie`;
  return `${TYPE_NAME[c1.t]} ${Math.max(c1.p, c2.p)} beats ${TYPE_NAME[c1.t]} ${Math.min(c1.p, c2.p)}`;
}

/** Deal a 6-card pool from a 18-card deck (3 types x powers 1-3 x 2 copies). */
function dealPool() {
  for (let attempt = 0; attempt < 50; attempt++) {
    const deck = [];
    for (let t = 0; t < 3; t++)
      for (let p = 1; p <= 3; p++)
        for (let c = 0; c < 2; c++) deck.push({ t, p });
    for (let i = deck.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    const pool = deck.slice(0, 6).map((c, i) => ({ id: i, t: c.t, p: c.p }));
    // reject dull pools (fewer than 2 distinct types)
    if (new Set(pool.map(c => c.t)).size >= 2) return pool;
  }
}

/* ============================================================================
   2. SOLVER — zero-sum matrix games
   Row player maximises. Returns { value, row, col } as mixed strategies.
   ============================================================================ */

const EPS = 1e-9;

/**
 * Exact solve by support enumeration: try every pair of equal-sized supports,
 * solve the indifference equations, keep the first candidate that survives
 * a best-response check. Matrices here are at most 3x3, so this is instant.
 */
function solveMatrix(A) {
  const m = A.length, n = A[0].length;
  if (m === 1 && n === 1) return { value: A[0][0], row: [1], col: [1] };

  const supports = k => {
    const out = [];
    for (let mask = 1; mask < (1 << k); mask++) {
      const s = [];
      for (let i = 0; i < k; i++) if (mask & (1 << i)) s.push(i);
      out.push(s);
    }
    return out.sort((a, b) => a.length - b.length);
  };

  for (const S of supports(m)) {
    for (const T of supports(n)) {
      if (S.length !== T.length) continue;
      const k = S.length;

      // row mix x over S making every column in T yield the same value v
      const x = indifference(T.map(j => S.map(i => A[i][j])), k);
      if (!x) continue;
      // column mix y over T making every row in S yield the same value v
      const y = indifference(S.map(i => T.map(j => A[i][j])), k);
      if (!y) continue;
      if (Math.abs(x.v - y.v) > 1e-7) continue;

      const v = x.v;
      const row = new Array(m).fill(0); S.forEach((i, q) => row[i] = x.w[q]);
      const col = new Array(n).fill(0); T.forEach((j, q) => col[j] = y.w[q]);

      // verify: row guarantees at least v, col concedes at most v
      let ok = true;
      for (let j = 0; j < n && ok; j++) {
        let s = 0; for (let i = 0; i < m; i++) s += row[i] * A[i][j];
        if (s < v - 1e-7) ok = false;
      }
      for (let i = 0; i < m && ok; i++) {
        let s = 0; for (let j = 0; j < n; j++) s += col[j] * A[i][j];
        if (s > v + 1e-7) ok = false;
      }
      if (ok) return { value: v, row, col };
    }
  }
  return fictitiousPlay(A, 8000);   // unreachable in practice; safety net
}

/**
 * Solve: sum_q w[q] * C[e][q] = v for every equation e, with sum(w) = 1.
 * Returns null if singular or if any weight is negative.
 */
function indifference(C, k) {
  const M = C.map(rowC => rowC.concat([-1]));
  const b = C.map(() => 0);
  M.push(new Array(k).fill(1).concat([0]));
  b.push(1);
  const sol = gauss(M, b);
  if (!sol) return null;
  const w = sol.slice(0, k);
  if (w.some(t => t < -EPS)) return null;
  return { w: w.map(t => Math.max(0, t)), v: sol[k] };
}

/** Gauss-Jordan with partial pivoting. */
function gauss(M, b) {
  const n = M.length;
  if (M.some(r => r.length !== n)) return null;
  const a = M.map((r, i) => r.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) piv = r;
    if (Math.abs(a[piv][c]) < 1e-12) return null;
    [a[c], a[piv]] = [a[piv], a[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      if (f === 0) continue;
      for (let q = c; q <= n; q++) a[r][q] -= f * a[c][q];
    }
  }
  return a.map((r, i) => r[n] / a[i][i]);
}

function fictitiousPlay(A, iters) {
  const m = A.length, n = A[0].length;
  const rc = new Array(m).fill(0), cc = new Array(n).fill(0);
  cc[0] = 1;
  for (let t = 0; t < iters; t++) {
    let bi = 0, best = -Infinity;
    for (let a = 0; a < m; a++) {
      let s = 0; for (let b = 0; b < n; b++) s += cc[b] * A[a][b];
      if (s > best) { best = s; bi = a; }
    }
    rc[bi]++;
    let bj = 0, worst = Infinity;
    for (let b = 0; b < n; b++) {
      let s = 0; for (let a = 0; a < m; a++) s += rc[a] * A[a][b];
      if (s < worst) { worst = s; bj = b; }
    }
    cc[bj]++;
  }
  const rt = rc.reduce((x, y) => x + y, 0), ct = cc.reduce((x, y) => x + y, 0);
  const row = rc.map(x => x / rt), col = cc.map(x => x / ct);
  // squeeze the value between the two best-response bounds
  let lo = Infinity, hi = -Infinity;
  for (let b = 0; b < n; b++) { let s = 0; for (let a = 0; a < m; a++) s += row[a] * A[a][b]; lo = Math.min(lo, s); }
  for (let a = 0; a < m; a++) { let s = 0; for (let b = 0; b < n; b++) s += col[b] * A[a][b]; hi = Math.max(hi, s); }
  return { value: (lo + hi) / 2, row, col };
}

/* ============================================================================
   3. AI — exact play, exact draft
   ============================================================================ */

/** Draw one item from `items` with the (unnormalised) weights `w`. */
function sample(items, w) {
  const total = w.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return items[(Math.random() * items.length) | 0];
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) { r -= w[i]; if (r <= 0) return items[i]; }
  return items[items.length - 1];
}

const AI = (() => {
  let cards = [];                 // the 6 cards of the current game
  let playMemo = new Map();
  let draftMemo = new Map();

  function bits(mask) {
    const out = [];
    for (let i = 0; i < 6; i++) if (mask & (1 << i)) out.push(i);
    return out;
  }

  /** Probability the PLAYER wins the match from this position. Draw = 0.5. */
  function value(pMask, aMask, margin) {
    if (pMask === 0) return margin > 0 ? 1 : margin < 0 ? 0 : 0.5;
    const key = (pMask << 9) | (aMask << 3) | (margin + 3);
    const hit = playMemo.get(key);
    if (hit !== undefined) return hit;
    const s = solveState(pMask, aMask, margin);
    playMemo.set(key, s.value);
    return s.value;
  }

  /** Full solve of one simultaneous-move node. */
  function solveState(pMask, aMask, margin) {
    const pi = bits(pMask), ai = bits(aMask);
    const A = pi.map(i => ai.map(j =>
      value(pMask ^ (1 << i), aMask ^ (1 << j), margin + clash(cards[i], cards[j]))
    ));
    const s = solveMatrix(A);
    return { value: s.value, row: s.row, col: s.col, pi, ai, A };
  }

  /* --------------------------------------------------------------------
     Opponent model. Equilibrium play is unexploitable but it never
     *punishes* a habit, so on its own the hardest setting would not feel
     like the hardest. This tracks one cheap tendency — whether you lead
     with your weakest, middle or strongest card — and lets the top of the
     dial best-respond to it. Counts survive across games in a session.
     -------------------------------------------------------------------- */
  let obs = [1, 1, 1];                       // Laplace prior: low / mid / high

  /** Which power-rank bucket a card falls in, within the hand it was played from. */
  function rankOf(id, handMask) {
    const hand = bits(handMask);
    if (hand.length < 2) return 1;
    const sorted = hand.slice().sort((a, b) => cards[a].p - cards[b].p || a - b);
    const k = sorted.indexOf(id);
    if (hand.length === 3) return k;
    return k === 0 ? 0 : 2;
  }

  /** Belief over which card the human is about to play: equilibrium, tilted by habit. */
  function belief(pMask, nashRow, pi) {
    const total = obs[0] + obs[1] + obs[2];
    const flat = 1 / pi.length;
    const raw = pi.map((id, k) => {
      const prior = 0.7 * nashRow[k] + 0.3 * flat;   // never write a card off entirely
      return Math.max(1e-9, prior) * (obs[rankOf(id, pMask)] / total);
    });
    const s = raw.reduce((a, b) => a + b, 0);
    return raw.map(x => x / s);
  }

  /** Player's win probability at the end of a given draft split. */
  function splitValue(pMask, aMask) { return value(pMask, aMask, 0); }

  /**
   * Draft lookahead. order[k] is 'P' or 'A' for pick k.
   * Player maximises their win probability, AI minimises it.
   */
  function draftValue(poolMask, pMask, aMask, order, idx) {
    if (idx === 6) return splitValue(pMask, aMask);
    const key = (poolMask << 12) | (pMask << 6) | aMask;
    const hit = draftMemo.get(key);
    if (hit !== undefined) return hit;
    const mine = order[idx] === "P";
    let best = mine ? -Infinity : Infinity;
    for (const i of bits(poolMask)) {
      const v = draftValue(
        poolMask ^ (1 << i),
        mine ? pMask | (1 << i) : pMask,
        mine ? aMask : aMask | (1 << i),
        order, idx + 1
      );
      best = mine ? Math.max(best, v) : Math.min(best, v);
    }
    draftMemo.set(key, best);
    return best;
  }

  return {
    reset(gameCards) { cards = gameCards; playMemo = new Map(); draftMemo = new Map(); },
    bits,
    value,
    solveState,

    /**
     * Which pool card the AI takes.
     * temp 0..1. At 1 it takes the minimax-best card; below that it samples from a
     * softmax over the true values, so it makes plausible mistakes rather than
     * random ones. At 0 the softmax is flat and the draft is pure chance.
     */
    draftPick(poolMask, pMask, aMask, order, idx, temp) {
      const options = bits(poolMask);
      const vals = options.map(i =>
        draftValue(poolMask ^ (1 << i), pMask, aMask | (1 << i), order, idx + 1));

      if (temp >= 1) {                       // exact: minimise the player's odds
        let bestI = options[0], bestV = Infinity;
        options.forEach((i, k) => { if (vals[k] < bestV - 1e-9) { bestV = vals[k]; bestI = i; } });
        return bestI;
      }
      const sharp = 60 * temp * temp;        // 0 = uniform, high = near-optimal
      const lo = Math.min(...vals);
      const w = vals.map(v => Math.exp(-(v - lo) * sharp));
      return sample(options, w);
    },

    /** Record a card the human played, for the opponent model. */
    observe(id, handMask) { obs[rankOf(id, handMask)]++; },

    /**
     * Which card the AI plays this round. The dial runs through three regimes:
     *   0 .. 0.66  coin flip  ->  exact equilibrium
     *   0.66       exact equilibrium (unexploitable, but forgiving)
     *   0.66 .. 1  equilibrium -> best response to the opponent model
     */
    playPick(pMask, aMask, margin, temp) {
      const ai = bits(aMask);
      if (temp <= 0) return ai[(Math.random() * ai.length) | 0];

      const s = solveState(pMask, aMask, margin);
      const nashW = Math.min(1, temp / 0.66);
      const flat = 1 / s.ai.length;
      const mix = s.col.map(p => nashW * p + (1 - nashW) * flat);

      const exploit = Math.max(0, (temp - 0.66) / 0.34);
      if (exploit > 0 && s.pi.length > 1 && Math.random() < exploit * 0.85) {
        const b = belief(pMask, s.row, s.pi);
        let bestJ = 0, bestEV = Infinity;
        for (let j = 0; j < s.ai.length; j++) {
          let ev = 0;
          for (let k = 0; k < s.pi.length; k++) ev += b[k] * s.A[k][j];
          if (ev < bestEV - 1e-12) { bestEV = ev; bestJ = j; }
        }
        return s.ai[bestJ];
      }
      return sample(s.ai, mix);
    }
  };
})();

/* ============================================================================
   4. VIEW
   ============================================================================ */

const $ = id => document.getElementById(id);
const el = { pool: $("pool"), pHand: $("pHand"), aiHand: $("aiHand"), arena: $("arena"),
             status: $("status"), odds: $("oddsLine"), poolLabel: $("poolLabel"),
             pPips: $("pPips"), aiPips: $("aiPips") };

function cardNode(card, opts = {}) {
  const d = document.createElement("div");
  d.className = "card " + TYPE_CLASS[card.t] + (opts.small ? " small" : "");
  d.dataset.id = card.id;
  if (opts.spent) d.classList.add("spent");
  if (opts.big) d.classList.add("big");
  d.innerHTML =
    `<span class="power">${"<i></i>".repeat(card.p)}</span>` +
    `<span class="glyph">${GLYPH[card.t]}</span>` +
    `<span class="name">${TYPE_NAME[card.t]}</span>` +
    `<span class="powernum">${card.p}</span>`;
  if (opts.onClick) {
    d.classList.add("pickable");
    d.addEventListener("click", opts.onClick);
  }
  return d;
}

function ghost(small) {
  const d = document.createElement("div");
  d.className = "card ghost" + (small ? " small" : "");
  return d;
}

/**
 * Send a card flying from `from` to `to`. It winds up with a short hop the
 * opposite way, then commits and travels straight there fast.
 */
function flyCard(card, from, to, destNode, done) {
  if (!from || !to || !destNode || typeof destNode.animate !== "function") { done(); return; }

  const node = cardNode(card);
  Object.assign(node.style, {
    position: "fixed",
    left: from.left + "px", top: from.top + "px",
    width: from.width + "px", height: from.height + "px",
    margin: "0", zIndex: "80", pointerEvents: "none"
  });
  document.body.appendChild(node);
  destNode.style.visibility = "hidden";

  const dx = to.left - from.left, dy = to.top - from.top;
  const len = Math.hypot(dx, dy) || 1;
  const bx = -dx / len * 32, by = -dy / len * 32;   // the wind-up, opposite the travel
  const tilt = dx >= 0 ? 8 : -8;

  const anim = node.animate([
    { transform: "translate(0,0) rotate(0deg) scale(1)",
      easing: "cubic-bezier(.3,1.5,.6,1)" },
    { transform: `translate(${bx}px,${by}px) rotate(${-tilt}deg) scale(1.07)`,
      offset: 0.34, easing: "cubic-bezier(.5,0,.2,1)" },
    { transform: `translate(${dx}px,${dy}px) rotate(0deg) scale(1)` }
  ], { duration: 430, fill: "forwards" });

  let fired = false;
  const finish = () => {
    if (fired) return;
    fired = true;
    node.remove();
    destNode.style.visibility = "";
    done();
  };
  anim.onfinish = finish;
  setTimeout(finish, 750);                          // safety net
}

function rectOf(container, id) {
  const n = container.querySelector('[data-id="' + id + '"]');
  return n ? { node: n, rect: n.getBoundingClientRect() } : null;
}

function renderPips(node, results) {
  node.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    const s = document.createElement("span");
    const r = results[i];
    s.className = "pip" + (r === undefined ? "" : r > 0 ? " w" : r < 0 ? " l" : " t");
    node.appendChild(s);
  }
}

/* ============================================================================
   5. CONTROLLER
   ============================================================================ */

const G = {
  cards: [], poolMask: 0, pMask: 0, aMask: 0,
  order: [], idx: 0,
  phase: "idle",          // idle | draft | play | over
  margin: 0, results: [],
  temp: 0.5,              // opponent sharpness, 0..1 (the slider)
  showOdds: true,
  playerStarts: true,
  tally: { w: 0, d: 0, l: 0 },
  busy: false
};

function newGame() {
  G.cards = dealPool();
  AI.reset(G.cards);
  G.poolMask = 0b111111;
  G.pMask = 0; G.aMask = 0;
  G.order = G.playerStarts ? ["P","A","A","P","P","A"] : ["A","P","P","A","A","P"];
  G.playerStarts = !G.playerStarts;
  G.idx = 0;
  G.phase = "draft";
  G.margin = 0;
  G.results = [];
  G.busy = false;
  resetArena();
  render();
  setStatus(G.order[0] === "A" ? "Opponent picks first." : "You pick first.");
  if (G.order[0] === "A") setTimeout(aiDraft, 500);
}

function resetArena() {
  el.arena.innerHTML = "";
  el.arena.appendChild(ghost(true));
  const vs = document.createElement("div");
  vs.className = "vs"; vs.textContent = "VS";
  el.arena.appendChild(vs);
  el.arena.appendChild(ghost(true));
}

function render() {
  // pool
  el.pool.innerHTML = "";
  const myTurn = G.phase === "draft" && G.order[G.idx] === "P";
  if (G.phase === "draft") {
    el.poolLabel.textContent = `Common pool — pick ${G.idx + 1} of 6 · ${myTurn ? "your turn" : "opponent choosing…"}`;
    for (let i = 0; i < 6; i++) {
      const taken = !(G.poolMask & (1 << i));
      const node = cardNode(G.cards[i], {
        spent: taken,
        onClick: (myTurn && !taken && !G.busy) ? () => playerDraft(i) : null
      });
      if (taken) node.classList.add(G.pMask & (1 << i) ? "taken-p" : "taken-a");
      el.pool.appendChild(node);
    }
  } else if (G.phase === "idle") {
    el.poolLabel.textContent = "Common pool";
    for (let i = 0; i < 6; i++) el.pool.appendChild(ghost());
  } else {
    el.poolLabel.textContent = `Round ${G.results.length + 1} of 3`;
    el.pool.innerHTML = '<div class="empty">Draft complete — play your cards.</div>';
  }

  // hands
  renderHand(el.pHand, G.pMask, true);
  renderHand(el.aiHand, G.aMask, false);

  renderPips(el.pPips, G.results);
  renderPips(el.aiPips, G.results.map(r => -r));

  renderOdds();
}

function renderHand(node, mask, isPlayer) {
  node.innerHTML = "";
  const ids = AI.bits(mask);
  if (!ids.length) {
    node.innerHTML = '<div class="empty">— empty —</div>';
    return;
  }
  ids.forEach(i => {
    const clickable = isPlayer && G.phase === "play" && !G.busy;
    node.appendChild(cardNode(G.cards[i], {
      onClick: clickable ? () => playerPlay(i) : null
    }));
  });
}

function renderOdds() {
  if (!G.showOdds || G.phase !== "play") { el.odds.textContent = ""; return; }
  const v = AI.value(G.pMask, G.aMask, G.margin);
  el.odds.innerHTML = `Against perfect play your odds are <b>${Math.round(v * 100)}%</b>`;
}

function setStatus(text, cls = "") {
  el.status.className = "status " + cls;
  el.status.innerHTML = text;
}

/* ---- draft ---- */

function playerDraft(i) {
  if (G.phase !== "draft" || G.busy || G.order[G.idx] !== "P") return;
  takeCard(i, true);
}

function aiDraft() {
  if (G.phase !== "draft") return;
  const i = AI.draftPick(G.poolMask, G.pMask, G.aMask, G.order, G.idx, G.temp);
  takeCard(i, false);
}

function takeCard(i, mine) {
  const src = rectOf(el.pool, i);

  G.poolMask ^= (1 << i);
  if (mine) G.pMask |= (1 << i); else G.aMask |= (1 << i);
  G.idx++;
  G.busy = true;
  render();

  const dst = rectOf(mine ? el.pHand : el.aiHand, i);

  flyCard(G.cards[i], src && src.rect, dst && dst.rect, dst && dst.node, () => {
    G.busy = false;
    if (G.idx === 6) {
      G.phase = "play";
      render();
      setStatus("Draft done. You both know every card — the only secret left is the order. <span class='hl'>Pick one.</span>");
      return;
    }
    render();
    if (G.order[G.idx] === "A") setTimeout(aiDraft, 220);
    else setStatus("Your pick.");
  });
}

/* ---- play ---- */

function playerPlay(i) {
  if (G.phase !== "play" || G.busy) return;
  G.busy = true;
  const j = AI.playPick(G.pMask, G.aMask, G.margin, G.temp);
  AI.observe(i, G.pMask);
  const pc = G.cards[i], ac = G.cards[j];

  G.pMask ^= (1 << i);
  G.aMask ^= (1 << j);
  const r = clash(pc, ac);
  G.margin += r;
  G.results.push(r);

  // reveal
  el.arena.innerHTML = "";
  el.arena.appendChild(cardNode(pc, { small: true, big: true }));
  const vs = document.createElement("div");
  vs.className = "vs"; vs.textContent = "VS";
  el.arena.appendChild(vs);
  el.arena.appendChild(cardNode(ac, { small: true, big: true }));

  render();
  const verdict = r > 0 ? "You take the round" : r < 0 ? "Opponent takes the round" : "Split";
  setStatus(`${clashText(pc, ac)} — <span class="hl">${verdict}</span>`,
            r > 0 ? "win" : r < 0 ? "lose" : "tie");

  setTimeout(() => {
    if (G.results.length === 3) endGame();
    else { G.busy = false; render(); }
  }, 1350);
}

function endGame() {
  G.phase = "over";
  G.busy = false;
  const wins  = G.results.filter(r => r > 0).length;
  const loses = G.results.filter(r => r < 0).length;
  let cls, text;
  if (wins > loses)      { cls = "win";  text = "You win the match"; G.tally.w++; }
  else if (loses > wins) { cls = "lose"; text = "You lose the match"; G.tally.l++; }
  else                   { cls = "tie";  text = "Match drawn";        G.tally.d++; }
  $("tw").textContent = G.tally.w;
  $("td").textContent = G.tally.d;
  $("tl").textContent = G.tally.l;
  setStatus(`<span class="hl">${text}</span> — ${wins}&ndash;${loses}. Press <b>New game</b> or <kbd>N</kbd>.`, cls);
  render();
}

/* ---- controls ---- */

$("newBtn").addEventListener("click", newGame);

/* difficulty is one continuous dial; the label names the zone you're in */
const ZONES = [
  { max: 33,  name: "Easy",    color: "var(--easy)" },
  { max: 66,  name: "Normal",  color: "var(--normal)" },
  { max: 99,  name: "Hard",    color: "var(--hard)" },
  { max: 100, name: "Perfect", color: "var(--perfect)" }
];

function applyTemp() {
  const raw = Number($("tempSlider").value);
  G.temp = raw / 100;
  const zone = ZONES.find(z => raw <= z.max);
  $("tempLabel").textContent = zone.name;
  document.documentElement.style.setProperty("--tempc", zone.color);
}

$("tempSlider").addEventListener("input", applyTemp);

$("oddsBtn").addEventListener("click", e => {
  G.showOdds = !G.showOdds;
  e.currentTarget.textContent = "Odds: " + (G.showOdds ? "on" : "off");
  renderOdds();
});

document.addEventListener("keydown", e => {
  if (e.key.toLowerCase() === "n") { newGame(); return; }
  const n = parseInt(e.key, 10);
  if (!n || n < 1 || n > 6) return;
  if (G.phase === "draft" && G.order[G.idx] === "P") {
    const avail = AI.bits(G.poolMask);
    if (avail[n - 1] !== undefined) playerDraft(avail[n - 1]);
  } else if (G.phase === "play") {
    const hand = AI.bits(G.pMask);
    if (hand[n - 1] !== undefined) playerPlay(hand[n - 1]);
  }
});

applyTemp();
render();
newGame();
