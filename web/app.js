/* toklab web visualizer.
 *
 * All tokenization happens in Python: the page loads toklab's source files into
 * Pyodide and calls toklab.viz, which returns JSON traces. This file only
 * renders those traces and animates them.
 */
"use strict";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

let viz = null;
let modelInfo = {};
let gpt2Loading = null;

// --------------------------------------------------------------- utilities
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Escape text and make whitespace visible. */
function visible(s) {
  return esc(s)
    .replace(/ /g, '<span class="ws">·</span>')
    .replace(/\n/g, '<span class="ws">↵</span>')
    .replace(/\t/g, '<span class="ws">⇥</span>')
    .replace(/\r/g, '<span class="ws">␍</span>');
}

/** GPT-2 byte <-> printable unicode table (same as toklab/byte_unicode.py). */
const U2B = (() => {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const m = new Map();
  bs.forEach((b, i) => m.set(String.fromCodePoint(cs[i]), b));
  return m;
})();

const reprToBytes = (r) => Uint8Array.from([...r].map((c) => U2B.get(c) ?? 63));
const hex = (b) => b.toString(16).padStart(2, "0").toUpperCase();
const strictDecoder = new TextDecoder("utf-8", { fatal: true });

/** Render a byte-level token (given as its GPT-2 repr string). */
function bpeLabel(repr) {
  if (state.rawBytes) return esc(repr);
  const bytes = reprToBytes(repr);
  try {
    return visible(strictDecoder.decode(bytes));
  } catch {
    // incomplete UTF-8 sequence: part of a multi-byte character
    return [...bytes].map((b) => `<span class="hexbyte">${hex(b)}</span>`).join(" ");
  }
}

function hue(key) {
  let h = 2166136261;
  for (const c of String(key)) h = Math.imul(h ^ c.codePointAt(0), 16777619);
  return ((h >>> 0) % 360);
}

function chip(label, key, { title = "", cls = "", sub = "" } = {}) {
  return `<span class="chip ${cls}" style="--h:${hue(key)}" title="${esc(title)}"><span class="lbl">${label}</span>${sub ? `<small>${esc(sub)}</small>` : ""}</span>`;
}

const isByteLevel = (kind) => kind === "bpe" || kind === "gpt2";

/** Chip for a final token from an encode trace. */
function tokenChip(kind, t, withId = false) {
  const special = t.special ? " special" : "";
  if (isByteLevel(kind)) {
    const title = `id ${t.id}\n${t.repr}\nbytes: ${(t.bytes || []).map(hex).join(" ")}`;
    return chip(t.special ? esc(t.repr) : bpeLabel(t.repr), t.repr, { title, cls: special, sub: withId ? t.id : "" });
  }
  const title = t.unk ? `unknown: "${t.source ?? ""}" -> id ${t.id}` : `id ${t.id}`;
  return chip(visible(t.text), t.text, { title, cls: t.unk ? "unk" : "", sub: withId ? t.id : "" });
}

function setStatus(text, kind = "busy") {
  const el = $("#status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.text();
}

// ------------------------------------------------------------------ player
/** A step player: play/pause, step, scrub. Calls render(step) on change. */
class Player {
  constructor(root, render) {
    this.root = root;
    this.render = render;
    this.max = 0;
    this.step = 0;
    this.timer = null;
    this.scrub = $(".scrub", root);
    this.speed = $(".speed", root);
    this.playBtn = $(".play", root);
    root.addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (!act) return;
      if (act === "play") return this.toggle();
      this.pause();
      if (act === "first") this.go(0);
      if (act === "prev") this.go(this.step - 1);
      if (act === "next") this.go(this.step + 1);
      if (act === "last") this.go(this.max);
    });
    this.scrub.addEventListener("input", () => { this.pause(); this.go(+this.scrub.value); });
    this.speed.addEventListener("change", () => { if (this.timer) { this.pause(); this.play(); } });
  }
  reset(max, step = 0) {
    this.pause();
    this.max = max;
    this.scrub.max = max;
    this.go(step);
  }
  go(step) {
    this.step = Math.max(0, Math.min(this.max, step));
    this.scrub.value = this.step;
    this.render(this.step);
  }
  play() {
    if (this.step >= this.max) this.go(0);
    this.playBtn.textContent = "❚❚";
    this.playBtn.setAttribute("aria-label", "Pause");
    this.timer = setInterval(() => {
      if (this.step >= this.max) return this.pause();
      this.go(this.step + 1);
    }, +this.speed.value);
  }
  pause() {
    clearInterval(this.timer);
    this.timer = null;
    this.playBtn.textContent = "▶";
    this.playBtn.setAttribute("aria-label", "Play");
  }
  toggle() { this.timer ? this.pause() : this.play(); }
}

// ================================================================== ENCODE
const state = { model: "bpe", rawBytes: false, trace: null, timeline: null };

const SAMPLES = [
  ["Hello world", "Hello world! Hello again."],
  ["Unseen words", "The quokka's antidisestablishmentarianism was unbelievable."],
  ["Emoji & scripts", "Tokenizers ❤️ emoji 🤖, café, 日本語 and తెలుగు."],
  ["Code", "def merge(a, b):\n    return a + b  # 42"],
  ["Numbers", "In 2019 GPT-2 had 50257 tokens; 3.14159 is pi."],
  ["Special token", "First document.<|endoftext|>Second document."],
];

const MODEL_NOTES = {
  word: "Word-level: lowercases, splits on words and punctuation, looks each word up. Unseen words become <unk>.",
  wordpiece: "WordPiece (BERT-style): normalizes, splits into words, then greedily takes the longest known piece. ## marks a piece that continues a word.",
  bpe: "Byte-level BPE trained in your browser on the sample corpus. Starts from 256 bytes, so nothing is ever unknown.",
  gpt2: "GPT-2's real 50,257-token vocabulary and 50,000 merges, applied by toklab's own BPE code. Ids match OpenAI's GPT-2 exactly.",
};

const STAGE_INFO = {
  input: ["Input text", "The raw string, exactly as typed."],
  normalize: ["Normalize", ""],
  pretokenize: ["Pre-tokenize", ""],
  bytes: ["UTF-8 bytes", "Each chunk is converted to bytes. Byte-level BPE starts from these 256 possible values, one token per byte."],
  merge: ["", ""],
  output: ["Token ids", "The final tokens and the integer ids the language model receives."],
};

function buildTimeline(trace, kind) {
  const keys = ["input", "normalize", "pretokenize"];
  if (isByteLevel(kind)) keys.push("bytes");
  keys.push("merge", "output");
  let mergeLen = 1;
  if (isByteLevel(kind) || kind === "wordpiece") {
    mergeLen = Math.max(1, ...trace.pretokens.map((p) => p.steps.length));
  }
  const stages = [];
  let t = 0;
  for (const key of keys) {
    const len = key === "merge" ? mergeLen : 1;
    stages.push({ key, start: t, len });
    t += len;
  }
  return { stages, total: t - 1, mergeLen };
}

function stageHTML(n, key, title, desc, body, current, entering) {
  return `<section class="stage${current ? " current" : ""}${entering ? " enter" : ""}" data-n="${n}" data-key="${key}">
    <h2>${esc(title)}</h2><p class="desc">${desc}</p>${body}</section>`;
}

function renderNormalize(trace, kind) {
  if (isByteLevel(kind)) {
    return ["Normalize", "Byte-level BPE (like GPT-2) skips normalization so that decoding gives back the exact input.", `<div class="textbox">${visible(trace.normalized)}</div>`];
  }
  const desc = kind === "word"
    ? "Unicode NFC normalization and lowercasing. Highlighted characters changed."
    : "Unicode normalization, accent stripping and lowercasing (like BERT uncased). Highlighted characters changed.";
  const a = [...trace.input];
  const b = [...trace.normalized];
  let html = "";
  if (a.length === b.length) {
    html = b.map((c, i) => (c !== a[i] ? `<span class="changed">${visible(c)}</span>` : visible(c))).join("");
  } else {
    html = visible(trace.normalized);
  }
  return ["Normalize", desc, `<div class="textbox">${html}</div>`];
}

function renderPretokenize(trace, kind) {
  const pieces = isByteLevel(kind) ? trace.pretokens.map((p) => p.text) : trace.pretokens.map((p) => (typeof p === "string" ? p : p.text));
  const desc = isByteLevel(kind)
    ? "GPT-2's regex splits text into words, numbers, punctuation and whitespace. A leading space stays attached to the word, and merges never cross these boundaries."
    : "Split on whitespace and punctuation into words. Whitespace is thrown away, so this step is lossy.";
  const chunks = pieces.map((p) => `<span class="chunk">${visible(p)}</span>`).join("");
  return ["Pre-tokenize", `${desc} <b>${pieces.length}</b> chunks.`, `<div class="chunks">${chunks}</div>`];
}

function renderBytes(trace) {
  const groups = trace.pretokens.map((p) => {
    if (p.special) return `<div class="bgroup"><div class="chars"><span class="ch">${esc(p.text)}</span></div><div class="hex"><span>special token</span></div></div>`;
    const chars = [];
    const hexes = [];
    const enc = new TextEncoder();
    for (const c of p.text) {
      const bs = [...enc.encode(c)];
      chars.push(`<span class="ch" style="flex:${bs.length}">${visible(c)}</span>`);
      hexes.push(...bs.map((b) => `<span>${hex(b)}</span>`));
    }
    return `<div class="bgroup"><div class="chars">${chars.join("")}</div><div class="hex">${hexes.join("")}</div></div>`;
  });
  return `<div class="bytes-row">${groups.join("")}</div>`;
}

function renderMergeBPE(trace, k) {
  const rows = trace.pretokens.map((p) => {
    if (p.special) {
      return `<div class="mrow done"><span class="src">${esc(p.text)}</span><div class="parts">${tokenChip("bpe", p.tokens[0])}<span class="note">special token, never split</span></div></div>`;
    }
    const steps = p.steps;
    const m = Math.min(k, steps.length);
    const parts = m === 0 ? [...p.repr] : steps[m - 1].parts;
    const active = k >= 1 && k <= steps.length ? steps[k - 1] : null;
    const merged = active ? active.pair[0] + active.pair[1] : null;
    const chips = parts.map((r) => chip(bpeLabel(r), r, { title: r, cls: r === merged ? "fresh" : "" })).join("");
    let note = "";
    if (active) note = `<span class="note">merge <b>${bpeLabel(active.pair[0])}</b> + <b>${bpeLabel(active.pair[1])}</b> · rank ${active.rank}</span>`;
    else if (steps.length === 0) note = `<span class="note">no learned merge applies</span>`;
    else note = `<span class="note">done · ${steps.length} merge${steps.length > 1 ? "s" : ""}</span>`;
    return `<div class="mrow${active ? "" : " done"}"><span class="src">${visible(p.text)}</span><div class="parts">${chips}${note}</div></div>`;
  });
  return `<div class="merge-rows">${rows.join("")}</div>`;
}

function renderMergeWordPiece(trace, k) {
  const rows = trace.pretokens.map((p) => {
    const steps = p.steps;
    const m = Math.min(k, steps.length);
    const active = k >= 1 && k <= steps.length ? steps[k - 1] : null;
    const failed = steps.some((s) => s.match === null);
    let body;
    if (m === steps.length && !active) {
      body = p.tokens.map((t) => tokenChip("wordpiece", t)).join("") + `<span class="note">${failed ? "no piece matched: whole word becomes [UNK]" : "done"}</span>`;
    } else {
      const matched = steps.slice(0, m).filter((s) => s.match).map((s) => s.match);
      const chips = matched.map((x, i) => chip(visible(x), x, { cls: active && i === matched.length - 1 && active.match ? "fresh" : "" })).join("");
      const next = steps[m] ? steps[m].start : p.text.length;
      const rest = p.text.slice(next);
      let tried = "";
      if (active) {
        const shown = active.tried.length > 8 ? ["…", ...active.tried.slice(-7)] : active.tried;
        tried = `<span class="tried">tried ${shown.map((c) => (c === active.match ? `<span class="hit">${visible(c)}</span>` : c === "…" ? "…" : `<s>${visible(c)}</s>`)).join(" ")}</span>`;
      }
      body = `${chips}${rest ? `<span class="rest">${visible(rest)}</span>` : ""}${tried}`;
    }
    return `<div class="mrow${active ? "" : " done"}"><span class="src">${visible(p.text)}</span><div class="parts">${body}</div></div>`;
  });
  return `<div class="merge-rows">${rows.join("")}</div>`;
}

function renderLookupWord(trace) {
  const rows = trace.tokens.map((t) => `<div class="mrow"><span class="src">${visible(t.source)}</span><div class="parts">${tokenChip("word", t, true)}${t.unk ? '<span class="note">not in vocabulary</span>' : ""}</div></div>`);
  return `<div class="merge-rows">${rows.join("")}</div>`;
}

function renderOutput(trace, kind) {
  const chips = trace.tokens.map((t) => tokenChip(kind, t, true)).join("");
  const chars = [...trace.input].length;
  const bytes = new TextEncoder().encode(trace.input).length;
  const n = trace.ids.length;
  const ratio = n ? (chars / n).toFixed(2) : "–";
  return `<div class="chips">${chips || '<span class="note">(no tokens)</span>'}</div>
    <div class="ids">[${trace.ids.join(", ")}]</div>
    <div class="stats">
      <div class="stat"><b>${chars}</b><span>characters</span></div>
      <div class="stat"><b>${bytes}</b><span>UTF-8 bytes</span></div>
      <div class="stat"><b>${n}</b><span>tokens</span></div>
      <div class="stat"><b>${ratio}</b><span>chars / token</span></div>
      <div class="stat${trace.unk_count ? " bad" : ""}"><b>${trace.unk_count}</b><span>unknown tokens</span></div>
      <div class="stat"><b>${trace.vocab_size.toLocaleString()}</b><span>vocab size</span></div>
    </div>`;
}

function renderEncode(step) {
  const { trace, timeline } = state;
  if (!trace) return;
  const kind = state.model;
  const html = [];
  let caption = "";
  timeline.stages.forEach((st, i) => {
    if (st.start > step) return;
    const current = step >= st.start && step < st.start + st.len;
    const n = i + 1;
    let [title, desc] = STAGE_INFO[st.key];
    let body = "";
    if (st.key === "input") body = `<div class="textbox">${visible(trace.input)}</div>`;
    if (st.key === "normalize") [title, desc, body] = renderNormalize(trace, kind);
    if (st.key === "pretokenize") [title, desc, body] = renderPretokenize(trace, kind);
    if (st.key === "bytes") body = renderBytes(trace);
    if (st.key === "merge") {
      const k = Math.min(step - st.start + 1, st.len);
      if (isByteLevel(kind)) {
        title = "Apply BPE merges";
        desc = `In every chunk, find the adjacent pair with the lowest merge rank (learned earliest) and merge it. Repeat until no learned merge applies. Step <b>${k}</b> of ${st.len}.`;
        body = renderMergeBPE(trace, k);
      } else if (kind === "wordpiece") {
        title = "Greedy longest-match";
        desc = `For each word, take the longest vocabulary piece that matches the start of what is left, then continue with a ## piece. Step <b>${k}</b> of ${st.len}.`;
        body = renderMergeWordPiece(trace, k);
      } else {
        title = "Vocabulary lookup";
        desc = "Each word is looked up in the vocabulary. Anything not seen during training maps to <unk>.";
        body = renderLookupWord(trace);
      }
      if (current) caption = `<b>Stage ${n}</b> · ${esc(title)}${st.len > 1 ? ` · step ${k} of ${st.len}` : ""}`;
    }
    if (st.key === "output") [desc, body] = [STAGE_INFO.output[1], renderOutput(trace, kind)];
    if (current && !caption) caption = `<b>Stage ${n}</b> · ${esc(title)}`;
    html.push(stageHTML(n, st.key, title, desc, body, current, step === st.start));
  });
  $("#pipeline").innerHTML = html.join("");
  $("#encode-player .caption").innerHTML = caption;
}

const encodePlayer = new Player($("#encode-player"), (step) => {
  renderEncode(step);
  if (encodePlayer.timer) $("#pipeline .stage.current")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
});

async function ensureGpt2() {
  if (modelInfo.gpt2) return;
  if (!gpt2Loading) {
    gpt2Loading = (async () => {
      setStatus("Loading GPT-2 vocabulary (1.5 MB)…");
      const [enc, merges] = await Promise.all([fetchText("data/gpt2/encoder.json"), fetchText("data/gpt2/vocab.bpe")]);
      await nextFrame();
      modelInfo = JSON.parse(viz.load_gpt2(enc, merges));
      setStatus("Ready", "ok");
    })();
  }
  await gpt2Loading;
}

let encodeTimer = null;
async function runEncode({ showAll = false } = {}) {
  if (!viz) return;
  const kind = state.model;
  if (kind === "gpt2") await ensureGpt2();
  const text = $("#encode-text").value;
  state.trace = JSON.parse(viz.encode(kind, text));
  state.timeline = buildTimeline(state.trace, kind);
  $("#model-note").textContent = MODEL_NOTES[kind] + (modelInfo[kind] ? ` Vocab size: ${modelInfo[kind].vocab_size.toLocaleString()}.` : "");
  encodePlayer.reset(state.timeline.total, showAll ? state.timeline.total : 0);
}

function initEncode() {
  $("#samples").innerHTML = SAMPLES.map(([label], i) => `<button data-i="${i}">${esc(label)}</button>`).join("");
  $("#samples").addEventListener("click", (e) => {
    const i = e.target.closest("button")?.dataset.i;
    if (i === undefined) return;
    $("#encode-text").value = SAMPLES[i][1];
    runEncode().then(() => encodePlayer.play());
  });
  $("#encode-model").addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("#encode-model button").forEach((x) => x.classList.toggle("on", x === b));
    state.model = b.dataset.model;
    await runEncode({ showAll: false });
    encodePlayer.play();
  });
  $("#encode-text").addEventListener("input", () => {
    clearTimeout(encodeTimer);
    encodeTimer = setTimeout(() => runEncode({ showAll: true }), 250);
  });
  $("#raw-bytes").addEventListener("change", (e) => {
    state.rawBytes = e.target.checked;
    renderEncode(encodePlayer.step);
    if (train.trace) renderTrain(trainPlayer.step);
  });
}

// =================================================================== TRAIN
const train = { kind: "bpe", trace: null, snaps: [], vocabAt: [], changedAt: [] };

function trainLabel(id) {
  const [text, repr] = train.trace.tokens[id];
  return train.trace.kind === "bpe" ? bpeLabel(repr) : visible(text);
}
const trainChip = (id, cls = "") => chip(trainLabel(id), train.trace.tokens[id][1], { title: `id ${id}`, cls });

function prepareTrain(trace) {
  train.trace = trace;
  const snaps = [trace.words.map((w) => w.parts)];
  const changed = [new Set()];
  const vocabAt = [];
  const base = trace.kind === "bpe" ? 256 : trace.corpus.alphabet + 5;
  const seen = new Set();
  vocabAt.push(base);
  trace.steps.forEach((s, i) => {
    const next = snaps[i].slice();
    const ch = new Set();
    for (const [w, parts] of Object.entries(s.updates)) { next[+w] = parts; ch.add(+w); }
    snaps.push(next);
    changed.push(ch);
    seen.add(s.new_id);
    vocabAt.push(base + seen.size);
  });
  train.snaps = snaps;
  train.changedAt = changed;
  train.vocabAt = vocabAt;
}

function renderTrain(step) {
  const t = train.trace;
  if (!t) return;
  const steps = t.steps;
  const isBpe = t.kind === "bpe";
  const next = steps[step];

  // next merge
  let nextHTML;
  if (next) {
    const metric = isBpe ? `occurs <b>${next.count}</b> times in the corpus` : `score <b>${next.score.toExponential(2)}</b> · pair count ${next.count}`;
    nextHTML = `<h3>Merge #${step + 1}</h3><div class="next-merge">${trainChip(next.pair[0])}<span class="op">+</span>${trainChip(next.pair[1])}<span class="op">→</span>${trainChip(next.new_id, "fresh")}<span class="meta">${metric}. New token id ${next.new_id}.</span></div>`;
  } else {
    nextHTML = `<h3>Training finished</h3><div class="next-merge"><span class="meta">${steps.length} merges learned · stopped because: ${esc(t.stop_reason)}. The Encode and Compare tabs now use this tokenizer.</span></div>`;
  }
  $("#train-next").innerHTML = nextHTML;

  // stats
  const start = t.corpus.tokens;
  const now = step === 0 ? start : steps[step - 1].total_tokens;
  const pct = ((now / start) * 100).toFixed(1);
  $("#train-stats").innerHTML = `<h3>Progress</h3>
    <div class="stats" style="margin-top:0">
      <div class="stat"><b>${train.vocabAt[step].toLocaleString()}</b><span>vocab size (target ${t.vocab_size})</span></div>
      <div class="stat"><b>${now.toLocaleString()}</b><span>corpus tokens (from ${start.toLocaleString()} ${isBpe ? "bytes" : "characters"})</span></div>
    </div>
    <div class="meter" title="corpus length relative to the start"><i style="width:${pct}%"></i></div>
    <span class="tried">corpus is now ${pct}% of its starting length</span>`;

  // candidates
  if (next) {
    const top = next.top;
    const val = (c) => (isBpe ? c[2] : c[3]);
    const max = val(top[0]) || 1;
    $("#train-candidates").innerHTML = `<h3>${isBpe ? "Most frequent pairs" : "Best-scoring pairs"}</h3><div class="cands">${top
      .map((c, i) => `<div class="cand${i === 0 ? " win" : ""}"><span class="pair">${trainChip(c[0])}${trainChip(c[1])}</span><span class="bar"><i style="width:${(val(c) / max) * 100}%"></i></span><span class="val">${isBpe ? c[2] : c[3].toExponential(2)}</span></div>`)
      .join("")}</div>`;
  } else {
    $("#train-candidates").innerHTML = `<h3>Candidate pairs</h3><p class="tried">No more merges.</p>`;
  }

  // tracked words
  const snap = train.snaps[step];
  const changed = train.changedAt[step];
  $("#train-words").innerHTML = `<h3>Most frequent corpus chunks</h3><div class="words">${t.words
    .map((w, i) => `<div class="wrow${changed.has(i) ? " flash" : ""}"><span class="f">${w.freq}×</span><span class="chips">${snap[i].map((id) => trainChip(id, changed.has(i) ? "" : "")).join("")}</span></div>`)
    .join("")}</div>`;

  // merges learned
  const learned = steps.slice(Math.max(0, step - 80), step).reverse();
  $("#train-merges").innerHTML = `<h3>Learned merges (${step})</h3><div class="merges">${learned
    .map((s) => `<div><span class="r">#${s.rank + 1}</span>${trainChip(s.pair[0])}+${trainChip(s.pair[1])}→${trainChip(s.new_id)}</div>`)
    .join("") || '<span class="tried">Nothing yet: press play.</span>'}</div>`;

  $("#train-player .caption").innerHTML = next
    ? `<b>Step ${step} of ${steps.length}</b> · next: merge ${trainLabel(next.pair[0])} + ${trainLabel(next.pair[1])}`
    : `<b>Done</b> · ${steps.length} merges learned`;
}

const trainPlayer = new Player($("#train-player"), renderTrain);

async function runTrain() {
  const btn = $("#train-run");
  btn.disabled = true;
  setStatus(`Training ${train.kind === "bpe" ? "BPE" : "WordPiece"}…`);
  await nextFrame();
  try {
    const trace = JSON.parse(viz.train(train.kind, $("#train-corpus").value, +$("#train-size").value));
    modelInfo = JSON.parse(viz.info());
    prepareTrain(trace);
    $("#train-player").hidden = false;
    $("#train-view").hidden = false;
    trainPlayer.reset(trace.steps.length, 0);
    trainPlayer.play();
    setStatus("Ready", "ok");
    if (state.model === train.kind) runEncode({ showAll: true });
  } catch (err) {
    console.error(err);
    setStatus("Training failed: see console", "error");
  } finally {
    btn.disabled = false;
  }
}

function initTrain(corpus) {
  $("#train-corpus").value = corpus;
  const size = $("#train-size");
  const sync = () => ($("#train-size-val").textContent = size.value);
  size.addEventListener("input", sync);
  $("#train-model").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("#train-model button").forEach((x) => x.classList.toggle("on", x === b));
    train.kind = b.dataset.model;
    if (train.kind === "wordpiece") { size.min = 120; size.value = 300; } else { size.min = 260; size.value = 400; }
    sync();
  });
  $("#train-run").addEventListener("click", runTrain);
}

// ================================================================= COMPARE
const NAMES = { word: "Word-level", wordpiece: "WordPiece", bpe: "Byte-level BPE", gpt2: "GPT-2 (50k BPE)" };

async function runCompare() {
  await ensureGpt2();
  const text = $("#compare-text").value;
  const rows = JSON.parse(viz.compare(text));
  const max = Math.max(1, ...rows.map((r) => r.ids.length));
  const chars = [...text].length;
  $("#compare-view").innerHTML = rows
    .map((r) => {
      const kind = r.model;
      const chips = r.tokens.map((t) => tokenChip(kind, t)).join("");
      const unk = r.unk_count ? ` · <span style="color:var(--bad)">${r.unk_count} unknown</span>` : "";
      return `<div class="card crow"><h3>${NAMES[kind]} <small>${r.ids.length} tokens · ${(chars / Math.max(1, r.ids.length)).toFixed(2)} chars/token · vocab ${r.vocab_size.toLocaleString()}${unk}</small></h3>
        <div class="cbar"><i style="width:${(r.ids.length / max) * 100}%"></i></div><div class="chips">${chips}</div></div>`;
    })
    .join("");
}

// ==================================================================== BOOT
function initTabs() {
  $(".tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".tab");
    if (!b) return;
    $$(".tab").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${b.dataset.tab}`));
    if (b.dataset.tab === "compare" && viz && !$("#compare-view").innerHTML) runCompare();
  });
}

async function boot() {
  initTabs();
  initEncode();
  $("#compare-run").addEventListener("click", runCompare);
  try {
    setStatus("Loading Python (Pyodide)…");
    const py = await loadPyodide({ indexURL: PYODIDE_URL });
    setStatus("Loading toklab…");
    const files = JSON.parse(await fetchText("py/manifest.json"));
    py.FS.mkdirTree("/home/pyodide/toklab");
    await Promise.all(files.map(async (f) => py.FS.writeFile(`/home/pyodide/toklab/${f}`, await fetchText(`py/toklab/${f}`))));
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    viz = py.pyimport("toklab.viz");
    const corpus = await fetchText("data/corpus.txt");
    initTrain(corpus);
    setStatus("Training default tokenizers…");
    await nextFrame();
    modelInfo = JSON.parse(viz.setup(corpus));
    setStatus("Ready", "ok");
    await runEncode();
    encodePlayer.play();
  } catch (err) {
    console.error(err);
    setStatus(`Failed to start: ${err.message}`, "error");
  }
}

boot();
