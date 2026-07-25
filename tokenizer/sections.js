"use strict";
/* ============================================================================
   sections.js — the course. Every demo below runs the real engine in bpe.js.
   ============================================================================ */

let RAW;      /* BPE trained with NO split pattern  (section 03) */
let TOK;      /* BPE trained with the GPT-4 pattern (sections 04+) */
const N_MERGES = 320;

/* the passage shown in the training film (real slice of the training corpus) */
const FILM = CORPUS.slice(0, 470).replace(/\n+/g, " ");

/* ---------- small shared renderers ---------- */

/** render a list of ids as coloured chips */
function renderStream(host, ids, tok, opt) {
  opt = opt || {};
  host.innerHTML = "";
  const hi = opt.hi;          /* {a,b} pair to highlight */
  const fresh = opt.fresh;    /* id to flash as newly created */
  const showIds = opt.showIds;
  let i = 0;
  while (i < ids.length) {
    const id = ids[i];
    const isPairHead = hi && id === hi.a && ids[i + 1] === hi.b;
    const mk = (v, cls) => {
      const c = el("span", "tk " + cls, esc(tokenLabel(v, tok)) + (showIds ? `<span class="sub">${v}</span>` : ""));
      c.title = `id ${v}  ·  ${v < 256 ? "raw byte" : "learned token"}  ·  ${JSON.stringify(strOf(tok.vocab[v] || []))}`;
      return c;
    };
    if (isPairHead) {
      host.appendChild(mk(id, "hit"));
      host.appendChild(mk(ids[i + 1], "hit"));
      i += 2;
    } else {
      host.appendChild(mk(id, fresh !== undefined && id === fresh ? "new" : id < 256 ? "b" : "m"));
      i += 1;
    }
  }
}

function legend(items) {
  return el(
    "div",
    "legend",
    items.map((x) => `<span><i style="background:${x.c}"></i>${x.l}</span>`).join("")
  );
}

function pyHi(src) {
  let s = esc(src);
  s = s.replace(/(#[^\n]*)/g, '<span class="cm">$1</span>');
  s = s.replace(/(&quot;[^&\n]*?&quot;)/g, '<span class="st">$1</span>');
  s = s.replace(
    /\b(def|for|in|while|if|else|elif|return|break|not|and|or|import|from|lambda|float)\b/g,
    '<span class="kw">$1</span>'
  );
  s = s.replace(/\b(range|len|list|max|min|zip|print|sorted|get|append|encode|decode|join)\b/g, '<span class="fn">$1</span>');
  return s;
}
const codeBlock = (src) => el("pre", "code", pyHi(src));

/** partial ranks map built from the first k merges of a tokenizer */
function ranksUpTo(tok, k) {
  const m = new Map();
  for (let i = 0; i < k && i < tok.merges.length; i++) m.set(KEY(tok.merges[i].a, tok.merges[i].b), tok.merges[i].idx);
  return m;
}
function countTokens(text, tok, k) {
  const r = k === undefined ? tok.ranks : ranksUpTo(tok, k);
  let n = 0;
  for (const ch of splitChunks(text, PATTERNS[tok.patName])) n += encodeChunk(bytesOf(ch), r).length;
  return n;
}

/* ============================================================================
   00 · THE FOOTGUN LIST
   ============================================================================ */
function sec00(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§0</div><div class="eqname">why this page exists</div>
    <div class="does"><b>A list of questions with the same answer.</b> Each of these is a real, reproducible
    weakness of production language models, and none of them is caused by the neural network.</div>`;
  const rows = [
    ["Why can't an LLM spell a word backwards?", "the letters were never separate things"],
    ["Why does counting letters in a word fail?", "the word arrived as one integer"],
    ["Why is it worse at Japanese, Hindi, Korean?", "3–4× more tokens for the same sentence"],
    ["Why is arithmetic unreliable?", "digits get chunked inconsistently"],
    ["Why did GPT-2 struggle with Python?", "every indent space was its own token"],
    ["Why does a trailing space break a prompt?", "it splits the next word off its leading space"],
    ["Why does <code>&lt;|endoftext|&gt;</code> in user text act strange?", "special tokens bypass the algorithm"],
    ["Why does <code>SolidGoldMagikarp</code> make models babble?", "a token the model was never trained on"],
  ];
  const g = el("div", "findings");
  rows.forEach((r) =>
    g.appendChild(el("div", "finding", `<div class="h">${r[0]}</div><div class="b">→ <b>tokenization</b> · ${r[1]}</div>`))
  );
  c.appendChild(g);
  c.appendChild(
    el(
      "div",
      "note warn",
      `<b>The one sentence to remember.</b> The tokenizer is a <i>separate program</i>, with its own training data
       and its own algorithm, trained <i>before</i> the model and then frozen. The model only ever sees its output.`
    )
  );
  sec.appendChild(c);

  /* live teaser */
  const t = el("div", "card");
  t.innerHTML = `<div class="eqno">live</div><div class="eqname">a first look</div>
    <div class="does">Here is a sentence as the model actually receives it — after being processed by a real BPE
    tokenizer trained further down this page. <span class="hl">Amber chips are learned multi-byte tokens</span>;
    grey chips are single raw bytes that never earned a merge. Notice that whole words are single units, and that
    the <b>leading space belongs to the word</b>.</div>`;
  const box = el("div", "stream");
  t.appendChild(box);
  t.appendChild(
    legend([
      { c: "#2a2110", l: "learned token" },
      { c: "#1b2437", l: "raw byte" },
      { c: "transparent", l: "· = space, ⏎ = newline" },
    ])
  );
  const inp = el("input");
  inp.type = "text";
  inp.value = "The model reads integers, not words. Tokenization is the whole story.";
  inp.style.marginTop = "12px";
  t.appendChild(inp);
  const st = statStrip([
    { k: "characters", v: "—", id: "t0chars" },
    { k: "utf-8 bytes", v: "—", id: "t0bytes", c: "c" },
    { k: "tokens", v: "—", id: "t0toks", c: "g" },
    { k: "bytes / token", v: "—", id: "t0cr", c: "v" },
  ]);
  t.appendChild(st);
  sec.appendChild(t);

  const upd = () => {
    const ids = encode(inp.value, TOK);
    renderStream(box, ids, TOK, {});
    const b = bytesOf(inp.value).length;
    $("t0chars").textContent = fmt([...inp.value].length);
    $("t0bytes").textContent = fmt(b);
    $("t0toks").textContent = fmt(ids.length);
    $("t0cr").textContent = ids.length ? (b / ids.length).toFixed(2) + "×" : "—";
  };
  inp.addEventListener("input", upd);
  setTimeout(upd, 0);
}

/* ============================================================================
   01 · TEXT IS NOT CHARACTERS
   ============================================================================ */
function sec01(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§1</div><div class="eqname">strings, code points, bytes</div>
    <div class="does"><b>Three different things, routinely confused.</b> A Python <code>str</code> is a sequence of
    <span class="hl">Unicode code points</span> — abstract numbers from 0 to 1,114,111 assigned to characters by the
    Unicode Consortium. A code point is not a byte. To store or transmit text you must <span class="hl">encode</span>
    it, and in practice that means <b>UTF-8</b>: a variable-length encoding that uses 1 byte for ASCII and up to
    4 bytes for everything else.</div>
    <div class="does">Why UTF-8 and not UTF-32 (fixed 4 bytes, no arithmetic needed)? Because UTF-32 would make every
    English document <span class="hl">4× longer</span> for nothing. UTF-8 is backwards compatible with ASCII, which
    is why it won the web. Its cost is the one that matters here: <b>different languages cost different numbers of
    bytes</b>, and tokenizers are built on bytes.</div>`;
  const tbl = el("div", "tblwrap");
  tbl.innerHTML = `<table class="sum">
    <tr><th>code point range</th><th>bytes</th><th>layout</th><th>covers</th></tr>
    <tr><td class="m2">U+0000 – U+007F</td><td class="mono">1</td><td class="mono">0xxxxxxx</td><td>ASCII — English letters, digits, punctuation</td></tr>
    <tr><td class="m2">U+0080 – U+07FF</td><td class="mono">2</td><td class="mono">110xxxxx 10xxxxxx</td><td>accented Latin, Greek, Cyrillic, Hebrew, Arabic</td></tr>
    <tr><td class="m2">U+0800 – U+FFFF</td><td class="mono">3</td><td class="mono">1110xxxx 10xxxxxx 10xxxxxx</td><td>CJK, Hangul, Devanagari, most living scripts</td></tr>
    <tr><td class="m2">U+10000 – U+10FFFF</td><td class="mono">4</td><td class="mono">11110xxx 10xxxxxx 10xxxxxx 10xxxxxx</td><td>emoji, rare CJK, historic scripts</td></tr>
  </table>`;
  c.appendChild(tbl);
  c.appendChild(
    el(
      "div",
      "note",
      `<b>Read the first byte.</b> A byte starting with <span class="mono-i">0</span> is a whole ASCII character.
       A byte starting with <span class="mono-i">10</span> is a <i>continuation</i> — it is never the start of anything.
       That self-synchronising property is why you can drop into the middle of a UTF-8 stream and recover.`
    )
  );
  sec.appendChild(c);

  /* live inspector */
  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">UTF-8 inspector</div>
    <div class="does">Type anything. Each card is <b>one code point</b>, with the bytes UTF-8 turns it into.
    <span class="hl">Amber = leading byte, blue = continuation byte, green = plain ASCII.</span></div>`;
  const presets = el("div", "btnrow");
  [
    ["English", "Hello world!"],
    ["Accents", "café naïve über"],
    ["Korean", "안녕하세요 👋"],
    ["Japanese", "こんにちは世界"],
    ["Emoji", "🦊🇰🇷👨‍👩‍👧"],
    ["Mixed", "Ｕｎｉｃｏｄｅ‽ 😄"],
  ].forEach(([l, v]) => {
    const b = el("button", "ghost tiny", l);
    b.onclick = () => {
      inp.value = v;
      upd();
    };
    presets.appendChild(b);
  });
  const inp = el("input");
  inp.type = "text";
  inp.value = "안녕하세요 👋 (hello!)";
  d.appendChild(inp);
  d.appendChild(presets);
  const st = statStrip([
    { k: "code points", v: "—", id: "u1cp" },
    { k: "utf-8 bytes", v: "—", id: "u1by", c: "c" },
    { k: "utf-16 units", v: "—", id: "u1u16", c: "v" },
    { k: "js .length", v: "—", id: "u1len", c: "r", s: "counts UTF-16 units, not characters" },
  ]);
  d.appendChild(st);
  const cps = el("div", "cps");
  d.appendChild(cps);
  const bl = el("div", "");
  bl.style.marginTop = "14px";
  bl.innerHTML = `<span class="field-lab">the flat byte sequence a tokenizer actually starts from</span>`;
  const flat = el("div", "stream");
  flat.style.maxHeight = "150px";
  bl.appendChild(flat);
  d.appendChild(bl);
  sec.appendChild(d);

  function upd() {
    const s = inp.value;
    const chars = [...s];
    cps.innerHTML = "";
    chars.slice(0, 40).forEach((ch) => {
      const bs = bytesOf(ch);
      const cp = ch.codePointAt(0);
      const cls = (b, i) => (bs.length === 1 ? "ascii" : i === 0 ? "lead" : "cont");
      cps.appendChild(
        el(
          "div",
          "cp",
          `<div class="ch">${esc(ch === " " ? "␣" : ch)}</div>
           <div class="u">U+${cp.toString(16).toUpperCase().padStart(4, "0")}</div>
           <div class="bs">${bs.map((b, i) => `<span class="by ${cls(b, i)}">${b.toString(2).padStart(8, "0")}</span>`).join("")}</div>
           <div class="n">${bs.map((b) => b).join(" ")} · ${bs.length}B</div>`
        )
      );
    });
    if (chars.length > 40) cps.appendChild(el("div", "cp", `<div class="ch">…</div><div class="u">+${chars.length - 40}</div>`));
    const bytes = bytesOf(s);
    $("u1cp").textContent = fmt(chars.length);
    $("u1by").textContent = fmt(bytes.length);
    $("u1u16").textContent = fmt(s.length);
    $("u1len").textContent = fmt(s.length);
    flat.innerHTML = "";
    bytes.slice(0, 200).forEach((b) => flat.appendChild(el("span", "tk b", String(b))));
  }
  inp.addEventListener("input", upd);
  setTimeout(upd, 0);
}

/* ============================================================================
   02 · WHY NOT JUST FEED IT BYTES?
   ============================================================================ */
function sec02(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§2</div><div class="eqname">the granularity trade</div>
    <div class="does"><b>The naive options both fail.</b> Feed the model raw bytes and the vocabulary is a tidy 256 —
    but an ordinary page of English becomes <span class="hl">thousands of positions</span>, and attention costs
    <span class="mono-i">O(n²)</span>. Your context window, your memory, and your latency all evaporate. Go the other
    way and give every <i>word</i> its own token, and you get short sequences — but a vocabulary in the millions,
    no way to spell an unseen word, and an <span class="mono-i">&lt;UNK&gt;</span> token that silently destroys
    information.</div>
    <div class="does"><b>Subword tokenization is the compromise</b>, and it is not a heuristic — it is
    <span class="hl">learned compression</span>. Frequent strings get short representations (one token), rare strings
    get spelled out of pieces. Exactly the Huffman instinct, applied to text, with the vocabulary size as the dial.</div>`;
  const tw = el("div", "tblwrap");
  tw.innerHTML = `<table class="sum">
    <tr><th>granularity</th><th>vocab</th><th>seq length</th><th>unseen words</th><th>verdict</th></tr>
    <tr><td>characters</td><td class="mono">~100</td><td class="mono">very long</td><td class="mono">fine</td><td class="m2">sequences too long; no semantic units</td></tr>
    <tr><td>bytes</td><td class="mono">256</td><td class="mono">longest</td><td class="mono">perfect</td><td class="m2">universal, but 4× worse for non-English</td></tr>
    <tr><td>words</td><td class="mono">10⁶+</td><td class="mono">shortest</td><td class="mono">&lt;UNK&gt;</td><td class="m2">huge embedding table; brittle</td></tr>
    <tr class="hi"><td><b>subword (BPE)</b></td><td class="mono">32k – 200k</td><td class="mono">good</td><td class="mono">spelled out</td><td class="m2">what essentially everyone ships</td></tr>
  </table>`;
  c.appendChild(tw);
  sec.appendChild(c);

  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">the same sentence, three ways</div>
    <div class="does">One paragraph, encoded at three granularities. Watch the position count collapse as the
    vocabulary grows.</div>`;
  const inp = el("textarea");
  inp.rows = 3;
  inp.value =
    "Attention is quadratic in sequence length, so every token you avoid is compute you keep. A good tokenizer is free speed.";
  d.appendChild(inp);
  const wrap = el("div");
  wrap.style.marginTop = "14px";
  ["bytes", "chars", "bpe"].forEach((k) => {
    const t = { bytes: "byte level · vocab 256", chars: "character level · vocab ≈ 100", bpe: `learned BPE · vocab ${256 + N_MERGES}` }[k];
    const b = el("div");
    b.style.marginBottom = "12px";
    b.innerHTML = `<span class="field-lab">${t} — <span id="g${k}n" style="color:var(--signal)">—</span> positions</span>`;
    const s = el("div", "stream");
    s.style.maxHeight = "120px";
    s.id = "g" + k;
    b.appendChild(s);
    wrap.appendChild(b);
  });
  d.appendChild(wrap);
  d.appendChild(
    el(
      "div",
      "note",
      `<b>The BPE row here uses a tokenizer trained on only 5.8 KB of text with ${N_MERGES} merges.</b> Production
       tokenizers learn 50,000–200,000 merges over hundreds of gigabytes, and compress roughly <b>4 bytes per
       token</b> on English. The shape of the win is the same; the magnitude is much larger.`
    )
  );
  sec.appendChild(d);

  function upd() {
    const s = inp.value;
    const by = bytesOf(s).slice(0, 260);
    const ch = [...s].slice(0, 260);
    const ids = encode(s, TOK);
    const gb = $("gbytes"),
      gc = $("gchars"),
      gp = $("gbpe");
    gb.innerHTML = "";
    by.forEach((b) => gb.appendChild(el("span", "tk b", String(b))));
    gc.innerHTML = "";
    ch.forEach((x) => gc.appendChild(el("span", "tk b", esc(x === " " ? "·" : x))));
    renderStream(gp, ids.slice(0, 260), TOK, {});
    $("gbytesn").textContent = fmt(bytesOf(s).length);
    $("gcharsn").textContent = fmt([...s].length);
    $("gbpen").textContent = fmt(ids.length);
  }
  inp.addEventListener("input", upd);
  setTimeout(upd, 0);
}

/* ============================================================================
   03 · BYTE PAIR ENCODING — THE TRAINER
   ============================================================================ */
function sec03(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§3</div><div class="eqname">the whole algorithm</div>
    <div class="does"><b>Two functions and a loop.</b> Count every adjacent pair. Take the most frequent one. Mint a
    new integer for it. Replace every occurrence. Repeat. That is the entire training procedure — originally a data
    compression algorithm (Gage, 1994), adapted to NLP by Sennrich et al. in 2015 and shipped by GPT-2 in 2019.</div>`;
  c.appendChild(
    codeBlock(`def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):        # every adjacent pair
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)            # collapse the pair
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

ids = list(text.encode("utf-8"))          # start from raw bytes
merges = {}
for i in range(vocab_size - 256):
    stats = get_stats(ids)
    pair  = max(stats, key=stats.get)     # the greedy choice
    idx   = 256 + i                       # the next free integer
    ids   = merge(ids, pair, idx)
    merges[pair] = idx                    # order matters. remember it.`)
  );
  c.appendChild(
    el(
      "div",
      "does",
      `<b>Every merge does two things at once:</b> it shortens the sequence by exactly the number of occurrences it
       replaced, and it grows the vocabulary by exactly one. Compression and vocabulary size trade off along a single
       dial, and that dial is the number of merges.`
    )
  );
  const e = el("div", "eq", eq(String.raw`\text{vocab\_size} = \underbrace{256}_{\text{raw bytes}} + \underbrace{n_{\text{merges}}}_{\text{learned}} + \underbrace{n_{\text{special}}}_{\text{added by hand}}`));
  c.appendChild(e);
  c.appendChild(
    el("div", "eq", eq(String.raw`\text{compression} = \frac{\text{bytes in}}{\text{tokens out}}`))
  );
  sec.appendChild(c);

  /* ---- the training film ---- */
  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">train it · merge by merge</div>
    <div class="does">This is the <b>real trainer</b>, running on a 5,832-byte corpus, no split pattern, ${N_MERGES} merges.
    The passage below is a slice of that corpus. <span class="hl">Amber-highlighted pairs are what the next merge is
    about to collapse</span>; green chips are the token that merge just created. Step through it, or hit play and
    watch the sequence eat itself.</div>`;

  const box = el("div", "stream");
  d.appendChild(box);
  d.appendChild(
    legend([
      { c: "#1b2437", l: "raw byte" },
      { c: "#2a2110", l: "learned token" },
      { c: "#F5A524", l: "about to merge" },
      { c: "#5BE3A6", l: "just created" },
    ])
  );

  const ctrls = el("div", "btnrow");
  const bStep = el("button", "act", "▶ Step");
  const bPlay = el("button", "ghost", "▶▶ Play");
  const bx10 = el("button", "ghost", "+10");
  const bReset = el("button", "ghost", "↺ Reset");
  ctrls.appendChild(bStep);
  ctrls.appendChild(bPlay);
  ctrls.appendChild(bx10);
  ctrls.appendChild(bReset);
  d.appendChild(ctrls);

  const sl = slider("mk", "merges applied", 0, N_MERGES, 0, 1);
  sl.style.marginTop = "14px";
  d.appendChild(sl);

  d.appendChild(
    statStrip([
      { k: "merges", v: "0", id: "m3k" },
      { k: "vocab size", v: "256", id: "m3v", c: "v" },
      { k: "corpus tokens", v: "—", id: "m3t", c: "c" },
      { k: "compression", v: "1.00×", id: "m3c", c: "g" },
      { k: "next pair count", v: "—", id: "m3n" },
    ])
  );

  const twoUp = el("div", "lab two");
  const left = el("div");
  left.innerHTML = `<span class="field-lab">learned vocabulary (newest first)</span>`;
  const led = el("div", "ledger", `<div class="empty">no merges yet — 256 raw bytes only</div>`);
  left.appendChild(led);
  const right = el("div");
  right.innerHTML = `<span class="field-lab">candidate pairs at this step</span>`;
  const cand = el("div", "ledger");
  right.appendChild(cand);
  twoUp.appendChild(left);
  twoUp.appendChild(right);
  d.appendChild(twoUp);

  const pw = el("div", "plotwrap");
  pw.style.marginTop = "16px";
  pw.innerHTML = `<canvas class="plot" id="cvTrain"></canvas>
    <div class="plot-cap">corpus length in tokens vs number of merges — the real measured curve</div>`;
  d.appendChild(pw);
  d.appendChild(
    el(
      "div",
      "note warn",
      `<b>Watch what goes wrong.</b> With no split pattern, the trainer happily learns tokens that straddle
       punctuation and sentence boundaries — <span class="mono-i">"·the"</span> is useful, but
       <span class="mono-i">"·is·a"</span> or <span class="mono-i">".·The"</span> wastes vocabulary on
       accidents of this corpus. Section 05 fixes exactly this.`
    )
  );
  sec.appendChild(d);

  /* --- state --- */
  const filmChunks = [bytesOf(FILM)];
  let k = 0,
    timer = null;

  function drawPlot() {
    const { g, W, H } = fitCanvas($("cvTrain"), 210);
    const pad = { l: 46, r: 14, t: 14, b: 26 };
    g.clearRect(0, 0, W, H);
    grid(g, W, H, pad, 4);
    axes(g, W, H, pad, "merges", "tokens");
    const ys = [RAW.base, ...RAW.merges.map((m) => m.total)];
    const maxY = RAW.base,
      minY = ys[ys.length - 1];
    const X = (i) => pad.l + ((W - pad.l - pad.r) * i) / (ys.length - 1);
    const Y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v - minY * 0.96) / (maxY - minY * 0.96));
    g.beginPath();
    ys.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
    g.strokeStyle = "#F5A524";
    g.lineWidth = 2;
    g.stroke();
    g.lineTo(X(ys.length - 1), H - pad.b);
    g.lineTo(X(0), H - pad.b);
    g.closePath();
    g.fillStyle = "#F5A52418";
    g.fill();
    /* marker */
    g.beginPath();
    g.arc(X(k), Y(ys[k]), 4.5, 0, 7);
    g.fillStyle = "#5BE3A6";
    g.fill();
    g.strokeStyle = "#5BE3A644";
    g.beginPath();
    g.moveTo(X(k), pad.t);
    g.lineTo(X(k), H - pad.b);
    g.stroke();
    g.fillStyle = "#5C6680";
    g.font = '10px "JetBrains Mono",monospace';
    g.textAlign = "right";
    g.fillText(fmt(maxY), pad.l - 6, Y(maxY) + 3);
    g.fillText(fmt(minY), pad.l - 6, Y(minY) + 3);
  }

  function renderLedger() {
    if (k === 0) {
      led.innerHTML = `<div class="empty">no merges yet — 256 raw bytes only</div>`;
      return;
    }
    led.innerHTML = "";
    for (let i = k - 1; i >= Math.max(0, k - 60); i--) {
      const m = RAW.merges[i];
      const A = pieceLabel(RAW.vocab[m.a]),
        B = pieceLabel(RAW.vocab[m.b]),
        lbl = pieceLabel(RAW.vocab[m.idx]);
      led.appendChild(
        el(
          "div",
          "r" + (i === k - 1 ? " fresh" : ""),
          `<span class="i">${m.idx}</span>
           <span class="p"><span class="o">${esc(A)}</span>+<span class="o">${esc(B)}</span> → <span class="n">${esc(lbl.length > 20 ? lbl.slice(0, 20) + "…" : lbl)}</span></span>
           <span class="c">×${fmt(m.count)}</span>`
        )
      );
    }
  }

  function renderCand() {
    const m = RAW.merges[k];
    cand.innerHTML = "";
    if (!m) {
      cand.innerHTML = `<div class="empty">training finished</div>`;
      return;
    }
    const top = m.top || [];
    const mx = top.length ? top[0].count : 1;
    top.forEach((t, i) => {
      const r = el("div", "r" + (i === 0 ? " fresh" : ""));
      r.style.gridTemplateColumns = "1fr 80px 46px";
      r.innerHTML = `<span class="p"><span class="${i === 0 ? "n" : "o"}">${esc(pieceLabel(RAW.vocab[t.a]))}</span>+<span class="${i === 0 ? "n" : "o"}">${esc(pieceLabel(RAW.vocab[t.b]))}</span></span>
        <span class="bar"><i style="width:${(100 * t.count) / mx}%"></i></span>
        <span class="c">${fmt(t.count)}</span>`;
      cand.appendChild(r);
    });
  }

  function render(freshId) {
    const ids = applyFirstK(filmChunks, RAW, k)[0];
    const next = RAW.merges[k];
    renderStream(box, ids, RAW, { hi: freshId === undefined && next ? { a: next.a, b: next.b } : null, fresh: freshId });
    $("m3k").textContent = fmt(k);
    $("m3v").textContent = fmt(256 + k);
    const total = k ? RAW.merges[k - 1].total : RAW.base;
    $("m3t").textContent = fmt(total);
    $("m3c").textContent = (RAW.base / total).toFixed(2) + "×";
    $("m3n").textContent = next ? "×" + fmt(next.count) : "done";
    $("mkv").textContent = k + " / " + N_MERGES;
    $("mk").value = k;
    renderLedger();
    renderCand();
    drawPlot();
  }

  function step() {
    if (k >= N_MERGES) {
      stop();
      return;
    }
    const idx = RAW.merges[k].idx;
    k++;
    render(idx);
    setTimeout(() => {
      if (!timer) render();
    }, 380);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    bPlay.textContent = "▶▶ Play";
    bPlay.classList.remove("on");
    render();
  }
  bStep.onclick = () => {
    stop();
    step();
  };
  bx10.onclick = () => {
    stop();
    k = Math.min(N_MERGES, k + 10);
    render();
  };
  bReset.onclick = () => {
    stop();
    k = 0;
    render();
  };
  bPlay.onclick = () => {
    if (timer) return stop();
    bPlay.textContent = "⏸ Pause";
    bPlay.classList.add("on");
    timer = setInterval(() => {
      if (k >= N_MERGES) return stop();
      const idx = RAW.merges[k].idx;
      k++;
      render(idx);
    }, 420);
  };
  bind("mk", (v) => {
    stop();
    k = v | 0;
    render();
  });
  window.__redraw = ((prev) => () => {
    if (prev) prev();
    drawPlot();
  })(window.__redraw);
  setTimeout(() => render(), 0);
}

/* ============================================================================
   04 · ENCODE & DECODE
   ============================================================================ */
function sec04(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§4</div><div class="eqname">inference is a replay of training</div>
    <div class="does"><b>Decoding is easy.</b> Every token id maps to a byte string; concatenate them and decode the
    result as UTF-8. Because every token bottoms out in raw bytes, <span class="hl">nothing is ever out of
    vocabulary</span> — a byte-level BPE tokenizer can represent any string in existence, including scripts it has
    never seen and byte sequences that aren't valid text at all.</div>
    <div class="does"><b>Encoding is subtle.</b> The naive instinct — "find the longest vocabulary entry that matches"
    — is <span class="hl">wrong</span>. Merges were learned in an order, and a late merge may consume a token that only
    exists because an earlier merge created it. So the encoder must replay that order: repeatedly find the present pair
    with the <b>lowest merge index</b> and apply that one, exactly as training did.</div>`;
  c.appendChild(
    codeBlock(`def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # the earliest-learned pair present wins — not the longest match
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                          # nothing left is mergeable
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids):
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")`)
  );
  c.appendChild(
    el(
      "div",
      "note warn",
      `<b>Why <span class="mono-i">errors="replace"</span> is not optional.</b> A model can emit any id sequence,
       including one whose concatenated bytes are not valid UTF-8 — for example the first half of a 3-byte character.
       Without the flag, your decoder throws mid-stream. With it, you get <span class="mono-i">�</span> until the next
       token completes the character. This is also why streaming chat UIs sometimes flicker a � before showing an emoji.`
    )
  );
  sec.appendChild(c);

  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">encode · step by step · then round-trip</div>
    <div class="does">Your text, byte by byte, with the merge ladder applied one rung at a time in the learned order.
    Each step highlights the pair being collapsed.</div>`;
  const inp = el("input");
  inp.type = "text";
  inp.value = "the tokenizer never sees the language model";
  d.appendChild(inp);
  const box = el("div", "stream");
  box.style.marginTop = "12px";
  d.appendChild(box);
  const row = el("div", "btnrow");
  const bS = el("button", "act", "▶ Step");
  const bP = el("button", "ghost", "▶▶ Play");
  const bF = el("button", "ghost", "⏭ Finish");
  const bR = el("button", "ghost", "↺ Reset");
  const bI = el("button", "ghost", "show ids");
  [bS, bP, bF, bR, bI].forEach((b) => row.appendChild(b));
  d.appendChild(row);
  d.appendChild(
    statStrip([
      { k: "merge step", v: "0", id: "e4s" },
      { k: "positions", v: "—", id: "e4n", c: "c" },
      { k: "applied merge", v: "—", id: "e4m" },
      { k: "round-trip", v: "—", id: "e4rt", c: "g" },
    ])
  );
  const out = el("div");
  out.style.marginTop = "12px";
  out.innerHTML = `<span class="field-lab">token ids the model receives</span>`;
  const ids = el("div", "stream");
  ids.style.maxHeight = "120px";
  out.appendChild(ids);
  d.appendChild(out);
  sec.appendChild(d);

  let steps = [],
    i = 0,
    timer = null,
    withIds = false;
  function build() {
    steps = encodeTrace(inp.value, TOK);
    i = 0;
    render();
  }
  function render() {
    const s = steps[i] || { ids: [] };
    renderStream(box, s.ids, TOK, {
      hi: steps[i + 1] && steps[i + 1].applied ? { a: steps[i + 1].applied.a, b: steps[i + 1].applied.b } : null,
      fresh: s.applied ? s.applied.idx : undefined,
      showIds: withIds,
    });
    $("e4s").textContent = i + " / " + (steps.length - 1);
    $("e4n").textContent = fmt(s.ids.length);
    $("e4m").textContent = s.applied ? esc(tokenLabel(s.applied.idx, TOK)) : "—";
    const final = encode(inp.value, TOK);
    const rt = decode(final, TOK) === inp.value;
    $("e4rt").textContent = rt ? "✓ exact" : "✗ lossy";
    $("e4rt").className = "v " + (rt ? "g" : "r");
    ids.innerHTML = "";
    final.slice(0, 220).forEach((x) => ids.appendChild(el("span", "tk " + (x < 256 ? "b" : "m"), String(x))));
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    bP.textContent = "▶▶ Play";
    bP.classList.remove("on");
  }
  bS.onclick = () => {
    stop();
    if (i < steps.length - 1) i++;
    render();
  };
  bF.onclick = () => {
    stop();
    i = steps.length - 1;
    render();
  };
  bR.onclick = () => {
    stop();
    i = 0;
    render();
  };
  bI.onclick = () => {
    withIds = !withIds;
    bI.classList.toggle("on", withIds);
    render();
  };
  bP.onclick = () => {
    if (timer) return stop();
    bP.textContent = "⏸ Pause";
    bP.classList.add("on");
    timer = setInterval(() => {
      if (i >= steps.length - 1) return stop();
      i++;
      render();
    }, 320);
  };
  inp.addEventListener("input", () => {
    stop();
    build();
  });
  setTimeout(build, 0);

  /* decode-anything demo */
  const dd = el("div", "card");
  dd.innerHTML = `<div class="eqno">live</div><div class="eqname">decode is total — even on nonsense</div>
    <div class="does">Feed the decoder a random id sequence. It always returns <i>something</i>: valid UTF-8 where the
    bytes happen to line up, <span class="mono-i">�</span> where they don't. There is no such thing as an unknown token.</div>`;
  const rnd = el("div", "btnrow");
  const bg = el("button", "act", "⚄ Roll random ids");
  rnd.appendChild(bg);
  dd.appendChild(rnd);
  const rout = el("div", "stream");
  rout.style.marginTop = "12px";
  dd.appendChild(rout);
  const rtxt = el("div");
  rtxt.style.marginTop = "12px";
  rtxt.innerHTML = `<span class="field-lab">decoded string</span>`;
  const rpre = el("pre", "code", "");
  rtxt.appendChild(rpre);
  dd.appendChild(rtxt);
  bg.onclick = () => {
    const n = 14 + ((Math.random() * 8) | 0);
    const arr = [];
    for (let j = 0; j < n; j++) arr.push((Math.random() * (256 + N_MERGES)) | 0);
    renderStream(rout, arr, TOK, { showIds: true });
    rpre.textContent = decode(arr, TOK);
  };
  setTimeout(() => bg.onclick(), 0);
  sec.appendChild(dd);
}

/* ============================================================================
   05 · THE SPLIT REGEX
   ============================================================================ */
function sec05(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§5</div><div class="eqname">forced splits — the part nobody explains</div>
    <div class="does"><b>Raw BPE merges anything adjacent.</b> Left alone it learns <span class="mono-i">dog.</span>,
    <span class="mono-i">dog!</span>, <span class="mono-i">dog?</span> and <span class="mono-i">dog,</span> as four
    unrelated tokens, and burns vocabulary on the difference. Worse, it learns tokens that span a sentence boundary,
    which are pure noise.</div>
    <div class="does"><b>GPT-2's fix:</b> chop the text with a regular expression <i>first</i>, then train and encode
    <span class="hl">inside each chunk independently</span>. Merges can never cross a chunk boundary — so a letter can
    never merge with a digit, and a word can never merge with the punctuation after it. This is a
    <span class="hl">hand-written, hard-coded rule</span> sitting underneath one of the most expensive machine
    learning systems ever built.</div>`;
  const pre = el("pre", "code");
  pre.innerHTML =
    `<span class="cm"># GPT-2 (Radford et al. 2019)</span>\n` +
    `<span class="nm">gpt2</span>  = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"""\n\n` +
    `<span class="cm"># GPT-4 / cl100k_base</span>\n` +
    `<span class="nm">gpt4</span>  = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"""`;
  c.appendChild(pre);
  const tw = el("div", "tblwrap");
  tw.innerHTML = `<table class="sum">
    <tr><th>change in GPT-4</th><th>effect</th></tr>
    <tr><td class="m2">contractions matched case-insensitively</td><td><span class="mono-i">'S</span> and <span class="mono-i">'s</span> now tokenize alike — GPT-2 wasted ids on the shouted variants</td></tr>
    <tr><td class="m2"><span class="mono-i">\\p{N}{1,3}</span></td><td>numbers are chunked into <b>at most 3 digits</b>, so no token spans an arbitrary-length number</td></tr>
    <tr><td class="m2"><span class="mono-i">[^\\r\\n\\p{L}\\p{N}]?\\p{L}+</span></td><td>a letter run may absorb one leading non-letter, but <b>never a newline</b></td></tr>
    <tr><td class="m2"><span class="mono-i">\\s*[\\r\\n]+</span></td><td>runs of newlines get their own chunks — much better for code and markdown</td></tr>
  </table>`;
  c.appendChild(tw);
  c.appendChild(
    el(
      "div",
      "note",
      `<b>A limitation worth knowing.</b> Because the chunking is a fixed regex, it is <i>not</i> learned and
       <i>not</i> language-aware. It assumes whitespace separates words — which is false for Chinese, Japanese and
       Thai, and part of why those languages tokenize expensively.`
    )
  );
  sec.appendChild(c);

  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">run the patterns</div>
    <div class="does">Each chip is one chunk. Merges are forbidden across chip boundaries — so whatever the regex
    separates here can <b>never</b> become a single token, no matter how often it appears.</div>`;
  const inp = el("textarea");
  inp.rows = 3;
  inp.value = "    hello world!!! 1234567890 How'S it going? dog. dog! dog, dog?";
  d.appendChild(inp);
  const ps = el("div", "btnrow");
  [
    ["indentation", "def f():\n    if x:\n        return 42"],
    ["numbers", "1234567890 and 12345678 and 999"],
    ["contractions", "How'S it going? I'VE seen it. don't"],
    ["punctuation", "dog. dog! dog, dog? dog... dog!!!"],
    ["newlines", "line one\n\n\nline two"],
    ["no spaces", "東京都は日本の首都です"],
  ].forEach(([l, v]) => {
    const b = el("button", "ghost tiny", l);
    b.onclick = () => {
      inp.value = v;
      upd();
    };
    ps.appendChild(b);
  });
  d.appendChild(ps);
  const which = el("div", "btnrow");
  let mode = "gpt4";
  which.appendChild(
    seg("segpat", [{ v: "none", l: "no pattern" }, { v: "gpt2", l: "GPT-2" }, { v: "gpt4", l: "GPT-4" }], "gpt4", (v) => {
      mode = v;
      upd();
    })
  );
  d.appendChild(which);
  const box = el("div", "stream");
  box.style.marginTop = "12px";
  d.appendChild(box);
  d.appendChild(
    statStrip([
      { k: "chunks", v: "—", id: "r5c", c: "c" },
      { k: "utf-8 bytes", v: "—", id: "r5b" },
      { k: "longest chunk", v: "—", id: "r5l", c: "v" },
    ])
  );
  sec.appendChild(d);

  function upd() {
    const chunks = splitChunks(inp.value, PATTERNS[mode]);
    box.innerHTML = "";
    chunks.slice(0, 300).forEach((ch) => {
      const s = el("span", "tk sp", esc(pieceLabel(bytesOf(ch))));
      s.title = JSON.stringify(ch);
      box.appendChild(s);
    });
    $("r5c").textContent = fmt(chunks.length);
    $("r5b").textContent = fmt(bytesOf(inp.value).length);
    $("r5l").textContent = chunks.length ? fmt(Math.max(...chunks.map((x) => bytesOf(x).length))) + "B" : "—";
  }
  inp.addEventListener("input", upd);
  setTimeout(upd, 0);
}

/* ============================================================================
   06 · SPECIAL TOKENS
   ============================================================================ */
function sec06(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§6</div><div class="eqname">tokens that were never learned</div>
    <div class="does"><b>Special tokens are appended by hand after training.</b> They are never the output of a merge,
    they have no byte string in the ordinary sense, and the encoder finds them by <span class="hl">searching the raw
    text for their literal spelling before any splitting happens</span>. GPT-2 shipped with exactly one:
    <span class="mono-i">&lt;|endoftext|&gt;</span>, id 50256, marking a document boundary so the model learns that
    what came before is finished.</div>`;
  c.appendChild(el("div", "eq", eq(String.raw`\underbrace{256}_{\text{raw bytes}} \;+\; \underbrace{50{,}000}_{\text{merges}} \;+\; \underbrace{1}_{\text{special}} \;=\; \underbrace{50{,}257}_{\text{GPT-2 vocab}}`)));
  c.appendChild(
    el(
      "div",
      "does",
      `That 50,257 is the width of GPT-2's embedding table and the width of its output softmax. Every special token you
       add costs you two more rows of <span class="mono-i">d_model</span> parameters, and — more importantly — requires
       <b>resizing and re-training</b> those rows if you add them to an existing model.`
    )
  );
  const tw = el("div", "tblwrap");
  tw.innerHTML = `<table class="sum">
    <tr><th>token</th><th>where</th><th>job</th></tr>
    <tr><td class="mono">&lt;|endoftext|&gt;</td><td class="m2">GPT-2 / GPT-3</td><td>document boundary; reset the context</td></tr>
    <tr><td class="mono">&lt;|im_start|&gt; &lt;|im_end|&gt;</td><td class="m2">chat models</td><td>delimit a conversation turn and its speaker</td></tr>
    <tr><td class="mono">&lt;|fim_prefix|&gt; &lt;|fim_middle|&gt; &lt;|fim_suffix|&gt;</td><td class="m2">code models</td><td>fill-in-the-middle: infill between two known halves</td></tr>
    <tr><td class="mono">[CLS] [SEP] [MASK] [PAD]</td><td class="m2">BERT</td><td>classification head, segment split, masked LM target, padding</td></tr>
    <tr><td class="mono">&lt;s&gt; &lt;/s&gt; &lt;unk&gt;</td><td class="m2">Llama / SentencePiece</td><td>sequence bounds and the unknown fallback</td></tr>
  </table>`;
  c.appendChild(tw);
  sec.appendChild(c);

  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">live</div><div class="eqname">a chat template is just string concatenation</div>
    <div class="does">The "conversation" your API exposes is a flat token stream. Roles, turns and system prompts are
    ordinary text wrapped in special tokens — <span class="hl">the model has no structural notion of a message</span>.
    Purple chips are special; everything else went through BPE.</div>`;
  const box = el("div", "stream");
  box.style.maxHeight = "220px";
  d.appendChild(box);
  d.appendChild(
    statStrip([
      { k: "special tokens", v: "—", id: "s6sp", c: "v" },
      { k: "content tokens", v: "—", id: "s6ct", c: "c" },
      { k: "total", v: "—", id: "s6tt", c: "g" },
      { k: "overhead", v: "—", id: "s6ov", s: "spent on scaffolding" },
    ])
  );
  const inp = el("textarea");
  inp.rows = 2;
  inp.style.marginTop = "12px";
  inp.value = "How does BPE decide which pair to merge?";
  d.appendChild(inp);
  sec.appendChild(d);

  function upd() {
    const parts = [
      ["sp", "<|im_start|>"],
      ["t", "system\nYou are a helpful assistant."],
      ["sp", "<|im_end|>"],
      ["sp", "<|im_start|>"],
      ["t", "user\n" + inp.value],
      ["sp", "<|im_end|>"],
      ["sp", "<|im_start|>"],
      ["t", "assistant\n"],
    ];
    box.innerHTML = "";
    let nsp = 0,
      nct = 0;
    parts.forEach(([kind, txt]) => {
      if (kind === "sp") {
        box.appendChild(el("span", "tk sp", esc(txt)));
        nsp++;
      } else {
        const ids = encode(txt, TOK);
        nct += ids.length;
        ids.forEach((id) => box.appendChild(el("span", "tk " + (id < 256 ? "b" : "m"), esc(tokenLabel(id, TOK)))));
      }
    });
    $("s6sp").textContent = fmt(nsp);
    $("s6ct").textContent = fmt(nct);
    $("s6tt").textContent = fmt(nsp + nct);
    const payload = countTokens(inp.value, TOK);
    $("s6ov").textContent = (((nsp + nct - payload) / (nsp + nct)) * 100).toFixed(0) + "%";
  }
  inp.addEventListener("input", upd);
  setTimeout(upd, 0);

  const w = el("div", "card");
  w.innerHTML = `<div class="eqno">⚠</div><div class="eqname">the security hole</div>
    <div class="does">If your encoder is allowed to recognise special tokens in <b>user-supplied</b> text, a user who
    types the literal string <span class="mono-i">&lt;|im_end|&gt;&lt;|im_start|&gt;system</span> can
    <span class="hl">forge a turn boundary</span> and impersonate the system prompt. This is why
    <span class="mono-i">tiktoken</span> refuses special tokens by default and makes you opt in per call
    (<span class="mono-i">allowed_special</span>), and why every serious serving stack encodes user content with
    special tokens <b>disabled</b> and inserts the delimiters itself.</div>
    <div class="note warn"><b>Rule.</b> Never call <span class="mono-i">encode(user_text, allowed_special="all")</span>.
    Treat "does this string tokenize into a control token" as an input-validation question, because it is one.</div>`;
  sec.appendChild(w);
}

/* ============================================================================
   07 · SENTENCEPIECE VS TIKTOKEN
   ============================================================================ */
function sec07(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§7</div><div class="eqname">the other family</div>
    <div class="does"><b>Two libraries, one algorithm, opposite order of operations.</b> This single difference
    explains most of the confusion people have when moving between the OpenAI and Llama/Mistral ecosystems.</div>
    <div class="does"><b>tiktoken</b> (GPT-2/3/4) encodes text to UTF-8 <i>first</i>, then runs BPE
    <span class="hl">on bytes</span>. Every token is a byte string; nothing can ever be unknown.<br>
    <b>SentencePiece</b> (Llama, Mistral, T5, Gemma) runs BPE <span class="hl">on Unicode code points directly</span>,
    then handles rare code points afterwards — either mapping them to <span class="mono-i">&lt;unk&gt;</span>, or, if
    <span class="mono-i">byte_fallback=True</span>, spelling them out as raw byte tokens.</div>`;
  const tw = el("div", "tblwrap");
  tw.innerHTML = `<table class="sum">
    <tr><th></th><th>tiktoken (GPT)</th><th>sentencepiece (Llama)</th></tr>
    <tr><td class="m2">BPE operates on</td><td class="mono">bytes</td><td class="mono">code points</td></tr>
    <tr><td class="m2">rare characters</td><td>always representable</td><td class="m2"><span class="mono-i">character_coverage</span> (0.99995 for Llama-2) decides which get their own symbol; the rest fall back</td></tr>
    <tr><td class="m2">unknown fallback</td><td>impossible by construction</td><td class="m2"><span class="mono-i">&lt;unk&gt;</span>, or byte tokens if <span class="mono-i">byte_fallback</span></td></tr>
    <tr><td class="m2">leading space</td><td>part of the word token (<span class="mono-i">"·the"</span>)</td><td class="m2"><span class="mono-i">add_dummy_prefix</span> prepends a space so "the" and " the" agree; spaces render as <span class="mono-i">▁</span></td></tr>
    <tr><td class="m2">can train?</td><td>no — inference only</td><td class="mono">yes, and fast</td></tr>
    <tr><td class="m2">normalisation</td><td>none</td><td class="m2">on by default (NFKC) — Llama turns it <b>off</b> with <span class="mono-i">identity</span></td></tr>
    <tr><td class="m2">typical vocab</td><td class="mono">50,257 · 100,277 · 200k</td><td class="mono">32,000 · 128,256</td></tr>
  </table>`;
  c.appendChild(tw);
  c.appendChild(
    el(
      "div",
      "note warn",
      `<b>The historical accident.</b> SentencePiece's defaults come from machine translation, not language modelling —
       aggressive normalisation, whitespace stripping, an <span class="mono-i">&lt;unk&gt;</span> token, a
       <span class="mono-i">▁</span> word-start marker. Llama-2 disables most of them one flag at a time. If you train
       a vocabulary with SentencePiece, <b>read every option</b>; the defaults will quietly mangle your text.`
    )
  );
  sec.appendChild(c);

  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">config</div><div class="eqname">Llama-2's actual settings, annotated</div>`;
  d.appendChild(
    codeBlock(`options = dict(
    model_type="bpe",                      # not unigram, the sentencepiece default
    vocab_size=32000,
    normalization_rule_name="identity",    # do NOT rewrite my text
    remove_extra_whitespaces=False,        # keep code indentation intact
    character_coverage=0.99995,            # rarest 0.005% of chars get no symbol
    byte_fallback=True,                    # ...they become raw byte tokens instead
    split_digits=True,                     # 2024 becomes 2 0 2 4, always
    allow_whitespace_only_pieces=True,     # runs of spaces may be one token
    add_dummy_prefix=True,                 # prepend a space to every sequence
    unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
)`)
  );
  d.appendChild(
    el(
      "div",
      "does",
      `<b>Note <span class="hl">split_digits=True</span>.</b> Llama forces every digit to be its own token — a
       deliberate sacrifice of compression to buy consistent arithmetic. GPT-4 takes the middle road with
       <span class="mono-i">\\p{N}{1,3}</span>. GPT-2 did neither, and its arithmetic is correspondingly worse.
       This is a design knob, not a law of nature.`
    )
  );
  sec.appendChild(d);
}

/* ============================================================================
   08 · CHOOSING vocab_size
   ============================================================================ */
function sec08(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§8</div><div class="eqname">the only hyperparameter you must pick by hand</div>
    <div class="does"><b>Bigger vocabulary, shorter sequences — up to a point.</b> Each extra merge buys less
    compression than the last, because you are always merging the <i>next</i> most frequent pair, and frequency has a
    long tail. Meanwhile the cost of the vocabulary grows perfectly linearly: two matrices of shape
    <span class="mono-i">V × d</span> (the embedding table and the output projection, if untied).</div>`;
  c.appendChild(el("div", "eq", eq(String.raw`P_{\text{vocab}} \;=\; 2\,V d \qquad\text{vs}\qquad P_{\text{blocks}} \;\approx\; 12\, L\, d^2`)));
  c.appendChild(
    el(
      "div",
      "does",
      `<b>Three forces pulling in different directions.</b>
       <span class="hl">Compression</span> wants V large — fewer positions, cheaper attention, more text per context.
       <span class="hl">Parameters</span> want V small — the embedding and softmax are pure overhead at small
       <span class="mono-i">d</span>. <span class="hl">Training signal</span> also wants V small — with a fixed corpus,
       doubling V halves the average number of times each token is seen, and the rare rows barely move off their
       initialisation.`
    )
  );
  sec.appendChild(c);

  /* measured compression curve */
  const d = el("div", "card");
  d.innerHTML = `<div class="eqno">measured</div><div class="eqname">diminishing returns, actually measured</div>
    <div class="does">Held-out text (never trained on) encoded with the first <i>k</i> merges of the tokenizer trained
    on this page. Real numbers, computed in your browser — note the knee.</div>`;
  const pw = el("div", "plotwrap");
  pw.innerHTML = `<div class="readout" id="ro8"></div><canvas class="plot" id="cvVocab"></canvas>
    <div class="plot-cap">bytes per token on held-out text vs vocabulary size</div>`;
  d.appendChild(pw);
  d.appendChild(
    el(
      "div",
      "note",
      `<b>Why the knee is so early here:</b> 5.8 KB of training text simply does not contain enough distinct frequent
       pairs to keep paying off. Production tokenizers are trained on <i>hundreds of gigabytes</i>, which pushes the
       knee out to tens of thousands of merges — which is exactly why 32k–100k is the industry range.`
    )
  );
  sec.appendChild(d);

  /* param cost calculator */
  const e = el("div", "card");
  e.innerHTML = `<div class="eqno">calc</div><div class="eqname">what a vocabulary costs you</div>
    <div class="does">Drag the vocabulary size and the model width. The bar shows how much of your parameter budget is
    spent on embedding and unembedding rather than on transformer blocks.</div>`;
  const lab = el("div", "lab");
  const pw2 = el("div", "plotwrap");
  pw2.innerHTML = `<canvas class="plot" id="cvCost"></canvas>
    <div class="plot-cap">share of total parameters spent on the vocabulary, by model width</div>`;
  const ctl = el("div", "controls");
  ctl.appendChild(slider("v8", "vocab size V", 8000, 256000, 100277, 1000));
  ctl.appendChild(slider("d8", "model width d", 256, 8192, 768, 128));
  ctl.appendChild(slider("l8", "layers L", 2, 96, 12, 1));
  lab.appendChild(pw2);
  lab.appendChild(ctl);
  e.appendChild(lab);
  e.appendChild(
    statStrip([
      { k: "vocab params", v: "—", id: "c8v", c: "v" },
      { k: "block params", v: "—", id: "c8b", c: "c" },
      { k: "total", v: "—", id: "c8t" },
      { k: "vocab share", v: "—", id: "c8s", c: "r" },
    ])
  );
  e.appendChild(
    el(
      "div",
      "note warn",
      `<b>Read the extremes.</b> At <span class="mono-i">d=768</span> (GPT-2 small) a 100k vocabulary is over a third
       of the model. At <span class="mono-i">d=8192</span> (65B-class) the same vocabulary is a rounding error — which
       is why large models can afford enormous vocabularies and small models cannot. <b>Weight tying</b> (sharing the
       embedding and the output matrix) halves this cost and is standard on small models.`
    )
  );
  sec.appendChild(e);

  const f = el("div", "card");
  f.innerHTML = `<div class="eqno">how-to</div><div class="eqname">extending a vocabulary you did not train</div>
    <div class="does">You <i>can</i> add tokens to a trained model — a domain vocabulary, new special tokens, or
    "gist" tokens that compress a long prompt into a few learned positions. The mechanics:</div>
    <div class="findings">
      <div class="finding"><div class="h">1 · resize both matrices</div><div class="b">Grow <b>wte</b> and the output head from <b>V</b> to <b>V+n</b> rows. Everything else is untouched.</div></div>
      <div class="finding"><div class="h">2 · initialise sensibly</div><div class="b">Not <b>N(0,1)</b>. Use the <b>mean of the existing embeddings</b>, or the average of the embeddings of the sub-tokens the new token replaces.</div></div>
      <div class="finding"><div class="h">3 · freeze, then thaw</div><div class="b">Train only the new rows first so they don't drag the model, then unfreeze if you have data to spare.</div></div>
      <div class="finding"><div class="h">4 · verify the round trip</div><div class="b">Confirm <b>decode(encode(x)) == x</b> over a real sample of your corpus <b>before</b> you spend a single GPU-hour.</div></div>
    </div>`;
  sec.appendChild(f);

  /* --- computations --- */
  const KS = [];
  for (let k = 0; k <= N_MERGES; k += 10) KS.push(k);
  const bytesHO = bytesOf(HELDOUT).length;
  const CURVE = KS.map((k) => ({ k, v: bytesHO / countTokens(HELDOUT, TOK, k) }));

  function drawVocab() {
    const { g, W, H } = fitCanvas($("cvVocab"), 230);
    const pad = { l: 46, r: 16, t: 16, b: 30 };
    g.clearRect(0, 0, W, H);
    grid(g, W, H, pad, 4);
    axes(g, W, H, pad, "vocabulary size (256 + merges)", "bytes / token");
    const maxV = Math.max(...CURVE.map((p) => p.v)) * 1.05,
      minV = 0.9;
    const X = (k) => pad.l + ((W - pad.l - pad.r) * k) / N_MERGES;
    const Y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (v - minV) / (maxV - minV));
    g.beginPath();
    CURVE.forEach((p, i) => (i ? g.lineTo(X(p.k), Y(p.v)) : g.moveTo(X(p.k), Y(p.v))));
    g.strokeStyle = "#5BE3A6";
    g.lineWidth = 2.2;
    g.stroke();
    CURVE.forEach((p) => {
      g.beginPath();
      g.arc(X(p.k), Y(p.v), 2.6, 0, 7);
      g.fillStyle = "#5BE3A6";
      g.fill();
    });
    g.fillStyle = "#5C6680";
    g.font = '10px "JetBrains Mono",monospace';
    g.textAlign = "right";
    [1, 1.5, 2, 2.5].forEach((v) => {
      if (v <= maxV) g.fillText(v.toFixed(1), pad.l - 6, Y(v) + 3);
    });
    g.textAlign = "center";
    [0, 80, 160, 240, 320].forEach((k) => g.fillText(fmt(256 + k), X(k), H - pad.b + 14));
    const last = CURVE[CURVE.length - 1];
    $("ro8").innerHTML = `held-out bytes <b>${fmt(bytesHO)}</b><br>at V=256 → <b>1.00</b> B/token<br>at V=${fmt(256 + N_MERGES)} → <b>${last.v.toFixed(2)}</b> B/token`;
  }

  function drawCost() {
    const V = parseFloat($("v8").value),
      d = parseFloat($("d8").value),
      L = parseFloat($("l8").value);
    const { g, W, H } = fitCanvas($("cvCost"), 230);
    const pad = { l: 46, r: 16, t: 16, b: 30 };
    g.clearRect(0, 0, W, H);
    grid(g, W, H, pad, 4);
    axes(g, W, H, pad, "model width d", "vocab share of parameters");
    const ds = [];
    for (let x = 256; x <= 8192; x += 128) ds.push(x);
    const share = (dd) => (2 * V * dd) / (2 * V * dd + 12 * L * dd * dd);
    const X = (dd) => pad.l + ((W - pad.l - pad.r) * (dd - 256)) / (8192 - 256);
    const Y = (s) => pad.t + (H - pad.t - pad.b) * (1 - s);
    g.beginPath();
    ds.forEach((dd, i) => (i ? g.lineTo(X(dd), Y(share(dd))) : g.moveTo(X(dd), Y(share(dd)))));
    g.strokeStyle = "#F5A524";
    g.lineWidth = 2.2;
    g.stroke();
    g.lineTo(X(8192), H - pad.b);
    g.lineTo(X(256), H - pad.b);
    g.closePath();
    g.fillStyle = "#F5A52418";
    g.fill();
    const s = share(d);
    g.beginPath();
    g.arc(X(d), Y(s), 5, 0, 7);
    g.fillStyle = "#5BE3A6";
    g.fill();
    g.fillStyle = "#5C6680";
    g.font = '10px "JetBrains Mono",monospace';
    g.textAlign = "right";
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => g.fillText((v * 100).toFixed(0) + "%", pad.l - 6, Y(v) + 3));
    g.textAlign = "center";
    [256, 2048, 4096, 6144, 8192].forEach((dd) => g.fillText(fmt(dd), X(dd), H - pad.b + 14));

    const pv = 2 * V * d,
      pb = 12 * L * d * d;
    const M = (x) => (x >= 1e9 ? (x / 1e9).toFixed(2) + "B" : x >= 1e6 ? (x / 1e6).toFixed(1) + "M" : fmt(x));
    $("c8v").textContent = M(pv);
    $("c8b").textContent = M(pb);
    $("c8t").textContent = M(pv + pb);
    $("c8s").textContent = (s * 100).toFixed(1) + "%";
    $("v8v").textContent = fmt(V);
    $("d8v").textContent = fmt(d);
    $("l8v").textContent = fmt(L);
  }

  ["v8", "d8", "l8"].forEach((id) => bind(id, drawCost));
  window.__redraw = ((prev) => () => {
    if (prev) prev();
    drawVocab();
    drawCost();
  })(window.__redraw);
  setTimeout(() => {
    drawVocab();
    drawCost();
  }, 0);
}

/* ============================================================================
   09 · THE FOOTGUN MUSEUM
   ============================================================================ */
function sec09(sec) {
  /* a) spelling */
  const a = el("div", "card");
  a.innerHTML = `<div class="eqno">exhibit A</div><div class="eqname">it cannot see letters</div>
    <div class="does">Ask a model how many <b>r</b>'s are in a word and you are asking it to introspect on something it
    was never shown. The word arrives as a small number of opaque integers. To answer, the model must have
    <span class="hl">memorised the spelling of each token</span> during training — which it does, imperfectly, for
    common tokens, and not at all for rare ones.</div>`;
  const ai = el("input");
  ai.type = "text";
  ai.value = "strawberry";
  a.appendChild(ai);
  const ab = el("div", "stream");
  ab.style.marginTop = "12px";
  a.appendChild(ab);
  const apr = el("div", "btnrow");
  [["strawberry"], ["unbelievable"], ["tokenization"], ["Ambassadorial"]].forEach(([w]) => {
    const b = el("button", "ghost tiny", w);
    b.onclick = () => {
      ai.value = w;
      aUpd();
    };
    apr.appendChild(b);
  });
  a.appendChild(apr);
  a.appendChild(
    statStrip([
      { k: "letters", v: "—", id: "f9l" },
      { k: "tokens", v: "—", id: "f9t", c: "c" },
      { k: "letters visible", v: "0", id: "f9v", c: "r", s: "as separate positions" },
    ])
  );
  a.appendChild(
    el(
      "div",
      "note",
      `<b>Reversing a string is the same problem.</b> The model must decompose each token into letters, reverse, and
       recompose — three operations it has no primitive for. Ask it to spell the word out with spaces first and
       accuracy jumps, because now the letters really are separate tokens.`
    )
  );
  sec.appendChild(a);
  function aUpd() {
    const ids = encode(ai.value, TOK);
    renderStream(ab, ids, TOK, { showIds: true });
    $("f9l").textContent = fmt([...ai.value].length);
    $("f9t").textContent = fmt(ids.length);
  }
  ai.addEventListener("input", aUpd);
  setTimeout(aUpd, 0);

  /* b) arithmetic */
  const b = el("div", "card");
  b.innerHTML = `<div class="eqno">exhibit B</div><div class="eqname">arithmetic dies at the chunk boundary</div>
    <div class="does">GPT-4's pattern caps digit runs at three — but it chops them <b>left to right</b>, so the grouping
    depends on the total length. <span class="hl">Two numbers of different lengths get incompatible groupings</span>,
    and column-wise addition — the algorithm every human uses — is not expressible over those tokens.</div>`;
  const bi = el("textarea");
  bi.rows = 3;
  bi.value = "1234567\n12345678\n999 + 1 = 1000";
  b.appendChild(bi);
  const bnums = el("div");
  bnums.style.marginTop = "12px";
  b.appendChild(bnums);
  b.appendChild(
    el(
      "div",
      "note warn",
      `<b>How the labs fix it.</b> Llama forces <span class="mono-i">split_digits=True</span> (every digit its own
       token). Some models tokenize numbers right-to-left so that place value lines up. Both trade compression for
       arithmetic — and both are decisions made in the tokenizer, not the model.`
    )
  );
  sec.appendChild(b);
  function bUpd() {
    bnums.innerHTML = "";
    bi.value.split("\n").slice(0, 6).forEach((line) => {
      if (!line.trim()) return;
      const row = el("div");
      row.style.margin = "0 0 10px";
      row.innerHTML = `<span class="field-lab">${esc(line)}</span>`;
      const s = el("div", "stream");
      s.style.maxHeight = "70px";
      splitChunks(line, PATTERNS.gpt4).forEach((ch) => s.appendChild(el("span", "tk sp", esc(pieceLabel(bytesOf(ch))))));
      row.appendChild(s);
      bnums.appendChild(row);
    });
  }
  bi.addEventListener("input", bUpd);
  setTimeout(bUpd, 0);

  /* c) trailing whitespace */
  const cw = el("div", "card");
  cw.innerHTML = `<div class="eqno">exhibit C</div><div class="eqname">the trailing space</div>
    <div class="does">In GPT tokenizers the space belongs to the <i>following</i> word:
    <span class="mono-i">"·hello"</span> is one common token. If your prompt ends with a space, the model must now
    predict a word <b>without</b> its usual leading space — a token distribution it has rarely seen. The completion
    degrades for a reason that looks like nothing at all in your code.</div>`;
  const cg = el("div");
  cw.appendChild(cg);
  cw.appendChild(
    el(
      "div",
      "note warn",
      `<b>This is the "trailing whitespace" warning</b> that OpenAI's playground used to show. Strip trailing
       whitespace from prompts. Always.`
    )
  );
  sec.appendChild(cw);
  ["Once upon a time,", "Once upon a time, ", "Once upon a time,  "].forEach((t) => {
    const row = el("div");
    row.style.margin = "0 0 10px";
    const ids = encode(t, TOK);
    row.innerHTML = `<span class="field-lab">${esc(JSON.stringify(t))} — ${ids.length} tokens</span>`;
    const s = el("div", "stream");
    s.style.maxHeight = "70px";
    renderStream(s, ids, TOK, {});
    row.appendChild(s);
    cg.appendChild(row);
  });

  /* d) non-English */
  const dd = el("div", "card");
  dd.innerHTML = `<div class="eqno">exhibit D</div><div class="eqname">the same sentence costs 3× more in Korean</div>
    <div class="does">Two compounding penalties. First, <b>UTF-8 itself</b>: Hangul and CJK cost 3 bytes per character
    against English's 1. Second, <b>the merge table</b>: it was trained on English, so those bytes never earned merges
    and stay as singletons. The result is a real, measurable tax on <span class="hl">context window, latency and API
    price</span> for most of the world's languages.</div>`;
  const dt = el("div", "tblwrap");
  const en = MULTILINGUAL[0];
  const enChars = [...en.text].length;
  const enTok = encode(en.text, TOK).length;
  dt.innerHTML =
    `<table class="sum"><tr><th>language</th><th>characters</th><th>utf-8 bytes</th><th>tokens</th><th>tokens / char</th><th>vs English</th></tr>` +
    MULTILINGUAL.map((m) => {
      const ch = [...m.text].length,
        by = bytesOf(m.text).length,
        tk = encode(m.text, TOK).length;
      const ratio = tk / ch / (enTok / enChars);
      return `<tr${m.lang === "English" ? ' class="hi"' : ""}><td>${m.flag} ${m.lang}</td><td class="m2">${ch}</td>
        <td class="mono">${by}</td><td class="mono">${tk}</td><td class="m2">${(tk / ch).toFixed(2)}</td>
        <td style="color:${ratio > 1.6 ? "var(--bad)" : ratio > 1.1 ? "var(--signal)" : "var(--good)"};font-family:'JetBrains Mono',monospace">${ratio.toFixed(2)}×</td></tr>`;
    }).join("") +
    `</table>`;
  dd.appendChild(dt);
  dd.appendChild(
    el(
      "div",
      "note",
      `<b>Measured with the tokenizer trained on this page</b> — English-only training data, so the effect here is
       stronger than in production models, which include multilingual data. But the effect is real in production too:
       GPT-4's tokenizer still costs roughly <b>2–3×</b> more tokens per character for Korean, Japanese and Hindi than
       for English.`
    )
  );
  sec.appendChild(dd);

  /* e) code */
  const ee = el("div", "card");
  ee.innerHTML = `<div class="eqno">exhibit E</div><div class="eqname">why GPT-2 was bad at Python</div>
    <div class="does">GPT-2's merge table had no tokens for runs of spaces, so a 4-space indent was
    <span class="hl">four separate tokens</span>, every line, in every file. Indentation-heavy code shredded the
    context window and spread each line's meaning across dozens of positions. GPT-4's tokenizer learned multi-space
    tokens, and Python performance jumped — <b>with no change to the architecture</b>.</div>`;
  const code = "def solve(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total";
  const eg = el("div");
  [["GPT-2 pattern", "gpt2"], ["GPT-4 pattern", "gpt4"]].forEach(([lab, p]) => {
    const chunks = splitChunks(code, PATTERNS[p]);
    const row = el("div");
    row.style.margin = "0 0 12px";
    row.innerHTML = `<span class="field-lab">${lab} — ${chunks.length} chunks</span>`;
    const s = el("div", "stream");
    s.style.maxHeight = "120px";
    chunks.forEach((ch) => s.appendChild(el("span", "tk sp", esc(pieceLabel(bytesOf(ch))))));
    row.appendChild(s);
    eg.appendChild(row);
  });
  ee.appendChild(eg);
  ee.appendChild(
    el(
      "div",
      "note",
      `<b>The chunking is only half the story.</b> The other half is whether the <i>merge table</i> contains a token for
       <span class="mono-i">"····"</span>. GPT-4's does; GPT-2's did not. Same regex family, very different vocabulary.`
    )
  );
  sec.appendChild(ee);

  /* f) SolidGoldMagikarp */
  const ff = el("div", "card");
  ff.innerHTML = `<div class="eqno">exhibit F</div><div class="eqname">SolidGoldMagikarp · the untrained token</div>
    <div class="does"><b>The tokenizer's training set is not the model's training set.</b> GPT-2's tokenizer was fit on
    a corpus that included Reddit scrapes, in which certain usernames appeared often enough to earn their own token —
    <span class="mono-i">SolidGoldMagikarp</span>, <span class="mono-i">petertodd</span>,
    <span class="mono-i">RandomRedditorWithNo</span>. The <i>model</i> was then trained on data where those strings had
    been filtered out.</div>
    <div class="does">So the embedding row for that token stayed near its <span class="hl">random
    initialisation</span> — never nudged by a gradient, sitting in a region of embedding space the model has no
    behaviour for. Feed it in and the model evades, insults, hallucinates, or repeats a different word entirely. It is
    not a jailbreak or a hidden personality. It is an <b>uninitialised variable</b>.</div>
    <div class="findings">
      <div class="finding"><div class="h">tokenizer corpus</div><div class="b">raw web scrape, unfiltered. Contains the usernames often enough to merge them into single tokens.</div></div>
      <div class="finding"><div class="h">model corpus</div><div class="b">cleaned, deduplicated, filtered. Those strings are gone.</div></div>
      <div class="finding"><div class="h">result</div><div class="b">a token that <b>exists</b> in the vocabulary but has <b>no learned meaning</b>. Under-trained by construction.</div></div>
      <div class="finding"><div class="h">the lesson</div><div class="b">Train the tokenizer on the <b>same distribution</b> as the model, and audit rare tokens for near-initialisation embedding norms before you ship.</div></div>
    </div>`;
  sec.appendChild(ff);
}

/* ============================================================================
   10 · MASTERY CHECK
   ============================================================================ */
const QUIZ = [
  {
    q: "A model is asked to reverse the word 'tokenization' and gets it wrong. What is the most direct cause?",
    o: [
      "The attention mechanism cannot represent reversal",
      "The word arrives as a few opaque token ids; the individual characters are not separate positions",
      "The model was not trained on enough reversal examples",
      "The softmax temperature is too high",
    ],
    a: 1,
    w: "The letters are simply not there as separate things to permute. Splitting the word into spaced letters first — so each letter really is its own token — measurably improves accuracy on exactly this task.",
  },
  {
    q: "During encoding, which pair does BPE collapse first?",
    o: [
      "The longest vocabulary entry that matches the text",
      "The most frequent pair in the input text",
      "The pair with the lowest merge index — the earliest one learned during training",
      "Whichever pair appears leftmost",
    ],
    a: 2,
    w: "Encoding replays training order. A later merge can consume a token that only exists because an earlier merge created it, so applying merges out of order produces a different — and wrong — tokenization.",
  },
  {
    q: "Why is a byte-level BPE tokenizer unable to produce an <UNK> token?",
    o: [
      "Because it has a very large vocabulary",
      "Because every token bottoms out in one of the 256 byte values, so any string is representable",
      "Because unknown characters are silently deleted",
      "Because the regex pattern rejects invalid input",
    ],
    a: 1,
    w: "The vocabulary is closed under the bytes. Worst case, a never-before-seen character is spelled out as its individual UTF-8 bytes — expensive, but never lossy.",
  },
  {
    q: "What is the purpose of the split regex applied before BPE?",
    o: [
      "To speed up training by parallelising across chunks",
      "To normalise Unicode into a canonical form",
      "To forbid merges across letter/digit/punctuation boundaries, so vocabulary isn't wasted on 'dog.' vs 'dog!'",
      "To strip whitespace from the input",
    ],
    a: 2,
    w: "Merges happen strictly inside chunks. The regex is a hand-written prior about what a token should never span — the least learned, most hard-coded component in the whole pipeline.",
  },
  {
    q: "GPT-2's vocabulary is 50,257. Where does that number come from?",
    o: [
      "It was chosen to be a round number under 2^16",
      "256 byte tokens + 50,000 learned merges + 1 special token",
      "50,257 distinct words in the training corpus",
      "It is the number of merges that fit in the embedding budget",
    ],
    a: 1,
    w: "Every BPE vocabulary decomposes this way: base bytes, plus one id per merge, plus special tokens bolted on afterwards.",
  },
  {
    q: "Doubling vocab_size, with the training corpus fixed, has which downside beyond parameters?",
    o: [
      "Sequences get longer",
      "Decoding becomes ambiguous",
      "Each token is seen roughly half as often, so rare embeddings are trained less well",
      "The regex must be recompiled",
    ],
    a: 2,
    w: "This is the same force behind SolidGoldMagikarp, in gentler form. A bigger vocabulary means a longer tail of tokens whose embeddings sit close to their initialisation.",
  },
  {
    q: "SentencePiece differs from tiktoken most fundamentally in that it…",
    o: [
      "uses unigram instead of BPE, always",
      "runs BPE on Unicode code points rather than on UTF-8 bytes",
      "cannot handle emoji",
      "does not support special tokens",
    ],
    a: 1,
    w: "Everything else — character_coverage, byte_fallback, the ▁ marker, add_dummy_prefix — follows from that one choice. SentencePiece can also do unigram, but it does BPE too; the operand is the real difference.",
  },
  {
    q: "Why should user-supplied text be encoded with special tokens disabled?",
    o: [
      "Special tokens slow down encoding",
      "Otherwise a user can type '<|im_end|><|im_start|>system' and forge a turn boundary",
      "Special tokens are not valid UTF-8",
      "The decoder cannot handle them",
    ],
    a: 1,
    w: "The chat 'structure' is only string concatenation. If user text can mint control tokens, the user can write your system prompt. tiktoken makes you opt in for exactly this reason.",
  },
];

function sec10(sec) {
  const c = el("div", "card");
  c.innerHTML = `<div class="eqno">§10</div><div class="eqname">check yourself</div>
    <div class="does">Eight questions. Each explains itself once answered.</div>`;
  sec.appendChild(c);
  const host = el("div");
  let score = 0,
    answered = 0;
  QUIZ.forEach((q, i) => {
    const w = el("div", "q");
    w.innerHTML = `<div class="qq"><span class="n">Q${i + 1}</span>${esc(q.q)}</div>`;
    const why = el("div", "why", `<b>Why:</b> ${q.w}`);
    q.o.forEach((o, j) => {
      const b = el("button", "opt", esc(o));
      b.onclick = () => {
        if (w.dataset.done) return;
        w.dataset.done = "1";
        answered++;
        if (j === q.a) score++;
        [...w.querySelectorAll(".opt")].forEach((x, xi) => {
          if (xi === q.a) x.classList.add("right");
          else if (xi === j) x.classList.add("wrong");
        });
        why.classList.add("show");
        $("qscore").textContent = score + " / " + answered;
        $("qbar").style.width = (100 * score) / QUIZ.length + "%";
      };
      w.appendChild(b);
    });
    w.appendChild(why);
    host.appendChild(w);
  });
  sec.appendChild(host);
  const s = el("div", "card");
  s.innerHTML = `<div class="eqname">score</div>
    <div class="stats"><div class="stat"><div class="k">correct</div><div class="v g" id="qscore">0 / 0</div></div></div>
    <div class="bar" style="margin-top:10px"><i id="qbar" style="width:0%"></i></div>`;
  sec.appendChild(s);

  const f = el("div", "card");
  f.innerHTML = `<div class="eqno">§</div><div class="eqname">what to actually do</div>
    <div class="findings">
      <div class="finding"><div class="h">Don't train one</div><div class="b">For almost every application, reuse <b>cl100k_base</b> or <b>o200k_base</b> via tiktoken. A tokenizer trained on your 50 MB of data will be worse than one trained on the internet.</div></div>
      <div class="finding"><div class="h">If you must</div><div class="b">Use SentencePiece BPE, copy Llama's flags, turn normalisation <b>off</b>, and verify round-trip on real data before training anything.</div></div>
      <div class="finding"><div class="h">Measure your domain</div><div class="b">Compute bytes-per-token on <b>your</b> corpus, not on English prose. If it's under 3, your inference bill is being set by the tokenizer.</div></div>
      <div class="finding"><div class="h">Audit the tail</div><div class="b">Rank vocabulary rows by embedding norm. Rows near the initialisation scale are tokens your model never learned — treat them as landmines.</div></div>
      <div class="finding"><div class="h">Strip trailing spaces</div><div class="b">One line of code. Removes an entire class of mysterious quality regressions.</div></div>
      <div class="finding"><div class="h">Never trust user text</div><div class="b">Encode it with special tokens disabled, every time, without exception.</div></div>
    </div>
    <div class="note warn"><b>The open problem.</b> Tokenization is a wart. It is the reason models can't spell, can't
    count, and cost triple in Hindi. Byte-level architectures (MEGABYTE, byte-latent transformers) are attempts to
    delete this stage entirely — trading a hand-written compressor for a learned one. Nobody has fully won yet.
    Eternal glory awaits whoever does.</div>`;
  sec.appendChild(f);
}

/* ============================================================================
   register
   ============================================================================ */
function buildAllSections() {
  RAW = trainBPE(CORPUS, N_MERGES, "none");
  TOK = trainBPE(CORPUS, N_MERGES, "gpt4");

  section("s0", "00", "the hook", "Everything is tokenization",
    "Eight famous LLM failures. One cause. Start here and the rest of the page is a series of explanations you already want.", sec00);
  section("s1", "01", "foundations", "Text is not characters",
    "Before bytes there are code points, and before code points there is a standards committee. Ten minutes here saves ten hours later.", sec01);
  section("s2", "02", "motivation", "Why not just feed it bytes?",
    "Both obvious answers — characters and whole words — fail badly. Subword tokenization is the compromise, and it is learned compression.", sec02);
  section("s3", "03", "the algorithm", "Byte Pair Encoding",
    "Count pairs, merge the winner, repeat. Train it here, one merge at a time, and watch a corpus compress itself.", sec03);
  section("s4", "04", "inference", "Encode & decode",
    "Turning text into ids is a replay of training in the learned order — and turning ids back into text is total, even when the ids are nonsense.", sec04);
  section("s5", "05", "the split", "The regex nobody explains",
    "A hand-written regular expression decides what a token may never span. It is the least glamorous and most consequential line in the pipeline.", sec05);
  section("s6", "06", "control", "Special tokens",
    "Document boundaries, chat turns, fill-in-the-middle — and the injection surface they create.", sec06);
  section("s7", "07", "the other family", "SentencePiece vs tiktoken",
    "Same algorithm, opposite order of operations. This is why the GPT and Llama ecosystems feel so different.", sec07);
  section("s8", "08", "economics", "Choosing vocab_size",
    "Compression, parameters, and training signal pull in three directions. Here is the actual measured trade.", sec08);
  section("s9", "09", "pathology", "The footgun museum",
    "Six classic failures, reproduced live. Once you can see the cause, you can never unsee it.", sec09);
  section("s10", "10", "mastery", "Check yourself",
    "Eight questions, then the short list of things to actually do on Monday.", sec10);
}
