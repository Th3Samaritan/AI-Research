"use strict";
/* ============================================================================
   bpe.js — a real, working Byte Pair Encoding tokenizer in the browser.
   No dependencies. Every number the site shows is computed by this file:
   the merges are actually learned, the encoder actually runs the merge
   ladder, and decode(encode(x)) actually round-trips.

   Mirrors karpathy/minbpe: train() -> merges, encode() -> ids, decode() -> str
   ============================================================================ */

const TE = new TextEncoder();
const TD = new TextDecoder("utf-8", { fatal: false });

/* ---------- byte helpers ---------- */
const bytesOf = (s) => Array.from(TE.encode(s));
const strOf = (bytes) => TD.decode(new Uint8Array(bytes));

/* pair -> single integer key. ids stay well under 65536 so this is exact. */
const KEY = (a, b) => a * 65536 + b;
const UNKEY = (k) => [Math.floor(k / 65536), k % 65536];

/* ---------- the two primitives from the lecture ---------- */

/** count how often each consecutive pair appears */
function getStats(ids, counts) {
  const c = counts || new Map();
  for (let i = 0; i + 1 < ids.length; i++) {
    const k = KEY(ids[i], ids[i + 1]);
    c.set(k, (c.get(k) || 0) + 1);
  }
  return c;
}

/** replace every occurrence of (a,b) with the new id idx */
function merge(ids, a, b, idx) {
  const out = [];
  let i = 0;
  while (i < ids.length) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
      out.push(idx);
      i += 2;
    } else {
      out.push(ids[i]);
      i += 1;
    }
  }
  return out;
}

/* ---------- the GPT split patterns ---------- */
/* GPT-2 (Radford et al. 2019) — the original. Numbers are NOT chunked. */
const GPT2_PAT =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
/* GPT-4 / cl100k_base — case-insensitive contractions, digits capped at 3,
   whitespace handled far more carefully. (JS has no inline (?i:) group, so the
   contraction alternatives are spelled out — behaviourally identical.) */
const GPT4_PAT =
  /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

const PATTERNS = { none: null, gpt2: GPT2_PAT, gpt4: GPT4_PAT };

function splitChunks(text, pat) {
  if (!pat) return [text];
  return text.match(new RegExp(pat.source, pat.flags)) || [];
}

/* ---------- training ---------- */
/**
 * Learn `numMerges` merges over `text`.
 * Returns { merges, ranks, vocab, base, chunks } where
 *   merges[i] = {a,b,idx,count,total,piece}
 *   ranks     = Map(pairKey -> idx)   (merge order == priority at encode time)
 *   vocab     = idx -> byte array
 */
function trainBPE(text, numMerges, patName) {
  const pat = PATTERNS[patName || "none"];
  const chunks = splitChunks(text, pat).map(bytesOf);
  let cur = chunks;
  const base = cur.reduce((s, c) => s + c.length, 0);

  const ranks = new Map();
  const merges = [];
  const vocab = [];
  for (let i = 0; i < 256; i++) vocab[i] = [i];

  for (let i = 0; i < numMerges; i++) {
    const stats = new Map();
    for (const ch of cur) getStats(ch, stats);
    let best = -1,
      bc = 1; // require count >= 2, otherwise merging is pointless
    for (const [k, c] of stats) if (c > bc) (bc = c), (best = k);
    if (best < 0) break;

    /* keep the runners-up so the UI can show what the merge competed against */
    const top = [...stats.entries()]
      .sort((x, y) => y[1] - x[1])
      .slice(0, 5)
      .map(([k, c]) => {
        const [p, q] = UNKEY(k);
        return { a: p, b: q, count: c, piece: strOf(vocab[p].concat(vocab[q])) };
      });

    const [a, b] = UNKEY(best);
    const idx = 256 + i;
    cur = cur.map((ch) => merge(ch, a, b, idx));
    const total = cur.reduce((s, c) => s + c.length, 0);

    vocab[idx] = vocab[a].concat(vocab[b]);
    ranks.set(best, idx);
    merges.push({ a, b, idx, count: bc, total, top, piece: strOf(vocab[idx]) });
  }
  return { merges, ranks, vocab, base, chunks, patName: patName || "none" };
}

/* ---------- inference ---------- */

/** encode one chunk, always applying the lowest-rank (earliest-learned) pair */
function encodeChunk(bytes, ranks) {
  let ids = bytes.slice();
  while (ids.length >= 2) {
    let bestRank = Infinity,
      bestKey = -1;
    for (let i = 0; i + 1 < ids.length; i++) {
      const r = ranks.get(KEY(ids[i], ids[i + 1]));
      if (r !== undefined && r < bestRank) (bestRank = r), (bestKey = KEY(ids[i], ids[i + 1]));
    }
    if (bestKey < 0) break;
    const [a, b] = UNKEY(bestKey);
    ids = merge(ids, a, b, bestRank);
  }
  return ids;
}

/** encode a whole string with a trained tokenizer */
function encode(text, tok) {
  const pat = PATTERNS[tok.patName];
  const out = [];
  for (const ch of splitChunks(text, pat)) {
    for (const id of encodeChunk(bytesOf(ch), tok.ranks)) out.push(id);
  }
  return out;
}

/** encode, but record every intermediate state so it can be animated */
function encodeTrace(text, tok) {
  const pat = PATTERNS[tok.patName];
  const chunks = splitChunks(text, pat);
  const steps = [];
  // merges happen inside chunks; the trace is replayed on the flattened view
  const perChunk = chunks.map(bytesOf);
  steps.push({ ids: perChunk.flat(), applied: null });
  let working = perChunk;
  for (let guard = 0; guard < 400; guard++) {
    let bestRank = Infinity,
      bestKey = -1;
    for (const ch of working)
      for (let i = 0; i + 1 < ch.length; i++) {
        const r = tok.ranks.get(KEY(ch[i], ch[i + 1]));
        if (r !== undefined && r < bestRank) (bestRank = r), (bestKey = KEY(ch[i], ch[i + 1]));
      }
    if (bestKey < 0) break;
    const [a, b] = UNKEY(bestKey);
    working = working.map((ch) => merge(ch, a, b, bestRank));
    steps.push({ ids: working.flat(), applied: { a, b, idx: bestRank } });
  }
  return steps;
}

/** ids -> text (lossless for anything the tokenizer can represent) */
function decode(ids, tok) {
  const bytes = [];
  for (const id of ids) {
    const v = tok.vocab[id];
    if (v) for (const b of v) bytes.push(b);
  }
  return strOf(bytes);
}

/** apply only the first k learned merges — used to scrub the training film */
function applyFirstK(chunks, tok, k) {
  let cur = chunks.map((c) => c.slice());
  for (let i = 0; i < k && i < tok.merges.length; i++) {
    const m = tok.merges[i];
    cur = cur.map((ch) => merge(ch, m.a, m.b, m.idx));
  }
  return cur;
}

/* ---------- display helpers ---------- */
/** printable form of a token: spaces/newlines/tabs made visible, bad bytes shown */
function pieceLabel(bytes) {
  const s = strOf(bytes);
  let out = "";
  for (const ch of s) {
    if (ch === " ") out += "·";
    else if (ch === "\n") out += "⏎";
    else if (ch === "\t") out += "⇥";
    else if (ch === "\r") out += "␍";
    else out += ch;
  }
  return out || "∅";
}
const tokenLabel = (id, tok) => pieceLabel(tok.vocab[id] || [63]);

/* ---------- the training corpus ---------- */
/* Original prose written for this page, plus a code sample, so the merges the
   trainer discovers are recognisable English/code subwords. ~5.8 KB. */
const CORPUS = `A tokenizer is a completely separate module from the language model. It has its own
training data, its own algorithm, and its own vocabulary. The language model never sees text.
It only ever sees a sequence of integers, and it only ever predicts the next integer.

The tokenizer is trained once, before the model exists, and then it is frozen. Everything the
model can possibly express is fixed at that moment. If the tokenizer never learned a token for
a word, the model must spell that word out of smaller pieces, one piece at a time, and it must
learn how those pieces combine from scratch.

The training text is first converted to bytes. Text is not characters, and characters are not
bytes. A string of text is a sequence of Unicode code points. Each code point is encoded to
one, two, three, or four bytes by the UTF-8 encoding. English letters take one byte each.
Accented letters take two. Most of the characters used for writing in Asia take three. Emoji
take four. This is why the same sentence, written in different languages, produces a very
different number of bytes, and therefore a very different number of tokens, and therefore a
very different cost to process.

Byte pair encoding starts from the 256 possible byte values and repeatedly does one thing:
find the most common adjacent pair of tokens in the training data, invent a new token for that
pair, and replace every occurrence of the pair with the new token. Each merge shortens the
sequence a little and grows the vocabulary by one. Run it a thousand times and the vocabulary
contains common letter pairs. Run it fifty thousand times and the vocabulary contains whole
words, common prefixes, common suffixes, and long runs of whitespace.

The order of the merges matters. The merges are learned in order, and at encoding time they
must be applied in the same order, because a later merge may depend on a token that an earlier
merge created. This is why the encoder does not simply look for the longest matching string in
the vocabulary. It repeatedly looks for the pair with the lowest merge index and applies that
one first. Encoding is a replay of training, restricted to a single piece of text.

Decoding is trivial by comparison. Every token is a sequence of bytes. Concatenate the bytes
of every token in the sequence, then decode the result as UTF-8. Because every token bottoms
out in raw bytes, the tokenizer can represent any possible string, including strings it has
never seen, including strings in languages that never appeared in the training data.

Before any of this happens, the text is usually split by a regular expression. Without a split
pattern, the trainer would happily learn a token for the string "dog." including the period,
and another for "dog!" and another for "dog?" and another for "dog," and the vocabulary would
fill up with punctuation variants of the same word. The split pattern forbids merges across a
letter boundary, a digit boundary, and a punctuation boundary. The tokenizer is then trained
independently inside each chunk, and the chunks are encoded independently and concatenated.

Special tokens sit outside the algorithm entirely. They are appended to the vocabulary by hand
after training, they never appear as the result of a merge, and the encoder handles them by
searching for them in the raw text before any splitting happens. The end of text token marks a
document boundary, and it teaches the model that whatever came before is finished and should
not influence what comes next. Chat models add more of them, to mark the start of a turn, the
end of a turn, and the identity of the speaker.

Choosing the size of the vocabulary is a trade. A larger vocabulary means each token carries
more text, so a document becomes a shorter sequence, so attention is cheaper and more text
fits inside the context window. But a larger vocabulary means a larger embedding table and a
larger output layer, both of which scale linearly with the size of the vocabulary, and it also
means every individual token is seen less often during training, so its embedding is trained
less well. Past some point the rare tokens are barely trained at all.

The pieces that are barely trained are the source of the strangest failures. A token can exist
in the vocabulary because the tokenizer training data contained it, while the model training
data contained almost none of it. The embedding for that token is then close to its random
initialisation. Feeding it to the model produces behaviour that looks like nonsense, because
in a real sense the model has never learned what that token means.

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges

The model is not reading the text. The model is reading the tokenizer's opinion of the text.
Every question about spelling, every question about counting letters, every question about
arithmetic, and every question about reversing a string is a question about the tokenizer
first and the model second. A model that cannot count the letters in a word is usually not
failing to count. It is failing to see the letters at all, because the word arrived as one
indivisible integer and the letters were never there to count.`;

/* short held-out passages used for the measurement plots (never trained on) */
const HELDOUT = `The tokenizer decides what the model is able to see. A held out passage is
useful because it measures compression on text the merges were not fitted to, which is the
number that actually matters when the model is deployed on documents nobody has read yet.
Every token in this paragraph must be reconstructed from merges learned somewhere else.`;

const MULTILINGUAL = [
  { lang: "English", flag: "EN", text: "The quick brown fox jumps over the lazy dog." },
  { lang: "French", flag: "FR", text: "Le vif renard brun saute par-dessus le chien paresseux." },
  { lang: "Korean", flag: "KO", text: "빠른 갈색 여우가 게으른 개를 뛰어넘습니다." },
  { lang: "Japanese", flag: "JA", text: "素早い茶色のキツネが怠け者の犬を飛び越えます。" },
  { lang: "Hindi", flag: "HI", text: "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।" },
  { lang: "Emoji", flag: "😀", text: "🦊 jumps over 🐕 — 👋 안녕!" },
];
