// The text layer, ported from en2thaana/text.py.
//
// The rule engine is NOT here and does not need to be: Stage 2 is
// deterministic, so every answer for a known word was precomputed into
// dict.txt at build time. The browser does lookup plus tokenisation, which is
// why this page needs no server.
//
// Everything data-shaped -- the abbreviation table, acronym letter names, the
// gazetteer, the punctuation map, even the tokenizer's pattern -- comes from
// data.js, generated from the Python source, so the two cannot drift.

import { DATA } from "./data.js";

const TOKEN = new RegExp(DATA.tokenRe, "g");
const DOTTED_ACRONYM = /\b(?:[A-Za-z]\.){2,}/g;

let DICT = null;

export async function load(onProgress) {
  const res = await fetch("dict.txt");
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress) onProgress(total ? got / total : 0);
  }
  const text = new TextDecoder().decode(
    chunks.reduce((acc, c) => (acc.set(c, acc.__o || 0), (acc.__o = (acc.__o || 0) + c.length), acc),
      Object.assign(new Uint8Array(got), { __o: 0 })));
  DICT = new Map();
  for (const line of text.split("\n")) {
    const i = line.indexOf("\t");
    if (i > 0) DICT.set(line.slice(0, i), line.slice(i + 1));
  }
  return DICT.size;
}

const arabicPunct = (s) =>
  [...s].map((c) => DATA.punct[c] || c).join("");

// U.S.A. -> USA, so the tokenizer sees one acronym rather than three letters.
export const joinDottedAcronyms = (t) =>
  t.replace(DOTTED_ACRONYM, (m) => m.replace(/\./g, "").toUpperCase());

const spellOut = (w) =>
  [...w].filter((c) => /[a-z]/i.test(c))
        .map((c) => DATA.letters[c.toLowerCase()] || c).join("");

function isAcronym(raw) {
  if (raw.length < 2 || raw !== raw.toUpperCase() || !/[A-Z]/.test(raw)) return false;
  // A shouted word is still a word: HOUSE must not become އެޗްއޯޔޫއެސްއީ.
  if (raw.length >= 4 && DICT.has(raw.toLowerCase())) return false;
  return true;
}

export function transcribeToken(raw, following = "") {
  if (!raw) return { text: raw, kind: "other", thaana: raw };
  if (!/[A-Za-z0-9]/.test(raw[0]))
    return { text: raw, kind: "other", thaana: arabicPunct(raw) };
  // Thaana text uses the same digits, and the comma in 120,000 is a separator.
  if (/[0-9]/.test(raw[0])) return { text: raw, kind: "number", thaana: raw };

  const low = raw.toLowerCase();

  // Only with a following period, so the surname `Ms` and the word `co` survive.
  if (DATA.abbrev[low] && following.startsWith(".")) {
    const t = DICT.get(DATA.abbrev[low]);
    if (t) return { text: raw, kind: "abbreviation", thaana: t, source: "lexicon" };
  }
  if (isAcronym(raw))
    return { text: raw, kind: "acronym", thaana: spellOut(raw), source: "spelled out" };

  // The lexicon first. The address registry holds hundreds of ordinary English
  // words -- fresh, beauty, camera -- because Maldivians name houses with them.
  const hit = DICT.get(low);
  if (hit) return { text: raw, kind: "word", thaana: hit, source: "lexicon" };

  // Not an English word: a Maldivian place name is now the likely reading, and
  // the registry spelling is a fact where a guess would only be a guess.
  const gz = DATA.gazetteer[low];
  if (gz) return { text: raw, kind: "placename", thaana: gz, source: "registry" };

  return { text: raw, kind: "word", thaana: null, source: "" };
}

export function transcribe(text) {
  const raw = joinDottedAcronyms(text).match(TOKEN) || [];
  return raw.map((tok, i) => transcribeToken(tok, raw[i + 1] || ""));
}
