// Stage 2 in the browser: IPA -> Thaana.
//
// Known words never reach this. Their answers were computed in Python at build
// time and shipped in dict.txt, because the rules are deterministic and a
// lookup is faster than any amount of clever code. This exists for the other
// case: a word no dictionary has, where the model predicts the phonemes and
// something still has to write them down.
//
// It is a port, so it can drift from the Python it was ported from. That is
// checked rather than hoped for: build_web.py runs both over all 139,219
// dictionary words and refuses to ship if a single one disagrees. The tables
// come from data.js, generated from the Python engine, including the
// orthographic table already filtered by its confidence guard, so the two
// cannot apply different thresholds.

import { DATA } from "./data.js";

const E = DATA.engine;
const FILI = new Set(E.fili);
const VOWEL_LETTERS = "aeiouy";
const RUNS = /[aeiouy]+/g;
const SONORANTS = new Set(["l", "n", "m", "r"]);

const isVowel = (p) =>
  [...p.split(":")[0]].every((c) => E.vowelChars.includes(c));

function normalise(ipa) {
  let out = "";
  for (const c of ipa) {
    if (E.drop.includes(c)) continue;
    out += E.normalise[c] ?? c;
  }
  return out;
}

// Longest match first, so /tS/ becomes one affricate rather than /t/ + /S/,
// which would be written ޓ + ޝ instead of ޗ.
function phonemes(ipa) {
  const s = normalise(ipa);
  const out = [];
  let i = 0;
  outer: while (i < s.length) {
    for (const p of E.phonemes) {
      if (s.startsWith(p, i)) { out.push(p); i += p.length; continue outer; }
    }
    out.push(s[i]); i += 1;
  }
  return out;
}

// English marks a long vowel with a final <e> that is not itself pronounced.
// Counting it makes the letter count exceed the sound count and silently
// disables the orthographic feature for a very large class of words.
function spellingForm(word) {
  const w = word.toLowerCase();
  if (w.length > 3 && w.endsWith("e") && !VOWEL_LETTERS.includes(w[w.length - 2]))
    return w.slice(0, -1);
  return w;
}

function vowelSpans(word) {
  const w = spellingForm(word);
  const out = [];
  RUNS.lastIndex = 0;
  let m;
  while ((m = RUNS.exec(w)) !== null) out.push([m.index, m.index + m[0].length, m[0]]);
  return out;
}

function orthHints(word, nVowelPhonemes) {
  const spans = vowelSpans(word);
  if (spans.length !== nVowelPhonemes) return null;
  const w = spellingForm(word);
  return spans.map(([start, end, text]) =>
    (start === 0 ? "^" : "") + (end === w.length ? "#" : "") + text);
}

// Re-insert the /r/ that non-rhotic RP drops but the spelling still shows,
// placed by position: a vowel run immediately followed by <r>.
function restoreR(word, phones) {
  const spans = vowelSpans(word);
  const vidx = [];
  phones.forEach((p, i) => { if (isVowel(p)) vidx.push(i); });
  if (spans.length !== vidx.length) return phones;
  const w = spellingForm(word);
  const after = new Set();
  spans.forEach(([, end], k) => {
    if (w[end] === "r" && phones[vidx[k] + 1] !== "r") after.add(vidx[k]);
  });
  if (!after.size) return phones;
  const out = [];
  phones.forEach((p, i) => { out.push(p); if (after.has(i)) out.push("r"); });
  return out;
}

// A schwa no letter accounts for: `able` is a-b-l, one vowel letter, but
// /eIb@l/ has two vowel sounds. The spelling writes no fili there.
function dropEpenthetic(word, phones) {
  const nRuns = vowelSpans(word).length;
  const idx = [];
  phones.forEach((p, i) => { if (isVowel(p)) idx.push(i); });
  const surplus = idx.length - nRuns;
  if (surplus <= 0) return phones;
  const cands = [];
  for (const i of idx) {
    if (phones[i] !== "ə") continue;
    const prev = i ? phones[i - 1] : null;
    const next = i + 1 < phones.length ? phones[i + 1] : null;
    if (prev !== null && !isVowel(prev) && SONORANTS.has(next)) cands.push(i);
  }
  if (cands.length < surplus) return phones;
  const drop = new Set(cands.slice(cands.length - surplus));
  return phones.filter((_, i) => !drop.has(i));
}

// Hints are matched BEFORE expansion: a spelling has one vowel run where /aI/
// has two halves. Matching after made every diphthong word lose its hint.
function prepare(word, phones) {
  phones = dropEpenthetic(word, phones);
  const nV = phones.filter(isVowel).length;
  const hints = orthHints(word, nV);
  phones = restoreR(word, phones);

  const outP = [], outH = [];
  let vi = 0;
  for (const p of phones) {
    if (!isVowel(p)) { outP.push(p); continue; }
    const h = hints ? hints[vi] : null;
    vi += 1;
    const parts = E.expand[p] ?? [p];
    outP.push(...parts);
    for (let k = 0; k < parts.length; k++) outH.push(h);
  }
  return [outP, hints ? outH : null];
}

function expandOnly(phones) {
  const out = [];
  for (const p of phones) out.push(...(E.expand[p] ?? [p]));
  return out;
}

export function convert(ipa, word) {
  const [phAll, hints] = word
    ? prepare(word, phonemes(ipa))
    : [expandOnly(phonemes(ipa)), null];

  const units = [];
  let vi = 0;

  const putVowel = (fili) => {
    if (units.length && units[units.length - 1][1] === "")
      units[units.length - 1][1] = fili;
    else units.push([E.alifu, fili]);
  };
  const filiFor = (p, hint) =>
    (hint ? E.orth[p + "|" + hint] : undefined) ?? E.vows[p] ?? "ަ";

  let i = 0;
  while (i < phAll.length) {
    const p = phAll[i];

    // /j/ between a consonant and a vowel is not a consonant here. It becomes
    // an i-fili on the letter before it, and the vowel opens its own carrier:
    // computer is ކޮމްޕިޔުޓަރ, not ކޮމްޕްޔޫޓަރ.
    if (p === "j" && i + 1 < phAll.length && isVowel(phAll[i + 1])
        && units.length && units[units.length - 1][1] === "") {
      units[units.length - 1][1] = E.glide.fili;
      const nxt = phAll[i + 1];
      const hint = hints ? hints[vi] : null;
      vi += 1;
      const f = filiFor(nxt, hint);
      units.push([E.glide.carrier, E.glide.shorten[f] ?? f]);
      i += 2;
      continue;
    }

    if (isVowel(p)) {
      const hint = hints ? hints[vi] : null;
      vi += 1;
      const split = E.split[p];
      if (split) { putVowel(split[0]); units.push([E.alifu, split[1]]); }
      else putVowel(filiFor(p, hint));
    } else {
      units.push([E.cons[p] ?? "", ""]);
    }
    i += 1;
  }

  // Every vowelless consonant closes with sukun, except the letters the data
  // says stand bare. That is ރ, 1,475 words against 2.
  for (const u of units)
    if (u[1] === "") u[1] = E.coda[u[0]] === "BARE" ? "" : E.sukun;

  return units.filter((u) => u[0]).map(([l, m]) => l + m).join("");
}
