// The out-of-vocabulary model, in the browser.
//
// Loaded lazily, on the first word the dictionary does not have, because most
// text never needs it and it is 5.7 MB. Once loaded it stays.
//
// It predicts PHONEMES, never Thaana. The rule engine writes those down exactly
// as it writes a dictionary word's, so nothing the model produces bypasses a
// rule. Results are tagged so the page can say which words were guessed.

import { convert } from "./engine.js";

const ORT_VERSION = "1.19.2";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ready = null;      // the in-flight or settled load
let enc = null, dec = null, V = null, stoi = null;

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = res;
    s.onerror = () => rej(new Error("could not load " + src));
    document.head.appendChild(s);
  });
}

async function init() {
  if (!window.ort) {
    await loadScript(ORT_BASE + "ort.min.js");
    // Point the runtime at the same CDN for its wasm, or it looks beside the
    // page and 404s.
    window.ort.env.wasm.wasmPaths = ORT_BASE;
    window.ort.env.wasm.numThreads = 1;   // no cross-origin isolation on Pages
    window.ort.env.logLevel = "error";
  }
  const ort = window.ort;
  V = await (await fetch("model/vocab.json")).json();
  stoi = new Map(V.src.map((c, i) => [c, i]));
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  [enc, dec] = await Promise.all([
    ort.InferenceSession.create("model/encoder.onnx", opts),
    ort.InferenceSession.create("model/decoder.onnx", opts),
  ]);
}

export function load() {
  if (!ready) ready = init().catch((e) => { ready = null; throw e; });
  return ready;
}

export const isLoaded = () => !!enc;

function srcTensor(word) {
  const ids = [...word.toLowerCase()]
    .map((c) => stoi.get(c))
    .filter((v) => v !== undefined)
    .slice(0, V.maxSrc);
  if (!ids.length) return null;
  const buf = new BigInt64Array(V.maxSrc).fill(BigInt(V.pad));
  ids.forEach((v, i) => { buf[i] = BigInt(v); });
  return new window.ort.Tensor("int64", buf, [1, V.maxSrc]);
}

/** English word -> IPA string, or null. */
export async function phonemesFor(word) {
  await load();
  const src = srcTensor(word);
  if (!src) return null;

  const { memory } = await enc.run({ src });
  const tgt = new BigInt64Array(V.maxTgt).fill(BigInt(V.pad));
  tgt[0] = BigInt(V.bos);

  const out = [];
  for (let step = 0; step < V.maxTgt - 1; step++) {
    const tgt_in = new window.ort.Tensor("int64", tgt.slice(), [1, V.maxTgt]);
    const { logits } = await dec.run({ memory, src, tgt_in });
    const width = V.tgt.length;
    const row = logits.data.subarray(step * width, (step + 1) * width);
    let best = 0;
    for (let k = 1; k < width; k++) if (row[k] > row[best]) best = k;
    if (best === V.eos) break;
    out.push(V.tgt[best]);
    tgt[step + 1] = BigInt(best);
  }
  return out.length ? out.join("") : null;
}

/** English word -> Thaana, via the model and then the rules. */
export async function transcribeUnknown(word) {
  const ipa = await phonemesFor(word);
  if (!ipa) return null;
  return { ipa, thaana: convert(ipa, word.toLowerCase()) };
}
