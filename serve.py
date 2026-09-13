"""Local server, for testing from another device on your own network.

    python serve.py                      # listen on 0.0.0.0:8765

It serves the same page as web/public, but transcribes through the Python
engine rather than the precomputed dictionary, so unknown words go to the
out-of-vocabulary model instead of coming back marked. That is the only
difference, and it is why this exists alongside the static build.

Stdlib only, like the rest of the package core -- no Flask, nothing to install,
nothing whose licence has to be tracked for the eventual release.

Bind note: it listens on all interfaces so another device on your network can
reach it. Do not expose this port to the open internet; there is no
authentication here.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from en2thaana.ipa import load_britfone, load_cmudict, load_en_uk  # noqa: E402
from en2thaana.text import normalise_dotted_acronyms, transcribe_text  # noqa: E402

STATIC = ROOT / "web" / "public"
PORT = 8765
MAX_BODY = 256 * 1024

TYPES = {".html": "text/html; charset=utf-8", ".woff2": "font/woff2",
         ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".txt": "text/plain; charset=utf-8"}


def lexicon_size() -> int:
    return len(set(load_en_uk()) | set(load_britfone()) | set(load_cmudict()))


def g2p_status() -> str | None:
    try:
        from en2thaana.g2p import status
        return status()
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "en2thaana"

    def log_message(self, fmt, *args):
        pass        # the default logger writes a line per asset; too noisy

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(),
                              TYPES[".html"])
        if path == "/api/status":
            return self._send(200, json.dumps({
                "lexicon": lexicon_size(), "g2p": g2p_status()}))
        # Any asset the page asks for, served from web/public. Resolve first
        # and confirm the result is still inside it, so a crafted path cannot
        # walk out.
        f = (STATIC / path.lstrip("/")).resolve()
        if f.is_file() and STATIC.resolve() in f.parents:
            return self._send(200, f.read_bytes(),
                              TYPES.get(f.suffix, "application/octet-stream"))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path.split("?")[0] != "/api/transcribe":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._send(413, json.dumps({"error": "too large"}))
        try:
            text = json.loads(self.rfile.read(n) or b"{}").get("text", "")
        except Exception:
            return self._send(400, json.dumps({"error": "bad json"}))

        toks = transcribe_text(normalise_dotted_acronyms(text))
        return self._send(200, json.dumps({"tokens": [
            {"text": t.text, "kind": t.kind, "thaana": t.thaana,
             "ipa": t.ipa, "source": t.source} for t in toks]},
            ensure_ascii=False))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"loading lexicons ...", flush=True)
    n = lexicon_size()
    g = g2p_status()
    print(f"  {n:,} words in the lookup stage")
    print(f"  OOV model: {g or 'not loaded'}")
    print(f"\nlistening on http://0.0.0.0:{port}")
    print(f"publish to your tailnet with:  tailscale serve --bg {port}\n", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
