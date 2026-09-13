"""The text layer: running English text -> Thaana transcription.

Stage 2 converts one word. Real text is not a list of words, and the sentence
test showed exactly where that breaks:

    Dr. Hassan Hameed published 120,000 entries in 2019
    -> ޑްރައިވް ...              "Dr." resolved through the lexicon as `drive`

So tokens are classified before anything is looked up, and only the ones that
are actually English words reach the pronunciation lexicon. Everything here is
rules -- there is nothing a model would do better, and a model in this layer
would make a deterministic failure into a mysterious one.

TOKEN CLASSES
-------------
abbreviation  Dr. Mr. St. etc.   expanded via ABBREV, then transcribed
acronym       PDF, NGO, U.S.     spelled letter by letter with Thaana letter names
number        120,000  2019      passed through unchanged; Thaana text uses the
                                 same digits, so converting them would be wrong
word          ordinary English   lexicon, then the rule engine
other         punctuation, space preserved verbatim so the text still reads
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from en2thaana.render import convert
from en2thaana.ipa import lookup_backend as lookup_backend_only
from en2thaana.ipa import lookup_with_g2p as lookup_backend
from en2thaana import gazetteer

# The Thaana spelling of each English letter name, for acronyms read aloud
# letter by letter (PDF -> ޕީ‧ޑީ‧އެފް). These are conventional spellings, not
# engine output: the engine would have to be told that <F> is pronounced "ef".
LETTER_NAMES = {
    "a": "އޭ",   "b": "ބީ",   "c": "ސީ",   "d": "ޑީ",   "e": "އީ",
    "f": "އެފް", "g": "ޖީ",   "h": "އެޗް", "i": "އައި", "j": "ޖޭ",
    "k": "ކޭ",   "l": "އެލް", "m": "އެމް", "n": "އެން", "o": "އޯ",
    "p": "ޕީ",   "q": "ކިއު", "r": "އާރ",  "s": "އެސް", "t": "ޓީ",
    "u": "ޔޫ",   "v": "ވީ",   "w": "ޑަބްލިއު", "x": "އެކްސް",
    "y": "ވައި", "z": "ޒެޑް",
}

# Abbreviations whose spelling collides with a real lexicon entry. `dr` is in
# CMUdict as "drive"; `st` as "street"; `no` as the word "no". Without this the
# lookup silently returns the wrong word rather than failing.
ABBREV = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss",
    "prof": "professor", "st": "saint", "ave": "avenue", "rd": "road",
    "vs": "versus", "etc": "etcetera", "approx": "approximately",
    "dept": "department", "govt": "government", "inc": "incorporated",
    "ltd": "limited", "co": "company", "jan": "january", "feb": "february",
    "mar": "march", "apr": "april", "jun": "june", "jul": "july",
    "aug": "august", "sep": "september", "sept": "september",
    "oct": "october", "nov": "november", "dec": "december",
    "mon": "monday", "tue": "tuesday", "wed": "wednesday",
    "thu": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday",
}

# Punctuation, per the Segha Standard Phonetic layout (jawish/jtk) -- the
# keyboard Dhivehi is actually typed on. Only three marks change: Thaana keeps the Latin full stop, colon and quotation
# mark, and parentheses mirror themselves under the bidi algorithm rather than
# being swapped in the text.
ARABIC_PUNCT = {
    ",": "،",      # ARABIC COMMA
    ";": "؛",      # ARABIC SEMICOLON
    "?": "؟",      # ARABIC QUESTION MARK
}


def arabic_punctuation(s: str) -> str:
    return "".join(ARABIC_PUNCT.get(c, c) for c in s)


# A word, a number, or a run of anything else. The number branch allows , and .
# only BETWEEN digits: a greedy `\d[\d,.]*` swallows the comma in "2019, and"
# into the number token, where the punctuation converter can never see it.
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:[,.]\d+)*|[^A-Za-z\d]+")

_ACRONYM_DOTTED = re.compile(r"^(?:[A-Za-z]\.){2,}$")


@dataclass
class Token:
    text: str
    kind: str          # word | acronym | abbreviation | number | other
    thaana: str | None
    ipa: str | None = None
    source: str = ""


def _acronym(word: str) -> str:
    return "".join(LETTER_NAMES.get(c.lower(), c) for c in word if c.isalpha())


def _is_acronym(raw: str, following: str) -> bool:
    """All-caps runs of 2+ letters are read out letter by letter -- but only
    when they are not simply a capitalised sentence or a shouted word. A word
    that the lexicon knows and that is 4+ letters long is treated as a word."""
    if len(raw) < 2 or not raw.isupper():
        return False
    if len(raw) >= 4 and lookup_backend(raw.lower())[0]:
        return False
    return True


def transcribe_token(raw: str, following: str = "") -> Token:
    """Classify one token and transcribe it."""
    if not raw or not raw[0].isalnum():
        return Token(raw, "other", arabic_punctuation(raw))
    if raw[0].isdigit():
        # Thaana text uses the same digits; converting them would be wrong.
        # The commas inside 120,000 are digit separators, not punctuation, and
        # are left alone because the whole number is one token.
        return Token(raw, "number", raw)

    low = raw.lower()

    # Dr. / Mr. -- only when actually followed by a period, so the surname
    # `Ms` and the word `co` are not mangled.
    if low in ABBREV and following.startswith("."):
        ipa, src = lookup_backend(ABBREV[low])
        if ipa:
            return Token(raw, "abbreviation", convert(ipa, ABBREV[low]), ipa, src)

    if _is_acronym(raw, following):
        return Token(raw, "acronym", _acronym(raw))

    # The English lexicon goes first. The address registry contains 409 of the
    # commonest English words -- fresh, beauty, best, black, camera -- because
    # Maldivians name houses with them, and consulting it first turned every
    # one of those into a house name mid-sentence.
    ipa, src = lookup_backend_only(low)
    if ipa:
        return Token(raw, "word", convert(ipa, low), ipa, src)

    # Not an English word: now a Maldivian place name is the likely reading,
    # and the registry spelling is a fact where the model would only guess.
    gz = gazetteer.lookup(low)
    if gz:
        return Token(raw, "placename", gz, None, "gazetteer")

    guess, gsrc = lookup_backend(low)
    if guess:
        return Token(raw, "word", convert(guess, low), guess, gsrc)
    return Token(raw, "word", None, None, "none")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def transcribe_text(text: str) -> list[Token]:
    """Running English text -> a list of classified, transcribed tokens."""
    raw = tokenize(text)
    out = []
    for i, tok in enumerate(raw):
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        out.append(transcribe_token(tok, nxt))
    return out


def render_text(text: str, unknown: str = "keep") -> str:
    """Transcribed text as a single string.

    unknown='keep'   leave an untranscribable word in Latin (honest, readable)
    unknown='mark'   wrap it in brackets so it is visible in testing
    """
    parts = []
    for t in transcribe_text(text):
        if t.thaana is not None:
            parts.append(t.thaana)
        elif unknown == "mark":
            parts.append("[" + t.text + "]")
        else:
            parts.append(t.text)
    return "".join(parts)


# Dotted acronyms (U.S.A.) arrive as separate tokens, so they are stitched back
# together here rather than in the tokenizer, which would need lookahead.
def normalise_dotted_acronyms(text: str) -> str:
    def repl(m):
        return "".join(c for c in m.group(0) if c.isalpha()).upper()
    return re.sub(r"\b(?:[A-Za-z]\.){2,}", repl, text)
