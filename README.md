# en2thaana

**Speak English in a British accent.** Type it, read it back in Thaana, said the
way they say it on the BBC.

English spelling lies. *Clerk* is said "clark". *Derby* is "darby". *Schedule*
starts with a "sh", *been* rhymes with "bin", and nobody warns you. If you read
Thaana, you already have a script that writes every vowel down. This uses it to
show you how the word actually sounds.

```
clerk       ކްލާރކް          schedule    ޝެޖޫލް
derby       ޑާރބީ            privacy     ޕްރައިވަސީ
been        ބިން             tomato      ޓޮމާޓޯ
again       އެގެން            vase        ވާސް
water       ވޯޓަރ            leisure     ލެޜަރ
```

Paste a whole paragraph and it handles the rest: punctuation becomes ، ؛ ؟,
`Dr.` becomes ޑޮކްޓޮރ instead of being read as "drive", `PDF` gets spelled out
as ޕީޑީއެފް, and numbers stay as numbers.

**Try it:** [the page](web/public) runs entirely in your browser. Nothing is
sent anywhere.

## This is not translation

ދަ ކޮމްޕިޔުޓަރ is the English word *computer*, written so you can say it. It is
not Dhivehi. Read it out loud and an English speaker will understand you. Show
it to a Dhivehi speaker as Dhivehi and they will be confused.

Think of the pronunciation key next to a headword in a dictionary. Same idea,
except the key is in a script you can already read.

## Which English?

Received Pronunciation, the accent of BBC newsreaders and English teachers. It
is non-rhotic, so *car* is ކާ and not ކާރ. Maldivian English leans British
anyway, so this should sound closer to what you are used to than an American
guide would.

## How it works

Two steps. First, look the word up in a pronunciation dictionary and get its
sounds. Second, write those sounds in Thaana.

The second step has no model in it and never guesses. It is a set of rules, and
the rules were not invented by me. I took a large body of English words that
Maldivians had already written in Thaana, lined each one up against its
pronunciation, and counted what letter each sound became. Every rule in
`rules/` carries the number of real words behind it, so you can see which ones
rest on a thousand examples and which rest on three.

Some of what fell out:

**English /t/ and /d/ are retroflex.** ޓ and ޑ, not ތ and ދ. The dentals are
reserved for the two *th* sounds, ތ for *think* and ދ for *this*, and the data
splits those 38 out of 38 without a single exception.

**Two dotted letters get in, and no more.** A dot marks an Arabic root, and an
English word has none. The exceptions are ޝ and ޜ, because *ship* and *measure*
have sounds Thaana has no plain letter for.

**ރ stands alone at the end of a syllable.** No sukun on it. 1,475 words write
it bare against 2 that do not.

**Clusters are written out.** *School* is ސްކޫލް. Dhivehi does not normally
allow two consonants to open a word, but these are English words being said in
English, so the ban does not apply.

**The spelling matters, not just the sound.** *Anthology* and *anvil* both have
the same weak vowel in the middle, and they are written differently: ޮ in one
and ި in the other, because one is spelled with an `o` and the other with an
`i`. So the engine gets to see the English spelling too, not only the sounds.

**/j/ is not a consonant here.** In *computer* it becomes a small ި on the ޕ
before it, which is why the word comes out ކޮމްޕިޔުޓަރ.

## How good is it

About half the time it produces exactly the spelling a Maldivian lexicographer
chose, on words it has never seen during fitting.

| split | exact match |
|---|---|
| train | 50.42% |
| dev | 47.73% |
| test | 51.02% |

Half sounds low until you look at what it is being marked against. The
reference disagrees with itself: it keeps the ރ in *assert* and drops it in
*first*, writes the /juː/ in *music* one way and in *computer* another. Where it
splits 124 to 94 on a question, no consistent engine can score better than the
larger half. Every disagreement is written to a file for a person to rule on
rather than quietly overwritten.

The gap between train and test is under a point, so the rules generalise instead
of memorising.

Each fix, measured as it went in:

| change | exact match |
|---|---|
| sounds only, no spelling | 36.35% |
| spelling used for weak vowels | 38.83% |
| ރ restored where the spelling has one | 44.35% |
| silent final `e` ignored | 44.65% |
| dropped the schwa no letter accounts for | 48.59% |
| preferred the hand-checked lexicon for consonants | 49.82% |
| the /juː/ rule | 51.02% |

## Words it has never seen

The dictionary holds 139,219 words. For anything outside it, names mostly, a
small model predicts the pronunciation and the same rules write it in Thaana. It
has 5.3 million parameters and trains in six minutes.

It gets the sounds exactly right 67% of the time, but the Thaana comes out right
**71%** of the time. Thaana has fewer vowels than the phonetic alphabet does, so
a good number of the model's mistakes land on the same letter anyway and stop
being mistakes.

The model runs **in the browser**. It is fetched only when a page actually meets
a word the dictionary lacks, because it is 5.7 MB and most text never needs it.

```
cryptocurrency  ކްރިޕްޓޮކަރަންސީ      blockchain  ބްލޮކްޗޭން
fintech         ފިންޓެކް              nasheed     ނެޝީޑް
```

That means the rules had to be ported to JavaScript, since the model produces
sounds and something has to write them down. A port drifts, so `build_web.py`
runs both engines over all 139,219 words and refuses to ship if one word
disagrees. Today none do.

Maldivian place names skip the model entirely. *Hithadhoo* is ހިތަދޫ because
that is how it is spelled, and no amount of guessing from the letters would land
on it.

## The page

`web/public/` is five files and no server. The rules never change their answer
for a given word, so every answer is worked out in advance and shipped as one
file. Your browser looks words up in it. There is no backend to go down and
nothing to pay for.

```
python build_web.py                     # rebuild the page's data
python -m http.server -d web/public     # or host the folder anywhere
```

`build_web.py` also writes `web/en2thaana-standalone.html`, the whole thing in a
single file you can open by double-clicking it, no internet needed.

## Running it yourself

```
python -m en2thaana schedule clerk derby
python -m en2thaana --explain computer
python -m en2thaana --build dist/en_thaana.jsonl
```

Rebuild the rules and score them, if you have a reference corpus to point
`EN2THAANA_GOLD` at:

```
python eval/induce.py
python eval/evaluate.py test
```

Without one the engine still works. You just cannot re-derive or re-score the
rules.

## Known problems

The /juː/ carrier is a coin flip. The reference writes *music* as މިޔުޒިކް and
*computer* as ކޮމްޕިއުޓަރ, 124 one way and 94 the other, and looking at the
English spelling does not predict which. A Dhivehi speaker picked ޔ and that is
what it does.

The model is quantized to 8 bits to get from 21 MB to 5.7 MB. On a couple of
words in twenty that changes which phoneme wins, so the page can give a slightly
different answer than the full model would. Both are guesses on words no
dictionary has. The float export is checked against PyTorch first, so a real
export bug cannot hide inside that allowance.

Word-initial consonants are the least well tested part, because the reference
corpus is thin there.

See `CREDITS.md` for where the pronunciation data comes from.
