# Credits and licences

## Pronunciation sources

The English → IPA stage is lookup against three pronunciation lexicons. The
dictionary this project ships (`dist/en_thaana.jsonl`, and `web/public/dict.txt`)
is derived from them.

### Britfone — MIT

British English (RP) pronunciation dictionary with stress, by Jose Llarena.
<https://github.com/JoseLlarena/Britfone>

Consulted first. It is human-curated, and where it disagrees with the
espeak-derived lists on a consonant it is right in the great majority of cases:
`/z/` not `/s/` in *absorb*, *acquisition* and *amuse*; `/ð/` not `/θ/` in
*algorithm*; `/ŋ/` before `/k/` in *anchor*. 2,122 entries in the shipped
dictionary.

```
MIT License. Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction...
```

### ipa-dict (`en_UK`) — MIT

Monolingual wordlists with IPA pronunciation, open-dict-data.
<https://github.com/open-dict-data/ipa-dict>

The repository is MIT; note its own caveat that third-party datasets retain
their original licences, and that the `en_UK` list is espeak-derived. 65,107
entries in the shipped dictionary.

### CMUdict — BSD-2-Clause

Carnegie Mellon University Pronouncing Dictionary.
<https://github.com/cmusphinx/cmudict>

General American, converted to RP by a deterministic transform
(`en2thaana/ipa.py`): non-prevocalic /r/ dropped, `/ɑr/→/ɑː/`, `/ɜr/→/ɜː/`,
`/ɔr/→/ɔː/`, LOT `/ɑ/→/ɒ/`. 71,990 entries in the shipped dictionary.

```
Copyright (c) 2015, Alexander Rudnicky. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met...
```

## Thaana orthography

### Hassan Hameed's English–Dhivehi dictionary — evaluation reference only

Dr. Hassan Hameed, 120,000 entries over 1,992 pages, self-funded and
commercially published in print. <https://hassanhameed.com>

Every spelling rule in `rules/*.tsv` is induced from his renderings, and the
accuracy figures in the README are measured against them. **His entries are read
in place from a local copy and are never copied into `data/`, written to
`dist/`, or published** — `.gitignore` enforces this. Rules are not
copyrightable; his entry list is. He has said publicly that a free online
version waits until his costs are recouped, so anyone extending this work should
contact him first. The conventions this engine reproduces are his.

### Dhivehi Bahuge Academy

The orthographic invariants — sukun placement, the prohibition on a bare noonu
(prenasalisation is phonemic), and the rule that a dotted letter marks an Arabic
etymon and so has no place in an English word — come from the Academy's
published rules.

### Gazetteer

240 Maldivian island and atoll names with their registered Thaana spellings,
derived from Maldives registry data.

## Font

**Faruma**, the standard Thaana typeface. Shipped as a 23 KB woff2. Its licence
terms are not distributed with the font file and should be confirmed before any
wide release.

## Method

The rule tables are this project's own work, induced by `eval/induce.py` by
aligning attested spellings to their pronunciations. Every row carries the count
of words supporting it, so no rule rests on an assertion.
