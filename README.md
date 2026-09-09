# SudoVoynichese

A generator that reproduces the Voynich manuscript's statistical fingerprint in a completely different, invented alphabet.

**What it is.** A corpus-driven generator: it reads the real Voynich transcription, learns each section's word-length distribution, letter transitions, and vocabulary, then produces new text from that model, rendered in Shavian glyphs instead of the transcription's Latin-letter EVA notation.

**What it does.** Zero characters of output overlap with the input's own alphabet. What survives the swap is measured directly: word-length spread, letter-transition entropy, vocabulary growth, local repetition, and how distinct one section's vocabulary is from another's, checked against the real corpus at matched sample sizes rather than against the whole book.

**Why it matters.** If a generator with no notation in common with the original still reproduces its statistical shape, that shape is a property of the underlying structure, not of which glyphs happen to carry it. That is the claim this script exists to check, section by section, run by run.

**How to use it.** Python 3.10+, standard library only. `python3 pseudo_voynich_v3.py LSI_ivtff_0d.txt`

## Quick start

```
python3 pseudo_voynich_v3.py LSI_ivtff_0d.txt
```

Parses the corpus, prints the statistical fingerprint, generates synthetic text for all six manuscript sections (botanical, astronomical, cosmological, pharmaceutical, balneological, recipe), and ends with a verification table comparing the synthetic text back against the real corpus.

Useful flags:

| Flag | Does |
|---|---|
| `--stats-only` | Parse and report the corpus fingerprint, skip generation |
| `--section botanical` | Generate one section only |
| `--lines N --words-per-line N` | How much synthetic text to render |
| `--mode eva\|shavian\|shavian-full` | Output alphabet |
| `--verify-seeds N` | Draws averaged into the verification table (default 5) |
| `--session --herbs N` | Run the pharmaceutical monograph engine on N synthetic herbs |
| `--recipes N` | Generate N synthetic recipes from the real recipe corpus |
| `--list-plants` | Print the built-in plant catalog |
| `--seed N` | Fix the random seed |

Run `python3 pseudo_voynich_v3.py --help` for the rest.

## The corpus

`LSI_ivtff_0d.txt` is Jorge Stolfi's interlinear IVTFF transcription of the manuscript. It is genuinely interlinear: a single physical line can carry several independent readings, one per transcriber, because different people transcribed overlapping stretches and disagree on some characters. The parser resolves this by keeping exactly one reading per physical line (transcriber priority is configurable with `--transcriber-priority`; the default prefers Takahashi's complete transcription), and treats three kinds of in-line ambiguity correctly rather than deleting them: a lone `?` (a character the transcriber could see but not identify) becomes its own token instead of vanishing, a dubious word break `,` splits a word the same as a definite break `.` does instead of injecting a stray glyph, and inline editorial comments are dropped as whole units so a guessed character inside one never leaks into the token stream.

`LSI_ivtff_0d_clean.txt` is a lightly filtered variant of the same transcription. Either can be passed as the corpus argument.

## Verification

The table at the end of a run compares the synthetic text against the real corpus on five corpus-wide metrics (Zipf exponent, bigram entropy, type-token ratio, local repetition rate, spectral gap) and, for every pair of sections, whether the real gap between those two sections' vocabularies is reproduced in the synthetic text. Every real-side value is read at the same sample size as the synthetic side it's compared against — these statistics move with sample size, so comparing a small synthetic draw against the whole book measures the size gap, not the generator. The table is averaged over several independent generations (`--verify-seeds`) so what prints is the generator's expected fidelity, not one draw's luck.

Current state: most rows land at 95-100% across repeated runs. The weakest are the pairs involving the cosmological section, which has only a few hundred words in the whole real corpus — a data-scarcity floor, not a generator defect.

## Files

| File | Role |
|---|---|
| `pseudo_voynich_v3.py` | The generator and CLI |
| `classifier.py`, `tokens.py` | IMASM structural fingerprinting, used to compare section fingerprints |
| `LSI_ivtff_0d.txt`, `LSI_ivtff_0d_clean.txt` | The corpus |
| `voynich_recipe_bio.json` | Recipe corpus backing `--recipes` |

## License

Unlicense. Public domain.
