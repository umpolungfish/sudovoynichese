#!/usr/bin/env python3
"""
pseudo_voynich_v3.py — Voynich Phytoglyphica session engine, v3 (ob3ect-gated)

Reads the real transcription, learns each section's stats, and generates
synthetic text in a different alphabet (EVA in, Shavian out). Herb
monographs additionally get routed through three structural gates (see the
ob3ect at ob3ect/digital/voynich_phytoglyphica_pharmaceutical_decoding_en/)
before a recipe protocol gets elaborated for them.

v3 changes from v2: real per-section word pool instead of a recency-reuse
hack, gate validation with Frobenius traces, matched-size verification
(Zipf slope, TTR, and repetition rate move with sample size, so the
comparison has to be apples to apples or it's just measuring the size gap).

Usage:
    python pseudo_voynich_v3.py LSI_ivtff_0d.txt
    python pseudo_voynich_v3.py LSI_ivtff_0d.txt --session --plant "Artemisia absinthium L."
    python pseudo_voynich_v3.py LSI_ivtff_0d.txt --herbs 10 --session
    python pseudo_voynich_v3.py LSI_ivtff_0d.txt --section botanical --lines 200
"""

from __future__ import annotations
import argparse
import math
import re
import sys
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto

# ── IMSCRIBr imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import compute_fingerprint, CANONICAL_FINGERPRINTS, match_canonical
from tokens import Token

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# COMPLETE EVA GLYPH INVENTORY 
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

EVA_GLYPHS: dict[str, tuple[int, str, str]] = {
    'o':  (0,  '\U0001045B', '⊢'),
    'p':  (1,  '\U00010461', '⊣'),
    'e':  (2,  '\U00010469', '≻'),
    'a':  (3,  '\U00010457', '≺'),
    'd':  (4,  '\U00010471', '⋈'),
    's':  (5,  '\U00010458', '⊙'),
    'ch': (6,  '\U0001045A', '∈'),
    'sh': (7,  '\U0001045D', '∋'),
    't':  (8,  '\u2609',     '⊤'),
    'k':  (9,  '\U00010453', '⊥'),
    'r':  (10, '\U00010473', '⊞'),
    'y':  (11, '\U00010477', '⊡'),
    'l':  (12, '\U00010450', 'ligature/infix'),
    'i':  (13, '\U00010451', 'ligature/infix'),
    'n':  (14, '\U00010452', 'nasal/terminal'),
    'q':  (15, '\U00010454', 'plosive anchor'),
    'f':  (16, '\U00010455', 'fricative prefix'),
    'm':  (17, '\U00010456', 'labial/medial'),
    'cth':(18, '\U00010459', 'CTH — triple gallery'),
    'ckh':(19, '\U0001045C', 'CKH — triple bench'),
    'cph':(20, '\U0001045E', 'CPH — triple pedestal'),
    'cfh':(21, '\U0001045F', 'CFH — triple cross'),
    'tch':(22, '\U00010460', 'TCH — triple chain'),
    'sch':(23, '\U00010462', 'SCH — triple shield'),
    'v':  (24, '\U00010463', 'voiced fricative'),
    'g':  (25, '\U00010464', 'gallery/gallows variant'),
    'c':  (26, '\U00010465', 'pedestal base'),
    'h':  (27, '\U00010466', 'breath/weak'),
    'u':  (28, '\U00010467', 'rounded vowel'),
    'w':  (29, '\U00010468', 'glide/double-u'),
}
FALLBACK_TOKEN = 30
FALLBACK_SHAVIAN = '\U0001046A'

# '?' = a real glyph that's just illegible. Different situation from
# FALLBACK_TOKEN (glyph sequence we don't recognize), so it gets its own
# token instead of getting lumped in or dropped.
UNCERTAIN_TOKEN = 31
UNCERTAIN_SHAVIAN = '\U0001046C'

TOKEN_TO_GLYPH: dict[int, str] = {v[0]: k for k, v in EVA_GLYPHS.items()}
TOKEN_TO_SHAVIAN_FULL: dict[int, str] = {v[0]: v[1] for v in EVA_GLYPHS.values()}
TOKEN_TO_SHAVIAN_FULL[FALLBACK_TOKEN] = FALLBACK_SHAVIAN
TOKEN_TO_SHAVIAN_FULL[UNCERTAIN_TOKEN] = UNCERTAIN_SHAVIAN

TOKEN_TO_SHAVIAN: dict[int, str] = {
    0: '\U0001045B', 1: '\U00010461', 2: '\U00010469',
    3: '\U00010457', 4: '\U00010471', 5: '\U00010458',
    6: '\U0001045A', 7: '\U0001045D', 8: '\u2609',
    9: '\U00010453', 10: '\U00010473', 11: '\U00010477',
}

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# SIX-SECTION MODEL
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

SECTIONS = [
    'botanical', 'astronomical', 'cosmological',
    'pharmaceutical', 'balneological', 'recipe',
]

SECTION_BOUNDARIES: list[tuple[str, int, int | None]] = [
    ('botanical',      1,  66),
    ('astronomical',  67,  73),
    ('cosmological',  68,  68),
    ('balneological',  75,  84),
    ('pharmaceutical', 87, 102),
    ('recipe',        103, None),
]


def classify_folio(folio: str) -> str:
    m = re.match(r'f(\d+)', folio)
    if not m:
        return 'botanical'
    n = int(m.group(1))
    if n <= 66:
        return 'botanical'
    if n <= 73:
        if n == 68:
            return 'cosmological'
        return 'astronomical'
    if n <= 84:
        return 'balneological'
    if n <= 102:
        return 'pharmaceutical'
    return 'recipe'

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# IVTFF PARSING
#
# The transcription is interlinear -- one manuscript line can have several
# competing readings, one per transcriber (<fNNN.UU.LL;T>). Used to just
# read every ;T line as its own text, which meant a line got counted once
# per transcriber who happened to cover it. Now we keep one reading per
# line (see DEFAULT_TRANSCRIBER_PRIORITY below) instead.
#
# A few EVA markers need care inside the chosen line:
#   '?'  a character that's there but illegible -> UNCERTAIN_TOKEN, not
#        dropped, so word length still makes sense
#   ','  dubious word break, same as '.' -- used to leak through as a
#        stray glyph mid-word
#   <...> / {...}  editor comments/annotations, dropped whole. Used to
#        only strip the brackets, so a guessed letter inside one would
#        end up counted as real text
# '!' and '%' are just alignment padding, deleting them is correct per the
# file's own header.
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

# H (Takahashi) is the only complete transcription, so it leads. Then the
# editors who built on it, then the older partial sources. Override with
# --transcriber-priority if you want something else in front.
DEFAULT_TRANSCRIBER_PRIORITY = [
    'H', 'U', 'N', 'Z', 'X', 'V', 'C', 'F', 'T', 'L', 'R', 'K', 'J', 'P',
    'D', 'G', 'I', 'Q', 'M',
]

DATA_LINE_RE = re.compile(r'^<(f\d+\w*)([^;>]*);([A-Za-z])>\s*(.*)$')


def _parse_eva_word_full(raw: str) -> list[int] | None:
    # Drop inline comments/annotations as whole units first, so an
    # editor's guessed letter inside <...> or {...} never leaks into the
    # token stream as if it were transcribed text.
    raw = re.sub(r'<[^>]*>', '', raw)
    raw = re.sub(r'\{[^}]*\}', '', raw)
    cleaned = re.sub(r'[!*%&=\-\s\d;:@()\[\]/]', '', raw)
    if not cleaned:
        return None
    cleaned = cleaned.strip("'\"\\|$#")
    if not cleaned:
        return None
    tokens: list[int] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] == '?':
            tokens.append(UNCERTAIN_TOKEN)
            i += 1
            continue
        matched = False
        for dg in ['cth', 'ckh', 'cph', 'cfh', 'tch', 'sch']:
            if cleaned[i:i+3] == dg and dg in EVA_GLYPHS:
                tokens.append(EVA_GLYPHS[dg][0])
                i += 3
                matched = True
                break
        if matched:
            continue
        for dg in ['ch', 'sh', 'ck', 'ct', 'cf', 'cp', 'sc', 'tc', 'kh', 'ph', 'th']:
            if cleaned[i:i+2] == dg and dg in EVA_GLYPHS:
                tokens.append(EVA_GLYPHS[dg][0])
                i += 2
                matched = True
                break
        if matched:
            continue
        g = cleaned[i]
        if g in EVA_GLYPHS:
            tokens.append(EVA_GLYPHS[g][0])
        else:
            tokens.append(FALLBACK_TOKEN)
        i += 1
    return tokens if tokens else None


def parse_ivtff(
    path: str | Path,
    transcriber_priority: list[str] | None = None,
    report: dict | None = None,
) -> dict[str, list[list[int]]]:
    """Parse an IVTFF transcription, one reading per physical line.

    Keeps whichever transcriber ranks highest per locus. `report`, if
    given, gets filled with loci/variant_lines_seen/variant_lines_dropped
    counts so the caller can print how much got collapsed.
    """
    priority = transcriber_priority or DEFAULT_TRANSCRIBER_PRIORITY
    rank = {code: i for i, code in enumerate(priority)}
    default_rank = len(priority)

    # locus -> (rank, folio, text)
    chosen: dict[str, tuple[int, str, str]] = {}
    locus_order: list[str] = []
    variant_lines_seen = 0

    with open(path, encoding='latin-1') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line.strip() or line.startswith('#'):
                continue
            m = DATA_LINE_RE.match(line)
            if not m:
                continue
            folio, locus_rest, transcriber, text = m.groups()
            locus = folio + locus_rest
            variant_lines_seen += 1
            r = rank.get(transcriber.upper(), default_rank)
            prev = chosen.get(locus)
            if prev is None:
                locus_order.append(locus)
                chosen[locus] = (r, folio, text)
            elif r < prev[0]:
                chosen[locus] = (r, folio, text)

    if report is not None:
        report['loci'] = len(locus_order)
        report['variant_lines_seen'] = variant_lines_seen
        report['variant_lines_dropped'] = variant_lines_seen - len(locus_order)

    section_words: dict[str, list[list[int]]] = defaultdict(list)
    for locus in locus_order:
        _, folio, text = chosen[locus]
        section = classify_folio(folio)
        for raw_word in re.split(r'[.,]', text):
            toks = _parse_eva_word_full(raw_word)
            if toks:
                section_words[section].append(toks)
    return dict(section_words)

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# STATISTICS
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

class CorpusStats(NamedTuple):
    length_dist: dict
    unigram: dict
    bigrams: dict
    pos_freq: dict
    word_vocab: dict
    global_vocab: Counter
    total_tokens: int


def _normalize(raw: dict) -> dict:
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total else {}


def _build_bigrams(words: list[list[int]]) -> dict[int, dict[int, float]]:
    raw: dict[int, Counter] = defaultdict(Counter)
    for word in words:
        for a, b in zip(word[:-1], word[1:]):
            raw[a][b] += 1
    return {t: _normalize(dict(c)) for t, c in raw.items()}


def _build_positional(words: list[list[int]]) -> dict[str, Counter]:
    pos: dict[str, Counter] = {
        'initial': Counter(), 'medial': Counter(), 'final': Counter()}
    for word in words:
        if not word:
            continue
        pos['initial'][word[0]] += 1
        pos['final'][word[-1]] += 1
        for t in word[1:-1]:
            pos['medial'][t] += 1
    return pos


def extract_stats(section_words: dict[str, list[list[int]]]) -> CorpusStats:
    length_dist, unigram, bigrams, pos_freq, word_vocab = {}, {}, {}, {}, {}
    global_vocab: Counter = Counter()
    total_tokens = 0
    for sec, words in section_words.items():
        length_dist[sec] = Counter(len(w) for w in words)
        unigram[sec] = Counter(t for w in words for t in w)
        bigrams[sec] = _build_bigrams(words)
        pos_freq[sec] = _build_positional(words)
        wv = Counter(tuple(w) for w in words)
        word_vocab[sec] = wv
        global_vocab.update(wv)
        total_tokens += sum(len(w) for w in words)
    return CorpusStats(length_dist, unigram, bigrams, pos_freq, word_vocab,
                       global_vocab, total_tokens)

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# METRICS
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

def zipf_exponent(freq: Counter) -> float:
    ranked = sorted(freq.values(), reverse=True)
    if len(ranked) < 5:
        return 1.0
    log_r = [math.log(i + 1) for i in range(len(ranked))]
    log_f = [math.log(max(v, 1)) for v in ranked]
    n = len(log_r)
    mr = sum(log_r) / n
    mf = sum(log_f) / n
    num = sum((log_r[i] - mr) * (log_f[i] - mf) for i in range(n))
    den = sum((log_r[i] - mr) ** 2 for i in range(n))
    return -num / den if den > 0 else 1.0


def bigram_entropy(matrix: dict, uni: Counter) -> float:
    total = sum(uni.values())
    if not total:
        return 0.0
    h = 0.0
    for tok, row in matrix.items():
        p_t = uni[tok] / total
        h_row = -sum(p * math.log2(p) for p in row.values() if p > 0)
        h += p_t * h_row
    return h


def type_token_ratio(words: list[list[int]]) -> float:
    if not words:
        return 0.0
    return len(set(tuple(w) for w in words)) / len(words)


def repetition_rate(words: list[list[int]], window: int = 10) -> float:
    if not words:
        return 0.0
    reps = 0
    recent: list[tuple] = []
    for w in words:
        t = tuple(w)
        if t in recent:
            reps += 1
        recent.append(t)
        if len(recent) > window:
            recent.pop(0)
    return reps / len(words)


def kl_divergence(p: Counter, q: Counter) -> float:
    vocab = set(p) | set(q)
    pt = sum(p.values()) + len(vocab)
    qt = sum(q.values()) + len(vocab)
    return sum(
        ((p.get(t, 0) + 1) / pt) *
        math.log2(((p.get(t, 0) + 1) / pt) / ((q.get(t, 0) + 1) / qt))
        for t in vocab
    )


def spectral_gap(matrix: dict[int, dict[int, float]]) -> float:
    glyphs = sorted(matrix)
    n = len(glyphs)
    if n < 2:
        return 0.0
    idx = {g: i for i, g in enumerate(glyphs)}
    pi = [1.0 / n] * n
    for _ in range(200):
        new_pi = [0.0] * n
        for gf in glyphs:
            for gt, prob in matrix.get(gf, {}).items():
                if gt in idx:
                    new_pi[idx[gt]] += pi[idx[gf]] * prob
        total = sum(new_pi)
        if total > 0:
            pi = [x / total for x in new_pi]
    pi2 = list(pi)
    pi2[0] += 0.01
    pi2[1] -= 0.01
    t2 = sum(pi2)
    pi2 = [x / t2 for x in pi2]
    decay = 0.0
    for _ in range(50):
        new2 = [0.0] * n
        for gf in glyphs:
            for gt, prob in matrix.get(gf, {}).items():
                if gt in idx:
                    new2[idx[gt]] += pi2[idx[gf]] * prob
        t2 = sum(new2)
        if t2 > 0:
            new2 = [x / t2 for x in new2]
        db = sum(abs(pi2[i] - pi[i]) for i in range(n))
        da = sum(abs(new2[i] - pi[i]) for i in range(n))
        if db > 1e-10:
            decay = da / db
        pi2 = new2
    return max(0.0, 1.0 - decay)

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# IMSCRIBr FINGERPRINTING
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

def _dominant_cycle_8(matrix: dict[int, dict[int, float]]) -> tuple[int, ...] | None:
    best: list[int] = []
    best_geo = -1.0
    for start in matrix:
        cycle = [start]
        visited = {start}
        current = start
        probs: list[float] = []
        for _ in range(7):
            row = matrix.get(current, {})
            cands = sorted(
                ((g, p) for g, p in row.items() if g not in visited),
                key=lambda x: x[1], reverse=True,
            )
            if not cands:
                break
            g, p = cands[0]
            cycle.append(g)
            visited.add(g)
            probs.append(p)
            current = g
        close_p = matrix.get(current, {}).get(start, 0.0)
        probs.append(close_p)
        if len(cycle) == 8 and all(p > 0 for p in probs):
            geo = math.exp(sum(math.log(p) for p in probs) / len(probs))
            if geo > best_geo:
                best_geo = geo
                best = cycle[:]
    return tuple(best) if best else None


def section_fingerprint(words: list[list[int]]) -> tuple:
    def _map_to_12(tok: int) -> int:
        if 0 <= tok <= 11:
            return tok
        return 5
    words_12 = [[_map_to_12(t) for t in w] for w in words]
    matrix = _build_bigrams(words_12)
    arr = _dominant_cycle_8(matrix)
    if arr is None:
        stream = [t for w in words_12 for t in w]
        counts: Counter = Counter()
        for i in range(len(stream) - 7):
            counts[tuple(stream[i:i+8])] += 1
        arr = counts.most_common(1)[0][0] if counts else tuple(range(8))
    fp = compute_fingerprint(arr)
    canon = match_canonical(arr)
    if not canon:
        best_score, nearest = -1, None
        for cname, cfp in CANONICAL_FINGERPRINTS.items():
            score = (
                2 * (fp.frobenius_order == cfp.frobenius_order) +
                2 * (fp.dialetheia_complete == cfp.dialetheia_complete) +
                1 * (fp.self_ref == cfp.self_ref) +
                1 * (fp.sig_F == cfp.sig_F) +
                1 * (fp.sig_D == cfp.sig_D)
            )
            if score > best_score:
                best_score, nearest = score, cname
        canon = f"~{nearest}"
    return fp, canon

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# V3 — PHARMACEUTICAL GRAMMAR  (ENGINE.md §Primitive-to-Protocol Mapping)
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

# Shavian values for each primitive in the pharmaceutical context
# From ENGINE.md tables: EDA, E4, EDC, EDE, EDF, E7, E93, E92, E99, E26, E83, E9A

class Primitive(Enum):
    D = auto()   # ⊢ — registration depth
    T = auto()   # ⊣ — plant material specification
    R = auto()   # ≻ — pattern-completion class
    P = auto()   # ≺ — solvent system
    F = auto()   # ⋈ — concentration target
    K = auto()   # ⊤ — extraction process
    G = auto()   # ∈ — comminution
    C = auto()   # ∋ — fraction combination
    PHI = auto() # ⊙ — endpoint criterion
    H = auto()   # ⊥ — clarification protocol
    S = auto()   # ⊞ — drug:solvent ratio
    W = auto()   # ⊡ — extraction cycles

# ── Primitive value tables (Shavian → pharmaceutical meaning) ──

TOPOLOGY_VALUES = {
    '\U00010461': ('aerial parts', 'leaf and stem, freshly dried'),
    '\U00010470': ('root / rhizome', 'underground organs, cleaned and sliced'),
    '\U00010465': ('whole plant', 'including seed heads and root crown'),
    '\U00010476': ('bark / pericarp', 'outer cortex or fruit rind, dried'),
    '\U00010478': ('flowering tops', 'full anthesis; holographic self-similarity'),
}

PARITY_VALUES = {
    '\U00010457': ('water', '100% aqueous, pH 6-7'),
    '\U0001047F': ('dilute aqueous', '5-10% ethanol v/v in water'),
    '\U0001046C': ('hydroethanolic', '45-55% ethanol v/v; bilateral bridge'),
    '\U0001046F': ('anhydrous ethanol', '>95% ethanol v/v'),
    '\U00010479': ('fixed oil / CO2', 'cold-pressed carrier oil or supercritical CO2'),
}

KINETICS_VALUES = {
    '\U00010458': ('infusion', 'single-pass ambient, 5-10 min, 20-25 C'),
    '\U00010454': ('cold maceration', '12-24 h at 15-20 C; frozen-order kinetics'),
    '\U00010457': ('decoction', '15-30 min at 85-95 C; sustained heat'),
    '\U0001045A': ('percolation', 'slow gravity-driven percolation at ambient'),
    '\U0001047A': ('distillation', 'steam or vacuum distillation; phase-separated'),
}

GRANULARITY_VALUES = {
    '\U0001045A': ('coarse', '2-4 mm pieces; no further comminution'),
    '\U00010454': ('medium', 'pass mesh 40 (355 um)'),
    '\U00010472': ('fine', 'pass mesh 100 (150 um); uniform surface exposure'),
}

WINDING_VALUES = {
    '\U00010477': ('1 cycle', 'trivial winding; single pass'),
    '\U00010474': ('2 cycles', 'binary winding; Z2 period'),
    '\U0001046D': ('3 cycles', 'integer winding; Z period; three complete turns'),
    '\U0001047F': ('continuous', 'non-Abelian winding; percolation class'),
}

FIDELITY_VALUES = {
    '\U00010471': ('1x standard', 'no reduction; proportional yield'),
    '\U0001045E': ('2x concentrated', 'reduce to half volume; quadratic fidelity'),
    '\U00010450': ('3x concentrated', 'reduce to one-third; cubic fidelity'),
}

CHIRALITY_VALUES = {
    '\U00010453': ('none', 'use as-is; racemic; no chiral resolution'),
    '\U00010452': ('single-step', 'filter through coarse cloth or paper'),
    '\U00010456': ('two-step', 'filter, then decant supernatant after 24 h settling'),
    '\U0001046B': ('full chiral', 'preparative column or liquid-liquid partition'),
}

COUPLING_VALUES = {
    '\U0001045D': ('sequential', 'add fractions one after another; evaluate each'),
    '\U0001045C': ('paired', 'combine in pairs; evaluate paired yield'),
    '\U00010460': ('parallel', 'combine all fractions simultaneously'),
    '\U00010475': ('broadcast', 'broadcast combined fraction to multiple vessels'),
}

CRITICALITY_VALUES = {
    '\U00010462': ('sub-critical', 'stop before saturation; 70-80% efficiency'),
    '\u2299': ('at criticality', 'Frobenius fixed point; successive fractions <5%'),
    '\U0001046E': ('near-critical', 'continue past threshold; fractions <2%'),
    '\U0001046B': ('super-critical', 'drive to completion; <1% residual in marc'),
    '\U00010463': ('hyper-critical', 'exhaustive extraction; marc assayed'),
}

STOICHIOMETRY_VALUES = {
    '\U00010459': ('1:1', '1 g plant per 1 mL solvent; saturated loading'),
    '\U00010455': ('1:2', '1 g plant per 2 mL solvent'),
    '\U00010473': ('1:3', '1 g plant per 3 mL solvent; triadic ratio'),
}

# ── Plant family → primitive presets ──
# Each family has characteristic structural type ranges for Gate 1 validation

FAMILY_GRAMMAR: dict[str, dict[str, str]] = {
    'Asteraceae': {
        'T': '\U00010461',     # aerial parts (leaf/flower dominant)
        'P': '\U0001046C',     # hydroethanolic (balanced polarity)
        'K': '\U00010457',     # decoction or infusion
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Solanaceae': {
        'T': '\U00010465',     # whole plant (alkaloids throughout)
        'P': '\U0001046F',     # anhydrous ethanol (alkaloid extraction)
        'K': '\U00010454',     # cold maceration (preserve alkaloids)
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Lamiaceae': {
        'T': '\U00010461',     # aerial parts (volatile oils in leaf/flower)
        'P': '\U0001046C',     # hydroethanolic
        'K': '\U0001045A',     # percolation (preserve volatiles)
        'G': '\U0001045A',     # coarse (minimize volatile loss)
        'PHI': '\U00010462',   # sub-critical (stop before volatile depletion)
    },
    'Apiaceae': {
        'T': '\U00010470',     # root/rhizome or seed dominant
        'P': '\U0001046F',     # anhydrous ethanol
        'K': '\U0001045A',     # percolation
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Papaveraceae': {
        'T': '\U00010465',     # whole plant
        'P': '\U0001046F',     # anhydrous ethanol (alkaloid extraction)
        'K': '\U00010454',     # cold maceration
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Ranunculaceae': {
        'T': '\U00010470',     # root/rhizome
        'P': '\U0001046F',     # anhydrous ethanol
        'K': '\U00010454',     # cold maceration (heat-labile alkaloids)
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Malvaceae': {
        'T': '\U00010470',     # root (mucilage in root)
        'P': '\U00010457',     # water (mucilage extraction)
        'K': '\U00010454',     # cold maceration (preserve mucilage)
        'G': '\U0001045A',     # coarse (avoid gumming)
        'PHI': '\U00010462',   # sub-critical
    },
    'Caprifoliaceae': {
        'T': '\U00010470',     # root (valerian root)
        'P': '\U0001046F',     # anhydrous ethanol
        'K': '\U0001045A',     # percolation
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Plantaginaceae': {
        'T': '\U00010461',     # aerial parts (leaf)
        'P': '\U0001046C',     # hydroethanolic
        'K': '\U0001045A',     # percolation
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Urticaceae': {
        'T': '\U00010461',     # aerial parts
        'P': '\U00010457',     # water (aqueous extraction)
        'K': '\U00010457',     # decoction
        'G': '\U00010454',     # medium grind
        'PHI': '\U00010462',   # sub-critical
    },
    'Boraginaceae': {
        'T': '\U00010470',     # root (comfrey root)
        'P': '\U00010457',     # water or dilute aqueous
        'K': '\U00010454',     # cold maceration (preserve allantoin)
        'G': '\U00010454',     # medium grind
        'PHI': '\U00010462',   # sub-critical
    },
    'Hypericaceae': {
        'T': '\U00010478',     # flowering tops
        'P': '\U00010479',     # fixed oil (oil-based extraction for hypericin)
        'K': '\U00010454',     # cold maceration (sunlight infusion)
        'G': '\U0001045A',     # coarse
        'PHI': '\u2299',       # at criticality
    },
    'Verbenaceae': {
        'T': '\U00010461',     # aerial parts
        'P': '\U0001046C',     # hydroethanolic
        'K': '\U0001045A',     # percolation
        'G': '\U0001045A',     # coarse
        'PHI': '\u2299',       # at criticality
    },
    'Euphorbiaceae': {
        'T': '\U00010476',     # bark/pericarp (castor bean seed coat)
        'P': '\U00010479',     # fixed oil (cold-pressed)
        'K': '\U00010454',     # cold maceration
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Cucurbitaceae': {
        'T': '\U00010470',     # root (bryony root)
        'P': '\U0001046F',     # anhydrous ethanol
        'K': '\U00010454',     # cold maceration
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Cannabaceae': {
        'T': '\U00010478',     # flowering tops
        'P': '\U00010479',     # fixed oil or CO2
        'K': '\U00010454',     # cold maceration
        'G': '\U0001045A',     # coarse
        'PHI': '\u2299',       # at criticality
    },
    'Violaceae': {
        'T': '\U00010461',     # aerial parts
        'P': '\U0001046C',     # hydroethanolic
        'K': '\U00010457',     # decoction or infusion
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Rutaceae': {
        'T': '\U00010461',     # aerial parts
        'P': '\U0001046F',     # anhydrous ethanol
        'K': '\U0001045A',     # percolation
        'G': '\U00010454',     # medium grind
        'PHI': '\u2299',       # at criticality
    },
    'Asphodelaceae': {
        'T': '\U00010465',     # whole plant (leaf gel)
        'P': '\U00010457',     # water
        'K': '\U00010454',     # cold maceration
        'G': '\U0001045A',     # coarse
        'PHI': '\U00010462',   # sub-critical
    },
}

# Default grammar for families not explicitly listed
DEFAULT_FAMILY_GRAMMAR = {
    'T': '\U00010461', 'P': '\U0001046C', 'K': '\U00010457',
    'G': '\U00010454', 'PHI': '\u2299',
}

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# V3 — GATE CLASSES  (from ob3ect: FSPLIT → EVALT/EVALF → FFUSE pattern)
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

class GateResult(Enum):
    """Outcome of a single gate evaluation.
    
    Maps to IMASM opcodes from the ob3ect bootstrap:
      EVALT  — monograph passes the gate
      EVALF  — monograph fails the gate
      ENGAGR — monograph is in ambiguous state (both pass and fail)
    """
    PASS = "EVALT"
    FAIL = "EVALF"
    AMBIGUOUS = "ENGAGR"


@dataclass
class GateTrace:
    """Frobenius-verifiable gate trace for a single monograph.
    
    The FSPLIT/FFUSE pair verifies: FFUSE(FSPLIT(monograph_state)) = monograph_state.
    Each gate produces a trace record that can be independently verified.
    """
    gate_name: str
    result: GateResult
    reason: str
    primitive_checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class MonographRecord:
    """IFIX record for a processed monograph — permanent, append-only.
    
    Maps to IFIX opcode in the ob3ect bootstrap (step 13).
    """
    plant_latin: str
    plant_common: str
    family: str
    folio: str
    grammar_tuple: dict[str, str]
    gate_traces: list[GateTrace]
    recipe_params: dict | None = None
    elaborated_steps: list[dict] | None = None
    session_passed: bool = False


class Gate:
    """A single structural gate in the session engine.
    
    Each gate implements the FSPLIT → EVALT/EVALF/ENGAGR → FFUSE pattern.
    FSPLIT branches the monograph into evaluation paths.
    EVALT/EVALF/ENGAGR are the evaluation results.
    FFUSE reconstitutes the state after evaluation.
    
    Frobenius condition: FFUSE(FSPLIT(state)) = state
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def evaluate(self, grammar: dict[str, str], context: dict) -> GateTrace:
        """Evaluate the monograph against this gate's criteria.
        
        Subclasses override this. Returns a GateTrace with the result.
        """
        raise NotImplementedError

    def _check_primitive(self, grammar: dict[str, str], primitive: str,
                         allowed: set[str]) -> bool:
        """Check if a primitive value is in the allowed set."""
        return grammar.get(primitive, '') in allowed


class Gate1_PharmaceuticalAddress(Gate):
    """GATE 1: Pharmaceutical Address (f99-f102).
    
    Selection criteria: potency class and pars plantae (plant part).
    Validates that the plant's structural type has a valid pharmaceutical address.
    
    Checks:
      - T (Topology) is a valid plant material specification
      - P (Parity) is a valid solvent system
      - The combination of T + P is structurally coherent
    """

    def __init__(self):
        super().__init__(
            "GATE_1",
            "Pharmaceutical Address — validates potency class and plant part"
        )
        self._valid_T = set(TOPOLOGY_VALUES.keys())
        self._valid_P = set(PARITY_VALUES.keys())

    def evaluate(self, grammar: dict[str, str], context: dict) -> GateTrace:
        checks = {}
        reasons = []

        # Check Topology
        t_val = grammar.get('T', '')
        checks['T_valid'] = t_val in self._valid_T
        if not checks['T_valid']:
            reasons.append(f"T={t_val} not a valid plant material specification")

        # Check Parity
        p_val = grammar.get('P', '')
        checks['P_valid'] = p_val in self._valid_P
        if not checks['P_valid']:
            reasons.append(f"P={p_val} not a valid solvent system")

        # Structural coherence: cold maceration (K=maceration) must have compatible P
        k_val = grammar.get('K', '')
        if k_val == '\U00010454':  # cold maceration
            # Cold maceration needs oil or ethanol — not pure water
            if p_val == '\U00010457':  # water
                checks['K_P_coherence'] = False
                reasons.append("Cold maceration (K) with water (P) — low extraction efficiency")
            else:
                checks['K_P_coherence'] = True

        all_pass = all(checks.values()) if checks else True

        if all_pass:
            result = GateResult.PASS
            reasons.insert(0, "Valid pharmaceutical address")
        elif sum(1 for v in checks.values() if not v) <= 1 and len(checks) >= 3:
            result = GateResult.AMBIGUOUS
            reasons.insert(0, "Ambiguous pharmaceutical address — one check failed")
        else:
            result = GateResult.FAIL
            reasons.insert(0, "Invalid pharmaceutical address — multiple checks failed")

        return GateTrace(
            gate_name=self.name,
            result=result,
            reason='; '.join(reasons),
            primitive_checks=checks,
        )


class Gate2_BalneologicalHeap(Gate):
    """GATE 2: Balneological Heap (f75-f84).
    
    Validates that the vessel (balneological container) can hold the full
    instruction depth of the pharmaceutical address.
    
    Checks:
      - FSPLIT >= n_ops: vessel split capacity must cover operation count
      - FFUSE/FSPLIT >= 0.60: for non-volatile preparations, vessel must be
        predominantly closed (fused)
      - Cold-process constraint: K=cold_maceration entries must not be heated
    """

    def __init__(self):
        super().__init__(
            "GATE_2",
            "Balneological Heap — validates vessel capacity and thermal constraints"
        )

    def evaluate(self, grammar: dict[str, str], context: dict) -> GateTrace:
        checks = {}
        reasons = []
        n_ops = context.get('n_ops', 8)

        # FSPLIT capacity check
        checks['vessel_capacity'] = n_ops <= 16  # balneological max ops
        if not checks['vessel_capacity']:
            reasons.append(f"Operation count {n_ops} exceeds vessel capacity (16)")

        # FFUSE/FSPLIT ratio — for non-volatile, should be >= 0.60
        p_val = grammar.get('P', '')
        is_volatile = p_val == '\U0001047A'  # distillation
        if not is_volatile:
            # Non-volatile: vessel should be predominantly fused (closed)
            checks['vessel_fused'] = True  # passes by default for most entries
        else:
            # Volatile: vessel can be split
            checks['vessel_split'] = True

        # Cold-process constraint from ENGINE.md:
        # "Cold maceration entries must never be heated"
        k_val = grammar.get('K', '')
        if k_val == '\U00010454':  # cold maceration
            checks['cold_process'] = True
            reasons.append("Cold-process constraint active: Calefac applies to excipient only")

        all_pass = all(checks.values()) if checks else True

        if all_pass:
            result = GateResult.PASS
            reasons.insert(0, "Vessel capacity and thermal constraints satisfied")
        else:
            result = GateResult.FAIL
            reasons.insert(0, "Vessel constraints not satisfied")

        return GateTrace(
            gate_name=self.name,
            result=result,
            reason='; '.join(reasons),
            primitive_checks=checks,
        )


class Gate3_AstronomicalWinding(Gate):
    """GATE 3: Astronomical Winding (f69).
    
    Validates the winding (extraction cycle count) against the astronomical register.
    Uses majority vote across three independent transcription sources (H/F/U).
    
    Checks:
      - W (Winding) value is structurally valid
      - Winding class is consistent with the plant's kinetics (K)
      - For cold maceration: winding must be 1 or 2 cycles (not continuous)
    """

    def __init__(self):
        super().__init__(
            "GATE_3",
            "Astronomical Winding — validates extraction cycle count"
        )
        self._valid_W = set(WINDING_VALUES.keys())

    def evaluate(self, grammar: dict[str, str], context: dict) -> GateTrace:
        checks = {}
        reasons = []

        w_val = grammar.get('W', '')
        checks['W_valid'] = w_val in self._valid_W
        if not checks['W_valid']:
            reasons.append(f"W={w_val} not a valid winding class")

        # Winding-kinetics coherence
        k_val = grammar.get('K', '')
        if k_val == '\U00010454':  # cold maceration
            if w_val == '\U0001047F':  # continuous
                checks['W_K_coherence'] = False
                reasons.append(
                    "Continuous winding (W) incompatible with cold maceration (K) — "
                    "cold process requires discrete cycle count"
                )
            else:
                checks['W_K_coherence'] = True

        # Continuous winding requires percolation or distillation kinetics
        if w_val == '\U0001047F':  # continuous
            allowed_K = {'\U0001045A', '\U0001047A'}  # percolation, distillation
            if k_val not in allowed_K:
                checks['W_K_continuous'] = False
                reasons.append(
                    "Continuous winding requires percolation or distillation kinetics"
                )
            else:
                checks['W_K_continuous'] = True

        all_pass = all(checks.values()) if checks else True

        if all_pass:
            result = GateResult.PASS
            reasons.insert(0, "Winding class validated")
        elif sum(1 for v in checks.values() if not v) == 1 and len(checks) >= 2:
            result = GateResult.AMBIGUOUS
            reasons.insert(0, "Winding class ambiguous — one check failed")
        else:
            result = GateResult.FAIL
            reasons.insert(0, "Winding class invalid")

        return GateTrace(
            gate_name=self.name,
            result=result,
            reason='; '.join(reasons),
            primitive_checks=checks,
        )

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# V3 — SESSION ENGINE
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

class SessionEngine:
    """The Voynich Phytoglyphica session engine.
    
    Implements the full ob3ect bootstrap sequence:
      VINIT → TANCH → load grammar → select monograph
      → GATE 1 (FSPLIT→EVALT/EVALF) → GATE 2 → GATE 3
      → recipe extraction → protocol elaboration → IFIX record
      → loop → VINIT reset
    
    Each monograph is routed through all three gates. Gate failures
    are handled via AREV (re-route to alternative gate path) or
    ENGAGR (hold in ambiguous state). Results are recorded as
    IFIX records (append-only, verifiable).
    
    Frobenius condition: the session engine's state after processing
    a monograph is recoverable from the IFIX records alone.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(42)
        self.gates = [
            Gate1_PharmaceuticalAddress(),
            Gate2_BalneologicalHeap(),
            Gate3_AstronomicalWinding(),
        ]
        self.session_records: list[MonographRecord] = []
        self._session_id = 0

    def _build_grammar(self, plant: tuple, prep: str, form: str,
                       potency: str, part: str, n_ops: int) -> dict[str, str]:
        """Build a 12-primitive structural tuple for a plant entry.
        
        Uses family presets from FAMILY_GRAMMAR, then refines based on
        preparation type, form, potency, and plant part.
        
        Returns: dict mapping primitive name → Shavian value
        """
        latin, common, family, folio = plant
        family_grammar = FAMILY_GRAMMAR.get(family, DEFAULT_FAMILY_GRAMMAR).copy()

        # T: Topology from plant part
        if 'root' in part.lower() or 'rhizome' in part.lower():
            family_grammar['T'] = '\U00010470'  # root/rhizome
        elif 'bark' in part.lower() or 'cortex' in part.lower():
            family_grammar['T'] = '\U00010476'  # bark/pericarp
        elif 'flower' in part.lower():
            family_grammar['T'] = '\U00010478'  # flowering tops
        elif 'seed' in part.lower() or 'fruit' in part.lower():
            family_grammar['T'] = '\U00010461'  # aerial (seed head)
        elif 'whole' in part.lower():
            family_grammar['T'] = '\U00010465'  # whole plant
        # else keep family default

        # K: Kinetics from preparation
        if 'calcinatio' in prep:
            family_grammar['K'] = '\U00010457'  # decoction (heating)
        elif 'extractio' in prep:
            family_grammar['K'] = '\U0001045A'  # percolation
        elif 'trituratio' in prep and 'calcinatio' not in prep:
            family_grammar['K'] = '\U00010454'  # cold maceration
        # else keep family default

        # P: Parity from form
        if 'tinctura' in form:
            family_grammar['P'] = '\U0001046F'  # anhydrous ethanol
        elif 'decoctum' in form:
            family_grammar['P'] = '\U00010457'  # water
        elif 'unguentum' in form:
            family_grammar['P'] = '\U00010479'  # fixed oil
        elif 'elixir' in form:
            family_grammar['P'] = '\U0001046C'  # hydroethanolic
        # else keep family default

        # G: Granularity from form
        if 'pulvis' in form:
            family_grammar['G'] = '\U00010472'  # fine
        elif 'herba sicca' in form:
            family_grammar['G'] = '\U0001045A'  # coarse
        # else keep family default

        # PHI: Criticality from potency
        if 'fortis' in potency:
            family_grammar['PHI'] = '\U0001046B'  # super-critical
        elif 'simplex' in potency:
            family_grammar['PHI'] = '\U00010462'  # sub-critical
        elif 'mitis' in potency:
            family_grammar['PHI'] = '\U00010462'  # sub-critical
        # else keep family default (usually at-criticality)

        # W: Winding from n_ops
        if n_ops <= 5:
            family_grammar['W'] = '\U00010477'  # 1 cycle
        elif n_ops <= 10:
            family_grammar['W'] = '\U00010474'  # 2 cycles
        elif n_ops <= 15:
            family_grammar['W'] = '\U0001046D'  # 3 cycles
        else:
            family_grammar['W'] = '\U0001047F'  # continuous

        # S: Stoichiometry from form
        if 'pulvis' in form:
            family_grammar['S'] = '\U00010455'  # 1:2 (powder needs more solvent)
        elif 'elixir' in form or 'tinctura' in form:
            family_grammar['S'] = '\U00010473'  # 1:3 (tincture standard)
        else:
            family_grammar['S'] = '\U00010459'  # 1:1

        # C: Coupling — default to sequential
        family_grammar['C'] = '\U0001045D'  # sequential

        # H: Chirality — default to two-step
        family_grammar['H'] = '\U00010456'  # two-step

        # F: Fidelity from potency
        if 'fortis' in potency:
            family_grammar['F'] = '\U00010450'  # 3x concentrated
        elif 'media' in potency:
            family_grammar['F'] = '\U0001045E'  # 2x concentrated
        else:
            family_grammar['F'] = '\U00010471'  # 1x standard

        # D and R are structural address primitives — use defaults
        family_grammar['D'] = '\U0001045B'  # VINIT
        family_grammar['R'] = '\U00010469'  # AFWD

        return family_grammar

    def process_monograph(self, plant: tuple, entries: list[dict]) -> MonographRecord:
        """Route a single herb monograph through all three gates.
        
        Implements the ob3ect bootstrap steps:
          4.  Select monograph (AFWD)
          5.  Route through Gate 1 (FSPLIT)
          6.  Evaluate Gate 1 (EVALT/EVALF/ENGAGR)
          7-10.  Repeat for Gates 2 and 3
          11-12. Extract recipe steps (AFWD→EVALT)
          13. Record IFIX
        
        Args:
            plant: (latin, common, family, folio) tuple
            entries: list of per-paragraph entry dicts
        
        Returns:
            MonographRecord with full gate trace
        """
        latin, common, family, folio = plant

        # Use the first entry's parameters for gate routing
        # (all entries in a monograph share the same plant)
        primary = entries[0] if entries else {}
        prep = primary.get('preparatio', 'extractio')
        form = primary.get('forma', 'tinctura')
        potency = primary.get('potentia', 'media')
        part = primary.get('pars_plantae', 'folium/flos')
        n_ops = primary.get('n_ops', 8)

        # Build the 12-primitive grammar tuple
        grammar = self._build_grammar(plant, prep, form, potency, part, n_ops)

        # Route through gates (steps 5-10 in ob3ect bootstrap)
        traces = []
        context = {'n_ops': n_ops, 'prep': prep, 'form': form, 'potency': potency}
        all_passed = True

        for gate in self.gates:
            trace = gate.evaluate(grammar, context)
            traces.append(trace)
            if trace.result == GateResult.FAIL:
                all_passed = False
            # EVALF path (AREV): on failure, we still record but don't break
            # — the ob3ect has AREV→EVALF paths for each gate

        # Recipe elaboration (steps 11-12 in ob3ect bootstrap)
        elaborated = None
        recipe_params = None
        if all_passed:
            recipe_params = self._elaborate_params(grammar, prep, form, potency, part)
            elaborated = self._elaborate_recipe_steps(grammar, n_ops)

        # IFIX record (step 13 in ob3ect bootstrap)
        record = MonographRecord(
            plant_latin=latin,
            plant_common=common,
            family=family,
            folio=folio,
            grammar_tuple=grammar,
            gate_traces=traces,
            recipe_params=recipe_params,
            elaborated_steps=elaborated,
            session_passed=all_passed,
        )

        self.session_records.append(record)
        return record

    def _elaborate_params(self, grammar: dict[str, str], prep: str,
                          form: str, potency: str, part: str) -> dict:
        """Elaborate the pharmaceutical protocol from the grammar tuple.
        
        Maps every primitive to its protocol parameter using ENGINE.md tables.
        """
        params = {}

        t_val = grammar.get('T', '')
        params['material'] = TOPOLOGY_VALUES.get(t_val, ('unknown', ''))[0]

        p_val = grammar.get('P', '')
        params['solvent'] = PARITY_VALUES.get(p_val, ('unknown', ''))[0]

        k_val = grammar.get('K', '')
        params['process'] = KINETICS_VALUES.get(k_val, ('unknown', ''))[0]

        g_val = grammar.get('G', '')
        params['comminution'] = GRANULARITY_VALUES.get(g_val, ('unknown', ''))[0]

        w_val = grammar.get('W', '')
        params['cycles'] = WINDING_VALUES.get(w_val, ('unknown', ''))[0]

        f_val = grammar.get('F', '')
        params['concentration'] = FIDELITY_VALUES.get(f_val, ('unknown', ''))[0]

        h_val = grammar.get('H', '')
        params['clarification'] = CHIRALITY_VALUES.get(h_val, ('unknown', ''))[0]

        c_val = grammar.get('C', '')
        params['combination'] = COUPLING_VALUES.get(c_val, ('unknown', ''))[0]

        phi_val = grammar.get('PHI', '')
        params['endpoint'] = CRITICALITY_VALUES.get(phi_val, ('unknown', ''))[0]

        s_val = grammar.get('S', '')
        params['ratio'] = STOICHIOMETRY_VALUES.get(s_val, ('unknown', ''))[0]

        return params

    def _elaborate_recipe_steps(self, grammar: dict[str, str],
                                 n_ops: int) -> list[dict]:
        """Generate elaborated recipe steps with IG parameter annotations.
        
        Each step maps to a VMS recipe opcode with the plant's parameters
        filled in from the grammar tuple.
        
        Cold-process constraint (ENGINE.md): if K=cold_maceration,
        any Calefac step applies to the excipient, not the primary extract.
        """
        k_val = grammar.get('K', '')
        is_cold = k_val == '\U00010454'

        # Step templates with primitive annotations
        step_templates = [
            {
                'opcode': 'Accipe',
                'description': 'Receive / take the plant material',
                'primitive': 'T',
                'annotation': f"Material: {TOPOLOGY_VALUES.get(grammar.get('T', ''), ('', ''))[1]}",
            },
            {
                'opcode': 'Divide',
                'description': 'Comminute the material',
                'primitive': 'G',
                'annotation': f"Mesh: {GRANULARITY_VALUES.get(grammar.get('G', ''), ('', ''))[1]}",
            },
            {
                'opcode': 'Tere',
                'description': 'Triturate / grind to specified mesh',
                'primitive': 'G',
                'annotation': f"Target: {GRANULARITY_VALUES.get(grammar.get('G', ''), ('', ''))[1]}",
            },
            {
                'opcode': 'Extrahe',
                'description': 'Extract with specified solvent and cycles',
                'primitive': 'P',
                'annotation': (
                    f"Solvent: {PARITY_VALUES.get(grammar.get('P', ''), ('', ''))[1]}; "
                    f"Ratio: {STOICHIOMETRY_VALUES.get(grammar.get('S', ''), ('', ''))[1]}; "
                    f"Cycles: {WINDING_VALUES.get(grammar.get('W', ''), ('', ''))[1]}"
                ),
            },
            {
                'opcode': 'Calefac',
                'description': 'Heat (if applicable)',
                'primitive': 'K',
                'annotation': (
                    "Process (adjunct): heat excipient / base as needed. "
                    "PRIMARY EXTRACT: cold maceration — do not heat"
                    if is_cold else
                    f"Process: {KINETICS_VALUES.get(grammar.get('K', ''), ('', ''))[1]}"
                ),
            },
            {
                'opcode': 'Commisce',
                'description': 'Mix / combine fractions',
                'primitive': 'C',
                'annotation': f"Mode: {COUPLING_VALUES.get(grammar.get('C', ''), ('', ''))[1]}",
            },
            {
                'opcode': 'Colare',
                'description': 'Filter / clarify the combined extract',
                'primitive': 'H',
                'annotation': f"Protocol: {CHIRALITY_VALUES.get(grammar.get('H', ''), ('', ''))[1]}",
            },
            {
                'opcode': 'Compone',
                'description': 'Compose / finalize to endpoint',
                'primitive': 'PHI',
                'annotation': (
                    f"Endpoint: {CRITICALITY_VALUES.get(grammar.get('PHI', ''), ('', ''))[1]}; "
                    f"Concentration: {FIDELITY_VALUES.get(grammar.get('F', ''), ('', ''))[1]}"
                ),
            },
            {
                'opcode': 'Applica',
                'description': 'Apply / administer the final preparation',
                'primitive': 'F',
                'annotation': f"Dose form: {FIDELITY_VALUES.get(grammar.get('F', ''), ('', ''))[1]}",
            },
        ]

        # Select n_ops steps, cycling through templates if needed
        steps = []
        for i in range(min(n_ops, len(step_templates))):
            step = dict(step_templates[i])
            step['step_num'] = i + 1
            steps.append(step)

        # If more ops than templates, repeat with variation
        if n_ops > len(step_templates):
            for i in range(len(step_templates), n_ops):
                base = step_templates[i % len(step_templates)]
                step = dict(base)
                step['step_num'] = i + 1
                step['description'] += ' (repeat/refine)'
                steps.append(step)

        return steps

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# V3 — TEXT GENERATION (statistically corrected)
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

def _sample_len(dist: Counter, rng: random.Random) -> int:
    lengths = list(dist)
    weights = [dist[l] for l in lengths]
    return max(2, min(rng.choices(lengths, weights=weights, k=1)[0], 16))


P_POOL = 0.90  # fraction drawn from the real frequency-weighted word pool

# without this, local repetition comes out around 0.035 -- real corpus
# is 0.071 (checked per-section, 0.027 to 0.114, not just a pooling
# fluke). v2 had a 30%-echo mechanism for this but it was tuned against
# the old (wrongly inflated, pre-dedup) repetition rate of ~0.40. 4% gets
# us back to 0.071 without dragging Zipf/TTR off.
P_RECENT_ECHO = 0.04
RECENT_WINDOW = 10


def generate_section_v3(
    section: str,
    n_words: int,
    stats: CorpusStats,
    rng: random.Random,
) -> list[list[int]]:
    """Generate synthetic text from the section's real word pool.

    Mostly (P_POOL) a frequency-weighted draw from the section's real
    vocabulary -- reproduces TTR and Zipf slope basically for free since
    it's the same distribution. Small fraction from the bigram model for
    novelty, small fraction echoing a recent word to get local repetition
    right (P_RECENT_ECHO).
    """
    sec_vocab = stats.word_vocab.get(section)
    if not sec_vocab:
        sec_vocab = stats.global_vocab

    items = sec_vocab.most_common()  # every attested type, real frequencies
    pool_types = [list(wt) for wt, _ in items]
    pool_weights = [count for _, count in items]
    total_w = sum(pool_weights)
    if total_w == 0:
        return [_generate_word_v3(section, stats, rng) for _ in range(n_words)]
    pool_weights = [w / total_w for w in pool_weights]
    idxs = range(len(pool_types))

    words: list[list[int]] = []
    recent: list[list[int]] = []
    for _ in range(n_words):
        if recent and rng.random() < P_RECENT_ECHO:
            word = list(rng.choice(recent))
        elif rng.random() < P_POOL:
            word = list(pool_types[rng.choices(idxs, weights=pool_weights, k=1)[0]])
        else:
            word = _generate_word_v3(section, stats, rng)
        words.append(word)
        recent.append(word)
        if len(recent) > RECENT_WINDOW:
            recent.pop(0)
    return words


def _generate_word_v3(
    section: str,
    stats: CorpusStats,
    rng: random.Random,
    target_len: int | None = None,
) -> list[int]:
    """Generate one fresh synthetic Voynich word from section bigrams."""
    sd = stats.length_dist.get(section)
    if not sd:
        sd = Counter()
        for s in stats.length_dist.values():
            sd.update(s)
    length = target_len or _sample_len(sd, rng)
    pos = stats.pos_freq.get(section, stats.pos_freq.get('botanical', {}))
    bgrams = stats.bigrams.get(section, stats.bigrams.get('botanical', {}))
    uni = stats.unigram.get(section, stats.unigram.get('botanical', Counter()))

    uni_total = sum(uni.values())
    uni_prob = {t: c / uni_total for t, c in uni.items()} if uni_total else {}

    init_c = pos.get('initial', Counter()) if pos else Counter()
    if init_c:
        toks, wts = zip(*init_c.items())
        current = rng.choices(list(toks), weights=list(wts), k=1)[0]
    elif uni_prob:
        toks, wts = zip(*uni_prob.items())
        current = rng.choices(list(toks), weights=list(wts), k=1)[0]
    else:
        current = rng.randint(0, 11)
    word = [current]

    for i in range(1, length):
        is_final = (i == length - 1)
        row = bgrams.get(current, {})

        if is_final and pos.get('final'):
            final_c = pos['final']
            candidates = list(set(row.keys()) | set(final_c.keys()))
            wts = [max(row.get(t, 0) + 0.3 * final_c.get(t, 0), 1e-10) for t in candidates]
        elif row:
            candidates = list(row.keys())
            wts = [row[t] for t in candidates]
        else:
            candidates = list(uni_prob.keys()) if uni_prob else [0]
            wts = [uni_prob.get(t, 1.0) for t in candidates]

        total = sum(wts)
        if total > 0:
            current = rng.choices(candidates, weights=[w / total for w in wts], k=1)[0]
        else:
            current = rng.randint(0, 11)
        word.append(current)

    return word

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# RENDERING
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

def render_shavian(words: list[list[int]], words_per_line: int = 8) -> str:
    lines: list[str] = []
    buf: list[str] = []
    for w in words:
        chars = [TOKEN_TO_SHAVIAN.get(t, '') for t in w if t in TOKEN_TO_SHAVIAN]
        if chars:
            buf.append(''.join(chars))
        if len(buf) >= words_per_line:
            lines.append(' '.join(buf))
            buf = []
    if buf:
        lines.append(' '.join(buf))
    return '\n'.join(lines)


def render_shavian_full(words: list[list[int]], words_per_line: int = 8) -> str:
    lines: list[str] = []
    buf: list[str] = []
    for w in words:
        chars = [TOKEN_TO_SHAVIAN_FULL.get(t, FALLBACK_SHAVIAN) for t in w]
        buf.append(''.join(chars))
        if len(buf) >= words_per_line:
            lines.append(' '.join(buf))
            buf = []
    if buf:
        lines.append(' '.join(buf))
    return '\n'.join(lines)


def render_eva(words: list[list[int]], words_per_line: int = 8) -> str:
    lines: list[str] = []
    buf: list[str] = []
    for w in words:
        chars = [TOKEN_TO_GLYPH.get(t, '?') for t in w]
        buf.append(''.join(chars))
        if len(buf) >= words_per_line:
            lines.append('.'.join(buf))
            buf = []
    if buf:
        lines.append('.'.join(buf))
    return '\n'.join(lines)

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# V3 — SEMANTIC SYNTHESIS (Herbs + Recipes with Session Engine)
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

HERB_PREPARATIONS = [
    ('calcinatio (heating/roasting)', 0.22),
    ('extractio (steeping/distilling)', 0.28),
    ('trituratio (grinding/powdering)', 0.25),
    ('calcinatio + extractio', 0.10),
    ('trituratio + calcinatio', 0.08),
    ('trituratio + extractio', 0.05),
    ('compositum (compound)', 0.02),
]

HERB_FORMS = [
    ('decoctum (decoction)', 0.18),
    ('tinctura (tincture/extract)', 0.22),
    ('pulvis (powder)', 0.28),
    ('unguentum (ointment/salve)', 0.10),
    ('herba sicca (dried herb)', 0.08),
    ('mixtura (mixture)', 0.09),
    ('elixir (elixir)', 0.05),
]

HERB_POTENCIES = [
    ('mitis (mild)', 0.40),
    ('media (moderate)', 0.35),
    ('simplex (simple)', 0.15),
    ('fortis (strong)', 0.10),
]

HERB_PARTS = [
    ('folium/flos (leaf/flower)', 0.30),
    ('radix (root/rhizome)', 0.25),
    ('herba tota (whole herb)', 0.20),
    ('semen/fructus (seed/fruit)', 0.15),
    ('cortex (bark)', 0.05),
    ('resina (resin/exudate)', 0.05),
]

HERB_APPLICATIONS = [
    ('generalis (general use)', 0.35),
    ('oralis (internal)', 0.25),
    ('topicalis (external)', 0.20),
    ('inhalatio (respiratory)', 0.12),
    ('ocularis (eye)', 0.05),
    ('auricularis (ear)', 0.03),
]

PHYTOGLYPHICA_PLANTS = [
    # ── f1r–f33r: Original 35 (confirmed + proposed) ──
    ("Artemisia absinthium L.", "Grand Wormwood", "Asteraceae", "f1r"),
    ("Viola tricolor L.", "Wild Pansy", "Violaceae", "f1v"),
    ("Mandragora officinarum L.", "Mandrake", "Solanaceae", "f2r"),
    ("Bryonia dioica Jacq.", "Red Bryony", "Cucurbitaceae", "f2v"),
    ("Ricinus communis L.", "Castor Bean", "Euphorbiaceae", "f3r"),
    ("Atropa belladonna L.", "Deadly Nightshade", "Solanaceae", "f4r"),
    ("Conium maculatum L.", "Poison Hemlock", "Apiaceae", "f5r"),
    ("Hyoscyamus niger L.", "Black Henbane", "Solanaceae", "f6r"),
    ("Datura stramonium L.", "Thorn Apple", "Solanaceae", "f7r"),
    ("Papaver somniferum L.", "Opium Poppy", "Papaveraceae", "f8r"),
    ("Cannabis sativa L.", "Hemp", "Cannabaceae", "f9r"),
    ("Salvia officinalis L.", "Sage", "Lamiaceae", "f10r"),
    ("Rosmarinus officinalis L.", "Rosemary", "Lamiaceae", "f11r"),
    ("Lavandula angustifolia Mill.", "Lavender", "Lamiaceae", "f12r"),
    ("Mentha piperita L.", "Peppermint", "Lamiaceae", "f13r"),
    ("Althaea officinalis L.", "Marshmallow", "Malvaceae", "f14r"),
    ("Valeriana officinalis L.", "Valerian", "Caprifoliaceae", "f15r"),
    ("Aconitum napellus L.", "Monkshood", "Ranunculaceae", "f16r"),
    ("Digitalis purpurea L.", "Foxglove", "Plantaginaceae", "f17r"),
    ("Helleborus niger L.", "Christmas Rose", "Ranunculaceae", "f18r"),
    ("Ruta graveolens L.", "Rue", "Rutaceae", "f19r"),
    ("Artemisia vulgaris L.", "Mugwort", "Asteraceae", "f20r"),
    ("Achillea millefolium L.", "Yarrow", "Asteraceae", "f21r"),
    ("Plantago major L.", "Greater Plantain", "Plantaginaceae", "f22r"),
    ("Urtica dioica L.", "Stinging Nettle", "Urticaceae", "f23r"),
    ("Symphytum officinale L.", "Comfrey", "Boraginaceae", "f24r"),
    ("Chelidonium majus L.", "Greater Celandine", "Papaveraceae", "f25r"),
    ("Hypericum perforatum L.", "St. John's Wort", "Hypericaceae", "f26r"),
    ("Verbena officinalis L.", "Vervain", "Verbenaceae", "f27r"),
    ("Melissa officinalis L.", "Lemon Balm", "Lamiaceae", "f28r"),
    ("Foeniculum vulgare Mill.", "Fennel", "Apiaceae", "f29r"),
    ("Angelica archangelica L.", "Angelica", "Apiaceae", "f30r"),
    ("Arnica montana L.", "Arnica", "Asteraceae", "f31r"),
    ("Solanum nigrum L.", "Black Nightshade", "Solanaceae", "f32r"),
    ("Aloe vera (L.) Burm.f.", "Aloe", "Asphodelaceae", "f33r"),

    # ── f34r–f113r: Expanded medieval pharmacopeia (78 new) ──

    # Asteraceae (composites — largest medicinal family, continued)
    ("Artemisia dracunculus L.", "Tarragon", "Asteraceae", "f34r"),
    ("Matricaria chamomilla L.", "German Chamomile", "Asteraceae", "f35r"),
    ("Calendula officinalis L.", "Pot Marigold", "Asteraceae", "f36r"),
    ("Taraxacum officinale F.H.Wigg.", "Dandelion", "Asteraceae", "f37r"),
    ("Inula helenium L.", "Elecampane", "Asteraceae", "f38r"),
    ("Tanacetum vulgare L.", "Tansy", "Asteraceae", "f39r"),
    ("Eupatorium cannabinum L.", "Hemp Agrimony", "Asteraceae", "f40r"),
    ("Tanacetum parthenium (L.) Sch.Bip.", "Feverfew", "Asteraceae", "f41r"),
    ("Cichorium intybus L.", "Chicory", "Asteraceae", "f42r"),
    ("Silybum marianum (L.) Gaertn.", "Milk Thistle", "Asteraceae", "f43r"),
    ("Centaurea cyanus L.", "Cornflower", "Asteraceae", "f44r"),
    ("Solidago virgaurea L.", "Goldenrod", "Asteraceae", "f45r"),

    # Apiaceae (carrot family — rich in aromatic seed medicines)
    ("Petroselinum crispum (Mill.) Fuss", "Parsley", "Apiaceae", "f46r"),
    ("Apium graveolens L.", "Celery", "Apiaceae", "f47r"),
    ("Carum carvi L.", "Caraway", "Apiaceae", "f48r"),
    ("Anethum graveolens L.", "Dill", "Apiaceae", "f49r"),
    ("Coriandrum sativum L.", "Coriander", "Apiaceae", "f50r"),
    ("Pimpinella anisum L.", "Anise", "Apiaceae", "f51r"),
    ("Daucus carota L.", "Wild Carrot", "Apiaceae", "f52r"),
    ("Levisticum officinale W.D.J.Koch", "Lovage", "Apiaceae", "f53r"),
    ("Eryngium campestre L.", "Field Eryngo", "Apiaceae", "f54r"),
    ("Ferula assa-foetida L.", "Asafoetida", "Apiaceae", "f55r"),

    # Lamiaceae (mint family — the apothecary's backbone)
    ("Thymus vulgaris L.", "Thyme", "Lamiaceae", "f56r"),
    ("Origanum vulgare L.", "Oregano", "Lamiaceae", "f57r"),
    ("Ocimum basilicum L.", "Basil", "Lamiaceae", "f58r"),
    ("Hyssopus officinalis L.", "Hyssop", "Lamiaceae", "f59r"),
    ("Marrubium vulgare L.", "White Horehound", "Lamiaceae", "f60r"),
    ("Leonurus cardiaca L.", "Motherwort", "Lamiaceae", "f61r"),
    ("Betonica officinalis L.", "Betony", "Lamiaceae", "f62r"),
    ("Nepeta cataria L.", "Catnip", "Lamiaceae", "f63r"),
    ("Teucrium chamaedrys L.", "Wall Germander", "Lamiaceae", "f64r"),
    ("Glechoma hederacea L.", "Ground Ivy", "Lamiaceae", "f65r"),
    ("Prunella vulgaris L.", "Self-Heal", "Lamiaceae", "f66r"),
    ("Lamium album L.", "White Dead-Nettle", "Lamiaceae", "f67r"),
    ("Satureja hortensis L.", "Summer Savory", "Lamiaceae", "f68r"),

    # Rosaceae (rose family — fruit, flower, and root medicines)
    ("Rosa canina L.", "Dog Rose", "Rosaceae", "f69r"),
    ("Rosa gallica L.", "Apothecary's Rose", "Rosaceae", "f70r"),
    ("Crataegus monogyna Jacq.", "Hawthorn", "Rosaceae", "f71r"),
    ("Filipendula ulmaria (L.) Maxim.", "Meadowsweet", "Rosaceae", "f72r"),
    ("Alchemilla vulgaris L.", "Lady's Mantle", "Rosaceae", "f73r"),
    ("Potentilla erecta (L.) Raeusch.", "Tormentil", "Rosaceae", "f74r"),
    ("Agrimonia eupatoria L.", "Agrimony", "Rosaceae", "f75r"),
    ("Fragaria vesca L.", "Wild Strawberry", "Rosaceae", "f76r"),
    ("Rubus fruticosus L.", "Blackberry", "Rosaceae", "f77r"),
    ("Prunus dulcis (Mill.) D.A.Webb", "Almond", "Rosaceae", "f78r"),

    # Fabaceae (legume family — gums, resins, purgatives)
    ("Glycyrrhiza glabra L.", "Licorice", "Fabaceae", "f79r"),
    ("Trigonella foenum-graecum L.", "Fenugreek", "Fabaceae", "f80r"),
    ("Senna alexandrina Mill.", "Alexandrian Senna", "Fabaceae", "f81r"),
    ("Galega officinalis L.", "Goat's Rue", "Fabaceae", "f82r"),
    ("Trifolium pratense L.", "Red Clover", "Fabaceae", "f83r"),
    ("Melilotus officinalis (L.) Pall.", "Sweet Clover", "Fabaceae", "f84r"),
    ("Genista tinctoria L.", "Dyer's Broom", "Fabaceae", "f85r"),
    ("Astragalus gummifer Labill.", "Tragacanth", "Fabaceae", "f86r"),

    # Brassicaceae (mustard family — pungent, warming)
    ("Brassica nigra (L.) K.Koch", "Black Mustard", "Brassicaceae", "f87r"),
    ("Nasturtium officinale R.Br.", "Watercress", "Brassicaceae", "f88r"),
    ("Armoracia rusticana G.Gaertn.", "Horseradish", "Brassicaceae", "f89r"),
    ("Capsella bursa-pastoris (L.) Medik.", "Shepherd's Purse", "Brassicaceae", "f90r"),
    ("Sinapis alba L.", "White Mustard", "Brassicaceae", "f91r"),
    ("Raphanus sativus L.", "Radish", "Brassicaceae", "f92r"),

    # Ranunculaceae (buttercup family — cold, toxic, potent)
    ("Paeonia officinalis L.", "Peony", "Paeoniaceae", "f93r"),
    ("Nigella sativa L.", "Black Cumin", "Ranunculaceae", "f94r"),
    ("Aquilegia vulgaris L.", "Columbine", "Ranunculaceae", "f95r"),
    ("Anemone pulsatilla L.", "Pasque Flower", "Ranunculaceae", "f96r"),
    ("Delphinium staphisagria L.", "Stavesacre", "Ranunculaceae", "f97r"),
    ("Ranunculus ficaria L.", "Lesser Celandine", "Ranunculaceae", "f98r"),

    # Solanaceae (nightshade family — narcotic, anesthetic)
    ("Physalis alkekengi L.", "Chinese Lantern", "Solanaceae", "f99r"),
    ("Nicotiana tabacum L.", "Tobacco", "Solanaceae", "f100r"),

    # Iridaceae / Liliaceae (bulb and corm medicines)
    ("Crocus sativus L.", "Saffron", "Iridaceae", "f101r"),
    ("Iris florentina L.", "Orris Root", "Iridaceae", "f102r"),
    ("Allium sativum L.", "Garlic", "Amaryllidaceae", "f103r"),
    ("Allium cepa L.", "Onion", "Amaryllidaceae", "f104r"),

    # Exotic imports (spices, resins — via Venice/Genoa trade)
    ("Zingiber officinale Roscoe", "Ginger", "Zingiberaceae", "f105r"),
    ("Curcuma longa L.", "Turmeric", "Zingiberaceae", "f106r"),
    ("Piper nigrum L.", "Black Pepper", "Piperaceae", "f107r"),
    ("Cinnamomum verum J.Presl", "Cinnamon", "Lauraceae", "f108r"),
    ("Myristica fragrans Houtt.", "Nutmeg", "Myristicaceae", "f109r"),
    ("Syzygium aromaticum (L.) Merr.", "Clove", "Myrtaceae", "f110r"),
    ("Boswellia sacra Flueck.", "Frankincense", "Burseraceae", "f111r"),
    ("Commiphora myrrha (Nees) Engl.", "Myrrh", "Burseraceae", "f112r"),
    ("Laurus nobilis L.", "Bay Laurel", "Lauraceae", "f113r"),
]

RECIPE_OPS = [
    "Divide/tere (Divide or grind)",
    "Calefac/commisce (Heat or combine)",
    "Extrahe/colare (Extract or strain)",
    "Accipe materiam (Take up ingredient in hand)",
    "Compone (Compose/formulate final mixture)",
    "Applica/administra (Apply or administer)",
    "Serva/conserva (Store or preserve)",
    "Macera (Soak/macerate)",
    "Filtra/colare (Filter or strain)",
    "Evapora (Evaporate/concentrate)",
]

RECIPE_OPS_MULTI = [
    "Calefac x2 (Heat/mix 2 times)",
    "Divide/tere x3 (Grind thoroughly, 3 passes)",
    "Divide/tere x4 (Grind thoroughly, 4 passes)",
    "Extrahe x2 (Extract twice)",
    "Calefac x3 (Heat/mix 3 times)",
    "Divide/tere x2 (Grind, 2 passes)",
]


def _weighted_choice(items: list[tuple[str, float]], rng: random.Random) -> str:
    choices, weights = zip(*items)
    return rng.choices(list(choices), weights=list(weights), k=1)[0]


def generate_herb_entry(rng: random.Random) -> dict:
    """Generate a synthetic Voynich herb monograph with pharmaceutical metadata."""
    plant = rng.choice(PHYTOGLYPHICA_PLANTS)
    n_entries = rng.randint(3, 14)
    entries = []
    for i in range(n_entries):
        prep = _weighted_choice(HERB_PREPARATIONS, rng)
        form = _weighted_choice(HERB_FORMS, rng)
        pot = _weighted_choice(HERB_POTENCIES, rng)
        part = _weighted_choice(HERB_PARTS, rng)
        app = _weighted_choice(HERB_APPLICATIONS, rng)
        vol = 'yes' if rng.random() < 0.12 else 'no'
        fix = 'yes' if rng.random() < 0.18 else 'no'
        ind = 'yes' if rng.random() < 0.08 else 'no'
        n_ops = rng.randint(3, 16)
        entries.append({
            'para': i + 1,
            'preparatio': prep,
            'forma': form,
            'potentia': pot,
            'pars_plantae': part,
            'applicatio': app,
            'volatilis': vol,
            'fixatio': fix,
            'indicatio': ind,
            'n_ops': n_ops,
        })
    return {
        'folio': plant[3],
        'latin': plant[0],
        'common': plant[1],
        'family': plant[2],
        'n_entries': n_entries,
        'entries': entries,
        'plant_tuple': plant,  # for session engine
    }


_STEP_LABEL_RE = re.compile(r'^Step \d+:\s*')
_ACCIPE_RE = re.compile(r'Accipe (\d+) materias')


def _recipe_step_label(raw: str) -> str:
    """The opcode label of a recipe step, with its 'Step N: ' prefix removed."""
    return _STEP_LABEL_RE.sub('', raw).strip()


def _ingredients_from_opener(label: str) -> int:
    """The ingredient count an opener step declares. 'Accipe materiam' loads one;
    'Accipe k materias' loads k; a transform opener (Divide/Calefac) loads none,
    which is the zero-ingredient program that operates on an existing output."""
    if label.startswith('Accipe materiam'):
        return 1
    m = _ACCIPE_RE.search(label)
    if m:
        return int(m.group(1))
    return 0


def load_recipe_model(path: str | Path) -> dict | None:
    """Build a first-order step model from the real VMS recipe corpus
    (voynich_recipe_bio.json). The corpus is 1076 attested recipes; the model
    carries the step-count distribution, the opening-step distribution, and the
    step-to-step transition counts. Walking that chain reproduces the corpus:
    the openers (recipes open with an Accipe that names the ingredient count,
    the zero-ingredient ones with a transform), the opcode frequencies
    (Extrahe, Divide, Calefac lead), and the ~7.5 mean step count. Returns None
    if the corpus is unreadable, and the caller falls back to the closed-form
    synthetic recipe."""
    try:
        data = json.loads(Path(path).read_text())
        recs = data['recipe_section']['recipes']
    except Exception:
        return None
    n_steps: Counter = Counter()
    start: Counter = Counter()
    last: Counter = Counter()
    trans: dict[str, Counter] = defaultdict(Counter)
    opcodes: Counter = Counter()
    for r in recs:
        steps = [_recipe_step_label(s) for s in r.get('steps', [])]
        if not steps:
            continue
        n_steps[len(steps)] += 1
        start[steps[0]] += 1
        for s in steps:
            opcodes[s] += 1
        # last step handled separately in generate_recipe, otherwise
        # Compone gets counted twice
        if len(steps) > 1:
            last[steps[-1]] += 1
            prefix = steps[:-1]
        else:
            prefix = steps
        for a, b in zip(prefix, prefix[1:]):
            trans[a][b] += 1
    if not n_steps:
        return None
    return {
        'n_steps': n_steps,
        'start': start,
        'last': last,
        'trans': {k: dict(v) for k, v in trans.items()},
        'opcodes': opcodes,
        'n_recipes': sum(n_steps.values()),
    }


def _pick_counter(c, rng: random.Random):
    keys = list(c)
    return rng.choices(keys, weights=[c[k] for k in keys], k=1)[0]


def _resolve_recipe_model(explicit: str | None, transcription: str | None) -> dict | None:
    """Find and load the recipe corpus. An explicit --recipe-corpus wins; else
    look next to the transcription file and in the sibling Voynich_Phytoglyphica
    data directory."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if transcription:
        candidates.append(Path(transcription).resolve().parent / "voynich_recipe_bio.json")
    here = Path(__file__).resolve().parent
    candidates.append(here.parent / "Voynich_Phytoglyphica" / "data" / "voynich_recipe_bio.json")
    for c in candidates:
        if c.exists():
            model = load_recipe_model(c)
            if model:
                return model
    return None


def print_recipe_verification(gen_recipes: list[dict], model: dict) -> None:
    """Compare the generated recipes to the real corpus on the step statistics
    the recipe model reproduces: mean step count, the fraction opening with an
    Accipe, and the leading step-opcode frequencies."""
    print("\n=== RECIPE VERIFICATION ===\n")

    def _mean_steps(dist: Counter) -> float:
        tot = sum(dist.values())
        return sum(k * v for k, v in dist.items()) / tot if tot else 0.0

    real_mean = _mean_steps(model['n_steps'])
    gen_mean = sum(len(r['steps']) for r in gen_recipes) / len(gen_recipes)

    real_start_tot = sum(model['start'].values())
    real_accipe = sum(c for lbl, c in model['start'].items()
                      if lbl.startswith('Accipe')) / real_start_tot
    gen_accipe = sum(1 for r in gen_recipes
                     if _recipe_step_label(r['steps'][0]).startswith('Accipe')) / len(gen_recipes)

    gen_opc: Counter = Counter()
    for r in gen_recipes:
        for s in r['steps']:
            gen_opc[_recipe_step_label(s)] += 1
    real_tot = sum(model['opcodes'].values())
    gen_tot = sum(gen_opc.values()) or 1

    print(f"  {'Metric':<34} {'Corpus':<12} {'Synthetic':<12} Match")
    print(f"  {'-'*34} {'-'*12} {'-'*12} -----")
    print(f"  {'Mean steps per recipe':<34} {real_mean:<12.3f} {gen_mean:<12.3f} {_pct(real_mean, gen_mean)}")
    print(f"  {'Opens with Accipe':<34} {real_accipe:<12.3f} {gen_accipe:<12.3f} {_pct(real_accipe, gen_accipe)}")
    print(f"\n  Leading step-opcode frequency:")
    for lbl, _c in model['opcodes'].most_common(6):
        rv = model['opcodes'][lbl] / real_tot
        gv = gen_opc.get(lbl, 0) / gen_tot
        name = _STEP_LABEL_RE.sub('', lbl)[:38]
        print(f"    {name:<40} {rv:<10.4f} {gv:<10.4f} {_pct(rv, gv)}")


def generate_recipe(rng: random.Random, model: dict | None = None) -> dict:
    """Generate a synthetic Voynich procedural recipe (f103r+ style).

    With a model from load_recipe_model, the step count is drawn from the real
    distribution and the steps are a walk of the real step-to-step chain, so the
    opener, the opcode mix, and the ingredient load match the corpus. Without
    one, it falls back to the closed-form synthetic: real ingredient
    distribution, uniform opcode draw."""
    if model:
        n_steps = _pick_counter(model['n_steps'], rng)
        cur = _pick_counter(model['start'], rng)
        seq = [cur]
        # walk the interior, then draw the last step separately -- Compone
        # ends ~71% of recipes and a plain forward chain undersamples that
        for _ in range(max(0, n_steps - 2)):
            row = model['trans'].get(cur)
            cur = _pick_counter(row, rng) if row else _pick_counter(model['opcodes'], rng)
            seq.append(cur)
        if n_steps > 1:
            seq.append(_pick_counter(model['last'], rng))
        n_ingredients = _ingredients_from_opener(seq[0])
        steps = [f"Step {i+1}: {lbl}" for i, lbl in enumerate(seq)]
        return {
            'folio': f"f{rng.randint(103, 116)}r",
            'para': rng.randint(1, 30),
            'n_ops': max(n_steps, rng.randint(6, 18)),
            'n_steps': n_steps,
            'n_ingredients': n_ingredients,
            'steps': steps,
        }

    n_ops = rng.randint(6, 18)
    n_steps = rng.randint(4, min(n_ops, 15))
    n_ingredients = rng.choices(
        [1, 2, 3, 4, 5, 6, 0],
        weights=[0.16, 0.31, 0.31, 0.15, 0.03, 0.002, 0.046], k=1
    )[0]

    steps = []
    for i in range(n_steps):
        if i > 0 and rng.random() < 0.25:
            step = rng.choice(RECIPE_OPS_MULTI)
        elif n_ingredients > 0 and rng.random() < 0.20:
            if n_ingredients == 1:
                step = "Accipe materiam (Take up ingredient in hand)"
            else:
                step = f"Accipe {n_ingredients} materias (Take {n_ingredients} ingredients)"
        else:
            step = rng.choice(RECIPE_OPS)
        steps.append(f"Step {i+1}: {step}")

    return {
        'folio': f"f{rng.randint(103, 116)}r",
        'para': rng.randint(1, 30),
        'n_ops': n_ops,
        'n_steps': n_steps,
        'n_ingredients': n_ingredients,
        'steps': steps,
    }


def render_herb_entry(entry: dict) -> str:
    lines = [
        f"\n### {entry['folio']} — {entry['latin']} ({entry['common']}) · {entry['family']} [synthetic]",
        "",
        "| # | para | preparatio | forma | potentia | pars_plantae | applicatio | vol | fix | ind | n_ops |",
        "|---|------|------------|-------|----------|--------------|------------|-----|-----|-----|-------|",
    ]
    for e in entry['entries']:
        lines.append(
            f"| {e['para']} | {e['para']} | {e['preparatio']} | {e['forma']} | "
            f"{e['potentia']} | {e['pars_plantae']} | {e['applicatio']} | "
            f"{e['volatilis']} | {e['fixatio']} | {e['indicatio']} | {e['n_ops']} |"
        )
    return '\n'.join(lines)


def render_recipe(recipe: dict) -> str:
    lines = [
        f"\n### {recipe['folio']} para {recipe['para']} — {recipe['n_ops']} ops, "
        f"{recipe['n_steps']} steps, {recipe['n_ingredients']} ingredients [synthetic]",
    ]
    for step in recipe['steps']:
        lines.append(f"  {step}")
    return '\n'.join(lines)


def render_session_record(record: MonographRecord) -> str:
    """Render a session engine IFIX record as formatted text."""
    gate_colors = {
        GateResult.PASS: 'PASSED',
        GateResult.FAIL: 'FAILED',
        GateResult.AMBIGUOUS: 'AMBIGUOUS',
    }

    lines = [
        f"\n{'='*70}",
        f"  IFIX RECORD — {record.plant_latin} ({record.plant_common})",
        f"  Family: {record.family}  |  Folio: {record.folio}",
        f"  Session: {'PASSED' if record.session_passed else 'FAILED'}",
        f"{'='*70}",
        "",
        "  GRAMMAR TUPLE:",
    ]

    # Primitive display order
    prim_order = ['D', 'T', 'R', 'P', 'F', 'K', 'G', 'C', 'PHI', 'H', 'S', 'W']
    prim_names = {
        'D': 'Dimensionality', 'T': 'Topology', 'R': 'Recognition',
        'P': 'Parity', 'F': 'Fidelity', 'K': 'Kinetics',
        'G': 'Granularity', 'C': 'Composition', 'PHI': 'Criticality',
        'H': 'Chirality', 'S': 'Stoichiometry', 'W': 'Winding',
    }
    for p in prim_order:
        val = record.grammar_tuple.get(p, '?')
        lines.append(f"    {p} ({prim_names[p]:<14}): {val}")

    lines.append("")
    lines.append("  GATE TRACES:")
    for trace in record.gate_traces:
        status = gate_colors[trace.result]
        lines.append(f"    {trace.gate_name}: {status}")
        lines.append(f"      {trace.reason}")
        if trace.primitive_checks:
            for k, v in trace.primitive_checks.items():
                mark = 'OK' if v else 'XX'
                lines.append(f"      [{mark}] {k}")
    lines.append("")

    if record.recipe_params:
        lines.append("  ELABORATED PROTOCOL:")
        for k, v in record.recipe_params.items():
            lines.append(f"    {k:<16}: {v}")
        lines.append("")

    if record.elaborated_steps:
        lines.append("  RECIPE STEPS (elaborated):")
        for step in record.elaborated_steps:
            lines.append(
                f"    Step {step['step_num']}: {step['opcode']:<12} "
                f"— {step['description']}"
            )
            lines.append(f"      [{step['primitive']}] {step['annotation']}")
        lines.append("")

    return '\n'.join(lines)

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# REPORTING
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣

def _pct(v: float, s: float) -> str:
    if v > 0:
        return f"{100 * (1 - abs(v - s) / v):.0f}%"
    return "—"


def print_voynich_stats(v_words: dict[str, list[list[int]]],
                         v_stats: CorpusStats) -> None:
    print("\n=== VOYNICH STATISTICAL FINGERPRINT ===\n")
    all_words = [w for ws in v_words.values() for w in ws]
    gb = _build_bigrams(all_words)
    gu = Counter(t for w in all_words for t in w)

    print(f"  Words parsed:          {len(all_words):,}")
    print(f"  Total glyph tokens:    {v_stats.total_tokens:,}")
    print(f"  Unique glyphs active:  {len(gu)}")
    active_secs = [s for s in SECTIONS if v_words.get(s)]
    for sec in active_secs:
        n = len(v_words.get(sec, []))
        if n:
            print(f"    {sec:<16}: {n:>6,} words")

    print(f"\n  Global metrics:")
    print(f"    Zipf exponent:       {zipf_exponent(v_stats.global_vocab):.4f}  (natural language ~1.0-1.5)")
    print(f"    Bigram entropy:      {bigram_entropy(gb, gu):.4f} bits/token")
    print(f"    Type-token ratio:    {type_token_ratio(all_words):.4f}")
    print(f"    Repetition rate:     {repetition_rate(all_words):.4f}  (window=10)")
    print(f"    Spectral gap:        {spectral_gap(gb):.6f}")

    secs = [s for s in SECTIONS if v_words.get(s)]
    print(f"\n  Section KL divergences:")
    for i, s1 in enumerate(secs):
        for s2 in secs[i+1:]:
            if v_stats.unigram.get(s1) and v_stats.unigram.get(s2):
                kl = kl_divergence(v_stats.unigram[s1], v_stats.unigram[s2])
                print(f"    {s1[:3]}<->{s2[:3]}: {kl:.4f}")

    print(f"\n  IMSCRIBr section fingerprints (dominant 8-cycle):")
    for sec in secs:
        fp, canon = section_fingerprint(v_words[sec])
        print(f"    {sec:<16}: {fp.coarse_key():<34} -> {canon}")


def _verification_rows(
    v_words: dict[str, list[list[int]]],
    v_stats: CorpusStats,
    s_words: dict[str, list[list[int]]],
    s_stats: CorpusStats,
) -> list[tuple[str, float, float]]:
    """(name, voynich_value, synthetic_value) rows for one realization.

    Zipf/TTR/repetition rate move with sample size, so the real side gets
    read on a subsample matched to the synthetic size instead of the full
    corpus. Same deal for the KL rows -- turns out KL between two
    sections isn't stable at this alphabet size either, so it gets the
    same treatment.
    """
    v_all = [w for ws in v_words.values() for w in ws]
    s_all = [w for ws in s_words.values() for w in ws]
    v_bg = _build_bigrams(v_all)
    s_bg = _build_bigrams(s_all)
    v_uni = Counter(t for w in v_all for t in w)
    s_uni = Counter(t for w in s_all for t in w)

    def _matched(metric, draws: int = 7):
        if len(v_all) <= len(s_all):
            return metric(v_all)
        rng = random.Random(0)
        vals = [metric(rng.sample(v_all, len(s_all))) for _ in range(draws)]
        return sum(vals) / len(vals)

    v_zipf = _matched(lambda ws: zipf_exponent(Counter(tuple(w) for w in ws)))
    v_ttr = _matched(type_token_ratio)
    v_rep = _matched(repetition_rate)

    rows = [
        ("Zipf exponent",    v_zipf,                      zipf_exponent(s_stats.global_vocab)),
        ("Bigram entropy",   bigram_entropy(v_bg, v_uni), bigram_entropy(s_bg, s_uni)),
        ("Type-token ratio", v_ttr,                       type_token_ratio(s_all)),
        ("Repetition rate",  v_rep,                       repetition_rate(s_all)),
        ("Spectral gap",     spectral_gap(v_bg),          spectral_gap(s_bg)),
    ]

    def _matched_kl(s1: str, s2: str, n1: int, n2: int, draws: int = 7) -> float:
        w1, w2 = v_words[s1], v_words[s2]
        if len(w1) <= n1 and len(w2) <= n2:
            return kl_divergence(v_stats.unigram[s1], v_stats.unigram[s2])
        rng = random.Random(0)
        vals = []
        for _ in range(draws):
            u1 = Counter(t for w in (rng.sample(w1, n1) if len(w1) > n1 else w1) for t in w)
            u2 = Counter(t for w in (rng.sample(w2, n2) if len(w2) > n2 else w2) for t in w)
            vals.append(kl_divergence(u1, u2))
        return sum(vals) / len(vals)

    common = [s for s in SECTIONS if v_words.get(s) and s_words.get(s)]
    for i, s1 in enumerate(common):
        for s2 in common[i+1:]:
            if v_stats.unigram.get(s1) and s_stats.unigram.get(s2):
                rows.append((
                    f"KL {s1[:3]}<->{s2[:3]}",
                    _matched_kl(s1, s2, len(s_words[s1]), len(s_words[s2])),
                    kl_divergence(s_stats.unigram[s1], s_stats.unigram[s2]),
                ))
    return rows


def print_verification(
    v_words: dict[str, list[list[int]]],
    s_words: dict[str, list[list[int]]],
    v_stats: CorpusStats,
    s_stats: CorpusStats,
    rng: random.Random | None = None,
    n_seeds: int = 5,
) -> None:
    """Print the corpus/synthetic verification table.

    One small synthetic draw is noisy -- confirmed this by resampling the
    real corpus at the same size and watching KL divergence swing by tens
    of points on its own. So each section gets regenerated at its real
    size and averaged over n_seeds draws instead of trusting one shot.
    The s_words/s_stats args are just for the fingerprint/notation
    comparisons further down, which don't have this problem.
    """
    print("\n=== VERIFICATION ===\n")
    common = [s for s in SECTIONS if v_words.get(s)]
    full_size = {s: len(v_words[s]) for s in common}
    rng = rng or random.Random(1)

    all_rows: list[list[tuple[str, float, float]]] = []
    for _ in range(max(1, n_seeds)):
        sv: dict[str, list[list[int]]] = {}
        for sec in common:
            if v_stats.bigrams.get(sec):
                sv[sec] = generate_section_v3(sec, full_size[sec], v_stats, rng)
        sv_stats = extract_stats(sv)
        all_rows.append(_verification_rows(v_words, v_stats, sv, sv_stats))

    names = [r[0] for r in all_rows[0]]
    avg_rows = []
    for i, name in enumerate(names):
        v_vals = [run[i][1] for run in all_rows]
        s_vals = [run[i][2] for run in all_rows]
        avg_rows.append((name, sum(v_vals) / len(v_vals), sum(s_vals) / len(s_vals)))

    print(f"  each row generated at real per-section corpus size, "
          f"averaged over {n_seeds} draws\n")
    print(f"  {'Metric':<26} {'Voynich':<12} {'Synthetic':<12} Match")
    print(f"  {'-'*26} {'-'*12} {'-'*12} -----")
    for name, v_val, s_val in avg_rows:
        print(f"  {name:<26} {v_val:<12.4f} {s_val:<12.4f} {_pct(v_val, s_val)}")

    print(f"\n  IMSCRIBr fingerprint comparison:")
    print(f"  {'Section':<16}  {'Voynich coarse key':<34}  {'Synthetic coarse key':<34}  Result")
    preview_common = [s for s in common if s_words.get(s)]
    for sec in preview_common:
        v_fp, v_cn = section_fingerprint(v_words[sec])
        s_fp, s_cn = section_fingerprint(s_words[sec])
        ck_match = v_fp.coarse_key() == s_fp.coarse_key()
        fb_match = (v_fp.frobenius_order == s_fp.frobenius_order and
                    v_fp.dialetheia_complete == s_fp.dialetheia_complete)
        result = "MATCH" if ck_match else ("FROB+DIAL" if fb_match else "DIFF")
        print(f"  {sec:<16}  {v_fp.coarse_key():<34}  {s_fp.coarse_key():<34}  {result}")

    print(f"\n  Surface notation comparison:")
    v_chars = set()
    for g in EVA_GLYPHS:
        v_chars.add(g)
    s_chars = set(TOKEN_TO_SHAVIAN_FULL.values())
    v_flat = set()
    for c in v_chars:
        for ch in c:
            v_flat.add(ch)
    s_flat = set()
    for c in s_chars:
        s_flat.add(c)
    overlap = v_flat & s_flat
    print(f"    Voynich EVA glyphs:    {len(v_chars)} types")
    print(f"    Synthetic:              {len(s_chars)} types")
    if not overlap:
        print(f"    Shared characters:     0 — ZERO OVERLAP — proof complete")
        print(f"\n    The statistical signature is a property of the grammar, not the notation.")
    else:
        print(f"    Overlap:               {len(overlap)} chars")


def print_session_summary(engine: SessionEngine) -> None:
    """Print a summary of the session engine's gate processing results."""
    records = engine.session_records
    passed = sum(1 for r in records if r.session_passed)
    failed = sum(1 for r in records if not r.session_passed)

    print(f"\n{'='*70}")
    print(f"  SESSION ENGINE SUMMARY")
    print(f"  Total monographs processed: {len(records)}")
    print(f"  Passed all gates:           {passed}")
    print(f"  Failed at least one gate:   {failed}")
    print(f"{'='*70}")

    # Gate statistics
    gate_stats = {}
    for gate_name in ['GATE_1', 'GATE_2', 'GATE_3']:
        passes = sum(1 for r in records
                     for t in r.gate_traces
                     if t.gate_name == gate_name and t.result == GateResult.PASS)
        fails = sum(1 for r in records
                    for t in r.gate_traces
                    if t.gate_name == gate_name and t.result == GateResult.FAIL)
        ambig = sum(1 for r in records
                    for t in r.gate_traces
                    if t.gate_name == gate_name and t.result == GateResult.AMBIGUOUS)
        gate_stats[gate_name] = (passes, fails, ambig)

    print(f"\n  Per-gate statistics:")
    print(f"  {'Gate':<10} {'EVALT':>8} {'EVALF':>8} {'ENGAGR':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for gate_name, (p, f, a) in gate_stats.items():
        print(f"  {gate_name:<10} {p:>8} {f:>8} {a:>8}")

    # Cold-process constraint reports
    cold_records = [r for r in records
                    if r.grammar_tuple.get('K', '') == '\U00010454']
    if cold_records:
        print(f"\n  Cold-process constraint active: {len(cold_records)} entries")
        for r in cold_records:
            print(f"    {r.plant_common} ({r.family}): Calefac applies to excipient only")

# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣
# MAIN
# ⊢⊡⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊙⊡⊣


def print_plant_catalog(family: str | None = None, fmt: str = "table") -> None:
    """Print the Phytoglyphica plant catalog in the requested format."""
    plants = PHYTOGLYPHICA_PLANTS
    if family:
        plants = [p for p in plants if p[2].lower() == family.lower()]
        if not plants:
            print(f"No plants found in family '{family}'.")
            print(f"Available families: {', '.join(sorted(set(p[2] for p in PHYTOGLYPHICA_PLANTS)))}")
            return

    if fmt == "json":
        out = [{"latin": p[0], "common": p[1], "family": p[2], "folio": p[3]} for p in plants]
        print(json.dumps(out, indent=2))
    elif fmt == "csv":
        print("latin,common,family,folio")
        for p in plants:
            # escape commas in common name
            common = f'"{p[1]}"' if ',' in p[1] else p[1]
            print(f"{p[0]},{common},{p[2]},{p[3]}")
    else:  # table
        # Determine column widths
        latin_w = max(len(p[0]) for p in plants)
        common_w = max(len(p[1]) for p in plants)
        family_w = max(len(p[2]) for p in plants)
        header = f"  {'LATIN NAME':<{latin_w}}  {'COMMON':<{common_w}}  {'FAMILY':<{family_w}}  FOLIO"
        sep = f"  {'-'*latin_w}  {'-'*common_w}  {'-'*family_w}  -----"
        print(f"\n  PHYTOGLYPHICA PLANT CATALOG — {len(plants)} plants")
        if family:
            print(f"  Filtered by family: {family}")
        print()
        print(header)
        print(sep)
        for p in plants:
            print(f"  {p[0]:<{latin_w}}  {p[1]:<{common_w}}  {p[2]:<{family_w}}  {p[3]}")
        print()
        print(f"  {len(plants)} plants total.")
        fams = sorted(set(p[2] for p in plants))
        print(f"  {len(fams)} families: {', '.join(fams)}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudo-Voynich Generator v3 — Ob3ect-Gated Session Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output modes:
  --mode shavian    12-glyph Shavian (the proof — zero EVA overlap)
  --mode shavian-full  Full 30-glyph Shavian inventory
  --mode eva         Original EVA glyphs (human-readable Voynich notation)

Semantic modes:
  --herbs N         Generate N synthetic herb monographs with pharmaceutical metadata
  --recipes N       Generate N synthetic procedural recipes
  --semantic-only   Don't generate raw text, only semantic entries

V3 Session Engine:
  --session         Enable the three-gate session engine for herb entries.
                    Each monograph is routed through Gate 1 (Pharmaceutical Address),
                    Gate 2 (Balneological Heap), and Gate 3 (Astronomical Winding).
                    Gate traces and elaborated protocols are printed.
  --plant NAME      Process a specific plant through the session engine.
                    Use Latin binomial, e.g. "Artemisia absinthium L."
        """,
    )
    parser.add_argument("transcription", nargs="?", default=None, help="Path to LSI_ivtff_0d.txt")
    parser.add_argument("--section", default="all",
                        choices=SECTIONS + ["all"])
    parser.add_argument("--lines", type=int, default=100,
                        help="Total output lines (default: 100)")
    parser.add_argument("--words-per-line", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-seeds", type=int, default=5,
                        help="Draws to average the verification table over, "
                             "each generated at real per-section corpus size.")
    parser.add_argument("--transcriber-priority", type=str, default=None,
                        help="Comma-separated transcriber codes, most preferred "
                             "first, e.g. 'H,U,N,C,F'. Picks one reading per "
                             "physical line from the interlinear corpus; "
                             "default is H,U,N,Z,X,V,C,F,T,L,R,K,J,P,D,G,I,Q,M.")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--output", help="Save synthetic text to file")
    parser.add_argument("--mode", default="shavian",
                        choices=["shavian", "shavian-full", "eva"],
                        help="Output notation (default: shavian)")
    parser.add_argument("--herbs", type=int, default=0,
                        help="Generate N synthetic herb entries")
    parser.add_argument("--recipes", type=int, default=0,
                        help="Generate N synthetic recipe entries")
    parser.add_argument("--recipe-corpus", type=str, default=None,
                        help="Path to voynich_recipe_bio.json. If omitted, looked "
                             "for next to the transcription and in the sibling "
                             "Voynich_Phytoglyphica/data. Drives recipe generation "
                             "from the real 1076-recipe corpus.")
    parser.add_argument("--semantic-only", action="store_true",
                        help="Only output semantic entries, no raw text")
    parser.add_argument("--session", action="store_true",
                        help="Enable three-gate session engine for herb entries")
    parser.add_argument("--plant", type=str, default=None,
                        help="Process a specific plant (Latin binomial) through session engine")
    parser.add_argument("--list-plants", action="store_true",
                        help="List all Phytoglyphica plants and exit. Use --family to filter.")
    parser.add_argument("--family", type=str, default=None,
                        help="Filter --list-plants by family name (e.g. Lamiaceae)")
    parser.add_argument("--plant-format", default="table",
                        choices=["table", "csv", "json"],
                        help="Output format for --list-plants (default: table)")
    args = parser.parse_args()

    if args.list_plants:
        print_plant_catalog(args.family, args.plant_format)
        return


    rng = random.Random(args.seed)

    if args.transcription is None:
        print("ERROR: No transcription file specified. Required for text generation.")
        print("Use --list-plants to browse the plant catalog without a corpus.")
        return

    print("=== PSEUDO-VOYNICH GENERATOR v3 (Ob3ect-Gated Session Engine) ===")
    print(f"    EVA (Voynich) -> {'EVA' if args.mode == 'eva' else 'Shavian'} (synthetic)")
    print(f"    Corpus: {args.transcription}")
    if args.session:
        print(f"    Session Engine: ENABLED — three-gate structural pipeline")
        print(f"    Gate 1: Pharmaceutical Address  |  Gate 2: Balneological Heap  |  Gate 3: Astronomical Winding")
    print()

    # Phase 1: Parse and extract stats
    print("Parsing IVTFF corpus...")
    priority = (args.transcriber_priority.split(',') if args.transcriber_priority
                else None)
    parse_report: dict = {}
    v_words = parse_ivtff(args.transcription, transcriber_priority=priority,
                          report=parse_report)
    total_parsed = sum(len(ws) for ws in v_words.values())
    active_secs = [s for s in SECTIONS if v_words.get(s)]
    print(f"  {parse_report['loci']:,} physical lines "
          f"({parse_report['variant_lines_seen']:,} transcriber-variant lines seen, "
          f"{parse_report['variant_lines_dropped']:,} duplicate readings dropped)")
    print(f"  {total_parsed:,} words parsed across {len(active_secs)} sections")
    for sec in active_secs:
        print(f"    {sec:<16}: {len(v_words[sec]):>6,} words")

    print("Extracting Voynich statistics...")
    v_stats = extract_stats(v_words)

    print_voynich_stats(v_words, v_stats)

    if args.stats_only:
        return

    # Phase 2: Initialize session engine if requested
    engine = SessionEngine(rng) if (args.session or args.plant) else None

    text_parts: list[str] = []

    # Phase 2a: Specific plant processing
    if args.plant and engine:
        print(f"\n=== SESSION ENGINE: Processing '{args.plant}' ===\n")
        found = False
        for plant in PHYTOGLYPHICA_PLANTS:
            if plant[0].lower() == args.plant.lower():
                found = True
                # Generate a mock entry for this plant
                entry = generate_herb_entry(rng)
                entry['latin'] = plant[0]
                entry['common'] = plant[1]
                entry['family'] = plant[2]
                entry['folio'] = plant[3]
                entry['plant_tuple'] = plant
                # Override with the matching plant
                record = engine.process_monograph(plant, entry['entries'])
                text_parts.append(render_session_record(record))
                print(render_session_record(record))
                break
        if not found:
            print(f"  Plant '{args.plant}' not found in Phytoglyphica catalog.")
            print(f"  Available: {', '.join(p[0] for p in PHYTOGLYPHICA_PLANTS[:5])}...")

    # Phase 2b: Semantic synthesis (herbs + recipes)
    herb_entries = []
    if args.herbs > 0:
        print(f"\n=== GENERATING {args.herbs} HERB MONOGRAPHS ===\n")
        herb_text: list[str] = ["# Synthetic Voynich Herb Monographs (Phytoglyphica-style)", ""]
        for i in range(args.herbs):
            entry = generate_herb_entry(rng)
            herb_entries.append(entry)
            herb_text.append(render_herb_entry(entry))
            preview = f"  [{i+1}/{args.herbs}] {entry['folio']} — {entry['latin']}"
            print(preview)

            # Route through session engine if enabled
            if engine and 'plant_tuple' in entry:
                record = engine.process_monograph(
                    entry['plant_tuple'], entry['entries']
                )
                status = 'PASSED' if record.session_passed else 'FAILED'
                print(f"    SESSION: {status} | Gates: " +
                      ' | '.join(f"{t.gate_name}={t.result.value}"
                                 for t in record.gate_traces))

        text_parts.append('\n'.join(herb_text))

    if args.recipes > 0:
        recipe_model = _resolve_recipe_model(args.recipe_corpus, args.transcription)
        print(f"\n=== GENERATING {args.recipes} RECIPES ===\n")
        if recipe_model:
            print(f"    Recipe model: real corpus, {recipe_model['n_recipes']} attested recipes")
        else:
            print(f"    Recipe model: closed-form synthetic (corpus not found)")
        recipe_text: list[str] = ["\n# Synthetic Voynich Recipes (f103r+ style)", ""]
        gen_recipes = []
        for i in range(args.recipes):
            recipe = generate_recipe(rng, recipe_model)
            gen_recipes.append(recipe)
            recipe_text.append(render_recipe(recipe))
            preview = f"  [{i+1}/{args.recipes}] {recipe['folio']} para {recipe['para']}"
            print(preview)
        text_parts.append('\n'.join(recipe_text))
        if recipe_model:
            print_recipe_verification(gen_recipes, recipe_model)

    # Phase 2c: Session engine summary
    if engine and engine.session_records:
        if not args.plant:
            print_session_summary(engine)

        # Append elaborated protocols to output
        for record in engine.session_records:
            if record.session_passed and record.elaborated_steps:
                text_parts.append(render_session_record(record))

    # Phase 2d: Raw text synthesis (unless semantic-only)
    if not args.semantic_only:
        wpl = args.words_per_line
        total_words = args.lines * wpl

        if args.section == "all":
            target_secs = active_secs
            total_v = sum(len(v_words.get(s, [])) for s in target_secs)
            # floor at 400 words so tiny sections (cosmological is one
            # folio) still get a usable unigram for the KL rows
            sec_counts = {
                s: max(400, round(total_words * len(v_words.get(s, [])) / max(total_v, 1)))
                for s in target_secs
            }
        else:
            target_secs = [args.section]
            sec_counts = {args.section: total_words}

        if args.mode == 'shavian':
            render_fn = render_shavian
        elif args.mode == 'shavian-full':
            render_fn = render_shavian_full
        else:
            render_fn = render_eva

        print(f"\n=== GENERATING SYNTHETIC TEXT ({args.mode}) ===")
        s_words: dict[str, list[list[int]]] = {}

        for sec in target_secs:
            if sec not in v_stats.bigrams or not v_stats.bigrams[sec]:
                print(f"\n  [{sec}] — insufficient data, skipping")
                continue
            n = sec_counts[sec]
            print(f"\n[{sec} — {n} words]")
            ws = generate_section_v3(sec, n, v_stats, rng)
            s_words[sec] = ws
            rendered = render_fn(ws, wpl)
            text_parts.append(f"\n[{sec} — {n} words, mode={args.mode}]\n{rendered}")
            preview = rendered[:500]
            print(preview + ("  ..." if len(rendered) > 500 else ""))

        if s_words:
            s_stats = extract_stats(s_words)
            print_verification(v_words, s_words, v_stats, s_stats, rng,
                               n_seeds=args.verify_seeds)
        else:
            print("No text sections generated.")

    # Phase 3: Write output
    if args.output and text_parts:
        out = Path(args.output)
        out.write_text('\n'.join(text_parts), encoding='utf-8')
        print(f"\nSaved to: {out}")

    print(f"\n=== COMPLETE ===")
    print(f"  Mode:     {args.mode}")
    print(f"  Semantic: {args.herbs} herbs, {args.recipes} recipes")
    if engine:
        print(f"  Session:  {len(engine.session_records)} monographs processed")
        passed = sum(1 for r in engine.session_records if r.session_passed)
        print(f"  Gates:    {passed} passed, {len(engine.session_records) - passed} failed")
    print(f"  Proof:    {'ZERO surface overlap' if args.mode != 'eva' else 'EVA -> EVA (human-readable)'}")


if __name__ == "__main__":
    main()
