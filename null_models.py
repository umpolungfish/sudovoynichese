#!/usr/bin/env python3
"""Null-model controls for SudoVoynichese -> IMASM compilation claim."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sudovoynichese import parse_ivtff, extract_stats, section_fingerprint, SECTIONS
from classifier import match_canonical, CANONICAL_CLASSES
RNG = random.Random(42)

LATIN_ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
ITALIAN_ALPHABET = list("abcdefghilmnopqrstuvz")

def latin_to_token(ch):
    return hash(ch) % 12

def italian_to_token(ch):
    return (hash(ch) * 7 + 3) % 12

def shuffled_words(section_words):
    out = {}
    for sec, words in section_words.items():
        w = list(words)
        RNG.shuffle(w)
        out[sec] = w
    return out

def shuffled_glyphs(section_words):
    out = {}
    for sec, words in section_words.items():
        out[sec] = [RNG.sample(w, len(w)) for w in words]
    return out

def latin_like_text(length_dist, n_words, rng):
    words = []
    lengths = list(length_dist.keys())
    weights = [length_dist[l] for l in lengths]
    for _ in range(n_words):
        L = rng.choices(lengths, weights=weights, k=1)[0]
        word = [latin_to_token(rng.choice(LATIN_ALPHABET)) for _ in range(L)]
        words.append(word)
    return words

def italian_like_text(length_dist, n_words, rng):
    words = []
    lengths = list(length_dist.keys())
    weights = [length_dist[l] for l in lengths]
    for _ in range(n_words):
        L = rng.choices(lengths, weights=weights, k=1)[0]
        word = [italian_to_token(rng.choice(ITALIAN_ALPHABET)) for _ in range(L)]
        words.append(word)
    return words

def random_text(length_dist, n_words, rng):
    words = []
    lengths = list(length_dist.keys())
    weights = [length_dist[l] for l in lengths]
    for _ in range(n_words):
        L = rng.choices(lengths, weights=weights, k=1)[0]
        word = [rng.randint(0, 11) for _ in range(L)]
        words.append(word)
    return words

def fp_key(fp):
    return fp.coarse_key()

def is_canonical(canon):
    return canon is not None and not canon.startswith("~")

def is_approx(canon):
    return canon is not None and canon.startswith("~")

def main():
    print("=" * 70)
    print("NULL-MODEL CONTROLS FOR SUDOVOYNICHESE -> IMASM CLAIM")
    print("=" * 70)
    report = {}
    section_words = parse_ivtff('LSI_ivtff_0d_clean.txt', report=report)
    print(f"\nReal Voynich: {report['loci']} loci parsed")
    v_stats = extract_stats(section_words)
    length_dists = {sec: dict(v_stats.length_dist[sec]) for sec in SECTIONS}

    def run_label(label, get_words_fn):
        print("-" * 70)
        print(label)
        print("-" * 70)
        results = {}
        hits = 0
        for sec in SECTIONS:
            words = get_words_fn(sec)
            if words is None:
                continue
            fp, canon = section_fingerprint(words)
            results[sec] = (fp, canon)
            if is_canonical(canon):
                hits += 1
                flag = "CANONICAL"
            elif is_approx(canon):
                flag = "~approx"
            else:
                flag = "NO MATCH"
            print(f"  {sec:16s} canon={str(canon):20s} fp={fp_key(fp):40s} [{flag}]")
        print(f"\n  Canonical hits: {hits}/{len(SECTIONS)}")
        return results, hits

    # Real Voynich
    v_results, v_hits = run_label("1. REAL VOYNICH CORPUS", lambda sec: section_words.get(sec))

    # Null 1: shuffled words
    sw = shuffled_words(section_words)
    sw_results, sw_hits = run_label("2. SHUFFLED WORDS (within section)", lambda sec: sw.get(sec))

    # Null 2: shuffled glyphs
    sg = shuffled_glyphs(section_words)
    sg_results, sg_hits = run_label("3. SHUFFLED GLYPHS (within word)", lambda sec: sg.get(sec))

    # Null 4: Latin-like
    lat = {}
    for sec in SECTIONS:
        n = sum(length_dists[sec].values())
        lat[sec] = latin_like_text(length_dists[sec], n, RNG)
    lat_results, lat_hits = run_label("4. LATIN-LIKE (matched word-length)", lambda sec: lat.get(sec))

    # Null 5: Italian-like
    it = {}
    for sec in SECTIONS:
        n = sum(length_dists[sec].values())
        it[sec] = italian_like_text(length_dists[sec], n, RNG)
    it_results, it_hits = run_label("5. ITALIAN-LIKE (matched word-length)", lambda sec: it.get(sec))

    # Null 6: Pure random
    rnd = {}
    for sec in SECTIONS:
        n = sum(length_dists[sec].values())
        rnd[sec] = random_text(length_dists[sec], n, RNG)
    rnd_results, rnd_hits = run_label("6. PURE RANDOM (matched word-length)", lambda sec: rnd.get(sec))

    print("\n" + "=" * 70)
    print("SUMMARY: Canonical class hits per model")
    print("=" * 70)
    print(f"  {'Model':<30} {'Hits':>6} / {len(SECTIONS):<3}")
    print(f"  {'-' * 30} {'-' * 10}")
    print(f"  {'Real Voynich':<30s} {v_hits:>6d} / {len(SECTIONS):<3d}")
    print(f"  {'Shuffled words':<30s} {sw_hits:>6d} / {len(SECTIONS):<3d}")
    print(f"  {'Shuffled glyphs':<30s} {sg_hits:>6d} / {len(SECTIONS):<3d}")
    print(f"  {'Latin-like':<30s} {lat_hits:>6d} / {len(SECTIONS):<3d}")
    print(f"  {'Italian-like':<30s} {it_hits:>6d} / {len(SECTIONS):<3d}")
    print(f"  {'Pure random':<30s} {rnd_hits:>6d} / {len(SECTIONS):<3d}")

if __name__ == '__main__':
    main()
