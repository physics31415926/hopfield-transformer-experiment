"""
LAMBADA Degradation Analysis

Analyze per-example predictions to understand why Hopfield patching
degrades LAMBADA accuracy while improving HellaSwag.
"""

import json
import os
import numpy as np
from collections import Counter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_predictions(mode, bench):
    path = os.path.join(RESULTS_DIR, f'{mode}_{bench}_predictions.json')
    with open(path) as f:
        return json.load(f)


def load_correct(mode, bench):
    path = os.path.join(RESULTS_DIR, f'{mode}_{bench}_correct.npy')
    return np.load(path)


def analyze_lambada():
    orig = load_predictions('original', 'lambada')
    hopf = load_predictions('hopfield', 'lambada')
    augm = load_predictions('augmented', 'lambada')

    orig_c = load_correct('original', 'lambada')
    hopf_c = load_correct('hopfield', 'lambada')
    augm_c = load_correct('augmented', 'lambada')

    n = len(orig)
    print(f"Total examples: {n}")
    print(f"Original acc:  {orig_c.mean()*100:.2f}%")
    print(f"Hopfield acc:  {hopf_c.mean()*100:.2f}%")
    print(f"Augmented acc: {augm_c.mean()*100:.2f}%")

    # Transition matrix: original -> hopfield
    print("\n" + "="*60)
    print("  TRANSITION MATRIX: Original -> Hopfield")
    print("="*60)
    oo = int(((orig_c == 1) & (hopf_c == 1)).sum())  # both correct
    ox = int(((orig_c == 1) & (hopf_c == 0)).sum())  # orig correct, hopf wrong
    xo = int(((orig_c == 0) & (hopf_c == 1)).sum())  # orig wrong, hopf correct
    xx = int(((orig_c == 0) & (hopf_c == 0)).sum())  # both wrong
    print(f"  Both correct:     {oo:5d} ({oo/n*100:.1f}%)")
    print(f"  Orig OK, Hopf BAD:{ox:5d} ({ox/n*100:.1f}%)  <-- degraded")
    print(f"  Orig BAD, Hopf OK:{xo:5d} ({xo/n*100:.1f}%)  <-- improved")
    print(f"  Both wrong:       {xx:5d} ({xx/n*100:.1f}%)")
    print(f"  Net change: {xo - ox:+d} examples")

    # Transition matrix: original -> augmented
    print("\n" + "="*60)
    print("  TRANSITION MATRIX: Original -> Augmented")
    print("="*60)
    oo2 = int(((orig_c == 1) & (augm_c == 1)).sum())
    ox2 = int(((orig_c == 1) & (augm_c == 0)).sum())
    xo2 = int(((orig_c == 0) & (augm_c == 1)).sum())
    xx2 = int(((orig_c == 0) & (augm_c == 0)).sum())
    print(f"  Both correct:     {oo2:5d} ({oo2/n*100:.1f}%)")
    print(f"  Orig OK, Augm BAD:{ox2:5d} ({ox2/n*100:.1f}%)  <-- degraded")
    print(f"  Orig BAD, Augm OK:{xo2:5d} ({xo2/n*100:.1f}%)  <-- improved")
    print(f"  Both wrong:       {xx2:5d} ({xx2/n*100:.1f}%)")
    print(f"  Net change: {xo2 - ox2:+d} examples")

    # Analyze degraded examples by context length
    print("\n" + "="*60)
    print("  DEGRADATION BY CONTEXT LENGTH (Original -> Hopfield)")
    print("="*60)

    ctx_lens = np.array([e['ctx_len'] for e in orig])
    text_lens = np.array([e['text_char_len'] for e in orig])
    n_last_tokens = np.array([e['n_last_tokens'] for e in orig])

    # Bin by context length quartiles
    quartiles = np.percentile(ctx_lens, [25, 50, 75])
    bins = [0, quartiles[0], quartiles[1], quartiles[2], ctx_lens.max() + 1]
    labels = ['Q1 (short)', 'Q2', 'Q3', 'Q4 (long)']

    print(f"\n  Context length quartiles: {quartiles}")
    print(f"  {'Bin':<15} {'N':>6} {'Orig%':>8} {'Hopf%':>8} {'Augm%':>8} {'Diff_H':>8} {'Diff_A':>8}")
    print(f"  {'-'*65}")

    for i in range(4):
        mask = (ctx_lens >= bins[i]) & (ctx_lens < bins[i+1])
        nm = mask.sum()
        if nm == 0:
            continue
        o_acc = orig_c[mask].mean() * 100
        h_acc = hopf_c[mask].mean() * 100
        a_acc = augm_c[mask].mean() * 100
        print(f"  {labels[i]:<15} {nm:>6} {o_acc:>7.1f}% {h_acc:>7.1f}% {a_acc:>7.1f}% {h_acc-o_acc:>+7.1f}% {a_acc-o_acc:>+7.1f}%")

    # Analyze by last word token count
    print("\n" + "="*60)
    print("  DEGRADATION BY LAST WORD TOKEN COUNT")
    print("="*60)
    print(f"  {'#Tokens':<10} {'N':>6} {'Orig%':>8} {'Hopf%':>8} {'Augm%':>8} {'Diff_H':>8} {'Diff_A':>8}")
    print(f"  {'-'*60}")

    for t in sorted(set(n_last_tokens)):
        mask = n_last_tokens == t
        nm = mask.sum()
        if nm < 20:
            continue
        o_acc = orig_c[mask].mean() * 100
        h_acc = hopf_c[mask].mean() * 100
        a_acc = augm_c[mask].mean() * 100
        print(f"  {t:<10} {nm:>6} {o_acc:>7.1f}% {h_acc:>7.1f}% {a_acc:>7.1f}% {h_acc-o_acc:>+7.1f}% {a_acc-o_acc:>+7.1f}%")

    # NLL analysis: compare per-example NLL
    print("\n" + "="*60)
    print("  NLL ANALYSIS")
    print("="*60)

    orig_nll = np.array([e['nll'] for e in orig])
    hopf_nll = np.array([e['nll'] for e in hopf])
    augm_nll = np.array([e['nll'] for e in augm])

    print(f"  Mean NLL:  orig={orig_nll.mean():.3f}  hopf={hopf_nll.mean():.3f}  augm={augm_nll.mean():.3f}")
    print(f"  Median NLL: orig={np.median(orig_nll):.3f}  hopf={np.median(hopf_nll):.3f}  augm={np.median(augm_nll):.3f}")

    nll_diff_h = hopf_nll - orig_nll
    nll_diff_a = augm_nll - orig_nll
    print(f"\n  NLL increase (hopfield - original):")
    print(f"    Mean: {nll_diff_h.mean():+.3f}  Median: {np.median(nll_diff_h):+.3f}")
    print(f"    % examples with higher NLL: {(nll_diff_h > 0).mean()*100:.1f}%")
    print(f"  NLL increase (augmented - original):")
    print(f"    Mean: {nll_diff_a.mean():+.3f}  Median: {np.median(nll_diff_a):+.3f}")
    print(f"    % examples with higher NLL: {(nll_diff_a > 0).mean()*100:.1f}%")

    # Degraded examples: what do they look like?
    print("\n" + "="*60)
    print("  DEGRADED EXAMPLES ANALYSIS (Orig correct -> Hopfield wrong)")
    print("="*60)

    degraded_mask = (orig_c == 1) & (hopf_c == 0)
    improved_mask = (orig_c == 0) & (hopf_c == 1)

    deg_ctx = ctx_lens[degraded_mask]
    imp_ctx = ctx_lens[improved_mask]
    all_ctx = ctx_lens

    print(f"  Degraded examples ({degraded_mask.sum()}):")
    print(f"    Mean ctx_len: {deg_ctx.mean():.1f}  (all: {all_ctx.mean():.1f})")
    print(f"    Mean text_len: {text_lens[degraded_mask].mean():.1f}  (all: {text_lens.mean():.1f})")
    print(f"    Mean n_last_tokens: {n_last_tokens[degraded_mask].mean():.2f}  (all: {n_last_tokens.mean():.2f})")

    if improved_mask.sum() > 0:
        print(f"  Improved examples ({improved_mask.sum()}):")
        print(f"    Mean ctx_len: {imp_ctx.mean():.1f}  (all: {all_ctx.mean():.1f})")
        print(f"    Mean text_len: {text_lens[improved_mask].mean():.1f}  (all: {text_lens.mean():.1f})")
        print(f"    Mean n_last_tokens: {n_last_tokens[improved_mask].mean():.2f}  (all: {n_last_tokens.mean():.2f})")

    # Sample some degraded examples
    print("\n  Sample degraded examples (last words):")
    deg_indices = np.where(degraded_mask)[0][:20]
    for idx in deg_indices:
        e = orig[idx]
        print(f"    [{idx}] last_word='{e['last_word']}' ctx_len={e['ctx_len']} nll_orig={orig[idx]['nll']:.2f} nll_hopf={hopf[idx]['nll']:.2f}")


def analyze_hellaswag():
    """Quick HellaSwag paired analysis for comparison."""
    orig_c = load_correct('original', 'hellaswag')
    hopf_c = load_correct('hopfield', 'hellaswag')
    augm_c = load_correct('augmented', 'hellaswag')

    n = len(orig_c)
    print("\n" + "="*60)
    print("  HELLASWAG TRANSITION MATRICES")
    print("="*60)

    for name, mod_c in [('hopfield', hopf_c), ('augmented', augm_c)]:
        oo = int(((orig_c == 1) & (mod_c == 1)).sum())
        ox = int(((orig_c == 1) & (mod_c == 0)).sum())
        xo = int(((orig_c == 0) & (mod_c == 1)).sum())
        xx = int(((orig_c == 0) & (mod_c == 0)).sum())
        print(f"\n  Original -> {name}:")
        print(f"    Both correct:     {oo:5d} ({oo/n*100:.1f}%)")
        print(f"    Orig OK, {name[:4]} BAD:{ox:5d} ({ox/n*100:.1f}%)")
        print(f"    Orig BAD, {name[:4]} OK:{xo:5d} ({xo/n*100:.1f}%)")
        print(f"    Both wrong:       {xx:5d} ({xx/n*100:.1f}%)")
        print(f"    Net change: {xo - ox:+d}")


if __name__ == '__main__':
    analyze_lambada()
    analyze_hellaswag()
