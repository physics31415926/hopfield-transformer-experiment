"""
Statistical Significance Tests for Benchmark Results

1. Binomial proportion confidence intervals (Wilson score)
2. Two-proportion z-test (comparing modes)
3. Paired bootstrap test (when per-example predictions available)
4. McNemar's test (paired binary outcomes)
"""

import json
import os
import sys
import math
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def wilson_ci(correct, total, confidence=0.95):
    """Wilson score confidence interval for binomial proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return center - margin, center + margin


def two_proportion_ztest(c1, n1, c2, n2):
    """Two-proportion z-test. Returns z-statistic and two-sided p-value."""
    p1, p2 = c1 / n1, c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * stats.norm.sf(abs(z))
    return z, p_val


def paired_bootstrap(preds_a, preds_b, n_bootstrap=10000, seed=42):
    """Paired bootstrap test for accuracy difference.
    preds_a, preds_b: arrays of 0/1 per-example correctness."""
    rng = np.random.RandomState(seed)
    n = len(preds_a)
    assert len(preds_b) == n
    observed_diff = preds_a.mean() - preds_b.mean()
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = preds_a[idx].mean() - preds_b[idx].mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_val = np.mean(np.abs(diffs - diffs.mean()) >= abs(observed_diff - diffs.mean()))
    return observed_diff, ci_lo, ci_hi, p_val


def mcnemar_test(preds_a, preds_b):
    """McNemar's test for paired binary outcomes."""
    b = np.sum((preds_a == 1) & (preds_b == 0))  # A correct, B wrong
    c = np.sum((preds_a == 0) & (preds_b == 1))  # A wrong, B correct
    if b + c == 0:
        return 0.0, 1.0, b, c
    chi2 = (abs(b - c) - 1)**2 / (b + c)  # with continuity correction
    p_val = stats.chi2.sf(chi2, df=1)
    return chi2, p_val, b, c


def analyze_aggregate(results_path):
    """Analyze statistical significance from aggregate results."""
    with open(results_path) as f:
        data = json.load(f)

    modes = ['original', 'hopfield', 'augmented']
    benchmarks = ['hellaswag', 'lambada']

    print("=" * 70)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    for bench in benchmarks:
        if bench not in data.get('original', {}):
            continue

        print(f"\n{'─' * 70}")
        print(f"  {bench.upper()}")
        print(f"{'─' * 70}")

        # 1. Confidence intervals
        print(f"\n  95% Wilson Score Confidence Intervals:")
        for m in modes:
            if bench not in data[m]:
                continue
            c = data[m][bench]['correct']
            n = data[m][bench]['total']
            acc = c / n * 100
            lo, hi = wilson_ci(c, n)
            print(f"    {m:12s}: {acc:.2f}%  [{lo*100:.2f}%, {hi*100:.2f}%]")

        # 2. Pairwise z-tests vs original
        print(f"\n  Two-Proportion Z-Tests (vs original):")
        c_orig = data['original'][bench]['correct']
        n_orig = data['original'][bench]['total']
        for m in ['hopfield', 'augmented']:
            if bench not in data[m]:
                continue
            c_m = data[m][bench]['correct']
            n_m = data[m][bench]['total']
            z, p = two_proportion_ztest(c_m, n_m, c_orig, n_orig)
            diff = c_m / n_m * 100 - c_orig / n_orig * 100
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {m:12s} vs original: diff={diff:+.2f}%  z={z:.3f}  p={p:.4f}  {sig}")

        # 3. Hopfield vs augmented
        if all(bench in data[m] for m in ['hopfield', 'augmented']):
            c_h = data['hopfield'][bench]['correct']
            n_h = data['hopfield'][bench]['total']
            c_a = data['augmented'][bench]['correct']
            n_a = data['augmented'][bench]['total']
            z, p = two_proportion_ztest(c_a, n_a, c_h, n_h)
            diff = c_a / n_a * 100 - c_h / n_h * 100
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"    {'augmented':12s} vs hopfield: diff={diff:+.2f}%  z={z:.3f}  p={p:.4f}  {sig}")

    # Effect sizes (Cohen's h)
    print(f"\n{'─' * 70}")
    print(f"  EFFECT SIZES (Cohen's h)")
    print(f"{'─' * 70}")
    for bench in benchmarks:
        if bench not in data.get('original', {}):
            continue
        p_orig = data['original'][bench]['correct'] / data['original'][bench]['total']
        for m in ['hopfield', 'augmented']:
            if bench not in data[m]:
                continue
            p_m = data[m][bench]['correct'] / data[m][bench]['total']
            h = 2 * (math.asin(math.sqrt(p_m)) - math.asin(math.sqrt(p_orig)))
            size = "negligible" if abs(h) < 0.2 else "small" if abs(h) < 0.5 else "medium" if abs(h) < 0.8 else "large"
            print(f"  {bench:12s} {m:12s} vs original: h={h:+.4f} ({size})")

    print()


def analyze_paired(predictions_dir):
    """Analyze with per-example predictions (paired tests)."""
    modes = ['original', 'hopfield', 'augmented']
    benchmarks = ['hellaswag', 'lambada']

    print("=" * 70)
    print("  PAIRED STATISTICAL TESTS")
    print("=" * 70)

    for bench in benchmarks:
        preds = {}
        for m in modes:
            path = os.path.join(predictions_dir, f'{m}_{bench}_predictions.npy')
            if os.path.exists(path):
                preds[m] = np.load(path)

        if 'original' not in preds:
            continue

        print(f"\n{'─' * 70}")
        print(f"  {bench.upper()} (Paired Tests)")
        print(f"{'─' * 70}")

        for m in ['hopfield', 'augmented']:
            if m not in preds:
                continue

            # Paired bootstrap
            diff, ci_lo, ci_hi, p_boot = paired_bootstrap(preds[m], preds['original'])
            print(f"\n  {m} vs original:")
            print(f"    Paired Bootstrap: diff={diff*100:+.2f}%  95% CI=[{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]  p={p_boot:.4f}")

            # McNemar's test
            chi2, p_mc, b, c = mcnemar_test(preds[m], preds['original'])
            print(f"    McNemar's test:   chi2={chi2:.3f}  p={p_mc:.4f}  (b={b}, c={c})")
            sig = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
            print(f"    Significance: {sig}")


def save_summary(results_path, output_path):
    """Save significance results as JSON."""
    with open(results_path) as f:
        data = json.load(f)

    summary = {}
    for bench in ['hellaswag', 'lambada']:
        if bench not in data.get('original', {}):
            continue
        bench_results = {}
        c_orig = data['original'][bench]['correct']
        n_orig = data['original'][bench]['total']
        lo, hi = wilson_ci(c_orig, n_orig)
        bench_results['original'] = {
            'accuracy': c_orig / n_orig,
            'ci_95': [lo, hi]
        }
        for m in ['hopfield', 'augmented']:
            if bench not in data[m]:
                continue
            c_m = data[m][bench]['correct']
            n_m = data[m][bench]['total']
            lo, hi = wilson_ci(c_m, n_m)
            z, p = two_proportion_ztest(c_m, n_m, c_orig, n_orig)
            p_orig = c_orig / n_orig
            p_m = c_m / n_m
            h = 2 * (math.asin(math.sqrt(p_m)) - math.asin(math.sqrt(p_orig)))
            bench_results[m] = {
                'accuracy': p_m,
                'ci_95': [lo, hi],
                'vs_original': {
                    'diff': p_m - p_orig,
                    'z_statistic': z,
                    'p_value': p,
                    'significant_05': bool(p < 0.05),
                    'significant_01': bool(p < 0.01),
                    'cohens_h': h
                }
            }
        summary[bench] = bench_results

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved significance summary to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/benchmark_results.json')
    parser.add_argument('--predictions', default=None, help='Dir with per-example .npy files')
    parser.add_argument('--output', default='results/significance_results.json')
    args = parser.parse_args()

    if os.path.exists(args.results):
        analyze_aggregate(args.results)
        save_summary(args.results, args.output)

    if args.predictions and os.path.exists(args.predictions):
        analyze_paired(args.predictions)
