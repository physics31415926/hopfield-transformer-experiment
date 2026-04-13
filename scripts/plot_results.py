"""
Visualization for Hopfield-Enhanced Transformer experiments.
Generates publication-quality comparison charts with training curves.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse
from collections import defaultdict

# Style config
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'vanilla': '#5B8DB8',
    'hopfield': '#E07B54',
    'augmented': '#6AAF6A',
}
LABELS = {
    'vanilla': 'Vanilla Transformer',
    'hopfield': 'Hopfield Attention (T-step)',
    'augmented': 'Hopfield + Memory Bank',
}
EXP_TITLES = {
    'recall': 'Associative Recall',
    'copy': 'Noisy Copy (Pattern Completion)',
    'lm': 'Structured Sequence LM',
}


def plot_training_curves(results_path, output_dir='results'):
    """Plot training & validation loss/accuracy curves for each experiment."""
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for exp_name, exp_data in results.items():
        modes = list(exp_data.keys())
        has_acc = any(
            exp_data[m].get('history', {}).get('val_acc') and
            any(v > 0 for v in exp_data[m]['history']['val_acc'])
            for m in modes
        )
        has_history = any('history' in exp_data[m] for m in modes)
        if not has_history:
            continue

        ncols = 3 if has_acc else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
        title = EXP_TITLES.get(exp_name, exp_name)
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

        for mode in modes:
            h = exp_data[mode].get('history', {})
            if not h:
                continue
            epochs = list(range(1, len(h['train_loss']) + 1))
            c = COLORS.get(mode, '#999')
            label = LABELS.get(mode, mode)

            # Train loss
            axes[0].plot(epochs, h['train_loss'], color=c, label=label,
                        linewidth=2, alpha=0.9)
            # Val loss
            axes[1].plot(epochs, h['val_loss'], color=c, label=label,
                        linewidth=2, linestyle='--', marker='o', markersize=3)

            if has_acc and ncols == 3:
                axes[2].plot(epochs, [a * 100 for a in h['val_acc']],
                           color=c, label=label, linewidth=2,
                           marker='o', markersize=3)

        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(frameon=False)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(frameon=False)
        axes[1].grid(True, alpha=0.3)

        if has_acc and ncols == 3:
            axes[2].set_title('Validation Accuracy')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy (%)')
            axes[2].legend(frameon=False)
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'{exp_name}_curves.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')


def plot_bar_comparison(results_path, output_dir='results'):
    """Bar charts comparing final metrics across models."""
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for exp_name, exp_data in results.items():
        modes = list(exp_data.keys())
        has_acc = any(exp_data[m].get('final_val_acc') is not None for m in modes)

        ncols = 3 if has_acc else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
        title = EXP_TITLES.get(exp_name, exp_name)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        x = np.arange(len(modes))
        width = 0.55
        colors = [COLORS.get(m, '#999') for m in modes]
        xlabels = [LABELS.get(m, m) for m in modes]

        # Best val loss
        ax = axes[0]
        vals = [exp_data[m].get('best_val_loss', 0) for m in modes]
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Best Validation Loss', fontweight='bold')
        ax.set_ylabel('Loss')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        # Highlight best
        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor('#333')
        bars[best_idx].set_linewidth(2)

        # Training time
        ax = axes[1]
        vals = [exp_data[m].get('time', 0) for m in modes]
        bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Training Time', fontweight='bold')
        ax.set_ylabel('Seconds')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # Accuracy (if applicable)
        if has_acc and ncols == 3:
            ax = axes[2]
            vals = [(exp_data[m].get('final_val_acc') or 0) * 100 for m in modes]
            bars = ax.bar(x, vals, width, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_title('Final Val Accuracy', fontweight='bold')
            ax.set_ylabel('Accuracy (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f'{exp_name}_comparison.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')


def plot_summary(results_path, output_dir='results'):
    """Single summary figure with all experiments."""
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    exp_names = list(results.keys())
    modes = list(results[exp_names[0]].keys())
    n_exp = len(exp_names)

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4.5))
    if n_exp == 1:
        axes = [axes]

    fig.suptitle('Hopfield-Enhanced Transformer: All Experiments',
                 fontsize=14, fontweight='bold', y=1.03)

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        exp_data = results[exp_name]
        x = np.arange(len(modes))
        vals = [exp_data[m].get('best_val_loss', 0) for m in modes]
        colors = [COLORS.get(m, '#999') for m in modes]
        xlabels = [LABELS.get(m, m) for m in modes]

        bars = ax.bar(x, vals, 0.55, color=colors, edgecolor='white')
        ax.set_title(EXP_TITLES.get(exp_name, exp_name), fontweight='bold')
        ax.set_ylabel('Best Val Loss')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor('#333')
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'summary_all.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_ablation(results_path, output_dir='results'):
    """Plot ablation study results from run_ablation.py output."""
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    exp_names = list(results.keys())

    # Separate by mode (hopfield vs augmented)
    mode_colors = {
        'hopfield': ('#E07B54', '#C4613A'),
        'augmented': ('#6AAF6A', '#4E8F4E'),
    }
    mode_labels = {
        'hopfield': 'Hopfield Attention',
        'augmented': 'Hopfield + Memory',
    }

    for exp_name in exp_names:
        exp_data = results[exp_name]

        # Group by mode
        grouped = {}
        for key, val in exp_data.items():
            mode = val.get('mode', key.split('_T')[0])
            T = val.get('hopfield_steps', int(key.split('_T')[1]))
            if mode not in grouped:
                grouped[mode] = []
            grouped[mode].append((T, val))

        n_modes = len(grouped)
        fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
        if n_modes == 1:
            axes = [axes]

        title = EXP_TITLES.get(exp_name, exp_name)
        fig.suptitle(f'Ablation: Hopfield Steps — {title}',
                     fontsize=14, fontweight='bold', y=1.03)

        for idx, (mode, entries) in enumerate(sorted(grouped.items())):
            ax = axes[idx]
            entries.sort(key=lambda x: x[0])
            steps = [e[0] for e in entries]
            losses = [e[1]['best_val_loss'] for e in entries]

            x = np.arange(len(steps))
            base_color = mode_colors.get(mode, ('#999', '#777'))
            colors_grad = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(steps)))

            bars = ax.bar(x, losses, 0.5, color=colors_grad, edgecolor='white')
            ax.set_title(mode_labels.get(mode, mode), fontweight='bold')
            ax.set_xlabel('Hopfield Steps (T)')
            ax.set_ylabel('Best Val Loss')
            ax.set_xticks(x)
            ax.set_xticklabels([f'T={s}' for s in steps])
            for bar, v in zip(bars, losses):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            best_idx = np.argmin(losses)
            bars[best_idx].set_edgecolor('#333')
            bars[best_idx].set_linewidth(2)

        plt.tight_layout()
        path = os.path.join(output_dir, f'ablation_{exp_name}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')

    # Also plot combined line chart: all experiments, loss vs T
    fig, axes = plt.subplots(1, len(exp_names), figsize=(5.5 * len(exp_names), 4.5))
    if len(exp_names) == 1:
        axes = [axes]

    fig.suptitle('Ablation: Best Val Loss vs Hopfield Steps',
                 fontsize=14, fontweight='bold', y=1.03)

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        exp_data = results[exp_name]

        grouped = {}
        for key, val in exp_data.items():
            mode = val.get('mode', key.split('_T')[0])
            T = val.get('hopfield_steps', int(key.split('_T')[1]))
            if mode not in grouped:
                grouped[mode] = []
            grouped[mode].append((T, val['best_val_loss']))

        for mode, entries in sorted(grouped.items()):
            entries.sort(key=lambda x: x[0])
            steps = [e[0] for e in entries]
            losses = [e[1] for e in entries]
            c = COLORS.get(mode, '#999')
            label = mode_labels.get(mode, mode)
            ax.plot(steps, losses, color=c, marker='o', linewidth=2,
                    markersize=6, label=label)

        ax.set_title(EXP_TITLES.get(exp_name, exp_name), fontweight='bold')
        ax.set_xlabel('Hopfield Steps (T)')
        ax.set_ylabel('Best Val Loss')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_combined.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_scaling(scaling_path, output_dir='results'):
    """Plot scaling experiment: val loss vs model size for each variant."""
    with open(scaling_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    exp_names = list(results.keys())
    n_exp = len(exp_names)

    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]

    fig.suptitle('Scaling: Effect of Model Size (d_model)',
                 fontsize=14, fontweight='bold', y=1.03)

    mode_labels = {
        'vanilla': 'Vanilla Transformer',
        'hopfield': 'Hopfield Attention',
        'augmented': 'Hopfield + Memory',
    }

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        exp_data = results[exp_name]

        # Group by mode
        grouped = defaultdict(list)
        for key, val in exp_data.items():
            mode = val['mode']
            d = val['d_model']
            grouped[mode].append((d, val['best_val_loss'], val['params']))

        for mode, entries in sorted(grouped.items()):
            entries.sort(key=lambda x: x[0])
            dims = [e[0] for e in entries]
            losses = [e[1] for e in entries]
            params = [e[2] for e in entries]
            c = COLORS.get(mode, '#999')
            label = mode_labels.get(mode, mode)
            ax.plot(dims, losses, color=c, marker='s', linewidth=2,
                    markersize=7, label=label)
            # Annotate param counts
            for d, l, p in zip(dims, losses, params):
                ax.annotate(f'{p//1000}K', (d, l), textcoords='offset points',
                           xytext=(0, 8), fontsize=7, ha='center', color=c)

        ax.set_title(EXP_TITLES.get(exp_name, exp_name), fontweight='bold')
        ax.set_xlabel('d_model')
        ax.set_ylabel('Best Val Loss')
        ax.set_xscale('log', base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scaling_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')

    # Also plot params vs loss
    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]
    fig.suptitle('Scaling: Parameters vs Performance',
                 fontsize=14, fontweight='bold', y=1.03)

    for i, exp_name in enumerate(exp_names):
        ax = axes[i]
        exp_data = results[exp_name]

        grouped = defaultdict(list)
        for key, val in exp_data.items():
            mode = val['mode']
            grouped[mode].append((val['params'], val['best_val_loss']))

        for mode, entries in sorted(grouped.items()):
            entries.sort(key=lambda x: x[0])
            params = [e[0] for e in entries]
            losses = [e[1] for e in entries]
            c = COLORS.get(mode, '#999')
            label = mode_labels.get(mode, mode)
            ax.plot(params, losses, color=c, marker='s', linewidth=2,
                    markersize=7, label=label)

        ax.set_title(EXP_TITLES.get(exp_name, exp_name), fontweight='bold')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Best Val Loss')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    plt.tight_layout()
    path = os.path.join(output_dir, 'scaling_params.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_wikitext(results_path, output_dir='results'):
    """Plot WikiText-2 experiment results."""
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    modes = list(results.keys())

    has_history = any('history' in results[m] for m in modes)

    # Training curves
    if has_history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('WikiText-2 Character-Level LM', fontsize=15, fontweight='bold', y=1.02)

        for mode in modes:
            h = results[mode].get('history', {})
            if not h:
                continue
            epochs = list(range(1, len(h['train_loss']) + 1))
            c = COLORS.get(mode, '#999')
            label = LABELS.get(mode, mode)

            axes[0].plot(epochs, h['train_loss'], color=c, label=label, linewidth=2)
            axes[1].plot(epochs, h['val_loss'], color=c, label=label, linewidth=2,
                        linestyle='--', marker='o', markersize=3)

        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend(frameon=False)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('Validation Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(frameon=False)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, 'wikitext_curves.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}')

    # Bar comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('WikiText-2: Model Comparison', fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(modes))
    colors = [COLORS.get(m, '#999') for m in modes]
    xlabels = [LABELS.get(m, m) for m in modes]

    vals = [results[m].get('best_val_loss', 0) for m in modes]
    bars = axes[0].bar(x, vals, 0.55, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Best Validation Loss', fontweight='bold')
    axes[0].set_ylabel('Loss (nats/char)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor('#333')
    bars[best_idx].set_linewidth(2)

    vals = [results[m].get('time', 0) for m in modes]
    bars = axes[1].bar(x, vals, 0.55, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title('Training Time', fontweight='bold')
    axes[1].set_ylabel('Seconds')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, rotation=15, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'wikitext_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def plot_pretrained(pretrained_path, output_dir='results'):
    """Plot pretrained model benchmark results with non-zero y-axis baseline."""
    with open(pretrained_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: PPL comparison (post-finetune focus) ---
    modes = []
    ppl_pre = []
    ppl_post = []
    colors = []
    mode_colors = {
        'original': COLORS['vanilla'],
        'hopfield': COLORS['hopfield'],
        'augmented': COLORS['augmented'],
    }
    mode_labels = {
        'original': 'Original',
        'hopfield': 'Hopfield Attn',
        'augmented': 'Augmented',
    }

    for mode in ['original', 'hopfield', 'augmented']:
        if mode not in data:
            continue
        d = data[mode]
        modes.append(mode_labels[mode])
        colors.append(mode_colors[mode])
        ppl_pre.append(d['pre_finetune']['perplexity'])
        if 'post_finetune' in d:
            ppl_post.append(d['post_finetune']['perplexity'])
        else:
            ppl_post.append(d['pre_finetune']['perplexity'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Qwen3-0.6B — Hopfield Attention Replacement (Last 4 Layers, 5 epochs)',
                 fontsize=14, fontweight='bold', y=1.02)

    # Left: Post-finetune PPL (non-zero baseline)
    x = np.arange(len(modes))
    width = 0.55
    bars = axes[0].bar(x, ppl_post, width, color=colors, edgecolor='white', linewidth=1.5)

    # Non-zero y-axis: set bottom to slightly below min value
    min_val = min(ppl_post)
    max_val = max(ppl_post)
    margin = (max_val - min_val) * 0.15
    y_bottom = max(0, min_val - margin * 2)
    axes[0].set_ylim(bottom=y_bottom, top=max_val + margin * 3)

    for bar, v in zip(bars, ppl_post):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + margin * 0.3,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, fontsize=11)
    axes[0].set_ylabel('Perplexity (lower is better)', fontsize=11)
    axes[0].set_title('Post-Finetune Perplexity', fontsize=12, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Add a break indicator on y-axis if baseline is cut
    if y_bottom > 0:
        axes[0].axhline(y=y_bottom, color='gray', linestyle='--', alpha=0.3)
        axes[0].text(0.02, 0.02, f'y-axis starts at {y_bottom:.0f}',
                     transform=axes[0].transAxes, fontsize=8, color='gray', style='italic')

    # Right: Pre vs Post finetune comparison
    width = 0.35
    bars_pre = axes[1].bar(x - width/2, ppl_pre, width, color=colors, alpha=0.35,
                           edgecolor='white', linewidth=1, label='Pre-finetune')
    bars_post = axes[1].bar(x + width/2, ppl_post, width, color=colors,
                            edgecolor='white', linewidth=1.5, label='Post-finetune')

    for bar, v in zip(bars_pre, ppl_pre):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{v:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold', alpha=0.6)
    for bar, v in zip(bars_post, ppl_post):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, fontsize=11)
    axes[1].set_ylabel('Perplexity (log scale)', fontsize=11)
    axes[1].set_yscale('log')
    axes[1].set_title('Pre vs Post Finetune', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pretrained_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')

    # --- Plot 2: Speed + Params overview ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Qwen3-0.6B — Efficiency Analysis',
                 fontsize=14, fontweight='bold', y=1.02)

    # Speed comparison
    speeds = []
    for mode in ['original', 'hopfield', 'augmented']:
        if mode not in data:
            continue
        d = data[mode]
        if 'post_finetune' in d:
            speeds.append(d['post_finetune']['tokens_per_sec'])
        else:
            speeds.append(d['pre_finetune']['tokens_per_sec'])

    bars = axes[0].bar(x, speeds, 0.55, color=colors, edgecolor='white', linewidth=1.5)
    min_spd = min(speeds)
    max_spd = max(speeds)
    spd_margin = (max_spd - min_spd) * 0.15
    axes[0].set_ylim(bottom=max(0, min_spd - spd_margin * 2), top=max_spd + spd_margin * 3)
    for bar, v in zip(bars, speeds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + spd_margin * 0.3,
                     f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, fontsize=11)
    axes[0].set_ylabel('Tokens/sec', fontsize=11)
    axes[0].set_title('Inference Speed', fontsize=12, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Param count
    params = []
    new_params = []
    for mode in ['original', 'hopfield', 'augmented']:
        if mode not in data:
            continue
        d = data[mode]
        params.append(d['params'] / 1e6)
        new_params.append(d['new_params'] / 1e6)

    bars_base = axes[1].bar(x, [p - n for p, n in zip(params, new_params)], 0.55,
                            color=colors, alpha=0.5, edgecolor='white', linewidth=1, label='Original params')
    bars_new = axes[1].bar(x, new_params, 0.55,
                           bottom=[p - n for p, n in zip(params, new_params)],
                           color=colors, edgecolor='white', linewidth=1.5, label='New params')

    min_p = min(params)
    max_p = max(params)
    p_margin = (max_p - min_p) * 0.15 if max_p > min_p else max_p * 0.05
    axes[1].set_ylim(bottom=max(0, min_p - p_margin * 3), top=max_p + p_margin * 3)
    for bar, v, nv in zip(bars_new, params, new_params):
        label_text = f'{v:.0f}M' if nv < 0.01 else f'{v:.0f}M (+{nv:.1f}M)'
        axes[1].text(bar.get_x() + bar.get_width()/2, v + p_margin * 0.3,
                     label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modes, fontsize=11)
    axes[1].set_ylabel('Parameters (M)', fontsize=11)
    axes[1].set_title('Model Size', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pretrained_efficiency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/experiment_results.json')
    parser.add_argument('--ablation', default=None, help='Path to ablation results JSON')
    parser.add_argument('--scaling', default=None, help='Path to scaling results JSON')
    parser.add_argument('--wikitext', default=None, help='Path to wikitext results JSON')
    parser.add_argument('--pretrained', default=None, help='Path to pretrained results JSON')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()

    if os.path.exists(args.results):
        plot_training_curves(args.results, args.output)
        plot_bar_comparison(args.results, args.output)
        plot_summary(args.results, args.output)

    if args.ablation:
        plot_ablation(args.ablation, args.output)

    if args.scaling:
        plot_scaling(args.scaling, args.output)

    if args.wikitext:
        plot_wikitext(args.wikitext, args.output)

    if args.pretrained:
        plot_pretrained(args.pretrained, args.output)

    print('\nAll plots generated.')
