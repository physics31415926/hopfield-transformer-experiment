"""
Visualization: plot experiment results as comparison charts.
Generates PNG figures for each experiment.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_experiment_results(results_path='results/experiment_results.json',
                            output_dir='results'):
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    colors = {'vanilla': '#4C72B0', 'hopfield': '#DD8452', 'augmented': '#55A868'}
    labels = {'vanilla': 'Vanilla Transformer', 'hopfield': 'Hopfield Attention', 'augmented': 'Hopfield + Memory'}

    for exp_name, exp_data in results.items():
        modes = list(exp_data.keys())

        # --- Bar chart: final metrics ---
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(f'Experiment: {exp_name}', fontsize=14, fontweight='bold')

        # Final loss
        ax = axes[0]
        vals = [exp_data[m].get('final_loss', 0) for m in modes]
        bars = ax.bar(modes, vals, color=[colors[m] for m in modes])
        ax.set_title('Final Loss')
        ax.set_ylabel('Loss')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9)

        # Final accuracy
        ax = axes[1]
        vals = [exp_data[m].get('final_acc', 0) * 100 for m in modes]
        bars = ax.bar(modes, vals, color=[colors[m] for m in modes])
        ax.set_title('Final Accuracy (%)')
        ax.set_ylabel('Accuracy (%)')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

        # Parameters
        ax = axes[2]
        vals = [exp_data[m].get('params', 0) for m in modes]
        bars = ax.bar(modes, vals, color=[colors[m] for m in modes])
        ax.set_title('Parameter Count')
        ax.set_ylabel('Parameters')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{v:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{exp_name}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {exp_name}_comparison.png')

    # --- Combined summary table ---
    fig, ax = plt.subplots(figsize=(10, 3 + len(results) * 1.2))
    ax.axis('off')

    table_data = []
    headers = ['Experiment', 'Model', 'Loss', 'Accuracy', 'Params', 'Time (s)']

    for exp_name, exp_data in results.items():
        for mode in exp_data:
            d = exp_data[mode]
            table_data.append([
                exp_name,
                labels.get(mode, mode),
                f"{d.get('final_loss', 0):.4f}",
                f"{d.get('final_acc', 0)*100:.1f}%",
                f"{d.get('params', 0):,}",
                f"{d.get('train_time', 0):.1f}",
            ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#f0f0f0']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color rows by model type
    for i, row in enumerate(table_data):
        mode_key = [k for k, v in labels.items() if v == row[1]]
        if mode_key:
            color = colors[mode_key[0]] + '30'  # light version
        for j in range(len(headers)):
            table[i+1, j].set_facecolor('#ffffff')

    fig.suptitle('Hopfield-Enhanced Transformer: Experiment Summary', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved summary_table.png')


if __name__ == '__main__':
    plot_experiment_results()
