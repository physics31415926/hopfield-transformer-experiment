"""
Scaling Experiment: Effect of model size on performance

Tests d_model = 64, 128, 256, 512 across all three model variants
on the associative recall and LM tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.run_synthetic import (
    AssociativeRecallDataset, CharLMDataset,
    collate_recall, train_epoch, evaluate
)
from src.model import build_model


def run_scaling(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    d_models = [64, 128, 256, 512]
    modes = ['vanilla', 'hopfield', 'augmented']

    experiments = {}
    if args.experiment in ('all', 'recall'):
        experiments['recall'] = {
            'task': 'recall',
            'cls': AssociativeRecallDataset,
            'kwargs': dict(num_samples=5000, num_pairs=8, vocab_size=64),
        }
    if args.experiment in ('all', 'lm'):
        experiments['lm'] = {
            'task': 'lm',
            'cls': CharLMDataset,
            'kwargs': dict(num_samples=5000, seq_len=64, vocab_size=64),
        }

    all_results = {}

    for exp_name, exp_cfg in experiments.items():
        print(f"\n{'='*60}")
        print(f"  Scaling: {exp_name}")
        print(f"{'='*60}")

        task = exp_cfg['task']
        train_ds = exp_cfg['cls'](seed=42, **exp_cfg['kwargs'])
        val_ds = exp_cfg['cls'](seed=123, **{**exp_cfg['kwargs'], 'num_samples': exp_cfg['kwargs']['num_samples'] // 5})

        collate_fn = collate_recall if task == 'recall' else None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

        exp_results = {}

        for mode in modes:
            for d_model in d_models:
                key = f"{mode}_d{d_model}"
                print(f"\n--- {key} ---")

                model_kwargs = dict(
                    vocab_size=64, d_model=d_model, num_heads=4,
                    d_ff=d_model * 2, num_layers=args.num_layers,
                    max_seq_len=256, hopfield_steps=3, num_memories=32,
                )

                model = build_model(mode=mode, **model_kwargs).to(device)
                param_count = model.count_parameters()
                print(f"  Parameters: {param_count:,}")

                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
                best_val_loss = float('inf')

                t0 = time.time()
                for epoch in range(1, args.epochs + 1):
                    train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, task)
                    val_loss, val_acc = evaluate(model, val_loader, device, task)
                    scheduler.step()

                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    if epoch % 10 == 0 or epoch == 1:
                        msg = f"  Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
                        if task == 'recall':
                            msg += f" | val_acc: {val_acc:.4f}"
                        print(msg)

                elapsed = time.time() - t0
                print(f"  Time: {elapsed:.1f}s | Best val_loss: {best_val_loss:.4f}")

                exp_results[key] = {
                    'mode': mode,
                    'd_model': d_model,
                    'params': param_count,
                    'best_val_loss': best_val_loss,
                    'final_val_loss': history['val_loss'][-1],
                    'final_val_acc': history['val_acc'][-1] if task == 'recall' else None,
                    'time': elapsed,
                    'history': history,
                }

        all_results[exp_name] = exp_results

    # Save
    os.makedirs('results', exist_ok=True)
    out_path = 'results/scaling_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nScaling results saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Scaling Summary")
    print(f"{'='*70}")
    for exp_name, exp_results in all_results.items():
        print(f"\n  {exp_name}:")
        print(f"  {'Config':<25} {'Params':>10} {'Best Val Loss':>14} {'Time':>8}")
        print(f"  {'-'*60}")
        for key, r in sorted(exp_results.items()):
            print(f"  {key:<25} {r['params']:>10,} {r['best_val_loss']:>14.4f} {r['time']:>7.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Scaling Experiment')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'recall', 'lm'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_layers', type=int, default=3)
    args = parser.parse_args()

    run_scaling(args)
