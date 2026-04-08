"""
Experiments to validate Hopfield-Enhanced Transformer

Experiment 1: Associative Recall — synthetic task testing pattern retrieval
Experiment 2: Copy/Reverse — sequence manipulation requiring precise memory
Experiment 3: Language Modeling — character-level LM on real text (WikiText-like)

Each experiment compares: vanilla, hopfield (multi-step attn), augmented (attn + memory)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import argparse
from collections import defaultdict
from model import build_model


# ============================================================
# Experiment 1: Associative Recall
# ============================================================
# Task: Given key-value pairs followed by a query key, predict the value.
# e.g., input: [k1 v1 k2 v2 k3 v3 ? k2] -> output: v2
# This directly tests associative memory capability.

class AssociativeRecallDataset(Dataset):
    """
    Sequences: [k1, v1, k2, v2, ..., kN, vN, QUERY_TOKEN, ki]
    Target: vi (the value associated with the queried key)
    """

    def __init__(self, num_samples: int, num_pairs: int = 8,
                 vocab_size: int = 64, seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        # Reserve tokens: 0=PAD, 1=QUERY
        self.key_range = (2, vocab_size // 2)
        self.val_range = (vocab_size // 2, vocab_size)

        rng = torch.Generator().manual_seed(seed)
        self.data = []
        self.targets = []

        for _ in range(num_samples):
            keys = torch.randint(self.key_range[0], self.key_range[1],
                                 (num_pairs,), generator=rng)
            vals = torch.randint(self.val_range[0], self.val_range[1],
                                 (num_pairs,), generator=rng)
            # Build sequence: k1 v1 k2 v2 ... kN vN QUERY ki
            seq = []
            for k, v in zip(keys, vals):
                seq.extend([k.item(), v.item()])

            # Pick a random pair to query
            query_idx = torch.randint(0, num_pairs, (1,), generator=rng).item()
            seq.append(1)  # QUERY token
            seq.append(keys[query_idx].item())

            self.data.append(torch.tensor(seq, dtype=torch.long))
            self.targets.append(vals[query_idx].item())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def collate_recall(batch):
    seqs, targets = zip(*batch)
    seqs = torch.stack(seqs)
    targets = torch.tensor(targets, dtype=torch.long)
    return seqs, targets


# ============================================================
# Experiment 2: Sequence Copy with Noise
# ============================================================
# Task: Given a sequence with some tokens masked, reconstruct the full sequence.
# Tests the model's ability to use context for pattern completion — core Hopfield.

class NoisyCopyDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 32,
                 vocab_size: int = 64, mask_ratio: float = 0.15, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = []
        self.targets = []
        mask_token = 0

        for _ in range(num_samples):
            seq = torch.randint(2, vocab_size, (seq_len,), generator=rng)
            target = seq.clone()
            # Mask some positions
            mask = torch.rand(seq_len, generator=rng) < mask_ratio
            noisy = seq.clone()
            noisy[mask] = mask_token
            self.data.append(noisy)
            self.targets.append(target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================================================
# Experiment 3: Character-level Language Modeling
# ============================================================

class CharLMDataset(Dataset):
    """Character-level LM on synthetic structured text."""

    def __init__(self, num_samples: int, seq_len: int = 64, vocab_size: int = 64, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = []

        # Generate structured sequences with repeating patterns
        # This tests whether Hopfield memory helps capture long-range repetitions
        for _ in range(num_samples):
            # Create a pattern and repeat it with variations
            pattern_len = torch.randint(4, 12, (1,), generator=rng).item()
            pattern = torch.randint(2, vocab_size, (pattern_len,), generator=rng)

            seq = []
            while len(seq) < seq_len + 1:
                # Sometimes repeat the pattern, sometimes add noise
                if torch.rand(1, generator=rng).item() < 0.7:
                    seq.extend(pattern.tolist())
                else:
                    noise_len = torch.randint(1, 5, (1,), generator=rng).item()
                    noise = torch.randint(2, vocab_size, (noise_len,), generator=rng)
                    seq.extend(noise.tolist())

            seq = torch.tensor(seq[:seq_len + 1], dtype=torch.long)
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target


# ============================================================
# Training & Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device, task='lm'):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        if task == 'recall':
            seqs, targets = batch
            seqs, targets = seqs.to(device), targets.to(device)
            result = model(seqs)
            # Only care about the last position prediction
            logits = result['logits'][:, -1, :]  # (B, V)
            loss = nn.functional.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.size(0)
        elif task == 'copy':
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            result = model(inputs, targets=targets)
            loss = result.get('total_loss', result['loss'])
            preds = result['logits'].argmax(dim=-1)
            # Only count masked positions (where input == 0)
            mask = (inputs == 0)
            if mask.any():
                total_correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()
        else:  # lm
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            result = model(inputs, targets=targets)
            loss = result.get('total_loss', result['loss'])
            total_tokens += targets.numel()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * (targets.size(0) if task == 'recall' else targets.numel())

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1) if total_tokens > 0 else 0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device, task='lm'):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in loader:
        if task == 'recall':
            seqs, targets = batch
            seqs, targets = seqs.to(device), targets.to(device)
            result = model(seqs)
            logits = result['logits'][:, -1, :]
            loss = nn.functional.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.size(0)
        elif task == 'copy':
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            result = model(inputs, targets=targets)
            loss = result.get('total_loss', result['loss'])
            preds = result['logits'].argmax(dim=-1)
            mask = (inputs == 0)
            if mask.any():
                total_correct += (preds[mask] == targets[mask]).sum().item()
                total_tokens += mask.sum().item()
        else:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            result = model(inputs, targets=targets)
            loss = result['loss']
            total_tokens += targets.numel()

        total_loss += loss.item() * (targets.size(0) if task == 'recall' else targets.numel())

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1) if total_tokens > 0 else 0
    return avg_loss, accuracy


def run_experiment(exp_name, task, dataset_cls, dataset_kwargs,
                   model_kwargs, modes, num_epochs, batch_size, lr, device):
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*60}")

    results = {}

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        # Build datasets
        train_ds = dataset_cls(seed=42, **dataset_kwargs)
        val_ds = dataset_cls(seed=123, **{**dataset_kwargs, 'num_samples': dataset_kwargs['num_samples'] // 5})

        collate_fn = collate_recall if task == 'recall' else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

        # Build model
        model = build_model(mode=mode, **model_kwargs).to(device)
        param_count = model.count_parameters()
        print(f"  Parameters: {param_count:,}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, task)
            val_loss, val_acc = evaluate(model, val_loader, device, task)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if epoch % 5 == 0 or epoch == 1:
                msg = f"  Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
                if task in ('recall', 'copy'):
                    msg += f" | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}"
                print(msg)

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s | Best val_loss: {best_val_loss:.4f}")

        results[mode] = {
            'params': param_count,
            'best_val_loss': best_val_loss,
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1] if task in ('recall', 'copy') else None,
            'time': elapsed,
            'history': history,
        }

    return results


def print_summary(results, task):
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<15} {'Params':>10} {'Best Val Loss':>14} {'Final Val Acc':>14} {'Time':>8}")
    print(f"  {'-'*61}")
    for mode, r in results.items():
        acc_str = f"{r['final_val_acc']:.4f}" if r['final_val_acc'] is not None else "N/A"
        print(f"  {mode:<15} {r['params']:>10,} {r['best_val_loss']:>14.4f} {acc_str:>14} {r['time']:>7.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Hopfield Transformer Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'recall', 'copy', 'lm'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hopfield_steps', type=int, default=3)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    modes = ['vanilla', 'hopfield', 'augmented']
    all_results = {}

    common_model_kwargs = dict(
        vocab_size=64, d_model=args.d_model, num_heads=4,
        d_ff=args.d_model * 2, num_layers=args.num_layers,
        max_seq_len=256, hopfield_steps=args.hopfield_steps,
        num_memories=32,
    )

    # Experiment 1: Associative Recall
    if args.experiment in ('all', 'recall'):
        results = run_experiment(
            exp_name="Associative Recall",
            task='recall',
            dataset_cls=AssociativeRecallDataset,
            dataset_kwargs=dict(num_samples=5000, num_pairs=8, vocab_size=64),
            model_kwargs=common_model_kwargs,
            modes=modes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        print_summary(results, 'recall')
        all_results['recall'] = results

    # Experiment 2: Noisy Copy (Pattern Completion)
    if args.experiment in ('all', 'copy'):
        results = run_experiment(
            exp_name="Noisy Copy (Pattern Completion)",
            task='copy',
            dataset_cls=NoisyCopyDataset,
            dataset_kwargs=dict(num_samples=5000, seq_len=32, vocab_size=64, mask_ratio=0.15),
            model_kwargs=common_model_kwargs,
            modes=modes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        print_summary(results, 'copy')
        all_results['copy'] = results

    # Experiment 3: Structured Sequence LM
    if args.experiment in ('all', 'lm'):
        results = run_experiment(
            exp_name="Structured Sequence Language Modeling",
            task='lm',
            dataset_cls=CharLMDataset,
            dataset_kwargs=dict(num_samples=5000, seq_len=64, vocab_size=64),
            model_kwargs=common_model_kwargs,
            modes=modes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        print_summary(results, 'lm')
        all_results['lm'] = results

    # Save results (including history for training curves)
    save_results = {}
    for exp_name, exp_results in all_results.items():
        save_results[exp_name] = {}
        for mode, r in exp_results.items():
            save_results[exp_name][mode] = {
                k: v for k, v in r.items()
            }

    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to results/experiment_results.json")


if __name__ == '__main__':
    main()
