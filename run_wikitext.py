"""
WikiText-2 Experiment: Real text language modeling

Downloads WikiText-2 and trains character-level LM to compare
vanilla, hopfield, and augmented transformers on real text data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import argparse
import urllib.request
import zipfile
from collections import defaultdict
from model import build_model


class WikiTextCharDataset(Dataset):
    """Character-level dataset from WikiText-2."""

    def __init__(self, text, seq_len=128, char2idx=None):
        self.seq_len = seq_len
        if char2idx is None:
            chars = sorted(set(text))
            self.char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for PAD
            self.char2idx['<unk>'] = 0
        else:
            self.char2idx = char2idx
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx) + 1

        # Encode text
        encoded = [self.char2idx.get(c, 0) for c in text]
        # Create sequences
        self.data = []
        for i in range(0, len(encoded) - seq_len - 1, seq_len // 2):
            chunk = encoded[i:i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.data.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def download_wikitext2(data_dir='data'):
    """Download and extract WikiText-2."""
    os.makedirs(data_dir, exist_ok=True)
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    zip_path = os.path.join(data_dir, 'wikitext-2-v1.zip')
    extract_dir = os.path.join(data_dir, 'wikitext-2')

    if os.path.exists(os.path.join(extract_dir, 'wiki.train.tokens')):
        print("WikiText-2 already downloaded.")
        return extract_dir

    print("Downloading WikiText-2...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    print("Done.")
    return extract_dir


def load_wikitext2(data_dir='data', seq_len=128, max_chars=500000):
    """Load WikiText-2 and create datasets."""
    wt_dir = download_wikitext2(data_dir)

    with open(os.path.join(wt_dir, 'wiki.train.tokens'), 'r', encoding='utf-8') as f:
        train_text = f.read()[:max_chars]
    with open(os.path.join(wt_dir, 'wiki.valid.tokens'), 'r', encoding='utf-8') as f:
        val_text = f.read()[:max_chars // 5]

    train_ds = WikiTextCharDataset(train_text, seq_len=seq_len)
    val_ds = WikiTextCharDataset(val_text, seq_len=seq_len, char2idx=train_ds.char2idx)

    return train_ds, val_ds, train_ds.vocab_size


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        result = model(inputs, targets=targets)
        loss = result.get('total_loss', result['loss'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        result = model(inputs, targets=targets)
        loss = result['loss']
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description='WikiText-2 Experiment')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--hopfield_steps', type=int, default=3)
    parser.add_argument('--max_chars', type=int, default=500000)
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Load data
    train_ds, val_ds, vocab_size = load_wikitext2(
        seq_len=args.seq_len, max_chars=args.max_chars
    )
    print(f"Vocab size: {vocab_size}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    modes = ['vanilla', 'hopfield', 'augmented']
    model_kwargs = dict(
        vocab_size=vocab_size, d_model=args.d_model, num_heads=4,
        d_ff=args.d_model * 2, num_layers=args.num_layers,
        max_seq_len=args.seq_len + 16, hopfield_steps=args.hopfield_steps,
        num_memories=32,
    )

    results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"  Mode: {mode}")
        print(f"{'='*50}")

        model = build_model(mode=mode, **model_kwargs).to(device)
        param_count = model.count_parameters()
        print(f"  Parameters: {param_count:,}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss = evaluate(model, val_loader, device)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if epoch % 5 == 0 or epoch == 1:
                bpc = val_loss / 0.6931  # bits per character (approx)
                print(f"  Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f} | bpc: {bpc:.3f}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s | Best val_loss: {best_val_loss:.4f}")

        results[mode] = {
            'params': param_count,
            'best_val_loss': best_val_loss,
            'final_val_loss': history['val_loss'][-1],
            'bpc': best_val_loss / 0.6931,
            'time': elapsed,
            'history': history,
        }

    # Save
    os.makedirs('results', exist_ok=True)
    with open('results/wikitext2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to results/wikitext2_results.json")

    # Summary
    print(f"\n{'='*60}")
    print(f"  WikiText-2 Summary")
    print(f"{'='*60}")
    print(f"  {'Mode':<15} {'Params':>10} {'Best Val Loss':>14} {'BPC':>8} {'Time':>8}")
    print(f"  {'-'*58}")
    for mode, r in results.items():
        print(f"  {mode:<15} {r['params']:>10,} {r['best_val_loss']:>14.4f} {r['bpc']:>8.3f} {r['time']:>7.1f}s")


if __name__ == '__main__':
    main()
