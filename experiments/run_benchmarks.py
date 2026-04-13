"""
Public Benchmark Evaluation: Qwen3-0.6B with Hopfield Attention

Evaluates three variants on standard NLP benchmarks:
1. WikiText-103 — perplexity (standard LM benchmark)
2. LAMBADA — accuracy + perplexity (long-range dependency, last-word prediction)
3. PTB (Penn Treebank) — perplexity (classic LM benchmark)

All datasets loaded via HuggingFace datasets library.
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.hf_integration import patch_model_attention, count_new_parameters


def load_model(model_path, device='cuda', dtype=torch.bfloat16):
    """Load Qwen3-0.6B model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Parameters: {n_params:,}")
    return model, tokenizer


def load_benchmark_data(benchmark_name):
    """Load benchmark dataset from HuggingFace."""
    print(f"Loading {benchmark_name} dataset...")

    if benchmark_name == 'wikitext-103':
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        text = '\n\n'.join([t for t in ds['text'] if t.strip()])
        return {'type': 'perplexity', 'text': text, 'name': 'WikiText-103'}

    elif benchmark_name == 'ptb':
        ds = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        text = '\n'.join(ds['sentence'])
        return {'type': 'perplexity', 'text': text, 'name': 'PTB'}

    elif benchmark_name == 'lambada':
        ds = load_dataset('lambada', split='test')
        return {'type': 'lambada', 'examples': ds, 'name': 'LAMBADA'}

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


@torch.no_grad()
def eval_perplexity(model, tokenizer, text, max_length=512, stride=256,
                    device='cuda', max_tokens=None):
    """Evaluate perplexity using sliding window on text."""
    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings.input_ids[0]

    if max_tokens and len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    total_tokens = len(input_ids)
    print(f"  Evaluating perplexity on {total_tokens:,} tokens (stride={stride})...")

    nlls = []
    t0 = time.time()

    for begin in range(0, total_tokens - 1, stride):
        end = min(begin + max_length, total_tokens)
        input_chunk = input_ids[begin:end].unsqueeze(0).to(device)

        outputs = model(input_chunk)
        logits = outputs.logits

        # Only count loss for tokens in the stride window (avoid double-counting)
        shift_start = 0 if begin == 0 else max_length - stride
        shift_logits = logits[0, shift_start:-1, :]
        shift_labels = input_chunk[0, shift_start + 1:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        nlls.append(loss)

        if begin > 0 and begin % (stride * 50) == 0:
            elapsed = time.time() - t0
            progress = begin / total_tokens * 100
            print(f"    {progress:.0f}% ({begin:,}/{total_tokens:,}) - {elapsed:.1f}s")

    elapsed = time.time() - t0
    all_nlls = torch.cat(nlls)
    avg_nll = all_nlls.mean().item()
    ppl = math.exp(avg_nll)
    tokens_evaluated = len(all_nlls)

    print(f"  Done in {elapsed:.1f}s. PPL={ppl:.2f}, NLL={avg_nll:.4f}, "
          f"tokens={tokens_evaluated:,}, speed={tokens_evaluated/elapsed:.0f} t/s")

    return {
        'perplexity': ppl,
        'avg_nll': avg_nll,
        'tokens_evaluated': tokens_evaluated,
        'time': elapsed,
        'tokens_per_sec': tokens_evaluated / elapsed,
    }


@torch.no_grad()
def eval_lambada(model, tokenizer, examples, device='cuda', max_examples=None):
    """Evaluate LAMBADA: predict the last word of each passage."""
    correct = 0
    total = 0
    total_nll = 0.0
    total_tokens = 0

    n = len(examples)
    if max_examples:
        n = min(n, max_examples)

    print(f"  Evaluating LAMBADA on {n} examples...")
    t0 = time.time()

    for i in range(n):
        text = examples[i]['text']

        # Split into context + last word
        words = text.rsplit(' ', 1)
        if len(words) != 2:
            continue
        context, last_word = words

        # Tokenize full text and context separately
        full_ids = tokenizer(text, return_tensors='pt').input_ids[0]
        ctx_ids = tokenizer(context, return_tensors='pt').input_ids[0]

        # The last word tokens are the difference
        n_last = len(full_ids) - len(ctx_ids)
        if n_last <= 0:
            continue

        input_ids = full_ids.unsqueeze(0).to(device)
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab)

        # Get predictions for the last word positions
        last_word_start = len(ctx_ids) - 1  # -1 because we predict next token
        pred_logits = logits[last_word_start:last_word_start + n_last]
        target_ids = full_ids[len(ctx_ids):len(ctx_ids) + n_last].to(device)

        # NLL for the last word
        loss = F.cross_entropy(pred_logits, target_ids, reduction='sum')
        total_nll += loss.item()
        total_tokens += n_last

        # Accuracy: check if greedy prediction matches
        pred_ids = pred_logits.argmax(dim=-1)
        if torch.equal(pred_ids, target_ids):
            correct += 1
        total += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            acc = correct / total * 100 if total > 0 else 0
            print(f"    {i+1}/{n} - acc={acc:.1f}% - {elapsed:.1f}s")

    elapsed = time.time() - t0
    accuracy = correct / total * 100 if total > 0 else 0
    avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_nll) if avg_nll < 100 else float('inf')

    print(f"  Done in {elapsed:.1f}s. Accuracy={accuracy:.2f}%, PPL={ppl:.2f}")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'perplexity': ppl,
        'avg_nll': avg_nll,
        'time': elapsed,
    }


def finetune(model, tokenizer, text, device='cuda', epochs=5, lr=1e-4,
             max_length=512, batch_tokens=4096):
    """Light fine-tune: train only Hopfield + patched attention params."""
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(k in name for k in ['log_beta', 'memory_bank', 'gate',
                                     'self_attn.original.']):
            param.requires_grad = True
            trainable.append(param)

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Fine-tuning: {n_trainable:,} / {n_total:,} params "
          f"({n_trainable/n_total*100:.2f}%)")

    if n_trainable == 0:
        print("  No trainable params, skipping fine-tune.")
        return model

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # Tokenize training text
    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings.input_ids[0]
    if len(input_ids) > 100000:
        input_ids = input_ids[:100000]

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_chunks = 0
        perm = torch.randperm(max(1, len(input_ids) - max_length))

        for i in range(0, min(len(perm), 200), 1):  # Cap at 200 chunks/epoch
            start = perm[i].item()
            chunk = input_ids[start:start + max_length].unsqueeze(0).to(device)
            if chunk.shape[1] < 2:
                continue

            outputs = model(chunk, labels=chunk)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_chunks += 1

        avg_loss = total_loss / max(n_chunks, 1)
        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({n_chunks} chunks)")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Public Benchmark Evaluation')
    parser.add_argument('--model_path', type=str,
                        default='/root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B',
                        help='Path to Qwen3-0.6B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--benchmarks', type=str, default='wikitext-103,lambada,ptb',
                        help='Comma-separated benchmark names')
    parser.add_argument('--modes', type=str, default='original,hopfield,augmented',
                        help='Comma-separated modes to evaluate')
    parser.add_argument('--patch_layers', type=str, default='24,25,26,27',
                        help='Comma-separated layer indices to patch')
    parser.add_argument('--num_steps', type=int, default=3)
    parser.add_argument('--num_memories', type=int, default=64)
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune patched layers before eval')
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Max tokens for perplexity eval (None=all)')
    parser.add_argument('--max_lambada', type=int, default=None,
                        help='Max LAMBADA examples (None=all)')
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',')]
    modes = [m.strip() for m in args.modes.split(',')]
    patch_layers = [int(x) for x in args.patch_layers.split(',')]

    # Pre-load all benchmark data
    bench_data = {}
    for bname in benchmarks:
        bench_data[bname] = load_benchmark_data(bname)

    # Load finetune text if needed
    ft_text = None
    if args.finetune:
        print("Loading fine-tune data (WikiText-103 train)...")
        ft_ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        ft_text = '\n\n'.join([t for t in ft_ds['text'] if t.strip()])

    results = {}

    for mode in modes:
        print(f"\n{'='*70}")
        print(f"  Mode: {mode}")
        print(f"{'='*70}")

        # Load fresh model
        model, tokenizer = load_model(args.model_path, device=args.device)
        n_total = sum(p.numel() for p in model.parameters())

        # Patch if needed
        n_new = 0
        if mode != 'original':
            model, n_patched = patch_model_attention(
                model, mode=mode, num_steps=args.num_steps,
                layers=patch_layers, num_memories=args.num_memories,
            )
            n_new = count_new_parameters(model)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"  Patched {n_patched} layers. New params: {n_new:,}, Total: {n_total:,}")

        mode_results = {
            'mode': mode,
            'params': n_total,
            'new_params': n_new,
            'patch_layers': patch_layers if mode != 'original' else [],
        }

        # Fine-tune if requested
        if args.finetune and mode != 'original' and ft_text:
            print(f"\n  Fine-tuning {mode}...")
            model = finetune(model, tokenizer, ft_text, device=args.device,
                             epochs=args.finetune_epochs)

        # Evaluate on each benchmark
        for bname in benchmarks:
            bd = bench_data[bname]
            print(f"\n  --- {bd['name']} ---")

            if bd['type'] == 'perplexity':
                result = eval_perplexity(
                    model, tokenizer, bd['text'],
                    device=args.device, max_tokens=args.max_tokens,
                )
            elif bd['type'] == 'lambada':
                result = eval_lambada(
                    model, tokenizer, bd['examples'],
                    device=args.device, max_examples=args.max_lambada,
                )

            mode_results[bname] = result

        results[mode] = mode_results

        del model
        torch.cuda.empty_cache()

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Public Benchmark Results — Qwen3-0.6B")
    print(f"{'='*80}")

    for bname in benchmarks:
        bd = bench_data[bname]
        print(f"\n  {bd['name']}:")
        if bd['type'] == 'perplexity':
            print(f"  {'Mode':<20} {'PPL':>10} {'NLL':>10} {'Speed':>12}")
            print(f"  {'-'*55}")
            for mode in modes:
                if bname in results[mode]:
                    r = results[mode][bname]
                    print(f"  {mode:<20} {r['perplexity']:>10.2f} {r['avg_nll']:>10.4f} "
                          f"{r['tokens_per_sec']:>10.0f} t/s")
        elif bd['type'] == 'lambada':
            print(f"  {'Mode':<20} {'Accuracy':>10} {'PPL':>10}")
            print(f"  {'-'*45}")
            for mode in modes:
                if bname in results[mode]:
                    r = results[mode][bname]
                    print(f"  {mode:<20} {r['accuracy']:>9.2f}% {r['perplexity']:>10.2f}")


if __name__ == '__main__':
    main()
