"""
Pretrained Model Benchmark: Qwen3-0.6B with Hopfield Attention

Compares three variants on perplexity:
1. Original Qwen3-0.6B (baseline)
2. Hopfield Attention (multi-step iterative attention)
3. Hopfield + Memory (original attention + associative memory bank)

Downloads model via modelscope, evaluates on WikiText-2 or custom text.
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.hf_integration import patch_model_attention


def load_model_and_tokenizer(model_path, device='cuda', dtype=torch.bfloat16):
    """Load model and tokenizer from local path."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",  # Need eager for our custom attention
    )
    model = model.to(device)
    model.eval()
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_eval_text(data_path=None, max_tokens=50000):
    """Load evaluation text. Uses WikiText-2 if available."""
    # Try WikiText-2 from our data directory
    wt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'wikitext-2', 'wiki.valid.tokens')
    if os.path.exists(wt_path):
        print(f"Using WikiText-2 validation set: {wt_path}")
        with open(wt_path, 'r', encoding='utf-8') as f:
            return f.read()

    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return f.read()

    # Generate evaluation text as fallback
    print("No eval data found. Generating synthetic evaluation text...")
    import random
    random.seed(42)
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "people", "into", "year", "your", "good", "some", "could", "them",
        "system", "network", "model", "data", "learning", "function", "memory",
    ]
    text = []
    for _ in range(5000):
        sent_len = random.randint(5, 20)
        sent = ' '.join(random.choice(words) for _ in range(sent_len))
        text.append(sent[0].upper() + sent[1:] + '.')
    return ' '.join(text)


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, text, max_length=512, stride=256,
                        device='cuda', max_tokens=50000):
    """Evaluate perplexity using sliding window."""
    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings.input_ids[0]

    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    total_len = len(input_ids)
    print(f"  Evaluating on {total_len} tokens (window={max_length}, stride={stride})...")

    nlls = []
    total_tokens = 0
    t0 = time.time()

    for begin in range(0, total_len - 1, stride):
        end = min(begin + max_length, total_len)
        target_begin = max(begin, 1) if begin == 0 else begin + (max_length - stride)

        input_chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :target_begin - begin] = -100  # Mask non-target tokens

        outputs = model(input_chunk, labels=target_chunk)
        neg_log_likelihood = outputs.loss

        num_tokens = (target_chunk != -100).sum().item()
        nlls.append(neg_log_likelihood.item() * num_tokens)
        total_tokens += num_tokens

        if end >= total_len:
            break

    elapsed = time.time() - t0
    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    tokens_per_sec = total_tokens / elapsed

    return {
        'perplexity': ppl,
        'avg_nll': avg_nll,
        'total_tokens': total_tokens,
        'time': elapsed,
        'tokens_per_sec': tokens_per_sec,
    }


@torch.no_grad()
def evaluate_perplexity_finetune(model, tokenizer, text, max_length=512,
                                  device='cuda', max_tokens=50000):
    """Simple perplexity evaluation for fine-tuned models."""
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    input_ids = encodings.input_ids[0]
    total_len = len(input_ids)

    nlls = []
    total_tokens = 0
    t0 = time.time()

    for i in range(0, total_len - 1, max_length):
        chunk = input_ids[i:i + max_length].unsqueeze(0).to(device)
        if chunk.size(1) < 2:
            continue
        targets = chunk.clone()
        outputs = model(chunk, labels=targets)
        num_tokens = chunk.size(1) - 1  # First token has no prediction
        nlls.append(outputs.loss.item() * num_tokens)
        total_tokens += num_tokens

    elapsed = time.time() - t0
    avg_nll = sum(nlls) / max(total_tokens, 1)
    ppl = math.exp(min(avg_nll, 100))  # Cap to avoid overflow

    return {
        'perplexity': ppl,
        'avg_nll': avg_nll,
        'total_tokens': total_tokens,
        'time': elapsed,
        'tokens_per_sec': total_tokens / max(elapsed, 0.001),
    }


def finetune(model, tokenizer, text, epochs=3, lr=2e-5, max_length=512,
             batch_size=4, device='cuda', max_tokens=100000):
    """Light fine-tuning on text data."""
    from torch.utils.data import Dataset, DataLoader

    class TextDataset(Dataset):
        def __init__(self, input_ids, max_length):
            self.chunks = []
            for i in range(0, len(input_ids) - max_length, max_length):
                self.chunks.append(input_ids[i:i + max_length])

        def __len__(self):
            return len(self.chunks)

        def __getitem__(self, idx):
            chunk = self.chunks[idx]
            return chunk, chunk.clone()

    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings.input_ids[0][:max_tokens]

    dataset = TextDataset(input_ids, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train new Hopfield params + attention projections in patched layers
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        # Train: Hopfield-specific params + all params inside patched attention layers
        is_hopfield = ('log_beta' in name or 'memory_bank' in name or 'hopfield' in name)
        is_patched_attn = ('self_attn.original.' in name or 'self_attn.memory_bank.' in name)
        if is_hopfield or is_patched_attn:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_params.append(param)

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Fine-tuning {n_trainable:,} / {n_total:,} parameters ({100*n_trainable/n_total:.2f}%)")

    if n_trainable == 0:
        print("  No trainable parameters (original model). Skipping fine-tune.")
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        n_batches = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}")

    model.eval()


def main():
    parser = argparse.ArgumentParser(description='Pretrained Model Benchmark')
    parser.add_argument('--model_path', type=str,
                        default='/root/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B',
                        help='Path to pretrained model')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--eval_data', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=50000)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--hopfield_steps', type=int, default=3)
    parser.add_argument('--num_memories', type=int, default=64)
    parser.add_argument('--finetune', action='store_true', help='Fine-tune new params')
    parser.add_argument('--finetune_epochs', type=int, default=3)
    parser.add_argument('--finetune_lr', type=float, default=2e-5)
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['original', 'hopfield', 'augmented'])
    parser.add_argument('--patch_layers', type=str, default=None,
                        help='Comma-separated layer indices to patch (default: all)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Load eval text
    eval_text = load_eval_text(args.eval_data, args.max_tokens)
    print(f"Eval text: {len(eval_text)} chars")

    # Load train text for fine-tuning
    train_text = None
    if args.finetune:
        wt_train = os.path.join(os.path.dirname(__file__), '..', 'data', 'wikitext-2', 'wiki.train.tokens')
        if os.path.exists(wt_train):
            with open(wt_train, 'r', encoding='utf-8') as f:
                train_text = f.read()
        else:
            train_text = eval_text  # Fallback

    patch_layers = None
    if args.patch_layers:
        patch_layers = [int(x) for x in args.patch_layers.split(',')]

    results = {}

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")

        # Load fresh model for each mode
        model, tokenizer = load_model_and_tokenizer(args.model_path, device)

        if mode == 'hopfield':
            model, n_patched = patch_model_attention(
                model, mode='hopfield',
                num_steps=args.hopfield_steps,
                layers=patch_layers,
            )
            print(f"  Patched {n_patched} layers with Hopfield attention (T={args.hopfield_steps})")
        elif mode == 'augmented':
            model, n_patched = patch_model_attention(
                model, mode='augmented',
                num_steps=args.hopfield_steps,
                num_memories=args.num_memories,
                layers=patch_layers,
            )
            print(f"  Patched {n_patched} layers with Hopfield + Memory")

        # Ensure model is on the right device
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters())
        new_params = sum(p.numel() for n, p in model.named_parameters()
                        if 'log_beta' in n or 'memory_bank' in n or 'hopfield' in n)
        print(f"  Total params: {param_count:,} (new: {new_params:,})")

        # Evaluate before fine-tuning
        print("\n  --- Eval (pre-finetune) ---")
        eval_result = evaluate_perplexity(
            model, tokenizer, eval_text,
            max_length=args.max_length, stride=args.stride,
            device=device, max_tokens=args.max_tokens,
        )
        print(f"  PPL: {eval_result['perplexity']:.2f} | "
              f"NLL: {eval_result['avg_nll']:.4f} | "
              f"Speed: {eval_result['tokens_per_sec']:.0f} tok/s | "
              f"Time: {eval_result['time']:.1f}s")

        result_entry = {
            'mode': mode,
            'params': param_count,
            'new_params': new_params,
            'pre_finetune': eval_result,
        }

        # Fine-tune if requested
        if args.finetune and mode != 'original' and train_text:
            print(f"\n  --- Fine-tuning ({args.finetune_epochs} epochs) ---")
            finetune(
                model, tokenizer, train_text,
                epochs=args.finetune_epochs, lr=args.finetune_lr,
                max_length=args.max_length, device=device,
            )

            print("\n  --- Eval (post-finetune) ---")
            eval_post = evaluate_perplexity(
                model, tokenizer, eval_text,
                max_length=args.max_length, stride=args.stride,
                device=device, max_tokens=args.max_tokens,
            )
            print(f"  PPL: {eval_post['perplexity']:.2f} | "
                  f"NLL: {eval_post['avg_nll']:.4f} | "
                  f"Speed: {eval_post['tokens_per_sec']:.0f} tok/s")
            result_entry['post_finetune'] = eval_post

        results[mode] = result_entry

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pretrained_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Pretrained Model Benchmark Summary (Qwen3-0.6B)")
    print(f"{'='*70}")
    print(f"  {'Mode':<15} {'Params':>12} {'New Params':>12} {'PPL':>10} {'NLL':>10} {'Speed':>12}")
    print(f"  {'-'*65}")
    for mode, r in results.items():
        pre = r['pre_finetune']
        print(f"  {mode:<15} {r['params']:>12,} {r['new_params']:>12,} "
              f"{pre['perplexity']:>10.2f} {pre['avg_nll']:>10.4f} "
              f"{pre['tokens_per_sec']:>10.0f} t/s")
        if 'post_finetune' in r:
            post = r['post_finetune']
            print(f"  {'  (finetuned)':<15} {'':>12} {'':>12} "
                  f"{post['perplexity']:>10.2f} {post['avg_nll']:>10.4f} "
                  f"{post['tokens_per_sec']:>10.0f} t/s")


if __name__ == '__main__':
    main()
