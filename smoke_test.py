import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.model import build_model

for mode in ['vanilla', 'hopfield', 'augmented']:
    m = build_model(mode, vocab_size=64, d_model=128, num_heads=4, d_ff=256, num_layers=2, max_seq_len=64).cuda()
    x = torch.randint(2, 64, (2, 32)).cuda()
    t = torch.randint(2, 64, (2, 32)).cuda()
    out = m(x, targets=t)
    p = m.count_parameters()
    loss = out['total_loss'].item()
    print(f"{mode:12s} | params: {p:>8,} | loss: {loss:.4f}")
print("Smoke test passed!")
