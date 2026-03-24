"""
rnn_att.py  —  Task 1, Model 3: RNN + Causal Attention
============================================================
Architecture : Embedding → Dropout(0.35)
               → RNN(H, 2 layers, dropout=0.35)
               → CausalAttention(H)
               → cat([rnn_out, context])  (B, T, 2H)
               → LayerNorm + Dropout(0.35)
               → Linear(2H, V)
Optimiser    : Adam  lr=0.001  weight_decay=1e-4
Scheduler    : CosineAnnealingLR  T_max=80  eta_min=1e-5
Epochs       : 80  |  Early-stop patience: 12
Saves        : models/rnn_attention.pt
               logs/training_log.txt  (appended)
               loss_rnn_att.json

NOTE: Single model trained on FULL names (e.g. "Rahul Sharma").
      The space between first and last name is treated as a regular
      character in the vocabulary — the model learns first name,
      the space, and last name all in one sequence.

Run:
    python rnn_att.py
"""

import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (DEVICE, Logger, load_full_names, train_model, count_params)

LOG_PATH   = "logs/training_log.txt"
sys.stdout = Logger(LOG_PATH)
print(f"[rnn_att.py]  Device: {DEVICE}  |  Log: {LOG_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class CausalAttention(nn.Module):
    """
    Bahdanau additive attention — CAUSAL (lower-triangular mask).

    At step t, context = weighted sum of RNN states 0..t.
    Future states masked to -inf before softmax.

    WHY causal masking is essential:
      Training processes full sequences (T tokens at once).
      Inference generates one token at a time.
      Without causal mask: training sees future chars → inference cannot →
      the model outputs repetition loops or garbage.
      With causal mask: same context at train AND inference → works correctly.
    """
    def __init__(self, H):
        super().__init__()
        self.Wq = nn.Linear(H, H, bias=False)
        self.Wk = nn.Linear(H, H, bias=False)
        self.v  = nn.Linear(H, 1, bias=False)

    def forward(self, Q, K):
        # Q, K: (B, T, H)
        scores = self.v(torch.tanh(
            self.Wq(Q).unsqueeze(2) + self.Wk(K).unsqueeze(1)
        )).squeeze(-1)                                         # (B, T, T)
        T      = scores.size(1)
        mask   = torch.tril(torch.ones(T, T, device=scores.device)).bool()
        scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
        alpha  = F.softmax(scores, dim=-1)                     # (B, T, T)
        return torch.bmm(alpha, K), alpha                      # (B, T, H)


class RNNWithAttention(nn.Module):
    """
    RNN with Causal Bahdanau Self-Attention — trained on full names as one sequence.

    The vocabulary includes the space character, so the model learns
    to generate "FirstName LastName" end-to-end in one pass.

    Architecture:
        Embedding → Dropout(0.35)
        → RNN(H, 2 layers, dropout=0.35)
        → CausalAttention(H)
        → cat([rnn_out, context])   (B, T, 2H)
        → LayerNorm + Dropout(0.35)
        → Linear(2H, V)

    Train/inference equivalence guaranteed by causal mask.

    Hyperparameters:
      Embedding dim  E = 64
      Hidden size    H = 256
      Layers         L = 2
      Dropout          = 0.35
      Optimiser      : Adam  lr=0.001  weight_decay=1e-4
      Scheduler      : CosineAnnealingLR  T_max=80  eta_min=1e-5
      Epochs           = 80
      Early-stop pat   = 12
    """
    def __init__(self, V, E=64, H=256, L=2):
        super().__init__()
        self.H, self.L = H, L
        self.emb      = nn.Embedding(V, E, padding_idx=0)
        self.drop_in  = nn.Dropout(0.35)
        self.rnn      = nn.RNN(E, H, num_layers=L, batch_first=True, dropout=0.35)
        self.attn     = CausalAttention(H)
        self.drop_out = nn.Dropout(0.35)
        self.norm     = nn.LayerNorm(H * 2)
        self.fc       = nn.Linear(H * 2, V)

    def forward(self, x, h=None):
        rnn_out, h = self.rnn(self.drop_in(self.emb(x)), h)
        ctx, _     = self.attn(rnn_out, rnn_out)
        out        = self.norm(self.drop_out(torch.cat([rnn_out, ctx], dim=-1)))
        return self.fc(out), h

    def init_h(self, B):
        return torch.zeros(self.L, B, self.H, device=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    full_names, ds, loader = load_full_names("TrainingNames.txt")

    E, H, L   = 64, 256, 2
    EPOCHS    = 80
    PATIENCE  = 12

    print("="*55)
    print("  TASK 1 — RNN + Attention  (single full-name model)")
    print("="*55)

    model = RNNWithAttention(ds.vocab_size, E, H, L).to(DEVICE)
    print(f"\nRNN+Attention  |  vocab={ds.vocab_size}  |  {count_params(model):,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    losses = train_model(model, loader, EPOCHS, optimizer, scheduler,
                         PATIENCE, "RNN+Attention", "rnn_attention")

    # ── Save loss history ─────────────────────────────────────────────────────
    with open("loss_rnn_att.json", "w") as f:
        json.dump({"RNN+Attention": losses}, f, indent=2)
    print("\nLoss history → loss_rnn_att.json")

    print(f"\n✓ rnn_att.py done.  Model: models/rnn_attention.pt")
    print(" \n \n Now  Run eval.py for generation, novelty & diversity metrics.\n")