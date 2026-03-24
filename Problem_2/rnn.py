import os, sys, json
import torch
import torch.nn as nn
from utils import (DEVICE, Logger, load_full_names, train_model, count_params)

LOG_PATH   = "logs/training_log.txt"
sys.stdout = Logger(LOG_PATH)
print(f"[rnn.py]  Device: {DEVICE}  |  Log: {LOG_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class VanillaRNN(nn.Module):
    
    def __init__(self, V, E=64, H=256, L=2):
        super().__init__()
        self.H, self.L = H, L
        self.emb  = nn.Embedding(V, E, padding_idx=0)
        self.drop = nn.Dropout(0.5)
        self.rnn  = nn.RNN(E, H, num_layers=L, batch_first=True, dropout=0.5)
        self.fc   = nn.Linear(H, V)

    def forward(self, x, h=None):
        out, h = self.rnn(self.drop(self.emb(x)), h)
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
    print("  TASK 1 — Vanilla RNN  (single full-name model)")
    print("="*55)

    model = VanillaRNN(ds.vocab_size, E, H, L).to(DEVICE)
    print(f"\nVanilla RNN  |  vocab={ds.vocab_size}  |  {count_params(model):,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=8, verbose=False)

    losses = train_model(model, loader, EPOCHS, optimizer, scheduler,
                         PATIENCE, "Vanilla RNN", "vanilla_rnn")

    # ── Save loss history ─────────────────────────────────────────────────────
    with open("loss_rnn.json", "w") as f:
        json.dump({"Vanilla RNN": losses}, f, indent=2)
    print("\nLoss history → loss_rnn.json")

    print(f"\n✓ rnn.py done.  Model: models/vanilla_rnn.pt")
    print(" \n Run blstm.py \n")