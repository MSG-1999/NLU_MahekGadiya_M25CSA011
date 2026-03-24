import os, sys, json
import torch
import torch.nn as nn
from utils import (DEVICE, Logger, load_full_names, train_model, count_params)

LOG_PATH   = "logs/training_log.txt"
sys.stdout = Logger(LOG_PATH)
print(f"[blstm.py]  Device: {DEVICE}  |  Log: {LOG_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class BLSTM(nn.Module):

    def __init__(self, V, E=64, H=256, L=2):
        super().__init__()
        self.H, self.L = H, L
        self.emb  = nn.Embedding(V, E, padding_idx=0)
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(E, H, num_layers=L, batch_first=True, dropout=0.5)
        self.norm = nn.LayerNorm(H)
        self.fc   = nn.Linear(H, V)

    def forward(self, x, h=None):
        e      = self.drop(self.emb(x))
        out, h = self.lstm(e, h)
        out    = self.drop(self.norm(out))
        return self.fc(out), h

    def init_h(self, B):
        z = torch.zeros(self.L, B, self.H, device=DEVICE)
        return (z.clone(), z.clone())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    full_names, ds, loader = load_full_names("TrainingNames.txt")

    E, H, L   = 64, 256, 2
    EPOCHS    = 80
    PATIENCE  = 12

    print("="*55)
    print("  TASK 1 — BLSTM  (single full-name model)")
    print("="*55)

    model = BLSTM(ds.vocab_size, E, H, L).to(DEVICE)
    print(f"\nBLSTM  |  vocab={ds.vocab_size}  |  {count_params(model):,} trainable parameters")

    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=0.001, alpha=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=8, verbose=False)

    losses = train_model(model, loader, EPOCHS, optimizer, scheduler,
                         PATIENCE, "BLSTM", "blstm")

    # ── Save loss history ─────────────────────────────────────────────────────
    with open("loss_blstm.json", "w") as f:
        json.dump({"BLSTM": losses}, f, indent=2)
    print("\nLoss history → loss_blstm.json")

    print(f"\n✓ blstm.py done.  Model: models/blstm.pt")
    print(" \n Run rnn_att.py \n")