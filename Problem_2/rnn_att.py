import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (DEVICE, Logger, load_full_names, train_model, count_params)

# logging everything into a file so we can check training later
LOG_PATH   = "logs/training_log.txt"
sys.stdout = Logger(LOG_PATH)

# printing device information
print(f"[rnn_att.py]  Device: {DEVICE}  |  Log: {LOG_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class CausalAttention(nn.Module):
    
    """
    This is Bahdanau-style attention with causal masking.

    Simple idea:
    At each time step, model looks at previous hidden states
    and decides which ones are important.

    Important:
    We use causal mask so model cannot see future characters.
    Without this, training and inference mismatch happens.
    """
    
    def __init__(self, H):
        super().__init__()
        
        # linear layers to compute attention scores
        self.Wq = nn.Linear(H, H, bias=False)
        self.Wk = nn.Linear(H, H, bias=False)
        self.v  = nn.Linear(H, 1, bias=False)

    def forward(self, Q, K):
        # Q, K: (B, T, H)
        
        # computing attention scores 
        scores = self.v(torch.tanh(
            self.Wq(Q).unsqueeze(2) + self.Wk(K).unsqueeze(1)
        )).squeeze(-1)                                         # (B, T, T)
        T      = scores.size(1)
        
        # creating lower triangular mask (causal)
        mask   = torch.tril(torch.ones(T, T, device=scores.device)).bool()
        
        # masking future positions
        scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
        
        # converting scores to probabilities (applying softmax)
        alpha  = F.softmax(scores, dim=-1)       
        
        # computing context vector
        return torch.bmm(alpha, K), alpha                    


class RNNWithAttention(nn.Module):
    
    def __init__(self, V, E=64, H=256, L=2):
        super().__init__()
        
        # saving hyperparameters
        self.H, self.L = H, L
        
        # embedding layer (characters --> vectors)
        self.emb      = nn.Embedding(V, E, padding_idx=0)
        
        # input dropout (slightly less than other models)
        self.drop_in  = nn.Dropout(0.35)
        
        # RNN layer (same as vanilla but combined with attention)
        self.rnn      = nn.RNN(E, H, num_layers=L, batch_first=True, dropout=0.35)
        
        # attention module
        self.attn     = CausalAttention(H)
        
        # output processing
        self.drop_out = nn.Dropout(0.35)
        self.norm     = nn.LayerNorm(H * 2)
        
        # final layer (2H because we concatenate)
        self.fc       = nn.Linear(H * 2, V)

    def forward(self, x, h=None):
        # embedding → dropout → RNN
        rnn_out, h = self.rnn(self.drop_in(self.emb(x)), h)
        
        ctx, _     = self.attn(rnn_out, rnn_out)
        out        = self.norm(self.drop_out(torch.cat([rnn_out, ctx], dim=-1)))
        
        # predicting next character
        return self.fc(out), h

    def init_h(self, B):
        # initializing hidden state
        return torch.zeros(self.L, B, self.H, device=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # loading dataset (full names)
    full_names, ds, loader = load_full_names("TrainingNames.txt")

    # hyperparameters
    E, H, L   = 64, 256, 2
    EPOCHS    = 80
    PATIENCE  = 12

    print("="*55)
    print("  TASK 1 — RNN + Attention  (single full-name model)")
    print("="*55)

     # creating model
    model = RNNWithAttention(ds.vocab_size, E, H, L).to(DEVICE)
    print(f"\nRNN+Attention  |  vocab={ds.vocab_size}  |  {count_params(model):,} trainable parameters")
 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # cosine scheduler (helps smoother learning)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

   # training model
    losses = train_model(model, loader, EPOCHS, optimizer, scheduler,
                         PATIENCE, "RNN+Attention", "rnn_attention")

    # ── Save loss history ─────────────────────────────────────────────────────
    with open("loss_rnn_att.json", "w") as f:
        json.dump({"RNN+Attention": losses}, f, indent=2)
    print("\nLoss history → loss_rnn_att.json")

    # final message
    print(f"\n✓ rnn_att.py done.  Model: models/rnn_attention.pt")
    print(" \n \n Now  Run eval.py for generation, novelty & diversity metrics.\n")