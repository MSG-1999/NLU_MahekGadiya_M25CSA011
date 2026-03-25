"""
utils.py  —  Shared utilities for all RNN variant scripts.
Provides: Logger, CharDataset, make_loader, gen_one, is_valid,
          gen_fullname, gen_batch, evaluate, plot_curves, DEVICE.
"""

import os, sys, json, datetime
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    # this class helps to print output both on terminal and log file
    
    def __init__(self, path):
        self.terminal = sys.stdout
         # append so each script adds to same log
        self.log      = open(path, "a", buffering=1)   
        
        # adding timestamp for each run 
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        self.log.write(f"\n=== {ts} ===\n\n")
    def write(self, m):  self.terminal.write(m);  self.log.write(m)
    def flush(self):     self.terminal.flush();   self.log.flush()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class CharDataset(Dataset):

    def __init__(self, words):
        
        # creating character vocabulary
        chars           = sorted(set(''.join(words).lower()))
        self.chars      = ['<PAD>', '<SOS>', '<EOS>'] + chars
        self.char2idx   = {c: i for i, c in enumerate(self.chars)}
        self.idx2char   = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data       = []
        
        # converting each word into sequence of indices
        for w in words:
            s  = [self.char2idx['<SOS>']]
            s += [self.char2idx[c] for c in w.lower() if c in self.char2idx]
            s += [self.char2idx['<EOS>']]
            self.data.append(torch.tensor(s, dtype=torch.long))

    def __len__(self):       return len(self.data)
    def __getitem__(self, i):
        s = self.data[i];   return s[:-1], s[1:]


def make_loader(words, batch=32):
    # creates dataloader with padding
    
    def collate(batch):
        inp, tgt = zip(*batch)
        
        # padding sequences to same length (this is helpful when we have variable-length names)
        return (nn.utils.rnn.pad_sequence(inp, batch_first=True, padding_value=0),
                nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0))
    ds = CharDataset(words)
    return ds, DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def count_params(m):
    
    # counts number of trainable parameters
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_model(model, loader, epochs=80, optimizer=None,
                scheduler=None, patience=12, name="model", save_name=None):
    """
    Training loop with:
      • CrossEntropyLoss (PAD=0 ignored)
      • Gradient clipping max_norm=5
      • Early stopping
      • Best checkpoint saved to models/<save_name>.pt
      • Data augmentation: reversed sequences for BLSTM
      • Every epoch logged
    """
    
    # loss function (ignoring padding)
    crit      = nn.CrossEntropyLoss(ignore_index=0)
    save_name = save_name or name.lower().replace(' ','_').replace('+','_').replace('__','_')
    best, pat = float('inf'), 0
    losses    = []

    print(f"\n{'='*55}")
    print(f"  {name}  |  {count_params(model):,} params")
    print(f"  Opt: {optimizer.__class__.__name__}  "
          f"Sched: {scheduler.__class__.__name__ if scheduler else 'None'}  "
          f"Epochs: {epochs}  Patience: {patience}")
    print(f"{'='*55}")
    print(f"  {'Ep':>4}  {'Loss':>8}  {'LR':>10}  {'Pat':>4}")
    print(f"  {'-'*35}")

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0; n_batches = 0

        for inp, tgt in loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

            # Forward pass (normal left-to-right)
            optimizer.zero_grad()
            logits, _ = model(inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            
            # using gradient clipping because loss was exploding initially
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot += loss.item(); n_batches += 1

            # Reversed-sequence augmentation (BLSTM only) — helps model learn better bidirectional dependencies
            if hasattr(model, 'lstm'):
                inp_rev = torch.flip(inp, dims=[1])
                tgt_rev = torch.flip(tgt, dims=[1])
                optimizer.zero_grad()
                logits_rev, _ = model(inp_rev)
                loss_rev = crit(logits_rev.reshape(-1, logits_rev.size(-1)),
                                tgt_rev.reshape(-1))
                loss_rev.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        avg = tot / n_batches
        losses.append(avg)
        
        # updating scheduler 
        if scheduler:
            try:    scheduler.step(avg)
            except TypeError: scheduler.step()

        improved = avg < best - 1e-4
        if improved:
            best = avg; pat = 0    # saving best model
            torch.save(model.state_dict(), f"models/{save_name}.pt")
            flag = " *"
        else:
            pat += 1; flag = ""

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  {ep:>4}  {avg:>8.4f}  {lr_now:>10.6f}  {pat:>4}{flag}")

        if pat >= patience:
            print(f"  [Early Stop] epoch {ep}")
            break

    print(f"  Best: {best:.4f}  →  models/{save_name}.pt\n")
    return losses


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def gen_one(model, ds, max_len=12, temp=0.8):
    """Auto-regressive character generation."""
    # generate one name character by character
    # temp controls randomness (lower = safe, higher = creative)
    
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([[ds.char2idx['<SOS>']]], dtype=torch.long, device=DEVICE)
        h   = None
        out = []
        for _ in range(max_len):
            logits, h = model(inp, h)
            
             # temperature controls randomness
            probs = F.softmax(logits[0, -1, :] / temp, dim=-1)
            idx   = torch.multinomial(probs, 1).item()
            if idx == ds.char2idx['<EOS>']:
                break
            ch = ds.idx2char[idx]
            if ch not in ('<PAD>', '<SOS>', '<EOS>'):
                out.append(ch)
            inp = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)
    return ''.join(out).strip().capitalize()


def is_valid(name, min_len=2, max_run=3):
         # simple filter to remove bad names
     
    if len(name) < min_len:
        return False
    nl = name.lower()
    
     # avoid repeated characters like "aaaa"
    for i in range(len(nl) - max_run):
        if len(set(nl[i:i+max_run+1])) == 1:
            return False
    c = Counter(nl.replace(' ', ''))
    if c and c.most_common(1)[0][1] / len(nl.replace(' ', '')) > 0.55:
        return False
    return True


def gen_fullname(fn_model, fn_ds, ln_model, ln_ds, temp=0.8):
    """Generate one full name: first_name + ' ' + last_name."""
    
    for _ in range(20):
        fn = gen_one(fn_model, fn_ds, max_len=10, temp=temp)
        ln = gen_one(ln_model, ln_ds, max_len=12, temp=temp)
        if is_valid(fn, min_len=2) and is_valid(ln, min_len=3):
            return f"{fn} {ln}"
    return None


def gen_batch(fn_model, fn_ds, ln_model, ln_ds, n=200, temp=0.8):
    """Generate n full names."""
    
    names = []
    attempts = 0
    while len(names) < n and attempts < n * 10:
        name = gen_fullname(fn_model, fn_ds, ln_model, ln_ds, temp=temp)
        if name:
            names.append(name)
        attempts += 1
    return names


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(fn_model, fn_ds, ln_model, ln_ds, train_names,
             n=200, model_name="model", temp=0.8):
    """
    Novelty Rate (%) : generated names NOT in training set
    Diversity    : unique / total
    """
    gen  = gen_batch(fn_model, fn_ds, ln_model, ln_ds, n=n, temp=temp)
    tr   = set(x.lower() for x in train_names)
    nov  = [g for g in gen if g.lower() not in tr]
    uniq = set(gen)
    N    = len(gen)
    nr   = len(nov)  / N * 100 if N else 0
    dr   = len(uniq) / N       if N else 0

    print(f"\n  [{model_name}]  ({N} names generated)")
    print(f"    Novelty Rate : {nr:.1f}%")
    print(f"    Diversity    : {dr:.3f}")
    print(f"    Sample (15)  : {gen[:15]}")
    return {"model": model_name, "generated": gen, "novelty": nr, "diversity": dr}


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(loss_dict, path="training_curves.png"):
    """Plot training loss curves for all models."""
    styles = {"Vanilla RNN": ("b-", 2), "BLSTM": ("g--", 2), "RNN+Attention": ("r-.", 2)}
    plt.figure(figsize=(11, 5))
    for nm, losses in loss_dict.items():
        st, lw = styles.get(nm, ("-", 1.5))
        plt.plot(losses, st, label=nm, linewidth=lw)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Training Loss — All Models (First Name corpus)", fontsize=13)
    plt.legend(fontsize=11); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Plot] → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_names(data_path="TrainingNames.txt"):
    """Load TrainingNames.txt and split into first/last name lists.
    Used by blstm.py and rnn_att.py (dual-model approach)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found.")

    with open(data_path) as f:
        full_names = [l.strip() for l in f if l.strip()]

    first_names, last_names = [], []
    for name in full_names:
        parts = name.split()
        if len(parts) >= 2:
            first_names.append(parts[0])
            last_names.append(parts[-1])

    first_names_uniq = sorted(set(first_names))
    last_names_uniq  = sorted(set(last_names))

    print(f"Loaded {len(full_names)} full names  →  "
          f"{len(first_names_uniq)} unique first names, "
          f"{len(last_names_uniq)} unique last names")

    fn_ds, fn_loader = make_loader(first_names_uniq, batch=16)
    ln_ds, ln_loader = make_loader(last_names_uniq,  batch=16)

    print(f"First-name vocab: {fn_ds.vocab_size}  |  Last-name vocab: {ln_ds.vocab_size}\n")
    return full_names, fn_ds, fn_loader, ln_ds, ln_loader


def load_full_names(data_path="TrainingNames.txt"):
    """Load TrainingNames.txt as FULL name sequences (e.g. 'Mahek Gadiya').
    The space character is part of the vocabulary — one model learns everything.
    Used by rnn.py (single-model approach)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found.")

    with open(data_path) as f:
        full_names = [l.strip() for l in f if l.strip()]

    # Deduplicate while preserving order
    seen = set()
    unique_names = []
    for n in full_names:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique_names.append(n)

    print(f"Loaded {len(full_names)} full names  →  {len(unique_names)} unique full names")

    ds, loader = make_loader(unique_names, batch=16)
    print(f"Full-name vocab: {ds.vocab_size}  (includes space character)\n")
    return full_names, ds, loader


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-MODEL GENERATION  (used by rnn.py / eval.py single-model path)
# ─────────────────────────────────────────────────────────────────────────────

def gen_one_fullname(model, ds, max_len=25, temp=0.8):
    """Auto-regressive generation of a full 'Firstname Lastname' sequence."""
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([[ds.char2idx['<SOS>']]], dtype=torch.long, device=DEVICE)
        h   = None
        out = []
        for _ in range(max_len):
            logits, h = model(inp, h)
            probs = F.softmax(logits[0, -1, :] / temp, dim=-1)
            idx   = torch.multinomial(probs, 1).item()
            if idx == ds.char2idx['<EOS>']:
                break
            ch = ds.idx2char[idx]
            if ch not in ('<PAD>', '<SOS>', '<EOS>'):
                out.append(ch)
            inp = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)
    raw = ''.join(out).strip()
    # Capitalise first letter of each word (first name and last name)
    return ' '.join(w.capitalize() for w in raw.split()) if raw else ''


def is_valid_fullname(name):
    """Validate a generated full name — must have exactly two words,
    each meeting minimum length and no repeated-character artefacts."""
    parts = name.strip().split()
    if len(parts) != 2:
        return False
    fn, ln = parts
    return is_valid(fn, min_len=2) and is_valid(ln, min_len=3)


def gen_batch_single(model, ds, n=200, temp=0.8):
    """Generate n full names using a single model."""
    names    = []
    attempts = 0
    while len(names) < n and attempts < n * 15:
        name = gen_one_fullname(model, ds, max_len=25, temp=temp)
        if is_valid_fullname(name):
            names.append(name)
        attempts += 1
    return names


def evaluate_single(model, ds, train_names, n=200, model_name="model", temp=0.8):
    """Evaluate a single full-name model.
    Novelty Rate (%) : generated names NOT in training set
    Diversity    : unique / total
    """
    gen  = gen_batch_single(model, ds, n=n, temp=temp)
    tr   = set(x.lower() for x in train_names)
    nov  = [g for g in gen if g.lower() not in tr]
    uniq = set(gen)
    N    = len(gen)
    nr   = len(nov)  / N * 100 if N else 0
    dr   = len(uniq) / N       if N else 0

    print(f"\n  [{model_name}]  ({N} names generated)")
    print(f"    Novelty Rate : {nr:.1f}%")
    print(f"    Diversity    : {dr:.3f}")
    print(f"    Sample (15)  : {gen[:15]}")
    return {"model": model_name, "generated": gen, "novelty": nr, "diversity": dr}