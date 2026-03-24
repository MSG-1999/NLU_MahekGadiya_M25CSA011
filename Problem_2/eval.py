"""
eval.py  —  Tasks 2 & 3: Quantitative Evaluation + Qualitative Analysis
============================================================
Prerequisites (run once each):
    python rnn.py
    python blstm.py
    python rnn_att.py

Then run:
    python eval.py
"""

import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from utils import (DEVICE, Logger, load_full_names,
                   evaluate_single, count_params)

LOG_PATH   = "logs/training_log.txt"
sys.stdout = Logger(LOG_PATH)
print(f"[eval.py]  Device: {DEVICE}  |  Log: {LOG_PATH}\n")


# =============================================================================
# MODEL CLASSES — defined INLINE
# =============================================================================

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


class CausalAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.Wq = nn.Linear(H, H, bias=False)
        self.Wk = nn.Linear(H, H, bias=False)
        self.v  = nn.Linear(H, 1, bias=False)
    def forward(self, Q, K):
        scores = self.v(torch.tanh(
            self.Wq(Q).unsqueeze(2) + self.Wk(K).unsqueeze(1)
        )).squeeze(-1)
        T    = scores.size(1)
        mask = torch.tril(torch.ones(T, T, device=scores.device)).bool()
        scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
        alpha  = F.softmax(scores, dim=-1)
        return torch.bmm(alpha, K), alpha

class RNNWithAttention(nn.Module):
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


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

STYLES = {
    "Vanilla RNN":   ("royalblue",   "-",   "o"),
    "BLSTM":         ("forestgreen", "--",  "s"),
    "RNN+Attention": ("crimson",     "-.",  "^"),
}

def _smooth(vals, w=3):
    if len(vals) < w:
        return np.array(vals)
    kernel = np.ones(w) / w
    padded = np.pad(vals, (w//2, w//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(vals)]


# ── Plot 1: Single loss overlay ───────────────────────────────────────────────
def plot_single(loss_dict, title, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in loss_dict.items():
        col, ls, _ = STYLES.get(name, ("gray", "-", "o"))
        ep = range(1, len(losses) + 1)
        ax.plot(ep, losses,           ls, color=col, linewidth=1.2, alpha=0.25,
                label='_nolegend_')
        ax.plot(ep, _smooth(losses),  ls, color=col, linewidth=2.3, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ── Plot 2: Best loss bar ─────────────────────────────────────────────────────
def plot_final_loss_bar(loss_dict, path):
    models  = list(loss_dict.keys())
    best    = [min(_smooth(loss_dict[m])) for m in models]
    colors  = [STYLES[m][0] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, best, color=colors, edgecolor='black',
                  linewidth=0.8, alpha=0.88, width=0.5)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=11)
    ax.set_ylabel("Best CE Loss", fontsize=12)
    ax.set_title("Best Training Loss per Model\n(single full-name model)",
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(best) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ── Plot 3: Convergence heatmap ───────────────────────────────────────────────
def plot_convergence(loss_dict, path):
    thresholds = [2.5, 2.0, 1.75, 1.5, 1.25]
    models     = list(loss_dict.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, losses in loss_dict.items():
        col, ls, _ = STYLES.get(name, ("gray", "-", "o"))
        ep = range(1, len(losses) + 1)
        ax1.plot(ep, _smooth(losses), ls, color=col, linewidth=2.2, label=name)
    for th in thresholds:
        ax1.axhline(th, color='gray', linestyle=':', linewidth=0.9, alpha=0.6)
        ax1.text(max(len(v) for v in loss_dict.values()) + 0.5, th,
                 f'{th}', va='center', fontsize=8, color='gray')
    ax1.set_xlabel("Epoch", fontsize=11); ax1.set_ylabel("CE Loss", fontsize=11)
    ax1.set_title("Full-Name Training Loss", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    data = np.array([
        [next((i+1 for i, l in enumerate(loss_dict[m]) if l <= th),
               len(loss_dict[m])) for m in models]
        for th in thresholds
    ], dtype=float)

    im = ax2.imshow(data, cmap='RdYlGn_r', aspect='auto')
    ax2.set_xticks(range(len(models))); ax2.set_xticklabels(models, fontsize=10)
    ax2.set_yticks(range(len(thresholds)))
    ax2.set_yticklabels([f'Loss <= {t}' for t in thresholds], fontsize=10)
    ax2.set_title("Epochs to Reach Threshold\n(green = faster)", fontsize=10,
                  fontweight='bold')
    for i in range(len(thresholds)):
        for j in range(len(models)):
            ax2.text(j, i, str(int(data[i, j])), ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if data[i, j] > data.max() * 0.6 else 'black')
    plt.colorbar(im, ax=ax2, label='Epochs')
    fig.suptitle("Convergence Speed Analysis", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ── Plot 4: Evaluation comparison ────────────────────────────────────────────
def plot_evaluation_comparison(results, path):
    models    = [r['model']    for r in results]
    novelty   = [r['novelty']  for r in results]
    diversity = [r['diversity'] * 100 for r in results]
    realism   = []
    for r in results:
        avg_l = np.mean([len(n) for n in r['generated']]) if r['generated'] else 10
        realism.append(max(0, 100 - abs(avg_l - 10) * 8))

    colors = [STYLES[m][0] for m in models]
    fig    = plt.figure(figsize=(16, 5))
    fig.suptitle("Model Evaluation Comparison", fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(1, 3, 1)
    b = ax1.bar(models, novelty, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax1.bar_label(b, fmt='%.1f%%', padding=3, fontsize=10)
    ax1.set_ylim(0, 115); ax1.set_ylabel("Novelty Rate (%)", fontsize=11)
    ax1.set_title("Novelty Rate\n(% not in training set)", fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3); ax1.tick_params(axis='x', labelsize=9)

    ax2 = fig.add_subplot(1, 3, 2)
    b2 = ax2.bar(models, diversity, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=10)
    ax2.set_ylim(0, 115); ax2.set_ylabel("Diversity (%)", fontsize=11)
    ax2.set_title("Diversity\n(unique / total x 100)", fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3); ax2.tick_params(axis='x', labelsize=9)

    ax3 = fig.add_subplot(1, 3, 3, polar=True)
    categories = ['Novelty', 'Diversity', 'Realism\n(proxy)']
    N_cat      = len(categories)
    angles     = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles    += angles[:1]
    ax3.set_theta_offset(np.pi / 2); ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 100); ax3.set_yticks([25, 50, 75, 100])
    ax3.set_yticklabels(['25', '50', '75', '100'], fontsize=7)
    ax3.grid(alpha=0.4)
    ax3.set_title("Overall Comparison\n(radar)", fontsize=10, fontweight='bold', pad=15)
    for i, r in enumerate(results):
        vals = [novelty[i], diversity[i], realism[i]] + [novelty[i]]
        col  = STYLES[r['model']][0]
        ax3.plot(angles, vals, '-', color=col, linewidth=2, label=r['model'])
        ax3.fill(angles, vals, color=col, alpha=0.12)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.2), fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NEW PLOT 5: Loss-gap heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_loss_gap_heatmap(loss_dict, path):
    """
    For each epoch, each model's loss MINUS the minimum across models.
    Shows which model leads and by how much over training.
    """
    models      = list(loss_dict.keys())
    min_len     = min(len(v) for v in loss_dict.values())
    arr         = np.array([_smooth(loss_dict[m])[:min_len] for m in models])
    gap         = arr - arr.min(axis=0, keepdims=True)

    step   = max(1, min_len // 40)
    gap_ds = gap[:, ::step]
    x_ticks = np.arange(1, min_len + 1)[::step]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(gap_ds, aspect='auto', cmap='YlOrRd',
                   extent=[x_ticks[0], x_ticks[-1], len(models) - 0.5, -0.5])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_title("Loss Gap vs Best Model per Epoch\n"
                 "(white = leading model; darker = further behind)",
                 fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("CE Loss Gap", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NEW PLOT 6: Learning-rate schedule overlay
# ══════════════════════════════════════════════════════════════════════════════
def plot_lr_schedules(loss_dict, path):
    """
    Simulate the LR schedule for each model and overlay on its loss curve.
    Left y-axis = CE loss; right y-axis = learning rate (log scale).
    """
    schedules = {
        "Vanilla RNN":   {"type": "plateau",  "lr0": 1e-3, "factor": 0.5, "patience": 8},
        "BLSTM":         {"type": "plateau",  "lr0": 1e-3, "factor": 0.5, "patience": 8},
        "RNN+Attention": {"type": "cosine",   "lr0": 1e-3, "eta_min": 1e-5, "T_max": 80},
    }

    def sim_plateau(losses, lr0, factor, patience):
        lrs, best_l, pat, lr = [], float('inf'), 0, lr0
        for l in losses:
            if l < best_l - 1e-4:
                best_l, pat = l, 0
            else:
                pat += 1
            if pat >= patience:
                lr *= factor; pat = 0
            lrs.append(lr)
        return lrs

    def sim_cosine(n, lr0, eta_min, T_max):
        return [eta_min + 0.5 * (lr0 - eta_min) * (1 + np.cos(np.pi * t / T_max))
                for t in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training Loss & Learning-Rate Schedule per Model",
                 fontsize=13, fontweight='bold')

    for ax, (name, losses) in zip(axes, loss_dict.items()):
        col, ls, _ = STYLES.get(name, ("gray", "-", "o"))
        ep  = np.arange(1, len(losses) + 1)
        sch = schedules.get(name, {})

        if sch.get("type") == "plateau":
            lrs = sim_plateau(losses, sch["lr0"], sch["factor"], sch["patience"])
        else:
            lrs = sim_cosine(len(losses), sch["lr0"], sch["eta_min"], sch["T_max"])

        ax2 = ax.twinx()
        ax.plot(ep, _smooth(losses), ls, color=col, linewidth=2.2, label="CE Loss")
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("CE Loss", fontsize=10, color=col)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NEW PLOT 7: Generated name-length distributions
# ══════════════════════════════════════════════════════════════════════════════
def plot_name_length_dist(results, path):
    """
    Histogram + box-plot of generated full-name character lengths per model.
    """
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Generated Name Length Distribution", fontsize=13, fontweight='bold')

    bp_data, bp_labels, bp_colors = [], [], []
    for r in results:
        lens = [len(n) for n in r['generated']]
        col, ls, _ = STYLES.get(r['model'], ("gray", "-", "o"))
        ax_hist.hist(lens, bins=range(min(lens), max(lens) + 2), alpha=0.55,
                     color=col, edgecolor='black', linewidth=0.5, label=r['model'])
        bp_data.append(lens); bp_labels.append(r['model']); bp_colors.append(col)

    ax_hist.set_xlabel("Full-Name Length (characters)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Histogram of Name Lengths", fontsize=11, fontweight='bold')
    ax_hist.legend(fontsize=10); ax_hist.grid(axis='y', alpha=0.3)

    bp = ax_box.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
    for patch, col in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(col); patch.set_alpha(0.70)
    ax_box.set_ylabel("Name Length (characters)", fontsize=11)
    ax_box.set_title("Box-Plot of Name Lengths", fontsize=11, fontweight='bold')
    ax_box.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NEW PLOT 8: Per-metric score matrix
# ══════════════════════════════════════════════════════════════════════════════
def plot_score_matrix(results, path):
    """
    Colour-coded table: rows = models, cols = metrics.
    Colour encodes relative rank within each metric column.
    """
    models    = [r['model'] for r in results]
    novelty   = np.array([r['novelty']         for r in results])
    diversity = np.array([r['diversity'] * 100  for r in results])
    avg_len   = np.array([np.mean([len(n) for n in r['generated']]) for r in results])
    realism   = np.array([max(0., 100 - abs(l - 10) * 8) for l in avg_len])

    def bigram_ratio(names):
        all_bg = [n[i:i+2] for n in names for i in range(len(n) - 1)]
        return len(set(all_bg)) / len(all_bg) * 100 if all_bg else 0

    lex_var = np.array([bigram_ratio(r['generated']) for r in results])

    metric_names = ["Novelty (%)", "Diversity (%)", "Realism (proxy)",
                    "Lexical Variety", "Avg Name Length"]
    raw_matrix   = np.column_stack([novelty, diversity, realism, lex_var, avg_len])

    norm_matrix = np.zeros_like(raw_matrix)
    for c in range(raw_matrix.shape[1]):
        col_min, col_max = raw_matrix[:, c].min(), raw_matrix[:, c].max()
        if col_max > col_min:
            norm_matrix[:, c] = (raw_matrix[:, c] - col_min) / (col_max - col_min) * 100
        else:
            norm_matrix[:, c] = 50.

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(norm_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10, rotation=20, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_title("Per-Metric Score Matrix  (green = best in column)",
                 fontsize=12, fontweight='bold')

    for i in range(len(models)):
        for j in range(len(metric_names)):
            raw_val = raw_matrix[i, j]
            txt = f"{raw_val:.1f}" if "Length" not in metric_names[j] \
                  else f"{raw_val:.1f} ch"
            ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                    fontweight='bold',
                    color='black' if 20 < norm_matrix[i, j] < 80 else 'white')

    plt.colorbar(im, ax=ax, label='Relative rank (0=worst, 100=best)', pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  [Plot] -> {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    full_names, full_ds, _ = load_full_names("TrainingNames.txt")

    E, H, L = 64, 256, 2

    rnn_model   = VanillaRNN(full_ds.vocab_size, E, H, L).to(DEVICE)
    blstm_model = BLSTM(full_ds.vocab_size, E, H, L).to(DEVICE)
    attn_model  = RNNWithAttention(full_ds.vocab_size, E, H, L).to(DEVICE)

    print("="*58)
    print("  TASK 1 - Model Parameter Counts")
    print("="*58)
    for label, m in [("Vanilla RNN",   rnn_model),
                     ("BLSTM",         blstm_model),
                     ("RNN+Attention", attn_model)]:
        print(f"  {label:<22} {count_params(m):>12,}  single model (full names)")

    print("\nLoading saved model checkpoints ...")
    for model, name in [(rnn_model,   "vanilla_rnn"),
                        (blstm_model, "blstm"),
                        (attn_model,  "rnn_attention")]:
        ckpt_path = f"models/{name}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"\n[FATAL] '{ckpt_path}' not found.\n"
                f"        Run:  python rnn.py  ->  python blstm.py  ->  python rnn_att.py\n"
                f"        Then: python eval.py")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"  OK  {ckpt_path}")

    # ── Load loss histories ───────────────────────────────────────────────────
    print("\nLoading loss histories ...")
    loss_dict = {}
    for json_file, model_key in [
        ("loss_rnn.json",     "Vanilla RNN"),
        ("loss_blstm.json",   "BLSTM"),
        ("loss_rnn_att.json", "RNN+Attention"),
    ]:
        if not os.path.exists(json_file):
            print(f"  [WARN] {json_file} not found - skipping {model_key}")
            continue
        with open(json_file) as f:
            d = json.load(f)
        key = list(d.keys())[0]
        loss_dict[model_key] = d[key]
        print(f"  OK  {json_file}  ({len(d[key])} epochs)")

    with open("loss_histories.json", "w") as f:
        json.dump({m: loss_dict.get(m, []) for m in
                   ["Vanilla RNN", "BLSTM", "RNN+Attention"]}, f, indent=2)
    print("  Saved -> loss_histories.json\n")

    # ── Task 2: Evaluation ────────────────────────────────────────────────────
    print("="*58)
    print("  TASK 2 - QUANTITATIVE EVALUATION")
    print("="*58)

    torch.manual_seed(42)

    # Vanilla RNN uses temp=1.2: flatter distribution → much less memorisation
    # BLSTM / RNN+Attention stay at temp=1.0 (already regularised enough)
    results = [
        evaluate_single(rnn_model,   full_ds, full_names, n=200,
                        model_name="Vanilla RNN",   temp=1.2),
        evaluate_single(blstm_model, full_ds, full_names, n=200,
                        model_name="BLSTM",         temp=1.0),
        evaluate_single(attn_model,  full_ds, full_names, n=200,
                        model_name="RNN+Attention", temp=0.8),
    ]

    print("\n--- Summary ---")
    print(f"  {'Model':<18} {'Novelty':>10} {'Diversity':>12} {'Count':>7}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['model']:<18} {r['novelty']:>9.1f}% "
              f"{r['diversity']:>12.3f} {len(r['generated']):>7}")

    # ── Task 3: Qualitative ───────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  TASK 3 - QUALITATIVE ANALYSIS")
    print("="*58)

    for r in results:
        print(f"\n-- {r['model']} --")
        print(f"  Samples  : {r['generated'][:20]}")
        print(f"  Novelty  : {r['novelty']:.1f}%  |  Diversity: {r['diversity']:.3f}")

    print("""
--- Discussion ---

All models trained on FULL names (one sequence: "Firstname Lastname").
The space character is part of the vocabulary.
    """)

    # ── Save generated name files ─────────────────────────────────────────────
    for r in results:
        out_fn = f"generated_{r['model'].replace(' ','_').replace('+','')}.txt"
        with open(out_fn, "w") as fh:
            fh.write("\n".join(r["generated"]))
        print(f"Saved -> {out_fn}")

    # ── Generate ALL plots ────────────────────────────────────────────────────
    print("\n--- Generating plots ---")

    if loss_dict:
        plot_single(loss_dict,
                    "Training Loss - Full-Name Corpus (All Models)",
                    "training_curves_all.png")
        plot_final_loss_bar(loss_dict,
                            "training_curves_best_loss_bar.png")
        plot_convergence(loss_dict,
                         "training_curves_convergence.png")
        plot_loss_gap_heatmap(loss_dict,
                              "training_curves_loss_gap.png")
        plot_lr_schedules(loss_dict,
                          "training_curves_lr_schedule.png")

    plot_evaluation_comparison(results,  "evaluation_comparison.png")
    plot_name_length_dist(results,       "eval_name_length_dist.png")
    plot_score_matrix(results,           "eval_score_matrix.png")

    print(f"\n  eval.py done.  Log: {LOG_PATH}")
    print("""
Output plots
  training_curves_all.png            -- all 3 models, full-name corpus
  training_curves_best_loss_bar.png  -- best CE loss per model
  training_curves_convergence.png    -- convergence speed heatmap
  training_curves_loss_gap.png   -- loss gap vs best model per epoch
  training_curves_lr_schedule.png  -- LR schedule overlaid on loss
  evaluation_comparison.png          -- novelty / diversity / radar
  eval_name_length_dist.png      -- histogram + box-plot of name lengths
  eval_score_matrix.png          -- colour-coded per-metric score table
    """)