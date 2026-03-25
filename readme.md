<div align="center">

# 🧠 NLU Assignment-2

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-NLP-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**Natural Language Understanding · 2nd Semester · 2026**

| 👤 Name | 🎓 Roll No | 🏛️ Department |
|:---:|:---:|:---:|
| Mahek Shankesh Gadiya | M25CSA011 | CSE (MTech-AI) |

[![GitHub](https://img.shields.io/badge/View%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MSG-1999/NLU_MahekGadiya_M25CSA011/tree/Assignment_2)

</div>

---

## 📑 Table of Contents

- [Q1 — Word Embeddings from IITJ Data](#-q1--word-embeddings-from-iitj-data)
  - [Project Structure](#-q1-project-structure)
  - [Requirements](#-requirements)
  - [How to Run](#%EF%B8%8F-how-to-run)
  - [Outputs](#-outputs)
  - [Key Results](#-key-results)
- [Q2 — Character-Level Name Generation using RNN Variants](#-q2--character-level-name-generation-using-rnn-variants)
  - [Project Structure](#-q2-project-structure)
  - [Requirements](#-requirements-1)
  - [How to Run](#%EF%B8%8F-how-to-run-1)
  - [Outputs](#-outputs-1)
  - [Key Results](#-key-results-1)
- [General Notes](#-general-notes)

---

## 🔤 Q1 — Word Embeddings from IITJ Data

> Word2Vec models **(CBOW and SGNS)** trained on text scraped from IIT Jodhpur websites, implemented both from scratch in PyTorch and using the Gensim library.

### 📁 Q1 Project Structure

```
Q1/
├── 📄 collect_data.py           # Task 1: Scrape text from IITJ websites
├── 📄 preprocess.py             # Task 1: Clean corpus + tokenization + word cloud
├── 📄 models_scratch.py         # Model definitions: CBOW, SGNS (PyTorch from scratch)
├── 📄 train_word2vec.py         # Task 2: Train all models + hyperparameter experiments
├── 📄 analyze_and_visualize.py  # Tasks 3 & 4: Semantic analysis + PCA/t-SNE plots
├── 📝 raw_corpus.txt            # Raw scraped text (auto-created)
├── 📝 cleaned_corpus.txt        # Cleaned tokenized corpus (auto-created)
├── 📂 logs/                     # All log files (auto-created)
├── 📂 models/                   # Gensim model files (auto-created)
└── 📂 models_scratch/           # Scratch model weights as .pkl (auto-created)
```

### 📦 Requirements

```bash
pip install torch numpy matplotlib requests beautifulsoup4 gensim scikit-learn wordcloud urllib3
```

### ▶️ How to Run

> ⚠️ **Run scripts in this exact order. Each step depends on the previous one.**

<details>
<summary><b>Step 1 — Collect Data</b></summary>

```bash
python collect_data.py
```

Scrapes text from **3 IITJ pages** (Academic Regulations, Circulars, CSE Projects). Strips HTML noise and retains only text blocks with 5 or more words.

📤 **Saves:** `raw_corpus.txt`, `logs/collect_data.log`

</details>

<details>
<summary><b>Step 2 — Preprocess Corpus</b></summary>

```bash
python preprocess.py
```

Cleans raw text (removes URLs, numbers, source markers), tokenizes into sentences, and generates a word cloud. Prints dataset statistics (vocab size, token count, top-10 words).

📤 **Saves:** `cleaned_corpus.txt`, `wordcloud.png`, `logs/preprocess.log`

</details>

<details>
<summary><b>Step 3 — Train Word2Vec Models</b></summary>

```bash
python train_word2vec.py
```

Trains **4 model variants** across **14 hyperparameter configurations:**

| Model | Implementation | Emb. Dim | Window | Neg Samples | Tag |
|:---:|:---:|:---:|:---:|:---:|:---:|
| CBOW | PyTorch (scratch) | 100 | 8 | 5 | PRIMARY |
| SGNS | PyTorch (scratch) | 200 | 8 | 5 | PRIMARY |
| CBOW | Gensim (library) | 100, 150, 200 | 5, 8, 10 | 5, 10, 15 | EXP |
| SGNS | Gensim (library) | 200, 250, 300 | 5, 8, 10 | 5, 10, 15 | EXP |

🕐 Scratch models: **15 epochs** | Gensim models: **50 epochs** | Batch size: **512**

📤 **Saves:** `models/*.model`, `models_scratch/*.pkl`, `experiment_results.json`, `logs/train.log`

</details>

<details>
<summary><b>Step 4 — Analyze and Visualize</b></summary>

```bash
python analyze_and_visualize.py
```

Runs semantic analysis (top-5 nearest neighbours + 5 analogy experiments) and generates **PCA and t-SNE** visualizations for all 4 model variants. Also produces hyperparameter comparison bar charts.

📤 **Saves:** PCA/t-SNE PNGs, hyperparameter plots, `logs/analyze.log`

</details>

### 📊 Outputs

| File | Description |
|:---|:---|
| `raw_corpus.txt` | Raw scraped text (~98,591 words from 3 IITJ sources) |
| `cleaned_corpus.txt` | Tokenized corpus — 5,457 sentences, 91,076 tokens, vocab 2,026 |
| `wordcloud.png` | Word cloud of most frequent content words |
| `experiment_results.json` | All 14 hyperparameter experiment results |
| `hyperparam_dim.png` | Embedding dimension study (CBOW & SGNS) |
| `hyperparam_window.png` | Context window size study |
| `hyperparam_neg.png` | Negative sample count study |
| `visualization_pca_{model_id}.png` | PCA 2D projection per model |
| `visualization_tsne_{model_id}.png` | t-SNE 2D projection per model |
| `word_embedding.txt` | Full 200-dim embedding vector for probe word `"academic"` |
| `logs/collect_data.log` | Data collection log |
| `logs/preprocess.log` | Preprocessing and statistics log |
| `logs/train.log` | Full training log with loss per epoch |
| `logs/analyze.log` | Nearest neighbours and analogy results |

### 🏆 Key Results

| Model | Similarity Gap | Analogy Hit Rate | Avg Top-1 Cosine | Composite Score |
|:---:|:---:|:---:|:---:|:---:|
| Scratch CBOW | 0.203 | 0.000 | 0.538 | 0.278 |
| Scratch SGNS | 0.281 | 0.250 | 0.545 | 0.386 |
| Library CBOW (Gensim) | **0.488** | 0.000 | **0.652** | **0.415 ✅** |
| Library SGNS (Gensim) | 0.239 | **0.250** | 0.619 | 0.384 |

> 🥇 **Recommended Model:** Library CBOW (Gensim) — highest composite score **(0.4149)**

---

## 🔤 Q2 — Character-Level Name Generation using RNN Variants

> Three character-level RNN models trained to generate realistic **Indian full names** from a dataset of 1,000 names.

### 📁 Q2 Project Structure

```
Q2/
├── 📄 rnn.py               # Vanilla RNN model + training
├── 📄 blstm.py             # BLSTM model + training
├── 📄 rnn_att.py           # RNN + Causal Attention model + training
├── 📄 eval.py              # Tasks 2 & 3: Evaluation + plot generation
├── 📄 utils.py             # Shared utilities (dataset, training loop, metrics)
├── 📝 TrainingNames.txt    # 1,000 Indian full names (training data)
├── 📂 logs/                # Training logs (auto-created)
└── 📂 models/              # Saved model checkpoints (auto-created)
```

### 📦 Requirements

```bash
pip install torch numpy matplotlib
```

> 💡 GPU is used automatically if available. CPU works fine too.

### ▶️ How to Run

> ⚠️ **Run scripts in this exact order. `eval.py` requires all 3 models to be trained first.**

<details>
<summary><b>Step 1 — Train Vanilla RNN</b></summary>

```bash
python rnn.py
```

Trains a **2-layer Vanilla RNN** with Adam optimizer (lr=0.001) and ReduceLROnPlateau scheduler. Early stopping patience = 12.

📤 **Saves:** `models/vanilla_rnn.pt`, `loss_rnn.json`

</details>

<details>
<summary><b>Step 2 — Train BLSTM</b></summary>

```bash
python blstm.py
```

Trains a **2-layer LSTM** with LayerNorm, RMSprop optimizer (lr=0.001), and ReduceLROnPlateau scheduler.

📤 **Saves:** `models/blstm.pt`, `loss_blstm.json`

</details>

<details>
<summary><b>Step 3 — Train RNN + Attention</b></summary>

```bash
python rnn_att.py
```

Trains a **2-layer RNN** augmented with Bahdanau-style causal self-attention. Uses Adam optimizer and CosineAnnealingLR scheduler (T_max=80).

📤 **Saves:** `models/rnn_attention.pt`, `loss_rnn_att.json`

</details>

<details>
<summary><b>Step 4 — Evaluate All Models</b></summary>

```bash
python eval.py
```

Loads all 3 saved checkpoints, generates **200 names per model**, computes novelty/diversity/realism metrics, and produces all evaluation plots.

> ⚠️ **Requires:** `models/vanilla_rnn.pt`, `models/blstm.pt`, `models/rnn_attention.pt`
> If any checkpoint is missing, re-run the corresponding training script first.

</details>

### 📊 Outputs

| File | Description |
|:---|:---|
| `models/vanilla_rnn.pt` | Trained Vanilla RNN checkpoint |
| `models/blstm.pt` | Trained BLSTM checkpoint |
| `models/rnn_attention.pt` | Trained RNN+Attention checkpoint |
| `loss_rnn.json` | Vanilla RNN per-epoch loss history |
| `loss_blstm.json` | BLSTM per-epoch loss history |
| `loss_rnn_att.json` | RNN+Attention per-epoch loss history |
| `loss_histories.json` | Combined loss history (all 3 models) |
| `generated_Vanilla_RNN.txt` | 200 names generated by Vanilla RNN |
| `generated_BLSTM.txt` | 200 names generated by BLSTM |
| `generated_RNNAttention.txt` | 200 names generated by RNN+Attention |
| `training_curves_all.png` | Loss curves for all 3 models |
| `training_curves_best_loss_bar.png` | Best CE loss per model (bar chart) |
| `training_curves_convergence.png` | Epochs needed to reach loss thresholds |
| `training_curves_loss_gap.png` | Per-epoch loss gap vs best model |
| `training_curves_lr_schedule.png` | LR schedule overlaid on loss |
| `evaluation_comparison.png` | Novelty / diversity / radar chart |
| `eval_name_length_dist.png` | Histogram + box-plot of name lengths |
| `eval_score_matrix.png` | Colour-coded per-metric score matrix |
| `logs/training_log.txt` | Full training log (all scripts append here) |

### 🏆 Key Results

| Model | Params | Novelty (%) | Diversity (%) | Realism (proxy) | Best CE Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla RNN | 223,325 | 82.0 | 98.0 | 85.8 | 0.760 |
| BLSTM | 865,885 | 95.5 | **99.5 ✅** | **86.1 ✅** | 1.113 |
| RNN+Attention | 363,101 | **97.5 ✅** | 97.5 | 39.2 | **0.720 ✅** |

> 🥇 **Best overall:** BLSTM — highest realism + diversity  
> ⚡ **Fast & practical:** Vanilla RNN  
> 🆕 **Maximum novelty:** RNN+Attention *(apply a length filter post-generation)*

---

## 📝 General Notes

| # | Note |
|:---:|:---|
| 🔒 | **Q1 SSL:** IITJ servers use self-signed SSL certificates — verification is auto-disabled in `collect_data.py` |
| 🎲 | **Q1 Reproducibility:** All random seeds are fixed to **42** |
| 📚 | **Q1 Vocab:** `min_count=1` is used to retain the full vocabulary on the small academic corpus |
| 🔡 | **Q2 Space char:** The space character is part of the vocab — all models are trained on full names (e.g., `"Mahek Gadiya"`) as a single sequence |
| 📋 | **Q2 Logs:** All Q2 training logs are appended to a single file: `logs/training_log.txt` |
| ⚙️ | **Q2 Hyperparams:** Shared across all models — embedding dim = **64**, hidden size = **256**, **2 layers**, max **80 epochs**, early stop patience = **12** |

---

<div align="center">

Made with ❤️ by **Mahek Shankesh Gadiya** · M25CSA011 · MTech-AI, CSE

</div>
