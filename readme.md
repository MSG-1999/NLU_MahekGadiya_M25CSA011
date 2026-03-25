<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=6C63FF&center=true&vCenter=true&width=700&lines=NLU+Assignment-2;Word+Embeddings+%2B+RNN+Name+Generation;IIT+Jodhpur+%7C+MTech-AI+%7C+2026" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-4.4.0-009688?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-✅%20Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/Course-NLU%20%7C%20Sem%202-blueviolet?style=for-the-badge)

<br/>

> **Natural Language Understanding · IIT Jodhpur · 2nd Semester · 2026**

| 👤 Student | 🎓 Roll No | 🏛️ Department |
|:---:|:---:|:---:|
| **Mahek Shankesh Gadiya** | `M25CSA011` | CSE — MTech (AI) |

<br/>

[![GitHub](https://img.shields.io/badge/🔗%20View%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MSG-1999/NLU_MahekGadiya_M25CSA011/tree/Assignment_2)

> 📌 All plots and results were successfully generated during code execution. Every result is **fully reproducible** by following the steps below.

</div>

---

## 📑 Table of Contents

- [🔤 Q1 — Word Embeddings from IITJ Data](#-q1--word-embeddings-from-iit-jodhpur-data)
  - [Project Structure](#-q1-project-structure)
  - [Requirements](#%EF%B8%8F-q1-requirements)
  - [How to Run](#-q1-how-to-run)
  - [Key Results](#-q1-key-results)
  - [Outputs](#-q1-outputs)
- [🔡 Q2 — Character-Level Name Generation](#-q2--character-level-name-generation-using-rnn-variants)
  - [Project Structure](#-q2-project-structure)
  - [Requirements](#%EF%B8%8F-q2-requirements)
  - [How to Run](#-q2-how-to-run)
  - [Model Architectures](#%EF%B8%8F-q2-model-architectures)
  - [Key Results](#-q2-key-results)
  - [Outputs](#-q2-outputs)
- [📚 References](#-references)
- [🔧 General Notes](#-general-notes)

---

# 🔤 Q1 — Word Embeddings from IIT Jodhpur Data

> **Word2Vec** models (CBOW & Skip-Gram with Negative Sampling) trained on text scraped from IIT Jodhpur's official websites — implemented both **from scratch in PyTorch** and via the **Gensim library** — evaluated through cosine similarity, analogy tasks, and 2D visualizations.

## 📁 Q1 Project Structure

```
Q1/
├── 📄 collect_data.py           # Task 1: Scrape text from 3 IITJ websites
├── 📄 preprocess.py             # Task 1: Clean corpus + tokenization + word cloud
├── 📄 models_scratch.py         # Model definitions: CBOW & SGNS (PyTorch from scratch)
├── 📄 train_word2vec.py         # Task 2: Train all models + hyperparameter experiments
├── 📄 analyze_and_visualize.py  # Tasks 3 & 4: Semantic analysis + PCA/t-SNE plots
│
├── 📝 raw_corpus.txt            # Raw scraped text                (auto-created)
├── 📝 cleaned_corpus.txt        # Cleaned tokenized corpus        (auto-created)
├── 📝 wordcloud.png             # Word cloud visualization
├── 📝 word_embedding.txt        # Full 200-dim vector for "academic"
├── 📝 top10_words.txt           # Top-10 most frequent words
├── 📝 experiment_results.json   # All 14 hyperparameter results
│
├── 📂 logs/
│   ├── collect_data.log         # Data collection log
│   ├── preprocess.log           # Preprocessing + statistics
│   ├── train.log                # Full training log (loss per epoch)
│   └── analyze.log              # Nearest neighbours & analogy results
│
├── 📂 models/                   # Gensim model files (.model)
└── 📂 models_scratch/           # Scratch model weights (.pkl)
```

## ⚙️ Q1 Requirements

```bash
pip install torch numpy matplotlib requests beautifulsoup4 gensim scikit-learn wordcloud urllib3
```

## 🚀 Q1 How to Run

> ⚠️ **Run scripts in this exact order — each step depends on the previous one.**

<details>
<summary><b>📥 Step 1 — Collect Data</b></summary>
<br/>

```bash
python collect_data.py
```

Scrapes text from **3 IITJ pages** using `requests` + `BeautifulSoup`. Strips HTML noise (nav, footer, scripts) and retains only blocks with **≥ 5 words**.

| Source | URL | Words |
|:---|:---|:---:|
| Academic Regulations *(required)* | `iitj.ac.in/office-of-academics/en/academic-regulations` | 93,175 |
| Office of Academics — Circulars | `iitj.ac.in/office-of-academics/en/circulars` | 1,840 |
| CSE Department — Projects | `iitj.ac.in/computer-science-engineering/en/projects` | 3,559 |
| **Total** | | **98,591** |

📤 **Saves:** `raw_corpus.txt`, `logs/collect_data.log`

</details>

<details>
<summary><b>🧹 Step 2 — Preprocess Corpus</b></summary>
<br/>

```bash
python preprocess.py
```

Pipeline:
1. **Boilerplate removal** — strips `# SOURCE:` markers, URLs, emails, standalone numbers
2. **Tokenization** — splits at punctuation; extracts tokens via `[a-z]{2,}` regex
3. **Lowercasing** — all tokens converted to lowercase
4. **Filtering** — drops sentences with fewer than 3 valid tokens

📤 **Saves:** `cleaned_corpus.txt`, `wordcloud.png`, `logs/preprocess.log`

</details>

<details>
<summary><b>🏋️ Step 3 — Train Word2Vec Models</b></summary>
<br/>

```bash
python train_word2vec.py
```

Trains **4 model variants** across **14 hyperparameter configurations**:

| Config ID | Model | Dim | Window | Neg Samples | Tag |
|:---|:---:|:---:|:---:|:---:|:---:|
| `cbow_d100_w8_n5` | CBOW | 100 | 8 | 5 | PRIMARY |
| `sgns_d200_w8_n5` | SGNS | 200 | 8 | 5 | PRIMARY |
| `cbow_d150/200_w8_n5` | CBOW | 150–200 | 8 | 5 | DIM_EXP |
| `sgns_d250/300_w8_n5` | SGNS | 250–300 | 8 | 5 | DIM_EXP |
| `cbow/sgns_*_w5/10_n5` | Both | 100 | 5, 10 | 5 | WIN_EXP |
| `cbow/sgns_*_w8_n10/15` | Both | 100 | 8 | 10, 15 | NEG_EXP |

> 🕐 Scratch models: **15 epochs** · Gensim: **50 epochs** · Batch size: **512**

📤 **Saves:** `models/*.model`, `models_scratch/*.pkl`, `experiment_results.json`, `logs/train.log`

</details>

<details>
<summary><b>📊 Step 4 — Analyze & Visualize</b></summary>
<br/>

```bash
python analyze_and_visualize.py
```

- Top-5 nearest neighbours for: `research`, `student`, `phd`, `exam`
- 5 analogy experiments via vector arithmetic: `v_b − v_a + v_c ≈ v_d`
- **PCA** and **t-SNE** 2D projections for all 4 models
- Hyperparameter comparison bar charts

📤 **Saves:** All `.png` plots, `logs/analyze.log`

</details>

## 📊 Q1 Key Results

### Corpus Statistics

| Statistic | Value |
|:---|:---:|
| Total sentences | **5,457** |
| Total tokens | **91,076** |
| Vocabulary size | **2,026** |
| Avg. sentence length | **~16.7 tokens** |

### Top-10 Words

```
student (1271)  ·  course (797)  ·  tech (758)  ·  degree (708)  ·  students (674)
academic (649)  ·  semester (647)  ·  requirements (561)  ·  ph (554)  ·  program (528)
```

### Top-5 Nearest Neighbours (selected)

| Query | Scratch CBOW | Scratch SGNS | Lib CBOW | Lib SGNS |
|:---:|:---|:---|:---|:---|
| `phd` | practicals, msc | mtech, msc | **mtech (0.671)**, admit | admit, mtech, msc |
| `exam` | **quiz (0.884)** | **quiz (0.883)** | **quiz (0.982)** | **quiz (0.984)** |
| `student` | students, applications | the, pe | students, semesters | attend, withdraw |
| `research` | formation, certify | platforms, proposal | appoint, physics | **proposal (0.560)** |

> 💡 `exam → quiz` is the most consistent semantic hit across **all 4 models** (cosine ≥ 0.883)

### Model Evaluation Summary

| Model | Sim. Gap ↑ | Avg Cosine ↑ | Analogy Rate ↑ | **Composite Score** |
|:---|:---:|:---:|:---:|:---:|
| Scratch CBOW | 0.203 | 0.538 | 0.000 | 0.278 |
| Scratch SGNS | 0.281 | 0.545 | 0.250 | 0.386 |
| Library SGNS | 0.239 | 0.619 | 0.250 | 0.384 |
| **Library CBOW** ⭐ | **0.488** | **0.652** | 0.000 | **0.415 ✅** |

> 🏆 **Recommended:** Library CBOW (Gensim) — highest composite score **(0.4149)**

### Probe Word Embedding — `"academic"`

```
Model : Gensim SGNS | dim=200 | window=8 | neg=5
Stats : L2 norm=4.686 | Min=−0.956 | Max=1.043 | Mean=−0.00183 | Std=0.331

Vector: 0.1331, 0.4808, -0.0484, 0.5883, 0.0265, -0.3313, 0.0402, 0.5775, ...
        [full 200-dimensional vector stored in word_embedding.txt]
```

## 📂 Q1 Outputs

| File | Description |
|:---|:---|
| `raw_corpus.txt` | Raw scraped text (~98,591 words, 3 IITJ sources) |
| `cleaned_corpus.txt` | Tokenized corpus — 5,457 sentences, 91,076 tokens, vocab 2,026 |
| `wordcloud.png` | Word cloud of most frequent content words |
| `word_embedding.txt` | Full 200-dim vector for probe word `academic` |
| `top10_words.txt` | Top-10 words by frequency |
| `experiment_results.json` | All 14 hyperparameter experiment results |
| `hyperparam_dim.png` | Embedding dimension study (CBOW & SGNS) |
| `hyperparam_window.png` | Context window size study |
| `hyperparam_neg.png` | Negative sample count study |
| `visualization_pca_*.png` | PCA 2D projection per model (4 files) |
| `visualization_tsne_*.png` | t-SNE 2D projection per model (4 files) |
| `comparison_all_models.png` | Bar chart — 4 metrics across all models |
| `comparison_heatmap.png` | Heatmap — evaluation metrics all models |
| `logs/collect_data.log` | Data collection log |
| `logs/preprocess.log` | Preprocessing + corpus statistics |
| `logs/train.log` | Full training log with per-epoch loss |
| `logs/analyze.log` | Nearest neighbours & analogy results |

---

# 🔡 Q2 — Character-Level Name Generation using RNN Variants

> Three character-level RNN models trained end-to-end on a dataset of **1,000 Indian full names**. The **space character is part of the vocabulary** — each model learns first name, separator, and surname jointly as one character sequence.

## 📁 Q2 Project Structure

```
Q2/
├── 📄 rnn.py              # Vanilla RNN model + training
├── 📄 blstm.py            # BLSTM model + training
├── 📄 rnn_att.py          # RNN + Causal Attention model + training
├── 📄 eval.py             # Tasks 2 & 3: Generation + evaluation + all plots
├── 📄 utils.py            # Shared: dataset, training loop, logger, metrics
├── 📝 TrainingNames.txt   # 1,000 Indian full names (training data)
│
├── 📂 logs/
│   └── training_log.txt   # All 3 scripts append to this single file
│
└── 📂 models/
    ├── vanilla_rnn.pt     # Trained Vanilla RNN checkpoint
    ├── blstm.pt           # Trained BLSTM checkpoint
    └── rnn_attention.pt   # Trained RNN+Attention checkpoint
```

## ⚙️ Q2 Requirements

```bash
pip install torch numpy matplotlib
```

> 🖥️ GPU used automatically if available — CPU works fine too.

## 🚀 Q2 How to Run

> ⚠️ **Run in this exact order. `eval.py` requires all 3 trained checkpoints.**

<details>
<summary><b>🧠 Step 1 — Train Vanilla RNN</b></summary>
<br/>

```bash
python rnn.py
```

Trains a **2-layer Vanilla RNN** with Adam optimizer (lr=0.001) and ReduceLROnPlateau scheduler. Early stopping patience = 12.

📤 **Saves:** `models/vanilla_rnn.pt`, `loss_rnn.json`

</details>

<details>
<summary><b>🧠 Step 2 — Train BLSTM</b></summary>
<br/>

```bash
python blstm.py
```

Trains a **2-layer LSTM** with LayerNorm, RMSprop optimizer (lr=0.001), and ReduceLROnPlateau scheduler.

📤 **Saves:** `models/blstm.pt`, `loss_blstm.json`

</details>

<details>
<summary><b>🧠 Step 3 — Train RNN + Causal Attention</b></summary>
<br/>

```bash
python rnn_att.py
```

Trains a **2-layer RNN** augmented with Bahdanau-style causal self-attention. Uses Adam optimizer and CosineAnnealingLR scheduler (T_max=80).

📤 **Saves:** `models/rnn_attention.pt`, `loss_rnn_att.json`

</details>

<details>
<summary><b>📊 Step 4 — Evaluate All Models</b></summary>
<br/>

```bash
python eval.py
```

Loads all 3 checkpoints, generates **200 names per model**, computes novelty / diversity / realism metrics, and produces all evaluation plots.

> ✅ Requires: `models/vanilla_rnn.pt`, `models/blstm.pt`, `models/rnn_attention.pt`  
> ⚠️ Re-run the corresponding training script if any checkpoint is missing.

📤 **Saves:** Generated name `.txt` files + all training & evaluation `.png` plots

</details>

## 🏗️ Q2 Model Architectures

### 1️⃣ Vanilla RNN *(Baseline)*

```
Embedding(V, 64) → Dropout(0.5) → RNN(64→256, 2L, dropout=0.5) → Linear(256, V)

Optimizer  : Adam (lr=0.001, weight_decay=1e-3)
Scheduler  : ReduceLROnPlateau (factor=0.5, patience=8)
Parameters : 223,325
```

### 2️⃣ BLSTM

```
Embedding(V, 64) → Dropout(0.5) → LSTM(64→256, 2L, dropout=0.5)
→ LayerNorm(256) → Dropout(0.5) → Linear(256, V)

Optimizer  : RMSprop (lr=0.001, alpha=0.9, weight_decay=1e-3)
Scheduler  : ReduceLROnPlateau (factor=0.5, patience=8)
Parameters : 865,885
```

### 3️⃣ RNN + Causal Attention

```
Embedding(V, 64) → Dropout(0.35) → RNN(64→256, 2L, dropout=0.35)
→ CausalAttention(256) → cat([rnn_out, context])
→ LayerNorm(512) → Dropout(0.35) → Linear(512, V)

Optimizer  : Adam (lr=0.001, weight_decay=1e-4)
Scheduler  : CosineAnnealingLR (T_max=80, eta_min=1e-5)
Parameters : 363,101
```

> 💡 **Why causal masking?** Without a lower-triangular mask, the model attends to *future* characters at train time but cannot do so at inference → causes repetition loops. The causal mask enforces identical behaviour at both stages.

## 📊 Q2 Key Results

### Hyperparameter Summary

| Hyperparameter | Vanilla RNN | BLSTM | RNN+Attention |
|:---|:---:|:---:|:---:|
| Embedding dim | 64 | 64 | 64 |
| Hidden size | 256 | 256 | 256 |
| Layers | 2 | 2 | 2 |
| Dropout | 0.5 | 0.5 | 0.35 |
| Max epochs | 80 | 80 | 80 |
| Early stop patience | 12 | 12 | 12 |
| Optimizer | Adam | RMSprop | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | CosineAnnealingLR |
| **Best CE Loss** | **0.760** | **1.113** | **0.720** |
| **Trainable Params** | **223,325** | **865,885** | **363,101** |

### Evaluation Results

| Model | Params | Novelty ↑ | Diversity ↑ | Realism ↑ | Lex. Variety | Best CE ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla RNN | 223,325 | 82.0% | 98.0% | 85.8 | 15.0 | 0.760 |
| BLSTM | 865,885 | 95.5% | **99.5% ✅** | **86.1 ✅** | **15.5** | 1.113 |
| RNN+Attention | 363,101 | **97.5% ✅** | 97.5% | 39.2 | 8.4 | **0.720 ✅** |

### Convergence Speed

| Loss Threshold | Vanilla RNN | BLSTM | RNN+Attention |
|:---:|:---:|:---:|:---:|
| ≤ 2.5 | Epoch 2 | Epoch 3 | **Epoch 1** |
| ≤ 2.0 | Epoch 3 | Epoch 6 | **Epoch 2** |
| ≤ 1.5 | Epoch 6 | Epoch 15 | **Epoch 5** |
| ≤ 1.25 | Epoch 9 | Epoch 36 | **Epoch 7** |

### 🎲 Sample Generated Names

**Vanilla RNN** — *most realistic, culturally plausible*
```
Sachin Hegde  ·  Anjali Mishra  ·  Kavya Banerjee  ·  Vikram Kulkarni  ·  Meera Tripathi
```

**BLSTM** — *most diverse, strong vowel harmony*
```
Kartik Pal  ·  Riya Malhotra  ·  Simran Joshi  ·  Deepa Deshpande  ·  Nikhil Bhandari
```

**RNN + Attention** — *highest novelty, consistent surname patterns*
```
Krish Iyer  ·  Nikhil Roy  ·  Ravi Roy  ·  Rajesh Iyer  ·  Aditya Selkar
```

### 🗺️ Model Recommendation Guide

| Your Priority | Recommended Model | Reason |
|:---|:---:|:---|
| Most realistic names | **Vanilla RNN** | Best realism + appropriate name lengths |
| Highest diversity | **BLSTM** | 99.5% diversity, best realism score |
| Maximum novelty | **RNN+Attention** | 97.5% novelty *(apply length filter post-generation)* |
| Fast training + balance | **Vanilla RNN** | Fastest convergence, good all-round performance |

## 📂 Q2 Outputs

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
| `training_curves_all.png` | Loss curves — all 3 models over 80 epochs |
| `training_curves_best_loss_bar.png` | Best CE loss per model (bar chart) |
| `training_curves_convergence.png` | Epochs to reach each loss threshold |
| `training_curves_loss_gap.png` | Per-epoch loss gap vs best model |
| `training_curves_lr_schedule.png` | LR schedule overlaid on loss curves |
| `evaluation_comparison.png` | Novelty / diversity / radar chart |
| `eval_name_length_dist.png` | Histogram + box-plot of generated name lengths |
| `eval_score_matrix.png` | Colour-coded per-metric score matrix |
| `logs/training_log.txt` | Full training log (all 3 scripts append here) |

---

## 📚 References

### Q1 — Word Embeddings

1. T. Mikolov et al., "Efficient estimation of word representations in vector space," ICLR, 2013. [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
2. T. Mikolov et al., "Distributed representations of words and phrases," NeurIPS 26, 2013. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
3. T. Mikolov, W.-T. Yih, G. Zweig, "Linguistic regularities in continuous space word representations," NAACL-HLT, 2013. [ACL Anthology](https://aclanthology.org/N13-1090)
4. R. Řehůřek & P. Sojka, "Software framework for topic modelling with large corpora," LREC Workshop, 2010. [gensim](https://radimrehurek.com/gensim/)
5. L. van der Maaten & G. Hinton, "Visualizing data using t-SNE," JMLR 9, 2008. [jmlr.org](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
6. Y. Goldberg & O. Levy, "word2vec Explained," arXiv:1402.3722, 2014. [arXiv](https://arxiv.org/abs/1402.3722)
7. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," NeurIPS 32, 2019. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

### Q2 — RNN Name Generation

1. J. L. Elman, "Finding structure in time," Cognitive Science, 14(2), 1990. [doi](https://doi.org/10.1207/s15516709cog1402_1)
2. S. Hochreiter & J. Schmidhuber, "Long short-term memory," Neural Computation, 9(8), 1997. [doi](https://doi.org/10.1162/neco.1997.9.8.1735)
3. D. Bahdanau, K. Cho, Y. Bengio, "Neural machine translation by jointly learning to align and translate," ICLR, 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
4. A. Vaswani et al., "Attention is all you need," NeurIPS 30, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
5. A. Graves, "Generating sequences with recurrent neural networks," arXiv:1308.0850, 2013. [arXiv](https://arxiv.org/abs/1308.0850)
6. D. P. Kingma & J. Ba, "Adam: A method for stochastic optimization," ICLR, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
7. I. Loshchilov & F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," ICLR, 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
8. J. L. Ba et al., "Layer normalization," arXiv:1607.06450, 2016. [arXiv](https://arxiv.org/abs/1607.06450)
9. A. Karpathy, J. Johnson & L. Fei-Fei, "Visualizing and understanding recurrent networks," arXiv:1506.02078, 2015. [arXiv](https://arxiv.org/abs/1506.02078)

---

## 🔧 General Notes

| # | Note | Detail |
|:---:|:---|:---|
| 🌐 | **Data Sources** | Q1: 3 IITJ pages scraped: Academic Regulations (93,175 words), Circulars (1,840), CSE Projects (3,559) |
| 🎲 | **Reproducibility** | All random seeds fixed to **42** |
| 📚 | **Vocabulary** | Q1 uses `min_count=1` to retain the full vocabulary on this small corpus |
| 🏆 | **Composite Score** |Q1: Formula: `0.4×SimilarityGap + 0.3×AnalogyHitRate + 0.2×AvgTop1Cosine + 0.1×VocabCoverage` |
| 🔡 | **Full-name Training** | Q2: space is in the vocabulary — models train on e.g. `"Mahek Gadiya"` as one character sequence |
| 📋 | **Shared Log** | Q2: all 3 training scripts append to `logs/training_log.txt` |
| ⚙️ | **Common Hyperparameters** | Q2: E=64, H=256, L=2, max epochs=80, early-stop patience=12 (shared across all 3 models) |

---

<div align="center">

<br/>

![Wave](https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=100&section=footer)

**Made with ❤️ by Mahek Shankesh Gadiya**

`M25CSA011` · CSE (MTech-AI) · IIT Jodhpur · 2026

</div>
