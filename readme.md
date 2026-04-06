<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=260&section=header&text=Natural%20Language%20Understanding&fontSize=44&fontAlignY=36&desc=Assignment%201%20%E2%80%94%20IIT%20Jodhpur%20%7C%20CSE%20MTech-AI%20%7C%202026&descAlignY=58&descSize=17&animation=fadeIn&fontColor=fff&descColor=c9d1ff" />

</div>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Space+Mono&weight=700&size=19&duration=2800&pause=900&color=A78BFA&center=true&vCenter=true&width=750&lines=💬+Regex+NLP+%7C+🔤+BPE+Tokenization+%7C+📊+Naive+Bayes;📰+TF-IDF+%2B+Linear+SVM+→+97.44%25+Test+Accuracy;⭐+ROC-AUC+%3D+0.99+%7C+IIT+Jodhpur+%7C+M25CSA011)](https://git.io/typing-svg)

</div>

<br/>

<div align="center">

![](https://img.shields.io/badge/👩‍🎓%20Student-Mahek%20Shankesh%20Gadiya-7C3AED?style=for-the-badge&logoColor=white)
![](https://img.shields.io/badge/🎫%20Roll-M25CSA011-A855F7?style=for-the-badge)
![](https://img.shields.io/badge/🏛%20IIT%20Jodhpur-CSE%20MTech--AI-6D28D9?style=for-the-badge&logo=google-scholar&logoColor=white)

<br/>

![](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/NLP-4%20Problems-059669?style=for-the-badge&logo=openai&logoColor=white)
![](https://img.shields.io/badge/Best%20Accuracy-97.44%25-DC2626?style=for-the-badge&logo=checkmarx&logoColor=white)
![](https://img.shields.io/badge/ROC--AUC-0.99-F59E0B?style=for-the-badge)

<br/>

[![GitHub](https://img.shields.io/badge/View%20on%20GitHub-%E2%86%92-181717?style=for-the-badge&logo=github)](https://github.com/MSG-1999/NLU_MahekGadiya_M25CSA011/tree/Assignment_1)
[![Stars](https://img.shields.io/github/stars/MSG-1999/NLU_MahekGadiya_M25CSA011?style=for-the-badge&logo=starship&color=F59E0B)](https://github.com/MSG-1999/NLU_MahekGadiya_M25CSA011)

</div>

---

## 🗂 Assignment Overview

<div align="center">

| # | Problem | Core Technique | Key Highlight |
|:--:|:--------|:--------------|:-------------|
| 💬 **1** | **Regex Chatbot — Reggy++** | Regular Expressions | 13 tested scenarios, typo-tolerant mood detection |
| 🔤 **2** | **BPE Tokenizer** | Byte Pair Encoding | Byte-level, GPT-style from scratch |
| 📊 **3** | **Sentiment Classifier** | Naive Bayes (scratch) | Laplace smoothing, 70/15/15 split |
| 📰 **4** | **News Classifier** | TF-IDF + Linear SVM | **97.44% accuracy · AUC 0.99** |

</div>

---

## 📁 Repository Structure

```
📦 NLU_MahekGadiya_M25CSA011  ── Branch: Assignment_1
│
├── 📂 Problem_1
│   ├── 🐍 M25CSA011_prob1.py        Regex chatbot — Reggy++
│   ├── 📋 M25CSA011_prob1.log       13 logged conversation runs
│   └── 📝 M25CSA011_prob1.txt       Reflection on naturalness
│
├── 📂 Problem_2
│   └── 🐍 M25CSA011_prob2.py        BPE tokenizer from scratch
│
├── 📂 Problem_3
│   └── 🐍 M25CSA011_prob3.py        Naive Bayes sentiment classifier
│
└── 📂 Problem_4
    ├── 📂 Result_Graphs
    │   ├── 🖼  Train_vs_Val_acc.png
    │   ├── 🖼  Test_acc.png
    │   ├── 🖼  Linear_SVM.png
    │   ├── 🖼  Naive.png
    │   ├── 🖼  logistic_regression.png
    │   ├── 🖼  Precision,Recall,f1_score.png
    │   └── 🖼  ROC_Curve.png
    ├── 🐍 M25CSA011_prob4.py         Full ML pipeline
    ├── 📑 M25CSA011_prob4.pdf        Detailed project report
    ├── 💾 linear_svm_model.pkl       Saved trained SVM
    └── 💾 tfidf_vectorizer.pkl       Saved TF-IDF vectorizer
```

---

## 💬 Problem 1 — Regex Chatbot `Reggy++`

<div align="center">

![](https://img.shields.io/badge/Type-Rule--based%20NLP-7C3AED?style=flat-square)
![](https://img.shields.io/badge/Deps-Standard%20Library%20Only-059669?style=flat-square)
![](https://img.shields.io/badge/Test%20Runs-13%20Scenarios-F59E0B?style=flat-square)
![](https://img.shields.io/badge/No%20ML-Pure%20Regex-0EA5E9?style=flat-square)

</div>

A fully rule-based conversational chatbot — **no external libraries**, built on Python's `re`, `datetime`, and file I/O modules.

### ✨ Features

| Feature | Detail |
|---------|--------|
| 📅 Multi-format date parsing | `dd-mm-yyyy` · `mm-dd-yy` · `dd/mm/yy` · `dd Mon yyyy` · `dd Month yyyy` |
| 🔀 Ambiguous date handling | Prompts `dd` or `mm`? when both parts ≤ 12 |
| 🎂 Accurate age calculation | Accounts for whether birthday has passed this year |
| 😊 Mood detection | Positive · Negative · Mixed — tolerates typos like `happi`, `angr`, `tirredd` |
| 🛡 Input validation | Rejects empty, numeric, symbol-only names with guided recovery |
| 📝 Full session logging | Every run auto-appended to `.log` |
| 🔁 Multi-run loop | Asks to continue after each conversation |

<details>
<summary><b>🧪 All 13 Test Scenarios — click to expand</b></summary>

<br/>

```
GROUP 1 — Date Format Parsing
──────────────────────────────────────────────────────────────────
RUN  1  ✅  dd/mm/yyyy  → "20/12/2004"             Age: 21
RUN  2  ✅  dd-mm-yyyy  → "20-12-2004"             Age: 21
RUN  3  ✅  dd Mon yyyy → "05 oct 1999"            Age: 26
RUN  4  ✅  dd Month yyyy → "27 february 2008"     Age: 17

GROUP 2 — Mood Detection (including typos)
──────────────────────────────────────────────────────────────────
RUN  5  ✅  "very happieee"    → POSITIVE
RUN  6  ✅  "angr"             → NEGATIVE
RUN  7  ✅  "happy and sad"    → MIXED (ambiguous)

GROUP 3 — Ambiguous Date + Clarification
──────────────────────────────────────────────────────────────────
RUN  8  ✅  "05-10-1999"  → asks dd/mm → resolved correctly
RUN  9  ✅  "03/03/2011"  → asks dd/mm → resolved correctly

GROUP 4 — Failure Cases + Recovery
──────────────────────────────────────────────────────────────────
RUN 10  ✅  Invalid names: "", "123", "###"  → re-prompts with guidance
RUN 11  ✅  Invalid DOB: "27/40/1999"        → re-prompts with format hint
RUN 12  ✅  Unknown mood: "hi", "great"      → re-prompts with examples

GROUP 5 — Combined Edge Cases
──────────────────────────────────────────────────────────────────
RUN 13  ✅  Invalid DOB + Ambiguous DOB + Unknown mood → full recovery
```

</details>

```bash
python M25CSA011_prob1.py
```

---

## 🔤 Problem 2 — Byte Pair Encoding (BPE) Tokenizer

<div align="center">

![](https://img.shields.io/badge/Type-Tokenization-0EA5E9?style=flat-square)
![](https://img.shields.io/badge/Level-Byte--level-0284C7?style=flat-square)
![](https://img.shields.io/badge/From-Scratch-059669?style=flat-square)
![](https://img.shields.io/badge/Used%20in-GPT%20%7C%20LLaMA%20%7C%20Claude-6D28D9?style=flat-square)

</div>

Implements the BPE algorithm **from scratch at byte level** — the same tokenization foundation powering GPT-4, LLaMA, and Claude.

### ⚙️ How BPE Works

```
Step 1 → Initialize vocabulary with 256 byte tokens  (IDs 0–255)
Step 2 → Encode every corpus line as a list of byte IDs
Step 3 → Count frequency of every adjacent token pair
Step 4 → Pick the most frequent pair    e.g. ("e", "r")
Step 5 → Merge → new Token ID starts at 256, then 257, 258 …
Step 6 → Replace all corpus occurrences with merged token
Step 7 → Repeat K times — print each merge + final vocabulary
```

### 🔑 Key Implementation Details

| Component | Description |
|-----------|-------------|
| `get_stats()` | Counts adjacent pair frequencies using `collections.Counter` |
| `merge()` | Replaces all occurrences of a pair with a new token ID |
| Vocabulary | Starts with 256 bytes; each merge adds one new token |
| Output | Prints each merge step + all final merged tokens |

```bash
python M25CSA011_prob2.py <K> <corpus.txt>

# Example — 20 merges on a corpus
python M25CSA011_prob2.py 20 corpus.txt
```

> ⚠️ Place `corpus.txt` in the working directory before running.

---

## 📊 Problem 3 — Naive Bayes Sentiment Classifier

<div align="center">

![](https://img.shields.io/badge/Type-Sentiment%20Analysis-DB2777?style=flat-square)
![](https://img.shields.io/badge/Built-100%25%20From%20Scratch-059669?style=flat-square)
![](https://img.shields.io/badge/Split-70%20%2F%2015%20%2F%2015-F59E0B?style=flat-square)
![](https://img.shields.io/badge/Smoothing-Laplace%20Add--1-7C3AED?style=flat-square)

</div>

Binary sentiment classifier **(POSITIVE / NEGATIVE)** built entirely from scratch — no scikit-learn, no NLTK.

### 🔬 Implementation Details

| Component | Detail |
|-----------|--------|
| 📂 Input | `pos.txt` and `neg.txt` — one sentence per line |
| ⚖️ Split | 70 / 15 / 15 — shuffled with `random.seed(42)` |
| 🧂 Smoothing | Laplace add-1 — handles unseen words gracefully |
| 🔢 Scoring | Log-probability — prevents floating-point underflow |
| 🧠 Prior | Computed from training class proportions |
| 💬 Mode | Interactive prediction loop after evaluation |

### 📐 Math Behind the Classifier

$$P(\text{class} \mid \text{sentence}) \propto \log P(\text{class}) + \sum_{w} \log P(w \mid \text{class})$$

**Laplace-smoothed word probability:**

$$P(w \mid \text{class}) = \frac{\text{count}(w, \text{class}) + 1}{\text{total words in class} + |\text{vocab}|}$$

```bash
# Place pos.txt and neg.txt in same directory first
python M25CSA011_prob3.py
```

---

## 📰 Problem 4 — News Classification: SPORTS vs POLITICS

<div align="center">

![](https://img.shields.io/badge/🏆%20Best%20Model-Linear%20SVM-059669?style=for-the-badge)
![](https://img.shields.io/badge/Test%20Accuracy-97.44%25-DC2626?style=for-the-badge)
![](https://img.shields.io/badge/ROC--AUC-0.99-F59E0B?style=for-the-badge)
![](https://img.shields.io/badge/Dataset-HuffPost%20209K-0EA5E9?style=for-the-badge)

</div>

End-to-end supervised ML pipeline classifying HuffPost news headlines as **SPORTS** or **POLITICS** — three models trained, tuned, compared, and evaluated.

### 📦 Dataset

```
Name    :  HuffPost News Category Dataset v3
Source  :  https://www.kaggle.com/datasets/rmisra/news-category-dataset
Format  :  JSON Lines — one article per line
Total   :  209,527 articles  →  filtered to 40,679  →  balanced to 10,154
```

> ⚠️ Download `News_Category_Dataset_v3.json` → place in `Problem_4/` before running.

---

### 🔧 Complete Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA LOADING & FILTERING                                   │
│  Load 209,527 JSON articles  →  Filter: SPORTS + POLITICS only       │
│  Result: 40,679 articles  (POLITICS: 35,602  |  SPORTS: 5,077)       │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CLASS BALANCING (Random Undersampling)                     │
│  Undersample POLITICS → 5,077  to match SPORTS count                 │
│  Final balanced dataset: 5,077 + 5,077 = 10,154 articles             │
│  Shuffle with random_state=42                                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: TEXT CONSTRUCTION & LABEL ENCODING                         │
│  text = headline + " " + short_description                           │
│  SPORTS → 0  |  POLITICS → 1                                         │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: TRAIN / VALIDATION / TEST SPLIT                            │
│  70% Train  |  20% Validation  |  10% Test  (stratified)            │
│  ~7,107 train  |  ~2,031 val  |  ~1,016 test                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: TF-IDF FEATURE EXTRACTION                                  │
│  lowercase=True  |  stop_words="english"                             │
│  ngram_range=(1,2) — unigrams + bigrams                              │
│  max_features=20,000  |  fit on TRAIN only (no leakage)              │
│  Output shape: (7107, 20000)                                         │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6: MODEL TRAINING & COMPARISON                                │
│  ① Multinomial Naive Bayes  (MultinomialNB)                          │
│  ② Logistic Regression      (max_iter=1000, random_state=42)         │
│  ③ Linear SVM               (LinearSVC, random_state=42)  ← BEST    │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 7: EVALUATION & VISUALIZATION                                 │
│  Accuracy · Confusion Matrix · Classification Report                 │
│  ROC Curve (AUC) · Precision/Recall/F1 bar plots                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 8: MODEL SAVING                                               │
│  linear_svm_model.pkl  +  tfidf_vectorizer.pkl  (via joblib)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

### ⚙️ Implementation Details

<details>
<summary><b>📥 Stage 1 — Data Loading & Filtering</b></summary>

```python
import json, pandas as pd

data = []
with open("News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            continue

df = pd.DataFrame(data)
# Filter to only Sports and Politics
df = df[df["category"].isin(["SPORTS", "POLITICS"])].reset_index(drop=True)
# Result: 40,679 articles  (POLITICS: 35,602 | SPORTS: 5,077)
```

</details>

<details>
<summary><b>⚖️ Stage 2 — Class Balancing via Random Undersampling</b></summary>

The original class ratio was severely imbalanced (7:1 — Politics dominates):

| Category | Count |
|----------|-------|
| POLITICS | 35,602 |
| SPORTS | 5,077 |

To prevent the classifier from being biased toward Politics, **random undersampling** was applied:

```python
df_sports   = df[df["category"] == "SPORTS"]
df_politics = df[df["category"] == "POLITICS"]

# Undersample Politics to match Sports count
df_politics_balanced = df_politics.sample(n=len(df_sports), random_state=42)

df_balanced = pd.concat([df_sports, df_politics_balanced])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
# Final: 5,077 + 5,077 = 10,154 articles
```

</details>

<details>
<summary><b>✍️ Stage 3 — Text Construction & Label Encoding</b></summary>

```python
# Combine headline and short_description for richer signal
df["text"]  = df["headline"] + " " + df["short_description"]

# Numeric label encoding
df["label"] = df["category"].map({"SPORTS": 0, "POLITICS": 1})
```

</details>

<details>
<summary><b>✂️ Stage 4 — Stratified Train / Val / Test Split</b></summary>

```python
from sklearn.model_selection import train_test_split

# Step 1: Hold out 10% for test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# Step 2: Split remaining 90% → 70% train, 20% val
# 20/90 ≈ 0.2222 gives the correct 70/20/10 overall ratio
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2222, random_state=42, stratify=y_temp
)
```

</details>

<details>
<summary><b>🔢 Stage 5 — TF-IDF Feature Extraction</b></summary>

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into numerical vectors, assigning higher weights to terms that are distinctive to individual documents but rare across the corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,        # normalize case
    stop_words="english",  # remove common stopwords
    ngram_range=(1, 2),    # unigrams + bigrams (e.g. "world cup", "election results")
    max_features=20000     # top 20K features
)

# IMPORTANT: fit ONLY on training data to prevent data leakage
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf   = tfidf_vectorizer.transform(X_val)
X_test_tfidf  = tfidf_vectorizer.transform(X_test)
# Output shape: (n_samples, 20000)
```

Bigrams capture meaningful phrases like `"world cup"`, `"election results"`, `"government policy"` that unigrams alone would miss.

</details>

<details>
<summary><b>🤖 Stage 6 — Three Models Trained & Compared</b></summary>

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

models = {
    "Naive Bayes":          MultinomialNB(),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM":           LinearSVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_tfidf)) * 100
    val_acc   = accuracy_score(y_val,   model.predict(X_val_tfidf))   * 100
    print(f"{name}: Train={train_acc:.2f}%  Val={val_acc:.2f}%")
```

**Why Linear SVM wins:**
- Finds the maximum-margin hyperplane in 20,000-dimensional TF-IDF space
- Highly effective on sparse, high-dimensional text data
- Less prone to overfitting than kernel SVMs on this scale
- Smallest generalization gap (train 99.88% → test 97.44%)

</details>

<details>
<summary><b>📊 Stage 7 — Evaluation, ROC Curve & Visualizations</b></summary>

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)

# --- Classification Report ---
best_model  = models["Linear SVM"]
y_test_pred = best_model.predict(X_test_tfidf)
print(classification_report(y_test, y_test_pred, target_names=["SPORTS","POLITICS"]))

# --- Normalized Confusion Matrix ---
for name, model in models.items():
    cm = confusion_matrix(y_test, model.predict(X_test_tfidf), normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=["SPORTS","POLITICS"]).plot(cmap="Blues")

# --- ROC Curve (AUC) ---
y_scores = best_model.decision_function(X_test_tfidf)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)   # → 0.99
```

</details>

<details>
<summary><b>💾 Stage 8 — Model Saving for Reuse</b></summary>

```python
import joblib

# Save both model and vectorizer — both needed for consistent inference
joblib.dump(best_model,       "linear_svm_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# --- Reload and predict on new text ---
model     = joblib.load("linear_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

new_text  = ["The team won the championship after a dramatic penalty shootout"]
vec       = vectorizer.transform(new_text)
prediction = model.predict(vec)    # → [0] = SPORTS
```

</details>

---

### 📊 Results Summary

#### Train vs Validation Accuracy

<div align="center">

| Model | Train Accuracy | Validation Accuracy |
|:------|:-------------:|:-------------------:|
| Naive Bayes | 96.99% | 95.98% |
| Logistic Regression | 97.11% | 95.93% |
| ⭐ **Linear SVM** | **99.88%** | **97.76%** |

</div>

#### Test Accuracy

<div align="center">

| Model | Test Accuracy |
|:------|:------------:|
| Naive Bayes | 95.48% |
| Logistic Regression | 96.02% |
| ⭐ **Linear SVM** | **97.44%** |

</div>

#### 🏆 Linear SVM — Final Classification Report

<div align="center">

| Class | Precision | Recall | F1-Score | Support |
|:------|:---------:|:------:|:--------:|:-------:|
| ⚽ SPORTS | 0.95 | 0.84 | 0.89 | 508 |
| 🗳️ POLITICS | 0.98 | 0.99 | 0.99 | 3,560 |
| Macro Avg | 0.96 | 0.92 | 0.94 | 4,068 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** | **4,068** |

**ROC-AUC = 0.99** — near-perfect class discrimination

</div>

#### 🔍 Confusion Matrix Analysis

| Model | Sports Recall | Politics Recall | Observation |
|:------|:------------:|:---------------:|:------------|
| Naive Bayes | 66% | 100% | Biased toward Politics |
| Logistic Regression | 71% | 100% | Improved but still biased |
| **Linear SVM** | **84%** | **99%** | **Best balance across both classes** |

### 📈 Result Graphs

| Graph | File |
|:------|:-----|
| Train vs Validation Accuracy | `Result_Graphs/Train_vs_Val_acc.png` |
| Test Accuracy Comparison | `Result_Graphs/Test_acc.png` |
| Linear SVM Confusion Matrix | `Result_Graphs/Linear_SVM.png` |
| Naive Bayes Confusion Matrix | `Result_Graphs/Naive.png` |
| Logistic Regression Confusion Matrix | `Result_Graphs/logistic_regression.png` |
| Precision · Recall · F1-Score | `Result_Graphs/Precision,Recall,f1_score.png` |
| ROC Curve (AUC = 0.99) | `Result_Graphs/ROC_Curve.png` |

```bash
# Run the full pipeline
python M25CSA011_prob4.py

# Outputs saved automatically:
#   linear_svm_model.pkl    — reuse without retraining
#   tfidf_vectorizer.pkl    — consistent feature space
```

---

## 📦 Dependencies

> ✅ **Problems 1, 2, 3** — pure Python standard library only, zero install needed.

```bash
# Required only for Problem 4
pip install scikit-learn pandas matplotlib numpy joblib
```

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Joblib](https://img.shields.io/badge/Joblib-grey?style=flat-square)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=180&section=footer&text=Mahek%20Shankesh%20Gadiya%20%E2%80%94%20M25CSA011&fontSize=24&fontAlignY=52&fontColor=fff&desc=IIT%20Jodhpur%20%C2%B7%20CSE%20MTech-AI%20%C2%B7%202026&descSize=15&descColor=c9d1ffcc&descAlignY=72" width="100%"/>

[![GitHub](https://img.shields.io/badge/GitHub-MSG--1999-181717?style=for-the-badge&logo=github)](https://github.com/MSG-1999)

*If this helped you, please ⭐ star the repository!*

</div>
