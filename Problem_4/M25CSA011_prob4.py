
#Note: 1st upload dataset file:"News_Category_Dataset_v3.json"

# ============================================================
# Name : Mahek Gadiya
# Roll : M25CSA011
# Subject : Natural Language Understanding
# Assignment : News Classification (SPORTS vs POLITICS)
# ============================================================


# Header & Dataset Loading

import json
import pandas as pd

data = []

with open("News_Category_Dataset_v3.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            continue  # skip broken lines if any

df = pd.DataFrame(data)

print("\nDataset loaded successfully!")
print("Total rows:", len(df))
df.head()

print("\nImmediately after loading:")
print(df["category"].value_counts())
print("Total:", len(df))

#----------------------------------------------------------------

# Filtering classes (Keeping only SPORTS and POLITICS news)
df = df[df["category"].isin(["SPORTS", "POLITICS"])]

# Reset index
df = df.reset_index(drop=True)
print("Filtered samples:", len(df))
df["category"].value_counts()

# Separate classes
df_sports = df[df["category"] == "SPORTS"]
df_politics = df[df["category"] == "POLITICS"]

#---------------------------------------------------

# (Class Balancing) Undersample POLITICS to match SPORTS count
# To Remove Baised
df_politics_balanced = df_politics.sample(
    n=len(df_sports),
    random_state=42
)

# Combine balanced dataset
df_balanced = pd.concat([df_sports, df_politics_balanced])

# Shuffle data
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced class distribution:")
print(df_balanced["category"].value_counts())

#---------------------------------------------------
# Text Preparation & Label Encoding

# Combine headline and short_description
df["text"] = df["headline"] + " " + df["short_description"]

# Create numeric labels
df["label"] = df["category"].map({
    "SPORTS": 0,
    "POLITICS": 1
})

# Keep only required columns
df = df[["text", "label"]]

df.head()

#---------------------------------------------------
# Train / Validation / Test Split

from sklearn.model_selection import train_test_split

# Features and labels
X = df["text"]
y = df["label"]

# Step 1: Split out TEST set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

# Step 2: Split remaining 90% into TRAIN (70%) and VALIDATION (20%)
# 20 / (70 + 20) = 20 / 90 ≈ 0.2222
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2222,
    random_state=42,
    stratify=y_temp
)

# Print sizes
print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

#-----------------------------------------------------------
# TF-IDF Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=20000
)

# Fit ONLY on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform validation and test data
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Check shapes
print("\nTrain TF-IDF shape:", X_train_tfidf.shape)
print("Validation TF-IDF shape:", X_val_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

#------------------------------------------------------------
# Train Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_tfidf, y_train)

    # Training accuracy
    train_pred = model.predict(X_train_tfidf)
    train_acc = accuracy_score(y_train, train_pred) * 100

    # Validation accuracy
    val_pred = model.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, val_pred) * 100

    results[name] = (train_acc, val_acc)

    print(f"\n{name} (Train Accuracy): {train_acc:.2f}%")
    print(f"{name} (Validation Accuracy): {val_acc:.2f}%")

# Test Accuracy
from sklearn.metrics import accuracy_score

print("\nTest Accuracy for all models:")

for name, model in models.items():
    y_test_pred = model.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    print(f"{name} (Test Accuracy): {test_acc:.2f}%")
    
#-----------------------------------------------------------
# Confusion Matrices

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for name, model in models.items():
    y_test_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_test_pred, normalize="true")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["SPORTS", "POLITICS"]
    )

    disp.plot(cmap="Blues", values_format=".2f")
    plt.title(f"{name} - Normalized Confusion Matrix (Test Set)")
    plt.show()

#--------------------------------------------------------------
# Final Model Selection (Linear SVM)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Final selected model
best_model = models["Linear SVM"]

# Predict on test set
y_test_pred = best_model.predict(X_test_tfidf)

# Test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred) * 100
print(f"\nTest Accuracy (Best Model- Linear SVM): {test_accuracy:.2f}%")

# Classification report (required)
print("\nClassification Report (Test Set):\n")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["SPORTS", "POLITICS"]
    )
)

#------------------------------------------------------------
#Save (Model & Vectorizer)

import joblib

# Save the trained Linear SVM model
joblib.dump(best_model, "linear_svm_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

#-----------------------------------------------------------
# Train vs Validation Accuracy Plot

import matplotlib.pyplot as plt
import numpy as np

# Model names
model_names = ["Naive Bayes", "Logistic Regression", "Linear SVM"]

# Accuracies (in %)
train_acc = [results[m][0] for m in model_names]
val_acc   = [results[m][1] for m in model_names]

x = np.arange(len(model_names))
width = 0.32

plt.figure(figsize=(10, 6))

# Softer, professional colors
bars_train = plt.bar(
    x - width/2,
    train_acc,
    width,
    label="Train Accuracy",
    color="#4C72B0"
)

bars_val = plt.bar(
    x + width/2,
    val_acc,
    width,
    label="Validation Accuracy",
    color="#DD8452"
)

# Axis labels & title
plt.xticks(x, model_names, fontsize=13)
plt.ylabel("Accuracy (%)", fontsize=13)
plt.xlabel("Models", fontsize=13)
plt.title("Train vs Validation Accuracy (Generalization Analysis) \n", fontsize=15, pad=12)

# Y-axis range (tight for clarity)
plt.ylim(94, 100)

# Subtle grid
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate bars
def annotate(bars):
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.15,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11
        )

annotate(bars_train)
annotate(bars_val)

plt.legend(fontsize=15, frameon=False)
plt.tight_layout()


plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Model names
model_names = ["Naive Bayes", "Logistic Regression", "Linear SVM"]

# Compute test accuracy in %
test_acc = []
for m in model_names:
    y_pred = models[m].predict(X_test_tfidf)
    test_acc.append(accuracy_score(y_test, y_pred) * 100)

x = np.arange(len(model_names))

plt.figure(figsize=(8, 5))

bars = plt.bar(
    x,
    test_acc,
    color="#4C72B0",
    edgecolor="black"
)

# Labels and title
plt.xticks(x, model_names, fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.title("Test Accuracy Comparison of Models", fontsize=14)

# Y-axis range
plt.ylim(85, 100)

# Grid for readability
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.3,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc

# Decision scores (not predict_proba for SVM)
y_scores = best_model.decision_function(X_test_tfidf)

fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Linear SVM)")
plt.legend()
plt.grid(alpha=0.4)
plt.show()

from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# Classification report as dict
report = classification_report(
    y_test,
    y_test_pred,
    target_names=["SPORTS", "POLITICS"],
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()

# Plot precision, recall, f1-score
df_report.loc[["SPORTS", "POLITICS"], ["precision", "recall", "f1-score"]].plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Precision, Recall and F1-score (Linear SVM)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()