# Name : Mahek Gadiya
# Roll : M25CSA011

#---------------------------------------------------------------------------

# Note : Before Running Please upload
       # 1) neg.txt
       # 2) pos.txt


# -------------------------------------------------
#import libraries &  Utility Functions
# -------------------------------------------------

import math
import random

def read_file(filename):
    """
    Reads a text file and returns a list of lowercase sentences.
    Each line is treated as one sentence.
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines if line.strip()]


def tokenize(sentence):
    """
    Tokenizes a sentence using simple whitespace tokenization.
    """
    return sentence.lower().split()


def split_data(data, train_ratio=0.72, val_ratio=0.15):
    """
    Splits data into train, validation, and test sets.
    Remaining data after train and validation is used as test data.
    """
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


# -------------------------------------------------
# Naive Bayes Training
# -------------------------------------------------

def train_naive_bayes(pos_sentences, neg_sentences):
    """
    Trains a Multinomial Naive Bayes model using word frequencies
    and Laplace smoothing.
    """
    pos_counts = {}
    neg_counts = {}
    vocabulary = set()

    pos_total_words = 0
    neg_total_words = 0

    # Count words in positive class
    for sentence in pos_sentences:
        for word in tokenize(sentence):
            vocabulary.add(word)
            pos_counts[word] = pos_counts.get(word, 0) + 1
            pos_total_words += 1

    # Count words in negative class
    for sentence in neg_sentences:
        for word in tokenize(sentence):
            vocabulary.add(word)
            neg_counts[word] = neg_counts.get(word, 0) + 1
            neg_total_words += 1

    model = {
        "pos_counts": pos_counts,
        "neg_counts": neg_counts,
        "pos_total": pos_total_words,
        "neg_total": neg_total_words,
        "vocab_size": len(vocabulary),
        "pos_prior": len(pos_sentences) / (len(pos_sentences) + len(neg_sentences)),
        "neg_prior": len(neg_sentences) / (len(pos_sentences) + len(neg_sentences))
    }

    return model


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict(sentence, model):
    """
    Predicts sentiment (POSITIVE or NEGATIVE) for a given sentence.
    """
    words = tokenize(sentence)

    log_pos = math.log(model["pos_prior"])
    log_neg = math.log(model["neg_prior"])

    # Laplace smoothing starts here
    for word in words:

        # Add 1 to word counts to avoid zero probability
        pos_wc = model["pos_counts"].get(word, 0) + 1
        neg_wc = model["neg_counts"].get(word, 0) + 1

        # Add vocabulary size to denominator
        pos_prob = pos_wc / (model["pos_total"] + model["vocab_size"])
        neg_prob = neg_wc / (model["neg_total"] + model["vocab_size"])

        log_pos += math.log(pos_prob)
        log_neg += math.log(neg_prob)

    return "POSITIVE" if log_pos > log_neg else "NEGATIVE"


# -------------------------------------------------
# Main Program
# -------------------------------------------------

def main():
    """
    Trains the Naive Bayes model and runs interactive sentiment prediction.
    """
    random.seed(42)

    pos_data = read_file("pos.txt")
    neg_data = read_file("neg.txt")

    # Split data
    pos_train, pos_val, pos_test = split_data(pos_data)
    neg_train, neg_val, neg_test = split_data(neg_data)

    # Train model
    model = train_naive_bayes(pos_train, neg_train)

    # ---------------- Validation ----------------
    val_correct = 0
    val_total = 0

    for s in pos_val:
        if predict(s, model) == "POSITIVE":
            val_correct += 1
        val_total += 1

    for s in neg_val:
        if predict(s, model) == "NEGATIVE":
            val_correct += 1
        val_total += 1

    print(f"\nValidation Accuracy: {val_correct / val_total:.2f}")

    # ---------------- Test (Optional) ----------------
    test_correct = 0
    test_total = 0

    for s in pos_test:
        if predict(s, model) == "POSITIVE":
            test_correct += 1
        test_total += 1

    for s in neg_test:
        if predict(s, model) == "NEGATIVE":
            test_correct += 1
        test_total += 1

    print(f"Test Accuracy: {test_correct / test_total:.2f}")

    # ---------------- Interactive Mode ----------------
    while True:
        print("\nEnter a sentence to classify sentiment (type 'exit' to quit):")
        user_input = input("-> ")

        if user_input.lower() == "exit":
            break

        result = predict(user_input, model)
        print("Predicted Sentiment:", result)


if __name__ == "__main__":
    main()