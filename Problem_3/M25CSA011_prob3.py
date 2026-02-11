# Name : Mahek Gadiya
# Roll : M25CSA011

#---------------------------------------------------------------------------
# Note : Before Running Please upload
#        1) neg.txt
#        2) pos.txt
#---------------------------------------------------------------------------


#-------------------------------------------------
# Import libraries
# -------------------------------------------------

import math
import random

#-------------------------------------------------
# Utility Functions
# -------------------------------------------------

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines if line.strip()]


def tokenize(sentence):
    return sentence.lower().split()


def split_data(data, train_ratio=0.7, val_ratio=0.15):
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

    pos_counts = {}
    neg_counts = {}
    vocabulary = set()

    pos_total_words = 0
    neg_total_words = 0

    for sentence in pos_sentences:
        for word in tokenize(sentence):
            vocabulary.add(word)
            pos_counts[word] = pos_counts.get(word, 0) + 1
            pos_total_words += 1

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

    words = tokenize(sentence)

    log_pos = math.log(model["pos_prior"])
    log_neg = math.log(model["neg_prior"])

    for word in words:
        pos_wc = model["pos_counts"].get(word, 0) + 1
        neg_wc = model["neg_counts"].get(word, 0) + 1

        pos_prob = pos_wc / (model["pos_total"] + model["vocab_size"])
        neg_prob = neg_wc / (model["neg_total"] + model["vocab_size"])

        log_pos += math.log(pos_prob)
        log_neg += math.log(neg_prob)

    return "POSITIVE" if log_pos > log_neg else "NEGATIVE"


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------

def evaluate(pos_data, neg_data, model, dataset_name):
    
    # counting correct predictions for both classes
    pos_correct = 0
    neg_correct = 0
   
    # checking positive sentences
    for s in pos_data:
        if predict(s, model) == "POSITIVE":
            pos_correct += 1

    # checking negative sentences
    for s in neg_data:
        if predict(s, model) == "NEGATIVE":
            neg_correct += 1
            
    # total samples in this dataset
    total = len(pos_data) + len(neg_data)
    total_correct = pos_correct + neg_correct

    # calculating accuracies in percentage
    overall_acc = (total_correct / total) * 100
    pos_acc = (pos_correct / len(pos_data)) * 100
    neg_acc = (neg_correct / len(neg_data)) * 100

    print(f"\n------ {dataset_name} Results ------")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(f"Positive Class Accuracy: {pos_acc:.2f}%")
    print(f"Negative Class Accuracy: {neg_acc:.2f}%")


# -------------------------------------------------
# Main Program
# -------------------------------------------------

def main():

    random.seed(42)

    pos_data = read_file("pos.txt")
    neg_data = read_file("neg.txt")

    # Split data
    pos_train, pos_val, pos_test = split_data(pos_data)
    neg_train, neg_val, neg_test = split_data(neg_data)

    # Print Split Information
    print("\n------ Dataset Split Information ------")
    print("train_val_test = (70% - 15% - 15%)\n")

    print(f"Positive Sentences: {len(pos_data)}")
    print(f"  Train: {len(pos_train)}")
    print(f"  Validation: {len(pos_val)}")
    print(f"  Test: {len(pos_test)}")

    print(f"\nNegative Sentences: {len(neg_data)}")
    print(f"  Train: {len(neg_train)}")
    print(f"  Validation: {len(neg_val)}")
    print(f"  Test: {len(neg_test)}")

    # Train model
    model = train_naive_bayes(pos_train, neg_train)

    # Evaluate Training
    evaluate(pos_train, neg_train, model, "Training")

    # Evaluate Validation
    evaluate(pos_val, neg_val, model, "Validation")

    # Evaluate Test
    evaluate(pos_test, neg_test, model, "Test")

    # Interactive Mode
    while True:
        print("\nEnter a sentence to classify sentiment (type 'exit' to quit):")
        user_input = input("-> ")

        if user_input.lower() == "exit":
            break

        result = predict(user_input, model)
        print("Predicted Sentiment:", result)


if __name__ == "__main__":
    main()