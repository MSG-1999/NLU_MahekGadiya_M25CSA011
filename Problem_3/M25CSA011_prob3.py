# Name : Mahek Gadiya
# Roll : M25CSA011
# Sub  : NLU
# Problem : 3

#---------------------------------------------------------------------------
# Note : Before Running Please upload
#        1) neg.txt
#        2) pos.txt
#---------------------------------------------------------------------------


#-------------------------------------------------
#  1) Import libraries
# -------------------------------------------------

import math  # using for logarithmic probability calculations
import random # Using for shuffling dataset before splitting

#-------------------------------------------------
# 2) Utility Functions
# -------------------------------------------------

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Remove empty lines and convert to lowercase
    return [line.strip().lower() for line in lines if line.strip()]

# Spliting sentence into words (tokens)
def tokenize(sentence):
    return sentence.lower().split()

 # spliting the data set into train/val/test ratio - 70%, 15%, 15 %
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
# 3) Naive Bayes Training
# -------------------------------------------------

def train_naive_bayes(pos_sentences, neg_sentences):

    pos_counts = {}  # Dictionary to store word counts in positive class
    neg_counts = {}  # Dictionary to store word counts in negative class
    vocabulary = set() # Set to store unique words

    pos_total_words = 0
    neg_total_words = 0
    
    # Count words in positive sentences
    for sentence in pos_sentences:
        for word in tokenize(sentence):
            vocabulary.add(word)
            pos_counts[word] = pos_counts.get(word, 0) + 1
            pos_total_words += 1
            
    # Count words in negative sentences
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
# 4) Prediction
# -------------------------------------------------

def predict(sentence, model):

    words = tokenize(sentence)
    
    # Starting with log prior probabilities
    log_pos = math.log(model["pos_prior"])
    log_neg = math.log(model["neg_prior"])

    for word in words:
        pos_wc = model["pos_counts"].get(word, 0) + 1     # Apply Laplace smoothing
        neg_wc = model["neg_counts"].get(word, 0) + 1     # Apply Laplace smoothing

        pos_prob = pos_wc / (model["pos_total"] + model["vocab_size"])
        neg_prob = neg_wc / (model["neg_total"] + model["vocab_size"])

        log_pos += math.log(pos_prob)
        log_neg += math.log(neg_prob)

    # Returning class with higher probability
    return "POSITIVE" if log_pos > log_neg else "NEGATIVE"


# -------------------------------------------------
# 5) Evaluation Function
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
# 6) Main Program
# -------------------------------------------------

def main():

    random.seed(42)
    
    # Loading datasets
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

    # Interactive prediction Mode
    while True:
        print("\nEnter a sentence to classify sentiment (type 'exit' to quit):")
        user_input = input("-> ")

        if user_input.lower() == "exit":
            break

        result = predict(user_input, model)
        print("Predicted Sentiment:", result)

if __name__ == "__main__":
    main()