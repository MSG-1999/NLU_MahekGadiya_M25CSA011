# Name : Mahek Gadiya
# Roll : M25CSA011
# Sub  : NLU
#                 Problem 2: Byte Pair Encoding (BPE)

# Note :1) first upload corpus.txt file
#       2) Run this program in terminal as below:
#          python RollNumber_prob2.py k corpus.txt
# ------------------------------------------------------

import sys
from collections import Counter


# This function counts how many times each adjacent pair appears
def get_stats(ids):
    counts = Counter()

    # Go through every line of tokens
    for sequence in ids:
        # zip(sequence, sequence[1:]) gives pairs like (a,b), (b,c)
        for pair in zip(sequence, sequence[1:]):
            counts[pair] += 1

    return counts


# This function merges the selected pair into a new token
def merge(ids, pair, idx):
    new_ids = []

    for sequence in ids:
        new_seq = []
        i = 0

        # Go through tokens one by one
        while i < len(sequence):

            # If current token and next token match the pair then
            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):
                new_seq.append(idx)   # Add new merged token
                i += 2                # Skip both tokens (since merged)
            else:
                new_seq.append(sequence[i])
                i += 1

        new_ids.append(new_seq)

    return new_ids


def main():

    # Checking if correct number of inputs are given
    if len(sys.argv) != 3:
        print("Usage: python RollNumber_prob2.py k corpus.txt")
        return

    # First input is number of merges (k), its depends on user
    K = int(sys.argv[1])

    # Second input is corpus file name
    corpus_path = sys.argv[2]

    # Print value of K (# Merges)
    print("\nNumber of merges (K):", K)

    # Try to open and read the file
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("File not found.")
        return

    # Convert each line into byte values 
    # its means we are doing byte-level BPE
    token_ids = [
        list(line.encode("utf-8"))
        for line in text.splitlines()
        if line.strip()
    ]

    # Create initial vocabulary of 256 byte values
    vocab = {i: bytes([i]) for i in range(256)}

    # New merged tokens will start from Token ID 256
    current_id = 256

    print(f"\nStarting BPE with {K} merges...\n")

    # Main BPE process
    for i in range(K):

        # Count frequency of all adjacent pairs
        stats = get_stats(token_ids)

        # If no pairs left, stop
        if not stats:
            break

        # Choose the most frequent pair
        best_pair = max(stats, key=stats.get)

        # Merge that pair
        token_ids = merge(token_ids, best_pair, current_id)

        # Add new merged token into vocabulary
        vocab[current_id] = (
            vocab[best_pair[0]] + vocab[best_pair[1]]
        )

        # here i am  Printing which merge happened
        p1 = vocab[best_pair[0]].decode('utf-8', errors='replace')
        p2 = vocab[best_pair[1]].decode('utf-8', errors='replace')
        print(f"Merge {i+1}: ({p1}, {p2}) -> ID {current_id}")

        current_id += 1

    # here i am Printing final merged tokens only
    print("\n--- Final Vocabulary (New Merged Tokens) ---")
    for tid in range(256, current_id):
        token_val = vocab[tid].decode('utf-8', errors='replace')
        print(f"Token ID {tid}: '{token_val}'")


# This ensures main() runs when we execute the file
if __name__ == "__main__":
    main()