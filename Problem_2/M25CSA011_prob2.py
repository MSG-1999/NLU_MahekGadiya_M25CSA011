#Name : Mahek Gadiya
#Roll : M25CSA011

#------------------------------------------------------

import sys
from collections import Counter

def get_stats(ids):
    """this function Calculates frequency of adjacent pairs of token IDs.""" 
    counts = Counter()
    for sequence in ids:
        for pair in zip(sequence, sequence[1:]):
            counts[pair] += 1
    return counts

# Once we find the best pair, this function replaces 
# every instance of that pair with a new single ID
def merge(ids, pair, idx):
    """Replaces occurrences of a pair with a new merged token ID."""
    new_ids = []
    for sequence in ids:
        new_seq = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i+1] == pair[1]:
                new_seq.append(idx)
                i += 2 # Skip two steps because we merged two items into one
            else:
                new_seq.append(sequence[i])
                i += 1
        new_ids.append(new_seq)
    return new_ids

def main():
    #1 Making sure the user provided a corpus file in the terminal
    if len(sys.argv) < 2:
        print("Usage: python M25CSA011_prob2.py corpus.txt")
        return

    corpus_path = sys.argv[1]
    K = 15  # Total number of merges we want to do !

    #2 Read the training data (Training Corpus)
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: {corpus_path} not found.")
        return

    #3 Initialization
    # We include the spaces from the text to match your desired output
    token_ids = [list(line.encode("utf-8")) for line in text.splitlines() if line.strip()]
    vocab = {i: bytes([i]) for i in range(256)}
    current_id = 256
    
    print(f"Starting BPE with {K} merges...\n")

    #4 Main loop for the BPE algorithm
    for i in range(K):
        stats = get_stats(token_ids)
        if not stats:
            break
        
        #Pick the pair that appears most frequently
        best_pair = max(stats, key=stats.get)
        
        #Create the new token and update the token list
        token_ids = merge(token_ids, best_pair, current_id)
        
        #Decoding the pair so we can see what characters were merged
        p1 = vocab[best_pair[0]].decode('utf-8', errors='replace')
        p2 = vocab[best_pair[1]].decode('utf-8', errors='replace')
        
        #Updating our vocab dictionary with the new merged bytes
        vocab[current_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        #Show the user which pair got merged into which ID (Merge X: (p1, p2) -> ID Y)
        print(f"Merge {i+1}: ({p1}, {p2}) -> ID {current_id}")
        current_id += 1

    #5 Print out all the new tokens we created
    print("\n--- Final Vocabulary (New Tokens) ---")
    for tid in range(256, current_id):
        token_val = vocab[tid].decode('utf-8', errors='replace')
        # EXACT FORMAT REQUESTED: Token ID X: 'val'
        print(f"Token ID {tid}: '{token_val}'")

if __name__ == "__main__":
    main()
