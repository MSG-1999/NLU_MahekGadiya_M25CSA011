from __future__ import annotations

import random
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# fixing seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
class Word2VecVocab:
    def __init__(self, sentences: List[List[str]], min_count: int = 1):
        
        # counting word frequency
        freq = Counter(tok for sent in sentences for tok in sent)
        
        # keeping words with frequency >= min_count
        vocab_items = [(w, c) for w, c in freq.most_common() if c >= min_count]
        
        # mapping word → index and index ---> word
        self.word2idx: Dict[str, int] = {w: i for i, (w, _) in enumerate(vocab_items)}
        self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}
        self.vocab_size: int = len(self.word2idx)
        
        # creating unigram distribution (for negative sampling)
        counts = np.array([c for _, c in vocab_items], dtype=np.float64)
        dist = counts ** 0.75
        self.unigram_dist = dist / dist.sum()

    def encode(self, word: str) -> Optional[int]:
        return self.word2idx.get(word)   

    def sample_negatives(self, n: int, exclude: Set[int]) -> List[int]:
        candidates = np.random.choice(self.vocab_size, size=n * 3, p=self.unigram_dist)
        result = [int(idx) for idx in candidates if int(idx) not in exclude]
        return result[:n]
 

class CBOWDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Word2VecVocab, window_size: int = 5):
        self.window_size = window_size
        self.vocab = vocab
        self.pairs: List[Tuple] = []
        fixed_ctx_len = 2 * window_size
        
        # creating (context --> target) pairs
        for sent in sentences:
            indices = [vocab.encode(w) for w in sent]
            indices = [i for i in indices if i is not None]
            for center_pos, target in enumerate(indices):
                ctx = [indices[j] for j in range(max(0, center_pos - window_size),
                       min(len(indices), center_pos + window_size + 1)) if j != center_pos]
                if not ctx:
                    continue
                ctx = ctx[:fixed_ctx_len]
                ctx += [0] * (fixed_ctx_len - len(ctx))
                self.pairs.append((torch.tensor(ctx, dtype=torch.long), target))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


class SGNSDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: Word2VecVocab,
                 window_size: int = 5, num_neg: int = 5):
        self.vocab = vocab
        self.num_neg = num_neg
        self.pairs: List[Tuple[int, int]] = []
        for sent in sentences:
            indices = [vocab.encode(w) for w in sent]
            indices = [i for i in indices if i is not None]
            for center_pos, target in enumerate(indices):
                for j in range(max(0, center_pos - window_size),
                               min(len(indices), center_pos + window_size + 1)):
                    if j == center_pos:
                        continue
                    self.pairs.append((target, indices[j]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        target, positive = self.pairs[idx]
        
        # sampling negative examples
        negatives = np.random.choice(self.vocab.vocab_size, size=self.num_neg, p=self.vocab.unigram_dist)
        return (torch.tensor(target, dtype=torch.long),
                torch.tensor(positive, dtype=torch.long),
                torch.tensor(negatives, dtype=torch.long))



# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOW, self).__init__()
        
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # output layer
        self.linear = nn.Linear(embedding_dim, vocab_size)
        nn.init.uniform_(self.embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, context_indices: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(context_indices)
        pooled = embeds.mean(dim=1)
        logits = self.linear(pooled)
        return F.log_softmax(logits, dim=1)

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings.weight.detach().cpu().numpy()


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(SkipGramNS, self).__init__()
        
        # separate embeddings for target and context
        self.target_embeddings  = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        nn.init.uniform_(self.target_embeddings.weight, -init_range, init_range)
        nn.init.constant_(self.context_embeddings.weight, 0.0)

    def forward(self, target: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        
        v_target    = self.target_embeddings(target).unsqueeze(1)
        v_positive  = self.context_embeddings(positive).unsqueeze(1)
        v_negatives = self.context_embeddings(negatives)
        pos_score = torch.bmm(v_target, v_positive.transpose(1, 2)).squeeze(-1).squeeze(-1)
        pos_loss  = F.logsigmoid(pos_score).mean()
        neg_scores = torch.bmm(v_target, v_negatives.transpose(1, 2)).squeeze(1)
        neg_loss   = F.logsigmoid(-neg_scores).sum(dim=1).mean()
        return -(pos_loss + neg_loss)

    def get_embeddings(self) -> np.ndarray:
        return self.target_embeddings.weight.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# WORD VECTOR UTILITY
# ─────────────────────────────────────────────────────────────────────────────
class ScratchWordVectors:
    def __init__(self, embeddings: np.ndarray, vocab: Word2VecVocab):
        
        
        # storing learned word embeddings
        self.vectors  = embeddings.astype(np.float32)
        self.vocab    = vocab
        
        # index --> word mapping (useful for output)
        self.index_to_key = [vocab.idx2word[i] for i in range(vocab.vocab_size)]
        
        # normalize vectors so cosine similarity becomes easy to compute
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self.vectors_norm = self.vectors / norms

    def __contains__(self, word: str) -> bool:
        return word in self.vocab.word2idx

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.vocab.word2idx[word]]

    def most_similar(self, word: Optional[str] = None, positive: Optional[List[str]] = None,
                     negative: Optional[List[str]] = None, topn: int = 5) -> List[Tuple[str, float]]:
        if word is not None:
            positive = [word]
        if positive is None: positive = []
        if negative is None: negative = []
        if isinstance(positive, str): positive = [positive]
        if isinstance(negative, str): negative = [negative]
        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        exclude: Set[int] = set()
        for w in positive:
            if w in self.vocab.word2idx:
                idx = self.vocab.word2idx[w]
                query += self.vectors_norm[idx]
                exclude.add(idx)
        for w in negative:
            if w in self.vocab.word2idx:
                idx = self.vocab.word2idx[w]
                query -= self.vectors_norm[idx]
                exclude.add(idx)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query /= q_norm
        similarities = self.vectors_norm @ query
        ranked = np.argsort(similarities)[::-1]
        results = []
        for idx in ranked:
            if int(idx) in exclude: continue
            results.append((self.index_to_key[idx], float(similarities[idx])))
            if len(results) >= topn: break
        return results
