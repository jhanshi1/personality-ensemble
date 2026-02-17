import os
import numpy as np
from collections import Counter


def build_vocab(texts, max_vocab_size=30000, min_freq=2):
    counter = Counter()

    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    words = [w for w, f in counter.items() if f >= min_freq]
    words = sorted(words, key=lambda w: counter[w], reverse=True)
    words = words[:max_vocab_size]

    word2idx = {"<PAD>": 0, "<UNK>": 1}

    for i, word in enumerate(words, start=2):
        word2idx[word] = i

    print("Vocabulary size:", len(word2idx))
    return word2idx


def encode_texts(texts, word2idx):
    encoded = []
    for text in texts:
        tokens = text.split()
        seq = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
        encoded.append(seq)
    return encoded


def pad_sequences(sequences, max_len):
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded


def analyze_lengths(texts):
    lengths = [len(text.split()) for text in texts]

    print("Mean length:", np.mean(lengths))
    print("95th percentile:", np.percentile(lengths, 95))
    print("Max length:", np.max(lengths))

    return lengths
