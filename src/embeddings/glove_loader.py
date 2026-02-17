import numpy as np
import os


def build_glove_embedding_matrix(word2idx, glove_path, embed_dim=300):
    vocab_size = len(word2idx)

    print("Building embedding matrix...")
    print("Vocab size:", vocab_size)

    # Initialize matrix with random values
    embedding_matrix = np.random.normal(
        scale=0.6,
        size=(vocab_size, embed_dim)
    )

    # PAD token should be zero vector
    embedding_matrix[word2idx["<PAD>"]] = np.zeros(embed_dim)

    found = 0

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]

            if word in word2idx:
                vector = np.asarray(values[1:], dtype="float32")
                embedding_matrix[word2idx[word]] = vector
                found += 1

    coverage = found / vocab_size * 100

    print(f"GloVe coverage: {found}/{vocab_size} ({coverage:.2f}%)")

    return embedding_matrix
