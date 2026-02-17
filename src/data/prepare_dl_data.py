import os
import numpy as np
from src.data.loader import load_pre_split_data
from src.data.dl_preprocess import (
    build_vocab,
    encode_texts,
    pad_sequences,
    analyze_lengths
)
from src.embeddings.glove_loader import build_glove_embedding_matrix


ARTIFACT_DIR = "artifacts/dl_data"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

GLOVE_PATH = "data/external/glove/glove.6B.300d.txt"
MAX_LEN = 50  # Based on your stats


def main():

    print("Loading split data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_pre_split_data()

    print("\nAnalyzing length distribution...")
    analyze_lengths(train_texts)

    print("\nBuilding vocabulary...")
    word2idx = build_vocab(train_texts)

    print("\nEncoding text...")
    train_encoded = encode_texts(train_texts, word2idx)
    val_encoded = encode_texts(val_texts, word2idx)
    test_encoded = encode_texts(test_texts, word2idx)

    print("\nPadding sequences...")
    X_train = pad_sequences(train_encoded, MAX_LEN)
    X_val = pad_sequences(val_encoded, MAX_LEN)
    X_test = pad_sequences(test_encoded, MAX_LEN)

    print("Train shape:", X_train.shape)

    print("\nSaving processed arrays...")
    np.save(os.path.join(ARTIFACT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(ARTIFACT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(ARTIFACT_DIR, "X_test.npy"), X_test)

    np.save(os.path.join(ARTIFACT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(ARTIFACT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(ARTIFACT_DIR, "y_test.npy"), y_test)

    np.save(os.path.join(ARTIFACT_DIR, "word2idx.npy"), word2idx)

    print("\nBuilding GloVe embedding matrix...")
    embedding_matrix = build_glove_embedding_matrix(
        word2idx,
        GLOVE_PATH,
        embed_dim=300
    )

    np.save(
        os.path.join(ARTIFACT_DIR, "glove_embedding_matrix.npy"),
        embedding_matrix
    )

    print("Embedding matrix shape:", embedding_matrix.shape)
    print("\nDL data preparation complete.")


if __name__ == "__main__":
    main()
