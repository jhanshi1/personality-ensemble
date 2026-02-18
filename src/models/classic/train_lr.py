from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, hstack
import numpy as np
import os
import json

from src.utils.evaluation import evaluate_model
from src.data.loader import load_pre_split_data
from src.features.pos_features import extract_pos_features
from src.features.emotion_features import (
    load_nrc_lexicon,
    extract_emotion_features
)


def build_features(X_train, X_val, X_test):

    # Word-level ngrams
    word_vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=3
    )

    # Character-level ngrams
    char_vectorizer = CountVectorizer(
        analyzer="char",
        max_features=2000,
        ngram_range=(3, 5),
        min_df=3
    )

    # Fit on train only
    X_train_word = word_vectorizer.fit_transform(X_train)
    X_val_word = word_vectorizer.transform(X_val)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_val_char = char_vectorizer.transform(X_val)
    X_test_char = char_vectorizer.transform(X_test)

    # POS features
    pos_train = csr_matrix(extract_pos_features(X_train))
    pos_val = csr_matrix(extract_pos_features(X_val))
    pos_test = csr_matrix(extract_pos_features(X_test))

    # NRC emotion features
    nrc = load_nrc_lexicon("data/external/NRC-Emotion-Lexicon.txt")

    emo_train = csr_matrix(extract_emotion_features(X_train, nrc))
    emo_val = csr_matrix(extract_emotion_features(X_val, nrc))
    emo_test = csr_matrix(extract_emotion_features(X_test, nrc))

    # Combine everything
    X_train_vec = hstack([X_train_word, X_train_char, pos_train, emo_train])
    X_val_vec = hstack([X_val_word, X_val_char, pos_val, emo_val])
    X_test_vec = hstack([X_test_word, X_test_char, pos_test, emo_test])

    return X_train_vec, X_val_vec, X_test_vec


def train_lr(X_train, y_train):

    model = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",
            max_iter=3000,
            C=1.0,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":

    ARTIFACT_DIR = "artifacts/classic"
    RESULTS_DIR = "artifacts/results"

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_pre_split_data()

    print("Building features...")
    X_train_vec, X_val_vec, X_test_vec = build_features(
        X_train, X_val, X_test
    )

    print("Training Logistic Regression...")
    model = train_lr(X_train_vec, y_train)

    # ---- Evaluation ----
    evaluate_model(model, X_train_vec, y_train, "Train")
    evaluate_model(model, X_val_vec, y_val, "Validation")
    metrics = evaluate_model(model, X_test_vec, y_test, "Test")

    # ---- Save Probabilities for Stacking ----
    print("\nSaving LR probabilities for stacking...")

    lr_train_probs = model.predict_proba(X_train_vec)
    lr_val_probs = model.predict_proba(X_val_vec)
    lr_test_probs = model.predict_proba(X_test_vec)

    # Handle OneVsRestClassifier output shape
    if isinstance(lr_train_probs, list):
        lr_train_probs = np.column_stack([p[:, 1] for p in lr_train_probs])

    if isinstance(lr_val_probs, list):
        lr_val_probs = np.column_stack([p[:, 1] for p in lr_val_probs])

    if isinstance(lr_test_probs, list):
        lr_test_probs = np.column_stack([p[:, 1] for p in lr_test_probs])

    np.save(f"{ARTIFACT_DIR}/lr_train_probs.npy", lr_train_probs)
    np.save(f"{ARTIFACT_DIR}/lr_val_probs.npy", lr_val_probs)
    np.save(f"{ARTIFACT_DIR}/lr_test_probs.npy", lr_test_probs)

    print("Saved stacking probabilities.")

    # ---- Save Metrics to JSON ----
    with open(f"{RESULTS_DIR}/lr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics to artifacts/results/lr_metrics.json")
