import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
import joblib
from src.utils.evaluation import evaluate_model
from src.data.loader import load_pre_split_data
from src.features.pos_features import extract_pos_features
from src.features.emotion_features import (
    load_nrc_lexicon,
    extract_emotion_features
)


def build_features(X_train, X_val, X_test):

    # Word-level TF-IDF
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        max_features=2000,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=10
    )

    # Character-level TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        max_features=1000,
        ngram_range=(2, 3),
        sublinear_tf=True,
        min_df=10
    )

    # Fit only on training data
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

    # Combine all features
    X_train_vec = hstack([X_train_word, X_train_char, pos_train, emo_train])
    X_val_vec = hstack([X_val_word, X_val_char, pos_val, emo_val])
    X_test_vec = hstack([X_test_word, X_test_char, pos_test, emo_test])

    # Scale sparse features
    scaler = MaxAbsScaler()
    X_train_vec = scaler.fit_transform(X_train_vec)
    X_val_vec = scaler.transform(X_val_vec)
    X_test_vec = scaler.transform(X_test_vec)

    return (
    X_train_vec, X_val_vec, X_test_vec,
    word_vectorizer, char_vectorizer, scaler )


def train_svm(X_train, y_train):

    base_svm = LinearSVC(
        C=0.01,
        class_weight="balanced",
        max_iter=5000
    )

    calibrated_svm = CalibratedClassifierCV(
        base_svm,
        method="sigmoid",
        cv=3
    )

    model = OneVsRestClassifier(calibrated_svm)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":

    ARTIFACT_DIR = "artifacts/classic"
    RESULTS_DIR = "artifacts/results"

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading pre-split data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_pre_split_data()

    print("Building features...")
    X_train_vec, X_val_vec, X_test_vec,word_vectorizer,char_vectorizer,scaler = build_features(
        X_train, X_val, X_test
    )

    print("Training LinearSVC (Calibrated)...")
    model = train_svm(X_train_vec, y_train)
    print("\nSaving SVM model, vectorizers, and scaler...")

    joblib.dump(model, f"{ARTIFACT_DIR}/svm_model.pkl")
    joblib.dump(word_vectorizer, f"{ARTIFACT_DIR}/svm_word_vectorizer.pkl")
    joblib.dump(char_vectorizer, f"{ARTIFACT_DIR}/svm_char_vectorizer.pkl")
    joblib.dump(scaler, f"{ARTIFACT_DIR}/svm_scaler.pkl")

    print("Saved SVM model and preprocessing artifacts.")
    # ---- Evaluation ----
    evaluate_model(model, X_train_vec, y_train, "Train")
    evaluate_model(model, X_val_vec, y_val, "Validation")
    metrics = evaluate_model(model, X_test_vec, y_test, "Test")

    # ---- Save Probabilities for Stacking ----
    print("\nSaving SVM probabilities for stacking...")

    svm_train_probs = model.predict_proba(X_train_vec)
    svm_val_probs = model.predict_proba(X_val_vec)
    svm_test_probs = model.predict_proba(X_test_vec)

    # Handle OneVsRestClassifier output shape
    if isinstance(svm_train_probs, list):
        svm_train_probs = np.column_stack([p[:, 1] for p in svm_train_probs])

    if isinstance(svm_val_probs, list):
        svm_val_probs = np.column_stack([p[:, 1] for p in svm_val_probs])

    if isinstance(svm_test_probs, list):
        svm_test_probs = np.column_stack([p[:, 1] for p in svm_test_probs])

    np.save(f"{ARTIFACT_DIR}/svm_train_probs.npy", svm_train_probs)
    np.save(f"{ARTIFACT_DIR}/svm_val_probs.npy", svm_val_probs)
    np.save(f"{ARTIFACT_DIR}/svm_test_probs.npy", svm_test_probs)

    print("Saved stacking probabilities.")

    # ---- Save Metrics ----
    with open(f"{RESULTS_DIR}/svm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics to artifacts/results/svm_metrics.json")
