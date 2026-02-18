import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.features.pos_features import extract_pos_features
from src.features.emotion_features import load_nrc_lexicon, extract_emotion_features


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIC_DIR = os.path.join(BASE_DIR, "artifacts", "classic")

# Load models
lr_model = joblib.load(os.path.join(CLASSIC_DIR, "lr_model.pkl"))
svm_model = joblib.load(os.path.join(CLASSIC_DIR, "svm_model.pkl"))

# Load vectorizers
lr_word_vec = joblib.load(os.path.join(CLASSIC_DIR, "word_vectorizer.pkl"))
lr_char_vec = joblib.load(os.path.join(CLASSIC_DIR, "char_vectorizer.pkl"))

svm_word_vec = joblib.load(os.path.join(CLASSIC_DIR, "svm_word_vectorizer.pkl"))
svm_char_vec = joblib.load(os.path.join(CLASSIC_DIR, "svm_char_vectorizer.pkl"))
svm_scaler = joblib.load(os.path.join(CLASSIC_DIR, "svm_scaler.pkl"))

# Load NRC lexicon
nrc = load_nrc_lexicon("data/external/NRC-Emotion-Lexicon.txt")


def build_lr_features(text):
    word = lr_word_vec.transform([text])
    char = lr_char_vec.transform([text])
    pos = csr_matrix(extract_pos_features([text]))
    emo = csr_matrix(extract_emotion_features([text], nrc))
    return hstack([word, char, pos, emo])


def build_svm_features(text):
    word = svm_word_vec.transform([text])
    char = svm_char_vec.transform([text])
    pos = csr_matrix(extract_pos_features([text]))
    emo = csr_matrix(extract_emotion_features([text], nrc))
    features = hstack([word, char, pos, emo])
    return svm_scaler.transform(features)


def predict(text):

    lr_features = build_lr_features(text)
    lr_probs = lr_model.predict_proba(lr_features)

    svm_features = build_svm_features(text)
    svm_probs = svm_model.predict_proba(svm_features)

    if isinstance(lr_probs, list):
        lr_probs = np.column_stack([p[:, 1] for p in lr_probs])

    if isinstance(svm_probs, list):
        svm_probs = np.column_stack([p[:, 1] for p in svm_probs])

    final_probs = (lr_probs + svm_probs) / 2

    return final_probs[0]


if __name__ == "__main__":
    text = "I love learning new ideas and exploring psychology and AI."
    probs = predict(text)

    print("\nPredicted Probabilities:")
    print("OPN:", probs[0])
    print("CON:", probs[1])
    print("EXT:", probs[2])
    print("AGR:", probs[3])
    print("NEU:", probs[4])
