from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from src.utils.evaluation import evaluate_model
from src.data.loader import load_pre_split_data
from scipy.sparse import csr_matrix
from src.features.pos_features import extract_pos_features
from src.features.emotion_features import (
    load_nrc_lexicon,
    extract_emotion_features
)
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

def build_features(X_train, X_val, X_test):

    # Word-level ngrams
    word_vectorizer = CountVectorizer(
        max_features=3000,
        ngram_range=(1,2),
        min_df=3
    )

    # Character-level ngrams
    char_vectorizer = CountVectorizer(
        analyzer="char",
        max_features=2000,
        ngram_range=(3,5),
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

    # NRC features
    nrc = load_nrc_lexicon("data/external/NRC-Emotion-Lexicon.txt")

    emo_train = csr_matrix(extract_emotion_features(X_train, nrc))
    emo_val = csr_matrix(extract_emotion_features(X_val, nrc))
    emo_test = csr_matrix(extract_emotion_features(X_test, nrc))

    # Combine everything
    X_train_vec = hstack([
        X_train_word,
        X_train_char,
        pos_train,
        emo_train
    ])

    X_val_vec = hstack([
        X_val_word,
        X_val_char,
        pos_val,
        emo_val
    ])

    X_test_vec = hstack([
        X_test_word,
        X_test_char,
        pos_test,
        emo_test
    ])

    return X_train_vec, X_val_vec, X_test_vec




def train_lr(X_train, y_train):

    model = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",          # sparse-friendly
            max_iter=3000,
            C=1.0,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":

    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_pre_split_data()

    print("Building TF features...")
    X_train_vec, X_val_vec, X_test_vec = build_features(
        X_train, X_val, X_test
    )

    print("Training Logistic Regression...")
    model = train_lr(X_train_vec, y_train)

    evaluate_model(model, X_train_vec, y_train, "Train")
    evaluate_model(model, X_val_vec, y_val, "Validation")
    evaluate_model(model, X_test_vec, y_test, "Test")
