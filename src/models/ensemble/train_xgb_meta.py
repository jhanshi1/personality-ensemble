import numpy as np
import os
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from src.utils.evaluation import evaluate_predictions
from src.data.loader import load_pre_split_data

# -------- Directories -------- #

CLASSIC_DIR = "artifacts/classic"
DL_DIR = "artifacts/dl_data"
ENSEMBLE_DIR = "artifacts/ensemble"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)


# -------- File Loaders -------- #

def load_classic(name):
    path = os.path.join(CLASSIC_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def load_dl(name):
    path = os.path.join(DL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


# -------- Interaction Features -------- #

def add_interaction_features(X):
    """
    Add pairwise trait interactions from FT-BERT probabilities.
    Assumes last 5 columns belong to FT-BERT.
    """

    ftbert_probs = X[:, -5:]

    interactions = []

    for i in range(5):
        for j in range(i + 1, 5):
            interactions.append(
                (ftbert_probs[:, i] * ftbert_probs[:, j]).reshape(-1, 1)
            )

    interactions = np.hstack(interactions)

    return np.hstack([X, interactions])


# -------- Main -------- #

def main():

    print("Loading validation probabilities (for training meta-model)...")

    # Classic validation probs
    val_lr = load_classic("lr_val_probs.npy")
    val_svm = load_classic("svm_val_probs.npy")

    # Deep validation probs
    val_bigru = load_dl("bigru_val_probs.npy")
    val_bilstm = load_dl("bilstm_bert_val_probs.npy")
    val_cnn = load_dl("cnn_bert_val_probs.npy")
    val_ftbert = load_dl("ftbert_val_probs.npy")

    # Concatenate
    X_val = np.hstack([
        val_lr,
        val_svm,
        val_bigru,
        val_bilstm,
        val_cnn,
        val_ftbert
    ])

    X_val = add_interaction_features(X_val)

    print("Meta training feature shape:", X_val.shape)

    # Load labels
    _, _, _, _, y_val, y_test = load_pre_split_data()

    # -------- Train Meta-Model -------- #

    print("Training XGBoost meta-classifier...")

    xgb = OneVsRestClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    )

    xgb.fit(X_val, y_val)

    # -------- Load Test Meta Features -------- #

    print("\nLoading test probabilities (for evaluation)...")

    test_lr = load_classic("lr_test_probs.npy")
    test_svm = load_classic("svm_test_probs.npy")

    test_bigru = load_dl("bigru_test_probs.npy")
    test_bilstm = load_dl("bilstm_bert_test_probs.npy")
    test_cnn = load_dl("cnn_bert_test_probs.npy")
    test_ftbert = load_dl("ftbert_test_probs.npy")

    X_test = np.hstack([
        test_lr,
        test_svm,
        test_bigru,
        test_bilstm,
        test_cnn,
        test_ftbert
    ])

    X_test = add_interaction_features(X_test)

    print("Meta test feature shape:", X_test.shape)

    # -------- Evaluate on TEST -------- #

    test_probs = xgb.predict_proba(X_test)
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_test,
        test_preds,
        name="XGBoost Stacked TEST"
    )

    print(metrics)

    # -------- Save Final Predictions -------- #

    np.save(os.path.join(ENSEMBLE_DIR, "xgb_test_probs.npy"), test_probs)
    np.save(os.path.join(ENSEMBLE_DIR, "xgb_test_preds.npy"), test_preds)

    print("Saved final ensemble test predictions.")


if __name__ == "__main__":
    main()
