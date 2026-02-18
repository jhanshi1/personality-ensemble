import numpy as np
import os
import json
from src.utils.evaluation import evaluate_predictions
from src.data.loader import load_pre_split_data

CLASSIC_DIR = "artifacts/classic"
DL_DIR = "artifacts/dl_data"
ENSEMBLE_DIR = "artifacts/ensemble"
RESULTS_DIR = "artifacts/results"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():

    print("Loading test probabilities...")

    # Load test probabilities
    lr = np.load(f"{CLASSIC_DIR}/lr_test_probs.npy")
    svm = np.load(f"{CLASSIC_DIR}/svm_test_probs.npy")

    bigru = np.load(f"{DL_DIR}/bigru_test_probs.npy")
    bilstm = np.load(f"{DL_DIR}/bilstm_bert_test_probs.npy")
    cnn = np.load(f"{DL_DIR}/cnn_bert_test_probs.npy")
    ftbert = np.load(f"{DL_DIR}/ftbert_test_probs.npy")

    # ---- Manually set weights based on validation macro F1 ----
    weights = {
        "lr": 0.5596,
        "svm": 0.5483,
        "bigru": 0.54,
        "bilstm": 0.57,
        "cnn": 0.565,
        "ftbert": 0.582
    }

    total = sum(weights.values())

    # Normalize weights
    for k in weights:
        weights[k] /= total

    print("Normalized Weights:", weights)

    # Weighted average
    final_probs = (
        weights["lr"] * lr +
        weights["svm"] * svm +
        weights["bigru"] * bigru +
        weights["bilstm"] * bilstm +
        weights["cnn"] * cnn +
        weights["ftbert"] * ftbert
    )

    final_preds = (final_probs >= 0.5).astype(int)

    # Load true labels
    _, _, _, _, _, y_test = load_pre_split_data()

    metrics = evaluate_predictions(
        y_test,
        final_preds,
        name="Weighted All-Models Ensemble"
    )

    print(metrics)

    np.save(f"{ENSEMBLE_DIR}/weighted_all_test_probs.npy", final_probs)
    np.save(f"{ENSEMBLE_DIR}/weighted_all_test_preds.npy", final_preds)

    with open(f"{RESULTS_DIR}/weighted_all_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved weighted all-model ensemble.")


if __name__ == "__main__":
    main()
