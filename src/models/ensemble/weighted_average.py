import numpy as np
import os
from src.utils.evaluation import evaluate_predictions
from src.data.loader import load_pre_split_data

CLASSIC_DIR = "artifacts/classic"
DL_DIR = "artifacts/dl_data"
ENSEMBLE_DIR = "artifacts/ensemble"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)


def main():

    # Load test probabilities
    ftbert = np.load(f"{DL_DIR}/ftbert_test_probs.npy")
    lr = np.load(f"{CLASSIC_DIR}/lr_test_probs.npy")

    # Weighted average
    final_probs = 0.7 * ftbert + 0.3 * lr

    final_preds = (final_probs >= 0.5).astype(int)

    # Load true labels
    _, _, _, _, _, y_test = load_pre_split_data()

    metrics = evaluate_predictions(
        y_test,
        final_preds,
        name="Weighted Ensemble (FTBERT + LR)"
    )

    print(metrics)

    np.save(f"{ENSEMBLE_DIR}/weighted_test_probs.npy", final_probs)
    np.save(f"{ENSEMBLE_DIR}/weighted_test_preds.npy", final_preds)

    print("Saved weighted ensemble predictions.")


if __name__ == "__main__":
    main()
