import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    classification_report
)


def evaluate_model(model, X, y, name="Dataset", verbose=True, thresholds=None):

    probs = model.predict_proba(X)

    if isinstance(probs, list):
        probs = np.column_stack([p[:, 1] for p in probs])

    if thresholds is None:
        thresholds = np.array([0.5] * probs.shape[1])

    preds = (probs >= thresholds).astype(int)

    acc = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average="macro", zero_division=0)
    f1_micro = f1_score(y, preds, average="micro", zero_division=0)
    hl = hamming_loss(y, preds)

    report_dict = classification_report(
        y,
        preds,
        target_names=["EXT", "NEU", "AGR", "CON", "OPN"],
        output_dict=True,
        zero_division=0
    )

    if verbose:
        print(f"\n{name} Results:")
        print("Accuracy:", round(acc, 4))
        print("F1 Macro:", round(f1_macro, 4))
        print("F1 Micro:", round(f1_micro, 4))
        print("Hamming Loss:", round(hl, 4))
        print("\nPer-Trait Report:")
        print(classification_report(
            y,
            preds,
            target_names=["EXT", "NEU", "AGR", "CON", "OPN"],
            zero_division=0
        ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hl,
        "per_trait": report_dict
    }


def find_best_thresholds(y_true, probs):

    n_labels = y_true.shape[1]
    thresholds = []

    for i in range(n_labels):

        best_thresh = 0.5
        best_f1 = 0

        # Restrict search range
        for t in np.arange(0.3, 0.7, 0.02):

            preds = (probs[:, i] >= t).astype(int)

            positive_ratio = preds.mean()

            # Prevent trivial all-0 or all-1 solutions
            if 0.05 < positive_ratio < 0.95:

                f1 = f1_score(y_true[:, i], preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t

        thresholds.append(best_thresh)

    return np.array(thresholds)


def evaluate_predictions(y_true, preds, name="Dataset", verbose=True):

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, preds, average="micro", zero_division=0)
    hl = hamming_loss(y_true, preds)

    report_dict = classification_report(
        y_true,
        preds,
        target_names=["EXT", "NEU", "AGR", "CON", "OPN"],
        output_dict=True,
        zero_division=0
    )

    if verbose:
        print(f"\n{name} Results:")
        print("Accuracy:", round(acc, 4))
        print("F1 Macro:", round(f1_macro, 4))
        print("F1 Micro:", round(f1_micro, 4))
        print("Hamming Loss:", round(hl, 4))
        print("\nPer-Trait Report:")
        print(classification_report(
            y_true,
            preds,
            target_names=["EXT", "NEU", "AGR", "CON", "OPN"],
            zero_division=0
        ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hl,
        "per_trait": report_dict
    }
