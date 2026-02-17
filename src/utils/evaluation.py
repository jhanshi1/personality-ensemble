from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    classification_report
)

def evaluate_model(model, X, y, name="Dataset", verbose=True):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1_macro = f1_score(y, preds, average="macro",zero_division=0)
    f1_micro = f1_score(y, preds, average="micro",zero_division=0)
    hl = hamming_loss(y, preds)
    report_dict = classification_report(
        y,
        preds,
        target_names=["EXT","NEU","AGR","CON","OPN"],
        output_dict=True,
        zero_division=0
    )
    if verbose:
        print(f"\n{name} Results:")
        print("Accuracy:", round(acc,4))
        print("F1 Macro:", round(f1_macro,4))
        print("F1 Micro:", round(f1_micro,4))
        print("Hamming Loss:", round(hl,4))
        print("\nPer-Trait Report:")
        print(classification_report(
            y,
            preds,
            target_names=["EXT","NEU","AGR","CON","OPN"],
            zero_division=0
        ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hl,
        "per_trait": report_dict
    }
def evaluate_predictions(y_true, preds, name="Dataset", verbose=True):
    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, preds, average="micro", zero_division=0)
    hl = hamming_loss(y_true, preds)

    report_dict = classification_report(
        y_true,
        preds,
        target_names=["EXT","NEU","AGR","CON","OPN"],
        output_dict=True,
        zero_division=0
    )

    if verbose:
        print(f"\n{name} Results:")
        print("Accuracy:", round(acc,4))
        print("F1 Macro:", round(f1_macro,4))
        print("F1 Micro:", round(f1_micro,4))
        print("Hamming Loss:", round(hl,4))
        print("\nPer-Trait Report:")
        print(classification_report(
            y_true,
            preds,
            target_names=["EXT","NEU","AGR","CON","OPN"],
            zero_division=0
        ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "hamming_loss": hl,
        "per_trait": report_dict
    }
