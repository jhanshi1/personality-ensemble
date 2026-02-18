import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = "artifacts/results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "model_comparison_bar.png")


def main():

    if not os.path.exists(RESULTS_DIR):
        raise FileNotFoundError("artifacts/results folder not found")

    models = []
    f1_scores = []

    for file in os.listdir(RESULTS_DIR):

        if file.endswith("_metrics.json"):

            path = os.path.join(RESULTS_DIR, file)

            with open(path, "r") as f:
                data = json.load(f)

            model_name = file.replace("_metrics.json", "")
            f1_macro = data.get("f1_macro", None)

            if f1_macro is not None:
                models.append(model_name)
                f1_scores.append(f1_macro)

    if not models:
        raise ValueError("No metrics JSON files found.")

    # Sort by F1 descending
    sorted_pairs = sorted(
        zip(models, f1_scores),
        key=lambda x: x[1],
        reverse=True
    )

    models_sorted, f1_sorted = zip(*sorted_pairs)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models_sorted, f1_sorted)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Macro F1 Score")
    plt.title("Model Comparison (Test Macro F1)")
    plt.ylim(min(f1_sorted) - 0.02, max(f1_sorted) + 0.02)

    for i, v in enumerate(f1_sorted):
        plt.text(i, v + 0.001, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.show()

    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
