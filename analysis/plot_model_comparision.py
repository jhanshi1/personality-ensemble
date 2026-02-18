import json
import os
import matplotlib.pyplot as plt

RESULTS_PATH = "artifacts/results/metrics_summary.json"
OUTPUT_DIR = "artifacts/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            "metrics_summary.json not found. "
            "Create it inside artifacts/results/"
        )

    # Load results
    with open(RESULTS_PATH, "r") as f:
        metrics = json.load(f)

    models = list(metrics.keys())
    f1_scores = list(metrics.values())

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

    # Annotate bars
    for i, v in enumerate(f1_sorted):
        plt.text(i, v + 0.001, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_bar.png")
    plt.show()

    print("Saved plot to artifacts/results/model_comparison_bar.png")


if __name__ == "__main__":
    main()
