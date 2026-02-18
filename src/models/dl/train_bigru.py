import numpy as np
import torch
import os
import json
from torch.utils.data import TensorDataset, DataLoader
from src.models.dl.bigru import BiGRUClassifier
from src.utils.evaluation import evaluate_predictions

ARTIFACT_DIR = "artifacts/dl_data"
RESULTS_DIR = "artifacts/results"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

device = torch.device("cpu")


def main():

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading processed DL data...")

    X_train = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/X_train.npy"),
        dtype=torch.long
    )
    X_val = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/X_val.npy"),
        dtype=torch.long
    )
    X_test = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/X_test.npy"),
        dtype=torch.long
    )

    y_train = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/y_train.npy"),
        dtype=torch.float32
    )
    y_val = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/y_val.npy"),
        dtype=torch.float32
    )
    y_test = np.load(f"{ARTIFACT_DIR}/y_test.npy")

    embedding_matrix = np.load(
        f"{ARTIFACT_DIR}/glove_embedding_matrix.npy"
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        TensorDataset(X_test, torch.zeros(len(X_test), 5)),
        batch_size=BATCH_SIZE
    )

    model = BiGRUClassifier(embedding_matrix)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    print("\nTraining complete.")

    model.eval()

    # ---- TRAIN PROBS ----
    train_probs = []
    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(model(batch_X))
            train_probs.append(probs.cpu())

    train_probs = torch.cat(train_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/bigru_train_probs.npy", train_probs)

    # ---- VAL PROBS ----
    val_probs = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(model(batch_X))
            val_probs.append(probs.cpu())

    val_probs = torch.cat(val_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/bigru_val_probs.npy", val_probs)

    # ---- TEST PROBS ----
    test_probs = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            probs = torch.sigmoid(model(batch_X))
            test_probs.append(probs.cpu())

    test_probs = torch.cat(test_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/bigru_test_probs.npy", test_probs)

    print("Saved stacking probabilities.")

    # ---- TEST EVALUATION ----
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_test,
        test_preds,
        name="BiGRU Test"
    )

    print(metrics)

    # ---- SAVE METRICS ----
    with open(f"{RESULTS_DIR}/bigru_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics to artifacts/results/bigru_metrics.json")


if __name__ == "__main__":
    main()
