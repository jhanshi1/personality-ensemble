import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from src.models.dl.cnn_bert import CNNBERTClassifier
from src.data.loader import load_pre_split_data
from src.data.bert_preprocess import tokenize_bert
from src.utils.evaluation import evaluate_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
MAX_LEN = 64

ARTIFACT_DIR = "artifacts/dl_data"
RESULTS_DIR = "artifacts/results"


def main():

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading split data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_pre_split_data()

    print("Tokenizing with BERT...")
    train_ids, train_mask = tokenize_bert(train_texts, max_len=MAX_LEN)
    val_ids, val_mask = tokenize_bert(val_texts, max_len=MAX_LEN)
    test_ids, test_mask = tokenize_bert(test_texts, max_len=MAX_LEN)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(train_ids, train_mask, y_train_tensor)
    val_dataset = TensorDataset(val_ids, val_mask, y_val_tensor)
    test_dataset = TensorDataset(test_ids, test_mask)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = CNNBERTClassifier().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for input_ids, attention_mask, labels in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
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
        for input_ids, attention_mask, _ in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            probs = torch.sigmoid(model(input_ids, attention_mask))
            train_probs.append(probs.cpu())

    train_probs = torch.cat(train_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/cnn_bert_train_probs.npy", train_probs)

    # ---- VAL PROBS ----
    val_probs = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            probs = torch.sigmoid(model(input_ids, attention_mask))
            val_probs.append(probs.cpu())

    val_probs = torch.cat(val_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/cnn_bert_val_probs.npy", val_probs)

    # ---- TEST PROBS ----
    test_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            probs = torch.sigmoid(model(input_ids, attention_mask))
            test_probs.append(probs.cpu())

    test_probs = torch.cat(test_probs, dim=0).numpy()
    np.save(f"{ARTIFACT_DIR}/cnn_bert_test_probs.npy", test_probs)

    print("Saved stacking probabilities.")

    # ---- TEST EVALUATION ----
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_test,
        test_preds,
        name="CNN+BERT Test"
    )

    print(metrics)

    # ---- SAVE METRICS ----
    with open(f"{RESULTS_DIR}/cnn_bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics to artifacts/results/cnn_bert_metrics.json")


if __name__ == "__main__":
    main()
