import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.dl.cnn_bert import CNNBERTClassifier
from src.data.loader import load_pre_split_data
from src.data.bert_preprocess import tokenize_bert
from src.utils.evaluation import evaluate_predictions

device = torch.device("cpu")

BATCH_SIZE = 16   # smaller for BERT
EPOCHS = 3        # start small
LR = 2e-4


def main():

    print("Loading split data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_pre_split_data()

    print("Tokenizing with BERT...")
    train_ids, train_mask = tokenize_bert(train_texts)
    val_ids, val_mask = tokenize_bert(val_texts)
    test_ids, test_mask = tokenize_bert(test_texts)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(train_ids, train_mask, y_train)
    val_dataset = TensorDataset(val_ids, val_mask, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = CNNBERTClassifier()
    model.to(device)

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

    print("\nEvaluating validation performance...")

    model.eval()
    val_probs = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            val_probs.append(probs.cpu())

    val_probs = torch.cat(val_probs, dim=0).numpy()
    val_preds = (val_probs >= 0.5).astype(int)

    metrics = evaluate_predictions(
        y_val.numpy(),
        val_preds,
        name="CNN+BERT Validation"
    )

    print(metrics)


if __name__ == "__main__":
    main()
