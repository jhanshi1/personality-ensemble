import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from src.data.loader import load_pre_split_data
from src.data.bert_preprocess import tokenize_bert
from src.utils.evaluation import evaluate_predictions

# -------------------- Config -------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
MAX_LEN = 64
NUM_LABELS = 5

ARTIFACT_DIR = "artifacts/dl_data"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# -------------------- Model -------------------- #

class FineTunedBERT(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, NUM_LABELS)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits


# -------------------- Training Script -------------------- #

def main():

    print("Loading split data...")
    train_texts, val_texts, test_texts, y_train, y_val, y_test = load_pre_split_data()

    print("Tokenizing with BERT...")
    train_ids, train_mask = tokenize_bert(train_texts, max_len=MAX_LEN)
    val_ids, val_mask = tokenize_bert(val_texts, max_len=MAX_LEN)
    test_ids, test_mask = tokenize_bert(test_texts, max_len=MAX_LEN)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(train_ids, train_mask, y_train)
    val_dataset = TensorDataset(val_ids, val_mask, y_val)
    test_dataset = TensorDataset(test_ids, test_mask)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = FineTunedBERT().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("\nStarting fine-tuning...")

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

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")

    print("\nTraining complete.")
        # -------------------- Train Probabilities -------------------- #

    print("\nGenerating train probabilities...")

    model.eval()
    train_probs = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            train_probs.append(probs.cpu())

    train_probs = torch.cat(train_probs, dim=0).numpy()

    np.save(f"{ARTIFACT_DIR}/ftbert_train_probs.npy", train_probs)

    print("Saved ftbert_train_probs.npy", train_probs.shape)

    # -------------------- Validation -------------------- #

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
        name="Fine-Tuned BERT Validation"
    )

    print(metrics)

    np.save(f"{ARTIFACT_DIR}/ftbert_val_probs.npy", val_probs)

    # -------------------- Test -------------------- #

    print("\nGenerating test probabilities...")

    test_probs = []

    with torch.no_grad():
        for input_ids, attention_mask in test_loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            test_probs.append(probs.cpu())

    test_probs = torch.cat(test_probs, dim=0).numpy()

    np.save(f"{ARTIFACT_DIR}/ftbert_test_probs.npy", test_probs)

    print("Saved FT-BERT val and test probabilities.")


if __name__ == "__main__":
    main()
