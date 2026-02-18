import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.models.dl.bigru import BiGRUClassifier


ARTIFACT_DIR = "artifacts/dl_data"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

device = torch.device("cpu")


def main():

    print("Loading processed DL data...")

    X_train = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/X_train.npy"),
        dtype=torch.long
    )
    X_val = torch.tensor(
        np.load(f"{ARTIFACT_DIR}/X_val.npy"),
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

    embedding_matrix = np.load(
        f"{ARTIFACT_DIR}/glove_embedding_matrix.npy"
    )

    print("Train shape:", X_train.shape)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
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


if __name__ == "__main__":
    main()
