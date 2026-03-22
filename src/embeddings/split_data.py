import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import PROCESSED_DATA_DIR, RANDOM_SEED

CLEAN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "clean.csv")

LABEL_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]


def split_dataset():
    df = pd.read_csv(CLEAN_DATA_PATH, encoding="latin1")

    original_shape = df.shape

    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Second split: 10% val, 20% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print("Original shape:", original_shape)
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:\n", train_df[LABEL_COLS].mean())
    print("\nValidation label distribution:\n", val_df[LABEL_COLS].mean())
    print("\nTest label distribution:\n", test_df[LABEL_COLS].mean())

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = split_dataset()

    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)

    print("\nDataset split completed successfully.")
