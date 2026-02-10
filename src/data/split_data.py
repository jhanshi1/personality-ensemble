import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import PROCESSED_DATA_DIR, RANDOM_SEED
CLEAN_DATA_PATH = PROCESSED_DATA_DIR + "clean.csv"
def split_dataset():
    df = pd.read_csv(CLEAN_DATA_PATH,encoding="latin")
    # First split: train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    # Second split: validation (10%) and test (20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=2/3,   # 20% of total
        random_state=RANDOM_SEED,
        shuffle=True
    )
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = split_dataset()
    train_df.to_csv(PROCESSED_DATA_DIR + "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR + "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR + "test.csv", index=False)
    print("Dataset split completed")
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
