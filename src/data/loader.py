import pandas as pd
import os
from src.utils.config import PROCESSED_DATA_DIR

LABEL_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

def load_pre_split_data():

    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Convert to list of strings explicitly
    X_train = train_df["status"].astype(str).tolist()
    y_train = train_df[LABEL_COLS].values

    X_val = val_df["status"].astype(str).tolist()
    y_val = val_df[LABEL_COLS].values

    X_test = test_df["status"].astype(str).tolist()
    y_test = test_df[LABEL_COLS].values

    return X_train, X_val, X_test, y_train, y_val, y_test
