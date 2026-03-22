import pandas as pd
import re
import os
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_DIR
KEEP_COLUMNS = [
    "STATUS",
    "cEXT", "cNEU", "cAGR", "cCON", "cOPN",
    "#AUTHID"
]
def basic_text_clean(text, aggressive=True):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    if aggressive:
        text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataset():
    df = pd.read_csv(RAW_DATA_PATH,encoding="latin1")
    # Keep only required columns
    df = df[KEEP_COLUMNS]
    # Rename STATUS column
    df = df.rename(columns={"STATUS": "status"})
    # Convert y/n to 1/0
    label_cols = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
    for col in label_cols:
        df[col] = df[col].map({"y": 1, "n": 0})
    # Clean text
    df["status"] = df["status"].astype(str).apply(basic_text_clean)
    df = df[df["status"].str.len() > 0]
    df = df[df["status"].str.split().str.len() >= 3]
    return df

if __name__ == "__main__":
    clean_df = clean_dataset()
    print("Clean dataset created")
    print("Shape:", clean_df.shape)
    print(clean_df.head())
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = PROCESSED_DATA_DIR + "clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"Saved clean dataset to {output_path}")
