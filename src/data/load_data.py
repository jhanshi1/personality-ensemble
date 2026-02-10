import pandas as pd
from src.utils.config import RAW_DATA_PATH
def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH,encoding="latin1")
    return df
if __name__ == "__main__":
    df=load_raw_data()
    print("Raw dataset loaded")
    print("Shape:", df.shape)
    print("Columns:")
    for col in df.columns:
        print(col)