import pandas as pd

def label_stats(df, name):
    print(f"\n{name}")
    print(df[["cEXT","cNEU","cAGR","cCON","cOPN"]].mean())

train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
test  = pd.read_csv("data/processed/test.csv")

label_stats(train, "TRAIN")
label_stats(val, "VALIDATION")
label_stats(test, "TEST")
