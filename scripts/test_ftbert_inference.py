import os
import torch
import numpy as np
from transformers import BertTokenizer
from src.models.dl.ft_bert import FineTunedBERT  # we will fix this import

# -------------------- Config -------------------- #

device = torch.device("cpu")
MAX_LEN = 64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "deep", "ft_bert_model.pt")

# -------------------- Load Tokenizer -------------------- #

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -------------------- Load Model -------------------- #

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = FineTunedBERT()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


def predict(text):

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs[0]


if __name__ == "__main__":

    text = "I love meeting new people and exploring creative ideas."

    probs = predict(text)

    print("\nFT-BERT Prediction:")
    print("OPN:", probs[0])
    print("CON:", probs[1])
    print("EXT:", probs[2])
    print("AGR:", probs[3])
    print("NEU:", probs[4])
