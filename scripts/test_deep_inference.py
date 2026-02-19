import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer

# ---------------- Project Path Fix ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ---------------- Model Imports ---------------- #

from src.models.dl.bigru import BiGRUClassifier
from src.models.dl.bilstm_bert import BiLSTMBERTClassifier
from src.models.dl.cnn_bert import CNNBERTClassifier
from src.models.dl.ft_bert import FineTunedBERT

from src.data.dl_preprocess import encode_texts, pad_sequences

# ---------------- Config ---------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN_RNN = 50
MAX_LEN_BERT = 64

DEEP_DIR = os.path.join(BASE_DIR, "artifacts", "deep")
DL_DATA_DIR = os.path.join(BASE_DIR, "artifacts", "dl_data")

TRAITS = ["OPN", "CON", "EXT", "AGR", "NEU"]

# ---------------- Load Shared Artifacts ---------------- #

word2idx = np.load(
    os.path.join(DL_DATA_DIR, "word2idx.npy"),
    allow_pickle=True
).item()

embedding_matrix = np.load(
    os.path.join(DL_DATA_DIR, "glove_embedding_matrix.npy")
)

# ---------------- Load Models ---------------- #

# 1️⃣ BiGRU (GloVe-based)
bigru_ckpt = torch.load(
    os.path.join(DEEP_DIR, "bigru_model.pt"),
    map_location=device
)

bigru_model = BiGRUClassifier(embedding_matrix)
bigru_model.load_state_dict(bigru_ckpt["model_state_dict"])
bigru_model.to(device)
bigru_model.eval()

# 2️⃣ BiLSTM-BERT
bilstm_ckpt = torch.load(
    os.path.join(DEEP_DIR, "bilstm_model.pt"),
    map_location=device
)

bilstm_model = BiLSTMBERTClassifier()
bilstm_model.load_state_dict(bilstm_ckpt["model_state_dict"])
bilstm_model.to(device)
bilstm_model.eval()

# 3️⃣ CNN-BERT
cnn_ckpt = torch.load(
    os.path.join(DEEP_DIR, "cnn_model.pt"),
    map_location=device
)

cnn_model = CNNBERTClassifier()
cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
cnn_model.to(device)
cnn_model.eval()

# 4️⃣ Fine-Tuned BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

ftbert_ckpt = torch.load(
    os.path.join(DEEP_DIR, "ft_bert_model.pt"),
    map_location=device
)

ftbert_model = FineTunedBERT()
ftbert_model.load_state_dict(ftbert_ckpt["model_state_dict"])
ftbert_model.to(device)
ftbert_model.eval()

# ---------------- Prediction Functions ---------------- #

def predict_glove(model, text):
    """
    For GloVe-based models (BiGRU)
    """
    encoded = encode_texts([text], word2idx)
    padded = pad_sequences(encoded, MAX_LEN_RNN)
    inputs = torch.tensor(padded, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs[0]


def predict_bert(model, text):
    """
    For BERT-based models (BiLSTM-BERT, CNN-BERT, FT-BERT)
    """
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN_BERT,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs[0]


def pretty_print(model_name, probs):
    print(f"\n{model_name} Predictions:")
    for trait, prob in zip(TRAITS, probs):
        print(f"{trait}: {prob:.4f}")


# ---------------- Test ---------------- #

if __name__ == "__main__":

    text = "I love exploring new ideas and meeting creative people."

    pretty_print("BiGRU (GloVe)", predict_glove(bigru_model, text))
    pretty_print("BiLSTM-BERT", predict_bert(bilstm_model, text))
    pretty_print("CNN-BERT", predict_bert(cnn_model, text))
    pretty_print("FT-BERT", predict_bert(ftbert_model, text))
