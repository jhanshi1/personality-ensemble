from transformers import BertTokenizer
import torch


def tokenize_bert(texts, max_len=50):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return encodings["input_ids"], encodings["attention_mask"]
