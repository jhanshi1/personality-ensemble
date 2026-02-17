import torch
import torch.nn as nn
from transformers import BertModel


class CNNBERTClassifier(nn.Module):
    def __init__(self, dropout=0.3, num_filters=128, kernel_size=3):
        super(CNNBERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze BERT (important for CPU)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv = nn.Conv1d(
            in_channels=768,
            out_channels=num_filters,
            kernel_size=kernel_size
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # (batch, seq_len, 768)
        last_hidden_state = outputs.last_hidden_state

        # CNN expects (batch, channels, seq_len)
        x = last_hidden_state.permute(0, 2, 1)

        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits
