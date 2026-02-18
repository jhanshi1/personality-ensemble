import torch
import torch.nn as nn
from transformers import BertModel


class BiLSTMBERTClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3):
        super(BiLSTMBERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze all BERT layers first
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze last 2 encoder layers
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)

        _, (hidden, _) = self.lstm(last_hidden_state)

        # hidden shape: (2, batch, hidden_dim)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.dropout(hidden)
        logits = self.fc(x)

        return logits
