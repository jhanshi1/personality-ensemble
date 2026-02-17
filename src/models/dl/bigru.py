import torch
import torch.nn as nn


class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, dropout=0.3):
        super(BiGRUClassifier, self).__init__()

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = True # freeze GloVe

        self.bigru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 5)  # 5 personality traits

    def forward(self, x):
        x = self.embedding(x)

        _, hidden = self.bigru(x)

        # hidden shape: (2, batch, hidden_dim)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        x = self.dropout(hidden)
        logits = self.fc(x)

        return logits
