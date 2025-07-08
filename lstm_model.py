
from torch import nn

# 模型定义 v1.0
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        out = self.fc(hidden[-1])
        out = self.dropout(out)
        return self.sigmoid(out)


import torch
# 模型定义 v2.0
class LSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)  # Attention层
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(emb)  # [batch, seq_len, hidden_dim]
        # Attention权重
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        # 加权求和
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        out = self.fc(context)
        out = self.dropout(out)
        return self.sigmoid(out)

# 模型定义 v3.0
class LSTMWithAttention2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, bidirectional=True, fc_hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attn = nn.Linear(lstm_output_dim, 1)  # Attention层
        self.fc1 = nn.Linear(lstm_output_dim, fc_hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(emb)  # [batch, seq_len, lstm_output_dim]
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # [batch, seq_len]
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, lstm_output_dim]
        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)