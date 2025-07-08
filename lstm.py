import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import string
import nltk
import numpy as np

from data import load_imdb_data, build_vocab, IMDBDataset, collate_fn
from lstm_model import LSTMModel, LSTMWithAttention, LSTMWithAttention2

DATA_DIR = "aclImdb_v1/aclImdb"
GLOVE_PATH = "./glove.42B.300d.txt"  # 请确保此文件下载并放置在此路径
MAX_LEN = 500
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 64
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_acc = 0

# # ---------- 加载原始IMDB数据 ----------
# def load_imdb_data(data_dir):
#     data, labels = [], []
#     for label in ['pos', 'neg']:
#         folder = os.path.join(data_dir, label)
#         if not os.path.exists(folder):
#             raise FileNotFoundError(f"找不到路径: {folder}，请确认aclImdb数据集是否正确解压至项目根目录")
#         for fname in os.listdir(folder):
#             if fname.endswith('.txt'):
#                 with open(os.path.join(folder, fname), encoding='utf-8') as f:
#                     text = f.read().strip()
#                     data.append(text)
#                     labels.append(1.0 if label == 'pos' else 0.0)
#     return data, labels

train_texts, train_labels = load_imdb_data(os.path.join(DATA_DIR, 'train'))
test_texts, test_labels = load_imdb_data(os.path.join(DATA_DIR, 'test'))

# # ---------- 构建词汇表 ----------
# def build_vocab(texts, max_vocab=20000):
#     counter = Counter()
#     for text in texts:
#         tokens = word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
#         counter.update(tokens)
#     most_common = counter.most_common(max_vocab - 2)
#     vocab = {"<PAD>": 0, "<UNK>": 1}
#     vocab.update({word: i + 2 for i, (word, _) in enumerate(most_common)})
#     return vocab

vocab = build_vocab(train_texts)


# ---------- 加载GloVe词向量 ----------
def load_glove_embeddings(glove_path, vocab, embedding_dim):
    embeddings = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim)).astype(np.float32)
    embeddings[0] = np.zeros(embedding_dim)
    found = 0
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[idx] = vec
                found += 1
    print(f"Loaded GloVe vectors for {found}/{len(vocab)} words.")
    return torch.tensor(embeddings)

# # ---------- 编码与Dataset ----------
# def encode(text, vocab, max_len=500):
#     tokens = word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
#     ids = [vocab.get(token, 1) for token in tokens[:max_len]]
#     return torch.tensor(ids, dtype=torch.long)
#
# class IMDBDataset(Dataset):
#     def __init__(self, texts, labels, vocab, max_len=500):
#         self.data = [(encode(text, vocab, max_len), torch.tensor(label, dtype=torch.float))
#                      for text, label in zip(texts, labels)]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
# def collate_fn(batch):
#     texts, labels = zip(*batch)
#     padded = pad_sequence(texts, batch_first=True, padding_value=0)
#     return padded.to(DEVICE), torch.stack(labels).to(DEVICE)

train_dataset = IMDBDataset(train_texts, train_labels, vocab)
test_dataset = IMDBDataset(test_texts, test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# # ---------- 模型定义 ----------
# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         emb = self.embedding(x)
#         _, (hidden, _) = self.lstm(emb)
#         out = self.fc(hidden[-1])
#         out = self.dropout(out)
#         return self.sigmoid(out)


model = LSTMWithAttention2(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
embedding_matrix = load_glove_embeddings(GLOVE_PATH, vocab, EMBEDDING_DIM)
model.embedding.weight.data.copy_(embedding_matrix)
model.embedding.weight.requires_grad = False
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()

#训练与评估
train_losses, train_accuracies, test_accuracies = [], [], []

def evaluate(model, loader, verbose=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = (model(x).squeeze() > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    if verbose:
        print(f"Test Accuracy: {acc:.4f}")
    return acc

def train(model, loader):
    global best_acc
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((preds > 0.5).float() == y).sum().item()
            total += y.size(0)
        avg_loss = total_loss / len(loader)
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, verbose=False)
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
        if train_acc > best_acc:
            best_acc = train_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'imdb_best.pth')
            print("Saved best model.")

#执行训练
train(model, train_loader)
evaluate(model, test_loader)

#可视化结果
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Acc', marker='o')
plt.plot(epochs, test_accuracies, label='Test Acc', marker='x')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


