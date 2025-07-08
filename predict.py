import torch
from lstm_model import LSTMWithAttention2
from data import build_vocab, encode, clean_html
import numpy as np

# 参数（需与训练时一致）
EMBEDDING_DIM = 300
HIDDEN_DIM = 64
NUM_LAYERS = 2
BIDIRECTIONAL = True
FC_HIDDEN_DIM = 32
MAX_LEN = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表
import pickle
# 假设你有保存vocab.pkl，如果没有可用build_vocab重新生成
# with open('vocab.pkl', 'rb') as f:
#     vocab = pickle.load(f)
# 或直接用训练时的build_vocab
from data import build_vocab
import os

DATA_DIR = "aclImdb_v1/aclImdb"
train_texts, _ = [], []
if os.path.exists(os.path.join(DATA_DIR, 'train/pos')):
    from data import load_imdb_data
    train_texts, _ = load_imdb_data(os.path.join(DATA_DIR, 'train'))
vocab = build_vocab(train_texts)

# 加载模型
model = LSTMWithAttention2(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, BIDIRECTIONAL, FC_HIDDEN_DIM)
model.load_state_dict(torch.load('imdb_best.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict(text):
    text = clean_html(text)
    ids = encode(text, vocab, MAX_LEN).unsqueeze(0).to(DEVICE)  # [1, seq_len]
    with torch.no_grad():
        prob = model(ids).item()
    label = "正面" if prob > 0.5 else "负面"
    if prob < 0.5:
        prob = 1-prob
    return label, prob

if __name__ == "__main__":
    while True:
        text = input("请输入一句英文影评（输入exit退出）：\n")
        if text.strip().lower() == "exit":
            break
        label, prob = predict(text)
        print(f"情感类型: {label}，置信度: {prob:.4f}")