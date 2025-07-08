
import os
import string
from collections import Counter
import re
import torch
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 去除所有HTML标签
def clean_html(text):
    return re.sub(r'<.*?>', '', text)


# 加载原始数据
def load_imdb_data(data_dir):
    data, labels = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"找不到路径，请确认aclImdb数据集是否正确")
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                with open(os.path.join(folder, fname), encoding='utf-8') as f:
                    text = clean_html(f.read().strip())
                    data.append(text)
                    labels.append(1.0 if label == 'pos' else 0.0)
    return data, labels

# 构建词汇表
def build_vocab(texts, max_vocab=20000):
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
        counter.update(tokens)
    most_common = counter.most_common(max_vocab - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: i + 2 for i, (word, _) in enumerate(most_common)})
    return vocab

# 编码与Dataset
def encode(text, vocab, max_len=500):
    tokens = word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))
    ids = [vocab.get(token, 1) for token in tokens[:max_len]]
    return torch.tensor(ids, dtype=torch.long)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=500):
        self.data = [(encode(text, vocab, max_len), torch.tensor(label, dtype=torch.float))
                     for text, label in zip(texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded.to(DEVICE), torch.stack(labels).to(DEVICE)


