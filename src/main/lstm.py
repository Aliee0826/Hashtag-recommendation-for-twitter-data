"""
This file contains the model structure of LSTM
"""


import torch
from torch import nn
from torch.autograd import Variable
from config.config import TrainConfig, LSTMConfig
from trainer import TextDataset

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# model config
lstm_config = LSTMConfig()
HIDDEN_SIZE = lstm_config.hidden_size

# training config
train_config = TrainConfig()
EPOCH = train_config.num_epoch
MAX_LEN = train_config.max_len
BATCH = train_config.batch_size

# device
DEVICE = train_config.device
device = torch.device(DEVICE)


# dataset
text_dataset_train = TextDataset('train')

# tokenizer
def build_text_pipeline():
    tokenizer = get_tokenizer(None) # if none, simply use split()
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(str(text))

    vocab = build_vocab_from_iterator(yield_tokens(text_dataset_train), specials=["<unk>"], min_freq=2)
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(str(x)))
    return text_pipeline, vocab

text_pipeline, vocab = build_text_pipeline()

def collate_batch_lstm(batch):
    
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        token = text_pipeline(_text)
        X = ([0]* (MAX_LEN-len(token)))+token if len(token)<MAX_LEN else token[:MAX_LEN]
        text_list.append(X)
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device)


# model building
class TextLSTM(nn.Module):

    def __init__(self, vocab_size=len(vocab), embed_dim=MAX_LEN, num_class=50, hidden_dim=HIDDEN_SIZE):
        super(TextLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.constant_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, num_class, dtype=torch.float)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = torch.cat((
        #     self.embedding(x), self.constant_embedding(x)), dim=2)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(1, self.embed_dim, self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(1, self.embed_dim, self.hidden_dim).to(device))
        torch.nn.init.xavier_normal_(h_0)
        torch.nn.init.xavier_normal_(c_0)
      
        lstm_out, (ht, ct) = self.lstm(x, (h_0, c_0))
        # ht = ht.view(-1, self.hidden_dim)
        x = self.relu(lstm_out[-1])
        output = self.fc(x)
        return self.sigmoid(output)

