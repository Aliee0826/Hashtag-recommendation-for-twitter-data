"""
This file contains the model structure of ResNet
"""


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from config.config import TrainConfig, RESNETConfig
from trainer import TextDataset
from d2l import torch as d2l


from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# model config
resnet_config = RESNETConfig()
NUM_RESIDUAL = [int(x) for x in resnet_config.num_residuals().split(',')]
NUM_CHANNEL = [int(x) for x in resnet_config.num_channels().split(',')]
EMBED_SIZE = resnet_config.embed_size()

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
tokenizer = get_tokenizer(None) # if none, simply use split()

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(str(text))

def build_text_pipeline():
    vocab1 = build_vocab_from_iterator(yield_tokens(text_dataset_train), min_freq=2, specials=["<unk>"])
    vocab1.set_default_index(vocab1["<unk>"])
    text_pipeline = lambda x: vocab1(tokenizer(str(x)))
    return text_pipeline, vocab1

text_pipeline, vocab1 = build_text_pipeline()

def collate_batch_resnet(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        token = text_pipeline(_text)
        X = token+([0]* (MAX_LEN-len(token))) if len(token)<MAX_LEN else token[:MAX_LEN]
        text_list.append(X)
    label_list = torch.tensor(label_list, dtype=torch.float)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    return text_list.to(device), label_list.to(device)

# Xavier weight initialization
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

def set_embedding(model):
    # glove embedding preparation
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[text_pipeline(vocab1)]
    # set two kinds of embedding
    model.embedding.weight.data.copy_(embeds)
    model.constant_embedding.weight.data.copy_(embeds)
    model.constant_embedding.weight.requires_grad = False
    model.apply(init_weights)
    return model


# model building
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv = True, strides = 2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class TextResNet(nn.Module):
    def __init__(self, vocab_size=len(vocab1), embed_size=EMBED_SIZE, 
                 num_channels=NUM_CHANNEL, num_residuals=NUM_RESIDUAL, **kwargs):
        super(TextResNet, self).__init__(**kwargs)

        '''1. Embedding layer'''
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # this constant embedding doesn't need training
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)

        '''2. First layer without residual'''
        self.conv_ini = nn.Conv1d(embed_size * 2, num_channels[0], kernel_size = 7, stride = 2, padding = 3)

        '''3. Four blocks of layers with residual'''
        self.resnet_block2 = nn.Sequential(*resnet_block(num_channels[0], num_channels[1], num_residuals[0], first_block=True))
        self.resnet_block3 = nn.Sequential(*resnet_block(num_channels[1], num_channels[2], num_residuals[1]))
        self.resnet_block4 = nn.Sequential(*resnet_block(num_channels[2], num_channels[3], num_residuals[2]))
        self.resnet_block5 = nn.Sequential(*resnet_block(num_channels[3], num_channels[4], num_residuals[3]))

        '''4. Final fully-connected layer'''
        # add linear layer convert to 50d (predicting class size)
        self.decoder = nn.Linear(num_channels[-1], 50, dtype=torch.float)

        '''Other functional layers'''
        # pool layer for first layer without residual
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # pool layer for residual blocks
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        # batch norm layer
        self.batchnorm = nn.BatchNorm1d(64)
        # activate function
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten() 
        self.sigmoid = nn.Sigmoid()
        # dropout function to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        '''layer aggragation'''
        self.resnet = nn.Sequential(
            self.conv_ini, self.batchnorm, self.relu, self.pool1, \
            self.resnet_block2, self.resnet_block3, self.resnet_block4, self.resnet_block5, self.pool2
            )

    def forward(self, inputs):
        # congragate two embedding methods
        # embedding shape: (batch size, number of words in a sentence, word embedding size)
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # reshape the torch(consisted of two embeddings) to accord with the input shape of conv1d
        embeddings = embeddings.permute(0, 2, 1)

        encoding = self.relu(self.flatten(self.resnet(embeddings)))
        outputs = self.decoder(self.dropout(encoding))

        return self.sigmoid(outputs)

if __name__ == "__main__":
    a = NUM_RESIDUAL
    print(a)
    print(NUM_CHANNEL)
    print(EMBED_SIZE)
    print(MAX_LEN)