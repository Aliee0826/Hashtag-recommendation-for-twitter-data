
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.config import CacheFile, TrainConfig
from trainer import TextDataset, Trainer, hashtag_weight
from bert import collate_fn_bert, TextBERT
from lstm import collate_batch_lstm, TextLSTM
from resnet import collate_batch_resnet, set_embedding, TextResNet


# training settings
train_config = TrainConfig()
EPOCH = train_config.num_epoch
MAX_LEN = train_config.max_len
BATCH = train_config.batch_size
THRE = train_config.threshold


# cache
cache = CacheFile()
LSTM_MODEL_PTH = cache.model_lstm
RESNET_MODEL_PTH = cache.model_resnet
BERT_MODEL_PTH = cache.model_bert
LSTM_VAL_TRAIN_PTH = cache.training_metrics_lstm
RESNET_VAL_TRAIN_PTH = cache.training_metrics_resnet
BERT_VAL_TRAIN_PTH = cache.training_metrics_bert
LSTM_PRED_PTH = cache.pred_lstm
RESNET_PRED_PTH = cache.pred_resnet
BERT_PRED_PTH = cache.pred_bert


# device
DEVICE = train_config.device
device = torch.device(DEVICE)


# dataset
text_dataset_train = TextDataset('train')
text_dataset_val = TextDataset('validation')
text_dataset_test = TextDataset('test')

# tag weight
tag_weight = hashtag_weight(50)


# ================================================= LSTM ================================================= #

# dataloader
lstm_dataloader_train = DataLoader(text_dataset_train, batch_size=BATCH, 
                            shuffle=True, collate_fn=collate_batch_lstm,
                            drop_last = True)

lstm_dataloader_val = DataLoader(text_dataset_val, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_batch_lstm,
                            drop_last = False)

lstm_dataloader_test = DataLoader(text_dataset_test, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_batch_lstm,
                            drop_last = False)

# model
def training_lstm():
    lstm = TextLSTM()
    lstm_30 = Trainer(
        lstm_dataloader_train, lstm_dataloader_val, lstm, EPOCH,  
        THRE, LSTM_MODEL_PTH, LSTM_VAL_TRAIN_PTH, device,
        tag_weight=tag_weight, bert=False, validation=False
    )




# ================================================= ResNet ================================================= #

# dataloader
resnet_dataloader_train = DataLoader(text_dataset_train, batch_size=BATCH, 
                            shuffle=True, collate_fn=collate_batch_resnet,
                            drop_last = True)

resnet_dataloader_val = DataLoader(text_dataset_val, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_batch_resnet,
                            drop_last = False)

resnet_dataloader_test = DataLoader(text_dataset_test, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_batch_resnet,
                            drop_last = False)

# model
def training_resnet():
    resnet = TextResNet()
    resnet = set_embedding(resnet)
    resnet_30 = Trainer(
        resnet_dataloader_train, resnet_dataloader_val, resnet, EPOCH,  
        THRE, RESNET_MODEL_PTH, RESNET_VAL_TRAIN_PTH, device,
        tag_weight=tag_weight, bert=False, validation=False
    )


# ================================================= BERT ================================================= #

# dataloader
bert_dataloader_train = DataLoader(text_dataset_train, batch_size=BATCH, 
                            shuffle=True, collate_fn=collate_fn_bert,
                            drop_last = True)

bert_dataloader_val = DataLoader(text_dataset_val, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_fn_bert,
                            drop_last = False)

bert_dataloader_test = DataLoader(text_dataset_test, batch_size=BATCH, 
                            shuffle=False, collate_fn=collate_fn_bert,
                            drop_last = False)

# model
def training_bert():
    bert = TextBERT()
    bert_30 = Trainer(
        bert_dataloader_train, bert_dataloader_val, bert, EPOCH,  
        THRE, BERT_MODEL_PTH, BERT_VAL_TRAIN_PTH, device,
        tag_weight=tag_weight, bert=True, validation=False
    )



if __name__ == "__main__":
    training_lstm()
    training_resnet()
    # training_bert()
