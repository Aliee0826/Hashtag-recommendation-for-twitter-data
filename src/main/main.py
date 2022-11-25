"""
This file combines the whole process of training and evaluation
"""

import numpy as np
import pandas as pd
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from config.config import CacheFile, TrainConfig, EvalConfig
from trainer import TextDataset, Trainer, hashtag_weight
from bert import collate_fn_bert, TextBERT
from lstm import collate_batch_lstm, TextLSTM
from resnet import collate_batch_resnet, set_embedding, TextResNet
from evaluation import GroundTruth, Predict, CalculateMetrics, StatisticalEval


# training settings
train_config = TrainConfig()
EPOCH = train_config.num_epoch
MAX_LEN = train_config.max_len
BATCH = train_config.batch_size
THRE = train_config.threshold

# evaluation
eval = EvalConfig()
NUM_SAMPLE = eval.n_sample()
TOP_N = [int(x) for x in eval.top_n().split(',')]
EPOCH_LS = [int(x) for x in eval.epoch_list().split(',')]

# device
DEVICE = train_config.device
device = torch.device(DEVICE)

# cache
cache = CacheFile()
# model
LSTM_MODEL_PTH = cache.model_lstm
RESNET_MODEL_PTH = cache.model_resnet
BERT_MODEL_PTH = cache.model_bert
# complex training validation
LSTM_VAL_TRAIN_PTH = cache.training_metrics_lstm
RESNET_VAL_TRAIN_PTH = cache.training_metrics_resnet
BERT_VAL_TRAIN_PTH = cache.training_metrics_bert
# predicted proba
LSTM_PROBA_PTH = cache.pred_proba_lstm
RESNET_PROBA_PTH = cache.pred_proba_resnet
BERT_PROBA_PTH = cache.pred_proba_bert
# predicted (top n) hashtags
LSTM_TOP_PTH = cache.pred_top_lstm
RESNET_TOP_PTH = cache.pred_top_resnet
BERT_TOP_PTH = cache.pred_top_bert
# metrics top 3/5/10 metrics of epoch 10/20/30
LSTM_METRICS_PTH = cache.metrics_lstm
RESNET_METRICS_PTH = cache.metrics_resnet
BERT_METRICS_PTH = cache.metrics_bert
# ground truth for evaluation
GT_VAL = cache.gt_val_50
GT_TEST = cache.gt_test_50
GT_STATS = cache.gt_stats
# statistical evaluation
STATS_PRED_BERT = cache.stats_pred_bert
STATS_PRED_LSTM = cache.stats_pred_lstm
STATS_PRED_RESNET = cache.stats_pred_resnet
STATS_METRICS_BERT = cache.stats_metrics_bert
STATS_METRICS_LSTM = cache.stats_metrics_lstm
STATS_METRICS_RESNET = cache.stats_metrics_resnet
STATS_CI_BERT = cache.stats_ci_bert
STATS_CI_LSTM = cache.stats_ci_lstm
STATS_CI_RESNET = cache.stats_ci_resnet




# dataset
text_dataset_train = TextDataset('train')
text_dataset_val = TextDataset('validation')
text_dataset_test = TextDataset('test')

# tag weight
tag_weight = hashtag_weight(50)


# predict and evaluation for each epoch in [10, 20, 30]
# metrics at top 3/5/10
def get_prediction_metrics(
    dataloader, model, model_pth, pred_proba_pth, pred_top_pth, \
    ground_truth, metrics_pth, epoch_list=EPOCH_LS, \
    top_n_list=TOP_N, device=device, bert=False):

    model.to(device)
    metrics = dict()
    for i in epoch_list:
        # load model
        model.load_state_dict(torch.load(model_pth.format(i), map_location=device))
        # predict proba
        pred = Predict(dataloader, model, device, bert)
        pred_proba = pred.get_pred_proba(save=True, save_pth=pred_proba_pth)
        # get top k recommendation
        metrics[i] = dict()
        for n in top_n_list:
            pred_tag = pred.get_pred_top_k(
                n, pred_proba_val=pred_proba, 
                save=True, save_pth=pred_top_pth.format(n)
                )
        me = CalculateMetrics(pred_tag, ground_truth)
        metrics[i][n] = me.get_metrics()

    # save metrics
    json_ = json.dumps(metrics)
    with open(metrics_pth, 'w') as json_file:
        json_file.write(json_)

    return 


# statistical evaluation 
def stats_eval(
    model, model_pth, collate_batch, ground_truth, pred_top_pth, 
    metrics_pth, ci_pth, device=device, 
    n=NUM_SAMPLE, top_list=TOP_N, bert=False
    ):
    # load model
    model.load_state_dict(torch.load(model_pth.format(30), map_location=device))
    stats_eval = StatisticalEval(ground_truth, device, n, top_list, bert)
    pred_group = stats_eval.predict_group(model, collate_batch, save=True, save_pth=pred_top_pth)
    df_metrics = stats_eval.get_group_metrics(ground_truth, group_result=pred_group, save=True, save_pth=metrics_pth)
    df_ci = stats_eval.get_group_metrics_ci(df_metrics, sig_level=0.05, save=True, save_pth=ci_pth)

    return 



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

# model training
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

# model training
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

# model training
def training_bert():
    bert = TextBERT()
    bert_30 = Trainer(
        bert_dataloader_train, bert_dataloader_val, bert, EPOCH,  
        THRE, BERT_MODEL_PTH, BERT_VAL_TRAIN_PTH, device,
        tag_weight=tag_weight, bert=True, validation=False
    )




if __name__ == "__main__":

    # train
    # training_lstm()
    # training_resnet()
    # training_bert()
    
    lstm = TextLSTM()
    resnet = TextResNet()
    bert = TextBERT()

    # ground truth
    gt_val = GroundTruth(text_dataset_val).get_ground_truth()
    gt_test = GroundTruth(text_dataset_test).get_ground_truth()
    gt_stats = GroundTruth(text_dataset_test).get_stats_ground_truth(50)

    # evaluation
    # LSTM
    lstm = TextLSTM()

    get_prediction_metrics(
        lstm_dataloader_val, lstm, LSTM_MODEL_PTH, LSTM_PROBA_PTH, LSTM_TOP_PTH, \
        gt_val, LSTM_METRICS_PTH, epoch_list=EPOCH_LS, \
        top_n_list=TOP_N, device=device, bert=False)
    
    stats_eval(
        lstm, LSTM_MODEL_PTH, collate_batch_lstm, gt_stats, STATS_PRED_LSTM, 
        STATS_METRICS_LSTM, STATS_CI_LSTM, device=device, 
        n=NUM_SAMPLE, top_list=TOP_N, bert=False
        )

    # ResNet
    resnet = TextResNet()

    get_prediction_metrics(
        resnet_dataloader_val, resnet, RESNET_MODEL_PTH, RESNET_PROBA_PTH, RESNET_TOP_PTH, \
        gt_val, RESNET_METRICS_PTH, epoch_list=EPOCH_LS, \
        top_n_list=TOP_N, device=device, bert=False)
    
    stats_eval(
        resnet, RESNET_MODEL_PTH, collate_batch_resnet, gt_stats, STATS_PRED_RESNET, 
        STATS_METRICS_RESNET, STATS_CI_RESNET, device=device, 
        n=NUM_SAMPLE, top_list=TOP_N, bert=False
        )
    
    # BERT
    bert = TextBERT()

    get_prediction_metrics(
        bert_dataloader_val, bert, BERT_MODEL_PTH, BERT_PROBA_PTH, BERT_TOP_PTH, \
        gt_val, BERT_METRICS_PTH, epoch_list=EPOCH_LS, \
        top_n_list=TOP_N, device=device, bert=True)
    
    stats_eval(
        bert, BERT_MODEL_PTH, collate_fn_bert, gt_stats, STATS_PRED_BERT, 
        STATS_METRICS_BERT, STATS_CI_BERT, device=device, 
        n=NUM_SAMPLE, top_list=TOP_N, bert=True
        )




