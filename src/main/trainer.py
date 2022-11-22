"""
This file contains model trainer for deep learning models
"""


import numpy as np
import pandas as pd
from datetime import datetime
import json

import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

from config.config import CacheFile, TrainConfig


cache = CacheFile()
# data with 50 hashtags
X_TRAIN_50_PTH = cache.X_train_50
X_VAL_50_PTH = cache.X_val_50
X_TEST_50_PTH = cache.X_test_50
Y_TRAIN_50_PTH = cache.y_train_50
Y_VAL_50_PTH = cache.y_val_50
Y_TEST_50_PTH = cache.y_test_50
HASHTAG_REPO = cache.hashtags_repository


# device 
train_config = TrainConfig()
DEVICE = train_config.device


# set device to gpu
# device = torch.device("cuda:0")
device = torch.device(DEVICE)


# data loading
def data_loading():
  # X
  X_train = np.load(X_TRAIN_50_PTH, allow_pickle=True)
  X_validation = np.load(X_VAL_50_PTH, allow_pickle=True)
  X_test = np.load(X_TEST_50_PTH, allow_pickle=True)
  # y
  y_train = np.load(Y_TRAIN_50_PTH, allow_pickle=True)
  y_validation = np.load(Y_VAL_50_PTH, allow_pickle=True)
  y_test = np.load(Y_TEST_50_PTH, allow_pickle=True)
  # combine
  data = {'train': {'text': X_train, 'labels': y_train},
          'validation': {'text': X_validation, 'labels': y_validation},
          'test': {'text': X_test, 'labels': y_test}}
  return data


# load Dataset
class TextDataset(Dataset):
    def __init__(self, set_name):
      """ 
      set_name: 'train'/'validation'/'test'
      """
      self.data = data_loading()[set_name]

    def __len__(self):
      x = self.data['labels']
      return len(x)

    def __getitem__(self, idx):
      text = self.data['text'][idx]
      labels = self.data['labels'][idx]
      return text, labels


# tags weight 
def hashtag_weight(top_n):
  tag_count = pd.read_csv(HASHTAG_REPO)
  total_count = tag_count.Count.sum()
  idf = np.log(total_count/tag_count.Count)
  tag_count['count_idf'] = idf/max(idf)
  tag_weight = tag_count.count_idf.tolist()[:top_n]
  return tag_weight


# trainer
def simple_trainer(
  data_loader, model, epochs, threshold, model_pth, 
  device, tag_weight=None, bert=False
  ):

  if tag_weight is not None:
    tag_weight = torch.reshape(torch.tensor([tag_weight]), (-1,)).to(device)
    criterion = nn.BCELoss(weight = tag_weight).to(device)
  else:
    criterion = nn.BCELoss(weight = tag_weight).to(device)

  j = 0
  model = model.to(device)
  model.train()

  optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
  
  start = datetime.now()
  for epoch in range(1, epochs+1):
    j += 1
    # BERT Trainer
    if bert == True:
      for i, data in enumerate(data_loader):
        if bert == True:
          input_ids, attention_mask, token_type_ids, labels = \
            data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
          out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        else:
          text, labels = data[0].to(device), data[1].to(device)
          out = model(text).to(device)
        
        # Compute the loss and its gradients
        loss = criterion(out, labels)
        # Adjust learning weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()    
      
        # Gather data and report
        if (i+1) % 20 == 0:
            # 1 if proba > 0.5, else 0
            out = (out>threshold).float()
            accuracy = (out == labels).sum().item() / (labels.shape[0]*labels.shape[1])
            f1 = f1_score(labels.to('cpu'), out.to('cpu'), average="samples")
            print("Epoch = {}  |  ".format(j),  
                  "Iteration = {}  |  ".format(i+1), 
                  "Loss = {}  |  ".format(loss.item()), 
                  "Accuracy = {}  |  ".format(accuracy),
                  "F1 = {}  |  ".format(f1),
                  "Runtime: {}".format((datetime.now() - start).seconds))

    if j % 10 == 0:
      # create checkpoint variable and add important data
      torch.save(model.state_dict(), model_pth.format(j))

  return model



def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
    

def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    """
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, checkpoint_path)



def complex_trainer(
  data_loader_train, data_loader_val, model, epochs,  
  threshold, checkpoint_path, cache_path, device,
  pre_num_epoch=0, pre_min_loss = np.Inf,
  tag_weight=None, bert=False):

  """
  Args:
    data_loader_train(_torch.utils.data.DataLoader_):
    data_loader_val(_torch.utils.data.DataLoader_):
    model(__): the model to be trained
    tag_weight(_list_): a list with all the hashtags' weight
                        deal with data imbalance
    epochs(_int_): number of epochs to be trained
    threshold(_float_): proba threshold to predict 0/1
    checkpoint_path(_str_): the path to save best model
    cache_path(_str_): the path to save metrics of each epoch (json file)
    pre_num_epoch(_int_): number of epochs the model have already been trained.
                          default to 0.
    pre_min_loss(_float_): the lowest evaluation loss among previous training                    
    device(_torch.device_): the device model running on

  Returns: 
    model: the best model after training 
    cache(_dict_): dictionary caching F1/loss of every epoch

  """

  j = 0
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  tag_weight = torch.reshape(torch.tensor([tag_weight]), (-1,)).to(device)
  criterion = nn.BCELoss(weight = tag_weight).to(device)

  # initialize
  # cache training process
  if pre_num_epoch==0:
      cache = {
          "train_loss": [],
          "eval_loss": [],
          "train_F1": [],
          "eval_F1": []
      }
  else:
      with open(cache_path, 'r') as json_file:
          cache = json.load(json_file)
  # loss
  train_loss = 0
  eval_loss = 0
  # F1/recall/precision
  train_f1 = 0
  eval_f1 = 0
  # update condition
  min_eval_loss = pre_min_loss
  early_sign = 0
  

  start = datetime.now()
  for epoch in range(1, epochs+1):
    j += 1
    # 1. training
    model.train()
    for i, data in enumerate(data_loader_train):

      if bert == True:
        input_ids, attention_mask, token_type_ids, labels = \
          data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        out = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
      else:
        text, labels = data[0].to(device), data[1].to(device)
        out = model(text).to(device)

      # loss
      loss = criterion(out, labels)
      train_loss += loss
      # f1
      out = (out>threshold).float()
      train_f1 += f1_score(
          labels.tolist(), 
          out.tolist(), 
          average="samples", 
          zero_division=0)
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    # cache
    train_avg_loss = train_loss/len(data_loader_train)
    train_avg_f1 = train_f1/len(data_loader_train)
    cache['train_loss'].append(float(train_avg_loss))
    cache['train_F1'].append(float(train_avg_f1))

    # reset 
    print("Average Training Loss = {}".format(train_avg_loss))
    print('======================== Epoch {}: Training  End ========================='.format(j+pre_num_epoch))
    print('\n')

    # 2. evaluation
    model.eval()
    for i, data in enumerate(data_loader_val):
      if bert == True:
        input_ids, attention_mask, token_type_ids, labels = \
          data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        out = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
      else:
        text, labels = data[0].to(device), data[1].to(device)
        out = model(text).to(device)

      loss = criterion(out, labels)
      eval_loss += loss    
      out = (out>threshold).float()

      # f1/recall/precision
      eval_f1 += f1_score(
          labels.tolist(), 
          out.tolist(), 
          average="samples", 
          zero_division=0)
    
    # cache
    eval_avg_loss = eval_loss/len(data_loader_val)
    eval_avg_f1 = eval_f1/len(data_loader_val)
    cache['eval_loss'].append(float(eval_avg_loss))
    cache['eval_F1'].append(float(eval_avg_f1))

    cache_json = json.dumps(cache)
    with open(cache_path, 'w') as json_file:
        json_file.write(cache_json)

    print("Average Evaluation Loss = {}".format(eval_avg_loss))
    print("F1 = {f1}".format(f1=round(eval_avg_f1, 6)))
    print('======================== Epoch {}: Validation End ========================'.format(j+pre_num_epoch))
    print('\n')

    # 3. update model
    # create checkpoint variable and add important data
    checkpoint = {
      'epoch': j + pre_num_epoch,
      'valid_loss_min': eval_avg_loss,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
        
    # save/update checkpoint if it becomes better 
    if eval_avg_loss <= min_eval_loss:
      print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_eval_loss, eval_avg_loss))
      # save checkpoint as best model
      save_ckp(checkpoint, checkpoint_path)
      min_eval_loss = eval_avg_loss
      early_sign = j
      print("Runtime: {}".format((datetime.now() - start).seconds))
      print('============================= Epoch {}: Done ============================='.format(j+pre_num_epoch))
      print('\n')
    elif j - early_sign > 3:
      print('============================= Epoch {}: Done ============================='.format(j+pre_num_epoch))
      print('\n')
      print('Loss has not been improved for 3 epochs. Stop Training.')
      break

    # 4. reset 
    # loss
    train_loss = 0
    eval_loss = 0
    # F1/recall/precision
    train_f1 = 0
    eval_f1 = 0
       
  return model, cache


def Trainer(data_loader_train, data_loader_val, model, epochs,  
  threshold, checkpoint_path, cache_path, device,
  pre_num_epoch=0, pre_min_loss=np.Inf,
  tag_weight=None, bert=False, validation=False):

  if validation is False:
    model = simple_trainer(
      data_loader_train, model, epochs, threshold, checkpoint_path, 
      device, tag_weight=tag_weight, bert=bert
      )
    return model 

  else:
    model, cache_metrics = complex_trainer(
      data_loader_train, data_loader_val, model, epochs,  
      threshold, checkpoint_path, cache_path, device,
      pre_num_epoch=pre_num_epoch, pre_min_loss=pre_min_loss,
      tag_weight=tag_weight, bert=bert)
    return model, cache_metrics
  
  