"""
This file contains the model structure of BERT
"""


import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from config.config import TrainConfig, BERTConfig
from trainer import TextDataset

bert_config = BERTConfig()
PRE_BERT = bert_config.pre_trained_model
train_config = TrainConfig()
EPOCH = train_config.num_epoch
MAX_LEN = train_config.max_len
BATCH = train_config.batch_size
DEVICE = train_config.device


# load pre-trained BERT
token_bert = BertTokenizer.from_pretrained(PRE_BERT)
bert = BertModel.from_pretrained(PRE_BERT)


# dataloader
# create PyTorch DataLoader with the models's tokenizer
def collate_fn_bert(data):
    text = [str(i[0]) for i in data]
    labels = [i[1] for i in data]

    # embedding
    data = token_bert.batch_encode_plus(batch_text_or_text_pairs=text,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=MAX_LEN,
                                        return_tensors='pt',
                                        return_length=True)

    #input_ids: text vector after tokenizer
    #attention_mask: padding positions = 0, other positions = 1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.FloatTensor(labels) 

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels



# model building
class TextBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.linear1 = torch.nn.Linear(768, 50)
        self.sig = nn.Sigmoid()


    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
        # BERT -> pick up the first vector of last hidden layer
        output = self.linear1(output.last_hidden_state[:, 0])
        return self.sig(output)


# class TextLabelModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = torch.nn.Linear(768, 50)
#         self.sig = nn.Sigmoid()

#     def forward(self, input_ids, attention_mask, token_type_ids):
      
#         with torch.no_grad():
#             output = bert(input_ids=input_ids,
#                           attention_mask=attention_mask,
#                           token_type_ids=token_type_ids)

#         # BERT -> pick up the first vector of last hidden layer
#         output = self.linear1(output.last_hidden_state[:, 0])
#         return self.sig(output)