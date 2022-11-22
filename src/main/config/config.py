"""
This file contains the classes for config.
To edit, values update config.ini
"""

import configparser
import os
import json

this_file = str(os.path.dirname(os.path.abspath(__file__ + "/..")))

config = configparser.ConfigParser()
config.read(this_file + "/config/config.ini")

class CacheFile:
    """
    The configs for cache files
    """
    @property
    def BEARER_TOKEN(self) -> str:
        """
        Path to the tweepy access token 
        """
        token_pth = this_file + config['cache']['BEARER_TOKEN']
        with open(token_pth, 'r') as f:
            token = json.load(f)
        return token['BEARER_TOKEN']

    @property
    def domain_id(self) -> str:
        """
        The domain id we would like to query 
        """
        return str(config['cache']['domain_id'])

    @property
    def domain_entity_pairs(self) -> str:
        """
        The path to Offical document about domain-entity pairs
        """
        return this_file + config['cache']['domain_entity_pairs']

    @property
    def raw_data(self) -> str:
        """
        The path to raw data crawled from Twitter
        """
        return this_file + config['cache']['raw_data']

    @property
    def all_data_dl(self) -> str:
        """
        The path to X train file
        """
        return this_file + config['cache']['all_data_dl']

    @property
    def X_train_all(self) -> str:
        """
        The path to X train file
        """
        return this_file + config['cache']['X_train_all']

    @property
    def X_val_all(self) -> str:
        """
        The path to X validation file
        """
        return this_file + config['cache']['X_val_all']

    @property
    def X_test_all(self) -> str:
        """
        The path to X test file
        """
        return this_file + config['cache']['X_test_all']

    @property
    def y_train_all(self) -> str:
        """
        The path to y train file
        """
        return this_file + config['cache']['y_train_all']

    @property
    def y_val_all(self) -> str:
        """
        The path to y validation file
        """
        return this_file + config['cache']['y_val_all']

    @property
    def y_test_all(self) -> str:
        """
        The path to y test file
        """
        return this_file + config['cache']['y_test_all']
    
    @property
    def X_train_50(self) -> str:
        """
        The path to X train 50 file
        """
        return this_file + config['cache']['X_train_50']

    @property
    def X_val_50(self) -> str:
        """
        The path to X validation 50 file
        """
        return this_file + config['cache']['X_val_50']

    @property
    def X_test_50(self) -> str:
        """
        The path to X test 50 file
        """
        return this_file + config['cache']['X_test_50']

    @property
    def y_train_50(self) -> str:
        """
        The path to y train 50 file
        """
        return this_file + config['cache']['y_train_50']

    @property
    def y_val_50(self) -> str:
        """
        The path to y validation 50 file
        """
        return this_file + config['cache']['y_val_50']

    @property
    def y_test_50(self) -> str:
        """
        The path to y test 50 file
        """
        return this_file + config['cache']['y_test_50']

    @property
    def X_train_50_non_dl(self) -> str:
        """
        The path to X train 50 file
        """
        return this_file + config['cache']['X_train_50_non_dl']

    @property
    def X_val_50_non_dl(self) -> str:
        """
        The path to X validation 50 file
        """
        return this_file + config['cache']['X_val_50_non_dl']

    @property
    def X_test_50_non_dl(self) -> str:
        """
        The path to X test 50 file
        """
        return this_file + config['cache']['X_test_50_non_dl']

    @property
    def y_train_50_non_dl(self) -> str:
        """
        The path to y train 50 file
        """
        return this_file + config['cache']['y_train_50_non_dl']

    @property
    def y_val_50_non_dl(self) -> str:
        """
        The path to y validation 50 file
        """
        return this_file + config['cache']['y_val_50_non_dl']

    @property
    def y_test_50_non_dl(self) -> str:
        """
        The path to y test 50 file
        """
        return this_file + config['cache']['y_test_50_non_dl']

    @property
    def hashtags_map(self) -> str:
        """
        The path to y test file
        """
        return this_file + config['cache']['hashtags_map']

    @property
    def hashtags_repository(self) -> str:
        """
        The path to y test file
        """
        return this_file + config['cache']['hashtags_repository']

    @property
    def model_bert(self) -> str:
        """
        The path to trained BERT model checkpoint
        """
        return this_file + config['cache']['model_bert']

    @property
    def model_lstm(self) -> str:
        """
        The path to trained lstm model checkpoint
        """
        return this_file + config['cache']['model_lstm']

    @property
    def model_resnet(self) -> str:
        """
        The path to trained resnet model checkpoint
        """
        return this_file + config['cache']['model_resnet']

    @property
    def pred_bert(self) -> str:
        """
        The path to trained bert predicted result
        """
        return this_file + config['cache']['pred_bert']

    @property
    def pred_lstm(self) -> str:
        """
        The path to trained lstm predicted result
        """
        return this_file + config['cache']['pred_lstm']

    @property
    def pred_resnet(self) -> str:
        """
        The path to trained resnet predicted result
        """
        return this_file + config['cache']['pred_resnet']
    
    @property
    def training_metrics_bert(self) -> str:
        """
        The path to validaiton metrics during training
        """
        return this_file + config['cache']['training_metrics_bert']
    
    @property
    def training_metrics_lstm(self) -> str:
        """
        The path to validaiton metrics during training
        """
        return this_file + config['cache']['training_metrics_lstm']
    
    @property
    def training_metrics_resnet(self) -> str:
        """
        The path to validaiton metrics during training
        """
        return this_file + config['cache']['training_metrics_resnet']




class TrainConfig:
    """
    The configs for model training
    """

    @property
    def device(self) -> str:
        """
        The device on which the model will be trained
        """
        return config['train']['device']

    @property
    def num_epoch(self) -> int:
        """
        The number of epoch for model training
        """
        return int(config['train']['num_epoch'])

    @property
    def max_len(self) -> int:
        """
        The length for text embedding
        """
        return int(config['train']['max_len'])

    @property
    def batch_size(self) -> int:
        """
        Number of data in one batch when building DataLoader
        """
        return int(config['train']['batch_size'])

    @property
    def threshold(self) -> float:
        """
        Probability threshold for prediciton
        """
        return float(config['train']['threshold'])


class BERTConfig:
    @property
    def pre_trained_model(self) -> str:
        """
        The name of pre-trained BERT model
        """
        return config['BERT']['pre_trained_model']


class LSTMConfig:
    @property
    def hidden_size(self) -> int:
        """
        The size of hidden layer
        """
        return int(config['LSTM']['hidden_size'])


class RESNETConfig:

    def num_residuals(self):
        """
        Number of residual block
        """
        return config['RESNET']['num_residuals']

    def num_channels(self):
        """
        Number of channels
        """
        return config['RESNET']['num_channels']

    def embed_size(self):
        """
        embedding size
        """
        return int(config['RESNET']['embed_size'])






















