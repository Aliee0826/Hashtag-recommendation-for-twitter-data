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
    def model_logistic(self) -> str:
        """
        The path to trained logistic model checkpoint
        """
        return this_file + config['cache']['model_logistic']

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

    @property
    def pred_proba_bert(self) -> str:
        """
        The path to trained bert predicted probability
        """
        return this_file + config['cache']['pred_proba_bert']

    @property
    def pred_proba_lstm(self) -> str:
        """
        The path to trained lstm predicted probability
        """
        return this_file + config['cache']['pred_proba_lstm']

    @property
    def pred_proba_resnet(self) -> str:
        """
        The path to trained resnet predicted probability
        """
        return this_file + config['cache']['pred_proba_resnet']
    
    @property
    def pred_top_bert(self) -> str:
        """
        The path to trained bert predicted top k
        """
        return this_file + config['cache']['pred_top_bert']

    @property
    def pred_top_lstm(self) -> str:
        """
        The path to trained lstm predicted top k
        """
        return this_file + config['cache']['pred_top_lstm']

    @property
    def pred_top_resnet(self) -> str:
        """
        The path to trained resnet predicted top k
        """
        return this_file + config['cache']['pred_top_resnet']

    @property
    def metrics_bert(self) -> str:
        """
        The path to top 3/5/10 metrics of epoch 10/20/30
        """
        return this_file + config['cache']['metrics_bert']

    @property
    def metrics_lstm(self) -> str:
        """
        The path top 3/5/10 metrics of epoch 10/20/30
        """
        return this_file + config['cache']['metrics_lstm']

    @property
    def metrics_resnet(self) -> str:
        """
        The path to top 3/5/10 metrics of epoch 10/20/30
        """
        return this_file + config['cache']['metrics_resnet']

    @property
    def stats_pred_bert(self) -> str:
        """
        The path to statistical evaluation predicted results of bert 
        """
        return this_file + config['cache']['stats_pred_bert']

    @property
    def stats_pred_lstm(self) -> str:
        """
        The path to statistical evaluation predicted results of lstm 
        """
        return this_file + config['cache']['stats_pred_lstm']

    @property
    def stats_pred_resnet(self) -> str:
        """
        The path to statistical evaluation predicted results of resnet
        """
        return this_file + config['cache']['stats_pred_resnet']

    @property
    def stats_metrics_bert(self) -> str:
        """
        The path to statistical evaluation metrics of bert
        """
        return this_file + config['cache']['stats_metrics_bert']

    @property
    def stats_metrics_lstm(self) -> str:
        """
        The path to statistical evaluation metrics of lstm
        """
        return this_file + config['cache']['stats_metrics_lstm']

    @property
    def stats_metrics_resnet(self) -> str:
        """
        The path to statistical evaluation metrics of resnet
        """
        return this_file + config['cache']['stats_metrics_resnet']


    @property
    def stats_ci_bert(self) -> str:
        """
        The path to confidence interval of metrics of bert
        """
        return this_file + config['cache']['stats_ci_bert']

    @property
    def stats_ci_lstm(self) -> str:
        """
        The path to confidence interval of metrics of lstm
        """
        return this_file + config['cache']['stats_ci_lstm']

    @property
    def stats_ci_resnet(self) -> str:
        """
        The path to confidence interval of metrics of resnet
        """
        return this_file + config['cache']['stats_ci_resnet']

    @property
    def gt_val_50(self) -> str:
        """
        The path to ground truth file of validation data
        """
        return this_file + config['cache']['gt_val_50']

    @property
    def gt_test_50(self) -> str:
        """
        The path to ground truth file of test data
        """
        return this_file + config['cache']['gt_test_50']

    @property
    def gt_stats(self) -> str:
        """
        The path to ground truth file for statistical evaluation
        """
        return this_file + config['cache']['gt_stats']


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

class TFIDFLogisticConfig:
    @property
    def tfidf_max_len(self) -> int:
        """
        The size of hidden layer
        """
        return int(config['tfidf_logistic']['tfidf_max_len'])


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



class EvalConfig:

    def n_sample(self):
        """
        Number of samples we would like to test on
        """
        return int(config['evaluate']['n_sample'])

    def top_n(self):
        """
        Number of top hashtags the model will return
        """
        return config['evaluate']['top_n']

    def epoch_list(self):
        """
        Models at which epochs we would like to validate
        """
        return config['evaluate']['epoch_list']























