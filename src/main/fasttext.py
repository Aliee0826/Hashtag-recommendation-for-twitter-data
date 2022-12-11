
import fasttext
import os
import time
import json
import re
import csv
import numpy as np
import pandas as pd
import pickle
import random

"""### load data"""

match_200 = ['poshmark','shopmycloset','fashion','style','ad','amazon','stocks','ebay','halloween','news','rt','aliexpress','stockmarket','crypto','investing','nowplaying','nft','usa','mahsaamini','bitcoin','deals','technology','jobs','nfl','business','nba','twitter','travel','ai','cybersecurity','hacked','tech','trading','stock','car','python','sweepstakes','deal','options','etsy','hiring','music','marketing','adsb','cryptocurrency','javascript','investment','gaming','nike','finance','free','wfh','earnings','worldseries','btc','remotework','trending','nhl','tiktok','photography','inflation','giveaway','dowjones','sneakers','pcas','food','youtube','art','happyhalloween','nfts','markets','womenwhocode','facebook','mlb','remotejob','job','blockchain','money','hacking','robinhood','acorns','opportunity','tesla','uk','cloud','golf','love','snapchat','data','forex','vanguard','metaverse','betterment','investor','shopping','stockpile','wealthfront','twitch','clink','amas','sports','nascar','unsubscribe','eth','twine','realestate','stop','football','gamblingtwitter','facebookdown','elonmusk','paypal','invest','ukraine','canada','opiran','adidas','retail','allyinvest','nasdaq','amc','sale','etrade','cars','christmas','stash','apple','tdameritrade','round','cnn','boeing','ransomware','iot','health','sales','basketball','vimeo','digitalmarketing','vintage','innovation','daytrading','gold','disney','fanduel','scotradar','marketbreadth','bnb','security','discount','layoffs','socialmedia','radio','icloud','xbox','investors','offer','instagram','blackboard','lasvegas','kellyservices','mexicogp','beauty','google','podcast','nftcommunity','clearthelist','dropbox','hospitality','stopwarontigray','datascience','audio','jobsearch','optionstrading','layoff','breakingnews','market','cyber','win','fitness','contest','metamask','baseball','arbitrage','weaimhigher','android','bwcdeals','jackinthebox','lisa','jewelry','financialmarkets','hereforhappy','zara','wsj','jeep','canon','essaypay','discord','chanel','digital','vans']
match_27 = match_200[4:31]
match_50 = match_200[0:50]


X_train_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/X_train_temp3_nonsampled.npy", allow_pickle=True)
X_validation_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/X_val_temp3.npy", allow_pickle=True)
X_test_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/X_test_temp3.npy", allow_pickle=True)
y_train_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/y_train_temp3_nonsampled.npy", allow_pickle=True)
y_validation_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/y_val_temp3.npy", allow_pickle=True)
y_test_50 = np.load("/content/gdrive/MyDrive/540/data/cleaned/y_test_temp3.npy", allow_pickle=True)

X_test_50[0]

def get_train_input(X_train,y_train,match_list):
    X = X_train.copy()
    y = y_train.copy()
    for i in range(len(X)):
        label_idx = [idx for idx in range(len(y[i])) if y[i][idx]==1]
        prefix_lab = ''
        for idx in label_idx:
            lab = ''.join(['__label__',match_list[idx]])
            prefix_lab = ''.join([prefix_lab,lab,' '])
            
        X[i] = ''.join([prefix_lab,X[i]])
    X = pd.DataFrame(X)
    return X

"""### train & validation"""
dim_list = list(np.linspace(100,300,num=3))
lr_list = [0.05,0.1,0.5]

gt_vald_50 = json.load(open('/content/gdrive/MyDrive/540/data/pred_results/ground_truth_50unsampled_forval.json','r'))


if __name__ == "__main__":

  # training set
  get_train_input(X_train_50,y_train_50,match_50).to_csv('/content/gdrive/MyDrive/540/data/cleaned/train_50.txt',sep='\t',quoting=csv.QUOTE_NONE,index=False,header=None,quotechar="",escapechar="\\")

  # validation set
  # https://flavioclesio.com/facebook-fasttext-automatic-hyperparameter-optimization-with-autotune
  get_train_input(X_validation_50,y_validation_50,match_50).to_csv('/content/gdrive/MyDrive/540/data/cleaned/validation_50.txt',sep='\t',quoting=csv.QUOTE_NONE,index=False,header=None,quotechar="",escapechar="\\")

    
  start_time_10 = time.time()
  ft_epoch10 = fasttext.train_supervised(input='/content/gdrive/My Drive/540/data/cleaned/train_50.txt',lr=0.5,epoch=10,dim=100,wordNgrams=2,loss='ova')
  end_time_10 = time.time()

  end_time_10 - start_time_10

  ft_epoch10.save_model('/content/gdrive/MyDrive/540/data/model/fasttext/ft_epoch10.bin')

  ft_epoch10.test_label('/content/gdrive/MyDrive/540/data/cleaned/validation_50.txt')


  start_time_20 = time.time()
  ft_epoch20 = fasttext.train_supervised(input='/content/gdrive/MyDrive/540/data/cleaned/train_50.txt',lr=0.5,epoch=20,dim=100,wordNgrams=2,loss='ova')
  end_time_20 = time.time()

  end_time_20 - start_time_20

  ft_epoch20.save_model('/content/gdrive/MyDrive/540/data/model/fasttext/ft_epoch20.bin')


  start_time_30 = time.time()
  ft_epoch30 = fasttext.train_supervised(input='/content/gdrive/My Drive/540/data/cleaned/train_50.txt',lr=0.5,epoch=30,dim=100,wordNgrams=2,loss='ova')
  end_time_30 = time.time()

  end_time_30 - start_time_30

  ft_epoch30.save_model('/content/gdrive/MyDrive/540/data/model/fasttext/ft_epoch30.bin')

  





