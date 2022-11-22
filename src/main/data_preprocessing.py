"""
This file contains:
1. data cleaning process for text and hashtags data
2. train/validation/test data generation
"""

import pandas as pd
import numpy as np
import random
import numpy as np
import pandas as pd
from operator import add
import csv, json, string, re, os

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

from config.config import CacheFile

cache = CacheFile()
# raw data
RAW_DATA_PTH = cache.raw_data

conf = SparkConf().setAppName("data").setMaster('local[*]')
sc = SparkContext(conf = conf)
sc.setLogLevel("WARN")
conf.set("spark.executor.memory", "32g")
conf.set("spark.driver.memory", "32g")


'''
step 1: initialize cleaning patterns
'''
wnl = WordNetLemmatizer()
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE_part = ""
for punc in PUNCT_TO_REMOVE:
    if punc not in [',','.','!','?',':',';']:
        PUNCT_TO_REMOVE_part += punc

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
url_pattern = re.compile(r'https?://\S+|www\.\S+')
html_pattern = re.compile('<.*?>')
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def filtered_hashtags(all_tags, hashtags_list):
    hashtags_set = set(hashtags_list)
    res_tags = []
    for tag in all_tags:
        if tag in hashtags_set:
            res_tags.append(tag)
    return res_tags

def clean_hashtags_col(hashtags_col):
    # get hashtag repository
    hashtags = str(hashtags_col)
    for x in ["'", " ", "[", "]"]:
        hashtags = hashtags.replace(x, "")
    return hashtags.split(',')


class DataCleaning_DL():
    def __init__(self, raw_data_path, raw_data_list):
        self.raw_data_path = raw_data_path
        self.raw_data_list = raw_data_list

    def get_hashtags_repo(self, top_n):
        hashtags = sc.parallelize([])
        for file in self.raw_data_list:
            data = pd.read_csv(self.raw_data_path + '/' + file, lineterminator='\n')
            data.hashtags = data.hashtags.str.lower()
            new_tags = sc.parallelize(data.values.tolist())\
                        .flatMap(lambda x: clean_hashtags_col(x[2]))\
                        .map(lambda x: (x, 1)).reduceByKey(add)\
                        .filter(lambda x: x[0] != 'nan' and x[0] \
                            not in STOPWORDS and x[0].isalpha())
            hashtags = hashtags.union(new_tags)
        hashtags = hashtags.reduceByKey(add).sortBy(lambda x: -x[1])           
        return hashtags.take(top_n)

    def text_cleaning(self, text, all_hashtags):
        '''
        remove url, html, emoji
        lower letter
        remove punctuations except ",.!?:;"
        word lemmatization
        retain stopwords
        remove hashtags at the end of the text, retain hashtags in the middle of text
        '''
        # remove url html emoji
        text = url_pattern.sub(r'', text)
        text = html_pattern.sub(r'', text)
        text = emoji_pattern.sub(r'', text)
        # remove punctuations
        text = text.translate(str.maketrans(
            PUNCT_TO_REMOVE_part, 
            ' '*len(PUNCT_TO_REMOVE_part)
            ))
        text = text.lower()

        # get tokens as a list
        tokens = nltk.word_tokenize(text)

        # remove hashtags at the end of the text
        hashtag_set = set(all_hashtags)
        store = []
        n_hashtag = len(hashtag_set)
        while n_hashtag:
            if not tokens:
                break
            last = tokens.pop()
            if last not in hashtag_set:
                store.append(last)
            n_hashtag -= 1
        while store:
            tokens.append(store.pop())

        # get part of speech
        tagged_sent = nltk.pos_tag(tokens)
        clean_token = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            # lemmatize words
            word = wnl.lemmatize(tag[0], pos = wordnet_pos)
            clean_token.append(word)
        return tokens

    def get_cleaned_data(self, threshold):
        # format -> (hashtag, count)
        hashtags = self.get_hashtags_repo(self.raw_data_list, 200)
        # top 200 hashtags list
        hashtags_list = [x[0] for x in hashtags]

        # format -> [clean text: str, clean hashtags: List[str]]
        cleaned_data = []
        for file in self.raw_data_list:
            df = pd.read_csv(self.raw_data_path + '/' + file, lineterminator='\n')
            text = df.text.astype(str)
            hashtags = df.hashtags.str.lower()
            for i in range(len(text)):
                all_hashtags = self.clean_hashtags_col(hashtags[i])
            # hashtags cleaning
            clean_hashtags = filtered_hashtags(all_hashtags, hashtags_list)
            if not clean_hashtags:
                continue

            # text cleaning
            clean_tokens = self.text_cleaning(text[i], all_hashtags)
            # filter candidate text: drop text with very short length 
            # comparing to its hashtags numbers
            if len(clean_tokens) / len(all_hashtags) < threshold:
                continue
            clean_text = ' '.join(clean_tokens)
            cleaned_data.append([clean_text, clean_hashtags])
        return cleaned_data

    def hashtag_encoding(self, hashtags_map, hashtags):
        hashtags = self.get_hashtags_repo(self.raw_data_list, 200)
        length = len(hashtags_map)
        encode = [0] * length
        for tag in hashtags:
            encode[hashtags_map[tag]] = 1
        return encode



class DataPreparation_DL:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def filter_top(self, top_n):
        X_new = []
        y_new = []
        for i in range(len(self.y)):
            y_new_val = self.y[i][: top_n]
            if sum(y_new_val) != 0:
                X_new.append(self.X[i])
                y_new.append(y_new_val)
        return X_new, y_new

    def get_shuffle_idx(self, top_n):
        X_new, y_new = self.filter_top(top_n)
        length = len(y_new)
        shuffle_id = [[i, random.random()] for i in range(length)]
        shuffle_id.sort(key = lambda x: x[1])
        return shuffle_id

    def dataset_split(self, train_par, val_par, top_n):
        X_new, y_new = self.filter_top(top_n)
        shuffle = self.get_shuffle_idx(top_n)
        X_sampled_train = []
        X_sampled_val = []
        X_sampled_test = []
        y_sampled_train = []
        y_sampled_val = []
        y_sampled_test = []
        n_train = round(len(shuffle) * train_par)
        n_val_1024 = len(shuffle) * val_par // 1024
        n_val = n_val_1024 * 1024
        n_test = len(shuffle) - n_train - n_val
        for i in range(len(shuffle)):
            idx = shuffle[i][0]
            if n_train > 0:
                X_sampled_train.append(X_new[idx])
                y_sampled_train.append(y_new[idx])
                n_train -= 1
            elif n_val > 0:
                X_sampled_val.append(X_new[idx])
                y_sampled_val.append(y_new[idx])
                n_val -= 1
            elif n_test > 0:
                X_sampled_test.append(X_new[idx])
                y_sampled_test.append(y_new[idx])
                n_test -= 1
        return X_sampled_train, X_sampled_val, X_sampled_test, \
               y_sampled_train, y_sampled_val, y_sampled_test


class Resampling:
    def __init__(self, X_sampled_train, X_sampled_val, X_sampled_test,\
        y_sampled_train, y_sampled_val, y_sampled_test):
        self.X_sampled_train = X_sampled_train
        self.X_sampled_val = X_sampled_val
        self.X_sampled_test = X_sampled_test
        self.y_sampled_train = y_sampled_train
        self.y_sampled_val = y_sampled_val
        self.y_sampled_test = y_sampled_test

    def get_hashtag_count(self, top_n):
        hashtag_count = {i: 0 for i in range(top_n)}
        for i in range(len(self.y_sampled_train)):
            for j in range(len(self.y_sampled_train[i])):
                if self.y_sampled_train[i][j]:
                    hashtag_count[j] += 1
        return hashtag_count

    def down_sampling(self, top_n):
        hashtag_count = self.get_hashtag_count(top_n)
        hashtag_count_down = hashtag_count.copy()
        X_downsampled = []
        y_downsampled = []

        for i in range(len(self.y_sampled_train)):
            y_encode = self.y_sampled_train[i]
            n_label = sum(y_encode)

            for j in range(len(y_encode)):
                # not delete num of hashtags less than 3000
                if y_encode[j] == 1 and hashtag_count_down[j] <= 3000:
                    delete = False
                    break
                delete = True
            
            if delete:
            # delete the data with hashtags of top 4 
            # (hashtags of top 4 is much greater than other hashtags)
                if sum(y_encode[:4]) != 0: 
                    # or n_label <= 2 or len(X_sampled_train[i]) > 300 
                    # or len(X_sampled_train[i]) < 50 or :
                    for j in range(len(y_encode)):
                        if y_encode[j] == 1:
                            # update the remaining hashtag_count_down
                            hashtag_count_down[j] -= 1
                    continue

            X_downsampled.append(self.X_sampled_train[i])
            y_downsampled.append(self.y_sampled_train[i])

        return X_downsampled, y_downsampled, hashtag_count_down

    def get_times(self, top_n):
        X_downsampled, y_downsampled, hashtag_count_down = \
            self.down_sampling(top_n)
        times = {i: 0 for i in range(top_n)}
        for key, val in hashtag_count_down.items():
            if hashtag_count_down[0] // val == 0:
                times[key] = 1
            else:
                times[key] = hashtag_count_down[0] // val
        return X_downsampled, y_downsampled, hashtag_count_down, times

    def over_sampling(self, top_n):
        X_downsampled_train, y_downsampled_train, hashtag_count_down, times = \
            self.get_times(top_n)
        hashtag_count_over = hashtag_count_down.copy()
        X_oversampled = []
        y_oversampled = []
        for i in range(len(y_downsampled_train)):
            y_encode = y_downsampled_train[i]
            add = False

            for j in range(len(y_encode)):
                # only add multiples of those with num of hashtags less than 2000
                if y_encode[j] == 1 and hashtag_count_over[j] <= 2000:
                    add = True
                    break

            if add:
                for j in range(len(y_encode)):
                    if y_encode[j] == 1:
                        # update hashtag_count_over
                        hashtag_count_over[j] += 1
                        for n in range(times[j]):
                            X_oversampled.append(X_downsampled_train[i])
                            y_oversampled.append(y_downsampled_train[i])

            # only add once     
            else:
                X_oversampled.append(X_downsampled_train[i])
                y_oversampled.append(y_downsampled_train[i])
        return X_oversampled, y_oversampled, hashtag_count_over



class DataPreparation_NON_DL:
    def __init__(self) -> None:
        pass

    def text_further_cleaning(self, text):
        # remove punctuation: include 
        text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
        # remove emoji
        text = emoji_pattern.sub(r'', text)
        # remove stopwords
        text = " ".join([word for word in str(text).split()\
                        if word not in STOPWORDS and word.isalpha()])
        # word Lemmatization, need to provide POS tag
        # Choose noun, example: running -> run, stripes -> stripe
        text = " ".join([wnl.lemmatize(word, 'n') for word in text.split()])
        return text
    
    def convert_to_df(self, X, y):
        df = pd.DataFrame(X, columns = ['text'])
        df['labels'] = y
        df['text'] = df['text'].apply(lambda x: self.text_further_cleaning(x))
        return df


if __name__ == "__main__":
    raw_data_list = os.listdir(RAW_DATA_PTH)[1:]
    d = DataCleaning_DL(RAW_DATA_PTH, raw_data_list)
    hashtags = d.get_hashtags_repo(30)
