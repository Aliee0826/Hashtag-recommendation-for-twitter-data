import pandas as pd
import numpy as np
import sklearn
from operator import add
import csv, json, string, re, os
from pyspark import SparkContext, SparkConf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, words, wordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')

'''
step 1: initialize cleaning patterns
'''
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


def clean_hashtags_col(hashtags_col):
    # get hashtag repository
    hashtags = str(hashtags_col)
    for x in ["'", " ", "[", "]"]:
        hashtags = hashtags.replace(x, "")
    return hashtags.split(',')
  
def get_hashtags(raw_data_list, top_n):
    hashtags = sc.parallelize([])
    for file in raw_data_list:
        data = pd.read_csv(raw_data_path + '/' + file, lineterminator='\n')
        data.hashtags = data.hashtags.str.lower()
        new_tags = sc.parallelize(data.values.tolist())\
                    .flatMap(lambda x: clean_hashtags_col(x[2]))\
                    .map(lambda x: (x, 1)).reduceByKey(add)\
                    .filter(lambda x: x[0] != 'nan' and x[0] not in STOPWORDS and x[0].isalpha())
        hashtags = hashtags.union(new_tags)
    hashtags = hashtags.reduceByKey(add).sortBy(lambda x: -x[1])           
    return hashtags.take(top_n)

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

def text_cleaning_dl(text, all_hashtags):
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

def filtered_hashtags(all_tags, hashtags_set):
    hashtags_set = set(hashtags_list)
    res_tags = []
    for tag in all_tags:
        if tag in hashtags_set:
            res_tags.append(tag)
    return res_tags

def get_cleaned_data(raw_data_list, threshold):
    # format -> (hashtag, count)
    hashtags = get_hashtags(raw_data_list, 200)
    # top 200 hashtags list
    hashtags_list = [x[0] for x in hashtags]

    # format -> [clean text: str, clean hashtags: List[str]]
    cleaned_data = []
    for file in raw_data_list:
        df = pd.read_csv(raw_data_path+ '/' + file, lineterminator='\n')
        text = df.text.astype(str)
        hashtags = df.hashtags.str.lower()
        for i in range(len(text)):
            all_hashtags = clean_hashtags_col(hashtags[i])
        # hashtags cleaning
        clean_hashtags = filtered_hashtags(all_hashtags, hashtags_list)
        if not clean_hashtags:
            continue

        # text cleaning
        clean_tokens = text_cleaning_dl(text[i], all_hashtags)
        # filter candidate text: drop text with very short length comparing to its hashtags numbers
        if len(clean_tokens) / len(all_hashtags) < threshold:
            continue
        clean_text = ' '.join(clean_tokens)
        cleaned_data.append([clean_text, clean_hashtags])
    return cleaned_data

def hashtag_encoding(hashtags_map, hashtags):
    length = len(hashtags_map)
    encode = [0] * length
    for tag in hashtags:
        encode[hashtags_map[tag]] = 1
    return encode




if __name__ == '__main__':
    conf = SparkConf().setAppName("data").setMaster('local[*]')
    sc = SparkContext(conf = conf)
    sc.setLogLevel("WARN")
    conf.set("spark.executor.memory", "32g")
    conf.set("spark.driver.memory", "32g")

    '''
    step 2: data cleaning
    '''
    # clean hashtags and get repository
    raw_data_path = '../../data/raw'
    raw_data_list = os.listdir(raw_data_path)
    # format -> (hashtag, count)
    hashtags = get_hashtags(raw_data_list, 200)
    # top 200 hashtags list
    hashtags_list = [x[0] for x in hashtags]

    wnl = WordNetLemmatizer()
    clean_data = get_cleaned_data(raw_data_list, 3)

    clean_data_df = pd.DataFrame(clean_data, columns = ['text', 'hashtags'])

    '''
    step 3: hashtag encoding
    '''
    hashtags_map = {hashtags_list[i]: i for i in range(len(hashtags_list))}
    clean_data_df['tashtag_encode'] = clean_data_df['hashtags'].apply(lambda x: hashtag_encoding(hashtags_map, x))
    # write hashtags map into json file
    hashtags_json = json.dumps(hashtags_map)
    with open('../../data/cleaned/hashtags_map.json', 'w') as json_file:
        json_file.write(hashtags_json)
    # write all data
    clean_data_df.to_csv('../../data/cleaned/all_data_dl.csv', index = False)

    '''
    step 4: split dataset (top 200 hashtags data)
    '''
    # stratify failed, cuz the least populated class in y has only 1 member
    X_all, X_test, y_all, y_test = sklearn.model_selection.train_test_split(\
                clean_data_df['text'], clean_data_df['tashtag_encode'],\
                test_size = 0.2, random_state = 0) 
    X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(\
            X_all, y_all, test_size = 0.25, random_state = 0)
    # cache y
    y_train_arr = np.array(y_train)
    np.save('../../data/cleaned/y_train.npy', y_train_arr) 
    y_validation_arr = np.array(y_validation)
    np.save('../../data/cleaned/y_validation.npy', y_validation_arr)
    y_test_arr = np.array(y_test)
    np.save('../../data/cleaned/y_test.npy', y_test_arr) 
    # cache X
    X_train_arr = np.array(X_train)
    np.save('../../data/cleaned/X_train.npy', X_train_arr) 
    X_validation_arr = np.array(X_validation)
    np.save('../../data/cleaned/X_validation.npy', X_validation_arr)
    X_test_arr = np.array(X_test)
    np.save('../../data/cleaned/X_test.npy', X_test_arr)

    print('Train size: {0}\nValidation size: {1}\nTest size: {2}'.format(len(X_train),len(X_validation),len(X_test)))

