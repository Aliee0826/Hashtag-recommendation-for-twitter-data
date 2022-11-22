"""
This file contains data cleaning and preparation process
"""
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sklearn

from config.config import CacheFile
from data_collection import get_date_data
from data_preprocessing import DataCleaning_DL, Resampling, DataPreparation_DL, DataPreparation_NON_DL


cache = CacheFile()
# raw data
RAW_DATA_PTH = cache.raw_data
DOMAIN_ENTITY = cache.domain_entity_pairs
DOMAIN_ID = cache.domain_id
BEARER_TOKEN = cache.BEARER_TOKEN
# data with 200 hashtags
ALL_DATA_DL = cache.all_data_dl
X_TRAIN_ALL_PTH = cache.X_train_all
X_VAL_ALL_PTH = cache.X_val_all
X_TEST_ALL_PTH = cache.X_test_all
Y_TRAIN_ALL_PTH = cache.y_train_all
Y_VAL_ALL_PTH = cache.y_val_all
Y_TEST_ALL_PTH = cache.y_test_all
# data with 50 hashtags
X_TRAIN_50_PTH = cache.X_train_50
X_VAL_50_PTH = cache.X_val_50
X_TEST_50_PTH = cache.X_test_50
Y_TRAIN_50_PTH = cache.y_train_50
Y_VAL_50_PTH = cache.y_val_50
Y_TEST_50_PTH = cache.y_test_50
# Non-DL data with 50 hashtags
X_TRAIN_50_NON_DL_PTH = cache.X_train_50_non_dl
X_VAL_50_NON_DL_PTH = cache.X_val_50_non_dl
X_TEST_50_NON_DL_PTH = cache.X_test_50_non_dl
Y_TRAIN_50_NON_DL_PTH = cache.y_train_50_non_dl
Y_VAL_50_NON_DL_PTH = cache.y_val_50_non_dl
Y_TEST_50_NON_DL_PTH = cache.y_test_50_non_dl
# hashtags
HASHTAG_MAP = cache.hashtags_map
HASHTAG_REPO = cache.hashtags_repository



# 1. Data Collection
def get_past_k_days_data(k, RAW_DATA_PTH, 
                         DOMAIN_ENTITY=DOMAIN_ENTITY, 
                         DOMAIN_ID=DOMAIN_ID, 
                         BEARER_TOKEN=BEARER_TOKEN):

    """get data of past 7 days
    """

    # Offical document about domain-entity pairs
    entity = pd.read_csv(DOMAIN_ENTITY)

    # Filter pairs for brand domain only (id: 47)
    entity2 = entity.assign(domains=entity.domains.str.split(',')).explode('domains')
    entity2 = entity2[entity2['domains'] == DOMAIN_ID].reset_index(drop=True)
    # Keep entity with english character
    english_entity = pd.DataFrame(columns=['entity_id', 'entity_name'])
    for i in range(entity2.shape[0]):
        name = entity2.loc[i, 'entity_name']
        if name.isascii():
            english_entity.loc[len(english_entity)] = [
                entity2.entity_id[i], entity2.entity_name[i]
                ]

    # Create time range for the past k days
    start_time = []
    end_time = []
    # past k days
    for i in range(1, k+1):
        d1 = datetime.today() - timedelta(days=i)
        d2 = datetime.today() - timedelta(days=i-1)
        start_time.append(d1.strftime("%Y-%m-%d")+"T00:00:00.000Z")
        end_time.append(d2.strftime("%Y-%m-%d")+"T00:00:00.000Z")

    # Get past 7 days data
    total_len = 0
    for i in range(len(start_time)):
        tweets = get_date_data(
            english_entity, start_time[i], 
            end_time[i], BEARER_TOKEN
            )
        total_len += tweets.shape[0]
        path = (RAW_DATA_PTH + '/data_{}_{}.csv')\
            .format(start_time[i][:10], end_time[i][:10])
        tweets.to_csv(path, header=True, index=False)
        print("Finished: %s to %s"%(start_time[i][:10], end_time[i][:10]))
        print("Volumn:", total_len)




# 2. Data Cleaning for all Data - 200 Hashtags
def get_all_data_ready_dl():

    '''
    step 2: data cleaning
    '''
    print("Start DL Data Cleaning..")
    # clean hashtags and get repository
    raw_data_list = os.listdir(RAW_DATA_PTH)[1:]
    data_clean = DataCleaning_DL(RAW_DATA_PTH, raw_data_list)
    # format -> (hashtag, count)
    hashtags = data_clean.get_hashtags_repo(200)
    # top 200 hashtags list
    hashtags_list = [x[0] for x in hashtags]
    clean_data = data_clean.get_cleaned_data(3)
    clean_data_df = pd.DataFrame(clean_data, columns = ['text', 'hashtags'])
    print("Finished.")

    '''
    step 3: hashtag encoding
    '''
    print("Start Preparing 200 Hashtags Data..")
    print("--> Get 200 Hashtags Repository..")
    hashtags_map = {hashtags_list[i]: i for i in range(len(hashtags_list))}
    clean_data_df['tashtag_encode'] = clean_data_df['hashtags'].\
        apply(lambda x: data_clean.hashtag_encoding(hashtags_map, x))
    # write hashtags map into json file
    hashtags_json = json.dumps(hashtags_map)
    with open(HASHTAG_MAP, 'w') as json_file:
        json_file.write(hashtags_json)
    # write all data
    clean_data_df.to_csv(ALL_DATA_DL, index = False)

    '''
    step 4: split dataset (top 200 hashtags data)
    '''
    print("--> Split Train/Validation/Test..")
    # stratify failed, cuz the least populated class in y has only 1 member
    X_all, X_test, y_all, y_test = sklearn.model_selection.train_test_split(\
                clean_data_df['text'], clean_data_df['tashtag_encode'],\
                test_size = 0.2, random_state = 0) 
    X_train, X_validation, y_train, y_validation = sklearn.model_selection.\
        train_test_split(X_all, y_all, test_size = 0.25, random_state = 0)
    # cache y
    y_train_arr = np.array(y_train)
    np.save(Y_TRAIN_ALL_PTH, y_train_arr) 
    y_validation_arr = np.array(y_validation)
    np.save(Y_VAL_ALL_PTH, y_validation_arr)
    y_test_arr = np.array(y_test)
    np.save(Y_TEST_ALL_PTH, y_test_arr) 
    # cache X
    X_train_arr = np.array(X_train)
    np.save(X_TRAIN_ALL_PTH, X_train_arr) 
    X_validation_arr = np.array(X_validation)
    np.save(X_VAL_ALL_PTH, X_validation_arr)
    X_test_arr = np.array(X_test)
    np.save(X_TEST_ALL_PTH, X_test_arr)

    print('Train size: {0}\nValidation size: {1}\nTest size: {2}'.\
        format(len(X_train),len(X_validation),len(X_test)))
    print("Finished.")



# 3. Data Preparation for 50 Hashtags 
def get_50tags_data_ready_dl():
    '''
    1. get the data with top 50 hashtags, 
       and split it into train, validation, and test dataset
    '''
    print("Start Preparing 50 Hashtags Data..")
    # top 200 dataset
    X_train = np.load(X_TRAIN_ALL_PTH, allow_pickle=True)
    X_validation = np.load(X_VAL_ALL_PTH, allow_pickle=True)
    X_test = np.load(X_TEST_ALL_PTH, allow_pickle=True)
    y_train = np.load(Y_TRAIN_ALL_PTH, allow_pickle=True)
    y_validation = np.load(Y_VAL_ALL_PTH, allow_pickle=True)
    y_test = np.load(Y_TEST_ALL_PTH, allow_pickle=True)

    X = X_train.tolist() + X_validation.tolist() + X_test.tolist()
    y = y_train.tolist() + y_validation.tolist() + y_test.tolist()

    data_prep = DataPreparation_DL(X, y)
    print("--> Split Train/Validation/Test..")
    X_sampled_train, X_sampled_val, X_sampled_test, y_sampled_train, y_sampled_val, y_sampled_test = \
        data_prep.dataset_split(0.5, 0.3, 50)

    # cache 50 hashtags data
    np.save(X_TRAIN_50_PTH, np.array(X_sampled_train))
    np.save(Y_TRAIN_50_PTH, np.array(y_sampled_train)) 
    np.save(X_VAL_50_PTH, np.array(X_sampled_val))
    np.save(Y_VAL_50_PTH, np.array(y_sampled_val))
    np.save(X_TEST_50_PTH, np.array(X_sampled_test))
    np.save(Y_TEST_50_PTH, np.array(y_sampled_test))

    print('Train size: {0}\nValidation size: {1}\nTest size: {2}'.\
        format(len(X_sampled_train), len(X_sampled_val), len(X_sampled_test)))
    print("Finished.")


    '''
    2. a tempt to use downsampling and oversampling to balance the dataset
    '''
    # re_sample = Resampling(X_sampled_train, X_sampled_val, X_sampled_test,\
    #                        y_sampled_train, y_sampled_val, y_sampled_test)
    # hashtag_count =  re_sample.get_hashtag_count(50)
    # X_downsampled_train, y_downsampled_train, hashtag_count_down, times = re_sample.get_times(50)
    # X_oversampled, y_oversampled, hashtag_count_over = re_sample.over_sampling(50)

    # np.save('../../data/cleaned/X_train_temp3_sampled.npy', np.array(X_oversampled))
    # np.save('../../data/cleaned/y_train_temp3_sampled.npy', np.array(y_oversampled)) 

    # print('Train size of sampling dataset: {0}'.format(len(X_oversampled)))



# 4. Non-DL Data Cleaning for 50 Hashtags
def get_50tags_data_ready_non_dl():
    print("Start Preparing 50 Hashtags NON-DL Data..")
    data_prep = DataPreparation_NON_DL()
    # X
    X_train = np.load(
        X_TRAIN_50_PTH, allow_pickle=True
        ).reshape(-1, 1).tolist()
    X_validation = np.load(
        X_VAL_50_PTH, allow_pickle=True
        ).reshape(-1, 1).tolist()
    X_test = np.load(
        X_TEST_50_PTH, allow_pickle=True
        ).reshape(-1, 1).tolist()
    
    # y
    y_train = np.load(
        Y_TRAIN_50_PTH, allow_pickle=True
        ).tolist()
    y_validation = np.load(
        Y_VAL_50_PTH, allow_pickle=True
        ).tolist()
    y_test = np.load(
        Y_TEST_50_PTH, allow_pickle=True
        ).tolist()
    
    # combine
    print("--> Further Cleaning..")
    cleaned_data = {
        'train': data_prep.convert_to_df(X_train, y_train),
        'validation': data_prep.convert_to_df(X_validation, y_validation),
        'test': data_prep.convert_to_df(X_test, y_test)
        }
    
    # cache y
    y_train_arr = np.array(cleaned_data['train']['labels'])
    np.save(Y_TRAIN_50_NON_DL_PTH, y_train_arr) 
    y_validation_arr = np.array(cleaned_data['validation']['labels'])
    np.save(Y_VAL_50_NON_DL_PTH, y_validation_arr)
    y_test_arr = np.array(cleaned_data['test']['labels'])
    np.save(Y_TEST_50_NON_DL_PTH, y_test_arr) 

    # cache x
    X_train_arr = np.array(cleaned_data['train']['text'])
    np.save(X_TRAIN_50_NON_DL_PTH, X_train_arr) 
    X_validation_arr = np.array(cleaned_data['validation']['text'])
    np.save(X_VAL_50_NON_DL_PTH, X_validation_arr)
    X_test_arr = np.array(cleaned_data['test']['text'])
    np.save(X_TEST_50_NON_DL_PTH, X_test_arr)
    print("Finished.")




if __name__ == "__main__":
    # get_past_k_days_data(
    #     1, RAW_DATA_PTH, 
    #     DOMAIN_ENTITY, 
    #     DOMAIN_ID, 
    #     BEARER_TOKEN
    #     )
    # get_all_data_ready_dl()
    get_50tags_data_ready_dl()
    get_50tags_data_ready_non_dl()
