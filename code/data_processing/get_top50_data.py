import os, random
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

def filter_top(X, y):
    X_new = []
    y_new = []
    for i in range(len(y)):
        y_new_val = y[i][: 50]
        if sum(y_new_val) != 0:
            X_new.append(X[i])
            y_new.append(y_new_val)
    return X_new, y_new

def get_shuffle_idx(length):
    shuffle_id = [[i, random.random()] for i in range(length)]
    shuffle_id.sort(key = lambda x: x[1])
    return shuffle_id

def dataset_split(X_new, y_new, shuffle, train_par, val_par):
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
    return X_sampled_train, X_sampled_val, X_sampled_test, y_sampled_train, y_sampled_val, y_sampled_test

def get_hashtag_count(y_sampled_train, top_n):
    hashtag_count = {i: 0 for i in range(top_n)}
    for i in range(len(y_sampled_train)):
        for j in range(len(y_sampled_train[i])):
            if y_sampled_train[i][j]:
                hashtag_count[j] += 1
    return hashtag_count

def down_sampling(X_sampled_train, y_sampled_train, hashtag_count):
    hashtag_count_down = hashtag_count.copy()
    X_downsampled = []
    y_downsampled = []

    for i in range(len(y_sampled_train)):
        y_encode = y_sampled_train[i]
        n_label = sum(y_encode)

        for j in range(len(y_encode)):
            # not delete num of hashtags less than 3000
            if y_encode[j] == 1 and hashtag_count_down[j] <= 3000:
                delete = False
                break
            delete = True
        
        if delete:
        # delete the data with hashtags of top 4 (hashtags of top 4 is much greater than other hashtags)
            if sum(y_encode[:4]) != 0: # or n_label <= 2 or len(X_sampled_train[i]) > 300 or len(X_sampled_train[i]) < 50 or :
                for j in range(len(y_encode)):
                    if y_encode[j] == 1:
                        # update the remaining hashtag_count_down
                        hashtag_count_down[j] -= 1
                continue



        X_downsampled.append(X_sampled_train[i])
        y_downsampled.append(y_sampled_train[i])

    return X_downsampled, y_downsampled, hashtag_count_down

def get_times(hashtag_count_down, top_n):
    times = {i: 0 for i in range(top_n)}
    for key, val in hashtag_count_down.items():
        if hashtag_count_down[0] // val == 0:
            times[key] = 1
        else:
            times[key] = hashtag_count_down[0] // val
    return times

def over_sampling(X_downsampled_train, y_downsampled_train, hashtag_count_down):
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


if __name__ == '__main__':
    '''
    1. get the data with top 50 hashtags, and split it into train, validation, and test dataset
    '''
    # top 200 dataset
    X_train = np.load("../../data/cleaned/X_train.npy", allow_pickle=True)
    X_validation = np.load("../../data/cleaned/X_validation.npy", allow_pickle=True)
    X_test = np.load("../../data/cleaned/X_test.npy", allow_pickle=True)
    y_train = np.load("../../data/cleaned/y_train.npy", allow_pickle=True)
    y_validation = np.load("../../data/cleaned/y_validation.npy", allow_pickle=True)
    y_test = np.load("../../data/cleaned/y_test.npy", allow_pickle=True)

    X = X_train.tolist() + X_validation.tolist() + X_test.tolist()
    y = y_train.tolist() + y_validation.tolist() + y_test.tolist()

    X_new, y_new = filter_top(X, y)
    shuffle_id = get_shuffle_idx(len(y_new))
    X_sampled_train, X_sampled_val, X_sampled_test, y_sampled_train, y_sampled_val, y_sampled_test = \
        dataset_split(X_new, y_new, shuffle_id, 0.5, 0.3)

    # cache data
    np.save('../../data/cleaned/X_train_temp3_nonsampled.npy', np.array(X_sampled_train))
    np.save('../../data/cleaned/y_train_temp3_nonsampled.npy', np.array(y_sampled_train)) 
    np.save('../../data/cleaned/X_val_temp3.npy', np.array(X_sampled_val))
    np.save('../../data/cleaned/y_val_temp3.npy', np.array(y_sampled_val))
    np.save('../../data/cleaned/X_test_temp3.npy', np.array(X_sampled_test))
    np.save('../../data/cleaned/y_test_temp3.npy', np.array(y_sampled_test))

    print('Train size: {0}\nValidation size: {1}\nTest size: {2}'.format(len(X_sampled_train), len(X_sampled_val), len(X_sampled_test)))


    '''
    2. a tempt to use downsampling and oversampling to balance the dataset
    '''
    hashtag_count = get_hashtag_count(y_sampled_train, 50)
    X_downsampled_train, y_downsampled_train, hashtag_count_down = down_sampling(X_sampled_train, y_sampled_train, hashtag_count)
    times = get_times(hashtag_count_down, 50)
    X_oversampled, y_oversampled, hashtag_count_over = over_sampling(X_downsampled_train, y_downsampled_train, hashtag_count_down)

    np.save('../../data/cleaned/X_train_temp3_sampled.npy', np.array(X_oversampled))
    np.save('../../data/cleaned/y_train_temp3_sampled.npy', np.array(y_oversampled)) 

    print('Train size of sampling dataset: {0}'.format(len(X_oversampled)))



