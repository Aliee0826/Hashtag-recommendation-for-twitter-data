"""
This file contains the model structure of TFIDF + Logistic Regression
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import GridSearchCV


from config.config import CacheFile, TFIDFLogisticConfig



cache = CacheFile()
# Non-DL data with 50 hashtags
X_TRAIN_50_NON_DL_PTH = cache.X_train_50_non_dl
X_VAL_50_NON_DL_PTH = cache.X_val_50_non_dl
X_TEST_50_NON_DL_PTH = cache.X_test_50_non_dl
Y_TRAIN_50_NON_DL_PTH = cache.y_train_50_non_dl
Y_VAL_50_NON_DL_PTH = cache.y_val_50_non_dl
Y_TEST_50_NON_DL_PTH = cache.y_test_50_non_dl
MODEL_PTH = cache.model_logistic

# tfidf logistic config
tfidf_logistic = TFIDFLogisticConfig()
MAX_LEN = tfidf_logistic.tfidf_max_len


def data_loading(path):
    return np.load(path, allow_pickle=True)


def get_dataset(X,y):
  df_x=pd.DataFrame(X,columns=['text'])
  df_y=pd.DataFrame(y.tolist(),
                    columns=['label_{}'.format(i) for i in range(len(y[0]))])
  data=pd.concat([df_x, df_y],axis=1)
  return data


def build_tfidf(df, max_len=MAX_LEN):
  corpus=df['text']
  # build vocab with train and return vectorizer
  vectorizer = TfidfVectorizer(max_features=max_len)
  X = vectorizer.fit_transform(corpus)
  X = coo_matrix(X, dtype=np.float32).toarray()
  return X, vectorizer


def get_tfidf_vec(df, vectorizer):
  corpus=df['text']
  X = vectorizer.transform(corpus)
  return coo_matrix(X, dtype=np.float32).toarray()


def training_lr(X, y, params, scoring):
    lr = LabelPowerset(LogisticRegression())
    gc = GridSearchCV(lr, params, scoring=scoring, cv=2)
    gc.fit(X, y)
    return gc.best_estimator_, gc.best_params_


def cache_model(model, model_pth):
    with open(model_pth, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    X_train = data_loading(X_TRAIN_50_NON_DL_PTH)
    X_validation = data_loading(X_VAL_50_NON_DL_PTH)
    X_test = data_loading(X_TEST_50_NON_DL_PTH)
    y_train = data_loading(Y_TRAIN_50_NON_DL_PTH)
    y_validation = data_loading(Y_VAL_50_NON_DL_PTH)
    y_test = data_loading(Y_TEST_50_NON_DL_PTH)

    df_train = get_dataset(X_train,y_train)
    df_validation = get_dataset(X_validation,y_validation)
    df_test = get_dataset(X_test, y_test)

    X_train_new, vec = build_tfidf(df_train)
    X_validation_new = get_tfidf_vec(df_validation, vec)
    X_test_new = get_tfidf_vec(df_test, vec)

    y_train = df_train.drop(columns = 'text')
    y_val = df_validation.drop(columns = 'text')
    y_test = df_test.drop(columns = 'text')

    params = {
        "classifier__penalty": ['none', 'l2'],
        "classifier__C": [0.1, 1, 10],
        "classifier__max_iter": [50, 100]
    }

    best_model, best_params = training_lr(X_train, y_train, params, 'f1_samples')
    cache_model(best_model, MODEL_PTH)
