"""
This file contains the process of crawling data from Twitter API
"""

import pandas as pd
import tweepy
from datetime import datetime, timedelta

from config.config import CacheFile

cache = CacheFile()
DOMAIN_ENTITY = cache.domain_entity_pairs
DOMAIN_ID = cache.domain_id
BEARER_TOKEN = cache.BEARER_TOKEN


def tweepy_query(query, start_time, end_time, BEARER_TOKEN=BEARER_TOKEN):
    """
    set up API credentital and get query from twitter api_

    Args:
        query (_str_): query string 
        start_time (_str_): query start time
        end_time (_str_): query end time

    Returns:
        _df_: query dataframe
    """

    # Obtained by sign up Twitter API account
    # Create a client
    client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
    # Get query
    tweets = client.search_recent_tweets(query=query,
                                    start_time=start_time,
                                    end_time=end_time, 
                                    tweet_fields=['created_at', 'entities'], 
                                    max_results=100)
    return tweets


def get_entity_data(entity_id, english_entity, start_time, 
                    end_time, BEARER_TOKEN=BEARER_TOKEN):
    '''
    Get tweets within a time range for specific entity id.
    Return data columns: text | created_time | hashtags | entity_name
    '''

    # Filter rule: domain entity pair, is not retweet, has hashtag, is english
    query = "context:47.{} -is:retweet has:hashtags lang:en".format(entity_id)

    # tweet fields:
        # created_at: post created time
        # entities: include tag name and other dict
    # recent tweets: can only get data within 7 days
    # max results: 100 tweets for each search
    tweets = tweepy_query(query, start_time, end_time, BEARER_TOKEN)
    
    # To dataframe: columns
    df = pd.DataFrame(columns = ['text', 'created_time', 'hashtags', 'entity_name'])

    entity_name = english_entity[english_entity['entity_id'] == entity_id]['entity_name'].values[0]
  
    # tweets not empty (because for some entities, no tweets are created during the time range)
    if tweets.data:
        for tweet in tweets.data:
            # To extract hashtags from the entities
            hashtag_lst = []
            # Although query specifies to get tweets with hashtags, 
            # some tweets still may not have hashtag key in entities
            if "hashtags" in tweet.entities:
                for ht in tweet.entities['hashtags']:
                    hashtag_lst.append(ht['tag'])
            # Restore data row
            ith_tweet = [tweet.text, tweet.created_at, hashtag_lst, entity_name]
            df.loc[len(df)] = ith_tweet
    return df


def get_date_data(english_entity, start_time, 
                  end_time, BEARER_TOKEN=BEARER_TOKEN):
    '''
    Get tweets within a time range for all domain-entity paires.
    Return data columns: text | created_time | hashtags | entity_name
    '''

    tweets = pd.DataFrame()
    n = english_entity.shape[0]
    print("Total entity names:", n)
    
    for i in range(n):
        entity_id = english_entity.loc[i, 'entity_id']
        df = get_entity_data(
            entity_id, english_entity, start_time, 
            end_time, BEARER_TOKEN
            )
        tweets = pd.concat([tweets, df], axis=0)
        if (i+1) % 10 == 0:
            print("Processed names:", i+1)
    return tweets



