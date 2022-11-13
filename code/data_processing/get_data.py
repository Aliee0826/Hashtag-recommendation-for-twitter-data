import pandas as pd
import tweepy
from datetime import datetime, timedelta

# Offical document about domain-entity pairs
entity = pd.read_csv("/content/gdrive/MyDrive/540/docs/evergreen-context-entities-20220601.csv")

# Filter pairs for brand domain only (id: 47)
entity2 = entity.assign(domains=entity.domains.str.split(',')).explode('domains')
entity2 = entity2[entity2['domains'] == '47'].reset_index(drop=True)
# Keep entity with english character
english_entity = pd.DataFrame(columns=['entity_id', 'entity_name'])
for i in range(entity2.shape[0]):
    name = entity2.loc[i, 'entity_name']
    if name.isascii():
        english_entity.loc[len(english_entity)] = [entity2.entity_id[i], entity2.entity_name[i]]


# Obtained by sign up Twitter API account
BEARER_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# Create a client
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def get_entity_data(entity_id, start_time, end_time):
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
    tweets = client.search_recent_tweets(query=query,
                                        start_time=start_time,
                                        end_time=end_time, 
                                        tweet_fields=['created_at', 'entities'], 
                                        max_results=100)
    
    # To dataframe: columns
    df = pd.DataFrame(columns = ['text', 'created_time', 'hashtags', 'entity_name'])

    entity_name = english_entity[english_entity['entity_id'] == entity_id]['entity_name'].values[0]
  
    # tweets not empty (because for some entities, no tweets are created during the time range)
    if tweets.data:
        for tweet in tweets.data:
            # To extract hashtags from the entities
            hashtag_lst = []
            # Although query specifies to get tweets with hashtags, some tweets still may not have hashtag key in entities
            if "hashtags" in tweet.entities:
                for ht in tweet.entities['hashtags']:
                    hashtag_lst.append(ht['tag'])
            # Restore data row
            ith_tweet = [tweet.text, tweet.created_at, hashtag_lst, entity_name]
            df.loc[len(df)] = ith_tweet
    return df

def get_date_data(start_time, end_time):
    '''
    Get tweets within a time range for all domain-entity paires.
    Return data columns: text | created_time | hashtags | entity_name
    '''
    tweets = pd.DataFrame()
    n = english_entity.shape[0]
    print("Total entity names:", n)
    
    for i in range(n):
        entity_id = english_entity.loc[i, 'entity_id']
        df = get_entity_data(entity_id, start_time, end_time)
        tweets = pd.concat([tweets, df], axis=0)
        if (i+1) % 10 == 0:
            print("Processed names:", i+1)
    return tweets

# Create time range for the past 7 days
start_time = []
end_time = []
# past 7 days
for i in range(1, 8):
    d1 = datetime.today() - timedelta(days=i)
    d2 = datetime.today() - timedelta(days=i-1)
    start_time.append(d1.strftime("%Y-%m-%d")+"T00:00:00.000Z")
    end_time.append(d2.strftime("%Y-%m-%d")+"T00:00:00.000Z")

# Get past 7 days data
total_len = 0
for i in range(len(start_time)):
    tweets = get_date_data(start_time[i], end_time[i])
    total_len += tweets.shape[0]
    path = "/content/gdrive/MyDrive/540/data/raw/data_{}_{}.csv".format(start_time[i][:10], end_time[i][:10])
    tweets.to_csv(path, header=True, index=False)
    print("Finished: %s to %s"%(start_time[i][:10], end_time[i][:10]))
    print("Volumn:", total_len)
