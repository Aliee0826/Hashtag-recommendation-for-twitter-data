# Hashtag-recommendation-for-twitter-data
Recommendations simulator for qualitative testing given mock personas fitting model requirements.

## Important Packages Requirement
`Python 3.8`
`PySpark 3.3.1`
`PyTorch 1.13.0`
`tweepy 4.12.1`
`huggingface-hub 0.11.0`

## Installing
Start up a fresh virtual environment in the same version as models you want to test, for example:
`conda create -n twitter_hashtag38 python=3.8`
`conda activate twitter_hashtag38`

Then run:
`pip install -r requirements.txt`

## Updating Data
Data: run `data_utils.py`
1. Data Collection:
   * You <b>MUST</b> have your own <b>TWitter API BEARER_TOKEN</b> and save it to `src/main/data/tweepy_token/BEARER_TOKEN.json`
2. Data Preparation:
   * Simply run `data_utils.py` to get cleaned data with 200 hashtags, cleaned data with 50 hashtags, and cleaned data with 50 hashtags for non-DL models

## Updating Models
Models must be added in `src/main` folder, for now we have `lstm.py`, `resnet.py`, `bert.py`.

## Model Training/Evaluation/Prediction
Simply run `main.py`