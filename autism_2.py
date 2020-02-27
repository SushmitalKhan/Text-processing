#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import os
from ast import literal_eval
from collections import defaultdict
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize


# In[2]:


base_path = 'D:\\autism_data\\2019-05_140 Chars'
all_data = []
for filename in os.listdir(base_path):
    if filename.endswith('.json'):
        with open(os.path.join(base_path, filename)) as f:
            myjson = json.loads(f.read())
            all_data.extend(myjson)
            print('Loaded {} JSON Objects'.format(len(myjson)))


# In[3]:


my_data_dict = defaultdict(list)
missing_retweet = []
for tweet in all_data:
    if 'rt' in tweet['text'].lower()[:4]:
        try:
            my_data_dict['text'].append(tweet['retweeted_status']['text'])
            my_data_dict['oc_id'].append(tweet['retweeted_status']['id'])
            my_data_dict['oc_date'].append(tweet['retweeted_status']['created_at'])
            my_data_dict['rt_id'].append(tweet['id'])
            my_data_dict['rt_date'].append(tweet['created_at'])
        except KeyError as e:
            missing_retweet.append(tweet)
            
print('{} Tweets did not contain retweet status but started with RT'.format(len(missing_retweet)))
tweet_frame = pd.DataFrame(my_data_dict).astype({'oc_date': np.datetime64, 'rt_date': np.datetime64})
tweet_frame['time_diff'] = tweet_frame[['oc_date', 'rt_date']].apply(lambda 
                                                                     row: row['rt_date'] - row['oc_date'], 
                                                                     axis=1)
tweet_frame.head()


# In[4]:


analyser = SentimentIntensityAnalyzer()

def get_sentiment(tweet_t):
    scores = analyser.polarity_scores(tweet_t)
    return scores
    
tweet_frame[['neg', 'neu', 'pos', 'compound']] = tweet_frame.apply(lambda row: pd.Series(get_sentiment(row['text'])), axis=1)
tweet_frame.head()


# In[5]:


tweet_frame.groupby(['time_diff']).max()


# In[9]:


out_path = 'D:\autism_data'

tweet_frame.to_csv('D:\\autism_data\\tweet_data.txt', sep=',', index=False)


# In[ ]:




