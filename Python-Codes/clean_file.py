import re
import glob
import nltk
import emoji
import string
import pandas as pd
from bs4 import BeautifulSoup

# we can insert as many patterns as we want to this list

zero_patterns = [['as','gym','reopen'], ['as','park','reopen'], ['as','salon','reopen'], ['as', 'state', 'reopen'],
                 ['as','salon','reopen'], ['as','restaurant','reopen'], ['as','business','reopen'], 
                 ['as', 'club', 'reopen'], ['as','bar','reopen'], ['as','continue','reopen'], ['as','begin','reopen'],
                 ['as','store','reopen'], ['announce','reopen','may'], ['announce','reopen','june'], 
                 ['announce','reopen','july'], ['announce','reopen','next week'], ['annouce', 'plan', 'reopen'], 
                 ['announce', 'beach', 'reopen'], ['announce','reopen','next month'], ['announce','reopen','today'], 
                 ['announce','reopen','tomorrow'], ['announce', 'bar', 'reopen'], ['announce', 'club', 'reopen'],
                 ['allow','reopen','june'], ['allow','reopen','july'], ['allow','reopen','may'],
                 ['allow','reopen','next month'], ['allow','reopen','today'], ['allow','reopen','tomorrow'],
                 ['allow','gym','reopen'], ['allow','park','reopen'], ['allow','salon','reopen'],  
                 ['allow','theater','reopen'], ['allow','salon','reopen'], ['allow','restaurant','reopen'], 
                 ['allow','business','reopen'], ['allow','store','reopen'], ['allow','church','reopen'], 
                 ['allow','hair','reopen'], ['allow','reopen','minute'], ['allow', 'bar', 'reopen'], 
                 ['committee to reopen'], ['softball','reopen'], ['basketball','reopen'], ['baseball','reopen'], 
                 ['sport','reopen'], ['team','facilit','reopen'], ['can reopen','bar'], ['can reopen','gym'], 
                 ['can reopen','store'], ['can reopen','salon'], ['can reopen','restaurant'], 
                 ['can reopen','business'], ['can reopen','club'], ['can reopen','beach'], ['can reopen','park'], 
                 ['can reopen','if','meet'], ["can't", 'wait', 'reopen'], ['on track', 'to reopen'], 
                 ['how to reopen'], ['reopen', 'november'], ['reopen', 'september'], ['reopen', 'october'], 
                 ['reopen','with limitation'], ['reopen with','restriction'], ['on track', 'when', 'reopen'],
                 ['help','reopen','safely'], ['ask', 'to reopen'], ['whether','safe','reopen'], ['when we reopen'], 
                 ['when','safe','reopen'], ['when','bar','reopen'], ['when','club','reopen'], 
                 ['when','salon','reopen'], ['when','store','reopen'],  ['when','gym','reopen'], 
                 ['when','beach','reopen'], ['reopen','if','meet'], ['create','safe','reopen'], 
                 ['can reopen','if'], ['begin','to reopen'], ['prepare','to reopen'], ['plan','to reopen'], 
                 ['in order to reopen'], ['ways to reopen'], ['seek', 'to reopen'], ['how we reopen'],
                 ['CDC','guideline','reopen'], ['CDC','document','reopen'], ['live look', 'reopen'], ['will reopen'], 
                 ['will not reopen'], ['will never reopen'], ['may not reopen'], ['if','don','reopen'], ["won't reopen"],
                 ['no way', 'partially reopen'], ['poll', '?'], ['vote', '?'], ['please vote'], ['keep','paul','prison'], 
                 ['paul','jail'], ['reopen congress'], ['reopen impeachment'], ['reopen', 'cold case'], 
                 ['reopen america spokesman'], ['israel', 'reopen'], ['disney', 'reopen']]

prefs = ['what', 'why', 'when', 'where', 'are', 'is', 'do',  'does']
keyword = 'reopen'

def clean_text(text):
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", "")
    text = text.replace("&amp;", "")
    text = text.replace("‘", "")
    text = text.replace("’", "")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace(" – ", " ")
    text = text.replace("—", " ")
    text = text.replace("•", " ")
    text = text.replace("…", " ")
    text = text.replace("\n", ". ")
    text = text.replace("\r", ". ")
    text = text.replace("\r\n", ". ")
    text = text.replace("----", ".")
    text = text.replace("...", ".")
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'bit.ly/\S+', '', text)
    text = re.sub('(RT\s@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', text)
    text = re.sub('(@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '', text)
    text = BeautifulSoup(text, "lxml").text 
    text = text.lower().strip()
    return text

def check_pattern(text, pattern):
    return 1 if all(element in text for element in pattern) else 0

def check_text(text):
    find_pattern = 0
    for pattern in zero_patterns:
        find_pattern += check_pattern(text, pattern)
    return 'remove' if find_pattern > 0 else text
        
def check_tweet(tweet):
    tweet = clean_text(tweet)
    text_list = re.split('[,;!.]', tweet)  
    last_text = text_list[-1]
    patterns = [check_text(text) for text in text_list]
    if ('remove' in patterns or len(tweet.split()) <= 3):
        return None
    elif last_text.startswith(tuple(prefs)) and last_text.endswith('?'):
        return None
    else:
        return tweet

def filter_tweet(tweet):
    return tweet if (keyword in str(tweet).lower()) else None

def clean_tweet(tweet, label):
    return check_tweet(tweet) if pd.isnull(label) else tweet

def clean_file(dft, dfs):
    print("----------start to clean file-----------")
    print("the number of tweet before filtering:", len(dft))
    dft = dft[['user', 'verified', 'location', 'state', 'followers', 'time', 'text', 'lang', 'sentiment',
               'rt_user', 'rt_verified', 'rt_location', 'rt_followers', 'rt_time', 'rt_lang']]
    df = pd.merge(dft, dfs, on='text', how="left")
    df = df.loc[df['label'] != 0]
    print("the number of tweet after removing:", len(df))
    df['clean_text'] = df['text'].apply(filter_tweet)
    df = df.dropna(subset=['clean_text'])
    print("the number of tweet after cleaning:", len(df))
    df['clean_text2'] = df.apply(lambda row: clean_tweet(str(row['clean_text']), row['label']), axis = 1)
    df = df.dropna(subset=['clean_text2'])
    print("the number of tweet after filtering:", len(df))
    dff = df[['user', 'verified', 'location', 'state', 'followers', 'time', 'text', 'lang', 'sentiment',
              'rt_user', 'rt_verified', 'rt_location', 'rt_followers', 'rt_time', 'rt_lang']]
    return dff