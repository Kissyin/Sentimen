#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import re
import argparse
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import TweetTokenizer

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# Setup argument parser
parser = argparse.ArgumentParser(description="Preprocess text files.")
parser.add_argument('--inputdir', type=str, required=True, help='Input directory path')
parser.add_argument('--outputdir', type=str, required=True, help='Output directory path')

args = parser.parse_args()

# Import stopwords and initialize stemmer
stopwords_indonesia = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])
 
# All emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

def load_data(file_path):
    return pd.read_excel(file_path)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def remove(tweet):
    tweet = re.sub('[0-9]+', '', tweet)
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet

def clean_tweets(tweet):
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r',', '', tweet)
    tweet = re.sub('[0-9]+', '', tweet)
    tweet = re.sub(r'\bhttps\b', '', tweet)
    tweet = re.sub(r'\bt\b', '', tweet)
    tweet = re.sub(r'\bco\b', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
 
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_indonesia and
              word not in emoticons and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
 
    return tweets_clean

def remove_punct(text):
    text = " ".join([char for char in text if char not in string.punctuation])
    return text

# Process each file in the input directory
input_dir = args.inputdir
output_dir = args.outputdir

for file_name in os.listdir(input_dir):
    if file_name.endswith('.xlsx'):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_Cleaned.csv")

        tweet_df = load_data(input_file_path)
        df = pd.DataFrame(tweet_df[['id_str', 'text']])
        df['text'] = df['text'].astype(str)
        df['remove_user'] = np.vectorize(remove_pattern)(df['text'], r"@[\w]*")
        df['remove_http'] = df['remove_user'].apply(lambda x: remove(x))
        df.sort_values("remove_http", inplace=True)
        df.drop_duplicates(subset="remove_http", keep='first', inplace=True)
        df['tweet_clean'] = df['remove_http'].apply(lambda x: clean_tweets(x))
        df['Tweet'] = df['tweet_clean'].apply(lambda x: remove_punct(x))
        df.sort_values("Tweet", inplace=True)
        df.drop(df.columns[[0, 1, 2, 3, 4]], axis=1, inplace=True)
        df.drop_duplicates(subset="Tweet", keep='first', inplace=True)
        df.to_csv(output_file_path, encoding='utf8', index=False)
