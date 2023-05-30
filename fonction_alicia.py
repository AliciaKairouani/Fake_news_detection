import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import *
from sklearn.feature_extraction.text import CountVectorizer


def preprocess(ReviewText):
    ReviewText = ReviewText.str.replace("(<br/>)", "")
    ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
    ReviewText = ReviewText.str.replace('(&amp)', '')
    ReviewText = ReviewText.str.replace('(&gt)', '')
    ReviewText = ReviewText.str.replace('(&lt)', '')
    ReviewText = ReviewText.str.replace('(\xa0)', ' ')  
    ReviewText = ReviewText.str.replace('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ')
    return ReviewText

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def stop_word_fortext(text):
    import nltk
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    result = [i for i in word_tokens if not i.isdigit()]
    filtered = [w.lower() for w in result]
    filtered_sentence = [w for w in filtered if not w.lower() in stop_words]
    return filtered_sentence

def remove_ponctuation(text):
    '''Remove ponctuation from a text'''
    punctuation = '''!@#$%^&*(){}[]'"|’._-``></“?:;"'\,—”–~mr.ms.\n\n—'sMr.Ms.'''
    no_ponctuation = [c for c in text if c not in punctuation]
    return no_ponctuation

def remove_url(thestring):
    URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', thestring)
    return URLless_string

def wordcloud(text):
    wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).generate(' '.join([a for a in text]))
    plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off');

def stop_word(text):
    import nltk
    nltk.download('punkt')
    text = text.astype(str)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(" ".join(text))
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    result = [i for i in word_tokens if not i.isdigit()]
    filtered = [w.lower() for w in result]
    filtered_sentence = [w for w in filtered if not w.lower() in stop_words]
    return filtered_sentence

