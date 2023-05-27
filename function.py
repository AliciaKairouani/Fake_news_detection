import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import string

def wordcloud(data):
    stopwords = set(STOPWORDS)
    data = data.astype(str)
    fake_text = ' '.join([text for text in data])
    print('Number of words in all_text:', len(fake_text))

    wordcloud = WordCloud(width=800, height=500, stopwords = stopwords,
                          random_state=21, max_font_size=110).generate(fake_text)
    plt.figure(figsize=(15, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off');

def bar_plot(data):
    data = data.astype(str)
    title_real = ' '.join([text for text in data])
    # Séparation du texte en une liste de mots
    title_real = title_real.split()

    # Suppression des stopwords
    stop_words = set(stopwords.words("english"))
    filtered_text_list2 = [word for word in title_real if word.lower() not in stop_words]

    filtered_text_list2 = [word for word in filtered_text_list2 if word not in string.punctuation]

    # Comptage des mots
    text_counts = Counter(filtered_text_list2)
    text_common_words = [word[0] for word in text_counts.most_common(25)]
    text_common_counts = [word[1] for word in text_counts.most_common(25)]
    print(text_common_words)
    # Création du graphique
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 12))

    sns.barplot(x=text_common_words, y=text_common_counts)
    plt.title('Most Common Words Used (excluding stopwords)')
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.show()
