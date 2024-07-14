import os
import pandas as pd
import nltk
import numpy as np
# nltk.download('all')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.util import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
import keras
from keras.models import load_model


tag_map = {
'CC': None,
'CD' :wn.NOUN,
'DT': None,
'EX' :wn.ADV,
'FW': None,
'IN' :wn.ADV,
'JJ':[wn.ADJ, wn.ADJ_SAT],
'JJR': [wn.ADJ, wn.ADJ_SAT],
'JJS': [wn.ADJ, wn.ADJ_SAT],
'LS' : None,
'MD': None,
'NN':wn.NOUN,
'NNS' :wn.NOUN,
'NNP': wn.NOUN,
'NNPS':wn.NOUN,
'PDT': [wn.ADJ, wn.ADJ_SAT],
'POS': None,
'PRP': None,
'PRP$' : None,
'RB':wn.ADV,
'RBR': wn.ADV,
'RBS' :wn.ADV,
'RP': [wn.ADJ, wn.ADJ_SAT],
'SYM': None,
'TO' : None,
'UH': None,
'VB':wn. VERB,
'VBD': wn.VERB,
'VBG': wn.VERB,
'VBN': wn.VERB,
'VBP' :wn.VERB,
'VBZ' :wn.VERB,
}


def data_cleaning(text):
    punct = '''!()-[]{};:"\,<>./?@#$%^&*_~'''
    no_punct = ''
    for char in text:
        if char not in punct:
            no_punct += char.lower()
    return no_punct

def tokenzation(cleaned_text):
    tokenz = word_tokenize(cleaned_text)
    return tokenz


def stop_words_removal(tokenz):
    stop_words = set(stopwords.words("english"))
    words = []
    for word in tokenz:
        if word.casefold() not in stop_words:
            words.append(word)
    return words

def lemmatization(words):
    lemma = WordNetLemmatizer()
    lemma_text = []
    text1 = words
    tags = pos_tag(text1)
    for tag in tags:
        pos_key = tag[1]
        pos_value = tag_map.get(pos_key)
        if pos_value:
            if isinstance(pos_value, list):
                lemma_text.append(lemma.lemmatize(tag[0], pos=pos_value[0]))
            else:
                lemma_text.append(lemma.lemmatize(tag[0], pos=pos_value))
        else:
            lemma_text.append(lemma.lemmatize(tag[0]))

    return lemma_text


def create_bigrams(lemmas):
    bi_grams = list(bigrams(lemmas))
    return bi_grams

def apply_tfidf(bi_grams):
    tfidf_loaded = joblib.load('bigram_tfidf_vectorizer.joblib')

    bigram_strings = [' '.join(bigram) for bigram in bi_grams]
    flattened_bigram_strings  = ' '.join(bigram_strings)

    features = tfidf_loaded.transform([flattened_bigram_strings])
    features = tf.sparse.reorder(tf.sparse.from_dense(features.toarray()))
    return features


def predict_sentiment(features):
    model = load_model('sequential_model.h5')
    prediction = model.predict(features)
    sentiment = (prediction > 0.5).astype(int)
    if sentiment == 1:
        return "Positive Review"
    else:
        return "Negative Review"


def processing(text):
    cleaned = data_cleaning(text)
    tokenz = tokenzation(cleaned)
    words = stop_words_removal(tokenz)
    lemmas = lemmatization(words)
    bi_grams = create_bigrams(lemmas)
    features = apply_tfidf(bi_grams)
    result = predict_sentiment(features)
    return result

