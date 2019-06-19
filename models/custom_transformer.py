import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import string
import functools
import operator
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,tokenizer=tokenize):
        self.tokenizer = tokenizer
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenizer(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag[0:2] == 'VB' or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class VerbRateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,tokenizer=tokenize):
        self.tokenizer = tokenizer
    def verb_rate(self,text):
        n = 0.0
        n_tok = 0.0
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            for word, tag in pos_tags:
                n_tok = n_tok +1
                if tag[0:2] == 'VB' or word == 'RT':
                    n = n +1
        if n_tok == 0:
            return 0
        else:
            return n/n_tok

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.verb_rate)
        return pd.DataFrame(X_tagged)

class WordCountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,tokenizer=tokenize):
        self.tokenizer = tokenizer
        
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(len)
        return pd.DataFrame(X_tagged)


class PuncRateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,tokenizer=tokenize):
        self.tokenizer = tokenizer
    def punc_rate(self,text):
        n_punc = len(list(filter(functools.partial(operator.contains, string.punctuation), text)))
        n_tok = len(self.tokenizer(text))
        if n_tok == 0:
            return 0
        else:
            return float(n_punc)/n_tok

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.punc_rate)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

