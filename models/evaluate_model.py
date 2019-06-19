# import libraries

import re
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.externals import joblib



import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from custom_transformer import StartingVerbExtractor,PuncRateExtractor,VerbRateExtractor,WordCountExtractor

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.scorer import make_scorer

import pickle
import cloudpickle

from datetime import datetime
from time import time


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages',engine)
    X = df['message']
    Y = df.iloc[:,4:].astype(int)
    min_labels = Y.sum().sort_values()
    Y.drop(min_labels[min_labels<2].index.values,1,inplace= True)
    return X,Y

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = RegexpTokenizer(r'\w+').tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if not clean_tok in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    # text processing and model pipeline
    ## The tranformation estimators are from the coustom_transformer.py file

    model_pipeline = Pipeline([
        ('features', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', RandomForestClassifier(n_estimators=100,n_jobs=-1))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'features__use_idf': [True,False],
        'features__ngram_range': [(1,1),(1,3)],
        'clf__n_estimators' : [100,150,200],
            
        
    }



    # create gridsearch object and return as final model pipeline
    custom_accuracy_score = make_scorer(mean_metric)
    cv = GridSearchCV(model_pipeline,parameters,verbose=100,cv=2,scoring=custom_accuracy_score,n_jobs=-1)


    return model_pipeline


def evaluate_model(model, X_test, Y_test):
    Y_test_pred = model.predict(X_test)
    report = evaluate(Y_test,Y_test_pred, avg_method = 'weighted avg')
    print(report.describe().drop(['count']).round(3))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    return filename





def mean_metric(Y,Y_pred,metric = accuracy_score):
    print('faris')
    metric_sum = 0
    for i in range(Y.shape[1]):
        metric_sum = metric_sum + metric(Y[:,i],Y_pred[:,i])
    return metric_sum / Y.shape[1]

def evaluate(Y_test,Y_test_pred, avg_method = 'weighted avg'):
    cr_sum = pd.DataFrame()
    for col in list(range(Y_test_pred.shape[1])):
        y_pred = Y_test_pred[:,col]
        y = Y_test.iloc[:,col].astype(int)
        cr = pd.DataFrame(classification_report(y,y_pred,output_dict=True)).round(2)
        cr.index.name = Y_test.columns.values[col]
        cr.drop(['support'],axis=0,inplace=True)
        cr = cr[avg_method].to_frame().T
        cr.index= [Y_test.columns.values[col]]
        cr['accuracy'] = round(accuracy_score(y,y_pred),2)
        if cr_sum.shape[0] == 0:
            cr_sum = cr
        else:
            cr_sum = cr_sum.append(cr)
            
    cr_sum.columns.name = avg_method
    return cr_sum
def load_model(model_filepath):
    model = joblib.load(model_filepath)
    return model

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Loading model...')
        model = load_model(model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()