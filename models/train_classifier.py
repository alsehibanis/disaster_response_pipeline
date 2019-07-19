import sys
# import libraries
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import re

import pickle

nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

def load_data(database_filepath):
    """Loads the data from database
    Parameters:
    database_filepath : string
        path of the database file
    
    Output:
    X : numpy.ndarray
        training data
    Y : numpy.ndarray
        labels of training data
    category_names: list
        list of the category names
    """       
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df.iloc[:,-36:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize text
    Parameters:
    text : string
        text to be tokenized
    
    Output:
    tokens : list
        list of tokens
    """           
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    return tokens

def build_model():
    """build and optimize the model using grid search
    Parameters:
    None
    
    Output:
    model : sklearn.model_selection._search.GridSearchCV
        the optimized model
    """               
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())

        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])
    
    parameters = {
              'clf__estimator__min_samples_split': [2, 3]}


    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names ):
    """Evaluate model performance
    Parameters:
    model : sklearn.model_selection._search.GridSearchCV
        model to be evaluated
    X_test : numpy.ndarray
        test data
    Y_test : numpy.ndarray
        labels of test data         
    category_names : list
        list of categories names

    Output:
    None
    """      
    y_pred=model.predict(X_test)

    for i, col in enumerate(Y_test.columns.values):
        print(col)
        print(classification_report(list(Y_test.loc[:,col]), y_pred[:,i]))    
    pass


def save_model(model, model_filepath):
    """Evaluate model performance
    Parameters:
    model : sklearn.model_selection._search.GridSearchCV
        model to be saved
    model_filepath : string
        path to the save the model file
    
    Output:
    None
    """      
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass
                    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()