import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
#######
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sqlalchemy
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """Load the dataset that we  have previously saved in sql format and split it into X and y """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
  
    df = pd.read_sql_table('Disaster',con =  engine)
    
    X = df['message']
    y = df.drop(['id','genre','message','original'],axis=1)
    category_names = list(y.columns)
    
    return X,y, category_names


def tokenize(text):
    
    """1. Tokenize the words, then  remove stop words and then take out the lemma of each word!!"""
    
    lemmatizer = WordNetLemmatizer()
    
    text_new = re.sub(r"[^a-zA-Z0-9]"," ", text)
    
    tokens = word_tokenize(text_new.lower())
    lemma = [ lemmatizer.lemmatize(word).strip() for word in tokens if word not in stopwords.words('english')]
    
    return lemma


def build_model():
    
    """We will create pipeline here and we will be using Decision Tree Classifier here"""
    base_model = DecisionTreeClassifier()
    
    pipeline = Pipeline([
    ('count', CountVectorizer(tokenizer = tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf', MultiOutputClassifier(base_model))])
    
#     parameters = {'count__min_df': [1, 2],
#                 'tfidf__use_idf': [True, False],
#                 'clf__estimator__max_depth': [3,4],
#                 'clf__estimator__min_samples_split': [2, 4]}

    #cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    """Here we have to calculate the precision, recall and f1- scores for each categories!!"""
    
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred,target_names = category_names)
    print(class_report)

def save_model(model, model_filepath):
    
    """Its time to save the model using dump command"""
    pickle.dump(model,open(model_filepath, 'wb'))


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