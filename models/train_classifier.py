import sys
from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    """
    Access database located at provided path and extracts attributes and
       target variables

    Parameters
    ----------
    database_filepath : str
        Database location.

    Returns
    -------
    X : numpy.ndarray
        Attributes of the training data.
    Y : numpy.ndarray
        Multilabel target variables of training data.
    Y_columns : pandas.Index
        Index of the multilabel column names.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', con=engine)
    X = df['message'].values
    Y_df = df.drop(['id','message','original','genre'], axis=1)
    Y_columns = Y_df.columns
    Y = Y_df.values
    
    return X, Y, Y_columns


def tokenize(text):
    """
    Applies tokenization logic to message.
       - Converts all words to lower-case
       - Removes any punctuation
       - Removes any stop words
       - Converts words to lemmatized root

    Parameters
    ----------
    text : str
        Character string containing words to be tokenized.

    Returns
    -------
    tokens : list
        List of tokenized words.

    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case
    text = text.lower()
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
 
    return tokens

def tokenize_keep_punctuation(text):
    """
    Applies tokenization logic to message.
       - Converts all words to lower-case
       - Does NOT remove any punctuation
       - Removes any stop words
       - Converts words to lemmatized root

    Parameters
    ----------
    text : str
        Character string containing words to be tokenized.

    Returns
    -------
    tokens : list
        List of tokenized words.

    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case
    text = text.lower()
    
    # remove punctuation
    # text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """
    Builds Multi Output classifier 
    
    Parameters
    -------
    None.

    
    Returns
    -------
    cv : sklearn.model_selection.GridSearchCV
        scikit-learn model based on defined pipeline and available parameters
        
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('class', MultiOutputClassifier(RandomForestClassifier()))
    #    ('class', MultiOutputClassifier(LogisticRegression()))
    #    ('class', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=2)))
    ])
    
    parameters = {
        'vect__tokenizer': [tokenize, tokenize_keep_punctuation],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'class__estimator__n_estimators': [10, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Creates predictions based on test data, displays the optimal parameters
       of the model, and displays the scores for each output label
    
    
    Parameters
    ----------
    model : sklearn.model_selection.GridSearchCV
        Trained machine learning model
    
    X_test : numpy.ndarray
        Test data attributes
        
    Y_test : numpy.ndarray
        Test data target variables
        
    category_names : list
        List of multi output variable target names

    Returns
    -------
    None
    
    """
    
    # predict model outcomes
    Y_pred = model.predict(X_test)
    
    # show best parameters
    print(model.best_params_)
    
    # show scores for each category
    for i in range(Y_pred.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    

def save_model(model, model_filepath):
    """
    Serializes the machine learning model to a pickle file
    
    Parameters
    ----------
    model : sklearn.model_selection.GridSearchCV
        Trained machine learning model
        
    model_filepath : str
        File location of the saved model
        
    Returns
    -------
    None
    
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
        
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