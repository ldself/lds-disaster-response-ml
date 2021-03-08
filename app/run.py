import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import pickle


app = Flask(__name__)

#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()
#
#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)
#
#    return clean_tokens

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
#model = joblib.load("../models/your_model_name.pkl")
model = pickle.load(open("../models/Classifier-RandomForest.pkl", "rb"))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_dist = categories.sum() / categories.shape[0]
    category_counts = list(category_dist)
    category_names = list(category_dist.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    y=category_counts,
                    x=category_names
                    #orientation='h'
                )
            ],
            
            'layout': {
                'title': 'Frequency of Each Label',
                'xaxis': {
                    'title': "Label"
                },
                'yaxis': {
                    'title': "Frequency",
                    'range': [0, 1]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
 #   app.run(debug=True)


if __name__ == '__main__':
    main()