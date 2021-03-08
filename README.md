# Disaster Response Machine Learning Project - Udacity

This project was created as part of the Udacity Data Science nanodegree.  The project is an attempt to create a machine learning pipeline that utilizes NLP to determine the various disaster responese related categories that a tweet or text may relate to in order to quickly determine the proper type of response.  The project consists of:

- A python script to load, clean, and persist the various messages
- A python script to process the messages through a machine learning pipeline to create a classifier to predict the categories of future messages
- A flask app to collect new messages and process the messages in the created machine learning model and to then display the predicted categories

## Installations
To install the flask app, you need to install the python packages (incuding python=3.6.3) in the requirements.txt file. Install those packages with 'pip install -r requirements.txt'

## Project Motivation
This was put together for the 'Disaster Response Pipeline' portfolio exercise as part of the Data Science nanodegree program.  The exercise provides an introduction to Natural Language Processing and Machine Learning Pipelines.  

I really enjoyed putting this together. I had seen some of the material previously in my own studies, but I had not really tried to put all of the pieces together.  It was interesting to see how a Pipeline is constructed using the scikit-learn package.  I will definitely try some more projects like this.

## File Descriptions
The folders contained in the project include:

app - html files to display the web pages and code to launch the flask app and interact with the web pages (taking input, displaying classification predictions)
data - csv files containing training data for the model, code to import and clean the training data, sqlite database to store cleaned data
model - code contating machine learning pipeline used to train and save the model and the trained machine learning model

## Current Status
The current project uses the word tokenizer and WordNetLemmatizer from the nltk package.  The project also uses the CountVectorizer, TfidfTransformer, RandomForestClassifier and GridSearchCV classes from the scikit-learn package.  The training Pipeline tries a couple of tokenizers (include or exclude punctuation), one-word and two-word n-grams, use or not use the idf, and 10 or 100 trees in the Random Forest.

Optimal parameters determined by the grid search are currently:
- n_estimators: 100
- use_idf = True
- ngram_range = (1, 2)
- tokenizer = ignore punctuation

Obviously, other combinations and models can be attempted.  Feel free to modify the code to try something else.

## Interaction
The program is launched from a terminal by executing 'python run.py' from the app folder.

## Licensing, Authors, Ackwledgements, etc.
I would like to acknowledge the Udacity Data Science nanodegree program for the inspration and tools provided to create this project.

MIT License is included.
