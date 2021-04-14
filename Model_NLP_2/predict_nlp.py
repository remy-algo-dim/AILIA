import pandas as pd
import sklearn.model_selection as sms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from spacy.lang.fr import French
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle

from train_nlp import spacy_tokenizer

""" Ce fichier a pour but de renvoyer les predictions du modele NLP de classification de blocs
pour de nouveaux samples en input """

def new_predict(model, sample, vectorizer, selector):
    """Input: model (pkl file), string text, vectorizer and selector pkls
       Output: prediction (block label)"""
    sample = pd.Series(sample)
    sample_preprocessed = vectorizer.transform(new_sample)
    sample_preprocessed = selector.transform(sample_preprocessed)
    prediction = model.predict(sample_preprocessed)
    return prediction


#TODO : utiliser argparse pour preciser le path des datas d'entrainement
path_data = "..."

if __name__ == "__main__":
	# load the model from disk
	loaded_model = pickle.load(open('Models/model_block_classification.pkl', 'rb'))
	loaded_vectorizer = pickle.load(open('Models/vectorizer_model.pkl', 'rb'))
	loaded_selector = pickle.load(open('Models/selector_model.pkl', 'rb'))

	# New datas preprocess
	new_samples = pd.read_csv(path_data)
	new_samples = new_samples.dropna()
	new_samples = new_samples.text.tolist()
	input_text = new_samples.copy()

	#Clean datas
	new_text = [spacy_tokenizer(s, stopwords) for s in new_samples]
	new_samples['new_text'] = [' '.join(s) for s in new_text]

	new_samples = loaded_vectorizer.transform(new_samples)
	new_samples = loaded_selector.transform(new_samples)
	new_predictions = loaded_model.predict(new_samples)

	df_pred = pd.DataFrame()
	df_pred['input'] = input_text
	df_pred['prediction'] = list(new_predictions)

