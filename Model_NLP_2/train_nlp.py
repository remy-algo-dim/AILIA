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

nltk.download('wordnet')
nltk.download('stopwords')

# On ne tiendra pas compte de l'ordre des mots
# On load les données : Nos données sont organisées sous forme de DF à 2 colonnes : texte  - label

#TODO : utiliser argparse pour preciser le path des datas d'entrainement
path_data = "../../Datas/db-text-for-bloc-labelization-nlp.csv"

nlp = French()

def spacy_tokenizer(sentence, stopwords):
    """Renvoie une sentence preprocessed"""
    # On crée notre objet token
    mytokens = nlp(sentence)
    # On lemmatize chaque token et on convertit en lowercase (lemmatization=forme cannonique/standard d'un mot)
    lemmatizer = WordNetLemmatizer()
    mytokens = [lemmatizer.lemmatize(str(word)).lower() for word in mytokens if not     
          word.is_punct and not word.like_num and word.text != 'n']
    # Removing stop words
    mytokens = [word for word in mytokens if word not in stopwords.words()]
    mytokens = [word for word in mytokens if word != 'sautdeligne']
    # Remove accents
    #mytokens = [strip_accents_ascii(word) for word in mytokens]
    return mytokens


# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)
# Limitation du nombre de features
TOP_K = 20000
# On split le texte en "mots" ou "n-grams"
# One of 'word', 'char'.
TOKEN_MODE = 'word'
# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.
    # Inputs
      train_texts: list, training text strings.
      train_labels: np.ndarray, training labels.
      val_texts: list, validation text strings.
    # Outputs 
       x_train, x_val: vectorized training and validation texts
    """
    # Paramètres qu'on passe au TF-IDF
    kwargs = {
      'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
      'dtype': 'int32',
      'strip_accents': 'unicode',
      'decode_error': 'replace', 
      'analyzer': TOKEN_MODE,  # Split text into word tokens.
       'min_df': MIN_DOCUMENT_FREQUENCY,
       }
    vectorizer = TfidfVectorizer(**kwargs)
    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)
    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)
    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val, vectorizer, selector



if __name__ == "__main__":

    df = pd.read_csv(path_data)
    df = df.dropna()
    text = df.text.tolist()

    #Clean datas
    new_text = [spacy_tokenizer(s, stopwords) for s in text]
    df['new_text'] = [' '.join(s) for s in new_text]

    # train test split 70%/30%
    tts = int(0.7*len(df))
    data_train = df[:tts]
    data_valid = df[tts:]

    #Training
    print("Training ...")
    x_train, x_val, vectorizer, selector = ngram_vectorize(data_train["new_text"], data_train["label"], data_valid["new_text"])
    y_train = data_train["label"]
    y_test = data_valid["label"]

    # Modeling
    bag_clf =BaggingClassifier(DecisionTreeClassifier(),n_estimators=50, n_jobs=-1)
    bag_clf.fit(x_train, y_train)
    predicted = bag_clf.predict(x_val)
    print("Metrics ...")
    print(metrics.classification_report(y_test, predicted))

    # Save models
    print("Saving models")
    model_filename = 'Models/model_block_classification.pkl'
    vectorizer_filename = 'Models/vectorizer_model.pkl'
    selector_filename = 'Models/selector_model.pkl'

    pickle.dump(bag_clf, open(model_filename, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
    pickle.dump(selector, open(selector_filename, 'wb'))


