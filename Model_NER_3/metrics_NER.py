from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.lang.fr import French
import pandas as pd
import json
import ast
from spacy_langdetect import LanguageDetector
import minibatch
import random
from spacy.util import minibatch, compounding
from spacy.training import Example
import numpy as np


""" Python script permettant de print les metrics du NER.
cmd : python metrics_NER.py  ---> 2 inputs : "path jsonl labelisé via Doccano" & "path modele NER en question"
                             ---> outputs : metrics printed sur le terminal
"""


def read_jsonl(path_json_file):
    with open(path_json_file, 'r') as json_file:
        json_list = list(json_file)
    return json_list


def reformate_datas(TRAIN_DATA):
    """Input: JSON file issu de Doccano
       Output: Datas mis au format du NER"""
    TRAIN = []
    for element in TRAIN_DATA:
        # null pose pb. On le remplace par un mot qqconque
        element = element.replace('null', '"ok"')
        d = eval(element)
        dictOfLabels = {}
        dictOfLabels['entities'] = d['labels']
        listOfTuples = []
        for element in dictOfLabels['entities']:
            element = tuple(element)
            listOfTuples.append(element)
        dictOfLabels['entities'] =  listOfTuples
        TRAIN.append((d['text'], dictOfLabels))
    return TRAIN


def X_test(DATASET):
    """Permet de recuperer le texte afin d'effectuer les predictions"""
    texts = []
    for element in DATASET:
        text = element[0]
        texts.append(text)
    return texts



def test_ner_model(output_dir, test_text):
    """Permet de tester un NER
    Input: directory du modele et texte a tester ainsi que le texte
    Output: Renvoie les labels predits ainsi que le texte correspondant a chaque label"""
    # Test the saved model
    nlp = spacy.load(output_dir)
    doc = nlp(test_text)
    predicted_labels = []
    corresponding_text = []
    for ent in doc.ents:
        predicted_labels.append(ent.label_)
        corresponding_text.append(ent.text)
    return predicted_labels, corresponding_text


def retrieve_trueLabelAndWord(DATASET):
    """Permet de recuperer deux listes: True label, et le word correspondant
    Ceci permettra de comparer les valeurs reelles aux valeurs predites du NER"""
    true_labels = []
    corresponding_text = []
    for element in DATASET:
        # on retrieve le texte
        text = element[0]
        label = element[1]
        label = label['entities']
        for e in label:
            start = e[0]
            end = e[1]
            target = e[2]
            true_labels.append(target)
            corresponding_word = text[start:end]
            corresponding_text.append(corresponding_word)
    return true_labels, corresponding_text



def make_tuples(true_token, true_labels):
    """Input: la liste de mots (token) ainsi que la liste de labels correspondante
       Output: liste de tuples ou chaque tuple est de la forme (token, label)"""
    true_tuple = []
    for w, p in zip(true_token, true_labels):
        true_tuple.append((w, p))
    return true_tuple


# Explication : supposons que dans le Test set il y ait 200 mots a labeliser, et que le modele en trouve 100, 
# alors, le coverage serait de 50%. Certes toutes les predictions peuvent etre bonnes, mais il en manque la moitié.
def get_metrics(y_pred, y_true):
    """Input: y_pred et y_true sous forme de listes de tuple (texte, label)
    Renvoie le poucentage de bons tuples (texte, label) détectés par le modele"""
    y_true_copy = y_true.copy()
    y_pred_copy = y_pred.copy()
    wrong_tuples = []
    good_tuples = []
    count = 0
    for tuple_ in y_pred_copy:
        count += 1
        if tuple_ in y_true_copy:
            y_true_copy.remove(tuple_)
            good_tuples.append(tuple_)
        else:
            # tuple qui na pas lieu d'exister
            wrong_tuples.append(tuple_)
    #print(count)
    accuracy_1 = np.round(len(good_tuples)/len(y_pred), 3) # cb de bonnes predictions par rap a ce que je predis
    accuracy_2 = np.round(len(good_tuples)/len(y_true), 3) # % cb de bonnes predictions par rap a ce quil fallait trouver
    
    #lets take a look on bad predictions
    # erreur de type 1 : mot trouvé, mais prédiction fausse
    label_error = {} # on spotera les labels où le modèle fait erreur
    count_error_I = 0
    for tuple_ in wrong_tuples:
        for TrueTuple in y_true_copy:
            if tuple_[0] == TrueTuple[0]:
                count_error_I += 1
                if (tuple_[1], TrueTuple[1]) not in label_error:
                    label_error[(tuple_[1], TrueTuple[1])] = 1
                else:
                    label_error[(tuple_[1], TrueTuple[1])] += 1
                y_true_copy.remove(TrueTuple)
                break
        # Si c'est pas une erreur de type I c'est qu'il s'agit d'une erreur de type II 
        # cad un mot releve qui n'aurait jamais du l'etre
    count_error_II = len(wrong_tuples) - count_error_I
    coverage = np.round(len(y_true) / len(y_pred) * 100, 1)
    coverage = np.round(np.abs(100 - coverage), 1)
    print('*************************************************************************************')
    print('             M  E  T  R  I  C  S                                 ')
    print('*************************************************************************************')
    print("Nombre de mots à prédire pour l'ensemble du test set : ", len(y_true))
    print("Nombre de mots prédits par le NER pour l'ensemble du test set : ", len(y_pred))
    print('Coverage : ', coverage, '% de difference')
    print('----------------------------------------------------------------------')
    print(len(good_tuples), ' mots ont bien été relevés ET prédits par le NER')
    print(len(wrong_tuples), " mots ont mal été prédits (erreur de type I ou type II)")
    print('Bonnes prédictions + Mauvaises prédictions =', len(good_tuples)+len(wrong_tuples))
    print('----------------------------------------------------------------------')
    print('Accuracy I (bonne pred / total pred) : ', accuracy_1*100, ' %')
    print('Accuracy II (bonne pred / total expected pred) : ', accuracy_2*100, ' %')
    print('----------------------------------------------------------------------')
    print('Nbe erreur type I : ', count_error_I, ' (mots bien relevés mais mal prédits)')
    print('Nbe erreur type II : ', count_error_II, " (mots qui n'auraient pas du être prédits)")
    print('----------------------------------------------------------------------')
    print("Confusions faites par l'algo (prediction, true_label): \n", label_error)


if __name__ == "__main__":
    #/Users/remyadda/Downloads/cv_shani_exp_pro.jsonl
    #/Users/remyadda/Desktop/AD/Projets/ALFRED/ALFRED_PROJECT/Model_NER_3
    path_json_file = input("Enter le path du fichier JSONL labélisé : ")
    output_dir = input("Entrer le path du NER modèle : ")

    DATA = read_jsonl(path_json_file)
    DATA = reformate_datas(DATA)
    #On ne recupere que le texte
    Xtest = X_test(DATA)

    #On recupere les true tokens et label sous forme de listes
    true_labels, true_corresponding_words = retrieve_trueLabelAndWord(DATA)
    #On met le tout dans une liste de tuples
    # Nbe de tuples a trouver
    true_tuple = make_tuples(true_corresponding_words, true_labels)

    # Prédictions
    preds = []
    predicted_corresponding_word_list = []
    for text in Xtest:
        predictions, predicted_corresponding_word = test_ner_model(output_dir, text)
        preds.extend(predictions)
        predicted_corresponding_word_list.extend(predicted_corresponding_word)

    # On met également les prédictions dans une liste de tuples
    predicted_tuple = make_tuples(predicted_corresponding_word_list, preds)

    #Metrics
    get_metrics(predicted_tuple, true_tuple)


