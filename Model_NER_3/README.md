# AILIA

Projet RH - Extraction de données de CVs

## Model NER

### Fichier python : metrics_NER.py

Pré recquis : Train et Save un modèle de NER dans un dossier au choix. Par exemple, un NER a déjà été train et saved dans le dossier : 
"ALFRED_PROJECT/Model_NER_3/Modeles"

Dans Model_NER_3.py on a un script permettant de print des metrics pertinentes du NER (attention, il se peut qu'avec des versions différentes de spacy on il soit impossible de run ce script)



**************** Comment RUN ce fichier ? ****************

commande : python metrics_NER.py

INPUTS :

2 inputs sont demandés : 
- Path du fichier doccano labelisé. Il s'agit du fichier JSONL exporté de DOCCANO après labelisation (le fichier contient donc le texte et les labels)
- Path du modele NER à tester ("ALFRED_PROJECT/Model_NER_3/Modeles" par exemple)

ATTENTION A BIEN MATCHER LE NER AVEC LE FICHIER DOCCANO EN INPUT. Exemple, si le fichier doccano entré en input concerne celui des expériences professionnelles, il est inutile d'entrer en input un NER trained sur le bloc des SKILLS. 


OUTPOUTS :
Voici un exemple:

Supposons qu'on ait un ces mots à relever : 
[('Formation', 'U-Diplome'), ('premiers', 'B-DiplomeSpec'), ('Secours', 'I-DiplomeSpec'), ('Civique', 'L-DiplomeSpec')]

Supponsons également que notre NER nous ait renvoye cela:
[('Formation', 'U-Diplome'), ('premiers', 'B-DiplomeSpec'), ('Secours', 'I-DiplomeSpec'), ('Civique', 'L-DiplomeSpec'), ('MASTER', 'I-DiplomeSpec')]

Voici ce que nous renvoie le script python

"""
Nombre de mots à prédire pour l'ensemble du test set :  4 			            (en effet il y a bien 4 mots qu'il fallait prédire)
Nombre de mots prédits par le NER pour l'ensemble du test set :  5 				(notre modèle en a trouvé 5)

Coverage :  20.0 % de difference 												(Il s'agit de la différence entre le nombre de mots qu'il fallait réellement prédire avec le nombre prédits par l'algo)
----------------------------------------------------------------------
3  mots ont bien été relevés ET prédits par le NER 							    (Il s'agit des trois premiers qui ont parfaitement été détectés par l'algo)
2  mots ont mal été prédits (erreur de type I ou type II)						(Les deux derniers mots prédits sont donc des erreurs)
Bonnes prédictions + Mauvaises prédictions = 5
----------------------------------------------------------------------
Accuracy I (bonne pred / total pred) :  60.0  %									(Formule claire)
Accuracy II (bonne pred / total expected pred) :  75.0  %						(Formule claire)
----------------------------------------------------------------------
Nbe erreur type I :  1  (mots bien relevés mais mal prédits)					(1 mot a bien été relevé, mais le label prédit par l'algo est incorrect)
Nbe erreur type II :  1  (mots qui n'auraient pas du être prédits)				(1 mot a été relevé par l'algo sans aucun raison)
----------------------------------------------------------------------
Confusions faites par l'algo (prediction, true_label): 							(Voici les labels qui ont fait l'objet d'une confusion par l'algo)
 {('I-DiplomeSpec', 'L-DiplomeSpec'): 1}
"""










