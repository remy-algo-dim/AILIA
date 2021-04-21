# ALFRED
Projet RH - Extraction de données de CVs

## Model NLP

Notebook/bloc_classification_NLP.ipynb --> Run un NLP sur des blocs de CV afin de pouvoir classifier ces blocs (experience pro,
skills, academic background ...) --> (utiliser python scripts suivants a la place)

train_nlp.py --> train un model de NLP de classification de blocs (save 3 pkl)
predict_nlp.py --> permet de renvoyer les predictions de ce modele sur de nouveaux samples

Dossier "Models" --> contient les pkl necessaires à l'utilisation du modele NLP


## Data cleaning

doccano_json_cleaning.ipynb --> Permet de clean les INPUTS et OUTPUTS de DOCCANO

## Model Word2Vec
Jupyter notebook avec plusieurs tests pour rassembler les mots clés de chaque filtres.(Word : Microsoft Word, Word, Woord, Wordd...)

## Model YoloV5
Fichier Python qui effectue les 3 premières étapes du produit : - PDF to JPG
																- YoloV5 : Détection des 4 BLOCs
																- Découpage du JPG en 4 BLOCs
																

## Model NER

Pré recquis : Train and save un modèle de NER.

Dans Model_NER_3.py on a un script permettant de print des metrics pertinentes du NER (attention, il se peut qu'avec des versions différentes de spacy on il soit impossible de run ce script)


## Téléchargement des CV via Outlook

Dans le dossier "CV_downloading" figure 2 fichier python:
	- CV_extraction_config.py --> fichier de config dans lequel sont définies plusieurs variables (email adress, PSW, nbe de mails a scrapper ...)
	- CV_extraction.py --> fichier permettant le telechargement de TOUTES les PJ se trouvant dans les mails choisis.

En cas d'abscence de PJ : aucune erreur n'est soulevée
En cas de présence de plusieurs PJ : Elles sont toutes téléchargées
TODO : 
- Rule base afin de ne télécharger QUE le CV dans le cas de présence de plusieurs PJ
- Gérer GMAIL