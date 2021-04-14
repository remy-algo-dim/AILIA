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

