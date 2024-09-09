# Diagnostic du Cancer du Sein avec un Réseau Neuronal Perceptron

Ce projet démontre l'utilisation d'un réseau neuronal Perceptron pour classer les diagnostics de cancer du sein sur la base du Jeu de Données de Cancer du Sein de Wisconsin.

## Table des Matières
- [Installation](#installation)
- [Jeu de Données](#jeu-de-donnees)
- [Exploration et Prétraitement](#exploration-et-pretraitement)
- [Détection des Valeurs Aberrantes](#detection-des-valeurs-aberrantes)
- [Réseau Neuronal](#reseau-neuronal)
- [Résultats](#resultats)
- [Conclusion](#conclusion)
- [Licence](#licence)

## Installation

Pour exécuter ce projet, vous devrez installer les bibliothèques nécessaires. Vous pouvez le faire en utilisant pip pour installer les éléments suivants :

```
scikit-learn
matplotlib
numpy
pandas
```

Sinon, vous pouvez utiliser l'environnement Anaconda.

## Jeu de Données

Le jeu de données utilisé est le Jeu de Données de Cancer du Sein de Wisconsin. Assurez-vous d'avoir le fichier de données nommé `BreastCancerWisconsinDataSet.csv` dans votre répertoire de travail.

## Exploration et Prétraitement

Après avoir chargé le jeu de données, le projet effectue une exploration initiale et un prétraitement :

- **Exploration :** L'en-tête du jeu de données et les informations de base sont inspectés pour comprendre sa structure et identifier les valeurs manquantes éventuelles.
- **Prétraitement :** La première colonne (numéros d'ID) et la dernière colonne (vide) sont supprimées du jeu de données.

## Détection des Valeurs Aberrantes (Outliers)

Les valeurs aberrantes (outliers) sont vérifiées à l'aide d'une fonction personnalisée pour s'assurer qu'il n'y a pas de valeurs extrêmes pouvant affecter les performances du réseau neuronal. Après avoir identifié les colonnes avec de nombreux zéros, une nouvelle version filtrée du jeu de données est créée.

## Réseau Neuronal

Deux modèles Perceptron sont construits et testés :

1. **Modèle 1 (p1) :** Entraîné sur le jeu de données original.
2. **Modèle 2 (p2) :** Entraîné sur le jeu de données filtré, excluant les lignes avec des valeurs nulles dans les colonnes liées à la concavité.

### Standardisation des Données

Les caractéristiques sont normalisées à l'aide de `MinMaxScaler` pour mettre les données à l'échelle entre 0 et 1.

### Visualisation des Données

Des graphiques de dispersion sont générés pour vérifier si les classes (Malin et Bénin) sont séparables linéairement, ce qui est une exigence clé pour l'utilisation d'un Perceptron.

### Entraînement et Test

Le jeu de données est divisé en ensembles d'entraînement (70%) et de test (30%). Les modèles sont ensuite entraînés et évalués en utilisant la précision, les rapports de classification et les matrices de confusion.

## Résultats

- **Modèle 1 (p1) :** Montre une bonne performance sur le jeu de données original.
- **Modèle 2 (p2) :** Montre une précision supérieure sur le jeu de données filtré, avec zéro faux négatifs et des faux positifs minimaux.

## Conclusion

- **Modèle 1** est recommandé pour les patients sans concavités dans leur masse mammaire (2,3% du jeu de données).
- **Modèle 2** est recommandé pour les patients avec des concavités dans leur masse mammaire (97,7% du jeu de données) en raison de sa plus grande précision et de son taux d'erreur plus faible.

## Licence

Ce projet est sous la Licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---
