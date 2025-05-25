# Paris Fire Brigade - Prédiction des Temps de Réponse

Ce projet vise à prédire les temps de réponse des véhicules de secours des pompiers de Paris en utilisant des techniques de machine learning et de deep learning.

##  Démarrage Rapide

### 1. Prérequis
- Python 3.8 ou plus récent
- Les données du challenge (à placer dans le dossier `data/`)

### 2. Installation
```bash

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Structure des données attendues
Placez vos fichiers de données dans le dossier `data/` :
```
data/
├── x_train.csv
├── y_train_u9upqBE.csv
├── x_train_additional_file.csv
├── x_test.csv
└── x_test_additional_file.csv
```

### 4. Exécution complète
```bash
# Prétraitement + Entraînement + Prédiction
python main.py --all
```

### 5. Résultats
- Modèles entraînés : `outputs/`
- Prédictions finales : `outputs/predictions.csv`
- Graphiques : `outputs/plots/`

## Objectifs

Prédire trois délais critiques pour les interventions d'urgence :
1. `delta selection-departure` : temps entre la sélection du véhicule et son départ
2. `delta departure-presentation` : temps entre le départ et l'arrivée sur site
3. `delta selection-presentation` : temps total entre la sélection et l'arrivée

## Structure du Projet

```
.
├── data/                           # Données brutes et prétraitées
│   ├── x_train.csv                # Données d'entraînement principales
│   ├── y_train_u9upqBE.csv        # Variables cibles
│   ├── x_train_additional_file.csv # Données additionnelles
│   ├── x_test.csv                 # Données de test
│   ├── x_test_additional_file.csv # Données de test additionnelles
│   ├── x_train_processed.csv      # Données d'entraînement prétraitées
│   └── x_test_processed.csv       # Données de test prétraitées
├── outputs/                        # Modèles entraînés et prédictions (ML classique)
├── scripts/                        # Scripts d'entraînement et de prédiction (ML classique)
│   ├── train.py                   # Script d'entraînement des modèles ML
│   └── predict.py                 # Script de prédiction ML
├── src/                           # Code source modulaire (ML classique)
│   ├── preprocessing.py           # Fonctions de prétraitement
│   ├── models.py                  # Définition des modèles ML
│   └── evaluate.py                # Fonctions d'évaluation
├── deep_learning/                 # Module Deep Learning
│   ├── data/
│   │   └── processed/             # Données préparées pour le DL
│   │       ├── X_train_seq.npy    # Séquences d'entraînement
│   │       ├── y_train_seq.npy    # Cibles séquentielles
│   │       ├── X_val_seq.npy      # Séquences de validation
│   │       ├── y_val_seq.npy      # Cibles de validation
│   │       ├── X_test_seq.npy     # Séquences de test
│   │       ├── y_test_seq.npy     # Cibles de test
│   │       ├── X_train_temp.npy   # Features temporelles d'entraînement
│   │       ├── y_train_temp.npy   # Cibles temporelles d'entraînement
│   │       ├── X_val_temp.npy     # Features temporelles de validation
│   │       ├── y_val_temp.npy     # Cibles temporelles de validation
│   │       ├── X_test_temp.npy    # Features temporelles de test
│   │       ├── y_test_temp.npy    # Cibles temporelles de test
│   │       ├── scaler_X_seq.joblib # Scaler pour les séquences
│   │       ├── scaler_y_seq.joblib # Scaler pour les cibles séquentielles
│   │       ├── scaler_X_temp.joblib # Scaler pour les features temporelles
│   │       ├── scaler_y_temp.joblib # Scaler pour les cibles temporelles
│   │       ├── features_temp.csv   # Noms des features temporelles
│   │       └── sector_encoder.joblib # Encodeur des secteurs géographiques
│   ├── models/                    # Modèles de deep learning entraînés
│   │   ├── lstm_model_*.keras     # Modèles LSTM standard
│   │   ├── bilstm_model_*.keras   # Modèles LSTM bidirectionnels
│   │   ├── gru_model_*.keras      # Modèles GRU
│   │   └── dense_model_*.keras    # Modèles Dense (MLP)
│   ├── outputs/                   # Résultats et visualisations DL
│   │   ├── plots/                 # Graphiques d'entraînement et prédictions
│   │   │   ├── *_history_*.png    # Courbes d'entraînement
│   │   │   └── *_predictions_*.png # Graphiques prédictions vs réel
│   │   ├── predictions/           # Résultats de prédictions
│   │   │   ├── *_results_*.json   # Métriques de performance
│   │   │   └── *_predictions_*.png # Visualisations des prédictions
│   │   ├── *_model_summary.txt    # Résumés des architectures
│   │   └── *_results_*.csv        # Comparaisons de modèles
│   ├── src/                       # Code source DL
│   │   ├── data_preparation_dl.py # Préparation des données pour DL
│   │   └── models_dl.py           # Architectures de modèles DL personnalisées
│   ├── train_dl.py                # Script d'entraînement DL principal
│   ├── predict_dl.py              # Script de prédiction DL
│   ├── setup_data_dl.py           # Configuration des données DL
│   ├── README.md                  # Documentation du module DL
│   └── rapport_projet_dl.md       # Rapport détaillé du projet DL
├── main.py                        # Script principal orchestrant le workflow ML
├── EDA_Paris_Fire_Challenge.ipynb # Analyse exploratoire des données
└── requirements.txt               # Dépendances Python
```

## Utilisation

### Machine Learning Classique

#### Pipeline complet
```bash
python main.py --all
```

#### Exécution par étapes
```bash
# Prétraitement uniquement
python main.py --preprocess

# Entraînement uniquement
python main.py --train

# Prédiction uniquement
python main.py --predict
```

### Deep Learning

#### Pipeline complet
```bash
cd deep_learning
python train_dl.py --all
```

#### Exécution par étapes
```bash
cd deep_learning

# Préparation des données pour DL
python train_dl.py --prepare

# Entraînement des modèles LSTM/GRU
python train_dl.py --train_lstm

# Entraînement du modèle Dense
python train_dl.py --train_dense

# Prédictions avec un modèle spécifique
python predict_dl.py --model bilstm  # ou lstm, gru, dense
```

## Approche Technique

### 1. Machine Learning Classique
   - **Prétraitement** : Encodage, gestion des valeurs manquantes, normalisation
   - **Feature Engineering** : Features temporelles, géographiques, d'interaction
   - **Modélisation** : RandomForest, XGBoost, Gradient Boosting
   - **Approches** : Multi-output et modèles séparés

### 2. Deep Learning
   - **Préparation des données** : Création de séquences temporelles par secteur géographique
   - **Architectures personnalisées** :
     - **LSTM Standard** : Analyse des séquences d'interventions
     - **LSTM Bidirectionnel** : Capture des dépendances temporelles bidirectionnelles
     - **GRU** : Alternative plus légère aux LSTM
     - **Réseau Dense** : Modèle MLP pour features temporelles
   - **Features** : Séquences de 10 interventions consécutives par secteur + caractéristiques temporelles

## Résultats

### Machine Learning Classique
Les meilleurs résultats ont été obtenus avec les modèles XGBoost, particulièrement avec l'approche de modèles séparés pour chaque cible.

### Deep Learning
| **Modèle** | **MAE** | **RMSE** | **MAPE** | **Performance** |
|------------|---------|----------|----------|-----------------|
| **BiLSTM** | **116.12** | **180.23** | **16.91%** | **Meilleur** |
| **LSTM** | 116.73 | 180.17 | 17.08% | Très bon |
| **GRU** | 116.54 | 180.53 | 17.00% | Bon |
| **Dense** | 115.98 | 221.06 | 25.87% | Plus d'erreur |

Le modèle LSTM Bidirectionnel obtient les meilleures performances avec un MAPE de 16.91%

## Évaluation

- **Métriques** : MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error)
- **Validation** : Split train/validation/test (60/20/20)
- **Visualisations** : Courbes d'entraînement, prédictions vs valeurs réelles

## Licence

Ce projet est sous licence MIT. 
