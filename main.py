#######################################################################
#                                                                     #
#            Pima Indians Diabetes Dataset Analysis                   #
#                                                                     #
# Authors: Augustin Chavanes & Corentin Salvi                         #
#######################################################################

#region Importation des bibliothèques necessaires
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')
#endregion

# Chargement des données
diabetes = pd.read_csv('./train.csv', na_values=['?'])

# Création des dossiers de sortie
output_dir = './DataTreatment'
output_dir_after = './DataAfterTreatment'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_after, exist_ok=True)

# ==================================================
# ÉTAPE 0: NETTOYAGE ET ETAT BRUT
# ==================================================

# Suppression de l'ID (aucune valeur prédictive)
if 'id' in diabetes.columns:
    diabetes = diabetes.drop(columns=['id'])

# Suppression des colonnes non souhaitées
columns_to_drop = ['education_level', 'income_level']
diabetes = diabetes.drop(columns=columns_to_drop, errors='ignore')

# --- SAUVEGARDE DES DISTRIBUTIONS BRUTES (DataTreatment) ---
print(f"[OK] Génération des distributions BRUTES dans {output_dir}...")
raw_numeric_cols = diabetes.select_dtypes(include=['int64', 'float64']).columns
for column in raw_numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(diabetes[column], kde=True, color='orange')
    plt.title(f'Distribution BRUTE de {column}')
    plt.savefig(os.path.join(output_dir, f'{column}_brut.png'))
    plt.close()

# ==================================================
# ÉTAPE 1: ENCODAGE DES VARIABLES CATÉGORIELLES
# ==================================================

categorical_columns = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
cols_to_encode = [c for c in categorical_columns if c in diabetes.columns]

diabetes = pd.get_dummies(diabetes, columns=cols_to_encode, drop_first=False)
print(f"[OK] One-Hot Encoding appliqué.")

# ==================================================
# ÉTAPE 2: Z-SCORE (STANDARDIZATION)
# ==================================================

target_col = 'diagnosed_diabetes'
numeric_cols = diabetes.select_dtypes(include=['int64', 'float64']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)

scaler = StandardScaler()
diabetes[numeric_cols] = scaler.fit_transform(diabetes[numeric_cols])
print(f"[OK] Z-Score appliqué.")

# ==================================================
# ÉTAPE 3: VISUALISATION APRÈS TRAITEMENT (DataAfterTreatment)
# ==================================================

print(f"[OK] Génération des distributions TRAITÉES dans {output_dir_after}...")
for column in numeric_cols:
    plt.figure(figsize=(8, 4))
    diabetes[column].hist(bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribution de {column} (après Z-Score)')
    plt.savefig(os.path.join(output_dir_after, f'{column}_standardise.png'))
    plt.close()


# ==================================================
# ÉTAPE 4: ANALYSE DE CORRÉLATION
# ==================================================

if target_col in diabetes.columns:
    # Calcul de la corrélation
    correlations = diabetes.corr()[target_col].sort_values(ascending=False)
    
    # Affichage console
    print("\n" + "="*50)
    print("TOP 10 DES VARIABLES LIÉES POSITIVEMENT :")
    print(correlations.head(11)) 

    # --- NOUVEAU : Sauvegarde des corrélations textuelles ---
    with open(os.path.join(output_dir_after, 'top_correlations.txt'), 'w') as f:
        f.write("TOP 10 DES VARIABLES LIÉES POSITIVEMENT AU DIABÈTE :\n")
        f.write(correlations.head(11).to_string())
        f.write("\n\nTOP 5 DES VARIABLES LIÉES NÉGATIVEMENT :\n")
        f.write(correlations.tail(5).to_string())

    # --- Génération de la Heatmap ---
    plt.figure(figsize=(20, 15))
    
    # Masque pour masquer la partie redondante (triangle supérieur)
    mask = np.triu(np.ones_like(diabetes.corr(), dtype=bool))
    
    sns.heatmap(diabetes.corr(), 
                mask=mask, 
                cmap='coolwarm', 
                center=0, 
                linewidths=0.1,
                annot=False) # Désactivé pour la clarté sur 35 colonnes
    
    plt.title(f'Matrice de Corrélation Finale', fontsize=16)
    
    # Sauvegarde sous le nom demandé
    plt.savefig(os.path.join("./", 'correlation_matrix.png'), dpi=300, bbox_inches='tight')


print(f"\n[OK] Matrice de corrélation enregistrée sous : correlation_matrix.png")