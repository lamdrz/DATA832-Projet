import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.logger import get_logger

logging = get_logger(__name__)

class DimensionalityReducer:
    def __init__(self):
        self.pca_features = [
            ...
        ]
        
        self.nmf_features = [
            ...
        ]

    def run_pca(self, df):
        logging.info("Démarrage de l'ACP...")
        
        X = df[self.pca_features]

        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ACP
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        variance_pc1_pc2 = pca.explained_variance_ratio_[:2].sum()
        logging.info(f"ACP terminée. Variance expliquée par PC1+PC2 : {variance_pc1_pc2:.2%}")

        return pca, X_pca, self.pca_features

    def plot_explained_variance(self, pca):
        """
        Affiche le graphe de la variance expliquée.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
        plt.title('Variance Cumulative Expliquée par l\'ACP')
        plt.xlabel('Nombre de Composantes')
        plt.ylabel('Variance Cumulative')
        plt.grid(True)
        plt.show()

    def plot_loadings(self, pca, features):
        """
        Affiche la heatmap des contributions (loadings) pour interpréter les axes.
        """
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(features))],
            index=features
        )

        plt.figure(figsize=(8, 4))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
        plt.title('Contributions des variables aux Composantes Principales')
        plt.tight_layout()
        plt.show()
        
    def find_optimal_k_nmf(self, df, max_k=10):
        """
        Calcule et trace l'erreur de reconstruction ||X - WH|| pour différents k
        afin de choisir le meilleur nombre de composantes.
        """
        logging.info("Recherche du K optimal pour la NMF...")
        features = [f for f in self.nmf_features if f in df.columns]
        
        X = df[features]

        # Standardisation (0 à 1)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        max_k = min(max_k, len(features))

        errors = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            # max_iter augmenté pour assurer la convergence
            nmf = NMF(n_components=k, init='nndsvda', max_iter=1000, random_state=42)
            nmf.fit(X_scaled)
            errors.append(nmf.reconstruction_err_)
            
        # Tracer la courbe de l'erreur (La méthode du Coude)
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, errors, marker='o', linestyle='-', color='g')
        plt.title('Erreur de reconstruction de la NMF en fonction de k')
        plt.xlabel('Nombre de composantes (k)')
        plt.ylabel('Erreur de reconstruction ||X - WH||')
        plt.xticks(K_range)
        plt.grid(True)
        plt.show()

    def run_nmf(self, df, k=3):
        """
        Applique la NMF avec le k choisi (par défaut 3).
        """
        logging.info(f"Démarrage de la NMF avec k={k}...")
        features = [f for f in self.nmf_features if f in df.columns]
        
        X = df[features]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        nmf = NMF(n_components=k, init='nndsvda', max_iter=1000, random_state=42)
        W = nmf.fit_transform(X_scaled) # Le poids de chaque profil par pays
        H = nmf.components_           # La composition des profils archétypaux

        logging.info("NMF terminée.")
        return nmf, W, H, features

    def plot_nmf_profiles(self, nmf_model, features):
        """
        Affiche la heatmap des profils (matrice H) pour interpréter les archétypes.
        """
        profiles = pd.DataFrame(
            nmf_model.components_,
            columns=features,
            index=[f'Profil {i+1}' for i in range(nmf_model.n_components)]
        )

        plt.figure(figsize=(10, 4))
        sns.heatmap(profiles, annot=True, cmap='YlOrRd', center=0)
        plt.title('Composition des profils énergétiques archétypaux (Matrice H)')
        plt.tight_layout()
        plt.show()