import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.logger import get_logger

logging = get_logger(__name__)

class DimensionalityReducer:
    def __init__(self):
        self.pca_features = [
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