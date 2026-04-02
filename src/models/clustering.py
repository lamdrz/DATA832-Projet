import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from config.logger import get_logger
from config.settings import PCA_FEATURES

logging = get_logger(__name__)

class ClusteringAnalyzer:
    def __init__(self):
        self.pca_features = list(PCA_FEATURES)

    def prepare_labels_and_data(self, df):
        """
        ÉTAPE 1 : Création des pseudo-labels et séparation des données.
        """
        logging.info("Préparation des pseudo-labels (Vérité terrain)...")
        df_cluster = df.copy()

        # Discrétisation du PIB par habitant en 4 classes (Q1, Q2, Q3, Q4)
        # pd.qcut coupe en tranches contenant le même nombre de pays
        df_cluster['gdp_class'] = pd.qcut(
            df_cluster['gdp_per_capita'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        # Extraction des labels (Vérité terrain)
        y_hdi = df_cluster['hdicode']
        y_gdp = df_cluster['gdp_class']

        # Extraction des caractéristiques pures (sans les labels !)
        X = df_cluster[self.pca_features]

        # Standardisation obligatoire avant le clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y_hdi, y_gdp, df_cluster

    def find_optimal_kmeans(self, X_scaled, max_k=10):
        """
        ÉTAPE 2a : Recherche du meilleur K pour K-Means (Coude + Silhouette)
        """
        logging.info("Recherche du K optimal pour K-Means...")
        inertias = []
        silhouettes = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))

        # Affichage des deux métriques sur une figure à 2 panneaux
        fig, ax1 = plt.subplots(1, 2, figsize=(14, 5))

        # Panneau 1 : Méthode du Coude (Inertie)
        ax1[0].plot(k_range, inertias, marker='o', color='b')
        ax1[0].set_title("Méthode du Coude (Inertie)")
        ax1[0].set_xlabel("Nombre de clusters (K)")
        ax1[0].set_ylabel("Inertie (Somme des carrés intra-cluster)")
        ax1[0].grid(True)

        # Panneau 2 : Score de Silhouette
        ax1[1].plot(k_range, silhouettes, marker='s', color='g')
        ax1[1].set_title("Score de Silhouette")
        ax1[1].set_xlabel("Nombre de clusters (K)")
        ax1[1].set_ylabel("Silhouette Score (Plus proche de 1 = Meilleur)")
        ax1[1].grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate_clustering(self, true_labels, predicted_labels, model_name="Modèle"):
        """
        Évalue le clustering via l'Indice de Rand Ajusté (ARI).
        """
        ari = adjusted_rand_score(true_labels, predicted_labels)
        logging.info(f"ARI pour {model_name} : {ari:.4f}")
        return ari