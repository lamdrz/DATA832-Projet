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
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import time

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

    def evaluate_clustering(self, true_labels, predicted_labels, model_name="Modèle", label_name="Vérité Terrain"):
        """
        Évalue le clustering via l'Indice de Rand Ajusté (ARI) et affiche la matrice d'alignement.
        """
        # 1. Calcul de l'ARI global
        ari = adjusted_rand_score(true_labels, predicted_labels)
        logging.info(f"ARI pour {model_name} vs {label_name} : {ari:.4f}")

        # 2. Création du tableau croisé (compte le nombre de pays par case)
        crosstab = pd.crosstab(
            true_labels, 
            predicted_labels, 
            rownames=[f'Vrais Labels ({label_name})'], 
            colnames=[f'Clusters Prédits ({model_name})']
        )

        # 3. Visualisation avec Seaborn (Heatmap)
        plt.figure(figsize=(8, 5))
        # annot=True pour afficher les nombres, fmt='d' pour des entiers (pas de virgules)
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', cbar=True)
        
        plt.title(f"Alignement : {model_name} vs {label_name}\n(Score ARI : {ari:.4f})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return ari

    def find_optimal_gmm(self, X_scaled, max_k=10):
        """
        Recherche du meilleur K pour GMM en utilisant les critères d'information BIC et AIC.
        Plus la valeur est basse, meilleur est le modèle.
        """
        logging.info("Recherche du K optimal pour GMM (AIC/BIC)...")
        bics = []
        aics = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
            gmm.fit(X_scaled)
            bics.append(gmm.bic(X_scaled))
            aics.append(gmm.aic(X_scaled))

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, bics, label='BIC', marker='o')
        plt.plot(k_range, aics, label='AIC', marker='s')
        plt.title("Critères d'Information GMM (AIC / BIC)")
        plt.xlabel("Nombre de composantes (K)")
        plt.ylabel("Valeur du score (Plus bas = Meilleur)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_k_distance(self, X_scaled, min_samples=5):
        """
        Affiche le k-distance plot pour aider à choisir le paramètre 'eps' de DBSCAN.
        On cherche le 'coude' sur la courbe des distances.
        """
        logging.info("Génération du K-distance plot pour DBSCAN...")
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, indices = neighbors_fit.kneighbors(X_scaled)
        
        # On trie les distances du k-ème voisin le plus proche
        distances = np.sort(distances[:, min_samples-1], axis=0)
        
        plt.figure(figsize=(8, 5))
        plt.plot(distances)
        plt.title(f"K-Distance Plot (min_samples={min_samples})")
        plt.xlabel("Points triés par distance")
        plt.ylabel(f"Distance au {min_samples}ème voisin le plus proche")
        plt.grid(True)
        plt.show()

    def run_all_clusterings_and_evaluate(self, X_scaled, y_hdi, y_gdp, k_chosen, eps_chosen, min_samples_chosen):
        """
        ÉTAPE 2 : Applique K-Means, GMM et DBSCAN, puis évalue avec l'ARI et affiche un graphique.
        """
        logging.info(f"--- Lancement des Clusterings avec K={k_chosen} et eps={eps_chosen} ---")
        
        # 1. K-Means
        kmeans = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X_scaled)
        
        # 2. Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=k_chosen, random_state=42, n_init=5)
        labels_gmm = gmm.fit_predict(X_scaled)
        
        # 3. DBSCAN
        dbscan = DBSCAN(eps=eps_chosen, min_samples=min_samples_chosen)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        
        # Log du nombre de clusters trouvés par DBSCAN (excluant le bruit : -1)
        n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        logging.info(f"DBSCAN a trouvé {n_clusters_dbscan} clusters (et du bruit).")

        # --- EVALUATION (ARI) ---
        results = {
            "K-Means": labels_kmeans,
            "GMM": labels_gmm,
            "DBSCAN": labels_dbscan
        }
        
        
        # Liste pour stocker les données pour le graphique
        viz_data = []

        for name, preds in results.items():
            ari_hdi = adjusted_rand_score(y_hdi, preds)
            ari_gdp = adjusted_rand_score(y_gdp, preds)
            print(f"{name:<15} | {ari_hdi:<15.4f} | {ari_gdp:<15.4f}")
            
            # Ajout des données pour Seaborn
            viz_data.append({"Modèle": name, "Référence": "HDI (Développement)", "Score ARI": ari_hdi})
            viz_data.append({"Modèle": name, "Référence": "GDP Class (Richesse)", "Score ARI": ari_gdp})
            
        # --- VISUALISATION GRAPHIQUE DES SCORES ---
        df_viz = pd.DataFrame(viz_data)
        
        plt.figure(figsize=(10, 6))
        # Utilisation d'un barplot groupé (hue="Référence")
        ax = sns.barplot(data=df_viz, x="Modèle", y="Score ARI", hue="Référence", palette=["#3498db", "#2ecc71"])
        
        plt.title("Performances des algorithmes de Clustering (Adjusted Rand Index)", fontsize=14, fontweight='bold')
        plt.ylabel("Score ARI (Plus proche de 1 = Meilleur)")
        plt.xlabel("Algorithmes Non Supervisés")
        
        # Ajout d'une ligne horizontale à 0 pour bien voir les scores négatifs (DBSCAN)
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
        
        # Ajout des valeurs exactes au-dessus de chaque barre pour plus de clarté
        for p in ax.patches:
            hauteur = p.get_height()
            # On décale le texte vers le bas si la valeur est négative
            offset = 5 if hauteur >= 0 else -15
            ax.annotate(f"{hauteur:.3f}", 
                        (p.get_x() + p.get_width() / 2., hauteur),
                        ha='center', va='center', 
                        xytext=(0, offset), 
                        textcoords='offset points',
                        fontsize=10)
            
        # Placement de la légende à l'extérieur
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
            
        return labels_kmeans, labels_gmm, labels_dbscan